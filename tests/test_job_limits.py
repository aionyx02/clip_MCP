"""Tests for the gate that keeps background jobs from exhausting the machine's memory.

The gate's whole decision is `JobManager._plan`, and it is exercised here
through the repository transaction it really runs in, so the queue, the
memory arithmetic, and the SQL that selects unfinished jobs are all covered
without launching a worker process.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pytest

from app.engine import renderer, resources
from app.engine.builder import FFmpegRenderer
from app.engine.ffmpeg import graph_from_file, run_ffmpeg
from app.engine.renderer import WARMUP, JobManager
from app.models.job import Job, JobKind, JobStatus
from app.server import _referenced_assets, import_asset, repo
from app.storage.repo import Repository
from helpers import build_project, insert, video_track

GIGABYTE = 1024 * resources.MEGABYTE

@pytest.fixture
def manager(tmp_path: Path) -> JobManager:
    """A job manager on a database of its own.

    Returns:
        The manager; `manager.repo` is the repository behind it.
    """
    return JobManager(Repository(str(tmp_path / "jobs.db")))

def queue(manager: JobManager, memory: int = 0, **fields) -> Job:
    """Store a queued job that no worker has been launched for.

    Args:
        manager: Manager whose repository receives the job.
        memory: The job's memory estimate in bytes.
        **fields: Other job fields, such as `status` or `admitted_at`.

    Returns:
        The stored job.
    """
    job = Job(kind=JobKind.ANALYZE, memory_estimate=memory, **fields)
    manager.repo.add_job(job)
    return job

def plan(manager: JobManager, spare: Optional[int], limit: int = 2, monkeypatch=None) -> List[Job]:
    """Run the gate once and report which jobs it admitted.

    Args:
        manager: Manager whose queue is planned.
        spare: Free memory the machine is pretending to have, or `None` for a
            machine that cannot be measured.
        limit: How many jobs may run at once.
        monkeypatch: Fixture used to stand in for the machine.

    Returns:
        The admitted jobs, oldest first.
    """
    monkeypatch.setattr(resources, "spare_bytes", lambda: spare)
    monkeypatch.setattr(resources, "max_concurrent_jobs", lambda: limit)
    return [job for job in manager.repo.update_active_jobs(manager._plan) if job.admitted_at is not None]

def stage_of(manager: JobManager, job: Job) -> Optional[str]:
    """Read back what a job says it is waiting for.

    Args:
        manager: Manager holding the job.
        job: Job to read.

    Returns:
        The job's stage.
    """
    return manager.repo.get_job(job.job_id).stage

def test_only_as_many_jobs_start_as_the_limit_allows(manager: JobManager, monkeypatch) -> None:
    first, second, third = (queue(manager) for _ in range(3))
    admitted = plan(manager, spare=8 * GIGABYTE, limit=2, monkeypatch=monkeypatch)
    assert [job.job_id for job in admitted] == [first.job_id, second.job_id]
    assert "waiting" in stage_of(manager, third)

def test_a_job_waits_rather_than_running_without_the_memory_it_needs(manager: JobManager, monkeypatch) -> None:
    small, large = queue(manager, memory=GIGABYTE), queue(manager, memory=6 * GIGABYTE)
    admitted = plan(manager, spare=2 * GIGABYTE, limit=4, monkeypatch=monkeypatch)
    assert [job.job_id for job in admitted] == [small.job_id]
    assert "6144 MB of free memory" in stage_of(manager, large)

def test_the_only_job_starts_even_when_the_machine_is_short_of_memory(manager: JobManager, monkeypatch) -> None:
    # Nothing is running, so waiting could only wait forever; a slow job beats no job.
    alone = queue(manager, memory=6 * GIGABYTE)
    assert [job.job_id for job in plan(manager, spare=0, limit=2, monkeypatch=monkeypatch)] == [alone.job_id]

def test_jobs_start_in_the_order_they_were_asked_for(manager: JobManager, monkeypatch) -> None:
    settled = datetime.now(timezone.utc) - WARMUP - timedelta(seconds=1)
    queue(manager, memory=GIGABYTE, status=JobStatus.RUNNING, admitted_at=settled)
    large, small = queue(manager, memory=6 * GIGABYTE), queue(manager, memory=GIGABYTE)
    # The small job fits in what is free, but the large one asked first and is waiting for it.
    plan(manager, spare=2 * GIGABYTE, limit=4, monkeypatch=monkeypatch)
    assert manager.repo.get_job(large.job_id).admitted_at is None
    assert manager.repo.get_job(small.job_id).admitted_at is None
    assert stage_of(manager, small) == "queued behind 1 job(s)"

def test_a_job_that_has_just_started_is_counted_before_it_claims_its_memory(manager: JobManager, monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    queue(manager, memory=3 * GIGABYTE, status=JobStatus.RUNNING, admitted_at=now)
    waiting = queue(manager, memory=3 * GIGABYTE)
    # The machine still reports 4 GB free because the running job has not allocated yet.
    plan(manager, spare=4 * GIGABYTE, limit=4, monkeypatch=monkeypatch)
    assert manager.repo.get_job(waiting.job_id).admitted_at is None

def test_a_job_that_has_been_running_a_while_is_already_in_what_the_machine_reports(manager: JobManager, monkeypatch) -> None:
    started = datetime.now(timezone.utc) - WARMUP - timedelta(seconds=1)
    queue(manager, memory=3 * GIGABYTE, status=JobStatus.RUNNING, admitted_at=started)
    waiting = queue(manager, memory=3 * GIGABYTE)
    admitted = plan(manager, spare=4 * GIGABYTE, limit=4, monkeypatch=monkeypatch)
    assert [job.job_id for job in admitted] == [waiting.job_id]

def test_a_job_whose_worker_stopped_reporting_no_longer_holds_a_slot(manager: JobManager, monkeypatch) -> None:
    queue(manager, status=JobStatus.RUNNING, admitted_at=datetime.now(timezone.utc))
    waiting = queue(manager)
    monkeypatch.setattr(renderer, "STALE_AFTER", timedelta(seconds=-1))
    admitted = plan(manager, spare=8 * GIGABYTE, limit=1, monkeypatch=monkeypatch)
    assert [job.job_id for job in admitted] == [waiting.job_id]

def test_a_job_waiting_for_a_slot_is_not_failed_for_being_quiet(manager: JobManager, monkeypatch) -> None:
    waiting = queue(manager)
    # get_job fails jobs whose worker has gone silent; one that never had a worker has not,
    # however long it sits there, which is the whole point of queueing rather than starting.
    monkeypatch.setattr(renderer, "STALE_AFTER", timedelta(seconds=-1))
    monkeypatch.setattr(resources, "max_concurrent_jobs", lambda: 0)
    assert manager.get_job(waiting.job_id).status == JobStatus.QUEUED

def test_a_cancelled_job_does_not_take_a_slot(manager: JobManager, monkeypatch) -> None:
    queue(manager, cancel_requested=True)
    running = queue(manager)
    admitted = plan(manager, spare=8 * GIGABYTE, limit=1, monkeypatch=monkeypatch)
    assert [job.job_id for job in admitted] == [running.job_id]

def test_finished_jobs_are_left_out_of_the_decision(manager: JobManager, monkeypatch) -> None:
    for status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        queue(manager, status=status, admitted_at=datetime.now(timezone.utc))
    waiting = queue(manager)
    admitted = plan(manager, spare=8 * GIGABYTE, limit=1, monkeypatch=monkeypatch)
    assert [job.job_id for job in admitted] == [waiting.job_id]

def test_an_unmeasurable_machine_still_limits_how_many_jobs_run(manager: JobManager, monkeypatch) -> None:
    first, second = queue(manager, memory=9 * GIGABYTE), queue(manager, memory=9 * GIGABYTE)
    admitted = plan(manager, spare=None, limit=1, monkeypatch=monkeypatch)
    assert [job.job_id for job in admitted] == [first.job_id]
    assert manager.repo.get_job(second.job_id).admitted_at is None

def _command_for_clips(media: Path, count: int) -> List[str]:
    """Build the render command for a timeline of `count` one-second clips.

    Args:
        media: Folder holding the test media.
        count: Number of clips to put on the video track.

    Returns:
        The FFmpeg arguments the renderer would run.
    """
    asset = import_asset(str(media / "wide.mp4"))
    operations = [video_track()] + [insert(f"c{index}", asset["id"], index, index + 1) for index in range(count)]
    project = repo.get_project(build_project(operations))
    return FFmpegRenderer().build_command(project, _referenced_assets(project), "out.mp4")

def test_a_long_timeline_holds_every_decoder_to_one_thread(media: Path) -> None:
    command = _command_for_clips(media, resources.MANY_INPUTS + 1)
    # One decoder per clip is opened at once, so each is held to a single decoded picture.
    assert command.count("-i") == resources.MANY_INPUTS + 1
    assert command.count("-threads") == resources.MANY_INPUTS + 1

def test_a_short_timeline_is_left_to_decode_at_full_speed(media: Path) -> None:
    assert "-threads" not in _command_for_clips(media, 2)

@pytest.mark.parametrize(("value", "expected"), [("6", 6), ("", 4), ("nonsense", 4), ("0", 1), ("-3", 1)])
def test_a_limit_falls_back_to_its_default_when_the_setting_is_not_a_number(
    value: str, expected: int, monkeypatch
) -> None:
    monkeypatch.setenv("CLIP_MCP_MAX_DECODERS", value)
    assert resources.max_decoders() == expected

def test_a_timeline_too_long_for_a_command_line_still_renders(media: Path, tmp_path: Path) -> None:
    # Each clip adds a few hundred characters of graph, so ninety of them pass the 32767
    # characters Windows allows a whole command line.
    asset = import_asset(str(media / "wide.mp4"))
    operations = [video_track()] + [
        insert(f"c{index}", asset["id"], (index % 45) * 0.2, (index % 45) * 0.2 + 0.2) for index in range(90)
    ]
    project = repo.get_project(build_project(operations))
    output = tmp_path / "long.mp4"
    command = FFmpegRenderer().build_command(project, _referenced_assets(project), str(output))
    assert len(command[command.index("-filter_complex") + 1]) > 32767
    run_ffmpeg(command, str(tmp_path / "ffmpeg.log"), 18.0, lambda fraction: None, lambda: False)
    assert output.stat().st_size > 0
    assert len(" ".join(graph_from_file(command, str(tmp_path / "graph.txt")))) < 32767
