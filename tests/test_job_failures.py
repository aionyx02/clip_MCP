"""Tests for what a job that did not complete says ended it (#107).

The kind is decided where the cause is known — the worker that went, the
file that cannot be read, FFmpeg's own failure, the cancellation — and never
read back out of the error's words, so each test makes a job end one way and
checks the kind recorded, with the message left as it was.
"""

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app import runtime
from app.engine import renderer
from app.engine.ffmpeg import FFmpegFailed, OperationCancelled, run_ffmpeg
from app.engine.renderer import SPEC_FILE_NAME, JobManager, run_worker
from app.models.job import FailureKind, Job, JobKind, JobStatus
from app.models.media import Asset
from app.storage.repo import Repository
from app.tools import job_queue, rendering
from app.tools.catalog import import_asset
from app.tools.rendering import _read_from, render_project
from helpers import build_project, insert, video_track

@pytest.fixture
def store(tmp_path: Path) -> Repository:
    """A repository of its own, so no other job is admitted when a worker finishes."""
    return Repository(str(tmp_path / "jobs.db"))

def queued(store: Repository, tmp_path: Path, spec: Dict[str, Any], **fields) -> Job:
    """Store a queued job with its specification written, as `start_job` leaves it, without a worker.

    Args:
        store: Repository receiving the job.
        tmp_path: Folder the job's work directory goes in.
        spec: The job's specification.
        **fields: Other job fields, such as `kind` or `asset_id`.

    Returns:
        The stored job.
    """
    job = Job(work_dir=str(tmp_path / "job"), **fields)
    os.makedirs(job.work_dir, exist_ok=True)
    with open(os.path.join(job.work_dir, SPEC_FILE_NAME), "w", encoding="utf-8") as handle:
        json.dump(spec, handle)
    store.add_job(job)
    return job

def render_spec(tmp_path: Path, source: Path) -> Dict[str, Any]:
    """A render whose one FFmpeg run reads `source` and nothing else."""
    return {
        "command": ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(source), "-f", "null", "-"],
        "duration_seconds": 1.0, "pieces": [], "render_path": str(tmp_path / "out.mp4"), "inputs": [str(source)],
    }

def ended(store: Repository, job: Job) -> Job:
    """Run a queued job's worker in this process and read back how it ended."""
    run_worker(store, job.job_id)
    return store.get_job(job.job_id)

def test_ffmpeg_failing_with_its_file_there_is_a_render_failure(store: Repository, tmp_path: Path) -> None:
    source = tmp_path / "not-a-video.mp4"
    source.write_text("these are words, not a picture", encoding="utf-8")
    job = ended(store, queued(store, tmp_path, render_spec(tmp_path, source)))
    assert (job.status, job.failure_kind) == (JobStatus.FAILED, FailureKind.RENDER)
    assert job.error_message

def test_ffmpeg_failing_on_a_file_that_is_gone_is_an_input_failure(store: Repository, tmp_path: Path) -> None:
    job = ended(store, queued(store, tmp_path, render_spec(tmp_path, tmp_path / "moved away.mp4")))
    assert (job.status, job.failure_kind) == (JobStatus.FAILED, FailureKind.INPUT)
    # The message is FFmpeg's own, as it always was: only the kind is new.
    assert job.error_message and "could not read" not in job.error_message

def test_a_file_taken_away_while_the_job_reads_it_is_put_down_to_the_file(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "talk.mp4"
    source.write_bytes(b"\0")

    def taken(job, spec, context, repo) -> None:
        source.unlink()
        raise RuntimeError("the program reading it stopped")

    monkeypatch.setattr(renderer, "_run_render", taken)
    job = ended(store, queued(store, tmp_path, render_spec(tmp_path, source)))
    assert (job.failure_kind, job.error_message) == (FailureKind.INPUT, "the program reading it stopped")

def test_an_analysis_of_a_file_that_is_gone_is_an_input_failure(store: Repository, tmp_path: Path) -> None:
    store.save_asset(Asset(id="gone", path=str(tmp_path / "gone.mp4"), duration=10, has_video=True, has_audio=True))
    job = ended(store, queued(store, tmp_path, {"transcribe": False}, kind=JobKind.ANALYZE, asset_id="gone"))
    assert (job.status, job.failure_kind) == (JobStatus.FAILED, FailureKind.INPUT)
    assert job.error_message.startswith(f"could not read {tmp_path / 'gone.mp4'}")

def test_an_analysis_of_an_asset_no_longer_in_the_library_is_an_input_failure(
    store: Repository, tmp_path: Path,
) -> None:
    job = ended(store, queued(store, tmp_path, {"transcribe": False}, kind=JobKind.ANALYZE, asset_id="forgotten"))
    assert (job.failure_kind, job.error_message) == (FailureKind.INPUT, "asset forgotten not found")

def test_an_error_of_no_known_kind_is_left_without_one(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "talk.mp4"
    source.write_bytes(b"\0")

    def broken(job, spec, context, repo) -> None:
        raise RuntimeError("something nobody foresaw")

    monkeypatch.setattr(renderer, "_run_render", broken)
    job = ended(store, queued(store, tmp_path, render_spec(tmp_path, source)))
    assert (job.status, job.failure_kind, job.error_message) == (
        JobStatus.FAILED, None, "something nobody foresaw",
    )

def test_a_job_stopped_while_running_is_cancelled(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stopped(job, spec, context, repo) -> None:
        raise OperationCancelled()

    monkeypatch.setattr(renderer, "_run_render", stopped)
    job = ended(store, queued(store, tmp_path, render_spec(tmp_path, tmp_path / "talk.mp4")))
    assert (job.status, job.failure_kind, job.error_message) == (JobStatus.CANCELLED, FailureKind.CANCELLED, None)

def test_a_job_cancelled_while_waiting_is_cancelled(store: Repository, tmp_path: Path) -> None:
    job = queued(store, tmp_path, {})
    cancelled = JobManager(store).cancel_job(job.job_id, wait_seconds=0)
    assert (cancelled.status, cancelled.failure_kind) == (JobStatus.CANCELLED, FailureKind.CANCELLED)

def test_a_worker_that_stopped_reporting_is_a_worker_failure(store: Repository, tmp_path: Path) -> None:
    job = queued(store, tmp_path, {}, status=JobStatus.RUNNING)
    when = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    store._conn.execute("UPDATE jobs SET data = json_set(data, '$.updated_at', ?) WHERE id = ?", (when, job.job_id))
    failed = JobManager(store).get_job(job.job_id)
    assert (failed.failure_kind, failed.error_message) == (FailureKind.WORKER, renderer.ABANDONED)

def test_a_worker_that_could_not_start_is_a_worker_failure(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused(*args, **kwargs) -> None:
        raise OSError("no more processes")

    monkeypatch.setattr(subprocess, "Popen", refused)
    job = Job(kind=JobKind.ANALYZE, work_dir=str(tmp_path / "job"))
    JobManager(store).start_job(job, {})
    failed = store.get_job(job.job_id)
    assert failed.failure_kind == FailureKind.WORKER
    assert failed.error_message.startswith("could not start worker")

def test_a_worker_that_crashed_is_a_worker_failure(store: Repository, tmp_path: Path) -> None:
    job = queued(store, tmp_path, {})
    os.remove(os.path.join(job.work_dir, SPEC_FILE_NAME))
    with pytest.raises(FileNotFoundError):
        renderer.main([store.db_path, job.job_id])
    failed = store.get_job(job.job_id)
    assert failed.failure_kind == FailureKind.WORKER
    assert failed.error_message.startswith("worker crashed")

def test_ffmpeg_failing_raises_its_own_error(tmp_path: Path) -> None:
    progress: List[float] = []
    with pytest.raises(FFmpegFailed):
        run_ffmpeg(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(tmp_path / "nothing.mp4"), "-f", "null", "-"],
                   str(tmp_path / "ffmpeg.log"), 1.0, progress.append, lambda: False)

def test_get_job_says_what_ended_a_job_and_nothing_for_one_that_finished() -> None:
    gone = Job(kind=JobKind.RENDER, status=JobStatus.FAILED, error_message="No such file or directory",
               failure_kind=FailureKind.INPUT)
    done = Job(kind=JobKind.ANALYZE, status=JobStatus.COMPLETED, progress=1.0)
    for job in (gone, done):
        runtime.repo.add_job(job)
    listed = job_queue.get_job([gone.job_id, done.job_id])["jobs"]
    assert (listed[0]["failure_kind"], listed[0]["error_message"]) == ("input", "No such file or directory")
    assert "failure_kind" not in listed[1]

def test_a_job_reads_every_file_its_commands_open_but_not_what_it_writes_itself() -> None:
    commands = [["ffmpeg", "-i", "talk.mp4", "-i", "cache/picture.mp4", "-i", "cache/new.mp4", "out.mp4"],
                ["ffmpeg", "-i", "talk.mp4", "cache/new.mp4.part"]]
    # A picture already in the cache is read; one this job makes is not there to be read before it runs.
    assert _read_from(commands, ["cache/new.mp4.part", "cache/new.mp4"]) == ["cache/picture.mp4", "talk.mp4"]

def test_a_render_is_queued_knowing_which_files_it_reads(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 1)], width=640, height=360)
    specs: List[Dict[str, Any]] = []

    def queue(job: Job, spec: Dict[str, Any]) -> Job:
        specs.append(spec)
        return job

    monkeypatch.setattr(runtime.job_manager, "start_job", queue)
    render_project(project)
    assert specs[0]["inputs"] == [runtime.repo.get_assets([wide])[wide].path]

def test_playback_is_prepared_knowing_which_files_it_reads(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 1)], width=640, height=360)
    specs: List[Dict[str, Any]] = []

    def queue(job: Job, spec: Dict[str, Any]) -> Job:
        specs.append(spec)
        return job

    monkeypatch.setattr(rendering, "PREPARE_PLAYBACK", True)
    monkeypatch.setattr(rendering, "_make_room", lambda: None)
    monkeypatch.setattr(rendering, "_being_prepared", set)
    # Whatever an earlier test left in the shared folder, the copy to play from is made again.
    monkeypatch.setattr(rendering.playback, "missing", lambda project, assets, present: [wide])
    monkeypatch.setattr(runtime.job_manager, "start_job", queue)
    rendering.prepare_playback(runtime.repo.get_project(project))
    # The copy to play from, and the loudness measured from a mix the job writes and reads back itself.
    assert [spec["steps"][-1]["stage"] for spec in specs] == ["making a copy to play from", "measuring loudness"]
    assert [spec["inputs"] for spec in specs] == [[runtime.repo.get_assets([wide])[wide].path]] * 2
