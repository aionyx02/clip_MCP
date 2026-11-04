"""Tests for telling a quiet worker from a gone one, and for what a gone one leaves behind (#232).

A long file's analysis failed at 88% with only "worker stopped reporting",
while its worker was alive and went on to finish it: the job was judged by
how long it had been quiet, and nothing else. A worker is now judged gone
only when its process is, and the failure says where it stopped, what the
worker last wrote, and what to do next.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.engine import renderer
from app.engine.renderer import SPEC_FILE_NAME, JobManager, run_worker
from app.models.job import FailureKind, Job, JobKind, JobStatus
from app.storage.repo import Repository


@pytest.fixture
def store(tmp_path: Path) -> Repository:
    """A repository of its own."""
    return Repository(str(tmp_path / "jobs.db"))


def quiet_for(store: Repository, job: Job, seconds: float) -> None:
    """Make a job look as if nothing has written to it for `seconds`."""
    when = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")
    store._conn.execute("UPDATE jobs SET data = json_set(data, '$.updated_at', ?) WHERE id = ?", (when, job.job_id))


def running(store: Repository, tmp_path: Path, **fields) -> Job:
    """Store a running analysis, as its worker leaves it partway through."""
    job = Job(kind=JobKind.ANALYZE, status=JobStatus.RUNNING, work_dir=str(tmp_path / "job"),
              admitted_at=datetime.now(timezone.utc), **fields)
    os.makedirs(job.work_dir, exist_ok=True)
    store.add_job(job)
    return job


def a_gone_pid() -> int:
    """The ID of a process that has ended."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_a_quiet_worker_whose_process_is_alive_is_not_failed(store: Repository, tmp_path: Path) -> None:
    # The analysis in #232 was failed while its worker was still listening, and finished it after.
    job = running(store, tmp_path, worker_pid=os.getpid(), progress=0.88, stage="telling the voices apart")
    quiet_for(store, job, renderer.STALE_AFTER.total_seconds() + 60)
    assert JobManager(store).get_job(job.job_id).status == JobStatus.RUNNING


def test_a_live_worker_quiet_for_too_long_is_given_up_on(store: Repository, tmp_path: Path) -> None:
    job = running(store, tmp_path, worker_pid=os.getpid(), progress=0.5, stage="transcribing speech")
    quiet_for(store, job, renderer.HUNG_AFTER.total_seconds() + 60)
    failed = JobManager(store).get_job(job.job_id)
    assert (failed.status, failed.failure_kind) == (JobStatus.FAILED, FailureKind.WORKER)
    assert "still running" in failed.error_message
    assert "transcribing speech at 50%" in failed.error_message


def test_a_worker_whose_process_is_gone_says_where_it_stopped_and_what_it_last_wrote(
    store: Repository, tmp_path: Path,
) -> None:
    log = tmp_path / "worker.log"
    log.write_text(
        "[ WARN:0@0.079] global net_impl_backend.cpp:345 noise\n"
        "Fatal Python error: Segmentation fault\n"
        "  File \"diarize.py\", line 249 in find_speakers\n",
        encoding="utf-8",
    )
    job = running(store, tmp_path, worker_pid=a_gone_pid(), worker_log=str(log),
                  progress=0.882, stage="telling the voices apart")
    quiet_for(store, job, renderer.STALE_AFTER.total_seconds() + 1)
    failed = JobManager(store).get_job(job.job_id)
    assert (failed.status, failed.failure_kind) == (JobStatus.FAILED, FailureKind.WORKER)
    message = failed.error_message
    assert message.startswith(renderer.ABANDONED)
    assert "telling the voices apart at 88%" in message
    assert "process is gone" in message
    assert "Segmentation fault" in message and "line 249 in find_speakers" in message
    # OpenCV's warnings are in every analysis's log, and would bury the line that says why.
    assert "WARN" not in message
    # What the AI can do about it.
    assert "transcribe false" in message and "once it has finished" in message


def test_a_worker_from_before_pids_were_kept_is_judged_by_quiet_alone(store: Repository, tmp_path: Path) -> None:
    job = running(store, tmp_path)
    quiet_for(store, job, renderer.STALE_AFTER.total_seconds() + 1)
    failed = JobManager(store).get_job(job.job_id)
    assert (failed.status, failed.failure_kind) == (JobStatus.FAILED, FailureKind.WORKER)
    assert failed.error_message.startswith(renderer.ABANDONED)


def test_a_gone_render_is_told_to_render_again_not_to_skip_speech(store: Repository, tmp_path: Path) -> None:
    job = running(store, tmp_path, worker_pid=a_gone_pid(), progress=0.4, stage="rendering")
    store.update_job(job.job_id, lambda current: setattr(current, "kind", JobKind.RENDER))
    quiet_for(store, job, renderer.STALE_AFTER.total_seconds() + 1)
    message = JobManager(store).get_job(job.job_id).error_message
    assert "rendering at 40%" in message and "transcribe" not in message


def test_the_worker_records_its_process_and_log_on_the_job(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = Job(kind=JobKind.RENDER, work_dir=str(tmp_path / "job"))
    os.makedirs(job.work_dir)
    with open(os.path.join(job.work_dir, SPEC_FILE_NAME), "w", encoding="utf-8") as handle:
        json.dump({}, handle)
    store.add_job(job)
    seen = {}

    def render(current, spec, context, repo) -> None:
        seen.update(store.get_job(current.job_id).model_dump(include={"worker_pid", "worker_log"}))

    monkeypatch.setattr(renderer, "_run_render", render)
    run_worker(store, job.job_id, log_path=str(tmp_path / "first" / "worker.log"))
    assert seen == {"worker_pid": os.getpid(), "worker_log": str(tmp_path / "first" / "worker.log")}


def test_the_voices_stage_is_named_before_the_engine_goes_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    # The voice separation's first pass holds Python's lock for most of a minute on a 13-minute
    # file, before its first progress callback. Named only then, the job read as the stage before.
    from app.engine import diarize

    seen = []

    class Engine:
        def process(self, samples, callback):
            seen.append("process")
            raise RuntimeError("stop here")

    monkeypatch.setattr(diarize, "_engine", lambda speakers: Engine())
    monkeypatch.setattr(diarize, "read_samples", lambda path, ffmpeg_bin: [0.0] * 16000)
    with pytest.raises(RuntimeError, match="stop here"):
        diarize.find_speakers("talk.mp4", lambda fraction: seen.append(fraction), lambda: False)
    assert seen == [0.0, "process"]


def test_a_new_stage_is_on_the_job_before_the_next_heartbeat(store: Repository, tmp_path: Path) -> None:
    job = running(store, tmp_path)
    context = renderer.JobContext(store, job.job_id)
    # Not entered: no heartbeat runs, so only the report itself can have written it.
    context.report(0.88, "telling the voices apart")
    assert store.get_job(job.job_id).stage == "telling the voices apart"

@pytest.mark.skipif(not os.path.isdir("/proc"), reason="a zombie is told by its /proc state, which only Linux has")
def test_a_worker_that_ended_and_was_never_waited_on_is_gone() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    # Ended, not waited on: a zombie, which answers kill(0) as if it were running.
    deadline = datetime.now() + timedelta(seconds=10)
    while datetime.now() < deadline and not Path(f"/proc/{child.pid}/stat").read_text().split(")")[-1].split()[0] == "Z":
        pass
    assert not renderer._process_alive(child.pid)
