"""Tests for timing each stage of a job and recording the memory it had in use."""

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from app import cli
from app.engine import machine, profile, renderer
from app.engine.profile import StageClock, stage_key, summarise
from app.engine.renderer import JobContext
from app.models.job import Job, JobKind, JobStatus
from app.models.media import Asset
from app.storage.repo import Repository

MIB = 1024 ** 2
GIB = 1024 ** 3

def stage_clock(
    now: List[float],
    memory=lambda: None,
    gpu=lambda: None,
) -> StageClock:
    """A clock with a fake time and fake readers.

    Its sampler takes one reading as it starts, for the card's baseline, then
    none for an hour; tests that count readings or need the baseline in place
    take their own with `_sample`.
    """
    return StageClock(sample_seconds=3600.0, clock=lambda: now[0], memory=memory, gpu=gpu)

def entry(stage: str, seconds: float, memory: Optional[int] = None, gpu: Optional[int] = None) -> Dict[str, Any]:
    """One profile entry as a job stores it."""
    return {"stage": stage, "seconds": seconds, "memory_rise_mb": memory, "gpu_rise_mb": gpu}

def finished(kind: JobKind, entries: List[Dict[str, Any]], asset_id: Optional[str] = None) -> Job:
    """A completed job carrying a profile."""
    return Job(kind=kind, status=JobStatus.COMPLETED, profile=entries, asset_id=asset_id)

def asset(asset_id: str, duration: Optional[Decimal]) -> Asset:
    """An asset of a known length, for the real-time factor."""
    return Asset(id=asset_id, path=f"/footage/{asset_id}.mp4", duration=duration, has_video=True, has_audio=True)

# --- the clock ---------------------------------------------------------------------------

def test_each_stage_is_timed_and_keeps_its_own_peak_rise() -> None:
    now = [0.0]
    memory = [1000 * MIB]
    gpu = [100 * MIB]
    clock = stage_clock(now, lambda: memory[0], lambda: gpu[0])
    clock.start()
    clock._sample()
    now[0] = 1.0
    clock.mark("transcribing")
    memory[0], gpu[0] = 1600 * MIB, 400 * MIB
    clock._sample()
    memory[0], gpu[0] = 1200 * MIB, 200 * MIB
    now[0] = 3.5
    clock.mark("detecting scenes")
    clock._sample()
    now[0] = 5.0
    assert clock.stop() == [
        entry("transcribing", 2.5, 600, 300),
        entry("detecting scenes", 1.5, 200, 100),
    ]

def test_a_stage_reported_twice_in_a_row_is_one_entry() -> None:
    now = [0.0]
    clock = stage_clock(now, lambda: 100 * MIB)
    clock.start()
    now[0] = 1.0
    clock.mark("reading")
    now[0] = 2.0
    clock.mark("reading")
    now[0] = 3.0
    clock.mark("writing")
    entries = clock.stop()
    assert [one["stage"] for one in entries] == ["reading", "writing"]
    assert entries[0]["seconds"] == 2.0

def test_a_stage_entered_again_later_is_a_new_entry() -> None:
    now = [0.0]
    clock = stage_clock(now, lambda: 100 * MIB)
    clock.start()
    now[0] = 1.0
    clock.mark("a")
    now[0] = 2.0
    clock.mark("b")
    now[0] = 3.0
    clock.mark("a")
    now[0] = 4.0
    entries = clock.stop()
    assert [one["stage"] for one in entries] == ["a", "b", "a"]
    assert [one["seconds"] for one in entries] == [1.0, 1.0, 1.0]

def test_a_stage_that_uses_less_memory_than_the_baseline_has_no_negative_rise() -> None:
    now = [0.0]
    memory = [2000 * MIB]
    clock = stage_clock(now, lambda: memory[0], lambda: None)
    clock.start()
    now[0] = 1.0
    clock.mark("reading")
    memory[0] = 500 * MIB
    clock._sample()
    now[0] = 2.0
    assert clock.stop()[0]["memory_rise_mb"] == 0

def test_a_reading_that_is_not_available_stays_none() -> None:
    now = [0.0]
    clock = stage_clock(now, lambda: None, lambda: None)
    clock.start()
    now[0] = 1.0
    clock.mark("reading")
    now[0] = 2.0
    entries = clock.stop()
    assert entries[0]["memory_rise_mb"] is None
    assert entries[0]["gpu_rise_mb"] is None

def test_a_card_that_cannot_be_read_is_not_asked_again() -> None:
    now = [0.0]
    calls = [0]

    def gpu() -> Optional[int]:
        calls[0] += 1
        return None

    clock = stage_clock(now, lambda: 100 * MIB, gpu)
    clock.start()
    clock._sample()
    clock._sample()
    now[0] = 1.0
    clock.mark("reading")
    clock._sample()
    clock.stop()
    assert calls[0] == 1

def test_a_stage_never_reported_leaves_no_entry() -> None:
    now = [0.0]
    clock = stage_clock(now, lambda: 100 * MIB)
    clock.start()
    now[0] = 2.0
    assert clock.stop() == []

class _StopAfter:
    """A stop event that answers False a set number of waits, then True."""

    def __init__(self, waits: int):
        self._left = waits

    def wait(self, seconds: float) -> bool:
        if self._left > 0:
            self._left -= 1
            return False
        return True

    def set(self) -> None:
        self._left = 0

    def is_set(self) -> bool:
        return False

def test_the_card_is_read_every_gpu_seconds() -> None:
    reads = {"memory": 0, "card": 0}

    def memory() -> int:
        reads["memory"] += 1
        return 100 * MIB

    def card() -> int:
        reads["card"] += 1
        return 50 * MIB * reads["card"]

    clock = StageClock(sample_seconds=0.25, gpu_seconds=1.0, memory=memory, gpu=card)
    clock._running = True
    # The clock's own event would stop at once: the sampler goes by the event it was handed.
    clock._stop = _StopAfter(0)
    clock._sample_forever(_StopAfter(12))
    # One reading of each as the sampler starts, the card's baseline among them, then memory every tick
    # and the card every fourth.
    assert reads["memory"] == 13
    assert reads["card"] == 4
    assert clock._baseline_gpu == 50 * MIB

def test_a_card_reading_in_progress_when_the_stage_changes_goes_to_neither_stage() -> None:
    now = [0.0]
    clock = stage_clock(now, lambda: 100 * MIB, lambda: 100 * MIB)
    clock.start()
    clock._sample()
    clock.mark("loading the model")

    def card_read_across_a_change() -> int:
        now[0] = 1.0
        clock.mark("transcribing")
        return 2000 * MIB

    clock._gpu = card_read_across_a_change
    clock._sample()
    now[0] = 2.0
    entries = clock.stop()
    assert [item["gpu_rise_mb"] for item in entries] == [None, None]

def test_a_stage_shorter_than_a_sample_still_has_the_reading_taken_as_it_began() -> None:
    now = [0.0]
    memory = [100 * MIB]
    clock = stage_clock(now, lambda: memory[0])
    clock.start()
    memory[0] = 600 * MIB
    clock.mark("quick")
    memory[0] = 100 * MIB
    now[0] = 0.1
    entries = clock.stop()
    assert entries[0]["memory_rise_mb"] == 500

def test_a_restarted_clock_does_not_revive_a_sampler_stop_gave_up_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile, "STOP_WAIT_SECONDS", 0.05)
    hung, release = threading.Event(), threading.Event()
    reads = {"old": 0, "new": 0}
    run = ["old"]

    def card() -> Optional[int]:
        reads[run[0]] += 1
        if run[0] == "old":
            hung.set()
            release.wait(5)
            # The old run's card reading fails late; the new run must not stop reading its card over it.
            return None
        return 100 * MIB

    clock = StageClock(sample_seconds=0.01, gpu_seconds=0.01, memory=lambda: 100 * MIB, gpu=card)
    clock.start()
    assert hung.wait(5)
    old = clock._thread
    clock.stop()
    run[0] = "new"
    clock.start()
    release.set()
    old.join(5)
    assert not old.is_alive()
    assert reads["old"] == 1
    waited = time.monotonic() + 5
    while clock._baseline_gpu is None and time.monotonic() < waited:
        time.sleep(0.001)
    assert clock._baseline_gpu == 100 * MIB
    assert clock._gpu_readable
    clock.stop()

def test_a_first_card_reading_across_the_first_stage_change_is_still_the_baseline() -> None:
    now = [0.0]
    clock = stage_clock(now, lambda: 100 * MIB)

    changes = [0]

    def card_read_across_a_change() -> int:
        # Every reading spans a change of stage, as the first one in a real job usually does.
        changes[0] += 1
        clock.mark(f"stage {changes[0]}")
        return 300 * MIB

    clock._gpu = card_read_across_a_change
    clock.start()
    clock._sample()
    assert clock._baseline_gpu == 300 * MIB
    clock.stop()

def test_a_stage_reported_again_reads_no_memory() -> None:
    reads = [0]

    def memory() -> int:
        reads[0] += 1
        return 100 * MIB

    clock = stage_clock([0.0], memory)
    clock.start()
    clock.mark("transcribing")
    before = reads[0]
    clock.mark("transcribing")
    assert reads[0] == before
    clock.stop()

def test_a_stopped_clock_takes_no_more_stages() -> None:
    now = [0.0]
    clock = stage_clock(now)
    clock.start()
    clock.mark("a")
    now[0] = 1.0
    entries = clock.stop()
    clock.mark("b")
    assert clock._stage is None
    assert [item["stage"] for item in clock.stop()] == [item["stage"] for item in entries] == ["a"]

def test_a_restarted_clock_starts_with_no_stages() -> None:
    clock = stage_clock([0.0])
    clock.start()
    clock.mark("a")
    clock.stop()
    clock.start()
    assert clock.stop() == []

def test_a_card_rise_is_none_for_a_stage_the_card_was_not_read_in() -> None:
    now = [0.0]
    gpu = [100 * MIB]
    clock = stage_clock(now, lambda: 100 * MIB, lambda: gpu[0])
    clock.start()
    now[0] = 1.0
    clock.mark("quick")
    now[0] = 2.0
    clock.mark("long")
    clock._sample()
    gpu[0] = 300 * MIB
    clock._sample()
    now[0] = 3.0
    entries = clock.stop()
    assert entries[0]["gpu_rise_mb"] is None
    assert entries[1]["gpu_rise_mb"] == 200

def test_stop_does_not_wait_past_the_stop_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_card() -> Optional[int]:
        entered.set()
        release.wait(5)
        return None

    clock = StageClock(sample_seconds=0.01, gpu_seconds=0.01, memory=lambda: 100 * MIB, gpu=blocking_card)
    monkeypatch.setattr(profile, "STOP_WAIT_SECONDS", 0.05)
    clock.start()
    try:
        assert entered.wait(2.0)
        began = time.monotonic()
        clock.stop()
        elapsed = time.monotonic() - began
        assert elapsed < 1.0
    finally:
        release.set()

# --- grouping ----------------------------------------------------------------------------

def test_stage_keys_group_a_counter() -> None:
    assert stage_key("rendering picture 3 of 12") == "rendering picture # of #"
    assert stage_key("rendering picture 4 of 12") == stage_key("rendering picture 3 of 12")
    assert stage_key("transcribing") == "transcribing"

# --- the readings ------------------------------------------------------------------------

def test_machine_memory_in_use_is_total_less_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile.resources, "memory_status", lambda: (4 * GIB, 16 * GIB))
    assert profile._machine_in_use() == 12 * GIB

def test_memory_that_cannot_be_measured_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile.resources, "memory_status", lambda: None)
    assert profile._machine_in_use() is None

def test_the_card_reading_is_mebibytes_as_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(machine, "_nvidia_smi", lambda query: "1234")
    assert profile._gpu_in_use() == 1234 * MIB

def test_a_card_that_says_nothing_has_no_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(machine, "_nvidia_smi", lambda query: None)
    assert profile._gpu_in_use() is None

@pytest.mark.parametrize("answer", ["[N/A]", "[Not Supported]"])
def test_a_card_that_answers_something_not_numeric_has_no_reading(
    monkeypatch: pytest.MonkeyPatch, answer: str,
) -> None:
    monkeypatch.setattr(machine, "_nvidia_smi", lambda query: answer)
    assert profile._gpu_in_use() is None

# --- the summary -------------------------------------------------------------------------

def summary() -> str:
    """Two analyze jobs and one render job, with one unfinished and one unprofiled job beside them."""
    jobs = [
        finished(JobKind.ANALYZE, [
            entry("transcription", 60.0, 512, 2048),
            entry("detecting scenes", 20.0, 128, None),
        ], asset_id="a"),
        finished(JobKind.ANALYZE, [entry("transcription", 30.0, 300, 1024)], asset_id="b"),
        finished(JobKind.RENDER, [
            entry("rendering picture 1 of 2", 10.0, 256, None),
            entry("rendering picture 2 of 2", 20.0, 256, None),
        ]),
        Job(kind=JobKind.ANALYZE, status=JobStatus.RUNNING, asset_id="c",
            profile=[entry("never finished", 99.0)]),
        Job(kind=JobKind.RENDER, status=JobStatus.COMPLETED, profile=None),
    ]
    assets = {"a": asset("a", Decimal("200")), "b": asset("b", Decimal("100")), "c": asset("c", None)}
    return summarise(jobs, assets)

def row(text: str, starts_with: str) -> List[str]:
    """The fields of the table row whose stage starts with a prefix."""
    return next(line for line in text.splitlines() if line.strip().startswith(starts_with)).split()

def test_the_summary_has_one_block_per_kind_and_leaves_the_rest_out() -> None:
    text = summary()
    assert "analyze: 2 completed jobs" in text
    assert "render: 1 completed job" in text
    assert "never finished" not in text

def test_a_stage_row_counts_jobs_totals_shares_and_medians() -> None:
    fields = row(summary(), "transcription")
    assert fields[:7] == ["transcription", "2", "90.000", "81.8%", "45.000", "512", "2048"]

def test_stages_are_ordered_by_total_seconds_largest_first() -> None:
    text = summary()
    assert text.index("transcription") < text.index("detecting scenes")

def test_a_stage_with_no_card_reading_shows_a_dash() -> None:
    assert row(summary(), "detecting scenes")[-1] == "-"

def test_a_counter_stage_groups_its_pictures_into_one_row() -> None:
    fields = row(summary(), "rendering picture")
    assert fields[-6:] == ["1", "30.000", "100.0%", "15.000", "256", "-"]

def test_analyze_jobs_report_the_real_time_factor() -> None:
    text = summary()
    assert "real-time factor: 0.37x (2 jobs, 300.0 s of footage)" in text

def test_analyze_jobs_without_a_duration_say_the_factor_cannot_be_worked_out() -> None:
    jobs = [finished(JobKind.ANALYZE, [entry("transcription", 10.0)], asset_id="a")]
    text = summarise(jobs, {"a": asset("a", None)})
    assert "real-time factor: - (no analyzed asset has a duration)" in text

def test_analyze_jobs_with_no_footage_say_the_factor_cannot_be_worked_out() -> None:
    jobs = [finished(JobKind.ANALYZE, [entry("transcription", 10.0)], asset_id="a")]
    text = summarise(jobs, {"a": asset("a", Decimal("0"))})
    assert "real-time factor: - (no analyzed asset has a duration)" in text

def test_nothing_to_summarise_says_so() -> None:
    nothing = [Job(kind=JobKind.RENDER, status=JobStatus.COMPLETED)]
    assert summarise(nothing, {}) == "No completed job has a profile yet."

# --- a job's own context -----------------------------------------------------------------

def test_a_job_context_records_its_stages_on_the_job(tmp_path: Path) -> None:
    repo = Repository(str(tmp_path / "jobs.db"))
    job = Job(work_dir=str(tmp_path / "work"))
    os.makedirs(job.work_dir, exist_ok=True)
    repo.add_job(job)
    with JobContext(repo, job.job_id) as context:
        context.report(0.1, "reading the file")
        context.report(0.5, "transcribing")
    stored = repo.get_job(job.job_id)
    assert stored is not None and stored.profile is not None
    assert [one["stage"] for one in stored.profile] == ["reading the file", "transcribing"]

def test_a_clock_that_cannot_start_leaves_the_profile_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class BrokenClock:
        """A clock whose start fails, as a machine without the means to read memory might."""

        def start(self) -> None:
            raise RuntimeError("no clock")

        def mark(self, stage: str) -> None:
            raise AssertionError("a clock that did not start is never marked")

        def stop(self) -> List[Dict[str, Any]]:
            raise AssertionError("a clock that did not start is never stopped")

    monkeypatch.setattr(renderer, "StageClock", BrokenClock)
    repo = Repository(str(tmp_path / "broken.db"))
    job = Job()
    repo.add_job(job)
    with JobContext(repo, job.job_id) as context:
        context.report(0.5, "transcribing")
    stored = repo.get_job(job.job_id)
    assert stored is not None
    assert stored.profile is None

class _MarkFails:
    """A clock whose mark fails while the job runs."""

    def start(self) -> None:
        pass

    def mark(self, stage: str) -> None:
        raise RuntimeError("no mark")

    def stop(self) -> List[Dict[str, Any]]:
        return []

class _StopFails:
    """A clock whose stop fails when the job is recorded."""

    def start(self) -> None:
        pass

    def mark(self, stage: str) -> None:
        pass

    def stop(self) -> List[Dict[str, Any]]:
        raise RuntimeError("no stop")

def test_a_clock_whose_mark_fails_never_fails_the_job(tmp_path: Path) -> None:
    repo = Repository(str(tmp_path / "mark.db"))
    job = Job()
    repo.add_job(job)
    context = JobContext(repo, job.job_id)
    context._clock = _MarkFails()
    with context as running:
        running.report(0.5, "transcribing")
    assert repo.get_job(job.job_id) is not None

def test_a_clock_whose_stop_fails_never_fails_the_job(tmp_path: Path) -> None:
    repo = Repository(str(tmp_path / "stop.db"))
    job = Job()
    repo.add_job(job)
    context = JobContext(repo, job.job_id)
    context._clock = _StopFails()
    with context as running:
        running.report(0.5, "transcribing")
    stored = repo.get_job(job.job_id)
    assert stored is not None
    assert stored.profile is None

def test_a_profile_that_cannot_be_written_never_fails_the_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repo = Repository(str(tmp_path / "write.db"))
    job = Job()
    repo.add_job(job)
    context = JobContext(repo, job.job_id)
    context.__enter__()
    context.report(0.5, "transcribing")
    original = repo.update_job

    def failing(job_id: str, change: Any) -> Any:
        if getattr(change, "__qualname__", "").endswith("_record_profile.<locals>.write"):
            raise RuntimeError("disk full")
        return original(job_id, change)

    monkeypatch.setattr(repo, "update_job", failing)
    context.__exit__(None, None, None)
    stored = repo.get_job(job.job_id)
    assert stored is not None
    assert stored.profile is None

# --- the command -------------------------------------------------------------------------

def test_the_profile_command_prints_a_summary(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["profile", "--days", "30"])
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "No completed job has a profile yet." in out or "total s" in out

def test_the_profile_command_reads_as_many_days_back_as_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = []

    def jobs_to_show(self, since):
        asked.append(since)
        return [], []

    monkeypatch.setattr(Repository, "jobs_to_show", jobs_to_show)
    with pytest.raises(SystemExit):
        cli.main(["profile", "--days", "5"])
    back = datetime.now(timezone.utc) - asked[0]
    assert timedelta(days=5) <= back < timedelta(days=5, minutes=1)

@pytest.mark.parametrize("days", ["-3", "0", "99999999", "a week"])
def test_the_profile_command_refuses_days_it_cannot_read(days: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["profile", "--days", days])
    assert caught.value.code == 2
    assert "a whole number of days from 1 to" in capsys.readouterr().err

def test_the_profile_command_makes_no_database_in_an_empty_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.workspace, "workspace_dir", lambda: str(tmp_path))
    with pytest.raises(SystemExit) as caught:
        cli.main(["profile"])
    assert caught.value.code == 0
    assert "No completed job has a profile yet." in capsys.readouterr().out
    assert not (tmp_path / "clip_mcp.db").exists()

def test_a_restarted_clock_reads_its_card_afresh() -> None:
    clock = stage_clock([0.0], lambda: 100 * MIB, lambda: 50 * MIB)
    clock.start()
    clock._sample()
    clock._gpu = lambda: None
    clock._sample()
    assert clock._baseline_gpu == 50 * MIB
    assert not clock._gpu_readable
    clock.stop()
    clock._gpu = lambda: 200 * MIB
    clock.start()
    clock._sample()
    assert clock._gpu_readable
    assert clock._baseline_gpu == 200 * MIB
    clock.stop()

def test_stages_of_the_same_total_are_listed_by_name() -> None:
    jobs = [finished(JobKind.RENDER, [entry("b stage", 5.0), entry("a stage", 5.0)])]
    text = summarise(jobs, {})
    assert text.index("a stage") < text.index("b stage")
