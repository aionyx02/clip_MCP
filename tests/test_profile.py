"""Tests for timing each stage of a job and recording the memory it had in use."""

import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

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
    """A clock with a fake time and fake memory readers, its sampler far away in the future."""
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
