"""Tests for one worker carrying on with the next analysis (#110).

An analysis loads a speech model of a few gigabytes, so analysing a folder
one file per worker loaded it again for every file. The worker that finishes
an analysis now keeps the next admitted analysis for itself, and the models
it loaded are kept in the process.
"""

import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.engine import diarize, models, renderer, resources
from app.engine.renderer import SPEC_FILE_NAME, JobManager, run_worker
from app.models.job import FailureKind, Job, JobKind, JobStatus
from app.storage.repo import Repository


def store(tmp_path: Path) -> Repository:
    """A repository of its own, so no other job is admitted when a worker finishes.

    Args:
        tmp_path: Folder the database goes in.

    Returns:
        The repository.
    """
    return Repository(str(tmp_path / "jobs.db"))


def queued(repo: Repository, tmp_path: Path, kind: JobKind = JobKind.ANALYZE, **fields) -> Job:
    """Store a queued job with its specification written, without a worker.

    Args:
        repo: Repository receiving the job.
        tmp_path: Folder the job's work directory goes in.
        kind: What the job does.
        **fields: Other job fields, such as `asset_id`.

    Returns:
        The stored job.
    """
    job = Job(kind=kind, **fields)
    job.work_dir = str(tmp_path / job.job_id)
    os.makedirs(job.work_dir, exist_ok=True)
    with open(os.path.join(job.work_dir, SPEC_FILE_NAME), "w", encoding="utf-8") as handle:
        json.dump({"transcribe": False}, handle)
    repo.add_job(job)
    return job


def room_for(monkeypatch: pytest.MonkeyPatch, jobs: int = 4) -> None:
    """Pretend the machine has room for several jobs and cannot be measured.

    Args:
        monkeypatch: Fixture used to stand in for the machine.
        jobs: How many jobs may run at once.
    """
    monkeypatch.setattr(resources, "spare_bytes", lambda: None)
    monkeypatch.setattr(resources, "max_concurrent_jobs", lambda: jobs)


def test_an_analysis_worker_keeps_the_next_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = store(tmp_path)
    first = queued(repo, tmp_path)
    second = queued(repo, tmp_path)
    monkeypatch.setattr(renderer, "_run_analysis", lambda *args, **kwargs: None)
    launched: List[str] = []
    monkeypatch.setattr(JobManager, "_launch_worker", lambda self, job: launched.append(job.job_id))
    room_for(monkeypatch)

    next_job = run_worker(repo, first.job_id)

    assert next_job == second.job_id
    # The second was admitted like any other job, but no worker was started for it.
    assert launched == []
    assert repo.get_job(second.job_id).admitted_at is not None


def test_a_render_waiting_behind_an_analysis_is_launched_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = store(tmp_path)
    analysis = queued(repo, tmp_path)
    render = queued(repo, tmp_path, kind=JobKind.RENDER)
    monkeypatch.setattr(renderer, "_run_analysis", lambda *args, **kwargs: None)
    launched: List[str] = []
    monkeypatch.setattr(JobManager, "_launch_worker", lambda self, job: launched.append(job.job_id))
    room_for(monkeypatch)

    assert run_worker(repo, analysis.job_id) is None
    assert launched == [render.job_id]


def test_a_render_worker_hands_on_the_analysis_it_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = store(tmp_path)
    render = queued(repo, tmp_path, kind=JobKind.RENDER)
    analysis = queued(repo, tmp_path)
    monkeypatch.setattr(renderer, "_run_render", lambda *args, **kwargs: None)
    launched: List[str] = []
    monkeypatch.setattr(JobManager, "_launch_worker", lambda self, job: launched.append(job.job_id))
    room_for(monkeypatch)

    assert run_worker(repo, render.job_id) is None
    assert launched == [analysis.job_id]


def test_main_runs_the_analysis_a_worker_keeps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = store(tmp_path)
    calls: List[str] = []

    def fake(repo: Repository, job_id: str) -> Any:
        """Stand in for a worker, handing the second job to the same process."""
        calls.append(job_id)
        return "b" if job_id == "a" else None

    monkeypatch.setattr(models, "_keep_speech_model", False)
    monkeypatch.setattr(renderer, "run_worker", fake)
    renderer.main([repo.db_path, "a"])
    assert calls == ["a", "b"]
    # A worker keeps its speech model between the analyses it runs.
    assert models._keep_speech_model


def test_a_crash_in_a_kept_analysis_fails_that_job_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = store(tmp_path)
    second = Job(job_id="b")
    repo.add_job(second)

    def fake(repo: Repository, job_id: str) -> Any:
        """Stand in for a worker that runs one job and crashes on the next."""
        if job_id == "a":
            return "b"
        raise RuntimeError("boom")

    monkeypatch.setattr(models, "_keep_speech_model", False)
    monkeypatch.setattr(renderer, "run_worker", fake)
    with pytest.raises(RuntimeError, match="boom"):
        renderer.main([repo.db_path, "a"])
    failed = repo.get_job("b")
    assert (failed.status, failed.failure_kind, failed.error_message) == (
        JobStatus.FAILED, FailureKind.WORKER, "worker crashed: boom",
    )


def test_the_speech_model_is_built_once_for_the_same_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: List[Dict[str, Any]] = []

    class FakeWhisperModel:
        """Stand in for faster-whisper's model, recording each construction."""

        def __init__(self, name: str, device: str, compute_type: str, download_root: str):
            built.append({
                "name": name, "device": device,
                "compute_type": compute_type, "download_root": download_root,
            })

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(models, "_speech_model", None)
    monkeypatch.setattr(models, "_speech_model_key", None)
    monkeypatch.setattr(models, "_keep_speech_model", True)

    first = models.speech_model("large-v3", "cuda", "models")
    again = models.speech_model("large-v3", "cuda", "models")

    assert again is first
    assert len(built) == 1
    assert built[0]["compute_type"] == "float16"

    other = models.speech_model("small", "cpu", "models")

    assert other is not first
    assert len(built) == 2
    assert built[1]["compute_type"] == "int8"
    # Only the newest model is kept.
    assert models._speech_model is other
    assert models._speech_model_key == ("small", "cpu", "models")


def test_outside_a_worker_the_speech_model_is_not_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # The server hears stretches again too; it must not hold the weights after that.
    built: List[str] = []

    class FakeWhisperModel:
        """Stand in for faster-whisper's model, recording each construction."""

        def __init__(self, name: str, device: str, compute_type: str, download_root: str):
            built.append(name)

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(models, "_speech_model", None)
    monkeypatch.setattr(models, "_speech_model_key", None)
    monkeypatch.setattr(models, "_keep_speech_model", False)

    first = models.speech_model("large-v3", "cuda", "models")
    again = models.speech_model("large-v3", "cuda", "models")

    assert again is not first
    assert built == ["large-v3", "large-v3"]
    assert models._speech_model is None


def test_the_diarizer_is_built_once_for_the_same_speaker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: List[Any] = []

    class Config:
        """Stand in for every sherpa-onnx configuration object."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def validate(self) -> bool:
            """Accept every configuration, as the real one would here."""
            return True

    class Engine:
        """Stand in for the diarization engine, recording each construction."""

        def __init__(self, config):
            built.append(config)

    module = types.ModuleType("sherpa_onnx")
    module.OfflineSpeakerDiarizationConfig = Config
    module.OfflineSpeakerSegmentationModelConfig = Config
    module.OfflineSpeakerSegmentationPyannoteModelConfig = Config
    module.SpeakerEmbeddingExtractorConfig = Config
    module.FastClusteringConfig = Config
    module.OfflineSpeakerDiarization = Engine
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    monkeypatch.setattr(diarize.models, "ensure", lambda model: "model.onnx")
    monkeypatch.setattr(diarize, "_engine_cache", None)

    first = diarize._engine(2)
    again = diarize._engine(2)

    assert again is first
    assert len(built) == 1

    other = diarize._engine(None)

    assert other is not first
    assert len(built) == 2
    assert diarize._engine_cache[0] is None
    assert diarize._engine_cache[1] is other
