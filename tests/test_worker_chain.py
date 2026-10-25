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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.engine import analysis, diarize, models, relisten, renderer, resources
from app.engine.renderer import SPEC_FILE_NAME, InputUnavailable, JobManager, run_worker
from app.models.job import FailureKind, Job, JobKind, JobStatus
from app.models.media import Asset
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


@pytest.mark.parametrize("error, kept", [
    (RuntimeError("CUDA error: an illegal memory access was encountered"), False),
    (InputUnavailable("the file is gone"), True),
])
def test_an_analysis_that_failed_for_a_reason_not_known_hands_the_next_to_a_fresh_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, kept: bool,
) -> None:
    repo = store(tmp_path)
    first = queued(repo, tmp_path)
    second = queued(repo, tmp_path)

    def fail(*args, **kwargs) -> None:
        raise error

    monkeypatch.setattr(renderer, "_run_analysis", fail)
    held, engine = object(), object()
    monkeypatch.setattr(models, "_speech_model", held)
    monkeypatch.setattr(models, "_speech_model_key", ("large-v3", "cpu", "models"))
    monkeypatch.setattr(diarize, "_engine_cache", (None, engine))
    launched: List[tuple] = []
    monkeypatch.setattr(JobManager, "_launch_worker", lambda self, job: launched.append(
        (job.job_id, models._speech_model, diarize._engine_cache)))
    room_for(monkeypatch)

    next_job = run_worker(repo, first.job_id)

    assert repo.get_job(first.job_id).status == JobStatus.FAILED
    # A file that could not be read says nothing about the process, so it still keeps the next.
    # Otherwise the models are let go before the fresh worker starts.
    assert (next_job, launched) == ((second.job_id, []) if kept else (None, [(second.job_id, None, None)]))


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
            # The old model is let go before a new one is built, so two are never held at once.
            assert models._speech_model is None
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

    # The same name on another device, as when a broken card falls back to the processor, is another model.
    fallen_back = models.speech_model("large-v3", "cpu", "models")
    assert fallen_back is not first
    assert built[1] == {"name": "large-v3", "device": "cpu", "compute_type": "int8", "download_root": "models"}
    elsewhere = models.speech_model("large-v3", "cpu", "other-models")
    assert elsewhere is not fallen_back and len(built) == 3

    other = models.speech_model("small", "cpu", "models")

    assert other is not first
    assert len(built) == 4
    assert built[3]["compute_type"] == "int8"
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
            # The old engine is let go before a new one loads the same models again.
            assert diarize._engine_cache is None
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


def fake_whisper(monkeypatch: pytest.MonkeyPatch, free: List[int]) -> List[str]:
    """Stand in for faster-whisper, each model taking 1500 MB of the memory `free` reports.

    Args:
        monkeypatch: The test's monkeypatch.
        free: The free memory, in bytes, in a list the fake models lower.

    Returns:
        The names of the models built, in order.
    """
    built: List[str] = []

    class FakeWhisperModel:
        """A model that takes its memory when built."""

        def __init__(self, name: str, device: str, compute_type: str, download_root: str):
            built.append(name)
            free[0] -= 1500 * resources.MEGABYTE

    class FakePipeline:
        """Transcribe nothing, as the real pipeline would a silent file."""

        def __init__(self, model):
            self.model = model

        def transcribe(self, path, **kwargs):
            return [], types.SimpleNamespace(language="en")

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    module.BatchedInferencePipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(resources, "memory_status", lambda: (free[0], 32 * 1024 * resources.MEGABYTE))
    monkeypatch.setattr(resources, "whisper_memory_bytes", lambda name: 2200 * resources.MEGABYTE)
    monkeypatch.setattr(models, "_speech_model", None)
    monkeypatch.setattr(models, "_speech_model_key", None)
    monkeypatch.setattr(models, "_keep_speech_model", True)
    return built


def test_a_model_the_worker_holds_is_not_counted_as_missing_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    free = [3000 * resources.MEGABYTE]
    built = fake_whisper(monkeypatch, free)

    def listen(model_name: str):
        return analysis._run_whisper("cpu", "speech.wav", 10.0, None, None, lambda fraction: None,
                                     lambda: False, model_name=model_name)

    listen("large-v3-turbo")
    # 1500 MB free now, under the 2200 MB a build needs; the model held needs nothing more.
    listen("large-v3-turbo")
    assert built == ["large-v3-turbo"]
    # Another model would be built, so it is checked, and there is not room for it.
    with pytest.raises(RuntimeError, match="needs about 2200 MB"):
        listen("small")
    assert built == ["large-v3-turbo"]


def test_before_build_runs_only_when_a_model_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    built = fake_whisper(monkeypatch, [8000 * resources.MEGABYTE])
    checked: List[Any] = []

    def check() -> None:
        checked.append(models._speech_model)

    models.speech_model("large-v3", "cpu", "models", before_build=check)
    models.speech_model("large-v3", "cpu", "models", before_build=check)
    models.speech_model("small", "cpu", "models", before_build=check)
    assert built == ["large-v3", "small"]
    # Each check ran with no model held: the old one is let go before memory is checked for the new.
    assert checked == [None, None]


def test_a_kept_analysis_cancelled_before_it_starts_ends_the_chain(tmp_path: Path) -> None:
    repo = store(tmp_path)
    kept = queued(repo, tmp_path, admitted_at=datetime.now(timezone.utc))
    assert JobManager(repo).cancel_job(kept.job_id, wait_seconds=0).status == JobStatus.CANCELLED
    assert run_worker(repo, kept.job_id) is None
    assert repo.get_job(kept.job_id).status == JobStatus.CANCELLED


def test_a_kept_analysis_whose_worker_died_is_failed_as_abandoned(tmp_path: Path) -> None:
    # Kept, it is queued with `admitted_at` set; if the worker dies before starting it, nothing writes to it.
    repo = store(tmp_path)
    kept = queued(repo, tmp_path, admitted_at=datetime.now(timezone.utc))
    when = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    repo._conn.execute("UPDATE jobs SET data = json_set(data, '$.updated_at', ?) WHERE id = ?", (when, kept.job_id))
    failed = JobManager(repo).get_job(kept.job_id)
    assert (failed.status, failed.failure_kind, failed.error_message) == (
        JobStatus.FAILED, FailureKind.WORKER, renderer.ABANDONED,
    )


def test_only_the_first_admitted_analysis_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = store(tmp_path)
    first, second, third = (queued(repo, tmp_path) for _ in range(3))
    monkeypatch.setattr(renderer, "_run_analysis", lambda *args, **kwargs: None)
    launched: List[str] = []
    monkeypatch.setattr(JobManager, "_launch_worker", lambda self, job: launched.append(job.job_id))
    room_for(monkeypatch)

    assert run_worker(repo, first.job_id) == second.job_id
    assert launched == [third.job_id]


def test_hearing_a_file_again_uses_the_model_the_worker_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: List[str] = []

    class Heard:
        """A model that hears nothing."""

        def transcribe(self, samples, **kwargs):
            return [], None

    def speech_model(name, device, download_root, before_build=None):
        asked.append(name)
        return Heard()

    monkeypatch.setattr(models, "speech_model", speech_model)
    monkeypatch.setattr(relisten, "read_samples", lambda path, ffmpeg_bin="ffmpeg": bytes(4 * 16000))
    monkeypatch.setattr(relisten, "stretches_to_transcribe", lambda voiced, duration: [(0.0, 1.0)])
    listener = relisten.Listener(name="large-v3-turbo", device="cpu", download_root="models")
    relisten.listen_again("speech.wav", "zh", listener, lambda fraction: None, lambda: False)
    assert asked == ["large-v3-turbo"]


def test_on_the_card_free_system_memory_is_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    built = fake_whisper(monkeypatch, [100 * resources.MEGABYTE])
    analysis._run_whisper("cuda", "speech.wav", 10.0, None, None, lambda fraction: None, lambda: False,
                          model_name="large-v3-turbo")
    assert built == ["large-v3-turbo"]


def test_a_transcribing_analysis_counts_the_speech_model_on_top_of_its_other_stages() -> None:
    whisper = resources.whisper_memory_bytes("large-v3-turbo")
    diarizing = resources.DIARIZATION_BYTES + int(3600 * resources.DIARIZATION_BYTES_PER_SECOND)
    assert resources.analysis_memory_bytes(True, "large-v3-turbo", 3600.0, diarize=True) == whisper + diarizing
    assert resources.analysis_memory_bytes(True, "large-v3-turbo") == whisper + resources.DETECTION_BYTES
    assert resources.analysis_memory_bytes(False, "large-v3-turbo", 3600.0) == resources.DETECTION_BYTES


@pytest.mark.parametrize("transcribe, diarizing, transcription, kept", [
    (False, True, "accurate", (False, False)),
    (True, False, "accurate", (True, False)),
    (True, True, "accurate", (True, True)),
    # Balanced and fast start with the small model, so the large one held would only wait to be dropped.
    (True, True, "balanced", (False, True)),
    (True, True, "fast", (False, True)),
])
def test_an_analysis_lets_go_of_the_held_models_it_will_not_start_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transcribe: bool, diarizing: bool, transcription: str,
    kept: tuple,
) -> None:
    _let_go_before(tmp_path, monkeypatch, transcribe, diarizing, transcription, kept,
                   analysis.whisper_model_for("accurate"))


def test_the_small_model_a_fast_analysis_left_is_kept_into_a_balanced_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Balanced starts with the small model, so the one held is used, not dropped and built again.
    _let_go_before(tmp_path, monkeypatch, True, True, "balanced", (True, True), analysis.FAST_WHISPER_MODEL)


def _let_go_before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transcribe: bool, diarizing: bool,
                   transcription: str, kept: tuple, held_name: str) -> None:
    """Run an analysis with a speech model and a diarizer held, and check which are held when its clock starts.

    Args:
        tmp_path: Folder for the job and its repository.
        monkeypatch: Fixture used to stand in for the models and the analysis.
        transcribe: Whether the analysis is expected to transcribe.
        diarizing: Whether it tells the speakers apart.
        transcription: Its transcription choice.
        kept: Whether the speech model and the diarizer are still held.
        held_name: The name of the speech model held.
    """
    repo = store(tmp_path)
    media = tmp_path / "song.mp3"
    media.write_bytes(b"")
    repo.save_asset(Asset(id="song", path=str(media), duration=10, has_video=False, has_audio=True))
    job = queued(repo, tmp_path, asset_id="song")
    held, engine = object(), object()
    monkeypatch.setattr(models, "_speech_model", held)
    monkeypatch.setattr(models, "_speech_model_key", (held_name, "cpu", "models"))
    monkeypatch.setattr(diarize, "_engine_cache", (None, engine))
    seen = []

    class Stop(Exception):
        """Ends the run once the analysis would start."""

    def analyze_media(*args, **kwargs):
        seen.append((models._speech_model, diarize._engine_cache))
        raise Stop()

    monkeypatch.setattr(renderer, "analyze_media", analyze_media)
    with open(os.path.join(job.work_dir, SPEC_FILE_NAME), "w", encoding="utf-8") as handle:
        json.dump({"transcribe": True, "with_speech": transcribe, "diarize": diarizing,
                   "transcription": transcription}, handle)
    clocked = []
    monkeypatch.setattr(renderer.StageClock, "start",
                        lambda self: clocked.append((models._speech_model, diarize._engine_cache)))
    run_worker(repo, job.job_id)
    # Let go before the clock reads its baseline, not only before the analysis.
    after = (held if kept[0] else None, (None, engine) if kept[1] else None)
    assert clocked == [after] and seen == [after]


@pytest.mark.parametrize("has_video, expected", [(False, False), (True, True)])
def test_an_analysis_specification_says_what_was_asked_and_whether_it_will_transcribe(
    monkeypatch: pytest.MonkeyPatch, has_video: bool, expected: bool,
) -> None:
    # A song is never transcribed; the worker reads this to let a held model go, and what was asked is kept.
    from app import runtime
    from app.tools import examining

    specs: List[Dict[str, Any]] = []
    monkeypatch.setattr(runtime.job_manager, "start_job", lambda job, spec: specs.append(spec) or job)
    asset = Asset(id="a", path="a.mp4", duration=10, has_video=has_video, has_audio=True)
    examining.start_analysis(asset, True, "accurate")
    assert specs[0]["with_speech"] is expected and specs[0]["transcribe"] is True
