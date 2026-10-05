"""Tests for remembering how fast the local models ran here, and estimating the next run from it."""

import array
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.engine import diarize, machine, speed
from app.engine.analysis import FAST_WHISPER_MODEL, whisper_model_name
from app.engine.diarize import speaker_model_name
from app.storage.repo import Repository

COMPUTER = "desk / Some CPU"
ACCURATE = whisper_model_name()

@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(str(tmp_path / "speed.db"))

def computer(tier: str = "mid", gpu=None) -> machine.Machine:
    return machine.Machine(tier=tier, gpu=gpu, gpu_memory_bytes=None, memory_bytes=None, logical_cores=8,
                           apple_silicon=False, mac=False, identity=COMPUTER)

def measured(stage: str = "transcription", seconds: float = 600.0, device: str = "cpu",
             model: str = ACCURATE) -> speed.Measured:
    return speed.Measured(stage=stage, device=device, model=model, seconds=seconds)

def samples(repo: Repository, stage: str = "transcription", device: str = "cpu", model: str = ACCURATE,
            computer_name: str = COMPUTER) -> list:
    return repo.speed_samples(computer_name, stage, device, model, limit=100)

# --- recording ---------------------------------------------------------------------------

def test_a_run_is_kept_against_this_computer(repo: Repository) -> None:
    speed.record(repo, [measured()], media_seconds=1200, computer=COMPUTER)
    assert samples(repo) == [0.5]
    assert samples(repo, computer_name="laptop / Other CPU") == []

def test_the_seconds_and_the_footage_length_are_both_kept(repo: Repository) -> None:
    speed.record(repo, [measured(seconds=300)], media_seconds=1200, computer=COMPUTER)
    assert repo._query("SELECT seconds, media_seconds FROM speed_samples") == [(300.0, 1200.0)]

def test_a_short_file_is_not_kept_since_loading_the_model_is_most_of_its_time(repo: Repository) -> None:
    speed.record(repo, [measured(seconds=30)], media_seconds=90, computer=COMPUTER)
    assert samples(repo) == []

def test_only_the_latest_runs_are_kept(repo: Repository) -> None:
    for index in range(speed.KEPT_SAMPLES + 5):
        speed.record(repo, [measured(seconds=float(index + 1))], media_seconds=600, computer=COMPUTER)
    kept = samples(repo)
    assert len(kept) == speed.KEPT_SAMPLES
    assert kept[0] == pytest.approx((speed.KEPT_SAMPLES + 5) / 600)

# --- estimating --------------------------------------------------------------------------

def test_with_nothing_measured_the_hardware_table_is_used_and_called_rough(repo: Repository) -> None:
    found = speed.estimate(repo, computer("mid"), {"accurate": 7200}, speakers=True)
    assert found["basis"] == speed.ROUGH
    assert found["transcription_minutes"] == [20, 80]
    assert found["speakers_minutes"] == [6, 16]

def test_a_low_tier_has_no_upper_bound_until_measured(repo: Repository) -> None:
    found = speed.estimate(repo, computer("low"), {"accurate": 3600}, speakers=False)
    assert found["transcription_minutes"] == [30, None]
    assert found["speakers_minutes"] is None

def test_fast_is_shorter_by_the_table(repo: Repository) -> None:
    accurate = speed.estimate(repo, computer("mid"), {"accurate": 3600}, False)["transcription_minutes"]
    fast = speed.estimate(repo, computer("mid"), {"fast": 3600}, False)["transcription_minutes"]
    assert fast == [accurate[0] // speed.FAST_SPEEDUP, accurate[1] // speed.FAST_SPEEDUP]

def test_a_batch_of_both_kinds_adds_up(repo: Repository) -> None:
    both = speed.estimate(repo, computer("mid"), {"accurate": 3600, "fast": 3600}, False)
    assert both["transcription_minutes"] == [15, 60]

def test_once_measured_the_median_of_recent_runs_is_used_with_room_to_spare(repo: Repository) -> None:
    for seconds in (600, 900, 1200):  # half, three quarters and all of a twenty-minute file
        speed.record(repo, [measured(seconds=seconds)], media_seconds=1200, computer=COMPUTER)
    found = speed.estimate(repo, computer("low"), {"accurate": 3600}, speakers=False)
    assert found["basis"] == speed.MEASURED
    assert found["transcription_minutes"] == [45, 68]

def test_runs_on_another_device_or_model_do_not_count(repo: Repository) -> None:
    speed.record(repo, [measured(device="cuda")], media_seconds=1200, computer=COMPUTER)
    speed.record(repo, [measured(model=FAST_WHISPER_MODEL)], media_seconds=1200, computer=COMPUTER)
    assert speed.estimate(repo, computer("mid"), {"accurate": 3600}, False)["basis"] == speed.ROUGH

def test_speakers_are_estimated_from_their_own_runs(repo: Repository) -> None:
    speed.record(repo, [measured(stage="speakers", model=speaker_model_name(), seconds=60)],
                 media_seconds=1200, computer=COMPUTER)
    found = speed.estimate(repo, computer("mid"), {"accurate": 3600}, True)
    assert found["speakers_minutes"] == [3, 5]
    # Transcription was never measured, so the whole estimate is still only as good as the table.
    assert found["basis"] == speed.ROUGH

def test_this_computer_is_named_without_looking_at_the_card() -> None:
    assert " / " in machine.identity()

# --- where the AI and the job runner meet it ---------------------------------------------

def test_the_reply_to_an_analysis_says_how_long_the_transcription_will_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import server
    from app.models.media import Asset

    monkeypatch.setattr(server.job_manager, "start_job", lambda job, spec: job)
    talk = Asset(id="speed-talk", path=str(tmp_path / "talk.mp4"), duration=1800, has_video=True, has_audio=True)
    silent = Asset(id="speed-silent", path=str(tmp_path / "silent.mp4"), duration=600, has_video=True, has_audio=False)
    for asset in (talk, silent):
        server.repo.save_asset(asset)

    found = server.analyze_asset([talk.id, silent.id], again=True)["estimate"]
    assert found["footage_minutes"] == 30.0  # the silent file is not transcribed
    assert found["tier"] == machine.profile().tier and found["basis"] in (speed.ROUGH, speed.MEASURED)
    assert found["speakers_minutes"] is not None
    assert server.analyze_asset([talk.id], again=True, diarize=False)["estimate"]["speakers_minutes"] is None
    assert "estimate" not in server.analyze_asset([talk.id], again=True, transcribe=False)

def test_the_instructions_use_this_computers_runs_once_there_are_some(repo: Repository) -> None:
    here = computer("mid")
    assert speed.ROUGH in machine.describe(here, speed.estimate(repo, here, {"accurate": 3600}, False))
    speed.record(repo, [measured(seconds=120)], media_seconds=1200, computer=COMPUTER)
    told = machine.describe(here, speed.estimate(repo, here, {"accurate": 3600}, False))
    assert speed.MEASURED in told and "6 to 9 minutes" in told

def test_a_finished_analysis_leaves_its_speed_behind_and_a_failed_one_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.engine import renderer
    from app.models.job import Job, JobKind
    from app.models.media import Asset, MediaAnalysis

    store = Repository(str(tmp_path / "runs.db"))
    asset = Asset(id="speed-run", path=str(tmp_path / "talk.mp4"), duration=1000, has_video=True, has_audio=True)
    store.save_asset(asset)
    job = Job(kind=JobKind.ANALYZE, asset_id=asset.id, work_dir=str(tmp_path))
    context = type("Context", (), {"report": lambda self, *a: None, "is_cancelled": lambda self: False})()

    def analyzed(asset, work_dir, on_measured, **_):
        on_measured("transcription", "cpu", ACCURATE, 250.0)
        on_measured("speakers", "cpu", "diar", 50.0)
        return MediaAnalysis(asset_id=asset.id, duration=1000)

    monkeypatch.setattr(renderer, "analyze_media", analyzed)
    renderer._run_analysis(store, job, {"transcribe": True}, context)
    assert store.speed_samples(machine.identity(), "transcription", "cpu", ACCURATE, limit=5) == [0.25]
    assert store.speed_samples(machine.identity(), "speakers", "cpu", "diar", limit=5) == [0.05]

    def broken(asset, work_dir, on_measured, **_):
        on_measured("transcription", "cpu", ACCURATE, 1.0)
        raise RuntimeError("speech recognition failed")

    monkeypatch.setattr(renderer, "analyze_media", broken)
    with pytest.raises(RuntimeError):
        renderer._run_analysis(store, job, {"transcribe": True}, context)
    assert store.speed_samples(machine.identity(), "transcription", "cpu", ACCURATE, limit=5) == [0.25]

def test_telling_voices_apart_is_timed_after_its_models_are_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Transcription leaves loading out; the two stages have to be measured the same way.
    def slow_to_load(speakers):
        time.sleep(0.3)
        return SimpleNamespace(process=lambda samples, callback: SimpleNamespace(sort_by_start_time=lambda: []))

    monkeypatch.setattr(diarize, "_engine", slow_to_load)
    monkeypatch.setattr(diarize, "read_samples", lambda path, ffmpeg_bin: array.array("f", [0.0] * 16))
    monkeypatch.setattr(diarize, "_voices", lambda samples, turns: [])
    taken = []
    diarize.find_speakers("talk.wav", lambda fraction: None, lambda: False, on_measured=taken.append)
    assert len(taken) == 1 and taken[0] < 0.3
