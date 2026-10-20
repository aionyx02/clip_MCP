"""Tests for choosing a fast transcription on a slow computer, and upgrading it later."""

from pathlib import Path

import pytest

from app import runtime
from app.tools import catalog, examining
from app.engine import analysis, renderer
from app.engine.diarize import speaker_model_name
from app.engine.rhythm import rhythm_model_name
from app.models.job import Job, JobKind
from app.models.media import Asset, MediaAnalysis, Transcript
from app.storage.repo import Repository

def analysed_with(asset_id: str, model: str) -> MediaAnalysis:
    """An analysis that is current in every way, its transcript made with `model`."""
    return MediaAnalysis(
        asset_id=asset_id, duration=600.0,
        transcript=Transcript(language="zh", model=model, segments=[]),
        recipe=analysis.current_recipe(model, speaker_model=speaker_model_name(), rhythm_model=rhythm_model_name()),
    )

@pytest.fixture
def talk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Asset:
    asset = Asset(id="quality-talk", path=str(tmp_path / "talk.mp4"), duration=600, has_video=True, has_audio=True)
    runtime.repo.save_asset(asset)
    return asset

@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list:
    specs: list = []
    monkeypatch.setattr(runtime.job_manager, "start_job", lambda job, spec: specs.append(spec) or job)
    return specs


def skipped(result: dict) -> list:
    """The assets an analysis left alone because theirs is current."""
    return [item["asset_id"] for item in result["skipped"] if "is current" in item["reason"]]

def test_each_choice_names_its_model() -> None:
    assert analysis.whisper_model_for("accurate") == analysis.whisper_model_name()
    assert analysis.whisper_model_for("fast") == analysis.FAST_WHISPER_MODEL != analysis.whisper_model_name()

def test_a_fast_transcript_is_not_out_of_date_for_being_fast(talk: Asset) -> None:
    assert not examining.is_stale(analysed_with(talk.id, analysis.FAST_WHISPER_MODEL))
    assert examining.is_stale(analysed_with(talk.id, "medium"))

def test_the_choice_reaches_the_job(talk: Asset, started: list) -> None:
    examining.analyze_asset([talk.id], again=True, transcription="fast")
    assert started[-1]["transcription"] == "fast"

def test_asking_for_accurate_upgrades_a_fast_transcript_without_again(talk: Asset, started: list) -> None:
    runtime.repo.save_analysis(analysed_with(talk.id, analysis.FAST_WHISPER_MODEL))
    result = examining.analyze_asset([talk.id])
    assert [job["asset_id"] for job in result["jobs"]] == [talk.id]
    assert result["upgrading"] == [talk.id]

def test_asking_for_fast_keeps_an_accurate_transcript(talk: Asset, started: list) -> None:
    runtime.repo.save_analysis(analysed_with(talk.id, analysis.whisper_model_name()))
    assert skipped(examining.analyze_asset([talk.id], transcription="fast")) == [talk.id]

def test_asking_for_fast_again_keeps_the_fast_transcript(talk: Asset, started: list) -> None:
    runtime.repo.save_analysis(analysed_with(talk.id, analysis.FAST_WHISPER_MODEL))
    assert skipped(examining.analyze_asset([talk.id], transcription="fast")) == [talk.id]

def test_an_out_of_date_accurate_transcript_is_redone_accurately_when_fast_is_asked(
    talk: Asset, started: list, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime.repo.save_analysis(analysed_with(talk.id, analysis.whisper_model_name()))
    monkeypatch.setattr(examining, "is_stale", lambda found: True)
    examining.analyze_asset([talk.id], transcription="fast")
    assert started[-1]["transcription"] == "accurate"

def test_again_never_turns_an_accurate_transcript_fast(talk: Asset, started: list) -> None:
    runtime.repo.save_analysis(analysed_with(talk.id, analysis.whisper_model_name()))
    examining.analyze_asset([talk.id], transcription="fast", again=True)
    assert started[-1]["transcription"] == "accurate"

def test_when_small_is_the_configured_model_nothing_counts_as_fast(
    talk: Asset, started: list, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIP_MCP_WHISPER_MODEL", analysis.FAST_WHISPER_MODEL)
    runtime.repo.save_analysis(analysed_with(talk.id, analysis.FAST_WHISPER_MODEL))
    assert analysis.transcription_of(analysis.FAST_WHISPER_MODEL) == "accurate"
    result = examining.analyze_asset([talk.id])
    assert skipped(result) == [talk.id] and result["upgrading"] == []

def test_a_dry_run_starts_nothing_and_gives_both_times(talk: Asset, started: list) -> None:
    found = examining.analyze_asset([talk.id], again=True, dry_run=True)
    assert started == []
    assert found["would_start"] == [talk.id]
    accurate, fast = found["estimates"]["accurate"], found["estimates"]["fast"]
    assert accurate["footage_minutes"] == fast["footage_minutes"] == 10.0
    assert fast["transcription_minutes"][0] <= accurate["transcription_minutes"][0]

def test_an_upgrade_says_the_whole_analysis_is_redone(talk: Asset, started: list) -> None:
    runtime.repo.save_analysis(analysed_with(talk.id, analysis.FAST_WHISPER_MODEL))
    assert "whole" in examining.analyze_asset([talk.id])["estimate"]["note"]

def test_the_library_says_which_transcripts_are_fast(talk: Asset) -> None:
    runtime.repo.save_analysis(analysed_with(talk.id, analysis.FAST_WHISPER_MODEL))
    listed = {asset["id"]: asset for asset in catalog.list_assets(text="talk.mp4", limit=200)["assets"]}
    assert listed[talk.id]["transcription"] == "fast"
    assert examining.get_analysis(talk.id)["transcription"] == "fast"

def test_the_editor_marks_a_fast_transcript(talk: Asset) -> None:
    from app.ui import app as editor

    runtime.repo.save_analysis(analysed_with(talk.id, analysis.FAST_WHISPER_MODEL))
    assert editor._asset_summary(talk)["fast_transcript"] is True
    runtime.repo.save_analysis(analysed_with(talk.id, analysis.whisper_model_name()))
    assert editor._asset_summary(talk)["fast_transcript"] is False

def test_the_job_runs_the_model_that_was_chosen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = Repository(str(tmp_path / "runs.db"))
    asset = Asset(id="quality-run", path=str(tmp_path / "talk.mp4"), duration=60, has_video=True, has_audio=True)
    store.save_asset(asset)
    seen = {}

    def analyzed(asset, work_dir, **keywords):
        seen.update(keywords)
        return MediaAnalysis(asset_id=asset.id, duration=60)

    monkeypatch.setattr(renderer, "analyze_media", analyzed)
    context = type("Context", (), {"report": lambda self, *a: None, "is_cancelled": lambda self: False})()
    job = Job(kind=JobKind.ANALYZE, asset_id=asset.id, work_dir=str(tmp_path))
    renderer._run_analysis(store, job, {"transcribe": True, "transcription": "fast"}, context)
    assert seen["speech_model"] == analysis.FAST_WHISPER_MODEL
    renderer._run_analysis(store, job, {"transcribe": True}, context)
    assert seen["speech_model"] == analysis.whisper_model_name()

def test_a_file_already_being_analyzed_is_left_to_that_job_and_says_so(talk: Asset, started: list) -> None:
    """#88: four files came back skipped with no reason, and a second job would have doubled the work."""
    from app.models.job import Job, JobKind, JobStatus

    waiting = Job(kind=JobKind.ANALYZE, asset_id=talk.id)
    runtime.repo.add_job(waiting)
    try:
        result = examining.analyze_asset([talk.id], again=True)
    finally:
        # Never left queued for a later test's queue to try to start.
        runtime.repo.update_job(waiting.job_id, lambda job: setattr(job, "status", JobStatus.CANCELLED))
    assert result["jobs"] == [] and started == []
    assert result["skipped"] == [{"asset_id": talk.id,
                                  "reason": f"already queued in job {waiting.job_id}; follow it with get_job"}]
