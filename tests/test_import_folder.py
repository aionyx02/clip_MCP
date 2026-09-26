"""Tests for importing a whole folder of footage."""

import os
from pathlib import Path

import pytest

from app import server
from app.engine.analysis import current_recipe, whisper_model_name
from app.engine.diarize import speaker_model_name
from app.engine.rhythm import rhythm_model_name
from app.models.job import Job, JobKind, JobStatus
from app.models.media import MediaAnalysis
from app.server import _natural_key, import_asset, import_folder

def names(result: dict) -> list:
    """Extract the imported file names from an `import_folder` result.

    Args:
        result: What `import_folder` returned.

    Returns:
        The base name of each imported asset, in the returned order.
    """
    return [os.path.basename(asset["path"]) for asset in result["assets"]]

def test_natural_key_orders_numbers_by_value() -> None:
    assert sorted(["clip10", "clip2", "clip1"], key=_natural_key) == ["clip1", "clip2", "clip10"]

def test_imports_media_in_natural_name_order(footage_folder: Path) -> None:
    assert names(import_folder(str(footage_folder))) == ["bgm.mp3", "clip1.mp4", "clip2.mp4", "clip10.mp4"]

def test_ignores_files_that_are_not_media(footage_folder: Path) -> None:
    assert "notes.txt" not in names(import_folder(str(footage_folder)))

def test_reports_unreadable_files_instead_of_failing_the_folder(footage_folder: Path) -> None:
    result = import_folder(str(footage_folder))
    skipped = [os.path.basename(entry["path"]) for entry in result["skipped"]]
    assert skipped == ["broken.mp4"]
    assert result["skipped"][0]["reason"]
    assert len(result["assets"]) == 4

def test_skips_sub_folders_unless_asked(footage_folder: Path) -> None:
    assert "nested.mp4" not in names(import_folder(str(footage_folder)))
    assert "nested.mp4" in names(import_folder(str(footage_folder), recursive=True))

def test_totals_the_durations_it_imported(footage_folder: Path) -> None:
    result = import_folder(str(footage_folder))
    assert float(result["total_duration"]) == pytest.approx(
        sum(float(asset["duration"]) for asset in result["assets"])
    )

def test_reports_which_files_have_video_and_audio(footage_folder: Path) -> None:
    assets = {os.path.basename(asset["path"]): asset for asset in import_folder(str(footage_folder))["assets"]}
    assert assets["bgm.mp3"]["has_video"] is False and assets["bgm.mp3"]["has_audio"] is True
    assert assets["clip1.mp4"]["has_video"] is True

def test_importing_the_same_folder_again_keeps_asset_ids(footage_folder: Path) -> None:
    first = {asset["path"]: asset["id"] for asset in import_folder(str(footage_folder))["assets"]}
    second = {asset["path"]: asset["id"] for asset in import_folder(str(footage_folder))["assets"]}
    assert first == second

def test_import_asset_returns_the_same_id_as_the_folder_import(footage_folder: Path) -> None:
    folder = {asset["path"]: asset["id"] for asset in import_folder(str(footage_folder))["assets"]}
    single = import_asset(str(footage_folder / "clip2.mp4"))
    assert single["id"] == folder[single["path"]]

def test_missing_folder_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        import_folder(str(tmp_path / "nope"))

def test_empty_folder_imports_nothing(tmp_path: Path) -> None:
    result = import_folder(str(tmp_path))
    assert result["assets"] == [] and result["skipped"] == [] and float(result["total_duration"]) == 0

def test_a_folder_is_analyzed_in_one_call_and_what_is_current_is_left_alone(footage_folder: Path, monkeypatch) -> None:
    started = []
    monkeypatch.setattr(server.job_manager, "start_job", lambda job, spec: started.append(job.asset_id) or job)
    assets = [asset["id"] for asset in import_folder(str(footage_folder))["assets"]]
    server.repo.save_analysis(MediaAnalysis(asset_id=assets[1], duration=1.0, recipe=current_recipe(
        whisper_model_name(), speaker_model=speaker_model_name(), rhythm_model=rhythm_model_name(),
    )))

    result = server.analyze_asset(assets)
    assert result["skipped"] == [assets[1]]
    assert [job["asset_id"] for job in result["jobs"]] == started == [assets[0], *assets[2:]]

    started.clear()
    assert [job["asset_id"] for job in server.analyze_asset(assets, again=True)["jobs"]] == assets

def test_a_batch_of_jobs_is_read_back_with_one_summary() -> None:
    done = Job(kind=JobKind.ANALYZE, status=JobStatus.COMPLETED, progress=1.0)
    broken = Job(kind=JobKind.ANALYZE, status=JobStatus.FAILED, error_message="no audio")
    running = Job(kind=JobKind.RENDER, status=JobStatus.RUNNING, progress=0.5)
    for job in (done, broken, running):
        server.repo.add_job(job)
    result = server.get_job([done.job_id, broken.job_id, running.job_id])
    assert (result["finished"], result["failed"], result["total"]) == (2, 1, 3)
    assert result["jobs"][1]["error_message"] == "no audio"
    assert "error_message" not in result["jobs"][0]
