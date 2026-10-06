"""Tests for where files go: the user's videos out of the workspace, and nothing piling up in it."""

import os
import time
from pathlib import Path

from app.server import (
    _OPERATIONS,
    WORKSPACE_DIR,
    apply_edits,
    create_project,
    get_job,
    import_asset,
    render_project,
    storage_usage,
    tidy_old_outputs,
)
from app.storage import housekeeping
from app.workspace import output_dir

ANYTHING = ["bad_picture", "clipping", "music", "mid_speech", "repeated", "continuity", "unplanned", "captions", "length"]

def finished(job: dict) -> dict:
    """Wait for a render to end.

    Args:
        job: What `render_project` returned.

    Returns:
        The job as `get_job` reports it at the end.
    """
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = get_job([job["job_id"]])["jobs"][0]
        if state["status"] in {"completed", "failed", "cancelled"}:
            return state
        time.sleep(0.3)
    raise AssertionError(f"render {job['job_id']} did not finish")

def one_clip_project(media: Path, name: str) -> str:
    """A project of one short clip, named.

    Args:
        media: The test media folder.
        name: The project's name.

    Returns:
        Its ID.
    """
    asset = import_asset(str(media / "wide.mp4"))["id"]
    project = create_project(name=name, width=320, height=180)
    apply_edits(project["id"], project["version"], _OPERATIONS.validate_python([
        {"action": "add_track", "track_id": "v", "track_type": "video"},
        {"action": "insert_clip", "track_id": "v", "clip_id": "c1", "asset_id": asset,
         "source_range": {"start": 0, "end": 2}},
    ]))
    return project["id"]

def test_a_name_already_taken_gets_the_next_number(tmp_path: Path) -> None:
    (tmp_path / "EP1.mp4").write_bytes(b"")
    (tmp_path / "EP1 (2).mp4").write_bytes(b"")
    assert housekeeping.unique_path(str(tmp_path), "EP1.mp4") == str(tmp_path / "EP1 (3).mp4")
    assert housekeeping.unique_path(str(tmp_path), "EP2.mp4") == str(tmp_path / "EP2.mp4")

def test_delivered_files_are_named_as_a_person_would() -> None:
    assert housekeeping.delivery_name("EP1 台北", "output", ".mp4") == "EP1 台北.mp4"
    assert housekeeping.delivery_name("EP1 台北", "output-portrait", ".mp4") == "EP1 台北（直式）.mp4"
    assert housekeeping.delivery_name("EP1 台北", "cover", ".jpg") == "EP1 台北 封面.jpg"
    # The same cut with its captions burned in is told apart by name, not by a (2).
    assert housekeeping.delivery_name("EP1 台北", "output", ".mp4", captioned=True) == "EP1 台北（字幕）.mp4"
    assert housekeeping.delivery_name("EP1 台北", "output-portrait", ".mp4", True) == "EP1 台北（直式、字幕）.mp4"

def test_pruning_keeps_only_what_it_is_told_to(tmp_path: Path) -> None:
    for name in ("old", "newest", "running"):
        (tmp_path / name).mkdir()
    housekeeping.prune(str(tmp_path), {"newest", "running"})
    assert sorted(os.listdir(tmp_path)) == ["newest", "running"]

def test_clearing_leaves_what_a_running_job_is_using(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    (jobs / "done").mkdir(parents=True)
    (jobs / "done" / "log.txt").write_text("x" * 100)
    (jobs / "busy").mkdir()
    freed = housekeeping.clean(str(tmp_path), ["work"], {"busy"})
    assert freed == 100
    assert sorted(os.listdir(jobs)) == ["busy"]

def test_the_old_outputs_keep_the_newest_of_each_project_and_nothing_else(tmp_path: Path) -> None:
    legacy = tmp_path / "outputs"
    files = {
        "a/EP1_output.mp4": 1, "b/EP1_output.mp4": 2, "b/job.json": 2, "c/EP1_output-portrait.mp4": 3,
        "d/EP1_cover.jpg": 4, "sound-x/EP1_sound.m4a": 5,
    }
    for relative, age in files.items():
        path = legacy / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        os.utime(path, (1_000_000 + age, 1_000_000 + age))
    plan = housekeeping.legacy_plan(str(tmp_path), str(tmp_path / "videos"))
    kept = {Path(item.path).relative_to(legacy).as_posix(): Path(item.keep_as).name for item in plan if item.keep_as}
    assert kept == {
        "b/EP1_output.mp4": "EP1.mp4", "c/EP1_output-portrait.mp4": "EP1（直式）.mp4", "d/EP1_cover.jpg": "EP1 封面.jpg",
    }
    assert len(plan) == len(files)
    duplicates = {Path(item.path).relative_to(legacy).as_posix() for item in plan if item.duplicate}
    assert duplicates == {"a/EP1_output.mp4"}

def test_a_finished_video_is_saved_with_the_users_videos_and_its_working_files_go(media: Path) -> None:
    project = one_clip_project(media, "交付測試")
    # Earlier tests may have transcribed this footage, and a two-second cut ends mid-sentence;
    # what is checked here is where the file goes, not whether the cut is any good.
    first = finished(render_project(project, allow=ANYTHING))
    second = finished(render_project(project, allow=ANYTHING))
    assert first["status"] == second["status"] == "completed", (first, second)
    assert first["output_path"] == str(output_dir() / "交付測試.mp4")
    assert second["output_path"] == str(output_dir() / "交付測試 (2).mp4")
    assert Path(first["output_path"]).stat().st_size > 0
    # Swept by the next render rather than by itself: its own log is open until its process ends.
    working = Path(housekeeping.render_dir(WORKSPACE_DIR)) / first["job_id"]
    deadline = time.monotonic() + 10
    while working.exists() and time.monotonic() < deadline:
        housekeeping.sweep_renders(WORKSPACE_DIR, lambda job_id: job_id == first["job_id"])
        time.sleep(0.2)
    assert not working.exists()

def test_only_the_newest_preview_of_a_project_is_kept(media: Path) -> None:
    project = one_clip_project(media, "預覽測試")
    first = finished(render_project(project, is_preview=True))
    second = finished(render_project(project, is_preview=True))
    assert second["status"] == "completed", second
    assert Path(second["output_path"]).exists()
    assert not Path(first["output_path"]).exists()
    assert os.listdir(housekeeping.preview_dir(WORKSPACE_DIR, project)) == [second["job_id"]]

def test_the_tools_report_space_and_never_tidy_without_being_told(tmp_path: Path) -> None:
    labels = {item["key"]: item for item in storage_usage()["items"]}
    assert labels["outputs"]["clearable"] is False
    assert labels["data"]["clearable"] is False
    legacy = Path(WORKSPACE_DIR) / "outputs" / "job"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "舊片_output.mp4").write_bytes(b"x")
    listed = tidy_old_outputs()
    assert [Path(item["to"]).name for item in listed["keep"]] == ["舊片.mp4"]
    assert (legacy / "舊片_output.mp4").exists()
