"""Tests for reading the library a page at a time, with seconds as numbers."""

import json
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from app import runtime
from app.tools import catalog, editing, planning
from app.models.media import Asset

def stored(name: str, folder: str, tmp_path: Path, seconds: float = 12.5) -> str:
    """Register an asset without probing a file, filed in a folder when one is named."""
    asset = Asset(id=f"list-{uuid.uuid4().hex[:8]}", path=str(tmp_path / folder / name),
                  duration=Decimal(str(seconds)), has_video=True, has_audio=True)
    runtime.repo.save_asset(asset)
    return asset.id

@pytest.fixture
def shelf(tmp_path: Path) -> dict:
    folder = runtime.repo.create_folder(f"f-{uuid.uuid4().hex[:8]}", "assets", f"抽杯架-{uuid.uuid4().hex[:12]}")
    inside = [stored(f"杯架{index}.mp4", "a", tmp_path) for index in range(3)]
    runtime.repo.file_items("assets", inside, folder["id"])
    name = f"台南-{uuid.uuid4().hex[:6]}"
    outside = stored(f"{name}.mp4", "b", tmp_path, seconds=30)
    return {"folder": folder["id"], "inside": inside, "outside": outside, "name": name}

def test_a_page_of_the_library_says_how_many_there_are_in_all(shelf: dict) -> None:
    page = catalog.list_assets(limit=2)
    assert len(page["assets"]) == 2 and page["total"] >= 4
    assert page["offset"] == 0
    rest = catalog.list_assets(limit=2, offset=2)
    assert {asset["id"] for asset in page["assets"]}.isdisjoint({asset["id"] for asset in rest["assets"]})

def test_the_library_can_be_read_one_folder_at_a_time(shelf: dict) -> None:
    found = catalog.list_assets(folder_id=shelf["folder"])
    assert sorted(asset["id"] for asset in found["assets"]) == sorted(shelf["inside"])
    assert any(folder["id"] == shelf["folder"] for folder in found["folders"])

def test_the_library_can_be_searched_by_file_name(shelf: dict) -> None:
    assert [asset["id"] for asset in catalog.list_assets(text=shelf["name"])["assets"]] == [shelf["outside"]]

def test_each_file_is_a_short_record_with_its_length_as_a_number(shelf: dict) -> None:
    asset = next(item for item in catalog.list_assets(folder_id=shelf["folder"])["assets"])
    assert isinstance(asset["duration"], float) and asset["duration"] == 12.5
    assert {"id", "name", "duration", "has_video", "has_audio", "analyzed", "transcription", "stale", "library",
            "folder_id"} == set(asset)

def test_seconds_leave_the_server_as_numbers(media: Path) -> None:
    imported = catalog.import_asset(str(media / "wide.mp4"))
    assert isinstance(imported["duration"], float)
    folder = catalog.import_folder(str(media))
    assert isinstance(folder["total_duration"], float)
    assert all(isinstance(asset["duration"], (float, type(None))) for asset in folder["assets"])
    project = editing.create_project(name="數字")
    editing.apply_edits(project["id"], project["version"], planning.OPERATIONS.validate_python([
        {"action": "add_track", "track_id": "main", "track_type": "video"},
        {"action": "insert_clip", "track_id": "main", "clip_id": "a", "asset_id": imported["id"],
         "source_range": {"start": 0, "end": 1.5}},
    ]))
    clip = editing.get_project(project["id"])["tracks"][0]["clips"][0]
    assert isinstance(clip["source_range"]["end"], float) and isinstance(clip["timeline_in"], float)
    json.dumps(editing.get_project(project["id"]))

def test_notes_on_how_a_file_may_be_used_are_read_wherever_the_file_is(shelf: dict) -> None:
    written = catalog.edit_asset(shelf["outside"], "  請消音  ")
    assert written["notes"] == "請消音"
    listed = catalog.list_assets(text=shelf["name"])["assets"]
    assert listed[0]["notes"] == "請消音"
    # A file without notes carries none, rather than an empty field on every line.
    assert all("notes" not in item for item in catalog.list_assets(folder_id=shelf["folder"])["assets"])
    catalog.edit_asset(shelf["outside"], "")
    assert "notes" not in catalog.list_assets(text=shelf["name"])["assets"][0]

def test_importing_a_file_again_keeps_its_notes(media: Path) -> None:
    asset_id = catalog.import_asset(str(media / "song.mp3"))["id"]
    catalog.edit_asset(asset_id, "只用前 20 秒")
    assert catalog.import_asset(str(media / "song.mp3"))["id"] == asset_id
    assert runtime.repo.get_assets([asset_id])[asset_id].notes == "只用前 20 秒"
    catalog.edit_asset(asset_id, "")

def test_the_editor_writes_the_same_notes(shelf: dict) -> None:
    from starlette.testclient import TestClient
    from app.ui import app as editor

    client = TestClient(editor.create_app())
    assert client.post("/api/assets/notes", json={"asset_id": shelf["outside"], "notes": "可不放"}).status_code == 200
    assert runtime.repo.get_assets([shelf["outside"]])[shelf["outside"]].notes == "可不放"
    shown = next(item for item in client.get("/api/assets?all=1").json()["assets"] if item["id"] == shelf["outside"])
    assert shown["notes"] == "可不放"
