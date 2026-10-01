"""Tests for the editor's web surface: what it shows, and what it refuses to do."""

import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app import server
from app.ui import app as editor
from app.ui import dialogs

@pytest.fixture
def client() -> TestClient:
    """The editor, served in-process."""
    return TestClient(editor.create_app())

def test_it_says_it_is_the_editor(client: TestClient) -> None:
    assert client.get("/api/ping").json() == {"app": "clip-mcp-editor"}

def test_a_new_project_comes_ready_for_footage_and_music(client: TestClient) -> None:
    made = client.post("/api/projects", json={"name": "新的", "width": 1080, "height": 1920}).json()
    project = server.repo.get_project(made["id"])
    assert (project.width, project.height) == (1080, 1920)
    assert [(track.id, track.track_type.value) for track in project.tracks] == [("video", "video"), ("music", "audio")]

def test_chosen_files_are_used_where_they_are_never_copied(client: TestClient, media: Path, monkeypatch) -> None:
    original = media / "tall.mp4"
    monkeypatch.setattr(dialogs, "pick", lambda kind: [str(original)])
    before = set(os.listdir(server.WORKSPACE_DIR))
    result = client.post("/api/pick", json={"kind": "files"}).json()
    assert result == {"chosen": 1, "imported": 1, "failed": []}
    listed = {asset["path"] for asset in client.get("/api/assets").json()["assets"]}
    assert str(original.resolve()) in {str(Path(path).resolve()) for path in listed}
    assert set(os.listdir(server.WORKSPACE_DIR)) - before <= {"cache"}

def test_a_cancelled_window_imports_nothing(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(dialogs, "pick", lambda kind: [])
    assert client.post("/api/pick", json={"kind": "folder"}).json() == {"chosen": 0, "imported": 0, "failed": []}

def test_a_project_whose_footage_is_gone_says_so(client: TestClient, media: Path, tmp_path: Path) -> None:
    moved = tmp_path / "moved.mp4"
    moved.write_bytes((media / "wide.mp4").read_bytes())
    asset = server.import_asset(str(moved))["id"]
    made = client.post("/api/projects", json={"name": "會不見的"}).json()
    project = server.repo.get_project(made["id"])
    client.post(f"/api/projects/{made['id']}/edits", json={"expected_version": project.version, "operations": [
        {"action": "insert_clip", "track_id": "video", "clip_id": "c1", "asset_id": asset,
         "source_range": {"start": 0, "end": 2}},
    ]})
    moved.unlink()
    listed = next(item for item in client.get("/api/projects").json()["projects"] if item["id"] == made["id"])
    assert listed["missing"] == ["moved.mp4"]
    assert listed["thumb"] == {"asset_id": asset, "t": 1.0}

def test_only_the_apps_own_folders_can_be_opened(client: TestClient, monkeypatch) -> None:
    opened = []
    monkeypatch.setattr(os, "startfile", opened.append, raising=False)
    assert client.post("/api/open-folder", json={"folder": "C:\\Windows"}).status_code == 404
    assert client.post("/api/open-folder", json={"folder": "outputs"}).status_code == 200
    assert opened == [str(server.output_dir())]

def test_clearing_ignores_anything_but_what_can_be_made_again(client: TestClient, monkeypatch) -> None:
    asked = []
    monkeypatch.setattr(server, "clean_storage", lambda kinds: asked.append(kinds) or {"freed_megabytes": 0})
    client.post("/api/storage", json={"kinds": ["previews", "data", "outputs", "model:whisper"]})
    assert asked == [["previews"]]

def test_undo_puts_back_the_last_edit_and_refuses_once_something_else_has_edited(client: TestClient, media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    made = client.post("/api/projects", json={"name": "復原"}).json()
    version = server.repo.get_project(made["id"]).version
    edited = client.post(f"/api/projects/{made['id']}/edits", json={"expected_version": version, "operations": [
        {"action": "insert_clip", "track_id": "video", "clip_id": "c1", "asset_id": asset,
         "source_range": {"start": 0, "end": 2}},
    ]}).json()
    assert client.get(f"/api/projects/{made['id']}").json()["can_undo"] is True
    assert client.post(f"/api/projects/{made['id']}/undo").status_code == 200
    assert not server.repo.get_project(made["id"]).base_video_track.clips

    current = server.repo.get_project(made["id"]).version
    client.post(f"/api/projects/{made['id']}/edits", json={"expected_version": current, "operations": [
        {"action": "insert_clip", "track_id": "video", "clip_id": "c2", "asset_id": asset,
         "source_range": {"start": 0, "end": 2}},
    ]})
    # The AI renames it meanwhile: undoing now would throw that away unseen.
    server.apply_edits(made["id"], current + 1, server._OPERATIONS.validate_python([
        {"action": "rename_project", "name": "AI 改的"},
    ]))
    assert client.post(f"/api/projects/{made['id']}/undo").status_code == 409
    assert edited["new_version"] == version + 1
