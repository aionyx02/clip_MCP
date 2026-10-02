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

def test_it_says_it_is_the_editor_and_for_which_workspace(client: TestClient) -> None:
    answer = client.get("/api/ping").json()
    assert answer["app"] == "clip-mcp-editor"
    assert answer["workspace"] == os.path.normcase(os.path.abspath(server.WORKSPACE_DIR))

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
    assert (result["chosen"], result["imported"], result["failed"], len(result["assets"])) == (1, 1, [], 1)
    listed = {asset["path"] for asset in client.get("/api/assets").json()["assets"]}
    assert str(original.resolve()) in {str(Path(path).resolve()) for path in listed}
    assert set(os.listdir(server.WORKSPACE_DIR)) - before <= {"cache"}

def test_a_cancelled_window_imports_nothing(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(dialogs, "pick", lambda kind: [])
    assert client.post("/api/pick", json={"kind": "folder"}).json() == {"chosen": 0, "imported": 0, "failed": [], "assets": []}

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

def test_the_version_can_be_asked_alone_to_notice_the_ai(client: TestClient) -> None:
    made = client.post("/api/projects", json={"name": "版本"}).json()
    assert client.get(f"/api/projects/{made['id']}/version").json() == {"version": server.repo.get_project(made["id"]).version}

def test_words_say_when_nothing_was_transcribed(client: TestClient, media: Path) -> None:
    asset = server.import_asset(str(media / "silent.mp4"))["id"]
    assert client.get(f"/api/words/{asset}?at=1").json() == {"words": [], "transcribed": False}

def test_an_empty_project_has_no_preview_to_make(client: TestClient) -> None:
    made = client.post("/api/projects", json={"name": "空的"}).json()
    assert client.post(f"/api/projects/{made['id']}/preview").json()["empty"] is True

def test_the_editor_watches_a_720_preview_of_the_version_it_shows(client: TestClient, media: Path) -> None:
    import subprocess
    import time

    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    made = client.post("/api/projects", json={"name": "預覽", "width": 1920, "height": 1080}).json()
    version = server.repo.get_project(made["id"]).version
    client.post(f"/api/projects/{made['id']}/edits", json={"expected_version": version, "operations": [
        {"action": "insert_clip", "track_id": "video", "clip_id": "c1", "asset_id": asset,
         "source_range": {"start": 0, "end": 2}},
    ]})
    started = client.post(f"/api/projects/{made['id']}/preview").json()
    assert started["version"] == version + 1
    # Asking again for the same version does not start another render.
    assert client.post(f"/api/projects/{made['id']}/preview").json()["job_id"] == started["job_id"]
    deadline = time.monotonic() + 120
    state = started
    while state.get("status") not in ("completed", "failed") and time.monotonic() < deadline:
        time.sleep(0.3)
        state = client.get(f"/api/projects/{made['id']}/preview").json()
    assert state["status"] == "completed", state
    path = server.job_manager.get_job(state["job_id"]).output_path
    size = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    assert size == "1280,720"
    assert client.get(state["url"]).status_code == 200

def captioned_project(client: TestClient, media: Path, footage: Path = None) -> tuple:
    """A project of one clip from 2 s into a file, with one caption on it.

    Args:
        client: The editor.
        media: The test media folder.
        footage: The file to use; `wide.mp4` when not given.

    Returns:
        The project's ID and the caption's ID.
    """
    asset = server.import_asset(str(footage or media / "wide.mp4"))["id"]
    made = client.post("/api/projects", json={"name": "字幕"}).json()
    version = server.repo.get_project(made["id"]).version
    client.post(f"/api/projects/{made['id']}/edits", json={"expected_version": version, "operations": [
        {"action": "insert_clip", "track_id": "video", "clip_id": "c1", "asset_id": asset,
         "source_range": {"start": 2, "end": 6}},
        {"action": "set_subtitles", "cues": [
            {"id": "q1", "asset_id": asset, "source_start": 3, "source_end": 4, "text": "我們出發吧"},
        ]},
    ]})
    return made["id"], "q1"

def test_captions_come_back_where_they_fall_in_the_cut(client: TestClient, media: Path) -> None:
    project, cue = captioned_project(client, media)
    listed = client.get(f"/api/projects/{project}/captions").json()["captions"]
    assert listed == [{"cue_id": cue, "start": 1.0, "end": 2.0, "text": "我們出發吧"}]

def test_a_caption_corrected_in_the_editor_is_corrected_in_the_project(client: TestClient, media: Path) -> None:
    project, cue = captioned_project(client, media)
    version = server.repo.get_project(project).version
    answer = client.post(f"/api/projects/{project}/edits", json={"expected_version": version, "operations": [
        {"action": "edit_subtitle", "cue_id": cue, "text": "我們出發囉"},
    ]})
    assert answer.status_code == 200
    assert client.get(f"/api/projects/{project}/captions").json()["captions"][0]["text"] == "我們出發囉"

def test_captioning_asks_before_transcribing_and_says_if_the_model_must_download(
    client: TestClient, media: Path, tmp_path: Path, monkeypatch,
) -> None:
    # A copy of its own, which no other test can have transcribed.
    fresh = tmp_path / "fresh.mp4"
    fresh.write_bytes((media / "wide.mp4").read_bytes())
    project, _ = captioned_project(client, media, fresh)
    asked = client.post(f"/api/projects/{project}/captions/make", json={"expected_version": 0}).json()
    assert [item["name"] for item in asked["needs"]] == ["fresh.mp4"]
    assert asked["replaces"] == 1
    assert isinstance(asked["model_missing"], bool)
    started = []
    monkeypatch.setattr(server, "analyze_asset", lambda ids, *rest: started.append((ids, rest)) or {"jobs": [{"job_id": "j1"}]})
    assert client.post(f"/api/projects/{project}/captions/make", json={"transcribe": True}).json() == {"jobs": ["j1"]}
    assert started[0][1][3] == "zh-TW"

def test_captioning_footage_already_transcribed_goes_straight_to_captions(client: TestClient, media: Path, monkeypatch) -> None:
    project, _ = captioned_project(client, media)
    monkeypatch.setattr(editor, "_untranscribed", lambda found: [])
    monkeypatch.setattr(server, "generate_subtitles", lambda project_id, version: {"new_version": version + 1, "captions": ["a", "b"]})
    answer = client.post(f"/api/projects/{project}/captions/make", json={"expected_version": 7}).json()
    assert answer == {"new_version": 8, "count": 2}

def test_a_chosen_song_comes_back_as_an_asset(client: TestClient, media: Path, monkeypatch) -> None:
    monkeypatch.setattr(dialogs, "pick", lambda kind: [str(media / "song.mp3")] if kind == "music" else [])
    answer = client.post("/api/pick", json={"kind": "music"}).json()
    assert answer["imported"] == 1 and len(answer["assets"]) == 1

def test_the_check_before_export_is_the_servers_own(client: TestClient, media: Path, monkeypatch) -> None:
    project, _ = captioned_project(client, media)
    asked = []
    monkeypatch.setattr(server, "check_render", lambda *args: asked.append(args) or {"ok": True, "findings": []})
    assert client.get(f"/api/projects/{project}/check?frame=portrait&captions=1").json()["ok"] is True
    assert asked == [(project, "portrait", True)]

def test_an_export_goes_ahead_only_over_what_the_user_was_shown(client: TestClient, media: Path, monkeypatch) -> None:
    project, _ = captioned_project(client, media)
    asked = []
    def fake_render(*args):
        asked.append(args)
        return {"job_id": "missing-job"}
    monkeypatch.setattr(server, "render_project", fake_render)
    answer = client.post(f"/api/projects/{project}/export", json={"frame": "", "captions": False, "allow": ["mid_speech"]})
    assert answer.status_code == 200
    project_id, preview, _, captions, frame, _, allow = asked[0]
    assert (preview, captions, frame, allow) == (False, False, None, ["mid_speech"])
    assert client.get(f"/api/projects/{project}/export").json()["job_id"] == "missing-job"

def test_a_finished_export_is_saved_where_the_user_keeps_videos(client: TestClient, media: Path) -> None:
    import time

    project, _ = captioned_project(client, media)
    client.post(f"/api/projects/{project}/edits", json={"expected_version": server.repo.get_project(project).version,
                                                        "operations": [{"action": "rename_project", "name": "交付"}]})
    findings = client.get(f"/api/projects/{project}/check").json()["findings"]
    kinds = sorted({finding["check"] for finding in findings})
    state = client.post(f"/api/projects/{project}/export", json={"allow": kinds}).json()
    deadline = time.monotonic() + 120
    while state.get("status") not in ("completed", "failed") and time.monotonic() < deadline:
        time.sleep(0.3)
        state = client.get(f"/api/projects/{project}/export").json()
    assert state["status"] == "completed", state
    assert state["file"].startswith("交付") and state["folder"] == str(server.output_dir())

def test_only_a_finished_render_can_be_opened(client: TestClient) -> None:
    assert client.post("/api/reveal", json={"job_id": "nothing"}).status_code == 404


def test_only_cherry_studio_is_connected_by_opening_a_link(client: TestClient, monkeypatch) -> None:
    opened = []
    monkeypatch.setattr(os, "startfile", opened.append, raising=False)
    assert client.post("/api/clients/open", json={"key": "codex"}).status_code == 404
    monkeypatch.setattr(editor.clients, "register_cherry_studio",
                        lambda command, dry_run: editor.clients.Registration("Cherry Studio", "confirm", "cherrystudio://x"))
    assert client.post("/api/clients/open", json={"key": "cherry-studio"}).status_code == 200
    assert opened == ["cherrystudio://x"]


def test_the_page_connects_and_disconnects_only_the_client_it_names(client: TestClient, monkeypatch) -> None:
    done = []
    monkeypatch.setattr(editor.clients, "register",
                        lambda command, only=None, dry_run=False: done.append(("connect", only, dry_run))
                        or [editor.clients.Registration("x", "unchanged", "")] * len(only or editor.clients.CLIENTS))
    monkeypatch.setattr(editor.clients, "unregister",
                        lambda only=None, dry_run=False: done.append(("disconnect", only, dry_run))
                        or [editor.clients.Registration("x", "removed", "")])
    assert client.post("/api/clients", json={}).status_code == 404
    assert client.post("/api/clients", json={"key": "cherry-studio"}).status_code == 404
    client.post("/api/clients", json={"key": "gemini-cli", "action": "connect"})
    client.post("/api/clients", json={"key": "codex", "action": "disconnect"})
    writes = [entry for entry in done if not entry[2]]
    assert writes == [("connect", ["gemini-cli"], False), ("disconnect", ["codex"], False)]
