"""Tests for the editor's web surface: what it shows, and what it refuses to do."""

import json
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
                        lambda only=None, dry_run=False, owner=None: done.append(("disconnect", only, dry_run))
                        or [editor.clients.Registration("x", "removed", "")])
    assert client.post("/api/clients", json={}).status_code == 404
    assert client.post("/api/clients", json={"key": "cherry-studio"}).status_code == 404
    client.post("/api/clients", json={"key": "gemini-cli", "action": "connect"})
    client.post("/api/clients", json={"key": "codex", "action": "disconnect"})
    writes = [entry for entry in done if not entry[2]]
    assert writes == [("connect", ["gemini-cli"], False), ("disconnect", ["codex"], False)]


# --- folders, and taking files out of the library ----------------------------------------

def _copy(media: Path, tmp_path: Path, name: str) -> str:
    """A file of its own to import, so taking it out never touches what other tests use."""
    copy = tmp_path / name
    copy.write_bytes((media / "wide.mp4").read_bytes())
    return server.import_asset(str(copy))["id"]

def _project_using(client: TestClient, asset_id: str, name: str) -> str:
    made = client.post("/api/projects", json={"name": name}).json()
    project = server.repo.get_project(made["id"])
    client.post(f"/api/projects/{made['id']}/edits", json={"expected_version": project.version, "operations": [
        {"action": "insert_clip", "track_id": "video", "clip_id": "c1", "asset_id": asset_id,
         "source_range": {"start": 0, "end": 2}},
    ]})
    return made["id"]

def _folder(client: TestClient, kind: str, name: str, parent_id=None) -> str:
    answer = client.post("/api/folders", json={"action": "create", "kind": kind, "name": name, "parent_id": parent_id})
    assert answer.status_code == 200, answer.text
    return answer.json()["id"]

def test_folders_nest_and_keep_their_names_apart(client: TestClient) -> None:
    trips = _folder(client, "projects", "旅行 資料夾")
    tainan = _folder(client, "projects", "台南", trips)
    assert client.post("/api/folders", json={"action": "create", "kind": "projects", "name": "台南", "parent_id": trips}).status_code == 400
    assert client.post("/api/folders", json={"action": "create", "kind": "projects", "name": "  "}).status_code == 400
    # The same name elsewhere, or for the other kind, is fine.
    _folder(client, "projects", "台南")
    _folder(client, "assets", "旅行 資料夾")
    assert client.post("/api/folders", json={"action": "create", "kind": "assets", "name": "x", "parent_id": tainan}).status_code == 404
    assert client.post("/api/folders", json={"action": "move", "id": trips, "parent_id": tainan}).status_code == 400
    listed = {folder["id"]: folder for folder in client.get("/api/folders?kind=projects").json()["folders"]}
    assert listed[tainan]["parent_id"] == trips
    assert client.post("/api/folders", json={"action": "rename", "id": tainan, "name": "台南 2026"}).status_code == 200
    assert client.get("/api/folders?kind=other").status_code == 400

def test_removing_a_folder_moves_what_was_in_it_up_and_deletes_nothing(client: TestClient) -> None:
    outer = _folder(client, "projects", "外層")
    inner = _folder(client, "projects", "內層", outer)
    made = client.post("/api/projects", json={"name": "收好的"}).json()["id"]
    assert client.post("/api/file", json={"kind": "projects", "ids": [made], "folder_id": inner}).status_code == 200
    assert next(item for item in client.get("/api/projects").json()["projects"] if item["id"] == made)["folder_id"] == inner
    client.post("/api/folders", json={"action": "delete", "id": inner})
    assert next(item for item in client.get("/api/projects").json()["projects"] if item["id"] == made)["folder_id"] == outer
    client.post("/api/folders", json={"action": "delete", "id": outer})
    assert next(item for item in client.get("/api/projects").json()["projects"] if item["id"] == made)["folder_id"] is None
    assert server.repo.get_project(made) is not None

def test_filing_refuses_a_folder_of_the_other_kind_or_something_unknown(client: TestClient, media: Path, tmp_path: Path) -> None:
    asset = _copy(media, tmp_path, "收.mp4")
    projects_folder = _folder(client, "projects", "只放專案")
    assert client.post("/api/file", json={"kind": "assets", "ids": [asset], "folder_id": projects_folder}).status_code == 404
    assert client.post("/api/file", json={"kind": "assets", "ids": ["nothing"], "folder_id": None}).status_code == 404
    assets_folder = _folder(client, "assets", "素材分類")
    assert client.post("/api/file", json={"kind": "assets", "ids": [asset], "folder_id": assets_folder}).status_code == 200
    listed = {item["id"]: item for item in client.get("/api/assets").json()["assets"]}
    assert listed[asset]["folder_id"] == assets_folder

def test_files_chosen_with_a_folder_open_are_filed_in_it(client: TestClient, media: Path, tmp_path: Path, monkeypatch) -> None:
    copy = tmp_path / "選進資料夾.mp4"
    copy.write_bytes((media / "tall.mp4").read_bytes())
    folder = _folder(client, "assets", "選進來")
    monkeypatch.setattr(dialogs, "pick", lambda kind: [str(copy)])
    result = client.post("/api/pick", json={"kind": "files", "folder_id": folder}).json()
    assert server.repo.folder_of("assets")[result["assets"][0]] == folder

def test_a_file_a_project_uses_cannot_be_taken_out_and_the_project_is_named(client: TestClient, media: Path, tmp_path: Path) -> None:
    asset = _copy(media, tmp_path, "還在用.mp4")
    _project_using(client, asset, "用著它的專案")
    answer = client.post("/api/assets/delete", json={"ids": [asset], "recycle": True})
    assert answer.status_code == 409
    assert answer.json()["in_use"] == [{"name": "還在用.mp4", "projects": ["用著它的專案"], "plans": []}]
    assert server.repo.get_assets([asset]) and (tmp_path / "還在用.mp4").exists()

def test_taking_a_file_out_keeps_the_original_unless_asked(client: TestClient, media: Path, tmp_path: Path, monkeypatch) -> None:
    recycled = []
    monkeypatch.setattr(editor.housekeeping, "to_recycle_bin", lambda paths: recycled.extend(paths) or True)
    kept = _copy(media, tmp_path, "留著原檔.mp4")
    folder = _folder(client, "assets", "要清掉的")
    server.repo.file_items("assets", [kept], folder)
    answer = client.post("/api/assets/delete", json={"ids": [kept], "recycle": False})
    assert answer.status_code == 200 and answer.json()["removed"] == [kept]
    assert not server.repo.get_assets([kept]) and kept not in server.repo.folder_of("assets")
    assert (tmp_path / "留著原檔.mp4").exists() and recycled == []
    gone = _copy(media, tmp_path, "丟回收筒.mp4")
    assert client.post("/api/assets/delete", json={"ids": [gone], "recycle": True}).status_code == 200
    assert recycled == [str((tmp_path / "丟回收筒.mp4").resolve())] or recycled == [os.path.abspath(tmp_path / "丟回收筒.mp4")]

def test_a_file_whose_original_is_gone_is_listed_so_it_can_be_taken_out(client: TestClient, media: Path, tmp_path: Path) -> None:
    asset = _copy(media, tmp_path, "不見了.mp4")
    (tmp_path / "不見了.mp4").unlink()
    assert asset not in {item["id"] for item in client.get("/api/assets").json()["assets"]}
    listed = {item["id"]: item for item in client.get("/api/assets?all=1").json()["assets"]}
    assert listed[asset]["missing"] is True
    assert client.post("/api/assets/delete", json={"ids": [asset], "recycle": True}).status_code == 200
    assert asset not in {item["id"] for item in client.get("/api/assets?all=1").json()["assets"]}

def test_a_file_taken_out_leaves_the_rest_of_a_timeline_that_covered_it(tmp_path: Path) -> None:
    from app.models.media import Asset
    from app.models.semantic import SemanticTimeline
    from app.storage.repo import Repository

    store = Repository(str(tmp_path / "library.db"))
    for asset_id in ("a", "b"):
        store.save_asset(Asset(id=asset_id, path=str(tmp_path / f"{asset_id}.mp4"), has_video=True, has_audio=True))
    store.save_semantic_timeline(SemanticTimeline(id="t", asset_ids=["a", "b"], input_hash="h", derivation_version=1), [])
    folder = store.create_folder("f", "assets", "放這")["id"]
    store.file_items("assets", ["a", "b"], folder)
    assert store.delete_unused_assets(["a"]) == {}
    assert set(store.get_assets(["a", "b"])) == {"b"}
    assert store.get_semantic_timeline("t").asset_ids == ["b"]
    assert store.folder_of("assets") == {"b": folder}

def test_a_file_the_ais_plan_uses_cannot_be_taken_out(tmp_path: Path) -> None:
    from app.models.media import Asset, Span
    from app.models.semantic import ClipKind, ClipLevel, SemanticClip, SemanticTimeline
    from app.storage.repo import Repository

    store = Repository(str(tmp_path / "planned.db"))
    for asset_id in ("picture", "song", "spare"):
        store.save_asset(Asset(id=asset_id, path=str(tmp_path / f"{asset_id}.mp4"), has_video=True, has_audio=True))
    store.save_semantic_timeline(
        SemanticTimeline(id="t", asset_ids=["picture", "spare"], input_hash="h", derivation_version=1),
        [SemanticClip(id="t:u0000", timeline_id="t", asset_id="picture", level=ClipLevel.UTTERANCE,
                      source_range=Span(start=0, end=2), kind=ClipKind.SPEECH, text="開場")])
    with store._transaction() as conn:
        conn.execute("INSERT INTO plans (id, timeline_id, version, data) VALUES (?, ?, ?, ?)", ("p", "t", 1, json.dumps(
            {"id": "p", "goal": "台南旅行 60 秒", "beats": [{"selections": [{"clip_id": "t:u0000"}]}],
             "music": {"cues": [{"asset_id": "song"}]}})))
    using = store.delete_unused_assets(["picture", "song", "spare"])
    assert using == {"picture": {"projects": [], "plans": ["台南旅行 60 秒"], "busy": []},
                     "song": {"projects": [], "plans": ["台南旅行 60 秒"], "busy": []}}
    # Refused as a whole: the one nothing uses is still there too.
    assert set(store.get_assets(["picture", "song", "spare"])) == {"picture", "song", "spare"}

def test_a_timeline_left_with_no_file_goes_with_the_last_one(tmp_path: Path) -> None:
    from app.models.media import Asset
    from app.models.semantic import SemanticTimeline
    from app.storage.repo import Repository

    store = Repository(str(tmp_path / "alone.db"))
    store.save_asset(Asset(id="only", path=str(tmp_path / "only.mp4"), has_video=True, has_audio=True))
    store.save_semantic_timeline(SemanticTimeline(id="t", asset_ids=["only"], input_hash="h", derivation_version=1), [])
    assert store.delete_unused_assets(["only"]) == {}
    assert store.get_semantic_timeline("t") is None

# --- deleting projects -------------------------------------------------------------------

def test_deleting_projects_takes_them_and_their_previews_and_keeps_what_was_delivered(
    client: TestClient, tmp_path: Path,
) -> None:
    from app.storage import housekeeping

    first, second = (server.create_project(name=f"要刪掉的{n}") for n in (1, 2))
    folder = _folder(client, "projects", "舊專案")
    server.repo.file_items("projects", [first["id"]], folder)
    previews = Path(housekeeping.preview_dir(server.WORKSPACE_DIR, first["id"])) / "old"
    previews.mkdir(parents=True)
    (previews / "preview.mp4").write_bytes(b"x")
    delivered = Path(server.output_dir()) / "要刪掉的1.mp4"
    delivered.parent.mkdir(parents=True, exist_ok=True)
    delivered.write_bytes(b"finished")

    answer = client.post("/api/projects/delete", json={"ids": [first["id"], second["id"]]})
    assert answer.status_code == 200 and sorted(answer.json()["deleted"]) == sorted([first["id"], second["id"]])
    assert server.repo.get_project(first["id"]) is None and server.repo.get_project(second["id"]) is None
    assert first["id"] not in server.repo.folder_of("projects")
    assert not previews.parent.exists()
    assert delivered.exists()

def test_a_project_being_rendered_is_not_deleted(client: TestClient, ended_after: list) -> None:
    from app.models.job import Job, JobKind, JobStatus

    project = server.create_project(name="輸出中")
    rendering = Job(kind=JobKind.RENDER, status=JobStatus.RUNNING, project_id=project["id"])
    server.repo.add_job(rendering)
    ended_after.append(rendering.job_id)
    answer = client.post("/api/projects/delete", json={"ids": [project["id"]]})
    assert answer.status_code == 409 and answer.json()["busy"] == ["輸出中"]
    assert server.repo.get_project(project["id"]) is not None

def test_the_ai_has_no_way_to_delete_a_project() -> None:
    import asyncio

    names = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
    assert not any("delete" in name and "project" in name for name in names)

# --- every job, whoever started it -------------------------------------------------------

def test_the_editor_sees_jobs_the_ai_started_with_what_they_are_for(
    client: TestClient, tmp_path: Path, ended_after: list,
) -> None:
    from app.models.job import Job, JobKind, JobStatus
    from app.models.media import Asset

    asset = Asset(id="job-asset", path=str(tmp_path / "訪談.mp4"), has_video=True, has_audio=True)
    server.repo.save_asset(asset)
    project = server.create_project(name="抽杯架 v3")
    analysing = Job(kind=JobKind.ANALYZE, status=JobStatus.RUNNING, asset_id=asset.id, progress=0.4)
    rendering = Job(kind=JobKind.RENDER, project_id=project["id"])
    for job in (analysing, rendering):
        server.repo.add_job(job)
        ended_after.append(job.job_id)
    listed = {job["job_id"]: job for job in client.get("/api/jobs/active").json()["jobs"]}
    assert listed[analysing.job_id]["name"] == "訪談.mp4" and listed[analysing.job_id]["progress"] == 0.4
    assert listed[rendering.job_id]["name"] == "抽杯架 v3" and listed[rendering.job_id]["status"] == "queued"

def test_a_job_can_be_stopped_from_the_editor(client: TestClient, ended_after: list) -> None:
    from app.models.job import Job, JobKind

    waiting = Job(kind=JobKind.RENDER, project_id=server.create_project(name="不要了")["id"])
    server.repo.add_job(waiting)
    ended_after.append(waiting.job_id)
    assert client.post("/api/jobs/cancel", json={"job_id": waiting.job_id}).status_code == 200
    assert server.repo.get_job(waiting.job_id).status.value == "cancelled"
    ended = {job["job_id"]: job for job in client.get("/api/jobs/active").json()["ended"]}
    assert ended[waiting.job_id]["status"] == "cancelled"
