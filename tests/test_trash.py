"""A deleted project waits in the trash, a video's other versions can be let go, and caches give room back."""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app import server
from app.models.job import Job, JobKind, JobStatus
from app.storage import housekeeping
from helpers import build_project, edit, insert, video_track


@pytest.fixture
def client() -> TestClient:
    from app.ui import app as editor

    return TestClient(editor.create_app())


def finished(project_id: str, version: int, name: str) -> Path:
    """A finished video of one version, as a full render leaves it."""
    path = Path(server.output_dir()) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"finished " + name.encode("utf-8"))
    server.repo.add_job(Job(kind=JobKind.RENDER, project_id=project_id, project_version=version,
                            status=JobStatus.COMPLETED, output_path=str(path)))
    return path


def test_a_deleted_project_waits_in_the_trash_and_comes_back_where_it_was(client: TestClient) -> None:
    project = server.create_project(name="垃圾桶裡的")["id"]
    folder = client.post("/api/folders", json={"action": "create", "kind": "projects", "name": "放回這裡"}).json()
    client.post("/api/file", json={"kind": "projects", "ids": [project], "folder_id": folder["id"]})

    assert client.post("/api/projects/delete", json={"ids": [project]}).status_code == 200
    assert server.repo.get_project(project) is None
    listed = client.get("/api/trash").json()
    waiting = next(item for item in listed["items"] if item["project_id"] == project)
    assert waiting["kind"] == "project" and waiting["days_left"] == listed["keep_days"] == 30

    back = client.post("/api/trash", json={"key": waiting["key"]}).json()
    assert back["project_id"] == project and server.repo.get_project(project).name == "垃圾桶裡的"
    assert server.repo.folder_of("projects")[project] == folder["id"]
    assert not any(item["key"] == waiting["key"] for item in client.get("/api/trash").json()["items"])


def test_the_trash_lets_go_after_thirty_days(tmp_path: Path) -> None:
    kept = housekeeping.put_in_trash(str(tmp_path), "project", "a" * 36, "很久以前", {"project.json": b"{}"}, [])
    housekeeping.empty_old_trash(str(tmp_path), now=datetime.now(timezone.utc) + timedelta(days=29))
    assert [item.key for item in housekeeping.in_trash(str(tmp_path))] == [kept.key]
    housekeeping.empty_old_trash(str(tmp_path), now=datetime.now(timezone.utc) + timedelta(days=31))
    assert housekeeping.in_trash(str(tmp_path)) == []


def test_other_versions_go_but_the_current_and_the_starred_stay(client: TestClient, media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], name="留下哪些")
    first = server.repo.get_project(project).version
    old = finished(project, first, "留下哪些 舊版.mp4")
    edit(project, [{"action": "rename_project", "name": "留下哪些"}, insert("b", asset, 4, 6)])
    current = finished(project, server.repo.get_project(project).version, "留下哪些 現在.mp4")

    found = client.get(f"/api/projects/{project}/storage").json()
    assert found["exported"] == 2 and [item["path"] for item in found["other_outputs"]] == [str(old)]

    # The name has to be typed right; then the old one goes to the trash and the current one stays.
    assert client.post(f"/api/projects/{project}/storage/trim", json={"name": "留下"}).status_code == 400
    assert old.exists()
    trimmed = client.post(f"/api/projects/{project}/storage/trim", json={"name": "留下哪些"}).json()
    assert trimmed["trashed"] == 1 and not old.exists() and current.exists()
    # Its version stays in the history, no longer exported.
    assert client.get(f"/api/projects/{project}/storage").json()["exported"] == 1

    # From the trash it goes back where it was.
    waiting = next(item for item in client.get("/api/trash").json()["items"]
                   if item["project_id"] == project and item["kind"] == "outputs")
    client.post("/api/trash", json={"key": waiting["key"]})
    assert old.exists()


def test_a_starred_version_keeps_its_finished_video(kept_history, client: TestClient, media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], name="標星號的")
    starred = server.project_history(project)["versions"][0]["commit"]
    finished(project, server.repo.get_project(project).version, "標星號的 星號.mp4")
    edit(project, [insert("b", asset, 4, 6)])
    server.mark_version(project, starred, "starred")
    assert client.get(f"/api/projects/{project}/storage").json()["other_outputs"] == []


def test_caches_give_room_back_oldest_first_when_the_disk_runs_low(tmp_path: Path) -> None:
    proxies = tmp_path / "cache" / "proxies"
    proxies.mkdir(parents=True)
    old, new = proxies / "old.mp4", proxies / "new.mp4"
    for path, age in ((old, 7200), (new, 10)):
        path.write_bytes(b"x" * 1000)
        os.utime(path, (time.time() - age, time.time() - age))
    # A floor just above what is free: one file is enough.
    free = __import__("shutil").disk_usage(tmp_path).free
    freed = housekeeping.give_cache_back(str(tmp_path), set(), floor=free + 1, headroom=0)
    assert freed == 1000 and not old.exists() and new.exists()
    # With room to spare nothing goes.
    assert housekeeping.give_cache_back(str(tmp_path), set(), floor=0) == 0 and new.exists()


def test_the_ai_files_footage_in_folders_and_deletes_nothing(media: Path) -> None:
    import asyncio
    from fastmcp import Client

    asset = server.import_asset(str(media / "tall.mp4"))["id"]

    async def call(arguments):
        async with Client(server.mcp) as client:
            return (await client.call_tool("organize_library", arguments)).structured_content

    # By file name, the way the AI reads the library.
    done = asyncio.run(call({"steps": [{"action": "create_folder", "name": "整理測試直式", "asset_ids": ["tall.mp4"]}]}))
    folder = next(item for item in done["folders"] if item["name"] == "整理測試直式")
    assert server.repo.folder_of("assets")[asset] == folder["id"]
    assert "整理測試直式" in done["done"][0]
    # There is no way for it to delete a folder or a file.
    assert not any(step for step in ("delete_folder", "delete") if step in str(server.LibraryStep))


def test_files_move_on_disk_only_once_agreed_and_never_over_another(media: Path, tmp_path: Path) -> None:
    import shutil

    source = tmp_path / "拍攝" / "要搬的.mp4"
    source.parent.mkdir()
    shutil.copy(media / "wide.mp4", source)
    asset = server.import_asset(str(source))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2)], name="搬檔")
    target = tmp_path / "整理好"
    target.mkdir()
    (target / "要搬的.mp4").write_bytes(b"already here")

    moves = [server.FileMove(asset_id=asset, to_folder=str(target))]
    shown = server.move_files(moves)
    planned = shown["planned"][0]
    # Nothing moves without agreement, and the taken name gets a number.
    assert source.exists() and planned["numbered"] and planned["to"].endswith("要搬的 (2).mp4")
    assert planned["projects"] == ["搬檔"]

    # Only the plan the user was shown is carried out.
    with pytest.raises(ValueError):
        server.move_files(moves, confirm_plan="something else")
    assert source.exists()
    moved = server.move_files(moves, confirm_plan=shown["plan"])["moved"][0]
    assert not source.exists() and (target / "要搬的.mp4").read_bytes() == b"already here"
    assert server.repo.get_assets([asset])[asset].path == moved["to"] == str(target / "要搬的 (2).mp4")
    # The project still plays it, from where it went.
    assert server.playback_of(server.repo.get_project(project))["tracks"][0]["clips"][0]["asset_id"] == asset


def test_a_folder_next_to_clip_mcp_s_own_is_not_taken_for_it(tmp_path: Path) -> None:
    assert server._inside(str(tmp_path / "work" / "a"), str(tmp_path / "work"))
    assert not server._inside(str(tmp_path / "work2"), str(tmp_path / "work"))


def test_the_ai_reading_the_trash_deletes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "output_dir", lambda: tmp_path)
    old = housekeeping.put_in_trash(str(tmp_path), "project", "b" * 36, "過期了", {"project.json": b"{}"}, [])
    meta = tmp_path / housekeeping.TRASH / old.key / "trashed.json"
    import json
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["trashed_at"] = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    meta.write_text(json.dumps(data), encoding="utf-8")
    assert [item["key"] for item in server.storage_usage()["trash"]] == [old.key]
    # The editor's own look at it is what lets it go.
    assert server.list_trash(expire=True)["items"] == []


def test_a_broken_project_in_the_trash_stays_there(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "output_dir", lambda: tmp_path)
    kept = housekeeping.put_in_trash(str(tmp_path), "project", "c" * 36, "壞掉的", {"project.json": b"not json"}, [])
    with pytest.raises(ValueError):
        server.restore_from_trash(kept.key)
    assert [item.key for item in housekeeping.in_trash(str(tmp_path))] == [kept.key]


def test_other_versions_of_a_video_with_no_name_are_not_let_go(media: Path) -> None:
    import uuid

    named = server.repo.get_project(server.create_project(name="有名字")["id"])
    nameless = named.model_copy(update={"id": str(uuid.uuid4()), "name": ""})
    server.repo.add_project(nameless)
    with pytest.raises(ValueError, match="name the video first"):
        server.trim_project(nameless.id, "")
