"""What the user says about a moment of a cut follows that footage, waits for the AI, and is theirs to delete."""

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client
from starlette.testclient import TestClient

from app import runtime, server
from app.tools import editing, versioning
from helpers import build_project, edit, insert, video_track


def call(name: str, arguments: dict):
    """Call a tool the way a client does; returns the structured result or the error's message."""
    async def go():
        async with Client(runtime.mcp) as client:
            result = await client.call_tool(name, arguments, raise_on_error=False)
            return result.content[0].text if result.is_error else result.structured_content
    return asyncio.run(go())


def two_shots(media: Path) -> str:
    """A cut of two shots: 0–4s of one file, then 2–6s of another, four seconds each."""
    first = server.import_asset(str(media / "wide.mp4"))["id"]
    second = server.import_asset(str(media / "tall.mp4"))["id"]
    return build_project([video_track(), insert("a", first, 0, 4), insert("b", second, 2, 6)], name="留言測試")


def test_a_comment_follows_the_footage_it_was_written_on(media: Path) -> None:
    project = two_shots(media)
    # 5s into the cut is 3s into the second file.
    written = server.add_comment(project, "這裡太拖", 5.0)
    assert (written["clip_id"], written["file_seconds"], written["start"]) == ("b", 3.0, 5.0)

    # The first shot is cut to one second: the moment moves three seconds earlier with its footage.
    edit(project, [{"action": "trim_clip", "track_id": "main", "clip_id": "a",
                    "new_source_range": {"start": 0, "end": 1}}])
    assert server.get_comments(project)["comments"][0]["start"] == 2.0

    # A stretch keeps both its ends on their footage.
    stretch = server.add_comment(project, "配樂太大聲", 1.5, 3.5)
    assert (stretch["start"], stretch["end"]) == (1.5, 3.5)

    # The footage taken out, the comment says so rather than pointing at whatever is there now.
    edit(project, [{"action": "delete_clip", "track_id": "main", "clip_id": "b"}])
    assert server.get_comments(project)["comments"][0]["gone"] is True


def test_the_ai_is_reminded_closes_it_with_what_it_did_and_cannot_delete_it(media: Path) -> None:
    project = two_shots(media)
    written = server.add_comment(project, "開頭太長", 1.0)
    # Anything the AI does to the video says the comment is waiting.
    looked = call("get_project", {"project_id": project})
    assert looked["open_comments"]["count"] == 1

    closed = call("resolve_comment", {"project_id": project, "comment_id": written["id"], "reply": "開頭縮短 2 秒"})
    assert closed["status"] == "resolved" and closed["replies"][0]["text"] == "開頭縮短 2 秒"
    assert closed["replies"][0]["by"].startswith("AI")
    assert "open_comments" not in call("get_project", {"project_id": project})
    assert server.get_comments(project)["comments"] == []
    assert len(server.get_comments(project, include_resolved=True)["comments"]) == 1
    tools = {tool.name for tool in asyncio.run(runtime.mcp.list_tools())}
    assert {"get_comments", "resolve_comment"} <= tools and not any("delete" in name and "comment" in name
                                                                     for name in tools)


def test_the_editor_writes_reopens_and_deletes_comments(media: Path) -> None:
    from app.ui import app as editor

    project = two_shots(media)
    client = TestClient(editor.create_app())
    made = client.post(f"/api/projects/{project}/comments", json={"text": "字幕太小", "start": 6.0}).json()
    url = f"/api/projects/{project}/comments/{made['id']}"
    client.post(url, json={"status": "resolved"})
    reopened = client.post(url, json={"status": "open", "reply": "還是太小"}).json()
    assert reopened["status"] == "open" and reopened["replies"][0]["by"] == "你"
    listed = client.get(f"/api/projects/{project}/comments").json()
    assert [item["id"] for item in listed["comments"]] == [made["id"]] and listed["open"] == 1
    assert client.post(url, json={"delete": True}).json() == {"deleted": made["id"]}
    assert client.get(f"/api/projects/{project}/comments").json()["comments"] == []


def test_comments_are_kept_in_the_history_beside_the_project(media: Path, kept_history) -> None:
    project = two_shots(media)
    written = server.add_comment(project, "換一首歌", 2.0)
    kept = kept_history.read(f"comments/{project}.json", kept_history.head())
    assert json.loads(kept)[0]["text"] == "換一首歌"
    # Closed, it says which version it was dealt with in.
    closed = server.change_comment(project, written["id"], "AI（測試）", "換成 Sunny", "resolved")
    assert closed["resolved_in"] == versioning.project_history(project)["versions"][0]["commit"]


def test_the_ai_asks_what_it_could_not_decide_on_its_moment_and_hears_the_answer(media: Path) -> None:
    from app.models.comment import Question

    project = two_shots(media)
    asked = server.ask_user(project, [Question(start=0.5, end=3.5, question="開頭 3 秒夠抓人嗎？",
                                               why="開場是試印失敗的畫面，沒有人講話")])["asked"]
    assert asked[0]["question"] is True and asked[0]["answered"] is False
    assert asked[0]["by"].startswith("AI") and "（開場是試印失敗的畫面" in asked[0]["text"]
    # Until the user answers, it is waiting on them, not on the AI.
    assert "open_comments" not in call("get_project", {"project_id": project})
    server.change_comment(project, asked[0]["id"], "你", "可以再短一點")
    assert call("get_project", {"project_id": project})["open_comments"]["count"] == 1
    [question] = server.get_comments(project)["comments"]
    assert question["answered"] is True and question["replies"][0]["text"] == "可以再短一點"


def test_a_preference_said_once_comes_back_with_every_project_until_it_is_forgotten() -> None:
    kept = server.remember_preference("開場不要超過三秒")["kept"]
    try:
        assert {"id": kept["id"], "text": "開場不要超過三秒"} in editing.list_projects()["preferences"]
        made = editing.create_project("下一支")
        assert kept in made["preferences"] and kept in editing.get_project(made["id"])["preferences"]
    finally:
        assert server.forget_preference(kept["id"]) == {"forgotten": kept["id"]}
    assert kept not in editing.list_projects().get("preferences", [])
    with pytest.raises(ValueError, match="there is no preference"):
        server.forget_preference(kept["id"])
