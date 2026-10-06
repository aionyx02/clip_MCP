"""Every version of a cut is kept, can be read back, gone back to, and branched from."""

import asyncio
import json
from pathlib import Path

from fastmcp import Client

from app import server
from app.storage.history import History, canonical, describe_project
from helpers import build_project, edit, insert, video_track


def call(name: str, arguments: dict):
    """Call a tool the way a client does; returns the structured result or the error's message."""
    async def go():
        async with Client(mcp_server()) as client:
            result = await client.call_tool(name, arguments, raise_on_error=False)
            return result.content[0].text if result.is_error else result.structured_content
    return asyncio.run(go())


def mcp_server():
    return server.mcp


def test_a_file_is_written_the_same_way_every_time() -> None:
    written = canonical({"b": 104.71668866666667, "a": [1.0, 2.5]})
    assert written.decode("utf-8") == '{\n  "a": [\n    1,\n    2.5\n  ],\n  "b": 104.717\n}\n'


def test_one_step_is_one_commit_however_many_files_it_wrote(tmp_path: Path) -> None:
    history = History(str(tmp_path / "history"), snapshot=lambda: {"library.json": b"{}\n"})
    with history.step("AI（Claude Code）", note="結尾的晚安多留一下"):
        history.write("projects/a-1.json", b"1\n", "改了 1 句字幕")
        history.write("plans/1.json", b"2\n", "計畫第 2 版")
    versions = history.versions(["projects", "plans", "library.json"])
    # The first version is how things stood when keeping them started.
    assert [version.subject for version in versions] == ["結尾的晚安多留一下", "開始記錄版本"]
    assert versions[0].author == "AI（Claude Code）"
    assert "改了 1 句字幕" in versions[0].body and "計畫第 2 版" in versions[0].body


def test_writing_what_is_already_there_keeps_no_version(tmp_path: Path) -> None:
    history = History(str(tmp_path / "history"))
    history.write("a.json", b"1\n")
    history.write("a.json", b"1\n")
    assert len(history.versions(["a.json"])) == 1


def test_a_change_is_said_in_words() -> None:
    before = {"name": "抽杯架", "tracks": [], "subtitles": [{"id": "c1", "text": "第三版"}]}
    after = {"name": "抽杯架", "tracks": [], "subtitles": [{"id": "c1", "text": "第三、四版"}]}
    assert describe_project(before, after) == "改了 1 句字幕"
    assert describe_project(None, after) == "建立了專案「抽杯架」"


def test_every_change_to_a_project_is_a_version_with_who_and_what(kept_history, media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = call("create_project", {"name": "版本測試"})["id"]
    call("apply_edits", {"project_id": project, "expected_version": 1, "note": "先放兩段", "operations": [
        video_track(), insert("a", asset, 0, 3), insert("b", asset, 4, 6)]})
    call("apply_edits", {"project_id": project, "expected_version": 2, "operations": [
        {"action": "rename_project", "name": "版本測試改名"}]})
    listed = call("project_history", {"project_id": project})["versions"]
    assert [version["what"] for version in listed[:3]] == ["改名為「版本測試改名」", "先放兩段", "建立了專案「版本測試」"]
    assert all(version["by"].startswith("AI（") for version in listed[:3])
    # Renaming the project renamed its file, and the history still follows it.
    assert kept_history.working_path("projects/", f"版本測試改名-{project}.json")


def test_going_back_is_one_more_version_and_can_be_undone(kept_history, media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], name="回到舊版")
    first = call("project_history", {"project_id": project})["versions"][0]["commit"]
    edit(project, [insert("b", asset, 4, 6)])
    current = server.get_project(project)["version"]
    restored = call("restore_version", {"project_id": project, "commit": first, "expected_version": current})
    assert [clip.id for clip in server.repo.get_project(project).tracks[0].clips] == ["a"]
    assert restored["plan_copied"] is False
    listed = call("project_history", {"project_id": project})["versions"]
    assert listed[0]["what"].startswith("回到「") and len(listed) == 4
    # The version that was gone back over is still there to go back to.
    again = call("restore_version", {"project_id": project, "commit": listed[1]["commit"],
                                     "expected_version": restored["new_version"]})
    assert [clip.id for clip in server.repo.get_project(project).tracks[0].clips] == ["a", "b"]
    assert again["new_version"] == restored["new_version"] + 1


def test_a_branch_is_a_project_of_its_own_beside_the_first(kept_history, media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], name="原本那支")
    branched = call("branch_project", {"project_id": project, "name": "原本那支 Reels 版"})
    branch = server.repo.get_project(branched["project_id"])
    assert branch.name == "原本那支 Reels 版" and branch.branched_from.project_id == project
    edit(branch.id, [{"action": "rename_project", "name": "Reels 版改過"}])
    assert server.repo.get_project(project).name == "原本那支"
    assert call("project_history", {"project_id": branch.id})["branched_from"]["project_id"] == project


def test_a_change_made_in_the_editor_says_so(kept_history, media: Path) -> None:
    from starlette.testclient import TestClient
    from app.ui import app as editor

    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], name="編輯器改的")
    response = TestClient(editor.create_app()).post(f"/api/projects/{project}/edits", json={
        "expected_version": 2, "operations": [{"action": "rename_project", "name": "編輯器改過"}]})
    assert response.status_code == 200, response.text
    newest = call("project_history", {"project_id": project})["versions"][0]
    assert newest["by"] == "編輯器" and newest["what"] == "改名為「編輯器改過」"


def test_the_history_never_signs_with_the_user_s_key(kept_history, media: Path) -> None:
    from dulwich.repo import Repo

    project = build_project([video_track()], name="不簽章")
    with Repo(kept_history.root) as repo:
        assert repo[repo.head()].gpgsig is None
    assert json.loads(kept_history.read(kept_history.working_path("projects/", f"{project}.json"),
                                        kept_history.head()))["name"] == "不簽章"


from test_edit_plan import planned  # noqa: E402,F401,F811  (the fixture, used by name below)


def test_going_back_leaves_another_video_compiled_from_the_same_plan_alone(kept_history, planned: dict) -> None:  # noqa: F811
    from app.models.plan import EditPlan

    portrait = server.create_project("直式版", width=1080, height=1920)["id"]
    landscape = server.create_project("橫式版")["id"]
    server.compile_plan(portrait, 1, planned["plan_id"])
    server.compile_plan(landscape, 1, planned["plan_id"])
    first = call("project_history", {"project_id": portrait})["versions"][0]["commit"]
    # The plan changes, and the portrait cut is compiled from it again.
    stored = EditPlan.model_validate(server.get_plan(planned["plan_id"])["plan"])
    server.save_plan(stored.model_copy(update={"goal": "改過的目標"}))
    server.compile_plan(portrait, 2, planned["plan_id"])

    restored = call("restore_version", {"project_id": portrait, "commit": first, "expected_version": 3})
    assert restored["plan_copied"] is True
    copy = {clip.from_plan_id for track in server.repo.get_project(portrait).tracks for clip in track.clips
            if clip.from_plan_id}
    assert copy and planned["plan_id"] not in copy
    assert server.repo.get_plan(copy.pop()).goal == "做一支短的"
    # The landscape cut and the shared plan are as they were.
    assert server.repo.get_plan(planned["plan_id"]).goal == "改過的目標"
    assert all(clip.from_plan_id == planned["plan_id"]
               for track in server.repo.get_project(landscape).tracks for clip in track.clips if clip.from_plan_id)


def test_the_version_panel_folds_one_sitting_into_one_dot(kept_history, media: Path) -> None:
    from app.models.job import Job, JobKind, JobStatus

    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], name="版本面板")
    for name in ("第二版", "第三版"):
        edit(project, [{"action": "rename_project", "name": name}])
    nodes = server.version_graph(project)["nodes"]
    # Creating it and every change after by the same AI, nothing rendered between: one dot.
    assert len(nodes) == 1 and len(nodes[0]["versions"]) > 3
    assert nodes[0]["thumb"]["asset_id"] == asset

    # Rendering one out marks it, and starts a new dot after it.
    rendered = server.repo.get_project(project).version
    server.repo.add_job(Job(kind=JobKind.RENDER, project_id=project, project_version=rendered,
                            status=JobStatus.COMPLETED, output_path=str(media / "out.mp4")))
    edit(project, [{"action": "rename_project", "name": "第四版"}])
    nodes = server.version_graph(project)["nodes"]
    assert [node["marks"] for node in nodes] == [[], ["exported"]]
    listed = call("project_history", {"project_id": project})["versions"]
    assert [version["marks"] for version in listed[:len(nodes[0]["versions"]) + 1]][-1] == ["exported"]

    # A star is the user's own, and a branch grows from the dot it started at.
    starred = call("mark_version", {"project_id": project, "commit": nodes[1]["commit"][:10], "mark": "starred"})
    assert starred["marks"] == ["exported", "starred"]
    branch = call("branch_project", {"project_id": project, "name": "版本面板短版", "commit": nodes[1]["commit"]})
    graph = server.version_graph(project)
    assert graph["branches"][0]["project_id"] == branch["project_id"]
    assert graph["branches"][0]["from"] == nodes[1]["commit"]
    assert server.version_graph(branch["project_id"])["branched_from"]["name"] == "第四版"
