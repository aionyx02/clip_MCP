"""Every version of a cut is kept, can be read back, gone back to, and branched from."""

import asyncio
import json
from pathlib import Path

from fastmcp import Client

from app import runtime
from app.tools import catalog, editing, planning, versioning
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
    return runtime.mcp


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


def added_on(monkeypatch) -> list:
    """Note what each count after the first did: added onto the last (a dict) or counted from the start (None)."""
    seen = []
    adding = History._count_since

    def noting(self, *args):
        seen.append(adding(self, *args))
        return seen[-1]

    monkeypatch.setattr(History, "_count_since", noting)
    return seen


def test_counting_versions_after_a_commit_counts_only_what_is_new_and_agrees_with_a_full_count(
        tmp_path: Path, monkeypatch) -> None:
    seen = added_on(monkeypatch)
    root = str(tmp_path / "history")
    history = History(root)
    first = "projects/片A-11111111-1111-1111-1111-111111111111.json"
    second = "projects/片B-22222222-2222-2222-2222-222222222222.json"
    a, b = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"

    def agrees() -> dict:
        counted = history.counts("projects/")
        # A History that never counted before walks everything: what the remembered count must match.
        assert counted == History(root).counts("projects/")
        return counted

    history.write(first, b"1\n")
    history.write(second, b"1\n")
    assert agrees() == {a: 1, b: 1}
    history.write(first, b"2\n")
    assert agrees() == {a: 2, b: 1}
    with history.step("編輯器"):
        history.write(first, b"3\n")
        history.write(second, b"2\n")
    history.write("library.json", b"{}\n")
    assert agrees() == {a: 3, b: 2}
    # A project deleted is one more change to it; one put back as it was is one more again.
    history.write(second, None)
    assert agrees() == {a: 3, b: 3}
    history.write(second, b"2\n")
    assert agrees() == {a: 3, b: 4}
    assert len(seen) == 4 and None not in seen


def test_a_history_started_again_is_counted_from_the_start(tmp_path: Path, monkeypatch) -> None:
    import shutil

    seen = added_on(monkeypatch)
    root = tmp_path / "history"
    history = History(str(root))
    for content in (b"1\n", b"2\n", b"3\n"):
        history.write("projects/a-1.json", content)
    assert history.counts("projects/") == {"a-1": 3}
    # Made again from nothing, the commit last counted at is not in it: adding on would count what is gone.
    shutil.rmtree(root)
    history.write("projects/a-1.json", b"1\n")
    assert history.counts("projects/") == {"a-1": 1}
    assert seen == [None]


def test_a_change_is_said_in_words() -> None:
    before = {"name": "抽杯架", "tracks": [], "subtitles": [{"id": "c1", "text": "第三版"}]}
    after = {"name": "抽杯架", "tracks": [], "subtitles": [{"id": "c1", "text": "第三、四版"}]}
    assert describe_project(before, after) == "改了 1 句字幕"
    assert describe_project(None, after) == "建立了專案「抽杯架」"


def test_trimming_one_clip_counts_as_one_however_many_slide_after_it() -> None:
    def clip(end: float, at: float) -> dict:
        return {"asset_id": "a", "source_range": {"start": 0, "end": end}, "timeline_in": at}

    def project(*clips: dict) -> dict:
        return {"name": "片", "tracks": [{"clips": list(clips)}]}

    before = project(clip(4, 0), clip(4, 4), clip(4, 8))
    assert describe_project(before, project(clip(3, 0), clip(4, 3), clip(4, 7))) == "調整了 1 段剪接"
    assert describe_project(before, project(clip(4, 1), clip(4, 5), clip(4, 9))) == "調整了 3 段剪接"


def test_every_change_to_a_project_is_a_version_with_who_and_what(kept_history, media: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
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
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], name="回到舊版")
    first = call("project_history", {"project_id": project})["versions"][0]["commit"]
    edit(project, [insert("b", asset, 4, 6)])
    current = editing.get_project(project)["version"]
    restored = call("restore_version", {"project_id": project, "commit": first, "expected_version": current})
    assert [clip.id for clip in runtime.repo.get_project(project).tracks[0].clips] == ["a"]
    assert restored["plan_copied"] is False
    listed = call("project_history", {"project_id": project})["versions"]
    assert listed[0]["what"].startswith("回到「") and len(listed) == 4
    # The version that was gone back over is still there to go back to.
    again = call("restore_version", {"project_id": project, "commit": listed[1]["commit"],
                                     "expected_version": restored["new_version"]})
    assert [clip.id for clip in runtime.repo.get_project(project).tracks[0].clips] == ["a", "b"]
    assert again["new_version"] == restored["new_version"] + 1


def test_a_branch_is_a_project_of_its_own_beside_the_first(kept_history, media: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], name="原本那支")
    branched = call("branch_project", {"project_id": project, "name": "原本那支 Reels 版"})
    branch = runtime.repo.get_project(branched["project_id"])
    assert branch.name == "原本那支 Reels 版" and branch.branched_from.project_id == project
    edit(branch.id, [{"action": "rename_project", "name": "Reels 版改過"}])
    assert runtime.repo.get_project(project).name == "原本那支"
    assert call("project_history", {"project_id": branch.id})["branched_from"]["project_id"] == project


def test_a_change_made_in_the_editor_says_so(kept_history, media: Path) -> None:
    from starlette.testclient import TestClient
    from app.ui import app as editor

    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
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


from edit_plan.shared import planned  # noqa: E402,F401,F811  (the fixture, used by name below)


def test_going_back_leaves_another_video_compiled_from_the_same_plan_alone(kept_history, planned: dict) -> None:  # noqa: F811
    from app.models.plan import EditPlan

    portrait = editing.create_project("直式版", width=1080, height=1920)["id"]
    landscape = editing.create_project("橫式版")["id"]
    planning.compile_plan(portrait, 1, planned["plan_id"])
    planning.compile_plan(landscape, 1, planned["plan_id"])
    first = call("project_history", {"project_id": portrait})["versions"][0]["commit"]
    # The plan changes, and the portrait cut is compiled from it again.
    stored = EditPlan.model_validate(planning.get_plan(planned["plan_id"])["plan"])
    planning.save_plan(stored.model_copy(update={"goal": "改過的目標"}))
    planning.compile_plan(portrait, 2, planned["plan_id"])

    restored = call("restore_version", {"project_id": portrait, "commit": first, "expected_version": 3})
    assert restored["plan_copied"] is True
    copy = {clip.from_plan_id for track in runtime.repo.get_project(portrait).tracks for clip in track.clips
            if clip.from_plan_id}
    assert copy and planned["plan_id"] not in copy
    assert runtime.repo.get_plan(copy.pop()).goal == "做一支短的"
    # The landscape cut and the shared plan are as they were.
    assert runtime.repo.get_plan(planned["plan_id"]).goal == "改過的目標"
    assert all(clip.from_plan_id == planned["plan_id"]
               for track in runtime.repo.get_project(landscape).tracks for clip in track.clips if clip.from_plan_id)


def test_a_version_that_opens_on_black_is_shown_past_it(kept_history, media: Path) -> None:
    from app.models.media import MediaAnalysis, Span

    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    runtime.repo.save_analysis(MediaAnalysis(asset_id=asset, duration=3.0, black_frames=[Span(start=0.0, end=1.5)]))
    project = build_project([video_track(), insert("a", asset, 0, 3)], name="黑場開頭的版本")
    # The same frame as the projects page: half a second past the black.
    assert versioning.version_graph(project)["nodes"][0]["thumb"] == {"asset_id": asset, "t": 2.0}


def test_the_version_panel_folds_one_sitting_into_one_dot(kept_history, media: Path) -> None:
    from app.models.job import Job, JobKind, JobStatus

    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], name="版本面板")
    for name in ("第二版", "第三版"):
        edit(project, [{"action": "rename_project", "name": name}])
    nodes = versioning.version_graph(project)["nodes"]
    # Creating it and every change after by the same AI, nothing rendered between: one dot.
    assert len(nodes) == 1 and len(nodes[0]["versions"]) > 3
    assert nodes[0]["thumb"]["asset_id"] == asset

    # Rendering one out marks it, and starts a new dot after it.
    rendered = runtime.repo.get_project(project).version
    finished = media / "版本面板.mp4"
    finished.write_bytes(b"finished")
    runtime.repo.add_job(Job(kind=JobKind.RENDER, project_id=project, project_version=rendered,
                             status=JobStatus.COMPLETED, output_path=str(finished)))
    edit(project, [{"action": "rename_project", "name": "第四版"}])
    nodes = versioning.version_graph(project)["nodes"]
    assert [node["marks"] for node in nodes] == [[], ["exported"]]
    listed = call("project_history", {"project_id": project})["versions"]
    assert [version["marks"] for version in listed[:len(nodes[0]["versions"]) + 1]][-1] == ["exported"]

    # A star is the user's own, and a branch grows from the dot it started at.
    starred = call("mark_version", {"project_id": project, "commit": nodes[1]["commit"][:10], "mark": "starred"})
    assert starred["marks"] == ["exported", "starred"]
    # A starred version is its own dot, not the head of the edits before it.
    marked = versioning.version_graph(project)["nodes"][1]
    assert marked["versions"] == marked["versions"][:1] and marked["outputs"][0]["name"] == "版本面板.mp4"
    # Its finished video gone, it says so rather than going quiet.
    finished.unlink()
    assert "removed" in versioning.version_graph(project)["nodes"][1]["marks"]
    branch = call("branch_project", {"project_id": project, "name": "版本面板短版", "commit": nodes[1]["commit"]})
    graph = versioning.version_graph(project)
    assert graph["branches"][0]["project_id"] == branch["project_id"]
    assert graph["branches"][0]["from"] == nodes[1]["commit"]
    assert versioning.version_graph(branch["project_id"])["branched_from"]["name"] == "第四版"
    # The editor's list folds the branch under the project it came from.
    from starlette.testclient import TestClient

    from app.ui import app as editor
    listed = {item["id"]: item for item in TestClient(editor.create_app()).get("/api/projects").json()["projects"]}
    assert listed[branch["project_id"]]["under"] == project and listed[project]["under"] is None
