"""Two versions of a cut compared part by part, the way a person would say what changed."""

from pathlib import Path

from app import server
from app.engine.compare import compare
from helpers import build_project, edit, insert, video_track


def three_shots(media: Path, **settings) -> str:
    """A cut in two parts: 開場 is one shot of 4s, 主體 two shots of 3s each."""
    shot = server.import_asset(str(media / "wide.mp4"))["id"]
    other = server.import_asset(str(media / "tall.mp4"))["id"]
    return build_project([
        video_track(), insert("a", shot, 0, 4), insert("b", other, 0, 3), insert("c", other, 3, 6),
        {"action": "set_markers", "markers": [{"id": "m1", "name": "開場", "timeline_in": 0},
                                              {"id": "m2", "name": "主體", "timeline_in": 4}]},
    ], name="比較", **settings)


def words(asset_id: str, start: float, end: float) -> str:
    return "嗨大家" if start < 1 else ""


def test_what_changed_is_said_part_by_part(media: Path) -> None:
    project = three_shots(media)
    before = server.repo.get_project(project)
    edit(project, [
        {"action": "trim_clip", "track_id": "main", "clip_id": "a", "new_source_range": {"start": 1, "end": 4}},
        {"action": "delete_clip", "track_id": "main", "clip_id": "c"},
    ])
    after = server.repo.get_project(project)
    compared = compare(before, after, words, lambda asset_id: "x.mp4")
    opening, body = compared["parts"]
    assert opening["name"] == "開場" and any("開頭剪掉 1.0 秒" in change["what"] and "嗨大家" in change["what"]
                                             for change in opening["changes"])
    # The second shot slid a second earlier with the trim, past the marker, and is still in 主體.
    assert body["changes"] == [{"what": "拿掉 x.mp4 的一段（3.0 秒）", "a_at": 7.0, "b_at": None}]
    assert compared["length"] == [10.0, 6.0]
    # The second shot is the same footage in both, a second earlier in the new one.
    assert [4.0, 7.0, 3.0, 6.0] in compared["aligned"]


def test_the_same_footage_under_new_clip_ids_is_not_a_change(media: Path) -> None:
    project = three_shots(media)
    before = server.repo.get_project(project)
    renamed = before.model_copy(deep=True)
    for index, clip in enumerate(renamed.tracks[0].clips):
        clip.id = f"p{index:03d}"
    compared = compare(before, renamed, words, lambda asset_id: "x.mp4")
    assert all(part["same"] for part in compared["parts"]) and compared["whole"] == []


def test_the_ai_compares_a_version_with_now(media: Path, kept_history) -> None:
    project = three_shots(media)
    first = server.project_history(project)["versions"][-1]["commit"]
    edit(project, [{"action": "delete_clip", "track_id": "main", "clip_id": "b"}])
    compared = server.compare_versions(project, first)
    assert compared["a"]["commit"] == first and "aligned" not in compared
    assert any(change["what"].startswith("拿掉") for part in compared["parts"] for change in part["changes"])
