"""Two versions made into one, a part from each, as a new version on top of both."""

from pathlib import Path

from app import server
from app.engine.merge import merge, merge_plans
from app.models.plan import Beat, EditPlan, Selection
from helpers import build_project, edit, insert, video_track


def three_shots(media: Path) -> str:
    """A cut in two parts: 開場 is one shot of 4s, 主體 two shots of 3s each."""
    shot = server.import_asset(str(media / "wide.mp4"))["id"]
    other = server.import_asset(str(media / "tall.mp4"))["id"]
    return build_project([
        video_track(), insert("a", shot, 0, 4), insert("b", other, 0, 3), insert("c", other, 3, 6),
        {"action": "set_markers", "markers": [{"id": "m1", "name": "開場", "timeline_in": 0},
                                              {"id": "m2", "name": "主體", "timeline_in": 4}]},
    ], name="合成")


def test_each_part_comes_whole_from_the_side_picked(media: Path) -> None:
    project = three_shots(media)
    before = server.repo.get_project(project)
    edit(project, [
        {"action": "trim_clip", "track_id": "main", "clip_id": "a", "new_source_range": {"start": 1, "end": 4}},
        {"action": "delete_clip", "track_id": "main", "clip_id": "c"},
    ])
    after = server.repo.get_project(project)
    # The new opening, the old body.
    merged = merge(before, after, {"主體": "a"})
    clips = merged.base_video_track.clips
    assert [(clip.id, float(clip.timeline_in), float(clip.source_range.start)) for clip in clips] == \
        [("a", 0.0, 1.0), ("b", 3.0, 0.0), ("c", 6.0, 3.0)]
    assert [(marker.name, float(marker.timeline_in)) for marker in merged.markers] == [("開場", 0.0), ("主體", 3.0)]
    # Nothing picked is the second side as it is.
    assert [clip.id for clip in merge(before, after, {}).base_video_track.clips] == ["a", "b"]


def test_a_merge_is_a_new_version_and_both_it_came_from_stay(media: Path, kept_history) -> None:
    project = three_shots(media)
    whole = server.project_history(project)["versions"][0]["commit"]
    edit(project, [{"action": "delete_clip", "track_id": "main", "clip_id": "c"}])
    version = server.repo.get_project(project).version
    before = len(server.project_history(project)["versions"])
    from test_comments import call

    merged = call("merge_version_parts", {"project_id": project, "commit_a": whole, "picks": {"主體": "a"},
                                          "expected_version": version})
    assert merged["new_version"] == version + 1 and merged["from_a"] == ["主體"] and merged["seconds"] == 10.0
    history = server.project_history(project)["versions"]
    assert history[0]["what"].startswith("合成兩版") and len(history) == before + 1


def test_a_part_nobody_has_cannot_be_picked(media: Path) -> None:
    import pytest

    project = three_shots(media)
    with pytest.raises(ValueError, match="nothing called 結尾"):
        server.merge_versions(project, "", None, {"結尾": "a"}, server.repo.get_project(project).version)


def test_the_plans_are_merged_beat_by_beat_with_their_reasons() -> None:
    beats = [Beat(id="b1", name="開場"), Beat(id="b2", name="結尾")]
    left = EditPlan(id="L", timeline_id="t", timeline_input_hash="", beats=beats, selections=[
        Selection(clip_id="u1", beat_id="b1", rationale="舊的開場比較直接"),
        Selection(clip_id="u9", beat_id="b2", rationale="舊結尾"),
    ])
    right_beats = [Beat(id="x1", name="開場"), Beat(id="x2", name="結尾")]
    right = EditPlan(id="R", timeline_id="t", timeline_input_hash="", beats=right_beats, selections=[
        Selection(clip_id="u2", beat_id="x1", rationale="新的開場"),
        Selection(clip_id="u5", beat_id="x2", rationale="新結尾"),
    ])
    merged = merge_plans(left, right, ["開場"])
    assert [(item.clip_id, item.beat_id, item.rationale) for item in merged.selections] == \
        [("u1", "x1", "舊的開場比較直接"), ("u5", "x2", "新結尾")]
    assert merged.id == "R"
