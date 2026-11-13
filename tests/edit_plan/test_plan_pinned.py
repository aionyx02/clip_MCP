"""Tests for a recompile over a pinned clip the plan has changed since (#301)."""

from pathlib import Path
from typing import Tuple

import pytest

from app.models.plan import Beat, EditPlan, Selection
from app.runtime import repo as runtime_repo
from app.tools.catalog import import_asset
from app.tools.editing import apply_edits, create_project, get_project
from app.tools.planning import amend_plan, compile_plan, get_plan, save_plan
from test_semantic_timeline import analysis

from edit_plan.test_plan_tools import amendments, main_clips, ops

CORNER = {"place": "top_right", "size": "small"}


def broll_clips(project_id: str) -> list:
    """Read the B-roll track's clips."""
    return next(track for track in get_project(project_id)["tracks"] if track["id"] == "broll")["clips"]


def compiled_with_broll(planned: dict, media: Path, monkeypatch: pytest.MonkeyPatch) -> Tuple[str, str, dict]:
    """Compile a plan with one full-frame shot of B-roll onto a fresh project.

    Returns:
        `(project_id, plan_id, shot)` — the shot as `add_broll` takes it.
    """
    from app.engine import meaning
    from app.tools.planning import propose_broll
    from app.tools.timelines import build_semantic_timeline, query_clips
    from test_meaning_search import _stand_in

    monkeypatch.setattr(meaning, "embed", _stand_in)
    quiet = import_asset(str(media / "silent.mp4"))["id"]
    runtime_repo.save_analysis(analysis(duration=4.0, scenes=[(0.0, 4.0)], asset_id=quiet))
    plan = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    built = build_semantic_timeline([clip["asset_id"] for clip in planned["clips"]] + [quiet], rebuild=True)
    speech = query_clips(timeline_id=built["timeline_id"], text="句話")["clips"]
    cover = query_clips(timeline_id=built["timeline_id"], asset_ids=[quiet])["clips"][0]
    saved = save_plan(EditPlan(
        timeline_id=built["timeline_id"], timeline_input_hash="", goal=plan.goal, beats=[Beat(id="b1", name="主體")],
        selections=[Selection(clip_id=speech[place]["clip_id"], beat_id="b1") for place in (0, 2)],
    ))
    offered = propose_broll(saved["plan_id"], over_clip_ids=[speech[0]["clip_id"]])
    shot = {"clip_id": cover["clip_id"], "over_clip_id": speech[0]["clip_id"], "seconds": 1.5,
            "candidate_hash": offered["candidate_hash"]}
    assert amend_plan(saved["plan_id"], 1, amendments({"action": "add_broll", "shot": shot}))["problems"] == []
    project = create_project(name="測試", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=saved["plan_id"])
    return project, saved["plan_id"], shot


def pin(project: str, track_id: str, clip_id: str, pinned: bool = True) -> None:
    apply_edits(project, get_project(project)["version"], ops({
        "action": "set_clip_pinned", "track_id": track_id, "clip_id": clip_id, "pinned": pinned,
    }))


def test_a_layout_the_plan_gives_a_pinned_shot_is_reported_not_dropped(
        planned: dict, media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project, plan_id, shot = compiled_with_broll(planned, media, monkeypatch)
    laid = broll_clips(project)[0]
    assert laid.get("layout") is None
    pin(project, "broll", laid["id"])
    version = get_plan(plan_id)["plan"]["version"]
    amend_plan(plan_id, version, amendments({"action": "add_broll", "shot": {**shot, "layout": CORNER}}))
    before = get_project(project)["version"]

    with pytest.raises(ValueError) as refused:
        compile_plan(project_id=project, expected_version=before, plan_id=plan_id)
    said = str(refused.value)
    assert f"clip {laid['id']} on track broll" in said
    assert "layout was the whole frame, the plan now asks for top_right small" in said
    assert "set_clip_pinned" in said and "change the plan back" in said
    # Only the field the plan changed: the range and the level are what they were.
    assert "source_range" not in said and "volume" not in said
    # Refused means nothing moved.
    assert get_project(project)["version"] == before
    assert broll_clips(project)[0].get("layout") is None

    # Unpinned, the plan's version is laid.
    pin(project, "broll", laid["id"], pinned=False)
    compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=plan_id)
    assert broll_clips(project)[0].get("layout") == CORNER


def test_changing_the_plan_back_lets_the_pinned_shot_compile_as_it_was(
        planned: dict, media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project, plan_id, shot = compiled_with_broll(planned, media, monkeypatch)
    laid = broll_clips(project)[0]
    pin(project, "broll", laid["id"])
    amend_plan(plan_id, get_plan(plan_id)["plan"]["version"],
               amendments({"action": "add_broll", "shot": {**shot, "layout": CORNER}}))
    amend_plan(plan_id, get_plan(plan_id)["plan"]["version"], amendments({"action": "add_broll", "shot": shot}))
    result = compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=plan_id)
    assert result["kept"] == 1
    assert broll_clips(project)[0].get("layout") is None and broll_clips(project)[0]["pinned"] is True


def test_a_pinned_clip_the_plan_did_not_change_is_not_reported(planned: dict) -> None:
    project = create_project(name="測試", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    first = main_clips(project)[0]
    # Held to a hand trim and a hand level: what a pin is for, so no report.
    shortened = {"start": float(first["source_range"]["start"]), "end": float(first["source_range"]["start"]) + 0.5}
    apply_edits(project, get_project(project)["version"], ops(
        {"action": "trim_clip", "track_id": "main", "clip_id": first["id"], "new_source_range": shortened},
        {"action": "set_clip_audio", "track_id": "main", "clip_id": first["id"], "volume": 0.25},
    ))
    result = compile_plan(project_id=project, expected_version=get_project(project)["version"],
                          plan_id=planned["plan_id"])
    assert result["kept"] == 1
    assert float(main_clips(project)[0]["source_range"]["end"]) == pytest.approx(shortened["end"])


def test_a_trim_the_plan_changes_on_a_pinned_clip_is_reported(planned: dict) -> None:
    project = create_project(name="測試", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    first = main_clips(project)[0]
    apply_edits(project, get_project(project)["version"], ops({
        "action": "set_clip_audio", "track_id": "main", "clip_id": first["id"], "volume": 0.25,
    }))
    amend_plan(planned["plan_id"], get_plan(planned["plan_id"])["plan"]["version"], amendments(
        {"action": "set_trim", "clip_id": planned["clips"][0]["clip_id"], "trim": {"kind": "head", "seconds": 1.0}},
    ))
    with pytest.raises(ValueError, match=rf"clip {first['id']} on track main .*source_range was"):
        compile_plan(project_id=project, expected_version=get_project(project)["version"],
                     plan_id=planned["plan_id"])



def test_what_the_plan_asked_of_a_clip_is_not_part_of_its_picture(planned: dict) -> None:
    from app.tools.rendering import picture_key

    project = create_project(name="測試", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    stored = runtime_repo.get_project(project)
    assert any(clip.plan_asked for track in stored.video_tracks for clip in track.clips)
    bare = stored.model_copy(deep=True)
    for track in bare.video_tracks:
        for clip in track.clips:
            clip.plan_asked = None
    # A recompile that only records what the plan asked does not make the picture render again.
    assert picture_key(stored) == picture_key(bare)
