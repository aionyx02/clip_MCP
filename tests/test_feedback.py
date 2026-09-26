"""Tests for the feedback loop: changes to the whole cut, and a history to go back through.

Global amendments are checked as arithmetic on the plan and on its compiled
length; the history is checked through the tools, since what matters is that
a version the user liked can be found and brought back.
"""

from typing import List

import pytest
from pydantic import TypeAdapter

from app.engine.plan import compiled_duration, plan_pieces
from app.models.plan import (
    Beat, BrollShot, DropBeatOp, EditPlan, MusicCue, MusicPlan, Pacing, PlanAmendment, PlanTarget, Selection,
    SetMusicLevelOp, SetPacingOp, SetTargetOp, apply_amendment,
)
from app.server import (
    amend_plan, diff_plan, get_plan, list_plan_versions, preview_plan_diff, revert_plan, save_plan,
)
from test_edit_plan import checked, footage, planned, sectioned, speech_of  # noqa: F401 — fixtures and helpers

def amendments(*operations: dict) -> list:
    """Validate amendments the way a client's would be.

    Args:
        *operations: Amendments as plain dictionaries.

    Returns:
        The parsed amendments.
    """
    return TypeAdapter(List[PlanAmendment]).validate_python(list(operations))

def two_parts() -> EditPlan:
    """A plan in two parts, with music under both and covering picture in the second.

    Returns:
        The plan.
    """
    return EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash",
        beats=[Beat(id="b1", name="開場"), Beat(id="b2", name="結尾")],
        selections=[Selection(clip_id="c1", beat_id="b1"), Selection(clip_id="c2", beat_id="b2"),
                    Selection(clip_id="c3", beat_id="b2")],
        broll=[BrollShot(clip_id="x", over_clip_id="c3", seconds=2.0)],
        music=MusicPlan(cues=[MusicCue(asset_id="song", volume=0.4),
                              MusicCue(beat_id="b2", asset_id="other", volume=0.5)]),
    )

# --- changes to the whole cut ------------------------------------------------------------

def test_the_music_can_be_turned_down_everywhere_at_once() -> None:
    plan = two_parts()
    apply_amendment(plan, SetMusicLevelOp(scale=0.5))
    assert [cue.volume for cue in plan.music.cues] == [0.2, 0.25]

def test_or_only_from_one_part_on() -> None:
    plan = two_parts()
    apply_amendment(plan, SetMusicLevelOp(scale=2.0, beat_id="b2"))
    assert [cue.volume for cue in plan.music.cues] == [0.4, 1.0]

def test_turning_up_stops_at_the_ceiling_rather_than_failing() -> None:
    plan = two_parts()
    for _ in range(5):
        apply_amendment(plan, SetMusicLevelOp(scale=3.0))
    assert all(cue.volume == 4.0 for cue in plan.music.cues)

def test_music_that_is_not_there_cannot_be_turned_down() -> None:
    plan = two_parts().model_copy(update={"music": None})
    with pytest.raises(ValueError, match="no music"):
        apply_amendment(plan, SetMusicLevelOp(scale=0.5))

def test_a_whole_part_comes_out_with_its_reason_and_everything_hung_on_it() -> None:
    plan = two_parts()
    apply_amendment(plan, DropBeatOp(beat_id="b2", reason="結尾太拖"))
    assert [beat.id for beat in plan.beats] == ["b1"]
    assert [selection.clip_id for selection in plan.selections] == ["c1"]
    assert {item.clip_id: item.reason for item in plan.rejected} == {"c2": "結尾太拖", "c3": "結尾太拖"}
    # The cover over a shot that is gone and the music cue for a part that is gone go with them.
    assert plan.broll == []
    assert [cue.beat_id for cue in plan.music.cues] == [None]

def test_a_part_the_plan_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError, match="no beat b9"):
        apply_amendment(two_parts(), DropBeatOp(beat_id="b9"))

def test_a_new_target_replaces_the_old() -> None:
    plan = two_parts()
    apply_amendment(plan, SetTargetOp(target=PlanTarget(seconds=30, platform="Reels")))
    assert (plan.target.seconds, plan.target.platform) == (30, "Reels")

def test_tighter_pacing_makes_the_whole_cut_shorter() -> None:
    by_id, children, assets, section = sectioned()
    plan = EditPlan(timeline_id="tl_test", timeline_input_hash="hash", beats=[Beat(id="b1", name="全部")],
                    selections=[Selection(clip_id=section, beat_id="b1")])
    from test_edit_plan import FOOTAGE, analysis, clean_cuts, ASSET_ID

    cuts = {ASSET_ID: clean_cuts(analysis(**FOOTAGE))}
    loose = compiled_duration(plan_pieces(plan, by_id, children, assets, cuts))
    apply_amendment(plan, SetPacingOp(pacing=Pacing(pause_seconds=0.45, breath_seconds=0.05)))
    tight = compiled_duration(plan_pieces(plan, by_id, children, assets, cuts))
    # The half-second pause after the first line is now dead air, and every cut keeps less.
    assert tight < loose - 0.4

def test_pacing_that_would_undo_itself_is_refused() -> None:
    """Taking a pause out leaves a breath either side; if what is left between them fits
    in the merge gap, the two halves are joined straight back and nothing happens."""
    by_id, _, assets, clips = footage()
    plan = EditPlan(timeline_id="tl_test", timeline_input_hash="hash", beats=[Beat(id="b1", name="主體")],
                    selections=[Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")],
                    pacing=Pacing(pause_seconds=0.4, breath_seconds=0.1))
    problems, _ = checked(plan, by_id, assets)
    assert any("pacing" in problem and "breath_seconds" in problem for problem in problems)

# --- the history -------------------------------------------------------------------------

def test_every_version_is_kept_with_what_it_was_for(planned: dict) -> None:  # noqa: F811
    first = planned["clips"][0]["clip_id"]
    amend_plan(planned["plan_id"], 1, amendments(
        {"action": "set_trim", "clip_id": first, "trim": {"kind": "head", "seconds": 1.0}},
    ), note="開頭太長")
    amend_plan(planned["plan_id"], 2, amendments(
        {"action": "set_pacing", "pacing": {"pause_seconds": 0.5, "breath_seconds": 0.05}},
    ), note="整體節奏太慢")
    history = list_plan_versions(planned["plan_id"])
    assert history["current"] == 3
    assert [item["version"] for item in history["versions"]] == [1, 2, 3]
    assert history["versions"][1]["note"].startswith("開頭太長 — retrimmed")
    assert history["versions"][2]["note"].startswith("整體節奏太慢 — pacing")
    assert history["versions"][1]["changed"] == {"selections": 1}
    assert history["versions"][2]["changed"] == {"goal": 1}

def test_an_earlier_version_can_be_read_and_compared(planned: dict) -> None:  # noqa: F811
    first = planned["clips"][0]["clip_id"]
    amend_plan(planned["plan_id"], 1, amendments({"action": "drop_selection", "clip_id": first, "reason": "不要"}))
    assert len(get_plan(planned["plan_id"], version=1)["plan"]["selections"]) == 2
    assert len(get_plan(planned["plan_id"])["plan"]["selections"]) == 1
    changes = diff_plan(planned["plan_id"], planned["plan_id"], before_version=1)["changes"]
    assert any(line.startswith("dropped") for line in changes["selections"])
    text = preview_plan_diff(planned["plan_id"], planned["plan_id"], before_version=1).content[0].text
    assert "1 dropped" in text

def test_going_back_writes_a_new_version_rather_than_losing_one(planned: dict) -> None:  # noqa: F811
    first = planned["clips"][0]["clip_id"]
    amend_plan(planned["plan_id"], 1, amendments({"action": "drop_selection", "clip_id": first}))
    back = revert_plan(planned["plan_id"], to_version=1, expected_version=2, note="剛剛那樣比較好")
    assert back["version"] == 3 and back["problems"] == []
    assert get_plan(planned["plan_id"])["plan"]["selections"] == get_plan(planned["plan_id"], version=1)["plan"]["selections"]
    # And going back is itself in the history, so it can be undone too.
    notes = [item["note"] for item in list_plan_versions(planned["plan_id"])["versions"]]
    assert notes[-1] == "back to version 1 — 剛剛那樣比較好"
    assert len(get_plan(planned["plan_id"], version=2)["plan"]["selections"]) == 1

def test_a_version_that_was_never_kept_is_refused(planned: dict) -> None:  # noqa: F811
    with pytest.raises(ValueError, match="versions kept are 1"):
        get_plan(planned["plan_id"], version=7)
    with pytest.raises(ValueError, match="version conflict"):
        revert_plan(planned["plan_id"], to_version=1, expected_version=5)
