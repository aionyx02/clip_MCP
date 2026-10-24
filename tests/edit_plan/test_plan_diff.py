"""Tests for what changed between two plans."""

from app.engine.plan.diff import diff_plans
from app.models.plan import MusicCue, MusicPlan, Selection, Trim, TrimKind

from edit_plan.shared import SONG, footage, parts, plan_for, speech_of


def test_a_diff_names_what_actually_changed() -> None:
    by_id, children, assets, clips = footage()
    first, second, third = speech_of(clips)[:3]
    before = plan_for([Selection(clip_id=first.id, beat_id="b1"), Selection(clip_id=second.id, beat_id="b1")],
                      goal="原本的目標")
    after = plan_for([
        Selection(clip_id=first.id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=1.0)),
        Selection(clip_id=third.id, beat_id="b1"),
    ], goal="改過的目標")
    changes = diff_plans(before, after)
    assert any("原本的目標" in line for line in changes["goal"])
    assert any(line.startswith("added") for line in changes["selections"])
    assert any(line.startswith("dropped") for line in changes["selections"])
    assert any(line.startswith("retrimmed") for line in changes["selections"])

def test_two_plans_that_are_the_same_diff_to_nothing() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")])
    assert diff_plans(plan, plan) == {}

def test_a_diff_names_the_music_that_changed() -> None:
    plan, _, _ = parts()
    before = plan.model_copy(update={"music": MusicPlan(cues=[MusicCue(asset_id=SONG)])})
    after = plan.model_copy(update={"music": MusicPlan(cues=[
        MusicCue(asset_id=SONG, cut_on_beat=True), MusicCue(beat_id="b3", asset_id="other"),
    ])})
    lines = diff_plans(before, after)["music"]
    assert "music from beat b3: other" in lines
    assert "music from the start: song → song, now cutting on the beat" in lines
