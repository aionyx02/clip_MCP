"""Tests for checking a plan, and for keeping what was changed by hand through a recompile."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

import pytest

from app.engine.plan.beats import plan_pieces
from app.engine.plan.checks import _story_problems, as_left, check_plan, check_recompile
from app.engine.plan.compile import compile_operations
from app.engine.plan.music import music_beds, plan_markers
from app.engine.plan.pieces import Piece, note_record
from app.engine.semantic import build_timeline, clean_cuts
from app.models.plan import (
    BeatRole, Beat, BeatTransition, EditPlan, MusicCue, MusicPlan, Established, Fact, PlanTarget, Selection, Structure,
    Trim, TrimKind,
)
from app.models.semantic import SemanticClip, SemanticTimeline
from app.models.timeline.parts import Clip, Project, Track
from test_semantic_timeline import ASSET_ID, analysis, asset

from edit_plan.shared import (
    BROLL_ASSET, FOOTAGE, SENTENCE, SONG, beats_plan, checked, covered, footage, joined_plan, music_by_hand, parts,
    plan_for, sectioned, song_of, sourced, speech_of, spoken_words,
)


def test_a_word_longer_than_the_snap_allows_is_left_alone_and_said_so() -> None:
    # One long word and no edge within SNAP_SECONDS of the middle of it. Moving the cut
    # further than that is a bigger change to the edit than the fault it fixes, so the
    # compiler leaves it — and says it left it, rather than quietly failing to fix it.
    made = spoken_words(("嗯" * 8, 2.0, 6.0))
    assets = {ASSET_ID: sourced()}
    timeline, clips = build_timeline(assets, {ASSET_ID: made})
    spoken = next(clip for clip in clips if clip.text)
    plan = plan_for([Selection(
        clip_id=spoken.id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.0),
    )]).model_copy(update={"timeline_id": timeline.id, "timeline_input_hash": timeline.input_hash})
    by_id, cuts = {clip.id: clip for clip in clips}, {ASSET_ID: clean_cuts(made)}

    assert plan_pieces(plan, by_id, {}, assets, cuts)[0].end == pytest.approx(4.0)
    _, notes = check_plan(plan, timeline, by_id, {}, assets, cuts)
    assert any("still land inside a word" in note for note in notes)
    # Which end, at which second, inside which word: a window has two edges to guess between.
    assert any("ends at 4.00s of its file, inside the word at 2.00–6.00s" in note for note in notes)

def test_moving_a_cut_is_reported_even_though_it_is_arithmetic() -> None:
    # It is still a change to somebody's edit. 「不要默默修正」 is about not doing things
    # silently, not only about validation.
    made = spoken_words(*SENTENCE)
    assets = {ASSET_ID: sourced()}
    timeline, clips = build_timeline(assets, {ASSET_ID: made})
    spoken = next(clip for clip in clips if clip.text)
    plan = plan_for([Selection(
        clip_id=spoken.id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.5),
    )]).model_copy(update={"timeline_id": timeline.id, "timeline_input_hash": timeline.input_hash})
    _, notes = check_plan(plan, timeline, {clip.id: clip for clip in clips}, {}, assets,
                          {ASSET_ID: clean_cuts(made)})
    assert any("moved off the middle of a word" in note for note in notes)

def test_a_cut_that_still_stops_mid_sentence_is_reported_not_repaired() -> None:
    # Finishing the sentence costs seconds, not fractions of one, so it changes how long
    # the video runs — which is the decision of whoever set the length, not the compiler's.
    made = spoken_words(*SENTENCE)
    assets = {ASSET_ID: sourced()}
    timeline, clips = build_timeline(assets, {ASSET_ID: made})
    spoken = next(clip for clip in clips if clip.text)
    plan = plan_for([Selection(
        clip_id=spoken.id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.5),
    )]).model_copy(update={"timeline_id": timeline.id, "timeline_input_hash": timeline.input_hash})

    by_id = {clip.id: clip for clip in clips}
    cuts = {ASSET_ID: clean_cuts(made)}
    problems, notes = check_plan(plan, timeline, by_id, {}, assets, cuts)
    assert problems == []
    assert any("stop before the sentence ends" in note for note in notes)
    # And it says what finishing it would cost, so the choice can be made on a number.
    assert any("2.0s" in note for note in notes)
    # One line however many windows it covers: fifteen near-identical notes is not a report.
    assert sum(1 for note in notes if "sentence ends" in note) == 1
    # But it names every one, with how short each is, so nobody has to work out which.
    record = next(note_record(note) for note in notes if "sentence ends" in note)
    assert record["kind"] == "stops_before_the_sentence_ends" and record["look"]
    assert record["clip_ids"] == [spoken.id] and record["seconds"] == 2.0
    assert f"{spoken.id} 2.0s" in record["message"]

def test_a_window_turned_all_the_way_down_is_not_said_to_stop_mid_sentence() -> None:
    """#81: a shot kept for its picture, under the music, has no sentence anybody hears stop."""
    made = spoken_words(*SENTENCE)
    assets = {ASSET_ID: sourced()}
    timeline, clips = build_timeline(assets, {ASSET_ID: made})
    spoken = next(clip for clip in clips if clip.text)
    plan = plan_for([Selection(
        clip_id=spoken.id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.5), volume=0,
    )]).model_copy(update={"timeline_id": timeline.id, "timeline_input_hash": timeline.input_hash})
    by_id = {clip.id: clip for clip in clips}
    problems, notes = check_plan(plan, timeline, by_id, {}, assets, {ASSET_ID: clean_cuts(made)})
    assert problems == []
    assert not any("sentence ends" in note for note in notes)

def context_for(plan: EditPlan, by_id: Dict[str, SemanticClip], children: dict, assets: dict) -> tuple:
    """Run the checks with a timeline matching the plan.

    Args:
        plan: Plan to check.
        by_id: The clips.
        children: Section members.
        assets: The assets.

    Returns:
        `(problems, notes)`.
    """
    from app.models.semantic import SemanticTimeline

    timeline = SemanticTimeline(
        id=plan.timeline_id, asset_ids=[ASSET_ID],
        input_hash=plan.timeline_input_hash, derivation_version=1,
    )
    return check_plan(plan, timeline, by_id, children, assets)

def test_a_clip_that_is_not_in_the_timeline_is_refused() -> None:
    by_id, children, assets, _ = footage()
    problems, _ = context_for(plan_for([Selection(clip_id="nope", beat_id="b1")]), by_id, children, assets)
    assert any("no such clip" in problem for problem in problems)

def test_a_beat_the_plan_does_not_have_is_refused() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="ghost")])
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("which the plan does not have" in problem for problem in problems)

def test_unusable_footage_cannot_be_selected() -> None:
    made = analysis(8.0, segments=[(4.0, 8.0, "終於開口")], silences=[(0.0, 4.0)], black=[(0.0, 3.5)])
    _, clips = build_timeline({ASSET_ID: sourced(seconds=8.0)}, {ASSET_ID: made})
    by_id = {clip.id: clip for clip in clips}
    plan = plan_for([Selection(clip_id=clips[0].id, beat_id="b1")])
    problems, _ = context_for(plan, by_id, {}, {ASSET_ID: sourced(seconds=8.0)})
    assert any("unusable" in problem for problem in problems)

def test_keep_needs_a_section_to_keep_things_from() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(
        clip_id=speech_of(clips)[0].id, beat_id="b1",
        trim=Trim(kind=TrimKind.KEEP, keep_clip_ids=["whatever"]),
    )])
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("nothing inside it to keep" in problem for problem in problems)

def test_keeping_something_that_is_not_inside_the_section_is_refused() -> None:
    by_id, children, assets, section_id = sectioned()
    plan = plan_for([Selection(
        clip_id=section_id, beat_id="b1",
        trim=Trim(kind=TrimKind.KEEP, keep_clip_ids=["somewhere-else"]),
    )])
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("is not inside this clip" in problem for problem in problems)

def test_asking_for_more_seconds_than_a_clip_has_is_refused() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(
        clip_id=speech_of(clips)[0].id, beat_id="b1",
        trim=Trim(kind=TrimKind.HEAD, seconds=30.0),
    )])
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("of a clip that runs" in problem for problem in problems)

def test_footage_analyzed_again_since_the_plan_was_written_is_refused() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")])
    stale = plan.model_copy(update={"timeline_input_hash": "something else"})
    from app.models.semantic import SemanticTimeline

    timeline = SemanticTimeline(id="tl_test", asset_ids=[ASSET_ID], input_hash="hash", derivation_version=1)
    problems, _ = check_plan(stale, timeline, by_id, children, assets)
    assert any("no longer mean the same moments" in problem for problem in problems)

def test_a_cut_that_misses_its_target_length_is_worth_saying() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")], target=PlanTarget(seconds=90.0))
    problems, notes = context_for(plan, by_id, children, assets)
    assert problems == []
    assert any("against a target of 90s" in note for note in notes)

def test_music_without_sound_is_refused() -> None:
    by_id, children, assets, clips = footage()
    assets = {**assets, "silent-song": asset("silent-song", has_audio=False)}
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")],
                    music=MusicPlan(cues=[MusicCue(asset_id="silent-song")]))
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("has no sound" in problem for problem in problems)

@pytest.mark.parametrize("cues, beats, heard, expected", [
    ([MusicCue(beat_id="b9", asset_id=SONG)], None, False, "which the plan does not have"),
    ([MusicCue(asset_id=SONG), MusicCue(asset_id="other")], None, False, "only the first cue"),
    ([MusicCue(asset_id=SONG), MusicCue(beat_id="b2", asset_id=SONG), MusicCue(beat_id="b2")], None, False,
     "already comes in on beat b2"),
    ([MusicCue(asset_id=SONG), MusicCue(beat_id="b3", asset_id=SONG), MusicCue(beat_id="b2")], None, False,
     "list the cues in the order"),
    ([MusicCue(asset_id=SONG, start=90.0)], None, False, "which runs 60.0s"),
    ([MusicCue(cut_on_beat=True)], None, False, "no beat to cut on"),
    ([MusicCue(asset_id=SONG, cut_on_beat=True)], None, False, "analyze it"),
    ([MusicCue(asset_id=SONG, cut_on_beat=True)], [], True, "no steady beat"),
    ([MusicCue(asset_id=SONG), MusicCue(beat_id="b4", asset_id=SONG)], None, False, "never begins"),
], ids=["unknown beat", "second without beat", "two on one beat", "out of order", "past the end",
        "silence on the beat", "never measured", "no pulse", "beat without footage"])
def test_music_cues_that_do_not_hold_up_are_refused(cues, beats, heard, expected) -> None:
    plan, by_id, assets = parts()
    plan = plan.model_copy(update={
        "beats": [*plan.beats, Beat(id="b4", name="空的", role=BeatRole.PAYOFF)],
        "music": MusicPlan(cues=cues),
    })
    cuts = {SONG: song_of(beats, 60.0)[1]} if heard else {}
    problems, _ = checked(plan, by_id, assets, cuts)
    assert any(expected in problem for problem in problems), problems

def test_music_turned_down_by_hand_stays_down_after_a_recompile() -> None:
    """It used to be let through the recompile check and then laid again at the plan's level."""
    plan, by_id, assets = parts()
    plan = plan.model_copy(update={"music": MusicPlan(cues=[
        MusicCue(asset_id=SONG, volume=0.4), MusicCue(beat_id="b3", asset_id="other", volume=0.5),
    ])})
    project, _ = music_by_hand(plan, by_id, assets, index=1, volume=0.1)
    pieces = plan_pieces(plan, by_id, {}, assets)
    assert check_recompile(plan, project, pieces, (), music_beds(plan, pieces, assets)) == []
    operations, provenance = compile_operations(plan, by_id, {}, assets, project)
    laid = [op for op in operations if op.get("track_id") == "music" and op["action"] == "add_clip"]
    assert [op["volume"] for op in laid] == [0.4, 0.1]
    assert [provenance[op["clip_id"]]["pinned"] for op in laid] == [False, True]

def test_music_lengthened_by_hand_into_the_next_stretch_is_refused() -> None:
    from app.models.timeline.parts import TimeRange

    plan, by_id, assets = parts()
    plan = plan.model_copy(update={"music": MusicPlan(cues=[
        MusicCue(asset_id=SONG), MusicCue(beat_id="b3", asset_id="other"),
    ])})
    project, laid = music_by_hand(plan, by_id, assets, index=0, source_range=TimeRange(start=Decimal(0), end=Decimal(30)))
    pieces = plan_pieces(plan, by_id, {}, assets)
    problems = check_recompile(plan, project, pieces, (), music_beds(plan, pieces, assets))
    assert any("lengthened by hand" in problem for problem in problems)

def test_a_cross_fade_longer_than_the_part_it_leaves_is_refused() -> None:
    plan, by_id, assets = parts()
    plan = plan.model_copy(update={"music": MusicPlan(crossfade_seconds=9.0, cues=[
        MusicCue(asset_id=SONG), MusicCue(beat_id="b2", asset_id="other"),
    ])})
    problems, _ = checked(plan, by_id, assets)
    assert any("cross-fade" in problem and "crossfade_seconds" in problem for problem in problems)

# --- story -------------------------------------------------------------------------------

def story(*roles: Optional[BeatRole]) -> List[Beat]:
    """Build parts with the roles given, in order.

    Args:
        *roles: One role per part, or nothing for a part that says none.

    Returns:
        The parts.
    """
    return [Beat(id=f"b{index}", name=f"第{index}段", role=role) for index, role in enumerate(roles, start=1)]

@pytest.mark.parametrize(("roles", "complaint"), [
    ((BeatRole.HOOK, None, BeatRole.PAYOFF), "do not say what they do"),
    ((BeatRole.HOOK, BeatRole.SETUP, BeatRole.PAYOFF), "no beat is a turn"),
    ((BeatRole.HOOK, BeatRole.TURN), "no beat is a payoff"),
    ((BeatRole.TURN, BeatRole.PAYOFF, BeatRole.SETUP), "end on the payoff"),
    ((BeatRole.SETUP, BeatRole.HOOK, BeatRole.TURN, BeatRole.PAYOFF), "open on it"),
])
def test_a_plan_that_does_not_tell_a_story_is_refused(roles: tuple, complaint: str) -> None:
    assert any(complaint in problem for problem in _story_problems(story(*roles)))

@pytest.mark.parametrize("roles", [
    (BeatRole.HOOK, BeatRole.SETUP, BeatRole.TURN, BeatRole.PAYOFF),
    (BeatRole.TURN, BeatRole.PAYOFF),
    (BeatRole.SETUP, BeatRole.TURN, BeatRole.TURN, BeatRole.PAYOFF),
])
def test_a_plan_that_begins_turns_and_ends_is_accepted(roles: tuple) -> None:
    assert _story_problems(story(*roles)) == []

R = BeatRole

@pytest.mark.parametrize(("structure", "roles"), [
    (Structure.FLASHBACK, (R.HOOK, R.SETUP, R.TURN, R.PAYOFF)),
    (Structure.PROBLEM_SOLUTION, (R.HOOK, R.PROBLEM, R.AGITATE, R.SOLUTION, R.RESULT)),
    (Structure.PROBLEM_SOLUTION, (R.PROBLEM, R.SOLUTION)),
    (Structure.LISTICLE, (R.HOOK, R.ITEM, R.ITEM, R.ITEM, R.RECAP)),
    (Structure.TUTORIAL, (R.HOOK, R.PREP, R.STEP, R.STEP, R.RESULT)),
    (Structure.BEFORE_AFTER, (R.HOOK, R.BEFORE, R.PROCESS, R.AFTER)),
    (Structure.REVIEW, (R.HOOK, R.FIRST_LOOK, R.DETAIL, R.DETAIL, R.VERDICT)),
    (Structure.MONTAGE, (R.HOOK, R.BUILD, R.PEAK, R.BUILD, R.PEAK, R.OUTRO)),
])
def test_every_structure_accepts_a_plan_in_its_own_shape(structure: Structure, roles: tuple) -> None:
    assert _story_problems(story(*roles), structure) == []

@pytest.mark.parametrize(("structure", "roles", "complaint"), [
    (Structure.LISTICLE, (R.HOOK, R.SETUP, R.TURN, R.PAYOFF), "清單式 has no place for"),
    (Structure.STORY, (R.HOOK, R.ITEM, R.ITEM), "起承轉合 has no place for"),
    (Structure.FLASHBACK, (R.SETUP, R.TURN, R.PAYOFF), "no beat is a hook"),
    (Structure.LISTICLE, (R.HOOK, R.ITEM, R.RECAP), "needs at least 2"),
    (Structure.TUTORIAL, (R.PREP, R.STEP, R.PREP, R.RESULT), "comes after a step"),
    (Structure.TUTORIAL, (R.PREP, R.STEP), "no beat is a result"),
    (Structure.BEFORE_AFTER, (R.AFTER, R.BEFORE), "comes after an after"),
    (Structure.PROBLEM_SOLUTION, (R.PROBLEM, R.SOLUTION, R.AGITATE), "end on the solution or the result"),
    (Structure.REVIEW, (R.DETAIL, R.FIRST_LOOK, R.VERDICT), "開箱評測 plays"),
    (Structure.MONTAGE, (R.HOOK, R.BUILD, R.OUTRO), "no beat is a peak"),
])
def test_a_plan_out_of_its_structures_shape_is_refused(structure: Structure, roles: tuple, complaint: str) -> None:
    assert any(complaint in problem for problem in _story_problems(story(*roles), structure))

def test_a_plan_says_its_structure_and_is_held_to_it() -> None:
    clips = lambda found: [Selection(clip_id=speech_of(found)[0].id, beat_id="b1"),
                           Selection(clip_id=speech_of(found)[1].id, beat_id="b2")]
    listed = story(R.ITEM, R.ITEM)
    assert any(problem.startswith("story") for problem in story_checked(clips, listed))
    plan_checked = story_checked(clips, listed, Structure.LISTICLE)
    assert not any(problem.startswith("story") for problem in plan_checked)

def story_checked(selections: List[Selection], beats: List[Beat], structure: Structure = Structure.STORY) -> List[str]:
    """Check a plan over the talk footage, with its timeline filled in.

    Args:
        selections: The chosen footage.
        beats: The parts.
        structure: The shape the plan says it takes.

    Returns:
        The problems found.
    """
    made = analysis(**FOOTAGE)
    assets = {ASSET_ID: sourced()}
    timeline, clips = build_timeline(assets, {ASSET_ID: made})
    plan = beats_plan(selections(clips) if callable(selections) else selections, beats).model_copy(
        update={"timeline_id": timeline.id, "timeline_input_hash": timeline.input_hash, "structure": structure},
    )
    problems, _ = check_plan(plan, timeline, {clip.id: clip for clip in clips}, {}, assets, None)
    return problems

def test_one_part_cut_from_one_recording_is_a_trim_and_needs_no_story() -> None:
    problems = story_checked(lambda clips: [Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")], story(None))
    assert not any(problem.startswith("story") for problem in problems)

def test_two_parts_are_a_story_and_have_to_say_so() -> None:
    problems = story_checked(
        lambda clips: [Selection(clip_id=speech_of(clips)[0].id, beat_id="b1"),
                       Selection(clip_id=speech_of(clips)[1].id, beat_id="b2")],
        story(None, None),
    )
    assert any(problem.startswith("story") for problem in problems)

def test_a_range_without_both_ends_is_handed_back() -> None:
    by_id, children, assets, clips = covered()
    walk = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = plan_for([Selection(clip_id=walk.id, beat_id="b1", trim=Trim(kind=TrimKind.RANGE, from_seconds=5.0))])
    problems, _ = check_plan(plan, SemanticTimeline(id="tl_test", asset_ids=[], input_hash="hash", derivation_version=6),
                             by_id, children, assets)
    assert any("`range` needs both" in problem for problem in problems)

def test_a_transition_with_no_picture_before_its_shot_is_refused_with_the_ways_to_fix_it() -> None:
    # It used to be left out with one line saying so, and every beat's transition went missing
    # in a cut whose parts each began at the top of a file.
    plan, by_id, children, assets = joined_plan(BeatTransition(kind="wipe", seconds=0.5), None, opening=0.0)
    timeline = SemanticTimeline(id="tl_test", asset_ids=[], input_hash="hash", derivation_version=6)
    problems, notes = check_plan(plan, timeline, by_id, children, assets)
    refused = [problem for problem in problems if "wipe" in problem]
    assert len(refused) == 1
    # Said once, as what stops the compile; not again as a note that it was dealt with (#95).
    assert not any(note_record(note)["kind"] == "join_left_out" for note in notes)
    assert "from_seconds" in refused[0] and "0.5" in refused[0]
    assert "dip" in refused[0] and "set_beat_join" in refused[0]
    # When the shot before is what is too short, moving this one later would not help: say so.
    from app.engine.plan import checks as compiler

    short, _ = compiler._unfit_transitions(plan, [
        Piece(asset_id="x", start=0.0, end=0.03, from_clip_ids=("a",), beat_id="b1"),
        Piece(asset_id="y", start=5.0, end=8.0, from_clip_ids=("b",), beat_id="b2"),
    ])
    assert short and "before it" in short[0] and "from_seconds" not in short[0]
    # A dip only reaches back half its length, so the same shot needs a quarter second for a 0.5s one.
    plan, by_id, children, assets = joined_plan(BeatTransition(kind="dip", seconds=0.5), None, opening=0.0)
    problems, _ = check_plan(plan, timeline, by_id, children, assets)
    assert any("0.25" in problem for problem in problems if "dip" in problem)

def test_what_is_laid_over_the_cut_follows_a_clip_sped_up_by_hand() -> None:
    by_id, children, assets, clips = footage()
    first, second = speech_of(clips)[0], speech_of(clips)[2]
    plan = EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash",
        beats=[Beat(id="b1", name="一"), Beat(id="b2", name="二")],
        selections=[Selection(clip_id=first.id, beat_id="b1"), Selection(clip_id=second.id, beat_id="b2")],
    )
    pieces = plan_pieces(plan, by_id, children, assets)
    fast = Clip(id="p001", asset_id=ASSET_ID, timeline_in=Decimal(0), pinned=True, speed=2.0,
                from_clip_ids=list(pieces[0].from_clip_ids),
                source_range={"start": pieces[0].start, "end": pieces[0].end})
    project = Project(id="x", tracks=[Track(id="main", track_type="video", clips=[fast])])
    left, _ = as_left(pieces, project)
    # The second part's marker lands where the sped-up first part now ends.
    assert plan_markers(plan, left)[1]["timeline_in"] == pytest.approx(pieces[0].played / 2, abs=0.001)

def test_a_piece_from_another_time_between_two_that_belong_together_is_handed_back() -> None:
    by_id, children, assets, clips = covered()
    first, second = speech_of(clips)[:2]
    walk = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    # The talking was shot in the morning, the walk that afternoon.
    assets = {
        ASSET_ID: assets[ASSET_ID].model_copy(update={"recorded_at": datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc)}),
        BROLL_ASSET: assets[BROLL_ASSET].model_copy(update={"recorded_at": datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc)}),
    }
    plan = plan_for([
        Selection(clip_id=first.id, beat_id="b1"),
        Selection(clip_id=walk.id, beat_id="b1"),
        Selection(clip_id=second.id, beat_id="b1"),
    ])
    timeline = SemanticTimeline(id="tl_test", asset_ids=[], input_hash="hash", derivation_version=6)
    problems, _ = check_plan(plan, timeline, by_id, children, assets)
    assert any(problem.startswith("continuity:") and walk.id in problem for problem in problems)
    # Said why, it stands.
    plan.selections[1].jump_reason = "先讓人看到下午要去哪"
    problems, _ = check_plan(plan, timeline, by_id, children, assets)
    assert not any(problem.startswith("continuity:") for problem in problems)

def test_a_part_that_opens_later_and_goes_back_in_time_is_handed_back() -> None:
    by_id, children, assets, clips = covered()
    first, second = speech_of(clips)[:2]
    walk = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    # The afternoon walk opens the part, and the morning's talking follows it.
    assets = {
        ASSET_ID: assets[ASSET_ID].model_copy(update={"recorded_at": datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc)}),
        BROLL_ASSET: assets[BROLL_ASSET].model_copy(update={"recorded_at": datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc)}),
    }
    selections = [Selection(clip_id=walk.id, beat_id="b1"), Selection(clip_id=first.id, beat_id="b1"),
                  Selection(clip_id=second.id, beat_id="b1")]
    timeline = SemanticTimeline(id="tl_test", asset_ids=[], input_hash="hash", derivation_version=6)
    problems, _ = check_plan(plan_for(selections), timeline, by_id, children, assets)
    assert [problem for problem in problems if problem.startswith("continuity:")] == [
        problem for problem in problems if problem.startswith(f"continuity: {first.id} was shot at")]
    assert any(problem.startswith("continuity:") for problem in problems)
    # Told backwards on purpose, it stands.
    problems, _ = check_plan(plan_for(selections, structure=Structure.FLASHBACK), timeline, by_id, children, assets)
    assert not any(problem.startswith("continuity:") for problem in problems)

def test_files_numbered_by_hand_played_out_of_their_order_are_noted() -> None:
    from app.engine.plan.checks import file_number

    assert [file_number(name) for name in ("1-開場.mp4", "01_a.mp4", "3.mov", "12 b.mp4", "2、c.mp4", "(4) d.mp4",
                                           "IMG_0001.mp4", "20260829_1.mp4", "v2.mp4")] == [1, 1, 3, 12, 2, 4,
                                                                                            None, None, None]
    by_id, children, assets, clips = covered()
    first = speech_of(clips)[0]
    walk = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    assets = {ASSET_ID: assets[ASSET_ID].model_copy(update={"path": "D:/拍攝/1-聊天.mp4"}),
              BROLL_ASSET: assets[BROLL_ASSET].model_copy(update={"path": "D:/拍攝/2-散步.mp4"})}
    timeline = SemanticTimeline(id="tl_test", asset_ids=[], input_hash="hash", derivation_version=6)

    def noted(*selections: Selection) -> list:
        _, notes = check_plan(plan_for(list(selections)), timeline, by_id, children, assets)
        return [note for note in notes if note.kind == "file_order"]

    found = noted(Selection(clip_id=walk.id, beat_id="b1"), Selection(clip_id=first.id, beat_id="b1"))
    assert len(found) == 1 and "1-聊天.mp4 after 2-散步.mp4" in found[0] and found[0].clip_ids == (first.id,)
    assert noted(Selection(clip_id=first.id, beat_id="b1"), Selection(clip_id=walk.id, beat_id="b1")) == []
    assert noted(Selection(clip_id=walk.id, beat_id="b1"),
                 Selection(clip_id=first.id, beat_id="b1", jump_reason="回到開頭")) == []

def test_a_chosen_clip_the_camera_shook_in_is_noted_with_how_much() -> None:
    by_id, children, assets, clips = covered()
    first = speech_of(clips)[0]
    walk = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    by_id = {**by_id, walk.id: walk.model_copy(update={"scores": {**walk.scores, "shake": 0.04}}),
             first.id: first.model_copy(update={"scores": {**first.scores, "shake": 0.002}})}
    timeline = SemanticTimeline(id="tl_test", asset_ids=[], input_hash="hash", derivation_version=6)
    _, notes = check_plan(plan_for([Selection(clip_id=first.id, beat_id="b1"), Selection(clip_id=walk.id, beat_id="b1")]),
                          timeline, by_id, children, assets)
    [shaky] = [note for note in notes if note.kind == "shaky"]
    assert f"{walk.id} (0.040)" in shaky and first.id not in shaky and shaky.clip_ids == (walk.id,)

def knowing(plan: EditPlan, by_id: dict, *said: dict) -> EditPlan:
    """Say, piece by piece in playing order, what each establishes and relies on.

    Args:
        plan: A plan of `parts`.
        by_id: Its clips, for where each piece's fact is said.
        *said: Per piece, `establishes` as a list of fact IDs, and `requires` and `ahead`.

    Returns:
        The plan, with a fact for every ID named.
    """
    named = sorted({fact for item in said for key in ("establishes", "requires", "ahead") for fact in item.get(key, [])})
    selections = []
    for selection, item in zip(plan.selections, said):
        clip = by_id[selection.clip_id]
        middle = (clip.source_range.start + clip.source_range.end) / 2
        selections.append(selection.model_copy(update={
            "establishes": [Established(fact=fact, evidence=(middle - 0.1, middle + 0.1))
                            for fact in item.get("establishes", [])],
            "requires": item.get("requires", []), "ahead": item.get("ahead", []),
        }))
    return plan.model_copy(update={"facts": [Fact(id=fact, says=fact) for fact in named], "selections": selections})

def test_a_plan_of_several_parts_says_what_the_viewer_has_to_know() -> None:
    plan, by_id, assets = parts()
    problems, _ = checked(plan.model_copy(update={"facts": []}), by_id, assets)
    assert any(problem.startswith("facts: a plan of more than one part") for problem in problems)
    # One part is a trim, and asks for nothing.
    one = plan_for([plan.selections[0].model_copy(update={"establishes": []})])
    assert not any("facts" in problem for problem in checked(one, by_id, assets)[0])

@pytest.mark.parametrize("said, expected", [
    # The result before the damage it repairs, with nothing to say it is on purpose.
    (({}, {"requires": ["damage"]}, {"establishes": ["damage"]}), "relies on damage, which nothing before it"),
    # On purpose, but the damage is never shown.
    (({}, {"ahead": ["damage"]}, {}), "nothing after it establishes it"),
    (({"establishes": ["damage"]}, {"ahead": ["damage"]}, {}), "has already established it"),
    # The hook may show what comes later, but it has to come.
    (({"requires": ["damage"]}, {}, {}), "relies on damage coming later"),
    (({"establishes": ["damage"]}, {"requires": ["cost"]}, {"establishes": ["cost"]}), "relies on cost"),
], ids=["out of order", "ahead never told", "ahead already told", "hook never told", "second fact"])
def test_a_fact_relied_on_before_it_is_established_is_handed_back(said: tuple, expected: str) -> None:
    plan, by_id, assets = parts()
    problems, _ = checked(knowing(plan, by_id, *said), by_id, assets)
    assert any(expected in problem for problem in problems), problems

@pytest.mark.parametrize("said", [
    ({"establishes": ["damage"]}, {"requires": ["damage"]}, {}),
    ({}, {"ahead": ["damage"]}, {"establishes": ["damage"]}),
    # The hook shows the result; the damage comes after it.
    ({"requires": ["damage"]}, {"establishes": ["damage"]}, {}),
], ids=["in order", "ahead on purpose", "hook"])
def test_a_fact_established_where_it_is_needed_passes(said: tuple) -> None:
    plan, by_id, assets = parts()
    problems, _ = checked(knowing(plan, by_id, *said), by_id, assets)
    assert problems == []

def test_a_fact_nobody_establishes_or_nobody_listed_is_handed_back() -> None:
    plan, by_id, assets = parts()
    plan = knowing(plan, by_id, {"establishes": ["damage"]}, {}, {})
    unlisted = plan.model_copy(update={"facts": [*plan.facts, Fact(id="cost", says="修了多少錢")]})
    assert "facts: nothing establishes cost; say which piece shows or says it, or take it out" in checked(
        unlisted, by_id, assets)[0]
    stranger = plan.model_copy(update={"selections": [
        plan.selections[0], plan.selections[1].model_copy(update={"requires": ["weather"]}), plan.selections[2],
    ]})
    assert any("names the fact weather, which is not in the plan's `facts`" in problem
               for problem in checked(stranger, by_id, assets)[0])

def test_evidence_outside_the_clip_or_trimmed_off_is_handed_back() -> None:
    plan, by_id, assets = parts()
    plan = knowing(plan, by_id, {"establishes": ["damage"]}, {}, {})
    first = plan.selections[0]
    clip = by_id[first.clip_id]
    opened, closed = clip.source_range.start, clip.source_range.end
    outside = first.model_copy(update={"establishes": [Established(fact="damage", evidence=(closed + 1, closed + 2))]})
    problems, _ = checked(plan.model_copy(update={"selections": [outside, *plan.selections[1:]]}), by_id, assets)
    assert any("is not in the clip" in problem for problem in problems), problems
    # Said near the end of the clip, and only its first second is played.
    late = first.model_copy(update={
        "establishes": [Established(fact="damage", evidence=(closed - 0.3, closed - 0.1))],
        "trim": Trim(kind=TrimKind.HEAD, seconds=1.0),
    })
    problems, _ = checked(plan.model_copy(update={"selections": [late, *plan.selections[1:]]}), by_id, assets)
    assert any("is not all played" in problem for problem in problems), problems

def test_tightening_one_sentence_is_refused_rather_than_doing_nothing() -> None:
    """#99: `tighten` on an utterance compiled exactly as it was, and said nothing."""
    by_id, _, assets, clips = footage()
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1", trim=Trim(kind=TrimKind.TIGHTEN))])
    problems, _ = checked(plan, by_id, assets)
    assert any("`tighten`" in problem and "`range`" in problem for problem in problems)
