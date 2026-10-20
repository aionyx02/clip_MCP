"""Tests for covering picture: where B-roll lands, what is refused, and the places offered for it."""

from typing import List

import pytest

from app.engine.plan.beats import plan_pieces
from app.engine.plan.broll import broll_covers, broll_slots
from app.engine.semantic import build_timeline, clean_cuts
from app.models.media import MediaAnalysis, Transcript, TranscriptSegment, TranscriptWord
from app.models.plan import (
    AddBrollOp, BrollShot, DropBrollOp, DropSelectionOp, Selection, Trim, TrimKind, apply_amendment,
)
from test_semantic_timeline import ASSET_ID, analysis, spans

from edit_plan.shared import BROLL_ASSET, covered, plan_for, sourced, speech_of


def laid_over(shots: List[dict], holds: List[int] = []) -> tuple:
    """Compile a cut of the four sentences with some picture laid over it.

    Args:
        shots: Each `{over: <index into the speech clips>, seconds, from: <clip id>}`.
        holds: Which of the four selections are held for what they show.

    Returns:
        `(covers, problems, notes)`.
    """
    by_id, children, assets, clips = covered()
    speech = speech_of(clips)
    cover_source = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = plan_for(
        [
            Selection(clip_id=clip.id, beat_id="b1", hold_picture=place in holds)
            for place, clip in enumerate(speech)
        ],
        broll=[
            BrollShot(
                clip_id=shot.get("from", cover_source.id),
                over_clip_id=shot["over"] if isinstance(shot["over"], str) else speech[shot["over"]].id,
                seconds=shot["seconds"],
            )
            for shot in shots
        ],
    )
    pieces = plan_pieces(plan, by_id, children, assets)
    return broll_covers(plan, pieces, by_id, assets)

def test_covering_picture_lands_where_the_words_it_belongs_to_are() -> None:
    covers, problems, _ = laid_over([{"over": 0, "seconds": 2.0}])
    assert problems == []
    # The first sentence runs 1.0 to 3.0 of a window that opens at 0.9, so it reaches the
    # timeline a tenth of a second in — and that is where the cover cuts in.
    assert (covers[0].timeline_in, covers[0].timeline_out) == (0.1, 2.1)
    # Played from the start of the clip it was taken from, for exactly as long as asked.
    assert (covers[0].start, covers[0].end) == (0.0, 2.0)

@pytest.mark.parametrize("seconds", [0.4, 9.0])
def test_a_shot_that_flickers_or_outstays_its_welcome_is_refused(seconds: float) -> None:
    _, problems, _ = laid_over([{"over": 0, "seconds": seconds}])
    assert any("may run" in problem for problem in problems)

def test_a_shot_may_not_cover_what_the_plan_holds_for_what_it_shows() -> None:
    # Nothing measures "the speaker is pointing at the thing", so the plan says it.
    covers, problems, _ = laid_over([{"over": 0, "seconds": 2.0}], holds=[0])
    assert covers == []
    assert any("holds for what it shows" in problem for problem in problems)

def test_two_shots_cannot_be_on_screen_at_once() -> None:
    covers, problems, _ = laid_over([{"over": 0, "seconds": 3.0}, {"over": 1, "seconds": 2.0}])
    # The first runs to 3.1s and the second would start at 2.6s.
    assert len(covers) == 1
    assert any("cannot be on screen at once" in problem for problem in problems)

def test_a_beat_made_mostly_of_covering_picture_is_refused() -> None:
    # The cut runs 8.6s, so four of them is nearly half of the only beat there is.
    _, problems, _ = laid_over([{"over": 0, "seconds": 4.0}])
    assert any("may be covered" in problem for problem in problems)

def test_a_shot_longer_than_the_footage_it_plays_is_refused() -> None:
    by_id, _, _, clips = covered()
    short = speech_of(clips)[0]
    _, problems, _ = laid_over([{"over": 2, "seconds": 5.0, "from": short.id}])
    assert any("which is not the 5s asked for" in problem for problem in problems)

def test_a_shot_over_footage_the_cut_does_not_use_is_refused() -> None:
    _, _, _, clips = covered()
    unused = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    _, problems, _ = laid_over([{"over": unused.id, "seconds": 2.0}])
    assert any("nowhere for this to start" in problem for problem in problems)

def test_a_shot_can_start_inside_a_window_rather_than_only_at_one() -> None:
    # Where a clip is in the cut is a question about seconds, not about selections: two
    # sentences merged into one window are still two places a cover can cut in on. That
    # is what lets a long shot be covered in part rather than not at all.
    covers, problems, _ = laid_over([{"over": 1, "seconds": 1.5}])
    assert problems == []
    # The second sentence runs 3.5 to 5.0 of a window that opened at 0.9.
    assert covers[0].timeline_in == 2.6

def test_a_shot_that_would_run_past_the_end_of_the_cut_is_refused() -> None:
    _, problems, _ = laid_over([{"over": 3, "seconds": 3.0}])
    assert any("of a cut that is" in problem for problem in problems)

def test_covering_footage_nobody_is_talking_over_is_said_rather_than_refused() -> None:
    # Swapping one silent picture for another is allowed; it is just rarely what is meant.
    _, _, _, clips = covered()
    quiet = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    by_id, children, assets, _ = covered()
    speech = speech_of(clips)
    plan = plan_for(
        [Selection(clip_id=speech[0].id, beat_id="b1"),
         Selection(clip_id=quiet.id, beat_id="b1")],
        broll=[BrollShot(clip_id=quiet.id, over_clip_id=quiet.id, seconds=2.0)],
    )
    covers, problems, notes = broll_covers(
        plan, plan_pieces(plan, by_id, children, assets), by_id, assets,
    )
    assert problems == [] and len(covers) == 1
    assert any("nobody is talking under it" in note for note in notes)

def test_the_places_worth_covering_are_the_ones_that_hold() -> None:
    by_id, children, assets, clips = covered()
    quiet = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    # The silent file is one twenty-second shot; the talk is three short windows.
    plan = plan_for([
        Selection(clip_id=speech_of(clips)[0].id, beat_id="b1"),
        Selection(clip_id=quiet.id, beat_id="b1"),
    ])
    slots = broll_slots(plan, plan_pieces(plan, by_id, children, assets), by_id)
    assert [slot["over_clip_id"] for slot in slots] == [quiet.id]
    # Offered at the length a shot may run, not the length the stretch holds for.
    assert slots[0]["seconds"] == 6.0 and slots[0]["held_seconds"] == 20.0
    assert "holds for" in slots[0]["why"]

def test_the_worst_stretch_is_offered_first() -> None:
    # Most of a cut made of static shots gets offered back, so the order has to carry the
    # priority: reading twenty slots to find the bad one is not an answer.
    by_id, children, assets, clips = covered()
    quiet = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = plan_for(
        [Selection(clip_id=clip.id, beat_id="b1") for clip in speech_of(clips)]
        + [Selection(clip_id=quiet.id, beat_id="b1")]
    )
    slots = broll_slots(plan, plan_pieces(plan, by_id, children, assets), by_id)
    held = [slot["held_seconds"] for slot in slots]
    # Twenty seconds of one silent shot comes before the 8.6s of talk from the other
    # file, however many places inside each of them are worth cutting in at.
    assert held == sorted(held, reverse=True)
    assert held[0] == 20.0 and set(held[1:]) == {8.6}

def test_a_sentence_with_no_boundary_inside_it_offers_one_place_to_cut_in() -> None:
    # A shot cuts in where a clip begins, and a fifteen-second sentence has one of those.
    # Covering it from anywhere else would be cutting in mid-sentence, which is the rule
    # this is built around rather than a limit of it.
    long_take = analysis(
        duration=40.0,
        segments=[(1.0, 3.0, "第一句話"), (20.0, 35.0, "很長的一段話")],
        silences=[(0.0, 1.0), (3.0, 20.0), (35.0, 40.0)],
    )
    assets = {ASSET_ID: sourced(seconds=40.0)}
    _, clips = build_timeline(assets, {ASSET_ID: long_take})
    by_id = {clip.id: clip for clip in clips}
    plan = plan_for([Selection(clip_id=clip.id, beat_id="b1") for clip in speech_of(clips)])
    slots = broll_slots(plan, plan_pieces(plan, by_id, {}, assets), by_id)
    assert len(slots) == 1 and slots[0]["seconds"] == 6.0

def test_nothing_is_offered_over_a_shot_the_plan_holds() -> None:
    by_id, children, assets, clips = covered()
    quiet = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = plan_for([Selection(clip_id=quiet.id, beat_id="b1", hold_picture=True)])
    assert broll_slots(plan, plan_pieces(plan, by_id, children, assets), by_id) == []

def test_dropping_a_shot_takes_the_picture_laid_over_it() -> None:
    _, _, _, clips = covered()
    speech = speech_of(clips)
    quiet = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = plan_for(
        [Selection(clip_id=clip.id, beat_id="b1") for clip in speech],
        broll=[BrollShot(clip_id=quiet.id, over_clip_id=speech[0].id, seconds=2.0)],
    )
    apply_amendment(plan, DropSelectionOp(clip_id=speech[0].id, reason="不要了"))
    # The place it named is gone, so the cover has nowhere to start. Leaving it would
    # only fail the next compile with a message about a clip nobody asked about.
    assert plan.broll == []

def test_adding_a_shot_over_the_same_place_twice_replaces_it() -> None:
    _, _, _, clips = covered()
    speech, quiet = speech_of(clips), next(c for c in clips if c.asset_id == BROLL_ASSET)
    plan = plan_for([Selection(clip_id=clip.id, beat_id="b1") for clip in speech])
    for seconds in (2.0, 3.0):
        apply_amendment(plan, AddBrollOp(shot=BrollShot(
            clip_id=quiet.id, over_clip_id=speech[0].id, seconds=seconds,
        )))
    assert [shot.seconds for shot in plan.broll] == [3.0]
    apply_amendment(plan, DropBrollOp(over_clip_id=speech[0].id))
    assert plan.broll == []

def test_covering_picture_may_not_cut_in_partway_through_a_sentence() -> None:
    """The half of the boundary rule that costs nothing to fix, so it is refused.

    Cutting back out is reported instead, because lengthening a shot to let a
    sentence finish decides how long the video runs.
    """
    made = MediaAnalysis(
        asset_id=ASSET_ID, duration=30.0,
        silences=spans([(0.0, 1.0), (3.0, 5.0), (15.0, 30.0)]),
        transcript=Transcript(language="zh", model="test", segments=[
            TranscriptSegment(start=1.0, end=3.0, text="開場", words=[
                TranscriptWord(start=1.0 + step * 0.5, end=1.5 + step * 0.5, text="字")
                for step in range(4)
            ]),
            TranscriptSegment(start=5.0, end=15.0, text="主體", words=[
                TranscriptWord(start=5.0 + step * 0.5, end=5.5 + step * 0.5, text="字")
                for step in range(20)
            ]),
        ]),
    )
    quiet = analysis(duration=20.0, silences=[(0.0, 20.0)], scenes=[(0.0, 20.0)], asset_id=BROLL_ASSET)
    assets = {ASSET_ID: sourced(seconds=30.0), BROLL_ASSET: sourced(BROLL_ASSET)}
    _, clips = build_timeline(assets, {ASSET_ID: made, BROLL_ASSET: quiet})
    by_id = {clip.id: clip for clip in clips}
    speech = speech_of(clips)
    cover = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    cuts = {ASSET_ID: clean_cuts(made), BROLL_ASSET: clean_cuts(quiet)}
    # A tail trim opens the first window two seconds into the sentence it belongs to.
    plan = plan_for(
        [Selection(clip_id=speech[0].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=1.0)),
         Selection(clip_id=speech[1].id, beat_id="b1")],
        broll=[BrollShot(clip_id=cover.id, over_clip_id=speech[0].id, seconds=1.5)],
    )
    pieces = plan_pieces(plan, by_id, {}, assets, cuts)
    covers, problems, _ = broll_covers(plan, pieces, by_id, assets, cuts)
    assert covers == []
    assert any("cuts in partway through a sentence" in problem for problem in problems)
    # And the server never offers such a place to begin with.
    assert all(slot["over_clip_id"] != speech[0].id
               for slot in broll_slots(plan, pieces, by_id, cuts))

def test_covering_picture_lands_where_its_words_are_played_at_the_new_pace() -> None:
    from app.models.plan import Pacing

    by_id, _, assets, clips = covered()
    speech = speech_of([clip for clip in clips if clip.asset_id == ASSET_ID])
    quiet = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = plan_for(
        [Selection(clip_id=clip.id, beat_id="b1") for clip in speech],
        broll=[BrollShot(clip_id=quiet.id, over_clip_id=speech[1].id, seconds=1.0)],
    )
    normal, _, _ = broll_covers(plan, plan_pieces(plan, by_id, {}, assets), by_id, assets)
    fast_plan = plan.model_copy(update={"pacing": Pacing(speed=1.25)})
    fast, refused, _ = broll_covers(fast_plan, plan_pieces(fast_plan, by_id, {}, assets), by_id, assets)
    assert refused == []
    assert fast[0].timeline_in == pytest.approx(normal[0].timeline_in / 1.25, abs=0.002)
    # The cover itself plays at normal speed, for as long as the plan says.
    assert fast[0].duration == normal[0].duration == 1.0

SAID_TWENTY = "今天我們來到台南的古著店看看有什麼好東西"

def _talking_with_a_black_moment(black: bool) -> tuple:
    """One sentence from 5s to 15s, with the picture black from 9s to 9.6s when asked, and a quiet shot."""
    made = MediaAnalysis(
        asset_id=ASSET_ID, duration=30.0,
        silences=spans([(0.0, 5.0), (15.0, 30.0)]),
        black_frames=spans([(9.0, 9.6)] if black else []),
        transcript=Transcript(language="zh", model="test", segments=[
            # Twenty different words: one word over and over is a line the recogniser made up.
            TranscriptSegment(start=5.0, end=15.0, text=SAID_TWENTY, words=[
                TranscriptWord(start=5.0 + step * 0.5, end=5.5 + step * 0.5, text=SAID_TWENTY[step])
                for step in range(20)
            ]),
        ]),
    )
    quiet = analysis(duration=20.0, silences=[(0.0, 20.0)], scenes=[(0.0, 20.0)], asset_id=BROLL_ASSET)
    assets = {ASSET_ID: sourced(seconds=30.0), BROLL_ASSET: sourced(BROLL_ASSET)}
    _, clips = build_timeline(assets, {ASSET_ID: made, BROLL_ASSET: quiet})
    cuts = {ASSET_ID: clean_cuts(made), BROLL_ASSET: clean_cuts(quiet)}
    return {clip.id: clip for clip in clips}, assets, cuts, speech_of(clips)[0],         next(clip for clip in clips if clip.asset_id == BROLL_ASSET)

def test_covering_picture_may_cut_in_partway_through_a_sentence_only_over_a_black_moment() -> None:
    """#99: a hand over the lens for 0.6s in the middle of a line; the sound cannot be cut."""
    for black in (True, False):
        by_id, assets, cuts, spoken, cover = _talking_with_a_black_moment(black)
        plan = plan_for([Selection(clip_id=spoken.id, beat_id="b1")],
                        broll=[BrollShot(clip_id=cover.id, over_clip_id=spoken.id, seconds=1.0, after_seconds=3.8)])
        pieces = plan_pieces(plan, by_id, {}, assets, cuts)
        covers, problems, _ = broll_covers(plan, pieces, by_id, assets, cuts)
        if black:
            assert problems == [] and len(covers) == 1
        else:
            assert covers == [] and any("partway through a sentence" in problem for problem in problems)
