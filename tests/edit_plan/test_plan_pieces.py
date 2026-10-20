"""Tests for the pieces a plan becomes: breath, snapping, trims, pauses, retakes and bad frames."""

from dataclasses import replace
from decimal import Decimal

import pytest

from app.engine.plan.beats import compile_pieces, plan_pieces
from app.engine.plan.checks import check_plan
from app.engine.plan.compile import compile_operations
from app.engine.plan.music import music_beds
from app.engine.plan.pieces import (
    BREATH_SECONDS, MERGE_GAP_SECONDS, PAUSE_SECONDS, SNAP_SECONDS, TAIL_TRIM_SECONDS, Piece, compiled_duration,
    note_record,
)
from app.engine.semantic import build_timeline, clean_cuts
from app.models.media import MediaAnalysis, Span, Transcript, TranscriptSegment, TranscriptWord
from app.models.plan import MusicCue, MusicPlan, Pacing, Selection, Trim, TrimKind
from app.models.semantic import SemanticTimeline
from test_semantic_timeline import ASSET_ID, analysis, spans

from edit_plan.shared import (
    BROLL_ASSET, FOOTAGE, SENTENCE, SONG, compiled, covered, cut_points, footage, parts, plan_for, sectioned, sourced,
    speech_of, spoken_words,
)


def test_a_selection_is_widened_into_the_silence_around_it() -> None:
    by_id, children, assets, clips = footage()
    first = speech_of(clips)[0]
    plan = plan_for([Selection(clip_id=first.id, beat_id="b1")])
    pieces = plan_pieces(plan, by_id, children, assets)
    # The sentence runs 1.0 to 3.0 and has a full second of silence in front of it,
    # so it opens a breath early rather than the instant the picture cuts.
    assert [(piece.asset_id, piece.start, piece.end) for piece in pieces] == [
        (ASSET_ID, round(1.0 - BREATH_SECONDS, 3), round(3.0 + BREATH_SECONDS, 3))
    ]
    # The piece knows what it was made of, which is what a second compile matches on.
    assert pieces[0].from_clip_ids == (first.id,)

def test_the_air_never_reaches_further_than_the_headroom_allows() -> None:
    # Two sentences with a fiftieth of a second of silence between them: less room
    # than a breath wants, so the breath has to give way rather than clip the words.
    tight = analysis(
        10.0,
        segments=[(1.0, 3.0, "前一句"), (3.05, 6.0, "後一句")],
        silences=[(0.0, 1.0), (3.0, 3.05), (6.0, 10.0)],
    )
    _, clips = build_timeline({ASSET_ID: sourced(seconds=10.0)}, {ASSET_ID: tight})
    by_id, assets = {clip.id: clip for clip in clips}, {ASSET_ID: sourced(seconds=10.0)}
    second = speech_of(clips)[1]
    assert second.safe_in == pytest.approx(0.05)

    pieces = plan_pieces(plan_for([Selection(clip_id=second.id, beat_id="b1")]), by_id, {}, assets)
    assert pieces[0].start == pytest.approx(second.source_range.start - second.safe_in)
    assert pieces[0].start > speech_of(clips)[0].source_range.end - BREATH_SECONDS

def test_a_window_that_reaches_the_end_of_the_file_stays_inside_it() -> None:
    # A file 36.266667s long, with the last sentence running right to the end. Rounding
    # that to milliseconds gives 36.267, which is past the end of the file: the whole
    # plan used to be refused for a window that was only ever asking for all of it.
    length = 36.266667
    to_the_end = analysis(
        length,
        segments=[(1.0, 3.0, "第一句"), (30.0, length, "最後一句")],
        silences=[(0.0, 1.0), (3.0, 30.0)],
    )
    assets = {ASSET_ID: sourced(seconds=length)}
    _, clips = build_timeline(assets, {ASSET_ID: to_the_end})
    by_id = {clip.id: clip for clip in clips}
    last = speech_of(clips)[-1]

    pieces = plan_pieces(plan_for([Selection(clip_id=last.id, beat_id="b1")]), by_id, {}, assets)
    # Compared the way the timeline compares it, which is where this used to blow up.
    assert Decimal(str(pieces[-1].end)) <= assets[ASSET_ID].duration

def head_piece(seconds: float, made: MediaAnalysis, snap: bool = True):
    """Compile a `head` trim over one spoken sentence.

    Args:
        seconds: How many seconds the trim asks for.
        made: The analysis to compile against.
        snap: Whether to hand the compiler the clean cut points.

    Returns:
        The single compiled window.
    """
    assets = {ASSET_ID: sourced()}
    _, clips = build_timeline(assets, {ASSET_ID: made})
    spoken = next(clip for clip in clips if clip.text)
    plan = plan_for([Selection(
        clip_id=spoken.id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=seconds),
    )])
    cuts = {ASSET_ID: clean_cuts(made)} if snap else None
    return plan_pieces(plan, {clip.id: clip for clip in clips}, {}, assets, cuts)[0]

def test_a_head_trim_does_not_stop_in_the_middle_of_a_word() -> None:
    made = spoken_words(*SENTENCE)
    # The sentence starts at 2.0, so 2.5s of it ends at 4.5 — inside 就起床.
    assert head_piece(2.5, made, snap=False).end == pytest.approx(4.5)
    # Given the word edges, it lands on one rather than through a syllable.
    # 4.5 is half a second from the edge at 4.0 and a full second from the one at 5.5,
    # so the nearest is 4.0 and the tie-break never comes into it.
    landed = head_piece(2.5, made).end
    assert landed == pytest.approx(4.0)
    assert not clean_cuts(made).splits_a_word(landed)

def test_snapping_a_cut_costs_a_fraction_of_a_second() -> None:
    # Whichever edge it takes, it may not wander: the point is a clean cut, not a
    # different edit. Measured on real footage this costs about a fifth of a second.
    made = spoken_words(*SENTENCE)
    assert abs(head_piece(2.5, made).end - 4.5) <= SNAP_SECONDS

def test_a_cut_already_between_words_is_left_where_it_is() -> None:
    made = spoken_words(*SENTENCE)
    # 2.0s of a sentence starting at 2.0 ends at 4.0, which is between two words.
    assert head_piece(2.0, made).end == pytest.approx(4.0)

def test_an_edge_that_breathes_is_not_snapped() -> None:
    # The opening of a `head` sits in measured silence with a breath on it. Moving it to
    # a word edge would spend the breath the rest of the compiler works to give it.
    made = spoken_words(*SENTENCE)
    assert head_piece(2.5, made).start == pytest.approx(2.0 - BREATH_SECONDS)

def test_footage_nobody_transcribed_is_cut_where_it_was_asked_for() -> None:
    # No word timings means nothing is known to break, which is not a licence to guess.
    made = analysis(20.0, silences=[(0.0, 20.0)])
    assets = {ASSET_ID: sourced()}
    _, clips = build_timeline(assets, {ASSET_ID: made})
    plan = plan_for([Selection(
        clip_id=clips[0].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=3.0),
    )])
    piece = plan_pieces(plan, {clip.id: clip for clip in clips}, {}, assets, {ASSET_ID: clean_cuts(made)})[0]
    assert piece.end == pytest.approx(3.0)

def test_a_tail_trim_does_not_start_in_the_middle_of_a_word() -> None:
    # The other end, and the other unpadded edge: `tail` opens wherever the arithmetic
    # lands, so it needs the same treatment as `head`.
    made = spoken_words(*SENTENCE)
    assets = {ASSET_ID: sourced()}
    _, clips = build_timeline(assets, {ASSET_ID: made})
    spoken = next(clip for clip in clips if clip.text)
    plan = plan_for([Selection(
        clip_id=spoken.id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=2.5),
    )])
    by_id = {clip.id: clip for clip in clips}
    # 2.5s back from the sentence end at 6.0 is 3.5, inside 四點 (3.0-4.0).
    assert plan_pieces(plan, by_id, {}, assets, None)[0].start == pytest.approx(3.5)
    landed = plan_pieces(plan, by_id, {}, assets, {ASSET_ID: clean_cuts(made)})[0].start
    assert not clean_cuts(made).splits_a_word(landed)

def test_a_note_from_outside_the_compiler_is_taken_as_worth_a_look() -> None:
    assert note_record("something") == {"kind": "note", "message": "something", "look": True}

def test_pieces_that_nearly_touch_become_one_clip() -> None:
    by_id, children, assets, clips = footage()
    first, second = speech_of(clips)[:2]
    plan = plan_for([
        Selection(clip_id=first.id, beat_id="b1"),
        Selection(clip_id=second.id, beat_id="b1"),
    ])
    pieces = plan_pieces(plan, by_id, children, assets)
    # Half a second apart, with breath on both sides, closes to under the merge gap.
    assert len(pieces) == 1
    assert pieces[0].end == pytest.approx(5.0 + BREATH_SECONDS, abs=0.01)
    # Merged, so the one clip remembers both sentences it was made from.
    assert pieces[0].from_clip_ids == (first.id, second.id)

def test_pieces_far_apart_stay_separate() -> None:
    by_id, children, assets, clips = footage()
    first, third = speech_of(clips)[0], speech_of(clips)[2]
    plan = plan_for([
        Selection(clip_id=first.id, beat_id="b1"),
        Selection(clip_id=third.id, beat_id="b1"),
    ])
    assert len(plan_pieces(plan, by_id, children, assets)) == 2

def test_head_and_tail_take_the_window_they_are_asked_for() -> None:
    by_id, children, assets, clips = footage()
    first = speech_of(clips)[0]
    head = plan_pieces(
        plan_for([Selection(clip_id=first.id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=1.0))]),
        by_id, children, assets,
    )
    tail = plan_pieces(
        plan_for([Selection(clip_id=first.id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=1.0))]),
        by_id, children, assets,
    )
    assert head[0].end == pytest.approx(2.0)
    assert tail[0].start == pytest.approx(2.0)

def test_keep_takes_only_the_sentences_it_names() -> None:
    by_id, children, assets, section_id = sectioned()
    inside = speech_of(children[section_id])
    plan = plan_for([Selection(
        clip_id=section_id, beat_id="b1",
        trim=Trim(kind=TrimKind.KEEP, keep_clip_ids=[inside[0].id, inside[2].id]),
    )])
    pieces = plan_pieces(plan, by_id, children, assets)
    assert len(pieces) == 2
    assert pieces[0].start == pytest.approx(inside[0].source_range.start - BREATH_SECONDS, abs=0.01)

def test_tighten_drops_the_pauses_out_of_a_section() -> None:
    by_id, children, assets, section_id = sectioned()
    plan = plan_for([Selection(clip_id=section_id, beat_id="b1", trim=Trim(kind=TrimKind.TIGHTEN))])
    tightened = compiled_duration(plan_pieces(plan, by_id, children, assets))
    whole = compiled_duration(plan_pieces(
        plan_for([Selection(clip_id=section_id, beat_id="b1")]), by_id, children, assets,
    ))
    # Three seconds of silence in the middle alone; the tightened cut has to be shorter.
    assert tightened < whole

# The mechanical cleanups: the things a rough cut needs that take no judgement, so they
# belong to the compiler rather than to a sentence in a guide hoping to be followed.
# Every one of them changes somebody's edit, so every one of them also says what it did.

LINE = ("我要", "講一", "件", "事")

def said_again(*at: float) -> MediaAnalysis:
    """Build a file where the same line is said over and over.

    Each go runs two seconds and is transcribed four words to the line, so a
    trim can land between words rather than through one, and the silence
    between two goes is what the semantic layer splits them on.

    Args:
        *at: Where each go starts, in seconds.

    Returns:
        The analysis.
    """
    segments, silences, after = [], [], 0.0
    for start in at:
        silences.append(Span(start=after, end=start))
        segments.append(TranscriptSegment(
            start=start, end=start + 2.0, text="".join(LINE),
            words=[
                TranscriptWord(start=start + place * 0.5, end=start + place * 0.5 + 0.5, text=word)
                for place, word in enumerate(LINE)
            ],
        ))
        after = start + 2.0
    silences.append(Span(start=after, end=after + 2.0))
    return MediaAnalysis(
        asset_id=ASSET_ID, duration=after + 2.0, silences=silences,
        transcript=Transcript(language="zh", model="test", segments=segments),
    )

def test_dead_air_in_the_middle_of_a_shot_is_taken_out() -> None:
    by_id, children, assets, section_id = sectioned()
    made = analysis(**FOOTAGE)
    plan = plan_for([Selection(clip_id=section_id, beat_id="b1")])
    pieces, notes = compile_pieces(plan, by_id, children, assets, {ASSET_ID: clean_cuts(made)})
    # The section covers the whole file, so it covers the three-second gap at 5s and the
    # two-second one at 10s. Each is taken out, leaving a breath on both sides of it.
    assert [(piece.start, piece.end) for piece in pieces] == [(0.9, 5.1), (7.9, 10.1), (11.9, 18.0)]
    assert any("2 pause(s)" in note for note in notes)

def test_a_pause_short_enough_to_be_a_breath_is_left_alone() -> None:
    by_id, children, assets, section_id = sectioned()
    made = analysis(**FOOTAGE)
    pieces, _ = compile_pieces(plan_for([Selection(clip_id=section_id, beat_id="b1")]),
                               by_id, children, assets, {ASSET_ID: clean_cuts(made)})
    # There is half a second of quiet at 3.0s. Cutting there would be cutting between
    # two sentences that belong together, so nothing opens or closes anywhere near it.
    assert not any(3.0 <= edge <= 3.5 for piece in pieces for edge in (piece.start, piece.end))

def test_a_pause_somebody_is_talking_across_is_left_in_and_said() -> None:
    # The two detectors disagree, which on real footage they do constantly: the stretch
    # was measured as quiet while the transcript has a word running the whole way over
    # it. Taking it out would cut through that word, which is the worse of the two.
    made = analysis(
        duration=12.0,
        segments=[(1.0, 2.0, "開頭"), (4.0, 8.0, "一直講"), (10.0, 11.0, "結尾")],
        silences=[(0.0, 1.0), (2.0, 4.0), (5.0, 7.0), (8.0, 10.0), (11.0, 12.0)],
    )
    pieces, notes = compiled(made)
    # Being inside a sentence is already reason enough to leave it in.
    assert (pieces[1].start, pieces[1].end) == (3.9, 8.1)
    assert not any("were taken out" in note for note in notes)

def test_a_shot_nobody_talks_in_keeps_its_quiet_stretches() -> None:
    # A hand reaching for a cup is quiet the whole way through, and the quiet is the
    # action, not the edit waiting: taking it out cut the pull itself out of the video.
    made = analysis(duration=12.0, silences=[(0.0, 0.4), (2.0, 5.0), (7.0, 10.0)])
    made.transcript = Transcript(language="zh", model="test", segments=[])
    pieces, notes = compiled(made, which="quiet")
    assert [(piece.start, piece.end) for piece in pieces] == [(0.0, 12.0)]
    assert not any("were taken out" in note for note in notes)

def test_a_pause_inside_one_sentence_is_kept() -> None:
    # Taking it out jumps inside the sentence, which the check before a render calls a cut in
    # the middle of somebody talking: the compiler made the cut and its own check refused it.
    made = analysis(duration=12.0, silences=[(0.0, 1.0), (3.0, 5.0), (9.0, 12.0)])
    made.transcript = Transcript(language="zh", model="test", segments=[
        TranscriptSegment(start=1.0, end=9.0, text="這個要很小心", words=[
            TranscriptWord(start=1.0, end=3.0, text="這個"), TranscriptWord(start=5.0, end=9.0, text="要很小心"),
        ]),
    ])
    pieces, notes = compiled(made)
    assert len(pieces) == 1 and pieces[0].start <= 1.0 and pieces[0].end >= 9.0
    assert not any("were taken out" in note for note in notes)
    assert not any("stop before the sentence ends" in note for note in notes)

def test_quiet_after_the_last_thing_said_is_not_dead_air() -> None:
    # Talking has to stop before a pause and start again after it for the pause to be the
    # edit waiting; quiet that runs on to the end of the window is what was being shown.
    made = analysis(duration=12.0, segments=[(0.5, 1.5, "這邊")], silences=[(1.5, 2.0), (4.0, 8.0)])
    whole = Selection(clip_id="", beat_id="b1")
    one = sourced(seconds=made.duration)
    _, clips = build_timeline({ASSET_ID: one}, {ASSET_ID: made})
    plan = plan_for([whole.model_copy(update={"clip_id": clip.id}) for clip in clips])
    pieces, notes = compile_pieces(plan, {clip.id: clip for clip in clips}, {}, {ASSET_ID: one},
                                   {ASSET_ID: clean_cuts(made)})
    assert not any("were taken out" in note for note in notes)
    assert not any(4.0 < piece.start < 8.0 for piece in pieces)

def test_taking_a_pause_out_has_to_outlast_the_merge_that_would_undo_it() -> None:
    # Both halves keep a breath, so what is left between them is the pause less two of
    # them. Any shorter than the merge gap and they are joined straight back together
    # and the stage is a no-op nobody could see.
    assert PAUSE_SECONDS > 2 * BREATH_SECONDS + MERGE_GAP_SECONDS

def test_the_last_complete_go_at_a_line_is_the_one_that_survives() -> None:
    pieces, notes = compiled(said_again(1.0, 4.0, 7.0))
    assert [(piece.start, piece.end) for piece in pieces] == [(6.9, 9.1)]
    assert any("said again" in note or "said a line" in note for note in notes)
    assert any("4.4s in all" in note for note in notes)

def test_where_the_last_go_at_a_line_is_unfinished_the_one_before_it_is_kept() -> None:
    # The third go is trimmed to a second and a half, so it stops partway through the
    # sentence. The second go is the last one that finishes.
    pieces, _ = compiled(said_again(1.0, 4.0, 7.0), {2: Trim(kind=TrimKind.HEAD, seconds=1.5)})
    assert [(piece.start, piece.end) for piece in pieces] == [(3.9, 6.1)]

def test_two_silent_windows_are_not_taken_for_two_goes_at_a_line() -> None:
    # Both say nothing, which compares as identical for having nothing to compare.
    # Silent B-roll is not a stammer, so neither of them goes.
    pieces, notes = compiled(said_again(1.0, 4.0, 7.0), which="quiet")
    assert len(pieces) > 1
    assert not any("said again" in note for note in notes)

def test_a_cut_landing_on_black_moves_off_it() -> None:
    made = analysis(
        duration=20.0,
        segments=[(1.0, 3.0, "第一句話"), (3.5, 5.0, "第二句話")],
        silences=[(0.0, 1.0), (3.0, 3.5), (5.0, 20.0)],
        black=[(3.3, 3.45)],
    )
    pieces, notes = compiled(made)
    # The second sentence opens a breath early, at 3.4, which is inside the black. The
    # first frame after it is 3.45, and that is where the cut goes.
    assert pieces[-1].start == pytest.approx(3.45)
    assert any("moved off black" in note for note in notes)

def test_a_cut_with_no_clean_frame_near_enough_is_left_and_said() -> None:
    made = analysis(
        duration=20.0,
        segments=[(1.0, 3.0, "第一句話"), (4.5, 6.0, "第二句話")],
        silences=[(0.0, 1.0), (3.0, 4.5), (6.0, 20.0)],
        black=[(2.5, 5.4)],
    )
    pieces, notes = compiled(made)
    # Nearly three seconds of black, so everything within a second of 4.4 is either
    # still black or in the middle of a word. The cut stays where it was asked for and
    # the fault is said out loud rather than hidden.
    assert pieces[-1].start == pytest.approx(4.4)
    assert any("still land on black" in note for note in notes)
    assert not any("moved off black" in note for note in notes)

def test_the_opening_does_not_start_on_somebody_about_to_speak() -> None:
    by_id, children, assets, section_id = sectioned()
    made = analysis(**FOOTAGE)
    pieces, notes = compile_pieces(plan_for([Selection(clip_id=section_id, beat_id="b1")]),
                                  by_id, children, assets, {ASSET_ID: clean_cuts(made)})
    # A second of nothing before the first word; the cut opens a breath before it instead.
    assert pieces[0].start == pytest.approx(1.0 - BREATH_SECONDS)
    assert any("before the first word" in note for note in notes)

def test_a_trim_that_hits_its_limit_says_how_much_is_still_there() -> None:
    by_id, children, assets, section_id = sectioned()
    made = analysis(**FOOTAGE)
    pieces, notes = compile_pieces(plan_for([Selection(clip_id=section_id, beat_id="b1")]),
                                  by_id, children, assets, {ASSET_ID: clean_cuts(made)})
    # Six seconds of nothing after the last word, and the trim will only take two of
    # them. Saying how much was taken without saying how much is left would read as done.
    assert pieces[-1].end == pytest.approx(20.0 - TAIL_TRIM_SECONDS)
    assert any("is still there" in note for note in notes)

def test_a_shot_held_silent_on_purpose_is_not_trimmed() -> None:
    # Nobody talks anywhere in it, so there is no run-up to measure against and nothing
    # to say the hold was a mistake.
    made = analysis(duration=10.0, silences=[(0.0, 10.0)], scenes=[(0.0, 10.0)])
    pieces, notes = compiled(made, which="quiet")
    assert pieces[0].start == pytest.approx(0.0)
    assert pieces[-1].end == pytest.approx(10.0)
    assert notes == []

def test_nothing_is_cleaned_when_the_compiler_knows_nothing_about_the_footage() -> None:
    by_id, children, assets, section_id = sectioned()
    plan = plan_for([Selection(clip_id=section_id, beat_id="b1")])
    pieces, notes = compile_pieces(plan, by_id, children, assets, None)
    # One window over the whole section, exactly as asked, and nothing claimed about it.
    assert [(piece.start, piece.end) for piece in pieces] == [(0.0, 20.0)]
    assert notes == []

def test_silences_and_black_are_known_even_where_nobody_was_transcribed() -> None:
    # The transcript and the picture detectors are separate passes. Refusing to report
    # the second because the first did not run would leave the compiler blind to what
    # it was told.
    made = analysis(duration=10.0, silences=[(0.0, 2.0)], black=[(4.0, 5.0)], frozen=[(4.5, 6.0)])
    known = clean_cuts(made)
    assert known.words == () and known.sentences == ()
    assert known.pauses == ((0.0, 2.0),)
    # Black and frozen overlap, and a cut point objects to either, so they count once.
    assert known.bad_picture == ((4.0, 6.0),)

# What the review found: three places where the compiler said one thing and did another.

def test_a_pause_the_merge_would_put_back_is_not_taken_out_or_claimed() -> None:
    """Both new edges settle inward, closing the gap to under the merge gap.

    The stage would split, `merge_pieces` would join the halves straight back, and the
    note would report seconds that stayed in the cut. A report that is wrong is
    worse than a pause that stayed.
    """
    made = MediaAnalysis(
        asset_id=ASSET_ID, duration=10.0,
        silences=spans([(0.0, 4.8), (5.0, 5.75), (6.0, 10.0)]),
        transcript=Transcript(language="zh", model="test", segments=[TranscriptSegment(
            start=4.8, end=6.0, text="一二",
            words=[TranscriptWord(start=4.8, end=5.3, text="一"),
                   TranscriptWord(start=5.45, end=6.0, text="二")],
        )]),
    )
    one = sourced(seconds=10.0)
    _, clips = build_timeline({ASSET_ID: one}, {ASSET_ID: made})
    plan = plan_for([Selection(clip_id=clip.id, beat_id="b1") for clip in speech_of(clips)])
    pieces, notes = compile_pieces(plan, {c.id: c for c in clips}, {}, {ASSET_ID: one},
                                   {ASSET_ID: clean_cuts(made)})
    # The silence runs 0.75s, well over the threshold, but settling the two new edges
    # leaves only 0.15s between them — inside the merge gap.
    assert [(piece.start, piece.end) for piece in pieces] == [(4.7, 6.1)]
    assert not any("taken out" in note for note in notes)

def test_speeding_up_the_whole_cut_lays_everything_out_at_the_new_pace() -> None:
    from app.models.plan import Pacing

    plan, by_id, assets = parts()
    plan = plan.model_copy(update={"music": MusicPlan(cues=[MusicCue(asset_id=SONG)])})
    normal = plan_pieces(plan, by_id, {}, assets)
    fast_plan = plan.model_copy(update={"pacing": Pacing(speed=1.25)})
    fast = plan_pieces(fast_plan, by_id, {}, assets)
    # The same footage, played quicker: the cut is shorter by exactly the speed.
    assert [(piece.start, piece.end) for piece in fast] == [(piece.start, piece.end) for piece in normal]
    assert compiled_duration(fast) == pytest.approx(compiled_duration(normal) / 1.25, abs=0.002)
    operations, _ = compile_operations(fast_plan, by_id, {}, assets)
    sequence = [op for op in operations if op.get("track_id") == "main" and op["action"] == "insert_clip"]
    assert all(op["speed"] == 1.25 for op in sequence)
    markers = next(op for op in operations if op["action"] == "set_markers")["markers"]
    assert [marker["timeline_in"] for marker in markers] == pytest.approx([0.0, *cut_points(fast)])
    beds = music_beds(fast_plan, fast, assets)
    assert beds[-1].timeline_in + beds[-1].end - beds[-1].start == pytest.approx(compiled_duration(fast))

def test_a_range_takes_a_stretch_out_of_the_middle_and_plays_it_at_its_own_speed() -> None:
    by_id, children, assets, clips = covered()
    walk = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = plan_for([Selection(
        clip_id=walk.id, beat_id="b1", speed=4.0, trim=Trim(kind=TrimKind.RANGE, from_seconds=5.0, to_seconds=9.0),
    )])
    pieces = plan_pieces(plan, by_id, children, assets)
    assert [(piece.start, piece.end, piece.speed) for piece in pieces] == [(5.0, 9.0, 4.0)]
    # Four seconds of walk at four times the speed is one second of the cut.
    assert compiled_duration(pieces) == 1.0

def test_a_piece_turned_down_on_purpose_is_not_joined_to_its_neighbour() -> None:
    by_id, children, assets, clips = footage()
    first, second = speech_of(clips)[:2]
    plan = plan_for([
        Selection(clip_id=first.id, beat_id="b1"),
        Selection(clip_id=second.id, beat_id="b1", volume=0.0),
    ])
    pieces = plan_pieces(plan, by_id, children, assets)
    # They sit half a second apart and would otherwise be one shot, played at one level.
    assert [piece.gain for piece in pieces] == [1.0, 0.0]

def test_a_selection_s_own_speed_is_not_multiplied_by_the_pacing_speed() -> None:
    by_id, children, assets, clips = covered()
    walk = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    talk = speech_of(clips)[0]
    plan = plan_for([Selection(clip_id=talk.id, beat_id="b1"), Selection(clip_id=walk.id, beat_id="b1", speed=8.0)],
                    pacing=Pacing(speed=1.5))
    # The talking takes the pacing; the timelapse keeps the x8 it was given, which x1.5 on top would take past x8.
    assert [piece.speed for piece in plan_pieces(plan, by_id, children, assets)] == [1.5, 8.0]
    problems, _ = check_plan(plan, SemanticTimeline(id="tl_test", asset_ids=[], input_hash="hash", derivation_version=6),
                             by_id, children, assets)
    assert not any("plays at" in problem for problem in problems)

def test_a_start_the_transcript_placed_late_is_moved_back_onto_the_sound() -> None:
    # 「抽杯」: the sound rises out of silence at 6.35, the transcript's first word is at 6.74.
    made = analysis(duration=12.0, segments=[(6.74, 9.0, "抽杯")], silences=[(0.0, 6.35), (9.0, 12.0)])
    pieces, notes = compiled(made)
    assert pieces[0].start == pytest.approx(6.35 - BREATH_SECONDS)
    assert any("where the sound really starts" in note for note in notes)

def test_a_word_stretched_across_a_silence_does_not_hide_the_sound_after_it() -> None:
    # What the transcriber actually did with 「抽杯」: 「抽」 from 5.44 to 6.74, straight
    # across a silence from 5.70 to 6.34, and 「杯」 from 6.74. The 「抽」 is heard at 6.34.
    made = analysis(duration=12.0, segments=[(6.74, 9.0, "杯子")], silences=[(5.70, 6.34), (9.0, 12.0)])
    made.transcript.segments.insert(0, TranscriptSegment(
        start=5.44, end=6.74, text="抽", words=[TranscriptWord(start=5.44, end=6.74, text="抽")],
    ))
    assert clean_cuts(made).sound_before(6.74) == pytest.approx(6.34)

def test_two_windows_of_one_file_played_out_of_order_stay_two() -> None:
    """#91: 10.17–17.66 then 0–8.92 of one file, next to each other, were joined into one window
    and the second was lost without a word."""
    from app.engine.plan.beats import _left_out
    from app.engine.plan.pieces import merge_pieces

    later = Piece(asset_id="a", start=9.98, end=18.49, from_clip_ids=("u0004",), beat_id="hook")
    earlier = Piece(asset_id="a", start=0.0, end=9.0, from_clip_ids=("u0002",), beat_id="setup")
    assert merge_pieces([later, earlier]) == [later, earlier]
    # In order, they still join.
    assert len(merge_pieces([earlier, replace(later, start=9.1)])) == 1
    # And a window lost anyway is said.
    [lost] = _left_out([later, earlier], [later], [])
    assert lost.kind == "selections_lost" and lost.clip_ids == ("u0002",) and lost.look
