"""Tests for the edit plan: what it refuses, and what it compiles to.

The compiler is the deterministic half of the design, so most of these are
arithmetic: given this plan and this footage, these are the seconds. The rest
check that a plan which does not hold up is handed back rather than quietly
repaired, because a compiler that fixes things is making decisions it was
built to stay out of.
"""

from decimal import Decimal
from pathlib import Path
from typing import Dict, List

import pytest
from pydantic import TypeAdapter

from app.engine.plan import (
    plan_markers,
    BREATH_SECONDS,
    broll_covers,
    broll_slots,
    MERGE_GAP_SECONDS,
    PAUSE_SECONDS,
    SNAP_SECONDS,
    TAIL_TRIM_SECONDS,
    check_plan,
    compile_operations,
    compile_pieces,
    compiled_duration,
    diff_plans,
    plan_pieces,
)
from app.engine.sections import build_sections
from app.engine.semantic import build_timeline, clean_cuts
from app.models.media import (
    Asset, MediaAnalysis, Span, Transcript, TranscriptSegment, TranscriptWord,
)
from app.models.plan import (
    AddBrollOp, Beat, BrollShot, DropBrollOp, DropSelectionOp, EditPlan, MusicPlan, PlanAmendment,
    PlanTarget, Rejection, Selection, Trim, TrimKind, apply_amendment,
)
from app.models.semantic import SectionChoice, SemanticClip
from app.models.timeline import Marker, Project
from app.server import (
    amend_plan,
    apply_edits,
    compile_plan,
    create_project,
    diff_plan,
    get_plan,
    get_project,
    import_asset,
    save_plan,
    validate_plan,
)
from app.server import repo as server_repo
from helpers import video_track
from test_semantic_timeline import ASSET_ID, analysis, asset

FOOTAGE = dict(
    duration=20.0,
    segments=[
        (1.0, 3.0, "第一句話"),
        (3.5, 5.0, "第二句話"),
        (8.0, 10.0, "第三句話"),
        (12.0, 14.0, "第四句話"),
    ],
    silences=[(0.0, 1.0), (3.0, 3.5), (5.0, 8.0), (10.0, 12.0), (14.0, 20.0)],
)

def sourced(asset_id: str = ASSET_ID, seconds: float = 20.0) -> Asset:
    """Build an asset that knows how long it is.

    Args:
        asset_id: ID for the asset.
        seconds: Its duration.

    Returns:
        The asset.
    """
    return asset(asset_id).model_copy(update={"duration": Decimal(str(seconds))})

def footage() -> tuple:
    """Build a timeline over one file of talk.

    Returns:
        `(clips_by_id, children, assets, ordered_clips)`.
    """
    made = analysis(**FOOTAGE)
    _, clips = build_timeline({ASSET_ID: sourced()}, {ASSET_ID: made})
    return {clip.id: clip for clip in clips}, {}, {ASSET_ID: sourced()}, clips

def sectioned() -> tuple:
    """Build a timeline with the whole file as one section.

    Returns:
        `(clips_by_id, children, assets, section_id)`.
    """
    made = analysis(**FOOTAGE)
    timeline, clips = build_timeline({ASSET_ID: sourced()}, {ASSET_ID: made})
    sections, utterances = build_sections(timeline.id, clips, [
        SectionChoice(first_clip_id=clips[0].id, last_clip_id=clips[-1].id, name="全部", summary="整段"),
    ])
    everything = [*utterances, *sections]
    children = {sections[0].id: [clip for clip in utterances]}
    return {clip.id: clip for clip in everything}, children, {ASSET_ID: sourced()}, sections[0].id

def plan_for(selections: List[Selection], **fields) -> EditPlan:
    """Build a plan over one beat.

    Args:
        selections: The chosen footage.
        **fields: Other plan fields.

    Returns:
        The plan.
    """
    return EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash",
        beats=[Beat(id="b1", name="主體")],
        selections=selections,
        **fields,
    )

def speech_of(clips: List[SemanticClip]) -> List[SemanticClip]:
    """Pick out the clips where someone is talking.

    Args:
        clips: Clips to filter.

    Returns:
        The speech clips in order.
    """
    return [clip for clip in clips if clip.text]

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

def spoken_words(*triples) -> MediaAnalysis:
    """Build an analysis of one sentence whose words are timed individually.

    `analysis()` gives each sentence a single word spanning it, which is
    enough for everything else but hides where the words actually are — and
    where the words are is the whole question here.

    Args:
        *triples: `(text, start, end)` per word.

    Returns:
        The analysis, with silence before the first word and after the last.
    """
    first, last = triples[0][1], triples[-1][2]
    made = analysis(
        20.0,
        segments=[(first, last, "".join(word for word, _, _ in triples))],
        silences=[(0.0, first), (last, 20.0)],
    )
    return made.model_copy(update={"transcript": made.transcript.model_copy(update={
        "segments": [made.transcript.segments[0].model_copy(update={
            "words": [TranscriptWord(text=word, start=start, end=end) for word, start, end in triples],
        })],
    })})

SENTENCE = (("今天", 2.0, 3.0), ("四點", 3.0, 4.0), ("就起床", 4.0, 5.5), ("了", 5.5, 6.0))

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

def test_beats_decide_the_order_the_footage_plays_in() -> None:
    by_id, children, assets, clips = footage()
    first, third = speech_of(clips)[0], speech_of(clips)[2]
    plan = EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash",
        beats=[Beat(id="open", name="開場"), Beat(id="body", name="主體")],
        selections=[
            Selection(clip_id=first.id, beat_id="body"),
            Selection(clip_id=third.id, beat_id="open"),
        ],
    )
    # Written body-first, but the beats say the opening comes first.
    assert plan_pieces(plan, by_id, children, assets)[0].start == pytest.approx(third.source_range.start, abs=0.2)

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

def compiled(made: MediaAnalysis, trims: Dict[int, Trim] = {}, which: str = "speech") -> tuple:
    """Compile every clip of one kind in a file, in order.

    Args:
        made: The analysis to derive the timeline from.
        trims: A trim for the selection at that position, where one is wanted.
        which: `speech` to select the talking, `quiet` to select the rest.

    Returns:
        `(pieces, notes)`.
    """
    one = sourced(seconds=made.duration)
    _, clips = build_timeline({ASSET_ID: one}, {ASSET_ID: made})
    chosen = speech_of(clips) if which == "speech" else [clip for clip in clips if not clip.text]
    plan = plan_for([
        Selection(clip_id=clip.id, beat_id="b1", trim=trims.get(place, Trim()))
        for place, clip in enumerate(chosen)
    ])
    return compile_pieces(plan, {clip.id: clip for clip in clips}, {}, {ASSET_ID: one},
                          {ASSET_ID: clean_cuts(made)})

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
    assert (pieces[1].start, pieces[1].end) == (3.9, 8.1)
    assert any("talking across them" in note for note in notes)
    assert not any("were taken out" in note for note in notes)

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
                    music=MusicPlan(asset_id="silent-song"))
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("has no sound" in problem for problem in problems)

def test_the_operations_build_one_video_track_in_order() -> None:
    by_id, children, assets, clips = footage()
    first, third = speech_of(clips)[0], speech_of(clips)[2]
    plan = plan_for([
        Selection(clip_id=first.id, beat_id="b1"),
        Selection(clip_id=third.id, beat_id="b1"),
    ])
    operations, provenance = compile_operations(plan, by_id, children, assets)
    assert operations[0]["action"] == "add_track"
    assert [origin["from_plan_id"] for origin in provenance.values()] == [plan.id, plan.id]
    assert [operation["action"] for operation in operations[1:]] == [
        "insert_clip", "insert_clip", "set_markers",
    ]
    assert operations[1]["source_range"]["start"] < operations[2]["source_range"]["start"]
    # Both selections are the one beat, so the timeline gets one marker, at its start.
    assert operations[3]["markers"] == [
        {"id": "b1", "name": "主體", "timeline_in": 0.0, "from_beat_id": "b1"},
    ]

def test_music_is_laid_under_the_picture_and_fitted_to_it() -> None:
    by_id, children, assets, clips = footage()
    assets = {**assets, "song": sourced("song", seconds=60.0).model_copy(update={"has_video": False})}
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")],
                    music=MusicPlan(asset_id="song", volume=0.4))
    actions = [operation["action"] for operation in compile_operations(plan, by_id, children, assets)[0]]
    assert actions[-3:] == ["add_track", "insert_clip", "fit_track"]

def test_the_same_plan_always_compiles_to_the_same_cut() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=clip.id, beat_id="b1") for clip in speech_of(clips)])
    assert compile_operations(plan, by_id, children, assets) == compile_operations(plan, by_id, children, assets)

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

# B-roll: picture laid over the cut while the sound underneath keeps running. The rules
# about how long it may run and what it may cover are arithmetic, so they live in the
# compiler and are refused with a reason rather than hoped for in a guide.

BROLL_ASSET = "asset-2"

def covered() -> tuple:
    """A timeline of talk, plus a second file of silent picture to lay over it.

    Returns:
        `(clips_by_id, children, assets, ordered_clips)`.
    """
    talk, quiet = analysis(**FOOTAGE), analysis(
        duration=20.0, silences=[(0.0, 20.0)], scenes=[(0.0, 20.0)], asset_id=BROLL_ASSET,
    )
    assets = {ASSET_ID: sourced(), BROLL_ASSET: sourced(BROLL_ASSET)}
    _, clips = build_timeline(assets, {ASSET_ID: talk, BROLL_ASSET: quiet})
    return {clip.id: clip for clip in clips}, {}, assets, clips

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

@pytest.fixture
def planned(media: Path) -> dict:
    """Import a real file, build a timeline over it, and save a plan for it.

    Returns:
        A dictionary with the `plan_id`, `timeline_id` and the chosen clips.
    """
    from app.server import build_semantic_timeline, query_clips

    asset_id = import_asset(str(media / "wide.mp4"))["id"]
    server_repo.save_analysis(analysis(asset_id=asset_id, **FOOTAGE))
    built = build_semantic_timeline([asset_id], rebuild=True)
    speech = query_clips(timeline_id=built["timeline_id"], text="句話")["clips"]
    saved = save_plan(EditPlan(
        timeline_id=built["timeline_id"], timeline_input_hash="",
        goal="做一支短的",
        beats=[Beat(id="b1", name="主體", intent="講完一件事")],
        selections=[
            Selection(clip_id=speech[0]["clip_id"], beat_id="b1", rationale="開場最清楚"),
            Selection(clip_id=speech[2]["clip_id"], beat_id="b1", rationale="收在這句"),
        ],
        rejected=[Rejection(clip_id=speech[1]["clip_id"], reason="重複了")],
    ))
    return {**saved, "clips": speech}

def test_a_saved_plan_is_stamped_with_the_footage_it_was_written_against(planned: dict) -> None:
    stored = get_plan(planned["plan_id"])["plan"]
    assert stored["timeline_input_hash"]
    assert stored["version"] == 1
    assert planned["problems"] == []
    # The reasons survive, which is the point of writing them down.
    assert stored["selections"][0]["rationale"] == "開場最清楚"
    assert stored["rejected"][0]["reason"] == "重複了"

def test_saving_a_plan_at_the_wrong_version_is_refused(planned: dict) -> None:
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    save_plan(stored)
    with pytest.raises(ValueError, match="version conflict"):
        save_plan(stored)

def amendments(*operations: dict) -> list:
    """Validate plan amendments the way a client's would be.

    Args:
        *operations: Amendments as plain dictionaries.

    Returns:
        The parsed amendments.
    """
    return TypeAdapter(List[PlanAmendment]).validate_python(list(operations))

def test_one_trim_can_be_changed_without_sending_the_plan_again(planned: dict) -> None:
    first = planned["clips"][0]["clip_id"]
    result = amend_plan(planned["plan_id"], 1, amendments(
        {"action": "set_trim", "clip_id": first, "trim": {"kind": "head", "seconds": 1.0}},
    ))
    assert result["version"] == 2 and result["problems"] == []
    stored = get_plan(planned["plan_id"])["plan"]
    assert stored["selections"][0]["trim"] == {"kind": "head", "keep_clip_ids": [], "seconds": 1.0}
    # Everything the plan was written to remember is still there.
    assert stored["selections"][0]["rationale"] == "開場最清楚"
    assert stored["rejected"][0]["reason"] == "重複了"

def test_dropping_a_piece_records_why_it_went(planned: dict) -> None:
    second = planned["clips"][2]["clip_id"]
    amend_plan(planned["plan_id"], 1, amendments(
        {"action": "drop_selection", "clip_id": second, "reason": "尾巴太拖"},
    ))
    stored = get_plan(planned["plan_id"])["plan"]
    assert [item["clip_id"] for item in stored["selections"]] == [planned["clips"][0]["clip_id"]]
    assert {item["clip_id"]: item["reason"] for item in stored["rejected"]}[second] == "尾巴太拖"

def test_a_piece_can_be_put_back_where_it_belongs(planned: dict) -> None:
    rejected, last = planned["clips"][1]["clip_id"], planned["clips"][2]["clip_id"]
    amend_plan(planned["plan_id"], 1, amendments(
        {"action": "add_selection", "before_clip_id": last,
         "selection": {"clip_id": rejected, "beat_id": "b1", "rationale": "還是需要這句"}},
    ))
    stored = get_plan(planned["plan_id"])["plan"]
    assert [item["clip_id"] for item in stored["selections"]][1] == rejected
    # It is in the cut now, so it is no longer an answer to "why is it not in there".
    assert [item["clip_id"] for item in stored["rejected"]] == []

def test_amending_a_plan_someone_else_has_moved_on_is_refused(planned: dict) -> None:
    first = planned["clips"][0]["clip_id"]
    change = amendments({"action": "set_trim", "clip_id": first, "trim": {"kind": "tighten"}})
    amend_plan(planned["plan_id"], 1, change)
    with pytest.raises(ValueError, match="version conflict"):
        amend_plan(planned["plan_id"], 1, change)

def test_amending_a_piece_the_plan_does_not_use_is_refused(planned: dict) -> None:
    with pytest.raises(ValueError, match="does not use clip"):
        amend_plan(planned["plan_id"], 1, amendments(
            {"action": "set_trim", "clip_id": "tl_nope:u9999", "trim": {"kind": "full"}},
        ))

def test_a_plan_checks_out_before_anything_is_rendered(planned: dict) -> None:
    result = validate_plan(planned["plan_id"])
    assert result["ok"] is True and result["problems"] == []
    assert result["clips"] >= 1 and result["duration"] > 0

def test_compiling_builds_the_cut_on_an_empty_project(planned: dict) -> None:
    project = create_project(width=640, height=360)["id"]
    result = compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    assert result["clips"] == result["clips"] and result["duration"] > 0
    state = get_project(project)
    assert [track["id"] for track in state["tracks"]] == ["main"]
    assert len(state["tracks"][0]["clips"]) == result["clips"]

def ops(*operations: dict) -> list:
    """Validate operations the way a client's would be.

    Args:
        *operations: Operations as plain dictionaries.

    Returns:
        The validated operations.
    """
    from app.server import _OPERATIONS

    return _OPERATIONS.validate_python(list(operations))

def main_clips(project_id: str) -> List[dict]:
    """Read the sequence track's clips, with every field.

    Args:
        project_id: Project to read.

    Returns:
        The clips in timeline order.
    """
    from app.models.timeline import Clip

    track = next(track for track in get_project(project_id)["tracks"] if track["id"] == "main")
    return [Clip.model_validate(clip).model_dump() for clip in track["clips"]]

def compiled_project(planned: dict) -> str:
    """Compile the plan onto a fresh project.

    Args:
        planned: The saved plan.

    Returns:
        The project ID.
    """
    project = create_project(width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    return project

def test_covering_picture_compiles_onto_a_track_of_its_own(planned: dict, media: Path) -> None:
    """Above the sequence, silent, and without changing how long the video runs."""
    from app.server import build_semantic_timeline, propose_broll, query_clips

    quiet = import_asset(str(media / "silent.mp4"))["id"]
    server_repo.save_analysis(analysis(duration=4.0, scenes=[(0.0, 4.0)], asset_id=quiet))
    plan = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    # Rebuilt over both files, so the plan is rewritten against the new timeline.
    built = build_semantic_timeline([clip["asset_id"] for clip in planned["clips"]] + [quiet], rebuild=True)
    speech = query_clips(timeline_id=built["timeline_id"], text="句話")["clips"]
    cover = query_clips(timeline_id=built["timeline_id"], asset_ids=[quiet])["clips"][0]
    saved = save_plan(EditPlan(
        timeline_id=built["timeline_id"], timeline_input_hash="",
        goal=plan.goal, beats=[Beat(id="b1", name="主體")],
        # The same two sentences the fixture chose: `wide.mp4` runs ten seconds and the
        # analysis describes twenty, so the later ones are not in the file at all.
        selections=[Selection(clip_id=speech[place]["clip_id"], beat_id="b1") for place in (0, 2)],
        broll=[BrollShot(clip_id=cover["clip_id"], over_clip_id=speech[0]["clip_id"], seconds=1.5)],
    ))
    assert saved["problems"] == []
    # Nothing in this cut holds long enough to be worth offering something to cover it.
    assert propose_broll(saved["plan_id"])["slots"] == []

    project = create_project(width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=saved["plan_id"])
    state = get_project(project)
    tracks = {track["id"]: track for track in state["tracks"]}
    assert list(tracks) == ["main", "broll"], list(tracks)
    laid = tracks["broll"]["clips"]
    assert len(laid) == 1 and laid[0]["volume"] == 0.0
    # It remembers what it is made of, so the next compile can match it if it gets pinned.
    assert laid[0]["from_clip_ids"] == [cover["clip_id"], speech[0]["clip_id"]]
    # The sequence decides the length, so covering picture never stretches the video.
    assert state["duration"] == sum(
        clip["source_range"]["end"] - clip["source_range"]["start"] for clip in tracks["main"]["clips"]
    )

def test_a_compiled_clip_records_where_it_came_from(planned: dict) -> None:
    clips = main_clips(compiled_project(planned))
    assert all(clip["from_plan_id"] == planned["plan_id"] for clip in clips)
    assert all(clip["from_clip_ids"] for clip in clips)
    assert not any(clip["pinned"] for clip in clips)

def test_compiling_again_rebuilds_what_the_plan_made(planned: dict) -> None:
    project = compiled_project(planned)
    before = main_clips(project)
    version = get_project(project)["version"]
    result = compile_plan(project_id=project, expected_version=version, plan_id=planned["plan_id"])
    after = main_clips(project)
    # The same plan gives the same cut, and nothing accumulates.
    assert result["kept"] == 0
    assert [(clip["asset_id"], clip["source_range"]) for clip in after] == [
        (clip["asset_id"], clip["source_range"]) for clip in before
    ]

def test_trimming_a_compiled_clip_by_hand_pins_it(planned: dict) -> None:
    project = compiled_project(planned)
    first = main_clips(project)[0]
    apply_edits(project, get_project(project)["version"], ops({
        "action": "trim_clip", "track_id": "main", "clip_id": first["id"],
        "new_source_range": {"start": float(first["source_range"]["start"]),
                             "end": float(first["source_range"]["start"]) + 0.5},
    }))
    # Nobody had to say "pin this": changing it by hand is what pinning records.
    assert main_clips(project)[0]["pinned"] is True

def test_a_hand_adjustment_survives_the_next_compile(planned: dict) -> None:
    project = compiled_project(planned)
    first = main_clips(project)[0]
    shortened = {"start": float(first["source_range"]["start"]), "end": float(first["source_range"]["start"]) + 0.5}
    apply_edits(project, get_project(project)["version"], ops(
        {"action": "trim_clip", "track_id": "main", "clip_id": first["id"], "new_source_range": shortened},
        {"action": "set_clip_audio", "track_id": "main", "clip_id": first["id"], "volume": 0.25},
    ))
    result = compile_plan(project_id=project, expected_version=get_project(project)["version"],
                          plan_id=planned["plan_id"])
    kept = main_clips(project)[0]
    assert result["kept"] == 1
    assert float(kept["source_range"]["end"]) == pytest.approx(shortened["end"])
    assert kept["volume"] == 0.25
    assert kept["pinned"] is True

def test_a_hand_made_transition_and_j_cut_survive_the_next_compile(planned: dict) -> None:
    """A pinned clip is rebuilt by inserting it again, and the insert used to drop these.

    `audio_lead`, `audio_lag` and the transition were not on the operation that
    creates a clip, so a recompile silently handed back a straight cut with its
    sound back on its picture — the one thing pinning promises not to do.
    """
    project = compiled_project(planned)
    clips = main_clips(project)
    second = clips[1]
    apply_edits(project, get_project(project)["version"], ops(
        {"action": "set_clip_audio", "track_id": "main", "clip_id": second["id"], "audio_lead": 0.2},
        {"action": "set_clip_look", "track_id": "main", "clip_id": second["id"],
         "transition_in": {"kind": "wipe", "seconds": 0.3, "direction": "up"}},
    ))
    compile_plan(project_id=project, expected_version=get_project(project)["version"],
                 plan_id=planned["plan_id"])
    kept = server_repo.get_project(project).tracks[0].clips[1]
    assert kept.pinned is True
    assert float(kept.audio_lead) == pytest.approx(0.2)
    assert kept.transition_in is not None
    assert (kept.transition_in.kind, float(kept.transition_in.seconds)) == ("wipe", 0.3)
    assert kept.transition_in.direction.value == "up"

def test_a_recompile_that_would_lose_a_pinned_clip_is_refused(planned: dict) -> None:
    project = compiled_project(planned)
    first = main_clips(project)[0]
    apply_edits(project, get_project(project)["version"], ops({
        "action": "set_clip_pinned", "track_id": "main", "clip_id": first["id"], "pinned": True,
    }))
    # Rewrite the plan so the footage that clip was made from is no longer in it.
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    without = stored.model_copy(update={"selections": stored.selections[1:]})
    save_plan(without)
    with pytest.raises(ValueError, match="adjusted by hand"):
        compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=without.id)
    # Refused means nothing moved.
    assert main_clips(project)[0]["id"] == first["id"]

def test_unpinning_hands_a_clip_back_to_the_plan(planned: dict) -> None:
    project = compiled_project(planned)
    first = main_clips(project)[0]
    original_end = float(first["source_range"]["end"])
    apply_edits(project, get_project(project)["version"], ops({
        "action": "trim_clip", "track_id": "main", "clip_id": first["id"],
        "new_source_range": {"start": float(first["source_range"]["start"]),
                             "end": float(first["source_range"]["start"]) + 0.5},
    }))
    apply_edits(project, get_project(project)["version"], ops({
        "action": "set_clip_pinned", "track_id": "main", "clip_id": first["id"], "pinned": False,
    }))
    compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=planned["plan_id"])
    assert float(main_clips(project)[0]["source_range"]["end"]) == pytest.approx(original_end)

def test_a_clip_the_plan_never_made_blocks_the_compile_until_it_is_pinned(planned: dict) -> None:
    project = create_project(width=640, height=360)["id"]
    asset_id = main_clips(compiled_project(planned))[0]["asset_id"]
    apply_edits(project, 1, ops(
        video_track(),
        {"action": "insert_clip", "track_id": "main", "clip_id": "mine", "asset_id": asset_id,
         "source_range": {"start": 0, "end": 1}},
    ))
    with pytest.raises(ValueError, match="compiling would remove it"):
        compile_plan(project_id=project, expected_version=get_project(project)["version"],
                     plan_id=planned["plan_id"])

def test_tracks_the_plan_does_not_own_are_left_alone(planned: dict) -> None:
    project = compiled_project(planned)
    asset_id = main_clips(project)[0]["asset_id"]
    apply_edits(project, get_project(project)["version"], ops(
        {"action": "add_track", "track_id": "cam", "track_type": "video"},
        {"action": "add_clip", "track_id": "cam", "clip_id": "inset", "asset_id": asset_id,
         "source_range": {"start": 0, "end": 1}, "timeline_in": 0,
         "layout": {"x": 0.6, "y": 0.1, "width": 0.3, "height": 0.3}},
    ))
    compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=planned["plan_id"])
    inset = next(track for track in get_project(project)["tracks"] if track["id"] == "cam")
    assert [clip["id"] for clip in inset["clips"]] == ["inset"]

def test_a_plan_that_does_not_check_out_never_compiles(planned: dict) -> None:
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    broken = stored.model_copy(update={
        "selections": [*stored.selections, Selection(clip_id="not-a-clip", beat_id="b1")],
    })
    save_plan(broken)
    project = create_project(width=640, height=360)["id"]
    with pytest.raises(ValueError, match="cannot be compiled yet"):
        compile_plan(project_id=project, expected_version=1, plan_id=broken.id)
    assert get_project(project)["tracks"] == []

def test_two_plans_can_be_compared_through_the_tools(planned: dict) -> None:
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    other = save_plan(stored.model_copy(update={"id": "plan-b", "version": 1, "goal": "另一種剪法"}))
    changes = diff_plan(planned["plan_id"], other["plan_id"])["changes"]
    assert any("另一種剪法" in line for line in changes["goal"])
    assert planned["plan_id"] in {plan["plan_id"] for plan in get_plan(other["plan_id"])["other_plans"]}


# --- structure markers ------------------------------------------------------------------------

def beats_plan(selections: List[Selection], beats: List[Beat]) -> EditPlan:
    """Build a plan whose beats are given rather than assumed.

    Args:
        selections: The chosen footage.
        beats: The parts of the video.

    Returns:
        The plan.
    """
    return EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash", beats=beats, selections=selections,
    )


def test_each_beat_becomes_a_marker_where_it_starts() -> None:
    """A plan's shape used to exist only in the plan, and be gone the moment it compiled."""
    by_id, children, assets, clips = footage()
    first, second, third = speech_of(clips)[:3]
    plan = beats_plan(
        [
            Selection(clip_id=first.id, beat_id="open"),
            Selection(clip_id=third.id, beat_id="middle"),
        ],
        [Beat(id="open", name="開場"), Beat(id="middle", name="主體")],
    )
    pieces = plan_pieces(plan, by_id, children, assets)
    markers = plan_markers(plan, pieces)
    assert [marker["name"] for marker in markers] == ["開場", "主體"]
    assert markers[0]["timeline_in"] == 0.0
    # The second part starts where the first one's footage ends.
    assert markers[1]["timeline_in"] == pytest.approx(pieces[0].duration, abs=0.001)


def test_a_beat_nothing_was_chosen_for_gets_no_marker() -> None:
    """A part of the plan with no footage in it is not a part of the video."""
    by_id, children, assets, clips = footage()
    plan = beats_plan(
        [Selection(clip_id=speech_of(clips)[0].id, beat_id="open")],
        [Beat(id="open", name="開場"), Beat(id="empty", name="沒拍到")],
    )
    markers = plan_markers(plan, plan_pieces(plan, by_id, children, assets))
    assert [marker["id"] for marker in markers] == ["open"]


def test_a_marker_somebody_added_survives_the_next_compile() -> None:
    """The compiler owns what it made. A marker with no beat behind it is not that."""
    by_id, children, assets, clips = footage()
    plan = beats_plan(
        [Selection(clip_id=speech_of(clips)[0].id, beat_id="open")],
        [Beat(id="open", name="開場")],
    )
    project = Project(id="p", width=1920, height=1080, markers=[
        Marker(id="mine", name="這裡要配樂", timeline_in=Decimal("1.5")),
        Marker(id="open", name="舊的開場", timeline_in=Decimal(0), from_beat_id="open"),
    ])
    operations, _ = compile_operations(plan, by_id, children, assets, project)
    written = next(op for op in operations if op["action"] == "set_markers")["markers"]
    names = {marker["id"]: marker["name"] for marker in written}
    # The compiler's own marker is rebuilt from the plan; the hand-made one is carried over.
    assert names["open"] == "開場"
    assert names["mine"] == "這裡要配樂"


def test_markers_land_on_the_timeline_in_order() -> None:
    """They are written as one set, and read back sorted by where they are."""
    project_id = create_project(width=1920, height=1080)["id"]
    apply_edits(project_id, 1, ops({"action": "set_markers", "markers": [
        {"id": "b", "name": "結尾", "timeline_in": 9},
        {"id": "a", "name": "開場", "timeline_in": 0},
    ]}))
    stored = server_repo.get_project(project_id)
    assert [marker.name for marker in stored.markers] == ["開場", "結尾"]


def test_two_markers_cannot_share_a_name_on_the_timeline() -> None:
    """An id is how a marker is referred to, so two of them is a broken set."""
    project_id = create_project(width=1920, height=1080)["id"]
    with pytest.raises(ValueError, match="share the id"):
        apply_edits(project_id, 1, ops({"action": "set_markers", "markers": [
            {"id": "a", "name": "開場", "timeline_in": 0},
            {"id": "a", "name": "結尾", "timeline_in": 9},
        ]}))
