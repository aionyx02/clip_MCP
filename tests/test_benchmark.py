"""Tests for L2: scoring a compiled cut against an annotated case.

The metrics are pure functions of the compiler's output and the analysis on
disk, so these are arithmetic against numbers chosen to exercise each rule, in
the same style as the compiler's own tests. One case at the end runs the whole
chain — analysis to timeline to plan to pieces to score — because a metric that
is right about hand-built pieces and wrong about real ones is worth nothing.
"""

import json
from pathlib import Path
from typing import List

import pytest

from app.benchmark.case import BenchmarkCase, MarkedSpan, load_case, load_cases
from app.benchmark.metrics import (
    MAX_LENGTH_ERROR,
    MIN_MUST_KEEP_COVERAGE,
    Scorecard,
    missing_analyses,
    score,
)
from app.engine.plan import Piece, plan_pieces
from app.engine.semantic import build_timeline
from app.models.plan import Beat, EditPlan, Selection
from test_edit_plan import FOOTAGE, sourced, speech_of
from test_semantic_timeline import ASSET_ID, analysis

FILE = "day1/clip.mp4"
ASSETS = {FILE: ASSET_ID}

def case_for(**fields) -> BenchmarkCase:
    """Build a case over one file.

    Args:
        **fields: Case fields to set beyond the footage and instruction.

    Returns:
        The case.
    """
    return BenchmarkCase(
        id="c1", instruction="剪成一支短的", footage=[FILE], **fields,
    )

def marked(start: float, end: float, why: str = "") -> MarkedSpan:
    """Build an annotation over the case's one file.

    Args:
        start: Start in that file, in seconds.
        end: End in that file, in seconds.
        why: The reason, carried through to the reader.

    Returns:
        The span.
    """
    return MarkedSpan(file=FILE, start=start, end=end, why=why)

def piece(start: float, end: float) -> Piece:
    """Build one compiled window over the case's one file.

    Args:
        start: Where it starts in the source, in seconds.
        end: Where it ends, in seconds.

    Returns:
        The window.
    """
    return Piece(ASSET_ID, start, end, ("tl_test:u0001",))

def scored(pieces: List[Piece], made=None, **fields) -> Scorecard:
    """Score windows against a case built from the given fields.

    Args:
        pieces: The compiled windows.
        made: The analysis to score against; a silent 20 s file by default.
        **fields: Case fields.

    Returns:
        The scorecard.
    """
    return score(case_for(**fields), pieces, {ASSET_ID: made or analysis(20.0)}, ASSETS)

def test_a_case_will_not_annotate_footage_it_does_not_have() -> None:
    with pytest.raises(ValueError, match="not in the case's footage"):
        BenchmarkCase(
            id="c1", instruction="剪短一點", footage=[FILE],
            must_keep=[MarkedSpan(file="day2/other.mp4", start=0, end=1)],
        )

def test_a_case_will_not_mark_the_same_stretch_both_ways() -> None:
    with pytest.raises(ValueError, match="both must-keep and must-drop"):
        case_for(must_keep=[marked(5, 10)], must_drop=[marked(8, 12)])

def test_a_span_that_ends_before_it_starts_is_refused() -> None:
    with pytest.raises(ValueError, match="not after its start"):
        MarkedSpan(file=FILE, start=10, end=4)

def test_cases_load_in_name_order_and_report_the_file_that_is_wrong(tmp_path: Path) -> None:
    (tmp_path / "b.json").write_text(json.dumps(
        {"id": "b", "instruction": "b", "footage": [FILE]}), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps(
        {"id": "a", "instruction": "a", "footage": [FILE]}), encoding="utf-8")
    assert [case.id for case in load_cases(tmp_path)] == ["a", "b"]

    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.json"):
        load_case(tmp_path / "broken.json")

def test_two_cases_cannot_share_an_id(tmp_path: Path) -> None:
    for name in ("one", "two"):
        (tmp_path / f"{name}.json").write_text(json.dumps(
            {"id": "same", "instruction": name, "footage": [FILE]}), encoding="utf-8")
    with pytest.raises(ValueError, match="both case 'same'"):
        load_cases(tmp_path)

def test_coverage_is_weighted_by_seconds_not_by_span() -> None:
    # A ten-second stretch kept whole and a one-second aside missed entirely:
    # counting spans would call that 50%, which would let an aside outvote the point.
    card = scored(
        [piece(0.0, 10.0)],
        must_keep=[marked(0.0, 10.0, "主線"), marked(15.0, 16.0, "補一句")],
    )
    assert card.must_keep_coverage == pytest.approx(10 / 11, abs=0.001)

def test_footage_that_had_to_go_and_got_in_is_counted() -> None:
    card = scored([piece(4.0, 8.0)], must_drop=[marked(6.0, 10.0, "講錯")])
    # Two of the four marked seconds are in the cut.
    assert card.must_drop_leakage == pytest.approx(0.5)
    assert any("had to go" in failure for failure in card.failures)

def test_an_unannotated_case_scores_nothing_rather_than_zero() -> None:
    card = scored([piece(0.0, 5.0)])
    # None and 0.0 are different answers: one is "nobody said", the other is "missed it all".
    assert card.must_keep_coverage is None and card.must_drop_leakage is None
    assert card.clean

def test_length_is_measured_against_the_target_the_case_names() -> None:
    assert scored([piece(0.0, 9.0)], target_seconds=10.0).length_error == pytest.approx(0.1)
    assert scored([piece(0.0, 9.0)]).length_error is None

def test_a_cut_through_the_middle_of_a_word_is_counted() -> None:
    made = analysis(20.0, segments=[(1.0, 3.0, "第一句話")], silences=[(3.0, 20.0)])
    assert scored([piece(0.0, 2.0)], made=made).words_cut == 1
    # Landing on the word's own edge keeps or drops the whole word, which is clean.
    assert scored([piece(0.0, 3.0)], made=made).words_cut == 0

def test_a_cut_on_black_or_frozen_picture_is_counted() -> None:
    made = analysis(20.0, black=[(5.0, 7.0)], frozen=[(12.0, 14.0)])
    # Opens on the first black frame and runs on into the frozen stretch.
    assert scored([piece(5.0, 13.0)], made=made).bad_frame_cuts == 2

def test_which_end_of_a_window_a_bad_frame_touches_decides_whether_it_is_seen() -> None:
    made = analysis(20.0, black=[(5.0, 7.0)], frozen=[(12.0, 14.0)])
    # Opens on the frame after the black ends and closes on the frame before the
    # freeze begins, so the viewer never sees either: both edges are clean.
    assert scored([piece(7.0, 12.0)], made=made).bad_frame_cuts == 0
    # The same two numbers the other way round are both on screen.
    assert scored([piece(6.0, 13.0)], made=made).bad_frame_cuts == 2

def test_each_cut_reports_the_room_it_had() -> None:
    made = analysis(20.0, segments=[(4.0, 8.0, "講話")], silences=[(0.0, 4.0), (8.0, 20.0)])
    card = scored([piece(2.0, 10.0)], made=made)
    # 2.0 sits in a silence running 0 to 4, so it could move two seconds either way;
    # 10.0 sits in one running 8 to 20 and is two seconds past its start.
    assert card.cut_margins == (2.0, 2.0)
    assert card.tightest_margin == 2.0 and card.median_margin == 2.0

def test_a_cut_that_is_not_in_a_silence_had_no_room() -> None:
    made = analysis(20.0, segments=[(4.0, 8.0, "講話")], silences=[(0.0, 4.0), (8.0, 20.0)])
    assert scored([piece(5.0, 6.0)], made=made).cut_margins == (0.0, 0.0)

def test_every_floor_that_was_missed_is_named(tmp_path: Path) -> None:
    made = analysis(20.0, segments=[(1.0, 5.0, "整段都在講")], silences=[(5.0, 20.0)], black=[(0.0, 1.0)])
    card = scored(
        [piece(0.5, 3.0)], made=made,
        target_seconds=10.0,
        must_keep=[marked(10.0, 20.0, "結論")],
        must_drop=[marked(2.0, 4.0, "講錯")],
    )
    assert not card.clean
    joined = " ".join(card.failures)
    assert "middle of a word" in joined
    assert "black or frozen" in joined
    assert "target" in joined
    assert "had to be kept" in joined and "had to go" in joined

def test_a_clean_cut_says_so() -> None:
    made = analysis(20.0, segments=[(4.0, 8.0, "講話")], silences=[(0.0, 4.0), (8.0, 20.0)])
    card = scored(
        [piece(3.9, 8.1)], made=made,
        target_seconds=4.2,
        must_keep=[marked(4.0, 8.0, "這句")],
        must_drop=[marked(15.0, 18.0, "不要")],
    )
    assert card.failures == () and card.clean
    assert card.must_keep_coverage == 1.0 and card.must_drop_leakage == 0.0
    assert card.length_error < MAX_LENGTH_ERROR

def test_scoring_against_footage_that_was_never_imported_is_refused() -> None:
    with pytest.raises(ValueError, match="annotated but was not imported"):
        score(case_for(must_keep=[marked(0, 5)]), [piece(0, 5)], {ASSET_ID: analysis(20.0)}, {})

def test_unanalyzed_footage_is_named_rather_than_scored_as_clean() -> None:
    card = score(case_for(), [piece(0.0, 5.0)], {}, ASSETS)
    # Nothing was measured, so nothing was found — which must not read as a pass.
    assert card.words_cut == 0 and card.bad_frame_cuts == 0 and card.cut_margins == ()
    assert missing_analyses([piece(0.0, 5.0)], {}) == [ASSET_ID]

def test_a_plan_compiled_from_real_footage_scores_end_to_end() -> None:
    # The whole chain, so a metric cannot be right about hand-built windows and
    # wrong about the ones the compiler actually produces.
    made = analysis(**FOOTAGE)
    assets = {ASSET_ID: sourced()}
    timeline, clips = build_timeline(assets, {ASSET_ID: made})
    spoken = speech_of(clips)
    plan = EditPlan(
        timeline_id=timeline.id,
        beats=[Beat(id="b1", name="主體", intent="講完一件事")],
        selections=[
            Selection(clip_id=spoken[0].id, beat_id="b1", rationale="開場"),
            Selection(clip_id=spoken[2].id, beat_id="b1", rationale="收尾"),
        ],
    )
    pieces = plan_pieces(plan, {clip.id: clip for clip in clips}, {}, assets)

    card = score(
        BenchmarkCase(
            id="end-to-end", instruction="留開頭和第三句", footage=[FILE],
            target_seconds=4.4,
            must_keep=[marked(1.0, 3.0, "第一句"), marked(8.0, 10.0, "第三句")],
            must_drop=[marked(3.5, 5.0, "第二句不要")],
        ),
        pieces, {ASSET_ID: made}, ASSETS,
    )
    # The compiler widens each sentence into the silence around it, so both
    # annotated stretches survive whole and the rejected one never gets in.
    assert card.must_keep_coverage == 1.0
    assert card.must_drop_leakage == 0.0
    # And it cuts in the measured silences, which is the whole point of safe_in/safe_out.
    assert card.words_cut == 0 and card.bad_frame_cuts == 0
    assert card.tightest_margin > 0
    assert card.must_keep_coverage > MIN_MUST_KEEP_COVERAGE
    assert card.clean
