"""Tests for L2: scoring a compiled cut against an annotated case.

The metrics are pure functions of the compiler's output and the analysis on
disk, so these are arithmetic against numbers chosen to exercise each rule, in
the same style as the compiler's own tests. One case at the end runs the whole
chain — analysis to timeline to plan to pieces to score — because a metric that
is right about hand-built pieces and wrong about real ones is worth nothing.

The corpus itself is not checked out in the repo: its footage lives on one
machine and its annotations quote somebody's own work. What is checked here is
that a case built by hand holds up, and that the seed command writes one.
"""

import json
from pathlib import Path
from typing import List

import pytest

from app.benchmark.case import (
    BenchmarkCase,
    CaptionLine,
    CaseWord,
    FileTranscript,
    FootageFile,
    MarkedSpan,
    file_digest,
    load_case,
    load_cases,
    verify_footage,
)
from app.benchmark.metrics import (
    MAX_LENGTH_ERROR,
    MIN_MUST_KEEP_COVERAGE,
    Scorecard,
    cer,
    characters,
    score,
    tokens,
    transcript_errors,
    transcript_score,
    wer,
)
from app.benchmark.seed import main as seed_main
from app.engine.plan.beats import plan_pieces
from app.engine.plan.pieces import BREATH_SECONDS, Piece
from app.engine.semantic import build_timeline, clean_cuts
from app.models.media import Asset, MediaAnalysis, Transcript
from app.models.plan import Beat, EditPlan, Selection
from edit_plan.shared import FOOTAGE, sourced, speech_of
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

def transcribed(made: MediaAnalysis) -> MediaAnalysis:
    """Mark an analysis as having been through speech recognition.

    `analysis()` leaves the transcript out when a test names no sentences, and
    the scorer reads a missing transcript as footage nobody transcribed —
    right in general, but these tests mean "transcribed, and nobody was
    talking", which is an answer rather than a gap.

    Args:
        made: The analysis to mark.

    Returns:
        The analysis, with an empty transcript if it had none.
    """
    if made.transcript is not None:
        return made
    return made.model_copy(update={"transcript": Transcript(language="zh", model="test", segments=[])})

def scored(pieces: List[Piece], made=None, **fields) -> Scorecard:
    """Score windows against a case built from the given fields.

    Args:
        pieces: The compiled windows.
        made: The analysis to score against; a silent 20 s file by default.
        **fields: Case fields.

    Returns:
        The scorecard.
    """
    return score(case_for(**fields), pieces, {ASSET_ID: transcribed(made or analysis(20.0))}, ASSETS)

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
    assert card.unmeasured == (f"{ASSET_ID} has no analysis, so nothing in it was measured",)
    assert not card.clean

def test_footage_transcribed_without_word_timings_cannot_clear_the_word_floor() -> None:
    # faster-whisper times every word, but an asset analyzed with transcribe off has
    # no transcript at all — and then every cut through it reads as not clipping a
    # word, because there are no words to compare against.
    made = analysis(20.0, segments=[(1.0, 5.0, "整段都在講")], silences=[(5.0, 20.0)])
    stripped = made.model_copy(update={"transcript": None})
    card = score(case_for(), [piece(2.0, 4.0)], {ASSET_ID: stripped}, ASSETS)
    assert card.words_cut == 0
    assert not card.clean
    assert "word timings" in card.unmeasured[0] or "no analysis" in card.unmeasured[0]

def test_footage_that_was_transcribed_and_had_nobody_talking_is_measured() -> None:
    # An empty transcript is an answer; a missing one is a gap. They must not be
    # treated alike, or silent B-roll would never clear a floor.
    card = scored([piece(0.0, 5.0)])
    assert card.unmeasured == () and card.clean

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
    pieces = plan_pieces(plan, {clip.id: clip for clip in clips}, {}, assets, {ASSET_ID: clean_cuts(made)})

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
    assert card.unmeasured == ()
    assert card.clean

def test_cuts_are_judged_against_the_windows_the_case_allows() -> None:
    made = analysis(20.0, segments=[(4.0, 8.0, "講話")], silences=[(0.0, 4.0), (8.0, 20.0)])
    allowed = [marked(0.0, 4.0, "開頭的靜音"), marked(8.0, 20.0, "結尾的靜音")]
    assert scored([piece(2.0, 10.0)], made=made, cut_windows=allowed).cuts_outside_windows == 0
    # Both ends land in the middle of the sentence the annotator ruled out.
    card = scored([piece(5.0, 6.0)], made=made, cut_windows=allowed)
    assert card.cuts_outside_windows == 2
    assert any("outside the windows" in failure for failure in card.failures)

def test_a_file_nobody_drew_windows_for_is_not_judged_on_them() -> None:
    # A gap in the corpus must not read as a bad plan.
    assert scored([piece(5.0, 6.0)]).cuts_outside_windows is None

def test_leakage_below_the_reported_precision_still_fails() -> None:
    # The target is zero, so a leak too small to print is still a leak; comparing on
    # the rounded number would let anything under 0.005% read as clean.
    card = scored([piece(0.0, 10.0)], must_drop=[marked(9.9999, 1009.9999, "不要")])
    assert card.must_drop_leakage == 0.0
    assert not card.clean and any("had to go" in failure for failure in card.failures)

def test_the_room_a_cut_had_is_the_headroom_the_semantic_layer_measures() -> None:
    # The two must not drift: safe_in/safe_out is what the compiler cuts by, and this
    # is what the score judges it on. A cut on a silence's own edge has no room in the
    # tight direction, and both say so.
    made = analysis(20.0, segments=[(4.0, 8.0, "講話")], silences=[(0.0, 4.0), (8.0, 20.0)])
    _, clips = build_timeline({ASSET_ID: sourced()}, {ASSET_ID: made})
    spoken = next(clip for clip in clips if clip.text)
    # The clip's headroom is the whole silence on each side; the compiler only ever
    # spends BREATH_SECONDS of it.
    assert spoken.safe_in == pytest.approx(4.0) and spoken.safe_out == pytest.approx(12.0)

    # So the cut opens a breath early and closes a breath late, and the room each cut
    # point reports is exactly the breath that was spent.
    card = scored([piece(4.0 - BREATH_SECONDS, 8.0 + BREATH_SECONDS)], made=made)
    assert card.cut_margins == (BREATH_SECONDS, BREATH_SECONDS)
    # And a cut on the edge itself has none, which is the number, not a missing one.
    assert scored([piece(4.0, 8.0)], made=made).cut_margins == (0.0, 0.0)

# --- provenance: a case says whose words it carries and who checked its annotations -------

def test_a_draft_case_scores_nothing_until_somebody_has_been_through_it() -> None:
    """Seeded from an existing cut, every chosen stretch is must-keep — a model answer.

    Scoring against that measures how closely a plan reproduces the old cut,
    which is the one thing the corpus is not for, so a draft belongs with the
    floors nobody could test rather than among the ones that passed.
    """
    assert case_for(must_keep=[marked(0, 4)], annotated_by="draft").is_draft
    card = scored([piece(0, 4)], must_keep=[marked(0, 4)], annotated_by="draft")
    assert any("unreviewed draft" in note for note in card.unmeasured)
    # Full coverage and no failures, and still not clean: nothing here is evidence.
    assert card.must_keep_coverage == 1.0
    assert not card.failures
    assert not card.clean

def test_a_reviewed_case_is_evidence() -> None:
    """The same annotations, once a person has stood behind them, do score."""
    assert not case_for(must_keep=[marked(0, 4)]).is_draft
    card = scored([piece(0, 4)], must_keep=[marked(0, 4)])
    assert not any("draft" in note for note in card.unmeasured)

def test_a_case_defaults_to_being_somebody_own_reviewed_work() -> None:
    """Hand-written cases are the normal kind, so they need neither field."""
    case = case_for()
    assert (case.instruction_by, case.annotated_by) == ("user", "user")
    assert not case.is_draft

def test_an_instruction_reconstructed_after_the_fact_says_so() -> None:
    """An instruction written by somebody who had already seen the answer is not the ask."""
    case = case_for(instruction_by="derived")
    assert case.instruction_by == "derived"

# --- the corpus that ships with the repo --------------------------------------------------

def test_the_corpus_holds_up_if_there_is_one() -> None:
    """The cases are edited by hand, so a malformed one should fail here and not mid-run.

    Skipped where the corpus is absent: the annotations quote somebody's own
    brief and point at footage on one machine, so they are not in the repo.
    """
    corpus = Path(__file__).resolve().parent.parent / "corpus"
    cases = sorted(corpus.glob("*.json")) if corpus.is_dir() else []
    if not cases:
        pytest.skip("no corpus checked out")
    loaded = load_cases(corpus)
    assert len(loaded) == len(cases)
    for case in loaded:
        assert case.footage, f"{case.id} lists no footage"
        # The finished exports live beside the footage. One reaching a case would be
        # handing the answer to whatever is being measured.
        for name in case.footage:
            stem = Path(name).stem
            assert not (stem[:4].isdigit() and len(stem) <= 8), f"{case.id}: {name} looks like a finished cut"

# --- what the footage is, and whether it is still the same footage ------------------------

def test_every_pinned_file_has_to_be_in_the_case_footage() -> None:
    with pytest.raises(ValueError, match="not in the case's footage"):
        case_for(footage_files=[
            FootageFile(file="day2/other.mp4", sha256="a" * 64, bytes=10),
        ])

def test_footage_cannot_be_pinned_twice() -> None:
    pinned = FootageFile(file=FILE, sha256="a" * 64, bytes=10)
    with pytest.raises(ValueError, match="pinned twice"):
        case_for(footage_files=[pinned, pinned])

def test_a_digest_that_is_not_sha256_hex_is_refused() -> None:
    with pytest.raises(ValueError):
        FootageFile(file=FILE, sha256="not a digest", bytes=10)
    with pytest.raises(ValueError):
        FootageFile(file=FILE, sha256="A" * 64, bytes=10)

def test_a_file_of_no_bytes_is_refused() -> None:
    with pytest.raises(ValueError):
        FootageFile(file=FILE, sha256="a" * 64, bytes=0)
    with pytest.raises(ValueError):
        FootageFile(file=FILE, sha256="a" * 64, bytes=10, duration=0)

def test_a_case_says_which_split_it_belongs_to() -> None:
    assert case_for().split == "tune"
    assert case_for(split="holdout").split == "holdout"
    with pytest.raises(ValueError):
        case_for(split="validation")

def test_a_case_says_what_sort_of_footage_it_is() -> None:
    assert case_for().kinds == []
    assert case_for(kinds=["solo vlog", "noisy outdoors"]).kinds == ["solo vlog", "noisy outdoors"]

def test_file_digest_reads_the_file_a_block_at_a_time() -> None:
    path = Path(__file__).resolve().parent / "test_benchmark.py"
    digest, size = file_digest(path)
    assert size == path.stat().st_size
    assert len(digest) == 64 and digest == digest.lower()

def test_verify_footage_says_nothing_when_every_file_matches(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.mp4").write_bytes(b"footage")
    digest, size = file_digest(tmp_path / "raw" / "a.mp4")
    case = BenchmarkCase(
        id="c1", instruction="剪成一支短的", footage=[FILE, "raw/a.mp4"],
        footage_files=[FootageFile(file="raw/a.mp4", sha256=digest, bytes=size)],
    )
    assert verify_footage(case, tmp_path) == []

def test_verify_footage_names_a_missing_file(tmp_path: Path) -> None:
    case = case_for(footage_files=[FootageFile(file=FILE, sha256="a" * 64, bytes=10)])
    assert "missing" in verify_footage(case, tmp_path)[0]

def test_verify_footage_names_a_file_whose_size_changed(tmp_path: Path) -> None:
    path = tmp_path / FILE
    path.parent.mkdir()
    path.write_bytes(b"a longer file than recorded")
    case = case_for(footage_files=[FootageFile(file=FILE, sha256="a" * 64, bytes=4)])
    assert "size" in verify_footage(case, tmp_path)[0]

def test_verify_footage_names_a_file_whose_digest_changed(tmp_path: Path) -> None:
    path = tmp_path / FILE
    path.parent.mkdir()
    path.write_bytes(b"abcd")
    case = case_for(footage_files=[FootageFile(file=FILE, sha256="a" * 64, bytes=4)])
    assert "digest" in verify_footage(case, tmp_path)[0]

def test_a_case_whose_footage_does_not_verify_is_not_scored(tmp_path: Path) -> None:
    # The file must be there for the case to be about it at all, so the cut is
    # measured; what is refused is calling any of it evidence.
    card = score(
        case_for(footage_files=[FootageFile(file=FILE, sha256="a" * 64, bytes=10)]),
        [piece(0.0, 5.0)], {ASSET_ID: transcribed(analysis(20.0))}, ASSETS, root=tmp_path,
    )
    assert any("missing" in note for note in card.unmeasured)
    assert not card.clean

def test_footage_not_checked_is_named_rather_than_assumed_good() -> None:
    card = scored(
        [piece(0.0, 5.0)],
        footage_files=[FootageFile(file=FILE, sha256="a" * 64, bytes=10)],
    )
    assert any("not checked" in note for note in card.unmeasured)
    assert not card.clean

# --- transcripts: checked ones score, drafts do not ---------------------------------------

def test_a_transcript_word_has_to_run_forwards() -> None:
    with pytest.raises(ValueError):
        CaseWord(text="a", start=2, end=1)
    with pytest.raises(ValueError, match="before the one in front"):
        FileTranscript(file=FILE, words=[
            CaseWord(text="b", start=2, end=3),
            CaseWord(text="a", start=1, end=1.5),
        ])

def test_a_transcript_of_footage_the_case_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError, match="not in the case's footage"):
        case_for(transcripts=[FileTranscript(file="day2/other.mp4")])

def test_a_caption_line_that_ends_before_it_starts_is_refused() -> None:
    with pytest.raises(ValueError, match="not after its start"):
        case_for(caption_lines=[CaptionLine(file=FILE, start=5, end=4, text="一句")])

def test_a_caption_line_about_footage_the_case_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError, match="not in the case's footage"):
        case_for(caption_lines=[CaptionLine(file="day2/other.mp4", start=0, end=1, text="一句")])

def test_tokens_break_chinese_into_characters_and_english_into_words() -> None:
    assert tokens("今天 we go 到台北!") == ["今", "天", "we", "go", "到", "台", "北"]

def test_tokens_strip_punctuation_and_symbols() -> None:
    # Punctuation and symbols are removed rather than turning into a break, so
    # "NT$100" is the letters and digits either side of the dollar.
    assert tokens("Hello,  world!! (2026) — NT$100") == ["hello", "world", "2026", "nt100"]

def test_characters_are_the_same_normalisation_one_character_at_a_time() -> None:
    assert characters("今天 we go 到台北!") == ["今", "天", "w", "e", "g", "o", "到", "台", "北"]

def test_transcript_errors_counts_substitutions_deletions_and_insertions() -> None:
    assert transcript_errors(["a", "b", "c"], ["a", "b", "c"]) == (0, 3)
    assert transcript_errors(["a", "b", "c"], ["a", "x", "c"]) == (1, 3)
    assert transcript_errors(["a", "b", "c"], ["a", "c"]) == (1, 3)
    assert transcript_errors(["a", "c"], ["a", "b", "c"]) == (1, 2)
    assert transcript_errors([], []) == (0, 0)

def test_wer_counts_a_substitution_deletion_and_insertion() -> None:
    assert wer("one two three", "one two three") == 0.0
    assert wer("one two three", "one too three") == pytest.approx(1 / 3)
    assert wer("one two three", "one three") == pytest.approx(1 / 3)
    assert wer("one three", "one two three") == pytest.approx(0.5)

def test_an_empty_reference_is_perfect_only_when_nothing_was_heard() -> None:
    assert wer("", "") == 0.0
    assert wer("", "something") == 1.0
    assert cer("", "") == 0.0
    assert cer("", "x") == 1.0

def test_cer_counts_characters_rather_than_words() -> None:
    assert cer("今天好", "今天很好") == pytest.approx(1 / 3)
    assert cer("abc", "axc") == pytest.approx(1 / 3)

def test_wer_over_chinese_counts_characters() -> None:
    # Said in the docstring so nobody reads a "word rate" over Chinese as words.
    assert wer("今天很好", "今天好") == pytest.approx(1 / 4)

def test_a_checked_transcript_is_scored_against_what_was_heard() -> None:
    made = analysis(20.0, segments=[(0.0, 2.0, "今天很好")])
    case = case_for(transcripts=[FileTranscript(
        file=FILE, checked_by="user",
        words=[CaseWord(text="今天很好", start=0.0, end=2.0)],
    )])
    scores = transcript_score(case, {FILE: made})
    assert scores[FILE]["wer"] == 0.0 and scores[FILE]["cer"] == 0.0
    assert scores[FILE]["words"] == 4

def test_a_draft_transcript_scores_nothing() -> None:
    made = analysis(20.0, segments=[(0.0, 2.0, "今天很好")])
    case = case_for(transcripts=[FileTranscript(
        file=FILE,
        words=[CaseWord(text="完全不一樣", start=0.0, end=2.0)],
    )])
    assert case.transcripts[0].checked_by == "draft"
    assert transcript_score(case, {FILE: made}) == {}
    card = score(case, [piece(0.0, 5.0)], {ASSET_ID: made}, ASSETS)
    assert card.transcripts == {}
    assert any("draft" in note for note in card.unmeasured)
    assert not card.clean

def test_a_checked_transcript_is_carried_on_the_scorecard() -> None:
    made = analysis(20.0, segments=[(0.0, 2.0, "今天很好")])
    case = case_for(transcripts=[FileTranscript(
        file=FILE, checked_by="user",
        words=[CaseWord(text="今天很好", start=0.0, end=2.0)],
    )])
    card = score(case, [piece(0.0, 5.0)], {ASSET_ID: made}, ASSETS)
    assert card.transcripts[FILE] == {"wer": 0.0, "cer": 0.0, "words": 4}
    assert card.unmeasured == () and card.clean

# --- seeding a draft from files on disk ---------------------------------------------------

class FakeRepository:
    """A workspace repository with only the two lookups seeding makes."""

    def __init__(self, assets: List[Asset], analyses: dict) -> None:
        self.assets = {asset.path: asset for asset in assets}
        self.analyses = analyses

    def find_asset_by_path(self, path: str):
        return self.assets.get(path)

    def get_analysis(self, asset_id: str):
        return self.analyses.get(asset_id)

def seed_argv(tmp_path: Path, *extra: str) -> List[str]:
    """Build the arguments for a seed run over two small files.

    Args:
        tmp_path: Corpus root to put the files under.
        *extra: More arguments, placed before the file names.

    Returns:
        The argument list.
    """
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "raw" / "A.MP4").write_bytes(b"first file")
    (tmp_path / "raw" / "B.MP4").write_bytes(b"second file, longer")
    return [
        "--root", str(tmp_path), "--case-id", "0828-trip", "--instruction", "剪成一分鐘",
        "--corpus", str(tmp_path / "corpus"), *extra,
        "raw/A.MP4", "raw/B.MP4",
    ]

def test_seed_writes_a_draft_case_with_hashes(tmp_path: Path) -> None:
    argv = seed_argv(tmp_path)
    repository = FakeRepository([], {})
    assert seed_main(argv, repository=repository) == 0
    written = json.loads((tmp_path / "corpus" / "0828-trip.json").read_text(encoding="utf-8"))
    assert written["id"] == "0828-trip"
    assert written["annotated_by"] == "draft" and written["split"] == "tune"
    assert written["footage"] == ["raw/A.MP4", "raw/B.MP4"]
    assert [entry["file"] for entry in written["footage_files"]] == ["raw/A.MP4", "raw/B.MP4"]
    for entry in written["footage_files"]:
        digest, size = file_digest(tmp_path / entry["file"])
        assert entry["sha256"] == digest and entry["bytes"] == size

def test_seed_copies_a_transcript_from_the_workspace_as_a_draft(tmp_path: Path) -> None:
    argv = seed_argv(tmp_path)
    file_path = str(tmp_path / "raw" / "A.MP4")
    made = analysis(2.0, segments=[(0.0, 1.0, "你好")])
    repository = FakeRepository(
        [Asset(id="asset-1", path=file_path, has_video=True, has_audio=True)],
        {"asset-1": made},
    )
    assert seed_main(argv, repository=repository) == 0
    written = json.loads((tmp_path / "corpus" / "0828-trip.json").read_text(encoding="utf-8"))
    assert len(written["transcripts"]) == 1
    transcript = written["transcripts"][0]
    assert transcript["file"] == "raw/A.MP4" and transcript["checked_by"] == "draft"
    assert [word["text"] for word in transcript["words"]] == ["你好"]

def test_seed_refuses_to_overwrite_a_case_without_force(tmp_path: Path) -> None:
    argv = seed_argv(tmp_path)
    repository = FakeRepository([], {})
    assert seed_main(argv, repository=repository) == 0
    target = tmp_path / "corpus" / "0828-trip.json"
    before = target.read_text(encoding="utf-8")
    assert seed_main(argv, repository=repository) == 1
    assert target.read_text(encoding="utf-8") == before
    assert seed_main([*argv, "--force"], repository=repository) == 0

def test_seed_records_the_split_and_whose_instruction_it_was(tmp_path: Path) -> None:
    argv = seed_argv(tmp_path, "--split", "holdout", "--instruction-by", "summarised")
    assert seed_main(argv, repository=FakeRepository([], {})) == 0
    written = json.loads((tmp_path / "corpus" / "0828-trip.json").read_text(encoding="utf-8"))
    assert written["split"] == "holdout" and written["instruction_by"] == "summarised"

def test_seed_refuses_a_file_that_is_not_under_the_root(tmp_path: Path) -> None:
    argv = seed_argv(tmp_path)
    argv[-1] = "raw/missing.mp4"
    assert seed_main(argv, repository=FakeRepository([], {})) == 1
    assert not (tmp_path / "corpus" / "0828-trip.json").exists()
