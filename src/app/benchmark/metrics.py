"""Measuring a compiled cut against what a case said it must not get wrong.

Everything here is a pure function of the windows the compiler produced and the
detections already on disk. No model, no sampling, no rendering: the same plan
over the same footage always scores the same, which is what makes a drop in the
score evidence rather than an impression.

The floors below come from the roadmap's quality table. They are not opinions
about taste — a cut through the middle of a word is wrong whoever made it — so
they are counted here rather than written into a guide and hoped for. What
cannot be settled this way, such as whether the cut is any good, is left to L3
and is not pretended to be measured.

The trap this module is built around is that **nothing measured looks exactly
like nothing found**. Footage nobody analyzed has no words to cut through and
no bad frames to land on, so a cut straight through it scores a flawless zero
on both. Every such gap is named in `Scorecard.unmeasured`, and a card with
anything in it is not `clean` — an untested floor has not been cleared.
"""

from dataclasses import dataclass
from statistics import median
from typing import List, Mapping, Optional, Sequence

from app.engine.plan import Piece, compiled_duration
from app.engine.semantic import EDGE_TOLERANCE_SECONDS, share_covered
from app.models.media import MediaAnalysis, Span
from app.benchmark.case import BenchmarkCase, MarkedSpan

# The floors a cut has to clear, from the roadmap's quality table. A cut that misses
# one of these is wrong in a way nobody has to be asked about.
#
# The length floor is deliberately its own number rather than `plan.LENGTH_TOLERANCE`,
# which happens to be the same today: that one decides when the planner mentions the
# drift to whoever is editing, this one decides whether a run passed. The corpus will
# move them apart, and tuning advice must not quietly move a pass mark.
MAX_LENGTH_ERROR = 0.10
MIN_MUST_KEEP_COVERAGE = 0.90
# The roadmap's target for footage that had to go is zero, so any of it is a failure.
MAX_MUST_DROP_LEAKAGE = 0.0

@dataclass(frozen=True)
class CutPoint:
    """One point in a source file where the cut opens or closes a window.

    Which end it is matters. The opening point is the first frame the viewer
    sees; the closing point is the first frame they do not.

    Attributes:
        asset_id: File the cut lands in.
        seconds: Where it lands, in that file.
        opens: Whether it opens the window rather than closing it.
    """

    asset_id: str
    seconds: float
    opens: bool

def _cuts(pieces: Sequence[Piece]) -> List[CutPoint]:
    """List every point in the source footage where the cut opens or closes a window.

    Args:
        pieces: The compiled windows.

    Returns:
        Two cut points per window, in compile order.
    """
    return [
        CutPoint(piece.asset_id, edge, opens)
        for piece in pieces
        for edge, opens in ((piece.start, True), (piece.end, False))
    ]

def _windows(pieces: Sequence[Piece], asset_id: str) -> List[Span]:
    """Collect the windows one asset contributes to the cut.

    Args:
        pieces: The compiled windows.
        asset_id: Asset to select.

    Returns:
        Its windows as spans, in the order they were compiled.
    """
    return [Span(start=piece.start, end=piece.end) for piece in pieces if piece.asset_id == asset_id]

def _share_in_cut(
    marked: Sequence[MarkedSpan],
    pieces: Sequence[Piece],
    asset_ids: Mapping[str, str],
) -> Optional[float]:
    """Measure how much of the annotated time ended up in the cut.

    Weighted by seconds rather than by span, so a two-second aside cannot
    outvote a minute that matters.

    Args:
        marked: The annotated stretches.
        pieces: The compiled windows.
        asset_ids: Asset ID for each of the case's file paths.

    Returns:
        The share from 0.0 to 1.0, or None when nothing was annotated — which
        is not the same answer as nothing being covered.
    """
    total = sum(span.duration for span in marked)
    if total <= 0:
        return None
    covered = sum(
        share_covered(span.start, span.end, _windows(pieces, asset_ids[span.file])) * span.duration
        for span in marked
    )
    return covered / total

def _has_word_timings(analysis: MediaAnalysis) -> bool:
    """Decide whether this analysis can answer whether a cut clips a word.

    A file that was analyzed without transcription has no transcript at all,
    and one transcribed by something that did not time its words has segments
    with no words in them. Either way there is nothing to compare a cut point
    against, and the answer that comes back is zero faults.

    Args:
        analysis: The stored analysis.

    Returns:
        True when the transcript can be asked where the words are. Footage that
        was transcribed and turned out to have nobody talking counts: an empty
        transcript is an answer, a missing one is not.
    """
    transcript = analysis.transcript
    if transcript is None:
        return False
    return not transcript.segments or any(segment.words for segment in transcript.segments)

def _inside_speech(seconds: float, analysis: MediaAnalysis) -> bool:
    """Decide whether a cut point falls in the middle of a spoken word.

    Strictly inside, and not within a rounding of either edge: a cut that lands
    on a word's boundary keeps or drops the whole word, which is clean. Times
    on both sides are rounded to milliseconds, so the same tolerance the
    semantic layer uses for snapped boundaries applies here.

    Args:
        seconds: The cut point in the source file.
        analysis: That file's analysis.

    Returns:
        True if a word is audibly cut through.
    """
    transcript = analysis.transcript
    words = [word for segment in (transcript.segments if transcript else []) for word in segment.words]
    return any(
        word.start + EDGE_TOLERANCE_SECONDS < seconds < word.end - EDGE_TOLERANCE_SECONDS
        for word in words
    )

def _on_bad_frame(cut: CutPoint, analysis: MediaAnalysis) -> bool:
    """Decide whether a cut point lands on a frame nobody wants to see.

    Half-open, unlike the word test, and which half is open depends on the end:
    a window's opening point is the first frame shown, so bad picture starting
    exactly there is seen; its closing point is the first frame *not* shown, so
    bad picture starting exactly there is not, while bad picture ending exactly
    there was on screen until the last moment.

    Args:
        cut: The cut point.
        analysis: That file's analysis.

    Returns:
        True if the picture the viewer actually sees there is black or frozen.
    """
    spans = [*analysis.black_frames, *analysis.frozen_frames]
    if cut.opens:
        return any(span.start <= cut.seconds < span.end for span in spans)
    return any(span.start < cut.seconds <= span.end for span in spans)

def _margin(seconds: float, analysis: MediaAnalysis) -> float:
    """Measure how much silence a cut point had around it before reaching sound.

    A cut sitting inside a measured silence can move either way before it
    reaches sound, and the nearer of those two distances is the room it really
    had. A cut that is not inside a silence had none. Zero therefore means the
    cut is in sound; whether that sound was a word is `words_cut`'s question,
    not this one.

    Unlike `_inside_speech`, this needs no rounding tolerance, and `_headroom`
    in the semantic layer agrees with it without one. The tolerance there
    decides whether a boundary counts as snapped to a silence; the quantity
    here is the distance to the very edge such a tolerance would relax, so a
    cut a millisecond past the edge has a millisecond less than no room, which
    is no room either way.

    Args:
        seconds: The cut point in the source file.
        analysis: That file's analysis.

    Returns:
        Seconds to the nearer edge of the silence around it, or 0.0.
    """
    room = [
        min(seconds - span.start, span.end - seconds)
        for span in analysis.silences
        if span.start <= seconds <= span.end
    ]
    return round(max(room), 3) if room else 0.0

def _outside_allowed(cuts: Sequence[CutPoint], case: BenchmarkCase, asset_ids: Mapping[str, str]) -> Optional[int]:
    """Count cut points that land where the case says a cut may not.

    Only files the case drew windows for are judged. A file with no windows
    annotated is one nobody has ruled on, and treating that as "every cut is
    wrong" would punish a plan for a gap in the corpus.

    Args:
        cuts: The cut points.
        case: The case, with its acceptable cut windows.
        asset_ids: Asset ID for each of the case's file paths.

    Returns:
        How many cut points fall outside every allowed window of a file that
        has them, or None when the case allows no windows anywhere.
    """
    if not case.cut_windows:
        return None
    allowed: dict[str, List[MarkedSpan]] = {}
    for window in case.cut_windows:
        allowed.setdefault(asset_ids[window.file], []).append(window)
    return sum(
        1 for cut in cuts
        if cut.asset_id in allowed
        and not any(
            window.start - EDGE_TOLERANCE_SECONDS <= cut.seconds <= window.end + EDGE_TOLERANCE_SECONDS
            for window in allowed[cut.asset_id]
        )
    )

def _unmeasured(pieces: Sequence[Piece], analyses: Mapping[str, MediaAnalysis]) -> List[str]:
    """Say which floors could not be tested, and on what footage.

    Called by `score` rather than offered to callers, because a gap nobody
    remembered to check for reads exactly like a clean pass.

    Args:
        pieces: The compiled windows.
        analyses: Each asset's stored analysis, keyed by asset ID.

    Returns:
        One sentence per piece of footage that could not be fully measured, in
        the order the cut uses it.
    """
    said: List[str] = []
    seen: set[str] = set()
    for piece in pieces:
        if piece.asset_id in seen:
            continue
        seen.add(piece.asset_id)
        analysis = analyses.get(piece.asset_id)
        if analysis is None:
            said.append(f"{piece.asset_id} has no analysis, so nothing in it was measured")
        elif not _has_word_timings(analysis):
            said.append(f"{piece.asset_id} was not transcribed with word timings, so cuts through words were not counted")
    return said

@dataclass(frozen=True)
class Scorecard:
    """What one plan scored on one case.

    Attributes:
        case_id: The case this was scored against.
        duration: How long the compiled cut runs, in seconds.
        length_error: How far that is from the case's target, as a share of the
            target. None when the case names no length.
        must_keep_coverage: Share of the must-keep time the cut contains. None
            when the case marks none.
        must_drop_leakage: Share of the must-drop time that got in. None when
            the case marks none.
        words_cut: Cut points landing in the middle of a spoken word.
        bad_frame_cuts: Cut points landing on black or frozen picture.
        cuts_outside_windows: Cut points landing where the case says a cut may
            not. None when the case draws no windows.
        cut_margins: Silence around each cut point, in seconds, in cut order.
        failures: The floors this cut missed, each said in a sentence.
        unmeasured: The floors that could not be tested, and on what footage.
    """

    case_id: str
    duration: float
    length_error: Optional[float]
    must_keep_coverage: Optional[float]
    must_drop_leakage: Optional[float]
    words_cut: int
    bad_frame_cuts: int
    cuts_outside_windows: Optional[int]
    cut_margins: tuple[float, ...]
    failures: tuple[str, ...]
    unmeasured: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """Whether the cut cleared every floor.

        A floor nobody could test has not been cleared, so footage listed in
        `unmeasured` keeps a card from reading as clean.

        Returns:
            True when nothing failed and nothing went unmeasured.
        """
        return not self.failures and not self.unmeasured

    @property
    def tightest_margin(self) -> Optional[float]:
        """The least silence any cut point had around it.

        Returns:
            The smallest margin in seconds, or None when the cut has no points.
        """
        return min(self.cut_margins) if self.cut_margins else None

    @property
    def median_margin(self) -> Optional[float]:
        """The silence a typical cut point had around it.

        Reported alongside the tightest one because a single tight cut in an
        otherwise roomy edit is a different problem from every cut being tight.

        Returns:
            The median margin in seconds, or None when the cut has no points.
        """
        return round(median(self.cut_margins), 3) if self.cut_margins else None

def _untested(
    case: BenchmarkCase,
    pieces: Sequence[Piece],
    analyses: Mapping[str, MediaAnalysis],
) -> List[str]:
    """Say what could not be tested, footage and annotations alike.

    A draft case belongs here rather than among the failures. Its floors were
    not missed; nobody has set them yet, and a card that reported one as
    cleared would be claiming a measurement that was never taken.

    Args:
        case: The case being scored.
        pieces: The windows the plan compiled to.
        analyses: Each asset's stored analysis, keyed by asset ID.

    Returns:
        One sentence per floor that went untested.
    """
    untested = []
    if case.is_draft:
        untested.append(
            f"case {case.id}: its annotations are an unreviewed draft, so nothing it reports is evidence"
        )
    return untested + _unmeasured(pieces, analyses)

def score(
    case: BenchmarkCase,
    pieces: Sequence[Piece],
    analyses: Mapping[str, MediaAnalysis],
    asset_ids: Mapping[str, str],
) -> Scorecard:
    """Measure a compiled cut against one case.

    Args:
        case: The case, with its target length and annotations.
        pieces: The windows the plan compiled to, already merged.
        analyses: Each asset's stored analysis, keyed by asset ID. Footage with
            no analysis, or transcribed without word timings, is reported in
            the card's `unmeasured` rather than scored as faultless.
        asset_ids: Asset ID for each of the case's file paths.

    Returns:
        The scorecard.

    Raises:
        ValueError: If the case annotates a file that `asset_ids` does not
            resolve. Scoring against footage that is not there would report a
            coverage of zero and look like a bad plan rather than a bad setup.
    """
    annotated = {span.file for span in [*case.must_keep, *case.must_drop, *case.cut_windows]}
    unresolved = sorted(annotated - set(asset_ids))
    if unresolved:
        raise ValueError(
            f"case {case.id}: {', '.join(unresolved)} is annotated but was not imported; "
            "import the case's footage before scoring it"
        )

    duration = compiled_duration(pieces)
    length_error = (
        abs(duration - case.target_seconds) / case.target_seconds
        if case.target_seconds else None
    )
    coverage = _share_in_cut(case.must_keep, pieces, asset_ids)
    leakage = _share_in_cut(case.must_drop, pieces, asset_ids)

    cuts = [cut for cut in _cuts(pieces) if cut.asset_id in analyses]
    words_cut = sum(1 for cut in cuts if _inside_speech(cut.seconds, analyses[cut.asset_id]))
    bad_frames = sum(1 for cut in cuts if _on_bad_frame(cut, analyses[cut.asset_id]))
    outside = _outside_allowed(_cuts(pieces), case, asset_ids)
    margins = tuple(_margin(cut.seconds, analyses[cut.asset_id]) for cut in cuts)

    # Said in sentences rather than returned as flags: a score that cannot say what
    # went wrong sends the reader back to the footage to find out.
    failures: List[str] = []
    if words_cut:
        failures.append(f"{words_cut} cut(s) land in the middle of a word")
    if bad_frames:
        failures.append(f"{bad_frames} cut(s) land on black or frozen picture")
    if outside:
        failures.append(f"{outside} cut(s) land outside the windows the case allows a cut in")
    if length_error is not None and length_error > MAX_LENGTH_ERROR:
        failures.append(
            f"the cut runs {duration}s against a target of {case.target_seconds}s, out by {length_error:.0%}"
        )
    if coverage is not None and coverage < MIN_MUST_KEEP_COVERAGE:
        failures.append(f"only {coverage:.0%} of the footage that had to be kept is in the cut")
    if leakage is not None and leakage > MAX_MUST_DROP_LEAKAGE:
        failures.append(f"{leakage:.2%} of the footage that had to go is in the cut")

    return Scorecard(
        case_id=case.id,
        duration=duration,
        length_error=None if length_error is None else round(length_error, 4),
        must_keep_coverage=None if coverage is None else round(coverage, 4),
        must_drop_leakage=None if leakage is None else round(leakage, 4),
        words_cut=words_cut,
        bad_frame_cuts=bad_frames,
        cuts_outside_windows=outside,
        cut_margins=margins,
        failures=tuple(failures),
        unmeasured=tuple(_untested(case, pieces, analyses)),
    )
