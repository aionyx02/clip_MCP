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
"""

from dataclasses import dataclass
from statistics import median
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from app.engine.plan import Piece, compiled_duration
from app.engine.semantic import EDGE_TOLERANCE_SECONDS
from app.models.media import MediaAnalysis, Span
from app.benchmark.case import BenchmarkCase, MarkedSpan

# The floors a cut has to clear, from the roadmap's quality table. A cut that
# misses one of these is wrong in a way nobody has to be asked about.
MAX_LENGTH_ERROR = 0.10
MIN_MUST_KEEP_COVERAGE = 0.90

def _overlap(start: float, end: float, spans: Iterable[Span]) -> float:
    """Measure how much of a stretch a set of spans covers.

    Args:
        start: Start of the stretch in seconds.
        end: End of the stretch in seconds.
        spans: Spans to measure against; they need not be sorted or disjoint,
            but overlapping ones would be counted twice, so callers pass
            windows that the compiler has already merged.

    Returns:
        The covered seconds, never more than the stretch is long.
    """
    covered = sum(max(0.0, min(end, span.end) - max(start, span.start)) for span in spans)
    return min(covered, max(0.0, end - start))

def _windows(pieces: Sequence[Piece], asset_id: str) -> List[Span]:
    """Collect the windows one asset contributes to the cut.

    Args:
        pieces: The compiled windows.
        asset_id: Asset to select.

    Returns:
        Its windows as spans, in the order they were compiled.
    """
    return [Span(start=piece.start, end=piece.end) for piece in pieces if piece.asset_id == asset_id]

def _share_covered(
    marked: Sequence[MarkedSpan],
    pieces: Sequence[Piece],
    asset_ids: Mapping[str, str],
) -> Optional[float]:
    """Measure how much of the annotated time the cut contains.

    Weighted by seconds rather than by span, so a two-second aside cannot
    outvote a minute that matters.

    Args:
        marked: The annotated stretches.
        pieces: The compiled windows.
        asset_ids: Asset ID for each of the case's file paths.

    Returns:
        The covered share from 0.0 to 1.0, or None when nothing was annotated,
        which is not the same as nothing being covered.
    """
    total = sum(span.duration for span in marked)
    if total <= 0:
        return None
    covered = sum(
        _overlap(span.start, span.end, _windows(pieces, asset_ids[span.file]))
        for span in marked
        if span.file in asset_ids
    )
    return round(covered / total, 4)

def _cuts(pieces: Sequence[Piece]) -> List[Tuple[str, float, bool]]:
    """List every point in the source footage where the cut opens or closes a window.

    Which end it is matters: the opening point is the first frame the viewer
    sees, while the closing point is the first frame they do not.

    Args:
        pieces: The compiled windows.

    Returns:
        `(asset_id, seconds, opens)` per cut point, two per window.
    """
    return [
        (piece.asset_id, edge, opens)
        for piece in pieces
        for edge, opens in ((piece.start, True), (piece.end, False))
    ]

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

def _on_bad_frame(seconds: float, opens: bool, analysis: MediaAnalysis) -> bool:
    """Decide whether a cut point lands on a frame nobody wants to see.

    Half-open, unlike the word test, and which half is open depends on the end:
    a window's opening point is the first frame shown, so bad picture starting
    exactly there is seen; its closing point is the first frame *not* shown, so
    bad picture starting exactly there is not, while bad picture ending exactly
    there was on screen until the last moment.

    Args:
        seconds: The cut point in the source file.
        opens: Whether this point opens the window rather than closing it.
        analysis: That file's analysis.

    Returns:
        True if the picture the viewer actually sees there is black or frozen.
    """
    spans = [*analysis.black_frames, *analysis.frozen_frames]
    if opens:
        return any(span.start <= seconds < span.end for span in spans)
    return any(span.start < seconds <= span.end for span in spans)

def _margin(seconds: float, analysis: MediaAnalysis) -> float:
    """Measure how much room a cut point had.

    A cut sitting inside a measured silence can move either way before it
    reaches sound; the nearer of those two distances is the room it actually
    had. A cut that is not inside a silence had none, whatever else is true of
    it.

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
        cut_margins: Room each cut point had, in seconds, in cut order.
        failures: The floors this cut missed, each said in a sentence.
    """

    case_id: str
    duration: float
    length_error: Optional[float]
    must_keep_coverage: Optional[float]
    must_drop_leakage: Optional[float]
    words_cut: int
    bad_frame_cuts: int
    cut_margins: Tuple[float, ...]
    failures: Tuple[str, ...]

    @property
    def clean(self) -> bool:
        """Whether the cut cleared every floor.

        Returns:
            True when nothing failed.
        """
        return not self.failures

    @property
    def tightest_margin(self) -> Optional[float]:
        """The least room any cut point had.

        Returns:
            The smallest margin in seconds, or None when the cut has no points.
        """
        return min(self.cut_margins) if self.cut_margins else None

    @property
    def median_margin(self) -> Optional[float]:
        """The room a typical cut point had.

        Reported alongside the tightest one because a single tight cut in an
        otherwise roomy edit is a different problem from every cut being tight.

        Returns:
            The median margin in seconds, or None when the cut has no points.
        """
        return round(median(self.cut_margins), 3) if self.cut_margins else None

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
        analyses: Each asset's stored analysis, keyed by asset ID. An asset
            with no analysis contributes no word or frame findings, so a cut
            through unanalyzed footage scores zero faults rather than a
            confident pass; `missing_analyses` says which.
        asset_ids: Asset ID for each of the case's file paths.

    Returns:
        The scorecard.

    Raises:
        ValueError: If the case annotates a file that `asset_ids` does not
            resolve. Scoring against footage that is not there would report a
            coverage of zero and look like a bad plan rather than a bad setup.
    """
    unresolved = sorted(
        {span.file for span in [*case.must_keep, *case.must_drop]} - set(asset_ids)
    )
    if unresolved:
        raise ValueError(
            f"case {case.id}: {', '.join(unresolved)} is annotated but was not imported; "
            "import the case's footage before scoring it"
        )

    duration = compiled_duration(pieces)
    length_error = (
        round(abs(duration - case.target_seconds) / case.target_seconds, 4)
        if case.target_seconds else None
    )
    coverage = _share_covered(case.must_keep, pieces, asset_ids)
    leakage = _share_covered(case.must_drop, pieces, asset_ids)

    cuts = [(asset_id, at, opens) for asset_id, at, opens in _cuts(pieces) if asset_id in analyses]
    words_cut = sum(1 for asset_id, at, _ in cuts if _inside_speech(at, analyses[asset_id]))
    bad_frames = sum(1 for asset_id, at, opens in cuts if _on_bad_frame(at, opens, analyses[asset_id]))
    margins = tuple(_margin(at, analyses[asset_id]) for asset_id, at, _ in cuts)

    # Said in sentences rather than returned as flags: a score that cannot say what
    # went wrong sends the reader back to the footage to find out.
    failures: List[str] = []
    if words_cut:
        failures.append(f"{words_cut} cut(s) land in the middle of a word")
    if bad_frames:
        failures.append(f"{bad_frames} cut(s) land on black or frozen picture")
    if length_error is not None and length_error > MAX_LENGTH_ERROR:
        failures.append(
            f"the cut runs {duration}s against a target of {case.target_seconds}s, out by {length_error:.0%}"
        )
    if coverage is not None and coverage < MIN_MUST_KEEP_COVERAGE:
        failures.append(f"only {coverage:.0%} of the footage that had to be kept is in the cut")
    if leakage:
        failures.append(f"{leakage:.0%} of the footage that had to go is in the cut")

    return Scorecard(
        case_id=case.id,
        duration=duration,
        length_error=length_error,
        must_keep_coverage=coverage,
        must_drop_leakage=leakage,
        words_cut=words_cut,
        bad_frame_cuts=bad_frames,
        cut_margins=margins,
        failures=tuple(failures),
    )

def missing_analyses(pieces: Sequence[Piece], analyses: Mapping[str, MediaAnalysis]) -> List[str]:
    """Name the footage in a cut that has not been analyzed.

    A score says nothing about words or frames in footage nobody measured, and
    the absence looks exactly like a clean result. Call this beside `score` and
    report what it returns rather than letting a setup mistake read as a pass.

    Args:
        pieces: The compiled windows.
        analyses: Each asset's stored analysis, keyed by asset ID.

    Returns:
        The asset IDs used by the cut that have no analysis, in cut order.
    """
    seen: List[str] = []
    for piece in pieces:
        if piece.asset_id not in analyses and piece.asset_id not in seen:
            seen.append(piece.asset_id)
    return seen
