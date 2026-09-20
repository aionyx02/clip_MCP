"""Deriving the semantic timeline's `utterance` level from an asset's analysis.

Everything here is a pure function of what `analyze_asset` already stored:
sentence boundaries from the transcript, shot changes from scene detection,
and measured silences. No model is involved and nothing is sampled, so the
same analysis always yields the same clips, and every rule below can be
checked by a unit test rather than by watching a video.

The coarser `section` and `topic` levels are not built here. Those need a
model to decide which candidate boundaries are real ones, and thresholds
that cannot be picked honestly without a corpus to pick them against.
"""

import hashlib
import json
from typing import List, Mapping, Optional, Sequence, Tuple

from app.models.media import Asset, MediaAnalysis, Span, TranscriptSegment
from app.models.semantic import ClipKind, ClipLevel, SemanticClip, SemanticTimeline

# Bumped whenever the rules below change, so timelines built by an older
# version are recognised as out of date rather than silently reused.
DERIVATION_VERSION = 1
# A stretch without speech shorter than this is not a thing on its own; it is the
# breath between two sentences, and it is already described by their headroom.
MIN_UTTERANCE_SECONDS = 0.4
# Silence edges and transcript boundaries are both rounded to milliseconds, so a
# boundary can land a hair outside the silence it was snapped to.
EDGE_TOLERANCE_SECONDS = 0.02
MOSTLY = 0.5
NEARLY_ALL = 0.9

def _fraction(start: float, end: float, spans: Sequence[Span]) -> float:
    """Measure how much of a stretch is covered by a set of spans.

    Args:
        start: Start of the stretch in seconds.
        end: End of the stretch in seconds.
        spans: Spans to measure against; they need not be sorted.

    Returns:
        The covered share, from 0.0 to 1.0.
    """
    length = end - start
    if length <= 0:
        return 0.0
    covered = sum(max(0.0, min(end, span.end) - max(start, span.start)) for span in spans)
    return min(1.0, covered / length)

def _headroom(start: float, end: float, silences: Sequence[Span], duration: float) -> Tuple[float, float]:
    """Measure how far a clip's in and out points may move before reaching sound.

    A boundary that sits inside a measured silence can move to that silence's
    far edge and no further, because past it someone is speaking. A boundary
    that does not sit in a silence has no room at all: sound runs straight up
    to it.

    Args:
        start: In point in seconds.
        end: Out point in seconds.
        silences: Measured silences in the asset.
        duration: Length of the asset in seconds.

    Returns:
        `(safe_in, safe_out)` in seconds, both at least 0 and both inside the
        asset.
    """
    safe_in = safe_out = 0.0
    for span in silences:
        length = span.end - span.start
        if span.start - EDGE_TOLERANCE_SECONDS <= start <= span.end + EDGE_TOLERANCE_SECONDS:
            safe_in = max(safe_in, min(start - span.start, length))
        if span.start - EDGE_TOLERANCE_SECONDS <= end <= span.end + EDGE_TOLERANCE_SECONDS:
            safe_out = max(safe_out, min(span.end - end, length))
    return max(0.0, min(safe_in, start)), max(0.0, min(safe_out, duration - end))

def _split_at_shots(start: float, end: float, scenes: Sequence[Span]) -> List[Tuple[float, float]]:
    """Cut a stretch without speech at the shot changes inside it.

    Two different shots are two different things even when neither has anyone
    talking over it. A cut is only taken when both sides of it are long enough
    to stand on their own, so a shot change a fifth of a second before someone
    speaks does not produce a fragment.

    Args:
        start: Start of the stretch in seconds.
        end: End of the stretch in seconds.
        scenes: Shots detected in the asset; each one's start is a cut.

    Returns:
        `(start, end)` pairs covering the stretch in order.
    """
    points = [start]
    for cut in sorted(scene.start for scene in scenes):
        if cut - points[-1] >= MIN_UTTERANCE_SECONDS and end - cut >= MIN_UTTERANCE_SECONDS:
            points.append(cut)
    points.append(end)
    return list(zip(points, points[1:]))

def _stretches(analysis: MediaAnalysis) -> List[Tuple[float, float, Optional[TranscriptSegment]]]:
    """Lay an asset out as speech and the stretches between it.

    Each transcribed sentence is one stretch. What is left over is cut at shot
    changes, and anything still shorter than `MIN_UTTERANCE_SECONDS` is
    dropped: a gap that short is the pause between two sentences, and both of
    them already carry it as headroom.

    Args:
        analysis: Analysis to lay out.

    Returns:
        `(start, end, segment)` in time order, where `segment` is the
        transcribed sentence for a speech stretch and `None` otherwise.
    """
    scenes = analysis.scenes
    duration = analysis.duration
    segments = analysis.transcript.segments if analysis.transcript is not None else []

    stretches: List[Tuple[float, float, Optional[TranscriptSegment]]] = []
    cursor = 0.0
    for segment in segments:
        start, end = max(cursor, segment.start), min(segment.end, duration)
        if end <= start:
            continue
        stretches.extend(
            (gap_start, gap_end, None)
            for gap_start, gap_end in _split_at_shots(cursor, start, scenes)
            if gap_end - gap_start >= MIN_UTTERANCE_SECONDS
        )
        stretches.append((start, end, segment))
        cursor = end
    stretches.extend(
        (gap_start, gap_end, None)
        for gap_start, gap_end in _split_at_shots(cursor, duration, scenes)
        if gap_end - gap_start >= MIN_UTTERANCE_SECONDS
    )
    return stretches

def _scores(
    start: float,
    end: float,
    analysis: MediaAnalysis,
    segment: Optional[TranscriptSegment],
) -> dict:
    """Measure a stretch against everything the analysis detected.

    These are measurements, not judgements: each one is a share of the stretch
    that some detector marked, so `query_clips` can filter on them and a
    benchmark can count them.

    Args:
        start: Start of the stretch in seconds.
        end: End of the stretch in seconds.
        analysis: Analysis the stretch comes from.
        segment: Transcribed sentence, for a speech stretch.

    Returns:
        Scores keyed by name, each rounded to three decimals. `speech` is the
        share carrying transcribed words, `silence`, `black` and `frozen` the
        shares each detector marked, and `words_per_second` the pace — whose
        unit is whatever the transcript counts as a word, so a Chinese
        transcript counts characters and the number compares within a
        language rather than across them.
    """
    words = segment.words if segment is not None else []
    scores = {
        "speech": _fraction(start, end, [Span(start=word.start, end=word.end) for word in words]),
        "silence": _fraction(start, end, analysis.silences),
        "black": _fraction(start, end, analysis.black_frames),
        "frozen": _fraction(start, end, analysis.frozen_frames),
    }
    if words and end > start:
        scores["words_per_second"] = len(words) / (end - start)
    return {name: round(value, 3) for name, value in scores.items()}

def _kind(has_speech: bool, has_audio: bool, scores: Mapping[str, float]) -> ClipKind:
    """Decide what a stretch is from what was measured in it.

    Args:
        has_speech: Whether the stretch carries transcribed words.
        has_audio: Whether the asset has an audio stream at all.
        scores: The stretch's measurements.

    Returns:
        The kind. A stretch with nobody talking over a picture that is black
        or frozen is `UNUSABLE`; one that is silent throughout is `SILENCE`;
        anything else without speech is `AMBIENT`, including shots from a file
        with no sound, because telling those apart from `ACTION` needs a
        motion measurement the analysis does not take.
    """
    if has_speech:
        return ClipKind.SPEECH
    if scores["black"] >= MOSTLY or scores["frozen"] >= MOSTLY:
        return ClipKind.UNUSABLE
    if has_audio and scores["silence"] >= NEARLY_ALL:
        return ClipKind.SILENCE
    return ClipKind.AMBIENT

def timeline_input_hash(analyses: Mapping[str, MediaAnalysis]) -> str:
    """Fingerprint everything a build depends on.

    The fingerprint covers each analysis's content and the version of the
    rules applied to it, but not when the analysis was taken: analyzing a file
    again with the same model gives the same footage back, and rebuilding over
    it would only invalidate plans for nothing.

    Args:
        analyses: Analyses the build would use, keyed by asset ID.

    Returns:
        The fingerprint as a hex digest.
    """
    payload = {
        "derivation_version": DERIVATION_VERSION,
        "analyses": {
            asset_id: hashlib.sha256(
                analyses[asset_id].model_dump_json(exclude={"analyzed_at"}).encode("utf-8")
            ).hexdigest()
            for asset_id in sorted(analyses)
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

def build_timeline(
    assets: Mapping[str, Asset],
    analyses: Mapping[str, MediaAnalysis],
) -> Tuple[SemanticTimeline, List[SemanticClip]]:
    """Derive the `utterance` level over a set of analyzed assets.

    The timeline's ID comes from the fingerprint of its input, so the same
    footage analyzed the same way always produces the same timeline and the
    same clip IDs — on this machine and on any other. That is what lets a plan
    refer to a clip by ID and mean the same moment tomorrow, and what lets a
    benchmark hold the timeline fixed while something else is being measured.

    Args:
        assets: Assets to cover, keyed by asset ID.
        analyses: Their analyses, keyed by asset ID. Every asset needs one.

    Returns:
        The timeline and its clips, the clips ordered by asset and then by
        time.

    Raises:
        KeyError: If an asset has no analysis.
    """
    input_hash = timeline_input_hash(analyses)
    timeline = SemanticTimeline(
        id=f"tl_{input_hash[:8]}",
        asset_ids=sorted(assets),
        input_hash=input_hash,
        derivation_version=DERIVATION_VERSION,
        levels=[ClipLevel.UTTERANCE],
    )

    clips: List[SemanticClip] = []
    for asset_id in timeline.asset_ids:
        asset, analysis = assets[asset_id], analyses[asset_id]
        for start, end, segment in _stretches(analysis):
            scores = _scores(start, end, analysis, segment)
            text = segment.text.strip() if segment is not None else ""
            safe_in, safe_out = _headroom(start, end, analysis.silences, analysis.duration)
            clips.append(SemanticClip(
                id=f"{timeline.id}:u{len(clips):04d}",
                timeline_id=timeline.id,
                asset_id=asset_id,
                level=ClipLevel.UTTERANCE,
                source_range=Span(start=round(start, 3), end=round(end, 3)),
                safe_in=round(safe_in, 3),
                safe_out=round(safe_out, 3),
                kind=_kind(bool(text), asset.has_audio, scores),
                text=text,
                scores=scores,
            ))
    return timeline, clips
