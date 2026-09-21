"""Where a section might begin, and whether a set of sections is well formed.

Sections cannot be found by rule alone: knowing that someone has finished one
subject and started another is exactly the judgement a model is for. What can
be done by rule is narrowing the field. This module over-produces candidate
boundaries from measurements — how long the pause is, how much the vocabulary
turns over, whether the shots change, whether the sentence opens with "那我們
接下來" — and a model then picks the real ones from that list and names them.

The limit is the point. A model that can only choose among candidates has a
bounded output space, so validating its answer is a membership test rather
than an interpretation, and its worst mistake is a badly placed section
rather than a cut through the middle of a word: every candidate sits on a
boundary that was measured from the audio in the first place.

The weights and thresholds below are provisional. They are recorded in the
timeline's `candidate_hash`, so tuning them against a corpus later produces a
new, clearly distinct build rather than quietly moving boundaries under
sections that were already agreed.
"""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine.semantic import combined_from_clips
from app.models.media import MediaAnalysis, Span
from app.models.semantic import ClipKind, ClipLevel, SectionChoice, SemanticClip

# Bumped whenever the signals or their weights change.
CANDIDATE_VERSION = 1
# How much each measurement contributes. The pause is the strongest single signal
# and vocabulary turnover the only one that knows anything about subject matter.
SIGNAL_WEIGHTS = {"pause": 0.45, "vocabulary": 0.30, "marker": 0.15, "shots": 0.10}
# A pause this long is as strong a signal as a pause gets.
PAUSE_SATURATION_SECONDS = 2.0
# How many clips on each side of a boundary the vocabulary comparison reads.
VOCABULARY_WINDOW = 3
# Shot changes this near the boundary count towards it, and this many saturate.
SHOT_WINDOW_SECONDS = 5.0
SHOT_SATURATION = 3
EDGE_TOLERANCE_SECONDS = 0.02
# Over-produce, but not without limit: an hour of talk is a few dozen sections, so a
# few candidates a minute leaves the model plenty to choose from and still fits a prompt.
CANDIDATES_PER_MINUTE = 3.0
MIN_CANDIDATES = 12
# Phrases people use when they are about to change the subject. Precise but rare,
# so this earns a boundary a good score without being needed for one.
DISCOURSE_MARKERS = (
    "那我們", "那我", "那接下來", "接下來", "好，那", "好那", "另外", "總之", "所以說",
    "再來", "然後呢", "首先", "第一", "第二", "最後", "回到", "講到", "說到", "對了",
    "so ", "okay", "ok,", "alright", "right,", "next,", "next ", "another", "finally",
    "first", "second", "lastly", "moving on", "let's", "now,", "anyway",
)
_LATIN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

@dataclass(frozen=True)
class Candidate:
    """One place a section could begin.

    Attributes:
        index: Position in the asset's clips; the boundary sits before the
            clip at this index.
        asset_id: Asset the boundary is in.
        at: Time of the boundary in the source file, in seconds.
        after_clip_id: The clip that would open the next section.
        signals: Each measurement, from 0.0 to 1.0.
        score: The weighted total, from 0.0 to 1.0.
    """

    index: int
    asset_id: str
    at: float
    after_clip_id: str
    signals: Dict[str, float]
    score: float

def _tokens(text: str) -> Counter:
    """Break text into comparable pieces, in any language.

    Chinese is not written with spaces, so character bigrams stand in for
    words — enough to notice that the vocabulary has turned over, and it needs
    no segmentation model to compute.

    Args:
        text: Text to break up.

    Returns:
        How often each piece appears.
    """
    lowered = text.lower()
    pieces = _LATIN.findall(lowered)
    characters = [character for character in lowered if _CJK.match(character)]
    pieces.extend(first + second for first, second in zip(characters, characters[1:]))
    return Counter(pieces)

def _cosine(left: Counter, right: Counter) -> float:
    """Measure how much two bags of pieces have in common.

    Args:
        left: Pieces on one side.
        right: Pieces on the other.

    Returns:
        Their cosine similarity, from 0.0 to 1.0.
    """
    shared = sum(left[piece] * right[piece] for piece in left.keys() & right.keys())
    if not shared:
        return 0.0
    return shared / math.sqrt(sum(count * count for count in left.values()) * sum(count * count for count in right.values()))

def _vocabulary_overlap(texts: Sequence[str], window: int = VOCABULARY_WINDOW) -> List[Optional[float]]:
    """Compare the vocabulary on either side of every boundary.

    Args:
        texts: One text per clip, empty where nothing is said.
        window: How many clips on each side to read.

    Returns:
        One value per boundary, where boundary `i` sits before clip `i`, or
        `None` where one side has no words to compare and the measurement says
        nothing. The first entry, before the first clip, is always `None`.
    """
    overlaps: List[Optional[float]] = [None]
    for index in range(1, len(texts)):
        left = sum((_tokens(text) for text in texts[max(0, index - window):index]), Counter())
        right = sum((_tokens(text) for text in texts[index:index + window]), Counter())
        overlaps.append(_cosine(left, right) if left and right else None)
    return overlaps

def _turnover(overlaps: Sequence[Optional[float]], index: int) -> float:
    """Measure how deep a dip in vocabulary overlap is.

    A boundary matters not because the overlap is low but because it is lower
    than what surrounds it: a whole passage of unfamiliar words is one subject,
    while a dip between two plateaus is the moment the subject changed. This is
    the depth score text tiling has used for the purpose since long before
    language models.

    Args:
        overlaps: Overlap at each boundary, with `None` where unmeasured.
        index: Boundary to score.

    Returns:
        The depth, from 0.0 to 1.0.
    """
    here = overlaps[index]
    if here is None:
        return 0.0
    peaks = []
    for step in (-1, 1):
        peak, position = here, index
        while 0 <= position + step < len(overlaps) and overlaps[position + step] is not None:
            position += step
            if overlaps[position] < peak:
                break
            peak = overlaps[position]
        peaks.append(peak)
    return min(1.0, (peaks[0] - here) + (peaks[1] - here))

def _pause_at(seconds: float, silences: Sequence[Span]) -> float:
    """Measure the silence a boundary falls in.

    Args:
        seconds: Time of the boundary.
        silences: Silences measured in the asset.

    Returns:
        The length of the silence around it, or 0.0 if sound runs through it.
    """
    for span in silences:
        if span.start - EDGE_TOLERANCE_SECONDS <= seconds <= span.end + EDGE_TOLERANCE_SECONDS:
            return span.end - span.start
    return 0.0

def _shot_density(seconds: float, scenes: Sequence[Span]) -> float:
    """Measure how busy the cutting is around a boundary.

    A run of shot changes usually means the place or the setup changed, which
    is a likely place for a section to start.

    Args:
        seconds: Time of the boundary.
        scenes: Shots detected in the asset.

    Returns:
        The density, from 0.0 to 1.0.
    """
    nearby = sum(1 for scene in scenes if abs(scene.start - seconds) <= SHOT_WINDOW_SECONDS)
    return min(1.0, nearby / SHOT_SATURATION)

def _opens_a_new_subject(text: str) -> float:
    """Check whether a sentence opens with a phrase that changes the subject.

    Args:
        text: The sentence.

    Returns:
        1.0 if it does, 0.0 otherwise.
    """
    stripped = text.strip().lower()
    return 1.0 if any(stripped.startswith(marker) for marker in DISCOURSE_MARKERS) else 0.0

def propose_candidates(
    clips: Sequence[SemanticClip],
    analyses: Mapping[str, MediaAnalysis],
) -> List[Candidate]:
    """Find every place a section could plausibly begin.

    Boundaries are proposed within each asset rather than between them: a file
    always starts a new section, and a subject that runs across two files is a
    topic, which is a label rather than a span.

    Args:
        clips: The timeline's `utterance` clips.
        analyses: The analyses they were derived from, keyed by asset ID.

    Returns:
        The candidates, in time order, the weakest dropped so the list stays
        readable. Every one of them sits on a boundary between two clips, and
        those were measured from the audio.
    """
    candidates: List[Candidate] = []
    for asset_id in sorted({clip.asset_id for clip in clips}):
        of_asset = [clip for clip in clips if clip.asset_id == asset_id]
        analysis = analyses[asset_id]
        overlaps = _vocabulary_overlap([clip.text for clip in of_asset])
        for index in range(1, len(of_asset)):
            at = of_asset[index].source_range.start
            signals = {
                "pause": min(1.0, _pause_at(at, analysis.silences) / PAUSE_SATURATION_SECONDS),
                "vocabulary": _turnover(overlaps, index),
                "marker": _opens_a_new_subject(of_asset[index].text),
                "shots": _shot_density(at, analysis.scenes),
            }
            score = sum(SIGNAL_WEIGHTS[name] * value for name, value in signals.items())
            if score <= 0:
                continue
            candidates.append(Candidate(
                index=index,
                asset_id=asset_id,
                at=at,
                after_clip_id=of_asset[index].id,
                signals={name: round(value, 3) for name, value in signals.items()},
                score=round(score, 3),
            ))

    duration = sum(analysis.duration for analysis in analyses.values())
    keep = max(MIN_CANDIDATES, round(duration / 60 * CANDIDATES_PER_MINUTE))
    if len(candidates) > keep:
        strongest = {id(candidate) for candidate in sorted(candidates, key=lambda item: -item.score)[:keep]}
        candidates = [candidate for candidate in candidates if id(candidate) in strongest]
    return candidates

def candidate_hash(candidates: Sequence[Candidate]) -> str:
    """Fingerprint the candidates a set of sections was chosen from.

    Args:
        candidates: The candidates offered.

    Returns:
        The fingerprint as a hex digest.
    """
    payload = {
        "candidate_version": CANDIDATE_VERSION,
        "weights": SIGNAL_WEIGHTS,
        "boundaries": [[candidate.asset_id, candidate.after_clip_id] for candidate in candidates],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

def check_sections(
    chosen: Sequence[Tuple[str, str]],
    clips: Sequence[SemanticClip],
    candidates: Sequence[Candidate],
) -> List[str]:
    """Check that a set of sections covers the footage and starts where it may.

    Nothing is corrected here. A set of sections that does not hold up is
    handed back with the reasons, because quietly repairing it would move the
    judgement out of the model and into the server without anyone deciding to.

    Args:
        chosen: One `(first_clip_id, last_clip_id)` per section, in the order
            they were given.
        clips: The timeline's `utterance` clips.
        candidates: The candidates that were offered.

    Returns:
        One message per problem, empty when the sections are sound.
    """
    by_id = {clip.id: clip for clip in clips}
    order: Dict[str, List[str]] = {}
    for clip in clips:
        order.setdefault(clip.asset_id, []).append(clip.id)
    allowed = {(candidate.asset_id, candidate.after_clip_id) for candidate in candidates}

    problems: List[str] = []
    consumed: Dict[str, int] = {asset_id: 0 for asset_id in order}
    for position, (first, last) in enumerate(chosen):
        where = f"section {position + 1}"
        if first not in by_id or last not in by_id:
            problems.append(f"{where}: {first if first not in by_id else last} is not a clip in this timeline")
            continue
        asset_id = by_id[first].asset_id
        if by_id[last].asset_id != asset_id:
            problems.append(f"{where}: starts in one file and ends in another; a section stays inside one file")
            continue
        within = order[asset_id]
        start, stop = within.index(first), within.index(last)
        if stop < start:
            problems.append(f"{where}: ends before it starts")
            continue
        if start != consumed[asset_id]:
            if consumed[asset_id] >= len(within):
                problems.append(f"{where}: every clip of that file is already in an earlier section")
            else:
                problems.append(
                    f"{where}: starts at {first}, but the previous section of that file left off at "
                    f"{within[consumed[asset_id]]}; sections follow one another with nothing in between"
                )
            continue
        if start > 0 and (asset_id, first) not in allowed:
            problems.append(f"{where}: {first} is not one of the candidate boundaries, so it cannot start a section")
            continue
        consumed[asset_id] = stop + 1

    for asset_id, position in consumed.items():
        if position < len(order[asset_id]):
            problems.append(
                f"{len(order[asset_id]) - position} clip(s) of asset {asset_id} are in no section; "
                "every clip belongs to one"
            )
    return problems

def build_sections(
    timeline_id: str,
    clips: Sequence[SemanticClip],
    chosen: Sequence[SectionChoice],
) -> Tuple[List[SemanticClip], List[SemanticClip]]:
    """Assemble the section clips and attach each utterance to the one holding it.

    Call `check_sections` first; this assumes the choices already hold up.

    Args:
        timeline_id: Timeline the sections belong to.
        clips: The timeline's `utterance` clips.
        chosen: The sections, in order.

    Returns:
        `(sections, utterances)` — the new section clips, and the utterance
        clips with their `parent_id` set, ready to be stored together.
    """
    by_id = {clip.id: clip for clip in clips}
    order: Dict[str, List[str]] = {}
    for clip in clips:
        order.setdefault(clip.asset_id, []).append(clip.id)

    sections: List[SemanticClip] = []
    attached: Dict[str, SemanticClip] = {clip.id: clip for clip in clips}
    for position, choice in enumerate(chosen):
        asset_id = by_id[choice.first_clip_id].asset_id
        within = order[asset_id]
        members = [
            by_id[clip_id]
            for clip_id in within[within.index(choice.first_clip_id):within.index(choice.last_clip_id) + 1]
        ]
        section_id = f"{timeline_id}:s{position:04d}"
        length = sum(clip.duration for clip in members) or 1.0
        sections.append(SemanticClip(
            id=section_id,
            timeline_id=timeline_id,
            asset_id=asset_id,
            level=ClipLevel.SECTION,
            source_range=Span(start=members[0].source_range.start, end=members[-1].source_range.end),
            safe_in=members[0].safe_in,
            safe_out=members[-1].safe_out,
            kind=section_kind(members),
            name=choice.name,
            topic=choice.topic,
            text=choice.summary,
            scores=combined_from_clips(members, length),
        ))
        for clip in members:
            attached[clip.id] = clip.model_copy(update={"parent_id": section_id})
    return sections, [attached[clip.id] for clip in clips]

def section_kind(members: Sequence[SemanticClip]) -> ClipKind:
    """Decide what a section is mostly made of.

    Args:
        members: The clips it holds.

    Returns:
        `SPEECH` if anyone talks in it, otherwise whichever kind covers the
        most of its length.
    """
    if any(clip.kind == ClipKind.SPEECH for clip in members):
        return ClipKind.SPEECH
    covered: Dict[ClipKind, float] = {}
    for clip in members:
        covered[clip.kind] = covered.get(clip.kind, 0.0) + clip.duration
    return max(covered, key=covered.get) if covered else ClipKind.AMBIENT
