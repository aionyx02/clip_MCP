"""Deriving, from an asset's analysis, the facts an edit is decided from.

Two of them. The semantic timeline's `utterance` level — a clip per sentence,
per shot, per long pause — and the seconds at which that file may be cut
without splitting a word. They share a module because they are the same kind
of thing: a pure function of what `analyze_asset` already stored, handed on as
numbers so that nothing downstream has to read a transcript and interpret it.

Everything here is a pure function of what `analyze_asset` already stored:
sentence boundaries from the transcript, shot changes from scene detection,
measured silences, what each shot and second measured, and which voice was
talking. No model is run here and nothing is sampled, so the same analysis
always yields the same clips, and every rule below can be checked by a unit
test rather than by watching a video.

The coarser `section` and `topic` levels are not built here. Those need a
model to decide which candidate boundaries are real ones, and thresholds
that cannot be picked honestly without a corpus to pick them against.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from app.models.media import SILENCE_DB, Asset, MediaAnalysis, Measured, Span, SpeakerTurn, TranscriptSegment
from app.models.semantic import ClipKind, ClipLevel, SemanticClip, SemanticTimeline

# Bumped whenever the rules below change, so timelines built by an older
# version are recognised as out of date rather than silently reused.
DERIVATION_VERSION = 6
# A stretch without speech shorter than this is not a thing on its own; it is the
# breath between two sentences, and it is already described by their headroom.
MIN_UTTERANCE_SECONDS = 0.4
# Silence edges and transcript boundaries are both rounded to milliseconds, so a
# boundary can land a hair outside the silence it was snapped to.
EDGE_TOLERANCE_SECONDS = 0.02
MOSTLY = 0.5
NEARLY_ALL = 0.9
# How much of the talking in a stretch one voice has to hold before the stretch is called
# theirs. A sentence that straddles a handover belongs to neither, and says so.
SPEAKER_MAJORITY = 0.8
# How far apart two voices have to sit before they are two people, when the question is
# whether a label in one file is the same person as a label in another. Cosine distance,
# so 0 is the same direction and 1 is unrelated. Kept apart from `diarize.SPEAKER_DISTANCE`,
# which asks a different question — that one splits one recording, this one joins several,
# and a recording made on another day in another room is further from itself than two
# people in one room are from each other. Provisional, like that one.
VOICE_DISTANCE = 0.45

def share_covered(start: float, end: float, spans: Sequence[Span]) -> float:
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

def _weighted_mean(parts: Sequence[Tuple[float, float]]) -> float:
    """Average values by how many seconds each one covers.

    Args:
        parts: `(value, seconds)` pairs.

    Returns:
        The weighted mean, or the plain mean when the parts cover no time at
        all, which nothing should produce but which must not divide by zero.
    """
    weight = sum(seconds for _, seconds in parts)
    if weight <= 0:
        return sum(value for value, _ in parts) / len(parts)
    return sum(value * seconds for value, seconds in parts) / weight

def _decibel_mean(parts: Sequence[Tuple[float, float]]) -> float:
    """Average levels by the energy they stand for, not by their decibel numbers.

    Decibels are logarithmic, so their arithmetic mean is not the level of
    anything. One second of digital silence — `SILENCE_DB`, which is -120 —
    among nine seconds of ordinary speech at -20 dB averages out to -30 dB,
    and a clip that is perfectly well recorded reads as far too quiet.
    Converted to energy first, that same second contributes a millionth of
    nothing and the clip reads as -20 dB, which is what it is.

    Args:
        parts: `(decibels, seconds)` pairs.

    Returns:
        The level of the whole stretch, in decibels.
    """
    energy = _weighted_mean([(10 ** (value / 10), seconds) for value, seconds in parts])
    return SILENCE_DB if energy <= 0 else max(SILENCE_DB, 10 * math.log10(energy))

def _highest(parts: Sequence[Tuple[float, float]]) -> float:
    """Take the largest value there was.

    For a peak or for how squared-off the waveform got, the whole question is
    whether it ever happened. Averaging it away is how a clipped second inside
    a clean clip stops being visible.

    Args:
        parts: `(value, seconds)` pairs.

    Returns:
        The largest value.
    """
    return max(value for value, _ in parts)

def _typical(parts: Sequence[Tuple[float, float]]) -> float:
    """Take the value holding the middle second of a stretch.

    What a noise floor is asked for is the hiss under most of a stretch. Both
    a mean and a minimum would answer with the quietest second instead — and a
    second of digital silence has no hiss at all, so it would report a
    beautifully quiet recording that does not exist.

    The middle is by time rather than by count, because the parts are not all
    the same length. A section made of twenty half-second lines and one shot
    running a minute is mostly that minute, and a plain median would let the
    twenty short ones outvote it.

    Args:
        parts: `(value, seconds)` pairs.

    Returns:
        The value covering the middle of the total time. Always one that was
        really measured, rather than the midpoint between two of them, which
        for a level is the more useful of the two answers.
    """
    ordered = sorted(parts)
    half = sum(seconds for _, seconds in ordered) / 2
    covered = 0.0
    for value, seconds in ordered:
        covered += seconds
        if covered >= half:
            return value
    return ordered[-1][0]

@dataclass(frozen=True)
class ScoreRule:
    """How one score is combined out of the parts it was measured in.

    Attributes:
        combine: Takes `(value, seconds)` pairs and returns the one number
            standing for all of them.
        precision: Decimal places to keep. Three is right for a share and
            wrong for shake, where an even pan and a handheld shot are a
            hundredfold apart and both round to zero.
        about_content: True for a score describing what is in a stretch —
            how much of it carries speech, how fast it is spoken — rather than
            how the stretch was shot or recorded. The distinction decides two
            things. A part carrying no content score really does carry none,
            so it counts as zero across its whole length, while a part nothing
            was measured in counts for nothing at all. And a search result
            shows the content scores, because they answer what a search asks;
            how well a shot was shot is a question for the few clips that
            survive it.
    """

    combine: Callable[[Sequence[Tuple[float, float]]], float] = _weighted_mean
    precision: int = 3
    about_content: bool = False

# Anything not named here is a weighted mean kept to three decimals, which is what an
# unremarkable new measurement will want.
SCORE_RULES: Dict[str, ScoreRule] = {
    "speech": ScoreRule(about_content=True),
    "silence": ScoreRule(about_content=True),
    "black": ScoreRule(about_content=True),
    "frozen": ScoreRule(about_content=True),
    "words_per_second": ScoreRule(about_content=True),
    "exposure": ScoreRule(precision=4),
    "contrast": ScoreRule(precision=4),
    "motion": ScoreRule(precision=4),
    "shake": ScoreRule(precision=5),
    "loudness": ScoreRule(combine=_decibel_mean, precision=2),
    "peak": ScoreRule(combine=_highest, precision=2),
    "noise_floor": ScoreRule(combine=_typical, precision=2),
    "flatness": ScoreRule(combine=_highest),
    "face_share": ScoreRule(precision=4),
    "face_x": ScoreRule(precision=4),
    "face_y": ScoreRule(precision=4),
}
DEFAULT_RULE = ScoreRule()

def rule_for(name: str) -> ScoreRule:
    """Find how a score combines.

    Args:
        name: The score's name.

    Returns:
        Its rule, or the default for a score nothing has been decided about.
    """
    return SCORE_RULES.get(name, DEFAULT_RULE)

def content_scores() -> Tuple[str, ...]:
    """Name the scores that describe what is in a stretch rather than how it was captured.

    Returns:
        Their names, in the order the rules declare them.
    """
    return tuple(name for name, rule in SCORE_RULES.items() if rule.about_content)

def _combined(parts: Mapping[str, Sequence[Tuple[float, float]]], whole: Optional[float] = None) -> dict:
    """Combine each score from the parts that carried it.

    Args:
        parts: `(value, seconds)` pairs per score name.
        whole: Length of the stretch in seconds. When given, a content score
            is spread across all of it rather than across the parts that
            carried it, so that a stretch nobody speaks over reads as quiet
            rather than as unmeasured.

    Returns:
        One number per score, rounded as its rule asks.
    """
    scores = {}
    for name in sorted(parts):
        rule = rule_for(name)
        value = rule.combine(parts[name])
        if whole is not None and rule.about_content and whole > 0:
            value *= sum(seconds for _, seconds in parts[name]) / whole
        scores[name] = round(value, rule.precision)
    return scores

def combined_over(start: float, end: float, measurements: Sequence[Measured]) -> dict:
    """Combine what was measured across a stretch, weighted by how much of it each covers.

    Measurements are taken per shot and per second, and a clip is neither: a
    sentence can run across two shots and end mid-second. Weighting by the
    overlap is the only reading that does not quietly favour a measurement
    that barely touches the clip.

    Args:
        start: Start of the stretch in seconds.
        end: End of the stretch in seconds.
        measurements: Records to combine; they need not be sorted, and a field
            of one that was not measured is skipped rather than counted as
            zero.

    Returns:
        One number per score. A score nothing overlapping measured is absent,
        which says it was not measured here — not that it was measured and
        came out unremarkable.
    """
    parts: Dict[str, List[Tuple[float, float]]] = {}
    for measurement in measurements:
        overlap = min(end, measurement.end) - max(start, measurement.start)
        if overlap <= 0:
            continue
        for name, value in measurement.model_dump(exclude={"start", "end"}).items():
            if value is not None:
                parts.setdefault(name, []).append((float(value), overlap))
    return _combined(parts)

def combined_from_clips(members: Sequence["SemanticClip"], whole: float) -> dict:
    """Combine the scores of the clips a coarser clip is made of.

    Args:
        members: The clips it covers, in order.
        whole: Its length in seconds.

    Returns:
        One number per score any member carried. A content score is spread
        across the whole stretch — a part with nobody talking really does
        carry no speech — while a measurement is combined only from the parts
        it was taken in, since a part whose blur nobody measured does not make
        the whole any sharper.
    """
    parts: Dict[str, List[Tuple[float, float]]] = {}
    for clip in members:
        for name, value in clip.scores.items():
            parts.setdefault(name, []).append((value, clip.duration))
    return _combined(parts, whole)

def speaker_at(
    turns: Sequence[SpeakerTurn],
    start: float,
    end: float,
    majority: float = SPEAKER_MAJORITY,
) -> Optional[str]:
    """Say whose stretch this is, when it is clearly one person's.

    Silence in the stretch is not held against anybody: what is weighed is the
    talking in it, so a sentence with a long pause in the middle still belongs
    to whoever said both halves of it.

    Args:
        turns: The file's speaker turns.
        start: Start of the stretch in seconds.
        end: End of the stretch in seconds.
        majority: Share of the talking one voice has to hold before the
            stretch is called theirs.

    Returns:
        The label, or `None` when nobody is speaking over the stretch or when
        it straddles a handover evenly enough that naming one of them would be
        a guess. An unlabelled clip is the honest answer there: the label goes
        on screen and into an edit, and a wrong one is worse than none.
    """
    held: dict = {}
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > 0:
            held[turn.speaker] = held.get(turn.speaker, 0.0) + overlap
    if not held:
        return None
    spoken = sum(held.values())
    longest = max(sorted(held), key=lambda speaker: held[speaker])
    return longest if held[longest] / spoken >= majority else None

def was_audible(silence_share: float, has_audio: bool) -> bool:
    """Decide whether words transcribed over a stretch were ever actually said.

    Speech recognition writes whole sentences over footage with nobody in it —
    a channel sign-off, a subtitle credit, whatever the model heard most in
    training — and hands them back with as much confidence as a real line. The
    silence detector measured the sound itself, so where it says there was
    nothing to hear, there was nothing said, whatever came back as text.

    An asset with no sound at all is not judged here: it has no silences
    measured against it, and no transcript either.

    Args:
        silence_share: How much of the stretch the silence detector marked,
            as `share_covered` measures it.
        has_audio: Whether the asset has an audio stream at all.

    Returns:
        False only when the stretch was measured as silent nearly all the way
        through.
    """
    return not has_audio or silence_share < NEARLY_ALL

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
        Scores keyed by name. `speech` is the share carrying transcribed
        words, `silence`, `black` and `frozen` the shares each detector
        marked, and `words_per_second` the pace — whose unit is whatever the
        transcript counts as a word, so a Chinese transcript counts characters
        and the number compares within a language rather than across them.
        The rest are the picture and sound measurements averaged over the
        stretch: `exposure`, `contrast`, `blur`, `motion` and `shake` from the
        shots it spans, `loudness`, `peak`, `noise_floor` and `flatness` from
        the seconds it spans, and `faces`, `face_share`, `face_x` and `face_y`
        from who was on screen during them. A measurement that was not taken
        is absent rather than zero, which is why `face_x` is missing from a
        clip with nobody in it rather than sitting in the middle of the frame.
    """
    words = segment.words if segment is not None else []
    scores = {
        "speech": share_covered(start, end, [Span(start=word.start, end=word.end) for word in words]),
        "silence": share_covered(start, end, analysis.silences),
        "black": share_covered(start, end, analysis.black_frames),
        "frozen": share_covered(start, end, analysis.frozen_frames),
    }
    if words and end > start:
        scores["words_per_second"] = len(words) / (end - start)
    scores = {name: round(value, rule_for(name).precision) for name, value in scores.items()}
    for measurements in (analysis.shots, analysis.sound, analysis.faces):
        scores.update(combined_over(start, end, measurements))
    return scores

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
        motion measurement the analysis does not take. Words transcribed over
        a stretch measured as silent do not make it `SPEECH`: see
        `was_audible`.
    """
    if has_speech and was_audible(scores["silence"], has_audio):
        return ClipKind.SPEECH
    if scores["black"] >= MOSTLY or scores["frozen"] >= MOSTLY:
        return ClipKind.UNUSABLE
    if has_audio and scores["silence"] >= NEARLY_ALL:
        return ClipKind.SILENCE
    return ClipKind.AMBIENT

def _cosine_distance(one: Sequence[float], two: Sequence[float]) -> float:
    """Measure how far apart two voices sit.

    Cosine rather than straight distance, because the embedding model's
    vectors carry loudness in their length: the same person recorded closer to
    the microphone is further away by every other measure and no different by
    this one.

    Args:
        one: The first embedding.
        two: The second.

    Returns:
        0.0 for the same direction, 1.0 for unrelated, 2.0 for opposite. Two
        vectors of different lengths are not comparable at all and answer 2.0.
    """
    if len(one) != len(two):
        return 2.0
    size = math.sqrt(sum(value * value for value in one)) * math.sqrt(sum(value * value for value in two))
    return 1.0 - sum(a * b for a, b in zip(one, two)) / size if size else 2.0

def join_voices(analyses: Mapping[str, MediaAnalysis]) -> Dict[Tuple[str, str], str]:
    """Work out which labels in different files are the same person.

    Diarization labels only mean something inside their own file, so an
    interview shot on two cameras comes back as four unrelated people. This is
    where they are joined up: each voice is compared with the ones already
    seen, and lands with the nearest of them when it is near enough.

    Walked in a fixed order — files by ID, labels within a file by name — so
    the same footage always produces the same joined labels. Assigning them by
    who speaks first would make the answer depend on the order the analyses
    happened to be read in.

    Greedy rather than a proper clustering: with a handful of voices the two
    agree, and a greedy pass can be read straight through. If a corpus ever
    shows them disagreeing, that is the moment to pay for the better one.

    Args:
        analyses: The analyses to join, keyed by asset ID.

    Returns:
        A joined label, `V1`, `V2` and so on, for each `(asset_id, label)`.
        Files whose voices were never measured are absent, which is not the
        same as their being nobody: it means nobody looked.
    """
    joined: Dict[Tuple[str, str], str] = {}
    known: List[Tuple[str, List[float]]] = []
    for asset_id in sorted(analyses):
        for voice in sorted(analyses[asset_id].voices, key=lambda item: item.speaker):
            nearest, distance = None, VOICE_DISTANCE
            for label, embedding in known:
                apart = _cosine_distance(voice.embedding, embedding)
                if apart < distance:
                    nearest, distance = label, apart
            if nearest is None:
                nearest = f"V{len(known) + 1}"
                known.append((nearest, list(voice.embedding)))
            joined[(asset_id, voice.speaker)] = nearest
    return joined

def voice_levels(
    analyses: Mapping[str, MediaAnalysis],
    joined: Mapping[Tuple[str, str], str],
) -> Dict[str, float]:
    """Measure how loudly each voice speaks, across every file they are in.

    Only the seconds that voice was actually holding the floor, so the room
    tone between two people's turns is not counted against either of them.
    Levels are combined as energy rather than as decibels, for the same reason
    every other level in this module is.

    Args:
        analyses: The analyses to measure, keyed by asset ID.
        joined: The joined label for each `(asset_id, per-file label)`, as
            `join_voices` gives it.

    Returns:
        A level in decibels per joined voice. A voice whose files carry no
        per-second sound measurements is absent: nobody measured it, which is
        not the same as its being silent.
    """
    parts: Dict[str, List[Tuple[float, float]]] = {}
    for asset_id, analysis in analyses.items():
        for turn in analysis.speakers:
            label = joined.get((asset_id, turn.speaker))
            if label is None:
                continue
            for record in analysis.sound:
                covered = min(turn.end, record.end) - max(turn.start, record.start)
                if covered > 0:
                    parts.setdefault(label, []).append((record.loudness, covered))
    return {label: round(_decibel_mean(heard), 2) for label, heard in parts.items()}

def voice_gains(levels: Mapping[str, float]) -> Dict[str, float]:
    """Work out what to turn each voice by so they match each other.

    Everybody is brought down to the quietest of them rather than up to the
    loudest. Which one is the reference makes no difference to the result — the
    render is normalized to one loudness afterwards, so a shift applied to
    every voice alike is undone — but turning down can never clip, and turning
    up would also bring up whatever hiss was under the quiet one.

    Args:
        levels: The level each voice speaks at, in decibels.

    Returns:
        A multiplier per voice, 1.0 for the quietest. Empty when fewer than
        two voices were measured: one voice is already consistent with itself,
        and this is not a loudness target.
    """
    if len(levels) < 2:
        return {}
    quietest = min(levels.values())
    return {label: round(10 ** ((quietest - level) / 20), 4) for label, level in levels.items()}

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

    # The first place that sees more than one file at once, so the first place a label
    # can mean the same person in two of them. The analyses keep their own `S1` and `S2`,
    # which is what diarization actually said; the joined labels are `V1`, `V2`, so that
    # nobody reads one for the other.
    voices = join_voices(analyses)
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
                speaker=voices.get((asset_id, said_by)) if (
                    said_by := speaker_at(analysis.speakers, start, end)
                ) else None,
                scores=scores,
            ))
    return timeline, clips

@dataclass(frozen=True)
class CleanCuts:
    """What the compiler is allowed to know about one file.

    Everything here is a number or a string lifted off the analysis, never the
    analysis itself. Reading a transcript is interpretation, and the compiler
    has to stay a pure function of what it is given, so the interpretation
    happens once in `clean_cuts` and the compiler is handed the result.

    The questions are kept apart because they have different answers and very
    different costs. Moving a cut off the middle of a word costs a fraction of
    a second and is never worth refusing. Moving it to the end of the sentence
    can cost many seconds, which changes how long somebody's video runs — so
    that one is reported rather than done to them.

    Empty is always read as not knowing rather than as nothing being there. A
    file nobody transcribed has no words to split and a file analyzed before
    the picture detectors ran has no bad frames to land on, and in neither
    case has anything been cleared.

    Attributes:
        words: Every transcribed word as `(start, end)` in source seconds, in
            order.
        sentences: Every transcribed sentence the same way.
        said: What each of those sentences says, in the same order, for
            telling two takes of one line apart.
        pauses: Every measured silence, which is where the dead air is.
        bad_picture: Every stretch the viewer should not be shown a frame of:
            black and frozen picture, merged into one list because a cut point
            has the same objection to either. Soft focus is not in here — it
            would need a threshold on the blur measurement, and that is one of
            the numbers the roadmap refuses to pick without a corpus.
    """

    words: Tuple[Tuple[float, float], ...] = ()
    sentences: Tuple[Tuple[float, float], ...] = ()
    said: Tuple[str, ...] = ()
    pauses: Tuple[Tuple[float, float], ...] = ()
    bad_picture: Tuple[Tuple[float, float], ...] = ()

    def splits_a_word(self, seconds: float) -> bool:
        """Say whether a cut here would land inside a word.

        The same rounding tolerance the rest of this module allows on a snapped
        boundary applies, so that this and the benchmark's `words_cut` — which
        asks the identical question of a finished cut — cannot answer it
        differently and leave the compiler chasing a fault the scorer does not
        see.

        Args:
            seconds: The time to check.

        Returns:
            True when a word is in progress. A file with no word timings
            answers False: nothing is *known* to break, which is not the same
            as nothing breaking.
        """
        return any(
            start + EDGE_TOLERANCE_SECONDS < seconds < end - EDGE_TOLERANCE_SECONDS
            for start, end in self.words
        )

    def splits_a_sentence(self, seconds: float) -> bool:
        """Say whether a cut here would leave a sentence unfinished.

        Args:
            seconds: The time to check.

        Returns:
            True when the time falls inside a transcribed sentence.
        """
        return any(start < seconds < end for start, end in self.sentences)

    def rest_of_sentence(self, seconds: float) -> float:
        """Measure how much longer the sentence being cut through runs.

        Args:
            seconds: The time the cut lands at.

        Returns:
            The seconds still to be spoken, or 0.0 when no sentence is in
            progress. This is what a note hands back to whoever chose the
            length, because spending it is their decision and not a
            compiler's.
        """
        inside = [end for start, end in self.sentences if start < seconds < end]
        return round(max(inside) - seconds, 3) if inside else 0.0

    def nearest_word_edge(self, seconds: float, within: float) -> Optional[float]:
        """Find the closest second near a time at which no word is in progress.

        Args:
            seconds: The time a cut was asked for.
            within: How far the cut may move, in seconds.

        Returns:
            The nearest such second, or None when the file offers none that
            close.
        """
        best: Optional[float] = None
        for edge in (edge for word in self.words for edge in word):
            if abs(edge - seconds) > within:
                continue
            # Nearest wins. A tie goes to the later edge, which keeps the word whole
            # rather than dropping it — hence the sort key falling on -edge.
            if best is None or (abs(edge - seconds), -edge) < (abs(best - seconds), -best):
                best = edge
        return best

    def shows_bad_picture(self, seconds: float, opens: bool) -> bool:
        """Say whether the frame a cut point puts on screen is one nobody wants to see.

        Half-open, and which half is open depends on the end. An opening point
        is the first frame the viewer sees, so black starting exactly there is
        seen. A closing point is the first frame they do not, so black
        starting exactly there is not, while black ending exactly there was on
        screen until the last moment. The benchmark's `bad_frame_cuts` asks
        this same question of a finished cut and has to get the same answer,
        or the compiler would be chasing a fault the scorer cannot see.

        Args:
            seconds: The time to check.
            opens: Whether the cut point opens the window rather than closing it.

        Returns:
            True when the frame shown there is black or frozen.
        """
        if opens:
            return any(start <= seconds < end for start, end in self.bad_picture)
        return any(start < seconds <= end for start, end in self.bad_picture)

    def is_clean(self, seconds: float, opens: bool) -> bool:
        """Say whether a cut may land here at all.

        Args:
            seconds: The time to check.
            opens: Whether the cut point opens the window rather than closing it.

        Returns:
            True when no word is in progress and no bad frame is on screen.
        """
        return not self.splits_a_word(seconds) and not self.shows_bad_picture(seconds, opens)

    def nearest_clean_point(self, seconds: float, within: float, opens: bool) -> Optional[float]:
        """Find the closest second near a time where a cut may land.

        The places worth trying are the edges of the things being avoided: the
        edge of a word keeps the word whole, and the edge of a black stretch
        is the first frame either side of it that is worth showing. Every
        candidate is checked against both objections, so moving off a bad
        frame cannot land in the middle of a word.

        Args:
            seconds: The time a cut was asked for.
            within: How far the cut may move, in seconds.
            opens: Whether the cut point opens the window rather than closing it.

        Returns:
            The nearest such second, or None when the file offers none that
            close.
        """
        best: Optional[float] = None
        for edge in {edge for span in (*self.words, *self.bad_picture) for edge in span}:
            if abs(edge - seconds) > within or not self.is_clean(edge, opens):
                continue
            # Nearest wins. A tie goes to the later edge, which keeps the word whole
            # rather than dropping it — hence the sort key falling on -edge.
            if best is None or (abs(edge - seconds), -edge) < (abs(best - seconds), -best):
                best = edge
        return best

    def pauses_inside(self, start: float, end: float, longer_than: float) -> Tuple[Tuple[float, float], ...]:
        """List the dead air strictly inside a window.

        Strictly inside, because a silence that reaches either edge is the
        breath the window was given rather than a pause in the middle of it,
        and trimming that is the head and tail trim's job, not this one.

        Args:
            start: Where the window opens, in source seconds.
            end: Where it closes.
            longer_than: How long a silence has to run to count as dead air.

        Returns:
            The qualifying silences, in order.
        """
        return tuple(
            (quiet, loud) for quiet, loud in self.pauses
            if quiet > start and loud < end and loud - quiet > longer_than
        )

    def spoken_between(self, start: float, end: float) -> str:
        """Collect what is said across a window.

        Args:
            start: Where the window opens, in source seconds.
            end: Where it closes.

        Returns:
            The text of every sentence that overlaps it, joined in order.
            Empty for a window with nobody talking in it, and also for a file
            nobody transcribed — which is why the retake pass refuses to act
            on an empty answer.
        """
        return "".join(
            text for (opened, closed), text in zip(self.sentences, self.said)
            if opened < end and start < closed
        )

    def first_word_in(self, start: float, end: float) -> Optional[float]:
        """Find where the talking starts inside a window.

        Args:
            start: Where the window opens, in source seconds.
            end: Where it closes.

        Returns:
            The start of the earliest word inside it, or None when nobody
            talks there.
        """
        inside = [opened for opened, closed in self.words if opened >= start and closed <= end]
        return min(inside) if inside else None

    def last_word_in(self, start: float, end: float) -> Optional[float]:
        """Find where the talking stops inside a window.

        Args:
            start: Where the window opens, in source seconds.
            end: Where it closes.

        Returns:
            The end of the latest word inside it, or None when nobody talks
            there.
        """
        inside = [closed for opened, closed in self.words if opened >= start and closed <= end]
        return max(inside) if inside else None

def _joined(spans: Sequence[Span]) -> Tuple[Tuple[float, float], ...]:
    """Merge overlapping stretches into one list, in order.

    Black and frozen picture are detected separately and overlap constantly —
    a frozen shot that is also black is reported by both. A cut point has the
    same objection to either, so they are counted once.

    Args:
        spans: The stretches, in any order.

    Returns:
        The stretches, sorted and with overlaps joined.
    """
    joined: List[Tuple[float, float]] = []
    for span in sorted(spans, key=lambda item: item.start):
        if joined and span.start <= joined[-1][1]:
            joined[-1] = (joined[-1][0], max(joined[-1][1], span.end))
        else:
            joined.append((span.start, span.end))
    return tuple(joined)

def clean_cuts(analysis: MediaAnalysis) -> CleanCuts:
    """Work out what the compiler may know about one file.

    A pure function of the detections already on disk, so the compiler can be
    handed numbers rather than transcripts to interpret.

    The transcript and the picture detectors are read separately on purpose. A
    file analyzed without transcription still has silences and black frames
    worth avoiding, and refusing to report them because nobody transcribed it
    would leave the compiler blind to things it was told.

    Args:
        analysis: The file's analysis.

    Returns:
        What may be known. Any part of it can be empty, and empty always means
        nothing is known rather than nothing is there.
    """
    transcript = analysis.transcript
    segments = transcript.segments if transcript is not None else []
    return CleanCuts(
        words=tuple(
            (word.start, word.end)
            for segment in segments
            for word in segment.words
        ),
        sentences=tuple((segment.start, segment.end) for segment in segments),
        said=tuple(segment.text for segment in segments),
        pauses=tuple((span.start, span.end) for span in analysis.silences),
        bad_picture=_joined([*analysis.black_frames, *analysis.frozen_frames]),
    )
