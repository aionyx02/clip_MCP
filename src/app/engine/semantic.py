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

import bisect
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from app.models.media import SILENCE_DB, Asset, MediaAnalysis, Measured, Span, SpeakerTurn, TranscriptSegment
from app.models.semantic import ClipKind, ClipLevel, SemanticClip, SemanticTimeline

# Bumped whenever the rules below change, so timelines built by an older
# version are recognised as out of date rather than silently reused.
# 8: lines the recogniser made up are told by more than their confidence, a stretch
# without speech has no speaker, and voices join across files by their average.
DERIVATION_VERSION = 8
# A stretch without speech shorter than this is not a thing on its own; it is the
# breath between two sentences, and it is already described by their headroom.
MIN_UTTERANCE_SECONDS = 0.4
# Silence edges and transcript boundaries are both rounded to milliseconds, so a
# boundary can land a hair outside the silence it was snapped to.
EDGE_TOLERANCE_SECONDS = 0.02
# A cut is clean where the speaker stopped: between two words at least this far apart.
# Shorter gaps are the spaces inside a phrase, and a cut there sounds like a word bitten
# off. Provisional.
CLEAN_GAP_SECONDS = 0.2
# A cut inside a measured silence at least this long is clean whatever the word timings
# say: the transcriber runs a word on across a pause often enough that taking its word
# for it refused cuts made in the middle of real silence. Provisional.
CLEAN_SILENCE_SECONDS = 0.3
# What a word ends on when the transcriber heard the phrase end there.
PHRASE_END_PUNCTUATION = "。！？!?…，,、；;：:"
# The transcriber places a word's start late, or its end early, often enough to cut a
# syllable off: the first edit made with the server lost the 「抽」 of 「抽杯」, which began
# 0.39s before the next word the transcript knew of. Sound this long between a measured
# silence and a cut, with no word over it, is somebody talking that the transcript
# missed; this far is as far back as it is looked for. Provisional.
UNHEARD_SECONDS = 0.15
ONSET_REACH_SECONDS = 0.6
# Where no silence was measured — a café, a fan, a street — the second listen is what
# says where the talking starts and stops. A voiced stretch has to start this far before
# a cut, or run this far past one, to have been cut into. The ends are given more room
# because the detector holds on after a voice stops: against the level measured every
# tenth of a second on the first edit, its ends ran 0.08s to 0.26s late, and its starts
# were within a window. Provisional (roadmap §13).
VOICED_MARGIN_SECONDS = 0.05
VOICED_TAIL_MARGIN_SECONDS = 0.15
# A heard word counts as the detector's to place once this much of it is voiced. The
# speech model stretches a word back across the pause in front of it — 「然後」 was heard
# from 18.02s where the voice resumes at 18.24s — so a word the detector hears is where
# the detector says; one it missed altogether, a soft 「我們」, is where the model says.
# Provisional (roadmap §13).
VOICED_SHARE = 0.25
# A word heard on the second listen counts only when the speech model was at least this
# sure of it. Transcribing near-silence on its own it hears 「字幕」 and 「放入冰箱」 at
# well under this; the words it was right about were well over it. Provisional (roadmap §13).
HEARD_PROBABILITY = 0.5
# Two heard words this close together are one phrase, so a cut between them takes the
# first off the second. Provisional (roadmap §13).
PHRASE_GAP_SECONDS = 0.3
# A cut this close to where the transcript has one word end and the next begin was put on
# that edge: the word on the far side is being taken off on purpose — 「各位」 off the front
# of 「各位我到高鐵了」 — however the sound runs on across it. Provisional.
WORD_EDGE_SECONDS = 0.05
MOSTLY = 0.5
NEARLY_ALL = 0.9
# How much of the talking in a stretch one voice has to hold before the stretch is called
# theirs. A sentence that straddles a handover belongs to neither, and says so.
# Provisional.
SPEAKER_MAJORITY = 0.8
# How far apart two voices have to sit before they are two people, when the question is
# whether a label in one file is the same person as a label in another. Cosine distance,
# so 0 is the same direction and 1 is unrelated. Kept apart from `diarize.SPEAKER_DISTANCE`,
# which asks a different question — that one splits one recording, this one joins several,
# and a recording made on another day in another room is further from itself than two
# people in one room are from each other. Provisional, like that one.
VOICE_DISTANCE = 0.45
# How far a file's only voice may sit from a voice already heard and still be them. A
# vlogger alone on a street measured 0.5 from themselves on another day, so a single
# voice is given more room than one of several. Provisional.
LONE_VOICE_DISTANCE = 0.6
# A voice talking less than this share of all the talking is worth asking about: one person
# heard on another microphone comes back as a voice of its own, a few seconds long beside
# the hours of the one it is (#82). Provisional.
MINOR_VOICE_SHARE = 0.1

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
    "warmth": ScoreRule(precision=4),
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

# Provisional (roadmap §13): a sentence whose words the speech model gave, on average, less
# than this chance of being right was made up — the recogniser writing 「3D列印、炒麵」 over a
# shot of a drone taking off. Real mumbling still averages well above it.
MADE_UP_PROBABILITY = 0.1

# Provisional: a sentence most of whose words the model gave no length at all — start and end
# the same instant — was not heard; the recogniser wrote it out of nothing, as it does
# 「中文字幕——YK」 over the hum of a café.
MADE_UP_EMPTY_SHARE = 0.5
# Provisional: a sentence whose first word the model gave under this chance, and that has a
# word of no length, was made up; a real line can open unsure, but not with nothing under it.
MADE_UP_FIRST_PROBABILITY = 0.02
# Provisional: the same word this many times running is the recogniser stuck in a loop —
# 「沒興趣,」 forty times over street noise — not someone repeating themselves.
MADE_UP_REPEATS = 6
# What the recogniser was trained on most: subtitle credits and a channel's sign-off and
# plea to subscribe, in either script, as it writes them rather than as anyone says them.
# 「謝謝大家」 is left out: people say it at the end of their own videos.
_MADE_UP_LINES = [
    re.compile(pattern) for pattern in (
        r"^中文字幕|字幕[組组由提供志]",
        # Its plea, word for word: 「請不吝點贊 訂閱 轉發 打賞支持明鏡與點點欄目」. A vlogger
        # asking for a subscription in their own words says neither 不吝 nor 點點欄目.
        r"不吝[點点][贊赞讚]|[點点]{2}[欄栏]目",
        r"^([謝谢][謝谢]|感[謝谢]|多[謝谢])(觀看|观看|收看|收聽|收听)$",
    )
]

def was_made_up(segment: TranscriptSegment) -> bool:
    """Decide whether the speech model invented a sentence nobody said.

    The silence detector catches sentences written over silence; this catches
    the ones written over sound that is not talking — wind, engines, a café —
    which the model hands back with almost no confidence in a single word,
    with words that take no time at all, stuck repeating one word, or as one
    of the lines it was trained on most: a subtitle credit, a sign-off.

    Args:
        segment: The sentence.

    Returns:
        True when any of those holds. Probabilities are only judged when
        every word carries one, so a transcript made before they were kept
        is judged by the rest.
    """
    words = segment.words
    bare = re.sub(r"[\s\W_]+", "", segment.text)
    if any(line.search(bare) for line in _MADE_UP_LINES):
        return True
    if not words:
        return False
    empty = sum(1 for word in words if word.end <= word.start)
    if empty / len(words) >= MADE_UP_EMPTY_SHARE:
        return True
    said = [re.sub(r"[\s\W_]+", "", word.text) for word in words]
    run = 1
    for before, word in zip(said, said[1:]):
        run = run + 1 if word and word == before else 1
        if run >= MADE_UP_REPEATS:
            return True
    chances = [word.probability for word in words]
    if any(chance is None for chance in chances):
        return False
    if chances[0] < MADE_UP_FIRST_PROBABILITY and empty:
        return True
    return sum(chances) / len(chances) < MADE_UP_PROBABILITY

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
    that some detector marked, so `query_clips` can filter on them.

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
    agree, and a greedy pass can be read straight through. Each joined voice
    is compared by where its files sit on average, weighted by how much
    talking each was measured over, so a week of one person's vlog gathers
    around them instead of drifting off a voice at a time. A file with only
    one voice in it is given more room, up to `LONE_VOICE_DISTANCE`: a person
    alone with a camera on a noisy street is further from themselves than two
    people in one room are from each other, and there is nobody else in the
    file for them to be.

    Args:
        analyses: The analyses to join, keyed by asset ID.

    Returns:
        A joined label, `V1`, `V2` and so on, for each `(asset_id, label)`.
        Files whose voices were never measured are absent, which is not the
        same as their being nobody: it means nobody looked.
    """
    joined: Dict[Tuple[str, str], str] = {}
    # Each joined voice: its label, the sum of its files' voices weighted by seconds, and the seconds.
    known: List[Tuple[str, List[float], float]] = []
    for asset_id in sorted(analyses):
        voices = analyses[asset_id].voices
        reach = LONE_VOICE_DISTANCE if len(voices) == 1 else VOICE_DISTANCE
        taken: set[str] = set()
        for voice in sorted(voices, key=lambda item: item.speaker):
            nearest, distance = None, reach
            for index, (label, total, _) in enumerate(known):
                if label in taken:
                    # Two voices the speaker split told apart in one file are two people.
                    continue
                apart = _cosine_distance(voice.embedding, total)
                if apart < distance:
                    nearest, distance = index, apart
            if nearest is None:
                known.append((f"V{len(known) + 1}", [0.0] * len(voice.embedding), 0.0))
                nearest = len(known) - 1
            # Joined only within `reach`, which a vector of another length never is, so the two always match.
            label, total, seconds = known[nearest]
            total = [held + value * voice.seconds for held, value in zip(total, voice.embedding)]
            known[nearest] = (label, total, seconds + voice.seconds)
            taken.add(label)
            joined[(asset_id, voice.speaker)] = label
    return joined

def maybe_one_person(analyses: Mapping[str, MediaAnalysis], seconds: Mapping[str, float]) -> List[dict]:
    """Point out the voices that may be the main one heard another way, for somebody to ask about.

    Every voice talking less than `MINOR_VOICE_SHARE` of all the talking,
    with how far it sits from the one who talks most. Not joined here: the
    distance is the evidence, and whether two microphones caught one person
    is for whoever knows the footage to say.

    Args:
        analyses: The analyses the voices were joined over, keyed by asset ID.
        seconds: How long each joined voice talks, as `voice_seconds` gives it.

    Returns:
        One `{voice, seconds, like, distance}` per such voice, the closest
        first; `distance` is None when either was never measured.
    """
    if len(seconds) < 2:
        return []
    whole = sum(seconds.values())
    main = max(seconds, key=lambda label: seconds[label])
    prints: Dict[str, List[float]] = {}
    for (asset_id, speaker), label in join_voices(analyses).items():
        voice = next((item for item in analyses[asset_id].voices if item.speaker == speaker), None)
        if voice is None:
            continue
        held = prints.setdefault(label, [0.0] * len(voice.embedding))
        if len(held) == len(voice.embedding):
            prints[label] = [value + part * voice.seconds for value, part in zip(held, voice.embedding)]
    found = []
    for label, talked in seconds.items():
        if label == main or talked >= MINOR_VOICE_SHARE * whole:
            continue
        apart = (_cosine_distance(prints[label], prints[main]) if label in prints and main in prints else None)
        found.append({"voice": label, "seconds": talked, "like": main,
                      "distance": None if apart is None or apart >= 2.0 else round(apart, 2)})
    return sorted(found, key=lambda item: (item["distance"] is None, item["distance"] or 0.0))

def voice_seconds(clips: Sequence[SemanticClip]) -> Dict[str, float]:
    """Add up how long each joined voice talks, so one person split in several is plain to see.

    Args:
        clips: The clips of a timeline.

    Returns:
        Seconds of talking by joined label, the most talkative first.
    """
    held: Dict[str, float] = {}
    for clip in clips:
        if clip.speaker is not None:
            held[clip.speaker] = held.get(clip.speaker, 0.0) + clip.source_range.end - clip.source_range.start
    return {label: round(seconds, 1) for label, seconds in sorted(held.items(), key=lambda item: -item[1])}

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
                analyses[asset_id].model_dump_json(exclude={"analyzed_at", "heard_again"}).encode("utf-8")
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
    refer to a clip by ID and mean the same moment tomorrow.

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
                kind=(kind := _kind(bool(text) and not was_made_up(segment), asset.has_audio, scores)),
                text=text,
                # Nobody is speaking in a stretch without words, whatever the speaker split heard there.
                speaker=voices.get((asset_id, said_by)) if kind == ClipKind.SPEECH and (
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
        spoken: What each word says, in the order of `words`.
        voices: Who says each word, in the order of `words`, as the speaker
            split labelled them within this file; None for a word no voice was
            told apart in, and empty when nobody split them.
        pauses: Every measured silence, which is where the dead air is.
        bad_picture: Every stretch the viewer should not be shown a frame of:
            black and frozen picture, merged into one list because a cut point
            has the same objection to either. Soft focus is not in here — it
            would need a threshold on the blur measurement, and that is one of
            the numbers the roadmap refuses to pick without a corpus.
        transcribed: Whether anybody transcribed the file at all. This is
            what tells an empty `words` apart: a transcribed file with no
            words in it has nobody talking, and a cut can move anywhere in it
            without splitting one; an untranscribed file has words nobody
            knows about.
        beats: Where the beat falls, for music. The one field where empty and
            unknown are kept apart, because the answers differ: None is a song
            nobody measured, and an empty tuple is one measured and found to
            have no steady pulse.
        energy: For a song, its energy each second on its own scale from 0
            to 1; None when it was not heard as one.
        loudness: For a song, its loudness each second in dBFS, which its
            loop is found from when its energy was never kept.
        voiced: Where the second listen's voice detector hears talking.
        heard: The words the second listen heard, each stretch transcribed on
            its own, as `(start, end, text)`; only those it was at least
            `HEARD_PROBABILITY` sure of.
        listened: Whether the file was listened to a second time at all, which
            tells an empty `voiced` apart from an unknown one.
        duration: How long the file is, when known, so that a window moved
            out to the end of the talking never runs past the end of it.
    """

    words: Tuple[Tuple[float, float], ...] = ()
    sentences: Tuple[Tuple[float, float], ...] = ()
    said: Tuple[str, ...] = ()
    spoken: Tuple[str, ...] = ()
    voices: Tuple[Optional[str], ...] = ()
    pauses: Tuple[Tuple[float, float], ...] = ()
    bad_picture: Tuple[Tuple[float, float], ...] = ()
    transcribed: bool = False
    beats: Optional[Tuple[float, ...]] = None
    energy: Optional[Tuple[float, ...]] = None
    loudness: Optional[Tuple[float, ...]] = None
    voiced: Tuple[Tuple[float, float], ...] = ()
    heard: Tuple[Tuple[float, float, str], ...] = ()
    listened: bool = False
    duration: Optional[float] = None

    def begins_phrase(self, index: int) -> bool:
        """Say whether a word is the first of a phrase: what somebody says between two stops.

        A transcriber's sentence is not always a sentence. Without punctuation
        to go on it runs several together, half a minute of them at a time, so
        where the speaker stopped is read off the words themselves: a pause
        long enough to cut in, a measured silence or a stretch the second
        listen heard nobody in between them however the word timings run,
        punctuation, a space the transcriber put between them, or the start
        of one of its sentences.

        Args:
            index: The word, by its place in `words`.

        Returns:
            True when a phrase begins on it. A file whose words carry no text
            answers on the timings alone.
        """
        if index <= 0:
            return True
        (earlier_start, before), (start, end) = self.words[index - 1], self.words[index]
        if start - before >= CLEAN_GAP_SECONDS:
            return True
        # The transcriber often lays words end to end and stretches one over the pause before it.
        # A stretch inside the earlier word belongs to whichever of its ends it lies nearer: nearer
        # its start, it was a pause before that word, not after it. The search starts at the first
        # stretch that could reach the earlier word's end, so one that began before the earlier word
        # and runs on past it is still weighed, however many others it overlaps.
        quiet = self._quiet_stretches
        at = min(bisect.bisect_left(quiet, (earlier_start, earlier_start)),
                 bisect.bisect_left(self._quiet_reach, before - EDGE_TOLERANCE_SECONDS))
        while at < len(quiet) and quiet[at][0] <= end:
            stops, resumes = quiet[at]
            if resumes <= end and (resumes >= before - EDGE_TOLERANCE_SECONDS
                                   or stops - earlier_start > before - resumes):
                return True
            at += 1
        opens = self._sentence_starts
        at = bisect.bisect_left(opens, before - EDGE_TOLERANCE_SECONDS)
        if at < len(opens) and opens[at] <= start + EDGE_TOLERANCE_SECONDS:
            return True
        if len(self.spoken) == len(self.words):
            earlier, this = self.spoken[index - 1], self.spoken[index]
            return (earlier.rstrip().endswith(tuple(PHRASE_END_PUNCTUATION))
                    or this[:1].isspace() or earlier[-1:].isspace())
        return False

    @cached_property
    def _quiet_stretches(self) -> Tuple[Tuple[float, float], ...]:
        """Every stretch long enough to begin a phrase after, in time order: a measured
        silence of `CLEAN_SILENCE_SECONDS`, or a gap of `CLEAN_GAP_SECONDS` between two
        stretches the second listen heard somebody in."""
        talking = sorted(self.voiced)
        return tuple(sorted(
            [(start, end) for start, end in self.pauses if end - start >= CLEAN_SILENCE_SECONDS]
            + [(stops, resumes) for (_, stops), (resumes, _) in zip(talking, talking[1:])
               if resumes - stops >= CLEAN_GAP_SECONDS]))

    @cached_property
    def _quiet_reach(self) -> Tuple[float, ...]:
        """How far any stretch up to and including each one of `_quiet_stretches` reaches, so a
        search can skip the ones that all end before a time; the stretches overlap and nest, so
        their own ends are not in order."""
        return tuple(itertools.accumulate((resumes for _, resumes in self._quiet_stretches), max))

    @cached_property
    def _sentence_starts(self) -> Tuple[float, ...]:
        """Where the transcriber's sentences start, in time order."""
        return tuple(sorted(opens for opens, _ in self.sentences))

    def word_split_at(self, seconds: float) -> Optional[Tuple[float, float]]:
        """Find the word a cut here would land inside, as `(start, end)`; None when it lands in none."""
        return next((word for word in self.words
                     if word[0] + EDGE_TOLERANCE_SECONDS < seconds < word[1] - EDGE_TOLERANCE_SECONDS), None)

    def splits_a_word(self, seconds: float) -> bool:
        """Say whether a cut here would land inside a word.

        The same rounding tolerance the rest of this module allows on a snapped
        boundary applies.

        Args:
            seconds: The time to check.

        Returns:
            True when a word is in progress. A file with no word timings
            answers False: nothing is *known* to break, which is not the same
            as nothing breaking.
        """
        return self.word_split_at(seconds) is not None

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
        screen until the last moment.

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

        In a transcribed file, a silence is dead air only where somebody stops
        talking before it and starts again after it, inside the window. Quiet
        in a shot nobody talks in — a hand reaching for a cup — is the action
        itself, and so is quiet after the last thing said: taking either out
        cuts away what the shot was chosen to show. And it has to fall between
        two sentences: a pause in the middle of one is how somebody talks, and
        taking it out jumps inside the sentence, which the check before a
        render rightly calls a cut in the middle of somebody talking. A file
        nobody transcribed cannot be told apart that way, so every long
        silence in it counts.

        Args:
            start: Where the window opens, in source seconds.
            end: Where it closes.
            longer_than: How long a silence has to run to count as dead air.

        Returns:
            The qualifying silences, in order.
        """
        def talked_around(quiet: float, loud: float) -> bool:
            before = any(begins < quiet and ends > start for begins, ends in self.sentences)
            after = any(ends > loud and begins < end for begins, ends in self.sentences)
            middle = (quiet + loud) / 2
            between = not any(begins < middle < ends for begins, ends in self.sentences)
            return before and after and between

        return tuple(
            (quiet, loud) for quiet, loud in self.pauses
            if quiet > start and loud < end and loud - quiet > longer_than
            and (not self.transcribed or talked_around(quiet, loud))
        )

    @property
    def loop(self) -> Optional[Tuple[float, float]]:
        """Where a song can loop without the jump being heard, as `music.loop_span` finds it; None when unknown."""
        from app.engine.music import energy_curve, loop_span

        # A song heard before its energy was kept still has its loudness, which the energy comes from.
        energy = self.energy or (tuple(energy_curve(self.loudness)) if self.loudness else ())
        return loop_span(self.beats or (), energy, self.duration)

    def sound_before(self, seconds: float) -> Optional[float]:
        """Find where talking the transcript missed really starts, just before a cut.

        Two things can say so. A measured silence: sound runs up to the cut for
        at least `UNHEARD_SECONDS` from a silence no more than
        `ONSET_REACH_SECONDS` back, and the transcript has no word for it. And
        the second listen, which never read the transcript: a voiced stretch
        that began no more than `ONSET_REACH_SECONDS` back and runs on through
        the cut, or a word heard right up against the first one kept, in the
        same phrase. The second is what answers where the room is never quiet
        enough to measure a silence in, which is where the first edit made
        with it lost 「我們」 and 「這個」 off the front of two lines.

        Args:
            seconds: Where a window opens, in source seconds.

        Returns:
            Where the sound really starts, the earliest either finds, when a
            cut here would take the start of it off. None otherwise, for a cut
            inside a measured silence, and for a file nobody transcribed.
        """
        if not self.transcribed or self._in_silence(seconds) or self.drops_words(seconds, opens=True):
            return None
        found = [onset for onset in (self._silence_onset(seconds), self._voice_onset(seconds)) if onset is not None]
        return min(found) if found else None

    def sound_after(self, seconds: float) -> Optional[float]:
        """Find where talking the transcript missed really stops, just after a cut.

        Args:
            seconds: Where a window closes, in source seconds.

        Returns:
            Where the sound really stops, when a cut here would take the end
            of it off; the mirror of `sound_before`.
        """
        if not self.transcribed or self._in_silence(seconds):
            return None
        found = [fall for fall in (self._silence_fall(seconds), self._voice_fall(seconds)) if fall is not None]
        return max(found) if found else None

    def drops_words(self, seconds: float, opens: bool) -> bool:
        """Say whether a cut opening a window sits on a word edge, taking whole words off the front of a line.

        The transcript knows the word being cut away — it ends where the first
        one kept begins — so the cut was put there to take it off, 「各位」
        off 「各位我到高鐵了」, and the sound running on up to the edge is
        that word, not one the transcript missed. A cut a missed word runs up
        to has no transcribed word right against it, and is still moved to
        the sound. Only the front of a line: a line stopped short of its end
        between two words that close together is bitten off, and is still
        found as one.

        Args:
            seconds: Where the cut is, in source seconds.
            opens: Whether it opens a window rather than closing one.

        Returns:
            True when the cut opens a window inside one sentence, a
            transcribed word starts within `WORD_EDGE_SECONDS` of it, and
            another ends within it, mostly before it and not run on across a
            measured silence.
        """
        if not opens:
            return False
        edge = WORD_EDGE_SECONDS
        kept = any(abs(start - seconds) <= edge for start, _ in self.words)
        gone = any(abs(end - seconds) <= edge and (start + end) / 2 < seconds and self.placed(start, end)
                   for start, end in self.words)
        within = any(begins < seconds < ends for begins, ends in self.sentences)
        return kept and gone and within

    def placed(self, start: float, end: float) -> bool:
        """Say whether a transcribed word is where the transcript put it.

        A word run on across a measured silence is misplaced: what it says is
        heard on one side of the silence, and where is not known.
        """
        return not any(min(end, loud) - max(start, quiet) > EDGE_TOLERANCE_SECONDS for quiet, loud in self.pauses)

    def last_sound_before(self, seconds: float) -> float:
        """Find where the last thing said or heard before a moment ends: what a window may not open onto.

        Args:
            seconds: The moment, in source seconds.

        Returns:
            The latest end at or before it of a transcribed word, a word the
            second listen heard, or a stretch its voice detector heard — or
            the moment itself when any of them runs through it; 0 when
            nothing was.
        """
        if self._sounding(seconds):
            return seconds
        ends = [end for _, end in self.words] + [end for _, end, _ in self.heard] + [end for _, end in self.voiced]
        return max((end for end in ends if end <= seconds + EDGE_TOLERANCE_SECONDS), default=0.0)

    def first_sound_after(self, seconds: float) -> float:
        """Find where the next thing said or heard after a moment starts: what a window may not close onto.

        Args:
            seconds: The moment, in source seconds.

        Returns:
            The mirror of `last_sound_before`: the earliest start at or after
            it, or the moment itself when anything runs through it;
            the end of the file when nothing comes, or infinity when its
            length is not known.
        """
        if self._sounding(seconds):
            return seconds
        starts = ([start for start, _ in self.words] + [start for start, _, _ in self.heard]
                  + [start for start, _ in self.voiced])
        last = self.duration if self.duration is not None else math.inf
        return min((start for start in starts if start >= seconds - EDGE_TOLERANCE_SECONDS), default=last)

    def _sounding(self, seconds: float) -> bool:
        """Say whether a word, a heard word or a voiced stretch runs through a moment, clear of its edges."""
        spans = [*self.words, *((start, end) for start, end, _ in self.heard), *self.voiced]
        return any(start < seconds - EDGE_TOLERANCE_SECONDS and end > seconds + EDGE_TOLERANCE_SECONDS
                   for start, end in spans)

    def room_before(self, onset: float) -> float:
        """Find how far before the start of some talking a window may open.

        Args:
            onset: Where the talking starts, as `sound_before` found it.

        Returns:
            The start of the silence it rises out of; or else the end of the
            last voiced stretch or heard word before it, so the breath left in
            front of a line never takes in the end of the one before; or a
            second before it when nothing is known to be there.
        """
        quiet = [start for start, loud in self.pauses if abs(loud - onset) <= EDGE_TOLERANCE_SECONDS]
        if quiet:
            return max(quiet)
        ended = [end for _, end in self.voiced if end <= onset] + [end for _, end, _ in self.heard if end <= onset]
        return max(ended, default=onset - 1.0)

    def room_after(self, fall: float) -> float:
        """Find how far after the end of some talking a window may close.

        Args:
            fall: Where the talking stops, as `sound_after` found it.

        Returns:
            The mirror of `room_before`, and never past the end of the file.
        """
        loud = [end for start, end in self.pauses if abs(start - fall) <= EDGE_TOLERANCE_SECONDS]
        if loud:
            return min(loud)
        begun = [start for start, _ in self.voiced if start >= fall] + [start for start, _, _ in self.heard if start >= fall]
        room = min(begun, default=fall + 1.0)
        return room if self.duration is None else min(room, self.duration)

    def quiet_on_second_listen(self, seconds: float) -> bool:
        """Say whether the second listen hears nobody at a moment.

        The transcript runs a word on across a pause often enough — 「這」 of
        「這他媽的」 was given 3.09s to 4.43s, over a third of a second of
        nothing — that a cut made in real quiet was taken for one through a
        word.

        Args:
            seconds: The moment, in source seconds.

        Returns:
            True when the file was listened to again and the moment is in no
            voiced stretch, and in no heard word the detector missed.
        """
        return (
            self.listened
            and not self._in_voice(seconds)
            and not any(start < seconds < end and not self._voiced_word(start, end) for start, end, _ in self.heard)
        )

    def _in_voice(self, seconds: float) -> bool:
        """Say whether the voice detector hears somebody at a moment, clear of its wobbly edges."""
        return any(
            start + VOICED_MARGIN_SECONDS < seconds < end - VOICED_TAIL_MARGIN_SECONDS for start, end in self.voiced
        )

    def _voiced_word(self, start: float, end: float) -> bool:
        """Say whether the voice detector heard enough of a heard word to be the one to place it."""
        covered = sum(max(0.0, min(end, until) - max(start, since)) for since, until in self.voiced)
        return covered >= VOICED_SHARE * max(end - start, 1e-6)

    def heard_cut_off(self, seconds: float, opens: bool) -> str:
        """Say which words the second listen heard that a cut takes off.

        Args:
            seconds: Where the cut is, in source seconds.
            opens: Whether it opens a window rather than closing one.

        Returns:
            The words as heard, or an empty string when it takes none off.
        """
        lost = self._lost_words(seconds, opens)
        if (not lost or self.drops_words(seconds, opens)
                or self._known_word(*((lost[0][0], seconds) if opens else (seconds, lost[-1][1])))):
            return ""
        return "".join(text for _, _, text in lost)

    def _voice_onset(self, seconds: float) -> Optional[float]:
        """Find talking the second listen hears running up to a cut that opens a window."""
        found = [
            start for start, end in self.voiced
            if seconds - ONSET_REACH_SECONDS <= start <= seconds - VOICED_MARGIN_SECONDS and end > seconds
        ]
        found += [start for start, _, _ in self._lost_words(seconds, opens=True)[:1]]
        found = [onset for onset in found if not self._known_word(onset, seconds)]
        return min(found) if found else None

    def _voice_fall(self, seconds: float) -> Optional[float]:
        """Find talking the second listen hears running on past a cut that closes a window."""
        found = [
            end for start, end in self.voiced
            if seconds + VOICED_TAIL_MARGIN_SECONDS <= end <= seconds + ONSET_REACH_SECONDS and start < seconds
        ]
        found += [end for _, end, _ in self._lost_words(seconds, opens=False)[-1:]]
        found = [fall for fall in found if not self._known_word(seconds, fall)]
        return max(found) if found else None

    def _lost_words(self, seconds: float, opens: bool) -> List[Tuple[float, float, str]]:
        """Find the words heard on the far side of a cut, in one phrase with the near side.

        Args:
            seconds: Where the cut is, in source seconds.
            opens: Whether it opens a window rather than closing one.

        Returns:
            The words cut off, in order: heard words each within
            `PHRASE_GAP_SECONDS` of the next, reaching no more than
            `ONSET_REACH_SECONDS` from the cut, running into a kept word that
            itself starts (or ends) within `PHRASE_GAP_SECONDS` of the cut.
            None where the detector hears a pause at the cut and hears the
            words either side of it: that is a real gap between two lines,
            whatever the speech model's word edges say.
        """
        lost = self._phrase_across(seconds, opens)
        if lost and not self._in_voice(seconds) and any(self._voiced_word(start, end) for start, end, _ in lost):
            return []
        return lost

    def _phrase_across(self, seconds: float, opens: bool) -> List[Tuple[float, float, str]]:
        """Find the heard words on the far side of a cut that run on from the near side, going by word edges alone."""
        words = sorted(self.heard)
        kept_side = [
            index for index, (start, end, _) in enumerate(words)
            if ((start + end) / 2 >= seconds if opens else (start + end) / 2 <= seconds)
        ]
        if not kept_side:
            return []
        if opens:
            first = kept_side[0]
            if words[first][0] - seconds > PHRASE_GAP_SECONDS:
                return []
            lost: List[Tuple[float, float, str]] = []
            after = words[first]
            for word in reversed(words[:first]):
                if after[0] - word[1] > PHRASE_GAP_SECONDS or seconds - word[0] > ONSET_REACH_SECONDS:
                    break
                lost.insert(0, word)
                after = word
            return lost
        last = kept_side[-1]
        if seconds - words[last][1] > PHRASE_GAP_SECONDS:
            return []
        lost = []
        before = words[last]
        for word in words[last + 1:]:
            if word[0] - before[1] > PHRASE_GAP_SECONDS or word[1] - seconds > ONSET_REACH_SECONDS:
                break
            lost.append(word)
            before = word
        return lost

    def _known_word(self, start: float, end: float) -> bool:
        """Say whether the transcript has a word lying wholly inside a stretch cut away.

        A cut next to a word the transcript knows about was made on purpose —
        a 「然後」 taken off the front of a line — and the second listen
        hearing that word is no reason to put it back. A word that runs on
        across the cut into what is kept does not count: that is the word
        misplaced, which is the case this is for.

        Args:
            start: Where the stretch cut away starts, in source seconds.
            end: Where it ends.

        Returns:
            True when a transcribed word starts and ends inside it.
        """
        tolerance = EDGE_TOLERANCE_SECONDS
        return any(
            opened >= start - tolerance and closed <= end + tolerance and closed - opened > tolerance
            for opened, closed in self.words
        )

    def _silence_onset(self, seconds: float) -> Optional[float]:
        """Find talking the transcript missed between a measured silence and a cut that opens a window."""
        ended = [loud for _, loud in self.pauses if seconds - ONSET_REACH_SECONDS <= loud < seconds]
        if not ended:
            return None
        onset = max(ended)
        quiet = max(start for start, loud in self.pauses if loud == onset)
        if seconds - onset < UNHEARD_SECONDS or self._accounted(onset, seconds, (quiet, onset)):
            return None
        return onset

    def _silence_fall(self, seconds: float) -> Optional[float]:
        """Find talking the transcript missed between a cut that closes a window and a measured silence."""
        begun =[quiet for quiet, _ in self.pauses if seconds < quiet <= seconds + ONSET_REACH_SECONDS]
        if not begun:
            return None
        fall = min(begun)
        loud = min(end for start, end in self.pauses if start == fall)
        if fall - seconds < UNHEARD_SECONDS or self._accounted(seconds, fall, (fall, loud)):
            return None
        return fall

    def _accounted(self, start: float, end: float, silence: Tuple[float, float]) -> bool:
        """Say whether the transcript has a word for the sound in a stretch next to a silence.

        A word that also runs into the silence does not count: it is itself
        misplaced. That is the case this is for — the transcript stretched
        「抽」 from 5.44s across a silence to 6.74s, and it was heard from 6.34s.

        Args:
            start: Where the stretch starts, in source seconds.
            end: Where it ends.
            silence: The measured silence it borders.

        Returns:
            True when a word covers part of the stretch and none of the silence.
        """
        tolerance = EDGE_TOLERANCE_SECONDS
        return any(
            opened < end - tolerance and closed > start + tolerance
            and not (opened < silence[1] - tolerance and closed > silence[0] + tolerance)
            for opened, closed in self.words
        )

    def _in_silence(self, seconds: float) -> bool:
        """Say whether a moment sits in a measured silence."""
        return any(
            quiet - EDGE_TOLERANCE_SECONDS <= seconds <= loud + EDGE_TOLERANCE_SECONDS for quiet, loud in self.pauses
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

def _voices(moments: Sequence[float], turns: Sequence[SpeakerTurn]) -> Tuple[Optional[str], ...]:
    """Say who is speaking at each moment, by the file's speaker turns.

    An hour of talk is thousands of words and turns, so each moment looks only
    at the turns that started late enough to still be running, rather than at
    all of them.

    Args:
        moments: Seconds into the file, one per word.
        turns: The file's speaker turns, in any order, possibly overlapping.

    Returns:
        A voice per moment, None where no turn covers it; empty when there are no turns.
    """
    if not turns:
        return ()
    ordered = sorted(turns, key=lambda turn: turn.start)
    starts = [turn.start for turn in ordered]
    longest = max(turn.end - turn.start for turn in ordered)
    voices: List[Optional[str]] = []
    for moment in moments:
        at = bisect.bisect_right(starts, moment) - 1
        while at >= 0 and starts[at] > moment - longest and moment >= ordered[at].end:
            at -= 1
        voices.append(ordered[at].speaker if at >= 0 and moment < ordered[at].end else None)
    return tuple(voices)

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
    # A sentence the recogniser made up is not somebody talking, and a cut through it bites nothing.
    segments = [segment for segment in (transcript.segments if transcript is not None else [])
                if not was_made_up(segment)]
    listened = analysis.second_listen
    return CleanCuts(
        words=tuple(
            (word.start, word.end)
            for segment in segments
            for word in segment.words
        ),
        sentences=tuple((segment.start, segment.end) for segment in segments),
        said=tuple(segment.text for segment in segments),
        spoken=tuple(word.text for segment in segments for word in segment.words),
        voices=_voices([(word.start + word.end) / 2 for segment in segments for word in segment.words],
                       analysis.speakers),
        pauses=tuple((span.start, span.end) for span in analysis.silences),
        bad_picture=_joined([*analysis.black_frames, *analysis.frozen_frames]),
        transcribed=transcript is not None,
        beats=None if analysis.rhythm is None else tuple(analysis.rhythm.beats),
        energy=tuple(analysis.music.energy) if analysis.music is not None and analysis.music.energy else None,
        loudness=tuple(second.loudness for second in analysis.sound) if analysis.rhythm is not None else None,
        voiced=tuple((span.start, span.end) for span in listened.voiced) if listened else (),
        heard=tuple(
            (word.start, word.end, word.text) for word in listened.words if word.probability >= HEARD_PROBABILITY
        ) if listened else (),
        listened=listened is not None,
        duration=analysis.duration,
    )
