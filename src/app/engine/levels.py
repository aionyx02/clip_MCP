"""Where the talking in a cut is, and how loud, so the music can be placed under it.

Music used to duck under the level of the footage's own sound: a compressor
listened to it and pulled the music down once it passed a fixed threshold.
That tied getting out of the way of a voice to how loud the voice happened to
be, and real edits broke it both ways. Voices turned down to match each other
fell under the threshold and the music came back up over them — ten decibels
over, in the first edit it happened to. A person talking to camera quietly
after a narration never crossed it at all.

The transcripts already say where somebody is talking, to the word. So the
music drops by a set amount wherever that is, and its own level is set against
the talking rather than against how loud the song was mastered: a clip's
volume then says how far under the voices it sits.

The footage is placed the same way. Every clip somebody talks in is brought
to one level of talking, measured over its sentences alone, so a line said far
from the camera and one shouted into it come out alike; every clip nobody
talks in — the street, the café, a timelapse — is kept a set distance under
that talking, turned down if it would be louder and left alone if it is
already under. The loudness normalization at the end then only has to move
the whole mix, rather than pumping it up and down to even out clips it cannot
tell apart.
"""

import functools
import math
import re
import subprocess
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from app.engine.builder import Talking
from app.engine.delivery import _on_timeline
from app.engine.ffmpeg import hidden_window_flags
from app.models.media import Asset, MediaAnalysis
from app.models.timeline import Clip, Project, TrackType

# The per-second measurements count digital silence as -120; anything this quiet is not
# part of what anybody hears as the level of the talking or of a song.
QUIET_DB = -70.0
# The level the talking in every clip is brought to, as a per-second RMS level in dBFS.
# Only a reference: the render normalizes the finished mix, so what matters is that every
# clip is brought to the same one. Close to where a narration is put (VOICE_KEY_DB).
TALK_DB = -20.0
# How far under the talking a clip nobody talks in is kept at most. Provisional.
AMBIENCE_UNDER_TALK_DB = 8.0
# The most a clip is turned up or down to reach its level. Past this, a quiet clip is
# mostly its own hiss, and a clip that loud is mostly clipping. Provisional.
LEVEL_RANGE_DB = 12.0
# Seconds of talking a clip needs for its own level to be measured from it. A clip with
# less takes the level of the talking in its whole file. Provisional.
MIN_TALK_SECONDS = 2.0
# How much of a second has to be under a sentence for its level to count as talking.
TALK_SHARE = 0.5
# How much of a song is listened to for its level: songs hold one level, and decoding a
# whole one for each clip laid from it would be most of a sound preview. Provisional.
SONG_SAMPLE_SECONDS = 60.0

def _power_mean(levels: List[float]) -> Optional[float]:
    """Average levels in decibels the way they add up as sound.

    Args:
        levels: Levels in dB.

    Returns:
        The mean power in dB, or None for no levels.
    """
    if not levels:
        return None
    return 10 * math.log10(sum(10 ** (level / 10) for level in levels) / len(levels))

def _talk_seconds(analysis: MediaAnalysis, low: float, high: float) -> List[float]:
    """Read the level of each second somebody is talking in a stretch of a file.

    Args:
        analysis: The file's analysis, transcribed.
        low: Where the stretch starts, in seconds.
        high: Where it ends.

    Returns:
        The level of each measured second at least `TALK_SHARE` under a
        sentence, in dBFS.
    """
    sentences = [
        (max(segment.start, low), min(segment.end, high))
        for segment in analysis.transcript.segments if segment.end > low and segment.start < high
    ]
    return [
        second.loudness for second in analysis.sound
        if second.loudness > QUIET_DB and second.start < high and second.end > low
        and sum(max(0.0, min(end, second.end) - max(start, second.start)) for start, end in sentences)
        >= TALK_SHARE * (second.end - second.start)
    ]

def _clip_gains(
    clips: List[Clip],
    assets: Mapping[str, Asset],
    analyses: Mapping[str, MediaAnalysis],
) -> Dict[str, float]:
    """Work out what to turn each clip of the sequence by, so its sound sits where it belongs.

    Args:
        clips: The sequence's clips.
        assets: The assets they play, by ID.
        analyses: Their analyses, by asset ID.

    Returns:
        A gain in dB per clip ID: the talking in a clip brought to `TALK_DB`,
        and a clip with no talking in it turned down to `AMBIENCE_UNDER_TALK_DB`
        under that if it is louder. A clip kept at its recorded level moves
        with the typical clip, so it stays where it was against the rest. A
        clip with nothing measured is left out.
    """
    gains: Dict[str, float] = {}
    kept: List[str] = []
    for clip in clips:
        asset, analysis = assets.get(clip.asset_id), analyses.get(clip.asset_id)
        if asset is None or not asset.has_audio or analysis is None or not analysis.sound:
            continue
        if clip.keep_level:
            kept.append(clip.id)
            continue
        low, high = float(clip.source_range.start), float(clip.source_range.end)
        talk = _talk_seconds(analysis, low, high) if analysis.transcript is not None else []
        if talk:
            if len(talk) < MIN_TALK_SECONDS:
                talk = _talk_seconds(analysis, 0.0, analysis.duration)
            gain = TALK_DB - _power_mean(talk)
        else:
            heard = _power_mean([
                second.loudness for second in analysis.sound
                if second.loudness > QUIET_DB and second.start < high and second.end > low
            ])
            if heard is None:
                continue
            # Only ever down: a quiet street is already where it belongs, and bringing it up
            # would bring its hiss up with it.
            gain = min(0.0, TALK_DB - AMBIENCE_UNDER_TALK_DB - heard)
        gains[clip.id] = round(max(-LEVEL_RANGE_DB, min(LEVEL_RANGE_DB, gain)), 2)
    moved = sorted(gains.values())
    for clip_id in kept:
        gains[clip_id] = moved[len(moved) // 2] if moved else 0.0
    return gains

def talking_in(
    project: Project,
    assets: Mapping[str, Asset],
    analyses: Mapping[str, MediaAnalysis],
    voices: Mapping[str, float],
    song: Callable[[Clip], Optional[float]],
) -> Optional[Talking]:
    """Work out where a cut has somebody talking, and place everything else against it.

    The talking is every transcribed sentence heard from the sequence and
    from the narration tracks. The sequence's clips are brought to it (see
    `_clip_gains`), and then its level is the per-second loudness measured
    under those sentences, after each clip's gain and volume and each
    narration's gain: what the viewer hears, not what was recorded. Each music
    track is brought to that level.

    Args:
        project: The project.
        assets: The assets it plays, by ID.
        analyses: Their analyses, by asset ID.
        voices: The narration tracks, with the gain in dB each is heard at.
        song: How loud a music clip's song is, in dB before its volume, or
            None when it cannot be measured.

    Returns:
        The talking, or None when nothing heard was transcribed, so where the
        talking is cannot be known. Among transcribed footage, a clip whose
        words were never asked for counts as nobody talking. A music track
        whose songs cannot be measured, or any track when nothing is said,
        gets no gain.
    """
    base = project.base_video_track
    sequence = [clip for clip in (base.clips if base else []) if clip.volume != 0]
    clip_gains = _clip_gains(sequence, assets, analyses)
    heard: List[Tuple[Clip, float]] = [(clip, clip_gains.get(clip.id, 0.0)) for clip in sequence]
    heard += [
        (clip, voices[track.id])
        for track in project.tracks if track.id in voices
        for clip in track.clips if clip.volume != 0
    ]
    spans: List[Tuple[float, float]] = []
    spoken: List[float] = []
    transcribed = False
    for clip, gain in heard:
        asset = assets.get(clip.asset_id)
        if asset is None or not asset.has_audio:
            continue
        analysis = analyses.get(clip.asset_id)
        if analysis is None or analysis.transcript is None:
            # Footage analyzed without its words is the cutaways — a street, a timelapse,
            # hands at work — which is why nobody asked for them. Counted as nobody talking.
            continue
        transcribed = True
        turned = gain + 20 * math.log10(clip.volume)
        for segment in analysis.transcript.segments:
            landed = _on_timeline(clip, segment.start, segment.end)
            if landed is not None:
                spans.append((round(landed[0], 3), round(landed[1], 3)))
        low, high = float(clip.source_range.start), float(clip.source_range.end)
        spoken += [level + turned for level in _talk_seconds(analysis, low, high)]

    if not transcribed:
        return None
    talking_level = _power_mean(spoken)
    gains = {}
    if talking_level is not None:
        for track in project.tracks:
            if track.track_type != TrackType.AUDIO or track.id in voices:
                continue
            levels = [level for clip in track.clips if (level := song(clip)) is not None]
            played = _power_mean(levels)
            if played is not None:
                gains[track.id] = round(talking_level - played, 2)
    return Talking(spans=tuple(sorted(spans)), music_gains=gains, clip_gains=clip_gains)

@functools.lru_cache(maxsize=256)
def _mean_volume(path: str, start: float, seconds: float) -> Optional[float]:
    """Measure the average level of a stretch of a file's sound.

    Args:
        path: The file.
        start: Where to start, in seconds.
        seconds: How long to listen.

    Returns:
        The mean level in dBFS, or None when it cannot be read or is silent.
    """
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", path,
         "-vn", "-af", "volumedetect", "-f", "null", "-"],
        stdin=subprocess.DEVNULL, capture_output=True, creationflags=hidden_window_flags(),
    )
    found = re.findall(r"mean_volume:\s+(-?[\d.]+) dB", result.stderr.decode("utf-8", errors="replace"))
    if result.returncode != 0 or not found or float(found[-1]) <= QUIET_DB:
        return None
    return float(found[-1])

def song_level(path: str, clip: Clip) -> Optional[float]:
    """Measure how loud the song under a music clip is, before the clip's volume.

    Args:
        path: The song's file.
        clip: The music clip.

    Returns:
        Its mean level in dBFS over the stretch the clip plays, up to
        `SONG_SAMPLE_SECONDS` of it; None when it cannot be measured.
    """
    start = float(clip.source_range.start)
    return _mean_volume(path, start, min(float(clip.source_range.end) - start, SONG_SAMPLE_SECONDS))
