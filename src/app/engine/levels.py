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
"""

import functools
import math
import re
import subprocess
from typing import Callable, List, Mapping, Optional, Tuple

from app.engine.builder import Talking
from app.engine.delivery import _on_timeline
from app.engine.ffmpeg import hidden_window_flags
from app.models.media import Asset, MediaAnalysis
from app.models.timeline import Clip, Project, TrackType

# The per-second measurements count digital silence as -120; anything this quiet is not
# part of what anybody hears as the level of the talking or of a song.
QUIET_DB = -70.0
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

def talking_in(
    project: Project,
    assets: Mapping[str, Asset],
    analyses: Mapping[str, MediaAnalysis],
    voices: Mapping[str, float],
    song: Callable[[Clip], Optional[float]],
) -> Optional[Talking]:
    """Work out where a cut has somebody talking, and bring each music track to that talking.

    The talking is every transcribed sentence heard from the sequence and
    from the narration tracks. Its level is the per-second loudness measured
    under those sentences, after each clip's volume and each narration's
    gain: what the viewer hears, not what was recorded.

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
    heard: List[Tuple[Clip, float]] = [(clip, 0.0) for clip in (base.clips if base else []) if clip.volume != 0]
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
        low, high = float(clip.source_range.start), float(clip.source_range.end)
        turned = gain + 20 * math.log10(clip.volume)
        for segment in analysis.transcript.segments:
            landed = _on_timeline(clip, segment.start, segment.end)
            if landed is None:
                continue
            spans.append((round(landed[0], 3), round(landed[1], 3)))
            opened, closed = max(segment.start, low), min(segment.end, high)
            spoken += [
                second.loudness + turned
                for second in analysis.sound
                if second.start < closed and second.end > opened and second.loudness > QUIET_DB
            ]

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
    return Talking(spans=tuple(sorted(spans)), music_gains=gains)

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
