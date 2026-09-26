"""Seeing what a cut sounds like, for whoever cannot listen to it.

A storyboard shows the rhythm of a cut and none of its sound, and the model
planning an edit cannot hear at all. Whether the music gets out of the way of
the talking, where a song changes, whether one part is much louder than the
next — all of that is in the sound and nowhere else. So the sound is rendered
on its own, in two halves kept apart, and measured: what the footage says,
and what the audio tracks play once ducking has had its way with them.

The numbers are levels, not verdicts. How far under the voice the music
should sit is a matter of taste and of the music, and a corpus would be
needed to put a number on it; what is measured here is how far under it
actually sits.
"""

import wave
from io import BytesIO
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models.timeline import Project

# Levels are read in windows this long: short enough to see a word, long enough not to
# see every syllable.
WINDOW_SECONDS = 0.1
# The quietest level drawn or counted; below it is silence as far as anyone listening
# is concerned.
FLOOR_DB = -60.0
# A window counts as somebody talking when the voice is within this much of its own loud
# end (its 95th percentile). Relative rather than absolute, because the stems are
# measured before the render's loudness normalization, so their absolute level is not
# what the viewer will hear. Provisional.
SPEAKING_BELOW_PEAK_DB = 20.0
# A part has somebody talking in it only with at least this much talking. Less is the
# edge of a line in the next part, or a cough, and a median over it describes nothing.
# Provisional.
SPEAKING_MIN_SECONDS = 0.5
CHART_WIDTH = 1400
CHART_HEIGHT = 360
VOICE_COLOUR = (90, 200, 120)
MUSIC_COLOUR = (240, 160, 60)

def levels(path: str) -> np.ndarray:
    """Measure a mono WAV's level window by window.

    Args:
        path: The file.

    Returns:
        RMS level per `WINDOW_SECONDS`, in dB relative to full scale, never
        below `FLOOR_DB`.
    """
    with wave.open(path, "rb") as file:
        rate = file.getframerate()
        samples = np.frombuffer(file.readframes(file.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
    size = max(1, int(round(rate * WINDOW_SECONDS)))
    count = len(samples) // size
    if count == 0:
        return np.full(0, FLOOR_DB)
    windows = samples[: count * size].reshape(count, size)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))
    return np.maximum(FLOOR_DB, 20 * np.log10(np.maximum(rms, 1e-9)))

@dataclass(frozen=True)
class PartSound:
    """How one part of a cut sounds.

    Attributes:
        name: The part.
        start: Where it starts, in seconds.
        end: Where it ends.
        voice: Median level of the voice while somebody is talking, or None
            when nobody talks in it.
        music_under_voice: Median level of the music while somebody talks.
        music_alone: Median level of the music in the gaps between.
    """

    name: str
    start: float
    end: float
    voice: Optional[float]
    music_under_voice: Optional[float]
    music_alone: Optional[float]

def _median(values: np.ndarray) -> Optional[float]:
    """Take a median of levels, ignoring silence.

    Args:
        values: Levels in dB.

    Returns:
        The median of those above the floor, to one place, or None when
        there are none.
    """
    heard = values[values > FLOOR_DB]
    return round(float(np.median(heard)), 1) if len(heard) else None

def parts_of(project: Project, voice: np.ndarray, music: np.ndarray) -> List[PartSound]:
    """Sum up the sound of each part of a cut.

    Args:
        project: The project; its markers are the parts, or the whole cut is
            one part when it has none.
        voice: The voice's levels, window by window.
        music: The music's levels, window by window.

    Returns:
        One summary per part, in order.
    """
    duration = float(project.duration)
    marks = sorted(project.markers, key=lambda marker: marker.timeline_in)
    bounds = [(float(marker.timeline_in), marker.name) for marker in marks] or [(0.0, "the whole cut")]
    length = min(len(voice), len(music))
    voice, music = voice[:length], music[:length]
    loud_end = np.percentile(voice, 95) if length else FLOOR_DB
    speaking = (voice > loud_end - SPEAKING_BELOW_PEAK_DB) & (voice > FLOOR_DB)
    summaries: List[PartSound] = []
    for index, (start, name) in enumerate(bounds):
        end = bounds[index + 1][0] if index + 1 < len(bounds) else duration
        # Rounded, not truncated: 3 / 0.1 is 30.000000000000004 in floating point.
        window = slice(int(round(start / WINDOW_SECONDS)), int(round(end / WINDOW_SECONDS)))
        talking = speaking[window]
        if talking.sum() * WINDOW_SECONDS < SPEAKING_MIN_SECONDS:
            talking = np.zeros_like(talking)
        summaries.append(PartSound(
            name=name, start=start, end=end,
            voice=_median(voice[window][talking]),
            music_under_voice=_median(music[window][talking]),
            music_alone=_median(music[window][~talking]),
        ))
    return summaries

def describe(parts: Sequence[PartSound]) -> List[str]:
    """Put the parts' sound into sentences.

    Args:
        parts: The summaries.

    Returns:
        One line per part.
    """
    lines = []
    for part in parts:
        said = [f"{part.name} ({part.start:.1f}–{part.end:.1f}s):"]
        if part.voice is None:
            said.append("nobody talks;")
        else:
            said.append(f"voice {part.voice:g} dB;")
        if part.music_under_voice is None and part.music_alone is None:
            said.append("no music")
        else:
            if part.music_under_voice is not None and part.voice is not None:
                said.append(
                    f"music {part.music_under_voice:g} dB under the talking "
                    f"({part.voice - part.music_under_voice:.0f} dB below the voice)"
                )
            if part.music_alone is not None:
                said.append(f"and {part.music_alone:g} dB in the gaps")
                if part.music_under_voice is not None:
                    said.append(f"(it drops {part.music_alone - part.music_under_voice:.0f} dB for the voice)")
        lines.append(" ".join(said))
    return lines

def chart(project: Project, voice: np.ndarray, music: np.ndarray) -> bytes:
    """Draw the two levels along the cut, with its cuts and parts marked.

    Args:
        project: The project.
        voice: The voice's levels, window by window.
        music: The music's levels, window by window.

    Returns:
        The chart as a PNG.
    """
    width, height = CHART_WIDTH, CHART_HEIGHT
    left, right, top, bottom = 50, 20, 30, 40
    image = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    duration = max(float(project.duration), WINDOW_SECONDS)
    x_of = lambda seconds: left + (width - left - right) * seconds / duration
    y_of = lambda level: top + (height - top - bottom) * (level / FLOOR_DB)

    for level in (0, -20, -40, -60):
        draw.line((left, y_of(level), width - right, y_of(level)), fill=(60, 60, 60))
        draw.text((6, y_of(level) - 8), f"{level}", fill=(160, 160, 160), font=font)
    base = project.base_video_track
    for clip in (base.clips if base else [])[1:]:
        draw.line((x_of(float(clip.timeline_in)), top, x_of(float(clip.timeline_in)), height - bottom),
                  fill=(80, 80, 80))
    for marker in project.markers:
        x = x_of(float(marker.timeline_in))
        draw.line((x, 10, x, height - bottom), fill=(220, 220, 220))
        draw.text((x + 3, 8), marker.name, fill=(255, 230, 0), font=font)
    for levels_, colour in ((music, MUSIC_COLOUR), (voice, VOICE_COLOUR)):
        points = [(x_of((index + 0.5) * WINDOW_SECONDS), y_of(level)) for index, level in enumerate(levels_)]
        if len(points) > 1:
            draw.line(points, fill=colour, width=2)
    step = max(1, int(round(duration / 10)))
    for seconds in range(0, int(duration) + 1, step):
        draw.text((x_of(seconds) - 8, height - bottom + 8), f"{seconds // 60}:{seconds % 60:02d}",
                  fill=(160, 160, 160), font=font)
    draw.text((width - 260, height - 22), "green: voice   orange: music", fill=(200, 200, 200), font=font)

    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()
