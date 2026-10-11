"""How a song's energy moves: where it is quiet, where it is fullest, how it ends.

Worked out from the loudness the analysis already measures every second, so
it costs nothing more to listen for. Each song is put on its own scale, from
its quiet passages to its fullest, since what matters is where in the song the
energy is, not how loudly it was mastered.
"""

import functools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Seconds the loudness is averaged over, so one drum hit is not a peak. Provisional.
SMOOTH_SECONDS = 3
# The song's own scale: from this percentile of its loudness (0) to this one (1). Provisional.
QUIET_PERCENTILE, FULL_PERCENTILE = 10, 95
# The two lines between quiet, middle and full on that scale. Provisional.
MIDDLE, FULL = 0.35, 0.7
# A stretch shorter than this is part of what is around it, not a section of its own. Provisional.
SECTION_SECONDS = 8
# How long the fullest stretch is looked for over. Provisional.
FULLEST_SECONDS = 8
# A song that falls under this in its last seconds fades out; one still above it ends full. Provisional.
FADED, ENDS_FULL = 0.25, 0.6
# How much of the end is looked at to say how a song ends. Provisional.
ENDING_SECONDS = 6
_LEVELS = ("quiet", "middle", "full")


def energy_curve(loudness: Sequence[float]) -> List[float]:
    """A song's energy, second by second, on its own scale from 0 to 1.

    Args:
        loudness: Its loudness each second, in dBFS.

    Returns:
        One value a second, rounded to two places; empty for no sound.
    """
    if not loudness:
        return []
    level = np.maximum(np.asarray(loudness, dtype=np.float64), -90.0)
    width = min(SMOOTH_SECONDS, len(level))
    smooth = np.convolve(np.pad(level, (width // 2, width - 1 - width // 2), mode="edge"),
                         np.ones(width) / width, mode="valid")
    low, high = np.percentile(smooth, QUIET_PERCENTILE), np.percentile(smooth, FULL_PERCENTILE)
    scaled = np.clip((smooth - low) / max(high - low, 1e-6), 0.0, 1.0)
    return [round(float(value), 2) for value in scaled]


def _level(value: float) -> str:
    return _LEVELS[0] if value < MIDDLE else _LEVELS[1] if value < FULL else _LEVELS[2]


def sections(energy: Sequence[float]) -> List[Dict[str, object]]:
    """The song in stretches of quiet, middle and full, each long enough to count as one.

    Args:
        energy: From `energy_curve`.

    Returns:
        Each with its `start` and `end` in seconds and its `level`.
    """
    runs: List[List] = []
    for second, value in enumerate(energy):
        level = _level(value)
        if runs and runs[-1][2] == level:
            runs[-1][1] = second + 1
        else:
            runs.append([second, second + 1, level])
    # A short run is folded into the longer of its neighbours, shortest first, until none is left.
    while len(runs) > 1:
        shortest = min(range(len(runs)), key=lambda index: runs[index][1] - runs[index][0])
        if runs[shortest][1] - runs[shortest][0] >= SECTION_SECONDS:
            break
        before = runs[shortest - 1] if shortest > 0 else None
        after = runs[shortest + 1] if shortest + 1 < len(runs) else None
        into = before if after is None or (before is not None and before[1] - before[0] >= after[1] - after[0]) else after
        into[0], into[1] = min(into[0], runs[shortest][0]), max(into[1], runs[shortest][1])
        del runs[shortest]
        merged: List[List] = []
        for run in runs:
            if merged and merged[-1][2] == run[2]:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        runs = merged
    return [{"start": start, "end": end, "level": level} for start, end, level in runs]


def fullest(energy: Sequence[float]) -> Optional[Tuple[int, int]]:
    """Where the song is fullest: the `FULLEST_SECONDS` with the most energy, as `(start, end)`."""
    if not energy:
        return None
    width = min(FULLEST_SECONDS, len(energy))
    totals = np.convolve(np.asarray(energy), np.ones(width), mode="valid")
    start = int(np.argmax(totals))
    return start, start + width


def ending(energy: Sequence[float]) -> Optional[str]:
    """How the song ends: `fades out`, `ends quietly`, or `ends full`, cut off or on a hit."""
    if len(energy) < ENDING_SECONDS:
        return None
    last = energy[-ENDING_SECONDS:]
    if max(last[-2:]) < FADED and max(last[:2]) > last[-1] + 0.2:
        return "fades out"
    if max(last[-2:]) >= ENDS_FULL:
        return "ends full"
    return "ends quietly"


def summary(energy: Sequence[float]) -> Dict[str, object]:
    """What the AI reads about a song's energy instead of the curve itself.

    Returns:
        Its `sections`, where it is `fullest`, and how it `ends`.
    """
    found = fullest(energy)
    return {"sections": sections(energy), "fullest": list(found) if found else None, "ends": ending(energy)}


def energy_at(energy: Sequence[float], second: float) -> Optional[float]:
    """The energy at one second of the song, or None past its end."""
    index = int(second)
    return energy[index] if 0 <= index < len(energy) else None


# A loop is whole bars of this many beats, counted from the first beat found: the beat
# tracker finds beats, not which of them begins a bar, so this assumes common time.
# Provisional (roadmap §13).
BAR_BEATS = 4
# The shortest loop: shorter, and the repeat is heard as a stuck record. Provisional
# (roadmap §13).
LOOP_SHORTEST_BARS = 8
# Energy averaged this long either side of a loop's two ends to compare them.
# Provisional (roadmap §13).
LOOP_COMPARE_SECONDS = 2
# Neither end of a loop may sit where the song is quieter than this on its own scale — an
# intro, a breakdown, a fade-out — since a jump there is heard. Provisional (roadmap §13).
LOOP_FLOOR = 0.35


def _mean(energy: Sequence[float], start: float, end: float) -> Optional[float]:
    window = energy[max(0, int(start)):max(0, int(end))]
    return sum(window) / len(window) if window else None


@functools.lru_cache(maxsize=256)
def loop_span(beats: Tuple[float, ...], energy: Tuple[float, ...], duration: Optional[float]) -> Optional[Tuple[float, float]]:
    """Find where a song can loop without the jump being heard.

    Both ends on bar lines of the same beat grid, so the loop is whole bars
    and keeps time; neither in a quiet intro, breakdown or fade-out; and the
    music either side of the jump as alike in energy as the song allows, so
    what comes after the loop's end sounds like what came after its start.
    Of the loops that match about equally well, the longest, so the repeat
    comes round as seldom as it can.

    Args:
        beats: The song's beats, in seconds.
        energy: Its energy each second, from `energy_curve`.
        duration: How long it is.

    Returns:
        `(start, end)` in song seconds: play to `end`, then carry on from
        `start`. None when the song is too short, has too few beats, or has
        no stretch loud enough to loop in.
    """
    if not beats or not energy or duration is None:
        return None
    bars = list(beats[::BAR_BEATS])
    width = LOOP_COMPARE_SECONDS
    best: Optional[Tuple[float, float, float]] = None
    for first, start in enumerate(bars):
        before_start, after_start = _mean(energy, start - width, start), _mean(energy, start, start + width)
        if after_start is None or after_start < LOOP_FLOOR:
            continue
        for end in bars[first + LOOP_SHORTEST_BARS:]:
            if end > duration - width:
                break
            before_end, after_end = _mean(energy, end - width, end), _mean(energy, end, end + width)
            if before_end is None or after_end is None or before_end < LOOP_FLOOR:
                continue
            # After the jump the song carries on from `start` where it would have from `end`;
            # and what led up to `end` is what will have led up to `start`.
            mismatch = abs(after_start - after_end) + abs((before_start or after_start) - before_end)
            # Rounded, so loops that match about equally well are told apart by their length.
            key = (round(mismatch, 1), -(end - start))
            if best is None or key < (round(best[0], 1), -(best[2] - best[1])):
                best = (mismatch, start, end)
    return None if best is None else (round(best[1], 3), round(best[2], 3))
