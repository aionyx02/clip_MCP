"""Bringing a finished mix to one loudness, measured rather than estimated.

A render used to end in FFmpeg's single-pass `loudnorm` and nothing else.
`loudnorm` earns its place — it evens out clips recorded at very different
levels, which is most of what a vlog is — but it gives up on loudness
whenever reaching it would break its peak ceiling, and real edits are exactly
that case: a cup set down, a laugh, a door, twenty decibels over the talking.
Every one of the first eleven edits made with this server came out between
-15.5 and -16.7 LUFS against a -14 target, with true peaks up to +0.3 dBFS.

So `loudnorm` stays, and what it leaves short is made up after it: the mix is
rendered once on its own and measured through the chain, the gain that closes
the gap is raised by whatever the peak limiter at the end takes back, and
measured again until it lands. The render then plays the mix through the same
chain at that gain. The chain is written once, in `chain`, so what is measured
here and what is rendered cannot drift apart.
"""

import math
import re
import subprocess
from typing import List, Tuple

from app.engine.ffmpeg import hidden_window_flags

# Stands in the render's filter graph for the gain until it has been measured.
GAIN_TOKEN = "@LOUDNESS_GAIN@"
# What `loudnorm` is asked for besides the loudness.
TRUE_PEAK_DB = -1.5
LOUDNESS_RANGE = 11.0
# The limiter's ceiling, in dBFS, sampled four times over so that it holds between samples
# as well. Lower than the -1 dBTP streaming platforms ask for, because AAC encoding puts
# peaks back up: at -3 two edits measured -1.8 and -0.9 dBTP once encoded. Provisional.
CEILING_DB = -3.5
# How close to the target counts as there, in LU, and how many rounds it may take.
# Measured on the first edit: 0, 2.4, 3.3 and 3.6 dB after loudnorm gave -16.4, -14.9,
# -14.3 and -14.2 LUFS. Provisional.
TOLERANCE_LU = 0.3
ROUNDS = 6
# Quieter than this, a mix is silence: there is nothing to bring up.
SILENT_LUFS = -69.0
# Never more gain than this after loudnorm, whatever the mix measures. Provisional.
MAX_GAIN_DB = 12.0

def chain(target: float, gain: str) -> str:
    """Write the filters a mix is played through to reach its loudness.

    Args:
        target: The loudness asked for, in LUFS.
        gain: The gain in dB that makes up what `loudnorm` leaves short, or
            `GAIN_TOKEN` while it is still to be found.

    Returns:
        A filter chain: `loudnorm`, the gain, then a peak limiter at four
        times the sample rate, then back to 48 kHz.
    """
    return (
        f"loudnorm=I={target:g}:TP={TRUE_PEAK_DB:g}:LRA={LOUDNESS_RANGE:g},"
        f"volume={gain}dB,aresample=192000,"
        f"alimiter=limit={10 ** (CEILING_DB / 20):.4f}:attack=1:release=60:level=false,aresample=48000"
    )

def measure(path: str, filters: str = "", ffmpeg_bin: str = "ffmpeg") -> Tuple[float, float]:
    """Measure the integrated loudness and the true peak of a sound file.

    Args:
        path: The file.
        filters: Filters to play it through first, if any.
        ffmpeg_bin: FFmpeg executable.

    Returns:
        `(lufs, peak_dbfs)`.

    Raises:
        RuntimeError: If FFmpeg cannot read it.
    """
    graph = f"{filters},ebur128=peak=true" if filters else "ebur128=peak=true"
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-nostats", "-i", path, "-af", graph, "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=hidden_window_flags(),
    )
    loud = re.findall(r"I:\s+(-?[\d.]+|-inf) LUFS", result.stderr)
    peak = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", result.stderr)
    if result.returncode != 0 or not loud or not peak:
        raise RuntimeError(result.stderr.strip()[-500:] or "ffmpeg could not measure the mix")
    return float(loud[-1]), float(peak[-1])

def settle_gain(mix_path: str, target: float, ffmpeg_bin: str = "ffmpeg") -> float:
    """Find the gain that brings a mix to a loudness through `chain`.

    Args:
        mix_path: The mix, rendered on its own and not yet normalized.
        target: Integrated loudness to reach, in LUFS.
        ffmpeg_bin: FFmpeg executable.

    Returns:
        The gain in dB after `loudnorm`, rounded to a hundredth. Zero for a
        silent mix.
    """
    loudness, _ = measure(mix_path, ffmpeg_bin=ffmpeg_bin)
    if not math.isfinite(loudness) or loudness <= SILENT_LUFS:
        return 0.0
    gain = 0.0
    for _ in range(ROUNDS):
        reached, _ = measure(mix_path, chain(target, f"{gain:.2f}"), ffmpeg_bin)
        if abs(reached - target) <= TOLERANCE_LU:
            break
        gain = min(gain + target - reached, MAX_GAIN_DB)
    return round(gain, 2)

def with_gain(command: List[str], gain: float) -> List[str]:
    """Put a settled gain into a command built with `GAIN_TOKEN` in its graph.

    Args:
        command: FFmpeg arguments.
        gain: The gain in dB.

    Returns:
        The command with the gain written in.
    """
    return [argument.replace(GAIN_TOKEN, f"{gain:.2f}") for argument in command]
