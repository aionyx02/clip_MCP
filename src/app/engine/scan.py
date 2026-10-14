"""Every frame of a finished render, looked at for what a look at a few frames would miss.

A render is looked at through a handful of frames a second, and a flaw two or
three frames long falls between them: a moment of black where two pieces meet,
a frame of the wrong shot, a picture that stops while the sound goes on. The
render is read once more, every frame of it, and where it goes black, stops,
or jumps in brightness and straight back is written down with its times. What
was meant — a fade, a card, a photo held still — is for whoever reads the
spots to say; they know the cut, and this only knows the picture.
"""

import json
import os
import re
from typing import Dict, List, Sequence, Tuple

from app.engine.analysis import _frame_records
from app.engine.ffmpeg import escape_filter_path

SPOTS_FILE_NAME = "scan.json"
LEVELS_FILE_NAME = "scan-levels.txt"
LOG_FILE_NAME = "scan.log"
# A frame this close to black, on 0 to 1, is black, as the analysis counts it.
BLACK_PIXEL = 0.10
# A picture that does not change for this long has stopped. Provisional: shorter than a
# held photo or a card, longer than the frame or two a speed change repeats.
FREEZE_SECONDS = 0.25
# A run of at most this many frames that jumps this far in brightness, on 0 to 1, from the
# frame before it and comes back to within half of it after: a flash. Provisional.
FLASH_FRAMES = 3
FLASH_LEVEL = 0.25
# Brightness is read off a small copy: a flash is the whole picture, not its detail.
LEVEL_WIDTH = 160

def command(ffmpeg_bin: str, path: str, work_dir: str) -> List[str]:
    """Write the FFmpeg command that reads every frame of a render.

    Args:
        ffmpeg_bin: The FFmpeg executable.
        path: The render.
        work_dir: Where the brightness of each frame is written.

    Returns:
        The command, its log to be read by `spots`.
    """
    levels = escape_filter_path(os.path.join(work_dir, LEVELS_FILE_NAME))
    graph = (
        "[0:V:0]split=2[detect][level];"
        f"[detect]blackdetect=d=0:pix_th={BLACK_PIXEL},freezedetect=n=-60dB:d={FREEZE_SECONDS}[seen];"
        f"[level]scale={LEVEL_WIDTH}:-2:out_range=full,signalstats,"
        f"metadata=mode=print:key=lavfi.signalstats.YAVG:file='{levels}'[levels]"
    )
    return [ffmpeg_bin, "-hide_banner", "-nostdin", "-loglevel", "info", "-i", path, "-an",
            "-filter_complex", graph, "-map", "[seen]", "-f", "null", "-", "-map", "[levels]", "-f", "null", "-"]

def _flashes(levels: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Find the runs of a few frames that jump in brightness and come straight back."""
    found: List[Tuple[float, float]] = []
    index = 1
    while index < len(levels) - 1:
        before = levels[index - 1][1]
        for length in range(1, FLASH_FRAMES + 1):
            if index + length >= len(levels):
                break
            run = [level for _, level in levels[index:index + length]]
            after = levels[index + length][1]
            jumps = [level - before for level in run]
            if (all(abs(jump) >= FLASH_LEVEL for jump in jumps) and (all(jump > 0 for jump in jumps)
                                                                     or all(jump < 0 for jump in jumps))
                    and abs(after - before) <= FLASH_LEVEL / 2):
                found.append((levels[index][0], levels[index + length][0]))
                index += length
                break
        index += 1
    return found

def _paired(log: str, start: str, end: str, duration: float) -> List[Tuple[float, float]]:
    """Pair a detector's start and end events, an unended one running to the end."""
    starts = [float(found) for found in re.findall(start, log)]
    ends = [float(found) for found in re.findall(end, log)]
    return [(begun, ends[number] if number < len(ends) else duration) for number, begun in enumerate(starts)]

def spots(work_dir: str, duration: float) -> List[Dict[str, object]]:
    """Read what the scan found, from its log and the brightness it wrote.

    Args:
        work_dir: Where the scan wrote.
        duration: Length of the render in seconds.

    Returns:
        `{kind, start, end}` each, `kind` `black`, `frozen` or `flash`, in
        time order.
    """
    try:
        with open(os.path.join(work_dir, LOG_FILE_NAME), encoding="utf-8", errors="replace") as handle:
            log = handle.read()
    except OSError:
        log = ""
    found = [("black", float(start), float(end))
             for start, end in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)]
    # Black does not move either: a stop inside black is the black, said once.
    found += [("frozen", start, end) for start, end in
              _paired(log, r"freeze_start: ([\d.]+)", r"freeze_end: ([\d.]+)", duration)
              if not any(kind == "black" and low <= start + 0.001 and end <= high + 0.001 for kind, low, high in found)]
    levels = [(seconds, values["lavfi.signalstats.YAVG"] / 255)
              for seconds, values in _frame_records(os.path.join(work_dir, LEVELS_FILE_NAME))
              if "lavfi.signalstats.YAVG" in values]
    found += [("flash", start, end) for start, end in _flashes(levels)]
    return [{"kind": kind, "start": round(start, 3), "end": round(end, 3)} for kind, start, end in sorted(
        found, key=lambda spot: spot[1])]

def keep(work_dir: str, duration: float) -> None:
    """Write what the scan found next to the render, and clear away what it was read from."""
    found = spots(work_dir, duration)
    with open(os.path.join(work_dir, SPOTS_FILE_NAME), "w", encoding="utf-8") as handle:
        json.dump(found, handle)
    for name in (LEVELS_FILE_NAME, LOG_FILE_NAME):
        if os.path.exists(os.path.join(work_dir, name)):
            os.remove(os.path.join(work_dir, name))

def kept(work_dir: str) -> List[Dict[str, object]]:
    """Read what a render's scan found, or nothing when it was not scanned."""
    try:
        with open(os.path.join(work_dir, SPOTS_FILE_NAME), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return []
