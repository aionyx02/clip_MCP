"""The mechanical baseline: every file in order, silences taken out, no judgement about what is said.

    python benchmark/baseline.py <output.mp4> <seconds> <file> [<file> ...]

What a cut with no understanding of the footage looks like. If the AI's film
is no better than this, understanding the footage solved nothing.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# Provisional: what counts as silence for this baseline, the usual defaults of a silence-cutting tool.
SILENCE_DB = -30
SILENCE_SECONDS = 0.5
MIN_PIECE_SECONDS = 0.5


def duration_of(path: str) -> float:
    """The length of a file in seconds."""
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True, check=True).stdout.strip())


def has_sound(path: str) -> bool:
    """Whether a file has a sound stream."""
    return bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip())


def sounding(path: str) -> List[Tuple[float, float]]:
    """The stretches of a file between its silences; all of it when it has no sound."""
    length = duration_of(path)
    if not has_sound(path):
        return [(0.0, length)]
    said = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af",
                           f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_SECONDS}", "-f", "null", "-"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    starts = [float(found) for found in re.findall(r"silence_start: (-?[\d.]+)", said)]
    ends = [float(found) for found in re.findall(r"silence_end: (-?[\d.]+)", said)]
    pieces, at = [], 0.0
    for start, end in zip(starts, ends + [length]):
        if start - at >= MIN_PIECE_SECONDS:
            pieces.append((at, start))
        at = end
    if length - at >= MIN_PIECE_SECONDS:
        pieces.append((at, length))
    return pieces


def main() -> None:
    output, seconds, files = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
    with tempfile.TemporaryDirectory() as folder:
        parts: List[str] = []
        left = seconds
        for path in files:
            for start, end in sounding(path):
                if left <= 0:
                    break
                length = min(end - start, left)
                left -= length
                part = str(Path(folder) / f"{len(parts):04d}.mp4")
                subprocess.run([
                    "ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", path,
                    "-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1,"
                           "fps=30,format=yuv420p",
                    "-map", "0:v:0", "-map", "0:a:0?" if has_sound(path) else "1:a:0",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-ar", "48000", "-ac", "2",
                    "-shortest", part], check=True)
                parts.append(part)
        listing = Path(folder) / "parts.txt"
        listing.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", output], check=True)


if __name__ == "__main__":
    main()
