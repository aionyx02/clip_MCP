"""What a shot looks like, as a fingerprint two shots can be compared by.

Two shots from different files can look so alike that a viewer cannot tell
which is which — the timelapse of the print that failed and the one of the
print that worked. Nothing about the files says so: they are different files,
shot at different times. What says so is the picture, so a few frames of each
are reduced to a difference hash: the frame shrunk to a 9x8 grey grid, and one
bit for each pair of neighbours in a row, set where the left is brighter. Two
pictures that look alike differ in few bits, whatever their size or encoding.
"""

from typing import Sequence, Tuple

from PIL import Image

from app.engine.frames import extract_frame

# Where in a clip its frames are taken, as shares of its length: one frame can catch the
# moment two otherwise different shots pass through the same picture. Provisional: three is
# what one decode each could afford, not a count anything was measured against.
SAMPLES = (0.25, 0.5, 0.75)
# Two frames this many bits apart of 64 look alike. Provisional: what a viewer would
# not tell apart has not been measured against anyone looking.
ALIKE_BITS = 10

def fingerprint(image: Image.Image) -> int:
    """Reduce a picture to its 64-bit difference hash."""
    grey = image.convert("L").resize((9, 8), Image.Resampling.BOX)
    pixels = list(grey.getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return bits

def look_of(path: str, seconds: float, ffmpeg_bin: str = "ffmpeg") -> int:
    """Fingerprint the frame of a file at a moment."""
    return fingerprint(extract_frame(path, seconds, 64, ffmpeg_bin))

def apart(first: int, second: int) -> int:
    """How many bits two fingerprints differ in."""
    return bin(first ^ second).count("1")

def alike(first: Sequence[int], second: Sequence[int]) -> Tuple[bool, int]:
    """Say whether two shots look alike, frame for frame at the same share of each.

    Args:
        first: One shot's fingerprints, at `SAMPLES`.
        second: The other's.

    Returns:
        `(alike, bits)`: whether most of their frames look alike, and how many
        bits the middle pair of the differences is.
    """
    distances = sorted(apart(one, other) for one, other in zip(first, second))
    middle = distances[len(distances) // 2]
    return middle <= ALIKE_BITS, middle
