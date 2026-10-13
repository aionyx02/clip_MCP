"""Photos in a video: which files are photos, how long one is shown, and how it moves while it is.

A photo has no length of its own. In the library it can be shown for as
long as anybody likes; a clip of it runs for as long as its source range
says, which starts at the default here. While it is on screen it moves a
little — a slow push in unless told otherwise — because a picture held
perfectly still in a video reads as the video having stopped.
"""

import os
from decimal import Decimal
from typing import Optional, Tuple

# What a photo is, by its extension: what FFmpeg reads as a single picture.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
# How long a clip of a photo may run: a photo is as long as it is shown, and this stands in
# for "as long as anybody would want" wherever a file has to have a length.
STILL_LONGEST = Decimal(3600)
# Provisional (roadmap §13): how long a photo is shown when nothing says otherwise.
STILL_SECONDS = 4.0
# Provisional (roadmap §13): how much closer a photo ends up than it starts, over the
# whole clip — slow enough to read as a look, not a zoom.
MOTION_AMOUNT = 0.08
# A photo that moves when nothing says how.
DEFAULT_MOTION = "push"


def is_image(path: str) -> bool:
    """Say whether a file is a photo rather than a video or a song.

    Args:
        path: The file.

    Returns:
        True for the picture formats in `IMAGE_EXTENSIONS`.
    """
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def motion_of(motion: Optional[str]) -> str:
    """The motion a photo clip has: its own, or the default slow push."""
    return motion or DEFAULT_MOTION


def motion_at(motion: Optional[str], progress: float) -> Tuple[float, float]:
    """Say how close in a photo is, and where across it the view is, part of the way through its clip.

    Args:
        motion: The clip's motion, or None for the default.
        progress: How far through the clip, 0 to 1.

    Returns:
        `(zoom, across)`: how much closer than filling the frame, and where
        the view's centre sits across the room the zoom leaves, 0 the left
        edge, 0.5 the middle and 1 the right.
    """
    moved = max(0.0, min(1.0, progress))
    kind = motion_of(motion)
    if kind == "push":
        return 1 + MOTION_AMOUNT * moved, 0.5
    if kind == "pull":
        return 1 + MOTION_AMOUNT * (1 - moved), 0.5
    if kind == "pan_left":
        return 1 + MOTION_AMOUNT, moved
    if kind == "pan_right":
        return 1 + MOTION_AMOUNT, 1 - moved
    return 1.0, 0.5


def zoompan(motion: Optional[str], frames: int, width: int, height: int, rate: str) -> str:
    """Write the FFmpeg filter that moves a photo over a clip's frames.

    The same arithmetic as `motion_at`, frame by frame: `on` is the frame
    being made, out of `frames`.

    Args:
        motion: The clip's motion, or None for the default.
        frames: How many frames the clip has.
        width: Width of the picture it makes.
        height: Height of the picture it makes.
        rate: The output frame rate, as FFmpeg writes it.

    Returns:
        The `zoompan` filter, with no commas inside its expressions.
    """
    kind = motion_of(motion)
    progress = f"on/{max(1, frames - 1)}"
    zoom = {
        "push": f"1+{MOTION_AMOUNT}*{progress}",
        "pull": f"{1 + MOTION_AMOUNT}-{MOTION_AMOUNT}*{progress}",
        "pan_left": f"{1 + MOTION_AMOUNT}",
        "pan_right": f"{1 + MOTION_AMOUNT}",
    }.get(kind, "1")
    across = {"pan_left": progress, "pan_right": f"(1-{progress})"}.get(kind, "0.5")
    return (f"zoompan=z={zoom}:x=(iw-iw/zoom)*{across}:y=(ih-ih/zoom)/2"
            f":d={frames}:s={width}x{height}:fps={rate}")
