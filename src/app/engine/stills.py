"""Photos in a video: which files are photos, how long one is shown, and how it moves while it is.

A photo has no length of its own. In the library it can be shown for as
long as anybody likes; a clip of it runs for as long as its source range
says, which starts at the default here. While it is on screen it moves a
little — a slow push in unless told otherwise — because a picture held
perfectly still in a video reads as the video having stopped. How the move
is drawn, the same way for photos and video, is in `moves`.
"""

import os
from decimal import Decimal
from typing import Optional

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


def held(frames: int, width: int, height: int, rate: str) -> str:
    """Write the FFmpeg filter that holds one picture still for a clip's frames.

    How a photo told not to move, and a title card's background, are made
    into video: the picture is brought down from the twice-size it was placed
    at to the box, for as many frames as the clip has. A photo that moves is
    moved by `moves.perspective` instead.

    Args:
        frames: How many frames the clip has.
        width: Width of the picture it makes.
        height: Height of the picture it makes.
        rate: The output frame rate, as FFmpeg writes it.

    Returns:
        The `zoompan` filter, with no commas inside its expressions.
    """
    return f"zoompan=z=1:x=(iw-iw/zoom)*0.5:y=(ih-ih/zoom)/2:d={frames}:s={width}x{height}:fps={rate}"
