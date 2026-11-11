"""Slow moves over a clip's picture: a push in, a pull back, or a pan across.

A photo always moves a little, because a picture held perfectly still in a
video reads as the video having stopped (see `stills`). A video clip moves
only when it is told to: a slow push in on somebody talking is the camera
move an editor adds when there was one camera and it never moved, and it is
a choice, not something every shot should do.

Both move by the same amount over their clip, `stills.MOTION_AMOUNT`, so a
photo and a talking shot cut together feel like one camera. The amount is
fixed per clip rather than per second: over a long clip the same push is
gentler, which is how a push reads anyway — a slow drift, never a zoom.

A push closes in on a point of the picture that stays where it is on the
screen while the frame closes in around it: the face, when one was seen,
and the middle otherwise. Pushing on the middle of a shot whose speaker
sits in its upper third drifts the speaker towards the top edge as the
frame closes; holding the face still is what pushing in on somebody means.

The move is linear. An ease in and out reads as a camera starting and
stopping, which a shot that is cut into and out of never shows: its first
and last frames are the cut. A steady drift also stays the same from one
half of a split clip to the next, so a jump cut that alternates a push with
a held shot reads as two framings rather than as a move that stalled.

Photos and video are both moved with FFmpeg's `perspective` filter, which
samples the moved window per frame with subpixel accuracy. `zoompan` rounds
the window to whole pixels, which on a slow push shows as the picture
stepping, even backwards: measured on a 5 s 1080x1920 push from 1.00 to
1.08, `zoompan` jittered 0.59 px RMS at the output size and 0.30 px at twice
it, and `perspective` 0.05 px with every step forwards, at 2.1 s of
rendering against 1.8 s for the shot held still. Through the render itself,
tracking a dot on that push: video 0.025 px with no step backwards; a photo
0.021 px, where the twice-size `zoompan` photos used to go through gave
0.39 px and stepped backwards on 131 of 149 frames.
"""

from fractions import Fraction
from typing import Optional, Tuple

from app.engine.stills import MOTION_AMOUNT, motion_of as photo_motion_of
from app.models.timeline.parts import MIN_MOVE_SECONDS, Clip

# The moves there are; `none` is no move at all.
MOTIONS = ("push", "pull", "pan_left", "pan_right")
# The point a push closes in on when nothing says where the subject is: the middle.
MIDDLE = (0.5, 0.5)


def motion_of(clip: Clip, still: bool) -> str:
    """Say how a clip's picture moves.

    A photo moves unless told not to, and a video clip only when told to; a
    title card holds its one frame.

    Args:
        clip: The clip.
        still: Its file is a photo.

    Returns:
        One of `MOTIONS`, or `none`.
    """
    if clip.card is not None:
        return "none"
    if still:
        return photo_motion_of(clip.motion)
    return clip.motion or "none"


def window_at(motion: str, progress: float, anchor: Tuple[float, float] = MIDDLE) -> Tuple[float, float, float]:
    """Say which part of a clip's placed picture is on screen, part of the way through its move.

    Args:
        motion: One of `MOTIONS`, or `none`.
        progress: How far through the clip, 0 to 1.
        anchor: The point of the picture, as fractions of its width and
            height, that a push or a pull keeps where it is on the screen;
            a pan crosses the picture and keeps only its height.

    Returns:
        `(zoom, left, top)`: how much closer than the placed picture, and the
        left and top edges of the part shown, as fractions of its width and
        height. The part shown is `1 / zoom` of each.
    """
    moved = max(0.0, min(1.0, progress))
    across, down = anchor
    if motion == "push":
        zoom = 1 + MOTION_AMOUNT * moved
    elif motion == "pull":
        zoom = 1 + MOTION_AMOUNT * (1 - moved)
    elif motion in ("pan_left", "pan_right"):
        zoom = 1 + MOTION_AMOUNT
        across = moved if motion == "pan_left" else 1 - moved
    else:
        return 1.0, 0.0, 0.0
    room = 1 - 1 / zoom
    return zoom, across * room, down * room


def progress_of(frame: int, frames: int) -> float:
    """Say how far through its move a clip is at one of its frames.

    The first frame is the start of the move and the last the end of it, the
    way `perspective` counts, so a move the length of its clip ends on the
    clip's last frame.

    Args:
        frame: The frame, counted from the first frame of the clip's picture,
            run-up included.
        frames: How many frames its picture has, run-up included.

    Returns:
        The progress, 0 to 1.
    """
    return max(0.0, min(1.0, frame / max(1, frames - 1)))


def picture_frames(clip: Clip, fps: Fraction) -> int:
    """Count the frames of a clip's picture, run-up included, as the render lays it out.

    Args:
        clip: The clip.
        fps: The output frame rate.

    Returns:
        The frames, at least one.
    """
    def frame(seconds) -> int:
        exact = Fraction(seconds) * fps
        return int(exact) + (1 if exact - int(exact) >= Fraction(1, 2) else 0)

    return max(1, frame(clip.timeline_out) - frame(clip.timeline_in) + frame(clip.run_up))


def window_at_source(
    motion: str, clip: Clip, seconds: float, fps: Fraction, anchor: Tuple[float, float] = MIDDLE,
) -> Tuple[float, float, float]:
    """Say which part of a clip's placed picture is on screen at one moment of its source.

    Args:
        motion: The clip's motion, as `motion_of` says.
        clip: The clip.
        seconds: A moment of its file, in seconds.
        fps: The output frame rate.
        anchor: As for `window_at`.

    Returns:
        As for `window_at`.
    """
    frame = round(Fraction(str(seconds - float(clip.video_source_start))) / Fraction(str(clip.speed)) * fps)
    return window_at(motion, progress_of(frame, picture_frames(clip, fps)), anchor)


def perspective(motion: str, frames: int, anchor: Tuple[float, float] = MIDDLE) -> Optional[str]:
    """Write the FFmpeg filter that moves a clip's placed picture over its frames.

    The same arithmetic as `window_at`, frame by frame: `on` is the frame
    being made, out of `frames`. The four corners of the part shown are
    stretched to the corners of the frame, and the picture between them is
    sampled at exactly where it falls, between pixels included.

    Args:
        motion: One of `MOTIONS`, or `none`.
        frames: How many frames the clip's picture has, run-up included.
        anchor: As for `window_at`.

    Returns:
        The `perspective` filter, with no commas inside its expressions, or
        None for a clip that does not move.
    """
    if motion not in MOTIONS:
        return None
    progress = f"on/{max(1, frames - 1)}"
    across, down = (f"{value:g}" for value in anchor)
    zoom = {
        "push": f"(1+{MOTION_AMOUNT}*{progress})",
        "pull": f"({1 + MOTION_AMOUNT}-{MOTION_AMOUNT}*{progress})",
    }.get(motion, f"{1 + MOTION_AMOUNT}")
    if motion == "pan_left":
        across = f"({progress})"
    elif motion == "pan_right":
        across = f"(1-{progress})"
    left = f"{across}*W*(1-1/{zoom})"
    top = f"{down}*H*(1-1/{zoom})"
    right = f"{left}+W/{zoom}"
    bottom = f"{top}+H/{zoom}"
    return (f"perspective=x0={left}:y0={top}:x1={right}:y1={top}:x2={left}:y2={bottom}:x3={right}:y3={bottom}"
            ":interpolation=cubic:eval=frame")
