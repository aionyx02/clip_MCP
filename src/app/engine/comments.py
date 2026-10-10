"""Where a comment sits on a cut: pinned to the footage it was written on, wherever that went.

Pure functions of the project, like the rest of the engine: the same cut and
the same comment always give the same place.
"""

from typing import Optional, Tuple

from app.models.comment import Anchor, Comment
from app.models.timeline import TOUCHING_SECONDS, Clip, Project

TOUCHING = float(TOUCHING_SECONDS)


def anchor_at(project: Project, seconds: float) -> Optional[Anchor]:
    """Say which moment of which file is on screen at a second of the cut.

    Args:
        project: The cut.
        seconds: Where on it.

    Returns:
        The file, its second and the clip; None in a gap, or past the end.
    """
    base = project.base_video_track
    for clip in base.clips if base else []:
        start, end = float(clip.timeline_in), float(clip.timeline_out)
        if start <= seconds < end or (seconds == end and clip is base.clips[-1]):
            source = float(clip.source_range.start) + (seconds - start) * clip.speed
            return Anchor(asset_id=clip.asset_id, source=round(source, 3), clip_id=clip.id)
    return None


def _second_of(clip: Clip, source: float) -> float:
    """Where a second of a clip's file lands on the cut."""
    return round(float(clip.timeline_in) + (source - float(clip.source_range.start)) / clip.speed, 3)


def place(project: Project, anchor: Anchor) -> Optional[Tuple[float, str]]:
    """Find where a moment of footage is on the cut now.

    Args:
        project: The cut as it is.
        anchor: The moment.

    Returns:
        The second it plays at and the clip it is in; the same clip is
        preferred when the file is used more than once. None when that moment
        is no longer in the cut.
    """
    base = project.base_video_track
    holding = [
        clip for clip in (base.clips if base else [])
        if clip.asset_id == anchor.asset_id
        and float(clip.source_range.start) - TOUCHING <= anchor.source <= float(clip.source_range.end) + TOUCHING
    ]
    if not holding:
        return None
    clip = next((item for item in holding if item.id == anchor.clip_id), holding[0])
    return _second_of(clip, anchor.source), clip.id


def where_now(project: Project, comment: Comment) -> dict:
    """Say where a comment is on the cut as it is now.

    Args:
        project: The cut as it is.
        comment: The comment.

    Returns:
        `start` and `end` (None for a moment) in seconds, the `clip_id` under
        its start, and `gone` when the footage it was written on has been
        cut out. A comment written over a gap keeps the second it was written
        at, since there was nothing to follow.
    """
    if comment.start_anchor is None:
        return {"start": comment.start, "end": comment.end, "clip_id": None, "gone": False}
    found = place(project, comment.start_anchor)
    ended = place(project, comment.end_anchor) if comment.end_anchor is not None else None
    if found is None and ended is None:
        return {"start": comment.start, "end": comment.end, "clip_id": None, "gone": True}
    start, clip_id = found if found is not None else ended
    end = None
    if comment.end is not None:
        end = ended[0] if ended is not None else start + (comment.end - comment.start)
        if end < start:
            start, end = end, start
    return {"start": start, "end": end, "clip_id": clip_id, "gone": False}
