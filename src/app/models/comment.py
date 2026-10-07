"""What the user says about a moment of a cut, written while watching it.

A comment is pinned to what was on screen, not to a second: the cut changes
under it, and 「這裡太拖」 has to keep meaning the shot it was written on. So
besides the second it was written at, it keeps which file was playing and at
which of that file's seconds, and finds its place again in whatever the cut
has become.
"""

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """The time now, in UTC."""
    return datetime.now(timezone.utc)


class Anchor(BaseModel):
    """The moment of footage under a comment's edge."""

    asset_id: str = Field(..., description="The file that was playing")
    source: float = Field(..., description="Which of that file's seconds")
    clip_id: str = Field(..., description="The clip it was in, preferred when the file is used twice")


class Reply(BaseModel):
    """Something said back on a comment: what the AI did about it, or the user adding to it."""

    by: str
    text: str
    at: datetime = Field(default_factory=_utc_now)
    commit: Optional[str] = Field(default=None, description="The version it was said at")


class Comment(BaseModel):
    """One thing the user said about a moment, or a stretch, of a cut."""

    id: str
    project_id: str
    text: str
    by: str = "你"
    created_at: datetime = Field(default_factory=_utc_now)
    commit: Optional[str] = Field(default=None, description="The version it was written on")
    start: float = Field(..., ge=0, description="Where on the cut it was written, in seconds")
    end: Optional[float] = Field(default=None, description="Where the stretch it is about ends; None for a moment")
    start_anchor: Optional[Anchor] = None
    end_anchor: Optional[Anchor] = None
    status: Literal["open", "resolved"] = "open"
    replies: List[Reply] = Field(default_factory=list)
    resolved_commit: Optional[str] = Field(default=None, description="The version it was dealt with in")
