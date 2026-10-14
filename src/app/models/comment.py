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


# Who the user is in a comment and its replies: the editor's own word for them.
USER = "你"


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
    by: str = USER
    created_at: datetime = Field(default_factory=_utc_now)
    commit: Optional[str] = Field(default=None, description="The version it was written on")
    start: float = Field(..., ge=0, description="Where on the cut it was written, in seconds")
    end: Optional[float] = Field(default=None, description="Where the stretch it is about ends; None for a moment")
    start_anchor: Optional[Anchor] = None
    end_anchor: Optional[Anchor] = None
    status: Literal["open", "resolved"] = "open"
    question: bool = Field(default=False, description="Whether the AI asked it, for the user to answer")
    replies: List[Reply] = Field(default_factory=list)
    resolved_commit: Optional[str] = Field(default=None, description="The version it was dealt with in")

    @property
    def answered(self) -> bool:
        """Whether the user has said something back on it: for a question the AI asked, its answer."""
        return any(reply.by == USER for reply in self.replies)

    @property
    def waiting(self) -> bool:
        """Whether it waits on the AI: open, and not a question still waiting on the user."""
        return self.status == "open" and (not self.question or self.answered)


class Question(BaseModel):
    """Something the AI is not sure of about a moment of the cut, put to the user."""

    start: float = Field(..., ge=0, description="Where on the cut, in seconds")
    end: Optional[float] = Field(default=None, description="Where the stretch it is about ends; None for a moment")
    question: str = Field(..., min_length=1, max_length=300, description="What to ask, in the user's language")
    why: str = Field(default="", max_length=300,
                     description="Why it is in doubt — what was weighed — so the user can answer the real question")


class Preference(BaseModel):
    """How the user likes their videos, in their words, kept for every project after."""

    id: str
    text: str = Field(..., min_length=1, max_length=300)
    said_at: datetime = Field(default_factory=_utc_now)
    project_id: Optional[str] = Field(default=None, description="The video it was said about")
