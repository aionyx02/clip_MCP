"""One request to make a video, kept so a new conversation can pick it up where the last one left off.

「直接做完」 used to live only in the conversation that heard it: a new one did not
know what was asked, or how far it had got. A production keeps only what cannot be
worked out again from what is already stored: the mode, the request as it was
said, the footage and the shapes it is for, the project and plan it is bound to,
and what the user said along the way. How far it has got, and whether it is
finished, are worked out each time from the footage, the plan, the project and its
renders (`engine/production.py`), so an edit made by hand in the editor never
leaves the two disagreeing.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# The shapes a cut is delivered in; None is the project's own.
Shape = Literal["landscape", "portrait", "square"]
Mode = Literal["interactive", "auto"]


def _utc_now() -> datetime:
    """The time now, in UTC."""
    return datetime.now(timezone.utc)


class Said(BaseModel):
    """Something the user said in the conversation, in their words, and when."""

    said: str = Field(..., min_length=1)
    at: datetime = Field(default_factory=_utc_now)


class Story(BaseModel):
    """The story the video tells, written down before the cut: a paper edit.

    Every part is required and has to say something, so the card is specific
    enough for the user to say it is wrong. Whether it is a good story is the
    user's to say, not the server's.
    """

    theme: str = Field(..., description="The video in one sentence")
    structure: str = Field(..., description="How it is built: 起承轉合, 清單式, 教學步驟, and in what order")
    opening: str = Field(..., description="How it opens: the first moment on screen")
    turn: str = Field(..., description="Where it turns: the moment things change")
    ending: str = Field(..., description="How it ends: the last moment on screen")
    left_out: str = Field(..., description="What is deliberately left out, and why")
    length: str = Field(..., description="How long it will run, roughly, as the user would say it")

    @field_validator("*")
    @classmethod
    def _says_something(cls, value: str, info) -> str:
        """Refuse a part left empty or blank; keep what was written, trimmed."""
        if not value.strip():
            raise ValueError(f"the story's {info.field_name} is empty: every part of the card has to say something")
        return value.strip()

    def digest(self) -> str:
        """A fingerprint of what the card says; any change to any part changes it."""
        blob = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class StoryConfirmed(BaseModel):
    """The user's yes to one story card, in their words."""

    story_hash: str = Field(..., description="The card they said yes to, by its fingerprint")
    said: str = Field(..., min_length=1)
    at: datetime = Field(default_factory=_utc_now)


class Production(BaseModel):
    """One request to make a video, from what was asked to the finished files."""

    id: str = Field(default_factory=lambda: "pr" + uuid.uuid4().hex[:8])
    mode: Mode = Field(..., description="`auto` when the user said to finish on your own; `interactive` otherwise")
    brief: str = Field(..., min_length=1, description="What the user asked for, as they said it")
    asset_ids: List[str] = Field(default_factory=list, description="The footage it is made from")
    outputs: List[Optional[Shape]] = Field(
        default_factory=lambda: [None],
        description="The shapes to deliver; None is the project's own",
    )
    project_id: Optional[str] = Field(default=None, description="The project it is cut on, once there is one")
    plan_id: Optional[str] = Field(
        default=None,
        description="The plan it is cut from. Kept because nothing else says it between `save_plan` and "
                    "`compile_plan`, when no project has been compiled from it yet",
    )
    said: List[Said] = Field(
        default_factory=list, description="What the user said along the way in the conversation, in their words",
    )
    story: Optional[Story] = Field(default=None, description="The story the video tells, once it is written")
    story_confirmed: Optional[StoryConfirmed] = Field(
        default=None,
        description="The user's yes to the card as it is now; cleared whenever the card changes",
    )
    story_confirmed_at: Optional[datetime] = Field(
        default=None,
        description="When the user last said yes to a story card. Kept when the card changes, and never moved "
                    "back: it is a point the user was last heard from",
    )
    created_at: datetime = Field(default_factory=_utc_now)
    closed: Optional[Said] = Field(default=None, description="The user's words when they called it off")
