"""The edit plan: what the cut is meant to do, and which footage is doing it.

A plan is a statement of intent, not a list of operations. It says which
semantic clips are used, what each one is there for, and why the ones left out
were left out. Turning that into seconds on a timeline is the compiler's job,
so the same plan always produces the same cut and the reasoning behind it
survives to be argued with later.

The one rule that keeps this true is that nothing here may be written as an
instruction to be interpreted. A trim says `keep these three sentences`, not
"tighten it up": a compiler that had to read prose would need a model inside
it, and then the plan would no longer determine the cut.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)

class TrimKind(str, Enum):
    """How much of a chosen clip is used.

    A closed set, because every one of these is implemented in the compiler.
    Anything finer is expressed by choosing `utterance` clips instead: the
    plan descends to the level it needs rather than asking for a judgement to
    be made further down the line.
    """

    FULL = "full"
    KEEP = "keep"
    HEAD = "head"
    TAIL = "tail"
    TIGHTEN = "tighten"

class Trim(BaseModel):
    """What to take from a chosen clip."""

    kind: TrimKind = TrimKind.FULL
    keep_clip_ids: List[str] = Field(
        default_factory=list,
        description="For `keep`: the clips inside this one to keep, in any order",
    )
    seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="For `head` and `tail`: how many seconds to take",
    )

class Beat(BaseModel):
    """One part of the finished video, and what it is there to do."""

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, description="What this part is, such as 開場 or 結尾")
    intent: str = Field(default="", description="What it has to achieve for the video to work")
    target_seconds: Optional[float] = Field(default=None, gt=0, description="Roughly how long it should run")

class Selection(BaseModel):
    """One piece of footage, placed in a beat, with the reason it is there."""

    clip_id: str = Field(..., description="Semantic clip this uses")
    beat_id: str = Field(..., description="Beat it belongs to")
    trim: Trim = Field(default_factory=Trim)
    rationale: str = Field(default="", description="Why this piece, here")
    role: Optional[str] = Field(default=None, description="What it does in the beat, such as 例子 or 結論")

class Rejection(BaseModel):
    """A piece of footage that was considered and not used.

    Kept because the question always comes: the user asks why the part about
    something is not in there, and the answer is either here or has to be
    worked out again from scratch.
    """

    clip_id: str
    reason: str = Field(default="", description="Why it was left out")

class MusicPlan(BaseModel):
    """A music bed under the whole cut."""

    asset_id: str
    volume: float = Field(default=0.35, ge=0, le=4)
    duck_under_speech: bool = Field(default=True, description="Drop the music while someone is talking")
    fade_in: float = Field(default=1.0, ge=0)
    fade_out: float = Field(default=2.0, ge=0)

class PlanTarget(BaseModel):
    """What the finished video has to be."""

    seconds: Optional[float] = Field(default=None, gt=0, description="How long it should run")
    platform: Optional[str] = Field(default=None, description="Where it is going, such as YouTube or Reels")
    note: str = Field(default="", description="Anything else the cut has to satisfy")

class EditPlan(BaseModel):
    """A cut, stated as intent rather than as operations."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = Field(default=1, ge=1, description="Raised on every save; `save_plan` takes it to detect a clash")
    timeline_id: str = Field(..., description="Semantic timeline the clip IDs belong to")
    timeline_input_hash: str = Field(
        ...,
        description="What that timeline was built from, so a plan cannot be compiled against different footage",
    )
    goal: str = Field(default="", description="What this video is for, in a sentence")
    target: PlanTarget = Field(default_factory=PlanTarget)
    beats: List[Beat] = Field(default_factory=list, description="The parts of the video, in order")
    selections: List[Selection] = Field(default_factory=list, description="The footage, in the order it is used")
    rejected: List[Rejection] = Field(default_factory=list)
    music: Optional[MusicPlan] = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
