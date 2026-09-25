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
from typing import Annotated, List, Literal, Optional, Union

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
    hold_picture: bool = Field(
        default=False,
        description="This shot is here for what it shows, not only for what is said over it, so B-roll may not "
                    "cover it. Set it on the shots where the speaker is doing something that has to be seen — "
                    "pointing at a thing, holding a thing up. Nothing measures that, so it is said here or not at all",
    )

class Rejection(BaseModel):
    """A piece of footage that was considered and not used.

    Kept because the question always comes: the user asks why the part about
    something is not in there, and the answer is either here or has to be
    worked out again from scratch.
    """

    clip_id: str
    reason: str = Field(default="", description="Why it was left out")

class BrollShot(BaseModel):
    """One stretch of the cut with different picture laid over it.

    The sound underneath keeps running, which is the whole point: B-roll is
    what somebody is talking about, shown while they talk about it. So a shot
    says where in the cut it starts and how long it runs, not where it sits on
    the timeline — the timeline moves every time the plan is compiled again,
    and a shot pinned to a second of it would be somewhere else by the next
    compile.

    It starts at the start of a semantic clip, so it always cuts in on a
    sentence boundary. Where it cuts back out is a length the person planning
    chose, and the compiler reports rather than moves it: lengthening a shot
    to let a sentence finish would be deciding how long the video runs, which
    is not the compiler's to decide.
    """

    clip_id: str = Field(..., description="Semantic clip the covering picture comes from, played from its start")
    over_clip_id: str = Field(
        ..., description="Semantic clip in the cut this starts over; the shot cuts in where that clip begins",
    )
    seconds: float = Field(..., gt=0, description="How long the covering picture runs")
    rationale: str = Field(default="", description="What this picture is doing here")

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
        default="",
        description=(
            "What that timeline was built from, so a plan cannot be compiled against different footage. "
            "Leave this out when writing a plan: `save_plan` stamps it from the timeline it names"
        ),
    )
    goal: str = Field(default="", description="What this video is for, in a sentence")
    target: PlanTarget = Field(default_factory=PlanTarget)
    beats: List[Beat] = Field(default_factory=list, description="The parts of the video, in order")
    selections: List[Selection] = Field(default_factory=list, description="The footage, in the order it is used")
    rejected: List[Rejection] = Field(default_factory=list)
    broll: List[BrollShot] = Field(
        default_factory=list,
        description="Picture laid over the sequence while its sound keeps running. A second pass over a rough cut, "
                    "not something to write at the same time as the selections",
    )
    music: Optional[MusicPlan] = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

class SetTrimOp(BaseModel):
    """Amendment that changes how much of one chosen clip is used."""

    action: Literal["set_trim"] = "set_trim"
    clip_id: str = Field(..., description="Semantic clip whose selection is being retrimmed")
    trim: Trim

class SetRationaleOp(BaseModel):
    """Amendment that rewrites why one piece is where it is; omitted fields stay unchanged."""

    action: Literal["set_rationale"] = "set_rationale"
    clip_id: str
    rationale: Optional[str] = Field(default=None, description="Why this piece, here")
    role: Optional[str] = Field(default=None, description="What it does in the beat, such as 例子 or 結論")
    beat_id: Optional[str] = Field(default=None, description="Move it into a different beat")

class AddSelectionOp(BaseModel):
    """Amendment that puts one more piece of footage into the cut."""

    action: Literal["add_selection"] = "add_selection"
    selection: Selection
    before_clip_id: Optional[str] = Field(
        default=None,
        description="Put it in front of this selection; appended to the end when left out",
    )

class DropSelectionOp(BaseModel):
    """Amendment that takes a piece out of the cut, and says why for the next person who asks."""

    action: Literal["drop_selection"] = "drop_selection"
    clip_id: str
    reason: str = Field(default="", description="Why it came out; kept with the plan's other rejections")

class AddBrollOp(BaseModel):
    """Amendment that lays one shot of covering picture over the cut."""

    action: Literal["add_broll"] = "add_broll"
    shot: BrollShot

class DropBrollOp(BaseModel):
    """Amendment that takes one shot of covering picture back off again."""

    action: Literal["drop_broll"] = "drop_broll"
    over_clip_id: str = Field(..., description="The clip the shot being removed starts over")

PlanAmendment = Annotated[
    Union[SetTrimOp, SetRationaleOp, AddSelectionOp, DropSelectionOp, AddBrollOp, DropBrollOp],
    Field(discriminator="action"),
]

def _selection_index(plan: EditPlan, clip_id: str) -> int:
    """Find a selection in a plan by the clip it uses.

    Args:
        plan: Plan to search.
        clip_id: Semantic clip the selection names.

    Returns:
        Its position in `plan.selections`.

    Raises:
        ValueError: If the plan does not select that clip.
    """
    for index, selection in enumerate(plan.selections):
        if selection.clip_id == clip_id:
            return index
    known = ", ".join(item.clip_id for item in plan.selections[:5]) or "nothing"
    raise ValueError(f"the plan does not use clip {clip_id}; it selects {known}")

def apply_amendment(plan: EditPlan, op: PlanAmendment) -> None:
    """Apply one amendment to a plan in place.

    This is how a plan changes: a trim that was too long, a piece that turned
    out not to work, one more shot that belongs in a beat. The alternative —
    sending the whole plan again to move one edge — loses the reasons written
    into every other selection to a typo.

    Args:
        plan: Plan to amend.
        op: The amendment.

    Raises:
        ValueError: If it names a selection the plan does not have, or adds
            one for a clip that is already in the cut.
    """
    if isinstance(op, AddBrollOp):
        # One shot per place it starts, so adding over the same clip twice replaces
        # rather than stacking: two covers starting together is not an edit anybody meant.
        plan.broll = [item for item in plan.broll if item.over_clip_id != op.shot.over_clip_id]
        plan.broll.append(op.shot)
        return

    if isinstance(op, DropBrollOp):
        kept = [item for item in plan.broll if item.over_clip_id != op.over_clip_id]
        if len(kept) == len(plan.broll):
            raise ValueError(f"no B-roll in this plan starts over clip {op.over_clip_id}")
        plan.broll = kept
        return

    if isinstance(op, SetTrimOp):
        plan.selections[_selection_index(plan, op.clip_id)].trim = op.trim
        return

    if isinstance(op, SetRationaleOp):
        selection = plan.selections[_selection_index(plan, op.clip_id)]
        if op.rationale is not None:
            selection.rationale = op.rationale
        if op.role is not None:
            selection.role = op.role
        if op.beat_id is not None:
            selection.beat_id = op.beat_id
        return

    if isinstance(op, AddSelectionOp):
        if any(item.clip_id == op.selection.clip_id for item in plan.selections):
            raise ValueError(f"clip {op.selection.clip_id} is already in the cut; retrim it rather than adding it twice")
        at = len(plan.selections) if op.before_clip_id is None else _selection_index(plan, op.before_clip_id)
        plan.selections.insert(at, op.selection)
        # Putting something back settles the question of why it was out.
        plan.rejected = [item for item in plan.rejected if item.clip_id != op.selection.clip_id]
        return

    dropped = plan.selections.pop(_selection_index(plan, op.clip_id))
    plan.rejected = [item for item in plan.rejected if item.clip_id != dropped.clip_id]
    plan.rejected.append(Rejection(clip_id=dropped.clip_id, reason=op.reason))
    # A cover that started over the shot just removed has nowhere left to start. Taking it
    # with the shot is arithmetic, not a judgement: the place it named is gone.
    plan.broll = [item for item in plan.broll if item.over_clip_id != dropped.clip_id]
