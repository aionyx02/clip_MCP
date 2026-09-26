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
    RANGE = "range"
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
    from_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="For `range`: where to start, in seconds from the clip's own start. With `to_seconds`, takes "
                    "a stretch out of the middle — a few seconds of a long timelapse or a walk. Both edges move "
                    "off a word if they land in one",
    )
    to_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="For `range`: where to stop, in seconds from the clip's own start",
    )

class BeatRole(str, Enum):
    """What a part does in the story the video tells: 起承轉合.

    A video that is only a run of moments, however good each one is, leaves
    nothing behind. These four are the least a story is made of, and saying
    which each part is makes a plan with no turn or no ending visible before
    anything is cut.
    """

    HOOK = "hook"
    SETUP = "setup"
    TURN = "turn"
    PAYOFF = "payoff"

class Beat(BaseModel):
    """One part of the finished video, and what it is there to do."""

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, description="What this part is, such as 開場 or 結尾")
    role: Optional[BeatRole] = Field(
        default=None,
        description="What this part does in the story (起承轉合): `hook` opens on the strongest moment or the "
                    "question the video answers; `setup` gives what the viewer needs to follow; `turn` is where "
                    "something changes — a problem, a surprise, a decision; `payoff` is what it came to. Every "
                    "part says one; a plan needs a turn and ends on its payoff",
    )
    intent: str = Field(default="", description="What it has to achieve for the video to work")
    target_seconds: Optional[float] = Field(default=None, gt=0, description="Roughly how long it should run")
    transition_in: Optional["BeatTransition"] = Field(
        default=None,
        description="How the picture passes from the beat before into this one; null is a straight cut. Ignored "
                    "on the first beat. Where the footage has less picture before this beat's first shot than the "
                    "transition asks for, it is shortened to what there is, and the notes say so",
    )
    sound_lead: Optional[float] = Field(
        default=None,
        gt=0,
        le=3,
        description="Let this beat's sound start this many seconds before its picture does, under the end of the "
                    "beat before (a J-cut): the next place is heard before it is seen. Shortened, with a note, "
                    "where the footage has less sound before the shot",
    )

class BeatTransition(BaseModel):
    """How one beat hands over to the next."""

    kind: Literal["dissolve", "dip", "wipe"] = Field(
        ..., description="`dissolve` mixes through, `dip` goes through a colour, `wipe` travels across the frame",
    )
    seconds: float = Field(default=0.5, gt=0, le=3, description="How long it runs, ending on the cut")
    through: str = Field(
        default="black",
        pattern=r"^(#[0-9a-fA-F]{6}|black|white)$",
        description="For `dip`: the colour it passes through",
    )
    direction: Literal["left", "right", "up", "down"] = Field(default="left", description="For `wipe`")

class Selection(BaseModel):
    """One piece of footage, placed in a beat, with the reason it is there."""

    clip_id: str = Field(..., description="Semantic clip this uses")
    beat_id: str = Field(..., description="Beat it belongs to")
    trim: Trim = Field(default_factory=Trim)
    rationale: str = Field(default="", description="Why this piece, here")
    role: Optional[str] = Field(default=None, description="What it does in the beat, such as 例子 or 結論")
    speed: Optional[float] = Field(
        default=None,
        ge=0.25,
        le=8,
        description="Play this piece faster or slower than shot: 4 or 8 for a timelapse of a long walk or a "
                    "street, 0.5 for slow motion. Everything laid over the cut — music, markers, covering "
                    "picture — is placed at the new length. Null for as shot. Multiplied by the pacing speed",
    )
    volume: Optional[float] = Field(
        default=None,
        ge=0,
        le=4,
        description="How loud this piece's own sound plays: 0 mutes it (a shout, wind, a street under music), "
                    "0.5 halves it. Null for as recorded",
    )
    keep_level: bool = Field(
        default=False,
        description="Leave this piece's talking at the level it was recorded instead of bringing it to the level "
                    "of the rest: a whisper, a shout, somebody far from the microphone on purpose",
    )
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

    It cuts in where the clip it names begins *in the cut*, which is usually
    that clip's own start and so a sentence boundary — but not always: a window
    that opens partway into a clip, because a pause was taken out of it or its
    head was trimmed, begins somewhere else. So the compiler checks rather than
    assumes, and refuses a shot that would cut in mid-sentence; moving it costs
    nothing. Where it cuts back out is a length the person planning chose, and
    that one is reported rather than moved: lengthening a shot to let a sentence
    finish would be deciding how long the video runs, which is not the
    compiler's to decide.
    """

    clip_id: str = Field(..., description="Semantic clip the covering picture comes from, played from its start")
    over_clip_id: str = Field(
        ..., description="Semantic clip in the cut this starts over; the shot cuts in where that clip begins",
    )
    seconds: float = Field(..., gt=0, description="How long the covering picture runs")
    rationale: str = Field(default="", description="What this picture is doing here")

class MusicCue(BaseModel):
    """One piece of music, from where one part of the video begins until the next cue takes over.

    A cue names the beat it comes in on rather than a second of the timeline,
    for the same reason B-roll names a clip: the timeline moves every time the
    plan is compiled, and a beat begins wherever its first shot ends up. So a
    change of song follows the part of the video it belongs to.
    """

    beat_id: Optional[str] = Field(
        default=None,
        description="The beat of the plan this comes in on, at the moment that beat begins in the cut. Leave it "
                    "out only on the first cue, to start the music with the video",
    )
    asset_id: Optional[str] = Field(
        default=None,
        description="The music. Null is silence from here until the next cue, for a part that should have none",
    )
    start: float = Field(
        default=0.0,
        ge=0,
        description="Where in the song to come in, in seconds; it loops back to here if the part outlasts it",
    )
    volume: float = Field(default=0.35, ge=0, le=4)
    fade_in: float = Field(default=1.0, ge=0)
    fade_out: float = Field(default=2.0, ge=0, description="Fade at the end of this cue, where the next one takes over")
    cut_on_beat: bool = Field(
        default=False,
        description="Move each picture cut while this plays onto the nearest beat of it. A cut only moves inside "
                    "the silence around it, never into a word, and never further than a fraction of a second; "
                    "one that cannot reach a beat stays where it was and is reported. The song also comes in on "
                    "a beat. Needs the song analyzed, since that is where the beat is found",
    )

class MusicPlan(BaseModel):
    """The music under the cut: one cue, or a run of them that changes with the parts of the video."""

    cues: List[MusicCue] = Field(
        ..., min_length=1, description="In the order they play; each runs until the next one comes in",
    )
    duck_under_speech: bool = Field(default=True, description="Drop the music while someone is talking")
    crossfade_seconds: float = Field(
        default=0.0,
        ge=0,
        le=10,
        description="When one song gives way to the next, overlap them this long: the next comes in early and "
                    "rises while this one fades. 0 meets them end to end, each with its own fades. How long is "
                    "a matter of taste, so it is only ever what the plan says",
    )

class Pacing(BaseModel):
    """How tight the whole cut is.

    The one place a note like 「整體節奏太慢」 lands. It is not a change to any
    one selection — every cut in the video gets tighter at once — so it is a
    setting of the plan rather than an edit to each of its pieces, and the
    compiler applies it everywhere the same way.
    """

    pause_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="Take out any silence inside a shot longer than this, in seconds. Lower is tighter. Null for "
                    "the default, 0.6. It has to leave room for a breath either side of the cut it makes, so it "
                    "cannot go below twice `breath_seconds` plus 0.3",
    )
    breath_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        le=0.5,
        description="Air left before and after every cut, in seconds, never more than the footage has. Lower is "
                    "tighter; 0 starts every line the instant its picture does. Null for the default, 0.1",
    )
    speed: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=2.0,
        description="Play the whole sequence this much faster, voices at their own pitch: 1.1 is a tenth quicker, "
                    "which most viewers do not notice as speed. Markers, covering picture, music and cuts on the "
                    "beat are all laid out at the new pace. Covering picture itself plays at normal speed. Null "
                    "for 1.0",
    )

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
    pacing: Pacing = Field(default_factory=Pacing, description="How tight the whole cut is")
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

class SetPacingOp(BaseModel):
    """Amendment that makes the whole cut tighter or looser at once.

    For feedback about the whole video rather than one part of it: 「整體節奏太慢」
    is shorter pauses and less air everywhere, not a trim to each piece.
    """

    action: Literal["set_pacing"] = "set_pacing"
    pacing: Pacing

class SetMusicLevelOp(BaseModel):
    """Amendment that turns the music up or down, everywhere or in one part."""

    action: Literal["set_music_level"] = "set_music_level"
    scale: float = Field(..., gt=0, le=4, description="What to multiply the music's volume by: 0.6 is quieter, 1.5 louder")
    beat_id: Optional[str] = Field(
        default=None, description="Only the music cue coming in on this beat; every cue when left out",
    )

class SetTargetOp(BaseModel):
    """Amendment that changes what the finished video has to be."""

    action: Literal["set_target"] = "set_target"
    target: PlanTarget

class DropBeatOp(BaseModel):
    """Amendment that takes a whole part out of the video, with the reason, for 「這整段不要」."""

    action: Literal["drop_beat"] = "drop_beat"
    beat_id: str
    reason: str = Field(default="", description="Why it came out; kept with each of its selections under `rejected`")

class SetPlaybackOp(BaseModel):
    """Amendment that changes how fast and how loud one selection plays, for 「這段快轉」 or 「這段靜音」."""

    action: Literal["set_playback"] = "set_playback"
    clip_id: str = Field(..., description="Semantic clip whose selection plays differently")
    speed: Optional[float] = Field(default=None, ge=0.25, le=8, description="As on a selection; null for as shot")
    volume: Optional[float] = Field(default=None, ge=0, le=4, description="As on a selection; null for as recorded")
    keep_level: bool = Field(default=False, description="As on a selection")

class SetBeatJoinOp(BaseModel):
    """Amendment that changes how the picture and the sound pass into a beat from the one before."""

    action: Literal["set_beat_join"] = "set_beat_join"
    beat_id: str
    transition_in: Optional[BeatTransition] = Field(default=None, description="As on a beat; null for a straight cut")
    sound_lead: Optional[float] = Field(default=None, gt=0, le=3, description="As on a beat; null for none")

PlanAmendment = Annotated[
    Union[
        SetTrimOp, SetRationaleOp, AddSelectionOp, DropSelectionOp, AddBrollOp, DropBrollOp,
        SetPacingOp, SetMusicLevelOp, SetTargetOp, DropBeatOp, SetPlaybackOp, SetBeatJoinOp,
    ],
    Field(discriminator="action"),
]

def describe_amendment(op: "PlanAmendment") -> str:
    """Say in a few words what one amendment did, for a version's history.

    Args:
        op: The amendment.

    Returns:
        The phrase.
    """
    if isinstance(op, SetTrimOp):
        return f"retrimmed {op.clip_id} to {op.trim.kind.value}"
    if isinstance(op, SetRationaleOp):
        return f"rewrote why {op.clip_id} is there" + (f", moved it to {op.beat_id}" if op.beat_id else "")
    if isinstance(op, AddSelectionOp):
        return f"added {op.selection.clip_id} to {op.selection.beat_id}"
    if isinstance(op, DropSelectionOp):
        return f"dropped {op.clip_id}" + (f" ({op.reason})" if op.reason else "")
    if isinstance(op, AddBrollOp):
        return f"covered {op.shot.over_clip_id} with {op.shot.clip_id}"
    if isinstance(op, DropBrollOp):
        return f"uncovered {op.over_clip_id}"
    if isinstance(op, SetPacingOp):
        return (f"pacing: pauses over {op.pacing.pause_seconds or 'default'}s out, "
                f"{op.pacing.breath_seconds if op.pacing.breath_seconds is not None else 'default'}s of air, "
                f"x{op.pacing.speed or 1:g}")
    if isinstance(op, SetMusicLevelOp):
        return f"music x{op.scale:g}" + (f" from {op.beat_id}" if op.beat_id else "")
    if isinstance(op, SetPlaybackOp):
        return f"{op.clip_id} plays at x{op.speed or 1:g}, volume {1 if op.volume is None else op.volume:g}"
    if isinstance(op, SetBeatJoinOp):
        joined = op.transition_in.kind if op.transition_in else "a straight cut"
        return f"{op.beat_id} comes in on {joined}" + (f", sound {op.sound_lead:g}s early" if op.sound_lead else "")
    if isinstance(op, SetTargetOp):
        return f"target {op.target.seconds or 'none'}s" + (f" for {op.target.platform}" if op.target.platform else "")
    return f"dropped the part {op.beat_id}" + (f" ({op.reason})" if op.reason else "")

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

    if isinstance(op, SetPacingOp):
        plan.pacing = op.pacing
        return

    if isinstance(op, SetPlaybackOp):
        selection = plan.selections[_selection_index(plan, op.clip_id)]
        selection.speed, selection.volume, selection.keep_level = op.speed, op.volume, op.keep_level
        return

    if isinstance(op, SetBeatJoinOp):
        beat = next((beat for beat in plan.beats if beat.id == op.beat_id), None)
        if beat is None:
            known = ", ".join(beat.id for beat in plan.beats) or "none"
            raise ValueError(f"the plan has no beat {op.beat_id}; its beats are {known}")
        beat.transition_in, beat.sound_lead = op.transition_in, op.sound_lead
        return

    if isinstance(op, SetMusicLevelOp):
        if plan.music is None:
            raise ValueError("the plan has no music to turn up or down")
        cues = [cue for cue in plan.music.cues if op.beat_id is None or cue.beat_id == op.beat_id]
        if not cues:
            raise ValueError(f"no music cue comes in on {op.beat_id}")
        for cue in cues:
            # Capped where the volume field is, rather than refused: turning something up
            # twice should stop at the ceiling, not fail the second time.
            cue.volume = round(min(4.0, cue.volume * op.scale), 3)
        return

    if isinstance(op, SetTargetOp):
        plan.target = op.target
        return

    if isinstance(op, DropBeatOp):
        if not any(beat.id == op.beat_id for beat in plan.beats):
            known = ", ".join(beat.id for beat in plan.beats) or "none"
            raise ValueError(f"the plan has no beat {op.beat_id}; its beats are {known}")
        going = [selection for selection in plan.selections if selection.beat_id == op.beat_id]
        plan.selections = [selection for selection in plan.selections if selection.beat_id != op.beat_id]
        plan.beats = [beat for beat in plan.beats if beat.id != op.beat_id]
        gone = {selection.clip_id for selection in going}
        plan.rejected = [item for item in plan.rejected if item.clip_id not in gone]
        plan.rejected += [Rejection(clip_id=selection.clip_id, reason=op.reason) for selection in going]
        # The same arithmetic a dropped selection gets: covers and music with nowhere to go.
        plan.broll = [item for item in plan.broll if item.over_clip_id not in gone]
        if plan.music is not None:
            plan.music.cues = [cue for cue in plan.music.cues if cue.beat_id != op.beat_id]
            if not plan.music.cues:
                plan.music = None
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

Beat.model_rebuild()
