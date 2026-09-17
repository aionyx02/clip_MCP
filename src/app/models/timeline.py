from decimal import Decimal
from enum import Enum
from typing import Annotated, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, Field, computed_field, model_validator

from app.models.media import Asset


class TimeRange(BaseModel):
    """A time span in seconds, where `end` must be greater than `start`."""

    start : Decimal = Field(..., ge=0, description="Start time (seconds)")
    end : Decimal = Field(..., ge=0, description="End time (seconds)")

    @model_validator(mode="after")
    def validate_range(self):
        """Ensure the range has a positive duration.

        Returns:
            The validated `TimeRange` instance.

        Raises:
            ValueError: If `end` is less than or equal to `start`.
        """
        if self.end <= self.start:
            raise ValueError("End time must be greater than start time")
        return self

    @property
    def duration(self) -> Decimal :
        """Length of the range in seconds.

        Returns:
            The difference between `end` and `start`.
        """
        return self.end - self.start

class Clip(BaseModel):
    """A segment of a source asset placed on a track."""

    id: str
    asset_id: str
    source_range: TimeRange = Field(..., description="Source time range of the clip")
    timeline_in: Decimal = Field(..., ge=0, description="Start time on the timeline")
    speed: float = Field(default=1.0, gt=0)
    volume: float = Field(default=1.0, ge=0, description="Audio gain; 1.0 keeps the original level, 0.5 halves it")
    audio_fade_in: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-in length at the clip start (seconds)")
    audio_fade_out: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-out length at the clip end (seconds)")

    @property
    def timeline_duration(self) -> Decimal:
        """Length of the clip on the timeline after applying playback speed.

        Returns:
            The source range duration divided by `speed`, in seconds.
        """
        return self.source_range.duration / Decimal(str(self.speed))

    @property
    def timeline_out(self) -> Decimal:
        """Time on the timeline at which the clip ends.

        Returns:
            `timeline_in` plus `timeline_duration`, in seconds.
        """
        return self.timeline_in + self.timeline_duration

class TrackType(str, Enum):
    """Kind of media a track carries."""

    VIDEO = "video"
    AUDIO = "audio"

class Track(BaseModel):
    """An ordered sequence of clips of a single media type."""

    id: str
    track_type: TrackType
    clips: List[Clip] = Field(default_factory=list)

class Project(BaseModel):
    """An editing project: output format settings plus its tracks."""

    id: str
    version: int = Field(default=1, description="Project version")
    fps_num: int = Field(default=30, gt=0, description="Frame rate numerator (e.g. 30000 for 29.97 fps)")
    fps_den: int = Field(default=1, gt=0, description="Frame rate denominator (e.g. 1001 for 29.97 fps)")
    width: int = Field(default=1920, gt=0, multiple_of=2, description="Output width (pixels); must be even")
    height: int = Field(default=1080, gt=0, multiple_of=2, description="Output height (pixels); must be even")
    tracks: List[Track] = Field(default_factory=list)

    @computed_field(description="Length of the edited video (seconds): the end of the last clip on the video track")
    @property
    def duration(self) -> Decimal:
        """Length of the edited video.

        Audio tracks do not extend the output; audio past this point is cut.

        Returns:
            The end of the last clip on the video track in seconds, or zero if
            the project has no video clips.
        """
        return max(
            (clip.timeline_out for track in self.tracks if track.track_type == TrackType.VIDEO for clip in track.clips),
            default=Decimal(0),
        )

class AddTrackOp(BaseModel):
    """Edit operation that appends an empty track to the project."""

    action: Literal["add_track"] = "add_track"
    track_id: str = Field(..., description="ID of the new track; must be unique within the project")
    track_type: TrackType

class _NewClipSpec(BaseModel):
    """Fields shared by operations that create a clip."""

    track_id: str
    clip_id: str = Field(..., description="ID of the new clip; must be unique within the track")
    asset_id: str = Field(..., description="ID returned by import_asset")
    source_range: TimeRange = Field(..., description="Segment of the asset to use")
    speed: float = Field(default=1.0, gt=0)
    volume: float = Field(default=1.0, ge=0, description="Audio gain; 1.0 keeps the original level, 0.5 halves it")
    audio_fade_in: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-in length at the clip start (seconds)")
    audio_fade_out: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-out length at the clip end (seconds)")

class AddClipOp(_NewClipSpec):
    """Edit operation that places a clip at an absolute timeline position without moving other clips."""

    action: Literal["add_clip"] = "add_clip"
    timeline_in: Decimal = Field(..., ge=0, description="Start time on the timeline (seconds)")

class InsertClipOp(_NewClipSpec):
    """Edit operation that inserts a clip into the sequence, shifting later clips to make room."""

    action: Literal["insert_clip"] = "insert_clip"
    before_clip_id: Optional[str] = Field(
        default=None,
        description="Insert at the start of this clip; omit to append after the last clip on the track",
    )

class TrimClipOp(BaseModel):
    """Edit operation that changes the source range of a clip."""

    action: Literal["trim_clip"] = "trim_clip"
    track_id: str
    clip_id: str
    new_source_range: TimeRange = Field(..., description="New segment of the asset to use")
    ripple: bool = Field(
        default=True,
        description="Shift later clips by the change in duration, so no gap or overlap is created",
    )

class DeleteOp(BaseModel):
    """Edit operation that removes a clip from a track."""

    action: Literal["delete_clip"] = "delete_clip"
    track_id: str
    clip_id: str
    ripple: bool = Field(
        default=True,
        description="Shift later clips earlier to close the space the clip occupied; false leaves a gap",
    )

class MoveClipOp(BaseModel):
    """Edit operation that moves a clip to an absolute timeline position without moving other clips."""

    action: Literal["move_clip"] = "move_clip"
    track_id: str
    clip_id: str
    new_timeline_in: Decimal = Field(..., ge=0, description="New start time on the timeline (seconds)")

class SetClipAudioOp(BaseModel):
    """Edit operation that changes a clip's audio level or fades; omitted fields stay unchanged."""

    action: Literal["set_clip_audio"] = "set_clip_audio"
    track_id: str
    clip_id: str
    volume: Optional[float] = Field(default=None, ge=0, description="Audio gain; 1.0 keeps the original level, 0.5 halves it")
    audio_fade_in: Optional[Decimal] = Field(default=None, ge=0, description="Audio fade-in length at the clip start (seconds)")
    audio_fade_out: Optional[Decimal] = Field(default=None, ge=0, description="Audio fade-out length at the clip end (seconds)")

EditOperation = Annotated[
    Union[AddTrackOp, AddClipOp, InsertClipOp, TrimClipOp, DeleteOp, MoveClipOp, SetClipAudioOp],
    Field(discriminator="action"),
]

def _find_track(project: Project, track_id: str) -> Track:
    """Look up a track by ID.

    Args:
        project: Project to search.
        track_id: ID of the track to find.

    Returns:
        The matching track.

    Raises:
        ValueError: If the project has no track with `track_id`.
    """
    for track in project.tracks:
        if track.id == track_id:
            return track
    raise ValueError(f"track {track_id} not found")

def _find_clip(track: Track, clip_id: str) -> Clip:
    """Look up a clip by ID.

    Args:
        track: Track to search.
        clip_id: ID of the clip to find.

    Returns:
        The matching clip.

    Raises:
        ValueError: If the track has no clip with `clip_id`.
    """
    for clip in track.clips:
        if clip.id == clip_id:
            return clip
    raise ValueError(f"clip {clip_id} not found on track {track.id}")

def _new_clip(track: Track, spec: _NewClipSpec, timeline_in: Decimal) -> Clip:
    """Create a clip from an operation's clip fields.

    Args:
        track: Track the clip will be added to.
        spec: Operation describing the clip.
        timeline_in: Start time of the clip on the timeline.

    Returns:
        The new clip. It is not added to the track.

    Raises:
        ValueError: If the track already has a clip with the same ID.
    """
    if any(clip.id == spec.clip_id for clip in track.clips):
        raise ValueError(f"clip {spec.clip_id} already exists on track {track.id}")
    return Clip(
        id=spec.clip_id,
        asset_id=spec.asset_id,
        source_range=spec.source_range,
        timeline_in=timeline_in,
        speed=spec.speed,
        volume=spec.volume,
        audio_fade_in=spec.audio_fade_in,
        audio_fade_out=spec.audio_fade_out,
    )

def _shift_clips(track: Track, starting_at: Decimal, delta: Decimal) -> None:
    """Move every clip that starts at or after a position by the same amount.

    Args:
        track: Track whose clips are moved.
        starting_at: Timeline position (seconds); clips starting earlier stay put.
        delta: Seconds to add to each affected clip's `timeline_in`.
    """
    for clip in track.clips:
        if clip.timeline_in >= starting_at:
            clip.timeline_in += delta

def apply_operation(project: Project, op: EditOperation) -> None:
    """Apply a single edit operation to a project in place.

    Rippling operations (`insert_clip`, and `trim_clip` and `delete_clip`
    unless `ripple` is false) shift the clips that follow the edit point on
    the same track, so the rest of that track keeps its order and spacing.
    Clips on other tracks never move. Clips on the
    affected track are kept sorted by `timeline_in`, the order in which they
    are rendered.

    Args:
        project: Project to modify.
        op: Operation to apply.

    Raises:
        ValueError: If the operation references a track or clip that does not
            exist, or adds a track or clip whose ID is already in use.
    """
    if isinstance(op, AddTrackOp):
        if any(track.id == op.track_id for track in project.tracks):
            raise ValueError(f"track {op.track_id} already exists")
        project.tracks.append(Track(id=op.track_id, track_type=op.track_type))
        return

    track = _find_track(project, op.track_id)
    if isinstance(op, AddClipOp):
        track.clips.append(_new_clip(track, op, op.timeline_in))
    elif isinstance(op, InsertClipOp):
        if op.before_clip_id is None:
            position = max((clip.timeline_out for clip in track.clips), default=Decimal(0))
        else:
            position = _find_clip(track, op.before_clip_id).timeline_in
        clip = _new_clip(track, op, position)
        _shift_clips(track, position, clip.timeline_duration)
        track.clips.append(clip)
    elif isinstance(op, TrimClipOp):
        clip = _find_clip(track, op.clip_id)
        old_out = clip.timeline_out
        clip.source_range = op.new_source_range
        if op.ripple:
            _shift_clips(track, old_out, clip.timeline_out - old_out)
    elif isinstance(op, MoveClipOp):
        _find_clip(track, op.clip_id).timeline_in = op.new_timeline_in
    elif isinstance(op, DeleteOp):
        clip = _find_clip(track, op.clip_id)
        track.clips.remove(clip)
        if op.ripple:
            _shift_clips(track, clip.timeline_out, -clip.timeline_duration)
    elif isinstance(op, SetClipAudioOp):
        clip = _find_clip(track, op.clip_id)
        for field in ("volume", "audio_fade_in", "audio_fade_out"):
            value = getattr(op, field)
            if value is not None:
                setattr(clip, field, value)
    track.clips.sort(key=lambda clip: clip.timeline_in)

def validate_project(project: Project, assets: Mapping[str, Asset]) -> None:
    """Check that a project's timeline is semantically consistent.

    These rules hold regardless of what the renderer supports: every clip
    references a registered asset, stays within that asset's duration, does
    not overlap another clip on the same track, and has audio fades that fit
    within its length.

    Args:
        project: Project to check.
        assets: Registered assets, keyed by asset ID.

    Raises:
        ValueError: If any rule is violated. The message names the offending
            clip and explains how to fix it.
    """
    for track in project.tracks:
        previous: Optional[Clip] = None
        for clip in sorted(track.clips, key=lambda c: c.timeline_in):
            asset = assets.get(clip.asset_id)
            if asset is None:
                raise ValueError(f"clip {clip.id}: asset {clip.asset_id} not found")
            if asset.duration is not None and clip.source_range.end > asset.duration:
                raise ValueError(
                    f"clip {clip.id}: source range ends at {clip.source_range.end}s, "
                    f"but asset {asset.id} is only {asset.duration}s long"
                )
            if clip.audio_fade_in + clip.audio_fade_out > clip.timeline_duration:
                raise ValueError(
                    f"clip {clip.id}: audio fades ({clip.audio_fade_in}s in + {clip.audio_fade_out}s out) "
                    f"are longer than the clip ({clip.timeline_duration}s)"
                )
            if previous is not None and clip.timeline_in < previous.timeline_out:
                raise ValueError(
                    f"clip {clip.id} starts at {clip.timeline_in}s and overlaps clip {previous.id}, "
                    f"which ends at {previous.timeline_out}s on track {track.id}"
                )
            previous = clip
