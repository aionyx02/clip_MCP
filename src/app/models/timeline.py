from decimal import Decimal
from enum import Enum
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field, model_validator


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
    timeline_in: Decimal = Field(..., description="Start time on the timeline")
    speed: float = Field(default=1.0, gt=0)
    volume: float = Field(default=1.0, ge=0)

    @property
    def timeline_duration(self) -> Decimal:
        """Length of the clip on the timeline after applying playback speed.

        Returns:
            The source range duration divided by `speed`, in seconds.
        """
        return Decimal(str(float(self.source_range.duration) / self.speed))

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
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    tracks: List[Track] = Field(default_factory=list)

class AddTrackOp(BaseModel):
    """Edit operation that appends an empty track to the project."""

    action: Literal["add_track"] = "add_track"
    track_id: str = Field(..., description="ID of the new track; must be unique within the project")
    track_type: TrackType

class AddClipOp(BaseModel):
    """Edit operation that places a segment of an asset on a track."""

    action: Literal["add_clip"] = "add_clip"
    track_id: str
    clip_id: str = Field(..., description="ID of the new clip; must be unique within the track")
    asset_id: str = Field(..., description="ID returned by import_asset")
    source_range: TimeRange = Field(..., description="Segment of the asset to use")
    timeline_in: Decimal = Field(..., ge=0, description="Start time on the timeline (seconds)")
    speed: float = Field(default=1.0, gt=0)
    volume: float = Field(default=1.0, ge=0)

class TrimClipOp(BaseModel):
    """Edit operation that changes the source range of a clip."""

    action: Literal["trim_clip"] = "trim_clip"
    track_id: str
    clip_id: str
    new_source_range: TimeRange = Field(..., description="New segment of the asset to use")

class DeleteOp(BaseModel):
    """Edit operation that removes a clip from a track."""

    action: Literal["delete_clip"] = "delete_clip"
    track_id: str
    clip_id: str

class MoveClipOp(BaseModel):
    """Edit operation that moves a clip to a new position on the timeline."""

    action: Literal["move_clip"] = "move_clip"
    track_id: str
    clip_id: str
    new_timeline_in: Decimal = Field(..., ge=0, description="New start time on the timeline (seconds)")

EditOperation = Annotated[
    Union[AddTrackOp, AddClipOp, TrimClipOp, DeleteOp, MoveClipOp],
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

def apply_operation(project: Project, op: EditOperation) -> None:
    """Apply a single edit operation to a project in place.

    Clips on the affected track are kept sorted by `timeline_in`, which is
    the order in which they are rendered.

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
        if any(clip.id == op.clip_id for clip in track.clips):
            raise ValueError(f"clip {op.clip_id} already exists on track {op.track_id}")
        track.clips.append(Clip(
            id=op.clip_id,
            asset_id=op.asset_id,
            source_range=op.source_range,
            timeline_in=op.timeline_in,
            speed=op.speed,
            volume=op.volume,
        ))
    elif isinstance(op, TrimClipOp):
        _find_clip(track, op.clip_id).source_range = op.new_source_range
    elif isinstance(op, MoveClipOp):
        _find_clip(track, op.clip_id).timeline_in = op.new_timeline_in
    elif isinstance(op, DeleteOp):
        track.clips.remove(_find_clip(track, op.clip_id))
    track.clips.sort(key=lambda clip: clip.timeline_in)
