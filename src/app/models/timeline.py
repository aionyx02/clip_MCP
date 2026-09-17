from decimal import Decimal
from enum import Enum
from typing import List, Literal, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Any


# create the TimeRange model
class TimeRange(BaseModel):
    start : Decimal = Field(..., ge=0, description="Start time (seconds)")
    end : Decimal = Field(..., ge=0, description="End time (seconds)")

    @model_validator(mode="after")
    def validate_range(self):
        if self.end <= self.start:
            raise ValueError("End time must be greater than start time")
        return self

    @property
    def duration(self) -> Decimal :
        return self.end - self.start

class Clip(BaseModel):
    id: str
    asset_id: str
    source_range: TimeRange = Field(..., description="Source time range of the clip")
    timeline_in: Decimal = Field(..., description="Start time on the timeline")
    speed: float = Field(default=1.0, gt=0)
    volume: float = Field(default=1.0, ge=0)

    @property
    def timeline_duration(self) -> Decimal:
        return Decimal(str(float(self.source_range.duration) / self.speed))

class TrackType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"

class Track(BaseModel):
    id: str
    track_type: TrackType
    clips: List[Clip] = Field(default_factory=list)

class Project(BaseModel):
    def __init__(self, /, **data: Any):
        super().__init__()
        self.fps_npms = None

    id: str
    version: int = Field(default=1, description="Project version")
    fps_npm: int = Field(default=30, description="Frames per minutes")
    fps_den: int = Field(default=1, description="Frames per second denominator")
    width: int = Field(default=1920)
    height: int = Field(default=1080)
    tracks: List[Track] = Field(default_factory=list)

class TrimClipOp(BaseModel):
    action: Literal["trim_clip"] = "trim_clip"
    track_id: str
    clip_id: str

class DeleteOp(BaseModel):
    action: Literal["delete_clip"] = "delete_clip"
    track_id: str
    clip_id: str

class MoveClipOp(BaseModel):
    action: Literal["move_clip"] = "move_clip"
    track_id: str
    clip_id: str
    new_timeline_in: Decimal

EditOperation = Union[TrimClipOp, DeleteOp, MoveClipOp]
