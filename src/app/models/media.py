from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field


class Asset(BaseModel):
    """A registered source media file and the probe facts editing relies on."""

    id: str
    path: str = Field(..., description="Absolute path to the media file")
    duration: Optional[Decimal] = Field(default=None, description="Container duration (seconds), if known")
    has_video: bool
    has_audio: bool

class Span(BaseModel):
    """A time span within a media file, in seconds."""

    start: float
    end: float

class TranscriptWord(BaseModel):
    """A transcribed word or character with its timing."""

    start: float
    end: float
    text: str

class TranscriptSegment(BaseModel):
    """A transcribed sentence or phrase with its timing."""

    start: float
    end: float
    text: str
    words: List[TranscriptWord] = Field(default_factory=list)

class Transcript(BaseModel):
    """Speech transcribed from a media file."""

    language: str = Field(..., description="Detected or requested language code, such as 'zh'")
    model: str = Field(..., description="Speech recognition model that produced the transcript")
    chinese_variant: Optional[str] = Field(default=None, description="Chinese variant the text was converted to, such as 'zh-TW'")
    segments: List[TranscriptSegment] = Field(default_factory=list)

class MediaAnalysis(BaseModel):
    """Machine-readable description of an asset's content, used to plan edits."""

    asset_id: str
    duration: float
    scenes: List[Span] = Field(default_factory=list, description="Shots between detected scene changes")
    black_frames: List[Span] = Field(default_factory=list, description="Stretches of (nearly) black picture")
    frozen_frames: List[Span] = Field(default_factory=list, description="Stretches where the picture does not change")
    silences: List[Span] = Field(default_factory=list, description="Stretches without audible sound")
    transcript: Optional[Transcript] = None
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
