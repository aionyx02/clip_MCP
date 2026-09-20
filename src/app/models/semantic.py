"""The semantic timeline: what is in the footage, as units an edit can be planned from.

A `SemanticClip` is one thing that can stand on its own — a sentence, a shot,
a pause. Edits are planned by referring to these by ID rather than by second,
so a plan says what it wants rather than where to cut, and the server works
out the seconds.

Every clip carries `safe_in` and `safe_out`: how far its in and out points may
move without running into sound. That turns "will this cut clip a word" from
something a model has to judge into something it can read.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.media import Span

class ClipLevel(str, Enum):
    """How coarse a semantic clip is.

    Both levels answer different questions, so both are kept: `utterance` is
    the only level precise enough to cut on, while `section` is small enough
    in number to read in full.

    A topic is not a third level. A topic can gather sections that are not
    next to each other and that come from different files, so it has no span
    of its own to be a clip of; it is carried as a section's `topic` label
    instead, and the topics of a shoot are the labels its sections share.
    """

    UTTERANCE = "utterance"
    SECTION = "section"

class ClipKind(str, Enum):
    """What a semantic clip is made of.

    `ACTION` is not produced yet: telling a shot where something happens from
    a shot that merely holds still needs the motion measurement the analysis
    layer does not take. Until then such shots are `AMBIENT`.
    """

    SPEECH = "speech"
    ACTION = "action"
    AMBIENT = "ambient"
    SILENCE = "silence"
    UNUSABLE = "unusable"

class TagSource(str, Enum):
    """Where a tag's claim came from, and therefore how far it can be trusted."""

    DETECTOR = "detector"
    MODEL = "model"
    USER = "user"

class Tag(BaseModel):
    """One claim about a clip's content, with its origin.

    A measurement and a guess are worth different amounts, so what wrote a tag
    is stored alongside it rather than left to be remembered.
    """

    value: str
    source: TagSource
    model_version: Optional[str] = Field(default=None, description="What produced it, when a model did")
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SemanticClip(BaseModel):
    """One thing in the footage that can be referred to on its own."""

    id: str
    timeline_id: str
    asset_id: str
    level: ClipLevel
    parent_id: Optional[str] = Field(default=None, description="The coarser clip this one belongs to")
    source_range: Span = Field(..., description="Seconds in the source file")
    safe_in: float = Field(default=0.0, ge=0, description="Seconds the in point may move earlier before it reaches sound")
    safe_out: float = Field(default=0.0, ge=0, description="Seconds the out point may move later before it reaches sound")
    kind: ClipKind
    name: Optional[str] = Field(default=None, description="What this section is called; utterances have no name")
    topic: Optional[str] = Field(default=None, description="The subject this section belongs to, shared across files")
    text: str = Field(default="", description="What is said here for an utterance, and the summary for a section")
    description: Optional[str] = Field(default=None, description="What is on screen here, as somebody who looked wrote it")
    described_by: Optional[str] = Field(default=None, description="Who or what wrote the description")
    speaker: Optional[str] = None
    scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Measurements taken from the analysis, such as how much of the span is silent or black",
    )
    tags: List[Tag] = Field(default_factory=list, description="Claims about the content, each with its origin")

    @property
    def duration(self) -> float:
        """Length of the clip in seconds.

        Returns:
            The source range's length.
        """
        return self.source_range.end - self.source_range.start

class ClipDescription(BaseModel):
    """What somebody saw in one clip, written back onto it."""

    clip_id: str
    description: str = Field(default="", description="What is on screen, in a sentence")
    tags: List[str] = Field(default_factory=list, description="Short labels to search on, such as 海邊 or 人臉特寫")

class SectionChoice(BaseModel):
    """One section, as chosen from the candidate boundaries offered."""

    first_clip_id: str = Field(..., description="Utterance the section opens on")
    last_clip_id: str = Field(..., description="Utterance the section ends on, inclusive")
    name: str = Field(..., min_length=1, description="What this part of the footage is, in a few words")
    topic: Optional[str] = Field(
        default=None,
        description="Subject it belongs to; the same label on sections of other files groups them",
    )
    summary: str = Field(default="", description="One line on what happens in it")

class SemanticTimeline(BaseModel):
    """One build of semantic clips over a set of assets.

    A build is pinned by `input_hash`, which covers the analyses it was
    derived from and the derivation itself. Building again over unchanged
    input returns the same timeline rather than a new one, so a plan that
    refers to a clip ID keeps referring to the same moment. When the input
    does change, the new build gets a new ID, and anything written against the
    old one can be told that it is out of date instead of silently pointing
    somewhere else.
    """

    id: str
    asset_ids: List[str]
    input_hash: str = Field(..., description="Covers the analyses and the derivation that produced this build")
    derivation_version: int = Field(..., description="Bumped when the derivation itself changes")
    levels: List[ClipLevel] = Field(default_factory=list, description="Levels that have been built so far")
    sections_by: Optional[str] = Field(
        default=None,
        description="What adjudicated the section boundaries, as the caller reported it",
    )
    candidate_hash: Optional[str] = Field(
        default=None,
        description="Covers the candidate boundaries the sections were chosen from",
    )
    built_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
