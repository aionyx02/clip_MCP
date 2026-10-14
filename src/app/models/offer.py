"""Songs put to the user to hear under a stretch of the cut, and pick from.

Whether a song suits the pictures is heard, and the AI cannot hear: it narrows
the music down to a few, and the user listens to each under the same stretch
in the editor and chooses. The choice comes back to the AI as a comment, the
way everything else the user says about the cut does.
"""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class OfferedSong(BaseModel):
    """One song to hear, from where in it."""

    asset_id: str = Field(..., description="The song, from the music library")
    start: float = Field(default=0.0, ge=0, description="Which second of the song to hear it from")
    why: str = Field(default="", description="What it is meant to do there, in a few words, for the user to read")


class SongOffer(BaseModel):
    """The songs put to the user for one stretch of a cut."""

    project_id: str
    start: float = Field(..., ge=0, description="Where the stretch begins, in seconds of the cut")
    end: float = Field(..., gt=0, description="Where it ends")
    volume: float = Field(default=0.35, ge=0, description="How loud each is played, as the music would be")
    songs: List[OfferedSong]
    offered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    beat_id: Optional[str] = Field(default=None, description="The part of the plan the stretch is, if it is one")
