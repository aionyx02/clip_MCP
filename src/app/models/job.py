import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)

# Provisional (roadmap §13): how far a job has to have gone before its pace says anything
# about the rest of it.
ESTIMATE_FROM_PROGRESS = 0.05

class JobKind(str, Enum):
    """Kinds of background work a job can perform."""

    RENDER = "render"
    ANALYZE = "analyze"
    # Proxies, repaired sound and a measured loudness, so the browser can play a cut.
    PREPARE = "prepare"

class JobStatus(str, Enum):
    """Lifecycle states of a background job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_active(self) -> bool:
        """Whether the job has not reached a terminal state yet.

        Returns:
            `True` for `QUEUED` and `RUNNING`; `False` otherwise.
        """
        return self in (JobStatus.QUEUED, JobStatus.RUNNING)

class FailureKind(str, Enum):
    """What kind of thing ended a job that did not complete, decided where the cause was known.

    There is no kind for a plan that does not hold: the plan, the project and
    the check are refused by the tool that would start a job, before any job
    exists.
    """

    # A file of the library the job reads was missing or could not be opened.
    INPUT = "input"
    # A file the server made for the job — a picture from the cache, the chapters, the words
    # drawn on — was missing; starting the job again makes it anew.
    MADE = "made"
    # FFmpeg ran and failed, with every file it reads there to be read.
    RENDER = "render"
    # The worker never started, crashed, or stopped reporting.
    WORKER = "worker"
    CANCELLED = "cancelled"

class Job(BaseModel):
    """A background job, such as a render or a media analysis, and its most recently recorded state."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: JobKind = JobKind.RENDER
    project_id: Optional[str] = Field(default=None, description="Project being rendered (render jobs)")
    asset_id: Optional[str] = Field(default=None, description="Asset being analyzed (analyze jobs)")
    status: JobStatus = JobStatus.QUEUED
    progress: float = Field(default=0.0, ge=0, le=1, description="Fraction of the work completed so far")
    stage: Optional[str] = Field(default=None, description="What the job is currently doing")
    output_path: Optional[str] = Field(default=None, description="Rendered file (render jobs)")
    project_version: Optional[int] = Field(
        default=None, description="The project's version when it was rendered, to tell whether it has changed since",
    )
    captioned: bool = Field(default=False, description="Whether the render burns the captions in (render jobs)")
    frame: Optional[str] = Field(
        default=None, description="The shape it was rendered in, when not the project's own (render jobs)",
    )
    preview: bool = Field(default=False, description="Whether the render is a preview rather than the finished video")
    work_dir: Optional[str] = Field(default=None, description="Directory holding the job's specification and logs")
    spots: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Where every frame of a finished render was read and found black, frozen or a flash: "
                    "`{kind, start, end}` in seconds of the render; null when it was not read (render jobs)",
    )
    sync: Optional[Dict[str, Any]] = Field(
        default=None,
        description="What a finished render's sound was found doing against its picture: `streams`, how far the "
                    "two streams' starts and lengths differ or null when they line up; `joins`, "
                    "`{at, clip_id, side, lag}` for each side of a join whose voicing could be lined up, lag in "
                    "seconds and positive when the sound is late; `sides_asked`, how many sides were offered to "
                    "the check; `unlined`, how many of those could not be lined up, as `{faded, other}`; "
                    "`left_out`, how many were left out, as `{not_listened, too_short, faded}`; and "
                    "`frame`, one frame of the render in seconds. Null when it was not checked (render jobs)",
    )
    error_message: Optional[str] = None
    failure_kind: Optional[FailureKind] = Field(
        default=None,
        description="What ended a failed or cancelled job; null when it did not end that way, or when a "
                    "failure's cause is none of the kinds",
    )
    cancel_requested: bool = False
    memory_estimate: int = Field(default=0, ge=0, description="Peak memory the job is expected to need, in bytes")
    priority: int = Field(
        default=0,
        description="Lower goes first; jobs of one priority go oldest first. Preparing playback waits behind "
                    "renders and analyses unless somebody is waiting to watch",
    )
    created_at: datetime = Field(default_factory=_utc_now)
    admitted_at: Optional[datetime] = Field(
        default=None,
        description="When a worker was launched for this job; null while it waits for a free slot",
    )
    updated_at: datetime = Field(default_factory=_utc_now, description="Last time the job record was written")

    def remaining_seconds(self, now: datetime) -> Optional[int]:
        """Say how long a running job has left, from how fast it has gone so far.

        The same figure for the AI and for the editor's panel, so the user is
        not told two times. Honest once the job has gone some way, so not before.

        Args:
            now: The time to count from.

        Returns:
            Seconds, or None while it is not running or has not gone far enough.
        """
        if self.status != JobStatus.RUNNING or self.progress < ESTIMATE_FROM_PROGRESS:
            return None
        spent = (now - (self.admitted_at or self.created_at)).total_seconds()
        return round(spent * (1 - self.progress) / self.progress)
