import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)

class JobKind(str, Enum):
    """Kinds of background work a job can perform."""

    RENDER = "render"
    ANALYZE = "analyze"

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
    work_dir: Optional[str] = Field(default=None, description="Directory holding the job's specification and logs")
    error_message: Optional[str] = None
    cancel_requested: bool = False
    memory_estimate: int = Field(default=0, ge=0, description="Peak memory the job is expected to need, in bytes")
    created_at: datetime = Field(default_factory=_utc_now)
    admitted_at: Optional[datetime] = Field(
        default=None,
        description="When a worker was launched for this job; null while it waits for a free slot",
    )
    updated_at: datetime = Field(default_factory=_utc_now, description="Last time the job record was written")
