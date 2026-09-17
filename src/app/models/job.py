import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states of a background render job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RenderJob(BaseModel):
    """A background render job and its most recently observed state."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    output_path: Optional[str] = None
    error_message: Optional[str] = None
