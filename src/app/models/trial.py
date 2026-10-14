"""A change to a cut put to a blind test before it is kept.

The AI that made a change is the worst judge of it: it reads the result through
the reason it wrote. A trial keeps the reason away from whoever judges. The
problem and the evidence for it are written down first, with what the change
was meant to do; the cut before and after are rendered and handed over as X and
Y, which is which known only here; the verdict is written down before it is
said which was the change; and the change is kept only if it made the problem
better and nothing around it worse.
"""

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    """What whoever judged said, before they knew which was the change."""

    better: Literal["X", "Y", "same"] = Field(..., description="Which handles the problem better, or neither")
    broke: bool = Field(..., description="Whether either made what is around the problem worse")
    why: str = Field(..., min_length=1, description="What they saw and heard, in their words")
    judged_by: Literal["subagent", "user", "same_ai"] = Field(
        ..., description="A fresh subagent that saw none of the change, the user, or the AI that made it",
    )
    judged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Trial(BaseModel):
    """One change to one problem, tried blind."""

    id: str
    project_id: str
    problem: str = Field(..., description="What is wrong, where, as it was seen or heard")
    evidence: str = Field(default="", description="What shows it: a join's words and levels, a comment, a finding")
    hypothesis: str = Field(..., description="What the change does and why it should help; never shown to the judge")
    expected: str = Field(..., description="What should be better afterwards; never shown to the judge")
    at: float = Field(..., ge=0, description="Where the problem starts in the cut before, in seconds")
    end: float = Field(..., gt=0, description="Where it ends")
    before_job: str
    before_version: Optional[int] = None
    candidate_at: Optional[float] = Field(default=None, description="Where the stretch starts in the cut after")
    candidate_end: Optional[float] = None
    after_job: Optional[str] = None
    after_version: Optional[int] = None
    candidate_is: Optional[Literal["X", "Y"]] = Field(default=None, description="Which label the change has")
    verdict: Optional[Verdict] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
