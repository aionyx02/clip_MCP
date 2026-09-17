import subprocess
from typing import Dict

from app.models.job import JobStatus, RenderJob


class JobManager:
    """In-memory registry of render jobs and the processes that run them."""

    def __init__(self):
        """Initialize an empty job registry."""
        self.jobs: Dict[str, RenderJob] = {}
        self.processes: Dict[str, subprocess.Popen] = {}

    def register_job(self, project_id: str, output_path: str) -> RenderJob:
        """Create a new job in the `QUEUED` state and store it.

        Args:
            project_id: ID of the project being rendered.
            output_path: Destination path of the rendered file.

        Returns:
            The newly registered job.
        """
        job = RenderJob(project_id=project_id, output_path=output_path)
        self.jobs[job.job_id] = job
        return job

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job, stopping its process if needed.

        The process is first asked to terminate and is killed if it has not
        exited within 3 seconds.

        Args:
            job_id: ID of the job to cancel.

        Returns:
            `True` if the job was cancelled; `False` if it does not exist or
            is no longer queued or running.
        """
        job = self.jobs.get(job_id)
        if not job or job.status not in [JobStatus.RUNNING, JobStatus.QUEUED]:
            return False

        proc = self.processes.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

        job.status = JobStatus.CANCELLED
        return True
