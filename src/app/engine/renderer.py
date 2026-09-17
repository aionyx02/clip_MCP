import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from app.engine.analysis import analyze_media
from app.engine.ffmpeg import OperationCancelled, run_ffmpeg
from app.models.job import Job, JobKind, JobStatus
from app.storage.repo import Repository

HEARTBEAT_INTERVAL_SECONDS = 1.0
STALE_AFTER = timedelta(seconds=30)
SPEC_FILE_NAME = "job.json"
FFMPEG_LOG_FILE_NAME = "ffmpeg.log"
WORKER_LOG_FILE_NAME = "worker.log"

def _detached_process_options() -> Dict[str, Any]:
    """Build `subprocess.Popen` options that detach a child from this process.

    The child gets no console window on Windows and its own session or
    process group, so it keeps running if the server's console is closed.

    Returns:
        Keyword arguments for `subprocess.Popen`.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}

def _fail_if_active(message: str) -> Callable[[Job], None]:
    """Build a job change that marks an unfinished job as failed.

    Args:
        message: Error message to record on the job.

    Returns:
        A change function for `Repository.update_job`. It leaves jobs that
        have already reached a terminal state untouched.
    """
    def change(job: Job) -> None:
        """Mark the job as failed if it is still queued or running."""
        if job.status.is_active:
            job.status = JobStatus.FAILED
            job.error_message = message
    return change

class JobManager:
    """Starts background jobs in detached worker processes and reports their state.

    Job state lives in the repository rather than in this process: workers
    record progress and outcomes there, so jobs survive server restarts and
    can be observed from any process that shares the database.
    """

    def __init__(self, repo: Repository):
        """Initialize the manager.

        Args:
            repo: Repository that stores job state.
        """
        self.repo = repo

    def start_job(self, job: Job, spec: Dict[str, Any]) -> Job:
        """Store a new job and launch a worker process that runs it.

        Args:
            job: New job with `work_dir` set to a directory dedicated to it.
            spec: JSON-serializable instructions for the worker, saved in the
                job directory. Render jobs need `command` and
                `duration_seconds`; analyze jobs need `transcribe`, `language`,
                `prompt`, and `chinese_variant`.

        Returns:
            The stored job, in the `QUEUED` state.

        Raises:
            OSError: If the job directory cannot be written or the worker
                cannot be started. The job is recorded as failed in the latter
                case.
        """
        os.makedirs(job.work_dir, exist_ok=True)
        with open(os.path.join(job.work_dir, SPEC_FILE_NAME), "w", encoding="utf-8") as spec_file:
            json.dump(spec, spec_file, indent=2, ensure_ascii=False)
        self.repo.add_job(job)

        try:
            with open(os.path.join(job.work_dir, WORKER_LOG_FILE_NAME), "wb") as worker_log:
                subprocess.Popen(
                    [sys.executable, "-m", "app.engine.renderer", self.repo.db_path, job.job_id],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=worker_log,
                    close_fds=True,
                    **_detached_process_options(),
                )
        except OSError as exc:
            self.repo.update_job(job.job_id, _fail_if_active(f"could not start worker: {exc}"))
            raise
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Fetch a job, first failing it if its worker has stopped reporting.

        Workers refresh an active job every `HEARTBEAT_INTERVAL_SECONDS`. An
        active job not updated within `STALE_AFTER` is assumed to have lost
        its worker, for example because the process was killed.

        Args:
            job_id: ID of the job to fetch.

        Returns:
            The job's latest state, or `None` if it does not exist.
        """
        job = self.repo.get_job(job_id)
        if job is not None and job.status.is_active and datetime.now(timezone.utc) - job.updated_at > STALE_AFTER:
            job = self.repo.update_job(job_id, _fail_if_active("worker stopped reporting; it may have been terminated"))
        return job

    def cancel_job(self, job_id: str, wait_seconds: float = 5.0) -> Optional[Job]:
        """Cancel a queued or running job.

        A queued job is cancelled immediately. A running job is flagged for
        cancellation, and this method waits up to `wait_seconds` for its
        worker to stop and confirm.

        Args:
            job_id: ID of the job to cancel.
            wait_seconds: Maximum time to wait for a running job to stop.

        Returns:
            The job's latest state, or `None` if it does not exist. The status
            may still be `running` if the worker did not respond in time.
        """
        def request(job: Job) -> None:
            """Cancel a queued job or flag a running one."""
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
            elif job.status == JobStatus.RUNNING:
                job.cancel_requested = True

        job = self.repo.update_job(job_id, request)
        deadline = time.monotonic() + wait_seconds
        while job is not None and job.status == JobStatus.RUNNING and time.monotonic() < deadline:
            time.sleep(0.2)
            job = self.repo.get_job(job_id)
        return job

class JobContext:
    """Reports a running job's progress and relays cancellation requests.

    While active, a background thread writes the latest progress and stage to
    the repository every `HEARTBEAT_INTERVAL_SECONDS`. The writes double as
    the job's heartbeat, and each one picks up cancellation requested by any
    process.
    """

    def __init__(self, repo: Repository, job_id: str):
        """Initialize the context.

        Args:
            repo: Repository that stores the job.
            job_id: ID of the running job.
        """
        self._repo = repo
        self._job_id = job_id
        self._progress = 0.0
        self._stage: Optional[str] = None
        self._cancelled = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)

    def __enter__(self) -> "JobContext":
        """Start sending heartbeats.

        Returns:
            This context.
        """
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        """Stop sending heartbeats and wait for the last one to be written."""
        self._stopped.set()
        self._thread.join()

    def report(self, progress: float, stage: Optional[str] = None) -> None:
        """Record progress to include in the next heartbeat.

        Progress never decreases and stays below 1.0 until the job finishes.

        Args:
            progress: Fraction of the work completed.
            stage: Short description of the current step; `None` keeps the
                previous one.
        """
        self._progress = max(self._progress, min(progress, 0.99))
        if stage is not None:
            self._stage = stage

    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested.

        Returns:
            `True` once any process has asked to cancel the job.
        """
        return self._cancelled.is_set()

    def _heartbeat(self) -> None:
        """Write progress and poll for cancellation until the context exits."""
        def write(job: Job) -> None:
            """Copy the latest progress and stage onto the job."""
            job.progress = max(job.progress, self._progress)
            if self._stage is not None:
                job.stage = self._stage

        while True:
            job = self._repo.update_job(self._job_id, write)
            if job is not None and job.cancel_requested:
                self._cancelled.set()
            if self._stopped.wait(HEARTBEAT_INTERVAL_SECONDS):
                return

def _run_render(job: Job, spec: Dict[str, Any], context: JobContext) -> None:
    """Render a project with the FFmpeg command stored in the job specification.

    Args:
        job: Render job being run.
        spec: Job specification with `command` and `duration_seconds`.
        context: Progress and cancellation for the job.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If FFmpeg fails.
    """
    context.report(0.0, "rendering")
    run_ffmpeg(
        spec["command"],
        os.path.join(job.work_dir, FFMPEG_LOG_FILE_NAME),
        float(spec["duration_seconds"]),
        context.report,
        context.is_cancelled,
    )

def _run_analysis(repo: Repository, job: Job, spec: Dict[str, Any], context: JobContext) -> None:
    """Analyze an asset and store the result.

    Args:
        repo: Repository holding the asset and receiving the analysis.
        job: Analyze job being run.
        spec: Job specification with `transcribe`, `language`, `prompt`, and
            `chinese_variant`.
        context: Progress and cancellation for the job.

    Raises:
        OperationCancelled: If cancellation was requested.
        ValueError: If the asset no longer exists.
        RuntimeError: If FFmpeg or speech recognition fails.
    """
    asset = repo.get_assets([job.asset_id]).get(job.asset_id)
    if asset is None:
        raise ValueError(f"asset {job.asset_id} not found")
    analysis = analyze_media(
        asset,
        job.work_dir,
        transcribe=spec["transcribe"],
        language=spec.get("language"),
        prompt=spec.get("prompt"),
        chinese_variant=spec.get("chinese_variant"),
        on_progress=context.report,
        is_cancelled=context.is_cancelled,
    )
    repo.save_analysis(analysis)

def run_worker(repo: Repository, job_id: str) -> None:
    """Run a queued job to completion and record its outcome.

    Returns without doing anything if the job is missing or no longer queued.
    A render that fails or is cancelled has its partial output file deleted.

    Args:
        repo: Repository that stores the job.
        job_id: ID of the job to run.
    """
    job = repo.get_job(job_id)
    if job is None or job.work_dir is None:
        return
    with open(os.path.join(job.work_dir, SPEC_FILE_NAME), encoding="utf-8") as spec_file:
        spec = json.load(spec_file)

    started = False

    def start(current: Job) -> None:
        """Move the job to running unless it was cancelled while queued."""
        nonlocal started
        if current.status == JobStatus.QUEUED and not current.cancel_requested:
            current.status = JobStatus.RUNNING
            started = True

    repo.update_job(job_id, start)
    if not started:
        return

    status, error_message = JobStatus.COMPLETED, None
    with JobContext(repo, job_id) as context:
        try:
            if job.kind == JobKind.RENDER:
                _run_render(job, spec, context)
            else:
                _run_analysis(repo, job, spec, context)
        except OperationCancelled:
            status = JobStatus.CANCELLED
        except Exception as exc:
            status, error_message = JobStatus.FAILED, str(exc) or type(exc).__name__

    if job.kind == JobKind.RENDER and status != JobStatus.COMPLETED and job.output_path:
        # Remove the partial file so it cannot be mistaken for a finished render.
        try:
            os.remove(job.output_path)
        except OSError:
            pass

    def finish(current: Job) -> None:
        """Record the job's final status."""
        current.status = status
        current.error_message = error_message
        current.stage = None
        if status == JobStatus.COMPLETED:
            current.progress = 1.0

    repo.update_job(job_id, finish)

def main(argv: Optional[List[str]] = None) -> None:
    """Run the worker process for one job.

    Usage: `python -m app.engine.renderer <db_path> <job_id>`. Any unexpected
    error is recorded on the job before it propagates.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]`.
    """
    db_path, job_id = argv if argv is not None else sys.argv[1:]
    repo = Repository(db_path)
    try:
        run_worker(repo, job_id)
    except Exception as exc:
        repo.update_job(job_id, _fail_if_active(f"worker crashed: {exc}"))
        raise

if __name__ == "__main__":
    main()
