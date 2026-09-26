import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from app.engine import loudness, resources
from app.engine.analysis import analyze_media
from app.engine.ffmpeg import OperationCancelled, run_ffmpeg
from app.models.job import Job, JobKind, JobStatus
from app.storage.repo import Repository

HEARTBEAT_INTERVAL_SECONDS = 1.0
STALE_AFTER = timedelta(seconds=30)
# A worker that has just been launched has not claimed its memory yet, so until it has
# had this long its whole estimate is counted against what the machine reports as free.
WARMUP = timedelta(seconds=60)
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
    """Admits background jobs into detached worker processes and reports their state.

    Job state lives in the repository rather than in this process: workers
    record progress and outcomes there, so jobs survive server restarts and
    can be observed from any process that shares the database.

    A job is not started when it is asked for, only queued. Renders and
    analyses are each heavy enough to fill a machine on their own — a render
    holds a decoder open per clip, and a transcription loads a speech model
    of a few gigabytes — so starting one for every call would exhaust the
    computer's memory rather than its patience. Queued jobs cost a database
    row and nothing else, and one is admitted only when both the job count
    and the free memory allow it.
    """

    def __init__(self, repo: Repository):
        """Initialize the manager.

        Args:
            repo: Repository that stores job state.
        """
        self.repo = repo

    def start_job(self, job: Job, spec: Dict[str, Any]) -> Job:
        """Store a new job and start it as soon as the machine has room for it.

        Args:
            job: New job with `work_dir` set to a directory dedicated to it,
                and `memory_estimate` set to what it is expected to need.
            spec: JSON-serializable instructions for the worker, saved in the
                job directory. Render jobs need `command` and
                `duration_seconds`; analyze jobs need `transcribe`, `language`,
                `prompt`, and `chinese_variant`.

        Returns:
            The stored job. It is `QUEUED` either way; its `stage` says what
            it is waiting for when it could not start immediately.

        Raises:
            OSError: If the job directory or its specification cannot be
                written.
        """
        os.makedirs(job.work_dir, exist_ok=True)
        with open(os.path.join(job.work_dir, SPEC_FILE_NAME), "w", encoding="utf-8") as spec_file:
            json.dump(spec, spec_file, indent=2, ensure_ascii=False)
        self.repo.add_job(job)
        self.launch_ready()
        return self.repo.get_job(job.job_id) or job

    def launch_ready(self) -> List[Job]:
        """Start workers for as many queued jobs as the machine can afford.

        Jobs are admitted oldest first and never out of order, so a small job
        cannot keep jumping ahead of a large one that is waiting for memory.
        The jobs that stay queued have their `stage` set to what they are
        waiting for, which is what `get_job` reports back.

        Returns:
            The jobs that were admitted and had a worker launched for them.
        """
        admitted = [job for job in self.repo.update_active_jobs(self._plan) if job.admitted_at is not None]
        for job in admitted:
            self._launch_worker(job)
        return admitted

    def _plan(self, jobs: List[Job]) -> List[str]:
        """Decide which queued jobs may start, and why the rest may not.

        Runs inside the repository's transaction, so the view of what is
        already running cannot change while the decision is being made.

        Args:
            jobs: Every queued and running job, oldest first. Jobs that are
                admitted or left waiting are modified in place.

        Returns:
            The IDs of the jobs that were changed.
        """
        now = datetime.now(timezone.utc)
        # A job whose worker stopped reporting has lost its memory back to the machine,
        # so it no longer holds a slot; `get_job` fails it the next time it is asked for.
        running = [
            job for job in jobs
            if (job.status == JobStatus.RUNNING or job.admitted_at is not None)
            and now - job.updated_at <= STALE_AFTER
        ]
        slots = resources.max_concurrent_jobs() - len(running)
        spare = resources.spare_bytes()
        if spare is not None:
            spare -= sum(
                job.memory_estimate for job in running
                if job.admitted_at is not None and now - job.admitted_at < WARMUP
            )

        changed, busy, behind = [], len(running), 0
        for job in jobs:
            if job.admitted_at is not None or job.status != JobStatus.QUEUED or job.cancel_requested:
                continue
            if behind:
                reason = f"queued behind {behind} job(s)"
            elif slots <= 0:
                reason = f"waiting for {len(running)} running job(s) to finish"
            # Nothing at all is running and this job still does not fit: start it anyway.
            # Refusing every job on a small machine is worse than running one job slowly,
            # and there is nothing whose finishing could free the memory it is waiting for.
            elif spare is not None and busy and job.memory_estimate > spare:
                reason = f"waiting for {job.memory_estimate // resources.MEGABYTE} MB of free memory"
            else:
                job.admitted_at, job.stage = now, None
                slots, busy = slots - 1, busy + 1
                if spare is not None:
                    spare = max(0, spare - job.memory_estimate)
                changed.append(job.job_id)
                continue
            behind += 1
            if job.stage != reason:
                job.stage = reason
                changed.append(job.job_id)
        return changed

    def _launch_worker(self, job: Job) -> None:
        """Start the detached worker process for an admitted job.

        Args:
            job: Job that was just admitted. It is recorded as failed if its
                worker cannot be started, which frees the slot again.
        """
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

    def get_job(self, job_id: str) -> Optional[Job]:
        """Fetch a job, first failing it if its worker has stopped reporting.

        Workers refresh an active job every `HEARTBEAT_INTERVAL_SECONDS`. A
        job whose worker has started but has not been updated within
        `STALE_AFTER` is assumed to have lost it, for example because the
        process was killed. A job still waiting for a free slot has no worker
        yet and is never failed for being quiet.

        Polling also moves the queue along, so a slot freed by a job that
        finished is filled without waiting for the next request.

        Args:
            job_id: ID of the job to fetch.

        Returns:
            The job's latest state, or `None` if it does not exist.
        """
        job = self.repo.get_job(job_id)
        if job is None:
            return None
        if job.status.is_active and job.admitted_at is not None and datetime.now(timezone.utc) - job.updated_at > STALE_AFTER:
            self.repo.update_job(job_id, _fail_if_active("worker stopped reporting; it may have been terminated"))
        self.launch_ready()
        return self.repo.get_job(job_id)

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
        # Whatever the cancelled job was holding is free again; let the queue use it.
        self.launch_ready()
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
        spec: Job specification with `command` and `duration_seconds`, and
            for a cached preview `pieces`: pictures to render into the cache
            first, each with its own `command`, `part`, `path` and
            `duration_seconds`. With `loudness` — its `mix_command`, the
            `mix_path` it writes, and the `target` — the mix is rendered and
            measured first, and the gain it settles on is written into
            `command`.
        context: Progress and cancellation for the job.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If FFmpeg fails.
    """
    log = os.path.join(job.work_dir, FFMPEG_LOG_FILE_NAME)
    pieces = spec.get("pieces", [])
    # The bar is shared out by seconds of picture: the pieces are rendered at full cost,
    # the final pass only puts them together, so it is weighted at a third of its length.
    weights = [float(piece["duration_seconds"]) for piece in pieces] + [float(spec["duration_seconds"]) / 3]
    total = sum(weights) or 1.0
    done = 0.0
    for number, piece in enumerate(pieces, start=1):
        start, share = done / total, weights[number - 1] / total
        context.report(start, f"rendering picture {number} of {len(pieces)}")
        try:
            run_ffmpeg(
                piece["command"], log, float(piece["duration_seconds"]),
                lambda fraction, start=start, share=share: context.report(start + fraction * share),
                context.is_cancelled,
            )
        except BaseException:
            # A half-written picture must never be found by the next render.
            if os.path.exists(piece["part"]):
                os.remove(piece["part"])
            raise
        # Moved into place only once whole. Another render may have finished the same
        # picture meanwhile; either copy is the same picture.
        os.replace(piece["part"], piece["path"])
        done += weights[number - 1]
    command = spec["command"]
    level = spec.get("loudness")
    if level is not None:
        context.report(done / total, "measuring loudness")
        try:
            run_ffmpeg(level["mix_command"], log, 0.0, lambda fraction: None, context.is_cancelled)
            gain = loudness.settle_gain(level["mix_path"], float(level["target"]))
        finally:
            if os.path.exists(level["mix_path"]):
                os.remove(level["mix_path"])
        command = loudness.with_gain(command, gain)
    start, share = done / total, weights[-1] / total
    context.report(start, "rendering")
    run_ffmpeg(
        command,
        log,
        float(spec["duration_seconds"]),
        lambda fraction: context.report(start + fraction * share),
        context.is_cancelled,
    )

def _run_analysis(repo: Repository, job: Job, spec: Dict[str, Any], context: JobContext) -> None:
    """Analyze an asset and store the result.

    Args:
        repo: Repository holding the asset and receiving the analysis.
        job: Analyze job being run.
        spec: Job specification with `transcribe`, `language`, `prompt`,
            `chinese_variant`, `diarize`, and `speakers`.
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
        diarize=spec.get("diarize", True),
        speakers=spec.get("speakers"),
    )
    repo.save_analysis(analysis)

def run_worker(repo: Repository, job_id: str) -> None:
    """Run a queued job to completion and record its outcome.

    Returns without doing anything if the job is missing or no longer queued.
    A render that fails or is cancelled has its partial output file deleted.
    Whatever queued job the freed memory allows is started before returning.

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
    # This worker's slot and its memory are free now, so hand them to whatever is waiting
    # rather than leaving the queue still until someone next asks about a job.
    JobManager(repo).launch_ready()

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
