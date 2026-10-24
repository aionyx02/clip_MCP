import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.engine import loudness, machine, models, resources, scan, speed
from app.engine.analysis import analyze_media, whisper_model_for
from app.engine.ffmpeg import FFmpegFailed, OperationCancelled, run_ffmpeg
from app.models.job import FailureKind, Job, JobKind, JobStatus
from app.storage import housekeeping
from app.storage.repo import Repository

HEARTBEAT_INTERVAL_SECONDS = 1.0
STALE_AFTER = timedelta(seconds=30)
# A worker that has just been launched has not claimed its memory yet, so until it has
# had this long its whole estimate is counted against what the machine reports as free.
WARMUP = timedelta(seconds=60)
SPEC_FILE_NAME = "job.json"
FFMPEG_LOG_FILE_NAME = "ffmpeg.log"
WORKER_LOG_FILE_NAME = "worker.log"

def _detached_process_options(breakaway: bool = True) -> Dict[str, Any]:
    """Build `subprocess.Popen` options that detach a child from this process.

    The child gets no console window on Windows and its own session or
    process group, so it keeps running if the server's console is closed.
    On Windows it also leaves the job object the server may have been started
    in: a client that closes its server kills everything in that job, and a
    transcription half done went with it (#88).

    Args:
        breakaway: Whether to leave the job object, which a job that does not
            allow it refuses.

    Returns:
        Keyword arguments for `subprocess.Popen`.
    """
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        return {"creationflags": flags | (subprocess.CREATE_BREAKAWAY_FROM_JOB if breakaway else 0)}
    return {"start_new_session": True}

ABANDONED = "worker stopped reporting; it may have been terminated"
# What a background preparation says while an export runs.
EXPORT_FIRST = "waiting for the export to finish"

def _abandoned(job: Job, now: datetime) -> bool:
    """Whether a job's worker is gone: it was running, or let in to run, and has gone quiet.

    A job still waiting for a slot has no worker, and is never abandoned for
    being quiet. Running without `admitted_at` is a job written before jobs
    were admitted; it has a worker like any other, and loses it the same way.

    Args:
        job: The job.
        now: The time to judge by.

    Returns:
        True when nothing has written to it for longer than `STALE_AFTER`.
    """
    started = job.status == JobStatus.RUNNING or (job.status.is_active and job.admitted_at is not None)
    return started and now - job.updated_at > STALE_AFTER

def _fail_if_active(message: str) -> Callable[[Job], None]:
    """Build a job change that marks an unfinished job as failed for want of its worker.

    Every caller knows the worker is what went: it could not be started,
    crashed, or stopped reporting. A job the user had already asked to stop
    is cancelled rather than failed: it ended as asked, and is not one to
    run again.

    Args:
        message: Error message to record on the job.

    Returns:
        A change function for `Repository.update_job`. It leaves jobs that
        have already reached a terminal state untouched.
    """
    def change(job: Job) -> None:
        """Mark the job as failed, or cancelled when that was asked for, if it is still queued or running."""
        if job.status.is_active and job.cancel_requested:
            job.status = JobStatus.CANCELLED
            job.failure_kind = FailureKind.CANCELLED
        elif job.status.is_active:
            job.status = JobStatus.FAILED
            job.error_message = message
            job.failure_kind = FailureKind.WORKER
    return change

class InputUnavailable(Exception):
    """Raised when a file a job reads is missing or cannot be opened."""

def _unreadable(paths: List[str]) -> Optional[str]:
    """Say which of a job's files cannot be opened for reading, if any.

    Args:
        paths: The files the job reads.

    Returns:
        Why the first one that cannot be read cannot, or None when all can.
    """
    for path in paths:
        try:
            with open(path, "rb"):
                pass
        except OSError as exc:
            return f"could not read {path}: {exc.strerror or exc}"
    return None

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

    def launch_ready(self, keep: Optional[JobKind] = None) -> Tuple[List[Job], Optional[Job]]:
        """Start workers for as many queued jobs as the machine can afford.

        Jobs are admitted oldest first and never out of order, so a small job
        cannot keep jumping ahead of a large one that is waiting for memory.
        The jobs that stay queued have their `stage` set to what they are
        waiting for, which is what `get_job` reports back.

        Args:
            keep: A kind of job whose first admitted job is returned instead of
                being launched, so that a worker already holding the models
                can run it in this process. It is admitted by `_plan` like any
                other, so it takes its place in the queue.

        Returns:
            `(launched, kept)`: the admitted jobs a worker was launched for,
            and the admitted job held back for the caller, if any.
        """
        admitted = [job for job in self.repo.update_active_jobs(self._plan)
                    if job.status == JobStatus.QUEUED and job.admitted_at is not None]
        launched: List[Job] = []
        kept: Optional[Job] = None
        for job in admitted:
            if kept is None and keep is not None and job.kind == keep:
                kept = job
                continue
            self._launch_worker(job)
            launched.append(job)
        return launched, kept

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
        # A job whose worker stopped reporting has given its memory back to the machine:
        # it is failed here, and holds no slot.
        changed = []
        for job in jobs:
            if _abandoned(job, now):
                _fail_if_active(ABANDONED)(job)
                changed.append(job.job_id)
        running = [
            job for job in jobs
            if job.status.is_active and (job.status == JobStatus.RUNNING or job.admitted_at is not None)
        ]
        slots = resources.max_concurrent_jobs() - len(running)
        spare = resources.spare_bytes()
        if spare is not None:
            spare -= sum(
                job.memory_estimate for job in running
                if job.admitted_at is not None and now - job.admitted_at < WARMUP
            )

        busy, behind = len(running), 0
        def exports(job: Job) -> bool:
            """Whether a job is a finished video being made, which background preparation waits for."""
            return job.kind == JobKind.RENDER and not job.preview

        exporting = any(exports(job) for job in running)
        # Oldest first within a priority: preparing playback waits behind the work somebody asked for.
        for job in sorted(jobs, key=lambda queued: queued.priority):
            if job.admitted_at is not None or job.status != JobStatus.QUEUED or job.cancel_requested:
                continue
            if exporting and job.kind == JobKind.PREPARE and job.priority > 0:
                # Getting a version ready to watch in the background never slows an export down.
                # Nobody is waiting on it, so nothing waits behind it either.
                if job.stage != EXPORT_FIRST:
                    job.stage = EXPORT_FIRST
                    changed.append(job.job_id)
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
                exporting = exporting or exports(job)
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
                for breakaway in (True, False):
                    try:
                        subprocess.Popen(
                            [sys.executable, "-m", "app.engine.renderer", self.repo.db_path, job.job_id],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=worker_log,
                            close_fds=True,
                            **_detached_process_options(breakaway),
                        )
                        break
                    except PermissionError:
                        # A job object that does not allow leaving it: the worker stays in it.
                        if not breakaway or os.name != "nt":
                            raise
        except OSError as exc:
            self.repo.update_job(job.job_id, _fail_if_active(f"could not start worker: {exc}"))

    def forget_abandoned(self) -> None:
        """Fail every job whose worker is gone, so nothing waits on it.

        Called before deciding anything on whether a job is still running —
        updating, clearing storage, removing a file — since a job whose
        worker was killed stays `running` until something says otherwise.
        """
        now = datetime.now(timezone.utc)

        def fail(jobs: List[Job]) -> List[str]:
            gone = [job for job in jobs if _abandoned(job, now)]
            for job in gone:
                _fail_if_active(ABANDONED)(job)
            return [job.job_id for job in gone]

        self.repo.update_active_jobs(fail)

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
        if _abandoned(job, datetime.now(timezone.utc)):
            self.repo.update_job(job_id, _fail_if_active(ABANDONED))
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
                job.failure_kind = FailureKind.CANCELLED
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

def _run_render(job: Job, spec: Dict[str, Any], context: JobContext, repo: Optional[Repository] = None) -> None:
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
        repo: Where this computer's pace at rendering is kept: the bar is
            shared out by it, and a finished render adds to it. Left out, the
            provisional proportions, and nothing is kept.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If FFmpeg fails.
    """
    log = os.path.join(job.work_dir, FFMPEG_LOG_FILE_NAME)
    pieces = spec.get("pieces", [])
    level = spec.get("loudness")
    length = float(spec["duration_seconds"])
    # The bar is shared out by how long each part is expected to take here, so that how far
    # it has come says how much time has gone: guessed from seconds of picture at first, a
    # finished video's one long pass at a third of a piece's cost, until this computer has
    # rendered something of the same kind and its own pace can be used.
    parts = (["pieces"] if pieces else []) + (["loudness"] if level is not None else []) + ["final", "scan"]
    computer = machine.identity() if repo is not None else ""
    costs = (speed.render_costs(repo, computer, job.preview, parts) if repo is not None
             else {part: speed.RENDER_COST[part] for part in parts})
    weights = [float(piece["duration_seconds"]) * costs.get("pieces", 0.0) for piece in pieces]
    weights.append(length * costs["final"])
    measuring = length * costs["loudness"] if level is not None else 0.0
    scanning = length * costs["scan"]
    total = (sum(weights) + measuring + scanning) or 1.0
    done = 0.0
    taken: Dict[str, Tuple[float, float]] = {}
    began = time.monotonic()
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
    if pieces:
        taken["pieces"] = (time.monotonic() - began, sum(float(piece["duration_seconds"]) for piece in pieces))
    command = spec["command"]
    if level is not None:
        began = time.monotonic()
        start, share = done / total, measuring / total
        context.report(start, "measuring loudness")
        try:
            run_ffmpeg(
                level["mix_command"], log, float(spec["duration_seconds"]),
                lambda fraction, start=start, share=share: context.report(start + fraction * share),
                context.is_cancelled,
            )
            gain = loudness.settle_gain(level["mix_path"], float(level["target"]))
        finally:
            if os.path.exists(level["mix_path"]):
                os.remove(level["mix_path"])
        command = loudness.with_gain(command, gain)
        done += measuring
        taken["loudness"] = (time.monotonic() - began, length)
    start, share = done / total, weights[-1] / total
    context.report(start, "rendering")
    began = time.monotonic()
    run_ffmpeg(
        command,
        log,
        length,
        lambda fraction: context.report(start + fraction * share),
        context.is_cancelled,
    )
    taken["final"] = (time.monotonic() - began, length)
    # Every frame read once more, for a flaw too short for a look at a few of them to catch.
    start, share = (done + weights[-1]) / total, scanning / total
    context.report(start, "checking every frame")
    began = time.monotonic()
    # The video is made by now: a scan that fails, or is cancelled, leaves it unscanned, never lost.
    try:
        run_ffmpeg(
            scan.command(command[0], spec["render_path"], job.work_dir),
            os.path.join(job.work_dir, scan.LOG_FILE_NAME),
            length,
            lambda fraction: context.report(start + fraction * share),
            context.is_cancelled,
        )
        scan.keep(job.work_dir, length)
        taken["scan"] = (time.monotonic() - began, length)
    except (RuntimeError, OSError, OperationCancelled):
        pass
    if repo is not None:
        speed.record_render(repo, computer, job.preview, taken)
    if spec.get("deliver"):
        # A finished video is the user's: it leaves the workspace for where they keep
        # their videos, under its own name, never over one already there.
        context.report(1.0, "saving to the videos folder")
        spec["delivered"] = housekeeping.deliver(spec["render_path"], spec["deliver"])

def _run_prepare(job: Job, spec: Dict[str, Any], context: JobContext) -> None:
    """Make what the browser needs to play a cut: each file in `steps`, written whole or not at all.

    A step is a command writing `part`, which is moved to `path` once it is
    whole; a step with `measure` instead renders the mix to `part` and keeps
    the gain to `target` loudness in `path`, as JSON.

    Args:
        job: The job.
        spec: Its `steps`, each with `command`, `part`, `path`, `seconds`
            and `stage`, and for a measurement `measure` and `target`.
        context: Progress and cancellation for the job.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If FFmpeg fails.
    """
    log = os.path.join(job.work_dir, FFMPEG_LOG_FILE_NAME)
    steps = spec.get("steps", [])
    total = sum(float(step["seconds"]) for step in steps) or 1.0
    done = 0.0
    for step in steps:
        if os.path.exists(step["path"]):
            done += float(step["seconds"])
            continue
        start, share = done / total, float(step["seconds"]) / total
        context.report(start, step["stage"])
        try:
            run_ffmpeg(step["command"], log, float(step["seconds"]),
                       lambda fraction, start=start, share=share: context.report(start + fraction * share),
                       context.is_cancelled)
            if step.get("measure"):
                lufs, _ = loudness.measure(step["part"])
                gain = 0.0 if lufs <= loudness.SILENT_LUFS else round(float(step["target"]) - lufs, 2)
                part = step["path"] + ".part"
                with open(part, "w", encoding="utf-8") as kept:
                    json.dump({"gain_db": gain, "lufs": lufs}, kept)
                os.replace(part, step["path"])
            else:
                os.replace(step["part"], step["path"])
        finally:
            if os.path.exists(step["part"]):
                os.remove(step["part"])
        done += float(step["seconds"])

def _run_analysis(repo: Repository, job: Job, spec: Dict[str, Any], context: JobContext) -> None:
    """Analyze an asset and store the result.

    Args:
        repo: Repository holding the asset and receiving the analysis.
        job: Analyze job being run.
        spec: Job specification with `transcribe`, `transcription`,
            `language`, `prompt`, `chinese_variant`, `diarize`, and `speakers`.
        context: Progress and cancellation for the job.

    Raises:
        OperationCancelled: If cancellation was requested.
        InputUnavailable: If the asset no longer exists, or its file cannot
            be read.
        RuntimeError: If FFmpeg or speech recognition fails.
    """
    asset = repo.get_assets([job.asset_id]).get(job.asset_id)
    if asset is None:
        raise InputUnavailable(f"asset {job.asset_id} not found")
    unreadable = _unreadable([asset.path])
    if unreadable:
        raise InputUnavailable(unreadable)
    runs: List[speed.Measured] = []
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
        on_measured=lambda *run: runs.append(speed.Measured(*run)),
        speech_model=whisper_model_for(spec.get("transcription", "accurate")),
        earlier=repo.get_analysis(asset.id),
    )
    repo.save_analysis(analysis)
    # Only once the analysis is kept: a cancelled or failed run says nothing about the next.
    speed.record(repo, runs, float(asset.duration or 0), machine.identity())

def _inputs(repo: Repository, job: Job, spec: Dict[str, Any]) -> List[str]:
    """The files a job reads: an analysis's file where the library has it now, otherwise its specification's `inputs`.

    Args:
        repo: Repository holding the assets.
        job: The job.
        spec: Its specification.

    Returns:
        The paths.
    """
    if job.kind == JobKind.ANALYZE:
        asset = repo.get_assets([job.asset_id]).get(job.asset_id)
        return [asset.path] if asset is not None else []
    return spec.get("inputs", [])

def run_worker(repo: Repository, job_id: str) -> Optional[str]:
    """Run a queued job to completion and record its outcome.

    Returns without doing anything if the job is missing or no longer queued.
    A render that fails or is cancelled has its partial output file deleted.
    A job that ends without completing has its `failure_kind` recorded from
    what stopped it, never from its error's words.
    Whatever queued job the freed memory allows is started before returning,
    except that after an analysis the next analysis is kept back and its ID
    returned, so this process can run it with the models already loaded.

    Args:
        repo: Repository that stores the job.
        job_id: ID of the job to run.

    Returns:
        The ID of the next analysis to run in this process, or None when this
        was not an analysis or no analysis was admitted.
    """
    job = repo.get_job(job_id)
    if job is None or job.work_dir is None:
        return None
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
        return None

    status, error_message, failure = JobStatus.COMPLETED, None, None
    with JobContext(repo, job_id) as context:
        try:
            if job.kind == JobKind.RENDER:
                _run_render(job, spec, context, repo)
            elif job.kind == JobKind.PREPARE:
                _run_prepare(job, spec, context)
            else:
                _run_analysis(repo, job, spec, context)
        except OperationCancelled:
            status, failure = JobStatus.CANCELLED, FailureKind.CANCELLED
        except InputUnavailable as exc:
            status, error_message, failure = JobStatus.FAILED, str(exc), FailureKind.INPUT
        except Exception as exc:
            status, error_message = JobStatus.FAILED, str(exc) or type(exc).__name__
            # A file the job reads that is gone or locked now is enough to have failed it,
            # whatever the error said. Otherwise only FFmpeg's own failure has a kind; any
            # other error's cause is not known here, and it is left without one.
            if _unreadable(_inputs(repo, job, spec)):
                failure = FailureKind.INPUT
            elif _unreadable(spec.get("made", [])):
                failure = FailureKind.MADE
            elif isinstance(exc, FFmpegFailed):
                failure = FailureKind.RENDER

    partial = spec.get("render_path") or job.output_path
    if job.kind == JobKind.RENDER and status != JobStatus.COMPLETED and partial:
        # Remove the partial file so it cannot be mistaken for a finished render.
        try:
            os.remove(partial)
        except OSError:
            pass
    if job.kind == JobKind.RENDER and status == JobStatus.COMPLETED:
        if spec.get("prune"):
            # Only the newest preview of a project is kept, and any still being rendered.
            housekeeping.prune(spec["prune"], repo.active_job_ids() | {job.job_id})

    def finish(current: Job) -> None:
        """Record the job's final status."""
        current.status = status
        current.error_message = error_message
        current.failure_kind = failure
        current.stage = None
        if status == JobStatus.COMPLETED:
            current.progress = 1.0
        if spec.get("delivered"):
            current.output_path = spec["delivered"]
        if job.kind == JobKind.RENDER and status == JobStatus.COMPLETED:
            # Kept on the job: a finished video's working folder is cleared away once it is saved.
            current.spots = scan.kept(job.work_dir)

    repo.update_job(job_id, finish)
    # This worker's slot and its memory are free now, so hand them to whatever is waiting
    # rather than leaving the queue still until someone next asks about a job. An analysis
    # worker keeps the next analysis for itself, because its models are already loaded.
    if job.kind == JobKind.ANALYZE:
        _, next_job = JobManager(repo).launch_ready(keep=JobKind.ANALYZE)
        return next_job.job_id if next_job is not None else None
    JobManager(repo).launch_ready()
    return None

def main(argv: Optional[List[str]] = None) -> None:
    """Run the worker process for one job, then any analysis it keeps for itself.

    Usage: `python -m app.engine.renderer <db_path> <job_id>`. Any unexpected
    error is recorded on the job before it propagates. An analysis worker runs
    the next admitted analysis in this same process, with its models already
    loaded, and keeps doing so until no analysis is waiting.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]`.
    """
    db_path, job_id = argv if argv is not None else sys.argv[1:]
    repo = Repository(db_path)
    # Only a worker keeps the speech model loaded between analyses; the server does not.
    models.keep_speech_model()
    current: Optional[str] = job_id
    while current is not None:
        try:
            current = run_worker(repo, current)
        except Exception as exc:
            repo.update_job(current, _fail_if_active(f"worker crashed: {exc}"))
            raise

if __name__ == "__main__":
    main()
