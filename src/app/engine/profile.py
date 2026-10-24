"""How long each stage of a job took, and how much memory the machine had in use during it.

A job reports the stage it is in while it runs. This times each of those
stages and, alongside it, samples the machine's memory and the NVIDIA card's
memory so that a stage can be read as a cost rather than only as a wait: the
transcription of an hour of talk is slow and takes several gigabytes, while
the same hour's scene detection is quick and takes almost nothing.

What it measures is the machine as a whole, not this process. Memory in use
is total minus available, so it includes FFmpeg and every other child process
a stage starts, and it also includes whatever else the person has open — a
browser, another editor, the rest of their day. Every rise is measured from
the reading taken when the job started, so a stage that runs after a large
model was loaded includes that model in its rise; the column is the most
memory in use during the stage, not what the stage itself added. It is
therefore an upper bound on the stage's cost, not what the stage would need
on an idle computer.

The numbers are sampled rather than tracked: memory is read every
`PROFILE_SAMPLE_SECONDS`, so a stage shorter than that can finish without a
reading of its own and show a rise of zero. The card is read less often, every
`GPU_SAMPLE_SECONDS`, because each reading starts nvidia-smi, which takes a
quarter of a second here: read as often as memory, it would keep a process
running for the whole job it is measuring. A card rise of None means the card
was not read during that stage: the stage was shorter than `GPU_SAMPLE_SECONDS`
or the card had stopped answering. Nothing here decides anything about a job;
it only records what was seen.
"""

import re
import threading
import time
from statistics import median
from typing import Any, Callable, Dict, List, Optional

from app.engine import machine, resources
from app.models.job import Job, JobKind, JobStatus
from app.models.media import Asset

MEGABYTE = 1024 * 1024
# Provisional: how often memory is read while a stage runs. A stage shorter than this
# may show no rise, since it can begin and end between two readings.
PROFILE_SAMPLE_SECONDS = 0.25
# Provisional: how often the card's memory is read. Each reading starts nvidia-smi, about
# 0.25 s on the machine this was written on, so this keeps it to about a tenth of the time.
GPU_SAMPLE_SECONDS = 2.0
# Provisional: how long `stop` waits for the sampler to end. A card reading in progress can take up to
# nvidia-smi's own 10 s timeout, and the job is not kept waiting for it: the thread is a
# daemon, so it is abandoned and ends with the process.
STOP_WAIT_SECONDS = 1.0

def _machine_in_use() -> Optional[int]:
    """Read how much of this computer's physical memory is in use.

    Machine-wide, so it includes FFmpeg and other child processes, and also
    everything else running on the computer at that moment.

    Returns:
        Bytes in use, or None when this machine's memory cannot be measured.
    """
    status = resources.memory_status()
    if status is None:
        return None
    available, total = status
    return max(0, total - available)

def _gpu_in_use() -> Optional[int]:
    """Read how much memory the first NVIDIA card has in use.

    Returns:
        Bytes in use, or None when there is no NVIDIA driver tool or it says
        nothing.
    """
    listed = machine._nvidia_smi("memory.used")
    if listed is None:
        return None
    text = listed.strip()
    if not text.isdigit():
        return None
    return int(text) * MEGABYTE

class StageClock:
    """Times the stages of one job and keeps the peak memory seen in each.

    A sampler thread reads the machine's memory and the card's memory every
    `sample_seconds` while the clock runs. `mark` is called from the job's own
    thread as it moves between stages, so it may run while the sampler does;
    both take the same lock.

    The clock can be given fake readers and a fake clock, which is how it is
    tested without a machine or a real thread.
    """

    def __init__(
        self,
        sample_seconds: float = PROFILE_SAMPLE_SECONDS,
        gpu_seconds: float = GPU_SAMPLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        memory: Callable[[], Optional[int]] = _machine_in_use,
        gpu: Callable[[], Optional[int]] = _gpu_in_use,
    ):
        """Initialize the clock without starting it.

        Args:
            sample_seconds: How often the sampler reads memory while running.
            gpu_seconds: How often it reads the card, rounded to a whole number
                of memory readings.
            clock: Reads the current time in seconds; `time.monotonic` unless
                a test says otherwise.
            memory: Reads the machine's memory in use, in bytes, or None.
            gpu: Reads the card's memory in use, in bytes, or None.
        """
        self._sample_seconds = sample_seconds
        self._gpu_every = max(1, int(round(gpu_seconds / sample_seconds)))
        self._clock = clock
        self._memory = memory
        self._gpu = gpu
        self._lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = []
        self._stage: Optional[str] = None
        self._stage_started = 0.0
        self._baseline_memory: Optional[int] = None
        self._baseline_gpu: Optional[int] = None
        self._peak_memory: Optional[int] = None
        self._peak_gpu: Optional[int] = None
        self._gpu_readable = True
        self._gpu_read_in_stage = False
        self._running = False
        # How many stages have begun, so a reading that was in progress across a change can be told apart.
        self._stages_begun = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Read the memory baseline, then sample in the background until `stop`.

        The card is not read here: a card reading starts nvidia-smi, which can
        take seconds, and the job's thread must not wait for it. Its baseline
        is read by the sampler instead, as soon as the sampler starts.
        """
        if self._running:
            return
        self._baseline_memory = self._memory()
        self._baseline_gpu = None
        # A machine with no card is not asked again, so it does not spawn a process per sample.
        self._gpu_readable = True
        self._gpu_read_in_stage = False
        self._peak_memory = self._baseline_memory
        self._peak_gpu = None
        self._entries = []
        self._stage = None
        self._running = True
        # A new event rather than clearing the old one: a sampler `stop` stopped waiting for keeps
        # the old one, already set, and ends when its reading returns instead of running again.
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_forever, args=(self._stop,), daemon=True)
        self._thread.start()

    def mark(self, stage: str) -> None:
        """Close the stage in progress, if any, and begin `stage`.

        A stage reported twice in a row stays one entry. A stage reported
        again after another stage has run is a new entry.

        Args:
            stage: What the job is doing now.
        """
        if stage == self._stage:
            return
        # Read before taking the lock, so the job's thread never waits on the sampler's reading.
        # Memory only: the card's reading starts a process, and the sampler takes it soon enough.
        memory = self._memory()
        with self._lock:
            if not self._running:
                return
            if stage == self._stage:
                return
            now = self._clock()
            self._close(now)
            self._stage = stage
            self._stages_begun += 1
            self._stage_started = now
            self._peak_memory = self._baseline_memory
            self._peak_gpu = self._baseline_gpu
            self._gpu_read_in_stage = False
            # One reading at the change, so a stage always has at least one.
            self._fold(memory, None)

    def stop(self) -> List[Dict[str, Any]]:
        """Close the last stage, stop the sampler, and return what each stage cost.

        Returns:
            One entry per stage, in order, each `{stage, seconds,
            memory_rise_mb, gpu_rise_mb}`. A rise is the stage's peak minus
            the baseline taken at `start`, never below zero, in MiB; None
            when that reading is not available on this machine.
        """
        if not self._running:
            return [dict(entry) for entry in self._entries]
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            # A card reading in progress can take up to nvidia-smi's own timeout. The thread is
            # a daemon, so the job does not wait for it beyond this.
            thread.join(timeout=STOP_WAIT_SECONDS)
        with self._lock:
            self._close(self._clock())
            self._running = False
        return [dict(entry) for entry in self._entries]

    def _sample(self, card: bool = True, stop: Optional[threading.Event] = None) -> None:
        """Take one reading of memory, and of the card when asked, and fold it into the current stage.

        The readings are taken outside the lock, so a slow one never holds up `mark`.
        A reading during which a new stage began belongs to neither stage for
        sure, so it is not folded into either; a card reading still serves as
        the baseline when there is none yet.

        Args:
            card: Whether to read the card as well.
            stop: The stop event of the sampler taking the reading. Once it is
                set, the reading is dropped: it is from a run already stopped,
                even if the clock has been started again since.
        """
        begun = self._stages_begun
        memory = self._memory()
        gpu = self._gpu() if card and self._gpu_readable else None
        with self._lock:
            if not self._running or (stop is not None and stop.is_set()):
                return
            if card and self._gpu_readable and gpu is None:
                # A card that could not be read once is not asked again.
                self._gpu_readable = False
            if gpu is not None and self._baseline_gpu is None:
                # The first card reading is the baseline every later rise is measured from.
                self._baseline_gpu = gpu
            if begun != self._stages_begun:
                return
            if gpu is not None and self._stage is not None:
                self._gpu_read_in_stage = True
            self._fold(memory, gpu)

    def _sample_forever(self, stop: threading.Event) -> None:
        """Read memory every `sample_seconds`, and the card every `gpu_seconds`, until `stop` is set.

        Args:
            stop: This sampler's own stop event, kept rather than read off the
                clock, so a sampler `stop` gave up waiting for ends when its
                reading returns instead of carrying on into the next run.
        """
        # The card's baseline is read here, as soon as the thread starts, rather than at the first scheduled
        # reading, by when the job may already have put a model on the card.
        self._sample(card=True, stop=stop)
        ticks = 0
        while not stop.wait(self._sample_seconds):
            ticks += 1
            self._sample(card=ticks % self._gpu_every == 0, stop=stop)
            if not self._running:
                return

    def _fold(self, memory: Optional[int], gpu: Optional[int]) -> None:
        """Fold readings into the current stage's peaks; caller holds the lock.

        Args:
            memory: The machine's memory in use, or None when not read.
            gpu: The card's memory in use, or None when not read.
        """
        if memory is not None:
            self._peak_memory = memory if self._peak_memory is None else max(self._peak_memory, memory)
        if gpu is not None:
            self._peak_gpu = gpu if self._peak_gpu is None else max(self._peak_gpu, gpu)

    def _close(self, now: float) -> None:
        """Record the stage in progress and clear it; caller holds the lock."""
        if self._stage is None:
            return
        self._entries.append({
            "stage": self._stage,
            "seconds": round(now - self._stage_started, 3),
            "memory_rise_mb": _rise(self._peak_memory, self._baseline_memory),
            "gpu_rise_mb": _rise(self._peak_gpu, self._baseline_gpu) if self._gpu_read_in_stage else None,
        })
        self._stage = None
        self._gpu_read_in_stage = False

def _rise(peak: Optional[int], baseline: Optional[int]) -> Optional[int]:
    """How far above the baseline a peak sits, in whole MiB, never below zero.

    Args:
        peak: The highest reading seen during the stage.
        baseline: The reading taken when the clock started.

    Returns:
        The rise in MiB, or None when either reading is not available.
    """
    if peak is None or baseline is None:
        return None
    return max(0, int(round((peak - baseline) / MEGABYTE)))

def stage_key(stage: str) -> str:
    """Group stages that differ only by a counter, such as one picture of twelve.

    Args:
        stage: The stage as the job reported it.

    Returns:
        The stage with every run of digits replaced by `#`.
    """
    return re.sub(r"\d+", "#", stage)

_KIND_ORDER = (JobKind.ANALYZE, JobKind.RENDER, JobKind.PREPARE)

def summarise(jobs: List[Job], assets: Dict[str, Asset]) -> str:
    """Summarise how long each stage took across completed jobs, one block per job kind.

    Only jobs that completed and carry a profile are read. Within a block,
    stages are grouped by `stage_key` and listed by total time, largest first.

    Args:
        jobs: The jobs to summarise.
        assets: The library's assets, by ID, for the length of the footage an
            analyze job was given.

    Returns:
        A plain-text table, or a sentence when no completed job has a profile.
    """
    done = [job for job in jobs if job.status == JobStatus.COMPLETED and job.profile]
    if not done:
        return "No completed job has a profile yet."
    present = {job.kind for job in done}
    kinds = [kind for kind in _KIND_ORDER if kind in present]
    kinds += sorted((kind for kind in present if kind not in _KIND_ORDER), key=lambda kind: kind.value)
    blocks = [_block(kind, [job for job in done if job.kind == kind], assets) for kind in kinds]
    return "\n\n".join(blocks)

def _block(kind: JobKind, jobs: List[Job], assets: Dict[str, Asset]) -> str:
    """Build the table for one kind of job.

    Args:
        kind: The kind of job the block is about.
        jobs: Completed jobs of that kind, each with a profile.
        assets: The library's assets, by ID.

    Returns:
        The block's lines, joined.
    """
    stages: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        seen = set()
        for entry in job.profile or []:
            key = stage_key(str(entry.get("stage", "")))
            row = stages.setdefault(key, {"jobs": 0, "seconds": 0.0, "times": [], "memory": None, "gpu": None})
            seconds = float(entry.get("seconds") or 0.0)
            row["seconds"] += seconds
            row["times"].append(seconds)
            if key not in seen:
                row["jobs"] += 1
                seen.add(key)
            memory = entry.get("memory_rise_mb")
            if memory is not None:
                row["memory"] = memory if row["memory"] is None else max(row["memory"], memory)
            gpu = entry.get("gpu_rise_mb")
            if gpu is not None:
                row["gpu"] = gpu if row["gpu"] is None else max(row["gpu"], gpu)
    total = sum(row["seconds"] for row in stages.values())
    lines = [f"{kind.value}: {len(jobs)} completed job{'' if len(jobs) == 1 else 's'}"]
    lines.append(f"  {'stage':<32} {'jobs':>5} {'total s':>10} {'share':>7} {'median s':>10} "
                 f"{'mem MB':>8} {'gpu MB':>8}")
    for key, row in sorted(stages.items(), key=lambda item: (-item[1]["seconds"], item[0])):
        share = f"{100 * row['seconds'] / total:.1f}%" if total else "-"
        memory = "-" if row["memory"] is None else str(int(row["memory"]))
        gpu = "-" if row["gpu"] is None else str(int(row["gpu"]))
        lines.append(f"  {key:<32} {row['jobs']:>5} {row['seconds']:>10.3f} {share:>7} "
                     f"{median(row['times']):>10.3f} {memory:>8} {gpu:>8}")
    if kind == JobKind.ANALYZE:
        lines.append(_real_time(jobs, assets))
    return "\n".join(lines)

def _real_time(jobs: List[Job], assets: Dict[str, Asset]) -> str:
    """Say how long analyzing took against the footage's own length.

    Args:
        jobs: Completed analyze jobs.
        assets: The library's assets, by ID.

    Returns:
        A line with the real-time factor and how many jobs it covers; a line
        saying it cannot be worked out when no asset has a duration.
    """
    seconds, footage, covered = 0.0, 0.0, 0
    for job in jobs:
        asset = assets.get(job.asset_id) if job.asset_id else None
        if asset is None or asset.duration is None:
            continue
        seconds += sum(float(entry.get("seconds") or 0.0) for entry in job.profile or [])
        footage += float(asset.duration)
        covered += 1
    if not covered or footage <= 0:
        return "  real-time factor: - (no analyzed asset has a duration)"
    return (f"  real-time factor: {seconds / footage:.2f}x "
            f"({covered} job{'' if covered == 1 else 's'}, {footage:.1f} s of footage)")
