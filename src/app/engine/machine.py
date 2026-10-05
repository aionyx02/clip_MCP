"""How fast this computer will be at the local models, said so the AI can plan around it.

Speech recognition is the slow part of reading footage, and how slow depends
on the computer far more than on anything the AI chooses: an NVIDIA card does
an hour of talk in minutes, an older laptop's processor can take longer than
the footage runs. The AI cannot see the computer, so this looks once and puts
it in words: a tier, what it rests on, and roughly how long an hour of footage
takes here.

The tier is read from the hardware alone, never from how past jobs went, so
the rule the AI follows on a slow computer (ask before a long transcription)
stays the same from one day to the next.
"""

import functools
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from app.engine import resources
from app.engine.ffmpeg import hidden_window_flags

GIB = 1024 ** 3

# Provisional (roadmap §13): where the tiers part. A card or computer sold as 4 GB or
# 16 GB reports a little less than that, so each line sits just under the round number.
HIGH_GPU_MEMORY_BYTES = int(3.5 * GIB)
MID_MEMORY_BYTES = 15 * GIB
MID_LOGICAL_CORES = 8

@dataclass(frozen=True)
class Machine:
    """What this computer brings to the local models.

    Attributes:
        tier: `high`, `mid` or `low`.
        gpu: The NVIDIA card speech recognition will use, or None.
        gpu_memory_bytes: That card's memory, or None when unknown.
        memory_bytes: Physical memory, or None when it cannot be read.
        logical_cores: Logical processors.
        apple_silicon: An Apple M-series Mac.
        mac: Any Mac.
        identity: Names this computer, so speeds measured on it stay with it.
    """

    tier: str
    gpu: Optional[str]
    gpu_memory_bytes: Optional[int]
    memory_bytes: Optional[int]
    logical_cores: int
    apple_silicon: bool
    mac: bool
    identity: str

    @property
    def device(self) -> str:
        """Where speech recognition runs here: `cuda` or `cpu`."""
        return "cuda" if self.gpu else "cpu"

def classify(
    gpu: Optional[str], gpu_memory_bytes: Optional[int], memory_bytes: Optional[int],
    logical_cores: int, apple_silicon: bool, mac: bool,
) -> str:
    """Put a computer in a tier by its hardware.

    A card whose memory cannot be read is not trusted to hold the model, and
    memory that cannot be read counts as too little: guessing high is the
    mistake that leaves a user waiting hours without having been asked.

    Args:
        gpu: The NVIDIA card speech recognition will use, or None.
        gpu_memory_bytes: That card's memory, or None when unknown.
        memory_bytes: Physical memory, or None when unknown.
        logical_cores: Logical processors.
        apple_silicon: An Apple M-series Mac.
        mac: Any Mac.

    Returns:
        `high`, `mid` or `low`.
    """
    if gpu and gpu_memory_bytes is not None and gpu_memory_bytes >= HIGH_GPU_MEMORY_BYTES:
        return "high"
    enough_memory = memory_bytes is not None and memory_bytes >= MID_MEMORY_BYTES
    if mac:
        return "mid" if apple_silicon and enough_memory else "low"
    return "mid" if enough_memory and logical_cores >= MID_LOGICAL_CORES else "low"

def whisper_device() -> str:
    """Choose the device for speech recognition.

    The `CLIP_MCP_WHISPER_DEVICE` environment variable selects `cuda` or
    `cpu`; the default, `auto`, uses CUDA when a GPU is available.

    Returns:
        `"cuda"` or `"cpu"`.
    """
    requested = os.environ.get("CLIP_MCP_WHISPER_DEVICE", "auto").lower()
    if requested != "auto":
        return requested
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

def _nvidia_card() -> Optional[Tuple[str, int]]:
    """Name and memory of the first NVIDIA card, from the tool its driver ships.

    Returns:
        `(name, bytes)`, or None when there is no driver tool or it says nothing.
    """
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return None
    try:
        listed = subprocess.run([tool, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
                                creationflags=hidden_window_flags())
    except (OSError, subprocess.SubprocessError):
        return None
    first = listed.stdout.strip().splitlines()[0] if listed.returncode == 0 and listed.stdout.strip() else ""
    name, _, mebibytes = first.rpartition(",")
    if not name or not mebibytes.strip().isdigit():
        return None
    return name.strip(), int(mebibytes) * 1024 ** 2

def _usable_card() -> Optional[Tuple[str, Optional[int]]]:
    """The NVIDIA card speech recognition will run on, if it runs on one.

    Returns:
        `(name, bytes or None)`, or None when it runs on the processor.
    """
    try:
        if whisper_device() != "cuda":
            return None
    except (ImportError, OSError, RuntimeError):
        return None
    return _nvidia_card() or ("NVIDIA GPU", None)

def _cpu_name() -> str:
    """The processor's model name, as the system reports it."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            named = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                   timeout=10)
            if named.returncode == 0 and named.stdout.strip():
                return named.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            found = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(encoding="utf-8"),
                              re.MULTILINE)
            if found:
                return found.group(1).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown processor"

def _apple_silicon() -> bool:
    """Whether this is an M-series Mac, even when Python itself runs translated for Intel."""
    return sys.platform == "darwin" and (platform.machine() == "arm64" or resources.sysctl_int("hw.optional.arm64") == 1)

@functools.lru_cache(maxsize=1)
def identity() -> str:
    """Name this computer, so speeds measured on it are not used on another; cheap, unlike `profile`."""
    return f"{platform.node()} / {_cpu_name()}"

@functools.lru_cache(maxsize=1)
def profile() -> Machine:
    """Look at this computer, once per process: the hardware does not change while it runs."""
    card = _usable_card()
    memory = resources.memory_status()
    found = dict(
        gpu=card[0] if card else None,
        gpu_memory_bytes=card[1] if card else None,
        memory_bytes=memory[1] if memory else None,
        logical_cores=os.cpu_count() or 1,
        apple_silicon=_apple_silicon(),
        mac=sys.platform == "darwin",
    )
    return Machine(tier=classify(**found), identity=identity(), **found)

def _gibibytes(count: Optional[int]) -> str:
    """Bytes as whole gigabytes, the way a computer's memory is usually stated."""
    return "unknown" if count is None else f"{round(count / GIB)} GB"

def _minutes_in_words(minutes: list) -> str:
    """A `[low, high]` range of minutes in words; high may be None for no upper bound."""
    low, high = minutes
    return f"{low} to {high} minutes" if high is not None else f"over {low} minutes, possibly much longer"

def describe(found: Machine, hour: dict) -> str:
    """Say what this computer means for the local models, in a few sentences for the AI.

    Args:
        found: The computer.
        hour: `speed.estimate` for an hour of footage transcribed
            accurately, measured here or from the hardware.

    Returns:
        The tier, what it rests on, and how long an hour of footage takes.
    """
    if found.gpu:
        runs_on = f"an NVIDIA card ({found.gpu}, {_gibibytes(found.gpu_memory_bytes)})"
    elif found.mac:
        runs_on = ("the processor only: speech recognition cannot use a Mac's graphics, Apple chips included"
                   f" ({'Apple silicon' if found.apple_silicon else 'Intel Mac'})")
    else:
        runs_on = ("the processor only: no NVIDIA card it can use, and AMD or Intel graphics do not help, since"
                   " speech recognition runs only on NVIDIA cards")
    return (
        f"This computer's speed tier for the local models is {found.tier}. Speech recognition runs on {runs_on};"
        f" {_gibibytes(found.memory_bytes)} memory, {found.logical_cores} logical cores. Transcribing an hour of"
        f" footage accurately takes about {_minutes_in_words(hour['transcription_minutes'])} here"
        f" ({hour['basis']}). Before analyzing, `analyze_asset` with `dry_run` gives both choices' times for"
        " the files at hand."
    )
