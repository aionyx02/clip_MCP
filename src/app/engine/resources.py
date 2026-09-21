"""How much memory this machine has to spare, and how much of it one job may take.

Nothing here decides what work to run. It measures the machine, turns the
`CLIP_MCP_*` settings into numbers, and hands out the few concurrency limits
that keep FFmpeg and speech recognition from claiming more memory than the
machine can give. Every estimate deliberately under-states what is free and
over-states what a job costs: being wrong in that direction queues a job,
while being wrong the other way freezes the user's computer.
"""

import ctypes
import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

MEGABYTE = 1024 * 1024

# Physical memory that is never spent on jobs, so the rest of the machine stays usable.
DEFAULT_MEMORY_RESERVE_MB = 2048
DEFAULT_MAX_JOBS = 2
DEFAULT_MAX_DECODERS = 4

# Rough peak resident memory per kind of work. A render holds one decoder per clip
# open at once, which is what makes a long timeline expensive; the rest is the
# filter graph and the encoder.
RENDER_BASE_BYTES = 256 * MEGABYTE
RENDER_PER_INPUT_BYTES = 64 * MEGABYTE
# The detection pass decodes one file once and splits it into branches no wider than
# 480 px, so its cost barely depends on how big the source is.
DETECTION_BYTES = 384 * MEGABYTE
# Diarization is the one stage whose cost follows the length of the file rather than the
# size of a model: the whole soundtrack is decoded to 16 kHz mono floats and handed over
# in one piece, so it is held twice over while the models read it. An hour of audio is
# about 440 MB of that, which is too much to leave out of the gate.
DIARIZATION_BYTES = 192 * MEGABYTE
DIARIZATION_BYTES_PER_SECOND = 16000 * 4 * 2
# Face detection reads stills a few hundred pixels wide, one at a time.
FACE_DETECTION_BYTES = 128 * MEGABYTE
# Weights plus CTranslate2's working set, by model name. int8 on the CPU is the expensive case.
WHISPER_MODEL_BYTES = {
    "tiny": 400 * MEGABYTE,
    "base": 550 * MEGABYTE,
    "small": 1100 * MEGABYTE,
    "medium": 2600 * MEGABYTE,
    "large-v3-turbo": 2200 * MEGABYTE,
    "large-v2": 4500 * MEGABYTE,
    "large-v3": 4500 * MEGABYTE,
}
DEFAULT_WHISPER_BYTES = 2600 * MEGABYTE

# Above this many inputs, each decoder is held to one thread: frame threading keeps
# several decoded pictures in flight per input, which a hundred inputs cannot afford.
MANY_INPUTS = 8

class _MemoryStatusEx(ctypes.Structure):
    """The `MEMORYSTATUSEX` structure that Windows fills in with memory figures."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

def _windows_memory() -> Optional[Tuple[int, int]]:
    """Read physical memory from the Windows kernel.

    Returns:
        `(available, total)` in bytes, or `None` if the call fails.
    """
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
    except (AttributeError, OSError):
        return None
    return int(status.ullAvailPhys), int(status.ullTotalPhys)

def _meminfo_memory() -> Optional[Tuple[int, int]]:
    """Read physical memory from `/proc/meminfo` on Linux.

    `MemAvailable` is the kernel's own estimate of what a new process can take
    without pushing the machine into swap, which is the question being asked
    here.

    Returns:
        `(available, total)` in bytes, or `None` if the file cannot be read.
    """
    try:
        values = {}
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                key, _, rest = line.partition(":")
                if key in ("MemAvailable", "MemTotal"):
                    values[key] = int(rest.split()[0]) * 1024
        return values["MemAvailable"], values["MemTotal"]
    except (OSError, KeyError, IndexError, ValueError):
        return None

def _sysconf_memory() -> Optional[Tuple[int, int]]:
    """Read physical memory through `sysconf`, where the platform offers it.

    Returns:
        `(available, total)` in bytes, or `None` if the platform does not
        report free pages.
    """
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        return os.sysconf("SC_AVPHYS_PAGES") * page, os.sysconf("SC_PHYS_PAGES") * page
    except (AttributeError, ValueError, OSError):
        return None

def memory_status() -> Optional[Tuple[int, int]]:
    """Measure physical memory on this machine.

    Returns:
        `(available, total)` in bytes, or `None` on a platform that cannot be
        measured. Callers treat `None` as unknown and fall back to the job
        count limit alone rather than guessing.
    """
    if os.name == "nt":
        return _windows_memory()
    return _meminfo_memory() or _sysconf_memory()

def _setting(name: str, default: int) -> int:
    """Read a whole-number setting from the environment.

    Args:
        name: Environment variable to read.
        default: Value used when it is unset, empty, or not a number.

    Returns:
        The setting, never below zero.
    """
    try:
        return max(0, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default

def memory_reserve_bytes() -> int:
    """Memory that jobs must leave free, set by `CLIP_MCP_MEMORY_RESERVE_MB`.

    Returns:
        The reserve in bytes.
    """
    return _setting("CLIP_MCP_MEMORY_RESERVE_MB", DEFAULT_MEMORY_RESERVE_MB) * MEGABYTE

def spare_bytes() -> Optional[int]:
    """Memory a new job may take right now.

    Returns:
        Free physical memory less the reserve, in bytes, never below zero; or
        `None` if this machine's memory cannot be measured.
    """
    status = memory_status()
    if status is None:
        return None
    return max(0, status[0] - memory_reserve_bytes())

def max_concurrent_jobs() -> int:
    """How many render or analyze jobs may run at once.

    Set by `CLIP_MCP_MAX_JOBS`. Both kinds of job saturate the CPU on their
    own, so running more of them at once finishes the batch no sooner while
    multiplying the memory it needs.

    Returns:
        The limit, at least 1.
    """
    return max(1, _setting("CLIP_MCP_MAX_JOBS", DEFAULT_MAX_JOBS))

def max_decoders() -> int:
    """How many frames may be decoded at once for contact sheets and storyboards.

    Set by `CLIP_MCP_MAX_DECODERS`. Each decode is its own short-lived FFmpeg
    process, and several tool calls can be in flight at the same time, so this
    is a limit across the whole server process rather than one per call.

    Returns:
        The limit, at least 1.
    """
    return max(1, _setting("CLIP_MCP_MAX_DECODERS", DEFAULT_MAX_DECODERS))

def whisper_memory_bytes(model_name: str) -> int:
    """Estimate what a speech recognition model needs while it runs.

    Args:
        model_name: Model as named for faster-whisper, such as
            `large-v3-turbo`. A local path or an unknown name gets the
            default.

    Returns:
        The estimate in bytes.
    """
    return WHISPER_MODEL_BYTES.get(model_name, DEFAULT_WHISPER_BYTES)

def analysis_memory_bytes(
    transcribe: bool,
    model_name: str,
    duration: float = 0.0,
    diarize: bool = False,
    detect_faces: bool = False,
) -> int:
    """Estimate what analyzing one asset needs.

    The stages do not all run at once, so this is the largest of them rather
    than their sum — except for diarization, which is counted on its own
    because what it holds is the file itself and not a model.

    Args:
        transcribe: Whether speech recognition runs as well as detection.
        model_name: Speech recognition model that would be loaded.
        duration: Length of the asset in seconds, which is what diarization's
            cost follows.
        diarize: Whether voices are told apart as well.
        detect_faces: Whether faces are looked for.

    Returns:
        The estimate in bytes.
    """
    stages = [DETECTION_BYTES]
    if transcribe:
        stages.append(whisper_memory_bytes(model_name))
    if detect_faces:
        stages.append(FACE_DETECTION_BYTES)
    if diarize:
        stages.append(DIARIZATION_BYTES + int(max(0.0, duration) * DIARIZATION_BYTES_PER_SECOND))
    return max(stages)

def render_memory_bytes(input_count: int) -> int:
    """Estimate what rendering a timeline needs.

    Args:
        input_count: Number of source files the render opens at once, which is
            one per clip.

    Returns:
        The estimate in bytes.
    """
    return RENDER_BASE_BYTES + max(0, input_count) * RENDER_PER_INPUT_BYTES

def decoder_threads(input_count: int) -> int:
    """Threads to give each decoder in a render.

    A decoder using frame threading keeps roughly one decoded picture per
    thread in memory. That is worth paying for a handful of inputs and ruinous
    for a hundred, so past `MANY_INPUTS` every decoder is held to one thread.
    The encoder and the filter graph keep all their threads either way.

    Args:
        input_count: Number of inputs the render opens.

    Returns:
        The thread count, or 0 to leave FFmpeg's own choice alone.
    """
    return 1 if input_count > MANY_INPUTS else 0

_decoder_slots = threading.BoundedSemaphore(max_decoders())

@contextmanager
def decoder_slot() -> Iterator[None]:
    """Hold one of the server's frame decoding slots for the duration of a block.

    Yields:
        Nothing. The slot is released on the way out, including on an error.
    """
    _decoder_slots.acquire()
    try:
        yield
    finally:
        _decoder_slots.release()
