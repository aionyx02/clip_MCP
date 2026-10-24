import functools
import json
import os
import re
import subprocess
import threading
import time
from typing import Callable, List, Optional

POLL_INTERVAL_SECONDS = 0.5
TERMINATE_GRACE_SECONDS = 3.0

class OperationCancelled(Exception):
    """Raised when a long-running operation stops because cancellation was requested."""

class FFmpegFailed(RuntimeError):
    """Raised when FFmpeg exits with an error; the message holds what it said."""

def hidden_window_flags() -> int:
    """Return `subprocess` creation flags that hide console windows on Windows.

    Returns:
        `CREATE_NO_WINDOW` on Windows; 0 elsewhere.
    """
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

def escape_filter_path(path: str) -> str:
    """Make a file path safe to put inside a quoted filtergraph argument.

    A path breaks a filtergraph three times over: on Windows the backslashes
    are escapes and the drive letter's colon separates filter options, and an
    apostrophe anywhere in the path closes the quoted section, which then
    leaves any comma or bracket in it to the graph parser.

    Args:
        path: Path to a file the filtergraph reads or writes, written between
            single quotes at the call site.

    Returns:
        The path with forward slashes, an escaped colon, and each apostrophe
        quoted out and escaped for both the filter and the graph.
    """
    return path.replace("\\", "/").replace("'", r"'\\\''").replace(":", r"\:")

def read_tail(path: str, limit: int = 500) -> str:
    """Read the end of a text file.

    Args:
        path: File to read.
        limit: Maximum number of characters to return.

    Returns:
        Up to `limit` trailing characters, or an empty string if the file
        cannot be read.
    """
    try:
        with open(path, "rb") as file:
            file.seek(0, os.SEEK_END)
            file.seek(max(0, file.tell() - limit * 4))
            return file.read().decode("utf-8", errors="replace").strip()[-limit:]
    except OSError:
        return ""

# Which files the scan decodes on the card (#149). Starting CUDA costs about half a second,
# which a cheap file does not earn back. Provisional, from one machine (RTX 4060 Laptop,
# FFmpeg 9.0), scan seconds in software against on the card: DJI HEVC 10-bit at 55 Mbps
# gained from 4.2 s long (1.18 → 0.96) to 50 s (15.4 → 8.5), and 263 Mbps HEVC 20 s gained
# (9.8 → 4.1); a 1.4 s DJI clip lost (0.39 → 0.64), and so did HEVC at 30 Mbps (40 s,
# 5.1 → 6.3), 9.8 Mbps (365 s, 39.2 → 40.9) and under 4 Mbps; H.264 never gained, not even
# at 218 Mbps (8.35 → 8.19). Only HEVC was measured gaining, so only HEVC is listed.
CARD_CODECS = frozenset({"hevc"})
# Between the 30 Mbps file that lost and the 55 Mbps footage that gained.
CARD_MIN_BITRATE = 40_000_000
# Between the 1.4 s clip that lost and the 4.2 s clip that gained.
CARD_MIN_SECONDS = 3.0
# Written into the workspace the first time a scan fails on the card, and read by every
# worker after: while it is there, nothing is decoded on the card. `clip-mcp check`
# reports it; deleting it lets the card be tried again.
CARD_FAILED_FILE_NAME = "card-decode-failed.txt"

def card_failed_path() -> str:
    """Where a failed decode on the card is recorded, in the workspace in use."""
    from app import workspace

    return os.path.join(workspace.workspace_dir(), CARD_FAILED_FILE_NAME)

def card_failure() -> Optional[str]:
    """Say what failed on the card, if a scan ever did.

    Returns:
        The file and FFmpeg's last line, as recorded; None when nothing has.
    """
    try:
        with open(card_failed_path(), encoding="utf-8", errors="replace") as handle:
            return handle.read().strip() or "a scan"
    except OSError:
        return None

def stop_decoding_on_card(path: str, error: str) -> None:
    """Record that a scan failed on the card, so no worker decodes on it again until the record is deleted.

    Args:
        path: The file whose scan failed.
        error: What FFmpeg said; its last line is kept.
    """
    lines = error.strip().splitlines()
    try:
        with open(card_failed_path(), "w", encoding="utf-8") as handle:
            handle.write(f"{path}: {lines[-1] if lines else 'FFmpeg failed'}\n")
    except OSError:
        pass

@functools.lru_cache(maxsize=None)
def _card_listed(ffmpeg_bin: str) -> bool:
    """Say whether this FFmpeg lists CUDA and an NVIDIA card is there to use it.

    Asked once per executable; see `can_decode_on_card` for the rest.
    """
    try:
        listed = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-hwaccels"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            creationflags=hidden_window_flags(),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    if "cuda" not in {line.strip() for line in listed.splitlines()}:
        return False
    try:
        from app.engine import machine

        return machine._nvidia_card() is not None
    except Exception:
        return False

def can_decode_on_card(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Say whether FFmpeg can decode the analysis scan on this computer's NVIDIA card.

    CUDA has to be among the hardware methods this FFmpeg was built with, and
    there has to be an NVIDIA card it was built for. Asking once and
    remembering the answer (in `_card_listed`) keeps an extra process off
    every scan; a check that failed at first use, such as a card still
    waking, is remembered too, until the process ends. `CLIP_MCP_HWACCEL=off`
    turns the card off, and so does a scan that once failed on it (see
    `stop_decoding_on_card`); both are read on every call.

    Args:
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        True when the card can be used.
    """
    if os.environ.get("CLIP_MCP_HWACCEL", "").strip().lower() == "off":
        return False
    if card_failure() is not None:
        return False
    return _card_listed(ffmpeg_bin)

def _ffprobe_beside(ffmpeg_bin: str) -> str:
    """The ffprobe that came with an FFmpeg: in the same folder when it was given as a path."""
    folder = os.path.dirname(ffmpeg_bin)
    return os.path.join(folder, "ffprobe") if folder else "ffprobe"

def _worth_decoding_on_card(path: str, ffmpeg_bin: str) -> bool:
    """Say whether a file's picture is costly enough to decode that the card earns its start-up."""
    try:
        result = subprocess.run(
            [_ffprobe_beside(ffmpeg_bin), "-v", "error", "-select_streams", "V:0",
             "-show_entries", "stream=codec_name:format=bit_rate", "-of", "json", path],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            creationflags=hidden_window_flags(),
        )
        found = json.loads(result.stdout or "{}")
        codec = (found.get("streams") or [{}])[0].get("codec_name")
        bitrate = int((found.get("format") or {}).get("bit_rate") or 0)
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return False
    return codec in CARD_CODECS and bitrate >= CARD_MIN_BITRATE

def card_decode_args(ffmpeg_bin: str, path: str, duration: float) -> List[str]:
    """The FFmpeg arguments that decode a file on the NVIDIA card, before `-i`, where that is faster.

    Only a file at least `CARD_MIN_SECONDS` long, whose picture is in one of
    `CARD_CODECS` at `CARD_MIN_BITRATE` or more, and only when the card can be
    used at all. The length is checked first, so a short file costs no probe.
    Without `-hwaccel_output_format`, FFmpeg hands ordinary frames back to the
    filters, and falls back to software for a codec the card cannot decode.

    Args:
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.
        path: The file to decode.
        duration: Its length in seconds.

    Returns:
        `["-hwaccel", "cuda"]`, or nothing.
    """
    if duration < CARD_MIN_SECONDS or not can_decode_on_card(ffmpeg_bin):
        return []
    return ["-hwaccel", "cuda"] if _worth_decoding_on_card(path, ffmpeg_bin) else []

class _ProgressReader(threading.Thread):
    """Background thread that tracks the media position FFmpeg reports."""

    def __init__(self, stream):
        """Initialize the reader.

        Args:
            stream: Text stream receiving FFmpeg's `-progress` output.
        """
        super().__init__(daemon=True)
        self._stream = stream
        self.seconds = 0.0

    def run(self) -> None:
        """Consume progress lines until the stream closes."""
        for line in self._stream:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and value.isdigit():
                self.seconds = int(value) / 1_000_000

@functools.lru_cache(maxsize=None)
def _graph_file_option(executable: str) -> str:
    """The option that hands this FFmpeg its filter graph in a file.

    FFmpeg 7 reads any option's value from a file when it is written `-/name`;
    FFmpeg 6, still what many Linux distributions ship, knows only
    `-filter_complex_script`, which 7 deprecates. Asked once per executable.

    Args:
        executable: The FFmpeg to ask.

    Returns:
        `-/filter_complex`, or `-filter_complex_script` for FFmpeg 6 and older.
    """
    try:
        first = subprocess.run([executable, "-hide_banner", "-version"], capture_output=True, text=True,
                               timeout=30, creationflags=hidden_window_flags()).stdout.split("\n", 1)[0]
    except (OSError, subprocess.SubprocessError):
        return "-/filter_complex"
    # "ffmpeg version 6.1.1-3ubuntu5", "ffmpeg version n7.1", "ffmpeg version 9.0.2-essentials_build-...".
    # A git snapshot ("N-118000-g...") has no release number, and is newer than any release.
    found = re.match(r"\S+ version n?(\d+)\.", first)
    return "-filter_complex_script" if found and int(found.group(1)) < 7 else "-/filter_complex"

def graph_from_file(command: List[str], graph_path: str) -> List[str]:
    """Move a command's filter graph out of the command line and into a file.

    A long timeline builds a graph of tens of thousands of characters, one
    chain per clip, and Windows refuses to start a process whose command line
    passes 32767 characters. FFmpeg reads an option's value from a file when
    the option is written `-/name` (`-filter_complex_script` before FFmpeg 7),
    so the graph goes there instead and the command line stays short however
    many clips there are.

    Args:
        command: FFmpeg arguments, starting with the executable.
        graph_path: File to write the graph to; a second graph in the same
            command, if there is one, goes next to it with a number added.

    Returns:
        The command with each `-filter_complex` value replaced by a file.
    """
    moved = list(command)
    written = 0
    for index in range(len(moved) - 1):
        if moved[index] != "-filter_complex":
            continue
        root, extension = os.path.splitext(graph_path)
        path = graph_path if written == 0 else f"{root}{written}{extension}"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(moved[index + 1])
        moved[index], moved[index + 1] = _graph_file_option(moved[0]), path
        written += 1
    return moved

def run_ffmpeg(
    command: List[str],
    log_path: str,
    duration_seconds: float,
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
) -> None:
    """Run FFmpeg to completion while reporting progress and honoring cancellation.

    FFmpeg's standard error is written to `log_path`, so callers can parse
    filter output from it afterwards. A filter graph is read from a file
    written next to the log, so no timeline is too long to start.

    Args:
        command: FFmpeg arguments, starting with the executable. Progress
            reporting options are added automatically.
        log_path: File that receives FFmpeg's standard error.
        duration_seconds: Duration of the media being processed, used to turn
            positions into fractions; progress is not reported if it is 0.
        on_progress: Called with the completed fraction, from 0.0 to 1.0,
            every `POLL_INTERVAL_SECONDS`.
        is_cancelled: Polled every `POLL_INTERVAL_SECONDS`; once it returns
            true, FFmpeg is terminated, and killed if it does not exit within
            `TERMINATE_GRACE_SECONDS`.

    Raises:
        OperationCancelled: If FFmpeg was stopped because of cancellation.
        FFmpegFailed: If FFmpeg exits with an error. The message holds the end
            of its log.
    """
    command = graph_from_file(command, f"{os.path.splitext(log_path)[0]}.graph.txt")
    graphs = [command[index + 1] for index in range(len(command) - 1)
              if command[index] in ("-/filter_complex", "-filter_complex_script")]
    try:
        _run(command, log_path, duration_seconds, on_progress, is_cancelled)
    finally:
        # Written for this one run; the folder it sits in may be one that has to end up empty.
        for graph in graphs:
            if os.path.exists(graph):
                os.remove(graph)

def _run(
    command: List[str],
    log_path: str,
    duration_seconds: float,
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
) -> None:
    """Run FFmpeg as `run_ffmpeg` describes, with its graph already in a file."""
    command = [command[0], "-progress", "pipe:1", *command[1:]]
    with open(log_path, "wb") as log_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log_file,
            encoding="utf-8",
            errors="replace",
            creationflags=hidden_window_flags(),
        )
        reader = _ProgressReader(process.stdout)
        reader.start()

        terminated_at = None
        while True:
            try:
                return_code = process.wait(timeout=POLL_INTERVAL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass
            if duration_seconds > 0:
                on_progress(min(reader.seconds / duration_seconds, 1.0))
            if terminated_at is None and is_cancelled():
                process.terminate()
                terminated_at = time.monotonic()
            elif terminated_at is not None and time.monotonic() - terminated_at > TERMINATE_GRACE_SECONDS:
                process.kill()

    if terminated_at is not None:
        raise OperationCancelled()
    if return_code != 0:
        raise FFmpegFailed(read_tail(log_path) or f"ffmpeg exited with code {return_code}")
