import os
import subprocess
import threading
import time
from typing import Callable, List

POLL_INTERVAL_SECONDS = 0.5
TERMINATE_GRACE_SECONDS = 3.0

class OperationCancelled(Exception):
    """Raised when a long-running operation stops because cancellation was requested."""

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

def run_ffmpeg(
    command: List[str],
    log_path: str,
    duration_seconds: float,
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
) -> None:
    """Run FFmpeg to completion while reporting progress and honoring cancellation.

    FFmpeg's standard error is written to `log_path`, so callers can parse
    filter output from it afterwards.

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
        RuntimeError: If FFmpeg exits with an error. The message holds the end
            of its log.
    """
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
        raise RuntimeError(read_tail(log_path) or f"ffmpeg exited with code {return_code}")
