import json
import os
import subprocess
from typing import Any, Dict


def probe_file(filepath: str, ffprobe_bin: str = "ffprobe") -> Dict[str, Any]:
    """Read container and stream metadata from a media file using ffprobe.

    Args:
        filepath: Path to the media file to inspect.
        ffprobe_bin: Path to, or name of, the ffprobe executable.

    Returns:
        The parsed ffprobe JSON output: a `format` object describing the
        container (duration, size, bit rate, ...) and a `streams` list
        describing each video, audio, and subtitle stream.

    Raises:
        FileNotFoundError: If `filepath` does not exist.
        RuntimeError: If ffprobe exits with a non-zero status.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"media file {filepath} not found")

    result = subprocess.run(
        [ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", filepath],
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.decode('utf-8', errors='replace').strip()}")
    return json.loads(result.stdout)
