import json
import os
import subprocess
from typing import Any, Dict, Optional, Tuple


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

def picture_size(info: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Read how big a file's picture is when it is played.

    A phone held upright records a landscape frame and a note to turn it, and
    FFmpeg turns it on the way out; so the size that matters for framing is
    the turned one, not the one stored.

    Args:
        info: What `probe_file` returned.

    Returns:
        `(width, height)` as shown, or None when the file has no picture. Cover
        art in a music file is not a picture.
    """
    for stream in info.get("streams", []):
        if stream.get("codec_type") != "video" or stream.get("disposition", {}).get("attached_pic"):
            continue
        width, height = stream.get("width"), stream.get("height")
        if not width or not height:
            return None
        turned = [
            data.get("rotation") for data in stream.get("side_data_list", []) if "rotation" in data
        ] or [stream.get("tags", {}).get("rotate")]
        if turned[0] is not None and round(abs(float(turned[0]))) % 180 == 90:
            width, height = height, width
        return int(width), int(height)
    return None
