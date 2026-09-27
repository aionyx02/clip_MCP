import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

# A timestamp written into a file name, as cameras and phones name their files:
# DJI_20260830121241_0017_D.MP4, VID20260830073357.mp4, video_2026-08-29_23-19-10.mp4.
_NAMED_TIME = re.compile(r"(20\d\d)-?(\d\d)-?(\d\d)[_\-T ]?(\d\d)-?(\d\d)-?(\d\d)")


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

def recorded_at(info: Dict[str, Any], filepath: str) -> Optional[datetime]:
    """Work out when a file was recorded.

    From the time the camera wrote into the file, or failing that from a time
    in the file's name, read as this machine's local time. Never from when
    the file was last modified: that is when it was copied, which for a
    folder of footage is one moment for all of it and says nothing about the
    order it was shot in.

    Args:
        info: What `probe_file` returned.
        filepath: The file's path, for its name.

    Returns:
        The moment recording started, in UTC, or None when nothing says.
    """
    tagged = [info.get("format", {}).get("tags", {}).get("creation_time")] + [
        stream.get("tags", {}).get("creation_time") for stream in info.get("streams", [])
    ]
    for value in tagged:
        if not value:
            continue
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        # Cameras with no clock set write 1970 or 2000; that is not a time anything was shot.
        if moment.year > 2000:
            return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    named = _NAMED_TIME.search(os.path.basename(filepath))
    if named:
        try:
            return datetime(*(int(part) for part in named.groups())).astimezone().astimezone(timezone.utc)
        except ValueError:
            return None
    return None

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

def timecode_start(info: Dict[str, Any]) -> Optional[float]:
    """Read the second a file's own timecode starts at.

    A professional camera stamps its files with the time of day, or a running
    count, so the first frame of a clip is 10:00:00:00 rather than zero; an
    editing program lines the file up by that. A phone does not, and a file
    without one starts at zero.

    Args:
        info: What `probe_file` returned.

    Returns:
        The seconds the first frame's timecode stands for, or None when the
        file carries none or it cannot be read. Drop-frame timecode — written
        with a semicolon — is counted the way it is meant: two frame numbers
        skipped every minute except every tenth.
    """
    tags = [info.get("format", {}).get("tags", {})] + [stream.get("tags", {}) for stream in info.get("streams", [])]
    written = next((value for tag in tags for key, value in tag.items() if key.lower() == "timecode"), None)
    if not written:
        return None
    video = next((stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"), None)
    rate_text = (video or {}).get("r_frame_rate") or (video or {}).get("avg_frame_rate") or ""
    try:
        numerator, denominator = (int(part) for part in rate_text.split("/"))
        rate = numerator / denominator
        hours, minutes, seconds, frames = (int(part) for part in written.replace(";", ":").replace(".", ":").split(":"))
    except (ValueError, ZeroDivisionError):
        return None
    nominal = round(rate)
    count = ((hours * 60 + minutes) * 60 + seconds) * nominal + frames
    if ";" in written or "." in written:
        dropped = 2 if nominal == 30 else 4 if nominal == 60 else 0
        total_minutes = hours * 60 + minutes
        count -= dropped * (total_minutes - total_minutes // 10)
    return count / rate


def speech_loudness(
    filepath: str, start: float, duration: float, ffmpeg_bin: str = "ffmpeg",
) -> Optional[float]:
    """Measure how loud a stretch of a file's sound is where there is any.

    Integrated loudness gates out the silence between sentences, so on a
    recording of somebody talking it is the level of the talking rather than
    of the talking averaged with the pauses. Only the sound is decoded.

    Args:
        filepath: Path to the media file.
        start: Where the stretch starts, in seconds.
        duration: How long it runs, in seconds.
        ffmpeg_bin: Path to, or name of, the ffmpeg executable.

    Returns:
        The integrated loudness in LUFS, or nothing when the file cannot be
        read or the stretch is silent.
    """
    from app.engine.ffmpeg import hidden_window_flags

    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", filepath,
         "-vn", "-af", "ebur128", "-f", "null", "-"],
        stdin=subprocess.DEVNULL, capture_output=True, creationflags=hidden_window_flags(),
    )
    if result.returncode != 0:
        return None
    summary = result.stderr.decode("utf-8", errors="replace").rsplit("Summary:", 1)[-1]
    for line in summary.splitlines():
        label, _, value = line.strip().partition(":")
        if label == "I":
            try:
                loudness = float(value.split()[0])
            except (IndexError, ValueError):
                return None
            # ebur128 reports a stretch with nothing above its gate as -70.
            return loudness if loudness > -70 else None
    return None
