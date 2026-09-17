import io
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List

from PIL import Image, ImageDraw, ImageFont

from app.engine.ffmpeg import hidden_window_flags

TILE_SIZE = 320
TILE_GAP = 6
LABEL_FONT_SIZE = 18

def format_timestamp(seconds: float) -> str:
    """Format a time for display, such as `1:02:03.4` or `02:03.4`.

    Args:
        seconds: Time in seconds.

    Returns:
        The formatted time with tenths of a second.
    """
    tenths = round(seconds * 10)
    hours, rest = divmod(tenths, 36000)
    minutes, rest = divmod(rest, 600)
    text = f"{minutes:02d}:{rest // 10:02d}.{rest % 10}"
    return f"{hours}:{text}" if hours else text

def extract_frame(path: str, seconds: float, max_size: int = TILE_SIZE, ffmpeg_bin: str = "ffmpeg") -> Image.Image:
    """Decode a single video frame, scaled to fit a square box.

    Args:
        path: Media file to read.
        seconds: Time of the frame in the file.
        max_size: Maximum width and height of the returned image in pixels.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        The frame as an RGB image.

    Raises:
        RuntimeError: If FFmpeg cannot decode a frame at that time.
    """
    result = subprocess.run(
        [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{max(seconds, 0):.3f}", "-i", path,
            "-map", "0:V:0", "-frames:v", "1",
            "-vf", f"scale={max_size}:{max_size}:force_original_aspect_ratio=decrease",
            "-f", "image2pipe", "-c:v", "png", "-",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        creationflags=hidden_window_flags(),
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"could not read a frame at {seconds:.3f}s: {result.stderr.decode('utf-8', errors='replace').strip()[-300:]}")
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")

def contact_sheet(path: str, times: List[float], columns: int = 4, ffmpeg_bin: str = "ffmpeg") -> bytes:
    """Render frames from a video as one labeled grid image.

    Frames are placed left to right, top to bottom, in the order of `times`.
    Each tile is labeled with its number and source time, so the image can be
    read on its own.

    Args:
        path: Video file to sample.
        times: Times of the frames to show, in seconds.
        columns: Maximum number of tiles per row.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        The contact sheet encoded as JPEG.

    Raises:
        RuntimeError: If a frame cannot be decoded.
    """
    with ThreadPoolExecutor(max_workers=6) as pool:
        frames = list(pool.map(lambda t: extract_frame(path, t, ffmpeg_bin=ffmpeg_bin), times))

    columns = max(1, min(columns, len(frames)))
    rows = math.ceil(len(frames) / columns)
    tile_width = max(frame.width for frame in frames)
    tile_height = max(frame.height for frame in frames)
    sheet = Image.new(
        "RGB",
        (columns * tile_width + (columns + 1) * TILE_GAP, rows * tile_height + (rows + 1) * TILE_GAP),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=LABEL_FONT_SIZE)

    for index, (frame, seconds) in enumerate(zip(frames, times)):
        row, column = divmod(index, columns)
        left = TILE_GAP + column * (tile_width + TILE_GAP)
        top = TILE_GAP + row * (tile_height + TILE_GAP)
        sheet.paste(frame, (left + (tile_width - frame.width) // 2, top + (tile_height - frame.height) // 2))
        label = f"#{index + 1}  {format_timestamp(seconds)}"
        box = draw.textbbox((left + 4, top + 4), label, font=font)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
        draw.text((left + 4, top + 4), label, fill=(255, 230, 0), font=font)

    output = io.BytesIO()
    sheet.save(output, "JPEG", quality=82)
    return output.getvalue()
