import io
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.engine import resources, stills
from app.engine.builder import WHOLE_DIM
from app.engine.ffmpeg import hidden_window_flags
from app.engine.reframe import Placement

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

    Waits for one of the server's decoding slots first, so several sheets
    being built at the same time cannot put an unbounded number of decoders
    on the machine at once.

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
    # A photo is the same picture at every moment, and seeking into one finds nothing.
    seek = [] if stills.is_image(path) else ["-ss", f"{max(seconds, 0):.3f}"]
    # Wait for a slot, then decode on one thread: a single frame is wanted, and frame
    # threading would hold several decoded pictures of a 4K source in memory to produce it.
    with resources.decoder_slot():
        result = subprocess.run(
            [
                ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-threads", "1", *seek, "-i", path,
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

def _crop_to_aspect(frame: Image.Image, aspect: float, placed: Optional[Placement] = None) -> Image.Image:
    """Fit a frame to a width-to-height ratio the way the render does.

    The picture is scaled to cover the frame and the overflow is cropped away
    — evenly on both sides, or around where the render's framing puts it, and
    closer in when it is zoomed. A shot shown whole is instead fitted inside,
    over a blurred and dimmed copy of itself.

    Args:
        frame: Frame to fit.
        aspect: Target width divided by target height.
        placed: How the render places it; None for a crop from the middle.

    Returns:
        A picture of `frame` with that ratio.
    """
    placed = placed or Placement()
    width, height = frame.size
    if placed.whole:
        cover = _crop_to_aspect(frame, aspect)
        background = cover.filter(ImageFilter.GaussianBlur(max(2, cover.width // 40)))
        background = ImageEnhance.Brightness(background).enhance(1 + WHOLE_DIM)
        scale = min(background.width / width, background.height / height)
        shown = frame.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
        background.paste(shown, ((background.width - shown.width) // 2, (background.height - shown.height) // 2))
        return background
    wide = width > height * aspect
    kept_w = max(1, round((height * aspect if wide else width) / placed.zoom))
    kept_h = max(1, round((height if wide else width / aspect) / placed.zoom))
    across, down = (placed.centre, placed.cross) if wide else (placed.cross, placed.centre)
    left = min(max(0, round(across * width - kept_w / 2)), width - kept_w)
    top = min(max(0, round(down * height - kept_h / 2)), height - kept_h)
    return frame.crop((left, top, left + kept_w, top + kept_h))

def still(
    path: str,
    seconds: float,
    width: int,
    height: int,
    placed: Optional[Placement] = None,
    ffmpeg_bin: str = "ffmpeg",
) -> Image.Image:
    """Take one frame at full size, framed the way the render frames it.

    Args:
        path: Video file.
        seconds: Time of the frame in the file.
        width: Width to deliver it at.
        height: Height to deliver it at.
        placed: How the render places it, as `reframe.placement_at` says;
            None for a crop from the middle.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        The frame, fitted to the delivery shape and scaled to its size.

    Raises:
        RuntimeError: If the frame cannot be decoded.
    """
    frame = extract_frame(path, seconds, max_size=max(width, height) * 2, ffmpeg_bin=ffmpeg_bin)
    return _crop_to_aspect(frame, width / height, placed).resize((width, height), Image.LANCZOS)

def _compose_sheet(
    frames: Sequence[Image.Image],
    labels: Sequence[str],
    columns: int,
    borders: Optional[Sequence[Optional[Tuple[int, int, int]]]] = None,
) -> bytes:
    """Lay decoded frames out as one labeled grid image.

    Frames are placed left to right, top to bottom, and each tile is labeled
    in its top-left corner, so the image can be read on its own.

    Args:
        frames: Decoded frames, in the order to show them.
        labels: Caption for each frame, in the same order.
        columns: Maximum number of tiles per row.
        borders: A colour to frame each tile in, or None for no frame, in the
            same order; None for no frames at all.

    Returns:
        The grid encoded as JPEG.
    """
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

    for index, (frame, label) in enumerate(zip(frames, labels)):
        row, column = divmod(index, columns)
        left = TILE_GAP + column * (tile_width + TILE_GAP)
        top = TILE_GAP + row * (tile_height + TILE_GAP)
        sheet.paste(frame, (left + (tile_width - frame.width) // 2, top + (tile_height - frame.height) // 2))
        colour = borders[index] if borders else None
        if colour:
            draw.rectangle((left, top, left + tile_width - 1, top + tile_height - 1), outline=colour, width=5)
        box = draw.textbbox((left + 4, top + 4), label, font=font)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
        draw.text((left + 4, top + 4), label, fill=(255, 230, 0), font=font)

    output = io.BytesIO()
    sheet.save(output, "JPEG", quality=82)
    return output.getvalue()

def storyboard_sheet(
    shots: Sequence[Tuple],
    aspect: Optional[float] = None,
    columns: int = 4,
    ffmpeg_bin: str = "ffmpeg",
    borders: Optional[Sequence[Optional[Tuple[int, int, int]]]] = None,
) -> bytes:
    """Render frames drawn from several videos as one labeled grid image.

    Each tile comes from its own file and carries its own caption, so an
    edited sequence can be shown in timeline order, and so can a run of
    separate files being looked over.

    Args:
        shots: One `(path, seconds, label)` per tile, in the order to show
            them, optionally with a fourth item: how the render places that
            tile, as `reframe.placement_at` says.
        aspect: Output width divided by height. When given, every tile is
            cropped to it, so the sheet shows the framing the render will have
            rather than the framing of the source files.
        columns: Maximum number of tiles per row.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.
        borders: A colour to frame each tile in, or None, per tile.

    Returns:
        The grid encoded as JPEG.

    Raises:
        RuntimeError: If a frame cannot be decoded.
    """
    with ThreadPoolExecutor(max_workers=resources.max_decoders()) as pool:
        frames = list(pool.map(lambda shot: extract_frame(shot[0], shot[1], ffmpeg_bin=ffmpeg_bin), shots))
    if aspect is not None:
        # One size for every tile, so sources of different shapes line up in the grid.
        size = (TILE_SIZE, max(1, round(TILE_SIZE / aspect))) if aspect >= 1 else (max(1, round(TILE_SIZE * aspect)), TILE_SIZE)
        frames = [
            _crop_to_aspect(frame, aspect, shot[3] if len(shot) > 3 else None).resize(size, Image.LANCZOS)
            for frame, shot in zip(frames, shots)
        ]
    return _compose_sheet(frames, [shot[2] for shot in shots], columns, borders)
