"""Tests for timestamp formatting, cropping, and sheet composition."""

import base64
import io
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from app.engine.frames import (
    SHEET_PIXEL_BUDGET,
    TILE_GAP,
    TILE_SIZE,
    Crop,
    _compose_sheet,
    _crop_to_aspect,
    extract_frame,
    format_timestamp,
    storyboard_sheet,
    tile_size_for,
)
from app.tools.catalog import import_asset
from app.tools.looking import view_frames
from app.tools.shared import MAX_STORYBOARD_TILES

@pytest.mark.parametrize(("seconds", "expected"), [
    (0, "00:00.0"),
    (5.04, "00:05.0"),
    (62.35, "01:02.4"),
    (3723.4, "1:02:03.4"),
])
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected

@pytest.mark.parametrize(("size", "aspect"), [
    ((640, 360), 9 / 16),
    ((360, 640), 16 / 9),
    ((640, 360), 1.0),
    ((400, 400), 9 / 16),
])
def test_crop_to_aspect_matches_the_target_ratio(size: tuple, aspect: float) -> None:
    cropped = _crop_to_aspect(Image.new("RGB", size), aspect)
    assert cropped.width / cropped.height == pytest.approx(aspect, rel=0.01)
    assert cropped.width <= size[0] and cropped.height <= size[1]

def test_crop_to_aspect_keeps_the_frame_when_it_already_matches() -> None:
    frame = Image.new("RGB", (640, 360))
    assert _crop_to_aspect(frame, 16 / 9).size == (640, 360)

def test_crop_to_aspect_crops_evenly_on_both_sides() -> None:
    frame = Image.new("RGB", (100, 100), "black")
    frame.paste(Image.new("RGB", (20, 100), "white"), (40, 0))
    cropped = _crop_to_aspect(frame, 0.5)
    # The centered white stripe must still be centered after cropping.
    assert cropped.size == (50, 100)
    assert cropped.getpixel((25, 50)) == (255, 255, 255)

def test_storyboard_sheet_uses_each_shots_own_file_and_label(media: Path) -> None:
    shots = [
        (str(media / "wide.mp4"), 1.0, "#1 wide"),
        (str(media / "tall.mp4"), 2.0, "#2 tall"),
    ]
    sheet = Image.open(io.BytesIO(storyboard_sheet(shots)))
    assert sheet.format == "JPEG"
    # Two tiles side by side, each at most TILE_SIZE across.
    assert sheet.width <= 2 * TILE_SIZE + 3 * 8

def test_storyboard_sheet_crops_every_tile_to_the_output_aspect(media: Path) -> None:
    shots = [
        (str(media / "wide.mp4"), 1.0, "#1"),
        (str(media / "tall.mp4"), 2.0, "#2"),
        (str(media / "silent.mp4"), 1.0, "#3"),
    ]
    portrait = Image.open(io.BytesIO(storyboard_sheet(shots, aspect=9 / 16, columns=3)))
    landscape = Image.open(io.BytesIO(storyboard_sheet(shots, aspect=16 / 9, columns=3)))
    # Sources of three different shapes must produce the same uniform grid,
    # so the sheet shows the render's framing rather than the sources'.
    assert portrait.height > landscape.height
    assert portrait.width < landscape.width

def assets_for(media: Path) -> tuple:
    """Import the three videos the sheet tests sample.

    Returns:
        `(wide, tall, silent)` asset IDs.
    """
    return tuple(import_asset(str(media / f"{name}.mp4"))["id"] for name in ("wide", "tall", "silent"))

def sheet_text(result) -> list:
    """Split a `view_frames` result into its listing lines.

    Args:
        result: What `view_frames` returned.

    Returns:
        The lines of the text block.
    """
    return result.content[0].text.splitlines()

def sheet_image(result) -> Image.Image:
    """Decode the contact sheet a `view_frames` result carries.

    Args:
        result: What `view_frames` returned.

    Returns:
        The sheet as an RGB image.
    """
    return Image.open(io.BytesIO(base64.b64decode(result.content[1].data))).convert("RGB")

def test_view_frames_samples_one_asset_across_a_range(media: Path) -> None:
    wide, _, _ = assets_for(media)
    lines = sheet_text(view_frames([wide], start=2, end=6, count=4))
    times = [float(line.split("|")[1].split("s")[0]) for line in lines if line.startswith("#")]
    assert len(times) == 4
    assert times == sorted(times) and 2 <= times[0] and times[-1] <= 6

def test_view_frames_covers_several_assets_in_one_call(media: Path) -> None:
    # 148 files used to mean 148 calls; this is the whole point of the change.
    wide, tall, silent = assets_for(media)
    lines = [line for line in sheet_text(view_frames([wide, tall, silent], count=3)) if line.startswith("#")]
    assert len(lines) == 9
    # Each tile says which file it came from, in the order they were asked for.
    assert [line.split("|")[0].split(": ")[1].strip() for line in lines[::3]] == [
        "wide.mp4", "tall.mp4", "silent.mp4",
    ]

def test_view_frames_refuses_more_frames_than_one_sheet_holds(media: Path) -> None:
    wide, tall, silent = assets_for(media)
    with pytest.raises(ValueError, match="more than one sheet holds"):
        view_frames([wide, tall, silent], count=MAX_STORYBOARD_TILES)

def test_view_frames_refuses_a_range_across_several_assets(media: Path) -> None:
    wide, tall, _ = assets_for(media)
    with pytest.raises(ValueError, match="only means something for one asset"):
        view_frames([wide, tall], start=1, end=3)

def test_view_frames_needs_at_least_one_asset() -> None:
    with pytest.raises(ValueError, match="at least one asset"):
        view_frames([])

def test_tile_size_for_keeps_the_default_at_twelve_or_more() -> None:
    assert tile_size_for(12, 16 / 9) == (4, TILE_SIZE)
    assert tile_size_for(36, 16 / 9) == (4, TILE_SIZE)

@pytest.mark.parametrize("count", [1, 2, 4])
def test_tile_size_for_makes_fewer_frames_larger_within_the_budget(count: int) -> None:
    columns, box = tile_size_for(count, 16 / 9)
    assert box > TILE_SIZE
    if count == 4:
        assert columns == 2
    height = round(box / (16 / 9))
    tiles = [Image.new("RGB", (box, height)) for _ in range(count)]
    sheet = Image.open(io.BytesIO(_compose_sheet(tiles, [f"#{index}" for index in range(count)], columns)))
    assert sheet.width * sheet.height <= SHEET_PIXEL_BUDGET

def test_view_frames_shows_fewer_frames_larger(media: Path) -> None:
    wide, _, _ = assets_for(media)
    few = sheet_image(view_frames([wide], count=4))
    many = sheet_image(view_frames([wide], count=12))
    few_tile = (few.width - 3 * TILE_GAP) // 2
    many_tile = (many.width - 5 * TILE_GAP) // 4
    assert few_tile > many_tile
    assert many_tile == TILE_SIZE

def corner_video(folder: Path) -> str:
    """A black landscape clip with a red square filling its top-left quarter.

    Args:
        folder: Directory to write it to.

    Returns:
        The asset ID.
    """
    path = folder / "corner.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=2",
         "-vf", "drawbox=x=0:y=0:w=160:h=90:color=red:t=fill",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return import_asset(str(path))["id"]

def test_view_frames_crop_zooms_into_the_named_part(tmp_path: Path) -> None:
    result = view_frames([corner_video(tmp_path)], count=1, crop=Crop(x=0, y=0, width=0.25, height=0.25))
    sheet = sheet_image(result)
    # The crop is 160x90 of the source, and it is not scaled past that.
    assert sheet.width - 2 * TILE_GAP <= 160
    assert sheet.height - 2 * TILE_GAP <= 90
    red, green, blue = sheet.getpixel((sheet.width // 2, sheet.height // 2))
    assert red > 180 and green < 80 and blue < 80
    assert any("zoomed to x 0.00–0.25" in line for line in sheet_text(result))

@pytest.mark.parametrize("crop", [
    Crop(width=0, height=1),
    Crop(width=1, height=0),
    Crop(x=-0.1, width=0.5),
    Crop(y=-0.1, height=0.5),
    Crop(x=0.8, width=0.4),
    Crop(y=0.8, height=0.4),
])
def test_view_frames_refuses_a_crop_outside_the_frame(media: Path, crop: Crop) -> None:
    wide, _, _ = assets_for(media)
    with pytest.raises(ValueError, match="crop"):
        view_frames([wide], count=1, crop=crop)

def test_view_frames_refuses_a_crop_across_several_assets(media: Path) -> None:
    wide, tall, _ = assets_for(media)
    with pytest.raises(ValueError, match="one asset"):
        view_frames([wide, tall], count=1, crop=Crop(width=0.5, height=0.5))

def test_extract_frame_does_not_enlarge_a_small_source(media: Path) -> None:
    frame = extract_frame(str(media / "silent.mp4"), 1.0, box=1000)
    assert frame.size == (320, 240)

def test_storyboard_sheet_default_keeps_the_old_box(media: Path) -> None:
    sheet = Image.open(io.BytesIO(storyboard_sheet([(str(media / "wide.mp4"), 1.0, "#1")])))
    assert sheet.width - 2 * TILE_GAP == TILE_SIZE
    assert sheet.height - 2 * TILE_GAP == round(TILE_SIZE * 9 / 16)
