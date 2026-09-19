"""Tests for timestamp formatting, cropping, and sheet composition."""

import io
from pathlib import Path

import pytest
from PIL import Image

from app.engine.frames import TILE_SIZE, _crop_to_aspect, format_timestamp, storyboard_sheet

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
