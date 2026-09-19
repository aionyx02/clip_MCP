"""Tests for per-clip colour adjustments and fades from and to black."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.engine.frames import extract_frame
from app.server import import_asset
from helpers import build_project, clips_of, edit, insert, render, video_track

def luma(path: Path, seconds: float) -> float:
    """Measure the average brightness of one frame.

    Args:
        path: Video to sample.
        seconds: Time of the frame.

    Returns:
        Mean luminance from 0 (black) to 255 (white).
    """
    frame = extract_frame(str(path), seconds).convert("L")
    return sum(frame.get_flattened_data()) / (frame.width * frame.height)

def colourfulness(path: Path, seconds: float) -> float:
    """Measure how far one frame is from grey.

    Args:
        path: Video to sample.
        seconds: Time of the frame.

    Returns:
        The mean spread between the strongest and weakest channel of a pixel.
    """
    pixels = list(extract_frame(str(path), seconds).get_flattened_data())
    return sum(max(pixel) - min(pixel) for pixel in pixels) / len(pixels)

@pytest.fixture
def colour_source(media: Path) -> str:
    """Import a bright, colourful clip.

    Returns:
        The asset ID.
    """
    return import_asset(str(media / "tall.mp4"))["id"]

def test_fade_in_starts_black_and_reaches_the_picture(colour_source: str, tmp_path: Path) -> None:
    project = build_project([video_track(), insert("a", colour_source, 0, 5, video_fade_in=2)], width=320, height=240)
    render(project, tmp_path / "out.mp4", loudness_target=None)
    assert luma(tmp_path / "out.mp4", 0.05) < 12
    assert luma(tmp_path / "out.mp4", 1.0) > luma(tmp_path / "out.mp4", 0.05)
    assert luma(tmp_path / "out.mp4", 4.0) > 40

def test_fade_out_ends_black(colour_source: str, tmp_path: Path) -> None:
    project = build_project([video_track(), insert("a", colour_source, 0, 5, video_fade_out=2)], width=320, height=240)
    render(project, tmp_path / "out.mp4", loudness_target=None)
    assert luma(tmp_path / "out.mp4", 0.5) > 40
    assert luma(tmp_path / "out.mp4", 4.95) < 12

def test_two_clips_can_dip_through_black_between_them(colour_source: str, tmp_path: Path) -> None:
    project = build_project([
        video_track(),
        insert("a", colour_source, 0, 3, video_fade_out=1),
        insert("b", colour_source, 3, 6, video_fade_in=1),
    ], width=320, height=240)
    render(project, tmp_path / "out.mp4", loudness_target=None)
    # Dark at the join, bright on either side, and the length is untouched.
    assert luma(tmp_path / "out.mp4", 2.98) < 15
    assert luma(tmp_path / "out.mp4", 1.0) > 40
    assert luma(tmp_path / "out.mp4", 5.0) > 40

def test_no_fade_leaves_the_first_frame_bright(colour_source: str, tmp_path: Path) -> None:
    project = build_project([video_track(), insert("a", colour_source, 0, 3)], width=320, height=240)
    render(project, tmp_path / "out.mp4", loudness_target=None)
    assert luma(tmp_path / "out.mp4", 0.05) > 40

def test_zero_saturation_renders_black_and_white(colour_source: str, tmp_path: Path) -> None:
    plain = build_project([video_track(), insert("a", colour_source, 0, 3)], width=320, height=240)
    grey = build_project([
        video_track(),
        insert("a", colour_source, 0, 3, color={"saturation": 0}),
    ], width=320, height=240)
    render(plain, tmp_path / "plain.mp4", loudness_target=None)
    render(grey, tmp_path / "grey.mp4", loudness_target=None)
    assert colourfulness(tmp_path / "plain.mp4", 1.0) > 40
    assert colourfulness(tmp_path / "grey.mp4", 1.0) < 8

def test_brightness_lifts_the_picture(colour_source: str, tmp_path: Path) -> None:
    plain = build_project([video_track(), insert("a", colour_source, 0, 3)], width=320, height=240)
    lifted = build_project([
        video_track(),
        insert("a", colour_source, 0, 3, color={"brightness": 0.3}),
    ], width=320, height=240)
    render(plain, tmp_path / "plain.mp4", loudness_target=None)
    render(lifted, tmp_path / "lifted.mp4", loudness_target=None)
    assert luma(tmp_path / "lifted.mp4", 1.0) > luma(tmp_path / "plain.mp4", 1.0) + 20

def test_colour_can_be_set_and_cleared_after_the_fact(colour_source: str, tmp_path: Path) -> None:
    project = build_project([video_track(), insert("a", colour_source, 0, 3)], width=320, height=240)
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "color": {"saturation": 0}}])
    render(project, tmp_path / "grey.mp4", loudness_target=None)
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "clear_color": True}])
    render(project, tmp_path / "back.mp4", loudness_target=None)
    assert colourfulness(tmp_path / "grey.mp4", 1.0) < 8
    assert colourfulness(tmp_path / "back.mp4", 1.0) > 40

def test_fades_longer_than_the_clip_are_refused(colour_source: str) -> None:
    with pytest.raises(ValueError, match="video fades"):
        build_project([
            video_track(),
            insert("a", colour_source, 0, 2, video_fade_in=1.5, video_fade_out=1.5),
        ], width=320, height=240)

def test_set_clip_look_keeps_the_adjustments_it_was_not_asked_about(colour_source: str) -> None:
    # Asking for more colour after asking for more light must not put the light back.
    project = build_project([video_track(), insert("a", colour_source, 0, 3)])
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "color": {"brightness": 0.3}}])
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "color": {"saturation": 1.5}}])
    colour = clips_of(project)[0]["color"]
    assert colour["brightness"] == 0.3 and colour["saturation"] == 1.5

def test_clearing_the_colour_leaves_the_clip_as_shot(colour_source: str) -> None:
    project = build_project([video_track(), insert("a", colour_source, 0, 3)])
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "color": {"brightness": 0.3}}])
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "clear_color": True}])
    assert clips_of(project)[0]["color"] is None

def test_set_clip_look_refuses_to_set_and_clear_the_same_thing(colour_source: str) -> None:
    project = build_project([video_track(), insert("a", colour_source, 0, 3)])
    with pytest.raises(ValidationError, match="not both"):
        edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a",
                        "color": {"saturation": 0}, "clear_color": True}])
