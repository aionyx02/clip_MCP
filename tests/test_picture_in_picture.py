"""Tests for drawing one video track on top of another."""

from pathlib import Path

import pytest

from app.engine.frames import extract_frame
from app.server import import_asset, preview_project
from helpers import build_project, clips_of, edit, insert, layout, level, render, video_track

def brightness(path: Path, seconds: float, box: tuple) -> float:
    """Measure the average brightness of a rectangle of one frame.

    Args:
        path: Video to sample.
        seconds: Time of the frame.
        box: Region as `(left, top, right, bottom)` fractions of the frame.

    Returns:
        Mean luminance from 0 to 255.
    """
    frame = extract_frame(str(path), seconds, max_size=640).convert("L")
    width, height = frame.size
    left, top, right, bottom = box
    region = frame.crop((int(width * left), int(height * top), int(width * right), int(height * bottom)))
    return sum(region.get_flattened_data()) / (region.width * region.height)

@pytest.fixture
def sources(media: Path) -> dict:
    """Import a black base clip and a bright clip to lay over it.

    Returns:
        The asset IDs keyed by role.
    """
    return {
        "base": import_asset(str(media / "black.mp4"))["id"],
        "top": import_asset(str(media / "tall.mp4"))["id"],
    }

@pytest.fixture
def inset(sources: dict) -> str:
    """Build a 6 s black base with a bright inset in the bottom right from 2 s to 4 s.

    Returns:
        The project ID.
    """
    return build_project([
        video_track("base"),
        insert("b", sources["base"], 0, 6, track_id="base"),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "pip", "asset_id": sources["top"],
         "source_range": {"start": 0, "end": 2}, "timeline_in": 2,
         "layout": {"x": 0.6, "y": 0.6, "width": 0.35, "height": 0.35}},
    ], width=640, height=360)

def test_the_inset_is_drawn_only_while_it_is_on_the_timeline(inset: str, tmp_path: Path) -> None:
    render(inset, tmp_path / "out.mp4", loudness_target=None)
    corner = (0.62, 0.62, 0.93, 0.93)
    assert brightness(tmp_path / "out.mp4", 1.0, corner) < 10
    assert brightness(tmp_path / "out.mp4", 3.0, corner) > 40
    assert brightness(tmp_path / "out.mp4", 5.0, corner) < 10

def test_the_inset_stays_inside_its_box(inset: str, tmp_path: Path) -> None:
    render(inset, tmp_path / "out.mp4", loudness_target=None)
    # The base is black, so anything bright outside the box would be the inset spilling out.
    assert brightness(tmp_path / "out.mp4", 3.0, (0.0, 0.0, 0.55, 1.0)) < 10
    assert brightness(tmp_path / "out.mp4", 3.0, (0.0, 0.0, 1.0, 0.55)) < 10

def test_a_clip_without_a_layout_covers_the_whole_frame(sources: dict, tmp_path: Path) -> None:
    project = build_project([
        video_track("base"),
        insert("b", sources["base"], 0, 6, track_id="base"),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "cover", "asset_id": sources["top"],
         "source_range": {"start": 0, "end": 2}, "timeline_in": 2},
    ], width=640, height=360)
    render(project, tmp_path / "out.mp4", loudness_target=None)
    assert brightness(tmp_path / "out.mp4", 3.0, (0.1, 0.1, 0.9, 0.9)) > 40

def test_the_base_track_still_decides_the_length(sources: dict) -> None:
    project = build_project([
        video_track("base"),
        insert("b", sources["base"], 0, 3, track_id="base"),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "long", "asset_id": sources["top"],
         "source_range": {"start": 0, "end": 5}, "timeline_in": 0},
    ], width=640, height=360)
    assert layout(project, "base")[-1][2] == 3.0

def test_the_inset_brings_its_own_sound(audio_media: Path, media: Path, tmp_path: Path) -> None:
    base = import_asset(str(media / "black.mp4"))["id"]
    talker = import_asset(str(audio_media / "loud.mp4"))["id"]
    project = build_project([
        video_track("base"),
        insert("b", base, 0, 5, track_id="base"),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "pip", "asset_id": talker,
         "source_range": {"start": 0, "end": 2}, "timeline_in": 1,
         "layout": {"x": 0.6, "y": 0.6, "width": 0.3, "height": 0.3}},
    ], width=640, height=360)
    render(project, tmp_path / "out.mp4", loudness_target=None)
    # The base clip is silent, so any sound at all came from the inset.
    assert level(tmp_path / "out.mp4", 1.5, 1.0) > level(tmp_path / "out.mp4", 4.0, 0.8) + 20

def test_a_box_outside_the_frame_is_refused(sources: dict) -> None:
    with pytest.raises(ValueError, match="outside the frame"):
        build_project([
            video_track("base"),
            insert("b", sources["base"], 0, 3, track_id="base"),
            {"action": "add_track", "track_id": "top", "track_type": "video"},
            {"action": "add_clip", "track_id": "top", "clip_id": "pip", "asset_id": sources["top"],
             "source_range": {"start": 0, "end": 2}, "timeline_in": 0,
             "layout": {"x": 0.8, "y": 0.1, "width": 0.5, "height": 0.3}},
        ], width=640, height=360)

def test_the_layout_can_be_changed_and_cleared_afterwards(inset: str) -> None:
    edit(inset, [{"action": "set_clip_look", "track_id": "top", "clip_id": "pip",
                  "layout": {"x": 0.05, "y": 0.05, "width": 0.25, "height": 0.25}}])
    assert clips_of(inset, "top")[0]["layout"]["x"] == 0.05
    edit(inset, [{"action": "set_clip_look", "track_id": "top", "clip_id": "pip", "clear_layout": True}])
    assert clips_of(inset, "top")[0]["layout"] is None

def test_the_storyboard_shows_the_sequence_and_names_the_insets(inset: str) -> None:
    lines = preview_project(inset).content[0].text.splitlines()
    # The inset must not be mixed in as if it were a shot in the sequence.
    assert all("clip pip" not in line for line in lines if line.startswith("#"))
    assert any(line.startswith("-- track top: clip pip") and "inset at" in line for line in lines)
    assert any("60% across and 60% down" in line for line in lines)

def test_the_storyboard_says_when_the_tiles_are_not_what_will_be_seen(inset: str) -> None:
    # Without a layout the clip fills the frame, so it replaces the tiles rather than
    # sitting in a corner of them. A storyboard of a B-roll pass that did not say so
    # would look exactly like a pass that never happened.
    edit(inset, [{"action": "set_clip_look", "track_id": "top", "clip_id": "pip", "clear_layout": True}])
    lines = preview_project(inset).content[0].text.splitlines()
    covering = next(line for line in lines if line.startswith("-- track top: clip pip"))
    assert "covering picture, over the whole frame" in covering
    assert "the sequence underneath" in covering

def test_a_layout_on_the_base_track_is_refused(sources: dict) -> None:
    # The base track is not drawn on top of anything, so a box there would be silently ignored.
    with pytest.raises(ValueError, match="above the base one"):
        build_project([
            video_track("base"),
            insert("b", sources["base"], 0, 3, track_id="base",
                   layout={"x": 0.6, "y": 0.6, "width": 0.3, "height": 0.3}),
        ], width=640, height=360)

def test_splitting_an_inset_keeps_the_box_on_both_halves(inset: str) -> None:
    edit(inset, [{"action": "split_clip", "track_id": "top", "clip_id": "pip", "new_clip_id": "pip2", "at": 3}])
    assert [clip["layout"]["x"] for clip in clips_of(inset, "top")] == [0.6, 0.6]

def test_a_clip_drawn_over_the_sequence_plays_at_its_speed(media: Path, tmp_path: Path) -> None:
    """It used to be read at its speed and played at normal speed, showing half of what it read."""
    import subprocess

    changing = tmp_path / "red_then_green.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:r=30:d=1", "-f", "lavfi", "-i", "color=c=green:s=320x240:r=30:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(changing),
    ], check=True, capture_output=True)
    over = import_asset(str(changing))["id"]
    base = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([
        video_track(), insert("a", base, 0, 2),
        video_track("broll"), {"action": "add_clip", "track_id": "broll", "clip_id": "b", "asset_id": over,
                               "source_range": {"start": 0, "end": 2}, "timeline_in": 0, "speed": 2.0},
    ], width=320, height=240)
    out = tmp_path / "fast_over.mp4"
    render(project, out, loudness_target=None)
    # Three quarters of a second in at double speed is a second and a half into the source.
    red, green, _ = extract_frame(str(out), 0.75).resize((1, 1)).getpixel((0, 0))
    assert green > red
