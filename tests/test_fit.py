"""A shot of another shape than its frame: cropped around a face, or shown whole over a blurred copy of itself."""

from fractions import Fraction
from pathlib import Path

import pytest

from app import server
from app.engine.frames import _crop_to_aspect
from app.engine.reframe import Placement, crop_filter, place_clip
from app.models.media import FaceMeasurement
from app.models.timeline import Clip, ClipFit, TimeRange
from helpers import build_project, clips_of, edit, insert, render, video_track

WIDE, TALL = (1920, 1080), (1080, 1920)


def shot(fit: ClipFit = None) -> Clip:
    """A ten-second clip of one file."""
    return Clip(id="a", asset_id="x", timeline_in=0, source_range=TimeRange(start=0, end=10), fit=fit)


def face_at(where: float) -> list:
    """A face seen in every second of the file, at one place across it."""
    return [FaceMeasurement(start=second, end=second + 1, faces=1, face_x=where, face_y=0.4, face_size=0.2)
            for second in range(10)]


def test_a_shot_decides_for_itself_unless_told() -> None:
    fps = Fraction(30)
    # Somebody in it: cropped around them.
    followed = place_clip(shot(), WIDE, TALL, face_at(0.8), fps)
    assert not followed.whole and followed.positions[0][1] > 0.6
    # Nobody in it, and a crop would keep a third: all of it, over a blurred copy.
    assert place_clip(shot(), WIDE, TALL, [], fps).whole
    # 4:3 in a 16:9 frame loses little: cropped from the middle, as always.
    assert place_clip(shot(), (1440, 1080), WIDE, [], fps) is None
    # The same shape has nothing to decide.
    assert place_clip(shot(), WIDE, WIDE, [], fps) is None


def test_what_the_shot_was_told_wins() -> None:
    fps = Fraction(30)
    assert place_clip(shot(ClipFit(mode="whole")), WIDE, TALL, face_at(0.5), fps).whole
    # Cropped anyway, and with no face, from the middle.
    assert place_clip(shot(ClipFit(mode="fill")), WIDE, TALL, [], fps) is None
    # Put somewhere by hand: it stays there, whatever the face does.
    pinned = place_clip(shot(ClipFit(mode="fill", center_x=0.2)), WIDE, TALL, face_at(0.9), fps)
    assert pinned.positions == ((0, 0.2),) and pinned.zoom == 1
    # Zoomed in, a shot of the same shape is cropped too, along both axes.
    zoomed = place_clip(shot(ClipFit(mode="fill", zoom=2, center_x=0.7, center_y=0.3)), WIDE, WIDE, [], fps)
    across, down = (zoomed.positions[0][1], zoomed.cross) if zoomed.axis == "x" else (zoomed.cross, zoomed.positions[0][1])
    assert zoomed.zoom == 2 and (across, down) == (0.7, 0.3)
    assert ":x=" in crop_filter(1920, 1080, zoomed) and ":y=" in crop_filter(1920, 1080, zoomed)


def test_a_still_is_drawn_the_way_the_render_draws_it() -> None:
    from PIL import Image

    frame = Image.new("RGB", (1920, 1080), (200, 30, 30))
    whole = _crop_to_aspect(frame, 9 / 16, Placement(whole=True))
    assert abs(whole.width / whole.height - 9 / 16) < 0.01
    zoomed = _crop_to_aspect(frame, 16 / 9, Placement(zoom=2))
    assert zoomed.size == (960, 540)


def test_a_landscape_shot_in_a_vertical_video_renders_whole_over_its_blur(media: Path, tmp_path: Path) -> None:
    from app.engine.frames import extract_frame

    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2)], width=360, height=640, name="直式")
    stored = server.repo.get_project(project)
    framing = server._framing(stored, server._referenced_assets(stored))
    assert framing["a"].whole
    render(project, tmp_path / "out.mp4", loudness_target=None, framing=framing)
    picture = extract_frame(str(tmp_path / "out.mp4"), 1.0, max_size=640).convert("L")
    # The bands above and below the shot are its blurred copy, not black.
    top = picture.crop((0, 0, picture.width, picture.height // 6))
    assert sum(top.getdata()) / (top.width * top.height) > 20


def test_the_ai_and_the_editor_set_how_a_shot_sits(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2)], width=360, height=640, name="放法")
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a",
                    "fit": {"mode": "fill", "center_x": 0.25, "zoom": 1.5}}])
    assert clips_of(project, "main")[0]["fit"] == {"mode": "fill", "center_x": 0.25, "center_y": None, "zoom": 1.5}
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "clear_fit": True}])
    assert clips_of(project, "main")[0]["fit"] is None
    with pytest.raises(ValueError, match="either fit or clear_fit"):
        edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a",
                        "fit": {"mode": "whole"}, "clear_fit": True}])
