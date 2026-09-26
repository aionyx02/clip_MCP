"""Tests for cropping a shot around the face in it when the frame is a different shape.

The decisions are arithmetic on the per-second face measurements, so most of
this checks them directly. Two renders check that the decision reaches the
picture: a white square stands in for the face, placed where a crop from the
middle would miss it.
"""

import subprocess
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from app.engine.builder import FFmpegRenderer
from app.engine.frames import extract_frame
from app.engine.reframe import (
    REFRAME_HOLD_SECONDS,
    REFRAME_MIN_SHOT_SECONDS,
    Framing,
    crop_filter,
    frame_clip,
)
from app.models.media import FaceMeasurement, MediaAnalysis
from app.models.timeline import Clip, TimeRange
from app.server import _framing, _referenced_assets, _reshaped, import_asset, repo
from helpers import build_project, insert, video_track

FPS = Fraction(30)
WIDE, TALL = (1920, 1080), (1080, 1920)

def shot(start: float = 0.0, end: float = 10.0, speed: float = 1.0) -> Clip:
    """A clip of a wide file.

    Args:
        start: In point, in seconds.
        end: Out point.
        speed: How fast it plays.

    Returns:
        The clip.
    """
    return Clip(id="c", asset_id="a", source_range=TimeRange(start=Decimal(str(start)), end=Decimal(str(end))),
                timeline_in=Decimal(0), speed=speed)

def seconds_at(*places: float, start: int = 0) -> list:
    """One face per second, across the frame at each place in turn.

    Args:
        *places: Where the face is, second by second, from 0 at the left to 1.
        start: The second the first one is measured at.

    Returns:
        The measurements.
    """
    return [
        FaceMeasurement(start=float(start + index), end=float(start + index + 1), faces=1, face_share=0.05,
                        face_x=place, face_y=0.5)
        for index, place in enumerate(places)
    ]

def test_a_face_off_centre_is_kept_in_the_crop() -> None:
    framing = frame_clip(shot(), WIDE, TALL, seconds_at(*[0.8] * 10), FPS)
    assert framing.axis == "x"
    assert framing.positions == ((0, 0.8),)

def test_the_crop_never_leaves_the_picture() -> None:
    """A face at the very edge: the crop stops at the edge rather than showing past it."""
    framing = frame_clip(shot(), WIDE, TALL, seconds_at(*[0.99] * 10), FPS)
    span = (9 / 16) / (16 / 9)
    assert framing.positions[0][1] == pytest.approx(1 - span / 2, abs=0.001)

def test_nobody_seen_is_cropped_from_the_middle() -> None:
    assert frame_clip(shot(), WIDE, TALL, [], FPS) is None
    empty = [FaceMeasurement(start=float(second), end=float(second + 1), faces=0) for second in range(10)]
    assert frame_clip(shot(), WIDE, TALL, empty, FPS) is None

def test_a_shot_the_shape_of_its_frame_has_nothing_to_decide() -> None:
    assert frame_clip(shot(), WIDE, WIDE, seconds_at(*[0.8] * 10), FPS) is None

def test_a_tall_shot_in_a_wide_frame_follows_the_face_down_it() -> None:
    faces = [FaceMeasurement(start=float(second), end=float(second + 1), faces=1, face_x=0.5, face_y=0.2)
             for second in range(10)]
    framing = frame_clip(shot(), TALL, WIDE, faces, FPS)
    assert framing.axis == "y"
    assert framing.positions[0][1] == pytest.approx(0.2, abs=0.2)

def test_the_crop_cuts_to_a_face_that_has_moved_and_stayed() -> None:
    framing = frame_clip(shot(), WIDE, TALL, seconds_at(*[0.2] * 5, *[0.8] * 5), FPS)
    # A cut, not a pan: two framings, the second from the second the face arrived.
    assert framing.positions == ((0, 0.2), (150, 0.8))

def test_a_glance_to_the_side_is_not_a_cut() -> None:
    glance = int(REFRAME_HOLD_SECONDS) - 1
    framing = frame_clip(shot(), WIDE, TALL, seconds_at(*[0.2] * 4, *[0.8] * glance, *[0.2] * 5), FPS)
    assert framing.positions == ((0, 0.2),)

def test_no_reframe_right_after_the_shots_own_cut_or_right_before_its_end() -> None:
    """Two cuts that close read as one fumbled one."""
    early = frame_clip(shot(end=8.0), WIDE, TALL, seconds_at(0.8, *[0.2] * 7), FPS)
    # The face moved a second in and stayed, so it gets its framing — but not until the
    # opening framing has lasted long enough to read as a shot of its own.
    assert early.positions[-1][1] == 0.2
    assert all(frame >= REFRAME_MIN_SHOT_SECONDS * 30 for frame, _ in early.positions[1:])
    late_start = 10 - int(REFRAME_MIN_SHOT_SECONDS)
    late = frame_clip(shot(), WIDE, TALL, seconds_at(*[0.2] * late_start, *[0.8] * (10 - late_start)), FPS)
    assert len(late.positions) == 1

def test_a_slowed_shot_counts_its_frames_as_they_play() -> None:
    """At half speed, the face's move five source seconds in lands ten seconds into the clip."""
    framing = frame_clip(shot(speed=0.5), WIDE, TALL, seconds_at(*[0.2] * 5, *[0.8] * 5), FPS)
    assert framing.positions[1] == (300, 0.8)

def test_the_crop_is_written_as_one_expression_per_framing() -> None:
    filtered = crop_filter(608, 1080, Framing("x", ((0, 0.3), (45, 0.7))))
    assert filtered == r"crop=608:1080:x=max(0\,min(iw-ow\,if(lt(n\,45)\,0.3\,0.7)*iw-ow/2))"
    assert crop_filter(608, 1080, None) == "crop=608:1080"

# --- the picture -------------------------------------------------------------------------

def square_video(folder: Path, name: str, boxes: str) -> str:
    """A black landscape clip with a white square standing in for a face.

    Args:
        folder: Directory to write it to.
        name: File name.
        boxes: The `drawbox` filters that place the square.

    Returns:
        The asset ID.
    """
    path = folder / name
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=8",
         "-vf", boxes, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return import_asset(str(path))["id"]

def portrait(project_id: str, path: Path, follow_faces: bool = True) -> None:
    """Render a project as a portrait video, the way `render_project` would.

    Args:
        project_id: The project.
        path: File to write.
        follow_faces: Whether the crop follows the face.
    """
    project = _reshaped(repo.get_project(project_id), "portrait")
    assets = _referenced_assets(project)
    command = FFmpegRenderer().build_command(
        project, assets, str(path), loudness_target=None,
        framing=_framing(project, assets) if follow_faces else None,
    )
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")[-800:]

def brightness(path: Path, seconds: float) -> float:
    """Measure how much of a frame is lit.

    Args:
        path: Video to sample.
        seconds: Time of the frame.

    Returns:
        Mean luminance, 0 to 255.
    """
    frame = extract_frame(str(path), seconds).convert("L")
    return sum(frame.get_flattened_data()) / (frame.width * frame.height)

def test_a_portrait_render_keeps_the_face_a_crop_from_the_middle_loses(tmp_path: Path) -> None:
    asset = square_video(tmp_path, "right.mp4", "drawbox=x=480:y=140:w=80:h=80:color=white:t=fill")
    repo.save_analysis(MediaAnalysis(asset_id=asset, duration=8.0, faces=seconds_at(*[0.8125] * 8)))
    project = build_project([video_track(), insert("a", asset, 0, 8)], width=640, height=360)

    followed, middle = tmp_path / "followed.mp4", tmp_path / "middle.mp4"
    portrait(project, followed)
    portrait(project, middle, follow_faces=False)
    assert brightness(followed, 2.0) > 20
    assert brightness(middle, 2.0) < 2

def test_a_portrait_render_cuts_to_the_face_when_it_moves(tmp_path: Path) -> None:
    asset = square_video(
        tmp_path, "moves.mp4",
        "drawbox=x=80:y=140:w=80:h=80:color=white:t=fill:enable='lt(t,4)',"
        "drawbox=x=480:y=140:w=80:h=80:color=white:t=fill:enable='gte(t,4)'",
    )
    repo.save_analysis(MediaAnalysis(
        asset_id=asset, duration=8.0, faces=seconds_at(*[0.1875] * 4, *[0.8125] * 4),
    ))
    project = build_project([video_track(), insert("a", asset, 0, 8)], width=640, height=360)
    out = tmp_path / "moves_portrait.mp4"
    portrait(project, out)
    assert brightness(out, 2.0) > 20
    assert brightness(out, 6.0) > 20

def test_the_shapes_keep_the_short_side() -> None:
    from app.models.timeline import Project

    wide = Project(id="p", width=1920, height=1080)
    assert (_reshaped(wide, "portrait").width, _reshaped(wide, "portrait").height) == (1080, 1920)
    assert (_reshaped(wide, "square").width, _reshaped(wide, "square").height) == (1080, 1080)
    assert _reshaped(wide, None) is wide

def test_the_storyboard_shows_the_framing_the_render_will_have(tmp_path: Path) -> None:
    """Checking a portrait cut before rendering is only worth anything if it is cropped the same way."""
    import base64
    import io

    from PIL import Image

    from app.server import preview_project

    asset = square_video(tmp_path, "board.mp4", "drawbox=x=480:y=140:w=80:h=80:color=white:t=fill")
    repo.save_analysis(MediaAnalysis(asset_id=asset, duration=8.0, faces=seconds_at(*[0.8125] * 8)))
    project = build_project([video_track(), insert("a", asset, 0, 8)], width=640, height=360)

    def lit(**options) -> float:
        _, image = preview_project(project, count=1, frame="portrait", **options).content
        sheet = Image.open(io.BytesIO(base64.b64decode(image.data))).convert("L")
        assert sheet.height > sheet.width
        # The bottom half of the tile, clear of the label in its corner.
        lower = sheet.crop((0, sheet.height // 2, sheet.width, sheet.height))
        return sum(lower.get_flattened_data()) / (lower.width * lower.height)

    # Against each other rather than against zero: the sheet's grey border lifts both.
    assert lit() > lit(follow_faces=False) + 10
