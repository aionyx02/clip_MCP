"""Slow moves over video: a push that closes in on the speaker, drawn the same in the render, the stills and the editor.

The arithmetic is checked directly. The renders check that it reaches the
picture: a soft white dot, or a white square standing in for a face, is
tracked frame by frame through the real render path.
"""

import subprocess
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from app import runtime
from app.engine import builder, moves, playback, stills
from app.engine.interchange import left_behind
from app.engine.reframe import box_of, frame_project, move_notes, placement_at
from app.models.media import FaceMeasurement, MediaAnalysis
from app.models.timeline.parts import Clip, TimeRange
from app.tools import catalog, rendering, shared
from helpers import build_project, clips_of, edit, insert, render, video_track

FPS = Fraction(30)


def dot_video(folder: Path, name: str, size: str, where: str, seconds: int = 3) -> str:
    """A dark clip with a soft white dot on it, imported.

    Args:
        folder: Directory to write it to.
        name: File name.
        size: Frame size, such as `360x640`.
        where: The dot's centre, as `x:y` in pixels.
        seconds: Length of the file.

    Returns:
        The asset ID.
    """
    x, y = where.split(":")
    path = folder / name
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=0x141414:s={size}:r=30:d={seconds}",
         "-vf", f"drawbox=x={int(x) - 6}:y={int(y) - 6}:w=12:h=12:color=white:t=fill,gblur=sigma=3",
         "-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return catalog.import_asset(str(path))["id"]


def frames_of(path: Path, width: int, height: int) -> np.ndarray:
    """Decode a render's every frame as grey."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width).astype(np.float64)


def centroids(frames: np.ndarray) -> np.ndarray:
    """Where the light is in each frame, by its weighted centre, in pixels."""
    ys, xs = np.mgrid[0:frames.shape[1], 0:frames.shape[2]]
    lit = np.clip(frames - 40, 0, None)
    weight = lit.sum(axis=(1, 2))
    return np.stack([(lit * xs).sum(axis=(1, 2)) / weight, (lit * ys).sum(axis=(1, 2)) / weight], axis=1)


def pushed(project: str, clip_id: str = "a", motion: str = "push") -> None:
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": clip_id, "motion": motion}])


def framing(project: str, frame=None):
    stored = rendering.reshaped(runtime.repo.get_project(project), frame)
    assets = shared.referenced_assets(stored)
    return stored, assets, rendering.framing_of(stored, assets)


# --- the arithmetic ----------------------------------------------------------------------

def test_a_push_closes_in_by_the_photo_amount_and_keeps_its_anchor_where_it_is() -> None:
    assert moves.window_at("push", 0) == (1.0, 0.0, 0.0)
    zoom, left, top = moves.window_at("push", 1, (0.3, 0.25))
    assert zoom == pytest.approx(1 + stills.MOTION_AMOUNT)
    # The anchor is as far into the part shown as it is into the whole picture: it does not move on screen.
    assert (0.3 - left) * zoom == pytest.approx(0.3) and (0.25 - top) * zoom == pytest.approx(0.25)
    assert moves.window_at("pull", 1)[0] == pytest.approx(1.0)
    assert moves.window_at("pan_left", 0.25)[1] == pytest.approx(0.25 * (1 - 1 / (1 + stills.MOTION_AMOUNT)))
    assert moves.window_at("none", 0.7) == (1.0, 0.0, 0.0)


def test_a_video_holds_still_unless_told_and_a_photo_moves_unless_told() -> None:
    clip = Clip(id="c", asset_id="a", source_range=TimeRange(start=Decimal(0), end=Decimal(4)), timeline_in=Decimal(0))
    assert moves.motion_of(clip, still=False) == "none"
    assert moves.motion_of(clip, still=True) == stills.DEFAULT_MOTION
    assert moves.motion_of(clip.model_copy(update={"motion": "pull"}), still=False) == "pull"
    assert moves.perspective("none", 60) is None
    # Counted over exactly the frames it is given, first to last.
    assert "on/59" in moves.perspective("push", 60) and "eval=frame" in moves.perspective("push", 60)


def test_a_sped_up_clip_pushes_over_its_length_on_screen() -> None:
    # Three seconds of file at 1.5x is two seconds on screen: 60 frames, so the push ends on frame 59.
    clip = Clip(id="c", asset_id="a", source_range=TimeRange(start=Decimal(0), end=Decimal(3)), timeline_in=Decimal(0),
                speed=1.5, motion="push")
    assert moves.picture_frames(clip, FPS) == 60
    chain = builder._picture_chain(clip, 60, 360, 640, FPS, None)
    assert "perspective=" in chain and "on/59" in chain
    # And the moment of the file that plays last on screen is the end of the push.
    assert moves.window_at_source("push", clip, 3 - 1.5 / 30, FPS)[0] == pytest.approx(1 + stills.MOTION_AMOUNT)
    assert moves.window_at_source("push", clip, 1.5, FPS)[0] == pytest.approx(1 + stills.MOTION_AMOUNT * 30 / 59)


def test_footage_with_pixels_to_spare_is_placed_larger_so_the_push_closes_in_on_real_detail() -> None:
    clip = Clip(id="c", asset_id="a", source_range=TimeRange(start=Decimal(0), end=Decimal(4)), timeline_in=Decimal(0),
                motion="push")
    assert builder.move_room(clip, 1080, 1920, None, (1080, 1920)) == 1.0
    assert builder.move_room(clip, 1080, 1920, None, (2160, 3840)) == pytest.approx(1 + stills.MOTION_AMOUNT)
    assert builder.move_room(clip, 1080, 1920, None, (1134, 2016)) == pytest.approx(1.05)
    chain = builder._picture_chain(clip, 120, 1080, 1920, FPS, None, source=(2160, 3840))
    assert "crop=1168:2074" in chain and chain.index("perspective=") < chain.index("scale=1080:1920")


# --- the render --------------------------------------------------------------------------

def test_a_pushed_video_clip_closes_in_smoothly_over_the_whole_clip(tmp_path: Path) -> None:
    """The dot drifts away from the middle by the push amount, one small step outwards on every frame."""
    asset = dot_video(tmp_path, "dot.mp4", "360x640", "90:160")
    project = build_project([video_track(), insert("a", asset, 0, 3)], width=360, height=640, name="推近")
    pushed(project)
    assert clips_of(project)[0]["motion"] == "push"
    out = tmp_path / "pushed.mp4"
    render(project, out, loudness_target=None)
    where = centroids(frames_of(out, 360, 640))
    assert len(where) == 90
    outward = (where - (180, 320)) / (where[0] - (180, 320))
    assert outward[-1] == pytest.approx([1 + stills.MOTION_AMOUNT] * 2, abs=0.01)
    steps = np.diff(where, axis=0)
    # Left of and above the middle, so every step is left and up; never a step back.
    assert (steps <= 0.02).all(), steps.max(axis=0)
    fitted = np.polyval(np.polyfit(np.arange(90), where[:, 0], 2), np.arange(90))
    assert np.sqrt(((where[:, 0] - fitted) ** 2).mean()) < 0.1


def test_a_push_closes_in_on_the_face_and_leaves_it_where_it_is(tmp_path: Path) -> None:
    """Pushing on the middle would drift a face in the top corner towards the edge; it stays put instead."""
    asset = dot_video(tmp_path, "face.mp4", "360x640", "90:192")
    runtime.repo.save_analysis(MediaAnalysis(asset_id=asset, duration=3.0, faces=[
        FaceMeasurement(start=float(second), end=float(second + 1), faces=1, face_share=0.05, face_x=0.25, face_y=0.3)
        for second in range(3)
    ]))
    project = build_project([video_track(), insert("a", asset, 0, 3)], width=360, height=640, name="推向臉")
    pushed(project)
    stored, assets, framed = framing(project)
    assert framed["a"].anchor == (0.25, 0.3)
    out = tmp_path / "face_pushed.mp4"
    render(project, out, loudness_target=None, framing=framed)
    where = centroids(frames_of(out, 360, 640))
    assert where[-1] == pytest.approx(where[0], abs=0.5)


def test_a_push_on_a_portrait_reframe_stays_on_the_face_the_crop_follows(tmp_path: Path) -> None:
    """A landscape shot cut down to a third of its width, around a face too near its right edge to centre."""
    asset = dot_video(tmp_path, "wide.mp4", "640x360", "580:120")
    runtime.repo.save_analysis(MediaAnalysis(asset_id=asset, duration=3.0, faces=[
        FaceMeasurement(start=float(second), end=float(second + 1), faces=1, face_share=0.05,
                        face_x=580 / 640, face_y=120 / 360)
        for second in range(3)
    ]))
    project = build_project([video_track(), insert("a", asset, 0, 3)], width=640, height=360, name="直式推近")
    pushed(project)
    stored, assets, framed = framing(project, "portrait")
    # The crop is held inside the picture, so the face sits right of the middle of it; the push closes in there.
    assert framed["a"].axis == "x" and framed["a"].anchor == pytest.approx((0.71, 0.333), abs=0.01)
    out = tmp_path / "portrait.mp4"
    command = builder.FFmpegRenderer().build_command(stored, assets, str(out), loudness_target=None, framing=framed)
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")[-800:]
    where = centroids(frames_of(out, stored.width, stored.height))
    assert where[-1] == pytest.approx(where[0], abs=0.6)


def test_the_stills_of_a_cut_show_the_push_as_far_as_it_has_gone(tmp_path: Path) -> None:
    from app.engine.frames import still

    asset = dot_video(tmp_path, "board.mp4", "360x640", "90:160")
    project = build_project([video_track(), insert("a", asset, 0, 3)], width=360, height=640, name="推近定格")
    pushed(project)
    stored, assets, framed = framing(project)
    clip = stored.base_video_track.clips[0]
    zooms = [placement_at(framed["a"], clip, seconds, FPS).window[0] for seconds in (0, 1.5, 3 - 1 / 30)]
    assert zooms[0] == 1.0 and zooms[0] < zooms[1] < zooms[2] == pytest.approx(1 + stills.MOTION_AMOUNT)
    dots = []
    for seconds in (0.0, 2.9):
        placed = placement_at(framed["a"], clip, seconds, FPS)
        picture = np.asarray(still(assets[asset].path, seconds, 360, 640, placed).convert("L"), dtype=np.float64)
        dots.append(centroids(picture[None])[0])
    # Further from the middle at the end, by as far as the push has gone by then.
    closer = placement_at(framed["a"], clip, 2.9, FPS).window[0]
    assert (dots[1] - (180, 320)) / (dots[0] - (180, 320)) == pytest.approx([closer] * 2, rel=0.01)


def test_a_photo_moves_through_the_same_filter_as_video(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "photo.png"
    Image.new("RGB", (720, 1280), (30, 30, 30)).save(path)
    asset = catalog.import_asset(str(path))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2)], width=360, height=640, name="照片推近")
    stored = runtime.repo.get_project(project)
    chain = builder._picture_chain(stored.base_video_track.clips[0], 60, 360, 640, FPS, None, still=True,
                                   source=(720, 1280))
    assert "perspective=" in chain and "zoompan" not in chain and "loop=loop=59" in chain
    held = builder._picture_chain(stored.base_video_track.clips[0].model_copy(update={"motion": "none"}),
                                  60, 360, 640, FPS, None, still=True)
    assert "perspective" not in held and "zoompan=z=1" in held


# --- what is refused and what is said ----------------------------------------------------

def test_a_move_too_short_to_read_and_one_on_a_title_card_are_refused(media: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 1.2), insert("b", asset, 2, 5)],
                            width=320, height=180, name="太短")
    with pytest.raises(ValueError, match="1.20s.*jolt"):
        pushed(project)
    pushed(project, "a", "none")
    edit(project, [{"action": "add_title_card", "track_id": "main", "clip_id": "t", "title": "開場",
                     "before_clip_id": "b"}])
    with pytest.raises(ValueError, match="title card"):
        pushed(project, "t")


def test_the_check_says_when_a_push_softens_footage_shot_at_the_output_size(media: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 4)], width=640, height=360, name="變軟")
    assert "moves" not in rendering.check_render(project)
    pushed(project)
    [note] = rendering.check_render(project)["moves"]
    assert "clip a" in note and "593x333 pixels of the 640x360 source" in note and "enlarged 8%" in note
    # A split can leave a half too short to push over, which nothing refuses: it is said instead.
    edit(project, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at": 1.0}])
    notes = rendering.check_render(project)["moves"]
    assert any("1.00s reads as a jolt" in note for note in notes)


def test_footage_shot_larger_than_the_frame_is_not_said_to_soften(media: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 4)], width=320, height=180, name="夠大")
    pushed(project)
    stored, assets, framed = framing(project)
    assert move_notes(stored, shared.sized(assets), framed) == []


def test_the_editor_is_told_how_a_video_clip_moves_and_what_it_closes_in_on(media: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 4), insert("b", asset, 4, 8)],
                            width=640, height=360, name="編輯器推近")
    pushed(project)
    stored, assets, framed = framing(project)
    described = playback.describe(stored, assets, playback.Prepared({}, {}, None), None, {}, framed)
    first, second = described["tracks"][0]["clips"]
    assert (first["motion"], first["motion_amount"], first["anchor"]) == ("push", stills.MOTION_AMOUNT, [0.5, 0.5])
    assert "motion" not in second


def test_an_export_says_a_push_over_video_does_not_come_across(media: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 4)], width=640, height=360, name="匯出推近")
    pushed(project)
    stored = runtime.repo.get_project(project)
    lines = left_behind(stored, (), shared.referenced_assets(stored))
    assert any(line.startswith("slow pushes, pulls and pans over video") and line.endswith(": a") for line in lines)


def test_only_a_moving_clip_gets_a_framing_written_down_for_its_move(media: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 4), insert("b", asset, 4, 8)],
                            width=640, height=360, name="只記會動的")
    pushed(project)
    stored = runtime.repo.get_project(project)
    framed = frame_project(stored, shared.sized(shared.referenced_assets(stored)), {})
    assert set(framed) == {"a"} and framed["a"].motion == "push"
    assert box_of(stored.base_video_track.clips[0], stored) == (640, 360)
