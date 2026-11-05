"""Steadying a handheld clip: asked for clip by clip, measured on the clip's own frames, never on a photo."""

import subprocess
from pathlib import Path

import numpy
import pytest

from app.engine.builder import FFmpegRenderer, can_stabilize
from app.engine.delivery import shaky_clips
from app.engine.probe import probe_file
from app.models.media import MediaAnalysis, ShotMeasurement
from app.runtime import repo
from app.tools import catalog, rendering, shared
from helpers import build_project, clips_of, edit, insert, render, video_track

needs_vidstab = pytest.mark.skipif(not can_stabilize(), reason="this FFmpeg was built without libvidstab")


@pytest.fixture(scope="module")
def shaky(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A still, detailed picture shot by a shaking hand: each frame cropped at a random offset, with a tone under it."""
    path = tmp_path_factory.mktemp("shaky") / "handheld.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "mandelbrot=size=960x540:rate=30:start_scale=0.5,trim=end_frame=1,"
                             "loop=loop=240:size=1,setpts=N/30/TB",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-vf", "crop=640:360:x='160+40*(random(1)-0.5)':y='90+40*(random(2)-0.5)'",
        "-t", "8", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    ], check=True, capture_output=True)
    return path


def motion(path: Path) -> float:
    """Measure how far the picture moves from one frame to the next, by phase correlation of its middle.

    Independent of vidstab, which made the steadied render: a shift found by another method.

    Returns:
        The mean shift between neighbouring frames, in pixels at 320 wide.
    """
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=320:180,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    frames = numpy.frombuffer(raw, numpy.uint8).reshape(-1, 180, 320).astype(numpy.float64)[:, 45:135, 80:240]
    shifts = []
    for before, after in zip(frames, frames[1:]):
        spectrum = numpy.fft.fft2(after) * numpy.conj(numpy.fft.fft2(before))
        peak = numpy.abs(numpy.fft.ifft2(spectrum / (numpy.abs(spectrum) + 1e-9)))
        y, x = numpy.unravel_index(numpy.argmax(peak), peak.shape)
        shifts.append(numpy.hypot(x - peak.shape[1] if x > peak.shape[1] // 2 else x,
                                  y - peak.shape[0] if y > peak.shape[0] // 2 else y))
    return float(numpy.mean(shifts))


def streams(path: Path) -> dict:
    """Each stream's length in seconds, and the picture's frame count."""
    found = probe_file(str(path))["streams"]
    video = next(stream for stream in found if stream["codec_type"] == "video")
    audio = next(stream for stream in found if stream["codec_type"] == "audio")
    return {"frames": int(video["nb_frames"]), "video": float(video["duration"]), "audio": float(audio["duration"])}


@needs_vidstab
@pytest.mark.parametrize("start", [2.0, 0.5], ids=["seeked", "read-from-start"])
def test_a_steadied_clip_moves_far_less_and_keeps_its_length_and_sound(shaky: Path, tmp_path: Path,
                                                                       start: float) -> None:
    # 2 s is read with -ss and 0.5 s from the file's start and cut in the chain: the
    # transforms have to be measured on the frames each read hands on, not on the file's.
    asset = catalog.import_asset(str(shaky))["id"]
    plain = build_project([video_track(), insert("a", asset, start, start + 4)], width=640, height=360)
    steady = build_project([video_track(), insert("a", asset, start, start + 4, stabilize=True)],
                           width=640, height=360)
    render(plain, tmp_path / "plain.mp4", loudness_target=None)
    render(steady, tmp_path / "steady.mp4", loudness_target=None)

    shaking, steadied = motion(tmp_path / "plain.mp4"), motion(tmp_path / "steady.mp4")
    assert shaking > 5 and steadied < shaking * 0.2
    assert streams(tmp_path / "steady.mp4") == streams(tmp_path / "plain.mp4")
    assert streams(tmp_path / "steady.mp4")["frames"] == 120


@needs_vidstab
def test_a_clip_is_measured_once_and_then_read_from_the_cache(shaky: Path, tmp_path: Path) -> None:
    asset = catalog.import_asset(str(shaky))["id"]
    project = repo.get_project(build_project([video_track(), insert("a", asset, 2, 6, stabilize=True)],
                                             width=640, height=360))
    renderer = FFmpegRenderer()
    command, passes = renderer.build_render(project, shared.referenced_assets(project), str(tmp_path / "o.mp4"),
                                            transforms_dir=str(tmp_path))
    [measuring] = passes
    assert measuring["kind"] == "transforms" and "vidstabdetect" in " ".join(measuring["command"])
    assert "vidstabtransform" in " ".join(command)
    subprocess.run(measuring["command"], check=True, capture_output=True)
    Path(measuring["part"]).replace(measuring["path"])
    _, again = renderer.build_render(project, shared.referenced_assets(project), str(tmp_path / "o.mp4"),
                                     transforms_dir=str(tmp_path))
    assert again == []


def test_a_clip_left_as_shot_is_rendered_without_any_measuring(media: Path, tmp_path: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = repo.get_project(build_project([video_track(), insert("a", asset, 0, 2)], width=640, height=360))
    command, passes = FFmpegRenderer().build_render(project, shared.referenced_assets(project),
                                                    str(tmp_path / "o.mp4"), transforms_dir=str(tmp_path))
    assert passes == [] and "vidstab" not in " ".join(command)


def test_steadying_a_photo_is_refused(tmp_path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (640, 360), (200, 40, 40)).save(tmp_path / "photo.png")
    photo = catalog.import_asset(str(tmp_path / "photo.png"))["id"]
    with pytest.raises(ValueError, match="photo"):
        build_project([video_track(), insert("a", photo, 0, 3, stabilize=True)], width=640, height=360)
    project = build_project([video_track(), insert("a", photo, 0, 3)], width=640, height=360)
    with pytest.raises(ValueError, match="photo"):
        edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "stabilize": True}])


def test_steadying_is_turned_on_and_off_by_hand(media: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2)], width=640, height=360)
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "stabilize": True}])
    assert clips_of(project)[0]["stabilize"] is True
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "stabilize": False}])
    assert clips_of(project)[0]["stabilize"] is False


def test_a_render_without_a_place_for_the_transforms_is_refused(media: Path, tmp_path: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    project = repo.get_project(build_project([video_track(), insert("a", asset, 0, 2, stabilize=True)],
                                             width=640, height=360))
    with pytest.raises(ValueError, match="steadied"):
        FFmpegRenderer().build_command(project, shared.referenced_assets(project), str(tmp_path / "o.mp4"))


def test_a_shaky_clip_left_as_shot_is_pointed_out_before_the_render(media: Path) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    repo.save_analysis(MediaAnalysis(asset_id=asset, duration=10.0, shots=[
        ShotMeasurement(start=0.0, end=4.0, exposure=0.5, contrast=0.4, shake=0.002),
        ShotMeasurement(start=4.0, end=10.0, exposure=0.5, contrast=0.4, shake=0.03),
    ]))
    project = build_project([
        video_track(), insert("steady", asset, 0, 3), insert("shaky", asset, 5, 8), insert("fixed", asset, 8, 10,
                                                                                            stabilize=True),
    ], width=640, height=360)
    found = shaky_clips(repo.get_project(project), rendering.analyses_of(repo.get_project(project)))
    assert len(found) == 1 and "clip shaky" in found[0] and "0.030" in found[0] and "stabilize" in found[0]
    checked = rendering.check_render(project)
    # Said beside the findings, never as one: a shake may be wanted.
    assert checked["shaky"] == found and all(finding["check"] != "shaky" for finding in checked["findings"])

