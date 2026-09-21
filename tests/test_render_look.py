"""Tests for per-clip colour adjustments, fades from and to black, and cross dissolves."""

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.engine.frames import extract_frame
from app.server import import_asset, repo
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


# --- cross dissolve -------------------------------------------------------------------------

def channels(path: Path, seconds: float) -> tuple:
    """Measure the average red, green and blue of one frame.

    Args:
        path: Video to sample.
        seconds: Time of the frame.

    Returns:
        `(red, green, blue)`, each from 0 to 255.
    """
    pixels = list(extract_frame(str(path), seconds).get_flattened_data())
    return tuple(sum(pixel[channel] for pixel in pixels) / len(pixels) for channel in range(3))


def seconds_long(path: Path) -> float:
    """Measure a rendered file's duration.

    Args:
        path: Video to probe.

    Returns:
        Its duration in seconds.
    """
    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return float(probed.stdout.strip())


@pytest.fixture
def red_then_green(tmp_path_factory: pytest.TempPathFactory):
    """Two solid-colour clips, cut at three seconds, the second with a second of run-up.

    Returns:
        A function taking the dissolve length and returning the project ID.
    """
    folder = tmp_path_factory.mktemp("colours")
    assets = []
    for name, colour in (("red.mp4", "red"), ("green.mp4", "green")):
        path = folder / name
        result = subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c={colour}:size=320x240:rate=30:duration=5",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
        ], capture_output=True)
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
        assets.append(import_asset(str(path))["id"])
    red, green = assets

    def build(dissolve: float) -> str:
        # The second clip takes its source from one second in, so a dissolve has picture
        # to run up from without moving the cut.
        project = build_project([video_track(), insert("a", red, 0, 3), insert("b", green, 1, 4)],
                                width=320, height=240)
        if dissolve:
            edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "b",
                            "dissolve_in": dissolve}])
        return project

    return build


def test_a_straight_cut_shows_one_clip_or_the_other(red_then_green, tmp_path: Path) -> None:
    """The baseline the dissolve is compared against: nothing is ever mixed."""
    render(red_then_green(0), tmp_path / "cut.mp4")
    before_red, before_green, _ = channels(tmp_path / "cut.mp4", 2.5)
    after_red, after_green, _ = channels(tmp_path / "cut.mp4", 3.5)
    assert before_red > 100 and before_green < 60
    assert after_green > 100 and after_red < 60


def test_a_dissolve_shows_both_clips_at_once(red_then_green, tmp_path: Path) -> None:
    """Half a second into a one-second dissolve, the frame is half of each clip."""
    straight, mixed = tmp_path / "cut.mp4", tmp_path / "mix.mp4"
    render(red_then_green(0), straight)
    render(red_then_green(1.0), mixed)
    # What each clip looks like on its own, so the mix is judged against them and not
    # against numbers that depend on what FFmpeg calls "red".
    only_red = channels(straight, 1.5)
    only_green = channels(straight, 3.5)
    halfway = channels(mixed, 2.5)
    for channel in range(3):
        assert halfway[channel] == pytest.approx((only_red[channel] + only_green[channel]) / 2, abs=4), channel


def test_a_dissolve_ends_on_the_cut_and_leaves_the_clips_where_they_are(
    red_then_green, tmp_path: Path
) -> None:
    """The mix runs up to the cut, so nothing after it moves and the length is unchanged."""
    straight, mixed = tmp_path / "cut.mp4", tmp_path / "mix.mp4"
    render(red_then_green(0), straight)
    project = red_then_green(1.0)
    render(project, mixed)
    assert seconds_long(mixed) == pytest.approx(seconds_long(straight), abs=0.05)
    assert [clip["timeline_in"] for clip in clips_of(project)] == [0, 3]
    # Before the dissolve begins the first clip is untouched, and after the cut the
    # second one is fully itself.
    assert channels(mixed, 1.5)[0] > 100 and channels(mixed, 1.5)[1] < 60
    assert channels(mixed, 3.5)[1] > 100 and channels(mixed, 3.5)[0] < 60


# --- variable speed --------------------------------------------------------------------------

@pytest.fixture
def counting(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A clip whose picture changes steadily, so playback speed is visible in it.

    Returns:
        The asset ID of an eight-second ramp from black to white with a tone on it.
    """
    path = tmp_path_factory.mktemp("ramp") / "ramp.mp4"
    result = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:size=320x240:rate=30:duration=8",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-vf", "geq=lum='T/8*255':cb=128:cr=128",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return import_asset(str(path))["id"]


def sped(asset: str, speed: float) -> str:
    """Put four seconds of the ramp on a timeline at a given speed.

    Args:
        asset: The ramp asset.
        speed: Playback speed.

    Returns:
        The project ID.
    """
    project = build_project([video_track(), insert("r", asset, 0, 4)], width=320, height=240)
    if speed != 1.0:
        edit(project, [{"action": "set_clip_speed", "track_id": "main", "clip_id": "r", "speed": speed}])
    return project


@pytest.mark.parametrize("speed, expected", [(1.0, 4.0), (2.0, 2.0), (0.5, 8.0)])
def test_speed_changes_how_long_a_clip_runs(counting: str, tmp_path: Path, speed: float, expected: float) -> None:
    """Four seconds of footage at double speed is two seconds of video."""
    project = sped(counting, speed)
    assert repo.get_project(project).tracks[0].clips[0].timeline_out == expected
    out = tmp_path / f"{speed}.mp4"
    render(project, out)
    assert seconds_long(out) == pytest.approx(expected, abs=0.1)


def test_the_picture_really_is_playing_faster(counting: str, tmp_path: Path) -> None:
    """The same moment of footage arrives earlier, rather than the clip being cut short."""
    normal, fast = tmp_path / "normal.mp4", tmp_path / "fast.mp4"
    render(sped(counting, 1.0), normal)
    render(sped(counting, 2.0), fast)
    # The ramp brightens steadily, so how bright a frame is says how far into the
    # footage it came from. Two seconds into the fast one is four seconds of footage.
    assert luma(fast, 1.9) == pytest.approx(luma(normal, 3.8), abs=12)
    assert luma(fast, 0.95) == pytest.approx(luma(normal, 1.9), abs=12)


def test_a_slower_clip_reaches_the_same_footage_later(counting: str, tmp_path: Path) -> None:
    """Half speed means twice as long to get to the same frame."""
    normal, slow = tmp_path / "normal.mp4", tmp_path / "slow.mp4"
    render(sped(counting, 1.0), normal)
    render(sped(counting, 0.5), slow)
    assert luma(slow, 3.8) == pytest.approx(luma(normal, 1.9), abs=12)


def test_what_follows_moves_up_when_a_clip_gets_shorter(counting: str) -> None:
    """A speed change is a length change, so the sequence stays tight by default."""
    project = build_project([video_track(), insert("a", counting, 0, 4), insert("b", counting, 0, 2)],
                            width=320, height=240)
    assert [clip["timeline_in"] for clip in clips_of(project)] == [0, 4]
    edit(project, [{"action": "set_clip_speed", "track_id": "main", "clip_id": "a", "speed": 2.0}])
    assert [clip["timeline_in"] for clip in clips_of(project)] == [0, 2]


def test_a_speed_change_can_leave_the_rest_where_it_is(counting: str) -> None:
    """Turning the ripple off leaves a gap, which is sometimes what is wanted."""
    project = build_project([video_track(), insert("a", counting, 0, 4), insert("b", counting, 0, 2)],
                            width=320, height=240)
    edit(project, [{"action": "set_clip_speed", "track_id": "main", "clip_id": "a",
                    "speed": 2.0, "ripple": False}])
    assert [clip["timeline_in"] for clip in clips_of(project)] == [0, 4]


@pytest.mark.parametrize("speed", [0.1, 20.0])
def test_a_speed_nobody_could_listen_to_is_refused(counting: str, speed: float) -> None:
    """The sound is stretched to match, and past these it stops being sound."""
    project = sped(counting, 1.0)
    with pytest.raises(ValidationError):
        edit(project, [{"action": "set_clip_speed", "track_id": "main", "clip_id": "r", "speed": speed}])
