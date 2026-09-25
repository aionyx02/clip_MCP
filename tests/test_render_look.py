"""Tests for per-clip colour adjustments, fades from and to black, transitions and speed."""

import subprocess
from pathlib import Path

import numpy
import pytest
from pydantic import ValidationError

from app.engine.diarize import SAMPLE_RATE, read_samples
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
        A function taking a transition, or None for a straight cut, and
        returning the project ID.
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

    def build(transition: dict = None) -> str:
        # The second clip takes its source from one second in, so a transition has
        # picture to run up from without moving the cut.
        project = build_project([video_track(), insert("a", red, 0, 3), insert("b", green, 1, 4)],
                                width=320, height=240)
        if transition:
            edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "b",
                            "transition_in": transition}])
        return project

    return build


def test_a_straight_cut_shows_one_clip_or_the_other(red_then_green, tmp_path: Path) -> None:
    """The baseline every transition is compared against: nothing is ever mixed."""
    render(red_then_green(), tmp_path / "cut.mp4")
    before_red, before_green, _ = channels(tmp_path / "cut.mp4", 2.5)
    after_red, after_green, _ = channels(tmp_path / "cut.mp4", 3.5)
    assert before_red > 100 and before_green < 60
    assert after_green > 100 and after_red < 60


def test_a_dissolve_shows_both_clips_at_once(red_then_green, tmp_path: Path) -> None:
    """Half a second into a one-second dissolve, the frame is half of each clip."""
    straight, mixed = tmp_path / "cut.mp4", tmp_path / "mix.mp4"
    render(red_then_green(), straight)
    render(red_then_green({"kind": "dissolve", "seconds": 1.0}), mixed)
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
    render(red_then_green(), straight)
    project = red_then_green({"kind": "dissolve", "seconds": 1.0})
    render(project, mixed)
    assert seconds_long(mixed) == pytest.approx(seconds_long(straight), abs=0.05)
    assert [clip["timeline_in"] for clip in clips_of(project)] == [0, 3]
    # Before the dissolve begins the first clip is untouched, and after the cut the
    # second one is fully itself.
    assert channels(mixed, 1.5)[0] > 100 and channels(mixed, 1.5)[1] < 60
    assert channels(mixed, 3.5)[1] > 100 and channels(mixed, 3.5)[0] < 60


# --- wipe and dip ---------------------------------------------------------------------------

def halves(path: Path, seconds: float) -> tuple:
    """Measure the average colour of the left and right halves of one frame.

    A wipe is the one transition whose whole point is that the two clips are
    somewhere different in the frame rather than mixed everywhere, so reading
    the frame as one average would say nothing about it.

    Args:
        path: Video to sample.
        seconds: Time of the frame.

    Returns:
        `(left, right)`, each `(red, green, blue)` from 0 to 255.
    """
    frame = extract_frame(str(path), seconds)
    pixels = list(frame.get_flattened_data())
    rows = [pixels[line * frame.width:(line + 1) * frame.width] for line in range(frame.height)]

    def mean(part: list) -> tuple:
        """Average one set of pixels per channel."""
        return tuple(sum(pixel[channel] for pixel in part) / len(part) for channel in range(3))

    return (
        mean([pixel for row in rows for pixel in row[: frame.width // 3]]),
        mean([pixel for row in rows for pixel in row[frame.width // 3 * 2:]]),
    )


def test_a_wipe_puts_the_two_clips_side_by_side(red_then_green, tmp_path: Path) -> None:
    """Halfway through, the frame is one clip on one side and the other on the other."""
    out = tmp_path / "wipe.mp4"
    render(red_then_green({"kind": "wipe", "seconds": 1.0, "direction": "left"}), out)
    left, right = halves(out, 2.5)
    # Nowhere in the frame is it a mix of the two: each side is one clip or the other,
    # which is what makes it a wipe rather than a dissolve.
    assert (left[0] > 100) != (right[0] > 100), (left, right)
    assert (left[1] > 100) != (right[1] > 100), (left, right)


def test_a_wipe_goes_the_way_it_is_told(red_then_green, tmp_path: Path) -> None:
    """The two directions are mirror images, so the direction is not being ignored."""
    leftwards, rightwards = tmp_path / "left.mp4", tmp_path / "right.mp4"
    render(red_then_green({"kind": "wipe", "seconds": 1.0, "direction": "left"}), leftwards)
    render(red_then_green({"kind": "wipe", "seconds": 1.0, "direction": "right"}), rightwards)
    near_left, near_right = halves(leftwards, 2.5)
    far_left, far_right = halves(rightwards, 2.5)
    assert near_left == pytest.approx(far_right, abs=8), (near_left, far_right)
    assert near_right == pytest.approx(far_left, abs=8), (near_right, far_left)


def test_a_dip_passes_through_its_colour(red_then_green, tmp_path: Path) -> None:
    """The picture goes to the colour and comes back out of it on the other side."""
    out = tmp_path / "dip.mp4"
    render(red_then_green({"kind": "dip", "seconds": 2.0, "through": "white"}), out)
    # The dip runs from 1s to the cut at 3s, so the colour is at its fullest halfway.
    assert min(channels(out, 2.0)) > 200, channels(out, 2.0)
    # And the clips either side are still themselves.
    assert channels(out, 0.5)[0] > 100 and channels(out, 0.5)[1] < 60
    assert channels(out, 3.5)[1] > 100 and channels(out, 3.5)[0] < 60


def test_a_dip_takes_a_colour_of_its_own(red_then_green, tmp_path: Path) -> None:
    """Black and white are the two worth naming; anything else is given as hex."""
    out = tmp_path / "blue.mp4"
    render(red_then_green({"kind": "dip", "seconds": 2.0, "through": "#0000ff"}), out)
    red, green, blue = channels(out, 2.0)
    assert blue > 200 and red < 60 and green < 60, (red, green, blue)


def test_a_dip_ends_on_the_cut_like_every_other_transition(red_then_green, tmp_path: Path) -> None:
    """Two mixes rather than one, but the sequence still keeps its length."""
    straight, dipped = tmp_path / "cut.mp4", tmp_path / "dip.mp4"
    render(red_then_green(), straight)
    project = red_then_green({"kind": "dip", "seconds": 2.0, "through": "black"})
    render(project, dipped)
    assert seconds_long(dipped) == pytest.approx(seconds_long(straight), abs=0.05)
    assert [clip["timeline_in"] for clip in clips_of(project)] == [0, 3]


def test_a_dip_with_no_room_for_both_halves_is_refused(red_then_green) -> None:
    """A dip needs a frame to go into the colour and a frame to come out of it."""
    project = red_then_green()
    with pytest.raises(ValueError, match="two frames"):
        edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "b",
                        "transition_in": {"kind": "dip", "seconds": 0.03, "through": "black"}}])


def test_a_transition_without_a_kind_is_refused(red_then_green) -> None:
    """Three kinds now, so which one is not something to be inferred from the length."""
    project = red_then_green()
    with pytest.raises(ValidationError):
        edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "b",
                        "transition_in": {"seconds": 1.0}}])


def test_a_transition_can_be_taken_off_again(red_then_green, tmp_path: Path) -> None:
    """Back to a straight cut, which is what null means and what clearing it gives."""
    project = red_then_green({"kind": "dissolve", "seconds": 1.0})
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "b",
                    "clear_transition": True}])
    assert repo.get_project(project).tracks[0].clips[1].transition_in is None
    render(project, tmp_path / "cut.mp4")
    assert channels(tmp_path / "cut.mp4", 2.5)[0] > 100


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


def sped(asset: str, speed: float, **fields) -> str:
    """Put four seconds of the ramp on a timeline at a given speed.

    Args:
        asset: The ramp asset.
        speed: Playback speed.
        **fields: Anything else the speed operation takes, such as `preserve_pitch`.

    Returns:
        The project ID.
    """
    project = build_project([video_track(), insert("r", asset, 0, 4)], width=320, height=240)
    if speed != 1.0 or fields:
        edit(project, [{"action": "set_clip_speed", "track_id": "main", "clip_id": "r",
                        "speed": speed, **fields}])
    return project


def tone_hz(path: Path) -> float:
    """Measure the strongest frequency in a file's sound.

    Args:
        path: Video to listen to.

    Returns:
        The frequency in hertz. The ramp carries one pure tone, so the
        strongest bin is that tone and nothing else.
    """
    samples = numpy.array(read_samples(str(path)), dtype=numpy.float64)
    # The middle half only: a render is loudness-normalized and can be faded at the
    # edges, and neither has anything to say about pitch.
    middle = samples[len(samples) // 4: len(samples) // 4 * 3]
    spectrum = numpy.abs(numpy.fft.rfft(middle * numpy.hanning(len(middle))))
    return float(numpy.fft.rfftfreq(len(middle), 1 / SAMPLE_RATE)[spectrum.argmax()])


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


def test_a_sped_up_clip_keeps_its_sound_all_the_way_through(counting: str, tmp_path: Path) -> None:
    """A speed change used to take most of the sound with it.

    `atempo` hands on a stream whose timing the `apad` and `atrim` that settle
    the clip's length then read wrong: they cut it to a quarter and padded the
    rest with silence, so any clip with a speed on it came out mostly silent.
    Measured in quarters rather than as one peak, because a peak at the front
    of a file that fades to nothing looks exactly like a file that is fine.
    """
    out = tmp_path / "fast.mp4"
    render(sped(counting, 2.0), out)
    samples = numpy.array(read_samples(str(out)), dtype=numpy.float64)
    quarters = [round(float(numpy.abs(part).max()), 3) for part in numpy.array_split(samples, 4)]
    assert min(quarters) > 0.05, quarters


def test_a_sped_up_voice_keeps_its_own_pitch_by_default(counting: str, tmp_path: Path) -> None:
    """The tone is 440Hz however fast it is played, which is what a person needs."""
    normal, fast = tmp_path / "normal.mp4", tmp_path / "fast.mp4"
    render(sped(counting, 1.0), normal)
    render(sped(counting, 2.0), fast)
    assert tone_hz(normal) == pytest.approx(440, abs=15)
    assert tone_hz(fast) == pytest.approx(440, abs=15)


def test_the_pitch_can_be_let_rise_with_the_speed(counting: str, tmp_path: Path) -> None:
    """What tape did: twice as fast is an octave up. A mistake on speech, the point of a gag."""
    out = tmp_path / "chipmunk.mp4"
    render(sped(counting, 2.0, preserve_pitch=False), out)
    assert tone_hz(out) == pytest.approx(880, abs=30)


def test_a_slowed_clip_can_be_let_fall_too(counting: str, tmp_path: Path) -> None:
    """The same in the other direction, so it is the speed being followed and not a trick."""
    out = tmp_path / "slow.mp4"
    render(sped(counting, 0.5, preserve_pitch=False), out)
    assert tone_hz(out) == pytest.approx(220, abs=15)


def test_the_pitch_setting_survives_a_speed_change_that_does_not_mention_it(counting: str) -> None:
    """Omitting it leaves the clip as it was, rather than quietly putting it back."""
    project = sped(counting, 2.0, preserve_pitch=False)
    edit(project, [{"action": "set_clip_speed", "track_id": "main", "clip_id": "r", "speed": 1.5}])
    clip = repo.get_project(project).tracks[0].clips[0]
    assert clip.speed == 1.5 and clip.preserve_pitch is False


@pytest.mark.parametrize("speed", [0.1, 20.0])
def test_a_speed_nobody_could_listen_to_is_refused(counting: str, speed: float) -> None:
    """The sound is stretched to match, and past these it stops being sound."""
    project = sped(counting, 1.0)
    with pytest.raises(ValidationError):
        edit(project, [{"action": "set_clip_speed", "track_id": "main", "clip_id": "r", "speed": speed}])


def test_a_dissolve_after_a_run_of_straight_cuts(tmp_path: Path) -> None:
    """A run of one clip and a run of several do not arrive on the same timebase.

    They meet at the crossfade, which refuses them unless both are settled
    first. Two clips alone never show it, because neither side is
    concatenated; this is the shape that does.
    """
    folder = tmp_path / "three"
    folder.mkdir()
    assets = []
    for name, colour in (("a.mp4", "red"), ("b.mp4", "blue"), ("c.mp4", "green")):
        path = folder / name
        result = subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c={colour}:size=320x240:rate=30:duration=5",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
        ], capture_output=True)
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
        assets.append(import_asset(str(path))["id"])
    first, second, third = assets

    # Two straight cuts, then a dissolve: the first run holds two segments and is
    # concatenated, the second holds one and is not.
    project = build_project([
        video_track(), insert("a", first, 0, 2), insert("b", second, 0, 2), insert("c", third, 1, 3),
    ], width=320, height=240)
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "c",
                    "transition_in": {"kind": "dissolve", "seconds": 0.6}}])
    render(project, tmp_path / "three.mp4")
    assert seconds_long(tmp_path / "three.mp4") == pytest.approx(6.0, abs=0.1)
    # Mid-dissolve the frame holds both the blue it is leaving and the green it is joining.
    blue, green = channels(tmp_path / "three.mp4", 3.7)[2], channels(tmp_path / "three.mp4", 3.7)[1]
    assert blue > 20 and green > 20, (blue, green)
