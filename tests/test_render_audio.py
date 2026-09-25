"""Tests for output loudness and for ducking music under speech.

These render real files with FFmpeg and measure the result, because a filter
graph that is wrong in a subtle way still produces a file.
"""

import subprocess
from pathlib import Path

import pytest

from app.server import import_asset
from app.server import repo
from helpers import build_project, edit, insert, level, loudness, render, video_track

@pytest.fixture
def quiet_project(audio_media: Path) -> str:
    """Build a project holding only the faint clip.

    Returns:
        The project ID.
    """
    asset = import_asset(str(audio_media / "quiet.mp4"))["id"]
    return build_project([video_track(), insert("q", asset, 0, 4)], width=320, height=240)

@pytest.fixture
def speech_over_music(audio_media: Path):
    """Build a project with speech starting at three seconds over a music bed.

    Returns:
        A function taking `duck` and returning the project ID.
    """
    talk = import_asset(str(audio_media / "talk.mp4"))["id"]
    music = import_asset(str(audio_media / "music.mp3"))["id"]

    def build(duck: bool) -> str:
        return build_project([
            video_track(),
            insert("t", talk, 0, 6),
            {"action": "add_track", "track_id": "music", "track_type": "audio", "duck_under_speech": duck},
            insert("m", music, 0, 6, track_id="music", volume=0.5),
        ], width=320, height=240)

    return build

def test_normalizes_the_finished_mix_to_the_target(quiet_project: str, tmp_path: Path) -> None:
    render(quiet_project, tmp_path / "out.mp4")
    assert loudness(tmp_path / "out.mp4") == pytest.approx(-14.0, abs=1.0)

def test_honours_a_different_target(quiet_project: str, tmp_path: Path) -> None:
    render(quiet_project, tmp_path / "out.mp4", loudness_target=-23.0)
    assert loudness(tmp_path / "out.mp4") == pytest.approx(-23.0, abs=1.0)

def test_leaves_the_mix_alone_when_normalization_is_off(quiet_project: str, tmp_path: Path) -> None:
    render(quiet_project, tmp_path / "out.mp4", loudness_target=None)
    assert loudness(tmp_path / "out.mp4") < -35

def test_evens_out_a_jump_between_clips_recorded_at_different_levels(audio_media: Path, tmp_path: Path) -> None:
    quiet = import_asset(str(audio_media / "quiet.mp4"))["id"]
    loud = import_asset(str(audio_media / "loud.mp4"))["id"]
    project = build_project(
        [video_track(), insert("q", quiet, 0, 4), insert("l", loud, 0, 5)], width=320, height=240
    )
    render(project, tmp_path / "raw.mp4", loudness_target=None)
    render(project, tmp_path / "norm.mp4")
    raw_jump = level(tmp_path / "raw.mp4", 4.5, 4.0) - level(tmp_path / "raw.mp4", 0.5, 3.0)
    normalized_jump = level(tmp_path / "norm.mp4", 4.5, 4.0) - level(tmp_path / "norm.mp4", 0.5, 3.0)
    assert raw_jump > 15
    assert normalized_jump < 5

def test_ducking_lowers_the_music_while_someone_speaks(speech_over_music, tmp_path: Path) -> None:
    render(speech_over_music(True), tmp_path / "ducked.mp4", loudness_target=None)
    quiet_part = level(tmp_path / "ducked.mp4", 0.5, 2.0, freq=200)
    speaking_part = level(tmp_path / "ducked.mp4", 3.5, 2.0, freq=200)
    assert quiet_part - speaking_part > 5

def test_music_is_untouched_without_the_flag(speech_over_music, tmp_path: Path) -> None:
    render(speech_over_music(False), tmp_path / "plain.mp4", loudness_target=None)
    quiet_part = level(tmp_path / "plain.mp4", 0.5, 2.0, freq=200)
    speaking_part = level(tmp_path / "plain.mp4", 3.5, 2.0, freq=200)
    assert abs(quiet_part - speaking_part) < 2

def test_ducking_can_be_turned_on_after_the_track_exists(audio_media: Path, tmp_path: Path) -> None:
    talk = import_asset(str(audio_media / "talk.mp4"))["id"]
    music = import_asset(str(audio_media / "music.mp3"))["id"]
    project = build_project([
        video_track(),
        insert("t", talk, 0, 6),
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        insert("m", music, 0, 6, track_id="music", volume=0.5),
    ], width=320, height=240)
    edit(project, [{"action": "set_track_audio", "track_id": "music", "duck_under_speech": True}])
    render(project, tmp_path / "out.mp4", loudness_target=None)
    assert level(tmp_path / "out.mp4", 0.5, 2.0, freq=200) - level(tmp_path / "out.mp4", 3.5, 2.0, freq=200) > 5

def test_a_video_track_cannot_duck_under_itself(audio_media: Path) -> None:
    asset = import_asset(str(audio_media / "quiet.mp4"))["id"]
    with pytest.raises(ValueError, match="only audio tracks can duck"):
        build_project([
            {"action": "add_track", "track_id": "main", "track_type": "video", "duck_under_speech": True},
            insert("q", asset, 0, 4),
        ], width=320, height=240)

def test_the_output_stays_as_long_as_the_video(quiet_project: str, tmp_path: Path) -> None:
    render(quiet_project, tmp_path / "out.mp4")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(tmp_path / "out.mp4")],
        capture_output=True,
    )
    assert float(probe.stdout.decode()) == pytest.approx(4.0, abs=0.1)

# Normalizing a mix that has no sound in it at all would ask for infinite gain. What
# comes out has to be silence, and the render has to finish: the failure this guards
# against used to take the whole job down at the first audio frame.
@pytest.mark.parametrize("source", ["silent.mp4", "muted.mp4"])
def test_a_project_with_no_sound_renders_silence(media: Path, source: str, tmp_path: Path) -> None:
    asset = import_asset(str(media / source))["id"]
    project = build_project([video_track(), insert("s", asset, 0, 3)], width=320, height=240)
    render(project, tmp_path / "out.mp4")
    # -91 dB is what the AAC encoder writes for digital silence; anything audible is a failure.
    assert level(tmp_path / "out.mp4", 0, 3) < -80

def test_the_silence_guard_reaches_nothing_but_nan(media: Path, audio_media: Path, tmp_path: Path) -> None:
    muted = import_asset(str(media / "muted.mp4"))["id"]
    loud = import_asset(str(audio_media / "loud.mp4"))["id"]
    project = build_project([video_track(), insert("m", muted, 0, 3), insert("l", loud, 0, 4)],
                            width=320, height=240)
    render(project, tmp_path / "out.mp4")
    # Silence where the muted clip is, and the clip that has sound comes through loudly:
    # the guard replaces NaN and leaves every sample that is a number alone.
    assert level(tmp_path / "out.mp4", 0, 2.5) < -80
    assert level(tmp_path / "out.mp4", 3.5, 3) > -30


@pytest.fixture
def two_tones(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """Two clips carrying tones far enough apart to be told apart in the mix.

    Returns:
        The asset IDs of a 440 Hz clip and a 1500 Hz clip, six seconds each.
    """
    folder = tmp_path_factory.mktemp("tones")
    assets = []
    for name, frequency in (("low.mp4", 440), ("high.mp4", 1500)):
        path = folder / name
        result = subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=6",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration=6",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ], capture_output=True)
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
        assets.append(import_asset(str(path))["id"])
    return tuple(assets)


def cut_at_three(low: str, high: str, **audio) -> str:
    """Put the low clip on screen for three seconds, then the high one.

    Args:
        low: Asset for the first clip.
        high: Asset for the second.
        **audio: Fields for a `set_clip_audio` on whichever clip is named.

    Returns:
        The project ID.
    """
    project = build_project([video_track(), insert("a", low, 0, 3), insert("b", high, 2, 5)],
                            width=320, height=240)
    if audio:
        edit(project, [{"action": "set_clip_audio", "track_id": "main", **audio}])
    return project


def test_a_j_cut_is_heard_before_it_is_seen(two_tones: tuple, tmp_path: Path) -> None:
    """The second clip's sound comes in under the tail of the first clip's picture."""
    low, high = two_tones
    straight = cut_at_three(low, high)
    render(straight, tmp_path / "straight.mp4", loudness_target=None)
    # Nothing of the second clip is audible while the first is still on screen.
    assert level(tmp_path / "straight.mp4", 1.5, 1.4, freq=1500) < -50

    jcut = cut_at_three(low, high, clip_id="b", audio_lead=1.5)
    render(jcut, tmp_path / "jcut.mp4", loudness_target=None)
    assert level(tmp_path / "jcut.mp4", 1.5, 1.4, freq=1500) > -30
    # And the picture has not moved: the cut is still at three seconds.
    assert repo.get_project(jcut).tracks[0].clips[1].timeline_in == 3


def test_an_l_cut_is_still_heard_after_it_has_gone(two_tones: tuple, tmp_path: Path) -> None:
    """The first clip's sound finishes over the shot that replaced it."""
    low, high = two_tones
    straight = cut_at_three(low, high)
    render(straight, tmp_path / "straight.mp4", loudness_target=None)
    assert level(tmp_path / "straight.mp4", 3.6, 1.4, freq=440) < -50

    lcut = cut_at_three(low, high, clip_id="a", audio_lag=1.5)
    render(lcut, tmp_path / "lcut.mp4", loudness_target=None)
    assert level(tmp_path / "lcut.mp4", 3.6, 1.4, freq=440) > -30


def test_the_two_are_heard_together_where_they_overlap(two_tones: tuple, tmp_path: Path) -> None:
    """An overlap is both of them at once, not one replacing the other."""
    low, high = two_tones
    jcut = cut_at_three(low, high, clip_id="b", audio_lead=1.5)
    render(jcut, tmp_path / "out.mp4", loudness_target=None)
    assert level(tmp_path / "out.mp4", 1.6, 1.2, freq=440) > -30
    assert level(tmp_path / "out.mp4", 1.6, 1.2, freq=1500) > -30


# --- repairing a recorded voice ---------------------------------------------------------

def band_level(path: Path, low: int, high: int) -> float:
    """Measure how much energy a file carries in one band of the spectrum.

    Args:
        path: The rendered file.
        low: Bottom of the band in hertz.
        high: Top of the band.

    Returns:
        The mean volume in that band, in decibels.
    """
    probed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"highpass=f={low},lowpass=f={high},volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    line = next(part for part in probed.splitlines() if "mean_volume" in part)
    return float(line.split("mean_volume:")[1].split("dB")[0])

@pytest.fixture
def rumbling(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A tone at speech pitch with a heavy low roar under it.

    Returns:
        The asset ID.
    """
    path = tmp_path_factory.mktemp("rumble") / "rumble.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:size=320x240:rate=30:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=40:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=500:duration=4",
        "-filter_complex", "[1:a][2:a]amix=inputs=2:normalize=0[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(path),
    ], check=True)
    return import_asset(str(path))["id"]

def test_taking_the_rumble_out_leaves_the_voice_where_it_was(rumbling: str, tmp_path: Path) -> None:
    """The point of a high-pass: the roar goes and the part that carries words does not."""
    plain = build_project([video_track(), insert("a", rumbling, 0, 4)], width=320, height=240)
    render(plain, tmp_path / "plain.mp4", loudness_target=None)

    cleaned = build_project([video_track(), insert("a", rumbling, 0, 4)], width=320, height=240)
    edit(cleaned, [{"action": "set_clip_audio", "track_id": "main", "clip_id": "a",
                    "cleanup": {"rumble": True}}])
    render(cleaned, tmp_path / "cleaned.mp4", loudness_target=None)

    # The 40Hz roar is far quieter; the 500Hz tone standing in for a voice is not.
    assert band_level(tmp_path / "cleaned.mp4", 20, 80) < band_level(tmp_path / "plain.mp4", 20, 80) - 10
    assert band_level(tmp_path / "cleaned.mp4", 400, 600) == pytest.approx(
        band_level(tmp_path / "plain.mp4", 400, 600), abs=2
    )

def test_nothing_is_repaired_unless_it_was_asked_for(rumbling: str) -> None:
    """Every repair throws part of the recording away, so none of them is a default."""
    project = build_project([video_track(), insert("a", rumbling, 0, 4)], width=320, height=240)
    assert repo.get_project(project).tracks[0].clips[0].cleanup.is_nothing

def test_asking_for_one_repair_does_not_put_another_back(rumbling: str) -> None:
    project = build_project([video_track(), insert("a", rumbling, 0, 4)], width=320, height=240)
    edit(project, [{"action": "set_clip_audio", "track_id": "main", "clip_id": "a",
                    "cleanup": {"rumble": True}}])
    edit(project, [{"action": "set_clip_audio", "track_id": "main", "clip_id": "a",
                    "cleanup": {"hiss": True}}])
    cleanup = repo.get_project(project).tracks[0].clips[0].cleanup
    assert cleanup.rumble and cleanup.hiss and not cleanup.sibilance
