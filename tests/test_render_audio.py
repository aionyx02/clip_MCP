"""Tests for output loudness and for ducking music under speech.

These render real files with FFmpeg and measure the result, because a filter
graph that is wrong in a subtle way still produces a file.
"""

import subprocess
from pathlib import Path

import pytest

from app.server import import_asset
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
