"""Tests for a voice recorded apart from the picture: the footage's own sound drops under it.

The case this was made for: a narration on a clip-on microphone laid over a
street shot, where the camera heard the street ten decibels louder than the
microphone heard the voice. These render real files and measure them, as the
other audio tests do.
"""

import subprocess
from pathlib import Path

import pytest

from app.engine.builder import FFmpegRenderer
from app.models.media import MediaAnalysis, Rhythm, Transcript, TranscriptSegment
from app.server import _referenced_assets, _voices, import_asset, repo
from helpers import build_project, edit, insert, level, render, video_track

@pytest.fixture(scope="module")
def narration(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Record a faint narration: nothing for two seconds, then a quiet 1 kHz voice.

    Returns:
        The file.
    """
    path = tmp_path_factory.mktemp("narration") / "narration.wav"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
        "-i", r"aevalsrc=if(gte(t\,2)\,0.03*sin(2*PI*1000*t)\,0):d=5", str(path),
    ], check=True)
    return path

def transcribed(asset_id: str, spoken_from: float = 2.0, tempo: float | None = None) -> None:
    """Store an analysis saying the file has sentences from some point to its end.

    Args:
        asset_id: The file.
        spoken_from: Where the sentences start, in seconds; the file runs five.
        tempo: A beat the analysis found, if it found one.
    """
    repo.save_analysis(MediaAnalysis(
        asset_id=asset_id,
        duration=5.0,
        transcript=Transcript(language="zh", model="test", segments=[
            TranscriptSegment(start=spoken_from, end=5.0, text="旁白"),
        ]),
        rhythm=None if tempo is None else Rhythm(tempo=tempo, beats=[0.5, 1.0]),
    ))

def street_and_narration(audio_media: Path, narration: Path, **track) -> str:
    """Lay the narration over a loud 800 Hz street.

    Args:
        audio_media: The shared audio fixtures.
        narration: The narration file.
        **track: Extra fields for the narration's track, such as `voice`.

    Returns:
        The project ID.
    """
    street = import_asset(str(audio_media / "loud.mp4"))["id"]
    voice = import_asset(str(narration))["id"]
    transcribed(voice)
    return build_project([
        video_track(),
        insert("street", street, 0, 5),
        {"action": "add_track", "track_id": "narr", "track_type": "audio", **track},
        insert("n", voice, 0, 5, track_id="narr"),
    ], width=320, height=240)

def test_the_footage_drops_under_a_narration(audio_media: Path, narration: Path, tmp_path: Path) -> None:
    project = street_and_narration(audio_media, narration)
    render(project, tmp_path / "out.mp4", loudness_target=None, voices=_voices(repo.get_project(project)))
    before = level(tmp_path / "out.mp4", 0.5, 1.0, freq=800)
    during = level(tmp_path / "out.mp4", 3.0, 1.5, freq=800)
    assert before - during > 8
    # And the voice ends up over the street, though it was recorded far under it.
    assert level(tmp_path / "out.mp4", 3.0, 1.5, freq=1000) - during > 6

def test_a_narration_is_found_without_being_labelled(audio_media: Path, narration: Path) -> None:
    project = repo.get_project(street_and_narration(audio_media, narration))
    voices = _voices(project)
    assert list(voices) == ["narr"]
    # Recorded about thirty decibels down, so it is brought a long way up.
    assert voices["narr"] > 5

def test_saying_it_is_not_a_voice_leaves_the_footage_alone(audio_media: Path, narration: Path, tmp_path: Path) -> None:
    project = street_and_narration(audio_media, narration, voice=False)
    assert _voices(repo.get_project(project)) == {}
    render(project, tmp_path / "out.mp4", loudness_target=None, voices=_voices(repo.get_project(project)))
    assert abs(level(tmp_path / "out.mp4", 0.5, 1.0, freq=800) - level(tmp_path / "out.mp4", 3.0, 1.5, freq=800)) < 2

def test_a_narration_the_beat_tracker_heard_a_beat_in_is_still_a_voice(audio_media: Path, narration: Path) -> None:
    # Syllables are strong onsets: the first real narration measured 135.7 BPM.
    project = street_and_narration(audio_media, narration)
    transcribed(repo.get_project(project).tracks[1].clips[0].asset_id, tempo=135.7)
    assert list(_voices(repo.get_project(project))) == ["narr"]

def test_a_recording_that_is_mostly_not_sentences_is_not_a_voice(audio_media: Path, narration: Path) -> None:
    # A song with one line the recogniser caught in its outro.
    project = street_and_narration(audio_media, narration)
    transcribed(repo.get_project(project).tracks[1].clips[0].asset_id, spoken_from=4.0)
    assert _voices(repo.get_project(project)) == {}

def test_a_track_set_to_duck_under_speech_is_music(audio_media: Path, narration: Path) -> None:
    project = street_and_narration(audio_media, narration, duck_under_speech=True)
    assert _voices(repo.get_project(project)) == {}

def test_a_track_never_transcribed_is_a_voice_only_when_said(audio_media: Path) -> None:
    music = import_asset(str(audio_media / "music.mp3"))["id"]
    street = import_asset(str(audio_media / "loud.mp4"))["id"]
    project = build_project([
        video_track(),
        insert("street", street, 0, 5),
        {"action": "add_track", "track_id": "bed", "track_type": "audio"},
        insert("m", music, 0, 5, track_id="bed"),
    ], width=320, height=240)
    assert _voices(repo.get_project(project)) == {}
    edit(project, [{"action": "set_track_audio", "track_id": "bed", "voice": True}])
    assert list(_voices(repo.get_project(project))) == ["bed"]
    edit(project, [{"action": "set_track_audio", "track_id": "bed", "voice_auto": True}])
    assert repo.get_project(project).tracks[1].voice is None

def test_music_ducks_under_the_narration_too(audio_media: Path, narration: Path, tmp_path: Path) -> None:
    project = street_and_narration(audio_media, narration)
    music = import_asset(str(audio_media / "music.mp3"))["id"]
    edit(project, [
        {"action": "add_track", "track_id": "music", "track_type": "audio", "duck_under_speech": True},
        insert("m", music, 0, 5, track_id="music", volume=0.1),
    ])
    stored = repo.get_project(project)
    render(project, tmp_path / "out.mp4", loudness_target=None, voices=_voices(stored))
    # The street itself is loud enough to duck the music throughout, so the narration can
    # only add to that; what matters is that the graph with all three renders, and that the
    # music is no louder while the narration speaks.
    assert level(tmp_path / "out.mp4", 3.0, 1.5, freq=200) <= level(tmp_path / "out.mp4", 0.5, 1.0, freq=200) + 1

def test_the_sound_preview_puts_the_narration_on_the_voice_side(
    audio_media: Path, narration: Path, tmp_path: Path,
) -> None:
    stored = repo.get_project(street_and_narration(audio_media, narration))
    voice, rest = tmp_path / "voice.wav", tmp_path / "rest.wav"
    command = FFmpegRenderer().build_sound(
        stored, _referenced_assets(stored), str(tmp_path / "mix.m4a"), str(voice), str(rest),
        loudness_target=None, voices=_voices(stored),
    )
    subprocess.run(command, check=True, capture_output=True)
    # Nothing on the voice side before the narration starts; the street is on the other.
    assert level(voice, 0.5, 1.0) < -60
    assert level(rest, 0.5, 1.0) > -30

@pytest.mark.parametrize(("operations", "message"), [
    ([{"action": "add_track", "track_id": "main", "track_type": "video", "voice": True}], "only an audio track"),
    ([video_track(), {"action": "add_track", "track_id": "n", "track_type": "audio", "voice": True,
                      "duck_under_speech": True}], "cannot duck under speech"),
])
def test_a_voice_track_that_makes_no_sense_is_refused(operations: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_project(operations, width=320, height=240)
