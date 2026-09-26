"""Tests for working out where a cut's talking is, and how loud, from its analyses."""

from decimal import Decimal

import pytest

from app.engine.builder import Talking
from app.engine.delivery import check_delivery
from app.engine.levels import talking_in
from app.models.media import Asset, MediaAnalysis, SoundMeasurement, Transcript, TranscriptSegment
from app.models.timeline import Clip, Project, Track

def sound(level: float, seconds: int = 10) -> list:
    """One measurement a second, all at one level."""
    return [
        SoundMeasurement(start=second, end=second + 1, loudness=level, peak=level + 6, noise_floor=-80, flatness=0)
        for second in range(seconds)
    ]

def said(*spans: tuple) -> Transcript:
    """A transcript with a sentence over each `(start, end)`."""
    return Transcript(language="zh", model="test", segments=[
        TranscriptSegment(start=start, end=end, text="一句話") for start, end in spans
    ])

ASSETS = {
    "cam": Asset(id="cam", path="cam.mp4", duration=Decimal(10), has_video=True, has_audio=True),
    "song": Asset(id="song", path="song.mp3", duration=Decimal(10), has_video=False, has_audio=True),
}

def project(volume: float = 1.0, speed: float = 1.0) -> Project:
    """Four seconds of camera from two seconds in, over a song."""
    return Project(id="p", tracks=[
        Track(id="main", track_type="video", clips=[Clip(
            id="c", asset_id="cam", timeline_in=Decimal(0), volume=volume, speed=speed,
            source_range={"start": 2, "end": 6},
        )]),
        Track(id="music", track_type="audio", duck_under_speech=True, clips=[Clip(
            id="m", asset_id="song", timeline_in=Decimal(0), source_range={"start": 0, "end": 4},
        )]),
    ])

def test_the_talking_is_placed_on_the_timeline_and_the_music_brought_to_it() -> None:
    analyses = {"cam": MediaAnalysis(asset_id="cam", duration=10, transcript=said((3, 5), (8, 9)), sound=sound(-30))}
    talking = talking_in(project(volume=0.5, speed=2.0), ASSETS, analyses, {}, lambda clip: -12.0)
    # Three seconds into the file is half a second into a clip played twice as fast from two
    # seconds in; the sentence at eight seconds is not in the clip at all.
    assert talking.spans == ((0.5, 1.5),)
    # Heard at -30 dB turned down by half (-6 dB): the song goes to that level.
    assert talking.music_gains["music"] == pytest.approx(-36.02 + 12.0, abs=0.01)

def test_footage_nobody_transcribed_leaves_the_talking_unknown() -> None:
    analyses = {"cam": MediaAnalysis(asset_id="cam", duration=10, sound=sound(-30))}
    assert talking_in(project(), ASSETS, analyses, {}, lambda clip: -12.0) is None

def test_a_cutaway_analyzed_without_its_words_counts_as_nobody_talking() -> None:
    cut = project()
    cut.tracks[0].clips.append(Clip(
        id="b", asset_id="street", timeline_in=Decimal(4), source_range={"start": 0, "end": 3},
    ))
    assets = {**ASSETS, "street": Asset(id="street", path="s.mp4", duration=Decimal(9), has_video=True, has_audio=True)}
    analyses = {
        "cam": MediaAnalysis(asset_id="cam", duration=10, transcript=said((3, 5)), sound=sound(-30)),
        "street": MediaAnalysis(asset_id="street", duration=9, sound=sound(-10)),
    }
    talking = talking_in(cut, assets, analyses, {}, lambda clip: -12.0)
    assert talking.spans == ((1.0, 3.0),)
    # The loud street is not part of the level the music is set against.
    assert talking.music_gains["music"] == pytest.approx(-30.0 + 12.0, abs=0.01)

def test_a_song_that_cannot_be_measured_is_left_at_its_own_level() -> None:
    analyses = {"cam": MediaAnalysis(asset_id="cam", duration=10, transcript=said((3, 5)), sound=sound(-30))}
    assert talking_in(project(), ASSETS, analyses, {}, lambda clip: None).music_gains == {}

def test_music_close_under_the_talking_is_found_before_the_render() -> None:
    loud = project()
    loud.tracks[1].clips[0].volume = 1.0
    loud.tracks[1].duck_under_speech = False
    talking = Talking(spans=((0.5, 1.5),), music_gains={"music": -18.0})
    found = [finding for finding in check_delivery(loud, {}, talking=talking) if finding.check == "music"]
    assert len(found) == 1 and "0 dB under the talking" in found[0].message
    # Ducked, and at the volume a plan gives music, it is well under.
    loud.tracks[1].duck_under_speech = True
    loud.tracks[1].clips[0].volume = 0.35
    assert not [finding for finding in check_delivery(loud, {}, talking=talking) if finding.check == "music"]
    # Nothing is said about music that was never brought to the talking.
    assert not check_delivery(loud, {}, talking=Talking(spans=((0.5, 1.5),)))
