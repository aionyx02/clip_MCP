"""Tests for working out where a cut's talking is, and how loud, from its analyses."""

from decimal import Decimal

import pytest

from app.engine.builder import Talking
from app.engine.delivery import check_delivery
from app.engine.levels import AMBIENCE_UNDER_TALK_DB, LEVEL_RANGE_DB, TALK_DB, talking_in
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
    # The talking is brought from -30 dB to TALK_DB, then turned down by half (-6 dB); the
    # song goes to that level.
    assert talking.clip_gains == {"c": pytest.approx(TALK_DB + 30.0)}
    assert talking.music_gains["music"] == pytest.approx(TALK_DB - 6.02 + 12.0, abs=0.01)

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
    # The loud street is not part of the level the music is set against, and it is turned
    # down under the talking — as far as it may be turned.
    assert talking.music_gains["music"] == pytest.approx(TALK_DB + 12.0, abs=0.01)
    assert talking.clip_gains["b"] == -LEVEL_RANGE_DB

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

def sequence(*clips: Clip) -> Project:
    """A project whose sequence is these clips."""
    return Project(id="p", tracks=[Track(id="main", track_type="video", clips=list(clips))])

def cut_from(asset_id: str, clip_id: str, at: float, low: float, high: float, **fields) -> Clip:
    """A clip of `asset_id` from `low` to `high` seconds, placed at `at`."""
    return Clip(id=clip_id, asset_id=asset_id, timeline_in=Decimal(str(at)),
                source_range={"start": low, "end": high}, **fields)

def test_a_voice_far_from_the_camera_and_one_close_to_it_come_out_alike() -> None:
    assets = {
        name: Asset(id=name, path=f"{name}.mp4", duration=Decimal(10), has_video=True, has_audio=True)
        for name in ("far", "near")
    }
    analyses = {
        "far": MediaAnalysis(asset_id="far", duration=10, transcript=said((0, 10)), sound=sound(-32)),
        "near": MediaAnalysis(asset_id="near", duration=10, transcript=said((0, 10)), sound=sound(-14)),
    }
    project = sequence(cut_from("far", "a", 0, 0, 5), cut_from("near", "b", 5, 0, 5))
    gains = talking_in(project, assets, analyses, {}, lambda clip: None).clip_gains
    assert -32 + gains["a"] == pytest.approx(TALK_DB) and -14 + gains["b"] == pytest.approx(TALK_DB)

def test_a_line_too_short_to_measure_takes_the_level_of_its_file() -> None:
    made = MediaAnalysis(asset_id="cam", duration=10, transcript=said((1, 2), (5, 9)), sound=[
        *sound(-20, 5), *[
            SoundMeasurement(start=second, end=second + 1, loudness=-30, peak=-24, noise_floor=-80, flatness=0)
            for second in range(5, 10)
        ],
    ])
    # One second of talking: not enough to measure, so the file's talking as a whole is used —
    # one second at -20 and four at -30, which add up as sound to -25.5.
    project = sequence(cut_from("cam", "a", 0, 0.5, 2.5))
    gain = talking_in(project, ASSETS, {"cam": made}, {}, lambda clip: None).clip_gains["a"]
    assert gain == pytest.approx(TALK_DB + 25.53, abs=0.01)

def test_a_quiet_place_is_not_brought_up_and_a_kept_clip_moves_with_the_rest() -> None:
    assets = {
        name: Asset(id=name, path=f"{name}.mp4", duration=Decimal(10), has_video=True, has_audio=True)
        for name in ("cam", "room")
    }
    analyses = {
        "cam": MediaAnalysis(asset_id="cam", duration=10, transcript=said((0, 10)), sound=sound(-26)),
        "room": MediaAnalysis(asset_id="room", duration=10, transcript=said(), sound=sound(-50)),
    }
    project = sequence(
        cut_from("cam", "a", 0, 0, 4), cut_from("room", "b", 4, 0, 3),
        cut_from("cam", "c", 7, 5, 9, keep_level=True),
    )
    gains = talking_in(project, assets, analyses, {}, lambda clip: None).clip_gains
    assert gains["b"] == 0.0
    assert -50 < TALK_DB - AMBIENCE_UNDER_TALK_DB
    # Kept at its recorded level against the others: moved as the typical clip was.
    assert gains["c"] == gains["a"] or gains["c"] == gains["b"]
