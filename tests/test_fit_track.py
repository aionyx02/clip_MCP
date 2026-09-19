"""Tests for making a music track cover the video exactly."""

from pathlib import Path

import pytest

from app.server import import_asset
from helpers import build_project, clips_of, edit, insert, layout, video_track

@pytest.fixture
def sources(media: Path) -> dict:
    """Import a video and a song.

    Returns:
        The asset IDs keyed by role.
    """
    return {
        "video": import_asset(str(media / "wide.mp4"))["id"],
        "song": import_asset(str(media / "song.mp3"))["id"],
        "jingle": import_asset(str(media / "jingle.mp3"))["id"],
    }

def music_project(sources: dict, video_seconds: float, song_seconds: float, song: str = "song", **clip_fields) -> str:
    """Build a project with one video clip and one music clip.

    Args:
        sources: Asset IDs from the `sources` fixture.
        video_seconds: Length of the video clip.
        song_seconds: Length of the music clip.
        **clip_fields: Extra fields for the music clip.

    Returns:
        The project ID.
    """
    return build_project([
        video_track(),
        insert("v", sources["video"], 0, video_seconds),
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        insert("song", sources[song], 0, song_seconds, track_id="music", **clip_fields),
    ])

def fit(project_id: str, **options) -> None:
    """Apply a `fit_track` operation to the music track.

    Args:
        project_id: Project to edit.
        **options: Extra fields for the operation.
    """
    edit(project_id, [{"action": "fit_track", "track_id": "music", **options}])

def test_music_longer_than_the_video_is_trimmed_to_the_end(sources: dict) -> None:
    project = music_project(sources, video_seconds=7, song_seconds=20)
    fit(project)
    assert layout(project, "music") == [("song", 0.0, 7.0, 0.0, 7.0)]

def test_a_song_that_was_trimmed_before_carries_on_rather_than_restarting(sources: dict) -> None:
    # The song is 20 s long; after a fit to 5 s it must play its next 4.5 s, not jump back to the top.
    project = music_project(sources, video_seconds=5, song_seconds=20)
    fit(project)
    edit(project, [insert("v2", sources["video"], 0, 4.5)])
    fit(project)
    assert layout(project, "music") == [("song", 0.0, 9.5, 0.0, 9.5)]

def test_music_shorter_than_the_video_is_repeated_to_cover_it(sources: dict) -> None:
    project = music_project(sources, video_seconds=9.5, song_seconds=4, song="jingle")
    fit(project)
    music = layout(project, "music")
    assert [entry[0] for entry in music] == ["song", "song-2", "song-3"]
    assert music[-1][2] == 9.5
    # The repeats start the song again from the top rather than continuing where it was cut.
    assert music[1][3] == 0.0 and music[2][3] == 0.0

def test_the_last_repeat_is_trimmed_rather_than_overrunning(sources: dict) -> None:
    project = music_project(sources, video_seconds=9.5, song_seconds=4, song="jingle")
    fit(project)
    assert layout(project, "music")[-1][:3] == ("song-3", 8.0, 9.5)

def test_music_that_starts_late_is_still_filled_to_the_end(sources: dict) -> None:
    project = build_project([
        video_track(),
        insert("v", sources["video"], 0, 10),
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        {"action": "add_clip", "track_id": "music", "clip_id": "song", "asset_id": sources["song"],
         "source_range": {"start": 0, "end": 3}, "timeline_in": 2},
    ])
    fit(project)
    music = layout(project, "music")
    assert music[0][1] == 2.0
    assert music[-1][2] == 10.0

def test_fitting_again_after_the_video_grows_extends_the_music(sources: dict) -> None:
    project = music_project(sources, video_seconds=5, song_seconds=20)
    fit(project)
    assert layout(project, "music")[-1][2] == 5.0
    edit(project, [insert("v2", sources["video"], 0, 4)])
    fit(project)
    assert layout(project, "music")[-1][2] == 9.0

def test_fitting_again_after_the_video_shrinks_drops_the_extra_music(sources: dict) -> None:
    project = music_project(sources, video_seconds=9.5, song_seconds=4, song="jingle")
    fit(project)
    assert len(clips_of(project, "music")) == 3
    edit(project, [{"action": "trim_clip", "track_id": "main", "clip_id": "v",
                    "new_source_range": {"start": 0, "end": 3}}])
    fit(project)
    music = layout(project, "music")
    assert len(music) == 1 and music[-1][2] == 3.0

def test_the_fade_out_moves_to_whatever_clip_is_last_now(sources: dict) -> None:
    project = music_project(sources, video_seconds=9.5, song_seconds=4, song="jingle",
                            audio_fade_in=1, audio_fade_out=2)
    fit(project)
    clips = {clip["id"]: clip for clip in clips_of(project, "music")}
    assert float(clips["song"]["audio_fade_in"]) == 1.0
    assert float(clips["song"]["audio_fade_out"]) == 0.0
    # song-3 only runs 8.0 to 9.5, and a fade cannot be longer than the clip it is on.
    assert float(clips["song-3"]["audio_fade_out"]) == 1.5
    assert float(clips["song-2"]["audio_fade_out"]) == 0.0

def test_a_fade_longer_than_the_last_clip_is_clamped_to_it(sources: dict) -> None:
    project = music_project(sources, video_seconds=9.5, song_seconds=4, song="jingle")
    fit(project, fade_out=3)
    last = clips_of(project, "music")[-1]
    length = float(last["source_range"]["end"]) - float(last["source_range"]["start"])
    assert float(last["audio_fade_out"]) == length

def test_fades_can_be_set_in_the_same_operation(sources: dict) -> None:
    project = music_project(sources, video_seconds=8, song_seconds=20)
    fit(project, fade_in=1.5, fade_out=3)
    clip = clips_of(project, "music")[0]
    assert float(clip["audio_fade_in"]) == 1.5 and float(clip["audio_fade_out"]) == 3.0

def test_fades_are_shrunk_to_fit_a_very_short_track(sources: dict) -> None:
    project = music_project(sources, video_seconds=2, song_seconds=20)
    fit(project, fade_in=3, fade_out=3)
    clip = clips_of(project, "music")[0]
    assert float(clip["audio_fade_in"]) + float(clip["audio_fade_out"]) <= 2.0

def test_looping_can_be_turned_off(sources: dict) -> None:
    # `loop` is about going back to the start: the jingle still plays out its last second,
    # it just does not start over afterwards.
    project = music_project(sources, video_seconds=10, song_seconds=3, song="jingle")
    fit(project, loop=False)
    assert layout(project, "music") == [("song", 0.0, 4.0, 0.0, 4.0)]

def test_fitting_a_video_track_is_refused(sources: dict) -> None:
    project = music_project(sources, video_seconds=5, song_seconds=10)
    with pytest.raises(ValueError, match="sets the length itself"):
        edit(project, [{"action": "fit_track", "track_id": "main"}])

def test_fitting_without_any_video_is_refused(sources: dict) -> None:
    project = build_project([
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        insert("song", sources["song"], 0, 5, track_id="music"),
    ])
    with pytest.raises(ValueError, match="nothing to fit"):
        fit(project)

def test_turning_looping_off_still_plays_the_rest_of_the_song(sources: dict) -> None:
    # `loop` is about going back to the start; a song with material left simply carries on.
    project = music_project(sources, video_seconds=9, song_seconds=4)
    fit(project, loop=False)
    assert layout(project, "music") == [("song", 0.0, 9.0, 0.0, 9.0)]
