"""A finished render's sound, checked against the sound of the files it was cut from."""

import json
import os
import subprocess
import wave
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from app import runtime
from app.engine import renderer, sync
from app.models.job import Job, JobKind, JobStatus
from app.models.timeline.parts import Clip, TimeRange, TitleCard, Track, TrackType
from app.runtime import repo
from app.storage.repo import Repository
from app.tools import rendering
from app.tools.catalog import import_asset
from app.tools.job_queue import get_job
from helpers import build_project, insert, video_track

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FRAME = 1 / 30

def make_clip(clip_id: str = "a", asset_id: str = "asset", source_start: float = 0, source_end: float = 2,
              timeline_in: float = 0, volume: float = 1.0, speed: float = 1.0, audio_lead: float = 0,
              audio_lag: float = 0, audio_fade_in: float = 0, audio_fade_out: float = 0,
              card: Optional[TitleCard] = None) -> Clip:
    """Build a clip of the base video track.

    Args:
        clip_id: The clip's ID.
        asset_id: The file it comes from.
        source_start: Where in the file its picture begins.
        source_end: Where in the file its picture ends.
        timeline_in: Where it begins on the timeline.
        volume: How loud its sound plays.
        speed: How fast it plays.
        audio_lead: How far its sound starts before its picture.
        audio_lag: How far its sound runs on after its picture.
        audio_fade_in: How long its sound fades in for.
        audio_fade_out: How long its sound fades out for.
        card: The title card it is, if any.

    Returns:
        The clip.
    """
    return Clip(
        id=clip_id,
        asset_id=asset_id,
        source_range=TimeRange(start=Decimal(str(source_start)), end=Decimal(str(source_end))),
        timeline_in=Decimal(str(timeline_in)),
        volume=volume,
        speed=speed,
        audio_lead=Decimal(str(audio_lead)),
        audio_lag=Decimal(str(audio_lag)),
        audio_fade_in=Decimal(str(audio_fade_in)),
        audio_fade_out=Decimal(str(audio_fade_out)),
        card=card,
    )

def video_tracks(*clips: Clip, above: Optional[List[Clip]] = None) -> List[Track]:
    """Build the project's tracks around one base video track.

    Args:
        *clips: The base video track's clips.
        above: Clips on a second video track drawn over the base, if any.

    Returns:
        The tracks, the base first.
    """
    tracks = [Track(id="main", track_type=TrackType.VIDEO, clips=list(clips))]
    if above:
        tracks.append(Track(id="over", track_type=TrackType.VIDEO, clips=list(above)))
    return tracks

def audio_tracks(*clips: Clip) -> List[Track]:
    """Build one audio track holding these clips."""
    return [Track(id="music", track_type=TrackType.AUDIO, clips=list(clips))]

def test_streams_that_line_up_are_not_reported() -> None:
    probed = {"streams": [
        {"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": "0.000000", "duration": "10.000000"},
    ], "format": {"duration": "10.000000"}}
    assert sync.stream_mismatch(probed, FRAME) is None

def test_an_audio_stream_that_starts_late_is_reported() -> None:
    probed = {"streams": [
        {"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": "0.200000", "duration": "10.000000"},
    ], "format": {"duration": "10.000000"}}
    found = sync.stream_mismatch(probed, FRAME)
    assert found == {"video_start": 0.0, "audio_start": 0.2, "video_length": 10.0, "audio_length": 10.0}

def test_a_file_without_sound_is_not_compared() -> None:
    probed = {"streams": [{"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"}],
              "format": {"duration": "10.000000"}}
    assert sync.stream_mismatch(probed, FRAME) is None

def test_a_stream_without_a_duration_is_measured_by_the_file() -> None:
    probed = {"streams": [
        {"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": "0.000000"},
    ], "format": {"duration": "10.200000"}}
    found = sync.stream_mismatch(probed, FRAME)
    assert found is not None and found["audio_length"] == 10.2

def test_streams_whose_length_nothing_says_are_not_compared() -> None:
    probed = {"streams": [{"codec_type": "video", "start_time": "0.000000"},
                          {"codec_type": "audio", "start_time": "0.300000"}], "format": {}}
    assert sync.stream_mismatch(probed, FRAME) is None

def test_an_audio_stream_one_and_a_half_frames_longer_is_reported() -> None:
    frame = 1 / 30
    probed = {"streams": [
        {"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": "0.000000", "duration": f"{10 + 1.5 * frame:.6f}"},
    ], "format": {"duration": "10.000000"}}
    found = sync.stream_mismatch(probed, frame)
    assert found is not None and found["audio_length"] == round(10 + 1.5 * frame, 3)

def test_an_audio_stream_half_a_frame_longer_is_not_reported() -> None:
    frame = 1 / 30
    probed = {"streams": [
        {"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": "0.000000", "duration": f"{10 + 0.5 * frame:.6f}"},
    ], "format": {"duration": "10.000000"}}
    assert sync.stream_mismatch(probed, frame) is None

def test_an_audio_stream_starting_one_and_a_half_frames_late_is_reported() -> None:
    frame = 1 / 30
    probed = {"streams": [
        {"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": f"{1.5 * frame:.6f}", "duration": "10.000000"},
    ], "format": {"duration": "10.000000"}}
    found = sync.stream_mismatch(probed, frame)
    assert found is not None and found["audio_start"] == round(1.5 * frame, 3)

def test_an_audio_stream_starting_half_a_frame_late_is_not_reported() -> None:
    frame = 1 / 30
    probed = {"streams": [
        {"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": f"{0.5 * frame:.6f}", "duration": "10.000000"},
    ], "format": {"duration": "10.000000"}}
    assert sync.stream_mismatch(probed, frame) is None

def test_a_join_next_to_a_muted_clip_skips_only_that_side() -> None:
    first = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    muted = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, volume=0)
    last = make_clip("c", "asset", source_start=4, source_end=6, timeline_in=4)
    offered, left_out = sync.sides(video_tracks(first, muted, last), [first, muted, last],
                                   {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before"), ("c", "after")]
    assert left_out == {"too_short": 0, "faded": 0}

def test_a_join_into_a_title_card_skips_the_card_side() -> None:
    first = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    card = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, card=TitleCard(title="開場"))
    offered, left_out = sync.sides(video_tracks(first, card), [first, card], {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before")]
    assert left_out == {"too_short": 0, "faded": 0}

def test_another_picture_tracks_sound_trims_the_side_it_runs_over() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=4, timeline_in=4)
    overlay = make_clip("c", "asset", source_start=0, source_end=1, timeline_in=4.2)
    offered, left_out = sync.sides(video_tracks(before, after, above=[overlay]), [before, after],
                                   {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before"), ("b", "after")]
    after_side = next(side for side in offered if side["clip_id"] == "b")
    # The overlay cuts the first 0.2s out of the after side, so the 0.8s past it is compared instead.
    assert (after_side["start"], after_side["end"]) == (5.2, 6.0)
    assert left_out == {"too_short": 0, "faded": 0}

def test_a_muted_picture_track_leaves_the_side_alone() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=4, timeline_in=4)
    overlay = make_clip("c", "asset", source_start=0, source_end=1, timeline_in=4.2, volume=0)
    offered, left_out = sync.sides(video_tracks(before, after, above=[overlay]), [before, after],
                                   {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before"), ("b", "after")]
    after_side = next(side for side in offered if side["clip_id"] == "b")
    assert (after_side["start"], after_side["end"]) == (4.0, 6.0)
    assert left_out == {"too_short": 0, "faded": 0}

def test_sound_from_an_audio_track_over_the_join_leaves_both_sides_out() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=2, timeline_in=2)
    music = make_clip("m", "asset", source_start=0, source_end=4, timeline_in=0)
    offered, left_out = sync.sides(video_tracks(before, after) + audio_tracks(music), [before, after],
                                   {"asset": "asset.wav"})
    assert offered == []
    assert left_out == {"too_short": 2, "faded": 0}

def test_a_clip_too_short_to_line_up_is_counted_on_its_side_of_the_join() -> None:
    # Nothing else plays before the join; the first clip's own sound is only 0.3s of it.
    short = make_clip("a", "asset", source_start=0, source_end=0.3, timeline_in=0)
    last = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=0.3)
    offered, left_out = sync.sides(video_tracks(short, last), [short, last], {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("b", "after")]
    assert left_out == {"too_short": 1, "faded": 0}

def test_a_fade_out_takes_the_end_of_the_before_stretch_off() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0, audio_fade_out=0.5)
    after = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert before_side["end"] == 1.5
    assert left_out == {"too_short": 0, "faded": 0}

def test_a_fade_out_leaving_too_little_stretch_is_counted_faded() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0, audio_fade_out=1.6)
    after = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("b", "after")]
    assert left_out == {"too_short": 0, "faded": 1}

def test_a_fade_in_takes_the_start_of_the_after_stretch_off() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    after = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, audio_fade_in=0.5)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    after_side = next(side for side in offered if side["clip_id"] == "b")
    assert after_side["start"] == 2.5
    assert left_out == {"too_short": 0, "faded": 0}

def test_a_fade_in_leaving_too_little_stretch_is_counted_faded() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    after = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, audio_fade_in=1.6)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before")]
    assert left_out == {"too_short": 0, "faded": 1}

def test_a_clip_too_short_even_without_its_fade_is_counted_too_short() -> None:
    short = make_clip("a", "asset", source_start=0, source_end=0.3, timeline_in=0, audio_fade_out=0.1)
    last = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=0.3)
    offered, left_out = sync.sides(video_tracks(short, last), [short, last], {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("b", "after")]
    assert left_out == {"too_short": 1, "faded": 0}

def test_a_side_of_exactly_the_minimum_length_has_no_room_to_move_and_is_too_short() -> None:
    short = make_clip("a", "asset", source_start=0, source_end=sync.MIN_SIDE_SECONDS, timeline_in=0)
    last = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=sync.MIN_SIDE_SECONDS)
    offered, left_out = sync.sides(video_tracks(short, last), [short, last], {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("b", "after")]
    assert left_out == {"too_short": 1, "faded": 0}

def test_a_side_of_the_minimum_length_plus_reach_is_offered() -> None:
    shortest = make_clip("a", "asset", source_start=0, source_end=sync.MIN_SIDE_SECONDS + sync.MIN_REACH_SECONDS,
                         timeline_in=0)
    last = make_clip("b", "asset", source_start=2, source_end=4,
                     timeline_in=sync.MIN_SIDE_SECONDS + sync.MIN_REACH_SECONDS)
    offered, left_out = sync.sides(video_tracks(shortest, last), [shortest, last], {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before"), ("b", "after")]
    assert left_out == {"too_short": 0, "faded": 0}

def test_a_fade_in_on_a_lead_is_measured_from_where_the_sound_comes_in() -> None:
    # The lead puts the sound in 0.5s before the picture, so its 0.3s fade is over before the join.
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    after = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, audio_lead=0.5, audio_fade_in=0.3)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    after_side = next(side for side in offered if side["clip_id"] == "b")
    assert after_side["start"] == 2.0
    assert left_out == {"too_short": 0, "faded": 0}

def test_a_fade_out_on_a_lag_is_measured_from_where_the_sound_stops() -> None:
    # The lag carries the sound 0.5s past the picture, so its 0.3s fade starts after the join.
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0, audio_lag=0.5, audio_fade_out=0.3)
    after = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert before_side["end"] == 2.0
    assert left_out == {"too_short": 0, "faded": 0}

def test_a_picture_track_without_sound_does_not_trim_the_side_it_runs_over() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=4, timeline_in=4)
    overlay = make_clip("c", "silent", source_start=0, source_end=1, timeline_in=4.2)
    offered, left_out = sync.sides(video_tracks(before, after, above=[overlay]), [before, after],
                                   {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before"), ("b", "after")]
    after_side = next(side for side in offered if side["clip_id"] == "b")
    assert (after_side["start"], after_side["end"]) == (4.0, 6.0)
    assert left_out == {"too_short": 0, "faded": 0}

def test_a_base_clip_without_sound_is_skipped_and_not_counted() -> None:
    heard = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    silent = make_clip("b", "silent", source_start=0, source_end=2, timeline_in=2)
    offered, left_out = sync.sides(video_tracks(heard, silent), [heard, silent], {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before")]
    assert left_out == {"too_short": 0, "faded": 0}

def test_a_title_card_on_another_track_is_not_other_sound() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=4, timeline_in=4)
    card = make_clip("c", "asset", source_start=0, source_end=1, timeline_in=4.2,
                     card=TitleCard(title="開場"))
    offered, left_out = sync.sides(video_tracks(before, after, above=[card]), [before, after],
                                   {"asset": "asset.wav"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before"), ("b", "after")]
    after_side = next(side for side in offered if side["clip_id"] == "b")
    assert (after_side["start"], after_side["end"]) == (4.0, 6.0)
    assert left_out == {"too_short": 0, "faded": 0}

def test_other_sound_that_ends_before_the_join_still_trims_the_before_side() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=4, source_end=8, timeline_in=4)
    overlay = make_clip("c", "asset", source_start=0, source_end=0.8, timeline_in=2.2)
    offered, _ = sync.sides(video_tracks(before, after, above=[overlay]), [before, after],
                            {"asset": "asset.wav"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert (before_side["start"], before_side["end"]) == (3.0, 4.0)

def test_a_short_clear_piece_next_to_the_join_gives_way_to_a_longer_one_further_out() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=4, source_end=8, timeline_in=4)
    broll = make_clip("c", "asset", source_start=0, source_end=0.8, timeline_in=3.0)
    offered, left_out = sync.sides(video_tracks(before, after, above=[broll]), [before, after],
                                   {"asset": "asset.wav"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert (before_side["start"], before_side["end"]) == (2.0, 3.0)
    assert left_out == {"too_short": 0, "faded": 0}

def test_the_nearest_of_two_pieces_long_enough_is_used() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=4, source_end=8, timeline_in=4)
    broll = make_clip("c", "asset", source_start=0, source_end=0.4, timeline_in=2.8)
    offered, left_out = sync.sides(video_tracks(before, after, above=[broll]), [before, after],
                                   {"asset": "asset.wav"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    # Both the 0.8s next to the join and the 0.8s before the overlay are long enough; the nearer one is used.
    assert (before_side["start"], before_side["end"]) == (3.2, 4.0)
    assert left_out == {"too_short": 0, "faded": 0}

def test_the_nearest_of_two_pieces_long_enough_is_used_on_the_after_side() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=4, source_end=8, timeline_in=4)
    broll = make_clip("c", "asset", source_start=0, source_end=0.4, timeline_in=4.8)
    offered, left_out = sync.sides(video_tracks(before, after, above=[broll]), [before, after],
                                   {"asset": "asset.wav"})
    after_side = next(side for side in offered if side["clip_id"] == "b")
    # Both the 0.8s next to the join and the 0.8s after the overlay are long enough; the nearer one is used.
    assert (after_side["start"], after_side["end"]) == (4.0, 4.8)
    assert left_out == {"too_short": 0, "faded": 0}

def test_a_side_whose_nearest_piece_is_cut_down_by_its_fade_is_faded() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0, audio_fade_out=0.9)
    after = make_clip("b", "asset", source_start=4, source_end=8, timeline_in=4)
    broll = make_clip("c", "asset", source_start=0, source_end=0.5, timeline_in=2.5)
    offered, left_out = sync.sides(video_tracks(before, after, above=[broll]), [before, after],
                                   {"asset": "asset.wav"})
    # The nearest piece is 1.0s before the fade and only 0.1s after it, so it is not compared; the
    # farther piece is 0.5s either way, so the side is still counted as faded rather than too short.
    assert [side["clip_id"] for side in offered] == ["b"]
    assert left_out == {"too_short": 0, "faded": 1}

def test_a_long_side_is_compared_over_exactly_the_join_window() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=6, timeline_in=0)
    after = make_clip("b", "asset", source_start=6, source_end=8, timeline_in=6)
    offered, _ = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert abs((before_side["end"] - before_side["start"]) - sync.JOIN_WINDOW_SECONDS) <= 1e-9

def test_base_clips_given_out_of_order_give_the_same_sides() -> None:
    first = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    second = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2)
    in_order, counts_in_order = sync.sides(video_tracks(first, second), [first, second],
                                           {"asset": "asset.wav"})
    out_of_order, counts_out_of_order = sync.sides(video_tracks(second, first), [second, first],
                                                   {"asset": "asset.wav"})
    assert out_of_order == in_order
    assert counts_out_of_order == counts_in_order

def test_a_lag_on_the_clip_before_gives_the_after_side_the_windows_rest() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0, audio_lag=1.0)
    after = make_clip("b", "asset", source_start=0, source_end=2, timeline_in=2)
    offered, _ = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    after_side = next(side for side in offered if side["clip_id"] == "b")
    # The lagged tail of the before clip's sound covers the start of the after side.
    assert after_side["start"] == 3.0

def test_a_lead_on_the_clip_after_ends_the_before_side_where_its_sound_starts() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=2, timeline_in=2, audio_lead=1.0)
    offered, _ = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert before_side["end"] == 1.0

def test_a_side_names_the_file_second_that_plays_where_it_starts() -> None:
    before = make_clip("a", "asset", source_start=2, source_end=4, timeline_in=0, speed=2.0)
    after = make_clip("b", "asset", source_start=4, source_end=6, timeline_in=1)
    offered, _ = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert (before_side["start"], before_side["end"], before_side["source_start"], before_side["speed"]) == (
        0.0, 1.0, 2.0, 2.0)

def test_the_after_side_of_a_j_cut_names_the_file_second_its_sound_plays() -> None:
    # The sound of b starts 0.5s before its picture, from 3.5s in its file, so at the join it is at 4.0s.
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=4, source_end=6, timeline_in=4, audio_lead=0.5)
    offered, _ = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    after_side = next(side for side in offered if side["clip_id"] == "b")
    assert (after_side["start"], after_side["end"], after_side["source_start"]) == (4.0, 6.0, 4.0)

def test_a_side_that_starts_inside_a_sped_up_clip_names_the_file_second_that_plays() -> None:
    before = make_clip("a", "asset", source_start=2, source_end=8, timeline_in=0, speed=1.5)
    after = make_clip("b", "asset", source_start=8, source_end=10, timeline_in=4)
    offered, _ = sync.sides(video_tracks(before, after), [before, after], {"asset": "asset.wav"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    # The side opens 2s into the clip, three file seconds in: 2.0 + (2.0 - 0.0) * 1.5.
    expected_source_start = 2.0 + (2.0 - 0.0) * 1.5
    assert (before_side["start"], before_side["end"]) == (2.0, 4.0)
    assert before_side["source_start"] == pytest.approx(expected_source_start)
    assert before_side["speed"] == 1.5

def make_sound(seconds: float = 6.0, seed: int = 1,
               bursts=((1.0, 1.4), (2.2, 2.7), (3.5, 4.0), (4.7, 5.2))) -> np.ndarray:
    """Build room tone with bursts of a tone in it, standing in for somebody talking.

    Args:
        seconds: How long the sound is.
        seed: The noise seed, so two sounds can be made unrelated.
        bursts: `(start, end)` of each burst of tone, in seconds.

    Returns:
        The samples, mono at `sync.SAMPLE_RATE`.
    """
    moments = np.arange(int(round(seconds * sync.SAMPLE_RATE))) / sync.SAMPLE_RATE
    samples = np.random.default_rng(seed).normal(0.0, 0.01, moments.size)
    for start, end in bursts:
        inside = (moments >= start) & (moments < end)
        samples[inside] += 0.4 * np.sin(2 * np.pi * 440 * moments[inside])
    return samples.astype(np.float32)

def write_sound(path: Path, samples: np.ndarray) -> Path:
    """Write mono samples as a WAV file FFmpeg can read exactly."""
    data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sync.SAMPLE_RATE)
        handle.writeframes(data.tobytes())
    return path

def shifted(samples: np.ndarray, seconds: float) -> np.ndarray:
    """Move the sound later by a positive number of seconds, or earlier by a negative one."""
    count = int(round(abs(seconds) * sync.SAMPLE_RATE))
    if seconds > 0:
        return np.concatenate((np.zeros(count, dtype=np.float32), samples))
    return samples[count:]

def side_of(start: float = 0.0, end: float = 4.0, source_start: float = 0.0, speed: float = 1.0) -> dict:
    """Build the side `lag_at` compares, with no file behind it."""
    return {"start": start, "end": end, "source_start": source_start, "speed": speed}

def periodic_envelope(values: List[int], seconds: float = 4.0, shift: int = 0) -> np.ndarray:
    """Build sound whose loudness repeats exactly every `len(values)` grid steps.

    Each grid step is filled with one constant, so its measured loudness is
    that constant and correlations between shifted copies come out exactly
    equal rather than differing by rounding.

    Args:
        values: The loudness of each step of the period, in order.
        seconds: How long the sound is.
        shift: How many steps later the period starts, to move the sound on.

    Returns:
        The samples, mono at `sync.SAMPLE_RATE`.
    """
    width = int(round(sync.LAG_STEP_SECONDS * sync.SAMPLE_RATE))
    blocks = int(round(seconds / sync.LAG_STEP_SECONDS))
    pattern = np.array(values, dtype=np.float32)
    samples = np.empty(blocks * width, dtype=np.float32)
    for block in range(blocks):
        samples[block * width:(block + 1) * width] = pattern[(block - shift) % len(pattern)]
    return samples

def test_identical_sound_has_no_lag() -> None:
    sound = make_sound()
    assert sync.lag_at(sound, 0.0, sound, side_of()) == 0.0

def test_sound_heard_later_than_the_cut_says_has_a_positive_lag() -> None:
    sound = make_sound()
    assert sync.lag_at(shifted(sound, 0.12), 0.0, sound, side_of()) == 0.12

def test_sound_heard_sooner_than_the_cut_says_has_a_negative_lag() -> None:
    sound = make_sound()
    assert sync.lag_at(shifted(sound, -0.12), 0.0, sound, side_of()) == -0.12

def test_a_side_reads_the_file_from_its_own_source_start() -> None:
    sound = make_sound()
    later = sound[sync.SAMPLE_RATE:]
    assert sync.lag_at(later, 0.0, sound, side_of(source_start=1.0), source_origin=0.0) == 0.0

def test_digital_silence_has_nothing_to_line_up() -> None:
    silence = np.zeros(6 * sync.SAMPLE_RATE, dtype=np.float32)
    assert sync.lag_at(silence, 0.0, silence, side_of()) is None
    assert sync.lag_at(make_sound(), 0.0, silence, side_of()) is None

def test_unrelated_sound_does_not_follow_closely_enough_to_be_believed() -> None:
    sound = make_sound()
    other = make_sound(seed=7, bursts=((0.3, 0.7), (1.8, 2.1), (3.1, 3.8), (4.9, 5.4)))
    render = other[sync.SAMPLE_RATE:]
    assert sync.lag_at(render, 0.0, sound, side_of(source_start=1.0)) is None

def test_a_lag_past_the_edge_of_the_search_is_not_reported() -> None:
    # The best shift inside the search leaves the sound 0.1s out of step, which does not follow closely
    # enough to be believed, so this is weak rather than beyond reach.
    sound = make_sound()
    assert sync.lag_at(shifted(sound, 0.6), 0.0, sound, side_of()) is None

def test_a_lag_exactly_at_the_edge_of_the_search_is_reported_as_beyond_reach() -> None:
    sound = make_sound()
    assert sync.lag_at(shifted(sound, sync.MAX_LAG_SECONDS), 0.0, sound, side_of()) == sync.BEYOND_REACH

def test_a_one_second_side_at_the_search_edge_is_beyond_reach() -> None:
    sound = make_sound()
    assert sync.lag_at(shifted(sound, sync.MAX_LAG_SECONDS), 0.0, sound, side_of(0.0, 1.0)) == sync.BEYOND_REACH

def test_a_one_second_side_one_step_inside_the_search_edge_is_reported_as_that_lag() -> None:
    # A 1s side can be moved `limit` steps and still keep MIN_SIDE_SECONDS compared. One step inside that
    # edge the sound is a real lag, not beyond reach.
    sound = make_sound()
    steps = int(round(1.0 / sync.LAG_STEP_SECONDS))
    least = int(round(sync.MIN_SIDE_SECONDS / sync.LAG_STEP_SECONDS))
    limit = min(int(round(sync.MAX_LAG_SECONDS / sync.LAG_STEP_SECONDS)), max(0, steps - least))
    offset = (limit - 1) * sync.LAG_STEP_SECONDS
    assert sync.lag_at(shifted(sound, offset), 0.0, sound, side_of(0.0, 1.0)) == pytest.approx(offset)

def test_a_short_side_does_not_report_an_offset_past_its_own_reach_as_a_smaller_one() -> None:
    # A side of 0.7s can only be moved 0.2s and still keep 0.5s compared. A best lag at the end of
    # that reach may be a bigger offset cut short, so it comes back beyond reach, as at the search's edge.
    sound = make_sound()
    assert sync.lag_at(shifted(sound, 0.2), 0.0, sound, side_of(1.0, 1.7, 1.0), source_origin=0.0) == sync.BEYOND_REACH
    # Past the reach, the best lag inside it is 0.1s out of step and does not follow closely enough.
    assert sync.lag_at(shifted(sound, 0.3), 0.0, sound, side_of(1.0, 1.7, 1.0), source_origin=0.0) is None
    assert sync.lag_at(shifted(sound, 0.1), 0.0, sound, side_of(1.0, 1.7, 1.0), source_origin=0.0) == 0.1

def test_a_window_reaching_past_the_end_of_the_sound_is_measured_against_the_whole_window() -> None:
    # Half a window of full-scale sound: its RMS over the whole window is the square root of a half, not 1.
    half = np.ones(int(sync.LAG_STEP_SECONDS * sync.SAMPLE_RATE) // 2, dtype=np.float32)
    assert sync._loudness(half, 0.0, [0.0]) == pytest.approx([np.sqrt(0.5)])

def test_a_render_with_nothing_varying_has_nothing_to_line_up() -> None:
    sound = make_sound()
    silence = np.zeros(6 * sync.SAMPLE_RATE, dtype=np.float32)
    assert sync.lag_at(silence, 0.0, sound, side_of()) is None

def test_a_periodic_envelope_in_step_keeps_the_zero_lag() -> None:
    pattern = list(range(1, 21))
    sound = periodic_envelope(pattern)
    assert sync.lag_at(sound, 0.0, sound, side_of()) == 0.0

def test_a_periodic_envelope_heard_a_tenth_early_prefers_the_negative_lag() -> None:
    pattern = list(range(1, 21))
    sound = periodic_envelope(pattern)
    early = periodic_envelope(pattern, shift=10)
    assert sync.lag_at(early, 0.0, sound, side_of()) == -0.1

def test_the_file_is_read_over_windows_scaled_by_its_speed() -> None:
    # At speed 2.0 each render step of 0.01s covers 0.02s of the file. The file's block is split into
    # halves h and s with h anti-correlated with the render and s = sqrt(2r² - h²), so only the full
    # 0.02s window has the render's loudness r; a 0.01s window would see h alone.
    steps = int(round(4.0 / sync.LAG_STEP_SECONDS))
    loud = np.random.default_rng(5).uniform(0.4, 0.8, steps)
    first = 0.9 - loud
    second = np.sqrt(2.0 * loud ** 2 - first ** 2)
    width = int(round(sync.LAG_STEP_SECONDS * sync.SAMPLE_RATE))
    render = np.repeat(loud.astype(np.float32), width)
    file = np.repeat(np.stack((first, second), axis=1).ravel().astype(np.float32), width)
    assert sync.lag_at(render, 0.0, file, side_of(0.0, 4.0, 0.0, speed=2.0)) == 0.0

def test_an_after_side_heard_early_is_measured_only_inside_the_side() -> None:
    a = make_sound(seconds=3.0, seed=3, bursts=((0.1, 0.5), (0.9, 1.3), (1.8, 2.2), (2.6, 2.9)))
    b = make_sound(seconds=3.0, seed=11, bursts=((0.05, 0.4), (0.7, 1.1), (1.5, 1.9), (2.4, 2.8)))
    # B is heard 0.2s early: at the join it has already played 0.2s, while A is still what plays before it.
    render = np.concatenate((a[:int(2.0 * sync.SAMPLE_RATE)],
                             b[int(0.2 * sync.SAMPLE_RATE):int(1.2 * sync.SAMPLE_RATE)]))
    assert sync.lag_at(render, 0.0, b[:int(1.0 * sync.SAMPLE_RATE)], side_of(2.0, 3.0)) == -0.2

def test_a_before_side_heard_late_is_measured_only_inside_the_side() -> None:
    a = make_sound(seconds=3.0, seed=3, bursts=((0.1, 0.5), (0.9, 1.3), (1.8, 2.2), (2.6, 2.9)))
    b = make_sound(seconds=3.0, seed=11, bursts=((0.05, 0.4), (0.7, 1.1), (1.5, 1.9), (2.4, 2.8)))
    # A is heard 0.2s late, so its first 0.8s runs from 0.2s and the after clip's sound takes the join.
    render = np.concatenate((np.zeros(int(0.2 * sync.SAMPLE_RATE), dtype=np.float32),
                             a[:int(0.8 * sync.SAMPLE_RATE)], b[:int(0.2 * sync.SAMPLE_RATE)]))
    assert sync.lag_at(render, 0.0, a[:int(1.0 * sync.SAMPLE_RATE)], side_of(0.0, 1.0)) == 0.2

def plain_render(tmp_path: Path) -> Path:
    """Write a short file with picture and sound for the check to read.

    Args:
        tmp_path: The test's temporary folder.

    Returns:
        The file.
    """
    path = tmp_path / "plain.mp4"
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return path

def test_check_compares_a_real_file_and_reports_its_streams(tmp_path: Path) -> None:
    path = tmp_path / "late.mp4"
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=3",
        "-itsoffset", "0.3", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True)

    def refuse(*args, **kwargs):
        raise AssertionError("the sound was decoded though there was nothing to compare it with")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(sync, "_read_stretch", refuse)
        found = sync.check(str(path), [], {}, FRAME, FFMPEG, "ffprobe")
    assert found["streams"] is not None
    assert found["sides_asked"] == 0
    assert found["unlined"] == unlined(0, 0)
    assert found["left_out"] == {}
    assert found["frame"] == round(FRAME, 4)

def test_check_does_not_decode_when_there_are_no_sides(tmp_path: Path) -> None:
    path = plain_render(tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("the sound was decoded though no side was asked about")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(sync, "_read_stretch", refuse)
        found = sync.check(str(path), [], {"too_short": 1, "faded": 0}, FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == []
    assert found["sides_asked"] == 0
    assert found["unlined"] == unlined(0, 0)
    assert found["left_out"] == {"too_short": 1, "faded": 0}

def test_check_reads_the_file_stretch_the_side_plays_at_its_speed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def read(path, start, length, ffmpeg_bin):
        calls.append((path, start, length))
        return np.zeros(sync.SAMPLE_RATE, dtype=np.float32)

    monkeypatch.setattr(sync, "_read_stretch", read)
    monkeypatch.setattr(sync, "probe_file", lambda *args: {})
    source = tmp_path / "source.wav"
    side = check_side(source, 0.0, 1.0, 2.0, speed=2.0)
    sync.check(str(tmp_path / "render.mp4"), [side], {"too_short": 0, "faded": 0}, FRAME, FFMPEG, "ffprobe")
    assert calls[0][1] == 0.0
    assert calls[1][0] == str(source)
    assert calls[1][1] >= 0.0
    assert calls[1][2] == pytest.approx(2.0 + sync.LAG_STEP_SECONDS * 2.0)

def ffmpeg(*args: str) -> None:
    """Run FFmpeg, failing the test if it cannot."""
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], check=True)

def source_file(tmp_path: Path) -> Path:
    """Write the source file the real-FFmpeg checks are made from."""
    return write_sound(tmp_path / "source.wav", make_sound())

def cut_render(tmp_path: Path, source: Path, start: float = 1.0, length: float = 4.0,
               audio_filter: Optional[str] = None) -> Path:
    """Build a render of one stretch of the source, in step unless a filter moves it."""
    render = tmp_path / "render.wav"
    arguments = ["-ss", f"{start}", "-t", f"{length}", "-i", str(source)]
    if audio_filter is not None:
        arguments += ["-af", audio_filter]
    ffmpeg(*arguments, "-c:a", "pcm_s16le", str(render))
    return render

def unlined(weak: int, beyond_reach: int, reach: float = 0.5) -> dict:
    """What `check` records for sides it could not line up, each searched `reach` seconds each way."""
    return {"weak": weak, "beyond_reach": beyond_reach,
            "searched": {"weak": [reach, reach] if weak else None,
                         "beyond_reach": [reach, reach] if beyond_reach else None}}

def check_side(path: Path, start: float, end: float, source_start: float, speed: float = 1.0,
               at: float = 0.0) -> dict:
    """Build one side as `sides` would offer it, pointing at a real file."""
    return {"at": at, "clip_id": "clip", "side": "after", "start": start, "end": end,
            "path": str(path), "source_start": source_start, "speed": speed}

def test_a_render_in_step_with_its_file_has_no_lag(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    render = cut_render(tmp_path, source)
    found = sync.check(str(render), [check_side(source, 0.0, 2.0, 1.0)], {"too_short": 0, "faded": 0},
                       FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == [{"at": 0.0, "clip_id": "clip", "side": "after", "lag": 0.0}]
    assert found["unlined"] == unlined(0, 0)

def test_a_render_delayed_a_tenth_of_a_second_has_a_lag_of_a_tenth(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    render = cut_render(tmp_path, source, audio_filter="adelay=100")
    found = sync.check(str(render), [check_side(source, 0.0, 2.0, 1.0)], {"too_short": 0, "faded": 0},
                       FRAME, FFMPEG, "ffprobe")
    assert found["joins"][0]["lag"] == pytest.approx(0.1, abs=0.01)

def test_a_render_a_tenth_of_a_second_early_has_a_lag_of_minus_a_tenth(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    render = cut_render(tmp_path, source, start=1.1)
    found = sync.check(str(render), [check_side(source, 0.0, 2.0, 1.0)], {"too_short": 0, "faded": 0},
                       FRAME, FFMPEG, "ffprobe")
    assert found["joins"][0]["lag"] == pytest.approx(-0.1, abs=0.01)

def test_a_render_two_seconds_of_room_tone_in_is_in_step(tmp_path: Path) -> None:
    # The voice detector used to hear the first burst after this much room tone late, though it was in step.
    source = source_file(tmp_path)
    render = cut_render(tmp_path, source)
    found = sync.check(str(render), [check_side(source, 2.0, 4.0, 3.0, at=2.0)], {"too_short": 0, "faded": 0},
                       FRAME, FFMPEG, "ffprobe")
    assert found["joins"][0]["lag"] == pytest.approx(0.0, abs=0.01)

def test_digital_silence_in_the_file_has_nothing_to_line_up(tmp_path: Path) -> None:
    samples = make_sound()
    samples[:3 * sync.SAMPLE_RATE] = 0.0
    source = write_sound(tmp_path / "silent_start.wav", samples)
    render = cut_render(tmp_path, source_file(tmp_path))
    found = sync.check(str(render), [check_side(source, 0.0, 2.0, 0.0)], {"too_short": 0, "faded": 0},
                       FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == []
    assert found["unlined"] == unlined(1, 0)

def test_a_clip_at_double_speed_lines_up_with_its_sped_up_file(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    render = cut_render(tmp_path, source, start=0.0, length=6.0, audio_filter="atempo=2.0")
    found = sync.check(str(render), [check_side(source, 0.0, 2.0, 0.0, speed=2.0)], {"too_short": 0, "faded": 0},
                       FRAME, FFMPEG, "ffprobe")
    assert found["joins"][0]["lag"] == pytest.approx(0.0, abs=0.02)

def test_a_delay_past_the_edge_of_the_search_is_not_reported(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    render = cut_render(tmp_path, source, audio_filter="adelay=600")
    found = sync.check(str(render), [check_side(source, 0.0, 2.0, 1.0)], {"too_short": 0, "faded": 0},
                       FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == []
    # Within the search the sound is 0.1s out of step, which does not follow closely enough to be believed.
    assert found["unlined"] == unlined(1, 0)

def test_sound_from_an_unrelated_file_does_not_line_up(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    other = write_sound(tmp_path / "other.wav",
                        make_sound(seed=7, bursts=((0.3, 0.7), (1.8, 2.1), (3.1, 3.8), (4.9, 5.4))))
    render = cut_render(tmp_path, other)
    found = sync.check(str(render), [check_side(source, 0.0, 2.0, 1.0)], {"too_short": 0, "faded": 0},
                       FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == []
    assert found["unlined"] == unlined(1, 0)

def test_a_side_whose_file_is_missing_counts_as_unlined_and_the_check_returns(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    render = cut_render(tmp_path, source)
    missing = tmp_path / "missing.wav"
    found = sync.check(str(render), [check_side(missing, 0.0, 2.0, 1.0)], {"too_short": 0, "faded": 0},
                       FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == []
    assert found["sides_asked"] == 1
    assert found["unlined"] == unlined(1, 0)

def test_a_side_searches_as_far_as_it_can_be_moved_with_enough_still_compared() -> None:
    assert sync.reach(side_of(0.0, 2.0)) == sync.MAX_LAG_SECONDS
    assert sync.reach(side_of(0.0, 1.0)) == 0.5
    assert sync.reach(side_of(0.0, 0.8)) == 0.3
    assert sync.reach(side_of(1.0, 1.7)) == 0.2

def test_the_check_records_how_far_the_sides_it_could_not_line_up_were_searched(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    render = cut_render(tmp_path, source)
    missing = tmp_path / "missing.wav"
    sides = [check_side(missing, 0.0, 2.0, 1.0), check_side(missing, 2.0, 2.8, 3.0)]
    found = sync.check(str(render), sides, {"too_short": 0, "faded": 0}, FRAME, FFMPEG, "ffprobe")
    assert found["unlined"] == {"weak": 2, "beyond_reach": 0,
                                "searched": {"weak": [0.3, 0.5], "beyond_reach": None}}

def test_a_short_side_late_past_its_reach_records_that_reach(tmp_path: Path) -> None:
    # 0.8s can be moved 0.3s: 0.3s late is at the edge of that, so it is beyond the 0.3s searched, not the 0.5s.
    source = source_file(tmp_path)
    render = cut_render(tmp_path, source, audio_filter="adelay=300:all=1")
    found = sync.check(str(render), [check_side(source, 1.0, 1.8, 2.0)], {"too_short": 0, "faded": 0},
                       FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == []
    assert found["unlined"] == unlined(0, 1, reach=0.3)

def test_an_oserror_reading_a_side_counts_that_side_and_the_rest_are_still_checked(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = source_file(tmp_path)
    render = cut_render(tmp_path, source)
    unreadable = tmp_path / "unreadable.wav"
    real = sync._read_stretch

    def read(path, start, length, ffmpeg_bin):
        if path == str(unreadable):
            raise OSError("the file cannot be read")
        return real(path, start, length, ffmpeg_bin)

    monkeypatch.setattr(sync, "_read_stretch", read)
    sides = [check_side(unreadable, 0.0, 2.0, 1.0), check_side(source, 0.0, 2.0, 1.0)]
    found = sync.check(str(render), sides, {"too_short": 0, "faded": 0}, FRAME, FFMPEG, "ffprobe")
    assert found["sides_asked"] == 2
    assert found["unlined"] == unlined(1, 0)
    assert found["joins"] == [{"at": 0.0, "clip_id": "clip", "side": "after", "lag": 0.0}]

def test_a_finished_render_keeps_what_its_sound_check_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = Repository(str(tmp_path / "jobs.db"))
    found = {"streams": None, "joins": [{"at": 2.0, "lag": 0.12}], "frame": round(FRAME, 4)}
    (tmp_path / renderer.SPEC_FILE_NAME).write_text(json.dumps({"render_path": str(tmp_path / "out.mp4")}))
    job = Job(kind=JobKind.RENDER, work_dir=str(tmp_path))
    store.add_job(job)
    monkeypatch.setattr(renderer, "_run_render", lambda job, spec, context, repo: sync.keep(job.work_dir, found))
    monkeypatch.setattr(renderer.JobManager, "launch_ready", lambda self: None)
    renderer.run_worker(store, job.job_id)
    assert store.get_job(job.job_id).sync == found

def make_finished_job(project_id: str, sync_data: Optional[dict]) -> Job:
    """Store a completed render with what its sound check found.

    Args:
        project_id: The project it was rendered from.
        sync_data: The check's result, or None when it was not checked.

    Returns:
        The stored job.
    """
    job = Job(kind=JobKind.RENDER, project_id=project_id, status=JobStatus.COMPLETED, sync=sync_data)
    repo.add_job(job)
    return job

def test_get_job_names_the_clip_and_the_side_whose_sound_is_heard_late(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.12}],
        "sides_asked": 1, "unlined": unlined(0, 0),
        "left_out": {"too_short": 0, "faded": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "out_of_step": ["at 2.00s, clip b, after the join: the sound is heard 0.12s late"]
    }

def test_get_job_says_nothing_about_sound_within_a_frame(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.02}],
        "sides_asked": 1, "unlined": unlined(0, 0),
        "left_out": {"too_short": 0, "faded": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_get_job_says_where_each_stream_starts_and_ends(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    streams = {"video_start": 0.0, "audio_start": 0.3, "video_length": 3.0, "audio_length": 2.7}
    job = make_finished_job(project, {"streams": streams, "joins": [], "sides_asked": 0,
                                      "unlined": unlined(0, 0),
                                      "left_out": {"too_short": 0, "faded": 0}, "frame": FRAME})
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {"out_of_step": ["the sound runs 0.30s to 3.00s while the picture runs 0.00s to 3.00s"]}

def test_get_job_ignores_a_lag_within_two_envelope_steps(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.02}],
        "sides_asked": 1, "unlined": unlined(0, 0),
        "left_out": {"too_short": 0, "faded": 0}, "frame": 1 / 60,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_get_job_reports_a_lag_beyond_two_envelope_steps(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.03}],
        "sides_asked": 1, "unlined": unlined(0, 0),
        "left_out": {"too_short": 0, "faded": 0}, "frame": 1 / 60,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "out_of_step": ["at 2.00s, clip b, after the join: the sound is heard 0.03s late"]
    }

def test_get_job_keeps_a_lag_under_one_frame_at_25_fps(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360, fps_num=25, fps_den=1)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.035}],
        "sides_asked": 1, "unlined": unlined(0, 0),
        "left_out": {"too_short": 0, "faded": 0}, "frame": 1 / 25,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_get_job_keeps_a_lag_under_one_frame_at_a_slower_frame_rate(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.08}],
        "sides_asked": 1, "unlined": unlined(0, 0),
        "left_out": {"too_short": 0, "faded": 0}, "frame": 1 / 10,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_get_job_says_when_the_sound_was_not_checked(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, None)
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_checked": "the sound was not checked against the picture, so whether it lines up is not known"
    }

# How get_job says a side could not be lined up at any shift searched.
WEAK = ("whose sound followed the file's closely enough at no shift within 0.5s: "
        "further off, too changed to compare, or its file unreadable")

def test_get_job_says_when_no_side_could_be_compared(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [], "sides_asked": 3, "unlined": unlined(3, 0),
        "left_out": {"too_short": 0, "faded": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_compared": "3 of the 3 sides of joins could not be lined up — 3 " + WEAK
                        + " — so whether they are in step is not known"
    }

def test_get_job_lists_only_the_reasons_that_left_sides_out(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None,
        "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.0}],
        "sides_asked": 3, "unlined": unlined(2, 0),
        "left_out": {"too_short": 2, "faded": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_compared": "4 of the 5 sides of joins could not be lined up — "
                        "2 with less than 0.7s of their own sound clear of other sound, "
                        "2 " + WEAK + " — "
                        "so whether they are in step is not known"
    }

def test_get_job_lists_weak_sound_and_beyond_reach_apart(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [], "sides_asked": 2,
        "unlined": unlined(1, 1),
        "left_out": {"too_short": 0, "faded": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_compared": "2 of the 2 sides of joins could not be lined up — "
                        "1 " + WEAK + ", "
                        "1 whose sound may be further off than the 0.5s searched — "
                        "so whether they are in step is not known"
    }

def test_get_job_names_how_far_short_sides_were_searched(media: Path) -> None:
    # A 0.8s side can only be moved 0.3s and keep 0.5s compared, so that is all that was searched.
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [], "sides_asked": 3,
        "unlined": {"weak": 2, "beyond_reach": 1,
                    "searched": {"weak": [0.3, 0.5], "beyond_reach": [0.3, 0.3]}},
        "left_out": {"too_short": 0, "faded": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_compared": "3 of the 3 sides of joins could not be lined up — "
                        "2 whose sound followed the file's closely enough at no shift within 0.3–0.5s "
                        "(a shorter side searches less): further off, too changed to compare, "
                        "or its file unreadable, "
                        "1 whose sound may be further off than the 0.3s searched — "
                        "so whether they are in step is not known"
    }

def test_get_job_puts_a_side_down_to_its_fade(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None,
        "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.0}],
        "sides_asked": 2, "unlined": unlined(1, 0),
        "left_out": {"too_short": 0, "faded": 1}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_compared": "2 of the 3 sides of joins could not be lined up — "
                        "1 where the clip's own fade leaves less than 0.7s to compare, "
                        "1 " + WEAK + " — "
                        "so whether they are in step is not known"
    }

def test_get_job_says_what_was_not_compared_when_some_sides_were_lined_up(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None,
        "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.0},
                  {"at": 4.0, "clip_id": "c", "side": "after", "lag": 0.0}],
        "sides_asked": 3, "unlined": unlined(1, 0),
        "left_out": {"too_short": 0, "faded": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_compared": "1 of the 3 sides of joins could not be lined up — 1 " + WEAK
                        + " — so whether they are in step is not known"
    }

def test_get_job_says_nothing_when_every_side_was_in_step(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None,
        "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.0},
                  {"at": 4.0, "clip_id": "c", "side": "after", "lag": 0.0}],
        "sides_asked": 2, "unlined": unlined(0, 0),
        "left_out": {"too_short": 0, "faded": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_render_project_asks_for_the_sound_check(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wide_asset = import_asset(str(media / "wide.mp4"))
    project = build_project([video_track(), insert("a", wide_asset["id"], 0, 2),
                             insert("b", wide_asset["id"], 2, 4)], width=640, height=360)
    started = []
    monkeypatch.setattr(runtime.job_manager, "start_job", lambda job, spec: started.append(spec) or job)
    monkeypatch.setattr(rendering, "analyses_of", lambda project, listened=False: {})
    monkeypatch.setattr(rendering, "talking_of", lambda project, voices: None)
    monkeypatch.setattr(rendering, "check_findings", lambda project, burn: [])
    rendering.render_project(project)
    assert started[-1]["sync"] == {
        "sides": [
            {"at": 2.0, "clip_id": "a", "side": "before", "start": 0.0, "end": 2.0,
             "path": wide_asset["path"], "source_start": 0.0, "speed": 1.0},
            {"at": 2.0, "clip_id": "b", "side": "after", "start": 2.0, "end": 4.0,
             "path": wide_asset["path"], "source_start": 2.0, "speed": 1.0},
        ],
        "left_out": {"too_short": 0, "faded": 0},
        "frame": FRAME,
    }

def test_render_project_asks_for_the_sound_check_at_the_project_frame_rate(
        media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wide_asset = import_asset(str(media / "wide.mp4"))
    project = build_project([video_track(), insert("a", wide_asset["id"], 0, 2),
                             insert("b", wide_asset["id"], 2, 4)], width=640, height=360,
                            fps_num=25, fps_den=1)
    started = []
    monkeypatch.setattr(runtime.job_manager, "start_job", lambda job, spec: started.append(spec) or job)
    monkeypatch.setattr(rendering, "analyses_of", lambda project, listened=False: {})
    monkeypatch.setattr(rendering, "talking_of", lambda project, voices: None)
    monkeypatch.setattr(rendering, "check_findings", lambda project, burn: [])
    rendering.render_project(project)
    assert started[-1]["sync"]["frame"] == 1 / 25

def test_render_project_leaves_files_without_sound_out_of_the_sound_check(
        media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A file with no sound track makes none; one with a track of silence still sounds, and is offered.
    wide_asset = import_asset(str(media / "wide.mp4"))
    silent_asset = import_asset(str(media / "silent.mp4"))
    muted_asset = import_asset(str(media / "muted.mp4"))
    project = build_project([video_track(), insert("a", wide_asset["id"], 0, 2),
                             insert("s", silent_asset["id"], 0, 2),
                             insert("m", muted_asset["id"], 0, 2)], width=640, height=360)
    started = []
    monkeypatch.setattr(runtime.job_manager, "start_job", lambda job, spec: started.append(spec) or job)
    monkeypatch.setattr(rendering, "analyses_of", lambda project, listened=False: {})
    monkeypatch.setattr(rendering, "talking_of", lambda project, voices: None)
    monkeypatch.setattr(rendering, "check_findings", lambda project, burn: [])
    rendering.render_project(project)
    asked = started[-1]["sync"]
    assert [(side["clip_id"], side["side"]) for side in asked["sides"]] == [("a", "before"), ("m", "after")]
    assert asked["sides"][1]["path"] == muted_asset["path"]
    assert asked["left_out"] == {"too_short": 0, "faded": 0}

def whole_sound(path: Path) -> np.ndarray:
    """Decode a whole file's sound the plain way, with no seek, as the reference a stretch must match."""
    result = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                             "-ar", str(sync.SAMPLE_RATE), "-"], capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)

def test_a_stretch_cannot_start_before_the_file() -> None:
    with pytest.raises(ValueError):
        sync._read_stretch("unused.wav", -0.1, 1.0, FFMPEG)

@pytest.mark.parametrize("start", [0.0, 0.2, 1.5])
def test_a_stretch_read_from_near_the_start_of_an_aac_file_matches_the_whole_file(media: Path, start: float) -> None:
    # A seek to the very start of an AAC file drops one more frame of its encoder delay; the stretch must not.
    whole = whole_sound(media / "wide.mp4")
    stretch = sync._read_stretch(str(media / "wide.mp4"), start, 1.0, FFMPEG)
    first = int(round(start * sync.SAMPLE_RATE))
    assert np.allclose(stretch[:8000], whole[first:first + 8000], atol=1e-3)

def sync_spec(work_dir: Path) -> dict:
    """Build the job specification a sound-check test needs, with no pieces to render.

    Args:
        work_dir: The job's work folder.

    Returns:
        The specification.
    """
    return {
        "pieces": [],
        "duration_seconds": 5.0,
        "command": ["ffmpeg"],
        "render_path": str(work_dir / "out.mp4"),
        "sync": {
            "sides": [{"at": 2.0, "clip_id": "b", "side": "after", "start": 2.0, "end": 4.0,
                       "path": str(work_dir / "source.wav"), "source_start": 2.0, "speed": 1.0}],
            "left_out": {"too_short": 1, "faded": 2},
            "frame": FRAME,
        },
    }

def run_render(monkeypatch: pytest.MonkeyPatch, work_dir: Path, spec: dict) -> None:
    """Run a render past its sound check with FFmpeg stubbed out.

    Args:
        monkeypatch: The test's monkeypatch.
        work_dir: Where the render's files go.
        spec: The job's specification.
    """
    class Reported:
        def report(self, *_):
            pass

        def is_cancelled(self):
            return False

    monkeypatch.setattr(renderer, "run_ffmpeg", lambda *args: None)
    renderer._run_render(Job(work_dir=str(work_dir)), spec, Reported())

def test_a_sound_check_that_breaks_leaves_the_render_unchecked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Broke(Exception):
        pass

    def broke(*args, **kwargs):
        raise Broke("the check broke")

    monkeypatch.setattr(sync, "check", broke)
    run_render(monkeypatch, tmp_path, sync_spec(tmp_path))
    assert sync.kept(str(tmp_path)) is None

def test_a_render_runs_the_sound_check_the_spec_asked_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    found = {"streams": None, "joins": [], "sides_asked": 1, "unlined": unlined(0, 0),
             "left_out": {"too_short": 0, "faded": 0}, "frame": round(FRAME, 4)}
    asked = []
    monkeypatch.setattr(sync, "check", lambda *args: asked.append(args) or found)
    spec = sync_spec(tmp_path)
    run_render(monkeypatch, tmp_path, spec)
    assert asked == [(spec["render_path"], spec["sync"]["sides"], spec["sync"]["left_out"],
                      FRAME, "ffmpeg", "ffprobe")]
    assert sync.kept(str(tmp_path)) == found
