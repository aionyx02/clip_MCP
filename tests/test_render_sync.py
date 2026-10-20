"""A finished render's sound, checked against where the cut says the talking is."""

import json
import os
import subprocess
import types
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

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

def test_a_clip_at_double_speed_has_its_sound_squeezed_with_it() -> None:
    clip = make_clip(source_start=2, source_end=6, timeline_in=10, speed=2.0)
    assert sync.expected_voicing([clip], {"asset": [(2.0, 3.0)]}) == [(10.0, 10.5)]

def test_a_lead_puts_the_sound_before_the_picture_and_takes_it_from_earlier() -> None:
    clip = make_clip(source_start=2, source_end=4, timeline_in=10, audio_lead=0.5)
    assert sync.expected_voicing([clip], {"asset": [(2.0, 3.0)]}) == [(10.0, 11.0)]

def test_a_muted_clip_contributes_no_voicing() -> None:
    clip = make_clip(source_start=0, source_end=2, timeline_in=0, volume=0)
    assert sync.expected_voicing([clip], {"asset": [(0.0, 2.0)]}) == []

def test_a_title_card_contributes_no_voicing() -> None:
    clip = make_clip(source_start=0, source_end=2, timeline_in=0, card=TitleCard(title="開場"))
    assert sync.expected_voicing([clip], {"asset": [(0.0, 2.0)]}) == []

def test_a_voiced_stretch_reaching_past_the_clip_is_cut_to_it() -> None:
    clip = make_clip(source_start=2, source_end=4, timeline_in=10)
    assert sync.expected_voicing([clip], {"asset": [(1.0, 3.0)]}) == [(10.0, 11.0)]

def test_touching_voicing_from_two_clips_is_merged() -> None:
    first = make_clip("a", "one", source_start=0, source_end=2, timeline_in=0)
    second = make_clip("b", "two", source_start=0, source_end=2, timeline_in=2)
    assert sync.expected_voicing([first, second], {"one": [(0.0, 2.0)], "two": [(0.0, 2.0)]}) == [(0.0, 4.0)]

def test_a_join_next_to_a_muted_clip_skips_only_that_side() -> None:
    first = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    muted = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, volume=0)
    last = make_clip("c", "asset", source_start=4, source_end=6, timeline_in=4)
    offered, left_out = sync.sides(video_tracks(first, muted, last), [first, muted, last],
                                   {"asset": [(0.0, 6.0)]}, {"asset"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before"), ("c", "after")]
    assert left_out == {"not_listened": 0, "too_short": 0}

def test_a_join_into_a_title_card_skips_the_card_side() -> None:
    first = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    card = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, card=TitleCard(title="開場"))
    offered, left_out = sync.sides(video_tracks(first, card), [first, card], {"asset": [(0.0, 4.0)]},
                                   {"asset"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before")]
    assert left_out == {"not_listened": 0, "too_short": 0}

def test_a_file_nobody_listened_to_for_voicing_is_counted_and_its_neighbour_still_offered() -> None:
    # Its talking would be silence to the cut; that is said, not hidden as a missing join.
    first = make_clip("a", "heard", source_start=0, source_end=2, timeline_in=0)
    second = make_clip("b", "heard", source_start=2, source_end=4, timeline_in=2)
    third = make_clip("c", "unheard", source_start=0, source_end=2, timeline_in=4.6)
    fourth = make_clip("d", "heard", source_start=4, source_end=10, timeline_in=6.6)
    offered, left_out = sync.sides(video_tracks(first, second, third, fourth), [first, second, third, fourth],
                                   {"heard": [(0.0, 10.0)]}, {"heard", "unheard"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [
        ("a", "before"), ("b", "after"), ("b", "before"), ("d", "after")]
    assert left_out == {"not_listened": 2, "too_short": 0}

def test_an_unlistened_clip_is_counted_even_when_it_starts_just_after_the_join() -> None:
    first = make_clip("a", "heard", source_start=0, source_end=2, timeline_in=0)
    second = make_clip("b", "heard", source_start=0, source_end=1, timeline_in=2)
    later = make_clip("c", "unheard", source_start=0, source_end=2, timeline_in=4.3)
    offered, left_out = sync.sides(video_tracks(first, second, later), [first, second, later],
                                   {"heard": [(0.0, 10.0)]}, {"heard", "unheard"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [
        ("a", "before"), ("b", "after"), ("b", "before")]
    assert left_out == {"not_listened": 1, "too_short": 0}

def test_another_picture_tracks_sound_trims_the_side_it_runs_over() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=4, timeline_in=4)
    overlay = make_clip("c", "asset", source_start=0, source_end=1, timeline_in=4.2)
    offered, left_out = sync.sides(video_tracks(before, after, above=[overlay]), [before, after],
                                   {"asset": [(0.0, 4.0)]}, {"asset"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before")]
    assert left_out == {"not_listened": 0, "too_short": 1}

def test_a_muted_picture_track_leaves_the_side_alone() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=4, timeline_in=4)
    overlay = make_clip("c", "asset", source_start=0, source_end=1, timeline_in=4.2, volume=0)
    offered, left_out = sync.sides(video_tracks(before, after, above=[overlay]), [before, after],
                                   {"asset": [(0.0, 4.0)]}, {"asset"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before"), ("b", "after")]
    assert left_out == {"not_listened": 0, "too_short": 0}

def test_sound_from_an_audio_track_over_the_join_leaves_both_sides_out() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=2, timeline_in=2)
    music = make_clip("m", "asset", source_start=0, source_end=4, timeline_in=0)
    offered, left_out = sync.sides(video_tracks(before, after) + audio_tracks(music), [before, after],
                                   {"asset": [(0.0, 4.0)]}, {"asset"})
    assert offered == []
    assert left_out == {"not_listened": 0, "too_short": 2}

def test_a_clip_too_short_to_line_up_is_counted_on_its_side_of_the_join() -> None:
    # Nothing else plays before the join; the first clip's own sound is only 0.3s of it.
    short = make_clip("a", "asset", source_start=0, source_end=0.3, timeline_in=0)
    last = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=0.3)
    offered, left_out = sync.sides(video_tracks(short, last), [short, last], {"asset": [(0.0, 4.0)]},
                                   {"asset"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("b", "after")]
    assert left_out == {"not_listened": 0, "too_short": 1}

def test_a_fade_out_takes_the_end_of_the_before_stretch_off() -> None:
    # Inside the fade the detector hears the voice end before the sound does.
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0, audio_fade_out=0.5)
    after = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": [(0.0, 4.0)]},
                                   {"asset"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert before_side["end"] == 1.5
    assert left_out == {"not_listened": 0, "too_short": 0}

def test_a_fade_out_leaving_too_little_stretch_is_counted_too_short() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0, audio_fade_out=1.6)
    after = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": [(0.0, 4.0)]},
                                   {"asset"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("b", "after")]
    assert left_out == {"not_listened": 0, "too_short": 1}

def test_a_fade_in_takes_the_start_of_the_after_stretch_off() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    after = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, audio_fade_in=0.5)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": [(0.0, 4.0)]},
                                   {"asset"})
    after_side = next(side for side in offered if side["clip_id"] == "b")
    assert after_side["start"] == 2.5
    assert left_out == {"not_listened": 0, "too_short": 0}

def test_a_fade_in_leaving_too_little_stretch_is_counted_too_short() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    after = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, audio_fade_in=1.6)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": [(0.0, 4.0)]},
                                   {"asset"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before")]
    assert left_out == {"not_listened": 0, "too_short": 1}

def test_a_picture_track_without_sound_does_not_trim_the_side_it_runs_over() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=4, timeline_in=4)
    overlay = make_clip("c", "silent", source_start=0, source_end=1, timeline_in=4.2)
    offered, left_out = sync.sides(video_tracks(before, after, above=[overlay]), [before, after],
                                   {"asset": [(0.0, 4.0)], "silent": [(0.0, 4.0)]}, {"asset"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before"), ("b", "after")]
    assert left_out == {"not_listened": 0, "too_short": 0}

def test_a_base_clip_without_sound_is_skipped_and_not_counted() -> None:
    heard = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    silent = make_clip("b", "silent", source_start=0, source_end=2, timeline_in=2)
    offered, left_out = sync.sides(video_tracks(heard, silent), [heard, silent],
                                   {"asset": [(0.0, 2.0)], "silent": [(0.0, 2.0)]}, {"asset"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before")]
    assert left_out == {"not_listened": 0, "too_short": 0}

def test_a_title_card_on_another_track_is_not_other_sound() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=4, timeline_in=4)
    card = make_clip("c", "asset", source_start=0, source_end=1, timeline_in=4.2,
                     card=TitleCard(title="開場"))
    offered, left_out = sync.sides(video_tracks(before, after, above=[card]), [before, after],
                                   {"asset": [(0.0, 4.0)]}, {"asset"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before"), ("b", "after")]
    assert left_out == {"not_listened": 0, "too_short": 0}

def test_other_sound_that_ends_before_the_join_still_trims_the_before_side() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=4, timeline_in=0)
    after = make_clip("b", "asset", source_start=4, source_end=8, timeline_in=4)
    overlay = make_clip("c", "asset", source_start=0, source_end=0.8, timeline_in=2.2)
    offered, _ = sync.sides(video_tracks(before, after, above=[overlay]), [before, after],
                            {"asset": [(0.0, 8.0)]}, {"asset"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert (before_side["start"], before_side["end"]) == (3.0, 4.0)

def test_a_long_side_is_compared_over_exactly_the_join_window() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=6, timeline_in=0)
    after = make_clip("b", "asset", source_start=6, source_end=8, timeline_in=6)
    offered, _ = sync.sides(video_tracks(before, after), [before, after], {"asset": [(0.0, 8.0)]}, {"asset"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert abs((before_side["end"] - before_side["start"]) - sync.JOIN_WINDOW_SECONDS) <= 1e-9

def test_base_clips_given_out_of_order_give_the_same_sides() -> None:
    first = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    second = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2)
    in_order, counts_in_order = sync.sides(video_tracks(first, second), [first, second],
                                           {"asset": [(0.0, 4.0)]}, {"asset"})
    out_of_order, counts_out_of_order = sync.sides(video_tracks(second, first), [second, first],
                                                   {"asset": [(0.0, 4.0)]}, {"asset"})
    assert out_of_order == in_order
    assert counts_out_of_order == counts_in_order

def test_a_clip_nobody_listened_to_is_counted_and_the_other_side_is_offered() -> None:
    heard = make_clip("a", "heard", source_start=0, source_end=2, timeline_in=0)
    unheard = make_clip("b", "unheard", source_start=0, source_end=2, timeline_in=2)
    offered, left_out = sync.sides(video_tracks(heard, unheard), [heard, unheard], {"heard": [(0.0, 2.0)]},
                                   {"heard", "unheard"})
    assert [(side["clip_id"], side["side"]) for side in offered] == [("a", "before")]
    assert left_out == {"not_listened": 1, "too_short": 0}

def test_a_lag_on_the_clip_before_gives_the_after_side_the_windows_rest() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0, audio_lag=1.0)
    after = make_clip("b", "asset", source_start=0, source_end=2, timeline_in=2)
    offered, _ = sync.sides(video_tracks(before, after), [before, after], {"asset": [(0.0, 3.0)]}, {"asset"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    after_side = next(side for side in offered if side["clip_id"] == "b")
    # The lagged tail is part of the before clip's voicing, and its sound covers the start of the after side.
    assert before_side["expected"] == [[0.0, 3.0]]
    assert after_side["start"] == 3.0

def test_a_lead_on_the_clip_after_ends_the_before_side_where_its_sound_starts() -> None:
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    after = make_clip("b", "asset", source_start=0, source_end=2, timeline_in=2, audio_lead=1.0)
    offered, _ = sync.sides(video_tracks(before, after), [before, after], {"asset": [(0.0, 3.0)]}, {"asset"})
    before_side = next(side for side in offered if side["clip_id"] == "a")
    assert before_side["end"] == 1.0

def test_identical_voicing_has_no_lag() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(1.0, 2.0)], 0.0, 4.0) == 0.0

def test_sound_heard_later_than_the_cut_says_has_a_positive_lag() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(1.12, 2.12)], 0.0, 4.0) == 0.12

def test_sound_heard_sooner_than_the_cut_says_has_a_negative_lag() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(0.88, 1.88)], 0.0, 4.0) == -0.12

def test_a_window_with_nothing_changing_has_nothing_to_line_up() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(0.0, 4.0)], 6.0, 10.0) is None

def test_a_window_whose_voicing_never_changes_has_no_lag() -> None:
    # The voice runs past both ends of the window, so there is no edge in it to line up.
    assert sync.lag_at([(0.0, 4.1)], [(0.0, 4.1)], 0.0, 4.0) is None

def test_unrelated_voicing_does_not_agree_enough_to_be_believed() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(3.0, 4.0)], 0.0, 4.0) is None

def test_a_short_sound_beside_a_short_word_does_not_give_a_lag() -> None:
    assert sync.lag_at([(1.0, 1.5)], [(2.0, 2.4)], 0.0, 4.0) is None

def test_a_lag_past_the_edge_of_the_search_is_not_reported() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(1.6, 2.6)], 0.0, 4.0) is None

def test_agreeing_equally_well_at_several_lags_takes_the_smallest_move() -> None:
    assert sync.lag_at([(0.0, 3.5)], [(-1.0, 10.0)], 0.0, 4.0) == 0.0

def test_agreeing_equally_either_way_at_one_size_takes_the_negative_lag() -> None:
    # Each shift covers the same stretch of the short blocks either side of the join.
    expected = [(0.0, 0.2), (0.4, 3.6), (3.8, 4.0)]
    measured = [(0.2, 3.8)]
    assert sync.lag_at(expected, measured, 0.0, 4.0) == -0.2

def test_agreement_is_intersection_over_union_not_share_of_the_expected_voice() -> None:
    # Recall of the short word alone would be 1.0, but the measured sound runs on
    # into the quiet around it, so the union is far larger than the agreement.
    assert sync.lag_at([(1.0, 1.6)], [(0.98, 2.2)], 0.0, 4.0) is None

def test_a_stretch_just_past_the_window_is_still_searched() -> None:
    # The voice is heard 0.4s late: the measured sound starts after the window
    # ends and must be kept by the prefilter to be found.
    assert sync.lag_at([(3.7, 4.0)], [(4.1, 4.4)], 0.0, 4.0) == 0.4

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
        patched.setattr(sync, "read_samples", refuse)
        found = sync.check(str(path), [], {}, FRAME, FFMPEG, "ffprobe")
    assert found["streams"] is not None
    assert found["sides_asked"] == 0
    assert found["left_out"] == {}
    assert found["frame"] == round(FRAME, 4)

def test_check_does_not_decode_when_there_are_no_sides(tmp_path: Path) -> None:
    path = plain_render(tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("the sound was decoded though no side was asked about")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(sync, "read_samples", refuse)
        found = sync.check(str(path), [], {"not_listened": 1, "too_short": 0}, FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == []
    assert found["sides_asked"] == 0
    assert found["left_out"] == {"not_listened": 1, "too_short": 0}

def test_check_lines_the_heard_voice_up_with_the_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = plain_render(tmp_path)
    side = {"at": 2.0, "clip_id": "a", "side": "before", "start": 0.0, "end": 2.0, "expected": [[1.0, 2.0]]}
    monkeypatch.setattr(sync, "voiced_stretches", lambda samples: [(1.12, 2.12)])
    found = sync.check(str(path), [side], {"not_listened": 0, "too_short": 0}, FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == [{"at": 2.0, "clip_id": "a", "side": "before", "lag": 0.12}]
    assert found["sides_asked"] == 1

def test_check_lines_up_each_side_of_a_join_on_its_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = plain_render(tmp_path)
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=2.0)
    after = make_clip("b", "asset", source_start=0, source_end=2, timeline_in=4.0)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": [(0.5, 1.5)]},
                                   {"asset"})
    monkeypatch.setattr(sync, "read_samples", lambda *args: b"")
    monkeypatch.setattr(sync, "voiced_stretches", lambda samples: [(2.5, 3.5), (4.6, 5.6)])
    found = sync.check(str(path), offered, left_out, FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == [
        {"at": 4.0, "clip_id": "a", "side": "before", "lag": 0.0},
        {"at": 4.0, "clip_id": "b", "side": "after", "lag": 0.1},
    ]
    assert found["sides_asked"] == 2

def test_an_l_cut_side_with_no_edge_of_its_own_voicing_is_not_compared(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = plain_render(tmp_path)
    before = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0, audio_lag=0.5)
    after = make_clip("b", "asset", source_start=0, source_end=4, timeline_in=2)
    offered, left_out = sync.sides(video_tracks(before, after), [before, after], {"asset": [(0.0, 10.0)]},
                                   {"asset"})
    monkeypatch.setattr(sync, "read_samples", lambda *args: b"")
    monkeypatch.setattr(sync, "voiced_stretches", lambda samples: [(0.0, 10.0)])
    found = sync.check(str(path), offered, left_out, FRAME, FFMPEG, "ffprobe")
    assert found["sides_asked"] == 2
    assert found["joins"] == []

def test_check_reports_how_many_sides_were_asked_and_what_was_left_out(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = plain_render(tmp_path)
    offered = [
        {"at": 2.0, "clip_id": "a", "side": "before", "start": 0.0, "end": 2.0, "expected": [[1.0, 2.0]]},
        {"at": 6.0, "clip_id": "b", "side": "before", "start": 4.0, "end": 6.0, "expected": [[5.0, 6.0]]},
        {"at": 10.0, "clip_id": "c", "side": "before", "start": 8.0, "end": 10.0, "expected": [[9.0, 10.0]]},
    ]
    monkeypatch.setattr(sync, "read_samples", lambda *args: b"")
    monkeypatch.setattr(sync, "voiced_stretches", lambda samples: [(1.0, 2.0)])
    left_out = {"not_listened": 1, "too_short": 0}
    found = sync.check(str(path), offered, left_out, FRAME, FFMPEG, "ffprobe")
    assert found["sides_asked"] == 3
    assert found["left_out"] == left_out
    assert found["joins"] == [{"at": 2.0, "clip_id": "a", "side": "before", "lag": 0.0}]

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

def test_get_job_names_the_clip_and_the_side_whose_voice_is_heard_late(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.12}],
        "sides_asked": 1, "left_out": {"not_listened": 0, "too_short": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "out_of_step": ["at 2.00s, clip b, after the join: the voice is heard 0.12s late"]
    }

def test_get_job_says_nothing_about_sound_within_a_frame(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.02}],
        "sides_asked": 1, "left_out": {"not_listened": 0, "too_short": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_get_job_says_where_each_stream_starts_and_ends(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    streams = {"video_start": 0.0, "audio_start": 0.3, "video_length": 3.0, "audio_length": 2.7}
    job = make_finished_job(project, {"streams": streams, "joins": [], "sides_asked": 0,
                                      "left_out": {"not_listened": 0, "too_short": 0}, "frame": FRAME})
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {"out_of_step": ["the sound runs 0.30s to 3.00s while the picture runs 0.00s to 3.00s"]}

def test_get_job_ignores_a_lag_within_the_edge_slack_at_a_high_frame_rate(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.03}],
        "sides_asked": 1, "left_out": {"not_listened": 0, "too_short": 0}, "frame": 1 / 60,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_get_job_ignores_a_lag_within_two_steps_of_the_voice_detector(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.06}],
        "sides_asked": 1, "left_out": {"not_listened": 0, "too_short": 0}, "frame": 1 / 30,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_get_job_reports_a_lag_beyond_two_steps_of_the_voice_detector(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.07}],
        "sides_asked": 1, "left_out": {"not_listened": 0, "too_short": 0}, "frame": 1 / 30,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "out_of_step": ["at 2.00s, clip b, after the join: the voice is heard 0.07s late"]
    }

def test_get_job_keeps_a_lag_under_one_frame_at_a_slower_frame_rate(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.08}],
        "sides_asked": 1, "left_out": {"not_listened": 0, "too_short": 0}, "frame": 1 / 10,
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

def test_get_job_says_when_no_side_could_be_compared(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None, "joins": [], "sides_asked": 3,
        "left_out": {"not_listened": 0, "too_short": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_compared": "3 of the 3 sides of joins could not be lined up — 3 where the voice does not start or "
                        "stop near the join or did not agree well enough — so whether they are in step is not known"
    }

def test_get_job_lists_only_the_reasons_that_left_sides_out(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None,
        "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.0}],
        "sides_asked": 3, "left_out": {"not_listened": 0, "too_short": 2}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_compared": "4 of the 5 sides of joins could not be lined up — "
                        "2 with less than 0.5s of their own sound clear of other sound, "
                        "2 where the voice does not start or stop near the join or did not agree well enough — "
                        "so whether they are in step is not known"
    }

def test_get_job_says_what_was_not_compared_when_some_sides_were_lined_up(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None,
        "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.0},
                  {"at": 4.0, "clip_id": "c", "side": "after", "lag": 0.0}],
        "sides_asked": 3, "left_out": {"not_listened": 0, "too_short": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_compared": "1 of the 3 sides of joins could not be lined up — 1 where the voice does not start or "
                        "stop near the join or did not agree well enough — so whether they are in step is not known"
    }

def test_get_job_says_nothing_when_every_side_was_in_step(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {
        "streams": None,
        "joins": [{"at": 2.0, "clip_id": "b", "side": "after", "lag": 0.0},
                  {"at": 4.0, "clip_id": "c", "side": "after", "lag": 0.0}],
        "sides_asked": 2, "left_out": {"not_listened": 0, "too_short": 0}, "frame": FRAME,
    })
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_render_project_asks_for_the_sound_check(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    started = []
    voices = {wide: types.SimpleNamespace(transcript=None, second_listen=types.SimpleNamespace(
        voiced=[types.SimpleNamespace(start=0.0, end=2.0)]))}
    monkeypatch.setattr(runtime.job_manager, "start_job", lambda job, spec: started.append(spec) or job)
    monkeypatch.setattr(rendering, "analyses_of", lambda project, listened=False: voices)
    monkeypatch.setattr(rendering, "talking_of", lambda project, voices: None)
    monkeypatch.setattr(rendering, "check_findings", lambda project, burn: [])
    rendering.render_project(project)
    assert started[-1]["sync"] == {
        "sides": [
            {"at": 2.0, "clip_id": "a", "side": "before", "start": 0.0, "end": 2.0,
             "expected": [[0.0, 2.0]]},
            {"at": 2.0, "clip_id": "b", "side": "after", "start": 2.0, "end": 4.0, "expected": []},
        ],
        "left_out": {"not_listened": 0, "too_short": 0},
        "frame": FRAME,
    }

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
                       "expected": [[2.0, 4.0]]}],
            "left_out": {"not_listened": 0, "too_short": 0},
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
    found = {"streams": None, "joins": [], "sides_asked": 1,
             "left_out": {"not_listened": 0, "too_short": 0}, "frame": round(FRAME, 4)}
    asked = []
    monkeypatch.setattr(sync, "check", lambda *args: asked.append(args) or found)
    spec = sync_spec(tmp_path)
    run_render(monkeypatch, tmp_path, spec)
    assert asked == [(spec["render_path"], spec["sync"]["sides"], spec["sync"]["left_out"],
                      FRAME, "ffmpeg", "ffprobe")]
    assert sync.kept(str(tmp_path)) == found
