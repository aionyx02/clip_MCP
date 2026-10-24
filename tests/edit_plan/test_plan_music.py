"""Tests for the music under a cut: where each cue plays, loops, cross-fades and markers."""

import pytest

from app.engine.plan.beats import plan_pieces
from app.engine.plan.music import music_beds, plan_markers
from app.engine.plan.pieces import compiled_duration
from app.models.plan import Beat, MusicCue, MusicPlan, Selection

from edit_plan.shared import SONG, beats_plan, cut_points, footage, parts, sourced, speech_of


def test_each_beat_becomes_a_marker_where_it_starts() -> None:
    """A plan's shape used to exist only in the plan, and be gone the moment it compiled."""
    by_id, children, assets, clips = footage()
    first, second, third = speech_of(clips)[:3]
    plan = beats_plan(
        [
            Selection(clip_id=first.id, beat_id="open"),
            Selection(clip_id=third.id, beat_id="middle"),
        ],
        [Beat(id="open", name="開場"), Beat(id="middle", name="主體")],
    )
    pieces = plan_pieces(plan, by_id, children, assets)
    markers = plan_markers(plan, pieces)
    assert [marker["name"] for marker in markers] == ["開場", "主體"]
    assert markers[0]["timeline_in"] == 0.0
    # The second part starts where the first one's footage ends.
    assert markers[1]["timeline_in"] == pytest.approx(pieces[0].duration, abs=0.001)


def test_a_beat_nothing_was_chosen_for_gets_no_marker() -> None:
    """A part of the plan with no footage in it is not a part of the video."""
    by_id, children, assets, clips = footage()
    plan = beats_plan(
        [Selection(clip_id=speech_of(clips)[0].id, beat_id="open")],
        [Beat(id="open", name="開場"), Beat(id="empty", name="沒拍到")],
    )
    markers = plan_markers(plan, plan_pieces(plan, by_id, children, assets))
    assert [marker["id"] for marker in markers] == ["open"]

def test_each_cue_plays_from_where_its_beat_begins_until_the_next() -> None:
    plan, by_id, assets = parts()
    plan = plan.model_copy(update={"music": MusicPlan(cues=[
        MusicCue(asset_id=SONG, fade_in=0.5, fade_out=0.5),
        MusicCue(beat_id="b2"),
        MusicCue(beat_id="b3", asset_id="other", start=10.0),
    ])})
    pieces = plan_pieces(plan, by_id, {}, assets)
    starts = [0.0, *cut_points(pieces)]
    beds = music_beds(plan, pieces, assets)
    # The middle part is silence: a gap on the track, not a clip of nothing.
    assert [(bed.asset_id, bed.timeline_in) for bed in beds] == [(SONG, starts[0]), ("other", starts[2])]
    assert beds[0].end - beds[0].start == pytest.approx(starts[1])
    assert beds[1].start == 10.0
    assert beds[1].timeline_in + beds[1].end - beds[1].start == pytest.approx(compiled_duration(pieces))
    assert (beds[0].fade_in, beds[0].fade_out) == (0.5, 0.5)

def test_a_song_shorter_than_its_part_loops_back_to_where_it_came_in() -> None:
    plan, by_id, assets = parts()
    assets = {**assets, "jingle": sourced("jingle", seconds=1.0).model_copy(update={"has_video": False})}
    plan = plan.model_copy(update={"music": MusicPlan(cues=[MusicCue(asset_id="jingle", start=0.2)])})
    pieces = plan_pieces(plan, by_id, {}, assets)
    beds = music_beds(plan, pieces, assets)
    assert len(beds) > 2
    assert all(bed.start == 0.2 for bed in beds)
    assert all(earlier.timeline_in + earlier.end - earlier.start == pytest.approx(later.timeline_in)
               for earlier, later in zip(beds, beds[1:]))
    assert beds[-1].timeline_in + beds[-1].end - beds[-1].start == pytest.approx(compiled_duration(pieces))
    # The fades are the cue's, not each loop's.
    assert beds[0].fade_in > 0 and all(bed.fade_in == 0 for bed in beds[1:])
    assert beds[-1].fade_out > 0 and all(bed.fade_out == 0 for bed in beds[:-1])

def test_two_songs_cross_fade_on_two_tracks_when_the_plan_asks() -> None:
    plan, by_id, assets = parts()
    plan = plan.model_copy(update={"music": MusicPlan(crossfade_seconds=1.0, cues=[
        MusicCue(asset_id=SONG), MusicCue(beat_id="b3", asset_id="other", start=10.0),
    ])})
    pieces = plan_pieces(plan, by_id, {}, assets)
    cut = cut_points(pieces)[1]
    first, second = music_beds(plan, pieces, assets)
    # The next song comes in a second early on the other track, and a second earlier in
    # itself, so at the cut it is at exactly the 10s the cue asked for.
    assert (first.track_id, second.track_id) == ("music", "music_b")
    assert second.timeline_in == pytest.approx(cut - 1.0)
    assert second.start == pytest.approx(9.0)
    assert first.timeline_in + first.end - first.start == pytest.approx(cut)
    assert first.fade_out == 1.0 and second.fade_in == 1.0

def test_a_song_coming_in_from_its_top_cross_fades_over_the_end_of_the_one_going_out() -> None:
    # It has nothing before its first second to come in early with, and the cross-fade used to
    # be dropped without a word. The song going out runs on past the cut instead.
    plan, by_id, assets = parts()
    plan = plan.model_copy(update={"music": MusicPlan(crossfade_seconds=1.0, cues=[
        MusicCue(asset_id=SONG), MusicCue(beat_id="b3", asset_id="other"),
    ])})
    pieces = plan_pieces(plan, by_id, {}, assets)
    cut = cut_points(pieces)[1]
    first, second = music_beds(plan, pieces, assets)
    assert (first.track_id, second.track_id) == ("music", "music_b")
    # The one coming in is exactly where the cue said at the cut: the top of the song.
    assert second.timeline_in == pytest.approx(cut) and second.start == 0.0
    assert first.timeline_in + first.end - first.start == pytest.approx(cut + 1.0)
    # The last stretch is short, so its fades share it in proportion; both are still there.
    assert first.fade_out == 1.0 and second.fade_in > 0.5

def test_without_a_cross_fade_the_songs_meet_end_to_end_on_one_track() -> None:
    plan, by_id, assets = parts()
    plan = plan.model_copy(update={"music": MusicPlan(cues=[
        MusicCue(asset_id=SONG), MusicCue(beat_id="b3", asset_id="other"),
    ])})
    beds = music_beds(plan, plan_pieces(plan, by_id, {}, assets), assets)
    assert {bed.track_id for bed in beds} == {"music"}
