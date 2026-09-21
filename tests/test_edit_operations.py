"""Tests for splitting a clip, reordering the sequence, and sound that runs outside its picture."""

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.server import import_asset, repo
from helpers import build_project, clips_of, edit, insert, layout, video_track

@pytest.fixture
def source(media: Path) -> str:
    """Import a ten-second video.

    Returns:
        The asset ID.
    """
    return import_asset(str(media / "wide.mp4"))["id"]

@pytest.fixture
def three_clips(source: str) -> str:
    """Build a project with clips A (3s), B (2s), and C (4s) in sequence.

    Returns:
        The project ID.
    """
    return build_project([
        video_track(),
        insert("a", source, 0, 3),
        insert("b", source, 4, 6),
        insert("c", source, 6, 10),
    ])

def test_split_at_a_timeline_time_divides_the_clip_in_place(three_clips: str) -> None:
    edit(three_clips, [{"action": "split_clip", "track_id": "main", "clip_id": "c", "new_clip_id": "c2", "at": 7}])
    assert layout(three_clips) == [
        ("a", 0.0, 3.0, 0.0, 3.0),
        ("b", 3.0, 5.0, 4.0, 6.0),
        ("c", 5.0, 7.0, 6.0, 8.0),
        ("c2", 7.0, 9.0, 8.0, 10.0),
    ]

def test_split_at_a_source_time_uses_the_time_transcripts_report(three_clips: str) -> None:
    edit(three_clips, [
        {"action": "split_clip", "track_id": "main", "clip_id": "c", "new_clip_id": "c2", "at_source": 8},
    ])
    assert layout(three_clips)[2:] == [("c", 5.0, 7.0, 6.0, 8.0), ("c2", 7.0, 9.0, 8.0, 10.0)]

def test_split_does_not_move_the_clips_around_it(three_clips: str) -> None:
    before = layout(three_clips)
    edit(three_clips, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at": 1}])
    after = layout(three_clips)
    assert [entry for entry in after if entry[0] in {"b", "c"}] == [entry for entry in before if entry[0] in {"b", "c"}]
    assert after[-1][2] == before[-1][2]

def test_split_copies_the_clips_settings_to_both_halves(source: str) -> None:
    project = build_project([video_track(), insert("a", source, 0, 6, volume=0.4)])
    edit(project, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at": 2}])
    halves = {clip["id"]: clip for clip in clips_of(project)}
    assert halves["a2"]["asset_id"] == halves["a"]["asset_id"] == source
    assert float(halves["a2"]["volume"]) == float(halves["a"]["volume"]) == 0.4

def test_split_moves_the_fade_out_to_the_second_half(source: str) -> None:
    project = build_project([video_track(), insert("a", source, 0, 8, audio_fade_in=1, audio_fade_out=2)])
    edit(project, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at": 4}])
    halves = {clip["id"]: clip for clip in clips_of(project)}
    assert float(halves["a"]["audio_fade_in"]) == 1 and float(halves["a"]["audio_fade_out"]) == 0
    assert float(halves["a2"]["audio_fade_in"]) == 0 and float(halves["a2"]["audio_fade_out"]) == 2

def test_split_shortens_a_fade_that_no_longer_fits_its_half(source: str) -> None:
    project = build_project([video_track(), insert("a", source, 0, 8, audio_fade_in=3, audio_fade_out=3)])
    edit(project, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at": 1}])
    halves = {clip["id"]: clip for clip in clips_of(project)}
    assert float(halves["a"]["audio_fade_in"]) == 1

@pytest.mark.parametrize("at", [3.0, 5.0, 9.0])
def test_split_outside_the_clip_is_refused_with_the_valid_range(three_clips: str, at: float) -> None:
    with pytest.raises(ValueError, match="must fall inside"):
        edit(three_clips, [{"action": "split_clip", "track_id": "main", "clip_id": "b", "new_clip_id": "b2", "at": at}])
    assert len(clips_of(three_clips)) == 3

def test_split_refuses_an_id_that_is_already_taken(three_clips: str) -> None:
    with pytest.raises(ValueError, match="already exists"):
        edit(three_clips, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "b", "at": 1}])

@pytest.mark.parametrize("point", [{}, {"at": 1, "at_source": 1}])
def test_split_needs_exactly_one_kind_of_cut_point(three_clips: str, point: dict) -> None:
    with pytest.raises(ValidationError, match="exactly one of at"):
        edit(three_clips, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "x", **point}])

def test_reorder_moves_a_clip_to_the_front_and_closes_the_space(three_clips: str) -> None:
    edit(three_clips, [{"action": "reorder_clip", "track_id": "main", "clip_id": "c", "before_clip_id": "a"}])
    assert layout(three_clips) == [
        ("c", 0.0, 4.0, 6.0, 10.0),
        ("a", 4.0, 7.0, 0.0, 3.0),
        ("b", 7.0, 9.0, 4.0, 6.0),
    ]

def test_reorder_moves_a_clip_to_the_end_when_no_target_is_given(three_clips: str) -> None:
    edit(three_clips, [{"action": "reorder_clip", "track_id": "main", "clip_id": "a"}])
    assert [entry[0] for entry in layout(three_clips)] == ["b", "c", "a"]
    assert layout(three_clips)[-1][:3] == ("a", 6.0, 9.0)

def test_reorder_keeps_the_cut_and_the_audio_settings(source: str) -> None:
    project = build_project([
        video_track(),
        insert("a", source, 0, 3),
        insert("b", source, 5, 7, volume=0.3, audio_fade_in=0.5, audio_fade_out=0.5),
    ])
    edit(project, [{"action": "reorder_clip", "track_id": "main", "clip_id": "b", "before_clip_id": "a"}])
    moved = {clip["id"]: clip for clip in clips_of(project)}["b"]
    assert (float(moved["source_range"]["start"]), float(moved["source_range"]["end"])) == (5.0, 7.0)
    assert float(moved["volume"]) == 0.3
    assert float(moved["audio_fade_in"]) == 0.5 and float(moved["audio_fade_out"]) == 0.5

def test_reorder_leaves_the_total_length_unchanged(three_clips: str) -> None:
    edit(three_clips, [{"action": "reorder_clip", "track_id": "main", "clip_id": "b", "before_clip_id": "a"}])
    assert layout(three_clips)[-1][2] == 9.0

def test_reorder_does_not_move_the_music(source: str, media: Path) -> None:
    song = import_asset(str(media / "song.mp3"))["id"]
    project = build_project([
        video_track(),
        insert("a", source, 0, 3),
        insert("b", source, 4, 6),
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        insert("m", song, 0, 5, track_id="music"),
    ])
    edit(project, [{"action": "reorder_clip", "track_id": "main", "clip_id": "b", "before_clip_id": "a"}])
    assert layout(project, "music") == [("m", 0.0, 5.0, 0.0, 5.0)]

def test_reorder_refuses_to_move_a_clip_before_itself(three_clips: str) -> None:
    with pytest.raises(ValueError, match="before itself"):
        edit(three_clips, [{"action": "reorder_clip", "track_id": "main", "clip_id": "a", "before_clip_id": "a"}])

def test_reorder_refuses_an_unknown_target(three_clips: str) -> None:
    with pytest.raises(ValueError, match="clip zz not found"):
        edit(three_clips, [{"action": "reorder_clip", "track_id": "main", "clip_id": "a", "before_clip_id": "zz"}])

def test_split_then_reorder_puts_the_ending_first(three_clips: str) -> None:
    # The flashback case: cut the last clip in two and lift its tail to the front.
    edit(three_clips, [
        {"action": "split_clip", "track_id": "main", "clip_id": "c", "new_clip_id": "ending", "at": 7},
        {"action": "reorder_clip", "track_id": "main", "clip_id": "ending", "before_clip_id": "a"},
    ])
    assert layout(three_clips) == [
        ("ending", 0.0, 2.0, 8.0, 10.0),
        ("a", 2.0, 5.0, 0.0, 3.0),
        ("b", 5.0, 7.0, 4.0, 6.0),
        ("c", 7.0, 9.0, 6.0, 8.0),
    ]

def test_split_keeps_the_look_on_both_halves(source: str) -> None:
    project = build_project([
        video_track(),
        insert("a", source, 0, 6, video_fade_in=1, video_fade_out=2, color={"saturation": 0}),
    ])
    edit(project, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at": 3}])
    halves = {clip["id"]: clip for clip in clips_of(project)}
    assert halves["a"]["color"]["saturation"] == halves["a2"]["color"]["saturation"] == 0
    # A fade belongs to the outer edge of the shot, not to the new cut in the middle of it.
    assert float(halves["a"]["video_fade_in"]) == 1 and float(halves["a"]["video_fade_out"]) == 0
    assert float(halves["a2"]["video_fade_in"]) == 0 and float(halves["a2"]["video_fade_out"]) == 2

def test_split_shortens_a_video_fade_that_no_longer_fits_its_half(source: str) -> None:
    project = build_project([video_track(), insert("a", source, 0, 8, video_fade_in=3, video_fade_out=3)])
    edit(project, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at": 1}])
    halves = {clip["id"]: clip for clip in clips_of(project)}
    assert float(halves["a"]["video_fade_in"]) == 1
    assert float(halves["a2"]["video_fade_out"]) == 3


# --- sound that runs outside its picture: the J and L cuts ---------------------------------

@pytest.fixture
def two_clips(media: Path) -> str:
    """Three clips of a ten-second file, laid end to end.

    Returns:
        The project ID. `a` takes the start of the file and sits at the start
        of the timeline, so it has room for neither a lead nor anything
        before it. `b` takes the middle, so it has room both ways. `c` takes
        the start of the file again but sits late on the timeline, which is
        the one case where a lead runs out of file before it runs out of
        timeline.
    """
    asset = import_asset(str(media / "wide.mp4"))["id"]
    return build_project([
        video_track(), insert("a", asset, 0, 4), insert("b", asset, 4, 8), insert("c", asset, 0, 2),
    ])


def set_audio(project: str, clip_id: str, **fields) -> None:
    """Change how far a clip's sound runs outside its picture.

    Args:
        project: Project to edit.
        clip_id: Clip to change.
        **fields: `audio_lead` or `audio_lag`.
    """
    edit(project, [{"action": "set_clip_audio", "track_id": "main", "clip_id": clip_id, **fields}])


def test_sound_may_overlap_where_picture_may_not(two_clips: str) -> None:
    """A J cut is two clips' sound overlapping. That is the whole point of it."""
    set_audio(two_clips, "b", audio_lead=1.5)
    clips = {clip.id: clip for clip in repo.get_project(two_clips).tracks[0].clips}
    # The pictures still butt up against each other at four seconds.
    assert clips["a"].timeline_out == clips["b"].timeline_in == 4
    # The sound does not: the second clip is already playing under the first.
    assert clips["b"].audio_timeline_in == Decimal("2.5")
    assert clips["b"].audio_timeline_in < clips["a"].timeline_out


def test_a_lead_that_reaches_before_the_file_is_refused(two_clips: str) -> None:
    """`c` sits late enough on the timeline, but its source starts at the start of the file."""
    with pytest.raises(ValueError, match="before the file starts"):
        set_audio(two_clips, "c", audio_lead=1.0)


def test_a_lag_that_reaches_past_the_end_of_the_file_is_refused(two_clips: str) -> None:
    """`wide.mp4` runs ten seconds, and the second clip already ends at eight."""
    with pytest.raises(ValueError, match="only 10"):
        set_audio(two_clips, "b", audio_lag=3.0)


def test_a_lead_that_starts_before_the_timeline_is_refused(two_clips: str) -> None:
    """Sound cannot begin before the video does."""
    with pytest.raises(ValueError, match="before the timeline begins"):
        set_audio(two_clips, "a", audio_lead=1.0)


def test_footage_with_no_sound_cannot_have_its_sound_run_on(media: Path) -> None:
    """Asking silently does nothing, which is worse than being told."""
    silent = import_asset(str(media / "silent.mp4"))["id"]
    project = build_project([video_track(), insert("s", silent, 0, 3)])
    with pytest.raises(ValueError, match="has no sound to run"):
        set_audio(project, "s", audio_lag=0.5)


def test_fades_are_measured_against_the_sound_not_the_picture(two_clips: str) -> None:
    """A lead and a lag make the sound longer than the clip, and a fade belongs to the sound."""
    set_audio(two_clips, "b", audio_lead=1.5, audio_fade_in=Decimal("5"))
    assert repo.get_project(two_clips).tracks[0].clips[1].audio_fade_in == 5
    with pytest.raises(ValueError, match="longer than the clip's sound"):
        set_audio(two_clips, "b", audio_fade_out=Decimal("2"))


def test_putting_the_sound_back_with_its_picture(two_clips: str) -> None:
    """Zero is how a J cut is undone."""
    set_audio(two_clips, "b", audio_lead=1.5)
    set_audio(two_clips, "b", audio_lead=Decimal(0))
    clip = repo.get_project(two_clips).tracks[0].clips[1]
    assert clip.audio_lead == 0
    assert clip.audio_timeline_in == clip.timeline_in
