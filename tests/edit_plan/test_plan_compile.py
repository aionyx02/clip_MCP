"""Tests for turning a plan into the edit operations that build its cut."""

from decimal import Decimal
from typing import List

import pytest
from pydantic import TypeAdapter

from app.engine.plan.beats import plan_pieces
from app.engine.plan.compile import compile_operations
from app.engine.plan.pieces import compiled_duration
from app.engine.semantic import clean_cuts
from app.models.plan import Beat, BrollShot, EditPlan, MusicCue, MusicPlan, Selection
from app.models.timeline.parts import Marker, Project
from test_semantic_timeline import ASSET_ID, analysis

from edit_plan.shared import (
    BROLL_ASSET, FOOTAGE, SONG, beats_plan, covered, footage, music_by_hand, parts, plan_for, sourced, speech_of,
)


def plan_duration(plan: EditPlan, by_id: dict, children: dict, assets: dict, cuts=None) -> float:
    """Measure how long a plan compiles to.

    Args:
        plan: The plan.
        by_id: The clips.
        children: Section members.
        assets: The assets.
        cuts: What is known about each file, if anything.

    Returns:
        The compiled length in seconds.
    """
    return compiled_duration(plan_pieces(plan, by_id, children, assets, cuts))

def test_the_operations_build_one_video_track_in_order() -> None:
    by_id, children, assets, clips = footage()
    first, third = speech_of(clips)[0], speech_of(clips)[2]
    plan = plan_for([
        Selection(clip_id=first.id, beat_id="b1"),
        Selection(clip_id=third.id, beat_id="b1"),
    ])
    operations, provenance = compile_operations(plan, by_id, children, assets)
    assert operations[0]["action"] == "add_track"
    assert [origin["from_plan_id"] for origin in provenance.values()] == [plan.id, plan.id]
    assert [operation["action"] for operation in operations[1:]] == [
        "insert_clip", "insert_clip", "set_markers",
    ]
    assert operations[1]["source_range"]["start"] < operations[2]["source_range"]["start"]
    # Both selections are the one beat, so the timeline gets one marker, at its start.
    assert operations[3]["markers"] == [
        {"id": "b1", "name": "主體", "timeline_in": 0.0, "from_beat_id": "b1"},
    ]

def test_music_is_laid_under_the_picture_and_fitted_to_it() -> None:
    by_id, children, assets, clips = footage()
    assets = {**assets, "song": sourced("song", seconds=60.0).model_copy(update={"has_video": False})}
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")],
                    music=MusicPlan(cues=[MusicCue(asset_id="song", volume=0.4)]))
    operations, provenance = compile_operations(plan, by_id, children, assets)
    assert [operation["action"] for operation in operations[-2:]] == ["add_track", "add_clip"]
    laid = operations[-1]
    assert laid["track_id"] == "music" and laid["volume"] == 0.4 and laid["timeline_in"] == 0.0
    assert laid["source_range"]["end"] == pytest.approx(plan_duration(plan, by_id, children, assets))
    assert provenance[laid["clip_id"]]["from_plan_id"] == plan.id

def test_the_same_plan_always_compiles_to_the_same_cut() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=clip.id, beat_id="b1") for clip in speech_of(clips)])
    assert compile_operations(plan, by_id, children, assets) == compile_operations(plan, by_id, children, assets)

def test_a_marker_somebody_added_survives_the_next_compile() -> None:
    """The compiler owns what it made. A marker with no beat behind it is not that."""
    by_id, children, assets, clips = footage()
    plan = beats_plan(
        [Selection(clip_id=speech_of(clips)[0].id, beat_id="open")],
        [Beat(id="open", name="開場")],
    )
    project = Project(id="p", width=1920, height=1080, markers=[
        Marker(id="mine", name="這裡要配樂", timeline_in=Decimal("1.5")),
        Marker(id="open", name="舊的開場", timeline_in=Decimal(0), from_beat_id="open"),
    ])
    operations, _ = compile_operations(plan, by_id, children, assets, project)
    written = next(op for op in operations if op["action"] == "set_markers")["markers"]
    names = {marker["id"]: marker["name"] for marker in written}
    # The compiler's own marker is rebuilt from the plan; the hand-made one is carried over.
    assert names["open"] == "開場"
    assert names["mine"] == "這裡要配樂"

def test_compiling_covering_picture_that_does_not_hold_up_is_refused() -> None:
    """The engine stands on its own, so it cannot leave a shot out and say nothing."""
    by_id, children, assets, clips = covered()
    speech = speech_of(clips)
    quiet = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = plan_for(
        [Selection(clip_id=clip.id, beat_id="b1") for clip in speech],
        broll=[BrollShot(clip_id=quiet.id, over_clip_id=speech[0].id, seconds=0.4)],
    )
    with pytest.raises(ValueError, match="does not hold up"):
        compile_operations(plan, by_id, children, assets, None, {ASSET_ID: clean_cuts(analysis(**FOOTAGE))})

def test_a_pinned_clip_is_matched_on_its_own_track() -> None:
    """Three kinds of clip mean different things by `from_clip_ids`, so one namespace clashes.

    A sequence window merged from exactly the two clips a cover is made of would
    take that cover's hand-made settings, and the music bed — made of no
    semantic clips at all — would match anything else made of none.
    """
    by_id, children, assets, clips = covered()
    speech = speech_of(clips)
    quiet = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = plan_for(
        [Selection(clip_id=clip.id, beat_id="b1") for clip in speech],
        broll=[BrollShot(clip_id=quiet.id, over_clip_id=speech[0].id, seconds=2.0)],
    )
    operations, provenance = compile_operations(
        plan, by_id, children, assets, None, {ASSET_ID: clean_cuts(analysis(**FOOTAGE))},
    )
    laid = [op for op in operations if op.get("track_id") == "broll" and op["action"] == "add_clip"]
    assert len(laid) == 1
    assert provenance[laid[0]["clip_id"]]["from_clip_ids"] == [quiet.id, speech[0].id]

def test_the_cues_become_clips_on_the_music_track() -> None:
    plan, by_id, assets = parts()
    plan = plan.model_copy(update={"music": MusicPlan(duck_under_speech=False, cues=[
        MusicCue(asset_id=SONG, volume=0.3), MusicCue(beat_id="b3", asset_id="other", volume=0.5),
    ])})
    operations, provenance = compile_operations(plan, by_id, {}, assets)
    track = next(op for op in operations if op["action"] == "add_track" and op["track_id"] == "music")
    assert track["duck_under_speech"] is False
    laid = [op for op in operations if op.get("track_id") == "music" and op["action"] == "add_clip"]
    assert [(op["asset_id"], op["volume"]) for op in laid] == [(SONG, 0.3), ("other", 0.5)]
    assert all(provenance[op["clip_id"]]["from_plan_id"] == plan.id for op in laid)

def test_a_looped_song_goes_onto_the_track_without_a_hair_of_overlap() -> None:
    """A song whose length is not a whole number of milliseconds: rounded the wrong way,
    one loop runs past the end of the file, or into the next one, and the track refuses it."""
    from app.models.timeline.apply import apply_operation
    from app.models.timeline.operations import EditOperation
    from app.models.timeline.validate import validate_project

    plan, by_id, assets = parts()
    assets = {**assets, "jingle": sourced("jingle", seconds=1.3337).model_copy(update={"has_video": False})}
    plan = plan.model_copy(update={"music": MusicPlan(cues=[
        MusicCue(asset_id="jingle", start=0.333), MusicCue(beat_id="b3", asset_id="jingle", start=0.333),
    ])})
    operations, _ = compile_operations(plan, by_id, {}, assets)
    project = Project(id="looped", width=640, height=360)
    for operation in TypeAdapter(List[EditOperation]).validate_python(operations):
        apply_operation(project, operation, assets)
    validate_project(project, assets)
    music = next(track for track in project.tracks if track.id == "music").clips
    assert len(music) > 4
    assert all(clip.source_range.end <= Decimal("1.3337") for clip in music)
    assert max(clip.timeline_out for clip in music) == project.duration

def test_a_music_clip_from_before_stretches_had_names_is_taken_as_the_first() -> None:
    plan, by_id, assets = parts()
    plan = plan.model_copy(update={"music": MusicPlan(cues=[MusicCue(asset_id=SONG, volume=0.4)])})
    project, _ = music_by_hand(plan, by_id, assets, volume=0.2, from_clip_ids=[])
    operations, _ = compile_operations(plan, by_id, {}, assets, project)
    assert [op["volume"] for op in operations if op.get("track_id") == "music" and op["action"] == "add_clip"] == [0.2]
