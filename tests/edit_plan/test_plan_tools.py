"""Tests for the planning tools end to end: saving, amending, copying and compiling a plan."""

from pathlib import Path
from typing import List

import pytest
from pydantic import TypeAdapter

from app.models.media import MediaAnalysis, Rhythm
from app.models.plan import BeatRole, Beat, BrollShot, EditPlan, MusicCue, MusicPlan, PlanAmendment, Selection
from app.runtime import repo as runtime_repo
from app.tools.catalog import import_asset
from app.tools.editing import apply_edits, create_project, get_project
from app.tools.planning import amend_plan, compile_plan, diff_plan, get_plan, save_plan, validate_plan
from helpers import video_track
from test_semantic_timeline import analysis

from edit_plan.shared import every, with_a_fact


def test_a_saved_plan_is_stamped_with_the_footage_it_was_written_against(planned: dict) -> None:
    stored = get_plan(planned["plan_id"])["plan"]
    assert stored["timeline_input_hash"]
    assert stored["version"] == 1
    assert planned["problems"] == []
    # The reasons survive, which is the point of writing them down.
    assert stored["selections"][0]["rationale"] == "開場最清楚"
    assert stored["rejected"][0]["reason"] == "重複了"

def test_saving_a_plan_at_the_wrong_version_is_refused(planned: dict) -> None:
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    save_plan(stored)
    with pytest.raises(ValueError, match="version conflict"):
        save_plan(stored)

def amendments(*operations: dict) -> list:
    """Validate plan amendments the way a client's would be.

    Args:
        *operations: Amendments as plain dictionaries.

    Returns:
        The parsed amendments.
    """
    return TypeAdapter(List[PlanAmendment]).validate_python(list(operations))

def test_one_trim_can_be_changed_without_sending_the_plan_again(planned: dict) -> None:
    first = planned["clips"][0]["clip_id"]
    result = amend_plan(planned["plan_id"], 1, amendments(
        {"action": "set_trim", "clip_id": first, "trim": {"kind": "head", "seconds": 1.0}},
    ))
    assert result["version"] == 2 and result["problems"] == []
    stored = get_plan(planned["plan_id"])["plan"]
    assert stored["selections"][0]["trim"] == {
        "kind": "head", "keep_clip_ids": [], "seconds": 1.0, "from_seconds": None, "to_seconds": None,
    }
    # Everything the plan was written to remember is still there.
    assert stored["selections"][0]["rationale"] == "開場最清楚"
    assert stored["rejected"][0]["reason"] == "重複了"

def test_dropping_a_piece_records_why_it_went(planned: dict) -> None:
    second = planned["clips"][2]["clip_id"]
    amend_plan(planned["plan_id"], 1, amendments(
        {"action": "drop_selection", "clip_id": second, "reason": "尾巴太拖"},
    ))
    stored = get_plan(planned["plan_id"])["plan"]
    assert [item["clip_id"] for item in stored["selections"]] == [planned["clips"][0]["clip_id"]]
    assert {item["clip_id"]: item["reason"] for item in stored["rejected"]}[second] == "尾巴太拖"

def test_a_piece_can_be_put_back_where_it_belongs(planned: dict) -> None:
    rejected, last = planned["clips"][1]["clip_id"], planned["clips"][2]["clip_id"]
    amend_plan(planned["plan_id"], 1, amendments(
        {"action": "add_selection", "before_clip_id": last,
         "selection": {"clip_id": rejected, "beat_id": "b1", "rationale": "還是需要這句"}},
    ))
    stored = get_plan(planned["plan_id"])["plan"]
    assert [item["clip_id"] for item in stored["selections"]][1] == rejected
    # It is in the cut now, so it is no longer an answer to "why is it not in there".
    assert [item["clip_id"] for item in stored["rejected"]] == []

def test_amending_a_plan_someone_else_has_moved_on_is_refused(planned: dict) -> None:
    first = planned["clips"][0]["clip_id"]
    change = amendments({"action": "set_trim", "clip_id": first, "trim": {"kind": "tighten"}})
    amend_plan(planned["plan_id"], 1, change)
    with pytest.raises(ValueError, match="version conflict"):
        amend_plan(planned["plan_id"], 1, change)

def test_amending_a_piece_the_plan_does_not_use_is_refused(planned: dict) -> None:
    with pytest.raises(ValueError, match="does not use clip"):
        amend_plan(planned["plan_id"], 1, amendments(
            {"action": "set_trim", "clip_id": "tl_nope:u9999", "trim": {"kind": "full"}},
        ))

def test_a_plan_checks_out_before_anything_is_rendered(planned: dict) -> None:
    result = validate_plan(planned["plan_id"])
    assert result["ok"] is True and result["problems"] == []
    assert result["clips"] >= 1 and result["duration"] > 0

def test_compiling_builds_the_cut_on_an_empty_project(planned: dict) -> None:
    project = create_project(name="測試", width=640, height=360)["id"]
    result = compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    assert result["clips"] == result["clips"] and result["duration"] > 0
    state = get_project(project)
    assert [track["id"] for track in state["tracks"]] == ["main"]
    assert len(state["tracks"][0]["clips"]) == result["clips"]

def ops(*operations: dict) -> list:
    """Validate operations the way a client's would be.

    Args:
        *operations: Operations as plain dictionaries.

    Returns:
        The validated operations.
    """
    from app.tools.planning import OPERATIONS

    return OPERATIONS.validate_python(list(operations))

def main_clips(project_id: str) -> List[dict]:
    """Read the sequence track's clips, with every field.

    Args:
        project_id: Project to read.

    Returns:
        The clips in timeline order.
    """
    from app.models.timeline.parts import Clip

    track = next(track for track in get_project(project_id)["tracks"] if track["id"] == "main")
    return [Clip.model_validate(clip).model_dump() for clip in track["clips"]]

def compiled_project(planned: dict) -> str:
    """Compile the plan onto a fresh project.

    Args:
        planned: The saved plan.

    Returns:
        The project ID.
    """
    project = create_project(name="測試", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    return project

def test_covering_picture_compiles_onto_a_track_of_its_own(planned: dict, media: Path) -> None:
    """Above the sequence, silent, and without changing how long the video runs."""
    from app.tools.planning import propose_broll
    from app.tools.timelines import build_semantic_timeline, query_clips

    quiet = import_asset(str(media / "silent.mp4"))["id"]
    runtime_repo.save_analysis(analysis(duration=4.0, scenes=[(0.0, 4.0)], asset_id=quiet))
    plan = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    # Rebuilt over both files, so the plan is rewritten against the new timeline.
    built = build_semantic_timeline([clip["asset_id"] for clip in planned["clips"]] + [quiet], rebuild=True)
    speech = query_clips(timeline_id=built["timeline_id"], text="句話")["clips"]
    cover = query_clips(timeline_id=built["timeline_id"], asset_ids=[quiet])["clips"][0]
    saved = save_plan(EditPlan(
        timeline_id=built["timeline_id"], timeline_input_hash="",
        goal=plan.goal, beats=[Beat(id="b1", name="主體")],
        # The same two sentences the fixture chose: `wide.mp4` runs ten seconds and the
        # analysis describes twenty, so the later ones are not in the file at all.
        selections=[Selection(clip_id=speech[place]["clip_id"], beat_id="b1") for place in (0, 2)],
        broll=[BrollShot(clip_id=cover["clip_id"], over_clip_id=speech[0]["clip_id"], seconds=1.5)],
    ))
    assert saved["problems"] == []
    # Nothing in this cut holds long enough to be worth offering something to cover it.
    assert propose_broll(saved["plan_id"])["slots"] == []

    project = create_project(name="測試", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=saved["plan_id"])
    state = get_project(project)
    tracks = {track["id"]: track for track in state["tracks"]}
    assert list(tracks) == ["main", "broll"], list(tracks)
    laid = tracks["broll"]["clips"]
    assert len(laid) == 1 and laid[0]["volume"] == 0.0
    # It remembers what it is made of, so the next compile can match it if it gets pinned.
    assert laid[0]["from_clip_ids"] == [cover["clip_id"], speech[0]["clip_id"]]
    # The sequence decides the length, so covering picture never stretches the video.
    assert state["duration"] == sum(
        clip["source_range"]["end"] - clip["source_range"]["start"] for clip in tracks["main"]["clips"]
    )

def test_a_compiled_clip_records_where_it_came_from(planned: dict) -> None:
    clips = main_clips(compiled_project(planned))
    assert all(clip["from_plan_id"] == planned["plan_id"] for clip in clips)
    assert all(clip["from_clip_ids"] for clip in clips)
    assert not any(clip["pinned"] for clip in clips)

def test_compiling_again_rebuilds_what_the_plan_made(planned: dict) -> None:
    project = compiled_project(planned)
    before = main_clips(project)
    version = get_project(project)["version"]
    result = compile_plan(project_id=project, expected_version=version, plan_id=planned["plan_id"])
    after = main_clips(project)
    # The same plan gives the same cut, and nothing accumulates.
    assert result["kept"] == 0
    assert [(clip["asset_id"], clip["source_range"]) for clip in after] == [
        (clip["asset_id"], clip["source_range"]) for clip in before
    ]

def test_trimming_a_compiled_clip_by_hand_pins_it(planned: dict) -> None:
    project = compiled_project(planned)
    first = main_clips(project)[0]
    apply_edits(project, get_project(project)["version"], ops({
        "action": "trim_clip", "track_id": "main", "clip_id": first["id"],
        "new_source_range": {"start": float(first["source_range"]["start"]),
                             "end": float(first["source_range"]["start"]) + 0.5},
    }))
    # Nobody had to say "pin this": changing it by hand is what pinning records.
    assert main_clips(project)[0]["pinned"] is True

def test_a_hand_adjustment_survives_the_next_compile(planned: dict) -> None:
    project = compiled_project(planned)
    first = main_clips(project)[0]
    shortened = {"start": float(first["source_range"]["start"]), "end": float(first["source_range"]["start"]) + 0.5}
    apply_edits(project, get_project(project)["version"], ops(
        {"action": "trim_clip", "track_id": "main", "clip_id": first["id"], "new_source_range": shortened},
        {"action": "set_clip_audio", "track_id": "main", "clip_id": first["id"], "volume": 0.25},
    ))
    result = compile_plan(project_id=project, expected_version=get_project(project)["version"],
                          plan_id=planned["plan_id"])
    kept = main_clips(project)[0]
    assert result["kept"] == 1
    assert float(kept["source_range"]["end"]) == pytest.approx(shortened["end"])
    assert kept["volume"] == 0.25
    assert kept["pinned"] is True

def test_a_hand_made_transition_and_j_cut_survive_the_next_compile(planned: dict) -> None:
    """A pinned clip is rebuilt by inserting it again, and the insert used to drop these.

    `audio_lead`, `audio_lag` and the transition were not on the operation that
    creates a clip, so a recompile silently handed back a straight cut with its
    sound back on its picture — the one thing pinning promises not to do.
    """
    project = compiled_project(planned)
    clips = main_clips(project)
    second = clips[1]
    apply_edits(project, get_project(project)["version"], ops(
        {"action": "set_clip_audio", "track_id": "main", "clip_id": second["id"], "audio_lead": 0.2},
        {"action": "set_clip_look", "track_id": "main", "clip_id": second["id"],
         "transition_in": {"kind": "wipe", "seconds": 0.3, "direction": "up"}},
    ))
    compile_plan(project_id=project, expected_version=get_project(project)["version"],
                 plan_id=planned["plan_id"])
    kept = runtime_repo.get_project(project).tracks[0].clips[1]
    assert kept.pinned is True
    assert float(kept.audio_lead) == pytest.approx(0.2)
    assert kept.transition_in is not None
    assert (kept.transition_in.kind, float(kept.transition_in.seconds)) == ("wipe", 0.3)
    assert kept.transition_in.direction.value == "up"

def test_a_recompile_that_would_lose_a_pinned_clip_is_refused(planned: dict) -> None:
    project = compiled_project(planned)
    first = main_clips(project)[0]
    apply_edits(project, get_project(project)["version"], ops({
        "action": "set_clip_pinned", "track_id": "main", "clip_id": first["id"], "pinned": True,
    }))
    # Rewrite the plan so the footage that clip was made from is no longer in it.
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    without = stored.model_copy(update={"selections": stored.selections[1:]})
    save_plan(without)
    with pytest.raises(ValueError, match="adjusted by hand"):
        compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=without.id)
    # Refused means nothing moved.
    assert main_clips(project)[0]["id"] == first["id"]

def test_unpinning_hands_a_clip_back_to_the_plan(planned: dict) -> None:
    project = compiled_project(planned)
    first = main_clips(project)[0]
    original_end = float(first["source_range"]["end"])
    apply_edits(project, get_project(project)["version"], ops({
        "action": "trim_clip", "track_id": "main", "clip_id": first["id"],
        "new_source_range": {"start": float(first["source_range"]["start"]),
                             "end": float(first["source_range"]["start"]) + 0.5},
    }))
    apply_edits(project, get_project(project)["version"], ops({
        "action": "set_clip_pinned", "track_id": "main", "clip_id": first["id"], "pinned": False,
    }))
    compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=planned["plan_id"])
    assert float(main_clips(project)[0]["source_range"]["end"]) == pytest.approx(original_end)

def test_a_clip_the_plan_never_made_blocks_the_compile_until_it_is_pinned(planned: dict) -> None:
    project = create_project(name="測試", width=640, height=360)["id"]
    asset_id = main_clips(compiled_project(planned))[0]["asset_id"]
    apply_edits(project, 1, ops(
        video_track(),
        {"action": "insert_clip", "track_id": "main", "clip_id": "mine", "asset_id": asset_id,
         "source_range": {"start": 0, "end": 1}},
    ))
    with pytest.raises(ValueError, match="compiling would remove it"):
        compile_plan(project_id=project, expected_version=get_project(project)["version"],
                     plan_id=planned["plan_id"])

def test_tracks_the_plan_does_not_own_are_left_alone(planned: dict) -> None:
    project = compiled_project(planned)
    asset_id = main_clips(project)[0]["asset_id"]
    apply_edits(project, get_project(project)["version"], ops(
        {"action": "add_track", "track_id": "cam", "track_type": "video"},
        {"action": "add_clip", "track_id": "cam", "clip_id": "inset", "asset_id": asset_id,
         "source_range": {"start": 0, "end": 1}, "timeline_in": 0,
         "layout": {"x": 0.6, "y": 0.1, "width": 0.3, "height": 0.3}},
    ))
    compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=planned["plan_id"])
    inset = next(track for track in get_project(project)["tracks"] if track["id"] == "cam")
    assert [clip["id"] for clip in inset["clips"]] == ["inset"]

def test_a_plan_that_does_not_check_out_never_compiles(planned: dict) -> None:
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    broken = stored.model_copy(update={
        "selections": [*stored.selections, Selection(clip_id="not-a-clip", beat_id="b1")],
    })
    save_plan(broken)
    project = create_project(name="測試", width=640, height=360)["id"]
    with pytest.raises(ValueError, match="cannot be compiled yet"):
        compile_plan(project_id=project, expected_version=1, plan_id=broken.id)
    assert get_project(project)["tracks"] == []

def test_two_plans_can_be_compared_through_the_tools(planned: dict) -> None:
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    other = save_plan(stored.model_copy(update={"id": "plan-b", "version": 1, "goal": "另一種剪法"}))
    changes = diff_plan(planned["plan_id"], other["plan_id"])["changes"]
    assert any("另一種剪法" in line for line in changes["goal"])
    assert planned["plan_id"] in {plan["plan_id"] for plan in get_plan(other["plan_id"])["other_plans"]}

def test_markers_land_on_the_timeline_in_order() -> None:
    """They are written as one set, and read back sorted by where they are."""
    project_id = create_project(name="測試", width=1920, height=1080)["id"]
    apply_edits(project_id, 1, ops({"action": "set_markers", "markers": [
        {"id": "b", "name": "結尾", "timeline_in": 9},
        {"id": "a", "name": "開場", "timeline_in": 0},
    ]}))
    stored = runtime_repo.get_project(project_id)
    assert [marker.name for marker in stored.markers] == ["開場", "結尾"]


def test_two_markers_cannot_share_a_name_on_the_timeline() -> None:
    """An id is how a marker is referred to, so two of them is a broken set."""
    project_id = create_project(name="測試", width=1920, height=1080)["id"]
    with pytest.raises(ValueError, match="share the id"):
        apply_edits(project_id, 1, ops({"action": "set_markers", "markers": [
            {"id": "a", "name": "開場", "timeline_in": 0},
            {"id": "a", "name": "結尾", "timeline_in": 9},
        ]}))

def tone(folder: Path, name: str, frequency: int, seconds: int = 20) -> str:
    """Write a steady tone to stand in for a song, and register it.

    Args:
        folder: Directory to write it to.
        name: File name.
        frequency: Pitch, so two songs can be told apart in the mix.
        seconds: Length.

    Returns:
        The asset ID.
    """
    import subprocess

    path = folder / name
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}",
         "-c:a", "libmp3lame", str(path)],
        check=True, capture_output=True,
    )
    return import_asset(str(path))["id"]

def test_the_music_changes_where_the_part_of_the_video_does(planned: dict, tmp_path: Path) -> None:
    """End to end: the first part plays one song, cut on its beat; the second part the other."""
    from helpers import level, render

    # Neither at 440Hz, which is what the footage itself sounds like.
    low, high = tone(tmp_path, "low.mp3", 700), tone(tmp_path, "high.mp3", 1500)
    # The beat is stamped on rather than measured: a steady tone has none, and the point
    # here is where the music goes, not how its beat is found.
    runtime_repo.save_analysis(MediaAnalysis(
        asset_id=low, duration=20.0, rhythm=Rhythm(tempo=120.0, beats=every(0.5, seconds=20.0)),
    ))
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    first, last = planned["clips"][0]["clip_id"], planned["clips"][2]["clip_id"]
    saved = save_plan(with_a_fact(stored.model_copy(update={
        "beats": [Beat(id="b1", name="開場", role=BeatRole.TURN), Beat(id="b2", name="結尾", role=BeatRole.PAYOFF)],
        "selections": [Selection(clip_id=first, beat_id="b1"), Selection(clip_id=last, beat_id="b2")],
        "music": MusicPlan(duck_under_speech=False, cues=[
            MusicCue(asset_id=low, volume=1.0, fade_in=0, fade_out=0, cut_on_beat=True),
            MusicCue(beat_id="b2", asset_id=high, volume=1.0, fade_in=0, fade_out=0),
        ]),
    }), 1.5, 2.5))
    assert saved["problems"] == [], saved["problems"]
    moved = next(note for note in saved["notes"] if note["kind"] == "cuts_moved_onto_the_beat")
    assert "moved onto the beat" in moved["message"] and not moved["look"]

    project = create_project(name="測試", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=saved["plan_id"])
    state = get_project(project)
    tracks = {track["id"]: track for track in state["tracks"]}
    cut = float(tracks["main"]["clips"][1]["timeline_in"])
    # On a beat of the song playing under it, and the second song comes in right there.
    assert cut % 0.5 == pytest.approx(0.0, abs=0.002)
    music = sorted(tracks["music"]["clips"], key=lambda clip: float(clip["timeline_in"]))
    assert [(clip["asset_id"], float(clip["timeline_in"])) for clip in music] == [(low, 0.0), (high, cut)]
    assert {marker["id"]: float(marker["timeline_in"]) for marker in state["markers"]}["b2"] == cut

    out = tmp_path / "out.mp4"
    render(project, out)
    heard_first, heard_last = (0.3, cut - 0.6), (cut + 0.3, 1.0)
    assert level(out, *heard_first, freq=700) > level(out, *heard_first, freq=1500) + 20
    assert level(out, *heard_last, freq=1500) > level(out, *heard_last, freq=700) + 20

def test_both_songs_are_heard_where_they_cross_fade(planned: dict, tmp_path: Path) -> None:
    from helpers import level, render

    low, high = tone(tmp_path, "low.mp3", 700), tone(tmp_path, "high.mp3", 1500)
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    first, last = planned["clips"][0]["clip_id"], planned["clips"][2]["clip_id"]
    saved = save_plan(with_a_fact(stored.model_copy(update={
        "beats": [Beat(id="b1", name="開場", role=BeatRole.TURN), Beat(id="b2", name="結尾", role=BeatRole.PAYOFF)],
        "selections": [Selection(clip_id=first, beat_id="b1"), Selection(clip_id=last, beat_id="b2")],
        "music": MusicPlan(duck_under_speech=False, crossfade_seconds=1.0, cues=[
            MusicCue(asset_id=low, volume=1.0, fade_in=0, fade_out=0),
            MusicCue(beat_id="b2", asset_id=high, start=5.0, volume=1.0, fade_in=0, fade_out=0),
        ]),
    }), 1.5, 2.5))
    assert saved["problems"] == [], saved["problems"]
    project = create_project(name="測試", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=saved["plan_id"])
    state = get_project(project)
    assert {track["id"] for track in state["tracks"]} >= {"music", "music_b"}
    cut = float(next(track for track in state["tracks"] if track["id"] == "main")["clips"][1]["timeline_in"])
    out = tmp_path / "crossfade.mp4"
    render(project, out)
    # Before the fade only the first song; halfway through it both; after the cut only the second.
    assert level(out, cut - 1.8, 0.4, freq=700) > level(out, cut - 1.8, 0.4, freq=1500) + 20
    both = (cut - 0.6, 0.3)
    assert abs(level(out, *both, freq=700) - level(out, *both, freq=1500)) < 12
    assert level(out, cut + 0.3, 0.5, freq=1500) > level(out, cut + 0.3, 0.5, freq=700) + 20

def test_a_sped_up_cut_compiles_to_the_length_the_plan_says(planned: dict) -> None:
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    from app.models.plan import Pacing

    saved = save_plan(stored.model_copy(update={"pacing": Pacing(speed=1.2)}))
    expected = validate_plan(saved["plan_id"])["duration"]
    project = create_project(name="測試", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=saved["plan_id"])
    assert float(get_project(project)["duration"]) == pytest.approx(expected, abs=0.01)

def test_a_clip_lengthened_by_hand_over_the_next_one_is_not_played_twice(planned: dict) -> None:
    project = compiled_project(planned)
    first, second = main_clips(project)
    apply_edits(project, get_project(project)["version"], ops(
        {"action": "trim_clip", "track_id": "main", "clip_id": first["id"],
         "new_source_range": {"start": float(first["source_range"]["start"]),
                              "end": float(second["source_range"]["end"])}},
        {"action": "delete_clip", "track_id": "main", "clip_id": second["id"]},
    ))
    result = compile_plan(project_id=project, expected_version=get_project(project)["version"],
                          plan_id=planned["plan_id"])
    assert len(main_clips(project)) == 1
    left = next(note for note in result["notes"] if note["kind"] == "left_out_for_a_hand_edit")
    assert "already plays them" in left["message"] and left["clip_ids"] and left["look"]

def test_how_a_beat_comes_in_and_how_a_piece_plays_reach_the_timeline(planned: dict) -> None:
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    save_plan(stored.model_copy(update={"beats": [
        stored.beats[0].model_copy(update={"role": BeatRole.TURN}), Beat(id="b2", name="結尾", role=BeatRole.PAYOFF),
    ]}))
    second = planned["clips"][2]["clip_id"]
    result = amend_plan(planned["plan_id"], 2, amendments(
        {"action": "set_rationale", "clip_id": second, "beat_id": "b2"},
        {"action": "set_beat_join", "beat_id": "b2", "transition_in": {"kind": "dissolve", "seconds": 0.4},
         "sound_lead": 0.3},
        {"action": "set_playback", "clip_id": second, "speed": 1.5, "volume": 0.5},
        {"action": "set_facts", "facts": [{"id": "topic", "says": "這支在講什麼"}]},
        {"action": "set_knowledge", "clip_id": planned["clips"][0]["clip_id"],
         "establishes": [{"fact": "topic", "evidence": [1.5, 2.5]}]},
    ))
    assert result["problems"] == []
    project = compiled_project(planned)
    kept = runtime_repo.get_project(project).tracks[0].clips[1]
    assert (kept.transition_in.kind, float(kept.transition_in.seconds)) == ("dissolve", 0.4)
    assert float(kept.audio_lead) == pytest.approx(0.3)
    assert (kept.speed, kept.volume) == (1.5, 0.5)

def test_a_plan_is_copied_onto_a_timeline_built_over_more_files(planned: dict, media: Path) -> None:
    from app.tools.planning import copy_plan
    from app.tools.timelines import build_semantic_timeline, get_semantic_clip

    quiet = import_asset(str(media / "silent.mp4"))["id"]
    runtime_repo.save_analysis(analysis(duration=4.0, scenes=[(0.0, 4.0)], asset_id=quiet))
    wider = build_semantic_timeline([planned["clips"][0]["asset_id"], quiet], rebuild=True)["timeline_id"]
    assert wider != planned["timeline_id"]

    copied = copy_plan(planned["plan_id"], timeline_id=wider, note="加上空景")
    plan = EditPlan.model_validate(get_plan(copied["plan_id"])["plan"])
    assert copied["plan_id"] != planned["plan_id"] and plan.timeline_id == wider
    # Same moments of the same file, under the IDs the wider timeline gave them.
    was = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    for before, after in zip(was.selections, plan.selections):
        assert after.clip_id.startswith(wider) and after.rationale == before.rationale
        assert get_semantic_clip(after.clip_id)["source_range"] == get_semantic_clip(before.clip_id)["source_range"]
    assert copied["problems"] == []

def test_a_plan_follows_its_sentences_after_the_footage_is_transcribed_again(planned: dict) -> None:
    from app.tools.planning import copy_plan
    from app.tools.timelines import build_semantic_timeline, get_semantic_clip

    asset_id = planned["clips"][0]["asset_id"]
    # The second listen puts every sentence edge a little elsewhere.
    runtime_repo.save_analysis(analysis(asset_id=asset_id, duration=20.0, segments=[
        (0.8, 3.1, "第一句話"), (3.5, 5.0, "第二句話"), (8.2, 10.4, "第三句話。"), (12.0, 14.0, "第四句話"),
    ], silences=[(0.0, 0.8), (3.1, 3.5), (5.0, 8.2), (10.4, 12.0), (14.0, 20.0)]))
    again = build_semantic_timeline([asset_id], rebuild=True)["timeline_id"]
    copied = copy_plan(planned["plan_id"], timeline_id=again)
    plan = EditPlan.model_validate(get_plan(copied["plan_id"])["plan"])
    assert [get_semantic_clip(item.clip_id)["source_range"]["start"] for item in plan.selections] == [0.8, 8.2]
    assert [match["is"] for match in copied["matched_nearby"]] == [[0.8, 3.1], [8.2, 10.4]]
    # The rejected sentence did not move, so it is carried exactly and not listed.
    assert len(plan.rejected) == 1

def heard_again(planned: dict, third: tuple = (8.2, 10.4)) -> str:
    """Transcribe the planned footage again so every sentence edge moves, and build a timeline over it."""
    from app.tools.timelines import build_semantic_timeline

    asset_id = planned["clips"][0]["asset_id"]
    runtime_repo.save_analysis(analysis(asset_id=asset_id, duration=20.0, segments=[
        (0.8, 3.1, "第一句話"), (3.5, 5.0, "第二句話"), (*third, "第三句話。"), (12.0, 14.0, "第四句話"),
    ], silences=[(0.0, 0.8), (3.1, 3.5), (5.0, third[0]), (third[1], 12.0), (14.0, 20.0)]))
    return build_semantic_timeline([asset_id], rebuild=True)["timeline_id"]

def test_a_trim_takes_the_same_seconds_of_the_file_after_its_sentence_moved(planned: dict) -> None:
    from app.tools.planning import copy_plan

    first, third = planned["clips"][0]["clip_id"], planned["clips"][2]["clip_id"]
    amend_plan(planned["plan_id"], 1, amendments(
        {"action": "set_trim", "clip_id": first, "trim": {"kind": "head", "seconds": 1.5}},
        {"action": "set_trim", "clip_id": third, "trim": {"kind": "tail", "seconds": 1.0}},
    ))
    copied = copy_plan(planned["plan_id"], timeline_id=heard_again(planned))
    plan = EditPlan.model_validate(get_plan(copied["plan_id"])["plan"])
    # 1.0–2.5s of the file, now that the sentence starts at 0.8; and 9.0–10.0s, now that it ends at 10.4.
    assert [item.trim.seconds for item in plan.selections] == [1.7, 1.4]
    assert [match["trim"] for match in copied["matched_nearby"]] == ["head 1.5s → 1.7s", "tail 1s → 1.4s"]

def test_a_trim_whose_seconds_are_no_longer_in_its_sentence_is_handed_back(planned: dict) -> None:
    from app.tools.planning import copy_plan

    third = planned["clips"][2]["clip_id"]
    amend_plan(planned["plan_id"], 1, amendments(
        {"action": "set_trim", "clip_id": third, "trim": {"kind": "head", "seconds": 0.4}},
    ))
    # 8.0–8.4s of the file, and the sentence now starts at 8.6.
    with pytest.raises(ValueError, match="no longer holds the seconds its trim takes"):
        copy_plan(planned["plan_id"], timeline_id=heard_again(planned, third=(8.6, 10.4)))

def test_a_song_picked_by_ear_under_the_plan_s_music_is_written_into_its_cue(planned: dict, media: Path) -> None:
    from app.models.offer import OfferedSong
    from app.tools.catalog import set_asset_library
    from app.tools.soundtrack import offer_songs, pick_song, song_trial

    song = import_asset(str(media / "song.mp3"))["id"]
    jingle = import_asset(str(media / "jingle.wav"))["id"]
    set_asset_library([song, jingle], "music")
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    save_plan(stored.model_copy(update={"music": MusicPlan(cues=[MusicCue(asset_id=jingle, volume=0.3, fade_in=0.4)])}))
    project = create_project(name="配樂", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    offer_songs(project, [OfferedSong(asset_id=song, start=2)])
    # Heard as the plan's cue would play it.
    offer = runtime_repo.song_offer(project)
    assert (offer.volume, offer.fade_in) == (0.3, 0.4)
    trial = song_trial(runtime_repo.get_project(project), offer, 1)
    assert any(clip.asset_id == song and float(clip.audio_fade_in) == 0.4 for track in trial.tracks for clip in track.clips)
    picked = pick_song(project, 1)
    assert "已寫進計畫" in picked["text"]
    cue = get_plan(planned["plan_id"])["plan"]["music"]["cues"][0]
    assert (cue["asset_id"], cue["start"], cue["volume"]) == (song, 2, 0.3)
