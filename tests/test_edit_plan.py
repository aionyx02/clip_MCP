"""Tests for the edit plan: what it refuses, and what it compiles to.

The compiler is the deterministic half of the design, so most of these are
arithmetic: given this plan and this footage, these are the seconds. The rest
check that a plan which does not hold up is handed back rather than quietly
repaired, because a compiler that fixes things is making decisions it was
built to stay out of.
"""

from decimal import Decimal
from pathlib import Path
from typing import Dict, List

import pytest
from pydantic import TypeAdapter

from app.engine.plan import (
    BREATH_SECONDS,
    check_plan,
    compile_operations,
    compiled_duration,
    diff_plans,
    plan_pieces,
)
from app.engine.sections import build_sections, propose_candidates
from app.engine.semantic import build_timeline
from app.models.media import Asset
from app.models.plan import (
    Beat, EditPlan, MusicPlan, PlanAmendment, PlanTarget, Rejection, Selection, Trim, TrimKind,
)
from app.models.semantic import SectionChoice, SemanticClip
from app.server import (
    amend_plan,
    apply_edits,
    compile_plan,
    create_project,
    diff_plan,
    get_plan,
    get_project,
    import_asset,
    save_plan,
    validate_plan,
)
from app.server import repo as server_repo
from helpers import video_track
from test_semantic_timeline import ASSET_ID, analysis, asset

FOOTAGE = dict(
    duration=20.0,
    segments=[
        (1.0, 3.0, "第一句話"),
        (3.5, 5.0, "第二句話"),
        (8.0, 10.0, "第三句話"),
        (12.0, 14.0, "第四句話"),
    ],
    silences=[(0.0, 1.0), (3.0, 3.5), (5.0, 8.0), (10.0, 12.0), (14.0, 20.0)],
)

def sourced(asset_id: str = ASSET_ID, seconds: float = 20.0) -> Asset:
    """Build an asset that knows how long it is.

    Args:
        asset_id: ID for the asset.
        seconds: Its duration.

    Returns:
        The asset.
    """
    return asset(asset_id).model_copy(update={"duration": Decimal(str(seconds))})

def footage() -> tuple:
    """Build a timeline over one file of talk.

    Returns:
        `(clips_by_id, children, assets, ordered_clips)`.
    """
    made = analysis(**FOOTAGE)
    _, clips = build_timeline({ASSET_ID: sourced()}, {ASSET_ID: made})
    return {clip.id: clip for clip in clips}, {}, {ASSET_ID: sourced()}, clips

def sectioned() -> tuple:
    """Build a timeline with the whole file as one section.

    Returns:
        `(clips_by_id, children, assets, section_id)`.
    """
    made = analysis(**FOOTAGE)
    timeline, clips = build_timeline({ASSET_ID: sourced()}, {ASSET_ID: made})
    candidates = propose_candidates(clips, {ASSET_ID: made})
    sections, utterances = build_sections(timeline.id, clips, [
        SectionChoice(first_clip_id=clips[0].id, last_clip_id=clips[-1].id, name="全部", summary="整段"),
    ])
    everything = [*utterances, *sections]
    children = {sections[0].id: [clip for clip in utterances]}
    return {clip.id: clip for clip in everything}, children, {ASSET_ID: sourced()}, sections[0].id

def plan_for(selections: List[Selection], **fields) -> EditPlan:
    """Build a plan over one beat.

    Args:
        selections: The chosen footage.
        **fields: Other plan fields.

    Returns:
        The plan.
    """
    return EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash",
        beats=[Beat(id="b1", name="主體")],
        selections=selections,
        **fields,
    )

def speech_of(clips: List[SemanticClip]) -> List[SemanticClip]:
    """Pick out the clips where someone is talking.

    Args:
        clips: Clips to filter.

    Returns:
        The speech clips in order.
    """
    return [clip for clip in clips if clip.text]

def test_a_selection_is_widened_into_the_silence_around_it() -> None:
    by_id, children, assets, clips = footage()
    first = speech_of(clips)[0]
    plan = plan_for([Selection(clip_id=first.id, beat_id="b1")])
    pieces = plan_pieces(plan, by_id, children, assets)
    # The sentence runs 1.0 to 3.0 and has a full second of silence in front of it,
    # so it opens a breath early rather than the instant the picture cuts.
    assert [(piece.asset_id, piece.start, piece.end) for piece in pieces] == [
        (ASSET_ID, round(1.0 - BREATH_SECONDS, 3), round(3.0 + BREATH_SECONDS, 3))
    ]
    # The piece knows what it was made of, which is what a second compile matches on.
    assert pieces[0].from_clip_ids == (first.id,)

def test_the_air_never_reaches_further_than_the_headroom_allows() -> None:
    # Two sentences with a fiftieth of a second of silence between them: less room
    # than a breath wants, so the breath has to give way rather than clip the words.
    tight = analysis(
        10.0,
        segments=[(1.0, 3.0, "前一句"), (3.05, 6.0, "後一句")],
        silences=[(0.0, 1.0), (3.0, 3.05), (6.0, 10.0)],
    )
    _, clips = build_timeline({ASSET_ID: sourced(seconds=10.0)}, {ASSET_ID: tight})
    by_id, assets = {clip.id: clip for clip in clips}, {ASSET_ID: sourced(seconds=10.0)}
    second = speech_of(clips)[1]
    assert second.safe_in == pytest.approx(0.05)

    pieces = plan_pieces(plan_for([Selection(clip_id=second.id, beat_id="b1")]), by_id, {}, assets)
    assert pieces[0].start == pytest.approx(second.source_range.start - second.safe_in)
    assert pieces[0].start > speech_of(clips)[0].source_range.end - BREATH_SECONDS

def test_a_window_that_reaches_the_end_of_the_file_stays_inside_it() -> None:
    # A file 36.266667s long, with the last sentence running right to the end. Rounding
    # that to milliseconds gives 36.267, which is past the end of the file: the whole
    # plan used to be refused for a window that was only ever asking for all of it.
    length = 36.266667
    to_the_end = analysis(
        length,
        segments=[(1.0, 3.0, "第一句"), (30.0, length, "最後一句")],
        silences=[(0.0, 1.0), (3.0, 30.0)],
    )
    assets = {ASSET_ID: sourced(seconds=length)}
    _, clips = build_timeline(assets, {ASSET_ID: to_the_end})
    by_id = {clip.id: clip for clip in clips}
    last = speech_of(clips)[-1]

    pieces = plan_pieces(plan_for([Selection(clip_id=last.id, beat_id="b1")]), by_id, {}, assets)
    # Compared the way the timeline compares it, which is where this used to blow up.
    assert Decimal(str(pieces[-1].end)) <= assets[ASSET_ID].duration

def test_pieces_that_nearly_touch_become_one_clip() -> None:
    by_id, children, assets, clips = footage()
    first, second = speech_of(clips)[:2]
    plan = plan_for([
        Selection(clip_id=first.id, beat_id="b1"),
        Selection(clip_id=second.id, beat_id="b1"),
    ])
    pieces = plan_pieces(plan, by_id, children, assets)
    # Half a second apart, with breath on both sides, closes to under the merge gap.
    assert len(pieces) == 1
    assert pieces[0].end == pytest.approx(5.0 + BREATH_SECONDS, abs=0.01)
    # Merged, so the one clip remembers both sentences it was made from.
    assert pieces[0].from_clip_ids == (first.id, second.id)

def test_pieces_far_apart_stay_separate() -> None:
    by_id, children, assets, clips = footage()
    first, third = speech_of(clips)[0], speech_of(clips)[2]
    plan = plan_for([
        Selection(clip_id=first.id, beat_id="b1"),
        Selection(clip_id=third.id, beat_id="b1"),
    ])
    assert len(plan_pieces(plan, by_id, children, assets)) == 2

def test_beats_decide_the_order_the_footage_plays_in() -> None:
    by_id, children, assets, clips = footage()
    first, third = speech_of(clips)[0], speech_of(clips)[2]
    plan = EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash",
        beats=[Beat(id="open", name="開場"), Beat(id="body", name="主體")],
        selections=[
            Selection(clip_id=first.id, beat_id="body"),
            Selection(clip_id=third.id, beat_id="open"),
        ],
    )
    # Written body-first, but the beats say the opening comes first.
    assert plan_pieces(plan, by_id, children, assets)[0].start == pytest.approx(third.source_range.start, abs=0.2)

def test_head_and_tail_take_the_window_they_are_asked_for() -> None:
    by_id, children, assets, clips = footage()
    first = speech_of(clips)[0]
    head = plan_pieces(
        plan_for([Selection(clip_id=first.id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=1.0))]),
        by_id, children, assets,
    )
    tail = plan_pieces(
        plan_for([Selection(clip_id=first.id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=1.0))]),
        by_id, children, assets,
    )
    assert head[0].end == pytest.approx(2.0)
    assert tail[0].start == pytest.approx(2.0)

def test_keep_takes_only_the_sentences_it_names() -> None:
    by_id, children, assets, section_id = sectioned()
    inside = speech_of(children[section_id])
    plan = plan_for([Selection(
        clip_id=section_id, beat_id="b1",
        trim=Trim(kind=TrimKind.KEEP, keep_clip_ids=[inside[0].id, inside[2].id]),
    )])
    pieces = plan_pieces(plan, by_id, children, assets)
    assert len(pieces) == 2
    assert pieces[0].start == pytest.approx(inside[0].source_range.start - BREATH_SECONDS, abs=0.01)

def test_tighten_drops_the_pauses_out_of_a_section() -> None:
    by_id, children, assets, section_id = sectioned()
    plan = plan_for([Selection(clip_id=section_id, beat_id="b1", trim=Trim(kind=TrimKind.TIGHTEN))])
    tightened = compiled_duration(plan_pieces(plan, by_id, children, assets))
    whole = compiled_duration(plan_pieces(
        plan_for([Selection(clip_id=section_id, beat_id="b1")]), by_id, children, assets,
    ))
    # Three seconds of silence in the middle alone; the tightened cut has to be shorter.
    assert tightened < whole

def context_for(plan: EditPlan, by_id: Dict[str, SemanticClip], children: dict, assets: dict) -> tuple:
    """Run the checks with a timeline matching the plan.

    Args:
        plan: Plan to check.
        by_id: The clips.
        children: Section members.
        assets: The assets.

    Returns:
        `(problems, notes)`.
    """
    from app.models.semantic import SemanticTimeline

    timeline = SemanticTimeline(
        id=plan.timeline_id, asset_ids=[ASSET_ID],
        input_hash=plan.timeline_input_hash, derivation_version=1,
    )
    return check_plan(plan, timeline, by_id, children, assets)

def test_a_clip_that_is_not_in_the_timeline_is_refused() -> None:
    by_id, children, assets, _ = footage()
    problems, _ = context_for(plan_for([Selection(clip_id="nope", beat_id="b1")]), by_id, children, assets)
    assert any("no such clip" in problem for problem in problems)

def test_a_beat_the_plan_does_not_have_is_refused() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="ghost")])
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("which the plan does not have" in problem for problem in problems)

def test_unusable_footage_cannot_be_selected() -> None:
    made = analysis(8.0, segments=[(4.0, 8.0, "終於開口")], silences=[(0.0, 4.0)], black=[(0.0, 3.5)])
    _, clips = build_timeline({ASSET_ID: sourced(seconds=8.0)}, {ASSET_ID: made})
    by_id = {clip.id: clip for clip in clips}
    plan = plan_for([Selection(clip_id=clips[0].id, beat_id="b1")])
    problems, _ = context_for(plan, by_id, {}, {ASSET_ID: sourced(seconds=8.0)})
    assert any("unusable" in problem for problem in problems)

def test_keep_needs_a_section_to_keep_things_from() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(
        clip_id=speech_of(clips)[0].id, beat_id="b1",
        trim=Trim(kind=TrimKind.KEEP, keep_clip_ids=["whatever"]),
    )])
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("nothing inside it to keep" in problem for problem in problems)

def test_keeping_something_that_is_not_inside_the_section_is_refused() -> None:
    by_id, children, assets, section_id = sectioned()
    plan = plan_for([Selection(
        clip_id=section_id, beat_id="b1",
        trim=Trim(kind=TrimKind.KEEP, keep_clip_ids=["somewhere-else"]),
    )])
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("is not inside this clip" in problem for problem in problems)

def test_asking_for_more_seconds_than_a_clip_has_is_refused() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(
        clip_id=speech_of(clips)[0].id, beat_id="b1",
        trim=Trim(kind=TrimKind.HEAD, seconds=30.0),
    )])
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("of a clip that runs" in problem for problem in problems)

def test_footage_analyzed_again_since_the_plan_was_written_is_refused() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")])
    stale = plan.model_copy(update={"timeline_input_hash": "something else"})
    from app.models.semantic import SemanticTimeline

    timeline = SemanticTimeline(id="tl_test", asset_ids=[ASSET_ID], input_hash="hash", derivation_version=1)
    problems, _ = check_plan(stale, timeline, by_id, children, assets)
    assert any("no longer mean the same moments" in problem for problem in problems)

def test_a_cut_that_misses_its_target_length_is_worth_saying() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")], target=PlanTarget(seconds=90.0))
    problems, notes = context_for(plan, by_id, children, assets)
    assert problems == []
    assert any("against a target of 90s" in note for note in notes)

def test_music_without_sound_is_refused() -> None:
    by_id, children, assets, clips = footage()
    assets = {**assets, "silent-song": asset("silent-song", has_audio=False)}
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")],
                    music=MusicPlan(asset_id="silent-song"))
    problems, _ = context_for(plan, by_id, children, assets)
    assert any("has no sound" in problem for problem in problems)

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
    assert [operation["action"] for operation in operations[1:]] == ["insert_clip", "insert_clip"]
    assert operations[1]["source_range"]["start"] < operations[2]["source_range"]["start"]

def test_music_is_laid_under_the_picture_and_fitted_to_it() -> None:
    by_id, children, assets, clips = footage()
    assets = {**assets, "song": sourced("song", seconds=60.0).model_copy(update={"has_video": False})}
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")],
                    music=MusicPlan(asset_id="song", volume=0.4))
    actions = [operation["action"] for operation in compile_operations(plan, by_id, children, assets)[0]]
    assert actions[-3:] == ["add_track", "insert_clip", "fit_track"]

def test_the_same_plan_always_compiles_to_the_same_cut() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=clip.id, beat_id="b1") for clip in speech_of(clips)])
    assert compile_operations(plan, by_id, children, assets) == compile_operations(plan, by_id, children, assets)

def test_a_diff_names_what_actually_changed() -> None:
    by_id, children, assets, clips = footage()
    first, second, third = speech_of(clips)[:3]
    before = plan_for([Selection(clip_id=first.id, beat_id="b1"), Selection(clip_id=second.id, beat_id="b1")],
                      goal="原本的目標")
    after = plan_for([
        Selection(clip_id=first.id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=1.0)),
        Selection(clip_id=third.id, beat_id="b1"),
    ], goal="改過的目標")
    changes = diff_plans(before, after)
    assert any("原本的目標" in line for line in changes["goal"])
    assert any(line.startswith("added") for line in changes["selections"])
    assert any(line.startswith("dropped") for line in changes["selections"])
    assert any(line.startswith("retrimmed") for line in changes["selections"])

def test_two_plans_that_are_the_same_diff_to_nothing() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_for([Selection(clip_id=speech_of(clips)[0].id, beat_id="b1")])
    assert diff_plans(plan, plan) == {}

@pytest.fixture
def planned(media: Path) -> dict:
    """Import a real file, build a timeline over it, and save a plan for it.

    Returns:
        A dictionary with the `plan_id`, `timeline_id` and the chosen clips.
    """
    from app.server import build_semantic_timeline, query_clips

    asset_id = import_asset(str(media / "wide.mp4"))["id"]
    server_repo.save_analysis(analysis(asset_id=asset_id, **FOOTAGE))
    built = build_semantic_timeline([asset_id], rebuild=True)
    speech = query_clips(timeline_id=built["timeline_id"], text="句話")["clips"]
    saved = save_plan(EditPlan(
        timeline_id=built["timeline_id"], timeline_input_hash="",
        goal="做一支短的",
        beats=[Beat(id="b1", name="主體", intent="講完一件事")],
        selections=[
            Selection(clip_id=speech[0]["clip_id"], beat_id="b1", rationale="開場最清楚"),
            Selection(clip_id=speech[2]["clip_id"], beat_id="b1", rationale="收在這句"),
        ],
        rejected=[Rejection(clip_id=speech[1]["clip_id"], reason="重複了")],
    ))
    return {**saved, "clips": speech}

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
    assert stored["selections"][0]["trim"] == {"kind": "head", "keep_clip_ids": [], "seconds": 1.0}
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
    project = create_project(width=640, height=360)["id"]
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
    from app.server import _OPERATIONS

    return _OPERATIONS.validate_python(list(operations))

def main_clips(project_id: str) -> List[dict]:
    """Read the sequence track's clips, with every field.

    Args:
        project_id: Project to read.

    Returns:
        The clips in timeline order.
    """
    from app.models.timeline import Clip

    track = next(track for track in get_project(project_id)["tracks"] if track["id"] == "main")
    return [Clip.model_validate(clip).model_dump() for clip in track["clips"]]

def compiled_project(planned: dict) -> str:
    """Compile the plan onto a fresh project.

    Args:
        planned: The saved plan.

    Returns:
        The project ID.
    """
    project = create_project(width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    return project

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
    project = create_project(width=640, height=360)["id"]
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
    project = create_project(width=640, height=360)["id"]
    with pytest.raises(ValueError, match="cannot be compiled yet"):
        compile_plan(project_id=project, expected_version=1, plan_id=broken.id)
    assert get_project(project)["tracks"] == []

def test_two_plans_can_be_compared_through_the_tools(planned: dict) -> None:
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    other = save_plan(stored.model_copy(update={"id": "plan-b", "version": 1, "goal": "另一種剪法"}))
    changes = diff_plan(planned["plan_id"], other["plan_id"])["changes"]
    assert any("另一種剪法" in line for line in changes["goal"])
    assert planned["plan_id"] in {plan["plan_id"] for plan in get_plan(other["plan_id"])["other_plans"]}
