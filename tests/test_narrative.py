"""The story check nominates where a cut may go wrong, and a final render of an `auto` production waits on rulings.

Nominations are made by rule from the order the project's clips actually play,
never judged: a hook or a flashback is nominated like a mistake, and is let
through by a `keep` with a reason. These tests hold each kind to a case it must
nominate and one it must not, that missing data is said rather than read as
nothing wrong, that a ruling is refused unless it is for a nomination there is
and a `keep` says why, and that a ruling stops counting once the order changes.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest

from app import runtime
from app.engine import narrative
from app.engine.narrative import Shot, nominate
from app.models.job import Job, JobStatus
from app.models.media import Transcript
from app.tools import rendering
from app.tools.editing import apply_edits, create_project, get_project
from app.tools.narrative import DoubtRuling, rule_doubts, story_doubts
from app.tools.planning import OPERATIONS
from app.tools.catalog import import_asset
from app.tools.production import start_production
from test_semantic_timeline import analysis

T0 = datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)


def kinds(found: narrative.Doubts) -> List[str]:
    """The kinds nominated, in play order."""
    return [doubt.kind for doubt in found.doubts]


def said(start: float, *words: str) -> tuple:
    """Words heard one second apart from a second of the cut."""
    return tuple((start + place, word) for place, word in enumerate(words))


# ------------------------------------------------------------------ each kind, by rule

def test_a_goodbye_with_more_than_ten_seconds_after_it_is_nominated_in_a_cut_of_one_part() -> None:
    shots = [Shot("c1", 0, 20, words=said(1, "我", "終於", "成功", "晚安")), Shot("c2", 20, 40, words=said(21, "失敗"))]
    found = nominate(shots)
    closing = [doubt for doubt in found.doubts if doubt.kind == "closing_too_early"]
    assert [doubt.id for doubt in closing] == ["closing_too_early:c1"]
    assert "晚安" in closing[0].says and closing[0].at == pytest.approx(4)


def test_a_goodbye_at_the_end_is_not() -> None:
    shots = [Shot("c1", 0, 20, words=said(1, "今天", "成功")), Shot("c2", 20, 30, words=said(25, "晚安", "掰掰"))]
    assert "closing_too_early" not in kinds(nominate(shots))


def test_going_back_more_than_twenty_minutes_within_one_part_is_nominated() -> None:
    shots = [Shot("c1", 0, 5, moment=T0, words=said(0, "好")), Shot("c2", 5, 10, moment=T0 - timedelta(minutes=21))]
    found = nominate(shots)
    assert [doubt.id for doubt in found.doubts if doubt.kind == "back_in_time"] == ["back_in_time:c1>c2"]


def test_going_back_less_than_twenty_minutes_or_forward_is_not() -> None:
    shots = [Shot("c1", 0, 5, moment=T0, words=said(0, "好")), Shot("c2", 5, 10, moment=T0 - timedelta(minutes=19)),
             Shot("c3", 10, 15, moment=T0 + timedelta(hours=3))]
    assert "back_in_time" not in kinds(nominate(shots))


def test_a_topic_left_and_come_back_to_is_nominated() -> None:
    shots = [Shot("c1", 0, 5, topics=("成功",), sourced=True, words=said(0, "好")),
             Shot("c2", 5, 10, topics=("失敗",), sourced=True), Shot("c3", 10, 15, topics=("成功",), sourced=True)]
    found = nominate(shots)
    returns = [doubt for doubt in found.doubts if doubt.kind == "topic_returns"]
    assert [doubt.id for doubt in returns] == ["topic_returns:c1>c3"] and returns[0].clip_ids == ("c1", "c2", "c3")


def test_topics_that_follow_one_another_are_not() -> None:
    shots = [Shot("c1", 0, 5, topics=("測試",), sourced=True, words=said(0, "好")),
             Shot("c2", 5, 10, topics=("測試",), sourced=True), Shot("c3", 10, 15, topics=("成功",), sourced=True),
             Shot("c4", 15, 20, card=True), Shot("c5", 20, 25, topics=("成功",), sourced=True)]
    found = nominate(shots)
    assert "topic_returns" not in kinds(found) and "topic_returns" not in found.unchecked


def test_first_words_late_are_nominated_and_early_ones_are_not() -> None:
    late = nominate([Shot("c1", 0, 27), Shot("c2", 27, 100, words=said(27, "各位"))])
    assert [doubt.id for doubt in late.doubts if doubt.kind == "late_subject"] == ["late_subject:words:c2"]
    # Sooner than 15 s, but later than 15% of a 40 s cut: late too.
    short = nominate([Shot("c1", 0, 7), Shot("c2", 7, 40, words=said(7, "各位"))])
    assert "late_subject:words:c2" in [doubt.id for doubt in short.doubts]
    early = nominate([Shot("c1", 0, 100, words=said(3, "各位"))])
    assert "late_subject" not in kinds(early)


def test_the_first_fact_landing_late_is_nominated_and_an_early_one_is_not() -> None:
    late = nominate([Shot("c1", 0, 30, words=said(1, "嗨"), planned=True, sourced=True),
                     Shot("c2", 30, 100, facts=((84.0, "subject"),), planned=True, sourced=True)])
    assert [doubt.id for doubt in late.doubts] == ["late_subject:fact:c2"]
    early = nominate([Shot("c1", 0, 100, words=said(1, "嗨"), facts=((5.0, "subject"),), planned=True,
                           sourced=True)])
    assert early.doubts == [] and "late_subject.facts" not in early.unchecked


def test_a_flashback_is_nominated_all_the_same() -> None:
    # The ending shown first, then the story from the start: legitimate, and still put to the model.
    shots = [Shot("hook", 0, 5, moment=T0 + timedelta(days=10), words=said(0, "成功了")),
             Shot("start", 5, 30, moment=T0, words=said(5, "一開始"))]
    assert kinds(nominate(shots)) == ["back_in_time"]


# ------------------------------------------------------------------ what could not be checked is said

def test_no_topics_or_recording_times_say_not_checked_rather_than_nothing() -> None:
    found = nominate([Shot("c1", 0, 5, sourced=True, words=said(0, "好")), Shot("c2", 5, 10, sourced=True)])
    assert found.doubts == []
    assert "set_sections" in found.unchecked["topic_returns"]
    assert "recording time" in found.unchecked["back_in_time"]
    assert "establish" in found.unchecked["late_subject.facts"]


def test_a_file_never_transcribed_is_not_read_as_silence() -> None:
    found = nominate([Shot("c1", 0, 30, heard=False)])
    assert "transcript" in found.unchecked["closing_too_early"]
    assert "transcript" in found.unchecked["late_subject.words"]
    assert "late_subject" not in kinds(found)


# ------------------------------------------------------------------ over a real project, through the tools

@pytest.fixture(autouse=True)
def no_workers(monkeypatch) -> None:
    """Keep every render a record: the gate is what is tested, not the file."""
    def kept(job: Job, spec: dict) -> Job:
        runtime.repo.add_job(job)
        return job

    monkeypatch.setattr(runtime.job_manager, "start_job", kept)
    monkeypatch.setattr(rendering, "_make_room", lambda: None)


def edit(project_id: str, operations: list) -> dict:
    """Apply edits the way the editor does: through `apply_edits`, with no production named."""
    return apply_edits(project_id, get_project(project_id)["version"], OPERATIONS.validate_python(operations))


@pytest.fixture
def by_hand(media: Path) -> str:
    """A project of two clips put in by hand: a talk recorded at ten that says goodnight, then one shot at nine.

    Returns:
        The project's ID.
    """
    talk = import_asset(str(media / "wide.mp4"))["id"]
    earlier = import_asset(str(media / "tall.mp4"))["id"]
    for asset_id, moment in ((talk, T0), (earlier, T0 - timedelta(hours=1))):
        runtime.repo.save_asset(runtime.repo.get_assets([asset_id])[asset_id].model_copy(update={"recorded_at": moment}))
    runtime.repo.save_analysis(analysis(10.0, segments=[(3.0, 4.0, "大家晚安")], silences=[(0.0, 3.0), (4.0, 10.0)],
                                        asset_id=talk))
    # Transcribed, and nobody talks in it.
    runtime.repo.save_analysis(analysis(6.0, silences=[(0.0, 6.0)], asset_id=earlier).model_copy(
        update={"transcript": Transcript(language="zh", model="test", segments=[])}))
    project = create_project(name="手放", width=640, height=360)["id"]
    edit(project, [
        {"action": "add_track", "track_id": "main", "track_type": "video"},
        {"action": "insert_clip", "track_id": "main", "clip_id": "talk", "asset_id": talk,
         "source_range": {"start": 0, "end": 9}},
        {"action": "insert_clip", "track_id": "main", "clip_id": "before", "asset_id": earlier,
         "source_range": {"start": 0, "end": 6}},
    ])
    return project


def test_clips_put_in_by_hand_are_nominated_from_their_words_and_times(by_hand: str) -> None:
    found = story_doubts(by_hand)
    assert sorted(doubt["id"] for doubt in found["doubts"]) == [
        "back_in_time:talk>before", "closing_too_early:talk", "late_subject:words:talk"]
    # Only what needs a clip's origin is not checked.
    assert set(found["not_checked"]) == {"topic_returns", "late_subject.facts"}
    assert "by hand" in found["not_checked"]["topic_returns"]


def test_rulings_need_a_production_an_id_on_the_list_and_a_reason_to_keep(by_hand: str) -> None:
    with pytest.raises(ValueError, match="start_production"):
        rule_doubts(by_hand, [DoubtRuling(id="closing_too_early:talk", verdict="change")])
    production = start_production(mode="auto", brief="直接做完", asset_ids=[], project_id=by_hand)["production_id"]
    with pytest.raises(ValueError, match="not among the nominations"):
        rule_doubts(by_hand, [DoubtRuling(id="closing_too_early:nothing", verdict="keep", reason="沒事")])
    with pytest.raises(ValueError, match="keep without a reason"):
        rule_doubts(by_hand, [DoubtRuling(id="closing_too_early:talk", verdict="keep", reason="  ")])
    # All or nothing: the good ruling beside a bad one is not kept either.
    with pytest.raises(ValueError, match="Nothing was saved"):
        rule_doubts(by_hand, [DoubtRuling(id="back_in_time:talk>before", verdict="keep", reason="倒敘"),
                              DoubtRuling(id="late_subject:words:nope", verdict="change")])
    assert runtime.repo.production(production).rulings == []


def previewed(project_id: str) -> None:
    """Preview the project as it is now, finished, so a final render is not refused for want of one."""
    started = rendering.render_project(project_id, is_preview=True)
    runtime.repo.update_job(started["job_id"], lambda job: setattr(job, "status", JobStatus.COMPLETED))


def final(project_id: str):
    """Start a final render through the tool, letting every finding of the check through."""
    from app.engine.delivery import CHECKS
    return rendering.render_project(project_id, is_preview=False, allow=list(CHECKS))


def keep_all(project_id: str) -> None:
    """Rule every nomination keep, with a reason."""
    rule_doubts(project_id, [DoubtRuling(id=doubt["id"], verdict="keep", reason="開場故意的倒敘")
                             for doubt in story_doubts(project_id)["doubts"]])


def test_a_final_render_of_an_auto_production_waits_until_every_nomination_is_kept(by_hand: str) -> None:
    start_production(mode="auto", brief="直接做完", asset_ids=[], project_id=by_hand)
    previewed(by_hand)
    with pytest.raises(ValueError) as refused:
        final(by_hand)
    assert "closing_too_early:talk (not ruled)" in str(refused.value)
    assert "back_in_time:talk>before" in str(refused.value)
    # Previews are never held up.
    rendering.render_project(by_hand, is_preview=True)

    rule_doubts(by_hand, [DoubtRuling(id="closing_too_early:talk", verdict="change", reason="把晚安剪掉")])
    with pytest.raises(ValueError, match="ruled change, and still there"):
        final(by_hand)

    keep_all(by_hand)
    assert story_doubts(by_hand)["unruled"] == []
    assert final(by_hand)["job_id"]


def test_reordering_the_cut_undoes_the_rulings_but_captions_do_not(by_hand: str) -> None:
    start_production(mode="auto", brief="直接做完", asset_ids=[], project_id=by_hand)
    keep_all(by_hand)
    sequence = story_doubts(by_hand)["sequence"]
    edit(by_hand, [{"action": "add_subtitle", "text": "大家晚安", "timeline_start": 3, "timeline_end": 4}])
    assert story_doubts(by_hand)["sequence"] == sequence and story_doubts(by_hand)["unruled"] == []

    # Moved in the editor: the same clips, another order.
    edit(by_hand, [{"action": "reorder_clip", "track_id": "main", "clip_id": "before", "before_clip_id": "talk"}])
    after = story_doubts(by_hand)
    assert after["sequence"] != sequence
    stale = [doubt for doubt in after["doubts"] if doubt["ruling"] is not None]
    assert stale and all(not doubt["ruling"]["on_this_sequence"] for doubt in stale)
    assert after["unruled"]
    previewed(by_hand)
    with pytest.raises(ValueError, match="ruled on the cut before it changed"):
        final(by_hand)


def test_an_interactive_production_or_none_is_not_held_to_the_story(by_hand: str) -> None:
    previewed(by_hand)
    assert final(by_hand)["job_id"]


def test_the_sequence_hash_follows_order_and_ranges_only() -> None:
    one = narrative.sequence_hash([("a", 0.0, 5.0, None), ("b", 0.0, 3.0, None)])
    assert one == narrative.sequence_hash([("a", 0.0, 5.0, None), ("b", 0.0, 3.0, None)])
    assert one != narrative.sequence_hash([("b", 0.0, 3.0, None), ("a", 0.0, 5.0, None)])
    assert one != narrative.sequence_hash([("a", 0.0, 4.5, None), ("b", 0.0, 3.0, None)])


def test_a_compiled_cut_reads_topics_and_facts_through_the_sections_its_pieces_belong_to(media: Path) -> None:
    from app.models.plan import Beat, EditPlan, Established, Fact, Selection
    from app.models.semantic import ClipLevel, SectionChoice
    from app.tools.planning import compile_plan, save_plan
    from app.tools.timelines import build_semantic_timeline, set_sections

    asset_id = import_asset(str(media / "wide.mp4"))["id"]
    runtime.repo.save_analysis(analysis(
        10.0, segments=[(1.0, 2.5, "杯架一抽就掉"), (4.0, 5.5, "中午吃什麼"), (7.5, 9.5, "這是我做的杯架")],
        silences=[(0.0, 1.0), (2.5, 4.0), (5.5, 7.5), (9.5, 10.0)], asset_id=asset_id))
    built = build_semantic_timeline([asset_id], rebuild=True)
    every = runtime.repo.query_semantic_clips(built["timeline_id"])
    every = sorted((clip for clip in every if clip.level != ClipLevel.SECTION), key=lambda clip: clip.source_range.start)
    # Talk about the cup rack, a stretch about lunch, and back: split where the pauses are.
    parts = [[clip for clip in every if clip.source_range.start < 3.9],
             [clip for clip in every if 3.9 <= clip.source_range.start < 7.4],
             [clip for clip in every if clip.source_range.start >= 7.4]]
    set_sections(timeline_id=built["timeline_id"], sections=[
        SectionChoice(first_clip_id=part[0].id, last_clip_id=part[-1].id, name=name, topic=topic)
        for part, name, topic in zip(parts, ("開場", "離題", "回來"), ("杯架", "午餐", "杯架"))
    ])
    sections = [clip for clip in runtime.repo.query_semantic_clips(built["timeline_id"])
                if clip.level == ClipLevel.SECTION]
    sections.sort(key=lambda clip: clip.source_range.start)
    plan = save_plan(EditPlan(
        timeline_id=built["timeline_id"], timeline_input_hash="", goal="杯架",
        beats=[Beat(id="b1", name="主體")], facts=[Fact(id="subject", says="這支在講杯架")],
        selections=[
            Selection(clip_id=sections[0].id, beat_id="b1"),
            # The plan's own check wants a reason for the stray; the story check still puts it to the model.
            Selection(clip_id=sections[1].id, beat_id="b1", jump_reason="對照"),
            Selection(clip_id=sections[2].id, beat_id="b1",
                      establishes=[Established(fact="subject", evidence=(7.5, 9.5))]),
        ],
    ))
    project = create_project(name="編譯", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=plan["plan_id"])

    found = story_doubts(project)
    assert {doubt["kind"] for doubt in found["doubts"]} >= {"topic_returns", "late_subject"}
    assert any(doubt["id"].startswith("late_subject:fact:") for doubt in found["doubts"])
    assert "topic_returns" not in found["not_checked"] and "late_subject.facts" not in found["not_checked"]

