"""A production keeps a request to make a video, and works out how far it has got from what is stored.

A new conversation knows nothing of the last one: what was asked, which footage,
how far it got. These tests hold that what was asked survives a restart, that how
far it got is read off the footage, plan, project and renders rather than kept
beside them, that it is finished by a finished video in every shape or by the
project going, and that a production follows its plan onto a project and refuses
to be moved where it would slip its own checks.
"""

from datetime import timedelta
from typing import List, Optional

import pytest
from pydantic import TypeAdapter

from app import runtime
from app.engine import production as progress
from app.models.plan import PlanAmendment
from app.models.job import Job, JobKind, JobStatus
from app.models.production import Production
from app.models.timeline.parts import Project
from app.engine.delivery import CHECKS
from app.storage.repo import Repository
from app.tools import editing, rendering, upkeep, versioning
from app.tools.editing import apply_edits, create_project, get_project
from app.tools.planning import OPERATIONS, amend_plan, compile_plan, save_plan
from app.tools.feedback import ask_user
from app.models.comment import Question
from app.tools.production import (
    close_production, confirm_story, get_production, set_production_mode, set_story, start_production,
)
from edit_plan.shared import planned  # noqa: F401 — the fixture


@pytest.fixture(autouse=True)
def no_workers(monkeypatch) -> None:
    """Keep every render a record: nothing here needs the file, only the job."""
    def kept(job: Job, spec: dict) -> Job:
        runtime.repo.add_job(job)
        return job

    monkeypatch.setattr(runtime.job_manager, "start_job", kept)
    monkeypatch.setattr(rendering, "_make_room", lambda: None)


def asset_of(planned: dict) -> str:
    """The file the plan is cut from."""
    return runtime.repo.get_plan(planned["plan_id"]).timeline_id and \
        runtime.repo.get_semantic_timeline(runtime.repo.get_plan(planned["plan_id"]).timeline_id).asset_ids[0]


def rendered(project_id: str, preview: bool = False, frame: Optional[str] = None,
             status: JobStatus = JobStatus.COMPLETED) -> Job:
    """A render of the project as it is now, ended as asked."""
    project = runtime.repo.get_project(project_id)
    job = Job(kind=JobKind.RENDER, project_id=project_id, project_version=project.version, preview=preview,
              frame=frame, status=status, picture=rendering.picture_key(project))
    runtime.repo.add_job(job)
    return job


def compiled(planned: dict, mode: str = "auto", outputs=None) -> tuple:
    """A production bound to a project its plan has been compiled onto.

    Returns:
        `(production_id, project_id)`.
    """
    production = start_production(mode=mode, brief="把這些剪成一支短片", asset_ids=[asset_of(planned)],
                                  outputs=outputs)["production_id"]
    plan = runtime.repo.get_plan(planned["plan_id"])
    save_plan(plan, production_id=production)
    project = create_project(name="短片", width=640, height=360, production_id=production)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    return production, project


def stage(production_id: str) -> str:
    """How far a production has got."""
    return get_production(production_id)["stage"]


# ------------------------------------------------------------------ how far it has got

def footing(**changes) -> progress.Footing:
    """What is stored, with everything done but what is changed."""
    project = Project(id="p1", name="短片", width=640, height=360)
    base = dict(analysed=frozenset({"a1"}), timelines=(frozenset({"a1"}),), plan_id="plan1", project=project,
                cut=True, picture="pic", renders=(), questions=())
    return progress.Footing(**{**base, **changes})


def test_each_step_is_read_off_what_is_stored() -> None:
    made = Production(mode="auto", brief="剪短", asset_ids=["a1"], project_id="p1")
    # Before there is a cut, every step up to it counts.
    assert progress.progress(made, footing(analysed=frozenset(), cut=False))["stage"] == "analysed"
    assert progress.progress(made, footing(timelines=(frozenset({"other"}),), cut=False))["stage"] == "timeline"
    assert progress.progress(made, footing(plan_id=None, project=None, cut=False))["stage"] == "plan"
    assert progress.progress(made, footing(project=None, cut=False))["stage"] == "project"
    assert progress.progress(made, footing(cut=False))["stage"] == "cut"
    assert progress.progress(made, footing())["stage"] == "previewed"
    preview = Job(kind=JobKind.RENDER, project_id="p1", project_version=1, preview=True, picture="pic",
                  status=JobStatus.COMPLETED, created_at=made.created_at + timedelta(seconds=1))
    assert progress.progress(made, footing(renders=(preview,), questions=("c1",)))["stage"] == "answered"
    assert progress.progress(made, footing(renders=(preview,)))["stage"] == "delivered"


def test_a_cut_already_on_the_project_needs_no_plan_before_it() -> None:
    made = Production(mode="auto", brief="把這支直接做完", project_id="p1")
    assert progress.progress(made, footing(plan_id=None, timelines=()))["stage"] == "previewed"


def test_progress_follows_the_production_through_every_step(planned: dict) -> None:
    production = start_production(mode="auto", brief="把這些剪成一支短片",
                                  asset_ids=[asset_of(planned)])["production_id"]
    said = get_production(production)
    assert said["stage"] == "plan" and said["project"] is None and "not bound to a project" in said["unbound"]

    save_plan(runtime.repo.get_plan(planned["plan_id"]), production_id=production)
    # The plan is saved and no project is made yet: only the production knows which plan it is.
    assert stage(production) == "project"
    assert get_production(production)["plan_id"] == planned["plan_id"]

    project = create_project(name="短片", width=640, height=360, production_id=production)["id"]
    assert stage(production) == "cut"
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    assert stage(production) == "previewed"
    rendered(project, preview=True)
    assert stage(production) == "delivered"
    assert "unbound" not in get_production(production)


def test_an_edit_made_in_the_editor_moves_the_progress_back(planned: dict) -> None:
    production, project = compiled(planned)
    rendered(project, preview=True)
    assert stage(production) == "delivered"

    # The editor edits through the same function the AI does, with no production in sight.
    state = get_project(project)
    last = state["tracks"][0]["clips"][-1]["id"]
    apply_edits(project, state["version"], OPERATIONS.validate_python([
        {"action": "delete_clip", "track_id": "main", "clip_id": last},
    ]))
    assert stage(production) == "previewed"


def test_a_production_and_how_far_it_got_survive_a_restart(planned: dict, monkeypatch) -> None:
    production, project = compiled(planned)
    monkeypatch.setattr(runtime, "repo", Repository(runtime.repo.db_path))
    said = get_production(production)
    assert said["brief"] == "把這些剪成一支短片" and said["mode"] == "auto"
    assert said["project"]["id"] == project and said["stage"] == "previewed"
    assert production in [listed["production_id"] for listed in get_production()["productions"]]


# ------------------------------------------------------------------ when it is finished

def test_a_finished_video_ends_it_and_the_project_can_start_another(planned: dict) -> None:
    production, project = compiled(planned)
    with pytest.raises(ValueError, match="already"):
        start_production(mode="interactive", brief="再一支", asset_ids=[], project_id=project)
    rendered(project, preview=True)
    rendered(project)
    assert get_production(production)["ended"] == "delivered"
    assert production not in [listed["production_id"] for listed in get_production()["productions"]]
    again = start_production(mode="interactive", brief="再一支", asset_ids=[], project_id=project)
    assert again["project"]["id"] == project


def test_a_video_exported_from_the_editor_ends_it_too(planned: dict) -> None:
    production, project = compiled(planned)
    # How the editor exports: the shared function, never the tool, with no production named.
    started = rendering.start_render(project, False, None, False, None, True, list(CHECKS),
                                     rendering.PREVIEW_SHORT_SIDE)
    assert runtime.repo.get_job(started["job_id"]).production_id is None
    runtime.repo.update_job(started["job_id"], lambda job: setattr(job, "status", JobStatus.COMPLETED))
    assert get_production(production)["ended"] == "delivered"


def test_one_shape_of_two_does_not_end_it(planned: dict) -> None:
    production, project = compiled(planned, outputs=[None, "portrait"])
    rendered(project, preview=True)
    rendered(project)
    said = get_production(production)
    assert said["ended"] is None and said["stage"] == "delivered" and said["missing"] == ["portrait"]
    rendered(project, frame="portrait")
    assert get_production(production)["ended"] == "delivered"


def test_a_failed_or_cancelled_final_does_not_end_it(planned: dict) -> None:
    production, project = compiled(planned)
    rendered(project, status=JobStatus.FAILED)
    rendered(project, status=JobStatus.CANCELLED)
    assert get_production(production)["ended"] is None


def test_a_landscape_project_rendered_as_it_is_delivers_landscape(planned: dict) -> None:
    production, project = compiled(planned, outputs=["landscape"])
    rendered(project, frame=None)
    assert get_production(production)["ended"] == "delivered"


def test_a_final_once_made_keeps_it_ended_after_another_edit(planned: dict) -> None:
    production, project = compiled(planned)
    rendered(project)
    state = get_project(project)
    apply_edits(project, state["version"], OPERATIONS.validate_python([
        {"action": "delete_clip", "track_id": "main", "clip_id": state["tracks"][0]["clips"][-1]["id"]},
    ]))
    assert get_production(production)["ended"] == "delivered"


def test_a_project_put_in_the_trash_ends_it_and_says_so(planned: dict) -> None:
    production, project = compiled(planned)
    assert upkeep.trash_projects([project])["deleted"] == [project]
    said = get_production(production)
    assert said["ended"] == "project_gone"
    assert "trash" in said["project_gone"]


# ------------------------------------------------------------------ binding it to a plan and a project

def test_a_production_started_on_a_project_learns_its_plan_when_it_is_compiled(planned: dict) -> None:
    project = create_project(name="既有", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    production = start_production(mode="auto", brief="把這支直接做完", asset_ids=[],
                                  project_id=project)["production_id"]
    assert get_production(production)["project"]["id"] == project
    assert runtime.repo.production(production).plan_id is None

    plan = runtime.repo.get_plan(planned["plan_id"])
    amend_plan(planned["plan_id"], plan.version, TypeAdapter(List[PlanAmendment]).validate_python([
        {"action": "set_target", "target": {"seconds": 8}},
    ]))
    compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=planned["plan_id"])
    assert runtime.repo.production(production).plan_id == planned["plan_id"]


def test_compiling_a_production_s_plan_binds_the_project(planned: dict) -> None:
    production = start_production(mode="auto", brief="剪短", asset_ids=[asset_of(planned)])["production_id"]
    save_plan(runtime.repo.get_plan(planned["plan_id"]), production_id=production)
    project = create_project(name="短片", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    assert runtime.repo.production(production).project_id == project


def test_an_auto_production_cannot_be_compiled_onto_another_project(planned: dict) -> None:
    production, project = compiled(planned)
    other = create_project(name="另一支", width=640, height=360)["id"]
    before = runtime.repo.production(production)
    with pytest.raises(ValueError, match="another project"):
        compile_plan(project_id=other, expected_version=1, plan_id=planned["plan_id"])
    assert get_project(other)["version"] == 1 and get_project(other)["tracks"] == []
    assert runtime.repo.production(production) == before


def test_an_interactive_production_s_plan_compiles_onto_its_branch(kept_history, planned: dict) -> None:
    production, project = compiled(planned, mode="interactive")
    # From before the cut, so the plan's own pieces are all the compile lays down.
    first = runtime.history.versions_of("projects/", f"{project}.json", 500)[-1].commit
    branch = versioning.branch_project(project, "Reels 版", commit=first)["project_id"]
    compile_plan(project_id=branch, expected_version=1, plan_id=planned["plan_id"])
    assert get_project(branch)["tracks"][0]["clips"]
    assert runtime.repo.production(production).project_id == project


def test_a_finished_production_s_plan_compiles_onto_a_new_project(planned: dict) -> None:
    production, project = compiled(planned)
    rendered(project)
    fresh = create_project(name="新的", width=640, height=360)["id"]
    compile_plan(project_id=fresh, expected_version=1, plan_id=planned["plan_id"])
    assert runtime.repo.production(production).project_id == project


def test_a_project_has_one_unfinished_production_at_a_time(planned: dict) -> None:
    production, project = compiled(planned)
    with pytest.raises(ValueError, match="already"):
        start_production(mode="auto", brief="又一支", asset_ids=[], project_id=project)
    second = start_production(mode="auto", brief="又一支", asset_ids=[asset_of(planned)])["production_id"]
    with pytest.raises(ValueError, match="already"):
        create_project(name="不會建", production_id=production)
    assert runtime.repo.production(second).project_id is None


# ------------------------------------------------------------------ the user's words, and what is refused

def test_calling_off_a_production_needs_the_user_s_words(planned: dict) -> None:
    production, _ = compiled(planned)
    with pytest.raises(ValueError, match="words"):
        close_production(production, said="  ")
    assert get_production(production)["ended"] is None
    close_production(production, said="這支先不要了")
    assert get_production(production)["ended"] == "closed"
    with pytest.raises(ValueError, match="ended"):
        set_production_mode(production, mode="interactive", said="還是我自己來")


def test_changing_the_mode_keeps_what_the_user_said(planned: dict) -> None:
    production, _ = compiled(planned)
    set_production_mode(production, mode="interactive", said="我想每一步都看")
    kept = runtime.repo.production(production)
    assert kept.mode == "interactive" and [said.said for said in kept.said] == ["我想每一步都看"]


def test_an_auto_production_refuses_a_branch_through_the_tool_but_not_the_editor(kept_history,
                                                                                 planned: dict) -> None:
    _, project = compiled(planned)
    with pytest.raises(ValueError, match="auto"):
        versioning.branch_project(project, "偷跑的分支")
    made = versioning.start_branch(project, "使用者自己開的")
    assert runtime.repo.get_project(made["project_id"]) is not None


def test_a_render_through_the_tool_names_the_unfinished_production(planned: dict) -> None:
    production, project = compiled(planned, mode="interactive")
    started = rendering.render_project(project, is_preview=True)
    assert runtime.repo.get_job(started["job_id"]).production_id == production


def test_a_project_with_no_production_renders_as_it_did(planned: dict) -> None:
    project = create_project(name="沒有製作", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    started = rendering.render_project(project, is_preview=True)
    assert runtime.repo.get_job(started["job_id"]).production_id is None
    assert editing.get_project(project)["id"] == project


# ------------------------------------------------------------------ the story card

CARD = dict(theme="第一次自己印杯架，從失敗到成功", structure="起承轉合，照時間順序",
            opening="開場是印壞、歪掉的第一個杯架", turn="杯子終於卡進去的那一下",
            ending="一整排杯子掛在架上", left_out="切片設定的畫面：太長，觀眾不需要", length="約 60 秒")


def with_story(planned: dict, mode: str = "auto") -> tuple:
    """A production with the card written.

    Returns:
        `(production_id, story_hash)`.
    """
    production = start_production(mode=mode, brief="把這些剪成一支短片", asset_ids=[asset_of(planned)])
    written = set_story(production["production_id"], **CARD)
    return production["production_id"], written["story_hash"]


@pytest.mark.parametrize("part", sorted(CARD))
def test_a_story_card_with_a_part_left_empty_is_refused(planned: dict, part: str) -> None:
    production = start_production(mode="auto", brief="剪短", asset_ids=[asset_of(planned)])["production_id"]
    with pytest.raises(ValueError, match=f"story's {part} is empty"):
        set_story(production, **{**CARD, part: "   "})
    assert get_production(production)["story"] is None


def test_a_yes_is_kept_in_the_user_s_words_and_holds_the_card(planned: dict) -> None:
    production, story_hash = with_story(planned)
    said = confirm_story(production, story_hash, said="好，就這樣剪")
    assert said["story"]["confirmed"] and said["story"]["confirmed_in"] == "好，就這樣剪"
    assert said["story"]["last_confirmed_at"] is not None
    with pytest.raises(ValueError, match="user's words"):
        confirm_story(production, story_hash, said="  ")


def test_changing_the_card_clears_the_yes_but_not_when_it_was_given(planned: dict) -> None:
    production, story_hash = with_story(planned)
    confirmed_at = confirm_story(production, story_hash, said="好")["story"]["last_confirmed_at"]

    changed = set_story(production, **{**CARD, "ending": "把杯架送給朋友"})
    assert "cleared" in changed and changed["story_hash"] != story_hash
    assert not changed["story"]["confirmed"] and changed["story"]["confirmed_in"] is None
    # When the user last said yes stays: it is when they were last heard from.
    assert changed["story"]["last_confirmed_at"] == confirmed_at
    stored = runtime.repo.production(production)
    assert stored.story_confirmed is None and stored.story_confirmed_at.isoformat() == confirmed_at


def test_the_same_card_written_again_keeps_the_yes(planned: dict) -> None:
    production, story_hash = with_story(planned)
    confirm_story(production, story_hash, said="好")
    again = set_story(production, **{**CARD, "turn": "  " + CARD["turn"] + "  "})
    assert again["story_hash"] == story_hash and "cleared" not in again
    assert again["story"]["confirmed"]


def test_a_yes_to_an_earlier_card_is_refused(planned: dict) -> None:
    production, old_hash = with_story(planned)
    set_story(production, **{**CARD, "opening": "開場是一整排杯子"})
    with pytest.raises(ValueError, match="not the card as it is now"):
        confirm_story(production, old_hash, said="好")
    assert not get_production(production)["story"]["confirmed"]


def test_there_is_no_yes_without_a_card(planned: dict) -> None:
    production = start_production(mode="auto", brief="剪短", asset_ids=[asset_of(planned)])["production_id"]
    with pytest.raises(ValueError, match="no story card"):
        confirm_story(production, "0" * 12, said="好")


def kinds(production_id: str) -> List[str]:
    """What is still to be put to the user, in order, by kind."""
    return [item["kind"] for item in get_production(production_id)["pending"]]


def test_an_auto_production_puts_a_missing_or_unconfirmed_card_first(planned: dict) -> None:
    production, project = compiled(planned)
    ask_user(project, [Question(start=0.0, question="開場會不會太長？")])
    # No card at all counts as not confirmed: skipping set_story does not get past it.
    assert kinds(production) == ["story", "question"]
    assert "set_story" in get_production(production)["pending"][0]["message"]

    story_hash = set_story(production, **CARD)["story_hash"]
    assert kinds(production) == ["story", "question"]
    assert "confirm_story" in get_production(production)["pending"][0]["message"]

    confirm_story(production, story_hash, said="可以")
    assert kinds(production) == ["question"]
    set_story(production, **{**CARD, "length": "約 90 秒"})
    assert kinds(production) == ["story", "question"]


def test_an_interactive_production_s_card_is_not_left_pending(planned: dict) -> None:
    production, _ = compiled(planned, mode="interactive")
    assert kinds(production) == []
