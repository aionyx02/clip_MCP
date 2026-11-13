"""Productions: a request to make a video, kept so a new conversation can pick it up.

Also the bindings the other tools make through it: a plan saved for a production,
a project made for one, a compile that carries a production's plan onto a
project, and a render that names the production it was made for. Only unfinished
productions take part; one that is finished is left as it was.
"""

from typing import List, Optional

from pydantic import ValidationError

from app.engine import production as progress
from app.models.comment import USER
from app.models.production import Mode, Production, Said, Shape, Story, StoryConfirmed
from app.storage import housekeeping
from app.workspace import output_dir
from app import runtime
from app.runtime import mcp
from app.tools import feedback, rendering

# ------------------------------------------------------------------ finding them

def _ended(production: Production) -> Optional[str]:
    """Whether a production is finished and how, as `engine.production.ended` says."""
    project = runtime.repo.get_project(production.project_id) if production.project_id else None
    renders = runtime.repo.renders_of(production.project_id) if project is not None else []
    return progress.ended(production, project, renders)

def unfinished() -> List[Production]:
    """Every production still going, oldest first."""
    return [production for production in runtime.repo.productions() if _ended(production) is None]

def of_project(project_id: str) -> Optional[Production]:
    """The unfinished production bound to a project; there is never more than one."""
    return next((production for production in unfinished() if production.project_id == project_id), None)

def of_plan(plan_id: str) -> List[Production]:
    """The unfinished productions whose plan it is."""
    return [production for production in unfinished() if production.plan_id == plan_id]

def going(production_id: str) -> Production:
    """Read a production that has not finished.

    Raises:
        ValueError: If there is none by that ID, or it has finished.
    """
    production = runtime.repo.production(production_id)
    if production is None:
        raise ValueError(f"there is no production {production_id}; get_production lists the unfinished ones")
    why = _ended(production)
    if why is not None:
        raise ValueError(f"production {production_id} has ended ({why}); start_production begins a new one")
    return production

def _taken(project_id: str, production_id: str = "") -> None:
    """Refuse a second unfinished production on one project.

    Raises:
        ValueError: If another unfinished production is bound to it.
    """
    owner = of_project(project_id)
    if owner is not None and owner.id != production_id:
        raise ValueError(f"project {project_id} already has an unfinished production, {owner.id} "
                         f"(「{owner.brief[:30]}」): a project has one at a time. Carry on with that one, or "
                         "close_production it in the user's words if they called it off")

# ------------------------------------------------------------------ what the other tools bind through it

def bind_project(production_id: str, project_id: str) -> None:
    """Bind a production to the project just made for it."""
    production = going(production_id)
    runtime.repo.save_production(production.model_copy(update={"project_id": project_id}))

def check_project_free(production_id: str) -> None:
    """Refuse to make a project for a production that already has one, before the project is made.

    Raises:
        ValueError: If the production is not going, or is bound to a project already.
    """
    production = going(production_id)
    if production.project_id is not None:
        raise ValueError(f"production {production_id} is already bound to project {production.project_id}; "
                         "it has one project. Carry on there, or start a new production for another video")

def bind_plan(production_id: str, plan_id: str) -> None:
    """Bind a production to the plan just saved for it."""
    production = going(production_id)
    runtime.repo.save_production(production.model_copy(update={"plan_id": plan_id}))

def compile_refusal(plan_id: str, project_id: str) -> Optional[str]:
    """Say why a plan may not be compiled onto a project, before anything is changed.

    An `auto` production bound to one project is refused another: a new
    project is a way round the questions its own still has to have answered.
    An `interactive` one is neither bound nor refused, since the user is
    there; a branch for Reels is cut that way. A production with no project
    yet takes this one, unless the project already has another.

    Args:
        plan_id: The plan.
        project_id: The project it is to be compiled onto.

    Returns:
        The reason, or None when it may go ahead.
    """
    owner = of_project(project_id)
    binding = []
    for production in of_plan(plan_id):
        if production.project_id is None:
            binding.append(production)
        elif production.project_id != project_id and production.mode == "auto":
            return (f"this plan belongs to production {production.id}, which is finishing project "
                    f"{production.project_id} on its own; it is not compiled onto another project. Branching "
                    "or a second video waits until the user has said so")
    claimants = {production.id for production in binding} | ({owner.id} if owner else set())
    if len(claimants) > 1:
        return (f"project {project_id} would be bound to more than one unfinished production "
                f"({', '.join(sorted(claimants))}); a project has one at a time")
    return None

def bind_compiled(plan_id: str, project_id: str) -> None:
    """After a compile, bind each side to the other.

    A production whose plan it is and that has no project yet takes the
    project; the production of the project takes the plan, which is how one
    started on an existing project, amended with `amend_plan`, learns its plan.

    Args:
        plan_id: The plan compiled.
        project_id: The project it was compiled onto.
    """
    for production in of_plan(plan_id):
        if production.project_id is None:
            runtime.repo.save_production(production.model_copy(update={"project_id": project_id}))
    owner = of_project(project_id)
    if owner is not None and owner.plan_id != plan_id:
        runtime.repo.save_production(owner.model_copy(update={"plan_id": plan_id}))

# ------------------------------------------------------------------ how far it has got

def _footing(production: Production) -> progress.Footing:
    """Gather what is stored about a production."""
    project = runtime.repo.get_project(production.project_id) if production.project_id else None
    base = project.base_video_track if project is not None else None
    plan = runtime.repo.get_plan(production.plan_id) if production.plan_id else None
    if plan is None and project is not None:
        plan = rendering.compiled_plan(project)
    comments = [comment for comment in (runtime.repo.comments(project.id) if project is not None else [])
                if comment.status == "open"]
    return progress.Footing(
        analysed=frozenset(asset_id for asset_id in production.asset_ids
                           if runtime.repo.get_analysis(asset_id) is not None),
        timelines=tuple(frozenset(listed[0].asset_ids) for listed in runtime.repo.list_semantic_timelines()),
        plan_id=plan.id if plan is not None else None,
        project=project,
        cut=bool(base and base.clips),
        picture=rendering.picture_key(project) if project is not None else None,
        renders=tuple(runtime.repo.renders_of(project.id)) if project is not None else (),
        questions=tuple(comment.id for comment in comments if comment.question and not comment.answered),
        opened=tuple(comment.id for comment in comments if comment.by == USER),
        texts={comment.id: comment.text for comment in comments},
    )

def owed(production: Production) -> List[dict]:
    """What is still owed between the AI and the user on a production, as `engine.production.pending` says."""
    return progress.pending(production, _footing(production))

def _gone(project_id: str) -> str:
    """Say where a production's project went: the trash, or out of it for good."""
    for item in housekeeping.in_trash(str(output_dir())):
        if item.kind == "project" and item.project_id == project_id:
            return (f"the project 「{item.name}」 was moved to the trash; it can be put back from the editor's "
                    "trash, and the production stays ended")
    return "the project no longer exists and is not in the trash; the production stays ended"

def _story_of(production: Production) -> Optional[dict]:
    """The story card as the AI reads it back, with its fingerprint and whether the user said yes to it."""
    if production.story is None and production.story_confirmed_at is None:
        return None
    confirmed = production.story_confirmed
    return {
        **(production.story.model_dump() if production.story is not None else {}),
        "story_hash": production.story.digest() if production.story is not None else None,
        "confirmed": progress.story_confirmed(production),
        "confirmed_in": confirmed.said if confirmed is not None and progress.story_confirmed(production) else None,
        "last_confirmed_at": production.story_confirmed_at.isoformat() if production.story_confirmed_at else None,
    }

def said_of(production: Production) -> dict:
    """A production as the AI reads it: what was asked, how far it has got, and what comes next."""
    footing = _footing(production)
    worked = progress.progress(production, footing)
    project = footing.project
    result = {
        "production_id": production.id, "mode": production.mode, "brief": production.brief,
        "asset_ids": production.asset_ids, "outputs": production.outputs,
        "project": {"id": project.id, "name": project.name, "version": project.version} if project else None,
        "plan_id": footing.plan_id,
        "story": _story_of(production),
        "said": [{"said": said.said, "at": said.at.isoformat()} for said in production.said],
        "started_at": production.created_at.isoformat(),
        **worked,
        "next": None if worked["ended"] or worked["stage"] is None else progress.NEXT[worked["stage"]],
    }
    previews = [job for job in footing.renders if job.preview]
    if previews:
        latest = max(previews, key=lambda job: job.created_at)
        result["latest_preview"] = {"job_id": latest.job_id, "output_path": latest.output_path,
                                    "shows_the_cut_now": latest.picture == footing.picture}
    if worked["stage"] == "analysed":
        result["not_analysed"] = sorted(set(production.asset_ids) - footing.analysed)
    if production.project_id is None:
        result["unbound"] = ("not bound to a project yet: pass production_id to create_project, or compile its "
                             "plan (saved with production_id) onto a project; until then nothing it is for is "
                             "held to it")
    if worked["ended"] == "project_gone":
        result["project_gone"] = _gone(production.project_id)
    if production.closed is not None:
        result["closed"] = {"said": production.closed.said, "at": production.closed.at.isoformat()}
    return result

# ------------------------------------------------------------------ the tools

@mcp.tool()
def start_production(
    mode: Mode,
    brief: str,
    asset_ids: List[str],
    outputs: Optional[List[Optional[Shape]]] = None,
    project_id: Optional[str] = None,
) -> dict:
    """Keep a request to make a video, so any later conversation can pick it up.

    Call it with `auto` when the user says to finish on your own —
    「直接做完」; with `interactive` once the footage is read and before the
    story is put to them. For 「把這支直接做完」 pass the existing project's
    `project_id`. Then pass the `production_id` to `save_plan` and
    `create_project`: a production that is not bound does not hold the work
    to it. How far it has got is never kept; `get_production` works it out
    from the footage, the plan, the project and its renders each time.

    Args:
        mode: `auto` or `interactive`.
        brief: What the user asked for, in their words.
        asset_ids: The footage it is made from; may be empty when
            `project_id` is given.
        outputs: The shapes to deliver — `landscape`, `portrait`, `square`,
            or null for the project's own; omit for the project's own only.
        project_id: An existing project to finish.

    Returns:
        The production as `get_production` gives it.

    Raises:
        ValueError: If the brief is empty, a file is not in the library, there
            is neither footage nor a project, or the project does not exist or
            already has an unfinished production.
    """
    if not brief.strip():
        raise ValueError("a production needs the user's request, in their words")
    missing = sorted(set(asset_ids) - set(runtime.repo.get_assets(asset_ids)))
    if missing:
        raise ValueError(f"not in the library: {', '.join(missing)}; import_asset them first")
    if not asset_ids and project_id is None:
        raise ValueError("a production is made from footage or finishes a project: give asset_ids or project_id")
    if project_id is not None:
        if runtime.repo.get_project(project_id) is None:
            raise ValueError(f"project {project_id} not found")
        _taken(project_id)
    shapes = list(dict.fromkeys(outputs)) if outputs else [None]
    production = Production(mode=mode, brief=brief.strip(), asset_ids=list(dict.fromkeys(asset_ids)),
                            outputs=shapes, project_id=project_id)
    runtime.repo.save_production(production)
    return said_of(production)

@mcp.tool()
def get_production(production_id: Optional[str] = None) -> dict:
    """Read what was asked and how far it has got; with no ID, every production still going.

    When the user wants to carry on — 「繼續上次那支」 — call this first,
    before anything else: it has the request in their words, the footage,
    the project and plan, which step comes next, and what they said along
    the way. A finished one says how it ended: `delivered`, `closed`, or
    `project_gone`, with whether the project is in the trash.

    Args:
        production_id: The production; omit to list every unfinished one.

    Returns:
        One production: `brief`, `mode`, `outputs`, `project`, `plan_id`,
        `story` (the card, its `story_hash`, whether it is `confirmed` as it
        is now, and `last_confirmed_at`), `steps` (each and whether it is
        done), `stage` (the next not done), `next` (what to do about it),
        `pending` (what is still owed between you and the user, in order: in
        an `auto` one an unconfirmed or missing story card first, then the
        unanswered questions, then comments the user wrote that are still
        open — each with its `comment_id` and `text`; while an `auto` one
        has any, its final render is refused), `missing` (shapes not
        delivered yet),
        `latest_preview`, `ended`, and `unbound` while it has no project.
        With no ID, `productions`: each unfinished one the same way.

    Raises:
        ValueError: If there is no production by that ID.
    """
    if production_id is None:
        return {"productions": [said_of(production) for production in unfinished()]}
    production = runtime.repo.production(production_id)
    if production is None:
        raise ValueError(f"there is no production {production_id}; call get_production without an ID to list them")
    return said_of(production)

@mcp.tool()
def set_production_mode(production_id: str, mode: Mode, said: str) -> dict:
    """Switch a production between `auto` and `interactive` when the user changes their mind.

    Args:
        production_id: The production.
        mode: The new mode.
        said: What the user said, in their words; kept with the production.

    Returns:
        The production as `get_production` gives it.

    Raises:
        ValueError: If it does not exist or has ended, or `said` is empty.
    """
    production = going(production_id)
    if not said.strip():
        raise ValueError("pass what the user said, in their words")
    changed = production.model_copy(update={"mode": mode, "said": [*production.said, Said(said=said.strip())]})
    runtime.repo.save_production(changed)
    return said_of(changed)

@mcp.tool()
def close_production(production_id: str, said: str) -> dict:
    """Call a production off, only when the user has said to.

    Never to tidy up, and never when it is finished: a finished video ends
    it by itself. Its project and plan are left as they are.

    Args:
        production_id: The production.
        said: The user's own words calling it off; required.

    Returns:
        The production as `get_production` gives it, `ended: closed`.

    Raises:
        ValueError: If it does not exist or has ended, or the user's words are missing.
    """
    production = going(production_id)
    if not said.strip():
        raise ValueError("a production is called off only in the user's words: pass what they said")
    closed = production.model_copy(update={"closed": Said(said=said.strip())})
    runtime.repo.save_production(closed)
    return said_of(closed)

@mcp.tool()
def set_story(
    production_id: str,
    theme: str,
    structure: str,
    opening: str,
    turn: str,
    ending: str,
    left_out: str,
    length: str,
) -> dict:
    """Write the story card: the story the video tells, before the plan is written.

    Write each part so the user could say it is wrong — name the moments in
    the footage, not the mood: 「開場是第一個印壞的杯架」, not 「輕鬆有趣」. A
    card that says something different from the one there clears the user's
    yes to it; the same card again keeps it. Put it to the user and, once
    they say yes, `confirm_story` with the `story_hash` returned here.

    Args:
        production_id: The production.
        theme: The video in one sentence.
        structure: How it is built — 起承轉合, 清單式, 教學步驟 — and in what order.
        opening: How it opens: the first moment on screen.
        turn: Where it turns: the moment things change.
        ending: How it ends: the last moment on screen.
        left_out: What is deliberately left out, and why.
        length: How long it will run, roughly — 「約 60 秒」.

    Returns:
        The production as `get_production` gives it, with `story_hash`, and
        `cleared` when this change took away a yes the user had given.

    Raises:
        ValueError: If the production does not exist or has ended, or a part is empty.
    """
    production = going(production_id)
    try:
        story = Story(theme=theme, structure=structure, opening=opening, turn=turn, ending=ending,
                      left_out=left_out, length=length)
    except ValidationError as error:
        raise ValueError("; ".join(str(problem["msg"]).removeprefix("Value error, ")
                                   for problem in error.errors())) from None
    same = production.story is not None and production.story.digest() == story.digest()
    cleared = not same and progress.story_confirmed(production)
    changed = production.model_copy(update={
        "story": story, "story_confirmed": production.story_confirmed if same else None,
    })
    runtime.repo.save_production(changed)
    result = {**said_of(changed), "story_hash": story.digest()}
    if cleared:
        result["cleared"] = ("the card changed, so the user's yes to the old one no longer holds: put this one "
                             "to them and confirm_story again")
    return result

@mcp.tool()
def confirm_story(production_id: str, story_hash: str, said: str) -> dict:
    """Keep the user's yes to the story card, only once they have said it.

    The `story_hash` is the one `set_story` or `get_production` gave for the
    card the user was shown. It is refused when the card has changed since:
    a yes to an earlier card is not a yes to this one, so put the new one to
    them first. Never confirm on the user's behalf.

    Args:
        production_id: The production.
        story_hash: The fingerprint of the card the user said yes to.
        said: The user's own words saying yes — 「好，就這樣剪」.

    Returns:
        The production as `get_production` gives it.

    Raises:
        ValueError: If the production does not exist or has ended, there is
            no card, the hash is not the card's as it is now, or the user's
            words are missing.
    """
    production = going(production_id)
    if production.story is None:
        raise ValueError("there is no story card to confirm: write it with set_story and put it to the user first")
    if not said.strip():
        raise ValueError("a story card is confirmed only in the user's words: pass what they said")
    now = production.story.digest()
    if story_hash != now:
        raise ValueError(f"story_hash {story_hash} is not the card as it is now ({now}): the user was shown "
                         "another card, or it changed after they saw it. Put the card as it is now to them, and confirm with its hash once "
                         "they say yes")
    yes = StoryConfirmed(story_hash=now, said=said.strip())
    last = production.story_confirmed_at
    confirmed = production.model_copy(update={
        "story_confirmed": yes, "story_confirmed_at": yes.at if last is None else max(last, yes.at),
    })
    runtime.repo.save_production(confirmed)
    return said_of(confirmed)

@mcp.tool()
def record_answer(production_id: str, said: str, comment_id: Optional[str] = None) -> dict:
    """Write down what the user said in the chat in answer, in their own words.

    For when the user answers in the conversation instead of in the editor.
    With `comment_id` — one of `pending`'s questions, or a comment of theirs
    they added to — it goes on that comment as theirs, so the question is
    answered just as a reply in the editor would make it. Without, it is kept
    with the production: 「再短一點」, said about the whole video. Pass
    exactly what they said: never answer on their behalf, never sum it up or
    rewrite it, and never pass what you decided yourself. Whatever is
    written here is taken as the user's words, as `allow` is.

    Args:
        production_id: The production.
        said: The user's words, as they said them.
        comment_id: The comment it answers, from `pending` or `get_comments`;
            omit when it answers none.

    Returns:
        The production as `get_production` gives it, with `comment`, the
        comment as `get_comments` shows it, when one was answered.

    Raises:
        ValueError: If the production does not exist or has ended, the
            user's words are missing, or the comment is not on its project.
    """
    production = going(production_id)
    if not said.strip():
        raise ValueError("an answer is written down only in the user's words: pass what they said")
    if comment_id is None:
        kept = production.model_copy(update={"said": [*production.said, Said(said=said.strip())]})
        runtime.repo.save_production(kept)
        return said_of(kept)
    if production.project_id is None:
        raise ValueError(f"production {production_id} has no project yet, so comment {comment_id} is not on it; "
                         "omit comment_id to keep the answer with the production")
    if not any(comment.id == comment_id for comment in runtime.repo.comments(production.project_id)):
        raise ValueError(f"project {production.project_id} has no comment {comment_id}: pass a comment_id from "
                         "get_production's pending or get_comments, or omit it")
    answered = feedback.change_comment(production.project_id, comment_id, USER, said, via="chat")
    return {**said_of(production), "comment": answered}
