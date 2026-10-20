"""What the user says back: comments on moments of a video, trials of alternatives, questions, and preferences."""

import os
import secrets
import shutil
import uuid
from typing import Annotated, List, Literal, Optional

from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field

from app.models.timeline import Project
from app.models.comment import Comment, Preference, Question, Reply
from app.models.trial import Trial, Verdict
from app.engine.probe import probe_file
from app.engine.comments import anchor_at, where_now
from app.engine.delivery import clock
from app.engine.frames import storyboard_sheet
from app.engine.listen import cut_sound
from app import runtime
from app.runtime import mcp
from app.tools import editing, rendering, shared

# Provisional (roadmap §13): a comment dragged over less than this is a moment, not a stretch.
SHORTEST_STRETCH_SECONDS = 0.05

# ------------------------------------------------------------------ what the user said about a moment

def _current_commit(project_id: str) -> Optional[str]:
    """The version a project is at now, as its latest commit; None before it has one."""
    found = runtime.history.versions_of("projects/", f"{project_id}.json", 1)
    return found[0].commit if found else None

def _comment_said(project: Project, comment: Comment) -> dict:
    """A comment as the AI and the editor read it: where it is on the cut now, and what was said back."""
    now = where_now(project, comment)
    asset = runtime.repo.get_assets([comment.start_anchor.asset_id]).get(comment.start_anchor.asset_id) \
        if comment.start_anchor else None
    return {
        "id": comment.id, "text": comment.text, "by": comment.by, "status": comment.status,
        **({"question": True, "answered": comment.answered}
           if comment.question else {}),
        "at": clock(now["start"]) + (f"–{clock(now['end'])}" if now["end"] is not None else ""),
        "start": now["start"], "end": now["end"], "clip_id": now["clip_id"], "gone": now["gone"],
        "file": os.path.basename(asset.path) if asset else None,
        "file_seconds": comment.start_anchor.source if comment.start_anchor else None,
        "written": comment.created_at.isoformat(timespec="seconds"),
        "replies": [{"by": reply.by, "text": reply.text, "at": reply.at.isoformat(timespec="seconds"),
                     "commit": reply.commit[:10] if reply.commit else None}
                    for reply in comment.replies],
        # The version it was dealt with in, to watch what was done about it.
        "resolved_in": comment.resolved_commit[:10] if comment.resolved_commit else None,
    }

# How far either side of the problem a judge sees, for whether the change broke what is
# around it. Provisional.
TRIAL_MARGIN_SECONDS = 2.0

def _render_length(path: str) -> Optional[float]:
    """How long a rendered file runs, or None when it cannot be read."""
    try:
        return float(probe_file(path)["format"]["duration"])
    except (OSError, RuntimeError, KeyError, ValueError):
        return None

def trial_dir(trial_id: str) -> str:
    """Where a trial keeps its own copies of the two renders."""
    return os.path.join(runtime.WORKSPACE_DIR, "trials", trial_id)

@mcp.tool()
def start_trial(
    project_id: str,
    before_job: str,
    problem: Annotated[str, Field(min_length=1)],
    hypothesis: Annotated[str, Field(min_length=1)],
    expected: Annotated[str, Field(min_length=1)],
    at: Annotated[float, Field(ge=0)],
    end: Annotated[float, Field(gt=0)],
    evidence: str = "",
) -> dict:
    """Put a change you are about to make to a blind test, before you make it.

    Whoever makes a change reads the result through their own reason for it,
    and says it worked. So when you change a cut to fix something — your own
    check found it, or the user's feedback — write the problem down first:
    where it is, what shows it, what you will change and what should be better.
    Then change the cut, render a preview and give it with `trial_candidate`;
    the two are handed over blind as X and Y with `trial_pack`; and only the
    verdict in `judge_trial` says whether the change is kept.

    Args:
        project_id: The video.
        before_job: A finished render of the cut before the change. A copy is
            kept, since a newer preview replaces it.
        problem: What is wrong and where, as it was seen or heard.
        hypothesis: What you will change and why it should help. Never shown
            to whoever judges.
        expected: What should be better afterwards. Never shown either.
        at: Where the problem starts in that render, in seconds.
        end: Where it ends.
        evidence: What shows it — a join's words and levels from
            `inspect_join`, a comment, a finding.

    Returns:
        The trial's `trial_id`, and what to do next.

    Raises:
        ValueError: If the render is not a finished one of this project, or
            the stretch is outside it.
    """
    job, path = rendering.finished_render(before_job)
    if job.project_id != project_id:
        raise ValueError(f"render {before_job} is not of project {project_id}")
    length = _render_length(path)
    if end <= at or (length is not None and end > length + 0.05):
        raise ValueError(f"the stretch {at:g}–{end:g}s is not in the {length or 0:.1f}s render")
    trial = Trial(id="t" + uuid.uuid4().hex[:7], project_id=project_id, problem=problem.strip(),
                  evidence=evidence.strip(), hypothesis=hypothesis.strip(), expected=expected.strip(),
                  at=at, end=end, before_job=before_job, before_version=job.project_version)
    os.makedirs(trial_dir(trial.id), exist_ok=True)
    shutil.copyfile(path, os.path.join(trial_dir(trial.id), "before.mp4"))
    runtime.repo.save_trial(trial)
    return {"trial_id": trial.id, "next": (
        "make the change, render a preview of it, and give it with `trial_candidate`")}

@mcp.tool()
def trial_candidate(
    trial_id: str,
    after_job: str,
    candidate_at: Optional[Annotated[float, Field(ge=0)]] = None,
    candidate_end: Optional[Annotated[float, Field(gt=0)]] = None,
) -> dict:
    """Give a trial the render of the cut after the change.

    Args:
        trial_id: The trial, from `start_trial`.
        after_job: A finished render of the cut after the change.
        candidate_at: Where the same stretch starts in it, when the change
            moved it; the same seconds as before when not given.
        candidate_end: Where it ends.

    Returns:
        What to do next: who judges it.

    Raises:
        ValueError: If the trial does not exist or has been judged, or the
            render is not a finished one of the same project after the first.
    """
    trial = runtime.repo.trial(trial_id)
    if trial is None:
        raise ValueError(f"there is no trial {trial_id}")
    if trial.verdict is not None:
        raise ValueError(f"trial {trial_id} has been judged; start another")
    job, path = rendering.finished_render(after_job)
    if (job.project_id != trial.project_id or after_job == trial.before_job
            or (job.project_version or 0) <= (trial.before_version or 0)):
        raise ValueError(f"render {after_job} is not a render of project {trial.project_id} made after the change; "
                         f"the cut before was version {trial.before_version}")
    shutil.copyfile(path, os.path.join(trial_dir(trial.id), "after.mp4"))
    trial.after_job, trial.after_version = after_job, job.project_version
    trial.candidate_at = trial.at if candidate_at is None else candidate_at
    trial.candidate_end = trial.end if candidate_end is None else candidate_end
    trial.candidate_is = secrets.choice(["X", "Y"])
    runtime.repo.save_trial(trial)
    return {"trial_id": trial.id, "next": (
        "have it judged by someone who has not seen the change: start a fresh subagent with no part in this "
        "conversation, give it only `trial_pack` and ask which of X and Y handles the problem better and which, "
        "if either, breaks what is around it; or show the pack to the user. Then record it with `judge_trial`. "
        "Judging it yourself is not blind — you know which you made — and the record says so")}

@mcp.tool()
def trial_pack(trial_id: str) -> ToolResult:
    """Hand over the two versions of a trial blind, as X and Y, for someone to judge.

    It holds the problem and its evidence, and for each of X and Y the frames
    of the stretch and a little either side, and its sound to hear — nothing
    that says which is the change or why it was made. Give it to whoever
    judges, as it is.

    Args:
        trial_id: The trial.

    Returns:
        The problem and evidence, where each sound file is, and the frames of
        X above Y.

    Raises:
        ValueError: If the trial does not exist or has no candidate yet.
    """
    trial = runtime.repo.trial(trial_id)
    if trial is None:
        raise ValueError(f"there is no trial {trial_id}")
    if trial.candidate_is is None:
        raise ValueError(f"trial {trial_id} has no change to compare yet; `trial_candidate` gives it one")
    folder = trial_dir(trial.id)
    files = {"before": (os.path.join(folder, "before.mp4"), trial.at, trial.end),
             "after": (os.path.join(folder, "after.mp4"), trial.candidate_at, trial.candidate_end)}
    labels = {trial.candidate_is: "after", ("Y" if trial.candidate_is == "X" else "X"): "before"}
    per_side = shared.MAX_STORYBOARD_TILES // 2
    shots, lines = [], [f"The problem: {trial.problem}"]
    if trial.evidence:
        lines.append(f"What shows it: {trial.evidence}")
    lines.append(f"Below, X above Y: the same stretch in two versions of the cut, with {TRIAL_MARGIN_SECONDS:g}s "
                 "either side. Say which handles the problem better, or that neither does, and which, if either, "
                 "makes what is around it worse.")
    for label in ("X", "Y"):
        path, start, finish = files[labels[label]]
        length = _render_length(path) or finish
        opened, closed = max(0.0, start - TRIAL_MARGIN_SECONDS), min(length, finish + TRIAL_MARGIN_SECONDS)
        step = (closed - opened) / (per_side - 1)
        for index in range(per_side):
            moment = round(min(opened + index * step, length - 0.05), 3)
            shots.append((path, moment, f"{label} {moment - start:+.1f}s"))
        heard = os.path.join(folder, f"{label}.wav")
        try:
            cut_sound(path, opened, closed, heard)
            lines.append(f"{label} sound: {heard}")
        except RuntimeError:
            lines.append(f"{label} has no sound here")
    return ToolResult(content=["\n".join(lines), Image(data=storyboard_sheet(shots, columns=per_side // 2),
                                                         format="jpeg")])

@mcp.tool()
def judge_trial(
    trial_id: str,
    better: Literal["X", "Y", "same"],
    broke: Literal["neither", "X", "Y", "both"],
    why: Annotated[str, Field(min_length=1)],
    judged_by: Literal["subagent", "user", "same_ai"],
) -> dict:
    """Write down the verdict on a trial, and only then hear which was the change.

    Args:
        trial_id: The trial.
        better: Which handles the problem better, as whoever judged said; `same`
            when neither does.
        broke: Which, if either, they said makes what is around the problem
            worse. Only the change breaking something counts against it.
        why: What they saw and heard, in their words.
        judged_by: `subagent` for a fresh one that saw none of the change,
            `user`, or `same_ai` when you judged it yourself — which is not
            blind, and the record says so.

    Returns:
        `change_was`, the change's label; `keep`, `change` only when it made
        the problem better and broke nothing, otherwise `original` — then put
        the cut back with `restore_version` to the version before; and the
        trial as recorded.

    Raises:
        ValueError: If the trial does not exist, has no change yet, or was
            judged already.
    """
    trial = runtime.repo.trial(trial_id)
    if trial is None:
        raise ValueError(f"there is no trial {trial_id}")
    if trial.candidate_is is None:
        raise ValueError(f"trial {trial_id} has no change to judge yet")
    if trial.verdict is not None:
        raise ValueError(f"trial {trial_id} was judged already: {trial.verdict.better}")
    trial.verdict = Verdict(better=better, broke=broke, why=why.strip(), judged_by=judged_by)
    runtime.repo.save_trial(trial)
    shutil.rmtree(trial_dir(trial.id), ignore_errors=True)
    keep = "change" if better == trial.candidate_is and broke not in (trial.candidate_is, "both") else "original"
    said = {
        "change_was": trial.candidate_is, "keep": keep, "blind": judged_by != "same_ai",
        "trial": _trial_said(trial),
    }
    if keep == "original":
        said["next"] = (f"the change is not kept: put the cut back as it was in version {trial.before_version} with "
                        "`project_history` and `restore_version`, and tell the user what was tried and why it was not "
                        "kept")
    return said

def _trial_said(trial: Trial) -> dict:
    """A trial as it is read back."""
    said = {"trial_id": trial.id, "problem": trial.problem, "hypothesis": trial.hypothesis,
            "expected": trial.expected, "before_version": trial.before_version, "after_version": trial.after_version}
    if trial.verdict is not None:
        said["verdict"] = {"better": trial.verdict.better, "change_was": trial.candidate_is,
                           "broke": trial.verdict.broke, "why": trial.verdict.why,
                           "judged_by": trial.verdict.judged_by}
    return said

@mcp.tool()
def get_trials(project_id: str) -> dict:
    """Read back the changes to a video that were tried blind, and what came of each.

    Args:
        project_id: The video.

    Returns:
        `trials`, oldest first: the problem, what was changed and expected,
        and the verdict with who gave it.
    """
    return {"trials": [_trial_said(trial) for trial in runtime.repo.trials(project_id)]}

@mcp.tool()
def ask_user(project_id: str, questions: Annotated[List[Question], Field(min_length=1, max_length=8)]) -> dict:
    """Put what you are not sure of about a cut to the user, each on the moment it is about.

    The checks say whether something is wrong; none says whether it is good —
    whether four seconds of opening hook anybody, whether a 57-second test
    loses them. Do not settle those by yourself: after a preview, ask about
    the few you weighed and could not decide, with why. Each shows in the
    editor as a question on its moment of the cut, and the answer comes back
    as a reply to it in `get_comments`. When an answer says how the user
    likes their videos in general, not only this one, `remember_preference`.

    Args:
        project_id: The video.
        questions: Each with where it is (`start`, and `end` for a stretch),
            the `question` in the user's language and `why` it is in doubt.

    Returns:
        The questions as `get_comments` shows them.

    Raises:
        ValueError: If the project does not exist.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    asked = []
    length = float(project.duration)
    for question in questions:
        if question.start > length or (question.end is not None and not (
                question.start + SHORTEST_STRETCH_SECONDS <= question.end <= length + 0.001)):
            raise ValueError(f"「{question.question[:20]}」 is asked at {question.start:g}"
                             + (f"–{question.end:g}" if question.end is not None else "")
                             + f"s, which is not a moment or a stretch of the {length:.1f}s cut; nothing was asked")
    for question in questions:
        end = question.end
        comment = Comment(
            id="c" + uuid.uuid4().hex[:7], project_id=project_id, by=f"AI（{runtime.calling_client.get()}）",
            text=question.question.strip() + (f"\n（{question.why.strip()}）" if question.why.strip() else ""),
            commit=_current_commit(project_id), question=True,
            start=round(question.start, 3), end=round(end, 3) if end is not None else None,
            start_anchor=anchor_at(project, question.start),
            end_anchor=anchor_at(project, end) if end is not None else None,
        )
        runtime.repo.save_comment(comment, f"AI 問：{question.question[:30]}")
        asked.append(_comment_said(project, comment))
    return {"asked": asked}

@mcp.tool()
def remember_preference(text: Annotated[str, Field(min_length=1, max_length=300)],
                        project_id: Optional[str] = None) -> dict:
    """Keep how the user likes their videos, for every project after this one.

    Only what the user said, in their words — 「開場不要超過三秒」, 「不要用
    轉場」 — and only when it is about their videos in general rather than
    this cut. What is kept comes back with `list_projects`, `create_project`
    and `get_project`, so the next video starts from it. Read it back to them
    when you keep it.

    Args:
        text: The preference, as they said it.
        project_id: The video it was said about, if one.

    Returns:
        `kept`, the preference with its `id`; and `preferences`, all of them.
    """
    preference = Preference(id="p" + uuid.uuid4().hex[:7], text=text.strip(), project_id=project_id)
    runtime.repo.save_preference(preference)
    return editing.with_preferences({"kept": {"id": preference.id, "text": preference.text}})

@mcp.tool()
def forget_preference(preference_id: str) -> dict:
    """Stop keeping a preference, when the user says it no longer holds.

    Args:
        preference_id: Its ID, from `preferences` in `list_projects`.

    Returns:
        `forgotten`, its ID; and `preferences`, what is still kept.

    Raises:
        ValueError: If no preference has that ID.
    """
    if not runtime.repo.delete_preference(preference_id):
        raise ValueError(f"there is no preference {preference_id}")
    return editing.with_preferences({"forgotten": preference_id})

def add_comment(project_id: str, text: str, start: float, end: Optional[float] = None) -> dict:
    """Write down what the user said about a moment of the cut, pinned to the footage under it.

    The editor's, never the AI's: a comment is the user speaking.

    Args:
        project_id: The video.
        text: What they said.
        start: Where on the cut, in seconds.
        end: Where the stretch it is about ends; None for a moment.

    Returns:
        The comment as `get_comments` shows it.

    Raises:
        ValueError: If the project does not exist or the text is empty.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    if not text.strip():
        raise ValueError("a comment needs something said")
    if end is not None and end < start:
        start, end = end, start
    if end is not None and end - start < SHORTEST_STRETCH_SECONDS:
        end = None
    comment = Comment(
        id="c" + uuid.uuid4().hex[:7], project_id=project_id, text=text.strip(), commit=_current_commit(project_id),
        start=round(max(0.0, start), 3), end=round(end, 3) if end is not None else None,
        start_anchor=anchor_at(project, start), end_anchor=anchor_at(project, end) if end is not None else None,
    )
    runtime.repo.save_comment(comment, f"留言：{comment.text[:30]}")
    return _comment_said(project, comment)

def change_comment(project_id: str, comment_id: str, by: str, reply: str = "",
                   status: Optional[Literal["open", "resolved"]] = None) -> dict:
    """Say something back on a comment, close it, or open it again.

    Args:
        project_id: The video.
        comment_id: The comment.
        by: Who is speaking: the AI program, or 你.
        reply: What is said; empty to only change its status.
        status: `resolved` to close it, `open` to open it again; None leaves it.

    Returns:
        The comment as `get_comments` shows it.

    Raises:
        ValueError: If the project or comment does not exist.
    """
    project = runtime.repo.get_project(project_id)
    comment = next((item for item in runtime.repo.comments(project_id) if item.id == comment_id), None)
    if project is None or comment is None:
        raise ValueError(f"project {project_id} has no comment {comment_id}")
    commit = _current_commit(project_id)
    if reply.strip():
        comment.replies.append(Reply(by=by, text=reply.strip(), commit=commit))
    if status == "resolved":
        comment.status, comment.resolved_commit = "resolved", commit
    elif status == "open":
        comment.status, comment.resolved_commit = "open", None
    said = {"resolved": "處理了一則留言", "open": "重開一則留言"}.get(status or "", "回覆一則留言")
    runtime.repo.save_comment(comment, f"{said}：{comment.text[:30]}")
    return _comment_said(project, comment)

@mcp.tool()
def get_comments(project_id: str, include_resolved: bool = False) -> dict:
    """Read what the user said about moments of a video while watching it in the editor.

    Each is a request written on the cut — 「這裡太拖」, 「這段配樂太大聲」 —
    pinned to the footage that was on screen, so it is shown where that
    footage is now even after the cut changed: `start` (and `end` for a
    stretch) in seconds, `at` as a timecode, the `clip_id` under it, and the
    `file` and `file_seconds`. `gone` means that footage has since been cut
    out. Read it, do what it asks — or ask the user if it can be read two
    ways — then close it with `resolve_comment`, saying what you did. One
    marked `question` is one you asked with `ask_user`: it waits on the user
    until it is `answered`, in its replies.

    Args:
        project_id: The video.
        include_resolved: Also list the ones already dealt with.

    Returns:
        `comments`, oldest first, each with its `id`, `text`, `status`,
        `replies` and, once closed, `resolved_in` — the version it was dealt
        with in; and `open`, how many are waiting.

    Raises:
        ValueError: If the project does not exist.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    every = runtime.repo.comments(project_id)
    return {
        "comments": [_comment_said(project, comment) for comment in every
                     if include_resolved or comment.status == "open"],
        "open": sum(comment.status == "open" for comment in every),
    }

@mcp.tool()
def resolve_comment(project_id: str, comment_id: str, reply: str, resolved: bool = True) -> dict:
    """Close a comment the user wrote, saying what was done about it.

    The reply is shown under the comment in the editor, in plain words the
    user reads: 「把開頭縮短 2 秒」, not the edits. The comment stays, greyed,
    and the user can open it again. With `resolved` false the reply is only
    said — to ask which of two things they meant, or why it was not done.
    Comments are the user's: there is no way to delete one from here.

    Args:
        project_id: The video.
        comment_id: The comment, from `get_comments`.
        reply: What was done, or what you need to know.
        resolved: False to reply and leave it open.

    Returns:
        The comment as `get_comments` shows it.

    Raises:
        ValueError: If the project or comment does not exist, or the reply
            is empty.
    """
    if not reply.strip():
        raise ValueError("say what was done, or what you need to know")
    return change_comment(project_id, comment_id, f"AI（{runtime.calling_client.get()}）", reply,
                          "resolved" if resolved else None)
