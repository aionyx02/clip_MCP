"""A project's versions: its history, going back, branching, comparing and merging versions, and marking them."""

import json
import os
import uuid
from typing import Annotated, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

from pydantic import Field

from app.models.plan import EditPlan
from app.models.timeline import BranchPoint, Project, validate_project
from app.models.job import Job
from app.engine.semantic import was_made_up
from app.engine.compare import compare
from app.engine.compare import parts_of as cut_parts
from app.engine.merge import WHOLE_PICKS, merge, merge_plans
from app.storage import housekeeping
from app.storage.history import Version
from app import runtime
from app.runtime import mcp
from app.tools import rendering, shared

# What the user can mark a version as. `exported` is a third mark, never set by hand: a
# version that was rendered out, not as a preview, wears it by itself.
VERSION_MARKS = ("starred", "published")

def project_json_at(project_id: str, commit: str) -> Optional[dict]:
    """A project as its file held it at one version, or None when it was not there then."""
    path = runtime.history.find("projects/", f"{project_id}.json", commit)
    content = runtime.history.read(path, commit) if path else None
    return json.loads(content) if content else None

def branches_of(project_id: str) -> List[Project]:
    """The projects started as branches of this one."""
    return [other for other in runtime.repo.list_projects()
            if other.branched_from is not None and other.branched_from.project_id == project_id]

def _version_record(version: Version, marks: Iterable[str] = ()) -> dict:
    """One version of a project as the AI reads it."""
    return {
        "commit": version.commit[:10],
        "when": version.when.astimezone().strftime("%Y-%m-%d %H:%M"),
        "by": version.author,
        "what": version.subject,
        "marks": sorted(marks),
    }

def _project_versions(project_id: str, limit: int) -> List[Tuple[Version, Optional[dict], set]]:
    """A project's versions, newest first, each with the project as it was then and its marks.

    Returns:
        `(version, project, marks)`, the project None where the version took
        it away. A version whose finished video is still there is `exported`;
        one whose finished video has gone since, `removed`.
    """
    marks = runtime.repo.version_marks(project_id)
    delivered = _delivered_renders(project_id)
    there = {job.project_version for job in delivered if os.path.exists(job.output_path)}
    gone = {job.project_version for job in delivered} - there
    found = []
    for version in runtime.history.versions_of("projects/", f"{project_id}.json", limit):
        data = project_json_at(project_id, version.commit)
        marked = set(marks.get(version.commit, ()))
        if data is not None and data.get("version") in there:
            marked.add("exported")
        elif data is not None and data.get("version") in gone:
            marked.add("removed")
        found.append((version, data, marked))
    return found

def _delivered_renders(project_id: str) -> List[Job]:
    """Every full render of a project that finished, whether its file is still there or not."""
    previews = os.path.abspath(housekeeping.preview_dir(runtime.WORKSPACE_DIR, project_id))
    return [job for job in runtime.repo.renders_of(project_id)
            if job.output_path and not os.path.abspath(job.output_path).startswith(previews)]

def outputs_of(project_id: str) -> List[Job]:
    """The finished videos of a project that are still where they were saved; previews are not among them."""
    return [job for job in _delivered_renders(project_id) if os.path.exists(job.output_path)]

def _base_clips(data: Optional[dict]) -> List[dict]:
    """A project's sequence, as its file holds it, in the order it plays."""
    base = next((track for track in (data or {}).get("tracks", []) if track.get("track_type") == "video"), None)
    return sorted(base.get("clips", []) if base else [], key=lambda clip: float(clip["timeline_in"]))

def _thumb_of(data: Optional[dict]) -> Optional[dict]:
    """Which frame stands for a version: its first shot, a second in, as the projects page shows it."""
    clips = _base_clips(data)
    if not clips:
        return None
    first = clips[0]
    start, end = float(first["source_range"]["start"]), float(first["source_range"]["end"])
    return {"asset_id": first["asset_id"], "t": round(min(start + 1.0, (start + end) / 2), 2)}

def _length_of(data: Optional[dict]) -> float:
    """How long a version runs: where its sequence ends."""
    return round(max((float(clip["timeline_in"])
                      + (float(clip["source_range"]["end"]) - float(clip["source_range"]["start"]))
                      / float(clip.get("speed") or 1) for clip in _base_clips(data)), default=0.0), 2)

def _version_nodes(entries: List[Tuple[Version, Optional[dict], set]], apart: set,
                   outputs: Mapping[int, List[Job]]) -> List[dict]:
    """Group versions into what the version panel shows as one dot each.

    A run of changes by one author with nothing rendered out between them is
    one dot: twenty nudges in the editor are one sitting, not twenty
    versions. A render ends a sitting, so a rendered version heads its own
    dot; a version the user starred or marked published, or one a branch
    started from, stands alone.

    Args:
        entries: From `_project_versions`, newest first.
        apart: Commits that must not be folded into another's dot.
        outputs: The finished videos still there, by the project version
            they were rendered from.

    Returns:
        The dots, newest first, each named by its newest version.
    """
    alone = set(VERSION_MARKS)
    nodes: List[dict] = []
    for version, data, marked in entries:
        record = {"commit": version.commit, "what": version.subject,
                  "when": version.when.astimezone().strftime("%m/%d %H:%M")}
        last = nodes[-1] if nodes else None
        if (last is not None and last["by"] == version.author and not marked and not alone & set(last["marks"])
                and version.commit not in apart and last["versions"][-1]["commit"] not in apart):
            last["versions"].append(record)
            continue
        number = data.get("version") if data else None
        nodes.append({
            "commit": version.commit, "when": record["when"], "by": version.author, "what": version.subject,
            "body": version.body, "marks": sorted(marked), "gone": data is None, "project_version": number,
            "thumb": _thumb_of(data), "seconds": _length_of(data),
            "outputs": [{"job_id": job.job_id, "name": os.path.basename(job.output_path)}
                        for job in outputs.get(number, [])],
            "versions": [record],
        })
    return nodes

def _nodes_of(project_id: str, limit: int, apart: set) -> List[dict]:
    """A project's dots for the version panel."""
    outputs: Dict[int, List[Job]] = {}
    for job in outputs_of(project_id):
        outputs.setdefault(job.project_version, []).append(job)
    return _version_nodes(_project_versions(project_id, limit), apart, outputs)

def version_graph(project_id: str, limit: int = 200) -> dict:
    """Everything the editor's version panel draws for one video.

    Args:
        project_id: The video.
        limit: Most versions to read.

    Returns:
        The `project`; its `nodes`, newest first; the `branches` started from
        it, each with the commit it started `from` and its own nodes; and
        `branched_from` when it is itself a branch.

    Raises:
        ValueError: If there is no such project.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    branches = branches_of(project_id)
    starts = {branch.branched_from.commit for branch in branches}
    parent = None
    if project.branched_from is not None:
        source = runtime.repo.get_project(project.branched_from.project_id)
        start = next((version for version in runtime.history.versions_of(
            "projects/", f"{project.branched_from.project_id}.json", 500)
            if version.commit == project.branched_from.commit), None)
        parent = {"project_id": project.branched_from.project_id, "commit": project.branched_from.commit,
                  "name": source.name if source else None, "what": start.subject if start else None}
    return {
        "project": {"id": project.id, "name": project.name, "version": project.version},
        "nodes": _nodes_of(project_id, limit, starts),
        "branches": [
            {"project_id": branch.id, "name": branch.name, "from": branch.branched_from.commit,
             "nodes": _nodes_of(branch.id, 30, set())}
            for branch in sorted(branches, key=lambda item: item.name or "")
        ],
        "branched_from": parent,
    }

@mcp.tool()
def project_history(project_id: str, limit: Annotated[int, Field(ge=1, le=200)] = 30) -> dict:
    """List a video's versions: every change to it, newest first, with who made it and what it was.

    Every change to a project — a compile, an edit, captions — is kept as a
    version, and nothing is ever taken out of the history: going back is one
    more version on top. Use it for 「剛剛那樣比較好」, 「回到昨天那版」 or
    「誰改了結尾」, then `restore_version` or `branch_project`. A client with a
    shell can also read the same history with git in the workspace's
    `history` folder.

    Args:
        project_id: The video.
        limit: Most versions to list.

    Returns:
        `versions`, each with its `commit`, `when`, `by` (the AI program or
        the editor), `what` changed in words, and its `marks`: `starred`
        (the user's favourite), `published` (the one that went out) and
        `exported` (rendered out); and `branched_from`, the project and
        version a branch started as, or null.

    Raises:
        ValueError: If there is no such project.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    return {
        "versions": [_version_record(version, marks) for version, _, marks in _project_versions(project_id, limit)],
        "branched_from": project.branched_from.model_dump() if project.branched_from else None,
    }

def project_at(project_id: str, commit: str) -> Project:
    """A project as it was at one version.

    Raises:
        ValueError: If there is no such version, or the project was not there then.
    """
    try:
        path = runtime.history.find("projects/", f"{project_id}.json", commit)
    except KeyError as exc:
        raise ValueError(str(exc).strip("'\"")) from exc
    content = runtime.history.read(path, commit) if path else None
    if content is None:
        raise ValueError(f"project {project_id} did not exist at version {commit}")
    return Project.model_validate_json(content)

def _plans_at(project: Project, commit: str) -> Dict[str, EditPlan]:
    """The plans a project's clips were compiled from, as they were at one version."""
    found: Dict[str, EditPlan] = {}
    for plan_id in {clip.from_plan_id for track in project.tracks for clip in track.clips if clip.from_plan_id}:
        content = runtime.history.read(f"plans/{plan_id}.json", commit)
        if content is not None:
            found[plan_id] = EditPlan.model_validate_json(content)
    return found

def _with_plans(project: Project, renamed: Mapping[str, str]) -> Project:
    """A project whose compiled clips name their plans by new IDs."""
    copy = project.model_copy(deep=True)
    for track in copy.tracks:
        for clip in track.clips:
            if clip.from_plan_id in renamed:
                clip.from_plan_id = renamed[clip.from_plan_id]
    return copy

def _copied_plan(plan: EditPlan, why: str) -> str:
    """Save a plan under a new ID, as the starting point of its own history; returns the ID."""
    stored = runtime.repo.save_plan(plan.model_copy(update={"id": str(uuid.uuid4()), "version": 1}), why)
    return stored.id

@mcp.tool()
def restore_version(project_id: str, commit: str, expected_version: int, note: str = "") -> dict:
    """Make a video what it was at one of its versions, as a new version on top.

    Nothing after that version is lost: it all stays in the history, and
    going back can itself be undone the same way. The plan the video was
    compiled from goes back with it, so the next compile does not undo this
    — and when another video was compiled from that same plan, this one gets
    a copy of it instead, so the other is left exactly as it is.

    Args:
        project_id: The video.
        commit: The version to go back to, from `project_history`.
        expected_version: The project's current `version`.
        note: The user's own words for why, kept in the history.

    Returns:
        The project's `new_version`, the `restored` version's `commit` and
        `what`, and `plan_copied` — true when the plan was shared and this
        video now has a copy of its own, which is worth telling the user.

    Raises:
        ValueError: If the project or version does not exist, or the project
            has moved on from `expected_version`.
    """
    current = runtime.repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    if current.version != expected_version:
        raise ValueError(f"version conflict: project {project_id} is at version {current.version}, not {expected_version}")
    earlier = project_at(project_id, commit)
    chosen = next((version for version in runtime.history.versions_of("projects/", f"{project_id}.json", 500)
                   if version.commit.startswith(commit.strip().lower())), None)
    if chosen is not None:
        shared.say_in_history(f"回到「{chosen.subject}」那一版")
    renamed: Dict[str, str] = {}
    copied = False
    others = [other for other in runtime.repo.list_projects() if other.id != project_id]
    for plan_id, plan in _plans_at(earlier, commit).items():
        used_elsewhere = any(clip.from_plan_id == plan_id
                             for other in others for track in other.tracks for clip in track.clips)
        if used_elsewhere:
            renamed[plan_id] = _copied_plan(plan, f"「{current.name or project_id}」回到舊版時另存的計畫")
            copied = True
            continue
        now = runtime.repo.get_plan(plan_id)
        if now is None:
            runtime.repo.save_plan(plan.model_copy(update={"version": 1}), "回到舊版時放回的計畫")
        elif now.model_dump(exclude={"version", "updated_at"}) != plan.model_dump(exclude={"version", "updated_at"}):
            runtime.repo.save_plan(plan.model_copy(update={"version": now.version}), "跟著影片回到舊版")
    restored = _with_plans(earlier, renamed).model_copy(update={
        "id": project_id, "version": current.version + 1, "branched_from": current.branched_from,
    })
    runtime.repo.update_project(restored, current.version)
    rendering.prepare_playback(runtime.repo.get_project(project_id))
    return {
        "new_version": restored.version,
        "restored": {"commit": chosen.commit[:10], "what": chosen.subject} if chosen else {"commit": commit},
        "plan_copied": copied,
    }

@mcp.tool()
def branch_project(project_id: str, name: str, commit: Optional[str] = None, note: str = "") -> dict:
    """Start another way of cutting a video from one of its versions, beside it rather than over it.

    For 「做一個 Reels 60 秒版」 or 「試試看沒有配樂的版本」: the branch is a project
    of its own, with its own copy of the plan, so it can be cut, captioned and
    rendered while the original is left alone. The editor draws it as a line
    running beside the original's from the version it started at.

    Args:
        project_id: The video to branch from.
        name: What to call the branch, in words: 「抽杯架 Reels 60 秒版」.
        commit: The version to start from, from `project_history`; omit for
            the video as it is now.
        note: The user's own words for why, kept in the history.

    Returns:
        The new `project_id`, its `name`, and `branched_from`.

    Raises:
        ValueError: If the project or version does not exist.
    """
    current = runtime.repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    if commit is None:
        versions = runtime.history.versions_of("projects/", f"{project_id}.json", 1)
        if not versions:
            raise ValueError(f"project {project_id} has no version in the history yet")
        commit = versions[0].commit
    start = project_at(project_id, commit)
    found = next((version.commit for version in runtime.history.versions_of("projects/", f"{project_id}.json", 500)
                  if version.commit.startswith(commit.strip().lower())), commit)
    renamed = {plan_id: _copied_plan(plan, f"「{name}」分支的計畫")
               for plan_id, plan in _plans_at(start, commit).items()}
    branch = _with_plans(start, renamed).model_copy(update={
        "id": str(uuid.uuid4()), "name": name.strip(), "version": 1,
        "branched_from": BranchPoint(project_id=project_id, commit=found),
    })
    shared.say_in_history(f"從「{current.name or project_id}」開分支「{branch.name}」")
    runtime.repo.add_project(branch)
    rendering.prepare_playback(branch)
    folder = runtime.repo.folder_of("projects").get(project_id)
    if folder:
        runtime.repo.file_items("projects", [branch.id], folder)
    return {"project_id": branch.id, "name": branch.name, "branched_from": branch.branched_from.model_dump()}

# Provisional (roadmap §13): a word counts as said in a stretch only when this much of it is
# inside, so a word the cut just grazes does not name the stretch.
WORD_INSIDE_SECONDS = 0.05

def said_between(asset_id: str, start: float, end: float) -> str:
    """What the transcript has said in a file between two of its seconds, glossary words corrected; empty when nothing was."""
    analysis = runtime.repo.get_analysis(asset_id)
    if analysis is None or analysis.transcript is None:
        return ""
    return shared.glossed("".join(
        word.text for segment in analysis.transcript.segments if not was_made_up(segment) for word in segment.words
        if word.start < end - WORD_INSIDE_SECONDS and word.end > start + WORD_INSIDE_SECONDS
    ).strip())

def version_at(project_id: str, commit: Optional[str]) -> Tuple[Project, dict]:
    """A project at one of its versions, or as it is now, with that version as the AI reads it.

    Args:
        project_id: The video.
        commit: The version, or a prefix of it; empty for the video as it is.

    Returns:
        The project as it was, and the version as `project_history` shows it.

    Raises:
        ValueError: If the project or the version does not exist.
    """
    current = runtime.repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    versions = runtime.history.versions_of("projects/", f"{project_id}.json", 500)
    if not commit:
        return current, (_version_record(versions[0]) if versions else {"commit": None, "what": "目前"})
    chosen = next((version for version in versions if version.commit.startswith(commit.strip().lower())), None)
    if chosen is None:
        raise ValueError(f"project {project_id} has no version {commit}")
    return project_at(project_id, chosen.commit), _version_record(chosen)

def compare_projects(before: Project, after: Project) -> dict:
    """What changed from one cut to another, files named by name and stretches by their words.

    Args:
        before: The left cut.
        after: The right cut.

    Returns:
        What `compare.compare` returns.
    """
    names = {asset_id: os.path.basename(asset.path) for asset_id, asset in runtime.repo.get_assets(
        {clip.asset_id for project in (before, after) for track in project.tracks for clip in track.clips}).items()}
    return compare(before, after, said_between, lambda asset_id: names.get(asset_id, asset_id))

@mcp.tool()
def compare_versions(project_id: str, commit_a: str, commit_b: Optional[str] = None) -> dict:
    """Say what changed between two versions of a video, part by part, in words the user reads.

    For 「第 3 版跟現在差在哪」 or before restoring or merging. Footage is matched
    by what it shows, not by clip IDs, so a recompile is not reported as
    everything changing. The editor's comparison view shows the same list.

    Args:
        project_id: The video.
        commit_a: The earlier version, from `project_history`.
        commit_b: The other one; omit for the video as it is now.

    Returns:
        `a` and `b` (each version's `commit`, `when`, `what`), `length` of
        each in seconds, `parts` — each part of the cut (its markers, one per
        beat) with `same` and `changes`, each a sentence with `a_at`/`b_at`,
        where it is in each version — and `whole`, changes to the video as a
        whole: songs, captions, caption style, frame.

    Raises:
        ValueError: If the project or a version does not exist.
    """
    before, a = version_at(project_id, commit_a)
    after, b = version_at(project_id, commit_b)
    compared = compare_projects(before, after)
    compared.pop("aligned")
    return {"a": a, "b": b, **compared}

def _merged_plan(before: Project, after: Project, commits: Tuple[Optional[str], Optional[str]], merged: Project,
                 from_left: List[str], name: str) -> Tuple[Project, str]:
    """Merge the plans the two sides were compiled from, beat by beat, when they can be.

    Each side's plan is read as it was at that side's version, so merging two
    old versions does not mix in the plan as it is today.

    Args:
        before: The left version.
        after: The right version.
        commits: The two versions, left and right; None for the video as it is.
        merged: The merged cut.
        from_left: Names of the parts taken from the left.
        name: The video's name, for a plan saved under a new ID.

    Returns:
        The merged cut with its compiled clips naming the plan they now come
        from, and what happened to the plans, in words; empty words when no
        plan was involved.
    """
    def plan_ids(project: Project) -> set:
        """The plans a cut's clips were compiled from."""
        return {clip.from_plan_id for track in project.tracks for clip in track.clips if clip.from_plan_id}

    def plan_at(project: Project, plan_id: str, commit: Optional[str]) -> Optional[EditPlan]:
        """A plan as it was at a version, or as it is."""
        return _plans_at(project, commit).get(plan_id) if commit else runtime.repo.get_plan(plan_id)

    left_ids, right_ids = plan_ids(before), plan_ids(after)
    if not from_left or not left_ids:
        return merged, ""
    if len(left_ids) != 1 or len(right_ids) != 1:
        return merged, "沒有合併剪輯計畫：兩邊不是各從一份計畫剪出來的"
    left_id, right_id = next(iter(left_ids)), next(iter(right_ids))
    left, right = plan_at(before, left_id, commits[0]), plan_at(after, right_id, commits[1])
    if left is None or right is None:
        return merged, "沒有合併剪輯計畫：找不到其中一份"
    if left.timeline_id != right.timeline_id:
        return merged, "沒有合併剪輯計畫：兩份計畫用的是不同的素材時間軸"
    combined = merge_plans(left, right, from_left)
    stored = runtime.repo.get_plan(right_id)
    used_elsewhere = any(clip.from_plan_id == right_id for other in runtime.repo.list_projects() if other.id != merged.id
                         for track in other.tracks for clip in track.clips)
    if used_elsewhere or stored is None:
        plan_id = _copied_plan(combined, f"「{name}」合成兩版時另存的計畫")
        said = "剪輯計畫也合併了；原本的計畫有別支影片在用，所以另存了一份" if used_elsewhere else "剪輯計畫也依段落合併了"
    else:
        # Saved on top of the plan as it is now, whichever version the right side was.
        plan_id = runtime.repo.save_plan(combined.model_copy(update={"version": stored.version}), "合成兩版").id
        said = "剪輯計畫也依段落合併了"
    relinked = merged.model_copy(update={"tracks": [
        track.model_copy(update={"clips": [
            clip.model_copy(update={"from_plan_id": plan_id}) if clip.from_plan_id in (left_id, right_id) else clip
            for clip in track.clips]})
        for track in merged.tracks]})
    return relinked, said

def merge_versions(project_id: str, commit_a: str, commit_b: Optional[str], picks: Mapping[str, str],
                   expected_version: int) -> dict:
    """Make the video one cut of two of its versions, a part from each, as a new version on top.

    The editor's comparison view and `merge_version_parts` both come here.

    Args:
        project_id: The video.
        commit_a: The left version.
        commit_b: The right version, which everything not picked comes
            from; None for the video as it is.
        picks: `a` or `b` for part names and the whole-video picks in
            `WHOLE_PICKS`; anything left out is `b`.
        expected_version: The version the video is at now.

    Returns:
        `new_version`, the merged cut's length in `seconds`, `from_a` — what
        was taken from the left — and `plan` when a plan was merged or could
        not be.

    Raises:
        ValueError: If the project or a version does not exist, a pick is
            not `a` or `b`, or the project moved on from `expected_version`.
    """
    current = runtime.repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    if current.version != expected_version:
        raise ValueError(f"version conflict: project {project_id} is at version {current.version}, "
                         f"not {expected_version}")
    before, a = version_at(project_id, commit_a)
    after, b = version_at(project_id, commit_b)
    names = {part.name for part in cut_parts(before) + cut_parts(after)} | set(WHOLE_PICKS)
    unknown = sorted(set(picks) - names)
    if unknown:
        raise ValueError(f"nothing called {', '.join(unknown)} to pick: the parts are "
                         f"{', '.join(sorted(names - set(WHOLE_PICKS)))}, and the whole video's "
                         f"{', '.join(WHOLE_PICKS)}")
    if any(side not in ("a", "b") for side in picks.values()):
        raise ValueError("each pick is a (the first version) or b (the second)")
    merged = merge(before, after, picks)
    from_left = [name for name, side in picks.items() if side == "a" and name not in WHOLE_PICKS]
    merged, plan_said = _merged_plan(before, after, (commit_a or None, commit_b), merged, from_left,
                                     current.name or project_id)
    taken = [name for name, side in picks.items() if side == "a"]
    shared.say_in_history(f"合成兩版：{'、'.join(taken) or '全部'}用「{a.get('what')}」" if taken else "合成兩版")
    result = merged.model_copy(update={"id": project_id, "version": current.version + 1,
                                       "name": current.name, "branched_from": current.branched_from})
    validate_project(result, runtime.repo.get_assets({clip.asset_id for track in result.tracks for clip in track.clips}))
    runtime.repo.update_project(result, current.version)
    rendering.prepare_playback(runtime.repo.get_project(project_id))
    said = {"new_version": result.version, "seconds": round(float(result.duration), 3), "from_a": taken}
    if plan_said:
        said["plan"] = plan_said
    return said

@mcp.tool()
def merge_version_parts(project_id: str, commit_a: str, picks: Dict[str, Literal["a", "b"]], expected_version: int,
                        commit_b: Optional[str] = None, note: str = "") -> dict:
    """Make a video one cut of two of its versions, taking each part from one or the other.

    For 「開頭用第 3 版的，其他用現在的」. The parts are the ones
    `compare_versions` lists; each is taken whole, and `music`, `captions`
    and `caption_style` can be picked for the video as a whole. The result is
    a new version on top of the video — both versions it came from stay in
    the history. When both were compiled from one plan, the plan is merged
    beat by beat too, with the reasons and trims of the beats taken.

    Args:
        project_id: The video.
        commit_a: The other version, from `project_history`.
        picks: `a` for each part, or `music`/`captions`/`caption_style`/`frame`, to
            take from `commit_a`; anything left out comes from `commit_b`.
        expected_version: The project's current version.
        commit_b: The version the rest comes from; omit for the video as it is.
        note: The user's own words, kept in the history.

    Returns:
        `new_version`, its length in `seconds`, `from_a` — what was taken
        from `commit_a` — and `plan`, what happened to the plan, when it did.

    Raises:
        ValueError: If a version does not exist, a pick names no part, or the
            project moved on from `expected_version`.
    """
    return merge_versions(project_id, commit_a, commit_b, picks, expected_version)

@mcp.tool()
def mark_version(project_id: str, commit: str, mark: Literal["starred", "published"], on: bool = True) -> dict:
    """Star a version of a video, or mark it as the one that was published; or take the mark off.

    For 「把這版標起來」 or 「這版已經發到 IG 了」. The marks show in the editor's
    version panel and in `project_history`, so 「回到我標星號那版」 can be
    found later.

    Args:
        project_id: The video.
        commit: The version, from `project_history`.
        mark: `starred` or `published`.
        on: False takes the mark off.

    Returns:
        The version's `commit` and its `marks` now.

    Raises:
        ValueError: If the project or version does not exist.
    """
    if runtime.repo.get_project(project_id) is None:
        raise ValueError(f"project {project_id} not found")
    chosen = next((version for version in runtime.history.versions_of("projects/", f"{project_id}.json", 500)
                   if version.commit.startswith(commit.strip().lower())), None)
    if chosen is None:
        raise ValueError(f"project {project_id} has no version {commit}")
    runtime.repo.set_version_mark(project_id, chosen.commit, mark, on)
    marks = next((marked for version, _, marked in _project_versions(project_id, 500)
                  if version.commit == chosen.commit), set(runtime.repo.version_marks(project_id).get(chosen.commit, ())))
    return {"commit": chosen.commit[:10], "marks": sorted(marks)}
