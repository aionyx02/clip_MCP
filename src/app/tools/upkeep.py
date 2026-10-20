"""Disk space: what each project holds, trimming and trashing projects, and tidying old outputs and caches."""

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import List, Literal

from app.models.timeline import Project
from app.storage import housekeeping
from app.workspace import output_dir
from app import runtime
from app.runtime import mcp
from app.tools import shared, versioning

def _project_caches(project_id: str) -> List[str]:
    """The folders of a project's previews and sound checks: made again when wanted."""
    return [housekeeping.preview_dir(runtime.WORKSPACE_DIR, project_id), housekeeping.sound_dir(runtime.WORKSPACE_DIR, project_id)]

def project_storage(project_id: str) -> dict:
    """What a video takes on disk, and what deleting its other versions would give back.

    Its other versions are the finished videos of every version but the one
    it is now and those the user starred or marked published; with them go
    its previews. The versions themselves stay in the history, marked as no
    longer exported.

    Args:
        project_id: The video.

    Returns:
        `versions`, `branches`, `exported` (versions with a finished video
        still there), `outputs_megabytes`, `cache_megabytes`,
        `freeable_megabytes`, and `other_outputs`, the files that would go.

    Raises:
        ValueError: If there is no such project.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    # Only the marked versions are read: the storage page asks this of every video at once.
    kept = {project.version}
    for commit, marks in runtime.repo.version_marks(project_id).items():
        if marks & set(versioning.VERSION_MARKS):
            data = versioning.project_json_at(project_id, commit)
            if data:
                kept.add(data.get("version"))
    outputs = versioning.outputs_of(project_id)
    others = [job for job in outputs if job.project_version not in kept]
    size = {job.job_id: housekeeping.size_of(job.output_path) for job in outputs}
    cache = _cache_bytes(project_id)
    return {
        "versions": runtime.history.counts("projects/").get(project_id, 0), "branches": len(versioning.branches_of(project_id)),
        "exported": len({job.project_version for job in outputs}),
        "outputs_megabytes": round(sum(size.values()) / 1e6, 1),
        "cache_megabytes": round(cache / 1e6, 1),
        "freeable_megabytes": round((sum(size[job.job_id] for job in others) + cache) / 1e6, 1),
        "other_outputs": [{"path": job.output_path, "megabytes": round(size[job.job_id] / 1e6, 1)} for job in others],
    }

def _cache_bytes(project_id: str) -> int:
    """What a project's previews and sound checks take."""
    return sum(housekeeping.size_of(folder) for folder in _project_caches(project_id))

def disk_of(project_id: str) -> float:
    """What a video takes on disk in megabytes, its finished videos and previews together, for its card."""
    outputs = sum(housekeeping.size_of(job.output_path) for job in versioning.outputs_of(project_id))
    return round((outputs + _cache_bytes(project_id)) / 1e6, 1)

def trim_project(project_id: str, typed_name: str) -> dict:
    """Delete a video's other versions: their finished videos go to the trash, its previews go.

    Only the editor calls this, once the user has typed the video's name: the
    AI has no tool that deletes anything.

    Args:
        project_id: The video.
        typed_name: The name the user typed to confirm.

    Returns:
        `trashed` (finished videos put in the trash) and `freed_megabytes`.

    Raises:
        ValueError: If there is no such project, or the name typed is not its name.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    if not (project.name or "").strip():
        raise ValueError("name the video first: its name is what is typed to confirm")
    if typed_name.strip() != project.name.strip():
        raise ValueError("the name typed is not the video's name")
    found = project_storage(project_id)
    paths = [item["path"] for item in found["other_outputs"]]
    freed = 0
    if paths:
        trashed = housekeeping.put_in_trash(str(output_dir()), "outputs", project_id, project.name or "", {}, paths)
        freed += trashed.bytes
    busy = runtime.repo.active_job_ids()
    for folder in _project_caches(project_id):
        if not os.path.isdir(folder):
            continue
        for entry in os.scandir(folder):
            if entry.name in busy:
                continue
            size = housekeeping.size_of(entry.path)
            shutil.rmtree(entry.path, ignore_errors=True)
            freed += size
    return {"trashed": len(paths), "freed_megabytes": round(freed / 1e6, 1)}

def trash_projects(project_ids: List[str]) -> dict:
    """Move projects to the trash, unless one of them is being rendered.

    Their finished videos stay where they were saved; their previews go,
    since they are made again. A project waits in the trash for
    `TRASH_KEEP_DAYS` days and can be put back until then. Each is in the
    trash before it leaves the database, so nothing is lost on the way.

    Args:
        project_ids: The projects.

    Returns:
        `deleted`, the IDs moved, and `busy`, the names of those being
        rendered; when `busy` is not empty nothing was moved.
    """
    runtime.job_manager.forget_abandoned()
    before = {project_id: project for project_id in project_ids if (project := runtime.repo.get_project(project_id))}
    filed = runtime.repo.folder_of("projects")
    named = [project.name for project in before.values() if project.name]
    shared.say_in_history(f"把「{'」「'.join(named[:3])}」移到垃圾桶" + (f"等 {len(named)} 支" if len(named) > 3 else ""))
    outputs = str(output_dir())
    trashed = {
        project_id: housekeeping.put_in_trash(outputs, "project", project_id, project.name or "", {
            "project.json": project.model_dump_json().encode("utf-8"),
            "place.json": json.dumps({"folder_id": filed.get(project_id)}).encode("utf-8"),
        }, [])
        for project_id, project in before.items()
    }
    deleted, busy = runtime.repo.delete_projects(list(before))
    for project_id, kept in trashed.items():
        if project_id not in deleted:
            housekeeping.let_go(outputs, kept.key)
    for project_id in deleted:
        for folder in _project_caches(project_id):
            shutil.rmtree(folder, ignore_errors=True)
    housekeeping.empty_old_trash(outputs)
    return {"deleted": deleted, "busy": busy}

def list_trash(expire: bool = False) -> dict:
    """What is in the trash.

    Args:
        expire: First remove for good what has waited longer than the trash
            keeps things. Only the editor asks for that: the AI reads the
            trash and never deletes anything.

    Returns:
        `items`, newest first, each with its `key`, `kind` (`project` or
        `outputs`), `name`, `days_left`, `megabytes` and `files`; and
        `keep_days`.
    """
    outputs = str(output_dir())
    if expire:
        housekeeping.empty_old_trash(outputs)
    now = datetime.now(timezone.utc)
    return {
        "items": [{"key": item.key, "kind": item.kind, "name": item.name, "project_id": item.project_id,
                   "days_left": item.days_left(now), "megabytes": round(item.bytes / 1e6, 1),
                   "files": len(item.files)} for item in housekeeping.in_trash(outputs)],
        "keep_days": housekeeping.TRASH_KEEP_DAYS,
    }

def restore_from_trash(key: str) -> dict:
    """Put something back from the trash: a project as it was, or finished videos where they were.

    The trash lets go of it only once the project is back in the database.

    Args:
        key: Its `key` from `list_trash`.

    Returns:
        Its `kind` and `name`, the `project_id` it is back as, and the `files`
        moved back.

    Raises:
        ValueError: If there is no such thing in the trash.
    """
    outputs = str(output_dir())
    try:
        what, kept = housekeeping.look_in_trash(outputs, key)
    except KeyError as exc:
        raise ValueError(f"{key} is not in the trash") from exc
    project_id = what.project_id
    if what.kind == "project" and "project.json" in kept:
        shared.say_in_history(f"從垃圾桶放回「{what.name}」")
        project = Project.model_validate_json(kept["project.json"])
        if runtime.repo.get_project(project.id) is not None:
            project = project.model_copy(update={"id": str(uuid.uuid4())})
        runtime.repo.add_project(project)
        project_id = project.id
        folder = json.loads(kept.get("place.json", b"{}")).get("folder_id")
        if folder and any(item["id"] == folder for item in runtime.repo.list_folders("projects")):
            runtime.repo.file_items("projects", [project.id], folder)
    back = housekeeping.take_out_of_trash(outputs, key)
    return {"kind": what.kind, "name": what.name, "project_id": project_id, "files": back}

@mcp.tool()
def storage_usage() -> dict:
    """Say how much room clip-mcp takes on this computer, and what can be cleared.

    Use it when the user asks how much space this takes or wants room back.
    Read the sizes out in plain words; never clear anything they did not ask
    to clear.

    Returns:
        `items`, each with a `key`, the `label` the user knows it by, its size
        in `megabytes`, the `path` of its folder, and whether it is
        `clearable` with `clean_storage` — only what can be made again is.
        `legacy` is the old outputs folder from before finished videos were
        saved to the videos folder: sort it out with `tidy_old_outputs`.
        `outputs` is where finished videos, covers and timeline exports are
        saved now, and is never cleared. `trash` is the app's trash: deleted
        projects and finished videos, kept for 30 days.

        `videos`, most room to give back first: each with its `versions`,
        how many are `exported`, the megabytes its finished videos and its
        previews take, and `freeable_megabytes` — what deleting its other
        versions would give back. That, emptying the trash and deleting a
        project are the user's to do in the editor (儲存空間, and each
        video's 版本 tab); suggest them, never claim to have done them.
        `trash` lists what is in the trash.
    """
    items = housekeeping.usage(runtime.WORKSPACE_DIR, str(output_dir()))
    videos = []
    for project in runtime.repo.list_projects():
        found = project_storage(project.id)
        videos.append({"project_id": project.id, "name": project.name, **{
            key: found[key] for key in ("versions", "exported", "outputs_megabytes", "cache_megabytes",
                                        "freeable_megabytes")}})
    return {"items": [
        {"key": item.key, "label": item.label, "megabytes": round(item.bytes / 1e6, 1), "path": item.path,
         "clearable": item.clearable}
        for item in items
    ], "videos": sorted(videos, key=lambda video: -video["freeable_megabytes"]),
        "trash": list_trash()["items"]}

@mcp.tool()
def clean_storage(kinds: List[Literal["previews", "thumbnails", "playback", "work"]]) -> dict:
    """Clear files that can be made again, to give the user disk space back.

    Nothing the user made is touched: projects, analyses and finished videos
    stay. A preview, a thumbnail or a working file still in use by a render is
    left. The next preview of a project renders its pictures again, which
    takes longer once.

    Args:
        kinds: `previews` (preview renders, sound checks and the pictures they
            are made from), `thumbnails` (the editor's thumbnails and
            waveforms), `playback` (the small copies the editor plays a cut
            from, made again when a project is opened), `work` (what analyses
            and renders leave behind).

    Returns:
        `freed_megabytes`.
    """
    runtime.job_manager.forget_abandoned()
    freed = housekeeping.clean(runtime.WORKSPACE_DIR, kinds, runtime.repo.active_job_ids())
    return {"freed_megabytes": round(freed / 1e6, 1)}

@mcp.tool()
def tidy_old_outputs(confirm: bool = False) -> dict:
    """Sort out the old outputs folder, where every render used to be kept.

    Without `confirm`, only says what would happen: the newest finished video,
    cover and export of each project move to the videos folder, and the older
    versions of them go to the recycle bin, from where they can still be
    restored. Sound checks and render logs stay where they are; `clean_storage`
    is not how they go either, so say the folder is left with them. Show the
    user that list and the sizes, and call again with `confirm` only once they
    have agreed to it.

    Args:
        confirm: Carry it out.

    Returns:
        Without `confirm`: `keep` (each file kept and where it goes),
        `recycle` (how many duplicates and megabytes go to the recycle bin) and
        `left_in_place` (what stays in the old folder). With
        it: `kept` (where each went), `recycled`, and `left` — files that
        could not be moved, such as one open in a player.
    """
    planned = housekeeping.legacy_plan(runtime.WORKSPACE_DIR, str(output_dir()))
    if not confirm:
        recycled = [item for item in planned if item.duplicate]
        stays = [item for item in planned if not item.keep_as and not item.duplicate]
        return {
            "keep": [{"from": item.path, "to": item.keep_as, "megabytes": round(item.bytes / 1e6, 1)}
                     for item in planned if item.keep_as],
            "recycle": {"files": len(recycled), "megabytes": round(sum(item.bytes for item in recycled) / 1e6, 1)},
            "left_in_place": {"files": len(stays), "megabytes": round(sum(item.bytes for item in stays) / 1e6, 1)},
        }
    return housekeeping.legacy_apply(runtime.WORKSPACE_DIR, planned)
