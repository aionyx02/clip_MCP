"""The editor's web server: the page, and a JSON surface over the same tools the AI uses.

Every edit goes through the server's own `apply_edits`, so a change made here
is checked by exactly the rules a change made by the AI is, pins the clip it
touched the same way, and is seen by the AI on its next read. Nothing here
writes a project directly except undo, which only ever puts back a state this
page saved a moment before.
"""

import hashlib
import io
import json
import mimetypes
import os
import re
from datetime import datetime, timedelta, timezone
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from app import clients, server
from app.engine.ffmpeg import hidden_window_flags
from app.engine import models
from app.engine import playback as playback_module
from app.engine.frames import extract_frame
from app.engine.subtitles import captioned_clips, place_cues
from app.models.timeline import Project
from app.storage import housekeeping
from app.engine.library import LIBRARIES, MUSIC, library_of
from app.storage.repo import FOLDER_KINDS, FolderNotFoundError
from app.ui import dialogs, updates

STATIC_DIR = Path(__file__).parent / "static"
# What `/api/ping` answers, so a second start can tell this editor from anything else on the port.
APP_ID = "clip-mcp-editor"
CACHE_DIR = Path(housekeeping.ui_cache_dir(server.WORKSPACE_DIR))
# How many states back undo reaches, per project. Each is one project's JSON.
UNDO_DEPTH = 50
# Samples per second kept for a waveform: enough to draw a clip a few pixels per
# tenth of a second wide, small enough that an hour of sound is a few hundred kilobytes.
WAVE_RATE = 50
THUMB_HEIGHTS = (54, 90, 180, 360)

# The editor watches a preview at this short side: enough to read captions and mouths by.
PREVIEW_SIDE = 720
# Words this far either side of a cut are shown while it is being dragged.
WORDS_AROUND = 4.0

_undo: Dict[str, List[Tuple[int, Project]]] = {}
# The preview the editor last asked for, per project: which version, and the render making it.
_previews: Dict[str, dict] = {}
# The finished file the editor last asked for, per project, kept so a page opened again finds it.
_exports: Dict[str, dict] = {}
_previews_lock = threading.Lock()
_undo_lock = threading.Lock()

def _error(message: str, status: int = 400) -> JSONResponse:
    """Answer a request that could not be done, with the reason as the tools give it."""
    return JSONResponse({"error": message}, status_code=status)

def _asset_summary(asset) -> dict:
    """What the page needs to show a file: its name, length and kind, and whether its transcript is a fast one."""
    return {
        "id": asset.id,
        "name": os.path.basename(asset.path),
        "path": asset.path,
        "duration": float(asset.duration) if asset.duration is not None else None,
        "has_video": asset.has_video,
        "has_audio": asset.has_audio,
        "width": asset.width,
        "height": asset.height,
        # Shown on the file, so the user checking captions later knows to expect misheard words.
        "fast_transcript": server.transcription_of(server.repo.transcript_model(asset.id)) == "fast",
        "notes": asset.notes,
        "library": library_of(asset),
    }

async def projects(request: Request) -> Response:
    """List the projects, newest edits first as the store keeps them, or make a new one."""
    if request.method == "POST":
        body = await request.json()
        try:
            made = server.create_project(
                name=str(body.get("name") or ""),
                width=int(body.get("width", 1920)),
                height=int(body.get("height", 1080)),
            )
        except (ValueError, TypeError) as error:
            return _error(str(error))
        server.apply_edits(made["id"], made["version"], server._OPERATIONS.validate_python([
            {"action": "add_track", "track_id": "video", "track_type": "video"},
            {"action": "add_track", "track_id": "music", "track_type": "audio", "duck_under_speech": True},
        ]))
        return JSONResponse({"id": made["id"]})
    listed, filed = [], server.repo.folder_of("projects")
    counts = server.history.counts("projects/")
    everything = server.repo.list_projects()
    branched = {}
    for project in everything:
        if project.branched_from is not None:
            branched[project.branched_from.project_id] = branched.get(project.branched_from.project_id, 0) + 1
    for project in everything:
        base = project.base_video_track
        clips = base.clips if base else []
        used = {clip.asset_id for track in project.tracks for clip in track.clips}
        found = server.repo.get_assets(used)
        missing = sorted(os.path.basename(asset.path) for asset in found.values() if not os.path.exists(asset.path))
        first = clips[0] if clips else None
        listed.append({
            "id": project.id, "name": project.name, "version": project.version,
            "duration": float(project.duration), "width": project.width, "height": project.height,
            "clips": len(clips),
            # A frame a second in, so a shot that opens on black still shows what it is.
            "thumb": {"asset_id": first.asset_id, "t": float(first.source_range.start) + 1.0} if first else None,
            "missing": missing,
            "folder_id": filed.get(project.id),
            "versions": counts.get(project.id, 0),
            "branches": branched.get(project.id, 0),
            "megabytes": server.disk_of(project.id),
        })
    return JSONResponse({"projects": listed})

async def project(request: Request) -> Response:
    """One project as the page draws it: its tracks and clips, its markers, and the files they play."""
    project_id = request.path_params["project_id"]
    found = server.repo.get_project(project_id)
    if not found:
        return _error(f"project {project_id} not found", 404)
    state = json.loads(found.model_dump_json(exclude={"subtitles"}))
    state["duration"] = float(found.duration)
    used = {clip.asset_id for track in found.tracks for clip in track.clips}
    state["assets"] = {asset_id: _asset_summary(asset) for asset_id, asset in server.repo.get_assets(used).items()}
    with _undo_lock:
        stack = _undo.get(project_id, [])
        state["can_undo"] = bool(stack) and stack[-1][0] == found.version
    return JSONResponse(state)

async def edits(request: Request) -> Response:
    """Apply a batch of edits, keeping the state before it for undo."""
    project_id = request.path_params["project_id"]
    body = await request.json()
    before = server.repo.get_project(project_id)
    if not before:
        return _error(f"project {project_id} not found", 404)
    try:
        operations = server._OPERATIONS.validate_python(body.get("operations", []))
        result = server.apply_edits(project_id, int(body["expected_version"]), operations)
    except (ValueError, KeyError, TypeError) as error:
        conflict = "version conflict" in str(error)
        return _error(str(error), 409 if conflict else 400)
    with _undo_lock:
        stack = _undo.setdefault(project_id, [])
        stack.append((result["new_version"], before))
        del stack[:-UNDO_DEPTH]
    return JSONResponse(result)

async def undo(request: Request) -> Response:
    """Put back the state before this page's last edit, if nothing else has edited since."""
    project_id = request.path_params["project_id"]
    current = server.repo.get_project(project_id)
    if not current:
        return _error(f"project {project_id} not found", 404)
    with _undo_lock:
        stack = _undo.get(project_id, [])
        if not stack or stack[-1][0] != current.version:
            # Something else — the AI, another tab — edited since, so the saved state
            # would throw its change away without anyone seeing it go.
            stack.clear()
            return _error("nothing to undo: the project was changed elsewhere since your last edit", 409)
        _, before = stack.pop()
        restored = before.model_copy(update={"version": current.version + 1})
        server.repo.update_project(restored, current.version)
        # The states further back now follow this version, not the one they were saved after.
        if stack:
            stack[-1] = (restored.version, stack[-1][1])
    return JSONResponse({"new_version": restored.version})

def _preview_state(project_id: str) -> dict:
    """Where the project's preview stands: which version it shows, and how far along it is."""
    with _previews_lock:
        entry = dict(_previews.get(project_id) or {})
    if not entry.get("job_id"):
        return entry
    job = server.job_manager.get_job(entry["job_id"])
    if job:
        entry.update({"status": job.status.value, "progress": round(job.progress, 3), "stage": job.stage})
        if job.status.value == "completed" and job.output_path and os.path.exists(job.output_path):
            entry["url"] = f"/output/{job.job_id}"
        if job.error_message:
            entry["error"] = job.error_message
    return entry

def _start_preview(project_id: str) -> dict:
    """Render a preview of the project as it is now, unless one of this version is already on its way.

    A preview of an older version still rendering is stopped: nobody is going to watch it.
    """
    project = server.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    with _previews_lock:
        current = _previews.get(project_id)
    if current and current.get("version") == project.version:
        state = _preview_state(project_id)
        if state.get("status") not in ("failed", "cancelled") or state.get("empty"):
            return state
    if current and current.get("job_id"):
        old = server.job_manager.get_job(current["job_id"])
        if old and old.status.is_active:
            server.cancel_job(old.job_id)
    if not project.base_video_track or not project.base_video_track.clips:
        with _previews_lock:
            _previews[project_id] = {"version": project.version, "empty": True}
        return _preview_state(project_id)
    def start(captions: bool) -> dict:
        return server._start_render(
            project_id, True, server.DEFAULT_LOUDNESS_TARGET, captions, None, True, None, PREVIEW_SIDE,
        )
    try:
        started = start(bool(project.subtitles))
    except ValueError:
        # Captions that no longer land on anything in the cut: preview the cut without them.
        started = start(False)
    with _previews_lock:
        _previews[project_id] = {"version": project.version, "job_id": started["job_id"]}
    return _preview_state(project_id)

async def preview(request: Request) -> Response:
    """The project's preview: POST makes sure one of the current version is coming, GET says how it is going."""
    project_id = request.path_params["project_id"]
    if request.method == "POST":
        try:
            return JSONResponse(await run_in_threadpool(_start_preview, project_id))
        except ValueError as error:
            return _error(str(error))
    return JSONResponse(_preview_state(project_id))

async def version(request: Request) -> Response:
    """The project's version alone, asked every few seconds to notice the AI changing it."""
    found = server.repo.get_project(request.path_params["project_id"])
    if not found:
        return _error("project not found", 404)
    return JSONResponse({"version": found.version})

async def words(request: Request) -> Response:
    """What is said in a stretch of a file, word by word, to see what a cut is about to take or give back."""
    asset_id = request.path_params["asset_id"]
    try:
        at = float(request.query_params.get("at", "0"))
    except ValueError:
        return _error("at is a number")
    analysis = server.repo.get_analysis(asset_id)
    if not analysis or not analysis.transcript:
        return JSONResponse({"words": [], "transcribed": False})
    found = []
    for segment in analysis.transcript.segments:
        if segment.end < at - WORDS_AROUND or segment.start > at + WORDS_AROUND:
            continue
        timed = segment.words or [segment]
        found += [{"start": word.start, "end": word.end, "text": word.text}
                  for word in timed if at - WORDS_AROUND <= word.start <= at + WORDS_AROUND]
    return JSONResponse({"words": found, "transcribed": True})

def _untranscribed(project: Project) -> List[dict]:
    """The files the captions would come from that nobody has transcribed yet."""
    missing, seen = [], set()
    for clip in captioned_clips(project):
        if clip.asset_id in seen:
            continue
        seen.add(clip.asset_id)
        analysis = server.repo.get_analysis(clip.asset_id)
        if analysis is None or analysis.transcript is None:
            asset = server.repo.get_assets([clip.asset_id]).get(clip.asset_id)
            if asset:
                missing.append(_asset_summary(asset))
    return missing

def _speech_model_missing() -> bool:
    """Whether transcribing would first have to download the speech model."""
    folder = models.whisper_dir()
    return not os.path.isdir(folder) or not any(os.scandir(folder))

async def captions(request: Request) -> Response:
    """The project's captions, each where it falls in the cut as it stands."""
    found = server.repo.get_project(request.path_params["project_id"])
    if not found:
        return _error("project not found", 404)
    placed = place_cues(found, found.subtitles)
    return JSONResponse({"captions": [
        {"cue_id": cue.cue_id, "start": float(cue.start), "end": float(cue.end), "text": cue.text}
        for cue in placed
    ]})

async def make_captions(request: Request) -> Response:
    """Caption the cut: first say what would have to be transcribed, then transcribe it, then caption.

    Transcribing is slow and, the first time, downloads the speech model, so it
    only starts once the page says the user agreed (`transcribe`).
    """
    project_id = request.path_params["project_id"]
    body = await request.json()
    found = server.repo.get_project(project_id)
    if not found:
        return _error("project not found", 404)
    missing = _untranscribed(found)
    if missing and not body.get("transcribe"):
        return JSONResponse({"needs": missing, "model_missing": _speech_model_missing(),
                             "replaces": len(found.subtitles)})
    if missing:
        # Captions here are read in Traditional Chinese, like the rest of the editor.
        started = await run_in_threadpool(
            server.analyze_asset, [asset["id"] for asset in missing], True, None, None, "zh-TW",
        )
        return JSONResponse({"jobs": [job["job_id"] for job in started["jobs"]]})
    try:
        made = await run_in_threadpool(server.generate_subtitles, project_id, int(body["expected_version"]))
    except (ValueError, KeyError) as error:
        return _error(str(error), 409 if "version conflict" in str(error) else 400)
    return JSONResponse({"new_version": made["new_version"], "count": len(made["captions"])})

async def jobs(request: Request) -> Response:
    """How a batch of background jobs is getting on, such as the transcriptions captions wait for."""
    ids = [job_id for job_id in request.query_params.get("ids", "").split(",") if job_id]
    try:
        return JSONResponse(server.get_job(ids))
    except ValueError as error:
        return _error(str(error), 404)

# Provisional (roadmap §13): how long a finished job is still reported, so the page can say it
# finished — longer than a window is usually left in the background.
ENDED_SHOWN_FOR = timedelta(seconds=90)

def _job_card(job, names: dict, ahead: int, now: datetime) -> dict:
    """One job as the floating panel shows it: what it is for, how far along, how long is left."""
    preview = bool(job.output_path and os.sep + "previews" + os.sep in job.output_path)
    card = {
        "job_id": job.job_id, "kind": job.kind.value, "preview": preview, "status": job.status.value,
        "progress": round(job.progress, 3), "stage": job.stage,
        "name": names.get(job.asset_id or job.project_id or "", ""),
        "asset_id": job.asset_id, "project_id": job.project_id, "ahead": ahead,
        "remaining_seconds": job.remaining_seconds(now),
        "delivered": bool(job.kind.value == "render" and not preview and job.status.value == "completed"
                          and job.output_path and os.path.exists(job.output_path)),
    }
    return card

def _jobs_to_show() -> dict:
    """Every job that is queued or running, whoever started it, and the ones that just ended."""
    server.job_manager.forget_abandoned()
    now = datetime.now(timezone.utc)
    active, ended = server.repo.jobs_to_show(now - ENDED_SHOWN_FOR)
    wanted = [job.asset_id for job in active + ended if job.asset_id]
    names = {asset_id: os.path.basename(asset.path) for asset_id, asset in server.repo.get_assets(wanted).items()}
    for project_id in {job.project_id for job in active + ended if job.project_id}:
        project = server.repo.get_project(project_id)
        if project is not None:
            names[project_id] = project.name or ""
    ahead = server.queue_positions(active)
    return {
        "jobs": [_job_card(job, names, ahead.get(job.job_id, 0), now) for job in active],
        "ended": [_job_card(job, names, 0, now) for job in ended],
    }

async def active_jobs(request: Request) -> Response:
    """The floating panel's view of the work going on: every source, not only this page's own."""
    return JSONResponse(await run_in_threadpool(_jobs_to_show))

async def cancel_job(request: Request) -> Response:
    """Stop a job the user chose to stop; the AI sees it cancelled when it next asks."""
    body = await request.json()
    try:
        found = await run_in_threadpool(server.cancel_job, str(body.get("job_id", "")))
    except ValueError as error:
        return _error(str(error), 404)
    return JSONResponse(found)

async def assets(request: Request) -> Response:
    """Every imported file the page can put on the timeline, and the folder each is filed in.

    With `all`, files that are no longer where they were imported from are
    listed too, marked `missing`, so the library page can take them out.
    """
    everything = request.query_params.get("all") == "1"
    library = request.query_params.get("library")
    filed = {**server.repo.folder_of("assets"), **server.repo.folder_of("music")}
    analyzing = {job.asset_id for job in server.repo.jobs_to_show(datetime.now(timezone.utc))[0]
                 if job.kind.value == "analyze"}
    listed = []
    heard = server.song_tags() if library == MUSIC else {}
    for asset in server.repo.list_assets():
        if library in LIBRARIES and library_of(asset) != library:
            continue
        present = os.path.exists(asset.path)
        if present or everything:
            entry = {**_asset_summary(asset), "folder_id": filed.get(asset.id), "missing": not present}
            if entry["library"] == MUSIC:
                analysis = server.repo.get_analysis(asset.id)
                entry["tempo"] = analysis.rhythm.tempo if analysis is not None and analysis.rhythm else None
                entry["analyzed"] = analysis is not None and analysis.rhythm is not None
                entry["analyzing"] = asset.id in analyzing
                if analysis is not None and analysis.music is not None:
                    entry.update(heard.get(asset.id, {}))
                    energy = analysis.music.energy
                    # Enough points for the bar on a card, not one a second.
                    step = max(1, len(energy) // ENERGY_POINTS)
                    entry["energy"] = [round(max(energy[index:index + step]), 2) for index in range(0, len(energy), step)]
            listed.append(entry)
    listed.sort(key=lambda asset: asset["name"].lower())
    return JSONResponse({"assets": listed})

# How many bars a song's energy is drawn with on its card.
ENERGY_POINTS = 48

async def move_library(request: Request) -> Response:
    """Move files to the footage or the music library."""
    body = await request.json()
    library = body.get("library")
    if library not in LIBRARIES:
        return _error("library is footage or music")
    try:
        result = await run_in_threadpool(server.set_asset_library, [str(item) for item in body.get("ids", [])], library)
    except ValueError as error:
        return _error(str(error), 404)
    return JSONResponse(result)

async def versions(request: Request) -> Response:
    """The version panel: a video's versions grouped into dots, and the branches started from it."""
    try:
        graph = await run_in_threadpool(server.version_graph, request.path_params["project_id"])
    except ValueError as error:
        return _error(str(error), 404)
    return JSONResponse(graph)

async def restore(request: Request) -> Response:
    """Make the video what it was at one version, as a new version on top."""
    body = await request.json()
    try:
        result = await run_in_threadpool(server.restore_version, request.path_params["project_id"],
                                          str(body["commit"]), int(body["expected_version"]))
    except (ValueError, KeyError, TypeError) as error:
        return _error(str(error), 409 if "version conflict" in str(error) else 400)
    return JSONResponse(result)

async def branch(request: Request) -> Response:
    """Start a branch of the video from one of its versions."""
    body = await request.json()
    try:
        result = await run_in_threadpool(server.branch_project, request.path_params["project_id"],
                                          str(body["name"]), body.get("commit"))
    except (ValueError, KeyError, TypeError) as error:
        return _error(str(error), 400)
    return JSONResponse(result)

async def mark(request: Request) -> Response:
    """Put a star or the published mark on a version, or take it off."""
    body = await request.json()
    if body.get("mark") not in server.VERSION_MARKS:
        return _error("a version is marked starred or published", 400)
    try:
        result = await run_in_threadpool(server.mark_version, request.path_params["project_id"],
                                          str(body["commit"]), body["mark"], bool(body.get("on", True)))
    except (ValueError, KeyError) as error:
        return _error(str(error), 400)
    return JSONResponse(result)

async def playback(request: Request) -> Response:
    """What the browser needs to play a cut as it is, or as it was at one version.

    Asking is also how what it still lacks gets made first: somebody is
    watching, so the copies it plays from go ahead of the other work waiting.
    """
    project_id = request.path_params["project_id"]
    commit = request.query_params.get("commit")
    try:
        project = (server._project_at(project_id, commit) if commit
                   else server.repo.get_project(project_id))
    except ValueError as error:
        return _error(str(error), 404)
    if project is None:
        return _error(f"project {project_id} not found", 404)
    return JSONResponse(await run_in_threadpool(server.playback_of, project))

# What a cached file may be called: what the server names them, and nothing that leaves the folder.
_CACHED_NAME = re.compile(r"^[0-9a-f-]{8,80}\.(mp4|m4a)$")

def _cached(folder: str, name: str, kind: str) -> Response:
    """Serve one prepared file, with ranges so the browser can seek in it."""
    if not _CACHED_NAME.match(name):
        return _error("not found", 404)
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return _error("not ready", 404)
    housekeeping.mark_used(path)
    return FileResponse(path, media_type=kind, headers={"Cache-Control": "max-age=31536000, immutable"})

async def proxy(request: Request) -> Response:
    """A file's small copy, for the browser to play the cut from."""
    name = request.path_params["name"]
    return _cached(playback_module.proxy_dir(server.WORKSPACE_DIR), name,
                   "video/mp4" if name.endswith(".mp4") else "audio/mp4")

async def sound(request: Request) -> Response:
    """A clip's repaired sound, for the browser to play in place of its own."""
    return _cached(playback_module.sound_dir(server.WORKSPACE_DIR), request.path_params["name"], "audio/mp4")

async def asset_notes(request: Request) -> Response:
    """Write the notes on how a file may be used, the ones the AI reads wherever it reads about the file.

    Args:
        request: `asset_id` and `notes`, as JSON.

    Returns:
        The file's `id`, `name` and `notes`, or 404 when it is not in the library.
    """
    body = await request.json()
    try:
        found = await run_in_threadpool(server.edit_asset, str(body.get("asset_id", "")), str(body.get("notes", "")))
    except ValueError as error:
        return _error(str(error), 404)
    return JSONResponse(found)

def _remove_assets(asset_ids: List[str], recycle: bool) -> dict:
    """Take files out of the library, and their originals to the recycle bin when asked.

    Refused as a whole, with nothing removed, when any of them is still in a
    project, in the AI's plan for a cut, or being analyzed: a project that
    loses its footage cannot be played or exported, and the user may not know
    which projects use what.
    """
    found = server.repo.get_assets(asset_ids)
    unknown = sorted(set(asset_ids) - set(found))
    if unknown:
        raise LookupError(f"asset {unknown[0]} not found")
    server.job_manager.forget_abandoned()
    using = server.repo.delete_unused_assets(found)

    def name(asset_id: str) -> str:
        return os.path.basename(found[asset_id].path)

    if using:
        return {
            "removed": [],
            "in_use": [{"name": name(asset_id), "projects": uses["projects"], "plans": uses["plans"]}
                       for asset_id, uses in using.items() if uses["projects"] or uses["plans"]],
            "busy": sorted(name(asset_id) for asset_id, uses in using.items() if uses["busy"]),
        }
    recycled = [asset.path for asset in found.values() if recycle and os.path.exists(asset.path)]
    # One trip to the recycle bin for them all; the library entries are gone either way.
    not_recycled = [] if not recycled or housekeeping.to_recycle_bin(recycled) else [os.path.basename(path) for path in recycled]
    return {"removed": sorted(found), "in_use": [], "busy": [], "not_recycled": not_recycled}

async def delete_assets(request: Request) -> Response:
    """Take files out of the library; `recycle` sends the originals to the recycle bin as well."""
    body = await request.json()
    ids = [str(asset_id) for asset_id in body.get("ids", []) if asset_id]
    if not ids:
        return _error("no files chosen")
    try:
        result = await run_in_threadpool(_remove_assets, ids, bool(body.get("recycle")))
    except LookupError as error:
        return _error(str(error), 404)
    return JSONResponse(result, status_code=409 if result["in_use"] or result["busy"] else 200)

async def delete_projects(request: Request) -> Response:
    """Move projects to the trash. Only from here: deleting is the user's to do, never the AI's."""
    body = await request.json()
    ids = [str(project_id) for project_id in body.get("ids", []) if project_id]
    if not ids:
        return _error("no projects chosen")
    result = await run_in_threadpool(server.trash_projects, ids)
    return JSONResponse(result, status_code=409 if result["busy"] else 200)

async def trash(request: Request) -> Response:
    """What is in the trash (GET), or put one thing back from it (POST)."""
    if request.method == "GET":
        return JSONResponse(await run_in_threadpool(server.list_trash, True))
    body = await request.json()
    try:
        return JSONResponse(await run_in_threadpool(server.restore_from_trash, str(body.get("key", ""))))
    except ValueError as error:
        return _error(str(error), 404)

async def project_storage(request: Request) -> Response:
    """What one video takes on disk, and what deleting its other versions would give back."""
    try:
        return JSONResponse(await run_in_threadpool(server.project_storage, request.path_params["project_id"]))
    except ValueError as error:
        return _error(str(error), 404)

async def trim(request: Request) -> Response:
    """Delete a video's other versions, once the user has typed its name."""
    body = await request.json()
    try:
        return JSONResponse(await run_in_threadpool(server.trim_project, request.path_params["project_id"],
                                                    str(body.get("name", ""))))
    except ValueError as error:
        return _error(str(error), 400)

async def folders(request: Request) -> Response:
    """The folders of the library or of the projects (GET), or make, rename, move or remove one (POST).

    Folders live only in clip-mcp: no file on disk is moved, so filing things
    away can never leave a project without its footage.
    """
    if request.method == "GET":
        kind = request.query_params.get("kind", "")
        if kind not in FOLDER_KINDS:
            return _error("kind is assets or projects")
        return JSONResponse({"folders": server.repo.list_folders(kind)})
    body = await request.json()
    action, folder_id = body.get("action"), body.get("id")
    try:
        if action == "create":
            made = server.repo.create_folder(uuid.uuid4().hex, str(body.get("kind", "")), str(body.get("name", "")),
                                             body.get("parent_id") or None)
            return JSONResponse(made)
        if action == "rename":
            server.repo.rename_folder(str(folder_id), str(body.get("name", "")))
        elif action == "move":
            server.repo.move_folder(str(folder_id), body.get("parent_id") or None)
        elif action == "delete":
            server.repo.delete_folder(str(folder_id))
        else:
            return _error("action is create, rename, move or delete")
    except FolderNotFoundError as error:
        return _error(str(error), 404)
    except ValueError as error:
        return _error(str(error))
    return JSONResponse({"ok": True})

async def file_items(request: Request) -> Response:
    """Put files or projects into a folder, or back at the top when `folder_id` is empty."""
    body = await request.json()
    kind = body.get("kind")
    ids = [str(item_id) for item_id in body.get("ids", []) if item_id]
    known = ({asset.id for asset in server.repo.list_assets()} if kind in ("assets", "music")
             else {project.id for project in server.repo.list_projects()} if kind == "projects" else None)
    if known is None:
        return _error("kind is assets or projects")
    if not ids or not set(ids) <= known:
        return _error("nothing to file, or something that is not there", 404)
    try:
        server.repo.file_items(kind, ids, body.get("folder_id") or None)
    except FolderNotFoundError as error:
        return _error(str(error), 404)
    return JSONResponse({"ok": True})

def _asset_or_none(asset_id: str):
    """The asset by its ID, or None when it is unknown or its file has gone."""
    found = server.repo.get_assets([asset_id]).get(asset_id)
    return found if found and os.path.exists(found.path) else None

async def media(request: Request) -> Response:
    """The source file itself, for the viewer to play; ranges are served so it can seek."""
    asset = _asset_or_none(request.path_params["asset_id"])
    if not asset:
        return _error("file not found", 404)
    kind = mimetypes.guess_type(asset.path)[0] or "application/octet-stream"
    return FileResponse(asset.path, media_type=kind)

def _write_thumb(path: str, seconds: float, height: int, cached: Path) -> None:
    """Decode one frame, shrink it, and keep it as a JPEG."""
    frame = extract_frame(path, seconds, max_size=height * 2)
    frame.thumbnail((height * 4, height))
    cached.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    frame.save(buffer, "JPEG", quality=78)
    partial = cached.with_suffix(".part")
    partial.write_bytes(buffer.getvalue())
    os.replace(partial, cached)

async def thumb(request: Request) -> Response:
    """One frame of a file, small, for the media bin and the clips on the timeline."""
    asset = _asset_or_none(request.path_params["asset_id"])
    if not asset or not asset.has_video:
        return _error("no picture", 404)
    try:
        seconds = round(float(request.query_params.get("t", "0")), 1)
        wanted = int(request.query_params.get("h", "90"))
    except ValueError:
        return _error("t and h are numbers")
    height = min(THUMB_HEIGHTS, key=lambda size: abs(size - wanted))
    key = hashlib.sha1(f"{asset.path}|{os.path.getmtime(asset.path)}|{seconds}|{height}".encode()).hexdigest()
    cached = CACHE_DIR / "thumbs" / f"{key}.jpg"
    if not cached.exists():
        try:
            # Decoding runs off the event loop: a timeline asks for dozens of these at once,
            # and every other request would wait behind them.
            await run_in_threadpool(_write_thumb, asset.path, seconds, height, cached)
        except RuntimeError:
            return _error("no frame there", 404)
    return FileResponse(cached, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

async def wave(request: Request) -> Response:
    """How loud a file is over time, as peaks the timeline draws its waveforms from."""
    asset = _asset_or_none(request.path_params["asset_id"])
    if not asset or not asset.has_audio:
        return JSONResponse({"rate": WAVE_RATE, "peaks": []})
    key = hashlib.sha1(f"{asset.path}|{os.path.getmtime(asset.path)}|{WAVE_RATE}".encode()).hexdigest()
    cached = CACHE_DIR / "waves" / f"{key}.json"
    if not cached.exists():
        await run_in_threadpool(_write_wave, asset.path, cached)
    return FileResponse(cached, media_type="application/json", headers={"Cache-Control": "max-age=86400"})

def _write_wave(path: str, cached: Path) -> None:
    """Measure a file's loudness peaks and keep them."""
    samples_per_peak = 40
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path,
         "-map", "0:a:0", "-ac", "1", "-ar", str(WAVE_RATE * samples_per_peak), "-f", "s16le", "-"],
        stdin=subprocess.DEVNULL, capture_output=True, creationflags=hidden_window_flags(),
    )
    samples = np.frombuffer(result.stdout, dtype=np.int16)
    usable = len(samples) - len(samples) % samples_per_peak
    peaks = np.abs(samples[:usable].reshape(-1, samples_per_peak)).max(axis=1) / 32768.0 if usable else np.zeros(0)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"rate": WAVE_RATE, "peaks": [round(float(peak), 3) for peak in peaks]}))

FRAMES = ("landscape", "portrait", "square")

def _export_settings(source) -> Tuple[Optional[str], bool]:
    """The shape and captions an export was asked for, from a query or a body."""
    frame = source.get("frame") or None
    return (frame if frame in FRAMES else None), str(source.get("captions", "")).lower() in ("1", "true", "yes")

async def check(request: Request) -> Response:
    """What the finished file would show that the user should hear about before it is made."""
    frame, captions = _export_settings(request.query_params)
    try:
        return JSONResponse(await run_in_threadpool(server.check_render, request.path_params["project_id"], frame, captions))
    except ValueError as error:
        return _error(str(error), 404)

def _export_state(project_id: str) -> dict:
    """The project's last export, as the page shows it: how far along, and where the file went."""
    with _previews_lock:
        entry = dict(_exports.get(project_id) or {})
    job = server.job_manager.get_job(entry["job_id"]) if entry.get("job_id") else None
    if job:
        entry.update({"status": job.status.value, "progress": round(job.progress, 3), "stage": job.stage,
                      "file": os.path.basename(job.output_path or ""), "folder": os.path.dirname(job.output_path or "")})
        if job.error_message:
            entry["error"] = job.error_message
    return entry

async def export(request: Request) -> Response:
    """Make the finished file (POST), or say how the last one is getting on (GET).

    Only kinds of finding the user chose to go ahead with are passed on; the
    server refuses the render over anything else, the same as for the AI.
    """
    project_id = request.path_params["project_id"]
    if request.method == "GET":
        return JSONResponse(_export_state(project_id))
    body = await request.json()
    frame, captions = _export_settings(body)
    try:
        started = await run_in_threadpool(
            server.render_project, project_id, False, server.DEFAULT_LOUDNESS_TARGET, captions, frame, True,
            body.get("allow") or None,
        )
    except ValueError as error:
        return _error(str(error))
    with _previews_lock:
        _exports[project_id] = {"job_id": started["job_id"]}
    return JSONResponse(_export_state(project_id))

def _finished_file(job_id: str) -> Optional[str]:
    """A finished render's file, by its job: the only files the page may ask to open."""
    job = server.job_manager.get_job(job_id)
    if job and job.status.value == "completed" and job.output_path and os.path.exists(job.output_path):
        return job.output_path
    return None

async def reveal(request: Request) -> Response:
    """Show a finished file or a file of the library in File Explorer, selected, or play it in the system's player."""
    body = await request.json()
    if body.get("asset_id"):
        # By the library's own record, so the page can only ever open a file the user imported.
        found = _asset_or_none(str(body["asset_id"]))
        path = found.path if found is not None else None
    else:
        path = _finished_file(str(body.get("job_id", "")))
    if not path:
        return _error("that file is not there", 404)
    if body.get("play"):
        os.startfile(path)
    else:
        subprocess.Popen(["explorer", "/select,", path])
    return JSONResponse({"path": path})

async def job(request: Request) -> Response:
    """How a render is getting on."""
    try:
        found = server.get_job([request.path_params["job_id"]])["jobs"][0]
    except ValueError as error:
        return _error(str(error), 404)
    if found.get("output_path") and found["status"] == "completed":
        found["url"] = f"/output/{found['job_id']}"
    return JSONResponse(found)

async def output(request: Request) -> Response:
    """A finished render, to play in the viewer or save."""
    found = server.job_manager.get_job(request.path_params["job_id"])
    if not found or not found.output_path or not os.path.exists(found.output_path):
        return _error("no file for that render", 404)
    return FileResponse(found.output_path, media_type="video/mp4")

async def pick(request: Request) -> Response:
    """Ask Windows for files or a folder, and import what was chosen where it is."""
    body = await request.json()
    kind = body.get("kind") if body.get("kind") in ("folder", "music") else "files"
    # Added on the music page, or as music for a project, a file is music whatever it sounds like.
    library = MUSIC if kind == "music" or body.get("library") == MUSIC else None
    chosen = await run_in_threadpool(dialogs.pick, kind)
    imported, failed, ids = 0, [], []
    for path in chosen:
        try:
            if kind == "folder":
                result = server.import_folder(path)
                imported += len(result["assets"])
                ids += [item["id"] for item in result["assets"]]
                failed += [os.path.basename(item["path"]) for item in result["skipped"]]
            else:
                ids.append(server._register_asset(path, library).id)
                imported += 1
        except (FileNotFoundError, ValueError, RuntimeError):
            failed.append(os.path.basename(path))
    if library and ids:
        server.set_asset_library(ids, library)
    found = server.repo.get_assets(ids)
    music = [asset_id for asset_id in ids if asset_id in found and library_of(found[asset_id]) == MUSIC]
    # Chosen with one of the library's folders open: filed there, where the user is looking.
    folder_id = body.get("folder_id")
    if folder_id and ids:
        here = music if body.get("library") == MUSIC else [asset_id for asset_id in ids if asset_id not in music]
        try:
            server.repo.file_items("music" if body.get("library") == MUSIC else "assets", here, str(folder_id))
        except FolderNotFoundError:
            # The folder went while the window was open; the files are in the library all the same.
            pass
    return JSONResponse({"chosen": len(chosen), "imported": imported, "failed": failed, "assets": ids,
                         "music": len(music)})

async def storage(request: Request) -> Response:
    """What the workspace holds, or clear the kinds the user chose."""
    if request.method == "POST":
        body = await request.json()
        kinds = [kind for kind in body.get("kinds", []) if kind in ("previews", "thumbnails", "playback", "work")]
        return JSONResponse(server.clean_storage(kinds))
    return JSONResponse(server.storage_usage())

def _openable() -> Dict[str, str]:
    """The folders the page may open, by name: never a path the page made up."""
    folders = {"outputs": str(server.output_dir())}
    folders.update({item.key: item.path for item in housekeeping.usage(server.WORKSPACE_DIR, folders["outputs"])})
    return folders

async def open_folder(request: Request) -> Response:
    """Open one of the app's folders in File Explorer."""
    body = await request.json()
    path = _openable().get(str(body.get("folder")))
    if not path:
        return _error("unknown folder", 404)
    os.makedirs(path, exist_ok=True)
    os.startfile(path)
    return JSONResponse({"opened": path})

def _clients() -> List[dict]:
    """Where each AI client stands. Only looks: nothing is written to find out."""
    states = []
    for key, result in zip(clients.CLIENTS, clients.register(clients.server_command(), dry_run=True)):
        states.append({
            "key": key, "name": result.client,
            "state": {"unchanged": "connected", "would register": "not_connected", "skipped": "not_installed",
                      "confirm": "confirm"}.get(result.status, "failed"),
            # The file a change would touch, to say so before asking; a link is the server's to open, never the page's.
            "detail": "" if result.status == "confirm" else result.detail,
        })
    return states

async def ai_clients(request: Request) -> Response:
    """Which AI clients can use clip-mcp; POST connects or disconnects the one the user chose.

    One client per request, and only the one named: changing another
    program's settings is something the user does, client by client.
    """
    if request.method == "POST":
        body = await request.json()
        key = body.get("key")
        if key not in clients.CLIENTS or key == "cherry-studio":
            return _error("that client is not connected from here", 404)
        if body.get("action") == "disconnect":
            result = (await run_in_threadpool(clients.unregister, [key], False, clients.server_command()))[0]
        else:
            result = (await run_in_threadpool(clients.register, clients.server_command(), [key]))[0]
        if result.status == "failed":
            return _error(result.detail)
    return JSONResponse({"clients": await run_in_threadpool(_clients)})

async def open_client(request: Request) -> Response:
    """Offer this server to a client that only takes one through a link it asks the user about."""
    body = await request.json()
    if body.get("key") != "cherry-studio":
        return _error("that client is connected from here, not by a link", 404)
    result = clients.register_cherry_studio(clients.server_command(), dry_run=True)
    if result.status != "confirm":
        return _error("Cherry Studio is not installed", 404)
    os.startfile(result.detail)
    return JSONResponse({"opened": True})

async def update(request: Request) -> Response:
    """Whether a newer version is out (GET), or install it now (POST).

    Never while a render is running: the installer closes clip-mcp, and a
    render half done would be lost.
    """
    if request.method == "GET":
        return JSONResponse(await run_in_threadpool(updates.latest))
    server.job_manager.forget_abandoned()
    if server.repo.active_job_ids():
        return _error("busy", 409)
    try:
        installing = await run_in_threadpool(updates.start_update, lambda: os._exit(0))
    except (OSError, RuntimeError) as error:
        return _error(str(error))
    return JSONResponse({"installing": installing})

async def ping(request: Request) -> Response:
    """Say this is the editor, and for which workspace, so a second start opens this one instead of another.

    The workspace matters: an editor for a test workspace and one for the real
    workspace are two different editors, even when one finds the other's port.
    """
    return JSONResponse({"app": APP_ID, "workspace": os.path.normcase(os.path.abspath(server.WORKSPACE_DIR))})

async def index(request: Request) -> Response:
    """The page."""
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

class Activity:
    """When the page last asked for anything; the launcher stops the server once it has been quiet."""

    last = time.monotonic()

# What each kind of change made in the editor is, for the version it leaves in the history.
EDITOR_DOING = (
    ("/edits", "在編輯器修改"), ("/undo", "復原上一步"), ("/captions/make", "產生字幕"),
    ("/api/projects/delete", "把專案移到垃圾桶"), ("/api/trash", "從垃圾桶放回"), ("/api/projects", "新增專案"), ("/api/assets/notes", "寫素材備註"),
    ("/api/assets/delete", "從素材庫拿掉"), ("/api/folders", "整理資料夾"), ("/api/file", "整理位置"),
    ("/api/pick", "加入素材"), ("/api/assets/library", "移動分類"), ("/versions/restore", "回到舊版本"), ("/versions/branch", "開分支"),
)

class _Stamp:
    """Note every request as a sign the window is still open, and keep each change as one version."""

    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.application(scope, receive, send)
            return
        Activity.last = time.monotonic()
        if scope["method"] != "POST":
            await self.application(scope, receive, send)
            return
        path = scope["path"]
        doing = next((said for ending, said in EDITOR_DOING if path.endswith(ending)), "在編輯器操作")
        with server.history.step("編輯器", doing=doing, later=True) as batch:
            await self.application(scope, receive, send)
        # Committing waits on the disk and a lock the MCP server shares: off the event loop.
        await run_in_threadpool(server.history.finish, batch)

class _FreshStatic(StaticFiles):
    """The page's own files, checked with the server every time.

    After an update the window must not keep running yesterday's script from
    its cache; asking costs a 304 on a machine-local server.
    """

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response

def create_app() -> Starlette:
    """Build the editor's web application."""
    return Starlette(middleware=[Middleware(_Stamp)], routes=[
        Route("/", index),
        Route("/api/projects", projects, methods=["GET", "POST"]),
        # Before the route that takes a project's ID, or "delete" would be read as one.
        Route("/api/projects/delete", delete_projects, methods=["POST"]),
        Route("/api/projects/{project_id}", project),
        Route("/api/projects/{project_id}/edits", edits, methods=["POST"]),
        Route("/api/projects/{project_id}/undo", undo, methods=["POST"]),
        Route("/api/projects/{project_id}/check", check),
        Route("/api/projects/{project_id}/export", export, methods=["GET", "POST"]),
        Route("/api/reveal", reveal, methods=["POST"]),
        Route("/api/projects/{project_id}/preview", preview, methods=["GET", "POST"]),
        Route("/api/projects/{project_id}/version", version),
        Route("/api/projects/{project_id}/playback", playback),
        Route("/api/projects/{project_id}/versions", versions),
        Route("/api/projects/{project_id}/versions/restore", restore, methods=["POST"]),
        Route("/api/projects/{project_id}/versions/branch", branch, methods=["POST"]),
        Route("/api/projects/{project_id}/versions/mark", mark, methods=["POST"]),
        Route("/api/projects/{project_id}/storage", project_storage),
        Route("/api/projects/{project_id}/storage/trim", trim, methods=["POST"]),
        Route("/api/trash", trash, methods=["GET", "POST"]),
        Route("/api/assets/library", move_library, methods=["POST"]),
        Route("/api/words/{asset_id}", words),
        Route("/api/projects/{project_id}/captions", captions),
        Route("/api/projects/{project_id}/captions/make", make_captions, methods=["POST"]),
        Route("/api/jobs", jobs),
        Route("/api/jobs/active", active_jobs),
        Route("/api/jobs/cancel", cancel_job, methods=["POST"]),
        Route("/api/assets", assets),
        Route("/api/assets/delete", delete_assets, methods=["POST"]),
        Route("/api/assets/notes", asset_notes, methods=["POST"]),
        Route("/api/folders", folders, methods=["GET", "POST"]),
        Route("/api/file", file_items, methods=["POST"]),
        Route("/api/pick", pick, methods=["POST"]),
        Route("/api/storage", storage, methods=["GET", "POST"]),
        Route("/api/open-folder", open_folder, methods=["POST"]),
        Route("/api/clients", ai_clients, methods=["GET", "POST"]),
        Route("/api/clients/open", open_client, methods=["POST"]),
        Route("/api/ping", ping),
        Route("/api/update", update, methods=["GET", "POST"]),
        Route("/api/jobs/{job_id}", job),
        Route("/media/{asset_id}", media),
        Route("/proxy/{name}", proxy),
        Route("/sound/{name}", sound),
        Route("/thumb/{asset_id}", thumb),
        Route("/wave/{asset_id}", wave),
        Route("/output/{job_id}", output),
        Mount("/static", _FreshStatic(directory=STATIC_DIR), name="static"),
    ])
