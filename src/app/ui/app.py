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
import subprocess
import threading
import time
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
from app.engine.frames import extract_frame
from app.models.timeline import Project
from app.storage import housekeeping
from app.ui import dialogs

STATIC_DIR = Path(__file__).parent / "static"
# What `/api/ping` answers, so a second start can tell this editor from anything else on the port.
APP_ID = "clip-mcp-editor"
CACHE_DIR = Path(housekeeping.ui_cache_dir(server.WORKSPACE_DIR))
# How many states back undo reaches, per project. Each is one project's JSON.
UNDO_DEPTH = 50
# Samples per second kept for a waveform: enough to draw a clip a few pixels per
# tenth of a second wide, small enough that an hour of sound is a few hundred kilobytes.
WAVE_RATE = 50
THUMB_HEIGHTS = (54, 90, 180)

_undo: Dict[str, List[Tuple[int, Project]]] = {}
_undo_lock = threading.Lock()

def _error(message: str, status: int = 400) -> JSONResponse:
    """Answer a request that could not be done, with the reason as the tools give it."""
    return JSONResponse({"error": message}, status_code=status)

def _asset_summary(asset) -> dict:
    """What the page needs to show a file: its name, length and kind."""
    return {
        "id": asset.id,
        "name": os.path.basename(asset.path),
        "path": asset.path,
        "duration": float(asset.duration) if asset.duration is not None else None,
        "has_video": asset.has_video,
        "has_audio": asset.has_audio,
        "width": asset.width,
        "height": asset.height,
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
    listed = []
    for project in server.repo.list_projects():
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

async def assets(request: Request) -> Response:
    """Every imported file the page can put on the timeline."""
    listed = [_asset_summary(asset) for asset in server.repo.list_assets() if os.path.exists(asset.path)]
    listed.sort(key=lambda asset: asset["name"].lower())
    return JSONResponse({"assets": listed})

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
            frame = extract_frame(asset.path, seconds, max_size=height * 2)
        except RuntimeError:
            return _error("no frame there", 404)
        frame.thumbnail((height * 4, height))
        cached.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        frame.save(buffer, "JPEG", quality=78)
        cached.write_bytes(buffer.getvalue())
    return FileResponse(cached, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

async def wave(request: Request) -> Response:
    """How loud a file is over time, as peaks the timeline draws its waveforms from."""
    asset = _asset_or_none(request.path_params["asset_id"])
    if not asset or not asset.has_audio:
        return JSONResponse({"rate": WAVE_RATE, "peaks": []})
    key = hashlib.sha1(f"{asset.path}|{os.path.getmtime(asset.path)}|{WAVE_RATE}".encode()).hexdigest()
    cached = CACHE_DIR / "waves" / f"{key}.json"
    if not cached.exists():
        sample_rate = WAVE_RATE * 40
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", asset.path,
             "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
            stdin=subprocess.DEVNULL, capture_output=True, creationflags=hidden_window_flags(),
        )
        samples = np.frombuffer(result.stdout, dtype=np.int16)
        usable = len(samples) - len(samples) % 40
        peaks = (np.abs(samples[:usable].reshape(-1, 40)).max(axis=1) / 32768.0) if usable else np.zeros(0)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"rate": WAVE_RATE, "peaks": [round(float(peak), 3) for peak in peaks]}))
    return FileResponse(cached, media_type="application/json", headers={"Cache-Control": "max-age=86400"})

async def render(request: Request) -> Response:
    """Start a render of the project, a preview unless the full file is asked for."""
    project_id = request.path_params["project_id"]
    body = await request.json()
    try:
        started = server.render_project(
            project_id,
            is_preview=bool(body.get("preview", True)),
            allow=body.get("allow") or None,
        )
    except ValueError as error:
        return _error(str(error))
    return JSONResponse(started)

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
    kind = "folder" if body.get("kind") == "folder" else "files"
    chosen = await run_in_threadpool(dialogs.pick, kind)
    imported, failed = 0, []
    for path in chosen:
        try:
            if kind == "folder":
                result = server.import_folder(path)
                imported += len(result["assets"])
                failed += [os.path.basename(item["path"]) for item in result["skipped"]]
            else:
                server.import_asset(path)
                imported += 1
        except (FileNotFoundError, ValueError, RuntimeError):
            failed.append(os.path.basename(path))
    return JSONResponse({"chosen": len(chosen), "imported": imported, "failed": failed})

async def storage(request: Request) -> Response:
    """What the workspace holds, or clear the kinds the user chose."""
    if request.method == "POST":
        body = await request.json()
        kinds = [kind for kind in body.get("kinds", []) if kind in ("previews", "thumbnails", "work")]
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

def _clients(dry_run: bool) -> List[dict]:
    """Where each AI client stands, or registering with the ones that are installed."""
    command = clients.server_command()
    states = []
    for key, result in zip(clients.CLIENTS, clients.register(command, dry_run=dry_run)):
        states.append({
            "key": key, "name": result.client,
            "state": {"unchanged": "connected", "registered": "connected", "would register": "not_connected",
                      "skipped": "not_installed"}.get(result.status, "failed"),
            "detail": result.detail,
        })
    return states

async def ai_clients(request: Request) -> Response:
    """Which AI clients can use clip-mcp; POST connects every installed one."""
    dry_run = request.method != "POST"
    return JSONResponse({"clients": await run_in_threadpool(_clients, dry_run)})

async def ping(request: Request) -> Response:
    """Say this is the editor, so a second start opens this one instead of another."""
    return JSONResponse({"app": APP_ID})

async def index(request: Request) -> Response:
    """The page."""
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

class Activity:
    """When the page last asked for anything; the launcher stops the server once it has been quiet."""

    last = time.monotonic()

class _Stamp:
    """Note every request as a sign the window is still open."""

    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            Activity.last = time.monotonic()
        await self.application(scope, receive, send)

def create_app() -> Starlette:
    """Build the editor's web application."""
    return Starlette(middleware=[Middleware(_Stamp)], routes=[
        Route("/", index),
        Route("/api/projects", projects, methods=["GET", "POST"]),
        Route("/api/projects/{project_id}", project),
        Route("/api/projects/{project_id}/edits", edits, methods=["POST"]),
        Route("/api/projects/{project_id}/undo", undo, methods=["POST"]),
        Route("/api/projects/{project_id}/render", render, methods=["POST"]),
        Route("/api/assets", assets),
        Route("/api/pick", pick, methods=["POST"]),
        Route("/api/storage", storage, methods=["GET", "POST"]),
        Route("/api/open-folder", open_folder, methods=["POST"]),
        Route("/api/clients", ai_clients, methods=["GET", "POST"]),
        Route("/api/ping", ping),
        Route("/api/jobs/{job_id}", job),
        Route("/media/{asset_id}", media),
        Route("/thumb/{asset_id}", thumb),
        Route("/wave/{asset_id}", wave),
        Route("/output/{job_id}", output),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ])
