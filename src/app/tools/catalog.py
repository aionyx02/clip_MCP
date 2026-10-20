"""The library: importing files and folders, listing and organising them, notes, and moving files on disk."""

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Dict, Iterable, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from app.models.media import Asset
from app.models.job import JobKind
from app.engine import stills
from app.engine.probe import picture_size, probe_file, recorded_at
from app.engine.library import FOLDER_KINDS, FOOTAGE, LIBRARIES, MUSIC, classify, library_of
from app.workspace import output_dir
from app import runtime
from app.runtime import mcp
from app.tools import examining, shared

MEDIA_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".mts", ".m2ts", ".wmv", ".flv", ".3gp",
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff",
}) | stills.IMAGE_EXTENSIONS

@mcp.tool()
def inspect_media(filepath: str) -> dict:
    """Inspect a local media file and return its technical metadata.

    Use this to check duration, resolution, frame rate, codecs, and audio
    streams before adding a file to a project.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.

    Returns:
        ffprobe output with a `format` object and a `streams` list.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    return probe_file(filepath)

def natural_key(name: str) -> List[object]:
    """Build a sort key that orders embedded numbers by value.

    Camera and phone exports are often numbered without padding, so plain
    text order would put `clip10` before `clip2`.

    Args:
        name: File name to build a key for.

    Returns:
        The name split into text and integer parts, lowercased.
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

def register_asset(filepath: str, library: Optional[str] = None) -> Asset:
    """Probe a media file and save it as an asset, in the footage or the music library.

    A new file without a picture is listened to: mostly somebody talking, it
    is footage; otherwise music, and a song is analyzed in the background
    straight away.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.
        library: Put it in this library rather than the one it sounds like,
            as when it is added on the editor's music page.

    Returns:
        The saved asset. A path that was imported before keeps its asset ID
        and its library.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    path = os.path.abspath(filepath)
    info = probe_file(path)

    existing = runtime.repo.find_asset_by_path(path)
    streams = info.get("streams", [])
    duration = info.get("format", {}).get("duration")
    photo = stills.is_image(path)
    if photo:
        # A photo is as long as it is shown.
        duration = stills.STILL_LONGEST
    asset = Asset(
        id=existing.id if existing else str(uuid.uuid4()),
        path=path,
        still=photo,
        duration=Decimal(duration) if duration is not None else None,
        has_video=any(
            stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")
            for stream in streams
        ),
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )
    size = picture_size(info)
    if size is not None:
        asset.width, asset.height = size
    asset.recorded_at = recorded_at(info, path)
    if existing is not None:
        asset.notes = existing.notes
        asset.licence = existing.licence
        asset.library = existing.library
        asset.source = existing.source
    if library in LIBRARIES:
        asset.library = library
    elif asset.library is None:
        asset.library = classify(asset)
    runtime.repo.save_asset(asset)
    _queue_music([asset])
    return asset

# Whether a song added to the music library starts being analyzed. Off where nothing should
# run in the background unasked, such as a test run.
ANALYZE_MUSIC_ON_ADD = True

def _queue_music(assets: Iterable[Asset]) -> List[str]:
    """Start analyzing songs of the music library that have no current analysis, behind other work.

    Args:
        assets: Files just added, moved to the music library, or found
            waiting; anything else among them is left alone.

    Returns:
        The IDs of the jobs started.
    """
    if not ANALYZE_MUSIC_ON_ADD:
        return []
    busy = {job.asset_id for job in runtime.repo.jobs_to_show(datetime.now(timezone.utc))[0] if job.kind == JobKind.ANALYZE}
    started = []
    for asset in assets:
        if library_of(asset) != MUSIC or asset.duration is None or asset.id in busy:
            continue
        existing = runtime.repo.get_analysis(asset.id)
        if existing is not None and existing.rhythm is not None and existing.music is not None \
                and not examining.is_stale(existing):
            continue
        # Behind the footage somebody asked about: a song can wait until it is wanted.
        started.append(examining.start_analysis(asset, False, "accurate", priority=1).job_id)
    return started

def tend_library() -> None:
    """Sort the files added before there were two libraries, and queue the songs not yet analyzed.

    Run once when the server or the editor starts, in the background: listening
    takes a moment per file. Whichever starts first does it; the other finds
    it taken and leaves it.
    """
    from filelock import FileLock, Timeout

    try:
        with FileLock(os.path.join(runtime.WORKSPACE_DIR, "library.lock"), timeout=0):
            for asset in runtime.repo.list_assets():
                if asset.library is None and os.path.exists(asset.path):
                    try:
                        asset.library = classify(asset)
                    except (OSError, RuntimeError, ValueError):
                        # Left where its picture says until it can be listened to; the next start tries again.
                        continue
                    runtime.repo.save_asset(asset)
            _queue_music(runtime.repo.list_assets())
    except Timeout:
        return

@mcp.tool()
def import_asset(filepath: str) -> dict:
    """Register a local media file as an asset that clips can reference.

    This is what puts a file in the library, the only files a project can use;
    the user sees the library in the editor app, files are never copied, and
    a file the user takes out of the library there loses its asset ID.
    Importing the same path again re-probes the file and keeps its asset ID.
    Video-track clips need `has_video`, and audio-track clips, such as
    background music, need `has_audio`. Cover art embedded in audio files is
    not counted as video. Video assets without audio contribute silence.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.

    Returns:
        The asset serialized as a dictionary: its `id`, absolute `path`,
        `duration` in seconds (or null if unknown, and for a photo, which
        lasts as long as its clip; `still` is true for one), and
        `has_video` / `has_audio` flags.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    return shared.plain(_asset_record(register_asset(filepath)))

def _asset_record(asset: Asset) -> dict:
    """An asset as a tool hands it back: a photo with no length, since it has none of its own."""
    record = asset.model_dump()
    if asset.still:
        record["duration"] = None
    return record

@mcp.tool()
def list_assets(
    kind: Optional[Literal["footage", "music"]] = None,
    folder_id: Optional[str] = None,
    text: Optional[str] = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> dict:
    """List imported assets a page at a time, and say which of them have been analyzed.

    A library grows into the hundreds, so read it the way the user sorted it:
    by the folders they made in the editor, or by part of a file name.

    Args:
        kind: `footage` — what was shot, and recordings of people talking —
            or `music`; omit for both. A song is listed with its `tempo` in
            beats per minute, the `moods` and `styles` that set it apart in
            this library, how it `ends` and where it is `fullest`, once
            analyzed; `analyzing` while it is. The mood is how it sounds to a
            model, right about half the time: say it as 「聽起來偏…」, never as
            a verdict.
        folder_id: Only the files filed in this folder, from `folders`. Its
            sub-folders are their own; ask for them by their own IDs.
        text: Only files whose name contains this.
        limit: Largest number of files to return.
        offset: How many matching files to skip, for the next page.

    Returns:
        A dictionary with `assets`, one short record per file — `id`, `name`,
        `duration` in seconds (null for a photo, which lasts as long as its
        clip; `photo` is true for one), `has_video`, `has_audio`, `analyzed`,
        `transcription` (`fast`, `balanced`, `accurate`, or null when speech was not
        transcribed), `stale`, `library`, `folder_id` and, where the file has
        them, its `notes` — how it may be used — in name order; `total`, how
        many match; `offset`; and `folders`, every folder of that library (the
        footage's when `kind` is omitted) with its `id`, `name` and
        `parent_id`. `stale` is true for an asset analyzed
        with detection settings or a speech model the server no longer uses:
        its analysis still works, but analyze those assets again before
        comparing them with each other. `inspect_media` has a file's full path,
        size and frame rate.
    """
    filed = {**runtime.repo.folder_of("assets"), **runtime.repo.folder_of("music")}
    matching = [
        asset for asset in runtime.repo.list_assets()
        if (kind is None or library_of(asset) == kind)
        and (folder_id is None or filed.get(asset.id) == folder_id)
        and (not text or text.lower() in os.path.basename(asset.path).lower())
    ]
    analyzing = examining.analyzing_assets()
    heard = examining.song_tags() if kind == MUSIC or any(library_of(asset) == MUSIC for asset in matching) else {}
    matching.sort(key=lambda asset: natural_key(os.path.basename(asset.path)))
    assets = []
    for asset in matching[offset:offset + limit]:
        analysis = runtime.repo.get_analysis(asset.id)
        assets.append({
            "id": asset.id,
            "name": os.path.basename(asset.path),
            # A photo has no length of its own: it lasts as long as a clip of it is made to.
            "duration": None if asset.duration is None or asset.still else float(asset.duration),
            "has_video": asset.has_video,
            "has_audio": asset.has_audio,
            **({"photo": True} if asset.still else {}),
            "analyzed": analysis is not None,
            "transcription": examining.transcription_kind(analysis),
            "stale": analysis is not None and examining.is_stale(analysis),
            "library": library_of(asset),
            "folder_id": filed.get(asset.id),
            **({"notes": asset.notes} if asset.notes else {}),
            **(examining.song_details(asset, analysis, heard, analyzing) if library_of(asset) == MUSIC
               else {"analyzing": True} if asset.id in analyzing else {}),
        })
    return {"assets": assets, "total": len(matching), "offset": offset,
            "folders": runtime.repo.list_folders(_folder_kind(kind or FOOTAGE))}

def _folder_kind(library: str) -> str:
    """The kind of folder a library's files are filed in."""
    return FOLDER_KINDS[library]

@mcp.tool()
def set_asset_library(asset_ids: List[str], library: Literal["footage", "music"], note: str = "") -> dict:
    """Move files between the footage and the music library.

    A file without a picture is put in one when it is added, by whether it is
    mostly somebody talking; that can be wrong — a song that is mostly a
    spoken intro, a recording of somebody singing. Move it when the user says
    so or when what you hear says otherwise, and say in your reply that you
    did. A song moved to music is analyzed for its beat in the background;
    footage is transcribed when analyzed, music never is.

    Args:
        asset_ids: The files.
        library: `footage` or `music`.
        note: The user's words for why, kept in the library's history.

    Returns:
        `moved`, the names of the files moved; a file already there is not
        among them. Each now sits at the top of that library's folders. A
        recording moved to footage is transcribed in the background.

    Raises:
        ValueError: If a file is not in the library, or a file with a
            picture is moved to music.
    """
    assets = [shared.get_asset(asset_id) for asset_id in asset_ids]
    pictured = [os.path.basename(asset.path) for asset in assets if asset.has_video]
    if library == MUSIC and pictured:
        raise ValueError(f"{', '.join(pictured)} has a picture, so it is footage: music is sound alone")
    moved = runtime.repo.move_to_library([asset.id for asset in assets], library)
    _queue_music(moved)
    if library == FOOTAGE and ANALYZE_MUSIC_ON_ADD:
        # Heard as a song, it was never transcribed: now it is listened to for what is said.
        for asset in moved:
            existing = runtime.repo.get_analysis(asset.id)
            if existing is not None and existing.transcript is None and asset.has_audio:
                examining.start_analysis(asset, True, "accurate", priority=1)
    return {"moved": [os.path.basename(asset.path) for asset in moved]}

@mcp.tool()
def edit_asset(asset_id: str, notes: str, note: str = "") -> dict:
    """Write down how a file may be used, so it is read wherever the file is.

    For an instruction about the footage itself — 「請消音」, 「可不放」,
    「0:12 以後手入鏡」. Users often put these in the file's name; when you see
    one there, follow it, and ask whether to keep it here too. The editor
    shows the same notes and the user can change them there.

    Args:
        asset_id: The file.
        notes: The notes, replacing what was there; empty clears them.
        note: Why, in the user's words, kept in the library's history.

    Returns:
        The file's `id`, `name` and `notes`.

    Raises:
        ValueError: If the file is not in the library.
    """
    asset = shared.get_asset(asset_id).model_copy(update={"notes": notes.strip()})
    runtime.repo.save_asset(asset)
    return {"id": asset.id, "name": os.path.basename(asset.path), "notes": asset.notes}

class CreateFolder(BaseModel):
    """Make a folder in the library, and put files in it."""

    action: Literal["create_folder"]
    name: str = Field(..., description="What to call it, in the user's words")
    library: Literal["footage", "music"] = Field(default="footage", description="Which library it is a folder of")
    parent_id: Optional[str] = Field(default=None, description="The folder it goes in; omit for the top")
    asset_ids: List[str] = Field(default_factory=list, description="Files to put in it straight away")

class RenameFolder(BaseModel):
    """Rename a folder of the library."""

    action: Literal["rename_folder"]
    folder_id: str
    name: str

class MoveFolder(BaseModel):
    """Put a folder inside another, or at the top."""

    action: Literal["move_folder"]
    folder_id: str
    parent_id: Optional[str] = Field(default=None, description="The folder to put it in; omit for the top")

class FileAssets(BaseModel):
    """Put files in a folder, or back at the top."""

    action: Literal["file_assets"]
    asset_ids: List[str]
    folder_id: Optional[str] = Field(default=None, description="The folder; omit for the top")

LibraryStep = Annotated[Union[CreateFolder, RenameFolder, MoveFolder, FileAssets], Field(discriminator="action")]

@mcp.tool()
def organize_library(steps: List[LibraryStep], note: str = "") -> dict:
    """Sort the library into folders: make, rename and move folders, and file the footage in them.

    Folders live only in clip-mcp — no file on disk moves — so this is yours
    to do when it helps, without asking first; then tell the user in a
    sentence what you made and filed. Nothing is ever deleted: footage the
    user does not want goes in a folder called 歸檔. The steps run in order
    and are kept as one version of the library. To move the files themselves
    on disk, see `move_files`.

    Args:
        steps: What to do, in order. `create_folder` can file footage in the
            new folder at once, since its ID is not known before.
        note: The user's words for why, kept in the library's history.

    Returns:
        `done`, each step in words, and `folders` and `music_folders`, every
        folder of each library now.

    Raises:
        ValueError: If a folder or file is unknown, or a name is taken.
    """
    names = {folder["id"]: folder["name"] for kind in ("assets", "music") for folder in runtime.repo.list_folders(kind)}
    done = []
    for step in steps:
        try:
            if isinstance(step, CreateFolder):
                made = runtime.repo.create_folder(uuid.uuid4().hex, _folder_kind(step.library), step.name, step.parent_id)
                names[made["id"]] = made["name"]
                if step.asset_ids:
                    _file_in(step.asset_ids, made["id"], step.library)
                done.append(f"新增資料夾「{made['name']}」" + (f"，放進 {len(step.asset_ids)} 個檔案" if step.asset_ids else ""))
            elif isinstance(step, RenameFolder):
                old = names.get(step.folder_id, step.folder_id)
                runtime.repo.rename_folder(step.folder_id, step.name)
                names[step.folder_id] = step.name
                done.append(f"資料夾「{old}」改名為「{step.name}」")
            elif isinstance(step, MoveFolder):
                runtime.repo.move_folder(step.folder_id, step.parent_id)
                done.append(f"把資料夾「{names.get(step.folder_id, step.folder_id)}」移到"
                            + (f"「{names.get(step.parent_id, step.parent_id)}」裡" if step.parent_id else "最上層"))
            else:
                _file_in(step.asset_ids, step.folder_id)
                done.append(f"把 {len(step.asset_ids)} 個檔案放進"
                            + (f"「{names.get(step.folder_id, step.folder_id)}」" if step.folder_id else "最上層"))
        except LookupError as exc:
            raise ValueError(str(exc).strip("'\"")) from exc
    shared.say_in_history("；".join(done))
    return {"done": done, "folders": runtime.repo.list_folders("assets"), "music_folders": runtime.repo.list_folders("music")}

def _file_in(asset_ids: List[str], folder_id: Optional[str], library: Optional[str] = None) -> None:
    """File footage or songs in a folder of their own library, or at its top.

    Args:
        asset_ids: The files.
        folder_id: The folder; its library's top when None.
        library: The library the folder belongs to, when it is known apart
            from the files.

    Raises:
        ValueError: If the files are of both libraries, or the folder is of the other one.
    """
    assets = [shared.get_asset(asset_id) for asset_id in asset_ids]
    kinds = {library_of(asset) for asset in assets} | ({library} if library else set())
    if len(kinds) > 1:
        raise ValueError("footage and music are filed in folders of their own: file them in separate steps, or "
                         "move a file to the other library first with set_asset_library")
    runtime.repo.file_items(_folder_kind(kinds.pop() if kinds else FOOTAGE), [asset.id for asset in assets], folder_id)

class FileMove(BaseModel):
    """One file to move on disk."""

    asset_id: str
    to_folder: str = Field(..., description="The folder on disk to move it to, as a full path; made if missing")
    name: Optional[str] = Field(default=None, description="A new file name; omit to keep its own. The extension "
                                                          "is kept when left off")

def _free_target(folder: str, name: str, taken: set) -> str:
    """Where a file can go without replacing anything: its name, or its name with a number added."""
    stem, extension = os.path.splitext(name)
    candidate, number = os.path.join(folder, name), 2
    while os.path.exists(candidate) or os.path.normcase(candidate) in taken:
        candidate = os.path.join(folder, f"{stem} ({number}){extension}")
        number += 1
    return candidate

def inside(path: str, folder: str) -> bool:
    """Whether a path is a folder or inside it, whole names only: `D:\\work2` is not inside `D:\\work`."""
    path, folder = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(folder))
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)

def _planned_moves(moves: List[FileMove]) -> List[dict]:
    """Where each file would go, and what that touches."""
    users: Dict[str, set] = {}
    using: Dict[str, set] = {}
    for project in runtime.repo.list_projects():
        for track in project.tracks:
            for clip in track.clips:
                users.setdefault(clip.asset_id, set()).add(project.name or project.id)
                using.setdefault(clip.asset_id, set()).add(project.id)
    working = runtime.repo.jobs_to_show(datetime.now(timezone.utc))[0]
    taken: set = set()
    planned = []
    for move in moves:
        asset = shared.get_asset(move.asset_id)
        if not os.path.isabs(move.to_folder):
            raise ValueError(f"{move.to_folder} is not a full folder path")
        folder = os.path.abspath(move.to_folder)
        if any(inside(folder, own) for own in (runtime.WORKSPACE_DIR, str(output_dir()))):
            raise ValueError("footage is not moved into clip-mcp's own folders")
        name = (move.name or os.path.basename(asset.path)).strip()
        if not name or os.path.basename(name) != name:
            raise ValueError(f"{name!r} is not a file name")
        if not os.path.splitext(name)[1]:
            name += os.path.splitext(asset.path)[1]
        wanted = os.path.join(folder, name)
        same = os.path.normcase(wanted) == os.path.normcase(asset.path)
        target = asset.path if same else _free_target(folder, name, taken)
        taken.add(os.path.normcase(target))
        planned.append({
            "asset_id": asset.id, "from": asset.path, "to": target, "missing": not os.path.exists(asset.path),
            "already_there": same, "numbered": not same and target != wanted,
            "new_folder": not os.path.isdir(folder),
            "megabytes": round(os.path.getsize(asset.path) / 1e6, 1) if os.path.exists(asset.path) else 0,
            "projects": sorted(users.get(asset.id, ())),
            # A render or an analysis has the file open: Windows will not let it move.
            "busy": any(job.asset_id == asset.id or job.project_id in using.get(asset.id, ()) for job in working),
        })
    return planned

def _plan_key(planned: List[dict]) -> str:
    """A short name for exactly this plan, so what is carried out is what the user was shown."""
    moves = [[plan["asset_id"], plan["from"], plan["to"]] for plan in planned]
    return hashlib.sha256(json.dumps(moves, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]

@mcp.tool()
def move_files(moves: List[FileMove], confirm_plan: Optional[str] = None, note: str = "") -> dict:
    """Move or rename footage on disk, keeping the library and every project pointing at it.

    Moving a user's own files is theirs to agree to. Call without
    `confirm_plan` first: it moves nothing, says where each file would go, and
    names that `plan`. Read it back — how many files, from where to where, any
    that get a number added — and call again with the same moves and
    `confirm_plan` set to that `plan` only once they agree; a plan that has
    changed since is refused. Nothing is overwritten: a name already taken
    gets ` (2)` added. Nothing is deleted: footage they do not want goes to an
    歸檔 folder like any other move. Projects keep working, since the library
    follows each file to where it went.

    Args:
        moves: Each file and the folder to move it to, with a new name if wanted.
        confirm_plan: The `plan` the user agreed to; carries it out.
        note: The user's words for why, kept in the library's history.

    Returns:
        Without `confirm_plan`: `plan`, and `planned`, each with `from`, `to`,
        `numbered` (a number was added), `new_folder`, `megabytes`, the
        `projects` using it, `busy` for a file a render or analysis has open,
        and `missing` for a file not found where the library has it. With it:
        `moved`, and `not_moved` with the reason.

    Raises:
        ValueError: If a file is not in the library, a folder is not a full
            path or is one of clip-mcp's own, or the plan is not the one shown.
    """
    planned = _planned_moves(moves)
    if confirm_plan is None:
        return {"plan": _plan_key(planned), "planned": planned}
    if confirm_plan.strip() != _plan_key(planned):
        raise ValueError("this is not the plan the user was shown: things have changed since, so show them the new "
                         "plan and ask again")
    shared.say_in_history(note or f"搬動了 {len(planned)} 個檔案")
    moved, not_moved = [], []
    for plan in planned:
        why = ("already there" if plan["already_there"] else "file not found" if plan["missing"]
               else "in use by a render or an analysis" if plan["busy"] else "")
        if why:
            not_moved.append({"from": plan["from"], "why": why})
            continue
        # Worked out again at the moment of moving: a file may have appeared there since the plan.
        target = _free_target(os.path.dirname(plan["to"]), os.path.basename(plan["to"]), set())
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.move(plan["from"], target)
        except OSError as exc:
            # Across drives a move is a copy first: one that stopped halfway is ours to take back.
            if os.path.exists(plan["from"]) and os.path.exists(target):
                try:
                    os.remove(target)
                except OSError:
                    pass
            not_moved.append({"from": plan["from"], "why": str(exc)})
            continue
        asset = shared.get_asset(plan["asset_id"])
        runtime.repo.save_asset(asset.model_copy(update={"path": target}))
        moved.append({"from": plan["from"], "to": target})
    return {"moved": moved, "not_moved": not_moved}

@mcp.tool()
def import_folder(folderpath: str, recursive: bool = False) -> dict:
    """Register every media file in a folder as an asset, in one call.

    Use this when the user points at a folder of footage instead of naming
    files. Files are returned in natural name order, so `clip2` comes before
    `clip10`, which is also the order to put them on the timeline unless the
    user says otherwise. A file that cannot be read is reported in `skipped`
    rather than failing the whole folder, and importing a folder again keeps
    the asset IDs it already handed out.

    Args:
        folderpath: Path to the folder, absolute or relative to the server's
            working directory.
        recursive: Whether to include media in sub-folders.

    Returns:
        A dictionary with `assets` (each in the format `import_asset` returns),
        `total_duration` in seconds, and `skipped`, a list of `path` and
        `reason` for files that could not be read.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    folder = os.path.abspath(folderpath)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"folder {folder} not found")

    paths: List[str] = []
    if recursive:
        for directory, _, names in os.walk(folder):
            paths.extend(os.path.join(directory, name) for name in names)
    else:
        paths = [os.path.join(folder, name) for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]
    paths = sorted(
        (path for path in paths if os.path.splitext(path)[1].lower() in MEDIA_EXTENSIONS),
        key=lambda path: (natural_key(os.path.dirname(path)), natural_key(os.path.basename(path))),
    )

    assets: List[Asset] = []
    skipped: List[dict] = []
    for path in paths:
        try:
            assets.append(register_asset(path))
        except (OSError, RuntimeError, ValueError) as error:
            skipped.append({"path": path, "reason": str(error)})

    # Photos have no length of their own: counted, an album would read as days of footage.
    total = sum((asset.duration for asset in assets if asset.duration is not None and not asset.still), Decimal(0))
    return shared.plain({"assets": [_asset_record(asset) for asset in assets], "total_duration": total, "skipped": skipped})
