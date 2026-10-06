"""Where everything the server writes goes, and how it is kept from piling up.

Two kinds of file come out of editing. What was made for the user — a
finished video, a cover, a timeline for another editor — is theirs, so it is
saved where they keep their videos and never touched again. Everything else
— previews, sound checks, a render's working files, thumbnails — can be made
again, so it lives in the workspace's `temp` and `cache` folders, only the
latest preview of each project is kept, and any of it can be cleared at any
time without losing work.

`outputs` is where every render used to go, finished or not. Nothing new is
written there; `legacy_plan` says what is in it and `legacy_apply` sorts it
out once the user has agreed to the list.
"""

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set

TEMP = "temp"
CACHE = "cache"
LEGACY = "outputs"

@dataclass(frozen=True)
class Usage:
    """How much room one kind of file takes, and whether it may be cleared.

    Attributes:
        key: What `clean` is told to clear it by.
        label: What the user calls it.
        bytes: Its size on disk.
        path: The folder it is in, to open.
        clearable: Whether clearing it loses nothing that cannot be made again.
    """

    key: str
    label: str
    bytes: int
    path: str
    clearable: bool

# What each model folder is for, as the user would say it.
MODEL_LABELS = {
    "whisper": "AI 模型：語音辨識",
    "diarization": "AI 模型：分辨說話的人",
    "faces": "AI 模型：人臉偵測",
    "search": "AI 模型：依意思搜尋片段",
    "tts": "AI 模型：配音（未啟用）",
}

def preview_dir(workspace: str, project_id: str) -> str:
    """The folder a project's previews are rendered into, one subfolder per render."""
    return os.path.join(workspace, TEMP, "previews", project_id)

def sound_dir(workspace: str, project_id: str) -> str:
    """The folder a project's sound checks are written into, one subfolder per check."""
    return os.path.join(workspace, TEMP, "sounds", project_id)

def render_dir(workspace: str) -> str:
    """The folder a full render does its work in before the file is saved for the user."""
    return os.path.join(workspace, TEMP, "renders")

def ui_cache_dir(workspace: str) -> str:
    """The editor's thumbnails and waveforms."""
    return os.path.join(workspace, CACHE, "editor")

def unique_path(folder: str, name: str) -> str:
    """A path in `folder` for `name` that does not exist yet: `name (2).mp4` when `name.mp4` does.

    Args:
        folder: Where the file will go.
        name: The file name wanted.

    Returns:
        The path; the folder is not created.
    """
    stem, suffix = os.path.splitext(name)
    candidate, number = os.path.join(folder, name), 2
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{stem} ({number}){suffix}")
        number += 1
    return candidate

def deliver(source: str, wanted: str) -> str:
    """Move a finished file to where the user keeps it, never over another one.

    Args:
        source: The finished file, in the workspace.
        wanted: Where it should go. Taken as it is unless something got there
            first, in which case the next free `(2)` name is used.

    Returns:
        Where it went.
    """
    folder = os.path.dirname(wanted)
    os.makedirs(folder, exist_ok=True)
    target = unique_path(folder, os.path.basename(wanted))
    # Across drives this is a copy and a delete, so it is copied to a hidden name and
    # renamed whole: a half-copied video must never sit there under its real name.
    partial = target + ".part"
    shutil.move(source, partial)
    os.replace(partial, target)
    return target

def prune(folder: str, keep: Iterable[str]) -> None:
    """Delete every subfolder of `folder` except the ones named.

    Anything that cannot be deleted — a preview still open in a player — is
    left for the next time.

    Args:
        folder: A project's previews or sound checks.
        keep: Subfolder names to leave: the newest, and any still being written.
    """
    kept = set(keep)
    if not os.path.isdir(folder):
        return
    for entry in os.scandir(folder):
        if entry.is_dir() and entry.name not in kept:
            shutil.rmtree(entry.path, ignore_errors=True)

def sweep_renders(workspace: str, delivered: Callable[[str], bool]) -> None:
    """Delete what finished renders left in their working folders.

    A render cannot do this itself: on Windows its own log is still open until
    its process ends. So the next render, or the next look at the workspace,
    does it.

    Args:
        workspace: The workspace folder.
        delivered: Says, by job ID, whether that render finished and its file
            was saved; a failed render's folder is kept for its logs.
    """
    folder = render_dir(workspace)
    if not os.path.isdir(folder):
        return
    for entry in os.scandir(folder):
        if entry.is_dir() and delivered(entry.name):
            shutil.rmtree(entry.path, ignore_errors=True)

def _size(path: str) -> int:
    """Bytes under a path, file or folder; zero when it is not there."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            # Model caches link one stored file under several names; count it where it is stored.
            file = os.path.join(root, name)
            try:
                if not os.path.islink(file):
                    total += os.path.getsize(file)
            except OSError:
                pass
    return total

def usage(workspace: str, outputs: str) -> List[Usage]:
    """Say what the workspace holds, kind by kind, and what the user's saved videos take.

    Args:
        workspace: The workspace folder.
        outputs: Where finished videos are saved.

    Returns:
        One entry per kind that has anything in it, models first.
    """
    found: List[Usage] = []
    models = os.path.join(workspace, "models")
    if os.path.isdir(models):
        for entry in sorted(os.scandir(models), key=lambda entry: entry.name):
            if entry.is_dir():
                label = MODEL_LABELS.get(entry.name, f"AI 模型：{entry.name}")
                found.append(Usage(f"model:{entry.name}", label, _size(entry.path), entry.path, False))
    database = sum(_size(os.path.join(workspace, name)) for name in os.listdir(workspace)
                   if name.startswith("clip_mcp.db")) if os.path.isdir(workspace) else 0
    found.append(Usage("data", "專案與素材分析", database, workspace, False))
    found.append(Usage("previews", "預覽暫存", _size(os.path.join(workspace, TEMP, "previews"))
                       + _size(os.path.join(workspace, TEMP, "sounds"))
                       + _size(os.path.join(workspace, CACHE, "pictures")),
                       os.path.join(workspace, TEMP), True))
    found.append(Usage("thumbnails", "縮圖快取", _size(ui_cache_dir(workspace)), ui_cache_dir(workspace), True))
    found.append(Usage("work", "工作暫存", _size(os.path.join(workspace, "jobs")) + _size(render_dir(workspace)),
                       os.path.join(workspace, "jobs"), True))
    legacy = os.path.join(workspace, LEGACY)
    if os.path.isdir(legacy) and any(os.scandir(legacy)):
        found.append(Usage("legacy", "舊版的輸出（待整理）", _size(legacy), legacy, False))
    found.append(Usage("outputs", "成品（影片資料夾）", _size(outputs), outputs, False))
    return [entry for entry in found if entry.bytes or entry.key in ("data", "outputs")]

def clean(workspace: str, keys: Iterable[str], busy: Set[str]) -> int:
    """Clear the kinds named, leaving anything a running job is using.

    Args:
        workspace: The workspace folder.
        keys: `previews`, `thumbnails` and `work`; anything else is refused by
            the caller and ignored here.
        busy: Job IDs still queued or running; their folders are left.

    Returns:
        Bytes freed.
    """
    folders: List[str] = []
    wanted = set(keys)
    if "previews" in wanted:
        for kind in ("previews", "sounds"):
            root = os.path.join(workspace, TEMP, kind)
            if os.path.isdir(root):
                folders += [entry.path for entry in os.scandir(root) if entry.is_dir()]
        folders.append(os.path.join(workspace, CACHE, "pictures"))
    if "thumbnails" in wanted:
        folders.append(ui_cache_dir(workspace))
    if "work" in wanted:
        folders += [os.path.join(workspace, "jobs"), render_dir(workspace)]
    freed = 0
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for entry in os.scandir(folder):
            if entry.name in busy:
                continue
            size = _size(entry.path)
            try:
                shutil.rmtree(entry.path) if entry.is_dir() else os.remove(entry.path)
                freed += size
            except OSError:
                pass
        # A project's previews folder whose last preview was in use stays; an empty one goes.
        if folder.startswith(os.path.join(workspace, TEMP)) and not any(os.scandir(folder)):
            os.rmdir(folder)
    return freed

# The shapes a cut can be delivered in, as the file name says them.
FRAME_WORDS = {"landscape": "橫式", "portrait": "直式", "square": "方形"}
# What follows the project's name for each kind of thing delivered.
KIND_WORDS = {"output": "", "cover": " 封面", "timeline": ""}

# Said after the shape when a video has its captions burned in. The same cut with and
# without them used to come out as `EP1.mp4` and `EP1 (2).mp4`, and nothing said which was which.
CAPTIONED_WORD = "字幕"

def delivery_name(stem: str, kind: str, extension: str, captioned: bool = False) -> str:
    """Name a delivered file the way a person would: `EP1 台北.mp4`, `EP1 台北（直式、字幕）.mp4`.

    Args:
        stem: The project's name, already safe for a file system.
        kind: `output`, `cover` or `timeline`, with `-portrait`, `-square` or
            `-landscape` for a cut delivered in another shape.
        extension: With its dot.
        captioned: Whether the captions are burned into it.

    Returns:
        The file name.
    """
    base, _, frame = kind.partition("-")
    differs = ([FRAME_WORDS[frame]] if frame else []) + ([CAPTIONED_WORD] if captioned else [])
    return f"{stem}{KIND_WORDS[base]}{f'（{'、'.join(differs)}）' if differs else ''}{extension}"

# What a legacy render's file was called: the project's name, then what kind of render.
LEGACY_NAME = re.compile(
    r"^(?P<stem>.+)_(?P<kind>output|cover|timeline)(?:-(?P<frame>landscape|portrait|square))?(?P<extension>\.\w+)$"
)

@dataclass(frozen=True)
class LegacyItem:
    """One file in the old outputs folder, and what will happen to it.

    Attributes:
        path: The file now.
        bytes: Its size.
        keep_as: Where it is moved to in the videos folder, or nothing when it
            is not kept.
        duplicate: An older version of a video, cover or export that is kept
            in a newer one; these, and only these, go to the recycle bin.
    """

    path: str
    bytes: int
    keep_as: Optional[str]
    duplicate: bool = False

def legacy_plan(workspace: str, outputs: str) -> List[LegacyItem]:
    """Decide what to do with the old outputs folder, without doing it.

    The newest finished video, cover and export of each project are kept and
    moved to the videos folder, and the older versions of them go to the
    recycle bin. Everything else — sound checks, a render's logs — is left
    where it is: the user asked for the duplicates to go, not for the folder
    to be emptied, and `clean_storage` is how the rest is cleared.

    Args:
        workspace: The workspace folder.
        outputs: Where finished videos are saved now.

    Returns:
        Every file under the old folder, kept ones first, newest first.
    """
    legacy = os.path.join(workspace, LEGACY)
    if not os.path.isdir(legacy):
        return []
    files = []
    for root, _, names in os.walk(legacy):
        for name in names:
            path = os.path.join(root, name)
            try:
                files.append((os.path.getmtime(path), path, os.path.getsize(path)))
            except OSError:
                pass
    files.sort(reverse=True)
    newest: Set[tuple] = set()
    planned: List[LegacyItem] = []
    for _, path, size in files:
        name = os.path.basename(path)
        matched = LEGACY_NAME.match(name)
        keep_as = None
        if matched:
            kind = matched["kind"] + (f"-{matched['frame']}" if matched["frame"] else "")
            group = (matched["stem"], kind, matched["extension"].lower())
            if group not in newest:
                newest.add(group)
                keep_as = os.path.join(outputs, delivery_name(matched["stem"], kind, matched["extension"]))
        planned.append(LegacyItem(path, size, keep_as, duplicate=bool(matched) and keep_as is None))
    planned.sort(key=lambda item: item.keep_as is None)
    return planned

def legacy_apply(workspace: str, plan: List[LegacyItem]) -> dict:
    """Carry out a plan `legacy_plan` made and the user agreed to.

    Args:
        workspace: The workspace folder.
        plan: The plan, as shown to the user.

    Returns:
        `kept`: where each kept file went; `recycled`: how many duplicates
        went to the recycle bin and how many bytes; `left`: files that could
        not be moved, such as one open in a player.
    """
    kept, left = [], []
    for item in plan:
        if item.keep_as and os.path.exists(item.path):
            try:
                kept.append(deliver(item.path, item.keep_as))
            except OSError:
                left.append(item.path)
    duplicates = [item for item in plan if item.duplicate and os.path.exists(item.path)]
    recycled = to_recycle_bin([item.path for item in duplicates])
    if not recycled:
        left += [item.path for item in duplicates]
    return {
        "kept": kept,
        "recycled": {"files": len(duplicates) if recycled else 0,
                     "bytes": sum(item.bytes for item in duplicates) if recycled else 0},
        "left": left,
    }

def to_recycle_bin(paths: List[str]) -> bool:
    """Send files or folders to the recycle bin, so a mistake can be undone.

    Args:
        paths: What to send.

    Returns:
        Whether they went. Off Windows nothing is sent, and the caller leaves
        them where they are rather than delete them for good.
    """
    if sys.platform != "win32" or not paths:
        return False
    import ctypes
    from ctypes import wintypes

    class FileOperation(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    delete, allow_undo, no_confirmation, silent, no_error_ui = 3, 0x40, 0x10, 0x4, 0x400
    # The list is ended by an empty string: each path, then one more terminator.
    listed = "".join(str(Path(path).resolve()) + "\0" for path in paths) + "\0"
    operation = FileOperation(None, delete, listed, None, allow_undo | no_confirmation | silent | no_error_ui)
    return ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation)) == 0 and not operation.fAnyOperationsAborted
