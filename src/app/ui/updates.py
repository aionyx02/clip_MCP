"""Finding a newer release, and replacing only what changed: the program alone, or everything by the installer.

Only a copy put on the machine by the installer is ever updated: a copy run
from a clone is updated with git, and offering it an installer would replace
the code being worked on. The check asks GitHub once per start and stays
quiet when it cannot reach it.

Each release carries, beside its installer, a zip of `program\\` alone and a
manifest (`clip-mcp-update.json`) with every asset's SHA-256 and a fingerprint
of the parts the program runs on: the runtime (Python and every package), the
launchers and FFmpeg. The install folder holds the same fingerprints for what
it has (`release.json`, written at build time). When all three match, only
`program\\` is replaced (`apply_update`); when any differs, or the install has
no record, the installer runs as before. Either download is checked against
the manifest's SHA-256 before anything is done with it.

Updating is the user's call unless they turned on automatic updates, which
are off until they do: then a newer release is downloaded when it is found
(at start and every few hours while the editor is open) and applied once no
render is running. Never while one is.
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from app.ui import apply_update

REPOSITORY = "aionyx02/clip_MCP"
LATEST = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
# The file the installer leaves in its folder, so a copy knows it may update itself.
MARKER = "installed-by-setup.txt"
MANIFEST = "clip-mcp-update.json"
# What the install folder's own build was: its version and the fingerprint of each part.
RELEASE = apply_update.RELEASE
SETTINGS = "update-settings.json"
# The parts a program-only update leaves in place; a release that changed any needs the installer.
PARTS = ("runtime", "launchers", "ffmpeg")
DOWNLOADS = Path(tempfile.gettempdir()) / "clip-mcp-update"
TIMEOUT = 5
# Provisional: how often an editor left open looks again when automatic updates are on, and how
# often it looks whether the renders it is waiting for have ended. Neither is from use yet.
CHECK_EVERY = 6 * 3600
IDLE_POLL = 60

_cache: dict = {}
_lock = threading.Lock()
# One update at a time, whether the user started it or the automatic one did.
_working = threading.Lock()
# How much of the download has arrived, for the editor to show while the user waits.
_progress = {"done": 0, "total": 0}
# What the automatic update is doing, for the editor to show.
_auto = {"state": "off", "version": None, "error": None, "staged": None}
_wake = threading.Event()
CHUNK = 256 * 1024

def current_version() -> str:
    """The version running now."""
    try:
        return version("clip-mcp")
    except PackageNotFoundError:
        return "0"

def _as_tuple(text: str) -> Tuple[int, ...]:
    """`v0.2.10` as (0, 2, 10), so versions compare as numbers."""
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])

def installed_by_setup() -> Optional[Path]:
    """The installer's folder, when this copy came from the installer; nothing otherwise."""
    for folder in list(Path(sys.prefix).resolve().parents)[:3]:
        if (folder / MARKER).exists():
            return folder
    return None

def _read(path: Path) -> dict:
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
        return found if isinstance(found, dict) else {}
    except (OSError, ValueError):
        return {}

def choose(installed: dict, manifest: dict) -> Tuple[str, List[str]]:
    """Which download an install needs for a release: `program` or `installer`, and why.

    Args:
        installed: The install folder's release.json; empty when it has none.
        manifest: The release's manifest.

    Returns:
        The plan, and the parts that differ (`unknown` when the install has no
        record to compare, `no_program_asset` when the release has no program zip).
    """
    if not installed:
        return "installer", ["unknown"]
    changed = [part for part in PARTS if not installed.get(part) or installed.get(part) != manifest.get(part)]
    if "program" not in manifest.get("assets", {}):
        changed.append("no_program_asset")
    return ("installer" if changed else "program"), changed

def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "clip-mcp-editor"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)

def latest(force: bool = False) -> dict:
    """Ask GitHub, once per start, whether there is a newer release, and which download it needs here.

    Returns:
        `current`, `available`, `managed`, and when there is one, its `version`,
        `notes`, `plan` (`program` or `installer`), `changed`, and the chosen
        download's `url`, `size` and `sha256`, with `installer_size` beside it.
        A newer release that cannot be checked is `available: False` with
        `refused` saying why. Unreachable or unpublished is simply not available.
    """
    with _lock:
        if _cache and not force:
            return dict(_cache)
    root = installed_by_setup()
    answer = {"current": current_version(), "available": False, "managed": root is not None}
    if root is not None:
        try:
            release = _get_json(LATEST)
            newer = _as_tuple(release.get("tag_name", "")) > _as_tuple(answer["current"])
            assets = {item.get("name"): item for item in release.get("assets", [])}
            if newer:
                tag = release["tag_name"].lstrip("v")
                answer.update(_offer(root, tag, assets, (release.get("body") or "").strip()[:4000]))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    with _lock:
        _cache.clear()
        _cache.update(answer)
    return dict(answer)

def _offer(root: Path, tag: str, assets: dict, notes: str) -> dict:
    """What updating to the release `tag` takes here, from its manifest."""
    if MANIFEST not in assets:
        return {"version": tag, "refused": "this release has no update manifest, so its downloads cannot be checked"}
    manifest = _get_json(assets[MANIFEST]["browser_download_url"])
    if manifest.get("version") != tag:
        return {"version": tag, "refused": f"the manifest is for {manifest.get('version')}, not {tag}"}
    plan, changed = choose(_read(root / RELEASE), manifest)
    chosen = manifest["assets"][plan]
    if chosen["name"] not in assets:
        return {"version": tag, "refused": f"the release does not have {chosen['name']}"}
    return {"available": True, "version": tag, "notes": notes, "plan": plan, "changed": changed,
            "url": assets[chosen["name"]]["browser_download_url"], "name": chosen["name"],
            "size": int(chosen["size"]), "sha256": chosen["sha256"],
            "installer_size": int(manifest["assets"]["installer"]["size"])}

def clear_old_downloads() -> None:
    """Remove what a past update downloaded; neither the installer nor the swap can remove themselves."""
    shutil.rmtree(DOWNLOADS, ignore_errors=True)

def _download(url: str, target: Path, size: int, sha256: str) -> Path:
    """Fetch `url` into `target`, keeping it only when its size and SHA-256 are the manifest's.

    Raises:
        RuntimeError: It did not arrive whole, or is not the file the manifest names.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "clip-mcp-editor"})
    digest = hashlib.sha256()
    _progress.update(done=0, total=int(size or 0))
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as handle:
        _progress["total"] = _progress["total"] or int(response.headers.get("Content-Length") or 0)
        while chunk := response.read(CHUNK):
            handle.write(chunk)
            digest.update(chunk)
            _progress["done"] += len(chunk)
    if target.stat().st_size != size:
        target.unlink(missing_ok=True)
        raise RuntimeError("the download did not arrive whole; try again")
    if digest.hexdigest() != sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"{target.name} is not the file the release lists (SHA-256 differs); nothing was changed")
    return target

def stage_program(archive: Path, root: Path, tag: str) -> Path:
    """Unpack a program zip into `<install>\\program.new`, refusing one that is not clip-mcp `tag`.

    Raises:
        RuntimeError: A path in the zip leaves the folder, or it holds no clip-mcp `tag`.
    """
    staged = root / "program.new"
    shutil.rmtree(staged, ignore_errors=True)
    with zipfile.ZipFile(archive) as unpacking:
        for name in unpacking.namelist():
            if name.startswith(("/", "\\")) or ":" in name or ".." in Path(name).parts:
                raise RuntimeError(f"the program zip holds a path outside its folder: {name}")
        unpacking.extractall(staged)
    if not (staged / "app" / "__init__.py").is_file() or not (staged / f"clip_mcp-{tag}.dist-info").is_dir():
        shutil.rmtree(staged, ignore_errors=True)
        raise RuntimeError(f"the program zip is not clip-mcp {tag}")
    return staged

def prepare(found: dict) -> dict:
    """Download what `found` needs and make it ready to apply: the program staged, or the installer saved.

    Returns:
        What `hand_over` needs: `plan`, `version`, and the installer's `path`.
    """
    root = installed_by_setup()
    if root is None:
        raise RuntimeError("only a copy installed by the installer updates itself")
    target = _download(found["url"], DOWNLOADS / found["name"], found["size"], found["sha256"])
    if found["plan"] == "program":
        stage_program(target, root, found["version"])
        target.unlink(missing_ok=True)
    return {"plan": found["plan"], "version": found["version"], "path": str(target)}

def _detached() -> int:
    return getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

def hand_over(staged: dict, on_started: Callable[[], None]) -> None:
    """Start what applies the update, then let the editor close so its files can be replaced."""
    root = installed_by_setup()
    if root is None:
        raise RuntimeError("only a copy installed by the installer updates itself")
    if staged["plan"] == "program":
        # Run from a copy: the original is in the folder being replaced.
        script = DOWNLOADS / "apply" / "apply_update.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(apply_update.__file__, script)
        subprocess.Popen([str(root / "runtime" / "python.exe"), "-I", "-S", str(script), str(root), staged["version"],
                          "--relaunch"], cwd=str(root), creationflags=_detached(), close_fds=True)
    else:
        # The installer says nothing when it is done, so the next start compares versions.
        (root / apply_update.RESULT).write_text(json.dumps({
            "ok": None, "kind": "installer", "version": staged["version"], "previous": current_version(),
            "announced": False, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
        # Silent, closing what holds clip-mcp's files, and opening the editor again at the end.
        subprocess.Popen([staged["path"], "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/RELAUNCH=1"],
                         creationflags=_detached(), close_fds=True)

    def close_soon() -> None:
        time.sleep(1.5)
        on_started()

    threading.Thread(target=close_soon, daemon=True).start()

def start_update(on_started: Callable[[], None]) -> str:
    """Download the newer release's program or installer, as this install needs, and apply it.

    Args:
        on_started: Called once the update is under way, to close the editor
            so its files can be replaced.

    Returns:
        The version being installed.

    Raises:
        RuntimeError: Nothing newer, another update under way, or the download
            is not what the release lists.
    """
    found = latest()
    if not found.get("available"):
        raise RuntimeError(found.get("refused") or "there is no newer version to install")
    if not _working.acquire(blocking=False):
        raise RuntimeError("an update is already under way")
    try:
        staged = _auto["staged"] if (_auto["staged"] or {}).get("version") == found["version"] else prepare(found)
        hand_over(staged, on_started)
    finally:
        _working.release()
    return found["version"]

def download_progress() -> dict:
    """How much of the download has arrived so far.

    Returns:
        `done` and `total` in bytes; `total` is 0 until the size is known.
    """
    return dict(_progress)

# --- automatic updates -----------------------------------------------------------------------

def auto_enabled() -> bool:
    """Whether the user turned automatic updates on; off for a copy not from the installer."""
    root = installed_by_setup()
    return root is not None and _read(root / SETTINGS).get("auto") is True

def set_auto(on: bool) -> None:
    """Turn automatic updates on or off for this installation.

    Raises:
        RuntimeError: This copy is not the installer's, and never updates itself.
    """
    root = installed_by_setup()
    if root is None:
        raise RuntimeError("only a copy installed by the installer updates itself")
    (root / SETTINGS).write_text(json.dumps({**_read(root / SETTINGS), "auto": bool(on)}), encoding="utf-8")
    _wake.set()

def tick(busy: Callable[[], bool], on_started: Callable[[], None]) -> float:
    """One look by the automatic updater: download a newer release, and apply it once nothing renders.

    Args:
        busy: Whether a render or other job is running.
        on_started: Closes the editor once the update is under way.

    Returns:
        Seconds until the next look.
    """
    if not auto_enabled():
        _auto.update(state="off", error=None)
        return CHECK_EVERY
    if not _working.acquire(blocking=False):
        return IDLE_POLL
    try:
        if _auto["staged"] is None:
            _auto.update(state="checking", error=None)
            found = latest(force=True)
            if not found.get("available"):
                _auto.update(state="current", error=found.get("refused"))
                return CHECK_EVERY
            _auto.update(state="downloading", version=found["version"])
            _auto["staged"] = prepare(found)
        if busy():
            _auto["state"] = "waiting"
            return IDLE_POLL
        _auto["state"] = "applying"
        hand_over(_auto["staged"], on_started)
        return CHECK_EVERY
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        _auto.update(state="failed", error=str(error), staged=None)
        return CHECK_EVERY
    finally:
        _working.release()

def watch(busy: Callable[[], bool], on_started: Callable[[], None]) -> None:
    """Run the automatic updater for as long as the editor runs; woken early when the setting changes."""
    while True:
        wait = tick(busy, on_started)
        _wake.wait(wait)
        _wake.clear()

def last_result() -> Optional[dict]:
    """The last update's outcome, with `fresh` the first time it is read, so the editor says it once."""
    root = installed_by_setup()
    if root is None:
        return None
    path = root / apply_update.RESULT
    found = _read(path)
    if not found:
        return None
    fresh = found.get("announced") is False
    if fresh:
        if found.get("ok") is None:
            found["ok"] = _as_tuple(current_version()) == _as_tuple(str(found.get("version")))
        try:
            path.write_text(json.dumps({**found, "announced": True}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return {**found, "fresh": fresh}

def overview(force: bool = False) -> dict:
    """Everything the editor shows about updates: the newer release, the setting, and the last outcome."""
    shown = {key: value for key, value in _auto.items() if key != "staged"}
    return {**latest(force), "auto": auto_enabled(), "auto_state": shown, "last": last_result()}
