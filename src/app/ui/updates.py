"""Finding a newer release, and handing over to its installer when the user says so.

Only a copy put on the machine by the installer is ever updated: a copy run
from a clone is updated with git, and offering it an installer would replace
the code being worked on. The check asks GitHub once per start and stays
quiet when it cannot reach it. Nothing is downloaded until the user chooses
to update, and never while a render is running.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Tuple

REPOSITORY = "aionyx02/clip_MCP"
LATEST = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
# The file the installer leaves in its folder, so a copy knows it may update itself.
MARKER = "installed-by-setup.txt"
INSTALLER = re.compile(r"^clip-mcp-setup-[\d.]+\.exe$")
DOWNLOADS = Path(tempfile.gettempdir()) / "clip-mcp-update"
TIMEOUT = 5

_cache: dict = {}
_lock = threading.Lock()

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

def latest(force: bool = False) -> dict:
    """Ask GitHub, once per start, whether there is a newer release with an installer.

    Returns:
        `current`, `available`, and when there is one, its `version`, `notes`,
        and the installer's `url` and `size`. Unreachable or unpublished is
        simply not available.
    """
    with _lock:
        if _cache and not force:
            return dict(_cache)
    answer = {"current": current_version(), "available": False, "managed": installed_by_setup() is not None}
    if answer["managed"]:
        try:
            request = urllib.request.Request(LATEST, headers={"Accept": "application/vnd.github+json",
                                                              "User-Agent": "clip-mcp-editor"})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                release = json.load(response)
            asset = next((item for item in release.get("assets", []) if INSTALLER.match(item.get("name", ""))), None)
            if asset and _as_tuple(release.get("tag_name", "")) > _as_tuple(answer["current"]):
                answer.update({
                    "available": True, "version": release["tag_name"].lstrip("v"),
                    "notes": (release.get("body") or "").strip()[:4000],
                    "url": asset["browser_download_url"], "size": asset.get("size", 0),
                })
        except (OSError, ValueError, KeyError):
            pass
    with _lock:
        _cache.clear()
        _cache.update(answer)
    return dict(answer)

def clear_old_downloads() -> None:
    """Remove the installer a past update downloaded; it cannot remove itself."""
    shutil.rmtree(DOWNLOADS, ignore_errors=True)

def start_update(on_started) -> str:
    """Download the newer installer and start it, then let the editor close for it.

    Args:
        on_started: Called once the installer is running, to close the editor
            so the installer can replace its files.

    Returns:
        The version being installed.

    Raises:
        RuntimeError: Nothing newer, or the download did not arrive whole.
    """
    found = latest()
    if not found.get("available"):
        raise RuntimeError("there is no newer version to install")
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    target = DOWNLOADS / f"clip-mcp-setup-{found['version']}.exe"
    request = urllib.request.Request(found["url"], headers={"User-Agent": "clip-mcp-editor"})
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as handle:
        shutil.copyfileobj(response, handle)
    if found.get("size") and target.stat().st_size != found["size"]:
        target.unlink(missing_ok=True)
        raise RuntimeError("the download did not arrive whole; try again")
    # Silent, closing what holds clip-mcp's files, and opening the editor again at the end.
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([str(target), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/RELAUNCH=1"],
                     creationflags=flags, close_fds=True)

    def close_soon() -> None:
        time.sleep(1.5)
        on_started()

    threading.Thread(target=close_soon, daemon=True).start()
    return found["version"]
