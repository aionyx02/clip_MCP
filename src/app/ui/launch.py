"""Opening the editor: one server, one window, gone when the window is.

The editor is a page served on this machine and shown in Edge or Chrome as
an app window — no address bar, its own taskbar entry — so nothing has to be
installed for it beyond what clip-mcp already needs. Starting it a second
time opens a window onto the one already running instead of a second
server. The page checks in every few seconds; once it has stopped, the
server stops too, so closing the window leaves nothing running behind it.
Renders it started carry on: they run in their own process.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

from app.workspace import workspace_dir

FIRST_PORT = 8765
PORTS_TRIED = 20
# The page checks in every 10 seconds; this long without a word means its window is closed.
# Long enough that a laptop waking from sleep, or a slow machine, is not taken for a closed window.
IDLE_SECONDS = 45
# How long a window gets to load before the server decides nobody is coming.
FIRST_LOAD_SECONDS = 120

def _state_file() -> Path:
    """Where the running editor writes its port, for the next start to find it."""
    return Path(workspace_dir()) / "temp" / "editor.json"

def _running_port() -> Optional[int]:
    """The port of an editor already running for this workspace, if there is one."""
    try:
        port = int(json.loads(_state_file().read_text(encoding="utf-8"))["port"])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1) as answer:
            found = json.load(answer)
        ours = os.path.normcase(os.path.abspath(workspace_dir()))
        if found.get("app") == "clip-mcp-editor" and found.get("workspace") == ours:
            return port
    except (OSError, ValueError, KeyError):
        pass
    return None

def _free_port() -> int:
    """The first port from 8765 on that nothing else is listening on."""
    for port in range(FIRST_PORT, FIRST_PORT + PORTS_TRIED):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"ports {FIRST_PORT} to {FIRST_PORT + PORTS_TRIED - 1} are all in use")

def _browser() -> Optional[str]:
    """Edge, which every Windows 11 has, or Chrome; nothing when neither is found."""
    roots = [os.environ.get(name) for name in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
    candidates = [
        Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe" for root in roots if root
    ] + [
        Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe" for root in roots if root
    ]
    return next((str(path) for path in candidates if path.is_file()), None)

def open_window(url: str) -> None:
    """Show the editor in an app window, or a browser tab where no app window can be made."""
    browser = _browser()
    if browser:
        subprocess.Popen([browser, f"--app={url}", "--window-size=1440,900"], close_fds=True)
    else:
        webbrowser.open(url)

def run(open_browser: bool = True) -> None:
    """Open the editor, starting its server unless one is already running.

    Args:
        open_browser: Whether to open a window; off for serving it alone.
    """
    running = _running_port()
    if running:
        if open_browser:
            open_window(f"http://127.0.0.1:{running}/")
        return

    import uvicorn

    from app.ui import app as editor

    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    state = _state_file()
    state.parent.mkdir(parents=True, exist_ok=True)
    server = uvicorn.Server(uvicorn.Config(editor.create_app(), host="127.0.0.1", port=port, log_level="warning"))

    def watch() -> None:
        """Stop the server once the window has stopped checking in."""
        started = time.monotonic()
        while not server.should_exit:
            time.sleep(2)
            quiet = time.monotonic() - editor.Activity.last
            loaded = editor.Activity.last > started
            if (loaded and quiet > IDLE_SECONDS) or (not loaded and time.monotonic() - started > FIRST_LOAD_SECONDS):
                server.should_exit = True

    def show() -> None:
        """Open the window once the server answers."""
        while not server.started and not server.should_exit:
            time.sleep(0.1)
        state.write_text(json.dumps({"port": port, "pid": os.getpid()}), encoding="utf-8")
        if open_browser:
            open_window(url)

    if sys.stdout is not None:
        print(f"clip-mcp editor: {url}  (closes with its window; Ctrl+C to stop now)")
    threading.Thread(target=show, daemon=True).start()
    if open_browser:
        threading.Thread(target=watch, daemon=True).start()
    try:
        server.run()
    finally:
        try:
            if json.loads(state.read_text(encoding="utf-8")).get("pid") == os.getpid():
                state.unlink()
        except (OSError, ValueError):
            pass

def main() -> None:
    """Start the editor with no console, as the shortcut does."""
    if sys.stdout is None:
        # Started without a console there is nowhere for a message to go, and a server
        # that writes to nowhere can fail on it. One small log, replaced on every start.
        log = Path(workspace_dir()) / "temp" / "editor.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(log, "w", encoding="utf-8", buffering=1)
    run()
