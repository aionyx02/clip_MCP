"""Where the workspace is: the database, the models, and everything rendered.

The server is started by whichever AI client the person uses, from whatever
folder that client happens to be in. So the workspace cannot be worked out
from where the process was started — that is how one person ended up with a
fresh, empty workspace in every folder they opened a client in. It is looked
up the same way every time, wherever the server was launched from:

1. `CLIP_MCP_WORKSPACE`, for pointing one run somewhere else on purpose.
2. The pointer file: a line in a small per-user config file saying where the
   workspace is, written by `clip-mcp setup`. Every client on the machine
   finds the same workspace through it, and moving the workspace is one line.
3. Neither set: `workspace/` inside the project when this is running from a
   copy of the source, so it follows wherever the project was cloned; and the
   per-user data folder when it is an installed package, since a folder
   inside the Python installation is wiped by the next upgrade.
"""

import json
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ENVIRONMENT_VARIABLE = "CLIP_MCP_WORKSPACE"
APP_NAME = "clip-mcp"

@dataclass(frozen=True)
class Workspace:
    """A workspace, and why it is the one in use.

    Attributes:
        path: The absolute path.
        source: What chose it: `environment`, `pointer`, `source checkout`
            or `user data`.
    """

    path: Path
    source: str

def config_dir() -> Path:
    """Find the per-user folder the pointer file lives in.

    Returns:
        `%APPDATA%\\clip-mcp` on Windows, `~/Library/Application Support/clip-mcp`
        on macOS, and `$XDG_CONFIG_HOME/clip-mcp` (by default `~/.config/clip-mcp`)
        elsewhere.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_NAME

def pointer_file() -> Path:
    """Find the pointer file.

    Returns:
        Its path, whether or not it exists yet.
    """
    return config_dir() / "config.toml"

def user_data_dir() -> Path:
    """Find the per-user folder an installed package keeps its workspace in.

    Returns:
        `%LOCALAPPDATA%\\clip-mcp\\workspace` on Windows,
        `~/Library/Application Support/clip-mcp/workspace` on macOS, and
        `$XDG_DATA_HOME/clip-mcp/workspace` (by default `~/.local/share/...`)
        elsewhere.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / APP_NAME / "workspace"

def source_checkout() -> Optional[Path]:
    """Find the project folder this is running from, when it is a copy of the source.

    Returns:
        The folder holding `pyproject.toml` and `src/app`, or nothing for an
        installed package. An editable install counts as a copy of the source:
        it runs from the clone.
    """
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "app").is_dir():
        return root
    return None

def read_pointer(path: Optional[Path] = None) -> Optional[Path]:
    """Read where the pointer file says the workspace is.

    Args:
        path: The pointer file; the per-user one when not given.

    Returns:
        The workspace it names, or nothing when there is no file, it cannot be
        read, or it names none. A pointer that cannot be read is ignored
        rather than fatal: the server still has to start.
    """
    path = path or pointer_file()
    try:
        with open(path, "rb") as handle:
            named = tomllib.load(handle).get("workspace")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return Path(named).expanduser().resolve() if isinstance(named, str) and named.strip() else None

def write_pointer(workspace: Path, path: Optional[Path] = None) -> Path:
    """Write the pointer file, naming a workspace.

    Args:
        workspace: The workspace to point at.
        path: The pointer file; the per-user one when not given.

    Returns:
        The file written.
    """
    path = path or pointer_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # A TOML basic string escapes the way JSON does, which covers a Windows backslash.
    path.write_text(
        "# Where clip-mcp keeps its projects, analyses, models and renders.\n"
        "# Written by `clip-mcp setup`; every AI client on this machine reads it.\n"
        f"workspace = {json.dumps(str(workspace.resolve()))}\n",
        encoding="utf-8",
    )
    return path

def resolve() -> Workspace:
    """Decide which workspace to use.

    Returns:
        The workspace and what chose it, by the order in this module's
        description.
    """
    override = os.environ.get(ENVIRONMENT_VARIABLE)
    if override:
        return Workspace(Path(override).expanduser().resolve(), "environment")
    pointed = read_pointer()
    if pointed is not None:
        return Workspace(pointed, "pointer")
    checkout = source_checkout()
    if checkout is not None:
        return Workspace(checkout / "workspace", "source checkout")
    return Workspace(user_data_dir(), "user data")

def workspace_dir() -> str:
    """Find the workspace in use, as a string for `os.path`.

    Returns:
        The absolute path.
    """
    return str(resolve().path)
