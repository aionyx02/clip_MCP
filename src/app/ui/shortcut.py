"""The editor's shortcuts on the desktop and in the Start menu.

They start `clip-mcp-editor`, the windowless command installed next to
`clip-mcp`, so opening the editor never leaves a console window behind it.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from app.engine.ffmpeg import hidden_window_flags

NAME = "clip-mcp 剪輯"
ICON = Path(__file__).parent / "static" / "icon.ico"

def editor_command() -> Optional[Path]:
    """Find the windowless editor command beside the `clip-mcp` the clients run.

    Returns:
        Its path, or nothing when it is not installed — an install from before
        the editor, which `uv tool install --editable .` brings up to date.
    """
    from app.clients import server_command

    executable = "clip-mcp-editor.exe" if sys.platform == "win32" else "clip-mcp-editor"
    found = Path(server_command()[0]).with_name(executable)
    return found if found.is_file() else None

def places() -> List[Path]:
    """Where the shortcuts go: the desktop and the Start menu's programs."""
    desktop = Path(os.environ.get("USERPROFILE") or Path.home()) / "Desktop"
    start_menu = Path(os.environ.get("APPDATA") or Path.home()) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    return [desktop / f"{NAME}.lnk", start_menu / f"{NAME}.lnk"]

def create(dry_run: bool = False) -> List[str]:
    """Make, or remake, both shortcuts.

    Args:
        dry_run: Say what would be made without making it.

    Returns:
        One line per shortcut saying what happened.
    """
    if sys.platform != "win32":
        return ["shortcuts: only made on Windows"]
    target = editor_command()
    if target is None:
        return ["shortcuts: clip-mcp-editor is not installed; run `uv tool install --editable .`, then setup again"]
    lines = []
    for place in places():
        if dry_run:
            lines.append(f"would make {place}")
            continue
        place.parent.mkdir(parents=True, exist_ok=True)
        # The Windows shell's own object makes the link, so it is exactly what Explorer would make.
        script = (
            "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LINK); "
            "$s.TargetPath = $env:TARGET; $s.IconLocation = $env:ICON; "
            "$s.Description = 'clip-mcp 剪輯：檢查和微調 AI 剪好的影片'; $s.Save()"
        )
        made = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            env={**os.environ, "LINK": str(place), "TARGET": str(target), "ICON": str(ICON)},
            capture_output=True, creationflags=hidden_window_flags(),
        )
        lines.append(f"made {place}" if made.returncode == 0 else
                     f"could not make {place}: {made.stderr.decode('utf-8', errors='replace').strip()[-200:]}")
    return lines

def remove() -> List[str]:
    """Take both shortcuts away again.

    Returns:
        The ones removed.
    """
    removed = []
    for place in places():
        if place.exists():
            place.unlink()
            removed.append(str(place))
    return removed
