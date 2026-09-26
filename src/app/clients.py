"""Registering the server with the AI clients installed on this machine.

Each client keeps its list of MCP servers in its own file, in its own format,
and none of them starts the server from the project folder. So what each is
given is the one thing that works from anywhere: the full path of the
`clip-mcp` command, found on this machine at the time of registering. Nothing
about where the workspace is goes into a client — the server looks that up
itself (`app.workspace`) — so moving the workspace never means registering
again.

Every file is backed up before it is changed, and a dry run says what would
change without changing it.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

SERVER_NAME = "clip-mcp"
# Names an earlier hand-written entry may have used for this server, replaced rather than
# left beside the new one so a client does not start two copies.
OLD_NAMES = ("clip_mcp",)

@dataclass(frozen=True)
class Registration:
    """What registering with one client did, or would do.

    Attributes:
        client: The client's name.
        status: `registered`, `unchanged`, `would register`, `skipped` or `failed`.
        detail: Where, or why not.
    """

    client: str
    status: str
    detail: str

def server_command() -> List[str]:
    """Find the command that starts this server from anywhere.

    The command installed by `uv tool install` comes first: it has an
    environment of its own, so a client starting it never collides with a
    development environment that another client — or a test run — is using
    at the same time. A copy inside a project's `.venv` is the last resort,
    because `uv sync` rewrites it and fails while any client has it running.

    Returns:
        The command as a list, first element an absolute path.
    """
    executable = "clip-mcp.exe" if sys.platform == "win32" else "clip-mcp"
    tool_bin = os.environ.get("UV_TOOL_BIN_DIR") or str(Path.home() / ".local" / "bin")
    installed = Path(tool_bin) / executable
    if installed.is_file():
        return [str(installed)]
    found = shutil.which(SERVER_NAME)
    if found:
        return [str(Path(found).resolve())]
    return [sys.executable, "-m", "app.server"]

def in_development_environment(command: List[str]) -> bool:
    """Tell whether a command runs from a project's own `.venv`.

    Args:
        command: The server command.

    Returns:
        True when it does, which is the setup that breaks while two clients
        run it.
    """
    return any(part == ".venv" for part in Path(command[0]).parts)

def _backup(path: Path) -> None:
    """Copy a file aside before changing it.

    Args:
        path: The file.
    """
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, path.with_name(f"{path.name}.{stamp}.bak"))

def _strip_jsonc(text: str) -> str:
    """Turn JSON with comments and trailing commas into plain JSON.

    Args:
        text: The file's text.

    Returns:
        The same data as strict JSON. Strings are left alone, so a URL's `//`
        is not taken for a comment.
    """
    out, index, length = [], 0, len(text)
    while index < length:
        character = text[index]
        if character == '"':
            end = index + 1
            while end < length and text[end] != '"':
                end += 2 if text[end] == "\\" else 1
            out.append(text[index:end + 1])
            index = end + 1
        elif text.startswith("//", index):
            index = text.find("\n", index)
            index = length if index < 0 else index
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = length if end < 0 else end + 2
        else:
            out.append(character)
            index += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))

def _update_json(
    path: Path, change: Callable[[dict], bool], dry_run: bool, comments: bool = False,
) -> str:
    """Change a JSON settings file in place.

    Args:
        path: The file; created when missing.
        change: Edits the parsed settings and says whether anything changed.
        dry_run: Leave the file alone.
        comments: The file may hold comments, which are lost on writing back.

    Returns:
        `registered`, `unchanged` or `would register`.
    """
    text = path.read_text(encoding="utf-8") if path.exists() else "{}"
    settings = json.loads(_strip_jsonc(text) if comments else text or "{}")
    if not change(settings):
        return "unchanged"
    if dry_run:
        return "would register"
    _backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return "registered"

def _replace_entry(servers: dict, entry: dict) -> bool:
    """Put this server's entry into a client's table of servers.

    Args:
        servers: The table, edited in place.
        entry: What the client needs to start the server.

    Returns:
        Whether anything changed.
    """
    stale = [name for name in OLD_NAMES if name in servers]
    if servers.get(SERVER_NAME) == entry and not stale:
        return False
    for name in stale:
        del servers[name]
    servers[SERVER_NAME] = entry
    return True

def register_opencode(command: List[str], dry_run: bool) -> Registration:
    """Register with opencode, in its global config.

    Args:
        command: The server command.
        dry_run: Say what would change without changing it.

    Returns:
        What happened.
    """
    folder = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "opencode"
    if not folder.is_dir() and shutil.which("opencode") is None:
        return Registration("opencode", "skipped", "not installed")
    path = next((folder / name for name in ("opencode.jsonc", "opencode.json") if (folder / name).exists()),
                folder / "opencode.json")
    entry = {"type": "local", "command": command, "enabled": True}

    def change(settings: dict) -> bool:
        settings.setdefault("$schema", "https://opencode.ai/config.json")
        return _replace_entry(settings.setdefault("mcp", {}), entry)

    try:
        status = _update_json(path, change, dry_run, comments=path.suffix == ".jsonc")
    except (OSError, ValueError) as error:
        return Registration("opencode", "failed", f"{path}: {error}")
    note = " (comments in it are not kept; a backup is)" if status == "registered" and path.suffix == ".jsonc" else ""
    return Registration("opencode", status, f"{path}{note}")

def register_claude_desktop(command: List[str], dry_run: bool) -> Registration:
    """Register with the Claude desktop app.

    Args:
        command: The server command.
        dry_run: Say what would change without changing it.

    Returns:
        What happened.
    """
    if sys.platform == "win32":
        folder = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Claude"
    elif sys.platform == "darwin":
        folder = Path.home() / "Library" / "Application Support" / "Claude"
    else:
        folder = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "Claude"
    if not folder.is_dir():
        return Registration("Claude Desktop", "skipped", "not installed")
    path = folder / "claude_desktop_config.json"
    entry = {"command": command[0], "args": command[1:]}
    try:
        status = _update_json(path, lambda settings: _replace_entry(settings.setdefault("mcpServers", {}), entry),
                              dry_run)
    except (OSError, ValueError) as error:
        return Registration("Claude Desktop", "failed", f"{path}: {error}")
    return Registration("Claude Desktop", status, str(path))

def _toml_section(command: List[str]) -> str:
    """Write this server's section of a Codex config.

    Args:
        command: The server command.

    Returns:
        The section's text.
    """
    return (
        f"[mcp_servers.{SERVER_NAME}]\n"
        f"command = {json.dumps(command[0])}\n"
        f"args = {json.dumps(command[1:])}\n"
    )

def register_codex(command: List[str], dry_run: bool) -> Registration:
    """Register with Codex CLI, in `~/.codex/config.toml`.

    Only this server's own section is touched; the rest of the file is kept
    as it was written, comments and all.

    Args:
        command: The server command.
        dry_run: Say what would change without changing it.

    Returns:
        What happened.
    """
    folder = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    if not folder.is_dir() and shutil.which("codex") is None:
        return Registration("Codex", "skipped", "not installed")
    path = folder / "config.toml"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = _toml_section(command)
    names = "|".join(re.escape(name) for name in (SERVER_NAME, *OLD_NAMES))
    # A section runs from its header to the next line that starts a header, or the end of
    # the file, and takes its own sub-tables (`.env`) with it. By line, not by bracket: the
    # `args` array holds brackets of its own.
    pattern = re.compile(
        rf'^\[mcp_servers\.(?:{names}|"(?:{names})")(?:\.[^\]\n]+)?\][^\n]*\n?(?:(?!\[).*\n?)*',
        re.MULTILINE,
    )
    found = pattern.findall(text)
    if found == [section] or [part.strip() for part in found] == [section.strip()]:
        return Registration("Codex", "unchanged", str(path))
    kept = pattern.sub("", text).rstrip()
    updated = (kept + "\n\n" if kept else "") + section
    if dry_run:
        return Registration("Codex", "would register", str(path))
    try:
        _backup(path)
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(updated, encoding="utf-8")
    except OSError as error:
        return Registration("Codex", "failed", f"{path}: {error}")
    return Registration("Codex", "registered", str(path))

def register_claude_code(command: List[str], dry_run: bool) -> Registration:
    """Register with Claude Code, for this user in every folder.

    Done through its own command rather than by editing its settings, which
    hold far more than servers and change shape between versions.

    Args:
        command: The server command.
        dry_run: Say what would change without changing it.

    Returns:
        What happened.
    """
    claude = shutil.which("claude")
    if claude is None:
        return Registration("Claude Code", "skipped", "not installed")
    listed = subprocess.run([claude, "mcp", "get", SERVER_NAME], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60)
    if listed.returncode == 0 and "Scope: User" in listed.stdout and command[0] in listed.stdout:
        return Registration("Claude Code", "unchanged", "user scope")
    if dry_run:
        return Registration("Claude Code", "would register", "user scope")
    subprocess.run([claude, "mcp", "remove", "--scope", "user", SERVER_NAME], capture_output=True, timeout=60)
    added = subprocess.run([claude, "mcp", "add", "--scope", "user", SERVER_NAME, "--", *command],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if added.returncode != 0:
        return Registration("Claude Code", "failed", (added.stderr or added.stdout).strip()[-300:])
    return Registration("Claude Code", "registered", "user scope")

CLIENTS: Dict[str, Callable[[List[str], bool], Registration]] = {
    "claude-code": register_claude_code,
    "claude-desktop": register_claude_desktop,
    "codex": register_codex,
    "opencode": register_opencode,
}

def register(command: List[str], only: Optional[List[str]] = None, dry_run: bool = False) -> List[Registration]:
    """Register with every client installed here, or the ones named.

    Args:
        command: The server command.
        only: Client keys from `CLIENTS`; all of them when not given.
        dry_run: Say what would change without changing anything.

    Returns:
        One result per client.
    """
    results = []
    for key in only or list(CLIENTS):
        try:
            results.append(CLIENTS[key](command, dry_run))
        except (OSError, subprocess.SubprocessError) as error:
            results.append(Registration(key, "failed", str(error)))
    return results
