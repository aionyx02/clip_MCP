"""Registering the server with the AI clients installed on this machine.

Each client keeps its list of MCP servers in its own file, in its own format,
and none of them starts the server from the project folder. So what each is
given is the one thing that works from anywhere: the full path of the
`clip-mcp` command, found on this machine at the time of registering. Nothing
about where the workspace is goes into a client — the server looks that up
itself (`app.workspace`) — so moving the workspace never means registering
again.

Nothing here runs on its own: every change to a client's settings is one the
user asked for, client by client — from `clip-mcp setup`, which asks, or from
the editor's 連接 AI page. What is added can be taken out again the same way,
and `clip-mcp uninstall` takes it out of every client at once.

A file is backed up before it is changed, into one `.clip-mcp.bak` beside it
that each change replaces, so the client's folder does not fill with copies.
A dry run says what would change without changing it.
"""

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
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
        status: For registering, `registered`, `unchanged`, `would register`,
            `skipped` (not installed), `failed`, or `confirm` for a client
            that only takes a server through a link the user approves in the
            client itself. For removing, `removed`, `absent`, `would remove`,
            `failed`, or `manual` for a client only the user can take it out of.
        detail: Where, or why not.
    """

    client: str
    status: str
    detail: str

def server_command() -> List[str]:
    """Find the command that starts this server from anywhere.

    The copy that is running comes first, when it is an installed one: the
    installer puts clip-mcp in a folder of its own, and a machine can hold a
    developer's copy as well, so the one asked is the one to hand out. Then
    the command `uv tool install` put in its default folder: it has an
    environment of its own, so a client starting it never collides with a
    development environment that another client — or a test run — is using
    at the same time. A copy inside a project's `.venv` is the last resort,
    because `uv sync` rewrites it and fails while any client has it running.

    Returns:
        The command as a list, first element an absolute path.
    """
    executable = "clip-mcp.exe" if sys.platform == "win32" else "clip-mcp"
    running = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if running and running.stem in (SERVER_NAME, f"{SERVER_NAME}-editor"):
        beside = running.resolve().with_name(executable)
        if beside.is_file() and ".venv" not in beside.parts:
            return [str(beside)]
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
    """Copy a file aside before changing it, over the copy the last change left.

    One copy, not one per change: these are other programs' folders, and a
    pile of dated copies in them is clutter nobody asked for.

    Args:
        path: The file.
    """
    if path.exists():
        shutil.copy2(path, path.with_name(f"{path.name}.clip-mcp.bak"))

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

def _chatgpt_desktop_installed() -> bool:
    """Whether the ChatGPT desktop app is installed, which keeps its servers where Codex does.

    It comes from the Microsoft Store, so it has a package folder rather than
    an install folder; its name is matched loosely so a renamed package is
    still found.
    """
    packages = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Packages"
    return sys.platform == "win32" and packages.is_dir() and any(packages.glob("OpenAI.ChatGPT*"))

def _codex_sections() -> "re.Pattern[str]":
    """Match this server's sections of a Codex config, under its name now or an old one.

    A section runs from its header to the next line that starts a header, or
    the end of the file, and takes its own sub-tables (`.env`) with it. By
    line, not by bracket: the `args` array holds brackets of its own.
    """
    names = "|".join(re.escape(name) for name in (SERVER_NAME, *OLD_NAMES))
    return re.compile(
        rf'^\[mcp_servers\.(?:{names}|"(?:{names})")(?:\.[^\]\n]+)?\][^\n]*\n?(?:(?!\[).*\n?)*',
        re.MULTILINE,
    )

def register_codex(command: List[str], dry_run: bool) -> Registration:
    """Register with Codex, in `~/.codex/config.toml`.

    The ChatGPT desktop app reads the same file — its Codex tab, not its
    ordinary chat, which only takes servers on the internet — so this one
    entry serves both. Only this server's own section is touched; the rest of
    the file is kept as it was written, comments and all.

    Args:
        command: The server command.
        dry_run: Say what would change without changing it.

    Returns:
        What happened.
    """
    folder = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    if not folder.is_dir() and shutil.which("codex") is None and not _chatgpt_desktop_installed():
        return Registration("Codex", "skipped", "not installed")
    path = folder / "config.toml"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = _toml_section(command)
    pattern = _codex_sections()
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

def _mcp_servers_file(name: str, folder: Path, file: str, command: List[str], dry_run: bool) -> Registration:
    """Register with a client that keeps its servers under `mcpServers` in a JSON file of its own.

    Args:
        name: The client's name.
        folder: Its settings folder; the client counts as installed when it exists.
        file: The settings file in that folder.
        command: The server command.
        dry_run: Say what would change without changing it.

    Returns:
        What happened.
    """
    path = folder / file
    entry = {"command": command[0], "args": command[1:]}
    try:
        status = _update_json(path, lambda settings: _replace_entry(settings.setdefault("mcpServers", {}), entry),
                              dry_run)
    except (OSError, ValueError) as error:
        return Registration(name, "failed", f"{path}: {error}")
    return Registration(name, status, str(path))

def register_gemini(command: List[str], dry_run: bool) -> Registration:
    """Register with Gemini CLI, in `~/.gemini/settings.json`, for this user in every folder.

    Args:
        command: The server command.
        dry_run: Say what would change without changing it.

    Returns:
        What happened.
    """
    folder = Path.home() / ".gemini"
    if not folder.is_dir() and shutil.which("gemini") is None:
        return Registration("Gemini CLI", "skipped", "not installed")
    return _mcp_servers_file("Gemini CLI", folder, "settings.json", command, dry_run)

def register_lm_studio(command: List[str], dry_run: bool) -> Registration:
    """Register with LM Studio, in `~/.lmstudio/mcp.json`; it loads the file again as soon as it changes.

    Args:
        command: The server command.
        dry_run: Say what would change without changing it.

    Returns:
        What happened.
    """
    folder = Path.home() / ".lmstudio"
    if not folder.is_dir():
        return Registration("LM Studio", "skipped", "not installed")
    return _mcp_servers_file("LM Studio", folder, "mcp.json", command, dry_run)

def cherry_studio_link(command: List[str]) -> str:
    """The link that offers this server to Cherry Studio, which asks the user before adding it.

    Cherry Studio keeps its servers in its own database rather than a file,
    so nothing here can write them; it takes them through this link instead,
    adds them switched off, and the user switches them on.

    Args:
        command: The server command.

    Returns:
        A `cherrystudio://` link.
    """
    payload = {"mcpServers": {SERVER_NAME: {"command": command[0], "args": command[1:]}}}
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
    return f"cherrystudio://mcp/install?servers={urllib.parse.quote(encoded, safe='')}"

def register_cherry_studio(command: List[str], dry_run: bool) -> Registration:
    """Say whether Cherry Studio is here, and give the link that adds this server to it.

    Args:
        command: The server command.
        dry_run: Unused: nothing is written either way.

    Returns:
        `skipped` when it is not installed; otherwise `confirm`, with the link.
    """
    if sys.platform == "win32":
        folder = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "CherryStudio"
    elif sys.platform == "darwin":
        folder = Path.home() / "Library" / "Application Support" / "CherryStudio"
    else:
        folder = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "CherryStudio"
    if not folder.is_dir():
        return Registration("Cherry Studio", "skipped", "not installed")
    return Registration("Cherry Studio", "confirm", cherry_studio_link(command))

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

# ----------------------------------------------------------------- taking it out again

def _drop_entry(servers: dict) -> bool:
    """Take this server, under its name now or an old one, out of a client's table of servers.

    Args:
        servers: The table, edited in place.

    Returns:
        Whether anything was there.
    """
    found = [name for name in (SERVER_NAME, *OLD_NAMES) if name in servers]
    for name in found:
        del servers[name]
    return bool(found)

def _remove_from_json(name: str, path: Path, table: str, dry_run: bool, comments: bool = False) -> Registration:
    """Take this server out of a client's JSON settings, leaving everything else as it was.

    Args:
        name: The client's name.
        path: Its settings file; a missing one means there is nothing to take out.
        table: The key its servers are under.
        dry_run: Say what would change without changing it.
        comments: The file may hold comments.

    Returns:
        What happened.
    """
    if not path.exists():
        return Registration(name, "absent", str(path))
    try:
        status = _update_json(path, lambda settings: isinstance(settings.get(table), dict) and _drop_entry(settings[table]),
                              dry_run, comments=comments)
    except (OSError, ValueError) as error:
        return Registration(name, "failed", f"{path}: {error}")
    return Registration(name, {"registered": "removed", "unchanged": "absent",
                               "would register": "would remove"}[status], str(path))

def unregister_claude_desktop(dry_run: bool) -> Registration:
    """Take this server out of the Claude desktop app's settings."""
    if sys.platform == "win32":
        folder = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Claude"
    elif sys.platform == "darwin":
        folder = Path.home() / "Library" / "Application Support" / "Claude"
    else:
        folder = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "Claude"
    return _remove_from_json("Claude Desktop", folder / "claude_desktop_config.json", "mcpServers", dry_run)

def unregister_opencode(dry_run: bool) -> Registration:
    """Take this server out of opencode's global config."""
    folder = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "opencode"
    path = next((folder / name for name in ("opencode.jsonc", "opencode.json") if (folder / name).exists()),
                folder / "opencode.json")
    return _remove_from_json("opencode", path, "mcp", dry_run, comments=path.suffix == ".jsonc")

def unregister_gemini(dry_run: bool) -> Registration:
    """Take this server out of Gemini CLI's settings."""
    return _remove_from_json("Gemini CLI", Path.home() / ".gemini" / "settings.json", "mcpServers", dry_run)

def unregister_lm_studio(dry_run: bool) -> Registration:
    """Take this server out of LM Studio's MCP list."""
    return _remove_from_json("LM Studio", Path.home() / ".lmstudio" / "mcp.json", "mcpServers", dry_run)

def unregister_codex(dry_run: bool) -> Registration:
    """Take this server's section out of the Codex config, which the ChatGPT desktop app shares."""
    path = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"
    if not path.exists():
        return Registration("Codex", "absent", str(path))
    text = path.read_text(encoding="utf-8")
    kept = _codex_sections().sub("", text)
    if kept == text:
        return Registration("Codex", "absent", str(path))
    if dry_run:
        return Registration("Codex", "would remove", str(path))
    try:
        _backup(path)
        path.write_text(kept.rstrip() + "\n" if kept.strip() else "", encoding="utf-8")
    except OSError as error:
        return Registration("Codex", "failed", f"{path}: {error}")
    return Registration("Codex", "removed", str(path))

def unregister_claude_code(dry_run: bool) -> Registration:
    """Take this server out of Claude Code, through its own command."""
    claude = shutil.which("claude")
    if claude is None:
        return Registration("Claude Code", "absent", "not installed")
    listed = subprocess.run([claude, "mcp", "get", SERVER_NAME], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60)
    if listed.returncode != 0:
        return Registration("Claude Code", "absent", "user scope")
    if dry_run:
        return Registration("Claude Code", "would remove", "user scope")
    removed = subprocess.run([claude, "mcp", "remove", "--scope", "user", SERVER_NAME],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if removed.returncode != 0:
        return Registration("Claude Code", "failed", (removed.stderr or removed.stdout).strip()[-300:])
    return Registration("Claude Code", "removed", "user scope")

def unregister_cherry_studio(dry_run: bool) -> Registration:
    """Say how to take this server out of Cherry Studio, which keeps it where nothing else can reach."""
    found = register_cherry_studio(["clip-mcp"], dry_run=True)
    if found.status == "skipped":
        return Registration("Cherry Studio", "absent", "not installed")
    return Registration("Cherry Studio", "manual", "remove clip-mcp in Cherry Studio: Settings → MCP Servers")

UNREGISTER: Dict[str, Callable[[bool], Registration]] = {
    "claude-code": unregister_claude_code,
    "claude-desktop": unregister_claude_desktop,
    "codex": unregister_codex,
    "opencode": unregister_opencode,
    "gemini-cli": unregister_gemini,
    "lm-studio": unregister_lm_studio,
    "cherry-studio": unregister_cherry_studio,
}

def unregister(only: Optional[List[str]] = None, dry_run: bool = False) -> List[Registration]:
    """Take this server out of every client, or the ones named.

    Args:
        only: Client keys; all of them when not given.
        dry_run: Say what would change without changing anything.

    Returns:
        One result per client.
    """
    results = []
    for key in only or list(UNREGISTER):
        try:
            results.append(UNREGISTER[key](dry_run))
        except (OSError, subprocess.SubprocessError) as error:
            results.append(Registration(key, "failed", str(error)))
    return results

CLIENTS: Dict[str, Callable[[List[str], bool], Registration]] = {
    "claude-code": register_claude_code,
    "claude-desktop": register_claude_desktop,
    "codex": register_codex,
    "opencode": register_opencode,
    "gemini-cli": register_gemini,
    "lm-studio": register_lm_studio,
    "cherry-studio": register_cherry_studio,
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
