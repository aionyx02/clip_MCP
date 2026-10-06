"""The `clip-mcp` command: run the server, set it up, or check it.

`clip-mcp` on its own is what every AI client runs, and it only ever speaks
MCP over stdio. `clip-mcp setup` and `clip-mcp check` are for the person
installing it. They are kept out of the server module so that neither loads
the server, and so that `setup` can move the workspace before anything opens
a database in the old one.
"""

import argparse
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import List, Optional

from app import clients, workspace
from app.engine.ffmpeg import hidden_window_flags

def _version() -> str:
    """Read the installed version.

    Returns:
        The version, or `unknown` when running from a bare copy of the source.
    """
    try:
        return version("clip-mcp")
    except PackageNotFoundError:
        return "unknown"

def _tool_version(name: str) -> str:
    """Find an external tool and the first line of its version.

    Args:
        name: `ffmpeg` or `ffprobe`.

    Returns:
        Where it is and its version, or that it is missing.
    """
    path = shutil.which(name)
    if path is None:
        return "not found on PATH — install it, or nothing can be analyzed or rendered"
    try:
        first = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=30,
                               creationflags=hidden_window_flags()).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return f"{path} (would not run)"
    return f"{path} ({first[0] if first else 'unknown version'})"

def check() -> int:
    """Say which workspace is in use and why, and whether what the server needs is there.

    Returns:
        The exit status: 0 when FFmpeg and ffprobe are both found.
    """
    chosen = workspace.resolve()
    pointed = workspace.read_pointer()
    command = clients.server_command()
    print(f"clip-mcp {_version()}")
    print(f"workspace:  {chosen.path}")
    print(f"  chosen by {chosen.source}; exists: {'yes' if chosen.path.is_dir() else 'no, created on first use'}")
    print(f"pointer:    {workspace.pointer_file()} ({'-> ' + str(pointed) if pointed else 'not set'})")
    print(f"command:    {' '.join(command)}")
    if clients.in_development_environment(command):
        print("  this is the project's own .venv, which fails when two clients run it at once;")
        print("  install it for every client with `uv tool install --editable .` in the project")
    missing = 0
    for tool in ("ffmpeg", "ffprobe"):
        found = _tool_version(tool)
        missing += found.startswith("not found")
        print(f"{tool + ':':<11} {found}")
    print("clients:")
    for result in clients.register(command, dry_run=True):
        state = {"unchanged": "registered", "would register": "not registered, or out of date",
                 "confirm": "add it from the editor's 連接 AI page, or open this link"}.get(
            result.status, result.status,
        )
        print(f"  {result.client:<15} {state} - {result.detail}")
    return 1 if missing else 0

# Set by `--no-prompt`: the installer runs these commands in a window nobody sees, where a
# question would wait forever, so it says outright that there is no one to ask.
NO_PROMPT = False

def _ask(question: str) -> bool:
    """Ask a yes-or-no question, with no for an answer unless the user says yes.

    Args:
        question: The question, without the `[y/N]`.

    Returns:
        Whether they said yes. No one to ask — the installer, a script — is a no.
    """
    if NO_PROMPT or not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False

def setup(
    workspace_path: Optional[str], only: Optional[List[str]], dry_run: bool, connect_all: bool, shortcuts: bool,
    ffmpeg_dir: Optional[str] = None,
) -> int:
    """Point clip-mcp at one workspace, and connect it to the AI clients the user agrees to.

    Another program's settings are changed only with the user's say-so: the
    clients named with `--client`, every one with `--all`, or each one the
    user answers yes for. Run where nobody can answer, it connects nothing it
    was not told to and says how to.

    Args:
        workspace_path: The workspace to point at; the one currently in use
            when not given, so running setup again changes nothing.
        only: Client keys the user chose to connect.
        dry_run: Say what would change without changing anything.
        connect_all: The user chose to connect every client installed here.
        shortcuts: The user chose the desktop and Start menu shortcuts.
        ffmpeg_dir: Where the installer put FFmpeg, to be found there from now on.

    Returns:
        The exit status: 0 unless a client could not be connected.
    """
    target = Path(workspace_path).expanduser().resolve() if workspace_path else workspace.resolve().path
    command = clients.server_command()
    if clients.in_development_environment(command):
        print("warning: the only clip-mcp found is inside the project's .venv. Clients will fail to start it")
        print("while another is running it. Run `uv tool install --editable .` in the project, then setup again.")
    if dry_run:
        print(f"would point the workspace at {target} ({workspace.pointer_file()})")
    else:
        written = workspace.write_pointer(target, ffmpeg=Path(ffmpeg_dir) if ffmpeg_dir else None)
        workspace.use_installed_ffmpeg()
        print(f"workspace: {target}")
        print(f"  written to {written}")
    print(f"command:   {' '.join(command)}")

    failed = False
    print("AI clients:")
    for key, found in zip(clients.CLIENTS, clients.register(command, dry_run=True)):
        if found.status == "skipped":
            print(f"  {found.client:<15} not installed")
            continue
        if found.status == "unchanged":
            print(f"  {found.client:<15} connected")
            continue
        if found.status == "confirm":
            print(f"  {found.client:<15} only takes a server through a link it asks you about; open it to add clip-mcp:")
            print(f"    {found.detail}")
            continue
        explicit = (only is not None and key in only) or connect_all
        if dry_run and not explicit and only is None:
            print(f"  {found.client:<15} not connected — would ask before adding an entry to {found.detail}")
            continue
        wanted = explicit or (
            only is None and _ask(f"  Connect {found.client}? This adds a clip-mcp entry to {found.detail}")
        )
        if not wanted:
            print(f"  {found.client:<15} not connected — `clip-mcp setup --client {key}`, or the editor's 連接 AI page")
            continue
        result = clients.register(command, [key], dry_run)[0]
        failed |= result.status == "failed"
        print(f"  {result.client:<15} {result.status} - {result.detail}")

    from app.ui import shortcut

    if dry_run and not shortcuts:
        print("editor: would ask before adding shortcuts to the desktop and the Start menu")
    elif shortcuts or _ask("Add the editor to the desktop and the Start menu?"):
        print("editor:")
        for line in shortcut.create(dry_run):
            print(f"  {line}")
    if not dry_run:
        print("Restart each client you connected, or reload its MCP servers, to pick this up.")
    return 1 if failed else 0

def uninstall(dry_run: bool, delete_data: bool) -> int:
    """Take clip-mcp out of everything it was put into, before the program itself is removed.

    This copy's entry in every AI client (one that starts another copy of
    clip-mcp is left connected), the shortcuts, and the pointer file go — the
    pointer stays when another copy is here, which reads it too, and so does
    the workspace its projects are in, whatever was asked. Otherwise the
    workspace — projects, analyses, models — goes only when the user says so,
    and then to the recycle bin. Finished videos are never touched.

    Args:
        dry_run: Say what would change without changing anything.
        delete_data: The user chose to delete the workspace too.

    Returns:
        The exit status: 0 unless something could not be taken out.
    """
    # Found before the pointer goes: after that, a different folder would answer to the name.
    data = workspace.resolve().path
    failed = False
    print("AI clients:")
    # Only this copy's entries: another copy on this machine, a developer's say, stays connected.
    owner = clients.server_command()
    others = clients.other_copies(owner)
    for result in clients.unregister(dry_run=dry_run, owner=owner):
        failed |= result.status == "failed"
        print(f"  {result.client:<15} {result.status} - {result.detail}")
        if result.status == "kept":
            others.append(result.detail)
    from app.ui import shortcut

    if dry_run:
        print(f"would remove the shortcuts: {', '.join(str(place) for place in shortcut.places() if place.exists()) or 'none'}")
    else:
        for removed in shortcut.remove():
            print(f"removed {removed}")
    pointer = workspace.pointer_file()
    if pointer.exists() and others:
        # Another copy reads it too, and would lose its workspace with it; only the FFmpeg
        # in this copy's folder, which goes with the folder, is taken out of it.
        print(f"kept {pointer}: another copy of clip-mcp uses it")
        if not dry_run:
            workspace.forget_ffmpeg(Path(owner[0]).resolve().parent.parent)
    elif pointer.exists():
        print(f"{'would remove' if dry_run else 'removed'} {pointer}")
        if not dry_run:
            pointer.unlink()
            if pointer.parent.exists() and not any(pointer.parent.iterdir()):
                pointer.parent.rmdir()
    if data.is_dir() and others:
        # The other copy's projects are in it; asked or not, it is not this uninstall's to take.
        print(f"kept the workspace: {data} - another copy of clip-mcp uses it")
    elif data.is_dir():
        if dry_run and not delete_data:
            print(f"would ask whether to delete the workspace at {data}; kept unless you say so")
            wanted = False
        else:
            wanted = delete_data or _ask(
                f"Delete the workspace at {data} (projects, analyses, models)? It goes to the recycle bin."
            )
        if wanted and not dry_run:
            from app.storage.housekeeping import to_recycle_bin

            print(f"workspace moved to the recycle bin: {data}" if to_recycle_bin([str(data)])
                  else f"could not move {data} to the recycle bin; delete it yourself if you want it gone")
        elif wanted:
            print(f"would move the workspace to the recycle bin: {data}")
        elif not dry_run:
            print(f"kept the workspace: {data}")
    print(f"Finished videos in {workspace.output_dir()} are yours and were left alone.")
    from app.ui.updates import installed_by_setup

    if installed_by_setup() is None:
        print("Last, remove the program itself: uv tool uninstall clip-mcp")
    return 1 if failed else 0

def main(argv: Optional[List[str]] = None) -> None:
    """Run the command.

    Args:
        argv: Arguments after the command name; the process's own when not given.
    """
    parser = argparse.ArgumentParser(
        prog="clip-mcp",
        description="Video editing for AI clients over MCP. With no command, runs the server on stdio.",
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve", help="run the MCP server on stdio (the default)")
    commands.add_parser("check", help="show the workspace in use, the command clients run, and what is missing")
    set_up = commands.add_parser("setup", help="choose the workspace, and connect the AI clients you agree to")
    set_up.add_argument("--workspace", help="the workspace to use from now on; the current one when omitted")
    set_up.add_argument("--client", action="append", choices=sorted(clients.CLIENTS), dest="clients",
                        help="connect this client without asking; may be given more than once")
    set_up.add_argument("--all", action="store_true", dest="connect_all",
                        help="connect every client installed here without asking")
    set_up.add_argument("--shortcuts", action="store_true", help="add the editor's shortcuts without asking")
    set_up.add_argument("--dry-run", action="store_true", help="say what would change, and change nothing")
    set_up.add_argument("--ffmpeg-dir", help="where the installer put ffmpeg.exe and ffprobe.exe")
    set_up.add_argument("--no-prompt", action="store_true", help="never ask; answer no to anything not given")
    remove = commands.add_parser("uninstall", help="take clip-mcp out of every AI client, and its shortcuts")
    remove.add_argument("--delete-data", action="store_true", help="also move the workspace to the recycle bin")
    remove.add_argument("--dry-run", action="store_true", help="say what would change, and change nothing")
    remove.add_argument("--no-prompt", action="store_true", help="never ask; answer no to anything not given")
    editor = commands.add_parser("ui", help="open the editor in the browser, to check and adjust a cut by hand")
    editor.add_argument("--no-browser", action="store_true", help="serve it without opening a window")
    arguments = parser.parse_args(argv)
    workspace.use_installed_ffmpeg()
    global NO_PROMPT
    NO_PROMPT = getattr(arguments, "no_prompt", False)

    if arguments.command == "check":
        sys.exit(check())
    if arguments.command == "setup":
        sys.exit(setup(arguments.workspace, arguments.clients, arguments.dry_run, arguments.connect_all,
                       arguments.shortcuts, arguments.ffmpeg_dir))
    if arguments.command == "uninstall":
        sys.exit(uninstall(arguments.dry_run, arguments.delete_data))
    if arguments.command == "ui":
        from app.ui.launch import run

        run(not arguments.no_browser)
        return
    from app.server import main as serve

    serve()

if __name__ == "__main__":
    main()
