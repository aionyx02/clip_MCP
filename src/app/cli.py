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
        first = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=30).stdout.splitlines()
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
        state = {"unchanged": "registered", "would register": "not registered, or out of date"}.get(
            result.status, result.status,
        )
        print(f"  {result.client:<15} {state} - {result.detail}")
    return 1 if missing else 0

def setup(workspace_path: Optional[str], only: Optional[List[str]], dry_run: bool) -> int:
    """Point every client at this installation, and at one workspace.

    Args:
        workspace_path: The workspace to point at; the one currently in use
            when not given, so running setup again changes nothing.
        only: Client keys to register with; every one installed when not given.
        dry_run: Say what would change without changing anything.

    Returns:
        The exit status: 0 unless a client could not be registered.
    """
    target = Path(workspace_path).expanduser().resolve() if workspace_path else workspace.resolve().path
    command = clients.server_command()
    if clients.in_development_environment(command):
        print("warning: the only clip-mcp found is inside the project's .venv. Clients will fail to start it")
        print("while another is running it. Run `uv tool install --editable .` in the project, then setup again.")
    if dry_run:
        print(f"would point the workspace at {target} ({workspace.pointer_file()})")
    else:
        written = workspace.write_pointer(target)
        print(f"workspace: {target}")
        print(f"  written to {written}")
    print(f"command:   {' '.join(command)}")
    failed = False
    for result in clients.register(command, only, dry_run):
        failed |= result.status == "failed"
        print(f"  {result.client:<15} {result.status} - {result.detail}")
    if not dry_run:
        print("Restart each client, or reload its MCP servers, to pick this up.")
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
    set_up = commands.add_parser("setup", help="register with the AI clients on this machine")
    set_up.add_argument("--workspace", help="the workspace to use from now on; the current one when omitted")
    set_up.add_argument("--client", action="append", choices=sorted(clients.CLIENTS), dest="clients",
                        help="only this client; may be given more than once")
    set_up.add_argument("--dry-run", action="store_true", help="say what would change, and change nothing")
    arguments = parser.parse_args(argv)

    if arguments.command == "check":
        sys.exit(check())
    if arguments.command == "setup":
        sys.exit(setup(arguments.workspace, arguments.clients, arguments.dry_run))
    from app.server import main as serve

    serve()

if __name__ == "__main__":
    main()
