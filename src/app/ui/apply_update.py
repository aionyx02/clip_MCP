"""Replace an installation's `program` folder with a staged one, and put the old one back if it fails.

Run by the editor as a separate process once it has staged a newer program in
`<install folder>\\program.new`, and then the editor quits:

    <install folder>\\runtime\\python.exe -I -S apply_update.py <install folder> <version> [--relaunch]

It runs from a copy outside `program`, with `-S` so the `.pth` that puts
`program` on the path is not read, and imports nothing but the standard
library: nothing it needs is in the folder it replaces.

The order, and what each failure leaves behind:

1. Stop whatever runs from the install folder (an AI client's MCP server, the
   editor), but not this process. Windows will not rename a folder whose files
   a running program holds.
2. `program` -> `program.old`. Failing here changes nothing.
3. `program.new` -> `program`. Failing here renames `program.old` back.
4. Smoke check: `bin\\clip-mcp.exe --version` must name the new version, and
   the server and editor modules must import. Failing here puts the old folder
   back, so the copy that started still starts.
5. `release.json` takes the new version; `program.old` is removed.

The outcome is written to `update-result.json` in the install folder for the
editor to show on its next start, and the steps to `update.log`.
"""

import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

RESULT = "update-result.json"
LOG = "update.log"
RELEASE = "release.json"
# Provisional: how long a smoke check may take, and how long to keep trying a rename that an
# antivirus scan or a process still closing holds up; neither has been measured beyond one machine.
SMOKE_TIMEOUT = 120
RENAME_TRIES = 20
RENAME_WAIT = 0.5

def _log(app: Path, line: str) -> None:
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        with open(app / LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {line}\n")
    except OSError:
        pass

def _rename(source: Path, target: Path) -> None:
    """Rename a folder, trying again for a while when Windows says something still holds it."""
    for attempt in range(RENAME_TRIES):
        try:
            os.rename(source, target)
            return
        except OSError:
            if attempt == RENAME_TRIES - 1:
                raise
            time.sleep(RENAME_WAIT)

def stop_running(app: Path) -> str:
    """Stop every process started from the install folder except this one, and wait for them to end.

    Returns:
        What was stopped, for the log.
    """
    here = str(app.resolve()).rstrip("\\") + "\\"
    script = (
        f"$here = '{here.replace(chr(39), chr(39) * 2)}'; "
        "$found = Get-Process | Where-Object { $_.Path -and $_.Id -ne " + str(os.getpid())
        + " -and $_.Path.StartsWith($here, [StringComparison]::OrdinalIgnoreCase) }; "
        "$found | ForEach-Object { \"$($_.Id) $($_.Path)\" }; "
        "$found | Stop-Process -Force -ErrorAction SilentlyContinue; "
        "$found | Wait-Process -Timeout 30 -ErrorAction SilentlyContinue"
    )
    done = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                          capture_output=True, text=True, timeout=90,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return done.stdout.strip()

def smoke(app: Path, version: str) -> Tuple[bool, str]:
    """Whether the installation starts as the new version.

    Returns:
        Whether it does, and what was seen when it does not.
    """
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        said = subprocess.run([str(app / "bin" / "clip-mcp.exe"), "--version"], capture_output=True, text=True,
                              timeout=SMOKE_TIMEOUT, cwd=str(app), env=environment, creationflags=flags)
        if said.returncode != 0 or said.stdout.strip() != f"clip-mcp {version}":
            return False, f"clip-mcp --version said {said.stdout.strip() or said.stderr.strip()[-300:]!r}"
        # Importing the server opens a workspace: a throwaway one, never the user's.
        scratch = tempfile.mkdtemp(prefix="clip-mcp-update-check-")
        try:
            imported = subprocess.run(
                [str(app / "runtime" / "python.exe"), "-I", "-B", "-c", "import app.server, app.ui.app"],
                capture_output=True, text=True, timeout=SMOKE_TIMEOUT, cwd=str(app),
                env=dict(environment, CLIP_MCP_WORKSPACE=scratch), creationflags=flags)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        if imported.returncode != 0:
            # The traceback's last line names the error, for the editor; all of it goes to update.log.
            _log(app, f"import check output:\n{imported.stderr.strip()[-4000:]}")
            lines = imported.stderr.strip().splitlines() or ["no output"]
            return False, f"the server does not import: {lines[-1][:300]}"
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"could not start: {error}"
    return True, ""

def apply(app: Path, version: str, stop: Callable[[], str], check: Callable[[], Tuple[bool, str]]) -> dict:
    """Swap `program.new` in for `program`, keeping the old one until the new one is seen to start.

    Args:
        app: The install folder.
        version: The version staged in `program.new`.
        stop: Stops what runs from the install folder.
        check: The smoke check, run once the new folder is in place.

    Returns:
        The outcome: `ok`, `version`, `previous`, and `reason` when it failed.
    """
    program, staged, old, failed = (app / name for name in ("program", "program.new", "program.old", "program.failed"))
    release = _read(app / RELEASE)
    outcome = {"ok": False, "version": version, "previous": release.get("version"), "kind": "program",
               "at": datetime.datetime.now().isoformat(timespec="seconds")}
    # A past update that stopped between its two renames left no program\: put the old one back.
    if not program.exists() and old.is_dir():
        _rename(old, program)
        _log(app, "put back program.old left by an update that did not finish")
    if not staged.is_dir():
        return {**outcome, "reason": "nothing staged"}
    for leftover in (old, failed):
        shutil.rmtree(leftover, ignore_errors=True)
    _log(app, f"stopping: {stop() or 'nothing was running'}")
    try:
        _rename(program, old)
    except OSError as error:
        shutil.rmtree(staged, ignore_errors=True)
        return {**outcome, "reason": f"program is still in use: {error}"}
    try:
        _rename(staged, program)
    except OSError as error:
        _rename(old, program)
        shutil.rmtree(staged, ignore_errors=True)
        return {**outcome, "reason": f"could not move the new program in: {error}"}
    started, why = check()
    if not started:
        _log(app, f"smoke check failed: {why}")
        _rename(program, failed)
        _rename(old, program)
        shutil.rmtree(failed, ignore_errors=True)
        return {**outcome, "reason": why}
    if release:
        (app / RELEASE).write_text(json.dumps({**release, "version": version}, indent=1), encoding="utf-8")
    shutil.rmtree(old, ignore_errors=True)
    return {**outcome, "ok": True}

def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

def main(argv: Optional[list] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    relaunch = "--relaunch" in arguments
    app, version = Path(arguments[0]), arguments[1]
    started = time.monotonic()
    _log(app, f"updating program to {version}")
    try:
        outcome = apply(app, version, lambda: stop_running(app), lambda: smoke(app, version))
    except Exception as error:  # noqa: BLE001 - whatever went wrong is told to the user, not lost
        outcome = {"ok": False, "version": version, "kind": "program", "reason": f"{type(error).__name__}: {error}",
                   "at": datetime.datetime.now().isoformat(timespec="seconds")}
    outcome["seconds"] = round(time.monotonic() - started, 1)
    outcome["announced"] = False
    _log(app, f"result: {json.dumps(outcome, ensure_ascii=False)}")
    (app / RESULT).write_text(json.dumps(outcome, ensure_ascii=False, indent=1), encoding="utf-8")
    if relaunch:
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen([str(app / "bin" / "clip-mcp-editor.exe")], cwd=str(app), creationflags=flags, close_fds=True)
    return 0 if outcome["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())
