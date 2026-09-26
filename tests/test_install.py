"""Tests for finding the workspace from anywhere, and for registering with AI clients.

Every client file here is written under a temporary home, never the real one.
"""

import json
import tomllib
from pathlib import Path

import pytest

from app import clients, workspace

@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every per-user folder this touches at a temporary one.

    Returns:
        The temporary home.
    """
    for variable, folder in (
        ("APPDATA", "roaming"), ("LOCALAPPDATA", "local"), ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"), ("CODEX_HOME", "codex"), ("UV_TOOL_BIN_DIR", "bin"),
    ):
        monkeypatch.setenv(variable, str(tmp_path / folder))
    monkeypatch.delenv(workspace.ENVIRONMENT_VARIABLE, raising=False)
    return tmp_path

# --- the workspace -----------------------------------------------------------------------

def test_a_copy_of_the_source_keeps_its_workspace_inside_the_project(home: Path) -> None:
    chosen = workspace.resolve()
    assert chosen.source == "source checkout"
    assert chosen.path == workspace.source_checkout() / "workspace"

def test_where_the_server_was_started_from_makes_no_difference(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fault this replaces: a client started in another folder got a new, empty workspace there.
    monkeypatch.chdir(tmp_path)
    assert workspace.resolve().path != tmp_path / "workspace"

def test_the_pointer_wins_over_where_the_project_is(home: Path, tmp_path: Path) -> None:
    target = tmp_path / "影片工作區" / "ws"
    workspace.write_pointer(target)
    assert workspace.resolve() == workspace.Workspace(target.resolve(), "pointer")

def test_the_environment_wins_over_the_pointer(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace.write_pointer(tmp_path / "pointed")
    monkeypatch.setenv(workspace.ENVIRONMENT_VARIABLE, str(tmp_path / "chosen"))
    assert workspace.resolve() == workspace.Workspace((tmp_path / "chosen").resolve(), "environment")

def test_a_pointer_that_cannot_be_read_is_passed_over(home: Path) -> None:
    workspace.pointer_file().parent.mkdir(parents=True)
    workspace.pointer_file().write_text("workspace = [not toml", encoding="utf-8")
    assert workspace.resolve().source == "source checkout"

def test_an_installed_package_keeps_its_workspace_with_the_user(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workspace, "source_checkout", lambda: None)
    assert workspace.resolve() == workspace.Workspace(workspace.user_data_dir(), "user data")
    assert str(home) in str(workspace.user_data_dir())

# --- the command clients run -------------------------------------------------------------

def test_the_command_installed_for_every_client_comes_first(home: Path) -> None:
    installed = home / "bin" / ("clip-mcp.exe" if clients.sys.platform == "win32" else "clip-mcp")
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    assert clients.server_command() == [str(installed)]
    assert not clients.in_development_environment([str(installed)])

def test_a_project_venv_is_recognised_as_the_fragile_setup() -> None:
    assert clients.in_development_environment([str(Path("D:/p/clip_MCP/.venv/Scripts/clip-mcp.exe"))])

# --- registering -------------------------------------------------------------------------

COMMAND = [str(Path("C:/Users/someone/.local/bin/clip-mcp.exe"))]

def test_opencode_gets_a_full_entry_and_the_stale_one_goes(home: Path) -> None:
    config = home / "config" / "opencode" / "opencode.jsonc"
    config.parent.mkdir(parents=True)
    config.write_text(
        '{\n  // mine\n  "$schema": "https://opencode.ai/config.json",\n'
        '  "mcp": { "clip_mcp": { "enabled": true }, },\n  "theme": "dark", /* kept */\n}\n',
        encoding="utf-8",
    )
    assert clients.register_opencode(COMMAND, dry_run=True).status == "would register"
    assert "clip_mcp" in config.read_text(encoding="utf-8")

    assert clients.register_opencode(COMMAND, dry_run=False).status == "registered"
    written = json.loads(config.read_text(encoding="utf-8"))
    assert written["mcp"] == {"clip-mcp": {"type": "local", "command": COMMAND, "enabled": True}}
    assert written["theme"] == "dark" and written["$schema"].startswith("https://")
    assert list(config.parent.glob("opencode.jsonc.*.bak"))
    assert clients.register_opencode(COMMAND, dry_run=False).status == "unchanged"

def test_codex_gets_its_section_and_keeps_everything_else(home: Path) -> None:
    config = home / "codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        '# my settings\nmodel = "gpt-5"\n\n[mcp_servers.clip-mcp]\ncommand = "uv"\nargs = ["run", "clip-mcp"]\n'
        '[mcp_servers.clip-mcp.env]\nX = "1"\n\n[mcp_servers.other]\ncommand = "other"\nargs = ["[a]"]\n',
        encoding="utf-8",
    )
    assert clients.register_codex(COMMAND, dry_run=False).status == "registered"
    text = config.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    assert parsed["mcp_servers"]["clip-mcp"] == {"command": COMMAND[0], "args": []}
    # The array's brackets are not a header, and the old entry's sub-table went with it.
    assert parsed["mcp_servers"]["other"] == {"command": "other", "args": ["[a]"]}
    assert parsed["model"] == "gpt-5" and text.startswith("# my settings")
    assert clients.register_codex(COMMAND, dry_run=False).status == "unchanged"

def test_claude_desktop_is_registered_only_where_it_is_installed(home: Path) -> None:
    assert clients.register_claude_desktop(COMMAND, dry_run=False).status == "skipped"
    folder = (home / "roaming" / "Claude") if clients.sys.platform == "win32" else None
    if folder is None:
        pytest.skip("the desktop app's folder is only under APPDATA on Windows")
    folder.mkdir(parents=True)
    assert clients.register_claude_desktop(COMMAND, dry_run=False).status == "registered"
    written = json.loads((folder / "claude_desktop_config.json").read_text(encoding="utf-8"))
    assert written["mcpServers"]["clip-mcp"] == {"command": COMMAND[0], "args": []}
