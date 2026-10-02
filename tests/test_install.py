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
        # Path.home(), which Gemini CLI's and LM Studio's folders hang off.
        ("USERPROFILE", "home"), ("HOME", "home"),
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
    assert (config.parent / "opencode.jsonc.clip-mcp.bak").exists()
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


def not_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every command look uninstalled, whatever is on this machine."""
    monkeypatch.setattr(clients.shutil, "which", lambda name: None)

@pytest.mark.parametrize(("register", "folder", "file"), [
    (clients.register_gemini, ".gemini", "settings.json"),
    (clients.register_lm_studio, ".lmstudio", "mcp.json"),
])
def test_gemini_and_lm_studio_get_an_mcp_servers_entry_and_keep_their_other_settings(
    home: Path, monkeypatch: pytest.MonkeyPatch, register, folder: str, file: str,
) -> None:
    not_on_path(monkeypatch)
    assert register(COMMAND, dry_run=False).status == "skipped"
    settings = home / "home" / folder
    settings.mkdir(parents=True)
    (settings / file).write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
    assert register(COMMAND, dry_run=False).status == "registered"
    written = json.loads((settings / file).read_text(encoding="utf-8"))
    assert written["theme"] == "dark" and written["mcpServers"]["other"] == {"command": "x"}
    assert written["mcpServers"]["clip-mcp"] == {"command": COMMAND[0], "args": COMMAND[1:]}
    assert register(COMMAND, dry_run=False).status == "unchanged"

def test_cherry_studio_is_offered_the_server_by_its_own_install_link(home: Path) -> None:
    import base64
    import urllib.parse

    assert clients.register_cherry_studio(COMMAND, dry_run=False).status == "skipped"
    if clients.sys.platform != "win32":
        pytest.skip("Cherry Studio's folder is only under APPDATA on Windows")
    (home / "roaming" / "CherryStudio").mkdir(parents=True)
    result = clients.register_cherry_studio(COMMAND, dry_run=False)
    assert result.status == "confirm" and result.detail.startswith("cherrystudio://mcp/install?servers=")
    encoded = urllib.parse.unquote(result.detail.split("servers=", 1)[1])
    payload = json.loads(base64.b64decode(encoded))
    # Cherry Studio checks each server strictly: only these fields are allowed.
    assert payload == {"mcpServers": {"clip-mcp": {"command": COMMAND[0], "args": COMMAND[1:]}}}

def test_the_chatgpt_desktop_app_counts_as_a_place_for_the_codex_entry(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    not_on_path(monkeypatch)
    assert clients.register_codex(COMMAND, dry_run=False).status == "skipped"
    if clients.sys.platform != "win32":
        pytest.skip("the ChatGPT desktop app is found by its Windows package")
    (home / "local" / "Packages" / "OpenAI.ChatGPT-Desktop_2p2nqsd0c76g0").mkdir(parents=True)
    assert clients.register_codex(COMMAND, dry_run=False).status == "registered"
    assert "[mcp_servers.clip-mcp]" in (home / "codex" / "config.toml").read_text(encoding="utf-8")


# --- consent, and taking it out again ---------------------------------------------------

def test_a_second_change_replaces_the_backup_rather_than_adding_one(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    not_on_path(monkeypatch)
    settings = home / "home" / ".gemini"
    settings.mkdir(parents=True)
    (settings / "settings.json").write_text("{}", encoding="utf-8")
    clients.register_gemini(COMMAND, dry_run=False)
    clients.unregister_gemini(dry_run=False)
    assert sorted(path.name for path in settings.iterdir()) == ["settings.json", "settings.json.clip-mcp.bak"]

@pytest.mark.parametrize(("register", "unregister", "folder", "file"), [
    (clients.register_gemini, clients.unregister_gemini, ".gemini", "settings.json"),
    (clients.register_lm_studio, clients.unregister_lm_studio, ".lmstudio", "mcp.json"),
])
def test_disconnecting_takes_out_only_this_server(
    home: Path, monkeypatch: pytest.MonkeyPatch, register, unregister, folder: str, file: str,
) -> None:
    not_on_path(monkeypatch)
    settings = home / "home" / folder
    settings.mkdir(parents=True)
    (settings / file).write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
    register(COMMAND, dry_run=False)
    assert unregister(dry_run=False).status == "removed"
    assert json.loads((settings / file).read_text(encoding="utf-8")) == {"theme": "dark", "mcpServers": {"other": {"command": "x"}}}
    assert unregister(dry_run=False).status == "absent"

def test_a_client_never_connected_is_not_given_a_file(home: Path) -> None:
    assert clients.unregister_lm_studio(dry_run=False).status == "absent"
    assert not (home / "home" / ".lmstudio").exists()

def test_disconnecting_codex_keeps_the_rest_of_its_config(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (home / "codex").mkdir()
    (home / "codex" / "config.toml").write_text('model = "gpt-5"\n\n[mcp_servers.other]\ncommand = "x"\n', encoding="utf-8")
    clients.register_codex(COMMAND, dry_run=False)
    assert clients.unregister_codex(dry_run=False).status == "removed"
    kept = tomllib.loads((home / "codex" / "config.toml").read_text(encoding="utf-8"))
    assert kept == {"model": "gpt-5", "mcp_servers": {"other": {"command": "x"}}}

def test_cherry_studio_is_never_taken_out_behind_the_users_back(home: Path) -> None:
    if clients.sys.platform != "win32":
        pytest.skip("Cherry Studio's folder is only under APPDATA on Windows")
    (home / "roaming" / "CherryStudio").mkdir(parents=True)
    assert clients.unregister_cherry_studio(dry_run=False).status == "manual"

def test_setup_with_nobody_to_ask_connects_nothing_it_was_not_told_to(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import cli

    not_on_path(monkeypatch)
    monkeypatch.setattr(cli, "_ask", lambda question: False)
    monkeypatch.setattr(clients, "server_command", lambda: COMMAND)
    (home / "home" / ".gemini").mkdir(parents=True)
    (home / "home" / ".lmstudio").mkdir(parents=True)
    cli.setup(None, None, dry_run=False, connect_all=False, shortcuts=False)
    assert not (home / "home" / ".gemini" / "settings.json").exists()
    assert not (home / "home" / ".lmstudio" / "mcp.json").exists()
    cli.setup(None, ["lm-studio"], dry_run=False, connect_all=False, shortcuts=False)
    assert not (home / "home" / ".gemini" / "settings.json").exists()
    assert (home / "home" / ".lmstudio" / "mcp.json").exists()

def test_uninstall_takes_it_out_of_every_client_and_keeps_the_workspace_unless_asked(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from app import cli

    not_on_path(monkeypatch)
    monkeypatch.setattr(cli, "_ask", lambda question: False)
    (home / "home" / ".gemini").mkdir(parents=True)
    clients.register_gemini(COMMAND, dry_run=False)
    workspace.write_pointer(tmp_path / "chosen")
    (tmp_path / "chosen").mkdir()
    cli.uninstall(dry_run=False, delete_data=False)
    assert json.loads((home / "home" / ".gemini" / "settings.json").read_text(encoding="utf-8"))["mcpServers"] == {}
    assert not workspace.pointer_file().exists()
    assert (tmp_path / "chosen").is_dir()

def test_uninstall_offers_to_delete_the_workspace_it_was_using_not_another(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from app import cli
    from app.storage import housekeeping

    asked, recycled = [], []
    monkeypatch.setattr(cli, "_ask", lambda question: asked.append(question) or True)
    monkeypatch.setattr(housekeeping, "to_recycle_bin", lambda paths: recycled.extend(paths) or True)
    monkeypatch.setattr(clients, "unregister", lambda only=None, dry_run=False: [])
    (tmp_path / "chosen").mkdir()
    workspace.write_pointer(tmp_path / "chosen")
    cli.uninstall(dry_run=False, delete_data=False)
    assert recycled == [str((tmp_path / "chosen").resolve())]
