"""Tests for what the installer relies on: its own FFmpeg, the right command, and updates."""

import io
import json
import os
import sys
from pathlib import Path

import pytest

from app import clients, workspace
from app.ui import updates

def test_the_installers_ffmpeg_is_put_first_on_this_process_path_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    folder = tmp_path / "ffmpeg"
    folder.mkdir()
    (folder / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")).write_bytes(b"")
    workspace.write_pointer(tmp_path / "workspace", ffmpeg=folder)
    monkeypatch.setenv("PATH", "elsewhere")
    assert workspace.use_installed_ffmpeg() == str(folder.resolve())
    assert os.environ["PATH"].split(os.pathsep)[0] == str(folder.resolve())

def test_moving_the_workspace_keeps_the_ffmpeg_the_installer_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    workspace.write_pointer(tmp_path / "first", ffmpeg=tmp_path / "ffmpeg")
    workspace.write_pointer(tmp_path / "second")
    text = workspace.pointer_file().read_text(encoding="utf-8")
    assert "second" in text and str((tmp_path / "ffmpeg").resolve()).replace("\\", "\\\\") in text

def test_no_ffmpeg_setting_leaves_the_path_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("PATH", "untouched")
    assert workspace.use_installed_ffmpeg() is None
    assert os.environ["PATH"] == "untouched"

def test_clients_are_given_the_copy_that_is_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = "clip-mcp.exe" if sys.platform == "win32" else "clip-mcp"
    installed = tmp_path / "Programs" / "clip-mcp" / "bin"
    installed.mkdir(parents=True)
    (installed / executable).write_bytes(b"")
    (installed / ("clip-mcp-editor.exe" if sys.platform == "win32" else "clip-mcp-editor")).write_bytes(b"")
    # A developer's copy in uv's default folder must not win over the one the editor runs from.
    default = tmp_path / "local-bin"
    default.mkdir()
    (default / executable).write_bytes(b"")
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(default))
    monkeypatch.setattr(sys, "argv", [str(installed / "clip-mcp-editor.exe")])
    assert clients.server_command() == [str((installed / executable).resolve())]

def test_versions_compare_as_numbers() -> None:
    assert updates._as_tuple("v0.10.0") > updates._as_tuple("0.9.9")
    assert updates._as_tuple("v0.1.0") == updates._as_tuple("0.1.0")

def test_a_copy_not_from_the_installer_is_never_offered_an_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(updates, "installed_by_setup", lambda: None)
    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("asked GitHub"))
    assert updates.latest(force=True)["available"] is False

def test_a_newer_release_with_an_installer_is_offered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = {"tag_name": "v99.0.0", "body": "新功能", "assets": [
        {"name": "clip-mcp-setup-99.0.0.exe", "browser_download_url": "https://example/setup.exe", "size": 10},
    ]}
    monkeypatch.setattr(updates, "installed_by_setup", lambda: tmp_path)
    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(release).encode()))
    found = updates.latest(force=True)
    assert found["available"] and found["version"] == "99.0.0" and found["notes"] == "新功能"

def test_no_network_is_simply_no_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(*args, **kwargs):
        raise OSError("no network")
    monkeypatch.setattr(updates, "installed_by_setup", lambda: tmp_path)
    monkeypatch.setattr(updates.urllib.request, "urlopen", offline)
    assert updates.latest(force=True)["available"] is False

def test_the_editor_will_not_update_while_a_render_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.testclient import TestClient

    from app import server
    from app.ui import app as editor

    monkeypatch.setattr(server.repo, "active_job_ids", lambda: {"job"})
    monkeypatch.setattr(updates, "start_update", lambda on_started: pytest.fail("started an update"))
    assert TestClient(editor.create_app()).post("/api/update", json={}).status_code == 409

def test_the_download_says_how_far_it_has_got(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    seen = []

    class Arriving(io.BytesIO):
        headers: dict = {}

        def read(self, size=-1):
            seen.append(updates.download_progress()["done"])
            return super().read(size)

    monkeypatch.setattr(updates, "DOWNLOADS", tmp_path)
    monkeypatch.setattr(updates, "CHUNK", 4)
    monkeypatch.setattr(updates, "latest", lambda: {"available": True, "version": "9", "url": "https://example/setup.exe",
                                                 "size": 10})
    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *args, **kwargs: Arriving(b"0123456789"))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: None)
    monkeypatch.setattr(updates.threading, "Thread", lambda **kwargs: type("T", (), {"start": lambda self: None})())
    updates.start_update(lambda: None)
    assert seen[:3] == [0, 4, 8] and updates.download_progress() == {"done": 10, "total": 10}

@pytest.mark.parametrize(("first_line", "option"), [
    ("ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers", "-filter_complex_script"),
    ("ffmpeg version n7.1 Copyright (c) 2000-2024 the FFmpeg developers", "-/filter_complex"),
    ("ffmpeg version 9.0.2-essentials_build-www.gyan.dev Copyright (c) 2000-2026", "-/filter_complex"),
    ("ffmpeg version N-118000-g1234abcd Copyright (c) 2000-2026", "-/filter_complex"),
])
def test_a_filter_graph_in_a_file_is_handed_over_the_way_that_ffmpeg_reads_it(
    first_line: str, option: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from app.engine import ffmpeg

    ffmpeg._graph_file_option.cache_clear()
    monkeypatch.setattr(ffmpeg.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=first_line + "\nbuilt with gcc"))
    moved = ffmpeg.graph_from_file(["ffmpeg-under-test", "-filter_complex", "[0:v]null[v]"], str(tmp_path / "graph.txt"))
    ffmpeg._graph_file_option.cache_clear()
    assert moved[1] == option and moved[2] == str(tmp_path / "graph.txt")

def test_the_installer_holds_every_package_to_the_tested_versions() -> None:
    root = Path(__file__).resolve().parent.parent
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    script = (root / "installer" / "install.ps1").read_text(encoding="utf-8")
    setup = (root / "installer" / "clip-mcp.iss").read_text(encoding="utf-8")
    # The lockfile's versions, exported where the installer picks them up...
    assert "uv export --locked" in workflow and r"installer\build\constraints.txt" in workflow
    assert r'Source: "build\constraints.txt"' in setup and "-Constraints" in setup
    # ...and the install held to them, not left to take whatever is newest that day.
    assert '"--constraints", $Constraints' in script

def test_pyav_is_held_below_the_release_that_breaks_transcription() -> None:
    import tomllib

    root = Path(__file__).resolve().parent.parent
    dependencies = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert any(entry.replace(" ", "").startswith("av") and "<19" in entry for entry in dependencies)
