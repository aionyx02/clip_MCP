"""Tests for updating an installed copy: which download it needs, checking it, and swapping the program safely."""

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

from app.ui import apply_update, updates

FINGERPRINTS = {"runtime": "r1", "launchers": "l1", "ffmpeg": "f1"}

def _manifest(**changed) -> dict:
    return {"version": "99.0.0", **FINGERPRINTS, **changed, "assets": {
        "program": {"name": "clip-mcp-program-99.0.0.zip", "size": 10, "sha256": "a"},
        "installer": {"name": "clip-mcp-setup-99.0.0.exe", "size": 100, "sha256": "b"},
    }}

def test_a_release_that_changed_only_the_program_replaces_only_the_program() -> None:
    assert updates.choose({"version": "1.0.0", **FINGERPRINTS}, _manifest()) == ("program", [])

@pytest.mark.parametrize("part", ["runtime", "launchers", "ffmpeg"])
def test_a_release_that_changed_what_the_program_runs_on_needs_the_installer(part: str) -> None:
    assert updates.choose({"version": "1.0.0", **FINGERPRINTS}, _manifest(**{part: "new"})) == ("installer", [part])

def test_an_install_with_nothing_to_compare_needs_the_installer() -> None:
    assert updates.choose({}, _manifest()) == ("installer", ["unknown"])

def _serve(monkeypatch: pytest.MonkeyPatch, routes: dict) -> list:
    asked = []

    def urlopen(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else request
        asked.append(url)
        body = routes[url]
        answer = io.BytesIO(body if isinstance(body, bytes) else json.dumps(body).encode())
        answer.headers = {}
        return answer
    monkeypatch.setattr(updates.urllib.request, "urlopen", urlopen)
    return asked

def _release(manifest: dict) -> dict:
    names = [MANIFEST_NAME, *(asset["name"] for asset in manifest["assets"].values())]
    return {"tag_name": "v99.0.0", "body": "新功能",
            "assets": [{"name": name, "browser_download_url": f"https://example/{name}"} for name in names]}

MANIFEST_NAME = updates.MANIFEST

def _installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: dict = None) -> Path:
    root = tmp_path / "app"
    (root / "program" / "app").mkdir(parents=True)
    (root / "program" / "app" / "__init__.py").write_text("old", encoding="utf-8")
    (root / updates.RELEASE).write_text(json.dumps(record if record is not None else {"version": "1.0.0", **FINGERPRINTS}),
                                        encoding="utf-8")
    monkeypatch.setattr(updates, "installed_by_setup", lambda: root)
    monkeypatch.setattr(updates, "DOWNLOADS", tmp_path / "downloads")
    return root

@pytest.mark.parametrize(("record", "plan"), [(None, "program"), ({"version": "1.0.0", **FINGERPRINTS, "runtime": "r0"}, "installer")])
def test_the_offer_names_the_download_this_install_needs(record, plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _installed(tmp_path, monkeypatch, record)
    manifest = _manifest()
    _serve(monkeypatch, {updates.LATEST: _release(manifest), f"https://example/{MANIFEST_NAME}": manifest})
    found = updates.latest(force=True)
    assert found["available"] and found["plan"] == plan and found["notes"] == "新功能"
    assert found["size"] == manifest["assets"][plan]["size"] and found["installer_size"] == 100

def test_a_newer_release_without_a_manifest_is_refused_with_the_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _installed(tmp_path, monkeypatch)
    release = {"tag_name": "v99.0.0", "assets": [{"name": "clip-mcp-setup-99.0.0.exe", "browser_download_url": "x"}]}
    _serve(monkeypatch, {updates.LATEST: release})
    found = updates.latest(force=True)
    assert found["available"] is False and "manifest" in found["refused"]

def _program_zip(version: str, extra: dict = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("app/__init__.py", "new")
        archive.writestr(f"clip_mcp-{version}.dist-info/METADATA", f"Version: {version}\n")
        for name, text in (extra or {}).items():
            archive.writestr(name, text)
    return buffer.getvalue()

def _found(data: bytes, sha256: str = None) -> dict:
    return {"available": True, "version": "99.0.0", "plan": "program", "name": "clip-mcp-program-99.0.0.zip",
            "url": "https://example/program.zip", "size": len(data), "sha256": sha256 or hashlib.sha256(data).hexdigest()}

def test_a_download_whose_checksum_differs_is_refused_and_nothing_is_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _installed(tmp_path, monkeypatch)
    data = _program_zip("99.0.0")
    _serve(monkeypatch, {"https://example/program.zip": data})
    with pytest.raises(RuntimeError, match="SHA-256"):
        updates.prepare(_found(data, sha256="0" * 64))
    assert not (root / "program.new").exists() and not list((tmp_path / "downloads").iterdir())

def test_a_checked_program_is_staged_beside_the_running_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _installed(tmp_path, monkeypatch)
    data = _program_zip("99.0.0")
    _serve(monkeypatch, {"https://example/program.zip": data})
    assert updates.prepare(_found(data))["plan"] == "program"
    assert (root / "program.new" / "app" / "__init__.py").read_text(encoding="utf-8") == "new"
    assert (root / "program" / "app" / "__init__.py").read_text(encoding="utf-8") == "old"

@pytest.mark.parametrize(("data", "match"), [
    (_program_zip("99.0.0", {"../escape.py": "x"}), "outside"),
    (_program_zip("98.0.0"), "not clip-mcp 99.0.0"),
])
def test_a_program_zip_that_is_not_the_release_is_refused(data: bytes, match: str, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    root = _installed(tmp_path, monkeypatch)
    _serve(monkeypatch, {"https://example/program.zip": data})
    with pytest.raises(RuntimeError, match=match):
        updates.prepare(_found(data))
    assert not (root / "program.new").exists() and not (tmp_path / "escape.py").exists()

# --- the swap ------------------------------------------------------------------------------------

def _staged(root: Path) -> None:
    (root / "program.new" / "app").mkdir(parents=True)
    (root / "program.new" / "app" / "__init__.py").write_text("new", encoding="utf-8")

def test_the_new_program_replaces_the_old_once_it_starts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _installed(tmp_path, monkeypatch)
    _staged(root)
    stopped = []
    outcome = apply_update.apply(root, "99.0.0", lambda: stopped.append(True) or "", lambda: (True, ""))
    assert outcome["ok"] and stopped and outcome["previous"] == "1.0.0"
    assert (root / "program" / "app" / "__init__.py").read_text(encoding="utf-8") == "new"
    assert not (root / "program.old").exists() and not (root / "program.new").exists()
    record = json.loads((root / updates.RELEASE).read_text(encoding="utf-8"))
    assert record == {"version": "99.0.0", **FINGERPRINTS}

def test_a_new_program_that_does_not_start_is_rolled_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _installed(tmp_path, monkeypatch)
    _staged(root)
    outcome = apply_update.apply(root, "99.0.0", lambda: "", lambda: (False, "ImportError: boom"))
    assert outcome["ok"] is False and outcome["reason"] == "ImportError: boom"
    assert (root / "program" / "app" / "__init__.py").read_text(encoding="utf-8") == "old"
    assert not any((root / name).exists() for name in ("program.old", "program.new", "program.failed"))
    assert json.loads((root / updates.RELEASE).read_text(encoding="utf-8"))["version"] == "1.0.0"

def test_a_program_still_in_use_is_left_as_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _installed(tmp_path, monkeypatch)
    _staged(root)
    real = apply_update.os.rename

    def held(source, target):
        if Path(source).name == "program":
            raise PermissionError("in use")
        return real(source, target)
    monkeypatch.setattr(apply_update.os, "rename", held)
    monkeypatch.setattr(apply_update, "RENAME_WAIT", 0)
    outcome = apply_update.apply(root, "99.0.0", lambda: "", lambda: pytest.fail("checked a program never moved in"))
    assert outcome["ok"] is False and "in use" in outcome["reason"]
    assert (root / "program" / "app" / "__init__.py").read_text(encoding="utf-8") == "old"

def test_an_update_cut_off_between_its_renames_puts_the_old_program_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _installed(tmp_path, monkeypatch)
    (root / "program").rename(root / "program.old")
    outcome = apply_update.apply(root, "99.0.0", lambda: "", lambda: (True, ""))
    assert outcome["reason"] == "nothing staged"
    assert (root / "program" / "app" / "__init__.py").read_text(encoding="utf-8") == "old"

def test_the_swap_needs_nothing_from_the_folder_it_replaces() -> None:
    """It runs with -I -S from a copy, so it may import the standard library only."""
    import ast

    tree = ast.parse(Path(apply_update.__file__).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {"app", "starlette", "fastmcp", "pydantic"}

# --- automatic updates ---------------------------------------------------------------------------

@pytest.fixture
def fresh_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(updates, "_auto", {"state": "off", "version": None, "error": None, "staged": None})

def test_with_automatic_updates_off_nothing_is_looked_for_or_downloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                        fresh_auto) -> None:
    _installed(tmp_path, monkeypatch)
    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("went online"))
    assert updates.auto_enabled() is False
    assert updates.tick(lambda: False, lambda: pytest.fail("updated")) == updates.CHECK_EVERY

def test_a_copy_not_from_the_installer_never_updates_itself(monkeypatch: pytest.MonkeyPatch, fresh_auto) -> None:
    monkeypatch.setattr(updates, "installed_by_setup", lambda: None)
    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("went online"))
    with pytest.raises(RuntimeError):
        updates.set_auto(True)
    assert updates.tick(lambda: False, lambda: pytest.fail("updated")) == updates.CHECK_EVERY
    with pytest.raises(RuntimeError):
        updates.prepare(_found(b"x"))

def test_automatic_updates_download_then_wait_for_the_render_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                     fresh_auto) -> None:
    _installed(tmp_path, monkeypatch)
    updates.set_auto(True)
    data = _program_zip("99.0.0")
    monkeypatch.setattr(updates, "latest", lambda force=False: _found(data))
    _serve(monkeypatch, {"https://example/program.zip": data})
    handed = []
    monkeypatch.setattr(updates, "hand_over", lambda staged, on_started: handed.append(staged["version"]))
    rendering = [True]
    assert updates.tick(lambda: rendering[0], lambda: None) == updates.IDLE_POLL
    assert updates._auto["state"] == "waiting" and handed == []
    rendering[0] = False
    updates.tick(lambda: rendering[0], lambda: None)
    assert handed == ["99.0.0"]

def test_the_editor_turns_automatic_updates_on_and_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.testclient import TestClient

    from app.ui import app as editor

    _installed(tmp_path, monkeypatch)
    monkeypatch.setattr(updates, "latest", lambda force=False: {"current": "1.0.0", "available": False, "managed": True})
    client = TestClient(editor.create_app())
    assert client.get("/api/update").json()["auto"] is False
    assert client.post("/api/update/settings", json={"auto": True}).status_code == 200
    assert client.get("/api/update").json()["auto"] is True
    assert client.post("/api/update/settings", json={"auto": "yes"}).status_code == 400

def test_the_last_outcome_is_said_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _installed(tmp_path, monkeypatch)
    (root / apply_update.RESULT).write_text(json.dumps({"ok": True, "version": "99.0.0", "announced": False}), encoding="utf-8")
    assert updates.last_result()["fresh"] is True
    assert updates.last_result()["fresh"] is False

def test_the_download_says_how_far_it_has_got(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _installed(tmp_path, monkeypatch)
    seen = []
    data = _program_zip("99.0.0")

    class Arriving(io.BytesIO):
        headers: dict = {}

        def read(self, size=-1):
            seen.append(updates.download_progress()["done"])
            return super().read(size)

    monkeypatch.setattr(updates, "CHUNK", 64)
    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *args, **kwargs: Arriving(data))
    updates.prepare(_found(data))
    assert seen[:3] == [0, 64, 128] and updates.download_progress() == {"done": len(data), "total": len(data)}

# --- what the build writes -----------------------------------------------------------------------

def _payload_tools():
    path = Path(__file__).resolve().parent.parent / "installer" / "payload_tools.py"
    spec = importlib.util.spec_from_file_location("payload_tools", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_the_runtime_fingerprint_changes_with_a_pin_and_not_with_a_comment() -> None:
    tools = _payload_tools()
    base = tools.runtime_fingerprint("# generated by uv\nav==16.0.1\n    # via clip-mcp\nnumpy==2.3.0\n", "3.14.5", True)
    assert tools.runtime_fingerprint("numpy==2.3.0\nav==16.0.1  # via x\n", "3.14.5", True) == base
    assert tools.runtime_fingerprint("av==16.0.1\nnumpy==2.3.1\n", "3.14.5", True) != base
    assert tools.runtime_fingerprint("av==16.0.1\nnumpy==2.3.0\n", "3.14.6", True) != base
    assert tools.runtime_fingerprint("av==16.0.1\nnumpy==2.3.0\n", "3.14.5", False) != base

def test_a_release_publishes_the_program_alone_with_checksums(tmp_path: Path) -> None:
    tools = _payload_tools()
    payload = tmp_path / "payload"
    for folder in ("program/app", "program/clip_mcp-2.0.0.dist-info", "bin", "ffmpeg"):
        (payload / folder).mkdir(parents=True)
    (payload / "program/app/__init__.py").write_text("", encoding="utf-8")
    (payload / "program/clip_mcp-2.0.0.dist-info/METADATA").write_text("Name: clip-mcp\nVersion: 2.0.0\n", encoding="utf-8")
    (payload / "bin/clip-mcp.exe").write_bytes(b"launcher")
    (payload / "ffmpeg/ffmpeg.exe").write_bytes(b"ffmpeg")
    (tmp_path / "requirements.txt").write_text("av==16.0.1\n", encoding="utf-8")
    record = tools.stamp(payload, tmp_path / "requirements.txt", "3.14.5", True)
    installer = tmp_path / "clip-mcp-setup-2.0.0.exe"
    installer.write_bytes(b"setup")
    manifest = tools.publish(payload, installer, tmp_path / "dist")
    program = tmp_path / "dist" / "clip-mcp-program-2.0.0.zip"
    assert manifest["assets"]["program"]["sha256"] == hashlib.sha256(program.read_bytes()).hexdigest()
    assert sorted(zipfile.ZipFile(program).namelist()) == ["app/__init__.py", "clip_mcp-2.0.0.dist-info/METADATA"]
    # What the editor compares: the same fingerprints on both sides choose the program alone.
    assert updates.choose(record, manifest) == ("program", [])
