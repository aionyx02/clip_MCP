"""Tests for finding cuBLAS for speech recognition on the card, and the GPU pack that brings it."""

import dataclasses
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from app import gpu
from app.engine import machine
from app.ui import updates

def libraries(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in gpu.LIBRARIES:
        (folder / name).write_bytes(b"")
    return folder

@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An installation folder, no system CUDA, and nothing loaded for real."""
    root = tmp_path / "Programs" / "clip-mcp"
    root.mkdir(parents=True)
    monkeypatch.setattr(updates, "installed_by_setup", lambda: root)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    for name in [name for name in gpu.os.environ if name.upper().startswith("CUDA_PATH")]:
        monkeypatch.delenv(name)
    monkeypatch.delenv("CLIP_MCP_WHISPER_DEVICE", raising=False)
    # Looked for as on Windows, and never loaded for real.
    monkeypatch.setattr(gpu, "_on_windows", lambda: True)
    monkeypatch.setattr(gpu, "_load", lambda folder: None)
    gpu.use.cache_clear()
    yield root
    gpu.use.cache_clear()

def test_the_bundled_pack_comes_before_the_systems(installed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    system = libraries(tmp_path / "toolkit" / "bin")
    monkeypatch.setenv("PATH", str(system))
    assert gpu.find() == gpu.Libraries("system", system)
    libraries(installed / "gpu")
    assert gpu.find() == gpu.Libraries("bundled", installed / "gpu")

def test_a_toolkit_named_only_by_cuda_path_is_found(installed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    libraries(tmp_path / "v12.6" / "bin")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "v12.6"))
    assert gpu.find().source == "system"

def test_cublas_that_will_not_load_is_no_cublas(installed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(folder: Path) -> None:
        raise OSError("not a valid Win32 application")

    monkeypatch.setattr(gpu, "_load", broken)
    monkeypatch.setattr(gpu, "card_count", lambda: 1)
    libraries(installed / "gpu")
    assert machine.whisper_device() == "cpu"
    assert gpu.status()["why"] == "wont_load"

def test_half_a_pack_is_no_pack(installed: Path) -> None:
    (installed / "gpu").mkdir()
    (installed / "gpu" / "cublas64_12.dll").write_bytes(b"")
    assert gpu.find() == gpu.Libraries("none", None)

def test_a_card_without_cublas_runs_on_the_processor_and_says_so(installed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "card_count", lambda: 1)
    assert machine.whisper_device() == "cpu"
    said = gpu.status()
    assert said["device"] == "cpu" and said["libraries"] == "none" and said["card"]
    assert "GPU pack" in said["reason"] and said["can_install"] and not said["pack_installed"]

def test_a_card_with_the_pack_runs_on_the_card(installed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "card_count", lambda: 1)
    libraries(installed / "gpu")
    assert machine.whisper_device() == "cuda"
    said = gpu.status()
    assert said["device"] == "cuda" and said["libraries"] == "bundled" and "GPU pack" in said["reason"]

def test_no_card_is_the_processor_whatever_libraries_are_here(installed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "card_count", lambda: 0)
    libraries(installed / "gpu")
    assert machine.whisper_device() == "cpu"
    said = gpu.status()
    assert said["device"] == "cpu" and "no NVIDIA card" in said["reason"] and not said["can_install"]

def test_a_copy_from_source_cannot_take_the_pack(installed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(updates, "installed_by_setup", lambda: None)
    with pytest.raises(RuntimeError, match="installed by the installer"):
        gpu.install_pack()

def a_wheel() -> bytes:
    built = io.BytesIO()
    with zipfile.ZipFile(built, "w") as archive:
        for member in gpu.PACK.members:
            archive.writestr(member, b"library " + member.encode())
        archive.writestr("nvidia/cublas/bin/nvblas64_12.dll", b"not wanted")
    return built.getvalue()

class Response(io.BytesIO):
    status = 200

def test_the_pack_keeps_only_the_libraries_from_a_checked_download(installed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = a_wheel()
    monkeypatch.setattr(gpu, "PACK", dataclasses.replace(gpu.PACK, sha256=hashlib.sha256(wheel).hexdigest(), size=len(wheel)))
    monkeypatch.setattr(gpu.urllib.request, "urlopen", lambda *args, **kwargs: Response(wheel))
    seen = []
    folder = gpu.install_pack(lambda done, total: seen.append((done, total)))
    assert folder == installed / "gpu"
    assert sorted(path.name for path in folder.iterdir()) == sorted([*gpu.LIBRARIES, gpu.STAMP])
    assert json.loads((folder / gpu.STAMP).read_text(encoding="utf-8"))["sha256"] == gpu.PACK.sha256
    assert seen[-1] == (len(wheel), len(wheel))
    assert not (installed / "gpu.download").exists() and not (installed / "gpu.new").exists()
    # A second time, nothing is downloaded.
    monkeypatch.setattr(gpu.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("downloaded again"))
    assert gpu.install_pack() == folder

def test_a_download_that_does_not_match_its_checksum_is_refused_and_deleted(installed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = a_wheel()
    monkeypatch.setattr(gpu, "PACK", dataclasses.replace(gpu.PACK, size=len(wheel)))
    monkeypatch.setattr(gpu.urllib.request, "urlopen", lambda *args, **kwargs: Response(wheel))
    with pytest.raises(RuntimeError, match="checksum"):
        gpu.install_pack()
    assert not (installed / "gpu").exists()
    assert not (installed / "gpu.download" / gpu.PACK.name).exists()

def test_check_says_where_speech_recognition_runs_and_why(installed: Path, monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture) -> None:
    from app import cli

    monkeypatch.setattr(gpu, "card_count", lambda: 1)
    monkeypatch.setattr(cli.clients, "register", lambda *args, **kwargs: [])
    cli.check()
    printed = capsys.readouterr().out
    assert "speech:     processor (cpu)" in printed and "cuBLAS 12 is not" in printed

def test_a_failure_on_the_card_is_said_not_silent(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from app.engine import analysis

    monkeypatch.setattr(machine, "whisper_device", lambda: "cuda")
    monkeypatch.setattr(gpu, "last_failure", None)

    def run(device: str) -> str:
        if device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return device

    assert analysis.on_whisper_device(run) == "cpu"
    assert "cublas64_12.dll" in gpu.last_failure and "fell back" in capsys.readouterr().err

def test_the_editor_says_where_speech_recognition_runs_and_refuses_a_pack_it_cannot_use(
    installed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.testclient import TestClient

    from app.ui import app as editor

    monkeypatch.setattr(gpu, "card_count", lambda: 0)
    client = TestClient(editor.create_app())
    said = client.get("/api/gpu").json()
    assert said["device"] == "cpu" and said["why"] == "no_card" and not said["install"]["running"]
    refused = client.post("/api/gpu", json={})
    assert refused.status_code == 400 and "no NVIDIA card" in refused.json()["error"]

def test_a_pack_downloaded_while_running_is_used_without_a_restart(installed: Path) -> None:
    assert gpu.use().source == "none"
    libraries(installed / "gpu")
    assert gpu.use().source == "bundled"

def test_an_update_never_takes_out_a_pack_the_user_fetched_later() -> None:
    script = (Path(__file__).parent.parent / "installer" / "install.ps1").read_text(encoding="utf-8")
    assert '"gpu", "remove"' not in script
