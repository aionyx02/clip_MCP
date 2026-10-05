"""Tests for telling how fast this computer will be at the local models, and saying so."""

import subprocess
from types import SimpleNamespace

import pytest

from app.engine import machine, resources

GIB = 1024 ** 3
HOUR = {"transcription_minutes": [10, 40], "basis": "rough, from the hardware; not yet measured on this computer"}

def computer(**changes) -> machine.Machine:
    """A computer with nothing special about it, changed where a test says."""
    base = dict(gpu=None, gpu_memory_bytes=None, memory_bytes=32 * GIB, logical_cores=16,
                apple_silicon=False, mac=False, identity="desk / Some CPU")
    base.update(changes)
    return machine.Machine(tier=machine.classify(**{k: v for k, v in base.items() if k != "identity"}), **base)

# --- tiers -------------------------------------------------------------------------------

def test_an_nvidia_card_with_room_for_the_model_is_high() -> None:
    assert computer(gpu="RTX 4060", gpu_memory_bytes=8188 * 1024 ** 2).tier == "high"

def test_a_4_gb_card_counts_even_though_it_reports_a_little_under() -> None:
    assert computer(gpu="GTX 1650", gpu_memory_bytes=3911 * 1024 ** 2).tier == "high"

def test_a_small_card_is_judged_by_the_processor_instead() -> None:
    assert computer(gpu="MX150", gpu_memory_bytes=2 * GIB).tier == "mid"
    assert computer(gpu="MX150", gpu_memory_bytes=2 * GIB, memory_bytes=8 * GIB).tier == "low"

def test_a_card_whose_memory_cannot_be_read_is_not_trusted() -> None:
    assert computer(gpu="NVIDIA GPU", gpu_memory_bytes=None, memory_bytes=8 * GIB).tier == "low"

def test_a_16_gb_computer_counts_though_windows_reports_less() -> None:
    assert computer(memory_bytes=int(15.7 * GIB), logical_cores=8).tier == "mid"

def test_few_cores_or_little_memory_is_low() -> None:
    assert computer(logical_cores=4).tier == "low"
    assert computer(memory_bytes=8 * GIB).tier == "low"
    assert computer(memory_bytes=None).tier == "low"

def test_apple_silicon_is_mid_with_16_gb_and_low_with_8() -> None:
    assert computer(mac=True, apple_silicon=True, memory_bytes=16 * GIB, logical_cores=8).tier == "mid"
    assert computer(mac=True, apple_silicon=True, memory_bytes=8 * GIB, logical_cores=8).tier == "low"

def test_every_intel_mac_is_low() -> None:
    assert computer(mac=True, memory_bytes=64 * GIB, logical_cores=16).tier == "low"

# --- what the AI is told -----------------------------------------------------------------

def test_the_ai_is_told_a_mac_transcribes_on_the_processor_only() -> None:
    told = machine.describe(computer(mac=True, apple_silicon=True, memory_bytes=16 * GIB, logical_cores=8), HOUR)
    assert "mid" in told and "processor only" in told and "Apple" in told

def test_the_ai_is_told_other_graphics_do_not_help() -> None:
    told = machine.describe(computer(memory_bytes=8 * GIB), HOUR)
    assert "low" in told and "AMD" in told and "rough" in told

def test_the_ai_is_told_which_card_is_used() -> None:
    told = machine.describe(computer(gpu="NVIDIA GeForce RTX 4060", gpu_memory_bytes=8 * GIB), HOUR)
    assert "high" in told and "RTX 4060" in told

# --- looking at the hardware -------------------------------------------------------------

def test_the_card_is_read_from_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(arguments, **keywords):
        assert keywords["creationflags"] == machine.hidden_window_flags()
        return SimpleNamespace(returncode=0, stdout="NVIDIA GeForce RTX 4060 Laptop GPU, 8188\n")

    monkeypatch.setattr(machine.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(machine.subprocess, "run", run)
    assert machine._nvidia_card() == ("NVIDIA GeForce RTX 4060 Laptop GPU", 8188 * 1024 ** 2)

def test_no_nvidia_smi_means_no_card_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(machine.shutil, "which", lambda name: None)
    assert machine._nvidia_card() is None

def test_a_hanging_nvidia_smi_means_no_card_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(arguments, **keywords):
        raise subprocess.TimeoutExpired(arguments, 10)

    monkeypatch.setattr(machine.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(machine.subprocess, "run", run)
    assert machine._nvidia_card() is None

def test_asking_for_the_processor_keeps_the_card_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIP_MCP_WHISPER_DEVICE", "cpu")
    assert machine._usable_card() is None

def test_this_computer_can_be_profiled() -> None:
    found = machine.profile()
    assert found.tier in ("high", "mid", "low") and found.logical_cores >= 1 and found.identity

# --- memory on a Mac ---------------------------------------------------------------------

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10000.
Pages active:                            300000.
Pages inactive:                          200000.
Pages speculative:                         5000.
Pages wired down:                        100000.
"""

def test_a_mac_counts_free_inactive_and_speculative_pages_as_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "sysctl_int", lambda name: 16 * GIB if name == "hw.memsize" else None)
    monkeypatch.setattr(resources.subprocess, "run",
                        lambda arguments, **keywords: SimpleNamespace(returncode=0, stdout=VM_STAT))
    assert resources._mac_memory() == ((10000 + 200000 + 5000) * 16384, 16 * GIB)

def test_a_mac_whose_vm_stat_fails_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "sysctl_int", lambda name: 16 * GIB)
    monkeypatch.setattr(resources.subprocess, "run",
                        lambda arguments, **keywords: SimpleNamespace(returncode=1, stdout=""))
    assert resources._mac_memory() is None

def test_the_ai_hears_the_tier_as_soon_as_it_connects() -> None:
    from app import server

    here = machine.profile()
    assert f"speed tier for the local models is {here.tier}" in server.mcp.instructions
    assert "dry_run" in server.mcp.instructions
