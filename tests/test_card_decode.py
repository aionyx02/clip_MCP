"""The analysis scan decodes on the NVIDIA card when there is one."""

import shutil
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from app.engine import analysis, machine
from app.engine import ffmpeg as engine_ffmpeg
from app.engine.ffmpeg import (
    FFmpegFailed,
    OperationCancelled,
    can_decode_on_card,
    card_decode_args,
)
from app.models.media import Asset

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

@pytest.fixture(autouse=True)
def fresh_card_answers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Forget what an earlier test learned about the card, and start with the switch unset."""
    monkeypatch.delenv("CLIP_MCP_HWACCEL", raising=False)
    engine_ffmpeg._card_listed.cache_clear()
    yield
    engine_ffmpeg._card_listed.cache_clear()

def _cards(monkeypatch: pytest.MonkeyPatch, listed: str, card=("NVIDIA GeForce RTX 4060", 8 * 1024 ** 3)) -> None:
    """Make FFmpeg list these hardware methods and the machine report this card."""
    monkeypatch.setattr(engine_ffmpeg.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout=listed, returncode=0))
    monkeypatch.setattr(machine, "_nvidia_card", lambda: card)

def test_a_card_ffmpeg_lists_cuda_for_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch, "Hardware acceleration methods:\ncuda\nvaapi\n")
    assert can_decode_on_card("ffmpeg") is True
    assert card_decode_args("ffmpeg") == ["-hwaccel", "cuda"]

def test_cuda_without_a_card_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch, "Hardware acceleration methods:\ncuda\nvaapi\n", card=None)
    assert can_decode_on_card("ffmpeg") is False

def test_an_ffmpeg_without_cuda_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch, "Hardware acceleration methods:\nvaapi\nvdpau\n")
    assert can_decode_on_card("ffmpeg") is False
    assert card_decode_args("ffmpeg") == []

def test_an_ffmpeg_that_will_not_list_its_methods_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args, **kwargs):
        raise OSError("ffmpeg is not there")

    monkeypatch.setattr(engine_ffmpeg.subprocess, "run", broken)
    monkeypatch.setattr(machine, "_nvidia_card", lambda: ("NVIDIA GeForce RTX 4060", 8 * 1024 ** 3))
    assert can_decode_on_card("ffmpeg") is False

def test_the_switch_turns_the_card_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch, "Hardware acceleration methods:\ncuda\n")
    monkeypatch.setenv("CLIP_MCP_HWACCEL", "OFF")
    assert can_decode_on_card("ffmpeg") is False
    assert card_decode_args("ffmpeg") == []

def test_the_switch_is_read_even_after_the_card_was_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch, "Hardware acceleration methods:\ncuda\n")
    assert can_decode_on_card("ffmpeg") is True
    monkeypatch.setenv("CLIP_MCP_HWACCEL", "off")
    assert can_decode_on_card("ffmpeg") is False

def _asset(media: Path) -> Asset:
    """A small real file the scan can decode."""
    return Asset(id="card", path=str(media / "silent.mp4"), duration=Decimal("4"), has_video=True, has_audio=False)

def test_the_scan_tries_the_card_and_falls_back_to_software(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Path
) -> None:
    calls = []
    real = analysis.run_ffmpeg

    def failing_on_card(command, log_path, duration, on_progress, is_cancelled):
        calls.append(list(command))
        if "-hwaccel" in command:
            raise FFmpegFailed("the card ran out of memory")
        real(command, log_path, duration, on_progress, is_cancelled)

    monkeypatch.setattr(analysis, "run_ffmpeg", failing_on_card)
    monkeypatch.setattr(analysis, "card_decode_args", lambda ffmpeg_bin="ffmpeg": ["-hwaccel", "cuda"])
    found = analysis.scan_media(_asset(media), str(tmp_path), lambda fraction: None, lambda: False, FFMPEG)
    assert len(calls) == 2
    assert "-hwaccel" in calls[0] and "-hwaccel" not in calls[1]
    assert calls[0].index("-hwaccel") < calls[0].index("-i")
    assert found.scenes, "the software pass should have measured the file"

def test_cancellation_is_never_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Path) -> None:
    calls = []

    def cancelled(command, log_path, duration, on_progress, is_cancelled):
        calls.append(list(command))
        raise OperationCancelled()

    monkeypatch.setattr(analysis, "run_ffmpeg", cancelled)
    monkeypatch.setattr(analysis, "card_decode_args", lambda ffmpeg_bin="ffmpeg": ["-hwaccel", "cuda"])
    with pytest.raises(OperationCancelled):
        analysis.scan_media(_asset(media), str(tmp_path), lambda fraction: None, lambda: False, FFMPEG)
    assert len(calls) == 1

def test_without_a_card_the_scan_runs_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Path
) -> None:
    calls = []
    real = analysis.run_ffmpeg

    def record(command, log_path, duration, on_progress, is_cancelled):
        calls.append(list(command))
        real(command, log_path, duration, on_progress, is_cancelled)

    monkeypatch.setattr(analysis, "run_ffmpeg", record)
    monkeypatch.setattr(analysis, "card_decode_args", lambda ffmpeg_bin="ffmpeg": [])
    analysis.scan_media(_asset(media), str(tmp_path), lambda fraction: None, lambda: False, FFMPEG)
    assert len(calls) == 1
    assert "-hwaccel" not in calls[0]
