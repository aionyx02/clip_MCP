"""The analysis scan decodes on the NVIDIA card where that is faster, and says so when the card fails."""

import json
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List, Optional

import pytest

from app.engine import analysis, machine
from app.engine import ffmpeg as engine_ffmpeg
from app.engine.ffmpeg import (
    CARD_MIN_BITRATE,
    CARD_MIN_SECONDS,
    FFmpegFailed,
    OperationCancelled,
    can_decode_on_card,
    card_decode_args,
    card_failure,
)
from app.models.media import Asset

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
LISTED = "Hardware acceleration methods:\ncuda\nvaapi\n"
CARD = ("NVIDIA GeForce RTX 4060", 8 * 1024 ** 3)

@pytest.fixture(autouse=True)
def fresh_card_answers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Forget what an earlier test learned about the card, start with the switch unset, and keep failures here."""
    monkeypatch.delenv("CLIP_MCP_HWACCEL", raising=False)
    monkeypatch.setattr(engine_ffmpeg, "card_failed_path", lambda: str(tmp_path / engine_ffmpeg.CARD_FAILED_FILE_NAME))
    engine_ffmpeg._card_listed.cache_clear()
    yield
    engine_ffmpeg._card_listed.cache_clear()

def _cards(monkeypatch: pytest.MonkeyPatch, listed: str = LISTED, card=CARD, codec: str = "hevc",
           bitrate: int = 55_000_000) -> List[List[str]]:
    """Make FFmpeg list these hardware methods, ffprobe report this picture, and the machine this card.

    Returns:
        Every command started, in order.
    """
    started: List[List[str]] = []

    def run(command, *args, **kwargs):
        started.append(list(command))
        if "-hwaccels" in command:
            return SimpleNamespace(stdout=listed, returncode=0)
        found = {"streams": [{"codec_name": codec}], "format": {"bit_rate": str(bitrate)}}
        return SimpleNamespace(stdout=json.dumps(found), returncode=0)

    monkeypatch.setattr(engine_ffmpeg.subprocess, "run", run)
    monkeypatch.setattr(machine, "_nvidia_card", lambda: card)
    return started

def test_a_card_ffmpeg_lists_cuda_for_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch)
    assert can_decode_on_card("ffmpeg") is True
    assert card_decode_args("ffmpeg", "clip.mp4", 50.0) == ["-hwaccel", "cuda"]

def test_cuda_without_a_card_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch, card=None)
    assert can_decode_on_card("ffmpeg") is False

def test_a_card_that_cannot_be_asked_about_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch)

    def broken():
        raise RuntimeError("nvidia-smi crashed")

    monkeypatch.setattr(machine, "_nvidia_card", broken)
    assert can_decode_on_card("ffmpeg") is False

def test_an_ffmpeg_without_cuda_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch, listed="Hardware acceleration methods:\nvaapi\nvdpau\n")
    assert can_decode_on_card("ffmpeg") is False
    assert card_decode_args("ffmpeg", "clip.mp4", 50.0) == []

@pytest.mark.parametrize("failure", [OSError("ffmpeg is not there"), subprocess.TimeoutExpired("ffmpeg", 30)])
def test_an_ffmpeg_that_will_not_list_its_methods_is_not_used(
    monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    def broken(*args, **kwargs):
        raise failure

    monkeypatch.setattr(engine_ffmpeg.subprocess, "run", broken)
    monkeypatch.setattr(machine, "_nvidia_card", lambda: CARD)
    assert can_decode_on_card("ffmpeg") is False

def test_what_the_card_offers_is_asked_once(monkeypatch: pytest.MonkeyPatch) -> None:
    started = _cards(monkeypatch)
    assert can_decode_on_card("ffmpeg") is True
    assert can_decode_on_card("ffmpeg") is True
    assert len(started) == 1

def test_the_switch_turns_the_card_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch)
    monkeypatch.setenv("CLIP_MCP_HWACCEL", "OFF")
    assert can_decode_on_card("ffmpeg") is False
    assert card_decode_args("ffmpeg", "clip.mp4", 50.0) == []

def test_the_switch_is_read_even_after_the_card_was_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch)
    assert can_decode_on_card("ffmpeg") is True
    monkeypatch.setenv("CLIP_MCP_HWACCEL", "off")
    assert can_decode_on_card("ffmpeg") is False

@pytest.mark.parametrize("codec, bitrate, seconds, used", [
    ("hevc", CARD_MIN_BITRATE, 50.0, True),
    ("hevc", CARD_MIN_BITRATE - 1, 50.0, False),
    ("h264", 218_000_000, 50.0, False),
    ("hevc", 55_000_000, CARD_MIN_SECONDS, True),
])
def test_only_costly_pictures_are_decoded_on_the_card(
    monkeypatch: pytest.MonkeyPatch, codec: str, bitrate: int, seconds: float, used: bool,
) -> None:
    _cards(monkeypatch, codec=codec, bitrate=bitrate)
    assert card_decode_args("ffmpeg", "clip.mp4", seconds) == (["-hwaccel", "cuda"] if used else [])

def test_a_short_file_is_not_decoded_on_the_card_and_costs_no_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    started = _cards(monkeypatch)
    assert card_decode_args("ffmpeg", "clip.mp4", CARD_MIN_SECONDS - 0.1) == []
    assert started == []

def test_a_probe_that_fails_means_no_card(monkeypatch: pytest.MonkeyPatch) -> None:
    _cards(monkeypatch)
    assert can_decode_on_card("ffmpeg") is True

    def broken(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 30)

    monkeypatch.setattr(engine_ffmpeg.subprocess, "run", broken)
    assert card_decode_args("ffmpeg", "clip.mp4", 50.0) == []

def test_ffprobe_is_taken_from_beside_an_ffmpeg_given_as_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    started = _cards(monkeypatch)
    card_decode_args(str(Path("tools") / "ffmpeg"), "clip.mp4", 50.0)
    assert started[-1][0] == str(Path("tools") / "ffprobe")

def _asset(media: Path, **fields) -> Asset:
    """A small real file the scan can decode."""
    found = dict(id="card", path=str(media / "silent.mp4"), duration=Decimal("4"), has_video=True, has_audio=False)
    found.update(fields)
    return Asset(**found)

def _scans(monkeypatch: pytest.MonkeyPatch, card: List[str], fail_on_card: bool = False) -> List[List[str]]:
    """Run the scan's FFmpeg for real, recording each command, failing on the card when asked.

    Returns:
        Every command run, in order.
    """
    calls: List[List[str]] = []
    real = analysis.run_ffmpeg

    def run(command, log_path, duration, on_progress, is_cancelled):
        calls.append(list(command))
        if fail_on_card and "-hwaccel" in command:
            with open(log_path, "w", encoding="utf-8") as log:
                log.write("cuda: out of memory\n")
            raise FFmpegFailed("decoding failed\nCUDA_ERROR_OUT_OF_MEMORY")
        real(command, log_path, duration, on_progress, is_cancelled)

    monkeypatch.setattr(analysis, "run_ffmpeg", run)
    monkeypatch.setattr(analysis, "card_decode_args", lambda ffmpeg_bin, path, duration: card)
    return calls

def test_the_scan_tries_the_card_and_falls_back_to_software(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Path
) -> None:
    calls = _scans(monkeypatch, ["-hwaccel", "cuda"], fail_on_card=True)
    found = analysis.scan_media(_asset(media), str(tmp_path), lambda fraction: None, lambda: False, FFMPEG)
    assert len(calls) == 2
    assert "-hwaccel" in calls[0] and "-hwaccel" not in calls[1]
    assert calls[0].index("-hwaccel") < calls[0].index("-i")
    assert found.scenes, "the software pass should have measured the file"

def test_a_card_that_failed_is_recorded_and_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Path
) -> None:
    _scans(monkeypatch, ["-hwaccel", "cuda"], fail_on_card=True)
    work = tmp_path / "work"
    work.mkdir()
    analysis.scan_media(_asset(media), str(work), lambda fraction: None, lambda: False, FFMPEG)
    kept = work / (analysis.DETECT_LOG_FILE_NAME + ".card")
    assert kept.read_text(encoding="utf-8") == "cuda: out of memory\n"
    assert card_failure() == f"{media / 'silent.mp4'}: CUDA_ERROR_OUT_OF_MEMORY"
    _cards(monkeypatch)
    assert can_decode_on_card("ffmpeg") is False

def test_a_scan_that_fails_without_the_card_runs_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Path
) -> None:
    calls: List[List[str]] = []

    def failing(command, log_path, duration, on_progress, is_cancelled):
        calls.append(list(command))
        raise FFmpegFailed("the file is broken")

    monkeypatch.setattr(analysis, "run_ffmpeg", failing)
    monkeypatch.setattr(analysis, "card_decode_args", lambda ffmpeg_bin, path, duration: [])
    with pytest.raises(FFmpegFailed):
        analysis.scan_media(_asset(media), str(tmp_path), lambda fraction: None, lambda: False, FFMPEG)
    assert len(calls) == 1
    assert card_failure() is None

def test_cancellation_is_never_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Path) -> None:
    calls: List[List[str]] = []

    def cancelled(command, log_path, duration, on_progress, is_cancelled):
        calls.append(list(command))
        raise OperationCancelled()

    monkeypatch.setattr(analysis, "run_ffmpeg", cancelled)
    monkeypatch.setattr(analysis, "card_decode_args", lambda ffmpeg_bin, path, duration: ["-hwaccel", "cuda"])
    with pytest.raises(OperationCancelled):
        analysis.scan_media(_asset(media), str(tmp_path), lambda fraction: None, lambda: False, FFMPEG)
    assert len(calls) == 1

def test_without_a_card_the_scan_runs_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Path) -> None:
    calls = _scans(monkeypatch, [])
    analysis.scan_media(_asset(media), str(tmp_path), lambda fraction: None, lambda: False, FFMPEG)
    assert len(calls) == 1
    assert "-hwaccel" not in calls[0]

@pytest.mark.parametrize("fields", [{"has_video": False, "has_audio": True}, {"still": True}])
def test_sound_alone_or_a_photo_never_asks_for_the_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Path, fields: dict,
) -> None:
    asked: List[str] = []
    monkeypatch.setattr(analysis, "run_ffmpeg", lambda *args: None)
    monkeypatch.setattr(analysis, "card_decode_args", lambda ffmpeg_bin, path, duration: asked.append(path) or [])
    try:
        analysis.scan_media(_asset(media, **fields), str(tmp_path), lambda fraction: None, lambda: False, FFMPEG)
    except OSError:
        pass  # No log was written by the stand-in; what matters is that the card was never asked about.
    assert asked == []

def _hevc_clip(folder: Path) -> Optional[Path]:
    """A 3 s 10-bit HEVC clip with a cut and a black stretch, or None when this FFmpeg cannot make one."""
    path = folder / "hevc.mp4"
    result = subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1.5",
        "-f", "lavfi", "-i", "color=c=black:size=320x240:rate=30:duration=0.5",
        "-f", "lavfi", "-i", "mandelbrot=size=320x240:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-filter_complex", "[2:v]trim=duration=1,setpts=PTS-STARTPTS[m];[0:v][1:v][m]concat=n=3:v=1[v]",
        "-map", "[v]", "-map", "3:a", "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error",
        "-c:a", "aac", str(path),
    ], capture_output=True)
    return path if result.returncode == 0 else None

def test_the_card_finds_what_software_finds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not can_decode_on_card(FFMPEG):
        pytest.skip("no NVIDIA card FFmpeg can decode on")
    clip = _hevc_clip(tmp_path)
    if clip is None:
        pytest.skip("this FFmpeg cannot encode 10-bit HEVC")
    asset = Asset(id="hevc", path=str(clip), duration=Decimal("3"), has_video=True, has_audio=True)
    found = {}
    for name, card in (("software", []), ("card", ["-hwaccel", "cuda"])):
        monkeypatch.setattr(analysis, "card_decode_args", lambda ffmpeg_bin, path, duration, card=card: card)
        work = tmp_path / name
        work.mkdir()
        scan = analysis.scan_media(asset, str(work), lambda fraction: None, lambda: False, FFMPEG)
        files = {path.name: path.read_bytes() for path in work.iterdir()
                 if path.is_file() and path.name != analysis.DETECT_LOG_FILE_NAME}
        found[name] = (scan, files)
    assert found["card"][0] == found["software"][0]
    assert found["card"][1] == found["software"][1]
