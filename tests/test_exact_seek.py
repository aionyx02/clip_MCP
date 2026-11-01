"""A stretch of sound that must be exact starts where the source says, on any AAC (#171)."""

import subprocess
import wave
from pathlib import Path

import numpy
import pytest

from app.engine.ffmpeg import SEEK_LEAD_SECONDS, exact_stretch_args
from app.engine.listen import cut_sound

RATE = 16000
CLICK_AT = 2.5
TOLERANCE_SAMPLES = 2


def _samples(path: Path) -> numpy.ndarray:
    with wave.open(str(path), "rb") as read:
        return numpy.frombuffer(read.readframes(read.getnframes()), dtype=numpy.int16).astype(numpy.float64)


def _onset(samples: numpy.ndarray) -> int:
    """Where the click first reaches half its height; its flat top has no one loudest sample."""
    level = numpy.abs(samples)
    return int(numpy.argmax(level >= level.max() / 2))


@pytest.fixture
def clicked(tmp_path: Path) -> Path:
    """Five seconds of quiet at 22.05 kHz AAC with one sharp click in it."""
    rate = 22050
    wav = numpy.random.default_rng(1).normal(0, 20, rate * 5)
    at = round(CLICK_AT * rate)
    wav[at:at + 30] = 30000
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(wav.astype(numpy.int16).tobytes())
    path = tmp_path / "clicked.m4a"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-c:a", "aac", "-b:a", "128k", str(path)],
                   check=True)
    return path


@pytest.mark.parametrize("start", [0.0, 0.4, 1.0, 1.37, 2.03])
def test_a_stretch_heard_again_has_the_click_where_the_whole_file_has_it(
    clicked: Path, tmp_path: Path, start: float,
) -> None:
    whole = tmp_path / "whole.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clicked), "-ac", "1", "-ar", str(RATE), str(whole)],
                   check=True)
    piece = tmp_path / "piece.wav"
    cut_sound(str(clicked), start, start + 2.6, str(piece))
    expected = _onset(_samples(whole)) - round(start * RATE)
    heard = _samples(piece)
    assert abs(_onset(heard) - expected) <= TOLERANCE_SAMPLES
    assert len(heard) == round(2.6 * RATE)


def test_a_stretch_near_the_start_is_not_sought_in_the_file_at_all() -> None:
    args = exact_stretch_args("a.m4a", SEEK_LEAD_SECONDS / 2, 2.0)
    assert args[:2] == ["-i", "a.m4a"]
    assert args[2:] == ["-ss", f"{SEEK_LEAD_SECONDS / 2:.3f}", "-t", f"{2.0 - SEEK_LEAD_SECONDS / 2:.3f}"]


def test_a_later_stretch_seeks_a_lead_early_and_cuts_after_decoding() -> None:
    args = exact_stretch_args("a.m4a", 10.0, 12.5)
    assert args == ["-ss", f"{10.0 - SEEK_LEAD_SECONDS:.3f}", "-i", "a.m4a",
                    "-ss", f"{SEEK_LEAD_SECONDS:.3f}", "-t", "2.500"]
