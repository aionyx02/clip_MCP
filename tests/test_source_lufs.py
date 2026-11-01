"""Integrated loudness of any range of a source file, gathered in the scan pass (#172)."""

import subprocess
from pathlib import Path
from typing import List

import pytest

from app.engine.analysis import _frame_records, _scan_graph, _sound_measurements, integrated_lufs, sound_note
from app.engine.probe import speech_loudness
from app.models.media import Asset, SoundMeasurement

TOLERANCE_LU = 0.3


def _tones(path: Path, channels: int) -> None:
    """Ten seconds of a -20 dBFS tone, then ten of a -30 dBFS one."""
    tone = "volume=enable='lt(t,10)':volume=0.1,volume=enable='gte(t,10)':volume=0.0316"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:d=20",
                    "-af", tone, "-ac", str(channels), "-c:a", "pcm_s16le", str(path)], check=True)


def _scanned(path: Path, work: Path) -> List[SoundMeasurement]:
    """What the scan's graph measures of a file, through the real filters."""
    asset = Asset(id="a", path=str(path), duration=20, has_video=False, has_audio=True)
    graph, outputs = _scan_graph(asset, str(work), False)
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-filter_complex", graph]
    for output in outputs:
        command += output
    subprocess.run(command, check=True)
    return _sound_measurements(_frame_records(str(work / "sound.txt")), 20.0,
                               _frame_records(str(work / "loudness.txt")))


@pytest.mark.parametrize("channels", [1, 2, 8])
def test_the_whole_file_reads_as_a_full_measurement_of_it_does(tmp_path: Path, channels: int) -> None:
    path = tmp_path / "tones.wav"
    _tones(path, channels)
    windows = _scanned(path, tmp_path)
    assert abs(integrated_lufs(windows) - speech_loudness(str(path), 0, 20)) <= TOLERANCE_LU


@pytest.mark.parametrize("channels", [1, 2])
def test_a_range_over_the_loud_half_reads_as_a_measurement_of_that_range_does(tmp_path: Path, channels: int) -> None:
    path = tmp_path / "tones.wav"
    _tones(path, channels)
    windows = [window for window in _scanned(path, tmp_path) if window.end <= 10]
    assert abs(integrated_lufs(windows) - speech_loudness(str(path), 0, 10)) <= TOLERANCE_LU


def test_the_quiet_half_is_gated_out_of_the_whole_but_measured_alone(tmp_path: Path) -> None:
    path = tmp_path / "tones.wav"
    _tones(path, 2)
    windows = _scanned(path, tmp_path)
    assert integrated_lufs(windows) > integrated_lufs([w for w in windows if w.start >= 11]) + 5


def test_silence_and_an_older_analysis_have_no_integrated_loudness() -> None:
    old = SoundMeasurement(start=0, end=1, loudness=-30, peak=-20, noise_floor=-60, flatness=0)
    assert integrated_lufs([old]) is None
    assert sound_note([old])["integrated_lufs"] is None


def test_blocks_go_in_the_window_they_fall_in_and_blocks_under_the_gate_are_left_out() -> None:
    records = [(0.0, {}), (1.0, {})]
    blocks = [(0.5, {"lavfi.r128.M": -20.04}), (1.2, {"lavfi.r128.M": -80.0}), (1.3, {"lavfi.r128.M": -15.0})]
    first, second = _sound_measurements(records, 2.0, blocks)
    assert first.momentary == [-20.0]
    assert second.momentary == [-15.0]
