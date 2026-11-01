"""The sound a benchmark judge reads about a film."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from app.engine.sound_quality import RATE

spec = importlib.util.spec_from_file_location(
    "benchmark_evidence", Path(__file__).parent.parent / "benchmark" / "evidence.py")
evidence = importlib.util.module_from_spec(spec)
sys.modules["benchmark_evidence"] = evidence
spec.loader.exec_module(evidence)


def tone(seconds: float, amplitude: float) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * 440 * np.arange(round(seconds * RATE)) / RATE)


def test_a_cut_is_reported_only_when_the_level_jumps():
    film = np.concatenate([tone(2, 0.5), tone(2, 0.05), tone(2, 0.05)])

    jumps = evidence.sound_jumps(film, [2.0, 4.0])

    assert [jump["at"] for jump in jumps] == [2.0]
    assert jumps[0]["before_dbfs"] - jumps[0]["after_dbfs"] == 20.0


def test_a_cut_at_the_very_end_is_compared_with_nothing_after_it():
    jumps = evidence.sound_jumps(tone(2, 0.5), [2.0])

    assert jumps[0]["after_dbfs"] == evidence.LEVEL_FLOOR_DBFS


def test_the_sound_table_has_a_row_per_window_and_a_short_last_row():
    rows = evidence.sound_table(tone(25, 0.1))

    assert [row["start"] for row in rows] == [0, 10, 20]
    assert rows[0]["level_dbfs"] == -23.0


def test_silence_still_running_at_the_end_reaches_the_end_of_the_film():
    log = "silence_start: 1.5\nsilence_end: 3 | silence_duration: 1.5\nsilence_start: 9\n"

    assert evidence.silences(log, 12.0) == [[1.5, 3.0], [9.0, 12.0]]
