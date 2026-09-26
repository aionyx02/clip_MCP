"""Tests for finding the beat in music.

Every signal here is synthesized, so where the beat is is known exactly rather
than annotated by ear. The tolerance is a frame of 60fps video, about 17ms,
because that is the finest a cut can land anyway.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.engine.analysis import analyze_media, current_recipe
from app.engine.rhythm import measure_rhythm, rhythm_model_name
from app.models.media import Asset

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FRAME_AT_60 = 1 / 60

def synthesize(folder: Path, name: str, expression: str, seconds: int = 20) -> Path:
    """Write a sound built from an expression, the way music is delivered.

    MP3 on purpose: it is what a music bed usually arrives as, and the encoder
    smears transients and adds a delay of its own, which the beat has to be
    found through.

    Args:
        folder: Directory to write it to.
        name: File name.
        expression: An `aevalsrc` expression in `t`.
        seconds: Length.

    Returns:
        The file.
    """
    path = folder / name
    result = subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"aevalsrc='{expression}':s=44100:d={seconds}",
         "-c:a", "libmp3lame", str(path)],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return path

def kick(period: float) -> str:
    """A kick drum: a pitch sweep that drops off fast, once a period.

    Args:
        period: Seconds between hits.

    Returns:
        The expression.
    """
    since = rf"mod(t\,{period})"
    return rf"0.8*sin(2*PI*(50+150*exp(-30*{since}))*{since})*exp(-8*{since})"

def hat(period: float) -> str:
    """A hi-hat on the off-beat: a burst of noise halfway between kicks.

    Args:
        period: Seconds between kicks.

    Returns:
        The expression.
    """
    return rf"0.15*(random(0)-0.5)*exp(-60*mod(t+{period / 2}\,{period}))"

PAD = r"0.15*sin(2*PI*220*t)+0.15*sin(2*PI*277*t)+0.1*sin(2*PI*330*t)"

def drift(beats, period: float, phase: float = 0.0) -> np.ndarray:
    """Measure how far each found beat sits from the true grid.

    Args:
        beats: The beats found.
        period: The true beat period.
        phase: Where the true grid starts.

    Returns:
        Each beat's distance from its nearest true beat, signed.
    """
    found = np.array(beats)
    return (found - phase + period / 2) % period - period / 2

def rhythm_of(path: Path):
    """Measure a file's rhythm with nobody waiting on it.

    Args:
        path: File to measure.

    Returns:
        The rhythm.
    """
    return measure_rhythm(str(path), lambda fraction: None, lambda: False, FFMPEG)

@pytest.mark.parametrize("period", [0.5, 0.4, 0.6])
def test_a_click_track_is_found_at_its_tempo_and_on_its_clicks(tmp_path: Path, period: float) -> None:
    """0.4s is the case that goes wrong first: its period is not a whole number of
    frames, and the envelope used to line up better at twice the period."""
    rhythm = rhythm_of(synthesize(tmp_path, "click.mp3", rf"sin(2*PI*1000*t)*lt(mod(t\,{period})\,0.03)"))
    assert rhythm.tempo == pytest.approx(60 / period, rel=0.02)
    assert np.abs(drift(rhythm.beats, period)).max() < FRAME_AT_60
    # Every click after the first is found; the chain only needs its first step to start.
    assert len(rhythm.beats) >= 20 / period - 2

def test_a_beat_that_starts_late_is_found_where_it_starts(tmp_path: Path) -> None:
    expression = rf"sin(2*PI*800*t)*lt(mod(t-0.2\,0.6)\,0.03)*gte(t\,0.2)"
    rhythm = rhythm_of(synthesize(tmp_path, "late.mp3", expression))
    assert np.abs(drift(rhythm.beats, 0.6, phase=0.2)).max() < FRAME_AT_60

def test_the_beat_is_the_kick_and_not_the_hat_between_them(tmp_path: Path) -> None:
    rhythm = rhythm_of(synthesize(tmp_path, "song.mp3", f"{kick(0.5)}+{hat(0.5)}+{PAD}"))
    assert rhythm.tempo == pytest.approx(120, rel=0.02)
    assert np.abs(drift(rhythm.beats, 0.5)).max() < FRAME_AT_60

def test_a_beat_under_hiss_is_still_found(tmp_path: Path) -> None:
    """Hiss makes every bin of the spectrum flicker; summed into bands, it averages out."""
    period = 60 / 128
    expression = f"{kick(period)}+{hat(period)}+{PAD}+0.3*(random(1)-0.5)"
    rhythm = rhythm_of(synthesize(tmp_path, "hiss.mp3", expression))
    assert rhythm.tempo == pytest.approx(128, rel=0.02)
    assert np.abs(drift(rhythm.beats, period)).max() < FRAME_AT_60

@pytest.mark.parametrize("expression", [
    r"sin(2*PI*200*t)",
    PAD,
    r"0.3*(random(0)-0.5)",
], ids=["steady tone", "sustained pad", "hiss"])
def test_music_without_a_pulse_has_no_beat(tmp_path: Path, expression: str) -> None:
    """Not a failure: an answer. A beat put on these would be made up."""
    rhythm = rhythm_of(synthesize(tmp_path, "flat.mp3", expression))
    assert rhythm.tempo is None
    assert rhythm.beats == []

@pytest.mark.parametrize("seconds", [0.05, 0.3, 1])
def test_a_sound_too_short_to_have_a_tempo_has_none(tmp_path: Path, seconds: float) -> None:
    """Shorter than the one-second smoothing either side, which it has to survive."""
    rhythm = rhythm_of(synthesize(tmp_path, "blip.mp3", rf"sin(2*PI*1000*t)*lt(mod(t\,0.1)\,0.01)", seconds))
    assert rhythm.tempo is None and rhythm.beats == []

def test_the_beat_is_measured_on_music_and_not_on_footage(tmp_path: Path) -> None:
    song = synthesize(tmp_path, "song.mp3", f"{kick(0.5)}+{PAD}", seconds=8)
    music = Asset(id="song", path=str(song), duration=8, has_video=False, has_audio=True)
    analysis = analyze_media(music, str(tmp_path), False, None, None, None, lambda *_: None, lambda: False, FFMPEG)
    assert analysis.rhythm is not None and analysis.rhythm.tempo == pytest.approx(120, rel=0.02)
    assert analysis.recipe.rhythm_model == rhythm_model_name()

    footage = tmp_path / "talk.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "smptebars=size=320x240:rate=30:duration=4",
         "-f", "lavfi", "-i", f"aevalsrc='{kick(0.5)}':d=4", "-shortest", "-c:v", "libx264", "-preset",
         "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(footage)],
        check=True, capture_output=True,
    )
    shot = Asset(id="talk", path=str(footage), duration=4, has_video=True, has_audio=True)
    analysis = analyze_media(shot, str(tmp_path), False, None, None, None, lambda *_: None, lambda: False, FFMPEG)
    # A beat tracker over somebody talking finds one in their syllables, so it is not asked.
    assert analysis.rhythm is None
    assert analysis.recipe.rhythm_model is None

def test_changing_how_beats_are_found_only_ages_the_music(monkeypatch: pytest.MonkeyPatch) -> None:
    music = current_recipe(None, FFMPEG, rhythm_model=rhythm_model_name())
    footage = current_recipe(None, FFMPEG)
    monkeypatch.setattr("app.engine.rhythm.MIN_CONTRAST", 5.0)
    now = current_recipe(None, FFMPEG, rhythm_model=rhythm_model_name())
    assert music.differs_from(now)
    assert not footage.differs_from(now)
