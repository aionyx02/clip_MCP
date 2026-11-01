"""What a stretch of sound measures as, on sound whose answer is known (#165)."""

import base64
import wave
from pathlib import Path

import numpy
import pytest

from app.engine.sound_quality import RATE, measure
from app.tools.catalog import import_asset
from app.tools.looking import MAX_INSPECT_SECONDS, inspect_sound

SECONDS = 6
GENERATOR = numpy.random.default_rng(7)


def noise(level: float = 0.05, seconds: float = SECONDS) -> numpy.ndarray:
    return GENERATOR.normal(0, level, round(seconds * RATE))


def tone(hertz: float, level: float = 0.05, seconds: float = SECONDS) -> numpy.ndarray:
    return level * numpy.sin(2 * numpy.pi * hertz * numpy.arange(round(seconds * RATE)) / RATE)


def lowpassed(samples: numpy.ndarray, hertz: float) -> numpy.ndarray:
    spectrum = numpy.fft.rfft(samples)
    spectrum[numpy.fft.rfftfreq(len(samples), 1 / RATE) > hertz] = 0
    return numpy.fft.irfft(spectrum, len(samples))


def wav(path: Path, samples: numpy.ndarray) -> Path:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((numpy.clip(samples, -1, 1) * 32767).astype(numpy.int16).tobytes())
    return path


@pytest.mark.parametrize("hertz", [50, 60])
def test_mains_hum_is_found_at_its_own_frequency(hertz: int) -> None:
    assert measure(noise() + tone(hertz))["hum"]["hertz"] == hertz


def test_noise_alone_has_no_hum() -> None:
    assert measure(noise())["hum"] is None


def test_a_recording_cut_off_at_8_khz_reads_near_8_khz() -> None:
    assert measure(lowpassed(noise(), 8000))["bandwidth_hz"] == pytest.approx(8000, rel=0.15)


def test_full_band_sound_reads_above_16_khz() -> None:
    assert measure(noise())["bandwidth_hz"] > 16000


def test_loud_bursts_over_quiet_noise_read_as_the_distance_between_them() -> None:
    samples = noise(0.001)
    burst = round(0.5 * RATE)
    # Half a second on and off, so at least a tenth of the frames are each.
    for at in range(0, len(samples), 2 * burst):
        samples[at:at + burst] += noise(0.1, 0.5)[:len(samples[at:at + burst])]
    assert measure(samples)["speech_band_snr_db"] == pytest.approx(40, abs=3)


def test_a_stretch_too_short_for_frame_figures_has_them_null() -> None:
    found = measure(noise(seconds=0.5))
    assert found["noise_floor_dbfs"] is None and found["speech_band_snr_db"] is None
    assert found["bandwidth_hz"] is not None


def test_sibilance_reads_as_where_the_energy_is() -> None:
    assert measure(tone(1000) + tone(7000, 0.5))["sibilance_db"] > measure(tone(1000) + tone(7000, 0.005))["sibilance_db"]


def test_digital_silence_measures_without_failing() -> None:
    found = measure(numpy.zeros(SECONDS * RATE))
    assert found["hum"] is None and found["bandwidth_hz"] is None


def _asset(path: Path) -> str:
    return import_asset(str(path))["id"]


def test_the_tool_returns_the_numbers_and_a_picture(tmp_path: Path) -> None:
    asset = _asset(wav(tmp_path / "hum.wav", noise() + tone(60)))
    result = inspect_sound(asset, 1, 5)
    assert result.structured_content["hum"]["hertz"] == 60
    picture = next(part for part in result.content if part.type == "image")
    assert base64.b64decode(picture.data).startswith(b"\x89PNG")


def test_a_stretch_longer_than_the_limit_is_refused_and_says_how_to_narrow_it(tmp_path: Path) -> None:
    asset = _asset(wav(tmp_path / "long.wav", numpy.zeros((MAX_INSPECT_SECONDS + 5) * RATE)))
    with pytest.raises(ValueError, match="start and end"):
        inspect_sound(asset)


def test_a_file_with_no_sound_is_refused(media: Path) -> None:
    with pytest.raises(ValueError, match="no sound"):
        inspect_sound(_asset(media / "silent.mp4"))
