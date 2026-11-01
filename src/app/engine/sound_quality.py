"""How a stretch of a source file sounds, measured: hum, bandwidth, noise and sibilance (#165).

`get_analysis` has levels per second, which say how loud something is and not
what is wrong with it. These numbers are measured from the spectrum, and what
they add up to, such as a muffled recording, is for the model to say.
"""

import subprocess
from typing import Dict, Optional

import numpy as np

from app.engine.ffmpeg import hidden_window_flags

RATE = 48000
# About 1 Hz between bins, which is what tells mains hum from the sound around it.
HUM_FFT = RATE
HUM_BASES = (50, 60)
HUM_HARMONICS = (1, 2, 3)
HUM_SURROUNDINGS_HZ = 20
# Provisional: how far a tone must stand above the bins around it to count as hum, in decibels.
# Averaged noise stays under it; not checked against real recordings yet.
HUM_MARGIN_DB = 12.0
# Provisional: how far under the strongest band sound still counts as there, in decibels. Measures
# where a recording's top end stops: a lowpass from an old codec or a phone reads at its cutoff.
BANDWIDTH_RANGE_DB = 60.0
BAND_LOWEST_HZ = 100.0
FRAME_SECONDS = 0.05
# Provisional: the share of frames taken as the quiet part and as the loud part of a stretch.
EDGE_SHARE = 0.1
MIN_FRAME_STATS_SECONDS = 1.0
SPEECH_BAND = (300, 3000)
SIBILANCE_BAND = (5000, 9000)
SILENCE_POWER = 1e-12


def read_sound(path: str, start: float, seconds: float, ffmpeg_bin: str = "ffmpeg") -> np.ndarray:
    """A stretch of a file's sound as mono samples between -1 and 1.

    The seek is the plain one: a stretch this is measured over is seconds long,
    and a seek an AAC frame early does not change a spectrum (#171).

    Args:
        path: The file.
        start: Where to start, in seconds.
        seconds: How long to read.
        ffmpeg_bin: FFmpeg executable.

    Returns:
        The samples at `RATE`.

    Raises:
        RuntimeError: If FFmpeg cannot read the sound.
    """
    decoded = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", path,
         "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
        stdin=subprocess.DEVNULL, capture_output=True, creationflags=hidden_window_flags(),
    )
    if decoded.returncode != 0:
        raise RuntimeError(f"could not read the sound: {decoded.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    return np.frombuffer(decoded.stdout, dtype=np.int16).astype(np.float64) / 32768.0


def _spectrum(samples: np.ndarray) -> np.ndarray:
    """The power in each 1 Hz bin, averaged over half-overlapping windows."""
    if len(samples) < HUM_FFT:
        samples = np.pad(samples, (0, HUM_FFT - len(samples)))
    window = np.hanning(HUM_FFT)
    starts = range(0, len(samples) - HUM_FFT + 1, HUM_FFT // 2)
    return np.mean([np.abs(np.fft.rfft(samples[at:at + HUM_FFT] * window)) ** 2 for at in starts], axis=0)


def _hum(spectrum: np.ndarray) -> Optional[Dict[str, float]]:
    """Which mains frequency stands above the sound around it, if either does."""
    best: Optional[Dict[str, float]] = None
    for base in HUM_BASES:
        for harmonic in HUM_HARMONICS:
            at = base * harmonic
            near = np.r_[spectrum[at - HUM_SURROUNDINGS_HZ:at - 2], spectrum[at + 3:at + HUM_SURROUNDINGS_HZ + 1]]
            floor = max(float(np.median(near)), SILENCE_POWER)
            above = 10 * np.log10(max(float(spectrum[at - 1:at + 2].max()), SILENCE_POWER) / floor)
            if above >= HUM_MARGIN_DB and (best is None or above > best["above_surroundings_db"]):
                best = {"hertz": base, "above_surroundings_db": round(float(above), 1)}
    return best


def _bandwidth(spectrum: np.ndarray) -> Optional[float]:
    """The highest third-octave band within `BANDWIDTH_RANGE_DB` of the strongest, by its centre."""
    centres, levels = [], []
    centre = BAND_LOWEST_HZ
    while centre * 2 ** (1 / 6) <= RATE / 2:
        low, high = int(centre / 2 ** (1 / 6)), int(centre * 2 ** (1 / 6))
        centres.append(centre)
        levels.append(float(spectrum[low:high + 1].mean()))
        centre *= 2 ** (1 / 3)
    strongest = max(levels)
    if strongest <= SILENCE_POWER:
        return None
    floor = strongest * 10 ** (-BANDWIDTH_RANGE_DB / 10)
    return round(max(centre for centre, level in zip(centres, levels) if level >= floor))


def _decibels(power: float) -> float:
    return round(10 * float(np.log10(max(power, SILENCE_POWER))), 1)


def _frame_figures(samples: np.ndarray) -> Dict[str, Optional[float]]:
    """Noise, signal over noise and sibilance, from the quietest and loudest frames."""
    size = round(FRAME_SECONDS * RATE)
    count = len(samples) // size
    if len(samples) / RATE < MIN_FRAME_STATS_SECONDS:
        return {"noise_floor_dbfs": None, "speech_band_snr_db": None, "sibilance_db": None}
    power = 2 * np.abs(np.fft.rfft(samples[:count * size].reshape(count, size), axis=1)) ** 2 / size ** 2
    hertz = np.fft.rfftfreq(size, 1 / RATE)
    speech = power[:, (hertz >= SPEECH_BAND[0]) & (hertz < SPEECH_BAND[1])].sum(axis=1)
    sibilance = power[:, (hertz >= SIBILANCE_BAND[0]) & (hertz < SIBILANCE_BAND[1])].sum(axis=1)
    order = np.argsort(speech)
    edge = max(1, round(count * EDGE_SHARE))
    quiet, loud = order[:edge], order[-edge:]
    noise = _decibels(float(speech[quiet].mean()))
    signal = _decibels(float(speech[loud].mean()))
    return {
        "noise_floor_dbfs": noise,
        "speech_band_snr_db": round(signal - noise, 1),
        "sibilance_db": round(_decibels(float(sibilance[loud].mean())) - signal, 1),
    }


def measure(samples: np.ndarray) -> Dict[str, object]:
    """Measure a stretch of sound.

    Args:
        samples: Mono samples at `RATE`.

    Returns:
        `hum`, `bandwidth_hz`, `noise_floor_dbfs`, `speech_band_snr_db` and
        `sibilance_db`; what could not be measured is null.
    """
    spectrum = _spectrum(samples)
    return {"hum": _hum(spectrum), "bandwidth_hz": _bandwidth(spectrum), **_frame_figures(samples)}


def spectrogram(path: str, start: float, seconds: float, ffmpeg_bin: str = "ffmpeg") -> bytes:
    """Draw a stretch of a file's sound as a spectrogram.

    Args:
        path: The file.
        start: Where to start, in seconds.
        seconds: How long to draw.
        ffmpeg_bin: FFmpeg executable.

    Returns:
        A PNG with time across and frequency up, legend included.

    Raises:
        RuntimeError: If FFmpeg cannot draw it.
    """
    drawn = None
    # A logarithmic frequency axis shows where voices sit, and older FFmpeg builds do not have it.
    for options in ("s=1024x512:legend=1:fscale=log", "s=1024x512:legend=1"):
        drawn = subprocess.run(
            [ffmpeg_bin, "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", path,
             "-vn", "-lavfi", f"showspectrumpic={options}", "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"],
            stdin=subprocess.DEVNULL, capture_output=True, creationflags=hidden_window_flags(),
        )
        if drawn.returncode == 0 and drawn.stdout:
            return drawn.stdout
    raise RuntimeError(f"could not draw the sound: {drawn.stderr.decode('utf-8', 'replace').strip()[-300:]}")
