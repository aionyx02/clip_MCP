"""Finding the beat in a piece of music.

The beat is where a listener would tap their foot, and a cut that lands on it
reads as meant while one just off it reads as a mistake. Finding it is a
measurement rather than a judgement, so it happens here, once, at analysis
time: the compiler is handed the times and never has to listen to anything.

Two steps, both standard. First an onset envelope — how much new sound
arrives in each short frame, which peaks at every drum hit and note start.
Then the tempo from how that envelope repeats, and the beats themselves by
dynamic programming over it (Ellis, "Beat Tracking by Dynamic Programming",
2007): the best chain of onsets that stays close to that tempo. A chain rather
than a grid, so a band that drifts a little is followed rather than slowly
lost.

Only for music. Speech has onsets too — every syllable — and tracking a beat
through somebody talking produces a confident answer to a question nobody
asked, so the analysis pass runs this on files with sound and no picture.
"""

import hashlib
import json
import subprocess
from typing import Callable, List, Optional, Tuple

import numpy as np

from app.engine.ffmpeg import OperationCancelled, hidden_window_flags
from app.models.media import Rhythm

SAMPLE_RATE = 11025
# About 46ms of sound per frame, moved on by about 12ms: short enough to place a beat
# within a frame of video at 60fps, long enough to see the spectrum a hit changes.
FRAME = 512
HOP = 128
FRAMES_PER_SECOND = SAMPLE_RATE / HOP
# The tempos worth considering, and the one to lean towards when two of them — one
# double the other — fit about equally well, which is the classic way beat tracking
# goes wrong. Provisional until there is a corpus to tune them against.
MIN_BPM = 60.0
MAX_BPM = 200.0
PRIOR_BPM = 120.0
PRIOR_OCTAVES = 1.0
# How hard the chain is pulled back to the tempo when an onset tempts it away. Ellis's
# own figure, on an envelope scaled to unit spread. Provisional.
TIGHTNESS = 100.0
# How far the strongest onsets have to stand above the typical frame before there is a
# beat at all: the 99th percentile of the envelope against the mean flux. A drum part
# sits far above it, and ambient sound, a sustained pad, a steady tone or hiss sits at
# about one — music without a pulse, where a beat would be made up. Provisional.
MIN_CONTRAST = 3.0
# The bands the spectrum is summed into, spaced evenly in pitch from a kick drum's
# fundamental to the top of a snare's crack.
BANDS = 24
LOWEST_HZ = 40.0
HIGHEST_HZ = 5000.0
# A piece this short cannot establish a tempo, whatever it sounds like.
MIN_BEATS = 4
# Where in a frame the sound that made it rise usually is, in samples from its start. The
# flux jumps once a hit is far enough into the window for the taper not to hide it,
# which is past the middle. Calibrated on click tracks and synthesized drum parts, where
# the middle of the frame read about 15ms early and this reads within a few milliseconds.
ONSET_OFFSET = 5 * FRAME // 8
# Frames of the spectrum handled at once, which bounds memory on a long song.
CHUNK_FRAMES = 4096

def rhythm_model_name() -> str:
    """Name the way beats are found right now, thresholds included.

    Stored in the analysis recipe the way the speech and speaker models are,
    so that beats found under one set of these numbers are recognised as a
    different measurement from beats found under another — and so that
    changing them makes only the music out of date, not every file.

    Returns:
        A short name ending in a fingerprint of every setting above.
    """
    settings = {
        "sample_rate": SAMPLE_RATE, "frame": FRAME, "hop": HOP, "min_bpm": MIN_BPM, "max_bpm": MAX_BPM,
        "prior_bpm": PRIOR_BPM, "prior_octaves": PRIOR_OCTAVES, "tightness": TIGHTNESS,
        "min_contrast": MIN_CONTRAST, "bands": BANDS, "lowest_hz": LOWEST_HZ, "highest_hz": HIGHEST_HZ,
        "min_beats": MIN_BEATS, "onset_offset": ONSET_OFFSET,
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"banded-flux+ellis-dp-{fingerprint}"

def _decode(path: str, ffmpeg_bin: str) -> np.ndarray:
    """Decode a file's sound to mono samples.

    Args:
        path: Media file to read.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        The samples as 32-bit floats at `SAMPLE_RATE`.

    Raises:
        RuntimeError: If FFmpeg cannot decode the file.
    """
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-nostdin", "-v", "error", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        creationflags=hidden_window_flags(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip()[-500:] or "ffmpeg could not read the sound"
        )
    return np.frombuffer(result.stdout, dtype=np.float32)

def _bands() -> np.ndarray:
    """Say which band each bin of the spectrum belongs to.

    Returns:
        A band index per bin, or -1 for a bin outside every band.
    """
    frequencies = np.fft.rfftfreq(FRAME, 1.0 / SAMPLE_RATE)
    edges = np.geomspace(LOWEST_HZ, HIGHEST_HZ, BANDS + 1)
    band = np.digitize(frequencies, edges) - 1
    band[band >= BANDS] = -1
    return band

def onset_envelope(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Measure how much new sound arrives in each frame.

    Spectral flux over bands spaced the way hearing spaces pitch: the rise in
    each band's level from one frame to the next, falls ignored, summed. A drum
    hit lights up many bands at once and a sustained note lights up none after
    it starts, which is the difference a beat tracker needs.

    Bands rather than single bins, because a single bin is noisy: hiss makes
    every one of them flicker, and a steady tone beats against the frame rate
    in the bins beside it. Either one, summed over a few hundred bins, looks
    like a pulse that is not there. Summed within a band first, it averages
    out.

    Args:
        samples: Mono samples at `SAMPLE_RATE`.

    Returns:
        `(flux, envelope)` — the flux as measured, and the same with the slow
        swell of the arrangement taken out, so a loud chorus does not outweigh
        the pulse running through it. Empty for anything shorter than a frame.
    """
    if len(samples) < FRAME:
        return np.zeros(0), np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP]
    window = np.hanning(FRAME).astype(np.float32)
    band = _bands()
    levels = np.zeros((len(frames), BANDS))
    for at in range(0, len(frames), CHUNK_FRAMES):
        power = np.abs(np.fft.rfft(frames[at:at + CHUNK_FRAMES] * window, axis=1)) ** 2
        for index in range(BANDS):
            levels[at:at + len(power), index] = power[:, band == index].sum(axis=1)
    levels = np.log1p(np.sqrt(levels))
    flux = np.concatenate([[0.0], np.maximum(0.0, np.diff(levels, axis=0)).sum(axis=1)])
    # About a second either side: the swell, not the pulse.
    width = int(round(FRAMES_PER_SECOND))
    # Sliced out of the full convolution rather than asked for as "same", which comes back
    # the length of the kernel instead when the sound is shorter than it.
    trend = np.convolve(flux, np.ones(2 * width + 1) / (2 * width + 1))[width:width + len(flux)]
    return flux, np.maximum(0.0, flux - trend)

def _smoothed(envelope: np.ndarray) -> np.ndarray:
    """Widen each onset by a frame or two either side.

    A beat period is rarely a whole number of frames, and two onsets a
    fraction of a frame out of step miss each other entirely when each is one
    frame wide — which makes twice the period, where the fractions happen to
    line up, look like the better fit.

    Args:
        envelope: The onset envelope.

    Returns:
        The envelope, blurred by a Gaussian about 20ms wide.
    """
    offsets = np.arange(-4, 5)
    kernel = np.exp(-0.5 * (offsets / 1.5) ** 2)
    return np.convolve(envelope, kernel / kernel.sum())[4:4 + len(envelope)]

def _period(envelope: np.ndarray) -> Optional[float]:
    """Find how many frames apart the beats are.

    Args:
        envelope: The onset envelope.

    Returns:
        The beat period in frames, with sub-frame precision, or None when the
        envelope is too short to hold even the slowest tempo twice.
    """
    centred = envelope - envelope.mean()
    size = 1 << int(np.ceil(np.log2(2 * len(centred))))
    spectrum = np.fft.rfft(centred, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:len(centred)]
    if correlation[0] <= 0:
        return None
    shortest = int(np.floor(FRAMES_PER_SECOND * 60.0 / MAX_BPM))
    longest = int(np.ceil(FRAMES_PER_SECOND * 60.0 / MIN_BPM))
    if longest + 1 >= len(correlation):
        return None
    lags = np.arange(shortest, longest + 1)
    bpm = FRAMES_PER_SECOND * 60.0 / lags
    prior = np.exp(-0.5 * (np.log2(bpm / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)
    best = int(np.argmax(correlation[lags] * prior))
    lag = int(lags[best])
    # The peak sits between two lags as often as on one; a parabola through it and its
    # neighbours says where, which matters over a three-minute song.
    left, middle, right = correlation[lag - 1], correlation[lag], correlation[lag + 1]
    bend = left - 2 * middle + right
    return float(lag + (0.5 * (left - right) / bend if bend < 0 else 0.0))

def _chain(envelope: np.ndarray, period: float) -> List[int]:
    """Find the best chain of onsets that keeps to a tempo.

    Args:
        envelope: The onset envelope, scaled to unit spread.
        period: The beat period in frames.

    Returns:
        The frames the beats fall on, in order.
    """
    steps = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    penalty = -TIGHTNESS * np.log(-steps / period) ** 2
    score = envelope.astype(float).copy()
    back = np.full(len(envelope), -1)
    for frame in range(len(envelope)):
        before = frame + steps
        usable = before >= 0
        if not usable.any():
            continue
        candidates = penalty[usable] + score[before[usable]]
        best = int(np.argmax(candidates))
        score[frame] = envelope[frame] + candidates[best]
        back[frame] = before[usable][best]
    tail = max(1, int(round(period)))
    frame = len(envelope) - tail + int(np.argmax(score[-tail:]))
    chain = []
    while frame >= 0:
        chain.append(frame)
        frame = back[frame]
    return chain[::-1]

def _trimmed(chain: List[int], envelope: np.ndarray, period: float) -> List[int]:
    """Drop the beats the chain carried through silence at either end.

    The chain keeps time through a quiet intro or a fade-out because nothing
    stops it; the beats it places there are ones nobody hears. Where nothing
    arrives near a beat, it goes.

    Args:
        chain: The beat frames.
        envelope: The onset envelope.
        period: The beat period in frames.

    Returns:
        The beats from the first heard one to the last heard one.
    """
    reach = max(1, int(round(period / 8)))
    heard = np.array([envelope[max(0, frame - reach):frame + reach + 1].max() for frame in chain])
    if not len(heard):
        return []
    loud = heard >= 0.5 * np.median(heard)
    kept = np.flatnonzero(loud)
    return chain[kept[0]:kept[-1] + 1] if len(kept) else []

def find_rhythm(samples: np.ndarray) -> Rhythm:
    """Find the tempo and the beats in some music.

    Args:
        samples: Mono samples at `SAMPLE_RATE`.

    Returns:
        The rhythm. `tempo` is None and `beats` empty when there is no steady
        pulse to find, which is an answer rather than a failure: it says a cut
        cannot be put on the beat of this, not that nobody looked.
    """
    flux, envelope = onset_envelope(samples)
    nothing = Rhythm(tempo=None, beats=[])
    if len(envelope) == 0 or flux.mean() == 0:
        return nothing
    if np.percentile(envelope, 99) / flux.mean() < MIN_CONTRAST:
        return nothing
    envelope = _smoothed(envelope)
    period = _period(envelope)
    if period is None:
        return nothing
    frames = _trimmed(_chain(envelope / envelope.std(), period), envelope, period)
    if len(frames) < MIN_BEATS:
        return nothing
    beats = [round((frame * HOP + ONSET_OFFSET) / SAMPLE_RATE, 3) for frame in frames]
    tempo = 60.0 / float(np.median(np.diff(beats)))
    return Rhythm(tempo=round(tempo, 1), beats=beats)

def measure_rhythm(
    path: str,
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
    ffmpeg_bin: str = "ffmpeg",
) -> Rhythm:
    """Find the tempo and the beats in a music file.

    Args:
        path: File to read.
        on_progress: Called with the completed fraction.
        is_cancelled: Polled before the work starts and after decoding.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        The rhythm.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If FFmpeg cannot decode the file.
    """
    if is_cancelled():
        raise OperationCancelled()
    samples = _decode(path, ffmpeg_bin)
    on_progress(0.5)
    if is_cancelled():
        raise OperationCancelled()
    rhythm = find_rhythm(samples)
    on_progress(1.0)
    return rhythm
