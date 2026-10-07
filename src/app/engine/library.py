"""Which library a file belongs in: footage, or music.

A file with a picture is footage. A file without one is music unless it is
mostly somebody talking — a voice memo, a narration, an interview recorded on
a phone — which is footage too, to be transcribed rather than beat-tracked.
Speech is found with the Silero VAD that ships inside faster-whisper, run over
a few short stretches rather than the whole file, so adding a folder of songs
stays quick.
"""

import subprocess
from typing import List, Optional

import numpy as np

from app.engine.ffmpeg import hidden_window_flags
from app.models.media import Asset

FOOTAGE = "footage"
MUSIC = "music"
LIBRARIES = (FOOTAGE, MUSIC)

# A file with no picture is a recording of somebody talking when at least this share of
# what is listened to is speech. Singing is heard as speech now and then, so it is well
# above what songs score. Provisional: set from the songs and narrations in one library.
SPEECH_SHARE = 0.5
# How much is listened to: this many stretches, each this long, spread over the file. Provisional.
STRETCHES = 3
STRETCH_SECONDS = 20.0
_RATE = 16000


def library_of(asset: Asset) -> str:
    """The library a file is in: what it was put in, or what it looks like until it is classified.

    Args:
        asset: The file.

    Returns:
        `footage` or `music`. A file with no picture that was never classified
        counts as music, which is how it has been treated so far.
    """
    if asset.library in LIBRARIES:
        return asset.library
    return MUSIC if asset.has_audio and not asset.has_video else FOOTAGE


def _stretches(duration: float) -> List[float]:
    """Where to start listening: spread over the file, clear of its very start and end."""
    if duration <= STRETCH_SECONDS * STRETCHES:
        return [0.0]
    return [duration * (index + 1) / (STRETCHES + 1) - STRETCH_SECONDS / 2 for index in range(STRETCHES)]


def speech_share(path: str, duration: float, ffmpeg_bin: str = "ffmpeg") -> Optional[float]:
    """How much of a file's sound is speech, over a few stretches of it.

    Args:
        path: The file.
        duration: Its length in seconds.
        ffmpeg_bin: FFmpeg executable.

    Returns:
        The share from 0 to 1, or None when the sound could not be read.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    heard = speech = 0.0
    for start in _stretches(duration):
        decoded = subprocess.run(
            [ffmpeg_bin, "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{STRETCH_SECONDS:.3f}",
             "-i", path, "-vn", "-ac", "1", "-ar", str(_RATE), "-f", "s16le", "-"],
            stdin=subprocess.DEVNULL, capture_output=True, creationflags=hidden_window_flags(),
        )
        if decoded.returncode != 0 or not decoded.stdout:
            continue
        samples = np.frombuffer(decoded.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        heard += len(samples) / _RATE
        for found in get_speech_timestamps(samples, VadOptions()):
            speech += (found["end"] - found["start"]) / _RATE
    return speech / heard if heard else None


def classify(asset: Asset, ffmpeg_bin: str = "ffmpeg") -> str:
    """Decide which library a newly added file belongs in.

    Args:
        asset: The file.
        ffmpeg_bin: FFmpeg executable.

    Returns:
        `footage` or `music`.
    """
    if asset.has_video or not asset.has_audio:
        return FOOTAGE
    share = speech_share(asset.path, float(asset.duration or 0), ffmpeg_bin)
    return FOOTAGE if share is not None and share >= SPEECH_SHARE else MUSIC
