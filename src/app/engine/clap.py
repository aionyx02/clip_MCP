"""What a song sounds like, in words: CLAP, run locally.

CLAP puts sound and text in one space, so a song can be compared with
"happy music" or "lo-fi hip hop music" — or with whatever the AI is asked for
— by how close they lie. The model is LAION's `larger_clap_music_and_speech`
(Apache-2.0), in the quantized ONNX export, downloaded the first time a song
is analyzed. Measured on one library of 39 songs: the style is right far more
often than not, the mood about half the time, so the mood is read out as how
a song sounds rather than used to decide whether it fits.

A song is heard as three stretches of ten seconds, at a fifth, under a half
and seven tenths of the way in, and the three are averaged: what it is like
through its middle, not in an intro that may be nothing like the rest.
"""

import subprocess
import threading
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.engine import models
from app.engine.ffmpeg import hidden_window_flags

# Pinned to a commit of the repository the ONNX export lives in, as the search model is.
_SOURCE = "https://huggingface.co/Xenova/larger_clap_music_and_speech/resolve/e9fd5ac1dbf3280936a7fc3ec8a020453ff184db"
AUDIO_MODEL = models.Model(
    name="music listening model",
    path="clap/larger-clap-music-speech-audio-q8.onnx",
    url=f"{_SOURCE}/onnx/audio_model_quantized.onnx",
    sha256="021dd4fb962b4ed20cc3a6730b09e0ccf9f9c49931047032118a3b64828513a6",
    size=75 * models.MEGABYTE,
)
TEXT_MODEL = models.Model(
    name="music description model",
    path="clap/larger-clap-music-speech-text-q8.onnx",
    url=f"{_SOURCE}/onnx/text_model_quantized.onnx",
    sha256="8f9f29c5f6adee917553d4b3a70729c731c0d18b88efca0ae67c1a1fc278f3b6",
    size=121 * models.MEGABYTE,
)
TOKENIZER = models.Model(
    name="music description tokenizer",
    path="clap/larger-clap-music-speech-tokenizer.json",
    url=f"{_SOURCE}/tokenizer.json",
    sha256="dc239041d98de27ffc3975473a1a23e3db4c937b23c138c38bbc66588bd247e5",
    size=2 * models.MEGABYTE,
)
# Stored with every analysis, so a song heard by another model is heard again.
MODEL_ID = "larger-clap-music-speech-q8"

# The words a song is described in, each with the English the model was taught. Every
# song keeps its score against each; which come first is decided against the library.
MOODS = {
    "歡樂": "happy", "振奮": "uplifting", "平靜": "calm", "放鬆": "relaxing", "哀傷": "sad", "浪漫": "romantic",
    "陰暗": "dark", "緊張": "tense", "神秘": "mysterious", "壯闊": "epic", "俏皮": "playful", "懷舊": "nostalgic",
    "有活力": "energetic", "夢幻": "dreamy", "溫暖": "warm", "逗趣": "funny",
}
STYLES = {
    "鋼琴": "piano", "原聲吉他": "acoustic guitar", "電子": "electronic", "Lo-fi": "lo-fi hip hop", "爵士": "jazz",
    "搖滾": "rock", "流行": "pop", "嘻哈": "hip hop", "管弦": "orchestral", "氛圍": "ambient", "放克": "funk",
    "拉丁": "latin", "民謠": "folk", "電影配樂": "cinematic", "電玩": "chiptune 8-bit", "企業簡報": "corporate",
}

# How the model wants its sound: ten seconds at 48 kHz as 64 mel bands, as its own
# feature extractor makes them.
_RATE, _FFT, _HOP, _MELS, _LOWEST, _HIGHEST, _SECONDS = 48000, 1024, 480, 64, 50.0, 14000.0, 10
# Where the three stretches are heard, as shares of the song. Provisional.
LISTEN_AT = (0.2, 0.45, 0.7)
# Fewer analyzed songs than this and the library says too little about what is usual:
# tags are ranked by their own scores instead. Provisional.
LIBRARY_MIN_SONGS = 8
TOP = 3

_lock = threading.Lock()
_loaded: Optional[tuple] = None
_described: Dict[str, np.ndarray] = {}


def download_size() -> int:
    """Bytes still to download before a song can be listened to; 0 once it is here."""
    return sum(model.size for model in (AUDIO_MODEL, TEXT_MODEL, TOKENIZER) if not models.is_downloaded(model))


def needed() -> List[models.Model]:
    """The files listening to a song needs."""
    return [AUDIO_MODEL, TEXT_MODEL, TOKENIZER]


def _load() -> tuple:
    """The two models and the tokenizer, kept in memory after the first song."""
    global _loaded
    with _lock:
        if _loaded is None:
            import onnxruntime
            from tokenizers import Tokenizer

            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            audio = onnxruntime.InferenceSession(models.ensure(AUDIO_MODEL), options,
                                                 providers=["CPUExecutionProvider"])
            text = onnxruntime.InferenceSession(models.ensure(TEXT_MODEL), options,
                                                providers=["CPUExecutionProvider"])
            _loaded = (audio, text, Tokenizer.from_file(models.ensure(TOKENIZER)))
        return _loaded


def _hertz_to_mel(hertz: np.ndarray) -> np.ndarray:
    """Slaney's mel scale: straight below 1 kHz, logarithmic above."""
    hertz = np.asarray(hertz, dtype=np.float64)
    return np.where(hertz >= 1000.0, 15.0 + np.log(np.maximum(hertz, 1e-10) / 1000.0) * (27.0 / np.log(6.4)),
                    3.0 * hertz / 200.0)


def _mel_to_hertz(mel: np.ndarray) -> np.ndarray:
    """The way back from Slaney's mel scale."""
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(mel >= 15.0, 1000.0 * np.exp(np.log(6.4) / 27.0 * (mel - 15.0)), 200.0 * mel / 3.0)


def _filters() -> np.ndarray:
    """The mel filter bank, Slaney-normalized: one column per band."""
    bins = np.linspace(0, _RATE // 2, _FFT // 2 + 1)
    edges = _mel_to_hertz(np.linspace(_hertz_to_mel(_LOWEST), _hertz_to_mel(_HIGHEST), _MELS + 2))
    widths = np.diff(edges)
    slopes = edges[None, :] - bins[:, None]
    bank = np.maximum(0, np.minimum(-slopes[:, :-2] / widths[:-1], slopes[:, 2:] / widths[1:]))
    return bank * (2.0 / (edges[2:_MELS + 2] - edges[:_MELS]))


_BANK = _filters()
_WINDOW = np.hanning(_FFT + 1)[:-1]


def _features(samples: np.ndarray) -> np.ndarray:
    """Ten seconds of sound as the model takes it: log mel bands, repeated to length if short."""
    length = _RATE * _SECONDS
    if len(samples) < length:
        samples = np.tile(samples, max(1, length // max(1, len(samples))))
        samples = np.pad(samples, (0, length - len(samples)))
    padded = np.pad(samples[:length], _FFT // 2, mode="reflect")
    count = 1 + (len(padded) - _FFT) // _HOP
    frames = padded[np.arange(_FFT)[None, :] + _HOP * np.arange(count)[:, None]]
    power = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2
    return (10.0 * np.log10(np.maximum(power @ _BANK, 1e-10))).astype(np.float32)[None, None]


def _decode(path: str, start: float, ffmpeg_bin: str) -> np.ndarray:
    """Ten seconds of a file from `start`, mono at the model's rate."""
    decoded = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", str(_SECONDS), "-i", path,
         "-vn", "-ac", "1", "-ar", str(_RATE), "-f", "f32le", "-"],
        stdin=subprocess.DEVNULL, capture_output=True, creationflags=hidden_window_flags(),
    )
    return np.frombuffer(decoded.stdout, dtype=np.float32) if decoded.returncode == 0 else np.zeros(0, np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def describe(text: str) -> np.ndarray:
    """A description of music, as a unit vector in the space songs are heard into.

    Args:
        text: In English, as the model was taught: "calm piano music".

    Returns:
        The vector.
    """
    if text not in _described:
        _, model, tokenizer = _load()
        # One at a time: the model takes no attention mask, so padding would be read as words.
        ids = np.array([tokenizer.encode(text).ids], dtype=np.int64)
        _described[text] = _unit(model.run(None, {"input_ids": ids})[0][0])
    return _described[text]


def listen(path: str, duration: float, ffmpeg_bin: str = "ffmpeg") -> Tuple[List[float], Dict[str, float], Dict[str, float]]:
    """Hear a song: what it is like as a vector, and how close it lies to each mood and style.

    Args:
        path: The file.
        duration: Its length in seconds.
        ffmpeg_bin: FFmpeg executable.

    Returns:
        `(vector, moods, styles)`: the unit vector, rounded, and the score
        against each word of `MOODS` and `STYLES`.
    """
    audio, _, _ = _load()
    starts = [0.0] if duration <= _SECONDS else [max(0.0, duration * share - _SECONDS / 2) for share in LISTEN_AT]
    name = audio.get_inputs()[0].name
    heard = [audio.run(None, {name: _features(_decode(path, start, ffmpeg_bin))})[0][0] for start in starts]
    vector = _unit(np.mean(heard, axis=0))

    def scores(table: Mapping[str, str]) -> Dict[str, float]:
        return {tag: round(float(describe(f"{english} music") @ vector), 4) for tag, english in table.items()}

    return [round(float(value), 5) for value in vector], scores(MOODS), scores(STYLES)


def ranked(scores: Mapping[str, Mapping[str, float]]) -> Dict[str, List[str]]:
    """The words that set each song apart from the rest of the library, best first.

    A word the model gives every song — `corporate`, `happy` — says nothing about
    any one of them, so each score is weighed against how that word scores
    across the library. With too few songs to know what is usual, the scores
    stand as they are.

    Args:
        scores: Each song's score against each word, by song.

    Returns:
        The `TOP` words of each song.
    """
    songs = list(scores)
    if not songs:
        return {}
    words = list(next(iter(scores.values())))
    if len(songs) < LIBRARY_MIN_SONGS:
        return {song: sorted(words, key=lambda word: -scores[song][word])[:TOP] for song in songs}
    table = np.array([[scores[song].get(word, 0.0) for word in words] for song in songs])
    mean, spread = table.mean(axis=0), np.maximum(table.std(axis=0), 1e-6)
    standing = (table - mean) / spread
    return {song: [words[index] for index in np.argsort(-standing[row])[:TOP]] for row, song in enumerate(songs)}


def closest(description: str, vectors: Mapping[str, Sequence[float]]) -> List[Tuple[str, float]]:
    """The songs closest to a description, closest first.

    Args:
        description: What is wanted, in English.
        vectors: Each song's vector, by song.

    Returns:
        `(song, score)` pairs.
    """
    wanted = describe(description)
    found = [(song, round(float(np.asarray(vector) @ wanted), 3)) for song, vector in vectors.items()]
    return sorted(found, key=lambda item: -item[1])
