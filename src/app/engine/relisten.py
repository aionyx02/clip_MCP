"""Listening to a file's talking again, without the transcript.

The transcript is made in one long decode, and a word in it is placed with the
words around it in mind. Usually that is what makes it right; now and then it
places a first word late, stretches a word across a pause, or leaves a soft
one out altogether. Everything downstream — the planner choosing sentences, the
compiler cutting on word edges, the check before a render — reads that same
transcript, so a word it misplaced is misplaced for all of them at once, and
nothing notices a syllable taken off. The first edit this was built for lost
「我們」 off 「我們可能要拆個30分鐘」 and 「這個」 off 「這個拆要很小心」, and
every check passed it.

So the talking is listened to a second time by two things that never saw the
transcript: a voice detector, which says where somebody is talking and
nothing about what, and the speech model again, run on each voiced stretch on
its own with nothing said before it to lean on. Where they hear talking right
up against a cut, the cut is moved or reported.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from app.engine.diarize import SAMPLE_RATE, read_samples
from app.engine.ffmpeg import OperationCancelled
from app.models.media import HeardWord, SecondListen, Span

VOICE_DETECTOR = "silero-vad"
# How sure the voice detector has to be before a window is somebody talking, and how
# long a gap or a sound has to be before it counts. Measured against the edges the first
# edit got wrong: at 0.5 with no padding it put the onset of 「終」 at 2.05s where the
# sound rises at 2.0s, and the end of 「囉」 at 9.82s where it falls away by 9.8s.
# Provisional (roadmap §13).
VOICE_THRESHOLD = 0.5
VOICE_MIN_SPEECH_MS = 100
VOICE_MIN_SILENCE_MS = 100
# Voiced stretches closer than this are transcribed together, so a phrase with a breath
# in it is heard as one phrase; a stretch is never longer than the speech model's window.
# Provisional (roadmap §13), the joining; the longest is the model's own limit.
JOIN_SECONDS = 0.8
LONGEST_STRETCH_SECONDS = 28.0
# Sound taken either side of a voiced stretch before transcribing it. The detector
# misses soft onsets — it put 「我們」 at 170.5s where it starts at 170.28s — and the
# speech model does not. Provisional (roadmap §13).
STRETCH_PAD_SECONDS = 0.4

@dataclass(frozen=True)
class Listener:
    """The speech model the second listen transcribes with.

    Attributes:
        name: The model's name, or a path to it.
        device: `"cuda"` or `"cpu"`.
        download_root: Where the weights are kept.
    """

    name: str
    device: str
    download_root: str

def hear(path: str, language: Optional[str], listener: Listener) -> str:
    """Transcribe a short recording on its own, with nothing to lean on.

    For checking what a speech model said: the words it was asked for are
    not given, so a word it swallowed is not heard anyway.

    Args:
        path: The recording.
        language: Its language, when known.
        listener: The speech model to transcribe with.

    Returns:
        What is said in it.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(
        listener.name, device=listener.device,
        compute_type="float16" if listener.device == "cuda" else "int8", download_root=listener.download_root,
    )
    segments, _ = model.transcribe(path, language=language, condition_on_previous_text=False, vad_filter=False)
    return "".join(segment.text for segment in segments).strip()

def voiced_stretches(samples: np.ndarray) -> List[Tuple[float, float]]:
    """Find where somebody is talking, by listening alone.

    Args:
        samples: Mono samples at `SAMPLE_RATE`.

    Returns:
        `(start, end)` of each voiced stretch, in seconds, in order.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    found = get_speech_timestamps(samples, VadOptions(
        threshold=VOICE_THRESHOLD,
        min_speech_duration_ms=VOICE_MIN_SPEECH_MS,
        min_silence_duration_ms=VOICE_MIN_SILENCE_MS,
        speech_pad_ms=0,
    ))
    return [(round(item["start"] / SAMPLE_RATE, 3), round(item["end"] / SAMPLE_RATE, 3)) for item in found]

def stretches_to_transcribe(voiced: Sequence[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    """Group voiced stretches into the pieces the speech model hears one at a time.

    Args:
        voiced: The voiced stretches, in order.
        duration: How long the file is.

    Returns:
        `(start, end)` of each piece, padded by `STRETCH_PAD_SECONDS` and never
        running into the next.
    """
    groups: List[List[float]] = []
    for start, end in voiced:
        if groups and start - groups[-1][1] <= JOIN_SECONDS and end - groups[-1][0] <= LONGEST_STRETCH_SECONDS:
            groups[-1][1] = end
        else:
            groups.append([start, end])
    pieces: List[Tuple[float, float]] = []
    for index, (start, end) in enumerate(groups):
        # Halfway to the neighbouring stretch at most, so two pieces never hear the same word.
        before = (groups[index - 1][1] + start) / 2 if index else 0.0
        after = (end + groups[index + 1][0]) / 2 if index + 1 < len(groups) else duration
        pieces.append((
            round(max(start - STRETCH_PAD_SECONDS, before, 0.0), 3),
            round(min(end + STRETCH_PAD_SECONDS, after, duration), 3),
        ))
    return pieces

def listen_again(
    path: str,
    language: Optional[str],
    listener: Listener,
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
    convert: Callable[[str], str] = lambda text: text,
    ffmpeg_bin: str = "ffmpeg",
) -> SecondListen:
    """Listen to a file's talking a second time, without its transcript.

    Args:
        path: Media file.
        language: The language the transcript found, so the second listen
            does not have to guess it from a second of sound.
        listener: The speech model to transcribe with.
        on_progress: Called with the completed fraction.
        is_cancelled: Polled between stretches to stop early.
        convert: Applied to every word heard, to write it the way the
            transcript is written, such as in traditional characters.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        Where the detector hears talking, and every word heard.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If the file cannot be decoded.
    """
    from faster_whisper import WhisperModel

    samples = np.frombuffer(read_samples(path, ffmpeg_bin), dtype=np.float32)
    duration = len(samples) / SAMPLE_RATE
    voiced = voiced_stretches(samples)
    pieces = stretches_to_transcribe(voiced, duration)
    words: List[HeardWord] = []
    if pieces:
        model = WhisperModel(
            listener.name,
            device=listener.device,
            compute_type="float16" if listener.device == "cuda" else "int8",
            download_root=listener.download_root,
        )
        for index, (start, end) in enumerate(pieces):
            if is_cancelled():
                raise OperationCancelled()
            # Nothing said before the stretch is given to it — no prompt, no previous
            # text — because leaning on what came before is what the first listen did.
            segments, _ = model.transcribe(
                samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)],
                language=language,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            for segment in segments:
                for word in segment.words or []:
                    words.append(HeardWord(
                        start=round(start + word.start, 3),
                        end=round(start + word.end, 3),
                        text=convert(word.word.strip()),
                        probability=round(float(word.probability), 3),
                    ))
            on_progress((index + 1) / len(pieces))
    return SecondListen(
        voiced=[Span(start=start, end=end) for start, end in voiced],
        words=[word for word in words if word.text],
        model=f"{VOICE_DETECTOR}+{listener.name}",
    )
