"""Telling one voice from another, and saying which stretches belong together.

Diarization answers "how many people are talking and when is each of them
talking", and nothing else. It does not know who they are: the labels are made
up per file, so `S1` in one recording has no relation to `S1` in the next.
Putting a name to a label is a judgement somebody who was there has to make,
and the server does not guess at it.

Two models do the work. A segmentation model marks where speech is and where
the voice changes; an embedding model turns each stretch into a vector so that
stretches from the same person can be clustered together. Both run on
onnxruntime, both are plain files kept in the workspace, and neither is gated
behind an account — which is why they are these models rather than the
better-known ones.
"""

import array
import subprocess
from typing import Callable, Dict, List, Optional, Tuple

from app.engine import models
from app.engine.ffmpeg import OperationCancelled, hidden_window_flags
from app.models.media import SpeakerTurn, Voice

SAMPLE_RATE = 16000
# A voice has to hold the floor this long to be a turn, and has to stop this long for the
# turn to end. Below either, what is being measured is the rhythm of a conversation rather
# than who is in it. Provisional, like every threshold that wants a corpus behind it.
MIN_SPEECH_SECONDS = 0.3
MIN_PAUSE_SECONDS = 0.5
# How far apart two voices have to be before they are two people. Only used when the
# number of speakers is not known; given one, the clustering is told to find exactly that
# many and this is ignored.
SPEAKER_DISTANCE = 0.5
THREADS = 2
# How much of a voice to measure before deciding what it sounds like. Longer is steadier,
# but a person does not become a different person after twenty seconds, and an hour-long
# interview would otherwise pay for the whole hour twice. The longest turns are taken
# first: a two-second interjection carries more of the room than of the speaker.
VOICE_SECONDS = 20.0

def speaker_model_name() -> str:
    """Name the models that produced a set of speaker turns.

    Stored in the analysis recipe, so that turns found by one pair of models
    are not compared with turns found by another as though they were the same
    measurement.

    Returns:
        A short name covering both models.
    """
    return "pyannote-segmentation-3.0+campplus-zh-en"

def read_samples(path: str, ffmpeg_bin: str = "ffmpeg") -> array.array:
    """Decode a file's sound to the mono samples the models expect.

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
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip()[-500:] or "ffmpeg could not read the sound")
    samples = array.array("f")
    samples.frombytes(result.stdout)
    return samples

def _engine(speakers: Optional[int]):
    """Build the diarization engine, downloading its models on first use.

    Args:
        speakers: How many people are talking, when that is known. `None`
            leaves it to be worked out from how far apart the voices are.

    Returns:
        A configured `OfflineSpeakerDiarization`.

    Raises:
        RuntimeError: If a model cannot be downloaded, or if the engine
            rejects the configuration.
    """
    import sherpa_onnx

    segmentation = models.ensure(models.SPEAKER_SEGMENTATION)
    embedding = models.ensure(models.SPEAKER_EMBEDDING)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation),
            num_threads=THREADS,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=embedding, num_threads=THREADS),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=speakers if speakers and speakers > 0 else -1,
            threshold=SPEAKER_DISTANCE,
        ),
        min_duration_on=MIN_SPEECH_SECONDS,
        min_duration_off=MIN_PAUSE_SECONDS,
    )
    if not config.validate():
        raise RuntimeError("the speaker models were downloaded but the diarizer would not accept them")
    return sherpa_onnx.OfflineSpeakerDiarization(config)

def _voices(samples: array.array, turns: List[SpeakerTurn]) -> List[Voice]:
    """Measure what each voice in a recording sounds like.

    One vector per label, taken from that label's longest turns up to
    `VOICE_SECONDS`. The samples are the ones diarization already decoded, so
    this costs a few forward passes of a small model and no extra decoding.

    Args:
        samples: The recording's mono samples at `SAMPLE_RATE`.
        turns: The turns diarization found.

    Returns:
        One `Voice` per label, in label order. Empty when the embedding model
        cannot be had — a file whose voices were not measured simply cannot be
        joined to another, which is the same answer as never having run the
        speaker split at all.

    Raises:
        RuntimeError: If the model cannot be downloaded.
    """
    import sherpa_onnx

    if not turns:
        return []
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=models.ensure(models.SPEAKER_EMBEDDING), num_threads=THREADS,
    ))
    by_label: Dict[str, List[SpeakerTurn]] = {}
    for turn in turns:
        by_label.setdefault(turn.speaker, []).append(turn)

    voices: List[Voice] = []
    for label in sorted(by_label):
        taken, heard = array.array("f"), 0.0
        for turn in sorted(by_label[label], key=lambda item: item.start - item.end):
            if heard >= VOICE_SECONDS:
                break
            first = max(0, int(turn.start * SAMPLE_RATE))
            last = min(len(samples), int(turn.end * SAMPLE_RATE))
            if last <= first:
                continue
            taken.extend(samples[first:last])
            heard += (last - first) / SAMPLE_RATE
        if heard <= 0:
            continue
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, taken)
        stream.input_finished()
        voices.append(Voice(
            speaker=label, embedding=list(extractor.compute(stream)), seconds=round(heard, 3),
        ))
    return voices

def find_speakers(
    path: str,
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
    speakers: Optional[int] = None,
    ffmpeg_bin: str = "ffmpeg",
) -> Tuple[List[SpeakerTurn], List[Voice]]:
    """Work out how many voices are in a recording and when each one is talking.

    Args:
        path: Media file to read.
        on_progress: Called with the completed fraction.
        is_cancelled: Polled as the work proceeds, to stop early.
        speakers: How many people are talking, when that is known. Saying so
            for an interview or a two-hander is worth doing: told the number,
            the clustering has to find exactly that many, and cannot split one
            person who moved closer to the microphone into two.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        `(turns, voices)`. The turns are in time order, labelled `S1`, `S2` and
        so on, and those labels mean nothing outside this file — which is what
        the voices are for: one vector per label, so the same person can be
        recognised in another recording. A file with nobody in it gives
        neither.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If the models cannot be downloaded or the file cannot be
            decoded.
    """
    engine = _engine(speakers)
    samples = read_samples(path, ffmpeg_bin)
    if not samples:
        return [], []

    def report(done: int, total: int) -> int:
        """Pass progress on, and tell the engine to stop when asked to.

        Args:
            done: Chunks processed.
            total: Chunks in total.

        Returns:
            Non-zero to abort, which is how this engine takes cancellation.
        """
        if is_cancelled():
            return 1
        if total:
            on_progress(min(done / total, 1.0))
        return 0

    result = engine.process(samples, callback=report)
    if is_cancelled():
        raise OperationCancelled()
    turns = [
        SpeakerTurn(start=round(segment.start, 3), end=round(segment.end, 3), speaker=f"S{segment.speaker + 1}")
        for segment in result.sort_by_start_time()
    ]
    return turns, _voices(samples, turns)
