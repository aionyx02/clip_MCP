import bisect
import os
import re
from typing import Callable, Dict, List, Optional

from app.engine.ffmpeg import OperationCancelled, run_ffmpeg
from app.models.media import Asset, MediaAnalysis, Span, Transcript, TranscriptSegment, TranscriptWord

SCENE_THRESHOLD = 10.0
BLACK_MIN_SECONDS = 0.5
FREEZE_MIN_SECONDS = 2.0
SILENCE_NOISE_DB = -35
SILENCE_MIN_SECONDS = 0.3
MERGE_GAP_SECONDS = 0.05
SNAP_TOLERANCE_SECONDS = 0.5
SENTENCE_PAUSE_SECONDS = 0.6
SENTENCE_END_PUNCTUATION = "。！？!?…"
DEFAULT_WHISPER_MODEL = "large-v3-turbo"
# OpenCC configurations by BCP 47 tag. zh-TW and zh-HK also convert vocabulary to regional usage.
CHINESE_VARIANT_CONFIGS = {"zh-TW": "s2twp", "zh-HK": "s2hk", "zh-Hant": "s2t", "zh-Hans": "t2s"}
CHINESE_LANGUAGES = {"zh", "yue"}

ProgressCallback = Callable[[float, str], None]

def _spans(pairs: List[tuple], duration: float) -> List[Span]:
    """Build spans from start/end pairs, clamped to the media duration.

    Args:
        pairs: `(start, end)` tuples in seconds; `end` may be `None` for a span
            that lasts until the end of the media.
        duration: Media duration in seconds.

    Returns:
        Spans with a positive length, rounded to milliseconds.
    """
    spans = []
    for start, end in pairs:
        start = max(0.0, start)
        end = duration if end is None else min(end, duration) if duration else end
        if end > start:
            spans.append(Span(start=round(start, 3), end=round(end, 3)))
    return spans

def _merge_spans(spans: List[Span], max_gap: float = MERGE_GAP_SECONDS) -> List[Span]:
    """Join spans that are separated by only a tiny gap.

    Detection filters compare individual samples or frames against a
    threshold, so a single noisy sample splits one long silence into
    back-to-back pieces. Merging restores the real span.

    Args:
        spans: Spans in time order.
        max_gap: Largest gap in seconds that is bridged.

    Returns:
        The merged spans.
    """
    merged: List[Span] = []
    for span in spans:
        if merged and span.start - merged[-1].end <= max_gap:
            merged[-1] = Span(start=merged[-1].start, end=max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return merged

def _paired_events(log: str, start_pattern: str, end_pattern: str) -> List[tuple]:
    """Pair start and end events that FFmpeg detection filters write to their log.

    Args:
        log: FFmpeg log text.
        start_pattern: Regular expression whose first group is a start time.
        end_pattern: Regular expression whose first group is an end time.

    Returns:
        `(start, end)` tuples in log order. A start without a matching end gets
        `None` as its end.
    """
    events = sorted(
        [(match.start(), "start", float(match.group(1))) for match in re.finditer(start_pattern, log)]
        + [(match.start(), "end", float(match.group(1))) for match in re.finditer(end_pattern, log)]
    )
    pairs, open_start = [], None
    for _, kind, seconds in events:
        if kind == "start":
            open_start = seconds
        elif open_start is not None:
            pairs.append((open_start, seconds))
            open_start = None
    if open_start is not None:
        pairs.append((open_start, None))
    return pairs

def detect_events(
    asset: Asset,
    log_path: str,
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
    ffmpeg_bin: str = "ffmpeg",
) -> Dict[str, List[Span]]:
    """Detect scene changes, black and frozen picture, and silence in one decoding pass.

    Video is analyzed at a reduced size; detection runs on the first video
    stream that is not cover art and on the first audio stream.

    Args:
        asset: Asset to analyze.
        log_path: File that receives FFmpeg's log, which is parsed for results.
        on_progress: Called with the completed fraction.
        is_cancelled: Polled to stop early.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        A dictionary with `scenes`, `black_frames`, `frozen_frames`, and
        `silences` span lists. Lists for a missing stream type are empty.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If FFmpeg fails.
    """
    duration = float(asset.duration or 0)
    command = [ffmpeg_bin, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", "-i", asset.path]
    if asset.has_video:
        command += [
            "-map", "0:V:0",
            "-vf", (
                f"scale=320:-2,scdet=threshold={SCENE_THRESHOLD},"
                f"blackdetect=d={BLACK_MIN_SECONDS}:pix_th=0.10,"
                f"freezedetect=n=-60dB:d={FREEZE_MIN_SECONDS}"
            ),
            "-f", "null", "-",
        ]
    if asset.has_audio:
        command += ["-map", "0:a:0", "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_SECONDS}", "-f", "null", "-"]
    run_ffmpeg(command, log_path, duration, on_progress, is_cancelled)

    with open(log_path, encoding="utf-8", errors="replace") as log_file:
        log = log_file.read()
    scenes: List[Span] = []
    if asset.has_video:
        cuts = sorted({round(float(t), 3) for t in re.findall(r"lavfi\.scd\.time: ([\d.]+)", log)})
        boundaries = [0.0, *[cut for cut in cuts if 0 < cut < duration], duration]
        scenes = _spans(list(zip(boundaries, boundaries[1:])), duration)
    return {
        "scenes": scenes,
        "black_frames": _merge_spans(_spans([(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)], duration)),
        # Frozen spans are not merged: two different still pictures in a row are separate spans.
        "frozen_frames": _spans(_paired_events(log, r"freeze_start: ([\d.]+)", r"freeze_end: ([\d.]+)"), duration),
        "silences": _merge_spans(_spans(_paired_events(log, r"silence_start: (-?[\d.]+)", r"silence_end: ([\d.]+)"), duration)),
    }

def _nearest(values: List[float], target: float) -> Optional[float]:
    """Find the value closest to a target in a sorted list.

    Args:
        values: Sorted values.
        target: Value to approach.

    Returns:
        The closest value, or `None` if `values` is empty.
    """
    index = bisect.bisect_left(values, target)
    candidates = values[max(0, index - 1):index + 1]
    return min(candidates, key=lambda value: abs(value - target)) if candidates else None

def snap_words_to_silences(words: List[TranscriptWord], silences: List[Span], tolerance: float = SNAP_TOLERANCE_SECONDS) -> List[TranscriptWord]:
    """Move word boundaries that border a pause onto the pause's measured edges.

    Whisper's word timings come from attention alignment and are often a few
    hundred milliseconds off. Silence edges measured from the audio signal are
    far more precise, so a word that starts within `tolerance` of the end of a
    silence starts there instead, and a word that ends within `tolerance` of
    the start of a silence ends there. Words inside continuous speech keep
    their timings, and word order is never changed.

    Args:
        words: Words in time order. They are modified in place.
        silences: Detected silences.
        tolerance: Maximum distance in seconds between a word boundary and a
            silence edge for the boundary to move.

    Returns:
        The same list of words.
    """
    silence_ends = sorted(span.end for span in silences)
    silence_starts = sorted(span.start for span in silences)
    previous_end = 0.0
    for index, word in enumerate(words):
        next_start = words[index + 1].start if index + 1 < len(words) else float("inf")
        onset = _nearest(silence_ends, word.start)
        if onset is not None and abs(onset - word.start) <= tolerance and previous_end <= onset < word.end:
            word.start = round(onset, 3)
        offset = _nearest(silence_starts, word.end)
        if offset is not None and abs(offset - word.end) <= tolerance and word.start < offset <= next_start:
            word.end = round(offset, 3)
        previous_end = word.end
    return words

def split_sentences(words: List[TranscriptWord], pause: float = SENTENCE_PAUSE_SECONDS) -> List[TranscriptSegment]:
    """Group words into sentence-like segments.

    A segment ends after a word with sentence-final punctuation, or before a
    pause of at least `pause` seconds between two words.

    Args:
        words: Words in time order.
        pause: Minimum gap in seconds that starts a new segment.

    Returns:
        Segments whose text is the concatenation of their words.
    """
    segments: List[TranscriptSegment] = []
    current: List[TranscriptWord] = []

    def flush() -> None:
        """Close the segment being built, if it has any words."""
        text = "".join(word.text for word in current).strip()
        if text:
            segments.append(TranscriptSegment(start=current[0].start, end=current[-1].end, text=text, words=list(current)))
        current.clear()

    for word in words:
        if current and word.start - current[-1].end >= pause:
            flush()
        current.append(word)
        if word.text.strip()[-1:] in SENTENCE_END_PUNCTUATION:
            flush()
    flush()
    return segments

def convert_chinese_variant(segments: List[TranscriptSegment], variant: str) -> List[TranscriptSegment]:
    """Convert transcript text to a Chinese script and regional variant with OpenCC.

    Each segment is converted as a whole, so ambiguous characters are resolved
    from context; for example, 头发 becomes 頭髮 rather than 頭發. When the
    conversion keeps the segment's length, the converted characters are
    mapped back onto the words one to one. When a vocabulary replacement
    changes the length, each word is converted on its own instead, so timings
    stay attached to the right text.

    Args:
        segments: Transcript segments. They and their words are modified in
            place.
        variant: Target variant; a key of `CHINESE_VARIANT_CONFIGS`.

    Returns:
        The same segments, with `text` always equal to the concatenated words.

    Raises:
        ValueError: If `variant` is not supported.
    """
    if variant not in CHINESE_VARIANT_CONFIGS:
        raise ValueError(f"unsupported Chinese variant {variant!r}; use one of {', '.join(CHINESE_VARIANT_CONFIGS)}")
    import opencc

    converter = opencc.OpenCC(f"{CHINESE_VARIANT_CONFIGS[variant]}.json")
    for segment in segments:
        joined = "".join(word.text for word in segment.words)
        converted = converter.convert(joined)
        if len(converted) == len(joined):
            position = 0
            for word in segment.words:
                length = len(word.text)
                word.text = converted[position:position + length]
                position += length
        else:
            for word in segment.words:
                word.text = converter.convert(word.text)
        segment.text = "".join(word.text for word in segment.words).strip()
    return segments

def _whisper_device() -> str:
    """Choose the device for speech recognition.

    The `CLIP_MCP_WHISPER_DEVICE` environment variable selects `cuda` or
    `cpu`; the default, `auto`, uses CUDA when a GPU is available.

    Returns:
        `"cuda"` or `"cpu"`.
    """
    requested = os.environ.get("CLIP_MCP_WHISPER_DEVICE", "auto").lower()
    if requested != "auto":
        return requested
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

def _run_whisper(
    device: str,
    path: str,
    duration: float,
    language: Optional[str],
    prompt: Optional[str],
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
) -> Transcript:
    """Transcribe a file with faster-whisper on one device, without post-processing.

    Args:
        device: `"cuda"` or `"cpu"`.
        path: Media file to transcribe.
        duration: Media duration in seconds, used for progress.
        language: Language code, or `None` to detect it.
        prompt: Optional text that guides vocabulary and writing style.
        on_progress: Called with the completed fraction.
        is_cancelled: Polled between segments to stop early.

    Returns:
        A transcript with one segment per decoded chunk and raw word timings.

    Raises:
        OperationCancelled: If cancellation was requested.
    """
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model_name = os.environ.get("CLIP_MCP_WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
    model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
    # Batched decoding on CPU overflows CTranslate2's native stack on Windows, so the CPU decodes one chunk at a time.
    segments, info = BatchedInferencePipeline(model).transcribe(
        path,
        language=language,
        initial_prompt=prompt,
        word_timestamps=True,
        batch_size=8 if device == "cuda" else 1,
    )
    result: List[TranscriptSegment] = []
    for segment in segments:
        if is_cancelled():
            raise OperationCancelled()
        words = [TranscriptWord(start=round(w.start, 3), end=round(w.end, 3), text=w.word) for w in segment.words or []]
        result.append(TranscriptSegment(start=round(segment.start, 3), end=round(segment.end, 3), text=segment.text, words=words))
        if duration > 0:
            on_progress(min(segment.end / duration, 1.0))
    return Transcript(language=info.language, model=model_name, segments=result)

def transcribe_speech(
    path: str,
    duration: float,
    language: Optional[str],
    prompt: Optional[str],
    chinese_variant: Optional[str],
    silences: List[Span],
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
) -> Transcript:
    """Transcribe speech with word-level timestamps that stay accurate on long recordings.

    Speech is first split into chunks by voice activity detection, and each
    chunk is decoded independently at its absolute position. Timing errors
    therefore cannot accumulate across a long recording, and stretches of
    music or silence are skipped rather than filled with hallucinated text.
    Word boundaries next to pauses are then snapped to the measured silence
    edges, and words are regrouped into sentence-like segments. Chinese text is
    finally converted to `chinese_variant`, if given.

    The model is chosen with `CLIP_MCP_WHISPER_MODEL` (default
    `large-v3-turbo`, downloaded on first use) and the device with
    `CLIP_MCP_WHISPER_DEVICE`. If the GPU cannot be used, recognition falls
    back to the CPU.

    Args:
        path: Media file to transcribe.
        duration: Media duration in seconds, used for progress.
        language: Language code such as `"zh"` or `"en"`, or `None` to detect it.
        prompt: Optional text that guides vocabulary and writing style.
        chinese_variant: Chinese script and regional variant to convert the
            text to, such as `"zh-TW"`; `None` keeps the model's output.
            Ignored unless the detected language is Chinese.
        silences: Silences detected in the same file.
        on_progress: Called with the completed fraction.
        is_cancelled: Polled to stop early.

    Returns:
        The transcript.

    Raises:
        OperationCancelled: If cancellation was requested.
    """
    device = _whisper_device()
    try:
        raw = _run_whisper(device, path, duration, language, prompt, on_progress, is_cancelled)
    except RuntimeError as exc:
        if device != "cuda" or not any(name in str(exc).lower() for name in ("cuda", "cublas", "cudnn")):
            raise
        raw = _run_whisper("cpu", path, duration, language, prompt, on_progress, is_cancelled)

    words = [word for segment in raw.segments for word in segment.words]
    snap_words_to_silences(words, silences)
    segments = split_sentences(words)
    if chinese_variant is not None and raw.language in CHINESE_LANGUAGES:
        convert_chinese_variant(segments, chinese_variant)
    else:
        chinese_variant = None
    return Transcript(language=raw.language, model=raw.model, chinese_variant=chinese_variant, segments=segments)

def analyze_media(
    asset: Asset,
    work_dir: str,
    transcribe: bool,
    language: Optional[str],
    prompt: Optional[str],
    chinese_variant: Optional[str],
    on_progress: ProgressCallback,
    is_cancelled: Callable[[], bool],
    ffmpeg_bin: str = "ffmpeg",
) -> MediaAnalysis:
    """Describe an asset's content so that edits can be planned from it.

    Args:
        asset: Asset to analyze.
        work_dir: Directory for logs.
        transcribe: Whether to transcribe speech. Ignored for assets without
            audio.
        language: Spoken language code, or `None` to detect it.
        prompt: Optional text that guides transcription.
        chinese_variant: Chinese script and regional variant for the
            transcript, such as `"zh-TW"`; `None` keeps the model's output.
        on_progress: Called with the completed fraction and a short
            description of the current stage.
        is_cancelled: Polled to stop early.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        The analysis: scenes, black and frozen frames, silences, and the
        transcript if requested.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If FFmpeg or speech recognition fails.
    """
    duration = float(asset.duration or 0)
    with_speech = transcribe and asset.has_audio
    detection_share = 0.2 if with_speech else 1.0
    events = detect_events(
        asset,
        os.path.join(work_dir, "detect.log"),
        lambda fraction: on_progress(fraction * detection_share, "detecting scenes, black and frozen frames, and silences"),
        is_cancelled,
        ffmpeg_bin,
    )
    transcript = None
    if with_speech:
        on_progress(detection_share, "loading the speech recognition model")
        transcript = transcribe_speech(
            asset.path,
            duration,
            language,
            prompt,
            chinese_variant,
            events["silences"],
            lambda fraction: on_progress(detection_share + fraction * (1 - detection_share), "transcribing speech"),
            is_cancelled,
        )
    return MediaAnalysis(asset_id=asset.id, duration=duration, transcript=transcript, **events)
