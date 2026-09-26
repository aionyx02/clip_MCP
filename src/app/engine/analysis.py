import bisect
import functools
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.engine import faces, models, resources
from app.engine.diarize import (
    MIN_PAUSE_SECONDS as SPEAKER_MIN_PAUSE_SECONDS,
    MIN_SPEECH_SECONDS as SPEAKER_MIN_SPEECH_SECONDS,
    SPEAKER_DISTANCE,
    find_speakers,
    speaker_model_name,
)
from app.engine.ffmpeg import OperationCancelled, escape_filter_path, hidden_window_flags, run_ffmpeg
from app.engine.rhythm import measure_rhythm, rhythm_model_name
from app.models.media import (
    AnalysisRecipe,
    Asset,
    FaceMeasurement,
    MediaAnalysis,
    Rhythm,
    ShotMeasurement,
    SILENCE_DB,
    SoundMeasurement,
    Span,
    SpeakerTurn,
    Voice,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)

# Bumped whenever this pass measures something new or measures it differently, so that
# an analysis taken by an older version is reported as out of date rather than compared
# with a newer one as if the two were the same measurement.
# 2: exposure is measured on the full luma range, and motion no longer counts the cut
# into a shot as movement inside it.
ANALYSIS_VERSION = 2
SCENE_THRESHOLD = 10.0
BLACK_MIN_SECONDS = 0.5
FREEZE_MIN_SECONDS = 2.0
SILENCE_NOISE_DB = -35
SILENCE_MIN_SECONDS = 0.3
# Picture measurements: four samples a second is far more than exposure or blur need,
# and enough for motion to tell a still shot from one where something is happening.
PICTURE_FPS = 4
PICTURE_WIDTH = 480
# Shake is a fast wobble, so it has to be sampled fast; the camera's own movement is
# measured on a small picture because a displacement in frame widths does not depend
# on how many pixels the frame has.
CAMERA_FPS = 15
CAMERA_WIDTH = 320
# Full accuracy is for stabilising footage. This only measures how much it moves.
CAMERA_ACCURACY = 9
SOUND_WINDOW_SECONDS = 1
# JPEG quality for the stills the face detector reads. They are thrown away with the
# job, and a detector does not care about the last few percent of fidelity.
STILL_QUALITY = 5
SOUND_SAMPLE_RATE = 48000
MERGE_GAP_SECONDS = 0.05
SNAP_TOLERANCE_SECONDS = 0.5
SENTENCE_PAUSE_SECONDS = 0.6
SENTENCE_END_PUNCTUATION = "。！？!?…"
DEFAULT_WHISPER_MODEL = "large-v3-turbo"
# OpenCC configurations by BCP 47 tag. zh-TW and zh-HK also convert vocabulary to regional usage.
CHINESE_VARIANT_CONFIGS = {"zh-TW": "s2twp", "zh-HK": "s2hk", "zh-Hant": "s2t", "zh-Hans": "t2s"}
CHINESE_LANGUAGES = {"zh", "yue"}
PICTURE_FILE_NAME = "picture.txt"
CAMERA_FILE_NAME = "camera.trf"
SOUND_FILE_NAME = "sound.txt"
DETECT_LOG_FILE_NAME = "detect.log"

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

@dataclass(frozen=True)
class Scan:
    """Everything one decoding pass over a file measured.

    Attributes:
        scenes: The shots the file is made of.
        black_frames: Stretches of (nearly) black picture.
        frozen_frames: Stretches where the picture does not change.
        silences: Stretches without audible sound.
        shots: How each shot was shot, one record per shot that got samples.
        sound: What the sound was like, one record per second.
    """

    scenes: List[Span] = field(default_factory=list)
    black_frames: List[Span] = field(default_factory=list)
    frozen_frames: List[Span] = field(default_factory=list)
    silences: List[Span] = field(default_factory=list)
    shots: List[ShotMeasurement] = field(default_factory=list)
    sound: List[SoundMeasurement] = field(default_factory=list)

@functools.lru_cache(maxsize=None)
def can_measure_shake(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Say whether this FFmpeg can measure how much the camera moves.

    Camera movement comes from `vidstabdetect`, which is only built in when
    FFmpeg was configured with libvidstab, and is read back from the transforms
    file in the plain-text format its `fileformat` option asks for — older
    builds only write the binary one. A missing filter or a missing option
    would both fail the whole graph rather than one measurement in it, so both
    are checked before the graph is written. Asking once and remembering the
    answer keeps an extra process off every analysis.

    Args:
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        True when the filter is there and takes the option.
    """
    try:
        help_text = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-loglevel", "quiet", "-h", "filter=vidstabdetect"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            creationflags=hidden_window_flags(),
        ).stdout
    except OSError:
        return False
    return "vidstabdetect AVOptions" in help_text and "fileformat" in help_text

def current_recipe(
    speech_model: Optional[str],
    ffmpeg_bin: str = "ffmpeg",
    speaker_model: Optional[str] = None,
    rhythm_model: Optional[str] = None,
) -> AnalysisRecipe:
    """Describe the way this pass measures things right now.

    Stored with every analysis, so that one taken with different thresholds,
    with a different speech model, or on a machine that could not measure
    shake is recognised as answering a slightly different question.

    Args:
        speech_model: Speech model the transcript was made with, or `None`
            when speech was not transcribed.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.
        speaker_model: Models the speaker turns were found with, or `None`
            when voices were not told apart.
        rhythm_model: How the beat was found, or `None` when it was not
            looked for.

    Returns:
        The recipe.
    """
    detectors = {
        "scene_threshold": SCENE_THRESHOLD,
        "black_min_seconds": BLACK_MIN_SECONDS,
        "freeze_min_seconds": FREEZE_MIN_SECONDS,
        "silence_noise_db": SILENCE_NOISE_DB,
        "silence_min_seconds": SILENCE_MIN_SECONDS,
        "picture_fps": PICTURE_FPS,
        "picture_width": PICTURE_WIDTH,
        "camera_fps": CAMERA_FPS,
        "camera_width": CAMERA_WIDTH,
        "camera_accuracy": CAMERA_ACCURACY,
        "sound_window_seconds": SOUND_WINDOW_SECONDS,
        "shake": can_measure_shake(ffmpeg_bin),
        "face_fps": faces.FRAME_FPS,
        "face_width": faces.FRAME_WIDTH,
        "face_score_threshold": faces.SCORE_THRESHOLD,
        # Every threshold diarization runs with. These are the provisional numbers §13 of
        # the roadmap expects to be tuned against a corpus, so an analysis taken under one
        # set of them has to be recognisable from one taken under another.
        "speaker_distance": SPEAKER_DISTANCE,
        "speaker_min_speech_seconds": SPEAKER_MIN_SPEECH_SECONDS,
        "speaker_min_pause_seconds": SPEAKER_MIN_PAUSE_SECONDS,
    }
    fingerprint = hashlib.sha256(json.dumps(detectors, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return AnalysisRecipe(
        version=ANALYSIS_VERSION,
        detectors=fingerprint.hexdigest()[:16],
        speech_model=speech_model,
        speaker_model=speaker_model,
        rhythm_model=rhythm_model,
    )

def _scan_graph(asset: Asset, work_dir: str, with_shake: bool) -> Tuple[str, List[List[str]]]:
    """Write the filtergraph that measures everything in one decoding pass.

    Each stream is split so that the detectors keep seeing the picture and the
    sound they were written against, while the measurements run beside them at
    their own sampling rates. Everything a filter writes to a file goes through
    `escape_filter_path`, because an ordinary Windows path is filtergraph
    syntax.

    Args:
        asset: Asset to scan.
        work_dir: Directory the measurement files are written to.
        with_shake: Whether to include the camera movement branch.

    Returns:
        The filtergraph, and the FFmpeg arguments for each output it feeds.
        Most of those outputs go nowhere — the filters they carry write their
        own files — but the stills the face detector reads are a real output,
        because no filter hands a whole frame back.
    """
    chains: List[str] = []
    outputs: List[List[str]] = []
    if asset.has_video:
        picture = escape_filter_path(os.path.join(work_dir, PICTURE_FILE_NAME))
        branches = ["[detect]", "[picture]", "[stills]"] + (["[camera]"] if with_shake else [])
        chains.append(f"[0:V:0]split={len(branches)}" + "".join(branches))
        chains.append(
            "[detect]scale=320:-2,"
            f"scdet=threshold={SCENE_THRESHOLD},"
            f"blackdetect=d={BLACK_MIN_SECONDS}:pix_th=0.10,"
            f"freezedetect=n=-60dB:d={FREEZE_MIN_SECONDS}[seen]"
        )
        # out_range=full, because most footage carries luma in the 16-235 studio range and
        # signalstats reports it raw: without this, black measures 0.06 and white 0.92, and a
        # full-range file and a studio-range one are not comparable at all.
        chains.append(
            f"[picture]fps={PICTURE_FPS},scale={PICTURE_WIDTH}:-2:out_range=full,signalstats,blurdetect,"
            f"metadata=mode=print:file='{picture}',nullsink"
        )
        chains.append(f"[stills]fps={faces.FRAME_FPS},scale={faces.FRAME_WIDTH}:-2[still]")
        if with_shake:
            camera = escape_filter_path(os.path.join(work_dir, CAMERA_FILE_NAME))
            chains.append(
                f"[camera]fps={CAMERA_FPS},scale={CAMERA_WIDTH}:-2,"
                f"vidstabdetect=accuracy={CAMERA_ACCURACY}:fileformat=ascii:result='{camera}',nullsink"
            )
        outputs.append(["-map", "[seen]", "-f", "null", "-"])
        # A path in an output argument is an ordinary argument, so it needs no escaping.
        outputs.append([
            "-map", "[still]", "-q:v", str(STILL_QUALITY), "-f", "image2",
            os.path.join(faces.frames_dir(work_dir), faces.FRAME_PATTERN),
        ])
    if asset.has_audio:
        sound = escape_filter_path(os.path.join(work_dir, SOUND_FILE_NAME))
        chains.append("[0:a:0]asplit=2[quiet][level]")
        chains.append(f"[quiet]silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_SECONDS}[heard]")
        # astats only reports an overall noise floor while it is also measuring one per
        # channel, so the per-channel request below is what makes the overall one appear.
        chains.append(
            f"[level]aresample={SOUND_SAMPLE_RATE},asetnsamples=n={SOUND_SAMPLE_RATE * SOUND_WINDOW_SECONDS},"
            "astats=metadata=1:reset=1:measure_perchannel=Noise_floor"
            ":measure_overall=Peak_level+RMS_level+Noise_floor+Flat_factor,"
            f"ametadata=mode=print:file='{sound}',anullsink"
        )
        outputs.append(["-map", "[heard]", "-f", "null", "-"])
    return ";".join(chains), outputs

def _frame_records(path: str) -> List[Tuple[float, Dict[str, float]]]:
    """Read what a `metadata` filter printed, one record per sampled frame.

    Args:
        path: File the filter wrote.

    Returns:
        `(seconds, values)` in time order, where `values` holds that frame's
        numeric metadata under its full key name. An empty list when the file
        is not there, which is what a stream the graph did not measure leaves
        behind. A value that is not a finite number is dropped: a filter
        reports one where there was nothing to measure, such as blur on a
        frame with no detail in it, and that is not a measurement of zero.

        A frame every one of whose values was dropped still gets a record,
        with nothing in it. A second of digital silence reads that way — every
        level in it is minus infinity decibels — and it has to stay
        distinguishable from a second that was never measured at all.
    """
    records: List[Tuple[float, Dict[str, float]]] = []
    values: Dict[str, float] = {}
    seconds = 0.0
    started = False
    try:
        lines = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return records
    with lines:
        for line in lines:
            if line.startswith("frame:"):
                if started:
                    records.append((seconds, values))
                match = re.search(r"pts_time:(-?[\d.]+)", line)
                values, seconds, started = {}, float(match.group(1)) if match else 0.0, True
            else:
                key, _, text = line.strip().partition("=")
                try:
                    number = float(text)
                except ValueError:
                    continue
                if math.isfinite(number):
                    values[key] = number
    if started:
        records.append((seconds, values))
    return records

_LOCAL_MOTION = re.compile(r"\(LM (-?\d+) (-?\d+) ")

def _camera_movement(path: str, fps: int = CAMERA_FPS, width: int = CAMERA_WIDTH) -> List[Tuple[float, float, float]]:
    """Read how far the camera itself moved between each pair of sampled frames.

    `vidstabdetect` writes one line per frame holding the motion it measured in
    a grid of small fields. Fields land on a moving subject as readily as on
    the background, so the camera's own movement is the median of them rather
    than their mean: half the picture has to agree before the camera is said to
    have moved. The file carries frame numbers and no times, which is why the
    branch that produced it was resampled to a fixed rate.

    Args:
        path: Transforms file the filter wrote, in its ASCII format.
        fps: Rate the branch was resampled to.
        width: Width the branch was scaled to.

    Returns:
        `(seconds, dx, dy)` in time order, the displacements in frame widths.
    """
    movement: List[Tuple[float, float, float]] = []
    try:
        lines = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return movement
    with lines:
        for line in lines:
            if not line.startswith("Frame "):
                continue
            seconds = (int(line.split(None, 2)[1]) - 1) / fps
            fields = _LOCAL_MOTION.findall(line)
            if not fields:
                movement.append((seconds, 0.0, 0.0))
                continue
            movement.append((
                seconds,
                statistics.median(int(x) for x, _ in fields) / width,
                statistics.median(int(y) for _, y in fields) / width,
            ))
    return movement

def _shake(movement: Sequence[Tuple[float, float, float]], start: float, end: float) -> Optional[float]:
    """Measure how unsteady the camera was over a stretch.

    A camera held still and a camera panning evenly can both be moving by the
    same amount from frame to frame, so the movement itself does not separate
    them. What separates them is how much that movement changes: an even pan
    barely changes, and a hand does nothing else.

    Measured from the second sample of the stretch onwards. The first one was
    taken across the cut into it and would make every shot after a cut look
    unsteady.

    Args:
        movement: Camera movement over the whole file, as `_camera_movement`
            returns it.
        start: Start of the stretch in seconds.
        end: End of the stretch in seconds.

    Returns:
        The mean change in displacement, in frame widths, or `None` when the
        stretch holds too few samples to measure a change from.
    """
    # The first sample of a shot holds the camera's movement across the cut into it, which
    # is the size of the edit rather than anything the camera did. It is dropped, and so is
    # the change that would otherwise be computed from it.
    inside = [(dx, dy) for seconds, dx, dy in movement if start <= seconds < end][1:]
    if len(inside) < 2:
        return None
    return statistics.fmean(
        math.hypot(inside[index][0] - inside[index - 1][0], inside[index][1] - inside[index - 1][1])
        for index in range(1, len(inside))
    )

def _shot_measurements(
    scenes: Sequence[Span],
    records: Sequence[Tuple[float, Dict[str, float]]],
    movement: Sequence[Tuple[float, float, float]],
) -> List[ShotMeasurement]:
    """Fold the sampled frames into one record per shot.

    Args:
        scenes: The shots of the file.
        records: Sampled picture measurements, as `_frame_records` returns them.
        movement: Camera movement, as `_camera_movement` returns it.

    Returns:
        One measurement per shot that a frame was sampled in. A shot shorter
        than the gap between two samples gets none at all, which reads as not
        measured rather than as measured and found unremarkable. A shot with
        only one sample in it gets no `motion`, and one with fewer than three
        no `shake`, for the same reason: both are changes between samples, and
        the change across the cut into the shot is not one of them.
    """
    shots: List[ShotMeasurement] = []
    for scene in scenes:
        inside = [
            values for seconds, values in records
            if scene.start <= seconds < scene.end and "lavfi.signalstats.YAVG" in values
        ]
        if not inside:
            continue
        blurs = [values["lavfi.blur"] for values in inside if "lavfi.blur" in values]
        # How much the picture changed is measured against the frame before it, so a shot's
        # first sample holds the size of the cut into it rather than anything happening in
        # it. Dropped: a solid white shot after a black one measured as 0.107 of a frame's
        # worth of movement, which is a shot where nothing happens at all.
        moving = inside[1:]
        shake = _shake(movement, scene.start, scene.end)
        shots.append(ShotMeasurement(
            start=scene.start,
            end=scene.end,
            exposure=round(statistics.fmean(v["lavfi.signalstats.YAVG"] for v in inside) / 255, 4),
            contrast=round(
                statistics.fmean(v["lavfi.signalstats.YHIGH"] - v["lavfi.signalstats.YLOW"] for v in inside) / 255, 4
            ),
            blur=round(statistics.fmean(blurs), 3) if blurs else None,
            motion=round(statistics.fmean(v["lavfi.signalstats.YDIF"] for v in moving) / 255, 4) if moving else None,
            shake=None if shake is None else round(shake, 5),
        ))
    return shots

def _sound_measurements(records: Sequence[Tuple[float, Dict[str, float]]], duration: float) -> List[SoundMeasurement]:
    """Turn the per-second sound statistics into records.

    Args:
        records: Sampled sound measurements, as `_frame_records` returns them.
        duration: Length of the file in seconds, which the last window is cut
            to.

    Returns:
        One measurement per window. Digital silence measures as minus infinity
        decibels, which `_frame_records` drops, so a window that reports no
        level at all is silence and is written as `SILENCE_DB`.
    """
    measurements: List[SoundMeasurement] = []
    for seconds, values in records:
        end = seconds + SOUND_WINDOW_SECONDS
        measurements.append(SoundMeasurement(
            start=round(seconds, 3),
            end=round(min(end, duration) if duration else end, 3),
            loudness=round(values.get("lavfi.astats.Overall.RMS_level", SILENCE_DB), 2),
            peak=round(values.get("lavfi.astats.Overall.Peak_level", SILENCE_DB), 2),
            noise_floor=round(values.get("lavfi.astats.Overall.Noise_floor", SILENCE_DB), 2),
            flatness=round(values.get("lavfi.astats.Overall.Flat_factor", 0.0), 3),
        ))
    return measurements

def sound_note(windows: Sequence[SoundMeasurement]) -> Optional[dict]:
    """Sum up a stretch of per-second sound measurements in a few numbers.

    An hour of footage is three and a half thousand windows, which is not
    something to read. What a decision actually rests on is the range: how
    quiet the quietest part got, how close the loudest came to the ceiling,
    and how high the hiss underneath sits.

    Args:
        windows: The per-second measurements overlapping the stretch.

    Returns:
        The summary, in decibels relative to full scale, or `None` when
        nothing was measured over the stretch. Narrow the range and ask again
        to find where a bad patch is.
    """
    if not windows:
        return None
    return {
        "windows": len(windows),
        "loudness_min": min(window.loudness for window in windows),
        "loudness_median": round(statistics.median(window.loudness for window in windows), 2),
        "loudness_max": max(window.loudness for window in windows),
        "noise_floor_median": round(statistics.median(window.noise_floor for window in windows), 2),
        "peak_max": max(window.peak for window in windows),
        "flatness_max": max(window.flatness for window in windows),
    }

def scan_media(
    asset: Asset,
    work_dir: str,
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
    ffmpeg_bin: str = "ffmpeg",
) -> Scan:
    """Detect events and measure picture and sound in one decoding pass.

    Decoding a long file is the expensive part, so everything that needs the
    frames reads them from the same pass rather than each scanning the file
    again: scene, black, frozen and silence detection, and alongside them
    exposure, contrast, blur, motion, camera shake and per-second levels.
    Detection runs on the first video stream that is not cover art and on the
    first audio stream.

    Args:
        asset: Asset to scan.
        work_dir: Directory that receives FFmpeg's log and the files its
            measuring filters write, both of which are parsed for the results.
        on_progress: Called with the completed fraction.
        is_cancelled: Polled to stop early.
        ffmpeg_bin: Path to, or name of, the FFmpeg executable.

    Returns:
        The scan. Lists for a missing stream type are empty.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If FFmpeg fails.
    """
    duration = float(asset.duration or 0)
    with_shake = asset.has_video and can_measure_shake(ffmpeg_bin)
    if asset.has_video:
        os.makedirs(faces.frames_dir(work_dir), exist_ok=True)
    graph, outputs = _scan_graph(asset, work_dir, with_shake)
    log_path = os.path.join(work_dir, DETECT_LOG_FILE_NAME)
    command = [ffmpeg_bin, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", "-i", asset.path]
    if graph:
        command += ["-filter_complex", graph]
    for output in outputs:
        command += output
    run_ffmpeg(command, log_path, duration, on_progress, is_cancelled)

    with open(log_path, encoding="utf-8", errors="replace") as log_file:
        log = log_file.read()
    scenes: List[Span] = []
    if asset.has_video:
        cuts = sorted({round(float(t), 3) for t in re.findall(r"lavfi\.scd\.time: ([\d.]+)", log)})
        boundaries = [0.0, *[cut for cut in cuts if 0 < cut < duration], duration]
        scenes = _spans(list(zip(boundaries, boundaries[1:])), duration)
    return Scan(
        scenes=scenes,
        black_frames=_merge_spans(_spans([(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)], duration)),
        # Frozen spans are not merged: two different still pictures in a row are separate spans.
        frozen_frames=_spans(_paired_events(log, r"freeze_start: ([\d.]+)", r"freeze_end: ([\d.]+)"), duration),
        silences=_merge_spans(_spans(_paired_events(log, r"silence_start: (-?[\d.]+)", r"silence_end: ([\d.]+)"), duration)),
        shots=_shot_measurements(
            scenes,
            _frame_records(os.path.join(work_dir, PICTURE_FILE_NAME)),
            _camera_movement(os.path.join(work_dir, CAMERA_FILE_NAME)) if with_shake else [],
        ),
        sound=_sound_measurements(_frame_records(os.path.join(work_dir, SOUND_FILE_NAME)), duration),
    )

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

def whisper_model_name() -> str:
    """Name of the speech recognition model that transcription will load.

    Set by `CLIP_MCP_WHISPER_MODEL`. Callers use it to size the memory a
    transcription needs before starting one.

    Returns:
        The model name, such as `large-v3-turbo`, or a path to a local model.
    """
    return os.environ.get("CLIP_MCP_WHISPER_MODEL", DEFAULT_WHISPER_MODEL)

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
        RuntimeError: If the machine has less free memory than the model
            needs, or if the device cannot run it.
    """
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model_name = whisper_model_name()
    # Only on the CPU do the weights sit in system memory; on the GPU they are in VRAM,
    # and a card that is too small raises its own error and falls back here.
    needed = resources.whisper_memory_bytes(model_name)
    available = resources.memory_status() if device == "cpu" else None
    if available is not None and available[0] < needed:
        raise RuntimeError(
            f"{model_name} needs about {needed // resources.MEGABYTE} MB to transcribe and only "
            f"{available[0] // resources.MEGABYTE} MB is free; close something, or set "
            "CLIP_MCP_WHISPER_MODEL to a smaller model such as small"
        )
    # Kept in the workspace rather than in a cache under the user's home directory, so
    # that the project holds everything it needs and deleting it takes the weights with it.
    model = WhisperModel(
        model_name,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        download_root=models.whisper_dir(),
    )
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
            # float(), because what the decoder reports is not always an ordinary one and
            # this number is stored on the job.
            on_progress(min(float(segment.end) / duration, 1.0))
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
    `large-v3-turbo`, downloaded into the workspace on first use) and the
    device with `CLIP_MCP_WHISPER_DEVICE`. If the GPU cannot be used,
    recognition falls back to the CPU.

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

def discard_scratch(work_dir: str) -> None:
    """Throw away everything the pass wrote for its own use.

    The stills the face detector reads, the frame and sound metadata and the
    camera transforms are all scaffolding: about 35 MB an hour between them,
    on top of a couple of hundred for the stills, and nothing reads any of it
    once the analysis has been built. Nothing else ever clears a job's directory,
    so leaving them means a workspace that grows with every analysis and never
    shrinks.

    FFmpeg's log stays. It is a few kilobytes, and it is the one thing worth
    having when an analysis has failed.

    Args:
        work_dir: The job's working directory.
    """
    faces.discard_frames(work_dir)
    for name in (PICTURE_FILE_NAME, CAMERA_FILE_NAME, SOUND_FILE_NAME):
        try:
            os.remove(os.path.join(work_dir, name))
        except OSError:
            pass

def _shares(
    with_faces: bool, with_speech: bool, with_speakers: bool, with_rhythm: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """Divide one progress bar among the stages this analysis will actually run.

    The weights are roughly what each stage costs against the others, measured
    loosely and not worth measuring precisely: they exist so that a bar does
    not sit at 20% for ten minutes, not so that it is accurate.

    Args:
        with_faces: Whether faces will be looked for.
        with_speech: Whether speech will be transcribed.
        with_speakers: Whether voices will be told apart.
        with_rhythm: Whether the beat will be looked for.

    Returns:
        `(start, span)` of the bar for each stage that will run, keyed by
        stage name, covering 0.0 to 1.0 between them.
    """
    weights = [("scan", 4.0)]
    if with_faces:
        weights.append(("faces", 2.0))
    if with_speech:
        weights.append(("speech", 20.0))
    if with_speakers:
        weights.append(("speakers", 4.0))
    if with_rhythm:
        weights.append(("rhythm", 1.0))
    total = sum(weight for _, weight in weights)
    shares: Dict[str, Tuple[float, float]] = {}
    cursor = 0.0
    for name, weight in weights:
        shares[name] = (cursor / total, weight / total)
        cursor += weight
    return shares

def _fetch_models(needed: Sequence[models.Model], on_progress: ProgressCallback) -> None:
    """Download whatever models this analysis needs and does not have yet.

    Reported as its own stage rather than as progress through the file,
    because that is what it is: a one-time setup step that the next analysis
    will not repeat. A model that cannot be downloaded fails the job and says
    which one — a download that fails is a passing problem worth retrying, and
    quietly producing an analysis with a measurement missing from it would
    leave something that looks finished and is not.

    Args:
        needed: Models the analysis will use.
        on_progress: Called with the completed fraction and the current stage.

    Raises:
        RuntimeError: If a model cannot be downloaded or is not what it should be.
    """
    for model in needed:
        if models.is_downloaded(model):
            continue
        megabytes = max(1, model.size // models.MEGABYTE)
        on_progress(0.0, f"downloading the {model.name} model, {megabytes} MB, once")
        models.ensure(model, lambda fraction: on_progress(
            0.0, f"downloading the {model.name} model, {megabytes} MB, once: {fraction:.0%}"
        ))

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
    diarize: bool = True,
    speakers: Optional[int] = None,
) -> MediaAnalysis:
    """Describe an asset's content so that edits can be planned from it.

    Args:
        asset: Asset to analyze.
        work_dir: Directory for the log, the measurement files, and the stills
            the face detector reads.
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
        diarize: Whether to tell the voices apart. Only done alongside a
            transcript, since a speaker label with nothing said under it has
            nowhere to go.
        speakers: How many people are talking, when that is known.

    Returns:
        The analysis: scenes, black and frozen frames, silences, how each shot
        was shot, what the sound was like second by second, who was on screen,
        the transcript and the speaker turns if they were asked for, where
        the beat falls if the file is music, and the recipe all of it was
        measured with.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If FFmpeg fails, a model cannot be downloaded, or speech
            recognition fails.
    """
    duration = float(asset.duration or 0)
    with_speech = transcribe and asset.has_audio
    with_speakers = with_speech and diarize
    with_faces = asset.has_video
    # Music is a file with sound and no picture. The beat is looked for there and nowhere
    # else: a beat tracker run over somebody talking finds one in their syllables.
    with_rhythm = asset.has_audio and not asset.has_video
    shares = _shares(with_faces, with_speech, with_speakers, with_rhythm)

    needed: List[models.Model] = []
    if with_faces:
        needed.append(models.FACE_DETECTION)
    if with_speakers:
        needed += [models.SPEAKER_SEGMENTATION, models.SPEAKER_EMBEDDING]
    _fetch_models(needed, on_progress)

    def report(stage: str, text: str) -> Callable[[float], None]:
        """Build the progress callback for one stage of the analysis.

        Args:
            stage: Which stage, as `_shares` keys it.
            text: What to tell the caller is happening.

        Returns:
            A callback taking that stage's own completed fraction.
        """
        start, span = shares[stage]
        return lambda fraction: on_progress(start + fraction * span, text)

    # Everything the pass writes for itself is thrown away on the way out, whichever way
    # out it takes. Cancelling during the scan is the likeliest moment of all, and it is
    # the only stage there is for footage analysed without speech.
    try:
        scan = scan_media(
            asset,
            work_dir,
            report("scan", "detecting scenes, silences and unusable frames, and measuring picture and sound"),
            is_cancelled,
            ffmpeg_bin,
        )
        on_screen: List[FaceMeasurement] = []
        if with_faces:
            on_screen = faces.detect_faces(work_dir, duration, report("faces", "looking for faces"), is_cancelled)
    finally:
        discard_scratch(work_dir)

    transcript = None
    turns: List[SpeakerTurn] = []
    voices: List[Voice] = []
    if with_speech:
        on_progress(shares["speech"][0], "loading the speech recognition model")
        transcript = transcribe_speech(
            asset.path,
            duration,
            language,
            prompt,
            chinese_variant,
            scan.silences,
            report("speech", "transcribing speech"),
            is_cancelled,
        )
    if with_speakers:
        turns, voices = find_speakers(
            asset.path, report("speakers", "telling the voices apart"), is_cancelled, speakers, ffmpeg_bin
        )
    rhythm: Optional[Rhythm] = None
    if with_rhythm:
        rhythm = measure_rhythm(asset.path, report("rhythm", "finding the beat"), is_cancelled, ffmpeg_bin)
    return MediaAnalysis(
        asset_id=asset.id,
        duration=duration,
        scenes=scan.scenes,
        black_frames=scan.black_frames,
        frozen_frames=scan.frozen_frames,
        silences=scan.silences,
        shots=scan.shots,
        sound=scan.sound,
        faces=on_screen,
        speakers=turns,
        voices=voices,
        rhythm=rhythm,
        transcript=transcript,
        recipe=current_recipe(
            transcript.model if transcript is not None else None,
            ffmpeg_bin,
            speaker_model_name() if with_speakers else None,
            rhythm_model_name() if with_rhythm else None,
        ),
    )
