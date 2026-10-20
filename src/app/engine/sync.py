"""A finished render's sound, checked against where the picture says it should be.

A file's picture and sound are two streams in one container, and they can
drift apart without anything looking wrong: the sound can start after the
picture and stay late for the whole cut, or a join can be a little to one
side of the voice it should land on. Neither shows up in a look at a frame,
and a viewer hears the second one as a cut that feels off without being able
to say why. The render is read once more, the sound alone, and where somebody
is talking is compared with where the cut said talking would be. What was
meant — a J cut, a line carried across a cut, a pause — is for whoever reads
the result to say; this only knows where the voice is, not why.
"""

import json
import os
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.engine.diarize import read_samples
from app.engine.probe import probe_file
from app.engine.relisten import voiced_stretches
from app.models.timeline.parts import Clip

SYNC_FILE_NAME = "sync.json"
# How far, in seconds either side of a join, the sound is compared. Provisional.
JOIN_WINDOW_SECONDS = 2.0
# The furthest the sound is looked for out of place, in seconds. Provisional.
MAX_LAG_SECONDS = 0.5
# The grid voicing is compared on, in seconds.
LAG_STEP_SECONDS = 0.01
# The voice detector decides in steps of this many seconds (512 samples at 16 kHz), so a lag
# smaller than one says nothing, whatever the frame rate.
VOICE_STEP_SECONDS = 0.032
# The share of the voice in the window that must line up at the best lag for it to be believed. Provisional.
MIN_AGREEMENT = 0.8

def stream_mismatch(probed: dict, frame: float) -> Optional[dict]:
    """Say how far a file's sound and picture streams differ, when they do.

    Args:
        probed: What `probe_file` read from the file.
        frame: One frame of the render, in seconds.

    Returns:
        `video_start`, `audio_start`, `video_length` and `audio_length`, each
        rounded to three places, when either start or either length differs by
        more than a frame; None when they line up, or a stream or how long it
        runs is not known.
    """
    streams = probed.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        return None
    whole = probed.get("format", {}).get("duration")
    video_length, audio_length = (float(length) if (length := stream.get("duration") or whole) else None
                                  for stream in (video, audio))
    if video_length is None or audio_length is None:
        return None
    video_start = float(video.get("start_time", 0) or 0)
    audio_start = float(audio.get("start_time", 0) or 0)
    if abs(video_start - audio_start) > frame or abs(video_length - audio_length) > frame:
        return {
            "video_start": round(video_start, 3),
            "audio_start": round(audio_start, 3),
            "video_length": round(video_length, 3),
            "audio_length": round(audio_length, 3),
        }
    return None

def expected_voicing(clips: Sequence[Clip],
                     voiced: Mapping[str, Sequence[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Work out where the cut says each source file's talking should be heard.

    A title card has no sound of its own, so its clip contributes nothing.

    Args:
        clips: The base video track's clips.
        voiced: Per asset ID, the source file's voiced stretches in seconds.

    Returns:
        `(start, end)` of each stretch on the timeline, in order, with
        overlapping or touching stretches merged and times rounded to three
        places.
    """
    found: List[Tuple[float, float]] = []
    for clip in clips:
        if clip.volume <= 0 or clip.card is not None or clip.asset_id not in voiced:
            continue
        begin = float(clip.timeline_in) - float(clip.audio_lead)
        source_start = float(clip.audio_source_start)
        source_end = float(clip.audio_source_end)
        speed = float(clip.speed)
        for stretch_start, stretch_end in voiced[clip.asset_id]:
            low, high = max(float(stretch_start), source_start), min(float(stretch_end), source_end)
            if high <= low:
                continue
            found.append((begin + (low - source_start) / speed, begin + (high - source_start) / speed))
    found.sort()
    merged: List[Tuple[float, float]] = []
    for start, end in found:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(round(start, 3), round(end, 3)) for start, end in merged]

def joins(clips: Sequence[Clip], voiced: Mapping[str, Sequence[Tuple[float, float]]]) -> List[float]:
    """Say where one shot hands over to the next, where both have sound and the voice there is known.

    A title card is silent, so a join into or out of one is left out. A file
    nobody listened to for voicing is silence to `expected_voicing`, so a
    voice heard from it would read as one out of place: a join is compared
    only when every clip heard within `JOIN_WINDOW_SECONDS` plus
    `MAX_LAG_SECONDS` of it was listened to. A clip up to `MAX_LAG_SECONDS`
    further out can still be pulled into the window `lag_at` searches, so the
    exclusion reaches that far too.

    Args:
        clips: The base video track's clips.
        voiced: Per asset ID, the source file's voiced stretches, as for
            `expected_voicing`.

    Returns:
        The `timeline_in` of each clip after the first, in timeline order,
        leaving out a join whose clip before or after it is muted or a card.
    """
    ordered = sorted(clips, key=lambda clip: clip.timeline_in)
    heard = [(float(clip.timeline_in) - float(clip.audio_lead), float(clip.timeline_out) + float(clip.audio_lag),
              clip.asset_id) for clip in ordered if clip.volume > 0 and clip.card is None]
    found: List[float] = []
    reach = JOIN_WINDOW_SECONDS + MAX_LAG_SECONDS
    for before, after in zip(ordered, ordered[1:]):
        at = float(after.timeline_in)
        if (before.volume > 0 and after.volume > 0 and before.card is None and after.card is None and all(
                asset_id in voiced for start, end, asset_id in heard
                if start < at + reach and end > at - reach)):
            found.append(at)
    return found

def _on(stretches: Sequence[Sequence[float]], moment: float) -> bool:
    """Whether a moment falls inside one of the stretches."""
    return any(start <= moment < end for start, end in stretches)

def lag_at(expected: Sequence[Sequence[float]], measured: Sequence[Sequence[float]], at: float) -> Optional[float]:
    """Say how far the sound at one join is heard from where the cut put it.

    Agreement at a lag is the share, among window samples where the expected
    voicing or the shifted measured voicing is on, of those where both are on.
    A lag with no sample on in either has an agreement of 0. A lag at the edge
    of the search may be a real offset larger than the search, so it is not
    reported.

    Args:
        expected: Where the cut says talking is, as `(start, end)` on the render.
        measured: Where the sound was heard talking, likewise.
        at: The join in seconds of the render.

    Returns:
        The lag in seconds whose voicing agrees with the cut's best, rounded to
        two places; positive when the sound is heard late. None when the window
        holds no change to line up, no lag agrees well enough, or the best lag
        is at the edge of what was searched.
    """
    # Only the stretches that can reach the window, so a long cut costs no more per join.
    reach = JOIN_WINDOW_SECONDS + MAX_LAG_SECONDS
    expected = [stretch for stretch in expected if stretch[1] > at - reach and stretch[0] < at + reach]
    measured = [stretch for stretch in measured if stretch[1] > at - reach and stretch[0] < at + reach]
    steps = int(round(JOIN_WINDOW_SECONDS * 2 / LAG_STEP_SECONDS)) + 1
    window = [at - JOIN_WINDOW_SECONDS + number * LAG_STEP_SECONDS for number in range(steps)]
    wanted = [_on(expected, moment) for moment in window]
    if all(value == wanted[0] for value in wanted):
        return None
    limit = int(round(MAX_LAG_SECONDS / LAG_STEP_SECONDS))
    best: Optional[Tuple[Tuple[float, float, int], float, float]] = None
    for number in range(-limit, limit + 1):
        lag = number * LAG_STEP_SECONDS
        heard = [_on(measured, moment + lag) for moment in window]
        either = sum(cut or sound for cut, sound in zip(wanted, heard))
        together = sum(cut and sound for cut, sound in zip(wanted, heard))
        agreement = together / either if either else 0.0
        # The best agreement first, then the smaller move, and a negative one before a positive.
        key = (-agreement, abs(lag), 0 if lag < 0 else 1)
        if best is None or key < best[0]:
            best = (key, lag, agreement)
    if best is None or best[2] < MIN_AGREEMENT:
        return None
    if abs(best[1]) >= MAX_LAG_SECONDS - LAG_STEP_SECONDS / 2:
        return None
    return round(best[1], 2)

def check(render_path: str, expected: Sequence[Sequence[float]], at: Sequence[float], frame: float,
          ffmpeg_bin: str, ffprobe_bin: str) -> dict:
    """Check a finished render's sound against its picture and the cut's own voicing.

    Args:
        render_path: The finished file.
        expected: Where the cut says talking is, as `(start, end)` on the render.
        at: The joins to compare, in seconds of the render.
        frame: One frame of the render, in seconds.
        ffmpeg_bin: The FFmpeg executable.
        ffprobe_bin: The ffprobe executable.

    Returns:
        `streams`, what `stream_mismatch` found or None; `joins`, `{at, lag}`
        for each join that could be lined up; `joins_asked`, how many joins were
        offered to the check; and `frame`, rounded to four places.
    """
    streams = stream_mismatch(probe_file(render_path, ffprobe_bin), frame)
    lags: List[dict] = []
    if expected and at:
        samples = np.frombuffer(read_samples(render_path, ffmpeg_bin), dtype=np.float32)
        measured = voiced_stretches(samples)
        lags = [{"at": round(moment, 3), "lag": lag}
                for moment in at if (lag := lag_at(expected, measured, moment)) is not None]
    return {"streams": streams, "joins": lags, "joins_asked": len(at), "frame": round(frame, 4)}

def keep(work_dir: str, found: dict) -> None:
    """Write what the check found next to the render."""
    with open(os.path.join(work_dir, SYNC_FILE_NAME), "w", encoding="utf-8") as handle:
        json.dump(found, handle)

def kept(work_dir: str) -> Optional[dict]:
    """Read what a render's check found; None when it was not checked, which is not the same as nothing found."""
    try:
        with open(os.path.join(work_dir, SYNC_FILE_NAME), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None
