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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.engine.diarize import read_samples
from app.engine.probe import probe_file
from app.engine.relisten import voiced_stretches
from app.models.timeline.parts import Clip, Track

SYNC_FILE_NAME = "sync.json"
# How far, in seconds either side of a join, the sound is compared. Provisional.
JOIN_WINDOW_SECONDS = 2.0
# The shortest stretch, in seconds, a side of a join is compared over; less holds too little voice to line up.
# Provisional.
MIN_SIDE_SECONDS = 0.5
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

def _merged(stretches: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Join stretches that overlap or touch into one each, in order."""
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(stretches):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged

def expected_voicing(clips: Sequence[Clip],
                     voiced: Mapping[str, Sequence[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Work out where the cut says each source file's talking should be heard.

    A title card has no sound of its own, so its clip contributes nothing.

    Args:
        clips: The clips whose voicing to work out, in timeline order.
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
    return [(round(start, 3), round(end, 3)) for start, end in _merged(found)]

def _sound_span(clip: Clip) -> Tuple[float, float]:
    """Say where on the timeline one clip's own sound plays."""
    return float(clip.audio_timeline_in), float(clip.audio_timeline_out)

def _clear_stretch(window: Tuple[float, float], at: float, others: Sequence[Tuple[float, float]],
                   side: str) -> Tuple[float, float]:
    """Find the part of a side's window nearest the join that holds none of the other sound.

    Sound reaching over the join from the other side pulls the clear stretch
    back to its boundary: a lead on the next clip leaves the rest of the
    before side to compare, and a lag on the last one leaves the rest of the
    after side. Other sound that covers the whole side leaves nothing.

    Args:
        window: The side's window, `(start, end)` on the timeline.
        at: Where the join is.
        others: Where every other heard clip's sound plays.
        side: `"before"` or `"after"`.

    Returns:
        `(start, end)` of the clear stretch, empty when there is none.
    """
    start, end = window
    others = _merged(others)
    if side == "before":
        for low, high in others:
            if low < at < high:
                end = min(end, low)
        clear = max((high for low, high in others if high <= end), default=start)
        return max(start, clear), min(at, end)
    for low, high in others:
        if low < at < high:
            start = max(start, high)
    clear = min((low for low, high in others if low >= start), default=end)
    return max(at, start), min(end, clear)

def sides(project_tracks: Sequence[Track], base_clips: Sequence[Clip],
          voiced: Mapping[str, Sequence[Tuple[float, float]]]) -> Tuple[List[dict], Dict[str, int]]:
    """Say which sides of which joins have a stretch of their own sound to line up.

    A join has two sides, the clip before it and the clip after it. Each side
    is compared over `JOIN_WINDOW_SECONDS` on its own side of the join, cut
    down to where that clip is the only sound playing: music or another
    picture track over it, and the neighbouring clip's own lead or lag, leave
    nothing there or a shorter stretch further from the join.

    Args:
        project_tracks: Every track of the project, so that sound from a
            picture track drawn above or from an audio track is counted.
        base_clips: The base video track's clips, which meet at the joins.
        voiced: Per asset ID, the source file's voiced stretches, as for
            `expected_voicing`.

    Returns:
        The sides offered to the check, in timeline order with `before` before
        `after` at one join: `{at, clip_id, side, start, end, expected}`, its
        `expected` being that clip's voicing only; and how many sides were
        left out, as `{not_listened, too_short}`.
    """
    counts = {"not_listened": 0, "too_short": 0}
    offered: List[dict] = []
    ordered = sorted(base_clips, key=lambda clip: clip.timeline_in)
    heard = [clip for track in project_tracks for clip in track.clips if clip.volume > 0 and clip.card is None]
    for before, after in zip(ordered, ordered[1:]):
        at = float(after.timeline_in)
        for side, clip in (("before", before), ("after", after)):
            if clip.volume <= 0 or clip.card is not None:
                continue
            if clip.asset_id not in voiced:
                counts["not_listened"] += 1
                continue
            window = (at - JOIN_WINDOW_SECONDS, at) if side == "before" else (at, at + JOIN_WINDOW_SECONDS)
            others = [_sound_span(other) for other in heard if other is not clip]
            start, end = _clear_stretch(window, at, others, side)
            own_start, own_end = _sound_span(clip)
            start, end = max(start, own_start), min(end, own_end)
            if end - start < MIN_SIDE_SECONDS:
                counts["too_short"] += 1
                continue
            offered.append({
                "at": round(at, 3), "clip_id": clip.id, "side": side,
                "start": round(start, 3), "end": round(end, 3),
                "expected": [list(stretch) for stretch in expected_voicing([clip], voiced)],
            })
    return offered, counts

def _on(stretches: Sequence[Sequence[float]], moment: float) -> bool:
    """Whether a moment falls inside one of the stretches."""
    return any(start <= moment < end for start, end in stretches)

def lag_at(expected: Sequence[Sequence[float]], measured: Sequence[Sequence[float]], start: float,
           end: float) -> Optional[float]:
    """Say how far the sound on one side of a join is heard from where the cut put it.

    Agreement at a lag is the share, among window samples where the expected
    voicing or the shifted measured voicing is on, of those where both are on.
    A lag with no sample on in either has an agreement of 0. A lag at the edge
    of the search may be a real offset larger than the search, so it is not
    reported.

    Args:
        expected: Where the cut says talking is, as `(start, end)` on the render.
        measured: Where the sound was heard talking, likewise.
        start: Where the side's window begins, in seconds of the render.
        end: Where it ends.

    Returns:
        The lag in seconds whose voicing agrees with the cut's best, rounded to
        two places; positive when the sound is heard late. None when the window
        holds no change to line up, no lag agrees well enough, or the best lag
        is at the edge of what was searched.
    """
    # Only the stretches that can reach the window, so a long cut costs no more per side.
    expected = [stretch for stretch in expected
                if stretch[1] > start - MAX_LAG_SECONDS and stretch[0] < end + MAX_LAG_SECONDS]
    measured = [stretch for stretch in measured
                if stretch[1] > start - MAX_LAG_SECONDS and stretch[0] < end + MAX_LAG_SECONDS]
    steps = int(round((end - start) / LAG_STEP_SECONDS)) + 1
    window = [start + number * LAG_STEP_SECONDS for number in range(steps)]
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

def check(render_path: str, offered_sides: Sequence[dict], left_out: Mapping[str, int], frame: float,
          ffmpeg_bin: str, ffprobe_bin: str) -> dict:
    """Check a finished render's sound against its picture, one side of a join at a time.

    Args:
        render_path: The finished file.
        offered_sides: The sides to compare, as `sides` offered them.
        left_out: How many sides were left out, as `sides` counted them.
        frame: One frame of the render, in seconds.
        ffmpeg_bin: The FFmpeg executable.
        ffprobe_bin: The ffprobe executable.

    Returns:
        `streams`, what `stream_mismatch` found or None; `joins`, `{at,
        clip_id, side, lag}` for each side that could be lined up;
        `sides_asked`, how many sides were offered to the check; `left_out`,
        which of them were left out and why; and `frame`, rounded to four
        places.
    """
    streams = stream_mismatch(probe_file(render_path, ffprobe_bin), frame)
    lags: List[dict] = []
    if offered_sides:
        samples = np.frombuffer(read_samples(render_path, ffmpeg_bin), dtype=np.float32)
        measured = voiced_stretches(samples)
        for side in offered_sides:
            lag = lag_at(side["expected"], measured, float(side["start"]), float(side["end"]))
            if lag is not None:
                lags.append({"at": side["at"], "clip_id": side["clip_id"], "side": side["side"], "lag": lag})
    return {"streams": streams, "joins": lags, "sides_asked": len(offered_sides),
            "left_out": dict(left_out), "frame": round(frame, 4)}

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
