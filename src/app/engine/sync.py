"""A finished render's sound, checked against the sound of the files it was cut from.

A file's picture and sound are two streams in one container, and they can
drift apart without anything looking wrong: the sound can start after the
picture and stay late for the whole cut, or a join can be a little to one
side of the sound it should land on. Neither shows up in a look at a frame,
and a viewer hears the second one as a cut that feels off without being able
to say why. Each side of a join is compared with the same stretch of its own
source file, by how its loudness rises and falls: a shape that survives
whatever is being said. Only the moments that stay inside the side are
compared at each lag, so a lag never reaches across the join into the other
clip's sound. What was meant — a J cut, a line carried across a cut, a pause —
is for whoever reads the result to say; this only says how far the sound has
moved.
"""

import json
import os
import subprocess
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from app.engine.ffmpeg import FFmpegFailed, hidden_window_flags
from app.engine.probe import probe_file
from app.models.timeline.parts import Clip, Track

SYNC_FILE_NAME = "sync.json"
# How far, in seconds either side of a join, the sound is compared. Provisional.
JOIN_WINDOW_SECONDS = 2.0
# The shortest stretch, in seconds, a side of a join is compared over; less holds too little sound to line up.
# Provisional.
MIN_SIDE_SECONDS = 0.5
# How far a side must be able to be moved, in seconds, with MIN_SIDE_SECONDS still compared: a side
# shorter than both together has too little room to say where its sound is. Provisional.
MIN_REACH_SECONDS = 0.2
# The furthest the sound is looked for out of place, in seconds. Provisional.
MAX_LAG_SECONDS = 0.5
# What `lag_at` returns when the best shift it found is at the edge of what it searched, so the sound may be
# further off than that.
BEYOND_REACH = "beyond_reach"
# The grid loudness is compared on, in seconds.
LAG_STEP_SECONDS = 0.01
# Sound is compared as its loudness every LAG_STEP_SECONDS, read at this many samples a second.
SAMPLE_RATE = 16000
# How closely the render's loudness must follow the file's at the best lag for it to be believed:
# a correlation, 1 being the same shape. Provisional.
MIN_CORRELATION = 0.8
# How far short of a stretch its file is sought before decoding, so the seek never lands at the file's very
# start; the rest is cut by timestamp.
SEEK_SHORT_SECONDS = 1.0

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

def _sound_span(clip: Clip) -> Tuple[float, float]:
    """Say where on the timeline one clip's own sound plays."""
    return float(clip.audio_timeline_in), float(clip.audio_timeline_out)

def _clear_stretch(window: Tuple[float, float], at: float, others: Sequence[Tuple[float, float]],
                   side: str) -> List[Tuple[float, float]]:
    """Find the parts of a side's window that hold none of the other sound, nearest the join first.

    Sound reaching over the join from the other side still ends the side where
    it starts: a lead on the next clip leaves the rest of the before side to
    compare, and a lag on the last one leaves the rest of the after side.
    Other sound inside the window cuts a piece out of it, and the pieces either
    side of that cut are both offered, so a short piece next to the join does
    not hide a longer one further out. Other sound that covers the whole side
    leaves nothing.

    Args:
        window: The side's window, `(start, end)` on the timeline.
        at: Where the join is.
        others: Where every other heard clip's sound plays.
        side: `"before"` or `"after"`.

    Returns:
        The clear stretches, `(start, end)` each, nearest the join first; empty
        when the other sound covers the whole side.
    """
    pieces = [window]
    for low, high in _merged(others):
        remaining: List[Tuple[float, float]] = []
        for piece_start, piece_end in pieces:
            if high <= piece_start or low >= piece_end:
                remaining.append((piece_start, piece_end))
                continue
            if low > piece_start:
                remaining.append((piece_start, min(low, piece_end)))
            if high < piece_end:
                remaining.append((max(high, piece_start), piece_end))
        pieces = remaining
    pieces = [(low, high) for low, high in pieces if high - low > 0]
    if side == "before":
        return sorted(pieces, key=lambda piece: at - piece[1])
    return sorted(pieces, key=lambda piece: piece[0] - at)

def sides(project_tracks: Sequence[Track], base_clips: Sequence[Clip],
          sources: Mapping[str, str]) -> Tuple[List[dict], Dict[str, int]]:
    """Say which sides of which joins have a stretch of their own sound to line up.

    A join has two sides, the clip before it and the clip after it. Each side
    is compared over `JOIN_WINDOW_SECONDS` on its own side of the join, cut
    down to where that clip is the only sound playing: music or another
    picture track over it, and the neighbouring clip's own lead or lag, cut
    the window into pieces, and the nearest piece of at least
    `MIN_SIDE_SECONDS + MIN_REACH_SECONDS` is used, so it has room to be
    moved while `MIN_SIDE_SECONDS` is still compared. The clip's own fades
    are taken off as well, since a fade changes the loudness's shape, so it
    is not compared. A clip whose asset has no sound makes none, so it
    neither offers a side nor counts as other sound.

    Args:
        project_tracks: Every track of the project, so that sound from a
            picture track drawn above or from an audio track is counted.
        base_clips: The base video track's clips, which meet at the joins.
        sources: Per asset ID whose file has sound, the file's path.

    Returns:
        The sides offered to the check, in timeline order with `before` before
        `after` at one join: `{at, clip_id, side, start, end, path,
        source_start, speed}`, where `source_start` is the second of the file
        that plays at `start`; and how many sides were left out, as
        `{too_short, faded}`.
    """
    counts = {"too_short": 0, "faded": 0}
    offered: List[dict] = []
    ordered = sorted(base_clips, key=lambda clip: clip.timeline_in)
    heard = [clip for track in project_tracks for clip in track.clips
             if clip.volume > 0 and clip.card is None and clip.asset_id in sources]
    for before, after in zip(ordered, ordered[1:]):
        at = float(after.timeline_in)
        for side, clip in (("before", before), ("after", after)):
            if clip.volume <= 0 or clip.card is not None or clip.asset_id not in sources:
                continue
            window = (at - JOIN_WINDOW_SECONDS, at) if side == "before" else (at, at + JOIN_WINDOW_SECONDS)
            others = [_sound_span(other) for other in heard if other is not clip]
            pieces = _clear_stretch(window, at, others, side)
            own_start, own_end = _sound_span(clip)
            fade_in, fade_out = float(clip.audio_fade_in), float(clip.audio_fade_out)
            chosen: Optional[Tuple[float, float]] = None
            without_fades: Optional[Tuple[float, float]] = None
            for piece_start, piece_end in pieces:
                start, end = max(piece_start, own_start), min(piece_end, own_end)
                if end - start >= MIN_SIDE_SECONDS + MIN_REACH_SECONDS and without_fades is None:
                    without_fades = (start, end)
                # Inside a fade the loudness is a ramp rather than the file's own, so a fade is not compared.
                clear = (max(start, own_start + fade_in), min(end, own_end - fade_out))
                if clear[1] - clear[0] >= MIN_SIDE_SECONDS + MIN_REACH_SECONDS:
                    chosen = clear
                    break
            if chosen is None:
                counts["faded" if without_fades is not None else "too_short"] += 1
                continue
            start, end = chosen
            source_start = float(clip.audio_source_start) + (start - float(clip.audio_timeline_in)) * float(clip.speed)
            offered.append({
                "at": round(at, 3), "clip_id": clip.id, "side": side, "start": round(start, 3),
                "end": round(end, 3), "path": sources[clip.asset_id], "source_start": round(source_start, 3),
                "speed": float(clip.speed),
            })
    return offered, counts

def _read_stretch(path: str, start: float, length: Optional[float], ffmpeg_bin: str) -> np.ndarray:
    """Decode a stretch of a file's sound to mono samples at `SAMPLE_RATE`.

    Args:
        path: The media file.
        start: The first second of the file to read; never negative.
        length: How many seconds to read; None reads to the end of the file.
        ffmpeg_bin: The FFmpeg executable.

    Returns:
        The samples as 32-bit floats, sample 0 at `start` seconds of the file.

    Raises:
        ValueError: If `start` is negative.
        FFmpegFailed: If FFmpeg cannot decode the file.
    """
    if start < 0:
        raise ValueError(f"cannot read a stretch from {start}s of a file")
    # A seek to the very start of an AAC file drops one more frame of its encoder delay than reading it from the
    # start does, so the file is only sought to well short of the stretch and the rest is cut by timestamp.
    coarse = start - SEEK_SHORT_SECONDS if start > SEEK_SHORT_SECONDS else 0.0
    command = [ffmpeg_bin, "-hide_banner", "-nostdin", "-v", "error"]
    if coarse > 0:
        command += ["-ss", f"{coarse:.6f}"]
    command += ["-i", path]
    if start - coarse > 0:
        command += ["-ss", f"{start - coarse:.6f}"]
    if length is not None:
        command += ["-t", f"{length:.6f}"]
    command += ["-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        creationflags=hidden_window_flags(),
    )
    if result.returncode != 0:
        raise FFmpegFailed(result.stderr.decode("utf-8", errors="replace").strip()[-500:]
                           or "ffmpeg could not read the sound")
    return np.frombuffer(result.stdout, dtype=np.float32)

def _loudness(samples: np.ndarray, origin: float, times: Sequence[float],
              window: float = LAG_STEP_SECONDS) -> np.ndarray:
    """Measure the RMS of the samples in a window starting at each of several moments.

    Sample `i` is at `origin + i / SAMPLE_RATE`. A window with no samples
    inside the sound has an RMS of 0; one reaching past either end is measured
    over the part that is there, against the whole window.

    Args:
        samples: The sound.
        origin: The second the first sample is at, in whatever time `times` is in.
        times: Where each window begins, in seconds.
        window: How long each window is, in seconds.

    Returns:
        One RMS per moment.
    """
    moments = np.asarray(times, dtype=np.float64)
    first = np.rint((moments - origin) * SAMPLE_RATE).astype(np.int64)
    width = max(1, int(round(window * SAMPLE_RATE)))
    low = np.clip(first, 0, len(samples))
    high = np.clip(first + width, 0, len(samples))
    if low.size == 0:
        return np.zeros(0)
    # Only the part the windows reach is summed, so a side of a long render costs no more than one of a short one.
    begin = int(low.min())
    values = np.asarray(samples[begin:int(high.max())], dtype=np.float64)
    squared = np.concatenate(([0.0], np.cumsum(values * values)))
    totals = squared[high - begin] - squared[low - begin]
    return np.sqrt(totals / width)

def _correlation(one: np.ndarray, other: np.ndarray) -> float:
    """Say how closely two shapes follow each other, from -1 to 1; 0 when either never varies."""
    one = one - one.mean()
    other = other - other.mean()
    scale = float(np.dot(one, one)) * float(np.dot(other, other))
    if scale == 0.0:
        return 0.0
    return float(np.dot(one, other) / np.sqrt(scale))

def _reach_steps(start: float, end: float) -> Tuple[int, int]:
    """Count a side's grid times, and how many steps it may be moved and still keep `MIN_SIDE_SECONDS` compared.

    Args:
        start: The side's start on the timeline.
        end: The side's end on the timeline.

    Returns:
        `(steps, limit)`: the side's grid times, and its reach in steps.
    """
    steps = max(1, int(round((end - start) / LAG_STEP_SECONDS)))
    least = int(round(MIN_SIDE_SECONDS / LAG_STEP_SECONDS))
    return steps, min(int(round(MAX_LAG_SECONDS / LAG_STEP_SECONDS)), max(0, steps - least))

def reach(side: Mapping[str, object]) -> float:
    """Say how far a side's sound was searched each way: `MAX_LAG_SECONDS`, or less for a short side.

    Args:
        side: The side's `start` and `end` on the timeline.

    Returns:
        Seconds, rounded to two places.
    """
    return round(_reach_steps(float(side["start"]), float(side["end"]))[1] * LAG_STEP_SECONDS, 2)

def lag_at(render: np.ndarray, render_origin: float, source: np.ndarray, side: Mapping[str, object],
           source_origin: Optional[float] = None) -> Union[float, str, None]:
    """Say how far the sound on one side of a join is heard from where the cut put it.

    The file's loudness is measured on a grid over the side, and the render's
    at the same moments, and the two shapes correlated at each lag. Only the
    grid times whose whole render window stays inside the side are compared at
    a lag, so the search never reaches across the join into the other clip's
    sound. A lag at the edge of the search may be a real offset larger than
    the search, so it is reported as `BEYOND_REACH` rather than as that edge.
    The render is held at its sounds, not at any voice in them, so a side of
    room tone lines up as well as one of talking.

    Args:
        render: The render's samples, sample 0 at `render_origin` on the timeline.
        render_origin: The timeline second of the render's first sample.
        source: The file's samples, sample 0 at `source_origin` in the file.
        side: The side's `start` and `end` on the timeline, the file second
            `source_start` that plays at `start`, and its `speed`.
        source_origin: The file second of the source's first sample; left out,
            it is the side's `source_start`.

    Returns:
        The lag in seconds whose loudness follows the file's best, rounded to
        two places; positive when the sound is heard late. None when no lag
        follows closely enough (a file stretch that never varies follows
        nothing); `BEYOND_REACH` when a lag that follows closely enough is at
        the edge of what was searched, so the sound may be further off.
    """
    start, end = float(side["start"]), float(side["end"])
    speed = float(side["speed"])
    # A lag must leave `MIN_SIDE_SECONDS` of the side compared, so a short side searches less far.
    steps, limit = _reach_steps(start, end)
    times = start + np.arange(steps, dtype=np.float64) * LAG_STEP_SECONDS
    file_start = float(side["source_start"])
    origin = file_start if source_origin is None else float(source_origin)
    offsets = (file_start - origin) + (times - start) * speed
    from_file = _loudness(source, 0.0, offsets, LAG_STEP_SECONDS * speed)
    # The render's loudness is measured once over the side; each lag is then a pair of slices of the two.
    heard = _loudness(render, render_origin, times)
    lag, correlation = 0.0, _correlation(from_file, heard)
    # In order of the smaller move, and a negative one before a positive, so
    # that equal correlations keep the lag preferred by the tie-break.
    moves = sorted((number for number in range(-limit, limit + 1) if number != 0),
                   key=lambda number: (abs(number), number))
    for number in moves:
        count = steps - abs(number)
        file_low = max(0, -number)
        heard_low = max(0, number)
        shape = _correlation(from_file[file_low:file_low + count], heard[heard_low:heard_low + count])
        if shape > correlation:
            lag, correlation = number * LAG_STEP_SECONDS, shape
    if correlation < MIN_CORRELATION:
        return None
    if limit == 0 or abs(lag) >= (limit - 0.5) * LAG_STEP_SECONDS:
        return BEYOND_REACH
    return round(lag, 2)

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
        `sides_asked`, how many sides were offered to the check; `unlined`,
        `{weak, beyond_reach, searched}`: how many of those could not be
        lined up because their sound did not follow the file's closely
        enough, or their file could not be read, and how many because the
        best shift was at the edge of the search; `searched` gives, for each
        of the two, `[least, most]` seconds those sides were searched each way
        (a short side searches less than `MAX_LAG_SECONDS`), or None when
        there are none; `left_out`, how many were left out
        before the check and why; and `frame`, rounded to four places.
    """
    streams = stream_mismatch(probe_file(render_path, ffprobe_bin), frame)
    lags: List[dict] = []
    unlined: Dict[str, object] = {"weak": 0, "beyond_reach": 0}
    searched: Dict[str, List[float]] = {"weak": [], "beyond_reach": []}
    if offered_sides:
        render = _read_stretch(render_path, 0.0, None, ffmpeg_bin)
        for side in offered_sides:
            speed = float(side["speed"])
            read_start = float(side["source_start"])
            length = (float(side["end"]) - float(side["start"])) * speed + LAG_STEP_SECONDS * speed
            try:
                source = _read_stretch(side["path"], read_start, length, ffmpeg_bin)
            except (FFmpegFailed, OSError):
                searched["weak"].append(reach(side))
                continue
            lag = lag_at(render, 0.0, source, side, source_origin=read_start)
            if lag == BEYOND_REACH:
                searched["beyond_reach"].append(reach(side))
            elif lag is not None:
                lags.append({"at": side["at"], "clip_id": side["clip_id"], "side": side["side"], "lag": lag})
            else:
                searched["weak"].append(reach(side))
    for kind, reaches in searched.items():
        unlined[kind] = len(reaches)
    unlined["searched"] = {kind: [min(reaches), max(reaches)] if reaches else None
                           for kind, reaches in searched.items()}
    return {"streams": streams, "joins": lags, "sides_asked": len(offered_sides), "unlined": unlined,
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
