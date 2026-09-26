"""Helpers for driving the server's tools from tests, rendering, and measuring the result."""

import re
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

from pydantic import TypeAdapter

from app.engine import loudness as levelling
from app.engine.builder import DEFAULT_LOUDNESS_TARGET, FFmpegRenderer
from app.models.timeline import Clip, EditOperation
from app.server import _referenced_assets, apply_edits, create_project, get_project, repo

_OPERATIONS = TypeAdapter(List[EditOperation])

def build_project(operations: List[dict], **settings) -> str:
    """Create a project and apply one batch of operations to it.

    Args:
        operations: Edit operations as plain dictionaries, as a client sends
            them.
        **settings: Output settings passed to `create_project`.

    Returns:
        The new project's ID.
    """
    project_id = create_project(**settings)["id"]
    apply_edits(project_id, 1, _OPERATIONS.validate_python(operations))
    return project_id

def video_track(track_id: str = "main") -> dict:
    """Build an `add_track` operation for a video track.

    Args:
        track_id: ID for the new track.

    Returns:
        The operation as a dictionary.
    """
    return {"action": "add_track", "track_id": track_id, "track_type": "video"}

def insert(clip_id: str, asset_id: str, start: float, end: float, track_id: str = "main", **fields) -> dict:
    """Build an `insert_clip` operation.

    Args:
        clip_id: ID for the new clip.
        asset_id: Asset the clip comes from.
        start: Source range start in seconds.
        end: Source range end in seconds.
        track_id: Track to append to.
        **fields: Extra clip fields, such as `volume` or `before_clip_id`.

    Returns:
        The operation as a dictionary.
    """
    return {
        "action": "insert_clip", "track_id": track_id, "clip_id": clip_id, "asset_id": asset_id,
        "source_range": {"start": start, "end": end}, **fields,
    }

def edit(project_id: str, operations: List[dict]) -> dict:
    """Apply a batch of operations to an existing project at its current version.

    Args:
        project_id: Project to edit.
        operations: Edit operations as plain dictionaries.

    Returns:
        What `apply_edits` returned.
    """
    version = get_project(project_id)["version"]
    return apply_edits(project_id, version, _OPERATIONS.validate_python(operations))

def clips_of(project_id: str, track_id: str = "main") -> List[dict]:
    """Read a track's clips in timeline order.

    Args:
        project_id: Project to read.
        track_id: Track whose clips are returned.

    Returns:
        Each clip as a dictionary, ordered by `timeline_in`.
    """
    project = get_project(project_id)
    track = next(item for item in project["tracks"] if item["id"] == track_id)
    # get_project omits fields still at their default; rebuild the whole clip so tests can read any field.
    clips = [Clip.model_validate(clip).model_dump() for clip in track["clips"]]
    return sorted(clips, key=lambda clip: clip["timeline_in"])

def layout(project_id: str, track_id: str = "main") -> List[tuple]:
    """Summarize a track as clip IDs with their timeline and source spans.

    Args:
        project_id: Project to read.
        track_id: Track to summarize.

    Returns:
        One `(clip_id, timeline_in, timeline_out, source_start, source_end)`
        per clip, in timeline order, with times as floats.
    """
    return [
        (
            clip["id"],
            float(clip["timeline_in"]),
            float(clip["timeline_in"]) + float(clip["source_range"]["end"]) - float(clip["source_range"]["start"]),
            float(clip["source_range"]["start"]),
            float(clip["source_range"]["end"]),
        )
        for clip in clips_of(project_id, track_id)
    ]

def render(project_id: str, path: Path, **options) -> None:
    """Render a project straight through FFmpeg, skipping the job worker.

    Args:
        project_id: Project to render.
        path: File to write.
        **options: Extra arguments for `build_command`.

    Raises:
        AssertionError: If FFmpeg rejects the filter graph or fails.
    """
    project = repo.get_project(project_id)
    renderer, assets = FFmpegRenderer(), _referenced_assets(project)
    command = renderer.build_command(project, assets, str(path), **options)
    target = options.get("loudness_target", DEFAULT_LOUDNESS_TARGET)
    if target is not None:
        # What the render worker does: measure the mix on its own, then write in its gain.
        mix = path.with_name(path.stem + "-mix.wav")
        measured = subprocess.run(
            renderer.build_mix(project, assets, str(mix), voices=options.get("voices"), talking=options.get("talking")),
            capture_output=True,
        )
        assert measured.returncode == 0, measured.stderr.decode("utf-8", errors="replace")[-800:]
        command = levelling.with_gain(command, levelling.settle_gain(str(mix), target))
        mix.unlink()
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")[-800:]

def _measure(path: Path, filters: str, pattern: str, before: Sequence[str] = ()) -> float:
    """Run an FFmpeg measuring filter and read the number it logs.

    Args:
        path: File to measure.
        filters: Audio filter chain that prints a measurement.
        pattern: Regular expression with one group capturing the number.
        before: Arguments placed before the input, such as a seek.

    Returns:
        The last matching value.
    """
    result = subprocess.run(
        ["ffmpeg", "-nostats", *before, "-i", str(path), "-af", filters, "-f", "null", "-"],
        capture_output=True,
    )
    return float(re.findall(pattern, result.stderr.decode("utf-8", errors="replace"))[-1])

def loudness(path: Path) -> float:
    """Measure integrated loudness.

    Args:
        path: File to measure.

    Returns:
        Integrated loudness in LUFS.
    """
    return _measure(path, "ebur128=framelog=quiet", r"I:\s+(-?[\d.]+) LUFS")

def true_peak(path: Path) -> float:
    """Measure the true peak.

    Args:
        path: File to measure.

    Returns:
        The true peak in dBFS.
    """
    return _measure(path, "ebur128=framelog=quiet:peak=true", r"Peak:\s+(-?[\d.]+) dBFS")

def level(path: Path, start: float, duration: float, freq: Optional[float] = None) -> float:
    """Measure the mean level of a stretch of a file, optionally in one band.

    Args:
        path: File to measure.
        start: Start of the stretch in seconds.
        duration: Length of the stretch in seconds.
        freq: Centre of a narrow band to isolate, such as the music tone.

    Returns:
        Mean volume in dB.
    """
    chain = "volumedetect" if freq is None else f"bandpass=f={freq}:width_type=h:w=30,volumedetect"
    return _measure(path, chain, r"mean_volume:\s+(-?[\d.]+) dB", ["-ss", str(start), "-t", str(duration)])
