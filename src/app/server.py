import os
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp.server.transforms import ResourcesAsTools
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field
from app.models.media import Asset, Span
from app.models.timeline import Project, EditOperation, apply_operation, validate_project
from app.models.job import Job, JobKind, JobStatus
from app.engine.probe import probe_file
from app.engine.builder import FFmpegRenderer
from app.engine.frames import contact_sheet, format_timestamp
from app.engine.renderer import JobManager
from app.storage.repo import Repository

WORKSPACE_DIR = os.path.abspath(os.environ.get("CLIP_MCP_WORKSPACE", "workspace"))
SKILLS_DIR = Path(__file__).parent / "skills"

mcp = FastMCP(
    "VideoEditingCore",
    instructions=(
        "Timeline-based video editing backed by FFmpeg. Before planning edits, "
        "read the clip-editing skill at skill://clip-editing/SKILL.md (clients "
        "without resource support can call read_resource with that URI). It "
        "explains how to turn editing requests into tool calls, how to read "
        "footage with analyze_asset, get_analysis, and view_frames, what to "
        "ask when a request is incomplete, and the current limits: one video "
        "track plus audio tracks for background music, no overlapping clips "
        "on a track, and speed fixed at 1.0. Projects, assets, analyses, and "
        "jobs are saved on disk and survive server restarts."
    ),
)
mcp.add_provider(SkillsDirectoryProvider(roots=SKILLS_DIR))
mcp.add_transform(ResourcesAsTools(mcp))

repo = Repository(os.path.join(WORKSPACE_DIR, "clip_mcp.db"))
job_manager = JobManager(repo)
renderer = FFmpegRenderer()

def _referenced_assets(project: Project) -> dict[str, Asset]:
    """Fetch the assets referenced by a project's clips.

    Args:
        project: Project whose clips are inspected.

    Returns:
        The registered assets used by the project, keyed by asset ID. IDs that
        are not registered are omitted.
    """
    return repo.get_assets(clip.asset_id for track in project.tracks for clip in track.clips)

def _get_asset(asset_id: str) -> Asset:
    """Fetch a registered asset.

    Args:
        asset_id: ID of the asset.

    Returns:
        The asset.

    Raises:
        ValueError: If no asset with `asset_id` exists.
    """
    asset = repo.get_assets([asset_id]).get(asset_id)
    if asset is None:
        raise ValueError(f"asset {asset_id} not found")
    return asset

def _overlapping(spans: List[Span], start: float, end: float) -> List[dict]:
    """Select the spans that overlap a time range.

    Args:
        spans: Spans to filter; anything with `start` and `end` attributes.
        start: Range start in seconds.
        end: Range end in seconds.

    Returns:
        The overlapping spans, serialized as dictionaries.
    """
    return [span.model_dump() for span in spans if span.end > start and span.start < end]

@mcp.tool()
def inspect_media(filepath: str) -> dict:
    """Inspect a local media file and return its technical metadata.

    Use this to check duration, resolution, frame rate, codecs, and audio
    streams before adding a file to a project.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.

    Returns:
        ffprobe output with a `format` object and a `streams` list.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    return probe_file(filepath)

@mcp.tool()
def import_asset(filepath: str) -> dict:
    """Register a local media file as an asset that clips can reference.

    Importing the same path again re-probes the file and keeps its asset ID.
    Video-track clips need `has_video`, and audio-track clips, such as
    background music, need `has_audio`. Cover art embedded in audio files is
    not counted as video. Video assets without audio contribute silence.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.

    Returns:
        The asset serialized as a dictionary: its `id`, absolute `path`,
        `duration` in seconds (or null if unknown), and `has_video` /
        `has_audio` flags.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    path = os.path.abspath(filepath)
    info = probe_file(path)

    existing = repo.find_asset_by_path(path)
    streams = info.get("streams", [])
    duration = info.get("format", {}).get("duration")
    asset = Asset(
        id=existing.id if existing else str(uuid.uuid4()),
        path=path,
        duration=Decimal(duration) if duration is not None else None,
        has_video=any(
            stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")
            for stream in streams
        ),
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )
    repo.save_asset(asset)
    return asset.model_dump()

@mcp.tool()
def list_assets() -> dict:
    """List all imported assets.

    Returns:
        A dictionary with an `assets` list, each item in the same format as
        returned by `import_asset`.
    """
    return {"assets": [asset.model_dump() for asset in repo.list_assets()]}

@mcp.tool()
def analyze_asset(
    asset_id: str,
    transcribe: bool = True,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    chinese_variant: Optional[Literal["zh-TW", "zh-HK", "zh-Hant", "zh-Hans"]] = None,
) -> dict:
    """Start analyzing what an asset contains, in the background.

    Detects scene changes, black and frozen picture, and silences. If
    `transcribe` is true and the asset has audio, also transcribes speech
    locally with word-level timestamps that stay accurate on long recordings.
    Returns immediately: poll `get_job` until the job completes, then read the
    results with `get_analysis`. Analyzing again replaces earlier results. The
    first transcription downloads the speech recognition model.

    Args:
        asset_id: ID of the asset to analyze.
        transcribe: Whether to transcribe speech.
        language: Spoken language code, such as "zh" or "en"; omit to detect it.
        prompt: Text that guides transcription, such as names and terms that
            appear in the recording.
        chinese_variant: Convert a Chinese transcript to this script and
            regional variant: "zh-TW" (Traditional, Taiwan characters and
            vocabulary, e.g. 視頻 becomes 影片), "zh-HK" (Traditional, Hong
            Kong), "zh-Hant" (Traditional, no vocabulary changes), or "zh-Hans"
            (Simplified). Omit to keep the speech model's output, which may mix
            scripts.

    Returns:
        A dictionary with the `job_id` and initial `status`.

    Raises:
        ValueError: If the asset does not exist or has no known duration.
    """
    asset = _get_asset(asset_id)
    if asset.duration is None:
        raise ValueError(f"asset {asset_id} has no known duration and cannot be analyzed")

    job = Job(kind=JobKind.ANALYZE, asset_id=asset_id)
    job.work_dir = os.path.join(WORKSPACE_DIR, "jobs", job.job_id)
    job_manager.start_job(job, {"transcribe": transcribe, "language": language, "prompt": prompt, "chinese_variant": chinese_variant})
    return {"job_id": job.job_id, "status": job.status.value}

@mcp.tool()
def get_analysis(
    asset_id: str,
    start: Annotated[float, Field(ge=0)] = 0,
    end: Optional[float] = None,
    include_words: bool = False,
) -> dict:
    """Return the analysis of an asset for a time range.

    Only items overlapping the range are returned. Transcripts of long
    recordings are large, so read them in windows of about 10 minutes. Times
    are seconds in the source file and can be used directly as a clip's
    `source_range`.

    Args:
        asset_id: ID of the analyzed asset.
        start: Range start in seconds.
        end: Range end in seconds; omit for the end of the asset.
        include_words: Whether to include word-level timings in transcript
            segments. Use them to cut exactly at a word; leave them out to
            save space.

    Returns:
        A dictionary with the asset `duration`, `analyzed_at`, and the
        overlapping `scenes`, `black_frames`, `frozen_frames`, `silences`, and
        `transcript` (`language`, `model`, `chinese_variant`, and `segments`
        with `start`, `end`, and `text`), or null for `transcript` if speech
        was not transcribed.

    Raises:
        ValueError: If the asset has not been analyzed yet.
    """
    analysis = repo.get_analysis(asset_id)
    if analysis is None:
        raise ValueError(f"asset {asset_id} has not been analyzed; call analyze_asset first")
    end = analysis.duration if end is None else end

    transcript = None
    if analysis.transcript is not None:
        exclude = None if include_words else {"words"}
        transcript = {
            "language": analysis.transcript.language,
            "model": analysis.transcript.model,
            "chinese_variant": analysis.transcript.chinese_variant,
            "segments": [
                segment.model_dump(exclude=exclude)
                for segment in analysis.transcript.segments
                if segment.end > start and segment.start < end
            ],
        }
    return {
        "asset_id": asset_id,
        "duration": analysis.duration,
        "analyzed_at": analysis.analyzed_at.isoformat(),
        "range": {"start": start, "end": end},
        "scenes": _overlapping(analysis.scenes, start, end),
        "black_frames": _overlapping(analysis.black_frames, start, end),
        "frozen_frames": _overlapping(analysis.frozen_frames, start, end),
        "silences": _overlapping(analysis.silences, start, end),
        "transcript": transcript,
    }

@mcp.tool()
def view_frames(
    asset_id: str,
    start: Annotated[float, Field(ge=0)] = 0,
    end: Optional[float] = None,
    count: Annotated[int, Field(ge=1, le=36)] = 12,
) -> ToolResult:
    """Look at frames from an asset as one labeled contact sheet image.

    Frames are sampled at evenly spaced times between `start` and `end`. Each
    tile is labeled with its number and source time. Use a wide range for an
    overview and a narrow range to inspect a moment before choosing a cut
    point. Works without `analyze_asset`.

    Args:
        asset_id: ID of an asset with video.
        start: Range start in seconds.
        end: Range end in seconds; omit for the end of the asset.
        count: Number of frames to sample.

    Returns:
        A text block listing each tile's source time, followed by the contact
        sheet as a JPEG image.

    Raises:
        ValueError: If the asset does not exist, has no video, or the range is
            empty.
        RuntimeError: If frames cannot be decoded.
    """
    asset = _get_asset(asset_id)
    if not asset.has_video or asset.duration is None:
        raise ValueError(f"asset {asset_id} has no video frames to show")
    duration = float(asset.duration)
    end = duration if end is None else min(end, duration)
    if end <= start:
        raise ValueError(f"empty range: start {start}s must be before end {end}s (asset is {duration}s long)")

    step = (end - start) / count
    # Sample the middle of each interval, and stay clear of the very last frame, which may not decode.
    times = [round(min(start + step * (index + 0.5), duration - 0.05), 3) for index in range(count)]
    listing = "\n".join(f"#{index + 1}: {seconds:.3f}s ({format_timestamp(seconds)})" for index, seconds in enumerate(times))
    return ToolResult(content=[
        f"Contact sheet of asset {asset_id}, {count} frames from {start:.3f}s to {end:.3f}s, in reading order:\n{listing}",
        Image(data=contact_sheet(asset.path, times), format="jpeg"),
    ])

@mcp.tool()
def create_project(
    width: Annotated[int, Field(gt=0, multiple_of=2)] = 1920,
    height: Annotated[int, Field(gt=0, multiple_of=2)] = 1080,
    fps_num: Annotated[int, Field(gt=0)] = 30,
    fps_den: Annotated[int, Field(gt=0)] = 1,
) -> dict:
    """Create an empty project.

    Add tracks and clips to the new project with `apply_edits`. When rendered,
    every source is scaled to fill the output size, center-cropped, and
    converted to the output frame rate.

    Args:
        width: Output width in pixels; must be even.
        height: Output height in pixels; must be even.
        fps_num: Frame rate numerator, e.g. 30000 for 29.97 fps.
        fps_den: Frame rate denominator, e.g. 1001 for 29.97 fps.

    Returns:
        The new project serialized as a dictionary, including its generated
        `id` and initial `version`.
    """
    project = Project(id=str(uuid.uuid4()), width=width, height=height, fps_num=fps_num, fps_den=fps_den)
    repo.add_project(project)
    return project.model_dump()

@mcp.tool()
def list_projects() -> dict:
    """List all projects.

    Returns:
        A dictionary with a `projects` list, each item holding a project's
        `id` and current `version`.
    """
    return {"projects": [{"id": project.id, "version": project.version} for project in repo.list_projects()]}

@mcp.tool()
def get_project(project_id: str) -> dict:
    """Return the full state of a project, including its tracks and clips.

    The returned `version` is required as `expected_version` when calling
    `apply_edits`.

    Args:
        project_id: ID of the project to retrieve.

    Returns:
        The project serialized as a dictionary, including `duration`, the
        length of the edited video in seconds.

    Raises:
        ValueError: If no project with `project_id` exists.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    return project.model_dump()

@mcp.tool()
def apply_edits(project_id: str, expected_version: int, operations: list[EditOperation]) -> dict:
    """Apply a batch of edit operations to a project with optimistic locking.

    Operations are applied in order and atomically: if any operation fails,
    or the resulting timeline is invalid, the project is left unchanged. On
    success the project version is incremented by one. Call `get_project`
    first to obtain the current version.

    Editing behaves like a magnetic timeline: `insert_clip` shifts later
    clips on the same track to make room, and `trim_clip` and `delete_clip`
    shift them to close or open space unless `ripple` is false. `add_clip`
    and `move_clip` use absolute positions and never move other clips. Clips
    on other tracks never move.

    Background music goes on an audio track. The video track defines the
    output length: gaps are rendered as black frames with silence, and audio
    past the last video clip is cut. `set_clip_audio` changes a clip's
    `volume` and `audio_fade_in` / `audio_fade_out`.

    The resulting timeline must satisfy these rules:
    - Clips stay within their asset's duration and do not overlap on a track.
    - Audio fades fit within their clip.
    - A project has at most one video track; audio tracks are unlimited.
    - Video-track clips use assets with video; audio-track clips use assets
      with audio.
    - Clips use speed 1.0.

    Args:
        project_id: ID of the project to edit.
        expected_version: Project version the caller last read. The request is
            rejected if the project has been modified since then.
        operations: Edit operations to apply, each identified by its `action`:
            `add_track`, `add_clip`, `insert_clip`, `trim_clip`, `move_clip`,
            `delete_clip`, or `set_clip_audio`.

    Returns:
        A dictionary with the `status` and the project's `new_version`.

    Raises:
        ValueError: If the project does not exist, the version does not match,
            an operation references a missing or duplicate track or clip, or
            the resulting timeline breaks a rule above.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if project.version != expected_version:
        raise ValueError(f"version conflict: current version is {project.version}, expected version is {expected_version}")

    for op in operations:
        apply_operation(project, op)
    assets = _referenced_assets(project)
    validate_project(project, assets)
    renderer.check_supported(project, assets)

    project.version += 1
    repo.update_project(project, expected_version)
    return {"status": "success", "new_version": project.version}

@mcp.tool()
def render_project(project_id: str, is_preview: bool = False) -> dict:
    """Start rendering a project to an MP4 file in the background.

    Returns immediately. Poll `get_job` with the returned `job_id` until the
    job reaches a terminal status, or stop it with `cancel_job`. Rendering
    runs in a separate worker process and continues even if this server
    restarts.

    Args:
        project_id: ID of the project to render.
        is_preview: If true, render a fast 480p preview instead of the
            full-quality output.

    Returns:
        A dictionary with the `job_id`, the initial `status`, and the absolute
        `output_path` the file will be written to. Each job writes to its own
        directory, so repeated renders never overwrite each other.

    Raises:
        ValueError: If the project does not exist, contains no clips, or uses
            an unsupported feature.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")

    kind = "preview" if is_preview else "output"
    job = Job(kind=JobKind.RENDER, project_id=project_id)
    job.work_dir = os.path.join(WORKSPACE_DIR, "outputs", job.job_id)
    job.output_path = os.path.join(job.work_dir, f"{project_id}_{kind}.mp4")

    command = renderer.build_command(project, _referenced_assets(project), job.output_path, is_preview=is_preview)
    job_manager.start_job(job, {"command": command, "duration_seconds": float(renderer.output_duration(project))})

    return {"job_id": job.job_id, "status": job.status.value, "output_path": job.output_path}

@mcp.tool()
def get_job(job_id: str) -> dict:
    """Return the current state of a background job.

    Args:
        job_id: ID of the job, as returned by `render_project` or
            `analyze_asset`.

    Returns:
        The job serialized as a dictionary, including `kind`, `status`,
        `progress` (0.0 to 1.0), the current `stage`, `output_path` for
        renders, and, for failed jobs, `error_message`.

    Raises:
        ValueError: If no job with `job_id` exists.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise ValueError(f"job {job_id} not found")
    return job.model_dump()

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Cancel a queued or running background job.

    Waits up to a few seconds for a running job to stop.

    Args:
        job_id: ID of the job, as returned by `render_project` or
            `analyze_asset`.

    Returns:
        A dictionary with the `job_id`, a `cancelled` flag, and the job's
        current `status` (null if the job does not exist). `cancelled` is
        false if the job does not exist or had already finished.
    """
    job = job_manager.cancel_job(job_id)
    return {
        "job_id": job_id,
        "cancelled": job is not None and job.status == JobStatus.CANCELLED,
        "status": job.status.value if job else None,
    }

def main():
    """Run the MCP server over the stdio transport."""
    mcp.run()

if __name__ == "__main__":
    main()
