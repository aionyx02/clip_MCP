import os
import re
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
from app.models.timeline import Project, TrackType, EditOperation, apply_operation, validate_project
from app.models.job import Job, JobKind, JobStatus
from app.engine.probe import probe_file
from app.engine.builder import DEFAULT_LOUDNESS_TARGET, FFmpegRenderer
from app.engine.frames import contact_sheet, format_timestamp, storyboard_sheet
from app.engine.subtitles import DEFAULT_MAX_CHARACTERS, DEFAULT_MAX_SECONDS, build_ass, timeline_cues
from app.engine.renderer import JobManager
from app.storage.repo import Repository

WORKSPACE_DIR = os.path.abspath(os.environ.get("CLIP_MCP_WORKSPACE", "workspace"))
SKILLS_DIR = Path(__file__).parent / "skills"
MAX_STORYBOARD_TILES = 36
MEDIA_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".mts", ".m2ts", ".wmv", ".flv", ".3gp",
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff",
})

mcp = FastMCP(
    "VideoEditingCore",
    instructions=(
        "Timeline-based video editing backed by FFmpeg. Before planning edits, "
        "read the clip-editing skill at skill://clip-editing/SKILL.md (clients "
        "without resource support can call read_resource with that URI). It "
        "explains how to turn editing requests into tool calls, how to read "
        "footage with analyze_asset, get_analysis, and view_frames, how to "
        "check an edit with preview_project before rendering, how to work "
        "through a folder of unedited footage when the user has no plan, "
        "what to ask when a request is incomplete, how music ducks under "
        "speech and how every render is normalized to one consistent "
        "loudness, how to caption an edit with generate_subtitles and "
        "burn_subtitles, and the current limits: no overlapping clips on a "
        "track, speed fixed at 1.0, no still images, and fades through black "
        "rather than cross dissolves. The person using this server is "
        "editing their own video, not writing code: reply in plain language "
        "with no tool names, IDs, or JSON, and read the plan back in one "
        "sentence for confirmation before the first render. Projects, assets, "
        "analyses, and jobs are saved on disk and survive server restarts."
    ),
)
# "resources" lists the supporting files alongside SKILL.md, so a client sees them
# without having to read the manifest first.
mcp.add_provider(SkillsDirectoryProvider(roots=SKILLS_DIR, supporting_files="resources"))
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

def _natural_key(name: str) -> List[object]:
    """Build a sort key that orders embedded numbers by value.

    Camera and phone exports are often numbered without padding, so plain
    text order would put `clip10` before `clip2`.

    Args:
        name: File name to build a key for.

    Returns:
        The name split into text and integer parts, lowercased.
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

def _register_asset(filepath: str) -> Asset:
    """Probe a media file and save it as an asset.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.

    Returns:
        The saved asset. A path that was imported before keeps its asset ID.

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
    return asset

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
    return _register_asset(filepath).model_dump()

@mcp.tool()
def list_assets() -> dict:
    """List all imported assets.

    Returns:
        A dictionary with an `assets` list, each item in the same format as
        returned by `import_asset`.
    """
    return {"assets": [asset.model_dump() for asset in repo.list_assets()]}

@mcp.tool()
def import_folder(folderpath: str, recursive: bool = False) -> dict:
    """Register every media file in a folder as an asset, in one call.

    Use this when the user points at a folder of footage instead of naming
    files. Files are returned in natural name order, so `clip2` comes before
    `clip10`, which is also the order to put them on the timeline unless the
    user says otherwise. A file that cannot be read is reported in `skipped`
    rather than failing the whole folder, and importing a folder again keeps
    the asset IDs it already handed out.

    Args:
        folderpath: Path to the folder, absolute or relative to the server's
            working directory.
        recursive: Whether to include media in sub-folders.

    Returns:
        A dictionary with `assets` (each in the format `import_asset` returns),
        `total_duration` in seconds, and `skipped`, a list of `path` and
        `reason` for files that could not be read.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    folder = os.path.abspath(folderpath)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"folder {folder} not found")

    paths: List[str] = []
    if recursive:
        for directory, _, names in os.walk(folder):
            paths.extend(os.path.join(directory, name) for name in names)
    else:
        paths = [os.path.join(folder, name) for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]
    paths = sorted(
        (path for path in paths if os.path.splitext(path)[1].lower() in MEDIA_EXTENSIONS),
        key=lambda path: (_natural_key(os.path.dirname(path)), _natural_key(os.path.basename(path))),
    )

    assets: List[Asset] = []
    skipped: List[dict] = []
    for path in paths:
        try:
            assets.append(_register_asset(path))
        except (OSError, RuntimeError, ValueError) as error:
            skipped.append({"path": path, "reason": str(error)})

    total = sum((asset.duration for asset in assets if asset.duration is not None), Decimal(0))
    return {"assets": [asset.model_dump() for asset in assets], "total_duration": total, "skipped": skipped}

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
    count: Annotated[int, Field(ge=1, le=MAX_STORYBOARD_TILES)] = 12,
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
def get_project(project_id: str, include_subtitles: bool = False) -> dict:
    """Return the state of a project: its tracks, clips, and version.

    The returned `version` is required as `expected_version` when calling
    `apply_edits`. Captions are left out unless asked for, because a long
    video has hundreds of them and they are not needed to edit the timeline;
    read them with `get_subtitles` instead. Clip fields that are still at
    their default are left out too, so a clip showing no `volume` is at 1.0
    and one showing no `color` is as shot.

    Args:
        project_id: ID of the project to retrieve.
        include_subtitles: Whether to include every caption in the result.

    Returns:
        The project serialized as a dictionary, including `duration`, the
        length of the edited video in seconds, and `subtitle_count`.

    Raises:
        ValueError: If no project with `project_id` exists.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    # Clips are dumped without the fields that are still at their defaults, which is most of them
    # on a normal cut. The project's own settings are put back, because the caller needs them every time.
    state = project.model_dump(exclude_defaults=True)
    state.update({
        "id": project.id,
        "version": project.version,
        "width": project.width,
        "height": project.height,
        "fps_num": project.fps_num,
        "fps_den": project.fps_den,
        "duration": project.duration,
        "subtitle_count": len(project.subtitles),
    })
    state["tracks"] = []
    for track in project.tracks:
        summary = track.model_dump(exclude_defaults=True)
        summary.update({"id": track.id, "track_type": track.track_type.value})
        summary["clips"] = [clip.model_dump(exclude_defaults=True) for clip in track.clips]
        state["tracks"].append(summary)
    if include_subtitles:
        state["subtitles"] = [cue.model_dump() for cue in project.subtitles]
    else:
        state.pop("subtitles", None)
    return state

@mcp.tool()
def apply_edits(project_id: str, expected_version: int, operations: list[EditOperation]) -> dict:
    """Apply a batch of edit operations to a project with optimistic locking.

    Operations are applied in order and atomically: if any operation fails,
    or the resulting timeline is invalid, the project is left unchanged. On
    success the project version is incremented by one. Call `get_project`
    first to obtain the current version.

    Editing behaves like a magnetic timeline: `insert_clip` shifts later
    clips on the same track to make room, and `trim_clip` and `delete_clip`
    shift them to close or open space unless `ripple` is false.
    `reorder_clip` moves a clip elsewhere in the sequence, closing the space
    it leaves and opening space where it lands, and it keeps the clip's cut
    and audio settings. `split_clip` cuts one clip into two halves that
    together occupy exactly what the original did, so nothing else moves.
    `add_clip` and `move_clip` use absolute positions and never move other
    clips. Clips on other tracks never move.

    Background music goes on an audio track. The first video track is the
    base and defines the output length: gaps are rendered as black frames
    with silence, and audio past its last clip is cut. Further video tracks
    are drawn on top of the base, each clip inside the box its `layout`
    gives, or over the whole frame without one. `set_clip_audio` changes a
    clip's `volume` and `audio_fade_in` / `audio_fade_out`, `set_clip_look`
    its fades through black and its `color`, and `set_track_audio` a whole
    track's `duck_under_speech`. `set_subtitles` replaces the captions
    `render_project` burns in.

    The resulting timeline must satisfy these rules:
    - Clips stay within their asset's duration and do not overlap on a track.
    - Audio and video fades fit within their clip.
    - Only clips above the base video track take a `layout`; base-track clips
      always fill the frame.
    - Only audio tracks duck under speech.
    - Video-track clips use assets with video; audio-track clips use assets
      with audio.
    - Clips use speed 1.0.

    Args:
        project_id: ID of the project to edit.
        expected_version: Project version the caller last read. The request is
            rejected if the project has been modified since then.
        operations: Edit operations to apply, each identified by its `action`:
            `add_track`, `add_clip`, `insert_clip`, `trim_clip`, `move_clip`,
            `delete_clip`, `split_clip`, `reorder_clip`, `fit_track`,
            `set_clip_audio`, `set_clip_look`, `set_track_audio`,
            `set_subtitles`, or `edit_subtitle`.

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

    # Assets are looked up first so operations that need source lengths, such as fit_track, can use them.
    known = repo.get_assets({clip.asset_id for track in project.tracks for clip in track.clips})
    for op in operations:
        apply_operation(project, op, known)
    assets = _referenced_assets(project)
    validate_project(project, assets)
    renderer.check_supported(project, assets)

    project.version += 1
    repo.update_project(project, expected_version)
    return {"status": "success", "new_version": project.version}

@mcp.tool()
def preview_project(
    project_id: str,
    count: Annotated[int, Field(ge=1, le=MAX_STORYBOARD_TILES)] = 12,
) -> ToolResult:
    """Look at the edited sequence as one labeled storyboard image.

    A storyboard takes seconds, while `render_project` takes minutes, so use
    it to check an edit before rendering: the order of the clips, what each
    one opens on, and whether a cut lands somewhere wrong. Every clip gets at
    least one tile and longer clips get more, so a short clip is never missed;
    only a project with more clips than the sheet holds is thinned out, and
    the text says how many were skipped.
    Tiles are labeled with their time in the edited result and the clip they
    come from, and cropped the way the render crops, so a vertical project
    shows vertical tiles. Tiles come from the base video track, which is the
    sequence itself. Black gaps, insets on higher video tracks, and audio
    tracks are listed in the text, since they cannot be seen in the frames.

    Args:
        project_id: ID of the project to look at.
        count: Number of frames to sample. Raised to one per clip when the
            project has more clips than this.

    Returns:
        A text block describing the output format, every tile, the black gaps,
        and the audio tracks, followed by the storyboard as a JPEG image.

    Raises:
        ValueError: If the project does not exist, has no clips on a video
            track, or a video clip uses an asset without video.
        RuntimeError: If frames cannot be decoded.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")

    # Tiles come from the base track, which is the sequence; tracks above it are insets, noted in the text.
    base = project.base_video_track
    clips = sorted(base.clips, key=lambda clip: clip.timeline_in) if base else []
    if not clips:
        raise ValueError(f"project {project_id} has no clips on a video track to preview")

    assets = _referenced_assets(project)
    for clip in clips:
        asset = assets.get(clip.asset_id)
        if asset is None:
            raise ValueError(f"clip {clip.id}: asset {clip.asset_id} not found")
        if not asset.has_video:
            raise ValueError(
                f"clip {clip.id}: asset {asset.id} has no video stream; "
                "put audio-only assets on an audio track"
            )

    # One tile per clip is the floor, so a one-second clip in a long edit is still shown.
    total_clips = len(clips)
    omitted = max(0, total_clips - MAX_STORYBOARD_TILES)
    if omitted:
        stride = total_clips / MAX_STORYBOARD_TILES
        clips = [clips[min(int(stride * index), total_clips - 1)] for index in range(MAX_STORYBOARD_TILES)]
    per_clip = [1] * len(clips)
    for _ in range(max(count, len(clips)) - len(clips)):
        # Give the next frame to whichever clip currently covers the most seconds per frame.
        longest = max(range(len(clips)), key=lambda index: float(clips[index].timeline_duration) / per_clip[index])
        per_clip[longest] += 1

    shots: List[tuple[str, float, str]] = []
    listing: List[str] = []
    previous_out = Decimal(0)
    for clip, frames in zip(clips, per_clip):
        if clip.timeline_in > previous_out:
            listing.append(
                f"-- black gap {format_timestamp(float(previous_out))} to "
                f"{format_timestamp(float(clip.timeline_in))} "
                f"({float(clip.timeline_in - previous_out):.3f}s)"
            )
        asset = assets[clip.asset_id]
        start, end = float(clip.source_range.start), float(clip.source_range.end)
        step = (end - start) / frames
        for offset in range(frames):
            # Sample the middle of each interval, and stay clear of the very last frame, which may not decode.
            seconds = start + step * (offset + 0.5)
            if asset.duration is not None:
                seconds = min(seconds, float(asset.duration) - 0.05)
            seconds = round(max(seconds, 0.0), 3)
            at = float(clip.timeline_in) + (seconds - start) / clip.speed
            number = len(shots) + 1
            label = clip.id if len(clip.id) <= 10 else f"{clip.id[:9]}~"
            shots.append((asset.path, seconds, f"#{number}  {format_timestamp(at)}  {label}"))
            listing.append(
                f"#{number}: clip {clip.id} | edit {format_timestamp(at)} | "
                f"source {seconds:.3f}s ({format_timestamp(seconds)})"
            )
        previous_out = clip.timeline_out

    for track in project.video_tracks[1:]:
        for clip in sorted(track.clips, key=lambda item: item.timeline_in):
            box = clip.layout
            where = (
                "over the whole frame" if box is None
                else f"at {box.x:.0%} across and {box.y:.0%} down, {box.width:.0%} x {box.height:.0%} of the frame"
            )
            listing.append(
                f"-- inset on track {track.id}: clip {clip.id}, "
                f"{format_timestamp(float(clip.timeline_in))} to {format_timestamp(float(clip.timeline_out))}, {where}"
            )

    for track in project.tracks:
        if track.track_type != TrackType.AUDIO or not track.clips:
            continue
        covered_from = min(clip.timeline_in for clip in track.clips)
        covered_to = max(clip.timeline_out for clip in track.clips)
        volumes = sorted({clip.volume for clip in track.clips})
        level = f"volume {volumes[0]}" if len(volumes) == 1 else f"volumes {', '.join(str(v) for v in volumes)}"
        listing.append(
            f"-- audio track {track.id}: {len(track.clips)} clip(s), "
            f"{format_timestamp(float(covered_from))} to {format_timestamp(float(covered_to))}, {level}"
        )

    header = (
        f"Storyboard of project {project_id} in timeline order: {project.width}x{project.height}, "
        f"{project.fps_num}/{project.fps_den} fps, {format_timestamp(float(project.duration))} long, "
        f"{len(shots)} frames from {total_clips} clip(s)."
    )
    if omitted:
        header += f" {omitted} clip(s) were skipped: the project has more clips than tiles."
    return ToolResult(content=[
        "\n".join([header, *listing]),
        Image(data=storyboard_sheet(shots, aspect=project.width / project.height), format="jpeg"),
    ])

@mcp.tool()
def generate_subtitles(
    project_id: str,
    max_characters: Annotated[int, Field(ge=8, le=120)] = DEFAULT_MAX_CHARACTERS,
    max_seconds: Annotated[float, Field(gt=0.2, le=15)] = DEFAULT_MAX_SECONDS,
) -> dict:
    """Propose captions for the edited sequence, from transcripts already made.

    Reads the transcript of every clip on the base video track, keeps only
    the speech that survived the edit, and moves each line to where it now
    falls in the result. Insets on video tracks above the base are pictures
    over that sound, so they are not captioned. Long sentences are broken between words. Nothing is saved:
    check the wording, fix any name the transcript misheard, and then store
    the captions with a `set_subtitles` operation in `apply_edits`. Render
    them into the picture with `render_project` and `burn_subtitles`.

    Args:
        project_id: ID of the project to caption.
        max_characters: Longest caption text before it is broken in two.
        max_seconds: Longest caption before it is broken in two.

    Returns:
        A dictionary with `cues`, each holding its `id`, `start` and `end` in
        seconds on the timeline, and its `text`, and
        `assets_without_transcript`, the assets that still need
        `analyze_asset` before they can be captioned.

    Raises:
        ValueError: If the project does not exist, or no clip on its video
            track has been transcribed.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")

    transcripts = {}
    untranscribed: List[str] = []
    base = project.base_video_track
    for clip in base.clips if base else []:
        if clip.asset_id in transcripts or clip.asset_id in untranscribed:
            continue
        analysis = repo.get_analysis(clip.asset_id)
        if analysis is not None and analysis.transcript is not None:
            transcripts[clip.asset_id] = analysis.transcript
        else:
            untranscribed.append(clip.asset_id)

    if not transcripts:
        raise ValueError(
            f"project {project_id} has no transcribed clips to caption; "
            "call analyze_asset on its sources first"
        )
    cues = timeline_cues(project, transcripts, max_characters=max_characters, max_seconds=max_seconds)
    return {
        "cues": [cue.model_dump() for cue in cues],
        "assets_without_transcript": untranscribed,
    }

@mcp.tool()
def get_subtitles(
    project_id: str,
    start: Annotated[float, Field(ge=0)] = 0,
    end: Annotated[Optional[float], Field(ge=0)] = None,
) -> dict:
    """Read the captions stored on a project, a window at a time.

    A long video has hundreds of captions, so read the stretch you need
    rather than all of them. Each caption's `id` is what `edit_subtitle`
    takes, so one wrong word can be corrected on its own.

    Omitting `end` reads every stored caption, including any left stranded
    past the end of the video by a later edit: those never appear in the
    render, and this is where you find them to fix or delete.

    Args:
        project_id: ID of the project to read.
        start: Start of the window in seconds on the timeline.
        end: End of the window in seconds; omit for every caption there is.

    Returns:
        A dictionary with `cues` in the window, the `total` number of captions
        on the project, and the `window` that was read.

    Raises:
        ValueError: If the project does not exist, or `end` is not after
            `start`.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if end is None:
        # Past the end of the video too: a caption you cannot see is one you cannot correct.
        limit = max([float(project.duration), *(float(cue.end) for cue in project.subtitles)])
    else:
        limit = end
        if limit <= start:
            raise ValueError(f"the window ends at {limit}s, which is not after its start at {start}s")
    window = [
        cue.model_dump() for cue in project.subtitles
        if float(cue.end) > start and float(cue.start) < limit
    ]
    return {"cues": window, "total": len(project.subtitles), "window": {"start": start, "end": limit}}

@mcp.tool()
def render_project(
    project_id: str,
    is_preview: bool = False,
    loudness_target: Annotated[Optional[float], Field(ge=-40, le=-5)] = DEFAULT_LOUDNESS_TARGET,
    burn_subtitles: bool = False,
) -> dict:
    """Start rendering a project to an MP4 file in the background.

    Returns immediately. Poll `get_job` with the returned `job_id` until the
    job reaches a terminal status, or stop it with `cancel_job`. Rendering
    runs in a separate worker process and continues even if this server
    restarts.

    Args:
        project_id: ID of the project to render.
        is_preview: If true, render a fast 480p preview instead of the
            full-quality output.
        loudness_target: Loudness of the finished file in LUFS. The default,
            -14, is what streaming platforms normalize to, so clips recorded
            on different devices come out at one consistent level instead of
            jumping. Pass null to leave the mix exactly as the clip volumes
            set it.
        burn_subtitles: If true, burn the project's stored captions into the
            picture. Store them first with a `set_subtitles` operation;
            `generate_subtitles` proposes them from the transcripts.

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

    subtitle_path = None
    if burn_subtitles:
        if not project.subtitles:
            raise ValueError(
                f"project {project_id} has no captions to burn; "
                "propose them with generate_subtitles and store them with a set_subtitles operation"
            )
        os.makedirs(job.work_dir, exist_ok=True)
        subtitle_path = os.path.join(job.work_dir, "subtitles.ass")
        with open(subtitle_path, "w", encoding="utf-8") as handle:
            handle.write(build_ass(project.subtitles, project.width, project.height))

    command = renderer.build_command(
        project, _referenced_assets(project), job.output_path,
        is_preview=is_preview, loudness_target=loudness_target, subtitle_path=subtitle_path,
    )
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
