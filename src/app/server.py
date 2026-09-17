import os
import uuid
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field
from app.models.timeline import Project, EditOperation, AddClipOp, apply_operation
from app.models.job import JobStatus
from app.engine.probe import probe_file
from app.engine.builder import FFmpegRenderer
from app.storage.repo import JobManager

mcp = FastMCP(
    "VideoEditingCore",
    instructions=(
        "Timeline-based video editing backed by FFmpeg. Typical workflow: "
        "import_asset for each source file, create_project, apply_edits with "
        "add_track and add_clip operations, then render_project and poll "
        "get_job until the job finishes. All state is kept in memory and is "
        "lost when the server restarts."
    ),
)

PROJECT_DB: dict[str, Project] = {}
ASSET_MAP: dict[str, str] = {}
job_manager = JobManager()
renderer = FFmpegRenderer()

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

    Importing the same path again returns the existing asset ID. Rendering
    requires every clip's asset to contain both a video and an audio stream.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.

    Returns:
        A dictionary with the `asset_id`, the absolute `path`, the `duration`
        in seconds (or null if unknown), and `has_video` / `has_audio` flags.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    path = os.path.abspath(filepath)
    info = probe_file(path)

    asset_id = next((aid for aid, asset_path in ASSET_MAP.items() if asset_path == path), None)
    if asset_id is None:
        asset_id = str(uuid.uuid4())
        ASSET_MAP[asset_id] = path

    stream_types = {stream.get("codec_type") for stream in info.get("streams", [])}
    duration = info.get("format", {}).get("duration")
    return {
        "asset_id": asset_id,
        "path": path,
        "duration": float(duration) if duration is not None else None,
        "has_video": "video" in stream_types,
        "has_audio": "audio" in stream_types,
    }

@mcp.tool()
def list_assets() -> dict:
    """List all imported assets.

    Returns:
        A dictionary with an `assets` list, each item holding an `asset_id`
        and its absolute `path`.
    """
    return {"assets": [{"asset_id": asset_id, "path": path} for asset_id, path in ASSET_MAP.items()]}

@mcp.tool()
def create_project(
    width: Annotated[int, Field(gt=0)] = 1920,
    height: Annotated[int, Field(gt=0)] = 1080,
    fps_num: Annotated[int, Field(gt=0)] = 30,
    fps_den: Annotated[int, Field(gt=0)] = 1,
) -> dict:
    """Create an empty project.

    Add tracks and clips to the new project with `apply_edits`.

    Args:
        width: Output width in pixels.
        height: Output height in pixels.
        fps_num: Frame rate numerator, e.g. 30000 for 29.97 fps.
        fps_den: Frame rate denominator, e.g. 1001 for 29.97 fps.

    Returns:
        The new project serialized as a dictionary, including its generated
        `id` and initial `version`.
    """
    project = Project(id=str(uuid.uuid4()), width=width, height=height, fps_num=fps_num, fps_den=fps_den)
    PROJECT_DB[project.id] = project
    return project.model_dump()

@mcp.tool()
def list_projects() -> dict:
    """List all projects.

    Returns:
        A dictionary with a `projects` list, each item holding a project's
        `id` and current `version`.
    """
    return {"projects": [{"id": project.id, "version": project.version} for project in PROJECT_DB.values()]}

@mcp.tool()
def get_project(project_id: str) -> dict:
    """Return the full state of a project, including its tracks and clips.

    The returned `version` is required as `expected_version` when calling
    `apply_edits`.

    Args:
        project_id: ID of the project to retrieve.

    Returns:
        The project serialized as a dictionary.

    Raises:
        ValueError: If no project with `project_id` exists.
    """
    project = PROJECT_DB.get(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    return project.model_dump()

@mcp.tool()
def apply_edits(project_id: str, expected_version: int, operations: list[EditOperation]) -> dict:
    """Apply a batch of edit operations to a project with optimistic locking.

    Operations are applied in order and atomically: if any operation fails,
    the project is left unchanged. On success the project version is
    incremented by one. Call `get_project` first to obtain the current
    version. Clips on each track are rendered in `timeline_in` order.

    Args:
        project_id: ID of the project to edit.
        expected_version: Project version the caller last read. The request is
            rejected if the project has been modified since then.
        operations: Edit operations to apply, each identified by its `action`:
            `add_track`, `add_clip`, `trim_clip`, `move_clip`, or `delete_clip`.

    Returns:
        A dictionary with the `status` and the project's `new_version`.

    Raises:
        ValueError: If the project does not exist, the version does not match,
            or an operation references a missing or duplicate track, clip, or
            asset.
    """
    project = PROJECT_DB.get(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if project.version != expected_version:
        raise ValueError(f"version conflict: current version is {project.version}, expected version is {expected_version}")

    for op in operations:
        if isinstance(op, AddClipOp) and op.asset_id not in ASSET_MAP:
            raise ValueError(f"asset {op.asset_id} not found")

    draft = project.model_copy(deep=True)
    for op in operations:
        apply_operation(draft, op)

    draft.version += 1
    PROJECT_DB[project_id] = draft
    return {"status": "success", "new_version": draft.version}

@mcp.tool()
def render_project(project_id: str, is_preview: bool = False) -> dict:
    """Start rendering a project to an MP4 file in the background.

    Returns immediately. Poll `get_job` with the returned `job_id` until the
    job reaches a terminal status, or stop it with `cancel_job`.

    Args:
        project_id: ID of the project to render.
        is_preview: If true, render a fast 480p preview instead of the
            full-quality output.

    Returns:
        A dictionary with the `job_id`, the initial `status`, and the absolute
        `output_path` the file will be written to.

    Raises:
        ValueError: If the project does not exist or contains no clips.
    """
    project = PROJECT_DB.get(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")

    out_ext = "preview.mp4" if is_preview else "output.mp4"
    output_path = os.path.abspath(f"./workspace/outputs/{project_id}_{out_ext}")

    cmd = renderer.build_command(project, ASSET_MAP, output_path, is_preview=is_preview)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    proc = renderer.execute(cmd, cancel_token={})
    job = job_manager.register_job(project_id, output_path)
    job_manager.processes[job.job_id] = proc
    job.status = JobStatus.RUNNING

    return {"job_id": job.job_id, "status": job.status.value, "output_path": output_path}

@mcp.tool()
def get_job(job_id: str) -> dict:
    """Return the current state of a render job.

    If the job's FFmpeg process has exited, the job is first updated to
    `completed` or `failed` accordingly.

    Args:
        job_id: ID of the job, as returned by `render_project`.

    Returns:
        The job serialized as a dictionary, including `status`, `progress`,
        `output_path`, and, for failed jobs, `error_message`.

    Raises:
        ValueError: If no job with `job_id` exists.
    """
    job = job_manager.jobs.get(job_id)
    if not job:
        raise ValueError(f"job {job_id} not found")

    proc = job_manager.processes.get(job_id)
    if proc and job.status == JobStatus.RUNNING:
        retcode = proc.poll()
        if retcode is not None:
            if retcode == 0:
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
            else:
                job.status = JobStatus.FAILED
                job.error_message = proc.stderr.read().decode("utf-8", errors="replace")[-500:]

    return job.model_dump()

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Cancel a queued or running render job.

    Args:
        job_id: ID of the job, as returned by `render_project`.

    Returns:
        A dictionary with the `job_id` and a `cancelled` flag, which is false
        if the job does not exist or has already finished.
    """
    success = job_manager.cancel_job(job_id)
    return {"job_id": job_id, "cancelled": success}

def main():
    """Run the MCP server over the stdio transport."""
    mcp.run()

if __name__ == "__main__":
    main()
