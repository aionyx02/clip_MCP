"""Background jobs: how each one is doing, where it stands in the queue, and cancelling it."""

import time
from datetime import datetime, timezone
from typing import Annotated, Dict, List, Optional, Sequence

from pydantic import Field

from app.models.timeline import Clip
from app.models.job import Job, JobKind, JobStatus
from app.engine import stills
from app import runtime
from app.runtime import mcp
from app.tools import looking, shared

# Provisional (roadmap §13): the longest `get_job` waits, under the same allowance.
MAX_JOB_WAIT_SECONDS = 50.0

@mcp.tool()
def get_job(
    job_ids: List[str], wait_seconds: Annotated[float, Field(ge=0, le=MAX_JOB_WAIT_SECONDS)] = 0,
) -> dict:
    """Return the current state of background jobs, waiting for one to end if asked.

    Pass every job of a batch at once — all the analyses of a folder, or a
    render — rather than polling them one by one. With `wait_seconds`, the
    answer comes back as soon as any of them finishes, fails or is cancelled,
    or when that long has passed: call it again with the same wait until the
    batch is done, instead of asking every few seconds.

    Args:
        job_ids: IDs of the jobs, as returned by `render_project` or
            `analyze_asset`.
        wait_seconds: How long to wait for one of them to end, at most 50 —
            under the time most clients allow a tool call. 0 answers at once.

    Returns:
        A dictionary with `jobs`, each with its `job_id`, `kind`, `status`,
        `progress` (0.0 to 1.0), the current `stage`, the `asset_id` of an
        analysis or the `output_path` of a render, `remaining_seconds` once a
        running job has gone far enough for its pace to say, `ahead` — how
        many queued jobs go before a queued one — for a failed job its
        `error_message`, for a failed or cancelled job its `failure_kind` —
        `input`: a file it reads is missing or cannot be opened, and the
        same call fails again until it can; `render`: FFmpeg failed with
        every file there; `worker`: its worker never started, crashed or
        was stopped, and the same call again may well work; `cancelled`;
        left out when the cause is none of these — for a finished analysis
        worth a second look a `note` to pass on, and for a finished render whose every frame was
        found black, frozen or a flash somewhere, `frames`: `found`, what the
        cut did not ask for — look at them with `view_render` `at: "spots"` —
        and `meant`, inside a fade, a dip, a card or a photo; or `not_read`
        when the check could not read it, which says nothing about its frames; and a summary
        over all of them: `finished` and
        `total` counts, `failed` counting those that failed or were
        cancelled, and `progress`, the average.

    Raises:
        ValueError: If a job does not exist.
    """
    ended = {JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}
    listed = _job_states(job_ids)
    deadline = time.monotonic() + wait_seconds
    waiting = {entry["job_id"] for entry in listed if entry["status"] not in ended}
    while waiting and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        listed = _job_states(job_ids)
        if any(entry["job_id"] in waiting and entry["status"] in ended for entry in listed):
            break
    return {
        "jobs": listed,
        "finished": sum(entry["status"] in ended for entry in listed),
        "failed": sum(entry["status"] in ended - {JobStatus.COMPLETED.value} for entry in listed),
        "total": len(listed),
        "progress": round(sum(entry["progress"] for entry in listed) / len(listed), 3) if listed else 1.0,
    }

def queue_positions(active: Sequence[Job]) -> Dict[str, int]:
    """Say how many queued jobs go before each queued one.

    Args:
        active: The jobs still queued or running, in the order they will run.

    Returns:
        Each queued job's place, 0 for the next to start.
    """
    waiting = sorted((job for job in active
                      if job.status == JobStatus.QUEUED and job.admitted_at is None and not job.cancel_requested),
                     key=lambda job: job.priority)
    return {job.job_id: place for place, job in enumerate(waiting)}

def _job_states(job_ids: List[str]) -> List[dict]:
    """Read each job's state, one record apiece, in the order asked.

    Args:
        job_ids: The jobs.

    Returns:
        Their records.

    Raises:
        ValueError: If a job does not exist.
    """
    listed: List[dict] = []
    now = datetime.now(timezone.utc)
    ahead = queue_positions(runtime.repo.jobs_to_show(now)[0])
    for job_id in dict.fromkeys(job_ids):
        job = runtime.job_manager.get_job(job_id)
        if not job:
            raise ValueError(f"job {job_id} not found")
        entry = {
            "job_id": job.job_id, "kind": job.kind.value, "status": job.status.value,
            "progress": round(job.progress, 3), "stage": job.stage,
        }
        if job.asset_id:
            entry["asset_id"] = job.asset_id
        if job.output_path:
            entry["output_path"] = job.output_path
        remaining = job.remaining_seconds(now)
        if remaining is not None:
            entry["remaining_seconds"] = remaining
        if job.job_id in ahead:
            entry["ahead"] = ahead[job.job_id]
        if job.error_message:
            entry["error_message"] = job.error_message
        if job.failure_kind:
            entry["failure_kind"] = job.failure_kind.value
        if job.kind == JobKind.RENDER and job.status == JobStatus.COMPLETED:
            if job.spots is None:
                entry["frames"] = {"not_read": "every frame was not read — the check failed or was stopped — so "
                                               "black, frozen or flashing frames in it are not known"}
            elif job.spots:
                entry["frames"] = _frame_spots(job)
        if job.kind == JobKind.ANALYZE and job.status == JobStatus.COMPLETED and job.asset_id:
            heard = runtime.repo.get_analysis(job.asset_id)
            voices = len({turn.speaker for turn in heard.speakers}) if heard else 0
            if voices > MANY_VOICES:
                entry["note"] = (
                    f"heard {voices} different voices, more than a video usually has — noisy talk splits one "
                    "person into many. If you know how many people talk, ask the user and analyze it again "
                    "with `speakers` and `again`"
                )
        listed.append(entry)
    return listed

def _frame_spots(job: Job) -> dict:
    """Say where every frame of a render was found black, frozen or a flash, and which of it the cut meant.

    Black inside a fade or a dip through black, and a card or a photo held
    still, are what the cut asked for; the rest is what nobody asked for — two
    pieces that do not quite meet, a frame of the wrong shot — and is said
    first, as `found`.

    Args:
        job: A finished render.

    Returns:
        `found` and `meant`, a line each: when, what, and the clips there.
    """
    project, _ = looking.project_as_rendered(job)
    clips = sorted(project.base_video_track.clips, key=lambda clip: clip.timeline_in) if (
        project is not None and project.base_video_track is not None) else []
    assets = shared.referenced_assets(project) if project is not None else {}
    slack = 1 / (project.fps_num / project.fps_den) if project is not None else 0.04

    def meant(kind: str, start: float, end: float, here: List[Clip]) -> Optional[str]:
        for clip in here:
            begin, finish = float(clip.timeline_in), float(clip.timeline_out)
            if clip.card is not None:
                return f"card {clip.id}"
            asset = assets.get(clip.asset_id)
            if kind == "frozen" and asset is not None and stills.is_image(asset.path):
                return f"photo {clip.id}"
            if kind == "black" and clip.video_fade_in and start >= begin - slack and end <= begin + float(
                    clip.video_fade_in) + slack:
                return f"fade in of {clip.id}"
            if kind == "black" and clip.video_fade_out and end <= finish + slack and start >= finish - float(
                    clip.video_fade_out) - slack:
                return f"fade out of {clip.id}"
        # A dip ends on the cut, so its black lies wholly inside the clip before the one it leads
        # into; every clip's dip is looked at, not only the clips the black falls in.
        for clip in clips:
            begin, dip = float(clip.timeline_in), clip.transition_in
            if (kind in ("black", "flash") and dip is not None and dip.kind == "dip"
                    and start >= begin - float(dip.seconds) - slack and end <= begin + slack):
                part = next((marker.name for marker in project.markers
                             if abs(float(marker.timeline_in) - begin) <= slack), None) if project is not None else None
                return f"dip into {clip.id}" + (f", where 「{part}」 begins" if part else "")
        return None

    found, kept = [], []
    for spot in job.spots:
        start, end = float(spot["start"]), float(spot["end"])
        here = [clip for clip in clips if float(clip.timeline_in) < max(end, start + slack)
                and float(clip.timeline_out) > start]
        where = " → ".join(f"clip {clip.id}" for clip in here) or "past the cut"
        line = f"{start:.2f}–{end:.2f}s {spot['kind']} ({end - start:.2f}s) in {where}"
        reason = meant(str(spot["kind"]), start, end, here)
        (kept if reason else found).append(line + (f": the {reason}" if reason else ""))
    return {"found": found, "meant": kept}

# Provisional (roadmap §13): more voices than this in one file is far more likely a noisy
# recording split apart than that many people — 40 for two friends talking over traffic.
MANY_VOICES = 8

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
    job = runtime.job_manager.cancel_job(job_id)
    return {
        "job_id": job_id,
        "cancelled": job is not None and job.status == JobStatus.CANCELLED,
        "status": job.status.value if job else None,
    }
