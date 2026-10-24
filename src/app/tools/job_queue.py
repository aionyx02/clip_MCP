"""Background jobs: how each one is doing, where it stands in the queue, and cancelling it."""

import os
import time
from datetime import datetime, timezone
from typing import Annotated, Dict, List, Optional, Sequence

from pydantic import Field

from app.models.timeline.parts import Clip
from app.models.job import Job, JobKind, JobStatus
from app.engine import stills, sync
from app.engine.analysis import CARD_LOG_SUFFIX, DETECT_LOG_FILE_NAME
from app.engine.ffmpeg import card_failed_path, card_failure
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
        `input`: a file of the library it reads is missing or cannot be
        opened, and the same call fails again until it can; `made`: a file
        the server made for it, a picture from the cache, the chapters or
        the words drawn on, was gone, and the same call again makes it anew; `render`: FFmpeg
        failed with every file it reads there; `worker`: its worker
        never started, crashed or was stopped, and the same call again may
        well work; `cancelled`; left out when the cause is none of these —
        for a finished analysis worth a second look a `note` to pass on — a
        crowd of voices, or a picture the NVIDIA card failed to decode, which
        makes later scans slower until the user deletes the record it names —
        and
        for a finished render whose every frame was found black, frozen or a
        flash somewhere, `frames`: `found`, what the
        cut did not ask for — look at them with `view_render` `at: "spots"` —
        and `meant`, inside a fade, a dip, a card or a photo; or `not_read`
        when the check could not read it, which says nothing about its frames; for a finished
        render whose sound does not line up with its picture, `sound`: `out_of_step`, a line
        each for a stream that starts or runs elsewhere than the picture and for a side of a
        join where the sound is heard early or late, naming the clip and the side; `not_compared`
        when some sides of joins could not be lined up, saying how many and why — left out before
        the check for too little of their own sound or for their own fade, their sound did not follow
        the file's closely enough within the shifts searched, their file could not be read, or their
        sound may be further off than was searched, each naming how far it was searched;
        or `not_checked`
        when the check could not be run, which says nothing about the sound; and a summary
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
            if job.sync is None:
                entry["sound"] = {
                    "not_checked": "the sound was not checked against the picture, so whether it lines up is not known"
                }
            else:
                sound = _sound_out_of_step(job)
                if sound:
                    entry["sound"] = sound
        if job.kind == JobKind.ANALYZE and job.status == JobStatus.COMPLETED and job.asset_id:
            heard = runtime.repo.get_analysis(job.asset_id)
            voices = len({turn.speaker for turn in heard.speakers}) if heard else 0
            notes = []
            if voices > MANY_VOICES:
                notes.append(
                    f"heard {voices} different voices, more than a video usually has — noisy talk splits one "
                    "person into many. If you know how many people talk, ask the user and analyze it again "
                    "with `speakers` and `again`"
                )
            card = _card_note(job)
            if card:
                notes.append(card)
            if notes:
                entry["note"] = "; ".join(notes)
        listed.append(entry)
    return listed

def _card_note(job: Job) -> Optional[str]:
    """What to tell the user when an analysis's picture could not be decoded on the NVIDIA card.

    Args:
        job: A finished analysis.

    Returns:
        The note, or None when the scan did not fail on the card.
    """
    if not job.work_dir or not os.path.exists(os.path.join(job.work_dir, DETECT_LOG_FILE_NAME + CARD_LOG_SUFFIX)):
        return None
    if card_failure() is None:
        return ("its picture could not be decoded on the NVIDIA card and was decoded in software; the card is "
                "not turned off now, so the next file will try it again")
    return ("its picture could not be decoded on the NVIDIA card and was decoded in software. Tell the user: "
            "scans no longer use the card, so long phone or drone footage now takes about twice as long to "
            f"analyze, until they delete {card_failed_path()} (`clip-mcp check` shows what failed)")

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

def _searched(span: List[float]) -> str:
    """Say how far some sides were searched each way, as `check` recorded it: one reach, or the range of them."""
    least, most = span
    return f"{most:g}s" if least == most else f"{least:g}–{most:g}s (a shorter side searches less)"

def _sound_out_of_step(job: Job) -> Dict[str, object]:
    """Say where a finished render's sound does not line up with its picture, and what was not known.

    Args:
        job: A finished render that was checked.

    Returns:
        `out_of_step`, a line each for a stream that starts or runs elsewhere
        than the picture, and for a side of a join where the sound is heard
        more than a frame, or two envelope steps, early or late, naming the
        clip and which side of the join it is; and `not_compared` when some sides
        could not be lined up, saying how many and why — sides left out
        before the check for too little of their own sound or for their own
        fade, sides whose sound did not follow the file's closely enough or
        whose file could not be read, and sides whose sound may be further off
        than was searched. How far those sides were searched is said as it
        was: a short side searches less than `MAX_LAG_SECONDS`.
        Empty when every side that was compared lines up and none were left out.
    """
    if job.sync is None:
        return {}
    frame = float(job.sync["frame"])
    threshold = max(frame, 2 * sync.LAG_STEP_SECONDS)
    lines: List[str] = []
    streams = job.sync["streams"]
    if streams:
        lines.append(
            f"the sound runs {streams['audio_start']:.2f}s to {streams['audio_start'] + streams['audio_length']:.2f}s "
            f"while the picture runs {streams['video_start']:.2f}s to "
            f"{streams['video_start'] + streams['video_length']:.2f}s"
        )
    joins = job.sync["joins"]
    over = [join for join in joins if abs(float(join["lag"])) > threshold]
    for join in over:
        lag = float(join["lag"])
        at = float(join["at"])
        lines.append(
            f"at {at:.2f}s, clip {join['clip_id']}, {join['side']} the join: "
            f"the sound is heard {abs(lag):.2f}s {'late' if lag > 0 else 'early'}"
        )
    found: Dict[str, object] = {}
    if lines:
        found["out_of_step"] = lines
    sides_asked = int(job.sync["sides_asked"])
    too_short = int(job.sync["left_out"]["too_short"])
    faded = int(job.sync["left_out"]["faded"])
    weak = int(job.sync["unlined"]["weak"])
    beyond_reach = int(job.sync["unlined"]["beyond_reach"])
    unknown = too_short + faded + weak + beyond_reach
    total = sides_asked + too_short + faded
    if unknown > 0:
        reasons: List[str] = []
        if too_short:
            reasons.append(f"{too_short} with less than "
                           f"{sync.MIN_SIDE_SECONDS + sync.MIN_REACH_SECONDS:g}s of their own sound "
                           "clear of other sound")
        if faded:
            reasons.append(f"{faded} where the clip's own fade leaves less than "
                           f"{sync.MIN_SIDE_SECONDS + sync.MIN_REACH_SECONDS:g}s to compare")
        searched = job.sync["unlined"]["searched"]
        if weak:
            # At no shift searched did the shapes agree: the sound may be further off than the search, or
            # changed too much (quiet room tone under the loudness levelling) to be followed; a file that
            # could not be read is counted here too.
            reasons.append(f"{weak} whose sound followed the file's closely enough at no shift within "
                           f"{_searched(searched['weak'])}: further off, too changed to compare, "
                           "or its file unreadable")
        if beyond_reach:
            reasons.append(f"{beyond_reach} whose sound may be further off than the "
                           f"{_searched(searched['beyond_reach'])} searched")
        found["not_compared"] = (
            f"{unknown} of the {total} sides of joins could not be lined up — " + ", ".join(reasons)
            + " — so whether they are in step is not known"
        )
    return found

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
