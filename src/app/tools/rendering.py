"""Turning a project into sound and picture: previews, renders, the check before a render, and playback copies.

Also what a render needs to know about a project first: who is talking where, how each
file is framed for the shape asked for, and the analyses it leans on.
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from typing import Annotated, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field

from app.models.media import Asset, MediaAnalysis
from app.models.plan import EditPlan
from app.models.timeline.parts import Clip, Project, TrackType
from app.models.job import Job, JobKind, JobStatus
from app.engine import cards, legibility, looks, playback, resources, sync, texts
from app.engine.analysis import second_listen, with_words_heard_again
from app.engine.probe import probe_file, speech_loudness
from app.engine.reframe import Framing, centred_note, frame_project, framed_size, placement_at
from app.engine.builder import DEFAULT_LOUDNESS_TARGET, Talking, voice_keys
from app.engine.levels import song_level, talking_in
from app.engine.delivery import (
    CHECKS, chapter_metadata, chapters, check_delivery, light_steps, look_candidates, run_on_cuts, shaky_clips,
)
from app.engine.frames import format_timestamp, still, storyboard_sheet
from app.engine.subtitles import build_ass, place_cues
from app.storage import housekeeping
from app.workspace import output_dir
from app import runtime
from app.runtime import mcp
from app.tools import production, shared

# Characters a Windows or POSIX filesystem will not take in a name, plus the control
# range. A project is named by whoever is editing, so a render cannot assume anything
# about what is in that name.
FORBIDDEN_IN_NAMES = frozenset('<>:"/\\|?*') | {chr(code) for code in range(32)}
# What each of those becomes in a file name: the full-width mark that looks like it, so
# 「8/29 暴雨」 is saved as 「8／29 暴雨」 rather than 「8 29 暴雨」 (#96); a control character
# has no look to keep and becomes a space.
NAME_LOOKALIKES = {"<": "＜", ">": "＞", ":": "：", '"': "＂", "/": "／", "\\": "＼", "|": "｜", "?": "？",
                   "*": "＊"}
MAX_OUTPUT_NAME = 60

def file_gone(job: Job) -> Optional[str]:
    """Say what became of a finished render's file, when it is no longer there.

    A preview is replaced by the next one, so a job can have finished and its
    file have gone; the job's status alone does not say that (#195).

    Args:
        job: The render.

    Returns:
        What to do instead, or None when the file is there.
    """
    if not job.output_path or os.path.exists(job.output_path):
        return None
    later = [other for other in (runtime.repo.renders_of(job.project_id) if job.project_id else [])
             if other.job_id != job.job_id and other.preview == job.preview and other.frame == job.frame
             and other.created_at > job.created_at and other.output_path and os.path.exists(other.output_path)]
    if not later:
        return f"the file render {job.job_id} made is no longer there; render again"
    newest = max(later, key=lambda other: other.created_at)
    edited = " (of a later edit of the cut)" if newest.project_version != job.project_version else ""
    return f"the file render {job.job_id} made is no longer there; render {newest.job_id}{edited} is the one to use"

def finished_render(job_id: str) -> Tuple[Job, str]:
    """Find a render that has finished, and the file it made.

    Args:
        job_id: The render's job.

    Returns:
        `(job, path)`.

    Raises:
        ValueError: If it is not a render, has not finished, or its file is gone.
    """
    job = runtime.job_manager.get_job(job_id)
    if job is None or job.kind != JobKind.RENDER:
        raise ValueError(f"render {job_id} not found; `render_project` gives the job ID of one")
    if job.status != JobStatus.COMPLETED:
        raise ValueError(f"render {job_id} is {job.status.value}; wait for it with `get_job` first")
    gone = file_gone(job) if job.output_path else f"render {job_id} made no file; render again"
    if gone:
        raise ValueError(gone)
    return job, job.output_path

def rendered_from(job: Job) -> Optional[Project]:
    """The project a render was made from, as it is now; None when it is gone."""
    return runtime.repo.get_project(job.project_id) if job.project_id else None

def changed_since(job: Job) -> bool:
    """Whether the project a render was made from has been edited since.

    Args:
        job: The render.

    Returns:
        True when its project's version has moved on from the one rendered.
    """
    project = rendered_from(job)
    return project is not None and job.project_version is not None and project.version != job.project_version

# What a clip carries that only its sound hears; `playback.mix_key` covers all of it.
SOUND_ONLY = {"volume", "audio_fade_in", "audio_fade_out", "audio_lead", "audio_lag", "cleanup", "keep_level",
              "preserve_pitch"}
# What says where a project or a clip came from, or what it is called, and is in no video made of it.
NOT_RENDERED = {"id", "name", "branched_from", "version", "deleted_subtitles"}
CLIP_BOOKKEEPING = {"id", "from_plan_id", "from_clip_ids", "pinned"}

def picture_key(project: Project) -> str:
    """Name a cut's picture by everything that decides what is on screen.

    Two versions of a project with the same key show the same thing, however
    many edits lie between them. Together with `playback.mix_key`, which names
    the sound, it covers all a render is made from but the shape and size it
    is rendered at, which a render records apart.

    Args:
        project: The project.

    Returns:
        The key.
    """
    picture = project.model_dump(mode="json", exclude=NOT_RENDERED | {"tracks"})
    picture["tracks"] = [
        [clip.model_dump(mode="json", exclude=SOUND_ONLY | CLIP_BOOKKEEPING) for clip in track.clips]
        for track in project.video_tracks
    ]
    return hashlib.sha256(json.dumps(picture, sort_keys=True).encode("utf-8")).hexdigest()[:32]

@mcp.tool()
def preview_project(
    project_id: str,
    count: Annotated[int, Field(ge=1, le=shared.MAX_STORYBOARD_TILES)] = 12,
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    follow_faces: bool = True,
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
    sequence itself. Black gaps and audio tracks are listed in the text, since
    they cannot be seen in the frames, and so is anything on a higher video
    track — an inset, or covering picture over the whole frame, in which case
    the tiles for that stretch show the sequence underneath rather than what
    the viewer will see.

    Args:
        project_id: ID of the project to look at.
        count: Number of frames to sample. Raised to one per clip when the
            project has more clips than this.
        frame: Show the cut in another shape, the way `render_project` with
            the same `frame` would render it; omit for the project's own.
        follow_faces: Crop around the face the way the render does; the
            same setting as `render_project`'s.

    Returns:
        A text block describing the output format, every tile, the black gaps,
        and the audio tracks, followed by the storyboard as a JPEG image.

    Raises:
        ValueError: If the project does not exist, has no clips on a video
            track, or a video clip uses an asset without video.
        RuntimeError: If frames cannot be decoded.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    project = reshaped(project, frame)

    # Tiles come from the base track, which is the sequence; tracks above it are insets, noted in the text.
    base = project.base_video_track
    clips = sorted(base.clips, key=lambda clip: clip.timeline_in) if base else []
    if not clips:
        raise ValueError(f"project {project_id} has no clips on a video track to preview")

    assets = shared.referenced_assets(project)
    for clip in clips:
        asset = assets.get(clip.asset_id)
        if asset is None:
            raise ValueError(f"clip {clip.id}: asset {clip.asset_id} not found")
        if not asset.has_video:
            raise ValueError(
                f"clip {clip.id}: asset {asset.id} has no video stream; "
                "put audio-only assets on an audio track"
            )

    framing = framing_of(project, assets, follow_faces)
    fps = Fraction(project.fps_num, project.fps_den)

    # Where the picture really goes black, found over the whole sequence rather than over
    # the tiles chosen below. A clip left out of the storyboard is not a hole in the edit,
    # and reading the gaps off the sampled list turned every skipped clip into one.
    total_clips = len(clips)
    gaps: List[List[tuple[Decimal, Decimal]]] = [[] for _ in clips]
    running_out = Decimal(0)
    for index, clip in enumerate(clips):
        if clip.timeline_in > running_out:
            gaps[index].append((running_out, clip.timeline_in))
        running_out = clip.timeline_out

    def gap_line(opened: Decimal, closed: Decimal) -> str:
        """Describe one stretch of black picture for the listing."""
        return (
            f"-- black gap {format_timestamp(float(opened))} to "
            f"{format_timestamp(float(closed))} ({float(closed - opened):.3f}s)"
        )

    # One tile per clip is the floor, so a one-second clip in a long edit is still shown.
    omitted = max(0, total_clips - shared.MAX_STORYBOARD_TILES)
    if omitted:
        stride = total_clips / shared.MAX_STORYBOARD_TILES
        chosen = sorted({min(int(stride * index), total_clips - 1) for index in range(shared.MAX_STORYBOARD_TILES)})
    else:
        chosen = list(range(total_clips))
    # Every tile carries what fell between the tile before it and itself, so a clip that is
    # not drawn is still counted and a real gap behind it is still reported.
    spans = list(zip([-1, *chosen], chosen))
    skipped = [later - earlier - 1 for earlier, later in spans]
    carried = [
        [gap for index in range(earlier + 1, later + 1) for gap in gaps[index]]
        for earlier, later in spans
    ]
    clips = [clips[index] for index in chosen]
    per_clip = [1] * len(clips)
    for _ in range(max(count, len(clips)) - len(clips)):
        # Give the next frame to whichever clip currently covers the most seconds per frame.
        longest = max(range(len(clips)), key=lambda index: float(clips[index].timeline_duration) / per_clip[index])
        per_clip[longest] += 1

    shots: List[tuple] = []
    listing: List[str] = []
    for position, (clip, frames) in enumerate(zip(clips, per_clip)):
        if skipped[position]:
            listing.append(f"-- {skipped[position]} clip(s) not shown")
        listing.extend(gap_line(opened, closed) for opened, closed in carried[position])
        asset = assets[clip.asset_id]
        start, end = float(clip.source_range.start), float(clip.source_range.end)
        step = (end - start) / frames
        if clip.card is not None:
            # One tile: the card's background. Its words are in the listing, not on the tile.
            number = len(shots) + 1
            at = float(clip.timeline_in)
            shots.append((asset.path, float(clip.card.at), f"#{number}  {format_timestamp(at)}  title card",
                          placement_at(framing.get(clip.id), clip, float(clip.card.at), fps), clip.card))
            words = clip.card.title + (f" / {clip.card.subtitle}" if clip.card.subtitle else "")
            listing.append(f"#{number}: clip {clip.id} | edit {format_timestamp(at)} | "
                           f"{'closing card' if clip.card.end else 'title card'} 「{words}」 "
                           f"for {end - start:g}s on {clip.card.background}"
                           + (f", logo {clip.card.logo_asset_id} above the words" if clip.card.logo_asset_id else "")
                           + " (words not drawn on the tile)")
            continue
        for offset in range(frames):
            # Sample the middle of each interval, and stay clear of the very last frame, which may not decode.
            seconds = start + step * (offset + 0.5)
            if asset.duration is not None:
                seconds = min(seconds, float(asset.duration) - 0.05)
            seconds = round(max(seconds, 0.0), 3)
            at = float(clip.timeline_in) + (seconds - start) / clip.speed
            number = len(shots) + 1
            label = clip.id if len(clip.id) <= 10 else f"{clip.id[:9]}~"
            framed = framing.get(clip.id)
            shots.append((
                asset.path, seconds, f"#{number}  {format_timestamp(at)}  {label}",
                placement_at(framed, clip, seconds, fps),
            ))
            # Drawn over the render, not over this tile: said instead.
            over = [text.text for text in clip.shown_texts if text.start <= seconds < text.end]
            listing.append(
                f"#{number}: clip {clip.id} | edit {format_timestamp(at)} | "
                f"source {seconds:.3f}s ({format_timestamp(seconds)})"
                + (f" | text on screen: {' / '.join(f'「{words}」' for words in over)}" if over else "")
            )

    # Sampling can stop short of the last clip, so say what runs on past the final tile.
    trailing = total_clips - 1 - chosen[-1]
    if trailing:
        listing.append(f"-- {trailing} clip(s) not shown")
        for index in range(chosen[-1] + 1, total_clips):
            listing.extend(gap_line(opened, closed) for opened, closed in gaps[index])

    for track in project.video_tracks[1:]:
        for clip in sorted(track.clips, key=lambda item: item.timeline_in):
            box = clip.layout
            # A clip with no layout fills the frame, which means it replaces what the
            # tiles show rather than sitting in a corner of it. Said plainly, because a
            # storyboard of a B-roll pass otherwise looks like the pass never happened.
            what = (
                "covering picture, over the whole frame — the tiles in this stretch show "
                "the sequence underneath, not this" if box is None
                else f"inset at {box.x:.0%} across and {box.y:.0%} down, "
                     f"{box.width:.0%} x {box.height:.0%} of the frame"
            )
            listing.append(
                f"-- track {track.id}: clip {clip.id}, "
                f"{format_timestamp(float(clip.timeline_in))} to {format_timestamp(float(clip.timeline_out))}, {what}"
            )

    if project.logo is not None:
        logo = project.logo
        listing.append(
            f"-- logo {logo.asset_id} in the {logo.corner.replace('_', '-')} corner, "
            f"{logo.width:.0%} of the frame wide at {logo.opacity:.0%} opacity, "
            "over the whole cut but a card with a logo of its own"
            + (f", from brand kit 「{logo.brand_kit}」" if logo.brand_kit else "") + " (not drawn on the tiles)"
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
    unplaced = centred_note(framing)
    if unplaced:
        header += f" {unplaced}."
    return ToolResult(content=[
        "\n".join([header, *listing]),
        Image(data=storyboard_sheet(shots, aspect=project.width / project.height), format="jpeg"),
    ])

def output_name(project: Project, kind: str) -> str:
    """Name a rendered file after the project it came from.

    A render used to be `<job uuid>/<project uuid>_output.mp4`, which is two
    identifiers and nothing a person can pick out of a folder. That is why
    finished cuts end up copied somewhere else under a name somebody made up
    — and next to the footage is the worst of the places they end up, because
    a cut sitting among its own sources is indistinguishable from source.

    Args:
        project: The project being rendered.
        kind: `output` or `preview`.

    Returns:
        A file name. The project's name where it has one, with anything a
        filesystem would refuse taken out, and its ID where the name is
        missing or survives sanitising as nothing. Each render has its own
        directory, so two of the same project do not collide.
    """
    stem = "".join(NAME_LOOKALIKES.get(character, " ") if character in FORBIDDEN_IN_NAMES else character
                   for character in (project.name or ""))
    stem = " ".join(stem.split())[:MAX_OUTPUT_NAME].strip(" .")
    return f"{stem or project.id}_{kind}.mp4"

def renamed(project: Project) -> dict:
    """Say when a render's file is not named exactly as its project is, and why.

    Args:
        project: The project rendered.

    Returns:
        `file_named` with the name the file takes, when a character a file
        name cannot hold was changed; empty otherwise.
    """
    if not project.name or not any(character in FORBIDDEN_IN_NAMES for character in project.name):
        return {}
    stem = output_name(project, "x")[: -len("_x.mp4")]
    return {"file_named": f"「{stem}」: a file name cannot hold some of the characters in 「{project.name}」, "
                          "so they were swapped for ones that look alike"}

def _delivered(job_id: str) -> bool:
    """Whether a full render finished and its file was saved, so its working folder can go."""
    job = runtime.repo.get_job(job_id)
    return job is not None and job.status == JobStatus.COMPLETED

def delivery_name(project: Project, kind: str, extension: str, captioned: bool = False) -> str:
    """Name a file saved for the user after the project: `EP1 台北.mp4`, `EP1 台北 封面.jpg`.

    Args:
        project: The project it came from.
        kind: `output`, `cover` or `timeline`, with `-portrait` and the like
            for another shape.
        extension: With its dot.
        captioned: Whether its captions are burned in.

    Returns:
        A file name; the caller makes it unique in its folder.
    """
    stem = output_name(project, "x")[: -len("_x.mp4")]
    return housekeeping.delivery_name(stem, kind, extension, captioned)

# A preview is rendered with its short side this long, from pictures cached at that size.
PREVIEW_SHORT_SIDE = 480
# Cached preview pictures nobody has used for this long are thrown away. Housekeeping,
# not a judgement: a picture is re-rendered in seconds if it is wanted again.
PICTURE_CACHE_DAYS = 14

def preview_sized(project: Project, side: int = PREVIEW_SHORT_SIDE) -> Project:
    """Shrink a project to the size its preview is rendered at.

    Args:
        project: The project.
        side: The preview's short side.

    Returns:
        A copy with its short side `side` long, or the project itself when it
        is already that small.
    """
    short = min(project.width, project.height)
    if short <= side:
        return project
    scale = side / short
    even = lambda value: max(2, int(round(value * scale / 2)) * 2)
    return project.model_copy(update={"width": even(project.width), "height": even(project.height)})

def _picture_cache() -> str:
    """Find the cache of preview pictures, clearing out what has not been used lately.

    Returns:
        The cache directory.
    """
    folder = os.path.join(runtime.WORKSPACE_DIR, "cache", "pictures")
    os.makedirs(folder, exist_ok=True)
    stale = time.time() - PICTURE_CACHE_DAYS * 86400
    for entry in os.scandir(folder):
        try:
            # A part file is a picture some render never finished; a day is long enough
            # for any render still writing one to be done with it.
            limit = time.time() - 86400 if entry.name.endswith(".part.mp4") else stale
            if entry.stat().st_mtime < limit:
                os.remove(entry.path)
        except OSError:
            pass
    return folder

def reshaped(project: Project, frame: Optional[str]) -> Project:
    """Give a project another shape for one render, leaving the stored one alone.

    Args:
        project: The project.
        frame: `landscape`, `portrait`, `square`, or None for as it is.

    Returns:
        A copy at the size `framed_size` gives, or the project itself when no
        shape was asked for.
    """
    if frame is None:
        return project
    width, height = framed_size(project.width, project.height, frame)
    return project.model_copy(update={"width": width, "height": height})

def framing_of(project: Project, assets: Mapping[str, Asset], follow_faces: bool = True) -> Dict[str, Framing]:
    """Work out how each clip sits in the frame: where its crop goes, or whether it is shown whole.

    Args:
        project: The project, at the size it is being rendered at.
        assets: Its files.
        follow_faces: Whether a crop may follow the face in a shot.

    Returns:
        A framing per clip that is not simply cropped from the middle, keyed
        by clip ID.
    """
    faces = {}
    for asset_id in assets:
        analysis = runtime.repo.get_analysis(asset_id)
        if analysis is not None and analysis.faces:
            faces[asset_id] = analysis.faces
    return frame_project(project, shared.sized(assets), faces, follow_faces)

@mcp.tool()
def render_project(
    project_id: str,
    is_preview: bool = True,
    loudness_target: Annotated[Optional[float], Field(ge=-40, le=-5)] = DEFAULT_LOUDNESS_TARGET,
    burn_subtitles: bool = False,
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    follow_faces: bool = True,
    # Every kind of finding the check makes, from the check's own list, so a new one can always be allowed.
    allow: Optional[List[Literal[CHECKS]]] = None,
) -> dict:
    """Start rendering a project to an MP4 file in the background.

    Before a final render, read skill://clip-editing/platform-conventions.md;
    after any render, look at it with `view_render` and `inspect_join`. A
    final render is refused until a preview of the cut's picture as it is
    now, in the same shape, has finished: that preview is what was looked at.
    What is compared is what the video would contain, not how many edits were
    made, so an edit undone or a new name asks for nothing. When only the
    sound changed — the music moved, a level — hearing it with
    `preview_sound` is enough.

    Before anything is rendered the cut is checked the way `check_render`
    checks it, and a full render is refused while it finds anything: black
    or frozen picture on screen, a voice recorded clipping, a caption too
    tall for the frame, or a length far from what the plan asked for. Fix
    the cut, or name the kinds of finding the user has decided to live with
    in `allow`. Previews are never refused — they are how you look — but
    say what they found.

    Returns immediately. Poll `get_job` with the returned `job_id` until the
    job reaches a terminal status, or stop it with `cancel_job`. Rendering
    runs in a separate worker process and continues even if this server
    restarts.

    Args:
        project_id: ID of the project to render.
        is_preview: True, the default, renders a fast preview, 480 pixels on
            its short side; false renders the full-quality output. A preview renders each
            clip's picture once and keeps it: after a change, only the clips
            the change touched are rendered again, so a second preview of a
            long cut takes a fraction of the first.
        loudness_target: Loudness of the finished file in LUFS. The default,
            -14, is what streaming platforms normalize to, so clips recorded
            on different devices come out at one consistent level instead of
            jumping. Pass null to leave the mix exactly as the clip volumes
            set it.
        burn_subtitles: If true, burn the project's stored captions into the
            picture. Store them first with a `set_subtitles` operation;
            `generate_subtitles` proposes them from the transcripts.
        frame: Render the same cut in another shape — `landscape` (16:9),
            `portrait` (9:16) or `square` — without touching the project.
            The short side stays the project's, so a 1920x1080 project comes
            out at 1080x1920 portrait and 1080x1080 square. Call once per
            shape to deliver one cut to several platforms. Omit it for the
            project's own size.
        follow_faces: Where a shot is wider (or taller) than the frame it is
            shown in, crop it around the face the analysis found rather than
            from the middle. The crop holds still while the face stays inside
            it and cuts to a new framing once the face has sat near the edge
            for a couple of seconds — it never pans. Shots with no face seen
            are cropped from the middle either way.
        allow: Kinds of finding to render in spite of, after the user has
            heard about them and said so. Never fill this in on your own.

    Returns:
        A dictionary with the `job_id`, the initial `status`, `stage`, and the
        absolute `output_path` the file will be written to. A full render is
        saved in the user's videos folder under the project's name, such as
        `Videos\\clip-mcp\\EP1 台北.mp4`, with what sets it apart from the
        project's other videos in brackets — `EP1 台北（直式、字幕）.mp4` — and
        `(2)` is added rather than overwrite one already there; `get_job` gives the final path once it is
        done. A preview is a working file in the workspace, and only the
        newest preview of each project is kept.
        Renders and analyses run one or two at a time, so that several of them
        cannot exhaust the machine's memory between them; `stage` says what a
        job that has not started yet is waiting for, and is null when it
        started immediately. `findings` lists what the check found that this
        render went ahead in spite of. The parts of the video, where it has
        markers, are carried in the file as chapters. A preview also says how
        many clip `pictures` it `rendered` and how many it `reused`.
        `framing` names any clip told to fill the frame that had no face to
        follow, so was cropped from the middle.

    Raises:
        ValueError: If the project does not exist, contains no clips, uses an
            unsupported feature, has no finished preview of its picture as it
            is now in this shape for a final render, or its sound as it is now
            has been neither previewed nor heard, or the check found something
            not allowed.
    """
    unseen = None if is_preview else _unseen(project_id, frame)
    if unseen:
        raise ValueError(unseen)
    # Named only here: the editor's renders are the user's own, and belong to no production.
    owner = production.of_project(project_id)
    return start_render(
        project_id, is_preview, loudness_target, burn_subtitles, frame, follow_faces, allow, PREVIEW_SHORT_SIDE,
        production_id=owner.id if owner is not None else None,
    )

def _unseen(project_id: str, frame: Optional[str]) -> Optional[str]:
    """Say what of a project as it is now nobody has looked at or heard, before a final render.

    The picture has to have been previewed in this shape. The sound has to
    have been previewed in any shape, since a shape does not change it, or
    heard with `preview_sound`; so after a change to the music alone the
    sound check, which takes a fraction of a preview, is all that is asked.

    Args:
        project_id: The project.
        frame: The shape, or None for the project's own.

    Returns:
        Why the final render is refused and the quickest way past it; None
        when there is nothing unseen, or when the project does not exist,
        which rendering says in its own words.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        return None
    previews = [job for job in runtime.repo.renders_of(project_id) if job.preview]
    shaped = [job for job in previews if job.frame == frame]
    shape = f", {frame}," if frame else ""
    if not shaped:
        return (f"no preview of the cut as it is now{shape} has finished: render one with is_preview: true, look "
                "at it with view_render and inspect_join, then render the final cut")
    if not any(job.picture == picture_key(project) for job in shaped):
        return (f"the picture has changed since the last preview that finished{shape}: render a preview with "
                "is_preview: true, look at it with view_render and inspect_join, then render the final cut")
    sound = playback.mix_key(project)
    if any(job.sound == sound for job in previews) or sound in runtime.repo.heard_sounds(project_id):
        return None
    return ("only the sound has changed since the last preview that finished; the picture is the same: hear it "
            "with preview_sound, then render the final cut (a new preview with is_preview: true does as well)")

def start_render(
    project_id: str,
    is_preview: bool,
    loudness_target: Optional[float],
    burn_subtitles: bool,
    frame: Optional[str],
    follow_faces: bool,
    allow: Optional[List[str]],
    preview_side: int,
    production_id: Optional[str] = None,
) -> dict:
    """Start a render; what `render_project` does, with the preview's size open.

    The editor app previews at 720 pixels so captions and mouths can be read;
    a client previews at 480, which is quicker and plenty to judge a cut by.
    The size is not on the tool, so a client is never asked to choose it.

    Args:
        project_id: As for `render_project`, and the rest likewise.
        preview_side: The short side of a preview, in pixels.
        production_id: The unfinished production `render_project` renders
            it for; None from the editor.

    Returns:
        As `render_project`.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    # Taken before the project is reshaped or sized down, so a final render compares the cut it is asked of.
    picture, sound = picture_key(project), playback.mix_key(project)

    project = reshaped(project, frame)
    findings = check_findings(project, burn_subtitles)
    # The words on screen are laid out at the size the video is delivered at, and libass shrinks
    # them with a preview's frame: laid out again at the preview's size, pixels rounded otherwise
    # break a caption into lines elsewhere than the finished video does.
    drawn_at = project
    if is_preview:
        project = preview_sized(project, preview_side)
    blocking = [finding for finding in findings if finding.check not in (allow or [])]
    if blocking and not is_preview:
        raise ValueError(
            "the cut is not ready to render:\n- " + "\n- ".join(finding.message for finding in blocking)
            + "\nFix these, or once the user has said to go ahead anyway, render again with allow="
            + str(sorted({finding.check for finding in blocking}))
        )
    kind = ("preview" if is_preview else "output") + (f"-{frame}" if frame else "")
    if is_preview:
        _make_room()
    housekeeping.sweep_renders(runtime.WORKSPACE_DIR, _delivered)
    job = Job(kind=JobKind.RENDER, project_id=project_id, project_version=project.version, picture=picture,
              sound=sound, captioned=burn_subtitles, preview=is_preview, frame=frame, production_id=production_id)
    delivered: Optional[str] = None
    if is_preview:
        job.work_dir = os.path.join(housekeeping.preview_dir(runtime.WORKSPACE_DIR, project_id), job.job_id)
        job.output_path = os.path.join(job.work_dir, output_name(project, kind))
    else:
        # Rendered in the workspace and moved out whole once finished, so the videos
        # folder never holds half a file.
        job.work_dir = os.path.join(housekeeping.render_dir(runtime.WORKSPACE_DIR), job.job_id)
        delivered = housekeeping.unique_path(str(output_dir()), delivery_name(project, kind, ".mp4", burn_subtitles))
    render_path = os.path.join(job.work_dir, output_name(project, kind))

    subtitle_path = None
    placed: list = []
    if burn_subtitles:
        if not project.subtitles:
            raise ValueError(
                f"project {project_id} has no captions to burn; "
                "propose them with generate_subtitles and store them with a set_subtitles operation"
            )
        placed = place_cues(project, project.subtitles)
        if not placed:
            raise ValueError(
                f"project {project_id} has captions, but none of the footage they transcribe is in the cut; "
                "run generate_subtitles again against the sequence as it stands"
            )
    # Title cards' words and the words over the footage go through the same file as the
    # captions, burned or not.
    titles = cards.ass_events(drawn_at, drawn_at.caption_style) + texts.ass_events(drawn_at, drawn_at.caption_style)
    if placed or titles:
        os.makedirs(job.work_dir, exist_ok=True)
        subtitle_path = os.path.join(job.work_dir, "subtitles.ass")
        with open(subtitle_path, "w", encoding="utf-8") as handle:
            handle.write(build_ass(placed, drawn_at.width, drawn_at.height, drawn_at.caption_style, titles,
                                   texts.ass_styles(drawn_at, drawn_at.caption_style)))

    chapters_path = None
    listed, _ = chapters(project)
    if listed:
        os.makedirs(job.work_dir, exist_ok=True)
        chapters_path = os.path.join(job.work_dir, "chapters.txt")
        with open(chapters_path, "w", encoding="utf-8") as handle:
            handle.write(chapter_metadata(listed))

    assets = shared.referenced_assets(project)
    framing = framing_of(project, assets, follow_faces)
    voices = voices_of(project)
    talking = talking_of(project, voices)
    pieces: list = []
    if is_preview:
        cache = _picture_cache()
        command, pieces = runtime.renderer.build_incremental(
            project, assets, render_path, cache,
            loudness_target=loudness_target, subtitle_path=subtitle_path, framing=framing,
            chapters_path=chapters_path, voices=voices, talking=talking,
        )
        # What this preview reads from the cache counts as used, so it is kept.
        for argument in command:
            if argument.startswith(cache) and os.path.exists(argument):
                os.utime(argument)
    else:
        # The camera movement measured to steady a clip is kept with the preview pictures, so a
        # preview's measuring is not done again for the finished video.
        command, pieces = runtime.renderer.build_render(
            project, assets, render_path, loudness_target=loudness_target, subtitle_path=subtitle_path,
            framing=framing, chapters_path=chapters_path, voices=voices, talking=talking,
            transforms_dir=_picture_cache(),
        )
    # One input per clip, all opened at once, is what a render's memory use is made of.
    job.memory_estimate = resources.render_memory_bytes(command.count("-i"))
    spec = {
        "command": command, "duration_seconds": float(runtime.renderer.output_duration(project)), "pieces": pieces,
        "render_path": render_path,
    }
    if delivered:
        job.output_path = delivered
        spec["deliver"] = delivered
    else:
        spec["prune"] = os.path.dirname(job.work_dir)
    if loudness_target is not None:
        mix_path = os.path.join(job.work_dir, "mix.wav")
        spec["loudness"] = {
            "mix_command": runtime.renderer.build_mix(project, assets, mix_path, voices=voices, talking=talking),
            "mix_path": mix_path, "target": loudness_target,
        }
    commands = [command, *(piece["command"] for piece in pieces)]
    if loudness_target is not None:
        commands.append(spec["loudness"]["mix_command"])
    spec["inputs"], spec["made"] = _read_from(commands, (asset.path for asset in assets.values()),
                                              (path for piece in pieces for path in (piece["part"], piece["path"])))
    # The captions and words drawn on are read through a filter rather than `-i`, and made here all the same.
    if subtitle_path is not None:
        spec["made"] = sorted({*spec["made"], subtitle_path})
    if project.base_video_track is not None:
        sources = {asset_id: asset.path for asset_id, asset in assets.items() if asset.has_audio}
        offered, left_out = sync.sides(project.tracks, project.base_video_track.clips, sources)
        spec["sync"] = {"sides": offered, "left_out": left_out, "frame": project.fps_den / project.fps_num}
    job = runtime.job_manager.start_job(job, spec)
    # Every clip on a picture track is one picture, the sequence's and the ones drawn over it.
    pictures = sum(len(track.clips) for track in project.video_tracks)

    result = {
        "job_id": job.job_id, "status": job.status.value, "stage": job.stage, "output_path": job.output_path,
        **renamed(project),
        "findings": [{"check": finding.check, "message": finding.message} for finding in findings],
    }
    unplaced = centred_note(framing)
    if unplaced:
        result["framing"] = unplaced
    if is_preview:
        rendered = sum(1 for piece in pieces if piece.get("kind") != "transforms")
        result["pictures"] = {"rendered": rendered, "reused": pictures - rendered}
    return result

def talking_of(project: Project, voices: Mapping[str, float]) -> Optional[Talking]:
    """Work out from the transcripts where a cut has somebody talking, and how loud.

    Args:
        project: The project about to be rendered.
        voices: Its narration tracks, from `voices_of`, with the gain each is
            heard at.

    Returns:
        Where somebody is heard talking, and the gain that brings each music
        track's songs to that talking, as `levels.talking` works them out.
        None when any clip whose words are heard was never transcribed: then
        nobody knows where all the talking is, and the music ducks by level.
    """
    assets = shared.referenced_assets(project)
    return talking_in(
        project, assets, analyses_of(project), voices, lambda clip: song_level(assets[clip.asset_id].path, clip),
    )

def voices_of(project: Project) -> Dict[str, float]:
    """Find the tracks a render treats as a voice recorded apart from the picture.

    Args:
        project: The project about to be rendered.

    Returns:
        As `voice_keys`, with each voice clip's talking measured from its file.
    """
    assets = shared.referenced_assets(project)

    def loudness(clip: Clip) -> Optional[float]:
        asset = assets.get(clip.asset_id)
        if asset is None:
            return None
        start, end = float(clip.source_range.start), float(clip.source_range.end)
        return speech_loudness(asset.path, start, end - start)

    return voice_keys(project, analyses_of(project), loudness)

def analyses_of(project: Project, listened: bool = False) -> Dict[str, MediaAnalysis]:
    """Read the analysis of every file a project plays.

    Args:
        project: The project.
        listened: Whether each transcribed file has to have had its second
            listen, which is taken now for one analyzed before there was one.

    Returns:
        The analyses that exist, keyed by asset ID.
    """
    found: Dict[str, MediaAnalysis] = {}
    budget = [BACKFILL_SECONDS]
    for asset_id in sorted({clip.asset_id for track in project.tracks for clip in track.clips}):
        analysis = runtime.repo.get_analysis(asset_id)
        if analysis is not None:
            found[asset_id] = listened_to(analysis, budget) if listened else analysis
    return found

# Provisional (roadmap §13): how much footage one tool call may give its second listen while
# the caller waits — a compile or a check is a tool call, not a background job, and on a
# processor a minute of footage takes most of a minute to hear again.
BACKFILL_SECONDS = 600.0

def listened_to(analysis: MediaAnalysis, budget: List[float]) -> MediaAnalysis:
    """A file's analysis as cuts are judged by: its second listen taken, its kept hearings put in.

    Args:
        analysis: The file's analysis.
        budget: As for `_second_listened`.

    Returns:
        The analysis, with the word times of every stretch heard again and kept.
    """
    return with_words_heard_again(_second_listened(analysis, budget))

def _second_listened(analysis: MediaAnalysis, budget: List[float]) -> MediaAnalysis:
    """Give a transcribed file its second listen, if it was analyzed before there was one.

    Once per file, and kept: it takes a few seconds for a short clip and
    about the length of a long recording divided by twenty on a GPU. Only
    as much footage as is left of `budget`, so a call is never held up for
    long: a file that does not fit is judged by its transcript alone this
    time and heard on a later call, and one longer than the whole budget
    gets its second listen when it is analyzed again. Where the file has
    gone or the listen fails, the analysis is handed back without it too —
    a check that cannot run is no reason to stop an edit.

    Args:
        analysis: The file's analysis.
        budget: Seconds of footage this call may still hear again, as a
            one-item list each file's listen is taken off.

    Returns:
        The analysis, with its second listen when it could be taken.
    """
    if analysis.transcript is None or analysis.second_listen is not None:
        return analysis
    if analysis.duration > budget[0]:
        return analysis
    asset = runtime.repo.get_assets([analysis.asset_id]).get(analysis.asset_id)
    if asset is None or not os.path.exists(asset.path):
        return analysis
    budget[0] -= analysis.duration
    try:
        listened = second_listen(asset.path, analysis.transcript, lambda fraction: None, lambda: False)
    except (RuntimeError, OSError, ValueError):
        return analysis
    # Read again before writing, so an analysis taken again meanwhile is not overwritten
    # with the one this started from.
    current = runtime.repo.get_analysis(analysis.asset_id)
    if current is None or current.analyzed_at != analysis.analyzed_at:
        return analysis.model_copy(update={"second_listen": listened})
    current.second_listen = listened
    runtime.repo.save_analysis(current)
    return current

def _plan_target(project: Project) -> Optional[float]:
    """Find how long the plan a project was compiled from asked it to run.

    Args:
        project: The project.

    Returns:
        The plan's target length, or None when the sequence did not all come
        from one plan or that plan named no length.
    """
    plan = compiled_plan(project)
    return plan.target.seconds if plan is not None else None

def compiled_plan(project: Project) -> Optional[EditPlan]:
    """The plan a project's whole sequence was compiled from; None when it did not all come from one."""
    base = project.base_video_track
    compiled_from = {clip.from_plan_id for clip in base.clips} if base else set()
    if len(compiled_from) != 1 or None in compiled_from:
        return None
    return runtime.repo.get_plan(next(iter(compiled_from)))

def check_findings(project: Project, burn_subtitles: bool) -> list:
    """Check a cut before it is rendered.

    Args:
        project: The project, at the size it will be rendered at.
        burn_subtitles: Whether its captions will be burned in.

    Returns:
        What `delivery.check_delivery` finds.
    """
    placed = place_cues(project, project.subtitles) if burn_subtitles and project.subtitles else None
    assets = shared.referenced_assets(project)
    dated = shared.dated(assets)
    analyses = analyses_of(project, listened=True)
    return check_delivery(
        project, analyses, _plan_target(project), placed, project.caption_style,
        talking_of(project, voices_of(project)), {asset_id: asset.recorded_at for asset_id, asset in dated.items()},
        _looks(look_candidates(project, analyses), assets),
    )

def _looks(clips: Sequence[Clip], assets: Mapping[str, Asset]) -> Dict[str, Tuple[int, ...]]:
    """Fingerprint clips at `looks.SAMPLES`, reading only the frames not fingerprinted before.

    Args:
        clips: The clips.
        assets: Their files, keyed by asset ID.

    Returns:
        The fingerprints by clip ID, for the clips whose file has a picture to read.
    """
    found: Dict[str, Tuple[int, ...]] = {}
    for asset_id in {clip.asset_id for clip in clips}:
        asset = assets.get(asset_id)
        if asset is None or not asset.has_video or not os.path.exists(asset.path):
            continue
        kept = runtime.repo.looks(asset_id)
        new: Dict[float, int] = {}
        for clip in (clip for clip in clips if clip.asset_id == asset_id):
            start, length = float(clip.source_range.start), float(clip.source_range.end - clip.source_range.start)
            moments = [round(start + length * share, 1) for share in looks.SAMPLES]
            try:
                for at in moments:
                    if at not in kept and at not in new:
                        new[at] = looks.look_of(asset.path, at)
            except RuntimeError:
                continue
            found[clip.id] = tuple(kept.get(at, new.get(at)) for at in moments)
        if new:
            runtime.repo.save_looks(asset_id, new)
    return found

def contrast_notes(project: Project) -> List[str]:
    """Measure the words of a cut, in their colours, against the picture behind them, and say where they are hard to read.

    A title card is measured on its background as the render treats it — the
    frame at its `at`, blurred, darkened or replaced by its colour — and words
    over the footage on the frame in the middle of the stretch they show for,
    through the name bar's box where they have one. Words on a picture laid
    over the sequence are left out: what is behind them moves about the frame.

    Args:
        project: The cut, at the size and shape it will be rendered at.

    Returns:
        One line for each card or text below `legibility.LOW_CONTRAST`.
    """
    base = project.base_video_track
    clips = sorted(base.clips, key=lambda clip: clip.timeline_in) if base else []
    if not any(clip.card is not None or clip.shown_texts for clip in clips):
        return []
    assets = shared.referenced_assets(project)
    framing = framing_of(project, assets)
    fps = Fraction(project.fps_num, project.fps_den)
    width, height = project.width, project.height

    def frame_at(clip: Clip, seconds: float):
        asset = assets.get(clip.asset_id)
        if asset is None or not asset.has_video or not os.path.exists(asset.path):
            return None
        try:
            return still(asset.path, seconds, width, height, placement_at(framing.get(clip.id), clip, seconds, fps))
        except RuntimeError:
            return None

    notes = []
    floor = f"{legibility.LOW_CONTRAST:g}:1"
    for clip in clips:
        when = format_timestamp(float(clip.timeline_in))
        if clip.card is not None:
            picture = frame_at(clip, float(clip.card.at))
            if picture is None:
                continue
            ratio = legibility.words_contrast(cards.backdrop_image(picture, clip.card),
                                              cards.placed(clip, width, height)["rows"])
            if ratio < legibility.LOW_CONTRAST:
                notes.append(
                    f"title card {clip.id} 「{clip.card.title}」 at {when}: its words stand {ratio:.1f}:1 against "
                    f"the {clip.card.background} background behind them (from {float(clip.card.at):g}s of its file), "
                    f"under the {floor} that large words need to read. Show another moment with `at` in "
                    "set_title_card, or put them on a dark `colour`"
                )
            continue
        for text in clip.shown_texts:
            shown = texts.shown(clip, text)
            if shown is None:
                continue
            middle = (max(text.start, clip.source_range.start) + min(text.end, clip.source_range.end)) / 2
            picture = frame_at(clip, float(middle))
            if picture is None:
                continue
            rows = texts.placed(text, width, height, project.caption_style)
            ratio = legibility.words_contrast(picture, rows, texts.BOX_OPACITY)
            if ratio < legibility.LOW_CONTRAST:
                notes.append(
                    f"words {text.id} 「{text.text}」 on clip {clip.id} at {format_timestamp(shown[0])}: they "
                    f"stand {ratio:.1f}:1 against the picture behind them at {float(middle):g}s of its file, under the "
                    f"{floor} that large words need to read. Move them with `x` and `y` in set_text to a darker "
                    "part of the frame, or make them a `name`, which has a dark box behind it"
                )
    return notes

@mcp.tool()
def check_render(
    project_id: str,
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    burn_subtitles: bool = True,
) -> dict:
    """Check a cut for what would be noticed in the finished file, without rendering it.

    The same check `render_project` runs first and refuses a render over. It
    looks for eleven things: `bad_picture`, black or frozen source picture that
    reaches the screen, as far as the analysis measures it: black of half a
    second or more, frozen of two seconds or more, while anything shorter
    shows only in a render's `frames` from `get_job`; `covered_card`,
    full-frame picture on a track above the sequence, such as B-roll, laid
    over a title card so its words are hidden, most likely after a card was
    added by hand, which moves the sequence but not the tracks above; `clipping`, a recording squared off at the ceiling,
    which no amount of turning down undoes; `music`, music heard too close
    under somebody talking; `mid_speech`, a cut that lands
    inside a word or between two words of one phrase, or takes the start or
    the end off a sound the microphone measured, with where to move it — a
    cut between two sentences is not one, even with no pause between them,
    where the transcript ends the first on 。！？ or a closing particle such
    as 吧 or 啊; `repeated`, the same stretch of a file shown twice;
    `look_alike`, two clips from different files that look so alike, with
    nothing said or written over them, that a viewer could not tell which is
    which; `continuity`, a clip put on by hand between two shot at another
    moment; `unplanned`, a sequence of three or more clips put together by hand
    rather than compiled from a plan, so nothing says how it opens, turns
    and ends; `captions`, a caption too tall for
    the frame — most likely when a landscape cut is rendered portrait — or on
    screen too briefly to read, with no room left in its shot to hold it; and
    `length`, a cut far from the length its plan asked for. Each is a fact
    about the cut, not a verdict on it: black may be meant, and the user may
    prefer the longer cut. Tell them, and let them decide.

    Args:
        project_id: ID of the project to check.
        frame: Check it in another shape, as `render_project` would render it.
        burn_subtitles: Check any captions as they would be burned in. False
            only for a render that leaves them out: a check that skips them
            passes what the render then refuses (#191).

    Returns:
        A dictionary with `ok` and `findings`, each with its `check` and a
        `message` saying what and where; and `light`, where the picture jumps
        in brightness or colour from one shot to the next within a part, with
        the adjustment that would even it out. `light` never stops a render:
        a jump may be the point. Offer the adjustment, and check again once
        it is set. And `run_on`, the cuts let through for ending a sentence
        that the next follows with no pause: nothing is cut off as the
        transcript has it, but only listening to the render tells whether the
        next sentence's first sound carries across. It never stops a render
        either. And `shaky`, the clips the camera shook in that are left as
        shot, with how to steady them; a shake may be wanted, so it never
        stops a render — ask before steadying, since it enlarges the picture.
        And `contrast`, the title cards and words over the footage
        whose white letters stand out too little from the picture behind
        them, measured on that picture as the render shows it, with what to
        change. It never stops a render: the words still have their shadow.

    Raises:
        ValueError: If the project does not exist.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if project.base_video_track is None or not project.base_video_track.clips:
        # Nothing checked is not the same as nothing wrong (#96).
        raise ValueError(f"project {project_id} has nothing on its sequence to check yet; compile a plan or "
                         "add clips first")
    findings = check_findings(reshaped(project, frame), burn_subtitles)
    checked = {
        "ok": not findings,
        "findings": [{"check": finding.check, "message": finding.message} for finding in findings],
    }
    light = light_steps(project, analyses_of(project))
    if light:
        checked["light"] = light
    shaky = shaky_clips(project, analyses_of(project))
    if shaky:
        checked["shaky"] = shaky
    base = project.base_video_track
    run_on = run_on_cuts(base.clips, analyses_of(project, listened=True))
    if run_on:
        checked["run_on"] = run_on
    contrast = contrast_notes(reshaped(project, frame))
    if contrast:
        checked["contrast"] = contrast
    return checked

# Whether edits start making what the browser needs to play them. Off where nobody will
# watch, such as a test run.
PREPARE_PLAYBACK = True
# What a preparation job is expected to hold in memory: one FFmpeg encoding a small picture.
# Provisional: an estimate, not measured.
PREPARE_MEMORY_BYTES = 300 * 1024 * 1024
# Giving cache room back looks at the disk at most this often, in seconds. Provisional.
ROOM_CHECK_SECONDS = 60

def _being_prepared() -> set:
    """The files preparation jobs still waiting or running will write, so none is asked for twice."""
    paths = set()
    for job in runtime.repo.jobs_to_show(datetime.now(timezone.utc))[0]:
        if job.kind != JobKind.PREPARE or not job.work_dir:
            continue
        try:
            with open(os.path.join(job.work_dir, "job.json"), encoding="utf-8") as spec:
                paths.update(step["path"] for step in json.load(spec).get("steps", []))
        except (OSError, ValueError):
            continue
    return paths

def _frame_rate(asset: Asset) -> float:
    """A file's frame rate as its first picture stream says, or 0 when it does not."""
    try:
        streams = probe_file(asset.path).get("streams", [])
    except (OSError, RuntimeError):
        return 0.0
    for stream in streams:
        if stream.get("codec_type") == "video":
            try:
                return float(Fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0"))
            except (ValueError, ZeroDivisionError):
                return 0.0
    return 0.0

def _read_from(commands: Sequence[List[str]], library: Iterable[str],
               written: Iterable[str] = ()) -> Tuple[List[str], List[str]]:
    """The files a job's FFmpeg commands open, so a failure while one cannot be read is put down to it.

    The user's footage and what the server made for the job are told apart:
    footage that is gone stays gone until the user puts it back, while a
    picture from the cache or the chapters, removed after the job was queued,
    is made anew by starting the job again. The words drawn on are read
    through a filter rather than `-i`, so the render adds them itself.

    Args:
        commands: The commands, each starting with the executable.
        library: The paths of the library's files.
        written: What the job's own steps write, which a later step opens
            and is not there to be read before the job runs.

    Returns:
        `(inputs, made)`: the library's files some command opens with `-i`,
        and the other files it opens, less what the job writes.
    """
    opened = {command[index + 1] for command in commands
              for index in range(len(command) - 1) if command[index] == "-i"} - set(written)
    footage = opened & set(library)
    return sorted(footage), sorted(opened - footage)

def _start_preparing(steps: List[dict], urgent: bool, read: Tuple[List[str], List[str]],
                     asset_id: Optional[str] = None, project_id: Optional[str] = None) -> Optional[str]:
    """Queue one preparation job over its steps, which read the footage and made files in `read`; its ID, or None."""
    if not steps:
        return None
    job = Job(kind=JobKind.PREPARE, asset_id=asset_id, project_id=project_id, priority=-1 if urgent else 1,
              memory_estimate=PREPARE_MEMORY_BYTES)
    job.work_dir = os.path.join(runtime.WORKSPACE_DIR, "jobs", job.job_id)
    return runtime.job_manager.start_job(job, {"steps": steps, "inputs": read[0], "made": read[1]}).job_id

def prepare_playback(project: Project, urgent: bool = False) -> List[str]:
    """Start making what the browser still needs to play a cut: proxies, repaired sound, its loudness.

    Each is made once and kept by what went into it, so this costs nothing
    for what is already there or already being made.

    Args:
        project: The cut.
        urgent: Somebody is waiting to watch it, so it goes ahead of the
            other work waiting; otherwise it waits behind it.

    Returns:
        The IDs of the jobs started.
    """
    if not PREPARE_PLAYBACK:
        return []
    _make_room()
    assets = shared.referenced_assets(project)
    busy = _being_prepared()
    proxies = playback.proxy_dir(runtime.WORKSPACE_DIR)
    started = []
    for asset_id in playback.missing(project, assets, os.listdir(proxies)):
        asset = assets[asset_id]
        path = os.path.join(proxies, playback.proxy_name(asset))
        if path in busy:
            continue
        step = {"command": playback.proxy_command(asset, path + ".part", _frame_rate(asset)), "part": path + ".part",
                "path": path, "seconds": float(asset.duration or 1), "stage": "making a copy to play from"}
        started.append(_start_preparing([step], urgent, ([asset.path], []), asset_id=asset_id))
    sound = playback.sound_dir(runtime.WORKSPACE_DIR)
    steps = []
    for track in project.tracks:
        for clip in track.clips:
            asset = assets.get(clip.asset_id)
            name = playback.repair_name(clip, asset) if asset is not None and playback.needs_repair(clip) else None
            path = os.path.join(sound, name) if name else None
            if path and not os.path.exists(path) and path not in busy:
                steps.append({"command": playback.repair_command(clip, asset, path + ".part"), "part": path + ".part",
                              "path": path, "seconds": float(clip.source_range.end - clip.source_range.start),
                              "stage": "repairing a voice"})
    gain = os.path.join(sound, playback.mix_key(project) + ".json")
    if project.tracks and float(project.duration) > 0 and not os.path.exists(gain) and gain not in busy:
        voices = voices_of(project)
        mix = gain + ".wav"
        steps.append({"command": runtime.renderer.build_mix(project, assets, mix, voices, talking_of(project, voices)),
                      "part": mix, "path": gain, "seconds": float(project.duration), "measure": True,
                      "target": playback.LOUDNESS_TARGET, "stage": "measuring loudness"})
    written = [path for step in steps for path in (step["part"], step["path"])]
    read = _read_from([step["command"] for step in steps], (asset.path for asset in assets.values()), written)
    started.append(_start_preparing(steps, urgent, read, project_id=project.id))
    return [job_id for job_id in started if job_id]

def playback_of(project: Project, urgent: bool = True) -> dict:
    """Describe a cut for the browser to play, and start making whatever it still lacks.

    Args:
        project: The cut, as it is now or as it was at a version.
        urgent: Whether somebody is waiting to watch it.

    Returns:
        What `playback.describe` returns.
    """
    assets = shared.referenced_assets(project)
    proxies = playback.proxy_dir(runtime.WORKSPACE_DIR)
    sound = playback.sound_dir(runtime.WORKSPACE_DIR)
    have = {}
    for asset_id, asset in assets.items():
        name = playback.proxy_name(asset)
        if name and os.path.exists(os.path.join(proxies, name)):
            have[asset_id] = f"/proxy/{name}"
    repaired = {}
    for track in project.tracks:
        for clip in track.clips:
            asset = assets.get(clip.asset_id)
            if asset is None or not playback.needs_repair(clip):
                continue
            name = playback.repair_name(clip, asset)
            if name and os.path.exists(os.path.join(sound, name)):
                repaired[clip.id] = f"/sound/{name}"
    gain = None
    measured = os.path.join(sound, playback.mix_key(project) + ".json")
    if os.path.exists(measured):
        with open(measured, encoding="utf-8") as kept:
            gain = json.load(kept).get("gain_db")
    voices = voices_of(project)
    described = playback.describe(
        project, assets, playback.Prepared(have, repaired, gain), talking_of(project, voices), voices,
        framing_of(project, assets),
    )
    if described["waiting"] or gain is None:
        prepare_playback(project, urgent=urgent)
    return described

_room_checked = 0.0

def _make_room() -> None:
    """Give cache room back if the disk is running low, at most once a minute."""
    global _room_checked
    if time.monotonic() - _room_checked < ROOM_CHECK_SECONDS:
        return
    _room_checked = time.monotonic()
    housekeeping.give_cache_back(runtime.WORKSPACE_DIR, runtime.repo.active_job_ids())
