"""Projects and their timelines: creating, listing and reading them, and applying edits to them."""

import uuid
from decimal import Decimal
from typing import Annotated, Callable, List, Optional, Sequence

from pydantic import Field

from app.models.timeline.apply import apply_operation
from app.models.timeline.operations import (
    AddSubtitleOp, EditOperation, EditSubtitleOp, FitTrackOp, SetCaptionStyleOp, SetSubtitlesOp,
)
from app.models.timeline.parts import Project, SubtitleCue, TrackType
from app.models.timeline.validate import validate_project
from app.engine.semantic import clean_cuts
from app.engine.frames import format_timestamp
from app.engine.subtitles import place_cues
from app import runtime
from app.runtime import mcp
from app.tools import rendering, shared

@mcp.tool()
def create_project(
    name: Annotated[str, Field(min_length=1, max_length=120)],
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
        name: What to call the project, such as 'EP1 台北' — required. It is
            what the editor app and `list_projects` show, and what the
            finished video is called, so name it for what the video is about,
            not 'test' or 'project'. If the user has not said, propose one in
            your read-back. Change it later with a `rename_project` operation.
        width: Output width in pixels; must be even.
        height: Output height in pixels; must be even.
        fps_num: Frame rate numerator, e.g. 30000 for 29.97 fps.
        fps_den: Frame rate denominator, e.g. 1001 for 29.97 fps.

    Returns:
        The new project serialized as a dictionary, including its generated
        `id` and initial `version`, and `preferences`: how the user likes
        their videos, as they said it on earlier ones — cut it that way.
    """
    if not name.strip():
        raise ValueError("a project needs a name: what the video is about, such as 'EP1 台北'")
    project = Project(
        id=str(uuid.uuid4()), name=name.strip(),
        width=width, height=height, fps_num=fps_num, fps_den=fps_den,
    )
    runtime.repo.add_project(project)
    return with_preferences(shared.plain(project.model_dump()))

def with_preferences(result: dict) -> dict:
    """Add how the user likes their videos to what a tool says, when anything is kept."""
    kept = runtime.repo.preferences()
    if kept:
        result["preferences"] = [{"id": preference.id, "text": preference.text} for preference in kept]
    return result

@mcp.tool()
def list_projects() -> dict:
    """List all projects.

    Returns:
        A dictionary with a `projects` list, each item holding a project's
        `id`, its `name` if it was given one, and its current `version`; and
        `preferences`, how the user likes their videos, when any are kept.
    """
    return with_preferences({"projects": [
        {"id": project.id, "name": project.name, "version": project.version}
        for project in runtime.repo.list_projects()
    ]})

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
        length of the edited video in seconds, and `subtitle_count`; and
        `preferences`, how the user likes their videos, when any are kept.

    Raises:
        ValueError: If no project with `project_id` exists.
    """
    project = runtime.repo.get_project(project_id)
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
    return with_preferences(shared.plain(state))

@mcp.tool()
def apply_edits(project_id: str, expected_version: int, operations: list[EditOperation], note: str = "") -> dict:
    """Apply a batch of edit operations to a project with optimistic locking.

    Operations are applied in order and atomically: if any operation fails,
    or the resulting timeline is invalid, the project is left unchanged. On
    success the project version is incremented by one. Call `get_project`
    first to obtain the current version.

    Editing behaves like a magnetic timeline: `insert_clip` shifts later
    clips on the same track to make room, and `trim_clip` and `delete_clip`
    shift them to close or open space unless `ripple` is false.
    `reorder_clip` moves a clip elsewhere in the sequence, keeping its cut
    and audio settings. `split_clip` cuts one clip into two halves that
    together occupy exactly what the original did, so nothing else moves.
    `add_clip` and `move_clip` use absolute positions and never move other
    clips. Clips on other tracks never move.

    The first video track is the base and defines the output length: gaps
    are rendered as black frames with silence, and audio past its last clip
    is cut. Further video tracks are drawn on top of the base, each clip
    inside the box its `layout` gives. Background music goes on an audio
    track. Each operation's own description says what it changes: a clip's
    sound running outside its picture (J and L cuts), its transition, colour
    and speed, a track's ducking, the structure markers, the captions and how
    they are drawn.

    The resulting timeline must satisfy these rules:
    - Clips stay within their asset's duration and do not overlap on a track.
      Sound is the exception: a lead or a lag overlaps the neighbouring
      clip's sound, which is what a J or an L cut is, but still comes from
      inside the file.
    - Audio and video fades fit within their clip.
    - Only clips above the base video track take a `layout`.
    - Only audio tracks duck under speech.
    - Video-track clips use assets with video; audio-track clips use assets
      with audio.

    Args:
        project_id: ID of the project to edit.
        expected_version: Project version the caller last read. The request is
            rejected if the project has been modified since then.
        note: The user's own words for this change — 「結尾的晚安多留一下」 —
            kept as what this version is in the project's history.
        operations: Edit operations to apply, each identified by its `action`:
            `add_track`, `add_clip`, `insert_clip`, `trim_clip`, `move_clip`,
            `delete_clip`, `split_clip`, `reorder_clip`, `fit_track`,
            `set_clip_audio`, `set_clip_look`, `set_track_audio`,
            `add_title_card`, `set_title_card`, `add_text`, `set_text`,
            `remove_text`,
            `set_subtitles`, `edit_subtitle`, or `add_subtitle`. Captions stored
            or corrected here are also remembered against the file they belong
            to, and the next `generate_subtitles` over that file, in any
            project, starts from them — but only those given
            `for_all_projects`; the rest stay in this project. Caption times can be given in the cut's own
            seconds with `timeline_start` and `timeline_end`.

    Returns:
        A dictionary with the `status` and the project's `new_version`, and,
        when captions were stored or corrected, `captions`: each one touched,
        as it now reads, one line apiece, marked `(this project only)` where it
        is — read them back rather than assume — and `captions_kept`, which says
        where the rest are now remembered; and, when the caption style was
        set, `caption_style`: the whole of it as it now stands.

    Raises:
        ValueError: If the project does not exist, the version does not match,
            an operation references a missing or duplicate track or clip, or
            the resulting timeline breaks a rule above.
    """
    before = runtime.repo.get_project(project_id)
    saved = apply(project_id, expected_version, operations)
    rendering.prepare_playback(saved)
    result = {"status": "success", "new_version": saved.version}
    gap = music_ends_early(saved)
    if gap:
        result["music"] = gap
    if any(isinstance(op, SetCaptionStyleOp) for op in operations):
        result["caption_style"] = saved.caption_style.model_dump(mode="json")
    touched = _touched_captions(before, saved, operations)
    if touched:
        shared_cues = [cue for cue in touched if not cue.local]
        if shared_cues:
            runtime.repo.remember_captions([cue.model_copy(update={"remembered_in": saved.id}) for cue in shared_cues])
        result["captions"] = [_caption_line(saved, cue) for cue in touched]
        if not shared_cues:
            result["captions_kept"] = "in this project only"
        elif len(shared_cues) < len(touched):
            result["captions_kept"] = ("remembered against their footage, so every project that captions it starts "
                                       "from them — except the ones marked (this project only)")
        else:
            result["captions_kept"] = "remembered against their footage: every project that captions it starts from them"
    return result

# Music stopping this much before the video does leaves an ending without it. Provisional.
MUSIC_GAP_SECONDS = 3.0

def music_ends_early(project: Project) -> Optional[str]:
    """Say when the music laid by hand stops well before the video ends, with what would fix it."""
    music = [track for track in project.tracks if track.track_type == TrackType.AUDIO and track.duck_under_speech]
    clips = [clip for track in music for clip in track.clips]
    if not clips:
        return None
    ends = max(float(clip.timeline_out) for clip in clips)
    left = float(project.duration) - ends
    if left <= MUSIC_GAP_SECONDS:
        return None
    return (f"the music stops at {ends:.1f}s and the last {left:.0f}s have none: fit_track with loop to repeat it, "
            f"a longer song (find_music with min_seconds), or say that is wanted")

def _touched_captions(before: Optional[Project], project: Project, operations: Sequence) -> List[SubtitleCue]:
    """Find the captions a batch of edits stored or corrected, as they now stand.

    By comparing the project's captions before and after rather than reading
    the operations: a caption added at a time in the cut does not say which
    file or second it lands on until it has been applied.

    Args:
        before: The project before the edits.
        project: The project after the edits.
        operations: The edits.

    Returns:
        Every caption set, added or corrected, in source order; a deleted one
        is left out.
    """
    if before is None or any(isinstance(op, SetSubtitlesOp) for op in operations):
        return list(project.subtitles)
    if not any(isinstance(op, (EditSubtitleOp, AddSubtitleOp)) for op in operations):
        return []
    was = {cue.id: cue for cue in before.subtitles}
    return [cue for cue in project.subtitles if was.get(cue.id) != cue]

def _caption_line(project: Project, cue: SubtitleCue) -> str:
    """Write one caption as a line to read back: where it lands and what it says.

    Args:
        project: The project it is on.
        cue: The caption.

    Returns:
        Its ID, its time in the cut (or that it is not in the cut), and its text.
    """
    placed = [item for item in place_cues(project, [cue])]
    at = f"{format_timestamp(float(placed[0].start))}" if placed else "not in the cut"
    return (f"{cue.id} {at} {cue.text}" + (f" / {cue.secondary}" if cue.secondary else "")
            + (" (this project only)" if cue.local else ""))

def apply(
    project_id: str,
    expected_version: int,
    operations: list,
    stamp: Optional[Callable[[Project], None]] = None,
) -> Project:
    """Apply a batch of operations to a project, atomically.

    Args:
        project_id: Project to edit.
        expected_version: Version the caller last read.
        operations: Operations to apply, in order.
        stamp: Called with the edited project before it is checked, for
            whatever the operations themselves cannot carry. `compile_plan`
            uses it to record where each clip came from, which keeps
            provenance off the tool surface and out of a caller's reach.

    Returns:
        The saved project, at its new version.

    Raises:
        ValueError: If the project does not exist, the version does not match,
            or the result breaks a timeline rule. Nothing is saved in that
            case.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if project.version != expected_version:
        raise ValueError(f"version conflict: current version is {project.version}, expected version is {expected_version}")

    # A caption names the file its words were spoken in. One naming a file nobody
    # imported would store cleanly and then never appear, which reads as a captioning
    # bug rather than a typo, so it is refused here where the typo still has a name.
    named = {cue.asset_id for op in operations if isinstance(op, SetSubtitlesOp) for cue in op.cues}
    unknown = sorted(named - set(runtime.repo.get_assets(named)))
    if unknown:
        raise ValueError(
            f"caption(s) name assets that are not imported: {', '.join(unknown)}; "
            "import the footage the words were spoken in first"
        )

    # Assets are looked up first so operations that need source lengths, such as fit_track, can use them.
    # Where each song on a track being fitted can loop on the beat: the timeline cannot read analyses.
    for op in operations:
        if isinstance(op, FitTrackOp):
            track = next((item for item in project.tracks if item.id == op.track_id), None)
            for asset_id in {clip.asset_id for clip in (track.clips if track else [])}:
                analysis = runtime.repo.get_analysis(asset_id)
                span = clean_cuts(analysis).loop if analysis is not None else None
                if span is not None:
                    op.loops[asset_id] = (Decimal(str(span[0])), Decimal(str(span[1])))
    # A photo a title card is to stand on is not in the cut until the card puts it there.
    known = runtime.repo.get_assets({clip.asset_id for track in project.tracks for clip in track.clips}
                                    | {op.photo_asset_id for op in operations if getattr(op, "photo_asset_id", None)})
    for op in operations:
        apply_operation(project, op, known)
    if stamp is not None:
        stamp(project)
    assets = shared.referenced_assets(project)
    validate_project(project, assets)
    runtime.renderer.check_supported(project, assets)

    project.version += 1
    runtime.repo.update_project(project, expected_version)
    return project
