"""Handing a finished cut over: the cut read back as text, chapters, covers, and timelines for other editors."""

import os
from fractions import Fraction
from typing import Annotated, Dict, List, Literal, Mapping, Optional

from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field

from app.models.media import Asset
from app.models.timeline import Clip
from app.engine import cards
from app.engine.probe import probe_file, timecode_start
from app.engine.reframe import placement_at
from app.engine.builder import voice_keys
from app.engine.delivery import chapters, clock, cover_candidates, frames_around, marked_parts, on_screen
from app.engine.frames import feed_preview, format_timestamp, still, storyboard_sheet
from app.engine.interchange import write_edl, write_fcpxml, write_otio, write_srt
from app.engine.subtitles import place_cues
from app.storage import housekeeping
from app.workspace import output_dir
from app import runtime
from app.runtime import mcp
from app.tools import rendering, shared, subtitles, versioning

@mcp.tool()
def read_cut(project_id: str) -> dict:
    """Read a cut back as a script: every shot in the order it plays, as a viewer meets it.

    The checks find what is wrong with a cut mechanically — a black frame, a
    cut inside a word. Whether the story holds is read, not measured: a line
    that says why somebody did something, landing after what they did, passes
    every check. Read this through before a final render and tell the user the
    story in a few sentences, as it plays, so they can say where it goes wrong.

    Args:
        project_id: The project.

    Returns:
        `script`: one line per shot of the sequence — when it plays, the part
        of the video it is in, the clip and what is on screen (the footage's
        description, or its file), what is said over it with the captions' glossary
        words corrected, the captions shown
        and the words on screen; what plays over it on a track above is under
        it, indented.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    base = project.base_video_track
    if base is None or not base.clips:
        raise ValueError(f"project {project_id} has nothing on its sequence to read")
    assets = shared.referenced_assets(project)
    parts = marked_parts(project)
    placed = place_cues(project, project.subtitles) if project.subtitles else []

    def seen(clip: Clip) -> str:
        if clip.card is not None:
            return f"card 「{clip.card.title}」" + (f" 「{clip.card.subtitle}」" if clip.card.subtitle else "")
        described = [runtime.repo.get_semantic_clip(clip_id) for clip_id in clip.from_clip_ids]
        words = "; ".join(found.description for found in described if found is not None and found.description)
        asset = assets.get(clip.asset_id)
        name = os.path.basename(asset.path) if asset is not None else clip.asset_id
        return f"{name} {float(clip.source_range.start):.1f}–{float(clip.source_range.end):.1f}s" + (
            f": {words}" if words else "")

    def line(clip: Clip) -> str:
        start, end = float(clip.timeline_in), float(clip.timeline_out)
        parts = [f"{clock(start)}–{clock(end)}", f"clip {clip.id}", seen(clip)]
        said = "" if clip.card is not None or clip.volume == 0 else versioning.said_between(
            clip.asset_id, float(clip.source_range.start), float(clip.source_range.end))
        if said:
            parts.append(f"said: 「{said}」")
        # More than a sliver: a caption starting where this shot ends is the next shot's (#80).
        shown = [cue.text for cue in placed if min(float(cue.end), end) - max(float(cue.start), start) > subtitles.CAPTION_SLIVER]
        if shown:
            parts.append("captions: " + " / ".join(dict.fromkeys(shown)))
        if clip.shown_texts:
            parts.append("on screen: " + " / ".join(text.text for text in clip.shown_texts))
        return " | ".join(parts)

    script: List[str] = []
    part = None
    above = [clip for track in project.video_tracks[1:] for clip in track.clips]
    for clip in sorted(base.clips, key=lambda clip: clip.timeline_in):
        # By its middle, as a cut's parts are read everywhere else: a shot coming in on a
        # transition can start a hair before its part's marker and is still that part's (#80).
        middle = float(clip.timeline_in) + float(clip.timeline_duration) / 2
        now = next((marker.name for opens, _, marker in reversed(parts) if opens <= middle), None)
        if now != part and now is not None:
            script.append(f"== {now}")
            part = now
        script.append(line(clip))
        for over in sorted((other for other in above
                            if other.timeline_in < clip.timeline_out and other.timeline_out > clip.timeline_in
                            and clip.timeline_in <= other.timeline_in),
                           key=lambda other: other.timeline_in):
            script.append("    over it: " + line(over))
    return {"project_id": project_id, "version": project.version, "script": script}

@mcp.tool()
def get_chapters(project_id: str) -> dict:
    """Read the parts of a cut back as chapters, ready for a video description.

    The parts are the project's markers — compiling a plan writes one per
    beat — so a chapter is a part and runs to the next. Rendering carries the
    same chapters inside the file. YouTube only shows chapters that start at
    0:00, number at least three and run at least ten seconds each; anything
    that breaks those is reported rather than changed, because merging or
    renaming parts changes the video.

    Args:
        project_id: ID of the project.

    Returns:
        A dictionary with `chapters` (`start`, `end`, `name`), the
        `description` text to paste — one `m:ss name` line each — and the
        `problems` that would stop YouTube showing them.

    Raises:
        ValueError: If the project does not exist.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    listed, problems = chapters(project)
    return {
        "chapters": [{"start": round(start, 3), "end": round(end, 3), "name": name} for start, end, name in listed],
        "description": "\n".join(f"{clock(start)} {name}" for start, _, name in listed),
        "problems": problems,
    }

@mcp.tool()
def propose_covers(
    project_id: str,
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    follow_faces: bool = True,
    per_shot: Annotated[int, Field(ge=1, le=4)] = 1,
    around: Optional[Annotated[float, Field(ge=0)]] = None,
    part: Optional[str] = None,
) -> ToolResult:
    """Lay out frames of the cut for you to choose a cover or thumbnail from.

    Choosing is yours, not this tool's: the cover has to show what the video
    is about, and nothing measured here knows that. So the listing opens with
    what the plan says the video is for and its parts, and every frame comes
    with what is being said over it. Pick the frame that shows the video's
    point — the thing it is about, the result it leads to — and that reads at
    a glance at the size a feed shows it: one subject, large in the frame,
    sharp, uncluttered, a face with its eyes open if there is one. Then look
    closer: call again with `around` set to that moment to see the frames
    either side of it, and choose the exact one. Offer the user your best one
    or two with the reason for each, and save the one they choose with
    `export_cover`.

    Args:
        project_id: ID of the project.
        frame: Crop the candidates to another shape, as it would be rendered.
        follow_faces: Crop around the face, as the render does.
        per_shot: How many moments to offer from each shot: the one its
            measurements favour and others spread through it.
        around: A moment of the cut, in seconds, to look at closely instead:
            every quarter second for two seconds either side of it.
        part: Only the frames of one part of the cut, by its name or the ID
            of the beat it came from — the payoff, where the point of the
            video usually is, or any part a long cut's sheet had no room for.

    Returns:
        What the video is for, a listing — each candidate's number, time in
        the cut, clip and what is said then — and the candidates as one
        labeled image.

    Raises:
        ValueError: If the project or the part does not exist, or there is
            nothing to offer.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    project = rendering.reshaped(project, frame)
    analyses = rendering.analyses_of(project)
    if around is None:
        offered = [
            (clip, seconds, float(clip.timeline_in) + (seconds - float(clip.source_range.start)) / clip.speed)
            for clip, seconds in cover_candidates(project, analyses, per_shot)
        ]
    else:
        offered = frames_around(project, analyses, around)
    if part is not None:
        parts = marked_parts(project)
        span = next(((opens, closes) for opens, closes, marker in parts
                     if part in (marker.name, marker.id, marker.from_beat_id)), None)
        if span is None:
            raise ValueError(f"no part called {part!r}; the parts are: "
                             f"{', '.join(marker.name for _, _, marker in parts) or 'none'}")
        opens, closes = span
        offered = [item for item in offered if opens <= item[2] < closes]
    if not offered:
        raise ValueError(f"project {project_id} has no picture to take a cover from")
    left_out = max(0, len(offered) - shared.MAX_STORYBOARD_TILES)
    offered = offered[:shared.MAX_STORYBOARD_TILES]
    assets = shared.referenced_assets(project)
    framing = rendering.framing_of(project, assets, follow_faces)
    fps = Fraction(project.fps_num, project.fps_den)
    said = place_cues(project, project.subtitles) if project.subtitles else []
    shots, listing = [], []
    for number, (clip, seconds, at) in enumerate(offered, start=1):
        shots.append((assets[clip.asset_id].path, seconds, f"#{number}  {format_timestamp(at)}",
                      placement_at(framing.get(clip.id), clip, seconds, fps)))
        words = next((cue.text for cue in said if float(cue.start) <= at < float(cue.end)), "")
        listing.append(f"#{number}: edit {at:.3f}s ({format_timestamp(at)}) | clip {clip.id}"
                       + (f" | 「{words}」" if words else ""))
    header = [f"{len(offered)} cover candidate(s) for project {project_id}, {project.width}x{project.height}"
              + (f", around {around:g}s" if around is not None else "") + ":"]
    plan = rendering.compiled_plan(project)
    if plan is not None and (plan.goal or plan.beats):
        parts = " → ".join(beat.name or beat.id for beat in plan.beats)
        header.insert(0, "What the video is for: " + (plan.goal or "(no goal written)")
                      + (f"\nIts parts: {parts}" if parts else ""))
    else:
        header.insert(0, "No plan says what this video is for; ask the user what it is about before choosing.")
    if left_out:
        header.append(f"{left_out} more were left out, from the end of the cut; ask for one `part` at a time.")
    return ToolResult(content=[
        "\n".join([*header, *listing]),
        Image(data=storyboard_sheet(shots, aspect=project.width / project.height), format="jpeg"),
    ])

@mcp.tool()
def export_cover(
    project_id: str,
    seconds: Annotated[float, Field(ge=0)],
    reason: Annotated[str, Field(min_length=1)],
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    follow_faces: bool = True,
) -> ToolResult:
    """Save one frame of the cut as a full-size JPEG, for a cover or thumbnail.

    The frame is the one the viewer sees at that moment — covering picture
    where there is some — at the size the cut is rendered at and cropped the
    way the render crops it. It comes back at the size a feed shows it too,
    and for a vertical cut cropped to the 3:4 middle a profile grid shows:
    look at both before telling the user it is done, and choose again if the
    subject is lost at that size or falls outside the middle.

    Args:
        project_id: ID of the project.
        seconds: The moment in the cut, as `propose_covers` lists it.
        reason: Why this frame: what of the video's point it shows, and why
            it reads at a glance. Written for the user, who decides.
        frame: Save it in another shape.
        follow_faces: Crop around the face, as the render does.

    Returns:
        The `output_path` of the JPEG, its `width` and `height`, the reason,
        and the cover at feed size.

    Raises:
        ValueError: If the project does not exist, nothing is on screen then,
            or what is on screen is a title card.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    project = rendering.reshaped(project, frame)
    clip = on_screen(project, seconds)
    if clip is None:
        raise ValueError(f"nothing is on screen at {seconds:g}s of project {project_id}")
    if clip.card is not None:
        raise ValueError(f"{seconds:g}s of project {project_id} is a title card; take the cover from a shot")
    assets = shared.referenced_assets(project)
    source = float(clip.source_range.start) + (seconds - float(clip.timeline_in)) * clip.speed
    framed = rendering.framing_of(project, assets, follow_faces).get(clip.id)
    fps = Fraction(project.fps_num, project.fps_den)
    picture = still(assets[clip.asset_id].path, source, project.width, project.height,
                    placement_at(framed, clip, source, fps))
    folder = str(output_dir())
    os.makedirs(folder, exist_ok=True)
    path = housekeeping.unique_path(folder, rendering.delivery_name(project, "cover" + (f"-{frame}" if frame else ""), ".jpg"))
    picture.save(path, "JPEG", quality=92)
    saved = {"output_path": path, "width": project.width, "height": project.height, "reason": reason}
    grid = ", and the 3:4 middle a profile grid crops a vertical video to" if project.height > project.width else ""
    return ToolResult(
        content=[
            f"Saved {path} ({project.width}x{project.height}). Why: {reason}\n"
            f"Below: the cover at the size a feed shows it{grid}.",
            Image(data=feed_preview(picture), format="jpeg"),
        ],
        structured_content=saved,
    )

def _timecodes(assets: Mapping[str, Asset]) -> Dict[str, float]:
    """Read the timecode each file starts at, for an editing program to line it up by.

    Read at export rather than kept on the asset: it is only wanted here, and a
    file that cannot be read now simply starts at zero.

    Args:
        assets: The files, keyed by asset ID.

    Returns:
        The second each file's timecode starts at, for the files that carry one.
    """
    found: Dict[str, float] = {}
    for asset_id, asset in assets.items():
        try:
            start = timecode_start(probe_file(asset.path))
        except (FileNotFoundError, RuntimeError):
            start = None
        if start:
            found[asset_id] = start
    return found

@mcp.tool()
def export_timeline(
    project_id: str,
    format: Literal["edl", "otio", "fcpxml", "srt"],
) -> dict:
    """Write the cut out for a professional editing program, pointed at the original files.

    For a user who wants to finish the edit in Premiere, DaVinci Resolve or
    Final Cut rather than start again from the rendered file. `fcpxml` is
    the fullest and opens in all three; `otio` is OpenTimelineIO, for Resolve
    and anything built on it; `edl` is the oldest and plainest — the sequence
    alone, one picture track with its sound. `srt` writes the stored captions
    as they fall in the cut, which every one of those programs imports.

    The edit comes across — every clip's in and out, its place and track,
    speed changes, and the parts of the video as markers, on each file's own
    timecode where it has one. `otio` also carries transitions and `fcpxml`
    each clip's level. What stays behind is what this server draws itself:
    colour, fades, voice repair, where an inset sits. `left_behind` says which of those this cut
    uses; tell the user, so they are not surprised in the other program.

    Args:
        project_id: ID of the project.
        format: `fcpxml`, `otio`, `edl`, or `srt`.

    Returns:
        A dictionary with the `output_path` of the file and `left_behind`,
        one line per kind of thing the export could not carry.

    Raises:
        ValueError: If the project does not exist, has nothing on its
            sequence, or has no captions to write as SRT.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if format == "srt":
        placed = place_cues(project, project.subtitles) if project.subtitles else []
        if not placed:
            raise ValueError(
                f"project {project_id} has no captions in the cut; propose them with generate_subtitles "
                "and store them with a set_subtitles operation"
            )
        text, behind = write_srt(placed), []
    else:
        if project.base_video_track is None or not project.base_video_track.clips:
            raise ValueError(f"project {project_id} has nothing on its sequence to export")
        writer = {"edl": write_edl, "otio": write_otio, "fcpxml": write_fcpxml}[format]
        # A title card is a frame and words this server draws; another program would read it as
        # footage of the file behind it. It goes across as the gap it fills, and is said to.
        titled = cards.cards_of(project)
        shown = project.model_copy(deep=True)
        if titled:
            base = shown.base_video_track
            base.clips = [clip for clip in base.clips if clip.card is None]
        assets = shared.referenced_assets(shown)
        text, behind = writer(shown, assets, _timecodes(assets))
        if titled:
            behind.append("title cards (they come across as gaps; add them as titles there): "
                          + ", ".join(clip.card.title for clip in titled[:6]) + (" …" if len(titled) > 6 else ""))
        # Decided at render time from the recordings, so the writers, which never see an
        # analysis, cannot say it. Only which tracks, not how loud: nothing is measured.
        if voice_keys(project, rendering.analyses_of(project), lambda clip: None):
            behind.append("the footage's sound ducking under a narration, and the narration brought up to level")
    folder = str(output_dir())
    os.makedirs(folder, exist_ok=True)
    path = housekeeping.unique_path(folder, rendering.delivery_name(project, "timeline", f".{format}"))
    # UTF-8 with a byte-order mark only for the EDL: the programs that read EDLs guess the
    # encoding, and a Chinese file name in a FROM CLIP NAME comes out garbled without it.
    with open(path, "w", encoding="utf-8-sig" if format == "edl" else "utf-8", newline="\n") as handle:
        handle.write(text)
    return {"output_path": path, "left_behind": behind}
