"""Looking at footage and renders: frames, storyboards, and a close look at each join in a cut."""

import os
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Sequence, Tuple

from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field

from app.models.timeline.parts import Project
from app.models.job import Job
from app.engine.delivery import Finding
from app.engine.joins import HeardWord, Join, joins_in, pause_across, words_heard
from app.engine.frames import Crop, format_timestamp, storyboard_sheet, tile_size_for
from app.engine.listen import WINDOW_SECONDS, cut_sound, levels_every
from app.engine.subtitles import place_cues
from app import runtime
from app.runtime import mcp
from app.tools import rendering, shared, versioning

@mcp.tool()
def view_frames(
    asset_ids: list[str],
    start: Annotated[float, Field(ge=0)] = 0,
    end: Optional[float] = None,
    count: Annotated[int, Field(ge=1, le=shared.MAX_STORYBOARD_TILES)] = 12,
    crop: Optional[Crop] = None,
) -> ToolResult:
    """Look at frames from one or more assets as one labeled contact sheet image.

    With a single asset, frames are sampled at evenly spaced times between
    `start` and `end`: use a wide range for an overview and a narrow range to
    inspect a moment before choosing a cut point. With several, `count` frames
    are taken across the whole of each one and they follow each other on the
    sheet, which is how a folder of footage is surveyed without a call per
    file. A range means nothing across files of different lengths, so `start`
    and `end` are only accepted for a single asset.

    Every tile is labeled with its number, its source time, and which file it
    came from. Works without `analyze_asset`. Fewer frames come out larger,
    since a sheet is kept to the same pixel budget, and `crop` zooms into part
    of the frame so small text or numbers can be read.

    Args:
        asset_ids: IDs of the assets to sample, in the order to show them.
        start: Range start in seconds; one asset only.
        end: Range end in seconds; one asset only, and the end of the asset
            when omitted.
        count: Number of frames per asset.
        crop: Part of the frame to zoom into, as fractions of it with `x` and
            `y` at the top-left corner; one asset only.

    Returns:
        A text block listing each tile's asset and source time, followed by
        the contact sheet as a JPEG image.

    Raises:
        ValueError: If an asset does not exist or has no video, the range is
            empty, a range is given for more than one asset, the crop is
            outside the frame or given for more than one asset, or the frames
            asked for do not fit on one sheet.
        RuntimeError: If frames cannot be decoded.
    """
    if not asset_ids:
        raise ValueError("name at least one asset to show frames from")
    if len(asset_ids) * count > shared.MAX_STORYBOARD_TILES:
        raise ValueError(
            f"{len(asset_ids)} assets at {count} frames each is more than one sheet holds; "
            f"ask for at most {shared.MAX_STORYBOARD_TILES} frames in total"
        )
    if len(asset_ids) > 1 and (start or end is not None):
        raise ValueError("a time range only means something for one asset; drop start and end, or ask about one file")
    if crop is not None:
        if len(asset_ids) > 1:
            raise ValueError("a crop only means something for one asset; drop crop, or ask about one file")
        if crop.width <= 0 or crop.height <= 0:
            raise ValueError(
                f"crop width and height must be greater than 0, not {crop.width} by {crop.height}; "
                "give the part of the frame as fractions, such as width 0.5 and height 0.5"
            )
        if crop.x < 0 or crop.y < 0:
            raise ValueError(
                f"crop x and y must not be negative, not {crop.x} and {crop.y}; "
                "0 is the left and top edge of the frame"
            )
        if crop.x + crop.width > 1:
            raise ValueError(
                f"crop runs past the right edge: x {crop.x} plus width {crop.width} is more than 1; "
                "move x left, or make width smaller"
            )
        if crop.y + crop.height > 1:
            raise ValueError(
                f"crop runs past the bottom edge: y {crop.y} plus height {crop.height} is more than 1; "
                "move y up, or make height smaller"
            )

    assets = [shared.get_asset(asset_id) for asset_id in asset_ids]
    first = assets[0]
    if first.width and first.height:
        aspect = first.width / first.height
        if crop is not None:
            aspect = (first.width * crop.width) / (first.height * crop.height)
    else:
        aspect = 16 / 9

    shots: List[tuple[str, float, str]] = []
    listing: List[str] = []
    for asset in assets:
        if not asset.has_video or asset.duration is None:
            raise ValueError(f"asset {asset.id} has no video frames to show")
        duration = float(asset.duration)
        last = duration if end is None else min(end, duration)
        if last <= start:
            raise ValueError(f"empty range: start {start}s must be before end {last}s (asset is {duration}s long)")
        step = (last - start) / count
        for index in range(count):
            # Sample the middle of each interval, and stay clear of the very last frame, which may not decode.
            seconds = round(min(start + step * (index + 0.5), duration - 0.05), 3)
            number = len(shots) + 1
            name = Path(asset.path).name
            shots.append((asset.path, seconds, f"#{number}  {format_timestamp(seconds)}  {name[:14]}"))
            listing.append(f"#{number}: {name} | {seconds:.3f}s ({format_timestamp(seconds)})")

    where = f"asset {asset_ids[0]}" if len(asset_ids) == 1 else f"{len(asset_ids)} assets, {count} frames each"
    columns, box = tile_size_for(len(shots), aspect)
    said = [f"Contact sheet of {where}, in reading order:"]
    if crop is not None:
        said.append(
            f"zoomed to x {crop.x:.2f}–{crop.x + crop.width:.2f}, "
            f"y {crop.y:.2f}–{crop.y + crop.height:.2f} of the frame"
        )
    return ToolResult(content=[
        "\n".join([*said, *listing]),
        Image(data=storyboard_sheet(shots, columns=columns, tile_size=box, crop=crop), format="jpeg"),
    ])

@mcp.tool()
def view_render(
    job_id: str,
    at: Literal["even", "captions", "spots"] = "even",
    times: Optional[List[Annotated[float, Field(ge=0)]]] = None,
    start: Annotated[float, Field(ge=0)] = 0,
    end: Optional[float] = None,
    count: Annotated[int, Field(ge=1, le=shared.MAX_STORYBOARD_TILES)] = 12,
) -> ToolResult:
    """Look at frames of a finished render: what the viewer will see, captions and transitions included.

    `preview_project` shows the footage a cut is made of; this shows the file
    that came out, so it is how to check what only rendering makes — captions
    burned in where they belong and readable, transitions that really happen,
    the framing of another shape. Render a preview, look here, fix what is
    wrong, and only then render the final cut. How one shot hands over to the
    next — every cut, transition and picture laid over — is `inspect_join`.

    The frames are returned to you as a tool result, which means they leave
    this machine if you are not running on it.

    Args:
        job_id: The render, preview or full, from `render_project`.
        at: `even` spreads `count` frames over the stretch; `captions` takes
            one from the middle of each caption on screen; `spots` takes, for
            each stretch reading every frame found black, frozen or a flash,
            the frame before it, its middle and the frame after.
        times: Exact seconds of the render to look at instead.
        start: Only look from here, in seconds of the render.
        end: Only look up to here; omit for the end.
        count: How many frames `even` takes.

    Returns:
        A text block naming each tile's time and what it is, then the frames
        as one JPEG. It says so when the project has changed since the render,
        when a render has no captions burned in, or when more frames were
        wanted than one sheet holds.

    Raises:
        ValueError: If the job is not a finished render or its file is gone.
    """
    job, path = rendering.finished_render(job_id)
    project = rendering.rendered_from(job)
    length = float(project.duration) if project is not None else None
    last = length if end is None else (end if length is None else min(end, length))
    said: List[str] = []
    wanted: List[tuple] = []
    if times:
        wanted = [(seconds, "") for seconds in times]
    elif at == "spots":
        if job.spots is None:
            said.append("this render was not read frame by frame — the check failed or was stopped — so nothing "
                        "is known about black, frozen or flashing frames in it; render again to have it read")
        elif not job.spots:
            said.append("every frame of this render was read and nothing was found black, frozen or a flash")
        step = 1 / (project.fps_num / project.fps_den) if project is not None else 0.04
        for spot in job.spots or []:
            begin, finish = float(spot["start"]), float(spot["end"])
            if finish <= start or (last is not None and begin >= last):
                continue
            name = f"{spot['kind']} {begin:.2f}–{finish:.2f}s"
            wanted += [(max(0.0, begin - step), f"{name}, the frame before"), ((begin + finish) / 2, f"{name}, middle"),
                       (finish + step, f"{name}, the frame after")]
    elif at == "captions":
        if not job.captioned:
            said.append("this render has no captions burned in; render with burn_subtitles to look at them")
        placed = place_cues(project, project.subtitles) if project else []
        wanted = [((float(cue.start) + float(cue.end)) / 2, f"{cue.cue_id} {cue.text}") for cue in placed
                  if float(cue.end) > start and (last is None or float(cue.start) < last)]
    else:
        stop = last if last is not None else start + 1
        step = (stop - start) / count
        wanted = [(start + step * (index + 0.5), "") for index in range(count)]
    if len(wanted) > shared.MAX_STORYBOARD_TILES:
        said.append(f"{len(wanted)} frames wanted, the first {shared.MAX_STORYBOARD_TILES} shown; look at the rest with "
                    "a later `start`")
        wanted = wanted[:shared.MAX_STORYBOARD_TILES]
    if rendering.changed_since(job):
        said.append("the project has been edited since this render, so it may no longer show what the cut is now")
    if length is not None:
        wanted = [(min(seconds, length - 0.05), what) for seconds, what in wanted]
    if not wanted:
        return ToolResult(content=["Nothing to show: " + "; ".join(said)])
    shots = [(path, round(seconds, 3), f"#{number}  {format_timestamp(seconds)}")
             for number, (seconds, _) in enumerate(wanted, start=1)]
    listing = [f"#{number}: {seconds:.2f}s ({format_timestamp(seconds)})" + (f" | {what}" if what else "")
               for number, (seconds, what) in enumerate(wanted, start=1)]
    return ToolResult(content=[
        "\n".join([*said, "Frames of the render, in reading order:", *listing]),
        Image(data=storyboard_sheet(shots), format="jpeg"),
    ])

# How much of a join `inspect_join` looks at unless asked for more: a second and a half
# either side, four frames a second. Enough for a movement to carry through or not, and
# one join is a sheet of fourteen frames.
JOIN_SECONDS = 1.5
JOIN_FPS = 4.0
# How the two frames either side of a cut are framed, so the cut can be found in the sheet.
CUT_BORDER = (230, 60, 60)

def project_as_rendered(job: Job) -> Tuple[Optional[Project], bool]:
    """The project a render was made from, as it was when it was rendered.

    Args:
        job: The render.

    Returns:
        `(project, exact)`: the project at the version rendered and True; or,
        when that version is no longer in its history, the project as it is
        now and False. `(None, False)` when the project is gone.
    """
    project = rendering.rendered_from(job)
    if project is None or not rendering.changed_since(job):
        return project, project is not None
    for version in runtime.history.versions_of("projects/", f"{project.id}.json", 50):
        data = versioning.project_json_at(project.id, version.commit)
        if data is not None and data.get("version") == job.project_version:
            return Project.model_validate(data), True
    return project, False

def found_at(join: Join, findings: Sequence[Finding]) -> List[Finding]:
    """The findings about the stretch of the cut that looking at a join shows."""
    opened, closed = join.at - JOIN_SECONDS, join.at + JOIN_SECONDS
    return [finding for finding in findings
            if any(start <= closed and end >= opened for start, end in finding.spans)]

def _join_line(number: int, join: Join, found: Sequence[Finding]) -> str:
    """Say where a join is, what kind, between what, and what was found there."""
    line = (f"#{number} {format_timestamp(join.at)} ({join.at:.2f}s) {join.kind}: {join.before} → {join.after}"
            + (f"; part 「{join.part}」 begins" if join.part else "")
            + (f"; the sound changes over at {join.sound_at:.2f}s" if join.sound_at is not None else ""))
    return "\n".join([line, *(f"    found: {finding.check}: {finding.message}" for finding in found)])

def _words_across(join: Join, heard: Sequence[HeardWord]) -> str:
    """Write the words heard around a join, in order, with ‖ where the picture cuts."""
    words: List[str] = []
    side, marked = None, False
    for word in heard:
        if not marked and word.start >= join.at:
            words.append("‖")
            marked = True
        if side != word.clip_id:
            words.append(f"[clip {word.clip_id}]")
            side = word.clip_id
        words.append(f"{word.start - join.at:+.2f} {word.text}" + ("✂" if word.cut else ""))
    if not heard:
        return "none transcribed"
    return " ".join(words + ([] if marked else ["‖"]))

def _levels_across(path: str, join: Join, opened: float, closed: float, keep_as: str) -> str:
    """Measure a render's sound around a join, keeping the stretch to play."""
    try:
        cut_sound(path, opened, closed, keep_as)
    except RuntimeError:
        return "the render has no sound here"
    measured = levels_every(keep_as, WINDOW_SECONDS)
    split = round((join.at - opened) / WINDOW_SECONDS)
    return (f"dB every {WINDOW_SECONDS:g}s from {opened - join.at:+.2f}s: "
            + " ".join(f"{level:g}" for level in measured[:split]) + " ‖ "
            + " ".join(f"{level:g}" for level in measured[split:]) + f"\n  sound: {keep_as}")

@mcp.tool()
def inspect_join(
    job_id: str,
    joins: Optional[List[Annotated[int, Field(ge=1)]]] = None,
    at: Optional[List[Annotated[float, Field(ge=0)]]] = None,
    seconds: Annotated[float, Field(gt=0, le=3)] = JOIN_SECONDS,
    fps: Annotated[float, Field(gt=0, le=8)] = JOIN_FPS,
) -> ToolResult:
    """Look at how a finished render hands over from one shot to the next.

    A storyboard shows each shot; whether a movement carries through a cut,
    or a sentence ends before the picture moves on, is in the second either
    side. Every join counts: plain cuts, transitions, B-roll coming in and
    going off, the start of a part.

    With neither `joins` nor `at`, it lists the render's joins and no frames:
    number, time, kind, what is either side, and what `check_render` found
    there. Then look at the ones that matter, by number or by time.

    The frames are returned to you as a tool result, which means they leave
    this machine if you are not running on it.

    Args:
        job_id: The render, preview or full, from `render_project`.
        joins: Numbers from the list; they belong to this render only.
        at: Seconds of the render; each looks at the join nearest it.
        seconds: How far either side of each join to look.
        fps: Frames a second to take.

    Returns:
        The list; or for each join its frames, a row before the cut and a row
        from it with the two either side framed red; the transcribed words
        laid onto the render's time, ✂ on any a cut takes into, and what
        they read as, corrected with the captions' glossary; the captions on
        screen there, as written or corrected by hand, which are what was
        said where the two differ; where a line
        ends and another begins across the cut, the pause between them beside
        the pauses the speaker makes between sentences; the render's
        sound level every 0.1s with ‖ at the cut; and a WAV of the stretch to
        play or hear again with `listen_again`. It says when the project has
        changed since the render, and which joins did not fit on the sheet.

    Raises:
        ValueError: If the job is not a finished render, a number is not in
            the list, or one join is more frames than a sheet holds.
    """
    job, path = rendering.finished_render(job_id)
    project, exact = project_as_rendered(job)
    if project is None:
        raise ValueError(f"the project render {job_id} was made from is gone, so its joins are not known")
    said: List[str] = []
    if not exact:
        said.append("the project has been edited since this render and the version rendered is no longer kept, "
                    "so these are the joins of the cut as it is now; render again to be sure")
    elif rendering.changed_since(job):
        said.append("the project has been edited since this render; these are the joins of the version rendered")
    listed = joins_in(project, {asset_id: os.path.basename(asset.path)
                                for asset_id, asset in shared.referenced_assets(project).items()})
    if not listed:
        return ToolResult(content=["\n".join([*said, f"render {job_id} has no joins: it is one shot"])])
    findings = [finding for finding in rendering.check_findings(project, job.captioned) if finding.spans]
    if not joins and not at:
        return ToolResult(content=["\n".join([
            *said, f"Joins of render {job_id}; look at any with `joins` or `at`:",
            *(_join_line(number, join, found_at(join, findings)) for number, join in enumerate(listed, start=1)),
        ])])

    wanted: List[int] = []
    for number in joins or []:
        if number > len(listed):
            raise ValueError(f"render {job_id} has {len(listed)} joins; there is no #{number}")
        wanted.append(number)
    for seconds_in in at or []:
        wanted.append(1 + min(range(len(listed)), key=lambda index: abs(listed[index].at - seconds_in)))
    wanted = list(dict.fromkeys(wanted))
    steps = round(seconds * fps)
    per_join = 2 * (steps + 1)
    if per_join > shared.MAX_STORYBOARD_TILES:
        raise ValueError(f"{seconds:g}s at {fps:g} frames a second is {per_join} frames a join, more than the "
                         f"{shared.MAX_STORYBOARD_TILES} a sheet holds; look less far, or more thinly")
    fits = shared.MAX_STORYBOARD_TILES // per_join
    if len(wanted) > fits:
        said.append(f"{', '.join(f'#{number}' for number in wanted[fits:])} did not fit on the sheet; "
                    "look at them in another call")
        wanted = wanted[:fits]

    length = float(project.duration)
    frame = project.fps_den / project.fps_num
    analyses = rendering.analyses_of(project)
    placed = place_cues(project, project.subtitles) if project.subtitles else []
    shots: list = []
    borders: list = []
    text: List[str] = []
    for number in wanted:
        join = listed[number - 1]
        before = [join.at - step / fps for step in range(steps, 0, -1)] + [join.at - frame]
        after = [join.at] + [join.at + step / fps for step in range(1, steps + 1)]
        for times, edge in ((before, len(before) - 1), (after, 0)):
            for index, moment in enumerate(times):
                shots.append((path, round(min(max(0.0, moment), length - 0.05), 3),
                              f"#{number} {moment - join.at:+.2f}s"))
                borders.append(CUT_BORDER if index == edge else None)
        opened, closed = max(0.0, join.at - seconds), min(length, join.at + seconds)
        keep_as = os.path.join(runtime.WORKSPACE_DIR, "temp", "joins", f"render-{job_id[:8]}-{join.at:.2f}.wav")
        heard = words_heard(project, analyses, opened, closed)
        text += [
            _join_line(number, join, found_at(join, findings)),
            f"  frames: a row of {steps + 1} up to the cut, then {steps + 1} from it",
            "  words: " + _words_across(join, heard),
        ]
        if heard:
            # The words above are as the transcriber timed them; this is what they say, corrected
            # with the glossary the captions use, so a misheard word fixed once reads fixed here.
            sides = [shared.glossed("".join(word.text for word in heard if (word.start >= join.at) == later))
                     for later in (False, True)]
            text.append(f"  reads: 「{sides[0]}」 ‖ 「{sides[1]}」")
        # The captions are what somebody read and corrected, word for word, where the transcript
        # and its glossary only fix what was misheard the same way every time.
        shown = [f"{float(cue.start) - join.at:+.2f} {cue.text}" for cue in placed
                 if float(cue.end) > opened and float(cue.start) < closed]
        if shown:
            text.append("  captions: " + " / ".join(shown))
        pause = pause_across(join.at, heard, analyses, join.part)
        if pause:
            text.append("  pause: " + pause)
        text += [
            "  levels: " + _levels_across(path, join, opened, closed, keep_as),
        ]
    return ToolResult(content=[
        "\n".join([*said, *text]),
        Image(data=storyboard_sheet(shots, columns=steps + 1, borders=borders), format="jpeg"),
    ])
