"""Deciding where to crop a shot shown in a frame of a different shape.

A landscape shot in a portrait video keeps about a third of its width, and
which third is the whole question. Cropping the middle loses anybody standing
off centre; following the face keeps them. The face is already measured,
second by second, at analysis time — this turns those seconds into where the
crop sits over the length of a shot.

When the face moves, the crop has a choice: pan after it, or cut to it. It
cuts. A pan is a camera move nobody made, and one driven by a detector
sampled twice a second either lags behind the face or swims with every jitter
in the measurement. A cut to a new framing is something editors do by hand
when they reframe for vertical, and it reads as one. So the crop holds still
while the face stays comfortably inside it, and jumps only once the face has
sat near an edge for a while — never so soon after the shot's own cut, or so
close to its end, that the two cuts read as a stumble.

Only the axis with room to move is followed: across for a wide shot in a
narrow frame, down for a tall shot in a wide one. A shot whose shape already
matches its frame has nothing to decide.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.models.media import Asset, FaceMeasurement
from app.models.timeline import Clip, Project

# How near the edge of the crop a face may sit before it counts as leaving: the outer
# quarter of the window either side of centre. Provisional until there is a corpus.
REFRAME_MARGIN = 0.25
# How long a face has to stay near the edge before the crop cuts to it. Shorter and a
# glance to one side is a cut; longer and somebody is half out of frame for all of it.
# Provisional.
REFRAME_HOLD_SECONDS = 2.0
# How long a framing has to last: a reframe no sooner than this after the shot's own cut
# or the last reframe, and no later than this before the shot ends. Two cuts closer than
# this read as one fumbled one. Provisional.
REFRAME_MIN_SHOT_SECONDS = 1.5
# A crop covering nearly the whole of an axis has nowhere to go.
SLACK = 0.001

@dataclass(frozen=True)
class Framing:
    """Where the crop sits over the length of one clip.

    Attributes:
        axis: `x` when the crop moves across the picture, `y` when it moves
            down it.
        positions: `(frame, centre)` pairs, in order: from that frame of the
            clip on, the crop is centred at that fraction of the picture's
            width or height. The first always starts at frame 0. Frames are
            counted from the first frame the clip reads, which is before its
            in point when a transition runs it in.
    """

    axis: str
    positions: Tuple[Tuple[int, float], ...]

def _span(source: Tuple[int, int], box: Tuple[int, int]) -> Tuple[str, float]:
    """Work out which axis a crop moves along, and how much of it the crop covers.

    Args:
        source: The picture's `(width, height)`.
        box: The `(width, height)` it is shown in.

    Returns:
        `(axis, span)` — the span a fraction of that axis, 1.0 when nothing is
        cropped from it.
    """
    source_aspect = Fraction(source[0], source[1])
    box_aspect = Fraction(box[0], box[1])
    if source_aspect > box_aspect:
        return "x", float(box_aspect / source_aspect)
    return "y", float(source_aspect / box_aspect)

def _clamped(centre: float, span: float) -> float:
    """Keep a crop's centre where the crop stays inside the picture.

    Args:
        centre: Where it would be centred.
        span: How much of the axis it covers.

    Returns:
        The centre, moved in as far as the edge requires, to three places.
    """
    return round(min(max(centre, span / 2), 1 - span / 2), 3)

def frame_clip(
    clip: Clip,
    source: Tuple[int, int],
    box: Tuple[int, int],
    faces: Sequence[FaceMeasurement],
    fps: Fraction,
) -> Optional[Framing]:
    """Work out where one clip's crop sits, following the face in it.

    Args:
        clip: The clip.
        source: Its picture's `(width, height)`, as shown.
        box: The `(width, height)` it is drawn at.
        faces: The per-second face measurements of its file.
        fps: The output frame rate.

    Returns:
        The framing, or None when the shot needs no crop along either axis or
        nobody's face was seen anywhere in it — then it is cropped from the
        middle, as it always was.
    """
    axis, span = _span(source, box)
    if span >= 1 - SLACK:
        return None
    opened = float(clip.video_source_start)
    closed = float(clip.source_range.end)
    seen: List[Tuple[float, float]] = []
    for second in faces:
        where = second.face_x if axis == "x" else second.face_y
        if second.end <= opened or second.start >= closed or not second.faces or where is None:
            continue
        seen.append((max(second.start, opened), where))
    if not seen:
        return None

    speed = float(clip.speed)
    # Opened on wherever the face is first seen. A median over the opening seconds would
    # split the difference between two places and frame the gap between them.
    centre = _clamped(seen[0][1], span)
    positions: List[Tuple[float, float]] = [(opened, centre)]
    reach = span / 2 * (1 - REFRAME_MARGIN)
    drifting: Optional[float] = None
    for at, where in seen:
        if abs(where - centre) <= reach:
            drifting = None
            continue
        drifting = at if drifting is None else drifting
        # Timed in the shot as it plays, so a slowed-down shot holds as long on screen.
        held = (at + 1.0 - drifting) / speed
        # Too soon after the last cut is a reason to wait, not to give up: a face that
        # moved in the first second and stayed there still gets its framing, just later.
        cut_at = max(drifting, positions[-1][0] + REFRAME_MIN_SHOT_SECONDS * speed)
        if held >= REFRAME_HOLD_SECONDS and cut_at <= at + 1.0:
            if (closed - cut_at) / speed < REFRAME_MIN_SHOT_SECONDS:
                break
            moved = _clamped(where, span)
            if moved != centre:
                centre = moved
                positions.append((cut_at, centre))
            drifting = None
    return Framing(axis=axis, positions=tuple(
        (round(Fraction(str(at - opened)) / Fraction(str(speed)) * fps) if index else 0, where)
        for index, (at, where) in enumerate(positions)
    ))

def centre_at(framing: Framing, clip: Clip, seconds: float, fps: Fraction) -> float:
    """Say where a clip's crop is centred at one moment of its source.

    Args:
        framing: The clip's framing.
        clip: The clip.
        seconds: A moment of its source, in seconds.
        fps: The output frame rate.

    Returns:
        The centre along the framing's axis, as a fraction of the picture.
    """
    frame = round(Fraction(str(seconds - float(clip.video_source_start))) / Fraction(str(clip.speed)) * fps)
    return [where for start, where in framing.positions if start <= frame][-1]

def box_of(clip: Clip, width: int, height: int) -> Tuple[int, int]:
    """Say what size a clip is drawn at.

    The same arithmetic the renderer uses, so the crop is worked out for the
    box the picture actually lands in.

    Args:
        clip: The clip.
        width: The frame's width.
        height: The frame's height.

    Returns:
        `(width, height)` of its box: the whole frame without a layout.
    """
    from app.engine.builder import overlay_box

    _, _, box_width, box_height = overlay_box(clip, width, height)
    return box_width, box_height

def frame_project(
    project: Project,
    assets: Mapping[str, Asset],
    faces: Mapping[str, Sequence[FaceMeasurement]],
) -> Dict[str, Framing]:
    """Work out the framing of every picture clip in a project.

    Args:
        project: The project, at the size it is being rendered at.
        assets: Its files, sized.
        faces: Each file's face measurements, keyed by asset ID.

    Returns:
        A framing for each clip that has one, keyed by clip ID. Clips left out
        are cropped from the middle.
    """
    fps = Fraction(project.fps_num, project.fps_den)
    framed: Dict[str, Framing] = {}
    for track in project.video_tracks:
        for clip in track.clips:
            asset = assets.get(clip.asset_id)
            if asset is None or not asset.width or not asset.height:
                continue
            framing = frame_clip(
                clip, (asset.width, asset.height), box_of(clip, project.width, project.height),
                faces.get(clip.asset_id, ()), fps,
            )
            if framing is not None:
                framed[clip.id] = framing
    return framed

def crop_filter(width: int, height: int, framing: Optional[Framing]) -> str:
    """Write the crop that cuts a scaled picture down to its box.

    The picture has already been scaled to cover the box, so exactly one axis
    has room left over; the crop's centre along it moves when the framing
    says to, frame by frame, and is held inside the picture.

    Args:
        width: Box width.
        height: Box height.
        framing: Where the crop sits, or None for the middle.

    Returns:
        The `crop` filter, with the commas inside its expressions escaped for
        a filtergraph.
    """
    if framing is None:
        return f"crop={width}:{height}"
    size, out = ("iw", "ow") if framing.axis == "x" else ("ih", "oh")
    # if(lt(n,f1), c0, if(lt(n,f2), c1, ... ck)), built from the last framing inwards.
    centre = f"{framing.positions[-1][1]:g}"
    for (frame, _), (_, before) in zip(reversed(framing.positions[1:]), reversed(framing.positions[:-1])):
        centre = rf"if(lt(n\,{frame})\,{before:g}\,{centre})"
    # Escaped rather than quoted, the way the rest of the graph is: inside quotes the
    # backslashes would reach the expression parser and it would refuse them.
    offset = rf"max(0\,min({size}-{out}\,{centre}*{size}-{out}/2))"
    return f"crop={width}:{height}:{framing.axis}={offset}"
