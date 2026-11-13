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

import statistics
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine import insets, moves
from app.models.media import Asset, FaceMeasurement
from app.models.timeline.parts import Clip, Project

# The shapes a cut is delivered in, as width to height.
FRAME_SHAPES = {"landscape": (16, 9), "portrait": (9, 16), "square": (1, 1)}

def framed_size(width: int, height: int, frame: Optional[str]) -> Tuple[int, int]:
    """Say how big a video of one size comes out in another shape.

    The short side is kept, because that is the side a platform's resolution
    is quoted by: 1080p is 1080 across the short side whichever way up it is.

    Args:
        width: The video's own width.
        height: Its own height.
        frame: `landscape`, `portrait`, `square`, or None for as it is.

    Returns:
        `(width, height)` in that shape.
    """
    if frame is None:
        return width, height
    across, down = FRAME_SHAPES[frame]
    short = min(width, height)
    even = lambda value: max(2, int(round(value / 2)) * 2)
    if across >= down:
        return even(short * across / down), short
    return short, even(short * down / across)

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
# Provisional (roadmap §13): a shot with no face that a crop would cut down to less than this
# share of its width (or height) is shown whole instead, over a blurred copy of itself — a
# landscape shot in a vertical video keeps a third. A smaller difference, 4:3 in 16:9, is cropped.
WHOLE_BELOW_SPAN = 0.7

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
        whole: Show the whole picture over a blurred copy of itself instead
            of cropping it; the positions are then unused.
        zoom: How much closer than covering the box the crop is; above 1 the
            other axis is cropped too.
        cross: Where the crop is centred along the other axis, when zoomed.
        centred: Told to fill, with no face seen to follow: cropped from the
            middle, which is a guess worth saying out loud rather than a
            framing anybody chose.
        motion: How the placed picture moves over the clip, as
            `moves.motion_of` says; `none` holds it.
        anchor: The point of the placed picture, as fractions of its width
            and height, that a push closes in on and keeps still on the
            screen: where the face sits in the box, or the middle.
    """

    axis: str
    positions: Tuple[Tuple[int, float], ...]
    whole: bool = False
    zoom: float = 1.0
    cross: float = 0.5
    centred: bool = False
    motion: str = "none"
    anchor: Tuple[float, float] = moves.MIDDLE

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

@dataclass(frozen=True)
class Placement:
    """How one frame of a clip sits in its box, for a still drawn the way the render draws it.

    Attributes:
        centre: Where the crop is centred along the axis it crops.
        whole: Shown whole over a blurred copy instead of cropped.
        zoom: How much closer than covering the box.
        cross: Where the crop is centred along the other axis, when zoomed.
        window: The part of the placed picture a move has on screen at
            that moment, as `moves.window_at` gives it: `(zoom, left, top)`.
    """

    centre: float = 0.5
    whole: bool = False
    zoom: float = 1.0
    cross: float = 0.5
    window: Tuple[float, float, float] = (1.0, 0.0, 0.0)

def placement_at(framing: Optional[Framing], clip: Clip, seconds: float, fps: Fraction) -> Optional[Placement]:
    """Say how a clip sits in its box at one moment of its source.

    Args:
        framing: The clip's framing, or None for a crop from the middle.
        clip: The clip.
        seconds: A moment of its source, in seconds.
        fps: The output frame rate.

    Returns:
        The placement, or None for a crop from the middle.
    """
    if framing is None:
        return None
    window = moves.window_at_source(framing.motion, clip, seconds, fps, framing.anchor)
    if framing.whole:
        return Placement(whole=True, window=window)
    return Placement(centre=centre_at(framing, clip, seconds, fps), zoom=framing.zoom, cross=framing.cross,
                     window=window)

def box_of(clip: Clip, project: Project, source: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """Say what size a clip is drawn at.

    The same arithmetic the renderer uses, so the crop is worked out for the
    box the picture actually lands in.

    Args:
        clip: The clip.
        project: The cut, at the size it is drawn at.
        source: The clip's picture size, which a corner picture keeps the
            shape of.

    Returns:
        `(width, height)` of its box: the whole frame without a layout.
    """
    from app.engine.builder import overlay_box

    _, _, box_width, box_height = overlay_box(clip, project, source)
    return box_width, box_height

def place_clip(
    clip: Clip,
    source: Tuple[int, int],
    box: Tuple[int, int],
    faces: Sequence[FaceMeasurement],
    fps: Fraction,
    follow_faces: bool = True,
) -> Optional[Framing]:
    """Decide how one clip sits in its box: where its crop goes, or whether it is shown whole.

    What the clip was told wins: `whole`, or `fill` with a centre and a zoom
    of somebody's choosing; `fill` alone follows a face, or crops from the
    middle when none was seen. Otherwise a face is followed; a shot with no face
    that would lose most of its width is shown whole; anything else is
    cropped from the middle.

    Args:
        clip: The clip.
        source: Its picture's `(width, height)`.
        box: The `(width, height)` it is drawn at.
        faces: The per-second face measurements of its file.
        fps: The output frame rate.
        follow_faces: Whether a face may be followed.

    Returns:
        The framing, or None to crop from the middle.
    """
    fit = clip.fit
    axis, span = _span(source, box)
    zoom = fit.zoom if fit is not None and fit.mode == "fill" else 1.0
    if span >= 1 - SLACK and zoom <= 1:
        return None
    if fit is not None and fit.mode == "whole":
        return Framing(axis=axis, positions=((0, 0.5),), whole=True)
    if fit is not None and (fit.center_x is not None or fit.center_y is not None or zoom > 1):
        # Placed by hand: it stays where it was put. Zooming in narrows both axes.
        along, other = (fit.center_x, fit.center_y) if axis == "x" else (fit.center_y, fit.center_x)
        narrowed = span / zoom
        return Framing(axis=axis, positions=((0, _clamped(0.5 if along is None else along, narrowed)),),
                       zoom=zoom, cross=_clamped(0.5 if other is None else other, 1 / zoom))
    followed = frame_clip(clip, source, box, faces, fps) if follow_faces else None
    if followed is not None:
        return followed
    if fit is not None:
        # Told to fill and nothing to follow. No other measure of where the subject is exists,
        # so the middle it is, marked so that the preview and the render can say so — but only
        # when faces were looked for: with following turned off, the middle is what was asked.
        return Framing(axis=axis, positions=((0, 0.5),), centred=follow_faces)
    seen = any(second.faces for second in faces
               if second.end > float(clip.video_source_start) and second.start < float(clip.source_range.end))
    if not seen and span < WHOLE_BELOW_SPAN:
        return Framing(axis=axis, positions=((0, 0.5),), whole=True)
    return None

def in_box(
    across: float,
    down: float,
    clip: Clip,
    seconds: float,
    source: Tuple[int, int],
    box: Tuple[int, int],
    framing: Optional[Framing],
    fps: Fraction,
) -> Tuple[float, float]:
    """Say where a point of a shot lands in its box, once the shot is placed the way the render places it.

    Args:
        across: The point across the shot, 0 the left edge and 1 the right.
        down: The point down it, 0 the top and 1 the bottom.
        clip: The clip.
        seconds: The moment of its file the point is at, for a crop that moves.
        source: The shot's `(width, height)`.
        box: The `(width, height)` it is drawn at.
        framing: How it is placed, or None for a crop from the middle.
        fps: The output frame rate.

    Returns:
        The point as fractions of the box's width and height, which are
        outside 0 to 1 when the crop leaves it out.
    """
    (width, height), (box_width, box_height) = source, box
    if framing is not None and framing.whole:
        # Fitted inside the box, in its middle.
        scale = min(box_width / width, box_height / height)
        return (0.5 + (across - 0.5) * width * scale / box_width,
                0.5 + (down - 0.5) * height * scale / box_height)
    scale = max(box_width / width, box_height / height) * (framing.zoom if framing is not None else 1.0)
    shown_x, shown_y = box_width / (width * scale), box_height / (height * scale)
    centre_x = centre_y = 0.5
    if framing is not None:
        along = centre_at(framing, clip, seconds, fps)
        centre_x, centre_y = (along, framing.cross) if framing.axis == "x" else (framing.cross, along)
        if framing.zoom <= 1:
            # Not zoomed, the other axis is cropped from the middle whatever `cross` says.
            centre_x, centre_y = (along, 0.5) if framing.axis == "x" else (0.5, along)
    left = min(max(centre_x - shown_x / 2, 0.0), 1 - shown_x)
    top = min(max(centre_y - shown_y / 2, 0.0), 1 - shown_y)
    return (across - left) / shown_x, (down - top) / shown_y

def anchor_of(
    clip: Clip,
    source: Tuple[int, int],
    box: Tuple[int, int],
    faces: Sequence[FaceMeasurement],
    framing: Optional[Framing],
    fps: Fraction,
) -> Tuple[float, float]:
    """Find the point of a clip's box that a push closes in on: where the face sits in it.

    Each second the face was seen in the clip is put where the placed shot
    puts it, under the crop of that second, so a crop that cuts to follow
    the face is accounted for; the median of those is the anchor. A median
    rather than a mean, so a second the detector caught somebody passing at
    the edge does not pull the push off the speaker.

    Args:
        clip: The clip.
        source: Its picture's `(width, height)`.
        box: The `(width, height)` it is drawn at.
        faces: The per-second face measurements of its file.
        framing: How it is placed, or None for a crop from the middle.
        fps: The output frame rate.

    Returns:
        The anchor as fractions of the box, held inside it; the middle when
        no face was seen in the clip.
    """
    opened = float(clip.video_source_start)
    closed = float(clip.source_range.end)
    seen: List[Tuple[float, float]] = []
    for second in faces:
        if (second.end <= opened or second.start >= closed or not second.faces
                or second.face_x is None or second.face_y is None):
            continue
        seen.append(in_box(second.face_x, second.face_y, clip, max(second.start, opened), source, box, framing, fps))
    if not seen:
        return moves.MIDDLE
    held = lambda value: round(min(max(value, 0.0), 1.0), 3)
    return held(statistics.median(x for x, _ in seen)), held(statistics.median(y for _, y in seen))

def frame_project(
    project: Project,
    assets: Mapping[str, Asset],
    faces: Mapping[str, Sequence[FaceMeasurement]],
    follow_faces: bool = True,
) -> Dict[str, Framing]:
    """Work out how every picture clip in a project sits in its box.

    Args:
        project: The project, at the size it is being rendered at.
        assets: Its files, sized.
        faces: Each file's face measurements, keyed by asset ID.
        follow_faces: Whether faces may be followed.

    Returns:
        A framing for each clip that has one, keyed by clip ID. Clips left out
        are cropped from the middle and hold still. A clip that moves always
        has one, which says how it moves and, for video, which point it
        closes in on: the face, when faces may be followed and one was seen.
    """
    fps = Fraction(project.fps_num, project.fps_den)
    framed: Dict[str, Framing] = {}
    base = project.base_video_track
    # The sequence laid into the other half of a split screen is framed in that half, on its own subject.
    for track in ([base] if base else []) + insets.drawn_overlays(project):
        for clip in track.clips:
            asset = assets.get(clip.asset_id)
            if asset is None or not asset.width or not asset.height:
                continue
            source = (asset.width, asset.height)
            box = box_of(clip, project, source)
            seen = faces.get(clip.asset_id, ())
            framing = place_clip(clip, source, box, seen, fps, follow_faces)
            motion = moves.motion_of(clip, asset.still)
            if motion != "none":
                # A crop from the middle, said out loud, so the move has somewhere to be written down.
                framing = framing or Framing(axis=_span(source, box)[0], positions=((0, 0.5),))
                # A photo's move is centred, as it always was; nothing measures where its subject is.
                anchor = (anchor_of(clip, source, box, seen, framing, fps)
                          if follow_faces and not asset.still else moves.MIDDLE)
                framing = replace(framing, motion=motion, anchor=anchor)
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
    cross = ""
    if framing.zoom > 1:
        # Zoomed in, the other axis has room too, and its centre is where it was put.
        other, other_out, name = ("ih", "oh", "y") if framing.axis == "x" else ("iw", "ow", "x")
        cross = rf":{name}=max(0\,min({other}-{other_out}\,{framing.cross:g}*{other}-{other_out}/2))"
    # if(lt(n,f1), c0, if(lt(n,f2), c1, ... ck)), built from the last framing inwards.
    centre = f"{framing.positions[-1][1]:g}"
    for (frame, _), (_, before) in zip(reversed(framing.positions[1:]), reversed(framing.positions[:-1])):
        centre = rf"if(lt(n\,{frame})\,{before:g}\,{centre})"
    # Escaped rather than quoted, the way the rest of the graph is: inside quotes the
    # backslashes would reach the expression parser and it would refuse them.
    offset = rf"max(0\,min({size}-{out}\,{centre}*{size}-{out}/2))"
    return f"crop={width}:{height}:{framing.axis}={offset}{cross}"


def centred_note(framed: Mapping[str, Framing]) -> Optional[str]:
    """Say which clips were told to fill the frame but had no face to follow.

    Args:
        framed: The project's framings, keyed by clip ID.

    Returns:
        One sentence naming them, or None when there are none.
    """
    named = [clip_id for clip_id, framing in framed.items() if framing.centred]
    if not named:
        return None
    return (f"clip {', '.join(named)}: told to fill the frame, but no face was seen to follow, so the crop sits "
            f"in the middle; set fit center_x / center_y to move it onto what matters")


def move_notes(project: Project, assets: Mapping[str, Asset], framed: Mapping[str, Framing]) -> List[str]:
    """Say what a slow move over video costs, clip by clip: a jolt on a short clip, softness past the source.

    A move closes in on the picture by `MOTION_AMOUNT`, so at its closest the
    frame shows that much less of the source. Footage shot at the output size
    then has fewer pixels than the frame and is enlarged to fill it, which
    softens it a little; footage shot larger has the pixels to spare. Said
    for every clip it happens to, because whether a little softness is worth
    the move is the user's call, and nothing here can see it. A move on a
    clip shorter than `MIN_MOVE_SECONDS`, which a split or a trim can leave
    behind, crosses the whole amount too fast to read as a move.

    Args:
        project: The cut, at the size it is rendered at.
        assets: Its files, sized.
        framed: Its framings, keyed by clip ID.

    Returns:
        One sentence per clip it applies to; empty when no video moves.
    """
    notes: List[str] = []
    # Clips softened alike are said in one sentence: a talking cut pushes on many clips of one file.
    softened: Dict[Tuple[int, int, int, int, int, int], List[str]] = {}
    for track in project.video_tracks:
        for clip in sorted(track.clips, key=lambda item: item.timeline_in):
            asset = assets.get(clip.asset_id)
            if asset is None or asset.still or moves.motion_of(clip, False) == "none":
                continue
            seconds = float(clip.timeline_duration)
            if seconds < moves.MIN_MOVE_SECONDS:
                notes.append(f"clip {clip.id}: a {clip.motion} over only {seconds:.2f}s reads as a jolt rather than "
                             f"a move; set motion none on it, or lengthen it past {moves.MIN_MOVE_SECONDS:g}s")
            if not asset.width or not asset.height:
                continue
            framing = framed.get(clip.id)
            width, height = box_of(clip, project, (asset.width, asset.height))
            fitted = min if framing is not None and framing.whole else max
            scale = fitted(width / asset.width, height / asset.height) * (framing.zoom if framing is not None else 1.0)
            closest = scale * (1 + moves.MOTION_AMOUNT)
            if closest > 1.005:
                key = (round(closest * 100), round(scale * 100), asset.width, asset.height, width, height)
                softened.setdefault(key, []).append(clip.id)
    for (closest, framed_at, source_width, source_height, width, height), clip_ids in softened.items():
        named = ", ".join(clip_ids[:8]) + (" …" if len(clip_ids) > 8 else "")
        notes.append(
            f"clip{'s' if len(clip_ids) > 1 else ''} {named}: at the closest point of the move the frame shows "
            f"{width * 100 / closest:.0f}x{height * 100 / closest:.0f} pixels of the {source_width}x{source_height} "
            f"source, enlarged {closest - 100}% to fill {width}x{height}, so it is slightly softer there"
            + (f" ({framed_at - 100}% of that is the framing's own zoom)" if framed_at > 100 else "")
            + "; footage shot larger than the frame would not be"
        )
    return notes
