"""Where a picture laid over the sequence is drawn: in a corner of the frame, or in one half of it.

A layout names a place, not a box, and the box is worked out here for the
frame the cut is drawn in, so the same plan lands in the same corner of a
landscape, portrait or square delivery. One place for the arithmetic, because
the render, the editor's player, the framing of each crop and the checks before
a render all need the same answer.
"""

import math
from decimal import Decimal
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from app.engine.subtitles import CAPTION_LINE_HEIGHT, caption_geometry
from app.models.timeline.parts import CaptionStyle, Clip, ClipLayout, Project, Track, TrackType

# How wide a corner picture's short edge is, as a share of the frame's short side. On a
# landscape frame with landscape footage that is the share of the frame's width the user
# decided on (about a quarter, a third, two fifths); on any other shape the picture keeps
# the same size against the short side rather than shrinking with the width. Provisional.
INSET_SHARES = {"small": 0.25, "medium": 0.33, "large": 0.40}
# Provisional: how far a corner picture stands in from the frame's edges, and from the top of
# the captions, as a share of the frame's short side. The same as a logo's: clear of the edge
# a TV or a rounded phone screen crops, not measured against anything.
INSET_MARGIN = 0.04
# Provisional: the line round a corner picture, and between the halves of a split screen, as a
# share of the frame's short side and never under 2 pixels: 2 on a 1080 frame. Enough to keep
# the picture's edge readable over a busy background without reading as a frame round it.
INSET_EDGE = 0.002
INSET_EDGE_COLOUR = "white"
INSET_EDGE_OPACITY = 0.85
# The track the sequence is drawn on a second time, in the other half of a split screen. Not a
# track of the project: it is worked out from the split each time the cut is drawn.
SPLIT_TRACK_ID = "split~"
OTHER_HALF = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
# A left or right half on a portrait frame is laid as the top or bottom one.
TURNED = {"left": "top", "right": "bottom"}

Box = Tuple[int, int, int, int]

def _even(value: float) -> int:
    """Round to an even number of pixels, never under 2: the encoder's chroma needs even sizes."""
    return max(2, int(round(value / 2)) * 2)

def place_in(layout: ClipLayout, width: int, height: int) -> str:
    """Say where a layout is drawn in a frame of one shape.

    Side by side on a portrait frame leaves two slivers each a quarter as wide
    as they are tall, too narrow for a face or a product to be seen in, so a
    left half is laid as the top one and a right half as the bottom one. A
    corner is the same corner on every shape.

    Args:
        layout: The layout.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        The place it is drawn at.
    """
    if height > width and layout.place in TURNED:
        return TURNED[layout.place]
    return layout.place

def captions_top(width: int, height: int, style: CaptionStyle) -> float:
    """Where two lines of captions reach up to, in pixels from the top: what a corner picture keeps above."""
    geometry = caption_geometry(width, height, style)
    return height - geometry.margin_v - 2 * geometry.font_size * CAPTION_LINE_HEIGHT

def inset_box(
    layout: ClipLayout,
    width: int,
    height: int,
    style: CaptionStyle,
    source: Optional[Tuple[int, int]] = None,
) -> Box:
    """Work out the box a layout is drawn in, in whole even pixels.

    A half is half the frame. A corner keeps its picture's own shape — a
    landscape shot in a landscape box — with its short edge the size's share of
    the frame's short side, `INSET_MARGIN` in from the two edges of its corner;
    a bottom corner sits that far above where two lines of captions reach, so
    the words are never drawn over it. A picture too big for the room between
    the margins is brought down, its shape kept.

    Args:
        layout: The layout.
        width: Frame width in pixels.
        height: Frame height in pixels.
        style: How the cut's captions are drawn, for where they reach up to.
        source: The picture's `(width, height)`; None gives the box the
            frame's own shape.

    Returns:
        `(x, y, width, height)`.
    """
    place = place_in(layout, width, height)
    if place in ("left", "right"):
        half = _even(width / 2)
        return (0 if place == "left" else width - half), 0, half, height
    if place in ("top", "bottom"):
        half = _even(height / 2)
        return 0, (0 if place == "top" else height - half), width, half
    short = min(width, height)
    margin = round(short * INSET_MARGIN)
    shape = source[0] / source[1] if source and source[0] and source[1] else width / height
    edge = short * INSET_SHARES[layout.size]
    box_width, box_height = (edge * shape, edge) if shape >= 1 else (edge, edge / shape)
    floor = captions_top(width, height, style) - margin
    room = min(1.0, (width - 2 * margin) / box_width, (floor - margin) / box_height)
    if room < 1:
        # Brought down to the room, and rounded down, so it stays inside it.
        box_width, box_height = max(2, int(box_width * room) // 2 * 2), max(2, int(box_height * room) // 2 * 2)
    else:
        box_width, box_height = _even(box_width), _even(box_height)
    x = margin if place.endswith("left") else width - margin - box_width
    y = margin if place.startswith("top") else int(math.floor(floor)) - box_height
    return x, max(0, y), box_width, box_height

def edges(layout: ClipLayout, box: Box, width: int, height: int) -> List[Box]:
    """Say where the line round a laid-over picture is drawn, as filled rectangles.

    All four sides of a corner picture; only the side between the halves of a
    split screen, since its other sides are the frame's own edges.

    Args:
        layout: The layout.
        box: The box it is drawn in, from `inset_box`.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        `(x, y, width, height)` of each rectangle, inside the box.
    """
    x, y, w, h = box
    line = _even(min(width, height) * INSET_EDGE)
    place = place_in(layout, width, height)
    if place == "left":
        return [(x + w - line, y, line, h)]
    if place == "right":
        return [(x, y, line, h)]
    if place == "top":
        return [(x, y + h - line, w, line)]
    if place == "bottom":
        return [(x, y, w, line)]
    return [(x, y, w, line), (x, y + h - line, w, line), (x, y, line, h), (x + w - line, y, line, h)]

def split_partners(project: Project) -> List[Clip]:
    """Lay the sequence into the other half of every split screen.

    A clip in one half leaves the other half to the sequence, cropped on its
    own subject rather than showing whichever half of its full-frame picture
    falls there. So for each stretch of the sequence under a split, a copy of
    that clip is drawn in the other half: silent, since its sound is heard from
    the sequence already, with no transition or fade, since the split begins and
    ends on its own clip. Worked out each time the cut is drawn, so a trim of
    the sequence is followed without anybody saying so. A title card is left
    out: it is not footage to crop, and a split over one is refused before
    a render.

    Args:
        project: The cut.

    Returns:
        The copies, in timeline order, each named after the split clip and the
        sequence clip it copies.
    """
    base = project.base_video_track
    if base is None:
        return []
    fps = Fraction(project.fps_num, project.fps_den)
    partners: List[Clip] = []
    for track in project.video_tracks[1:]:
        for split in track.clips:
            if split.layout is None or not split.layout.is_half:
                continue
            for under in sorted(base.clips, key=lambda clip: clip.timeline_in):
                if under.card is not None:
                    continue
                opened = max(split.timeline_in, under.timeline_in)
                closed = min(split.timeline_out, under.timeline_out)
                if Fraction(closed - opened) * fps < 1:
                    continue
                speed = Decimal(str(under.speed))
                start = under.source_range.start + (opened - under.timeline_in) * speed
                end = min(under.source_range.end, start + (closed - opened) * speed)
                partners.append(under.model_copy(update={
                    "id": f"{split.id}~{under.id}",
                    "source_range": under.source_range.model_copy(update={"start": start, "end": end}),
                    "timeline_in": opened,
                    "volume": 0.0,
                    "audio_lead": Decimal(0), "audio_lag": Decimal(0),
                    "audio_fade_in": Decimal(0), "audio_fade_out": Decimal(0),
                    "video_fade_in": Decimal(0), "video_fade_out": Decimal(0),
                    "transition_in": None,
                    "texts": [],
                    "layout": ClipLayout(place=OTHER_HALF[split.layout.place]),
                }, deep=True))
    return sorted(partners, key=lambda clip: clip.timeline_in)

def drawn_overlays(project: Project) -> List[Track]:
    """The video tracks drawn over the sequence, in the order they are drawn.

    The sequence laid into the other half of each split screen comes first,
    then the project's own tracks above the base.

    Args:
        project: The cut.

    Returns:
        The tracks.
    """
    partners = split_partners(project)
    split = [Track(id=SPLIT_TRACK_ID, track_type=TrackType.VIDEO, clips=partners)] if partners else []
    return split + project.video_tracks[1:]

def covered(first: Box, second: Box) -> float:
    """How much of `first` lies under `second`, as a share of `first`'s area: 0 when they do not meet."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    if right <= left or bottom <= top or first[2] <= 0 or first[3] <= 0:
        return 0.0
    return (right - left) * (bottom - top) / (first[2] * first[3])

def spans_of(moments: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Join stretches that touch or overlap into the fewest that cover the same time."""
    joined: List[List[float]] = []
    for start, end in sorted(moments):
        if joined and start <= joined[-1][1] + 1e-3:
            joined[-1][1] = max(joined[-1][1], end)
        else:
            joined.append([start, end])
    return [(start, end) for start, end in joined]
