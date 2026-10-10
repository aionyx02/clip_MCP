"""Words over footage: where each kind sits, and the arithmetic the render and the editor both follow.

A text layer is pinned to seconds of its clip's file, so it shows for as
much of that stretch as the cut plays, wherever the cut has moved it. Each
kind has a place of its own — a name low on the left, a place high on the
left, a headline in the middle — kept clear of the captions below and of
what a phone's app draws over a vertical video, the way captions are.
Everything is placed here once, in frame pixels: the render writes it into
the captions' subtitle file and the editor's player paints the same rows.
"""

from decimal import Decimal
from typing import List, Optional, Tuple

from app.engine.subtitles import (
    CAPTION_LINE_HEIGHT, DEFAULT_FONT, caption_geometry, escape_ass_text, format_ass_time, wrap_caption,
)
from app.models.timeline import CaptionStyle, Clip, Project, TextLayer

# Provisional (roadmap §13): each kind's size, as a share of the frame's short side.
SIZES = {"name": 0.05, "place": 0.042, "headline": 0.085, "free": 0.06}
# The second line, against the first.
SECOND_SCALE = 0.7
# Provisional (roadmap §13): how much of the top a phone's app covers on a vertical video,
# and how much a player's title bar might on a landscape one.
TOP_SAFE_PORTRAIT = 0.12
TOP_SAFE_LANDSCAPE = 0.06
# How much of the frame's width a line may take before it wraps.
LINE_WIDTH = 0.8
# How long the words take to come and go.
FADE_SECONDS = 0.2
# The name bar's box: how far it reaches past the words, against their size, and how see-through it is.
BOX_PAD = 0.3
BOX_OPACITY = 0.62


def shown(clip: Clip, text: TextLayer) -> Optional[Tuple[float, float]]:
    """When a text layer is on screen in the cut: the part of its stretch the clip still plays.

    Args:
        clip: The clip it is on.
        text: The text.

    Returns:
        `(start, end)` in seconds of the cut, or None when the clip no longer
        plays any of it.
    """
    start = max(text.start, clip.source_range.start)
    end = min(text.end, clip.source_range.end)
    if end <= start:
        return None
    speed = Decimal(str(clip.speed))
    return (float(clip.timeline_in + (start - clip.source_range.start) / speed),
            float(clip.timeline_in + (end - clip.source_range.start) / speed))


def _wrapped(text: str, size: int, width: int) -> List[str]:
    return wrap_caption(" ".join(text.split()), width * LINE_WIDTH / size)


def placed(text: TextLayer, width: int, height: int, style: CaptionStyle) -> List[dict]:
    """Place a text layer's lines in the frame.

    Args:
        text: The text.
        width: Frame width in pixels.
        height: Frame height in pixels.
        style: The captions' style, whose margins it keeps clear of.

    Returns:
        Rows, each `{text, size, x, y, align, bold, box, outline}`: `x` the
        left edge for `align` `left` or the centre for `center`, `y` the
        row's centre, `box` the padding of the box behind it or 0, and
        `outline` its outline width.
    """
    geometry = caption_geometry(width, height, style)
    short = min(width, height)
    size = round(short * SIZES[text.style])
    small = max(8, round(size * SECOND_SCALE))
    lines = [(line, size, True) for line in _wrapped(text.text, size, width)]
    if text.second:
        lines += [(line, small, False) for line in _wrapped(text.second, small, width)]
    steps = [row_size * CAPTION_LINE_HEIGHT for _, row_size, _ in lines]
    total = sum(steps)
    top_safe = height * (TOP_SAFE_PORTRAIT if height > width else TOP_SAFE_LANDSCAPE)
    box = round(size * BOX_PAD) if text.style == "name" else 0
    outline = max(1, round(size / 14)) if text.style in ("headline", "free") else 0
    if text.style == "name":
        # Above where two lines of captions sit, with a little air: a name bar under a caption
        # would be covered by it, and the captions are what the viewer is reading.
        captions_top = height - geometry.margin_v - 2 * geometry.font_size * CAPTION_LINE_HEIGHT
        top, x, align = captions_top - size * 0.6 - total - box, geometry.margin_h + box, "left"
    elif text.style == "place":
        top, x, align = top_safe, geometry.margin_h, "left"
    elif text.style == "headline":
        top, x, align = height / 2 - total / 2, width / 2, "center"
    else:
        across = width * (text.x if text.x is not None else 0.5)
        down = height * (text.y if text.y is not None else 0.5)
        top = min(max(down - total / 2, 0), height - total)
        x, align = across, "center"
    rows = []
    for (line, row_size, bold), step in zip(lines, steps):
        rows.append({"text": line, "size": row_size, "x": round(x), "y": round(top + step / 2), "align": align,
                     "bold": bold, "box": box, "outline": outline})
        top += step
    return rows


def _layers(project: Project) -> List[Tuple[Clip, TextLayer, Tuple[float, float]]]:
    """Every text layer the cut shows, with its clip and when, in the order they appear."""
    found = []
    for track in project.video_tracks:
        for clip in track.clips:
            for text in clip.texts:
                when = shown(clip, text)
                if when is not None:
                    found.append((clip, text, when))
    return sorted(found, key=lambda item: item[2][0])


# libass draws a box behind each line of an event in this style: the name bar.
BOX_STYLE = "TextBox"


def ass_styles(project: Project, style: Optional[CaptionStyle] = None) -> List[str]:
    """The extra subtitle style the text layers need: the name bar's box, see-through black."""
    font = (style.font if style else None) or DEFAULT_FONT
    alpha = f"{round(255 * (1 - BOX_OPACITY)):02X}"
    return [f"Style: {BOX_STYLE},{font},20,&H00FFFFFF,&H00FFFFFF,&H{alpha}000000,&H{alpha}000000,"
            "-1,0,0,0,100,100,0,0,3,0,0,4,0,0,0,1"]


def ass_events(project: Project, style: Optional[CaptionStyle] = None) -> List[str]:
    """Write a cut's text layers as subtitle events, each line pinned to its place.

    Args:
        project: The cut.
        style: The captions' style: their font, and the margins kept clear.

    Returns:
        `Dialogue` lines for the `[Events]` section.
    """
    style = style or CaptionStyle()
    font = style.font or DEFAULT_FONT
    fade = round(FADE_SECONDS * 1000)
    events = []
    for _, text, (start, end) in _layers(project):
        for row in placed(text, project.width, project.height, style):
            anchor = 4 if row["align"] == "left" else 5
            look = (f"\\bord{row['box']}" if row["box"] else
                    f"\\bord{row['outline']}\\shad{max(1, row['size'] // 24)}\\3c&H00000000&\\4c&H80000000&")
            events.append(
                f"Dialogue: 2,{format_ass_time(start)},{format_ass_time(end)},"
                f"{BOX_STYLE if row['box'] else 'Default'},,0,0,0,,"
                f"{{\\an{anchor}\\pos({row['x']},{row['y']})\\fn{font}\\fs{row['size']}\\b{int(row['bold'])}"
                f"\\c&H00FFFFFF&{look}\\fad({fade},{fade})}}{escape_ass_text(row['text'])}"
            )
    return events


def drawn(project: Project, style: Optional[CaptionStyle] = None) -> dict:
    """Describe a cut's text layers for the editor's player to paint, as `placed` places them.

    Returns:
        `events`, each with `start`, `end`, the clip and text IDs, and its
        `rows`; and the `fade` and the box's `opacity`.
    """
    style = style or CaptionStyle()
    return {
        "fade": FADE_SECONDS, "box_opacity": BOX_OPACITY,
        "events": [{"start": start, "end": end, "clip_id": clip.id, "text_id": text.id,
                    "rows": placed(text, project.width, project.height, style)}
                   for clip, text, (start, end) in _layers(project)],
    }
