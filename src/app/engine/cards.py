"""Title cards: a frame behind, a title over it, and the arithmetic both the render and the editor follow.

A card's clip reads one frame of a file — the shot it leads into, or a
photo — and holds it for as long as the card lasts. How that frame is
treated is the card's background; what is written over it is placed here,
in frame pixels, once: the render writes it into the same subtitle file as
the captions, and the editor's player paints the same lines at the same
places, so a title cannot wrap differently in one than in the other.
"""

from typing import List, Optional

from PIL import Image, ImageEnhance, ImageFilter

from app.engine.subtitles import CAPTION_LINE_HEIGHT, DEFAULT_FONT, escape_ass_text, format_ass_time, wrap_caption
from app.models.timeline import CaptionStyle, Clip, Project, TitleCard

# Provisional (roadmap §13): how big a card's title is, as a share of the frame's short side.
TITLE_SHARE = 0.1
# How big the line under the title is, against the title.
SUBTITLE_SCALE = 0.5
# How much of the frame's width a line of the title may take before it wraps.
TITLE_WIDTH = 0.8
# Provisional (roadmap §13): how long the words take to fade in, and out again.
TEXT_FADE_SECONDS = 0.3
# How far the blur reaches, as a share of the picture's short side: far enough that the
# shot behind is colour and shape, not something to read.
BLUR_SHARE = 1 / 25
# How much darker the frame behind is made, so white words read on it.
BLUR_DIM = -0.25
PICTURE_DIM = -0.15


def backdrop_filter(card: TitleCard, width: int, height: int) -> str:
    """Write the filters that turn the frame behind a card into its background.

    Args:
        card: The card.
        width: Width of the picture they run on.
        height: Height of the picture they run on.

    Returns:
        The filters, without a leading comma.
    """
    if card.background == "colour":
        return f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{card.colour[1:]}@1:t=fill"
    if card.background == "picture":
        return f"eq=brightness={PICTURE_DIM}"
    # boxblur's radius cannot pass half the picture's short side.
    radius = max(2, min(round(min(width, height) * BLUR_SHARE), min(width, height) // 2 - 1))
    return f"boxblur={radius}:2,eq=brightness={BLUR_DIM}"


def backdrop_image(frame: Image.Image, card: TitleCard) -> Image.Image:
    """Treat a still of the frame behind a card the way `backdrop_filter` treats the render's.

    For pictures of the cut — a storyboard tile, a cover — that show the card's
    background; its words are not drawn on them.

    Args:
        frame: The frame, already fitted to the delivery shape.
        card: The card.

    Returns:
        The card's background.
    """
    if card.background == "colour":
        return Image.new("RGB", frame.size, card.colour)
    if card.background == "picture":
        return ImageEnhance.Brightness(frame).enhance(1 + PICTURE_DIM)
    blurred = frame.filter(ImageFilter.GaussianBlur(max(2, round(min(frame.size) * BLUR_SHARE))))
    return ImageEnhance.Brightness(blurred).enhance(1 + BLUR_DIM)

def _lines(text: str, size: int, width: int) -> List[str]:
    return wrap_caption(" ".join(text.split()), width * TITLE_WIDTH / size)


def placed(clip: Clip, width: int, height: int) -> dict:
    """Place a card's words in the frame.

    The title is centred, the line under it smaller; the two together sit
    in the middle of the frame, however many lines the title wraps to.

    Args:
        clip: The card's clip.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        `start` and `end` on the timeline, `fade` in seconds, and `rows`:
        each `{text, size, y}`, `y` the centre of the line in frame pixels.
    """
    card = clip.card
    size = round(min(width, height) * TITLE_SHARE)
    small = max(8, round(size * SUBTITLE_SCALE))
    rows = [(line, size) for line in _lines(card.title, size, width)]
    if card.subtitle:
        rows += [(line, small) for line in _lines(card.subtitle, small, width)]
    # A little air between the title and the line under it, on top of the line spacing.
    gap = round(small * 0.5) if card.subtitle else 0
    total = sum(row_size * CAPTION_LINE_HEIGHT for _, row_size in rows) + gap
    y = height / 2 - total / 2
    placed_rows = []
    for index, (text, row_size) in enumerate(rows):
        if index and row_size != rows[index - 1][1]:
            y += gap
        step = row_size * CAPTION_LINE_HEIGHT
        placed_rows.append({"text": text, "size": row_size, "y": round(y + step / 2)})
        y += step
    return {
        "start": float(clip.timeline_in), "end": float(clip.timeline_out), "fade": TEXT_FADE_SECONDS,
        "rows": placed_rows,
    }


def cards_of(project: Project) -> List[Clip]:
    """The title cards of a cut, in the order they play."""
    base = project.base_video_track
    return sorted((clip for clip in (base.clips if base else []) if clip.card is not None),
                  key=lambda clip: clip.timeline_in)


def ass_events(project: Project, style: Optional[CaptionStyle] = None) -> List[str]:
    """Write a cut's title cards as subtitle events, in the captions' font.

    Each line is its own event, pinned to its place: libass would otherwise
    lay the lines out by its own rules, and the editor could not follow them.

    Args:
        project: The cut.
        style: The captions' style, whose font the cards share.

    Returns:
        `Dialogue` lines for the `[Events]` section; none for a cut without cards.
    """
    font = (style.font if style else None) or DEFAULT_FONT
    fade = round(TEXT_FADE_SECONDS * 1000)
    events = []
    for clip in cards_of(project):
        words = placed(clip, project.width, project.height)
        for row in words["rows"]:
            shadow = max(1, row["size"] // 24)
            events.append(
                f"Dialogue: 1,{format_ass_time(words['start'])},{format_ass_time(words['end'])},Default,,0,0,0,,"
                f"{{\\an5\\pos({project.width // 2},{row['y']})\\fn{font}\\fs{row['size']}\\b1"
                f"\\c&H00FFFFFF&\\bord0\\shad{shadow}\\4c&H80000000&\\fad({fade},{fade})}}"
                f"{escape_ass_text(row['text'])}"
            )
    return events


def drawn(project: Project) -> List[dict]:
    """Describe a cut's title cards for the editor's player to paint, as `placed` places them."""
    return [placed(clip, project.width, project.height) for clip in cards_of(project)]
