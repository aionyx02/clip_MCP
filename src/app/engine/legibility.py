"""How well the words drawn over a cut stand out from what is behind them.

A title card's words and the words over the footage are drawn white unless a
brand kit gives them its colours, so a card read off a bright frame, or a
headline over a bright sky, can be lost where a dark frame would carry them.
This measures the picture behind the words, the way the render will show it,
against the words' own colour; whether the frame is the wrong one is for
whoever chose it.
"""

from typing import Sequence, Tuple

import numpy as np
from PIL import Image

from app.engine.subtitles import CAPTION_LINE_HEIGHT, text_width

# Provisional: below this, words are taken to be hard to read on what is behind them.
# WCAG 2's contrast ratio for large text (3:1), measured between white and the bright end of
# the background under the words, without the soft shadow or outline drawn round them.
LOW_CONTRAST = 3.0
# Provisional: the bright end of the background is where this share of its pixels stay under,
# so a bright patch behind a few letters counts and a stray highlight does not. Dark words are
# measured against the dark end the same way.
BRIGHT_PERCENTILE = 90


def relative_luminance(pixels: np.ndarray) -> np.ndarray:
    """WCAG 2's relative luminance of sRGB pixels, 0 for black to 1 for white.

    Args:
        pixels: An array of RGB values 0–255, the last axis the channels.

    Returns:
        One luminance per pixel.
    """
    channels = pixels.astype(np.float64) / 255
    linear = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def contrast_ratio(lighter: float, darker: float) -> float:
    """WCAG 2's contrast ratio between two relative luminances: 1 for none, 21 for white on black."""
    return (max(lighter, darker) + 0.05) / (min(lighter, darker) + 0.05)


def row_box(row: dict, width: int) -> Tuple[int, int, int, int]:
    """Where a placed row of words covers the frame, as `(left, top, right, bottom)` in pixels.

    Args:
        row: A row as `cards.placed` or `texts.placed` gives it: `text`,
            `size`, `y`, the centre of the line, and `colour`; and, from `texts.placed`,
            `x` with its `align` and the `box` padding round it. A row
            without `x` is centred, as a card's are.
        width: Frame width in pixels.
    """
    reach = text_width(row["text"]) * row["size"]
    pad = row.get("box", 0)
    x, align = row.get("x", width / 2), row.get("align", "center")
    left = x - pad if align == "left" else x - reach / 2 - pad
    half = row["size"] * CAPTION_LINE_HEIGHT / 2
    return round(left), round(row["y"] - half), round(left + reach + 2 * pad), round(row["y"] + half)


def words_contrast(picture: Image.Image, rows: Sequence[dict], box_opacity: float = 0.0) -> float:
    """Measure words against the picture behind them, in their own colour.

    Args:
        picture: The frame behind the words, at the size and framing they are
            placed in, already treated as the render treats it.
        rows: The rows of words, as `row_box` reads them; a row without a
            `colour` is white.
        box_opacity: How opaque the black box drawn behind a row with a `box`
            is; rows without one stand on the picture itself.

    Returns:
        The lowest contrast ratio, among the words of each colour, between
        that colour and the end of the picture under them nearer to it in
        lightness — the bright end under light words, the dark end under
        dark ones — rounded to one decimal; 21 when no row is inside the
        frame.
    """
    pixels = np.asarray(picture.convert("RGB"))
    height, width = pixels.shape[:2]
    behind: dict = {}
    for row in rows:
        left, top, right, bottom = row_box(row, width)
        part = pixels[max(0, top):min(height, bottom), max(0, left):min(width, right)].reshape(-1, 3)
        if row.get("box"):
            part = part * (1 - box_opacity)
        behind.setdefault(row.get("colour") or "#ffffff", []).append(part)
    worst = 21.0
    for colour, parts in behind.items():
        if not sum(len(part) for part in parts):
            continue
        under = relative_luminance(np.concatenate(parts))
        hexed = {"black": "#000000", "white": "#ffffff"}.get(colour, colour)
        words = float(relative_luminance(np.array([int(hexed[index:index + 2], 16) for index in (1, 3, 5)])))
        end = BRIGHT_PERCENTILE if words >= float(np.median(under)) else 100 - BRIGHT_PERCENTILE
        worst = min(worst, contrast_ratio(words, float(np.percentile(under, end))))
    return round(worst, 1)
