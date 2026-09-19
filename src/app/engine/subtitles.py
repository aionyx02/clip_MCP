"""Turn subtitle cues into an ASS file that FFmpeg can burn into the picture."""

import os
import unicodedata
from decimal import Decimal
from typing import Iterable, Iterator, List, Mapping, Tuple

from app.models.media import Transcript, TranscriptSegment, TranscriptWord
from app.models.timeline import Project, SubtitleCue, number_cues

# Burned captions sit larger than broadcast subtitles, at about six percent of the shorter side.
FONT_DIVISOR = 16
# Phones cover roughly the bottom eighth of a vertical video with their own controls and captions.
PORTRAIT_BOTTOM_FRACTION = 0.13
LANDSCAPE_BOTTOM_FRACTION = 0.06
SIDE_FRACTION = 0.08
# libass falls back to a font that has the glyphs, but the starting point can be set for unusual systems.
DEFAULT_FONT = os.environ.get("CLIP_MCP_SUBTITLE_FONT", "Arial")
# A caption much longer than this is hard to read at a glance, and one much shorter flickers past.
DEFAULT_MAX_CHARACTERS = 24
DEFAULT_MAX_SECONDS = 6.0
MINIMUM_CUE_SECONDS = 0.05

def format_ass_time(seconds: float) -> str:
    """Format a time the way an ASS event line expects it.

    Args:
        seconds: Time in seconds.

    Returns:
        The time as `H:MM:SS.cc`, rounded to centiseconds.
    """
    hundredths = max(0, round(float(seconds) * 100))
    hours, rest = divmod(hundredths, 360000)
    minutes, rest = divmod(rest, 6000)
    return f"{hours}:{minutes:02d}:{rest // 100:02d}.{rest % 100:02d}"

def escape_ass_text(text: str) -> str:
    """Make a line of text safe to put in an ASS event.

    Braces start an override block in ASS and a raw newline would end the
    event, so both are neutralized.

    Args:
        text: Text to escape.

    Returns:
        The escaped text on one event line.
    """
    cleaned = text.replace("{", "(").replace("}", ")").replace("\\", "/")
    return "\\N".join(part.strip() for part in cleaned.splitlines() if part.strip())

def _display_width(character: str) -> float:
    """Estimate how much of a line one character takes up.

    Args:
        character: The character to measure.

    Returns:
        Its width as a fraction of the font size: a full unit for the wide
        East Asian forms, half a unit for everything else.
    """
    return 1.0 if unicodedata.east_asian_width(character) in ("W", "F") else 0.5

def wrap_caption(text: str, max_units: float) -> List[str]:
    """Break one caption into lines that fit the frame.

    libass only breaks lines at spaces, so Chinese and Japanese captions run
    off both edges of the picture unless they are broken here first.

    Args:
        text: A single line of caption text.
        max_units: Line width as a multiple of the font size.

    Returns:
        The text as one or more lines, none wider than `max_units`.
    """
    lines: List[str] = []
    current, width = "", 0.0
    for character in text:
        size = _display_width(character)
        if current and width + size > max_units:
            space = current.rfind(" ")
            if space > 0:
                lines.append(current[:space])
                current = current[space + 1:]
                width = sum(_display_width(item) for item in current)
            else:
                lines.append(current)
                current, width = "", 0.0
            if character == " ":
                continue
        current += character
        width += size
    if current:
        lines.append(current)
    return lines or [""]

def build_ass(cues: Iterable[SubtitleCue], width: int, height: int, font: str = DEFAULT_FONT) -> str:
    """Render subtitle cues as a complete ASS subtitle file.

    The style is sized and positioned from the output format, so captions
    keep clear of the controls a phone draws over a vertical video.

    Args:
        cues: Cues to write, in timeline order.
        width: Output width in pixels.
        height: Output height in pixels.
        font: Font family name to ask for.

    Returns:
        The ASS file contents.
    """
    font_size = max(12, round(min(width, height) / FONT_DIVISOR))
    bottom = PORTRAIT_BOTTOM_FRACTION if height > width else LANDSCAPE_BOTTOM_FRACTION
    margin_v = round(height * bottom)
    margin_h = round(width * SIDE_FRACTION)
    outline = max(1, round(font_size / 16))

    lines: List[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},{max(1, outline // 2)},2,{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    # libass measures in the ASS resolution, so the usable width is the frame minus both margins.
    max_units = max(4.0, (width - 2 * margin_h) / font_size)
    for cue in cues:
        parts = [part for part in escape_ass_text(cue.text).split("\\N") if part]
        text = "\\N".join(line for part in parts for line in wrap_caption(part, max_units))
        if not text:
            continue
        lines.append(
            f"Dialogue: 0,{format_ass_time(cue.start)},{format_ass_time(cue.end)},Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"

def _pieces(
    segment: TranscriptSegment,
    start: float,
    end: float,
    max_characters: int,
    max_seconds: float,
) -> Iterator[Tuple[float, float, str]]:
    """Split one transcript segment into caption-sized pieces inside a clip.

    Word timings are used when the transcript has them, so a long sentence
    breaks between words rather than mid-word. Pieces are clamped to the
    clip's source range, so a caption never runs past a cut.

    Args:
        segment: Transcript segment to split.
        start: Start of the clip's source range in seconds.
        end: End of the clip's source range in seconds.
        max_characters: Longest caption text before it is broken up.
        max_seconds: Longest caption before it is broken up.

    Yields:
        One `(start, end, text)` per caption, in source-file seconds.
    """
    words = [word for word in segment.words if word.end > start and word.start < end]
    if not words:
        text = segment.text.strip()
        piece_start, piece_end = max(segment.start, start), min(segment.end, end)
        if text and piece_end - piece_start > MINIMUM_CUE_SECONDS:
            yield piece_start, piece_end, text
        return

    chunks: List[List[TranscriptWord]] = []
    current: List[TranscriptWord] = []
    for word in words:
        length = len("".join(item.text for item in [*current, word]).strip())
        span = word.end - (current[0].start if current else word.start)
        if current and (length > max_characters or span > max_seconds):
            chunks.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(current)

    for chunk in chunks:
        text = "".join(word.text for word in chunk).strip()
        piece_start, piece_end = max(chunk[0].start, start), min(chunk[-1].end, end)
        if text and piece_end - piece_start > MINIMUM_CUE_SECONDS:
            yield piece_start, piece_end, text

def timeline_cues(
    project: Project,
    transcripts: Mapping[str, Transcript],
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> List[SubtitleCue]:
    """Map the transcripts of a project's source files onto its edited timeline.

    Only the speech that survived the edit is kept, and each caption is moved
    to where that speech now falls in the result. Captions come from the base
    video track, the sequence itself; video tracks drawn on top of it are
    pictures over that sound, not extra speech.

    Args:
        project: Project whose base video track is captioned.
        transcripts: Transcript of each asset, keyed by asset ID; assets that
            were never transcribed are simply skipped.
        max_characters: Longest caption text before it is broken up.
        max_seconds: Longest caption before it is broken up.

    Returns:
        The captions in timeline order.
    """
    cues: List[SubtitleCue] = []
    track = project.base_video_track
    # Only the base track: an inset drawn over it is a second picture, not a second voice to caption.
    for clip in sorted(track.clips, key=lambda item: item.timeline_in) if track else []:
        transcript = transcripts.get(clip.asset_id)
        if transcript is None:
            continue
        start, end = float(clip.source_range.start), float(clip.source_range.end)
        base, speed = float(clip.timeline_in), float(clip.speed)
        for segment in transcript.segments:
            if segment.end <= start or segment.start >= end:
                continue
            for piece_start, piece_end, text in _pieces(segment, start, end, max_characters, max_seconds):
                cues.append(SubtitleCue(
                    start=Decimal(str(round(base + (piece_start - start) / speed, 3))),
                    end=Decimal(str(round(base + (piece_end - start) / speed, 3))),
                    text=text,
                ))
    # Numbered here, so a single line can be corrected later without resending the rest,
    # and by the same helper `set_subtitles` uses, so the IDs survive being stored.
    return number_cues(sorted(cues, key=lambda cue: cue.start))
