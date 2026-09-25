"""Turn subtitle cues into an ASS file that FFmpeg can burn into the picture."""

import os
import unicodedata
from decimal import Decimal
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from app.engine.semantic import share_covered, speaker_at, was_audible
from app.models.media import Span, SpeakerTurn, Transcript, TranscriptSegment, TranscriptWord
from app.models.timeline import (
    CaptionStyle,
    Clip,
    CueOrder,
    CueWord,
    PlacedCue,
    Project,
    SpeakerMark,
    SubtitleCue,
    TrackType,
    cue_order,
    number_cues,
)

# Where captions sit when the style names no platform to keep clear of. Phones cover
# roughly the bottom eighth of a vertical video with their own controls and captions; a
# landscape one only has a player's control bar. The size and the side margins moved onto
# `CaptionStyle`, because a platform changes those and does not change this.
PORTRAIT_BOTTOM_FRACTION = 0.13
LANDSCAPE_BOTTOM_FRACTION = 0.06
# libass falls back to a font that has the glyphs, but the starting point can be set for unusual systems.
DEFAULT_FONT = os.environ.get("CLIP_MCP_SUBTITLE_FONT", "Arial")
# A caption much longer than this is hard to read at a glance, and one much shorter flickers past.
DEFAULT_MAX_CHARACTERS = 24
DEFAULT_MAX_SECONDS = 6.0
MINIMUM_CUE_SECONDS = 0.05
# The same floor in Decimal, for placing stored captions on a timeline that counts in Decimal.
MINIMUM_CUE = Decimal(str(MINIMUM_CUE_SECONDS))
# Times on the timeline are kept to milliseconds; Decimal division does not stop there by itself.
MILLISECOND = Decimal('0.001')
# ASS writes colours as &HBBGGRR, backwards from everywhere else. White, yellow, cyan and
# magenta: four that stay apart from each other over any picture, and over each other's
# outlines. A fifth speaker starts round again, which is better than inventing a colour
# nobody can tell from the first four.
SPEAKER_COLOURS = ("&H00FFFFFF", "&H0000FFFF", "&H00FFFF00", "&H00FF00FF")
# What a word looks like before it is said, when the caption lights up word by word.
UNSPOKEN_COLOUR = "&H00909090"

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

def captioned_clips(project: Project) -> List[Clip]:
    """Collect the clips whose sound the viewer hears, in timeline order.

    The base video track carries the sequence and the audio tracks carry any
    narration; video tracks above the base are pictures drawn over that sound,
    so they bring no words of their own. A clip turned all the way down brings
    none either.

    Args:
        project: Project to read.

    Returns:
        The clips that can put words on screen.
    """
    base = project.base_video_track
    tracks = ([base] if base else []) + [
        track for track in project.tracks if track.track_type == TrackType.AUDIO
    ]
    audible = [clip for track in tracks for clip in track.clips if clip.volume != 0]
    return sorted(audible, key=lambda clip: clip.timeline_in)

def _at(timeline_in: Decimal, into_clip: Decimal, speed: Decimal) -> Decimal:
    """Work out when a moment inside a clip falls on the timeline.

    Args:
        timeline_in: Where the clip starts on the timeline.
        into_clip: How far into the clip's source the moment is.
        speed: The clip's playback speed.

    Returns:
        The time on the timeline, to the millisecond the rest of the timeline
        is kept to. Dividing by a speed other than 1.0 otherwise leaves a
        Decimal carrying every digit it can hold.
    """
    return (timeline_in + into_clip / speed).quantize(MILLISECOND)

def place_cues(project: Project, cues: Sequence[SubtitleCue]) -> List[PlacedCue]:
    """Work out where each stored caption falls in the cut as it stands.

    Captions are anchored to the source file and second the words were spoken,
    so this is where that becomes a time on screen. A caption whose words were
    cut out is placed nowhere and simply does not appear; one whose words were
    split across two clips is placed twice, because they are said twice. Each
    placement is clipped to the window it lands in, so a line never runs past
    the cut that ends it.

    Args:
        project: Project whose timeline the captions are placed on.
        cues: The stored captions. A sequence rather than an iterable on
            purpose: they are read once per clip, and a generator would be
            spent on the first one and leave every later clip captionless.

    Returns:
        The placements in timeline order.
    """
    by_asset: Dict[str, List[SubtitleCue]] = {}
    for cue in cues:
        by_asset.setdefault(cue.asset_id, []).append(cue)

    placed: List[PlacedCue] = []
    for clip in captioned_clips(project):
        speed = Decimal(str(clip.speed))
        window_start, window_end = clip.source_range.start, clip.source_range.end
        for cue in by_asset.get(clip.asset_id, ()):
            first = max(cue.source_start, window_start)
            last = min(cue.source_end, window_end)
            if last - first <= MINIMUM_CUE:
                continue
            placed.append(PlacedCue(
                cue_id=cue.id,
                start=_at(clip.timeline_in, first - window_start, speed),
                end=_at(clip.timeline_in, last - window_start, speed),
                text=cue.text,
                secondary=cue.secondary,
                speaker=cue.speaker,
                # Only the words still on screen: a caption cut in half lights up the
                # half that survived, and the rest belongs to the other placement.
                words=[
                    CueWord(
                        start=_at(clip.timeline_in, max(word.start, first) - window_start, speed),
                        end=_at(clip.timeline_in, min(word.end, last) - window_start, speed),
                        text=word.text,
                    )
                    for word in cue.words
                    if word.end > first and word.start < last
                ],
            ))
    return sorted(placed, key=lambda item: (item.start, item.end))

def escape_ass_inline(text: str) -> str:
    """Make one word safe to drop into an ASS event without joining it to anything.

    `escape_ass_text` also strips and rejoins lines, which is right for a whole
    caption and wrong for a word: the spacing between words is what a line is
    made of.

    Args:
        text: Text to escape.

    Returns:
        The escaped text, spacing intact.
    """
    return text.replace("{", "(").replace("}", ")").replace("\\", "/").replace("\n", " ")

def _speaker_colours(cues: Sequence[PlacedCue]) -> Dict[str, str]:
    """Give each voice a colour, the same one every time.

    Args:
        cues: The placed captions.

    Returns:
        An ASS colour per speaker label. Assigned in label order rather than in
        the order they happen to speak, so the same footage always comes out
        the same way and a re-render does not swap two people's colours.
    """
    labels = sorted({cue.speaker for cue in cues if cue.speaker})
    return {label: SPEAKER_COLOURS[index % len(SPEAKER_COLOURS)] for index, label in enumerate(labels)}

def _karaoke_text(cue: PlacedCue, max_units: float) -> str:
    """Write a caption as word-by-word timings inside one event.

    Wrapped here rather than by `wrap_caption`, which measures plain text: the
    override tags between the words are not on screen and must not count
    towards the width of a line.

    Args:
        cue: The placed caption, with word timings in timeline seconds.
        max_units: Line width as a multiple of the font size.

    Returns:
        The event text, with a `\\k` per word and the gaps between them.
    """
    parts: List[str] = []
    width, cursor = 0.0, cue.start
    for word in cue.words:
        text = escape_ass_inline(word.text)
        if not text.strip():
            continue
        size = sum(_display_width(character) for character in text)
        if parts and width + size > max_units:
            parts.append("\\N")
            width = 0.0
        # The silence before a word is timed too, or every word would light up early by
        # however long the pause before it ran.
        gap = round(float(word.start - cursor) * 100)
        if gap > 0:
            parts.append(f"{{\\k{gap}}}")
        parts.append(f"{{\\k{max(1, round(float(word.end - word.start) * 100))}}}{text}")
        cursor = word.end
        width += size
    return "".join(parts)

def _wrapped(text: str, max_units: float) -> str:
    """Escape a caption and break it into lines that fit the frame.

    Args:
        text: The caption text.
        max_units: Line width as a multiple of the font size.

    Returns:
        The lines joined the way an ASS event wants them, empty when the text
        holds nothing to show.
    """
    parts = [part for part in escape_ass_text(text).split("\\N") if part]
    return "\\N".join(line for part in parts for line in wrap_caption(part, max_units))

def build_ass(
    cues: Sequence[PlacedCue],
    width: int,
    height: int,
    style: Optional[CaptionStyle] = None,
) -> str:
    """Render subtitle cues as a complete ASS subtitle file.

    The style is sized and positioned from the output format, so captions keep
    clear of the controls a phone draws over a vertical video, and from the
    project's caption style, which says which platform's furniture to stay out
    of the way of.

    Args:
        cues: Placed cues to write, in timeline order.
        width: Output width in pixels.
        height: Output height in pixels.
        style: How to draw them; the plain style when not given.

    Returns:
        The ASS file contents.
    """
    style = style or CaptionStyle()
    font = style.font or DEFAULT_FONT
    font_size = max(12, round(min(width, height) * style.size_fraction))
    bottom = style.bottom_fraction
    if bottom is None:
        bottom = PORTRAIT_BOTTOM_FRACTION if height > width else LANDSCAPE_BOTTOM_FRACTION
    margin_v = round(height * bottom)
    margin_h = round(width * style.side_fraction)
    outline = max(1, round(font_size * style.outline_fraction))
    # With karaoke the secondary colour is what a word looks like before it is said, so it
    # has to differ from the primary or nothing appears to happen. Without it the two are
    # the same, which is what a caption that simply sits there wants.
    waiting = UNSPOKEN_COLOUR if style.karaoke else "&H00FFFFFF"
    colours = _speaker_colours(cues) if style.speaker_mark in (SpeakerMark.COLOUR, SpeakerMark.BOTH) else {}
    named = style.speaker_mark in (SpeakerMark.NAME, SpeakerMark.BOTH)

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
        f"Style: Default,{font},{font_size},&H00FFFFFF,{waiting},&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},{max(1, outline // 2)},2,{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    # libass measures in the ASS resolution, so the usable width is the frame minus both margins.
    max_units = max(4.0, (width - 2 * margin_h) / font_size)
    for cue in cues:
        if style.karaoke and cue.words:
            text = _karaoke_text(cue, max_units)
        else:
            text = _wrapped(cue.text, max_units)
        if not text:
            continue
        if named and cue.speaker:
            # The label when there is no name for it: a caption reading `S2` is still
            # better than one putting the words in the wrong person's mouth.
            who = style.speaker_names.get(cue.speaker, cue.speaker)
            text = f"{escape_ass_inline(who)}：{text}"
        if cue.secondary:
            # Measured against its own size: a smaller line fits more before it wraps.
            second = _wrapped(cue.secondary, max_units / style.secondary_scale)
            if second:
                text = f"{text}\\N{{\\fs{max(8, round(font_size * style.secondary_scale))}}}{second}"
        colour = colours.get(cue.speaker or "")
        if colour:
            text = f"{{\\c{colour}}}{text}"
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
        One `(start, end, text, words)` per caption, in source-file seconds.
        The words are what lets a caption light up as it is said; a segment
        transcribed without them yields none, and such a caption lights up
        whole.
    """
    words = [word for word in segment.words if word.end > start and word.start < end]
    if not words:
        text = segment.text.strip()
        piece_start, piece_end = max(segment.start, start), min(segment.end, end)
        if text and piece_end - piece_start > MINIMUM_CUE_SECONDS:
            yield piece_start, piece_end, text, []
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
            timed = [
                CueWord(
                    start=Decimal(str(round(max(word.start, piece_start), 3))),
                    end=Decimal(str(round(min(word.end, piece_end), 3))),
                    text=word.text,
                )
                for word in chunk
                if min(word.end, piece_end) > max(word.start, piece_start)
            ]
            yield piece_start, piece_end, text, timed

def timeline_cues(
    project: Project,
    transcripts: Mapping[str, Transcript],
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    silences: Optional[Mapping[str, Sequence[Span]]] = None,
    speakers: Optional[Mapping[str, Sequence[SpeakerTurn]]] = None,
) -> List[SubtitleCue]:
    """Propose captions for the speech that survived the edit.

    Only speech the viewer can hear is captioned: the base video track carries
    the sequence, the audio tracks carry any narration recorded separately, and
    video tracks drawn on top of the base are pictures over that sound rather
    than extra voices. Anything turned all the way down is left out, and so is
    anything never transcribed — which is what keeps a music bed out of the
    captions.

    The cut decides which words are captioned; it does not decide where the
    captions sit. Each one is anchored to the file and the second the words
    were spoken, so rearranging the cut afterwards moves them with the footage
    instead of stranding them.

    Args:
        project: Project whose sound is captioned.
        transcripts: Transcript of each asset, keyed by asset ID; assets that
            were never transcribed are simply skipped.
        max_characters: Longest caption text before it is broken up.
        max_seconds: Longest caption before it is broken up.
        silences: Measured silences per asset. Given them, a sentence the
            transcriber wrote over a stretch that was silent is left out
            rather than captioned: see `was_audible`.
        speakers: The speaker split's turns per asset. Given them, each caption
            records whose stretch it is when it is clearly one person's — the
            same eighty percent rule the semantic layer uses, so a caption and
            the clip it sits in cannot credit different people. A label is
            carried whether or not the style draws it: whether to show it is a
            decision for later, and losing it here would mean transcribing
            again to get it back.

    Returns:
        The captions, ordered by the footage they come from.
    """
    measured = silences or {}
    turns = speakers or {}
    cues: List[SubtitleCue] = []
    seen: set[CueOrder] = set()
    for clip in captioned_clips(project):
        transcript = transcripts.get(clip.asset_id)
        if transcript is None:
            continue
        start, end = float(clip.source_range.start), float(clip.source_range.end)
        quiet = measured.get(clip.asset_id, ())
        for segment in transcript.segments:
            if segment.end <= start or segment.start >= end:
                continue
            # A sentence the recogniser invented over an empty shot is not a caption.
            if quiet and not was_audible(share_covered(segment.start, segment.end, quiet), True):
                continue
            for piece_start, piece_end, text, timed in _pieces(
                segment, start, end, max_characters, max_seconds,
            ):
                cue = SubtitleCue(
                    asset_id=clip.asset_id,
                    source_start=Decimal(str(round(piece_start, 3))),
                    source_end=Decimal(str(round(piece_end, 3))),
                    text=text,
                    speaker=speaker_at(turns.get(clip.asset_id, ()), piece_start, piece_end),
                    words=timed,
                )
                # The same words can survive in two places — a clip split in two, or used
                # twice — and that is one caption placed twice, not two to proofread. The
                # key is the one the store orders by, so the two cannot disagree.
                if cue_order(cue) in seen:
                    continue
                seen.add(cue_order(cue))
                cues.append(cue)
    # Numbered here, so a single line can be corrected later without resending the rest,
    # and by the same helper `set_subtitles` uses, so the IDs survive being stored.
    return number_cues(sorted(cues, key=cue_order))
