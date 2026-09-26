"""Turn subtitle cues into an ASS file that FFmpeg can burn into the picture."""

import os
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

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
# How tall a line of captions stands, as a multiple of the font size: the glyphs plus
# the gap libass leaves between lines. An estimate of the renderer's own spacing rather
# than a taste, used only to tell whether a block of lines fits in the frame.
CAPTION_LINE_HEIGHT = 1.25
# Where a long caption is broken into lines: how much a pause is worth against lines of
# uneven length, and how long a pause earns the whole of it. Half a second is the gap
# between two phrases; two lines, one three times the length of the other, cost as much
# as that pause is worth. Provisional.
PHRASE_PAUSE_WEIGHT = 0.5
PHRASE_PAUSE_SECONDS = 0.5
# Spoken Chinese comes back from the recogniser with few pauses and no punctuation, so the
# words themselves say where a phrase ends: after a particle that closes one, before a
# word that opens the next, and never between a number and what it counts (`8|點`,
# `一|整天`). Weighed against the pause and against uneven lines as above. Provisional.
PHRASE_CLOSERS = tuple("啊了吧呢囉喔嗎呀啦耶欸哦嘛，。！？、,.!?")
PHRASE_OPENERS = ("但是", "可是", "不過", "然後", "所以", "因為", "而且", "其實", "結果", "還有", "那", "我", "你", "他", "她")
PHRASE_NUMERALS = tuple("0123456789一二三四五六七八九十百千萬兩幾半")
PHRASE_CLOSER_WEIGHT = 0.4
PHRASE_OPENER_WEIGHT = 0.3
PHRASE_NUMERAL_PENALTY = 0.8
# The recogniser hands Chinese back a character at a time, so its words say nothing about
# where `營業` begins. A dictionary does, and a break inside one of its words is worse than
# any line being uneven. Heavy rather than forbidden, because a word longer than the room
# left on a line still has to be broken somewhere. Provisional.
PHRASE_INSIDE_WORD_PENALTY = 1.5
# Short lines read faster than full ones, so a line more is worth it to break in the right
# place, up to one extra line in three. Each extra line costs this much. Provisional.
PHRASE_EXTRA_LINE_COST = 0.3

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
    the cut that ends it. A clip turned all the way down shows only captions
    written by hand for its picture, the ones with no word timings: nobody
    hears what was said on it.

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
    heard = captioned_clips(project)
    # A shot turned all the way down says nothing, but a caption written for its picture —
    # one with no word timings, typed rather than transcribed — still belongs on it.
    muted = [
        clip for clip in (project.base_video_track.clips if project.base_video_track else [])
        if clip.volume == 0
    ]
    for clip in sorted([*heard, *muted], key=lambda clip: clip.timeline_in):
        speed = Decimal(str(clip.speed))
        window_start, window_end = clip.source_range.start, clip.source_range.end
        for cue in by_asset.get(clip.asset_id, ()):
            if clip.volume == 0 and cue.words:
                continue
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

@dataclass(frozen=True)
class CaptionGeometry:
    """How big captions are drawn in one frame, and how much room they have.

    Attributes:
        font_size: Font size in pixels.
        margin_h: Space kept clear at each side, in pixels.
        margin_v: Space kept clear below the text, in pixels.
        outline: Outline width in pixels.
        max_units: How wide a line may run, as a multiple of the font size.
    """

    font_size: int
    margin_h: int
    margin_v: int
    outline: int
    max_units: float

def caption_geometry(width: int, height: int, style: CaptionStyle) -> CaptionGeometry:
    """Work out the size and margins captions get in a frame.

    One place for this, because two things need the same answer: the file
    that draws the captions, and the check that they fit.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        style: How the captions are drawn.

    Returns:
        The geometry.
    """
    font_size = max(12, round(min(width, height) * style.size_fraction))
    bottom = style.bottom_fraction
    if bottom is None:
        bottom = PORTRAIT_BOTTOM_FRACTION if height > width else LANDSCAPE_BOTTOM_FRACTION
    margin_h = round(width * style.side_fraction)
    return CaptionGeometry(
        font_size=font_size,
        margin_h=margin_h,
        margin_v=round(height * bottom),
        # At least a pixel when one is asked for at all; none when none is.
        outline=max(1, round(font_size * style.outline_fraction)) if style.outline_fraction else 0,
        # libass measures in the ASS resolution, so the usable width is the frame minus both margins.
        max_units=max(4.0, (width - 2 * margin_h) / font_size),
    )

def _speaker_prefix(cue: PlacedCue, style: CaptionStyle) -> str:
    """Write the name that goes in front of a caption, when the style shows one.

    Args:
        cue: The placed caption.
        style: How captions are drawn.

    Returns:
        The escaped name and its colon, or nothing.
    """
    if style.speaker_mark not in (SpeakerMark.NAME, SpeakerMark.BOTH) or not cue.speaker:
        return ""
    # The label when there is no name for it: a caption reading `S2` is still
    # better than one putting the words in the wrong person's mouth.
    return f"{escape_ass_inline(style.speaker_names.get(cue.speaker, cue.speaker))}："

def _room(prefix: str, geometry: CaptionGeometry) -> float:
    """Work out how wide the words of a caption may run beside the name in front of them.

    Args:
        prefix: The speaker's name as drawn, or nothing.
        geometry: The frame's caption geometry.

    Returns:
        The width left, as a multiple of the font size.
    """
    return max(4.0, geometry.max_units - sum(_display_width(character) for character in prefix))

def _text_width(text: str) -> float:
    """Measure a run of text the way a line is measured.

    Args:
        text: The text.

    Returns:
        Its width as a multiple of the font size.
    """
    return sum(_display_width(character) for character in text)

def _phrased(
    count: int,
    fits: Callable[[int, int], bool],
    width: Callable[[int, int], float],
    edge: Callable[[int], float],
) -> List[Tuple[int, int]]:
    """Break a run of words into lines that end where a phrase does.

    Filling each line to the brim breaks wherever the room runs out: between
    `下午2` and `點`, or with one character left over to flash past on its own.
    So every way of breaking into the fewest lines that fit, or a few more, is
    weighed: lines about as long as each other, each break where a phrase
    ends, and no more lines than that is worth.

    Args:
        count: How many words.
        fits: Whether words `i` up to `j` (not included) fit on one line. A
            single word always does, since it cannot be broken.
        width: How wide words `i` up to `j` are.
        edge: How good a place the start of word `i` is to break, from about
            -1 (inside a phrase) to 1 (between two).

    Returns:
        Each line as `(first, past_last)` word indices, in order.
    """
    if count == 0:
        return []
    fewest, start = 0, 0
    while start < count:
        end = start + 1
        while end < count and fits(start, end + 1):
            end += 1
        fewest, start = fewest + 1, end
    most = min(count, fewest + (fewest + 2) // 3)
    infinity = float("inf")
    chosen: List[Tuple[int, int]] = []
    cheapest = infinity
    for wanted in range(fewest, most + 1):
        target = width(0, count) / wanted

        def cost(first: int, past: int) -> float:
            uneven = ((width(first, past) - target) / target) ** 2 if target else 0.0
            return uneven - (edge(first) if first else 0.0)

        # best[lines][end]: the cheapest way to put the first `end` words on that many lines.
        best = [[infinity] * (count + 1) for _ in range(wanted + 1)]
        back = [[0] * (count + 1) for _ in range(wanted + 1)]
        best[0][0] = 0.0
        for lines in range(1, wanted + 1):
            for end in range(1, count + 1):
                for first in range(end - 1, -1, -1):
                    if end - first > 1 and not fits(first, end):
                        break
                    if best[lines - 1][first] == infinity:
                        continue
                    total = best[lines - 1][first] + cost(first, end)
                    if total < best[lines][end]:
                        best[lines][end], back[lines][end] = total, first
        total = best[wanted][count] + PHRASE_EXTRA_LINE_COST * (wanted - fewest)
        if total < cheapest:
            breaks: List[Tuple[int, int]] = []
            end = count
            for lines in range(wanted, 0, -1):
                first = back[lines][end]
                breaks.append((first, end))
                end = first
            cheapest, chosen = total, breaks[::-1]
    return chosen

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

@lru_cache(maxsize=1)
def _segmenter() -> Callable[[str], List[str]]:
    """Load the Chinese word splitter once, the first time a Chinese caption is broken.

    Returns:
        A function from text to its words. The text is read in Simplified,
        which is what the dictionary is built from: it finds `咖啡厅` whole
        and `咖啡廳` in two. OpenCC converts one character to one here, so the
        words map straight back onto the Traditional text.
    """
    import logging
    import warnings

    import opencc

    with warnings.catch_warnings():
        # jieba's own regular expressions are written in a way Python now warns about.
        warnings.simplefilter("ignore", SyntaxWarning)
        import jieba
    jieba.setLogLevel(logging.WARNING)
    simplify = opencc.OpenCC("t2s.json").convert

    def split(text: str) -> List[str]:
        simple = simplify(text)
        return list(jieba.cut(simple if len(simple) == len(text) else text))

    return split

def word_edges(text: str) -> Optional[set[int]]:
    """Find where the words of a Chinese caption begin and end.

    Args:
        text: The caption text.

    Returns:
        Every character offset a word begins or ends at, or nothing for text
        with no Chinese in it: other languages already come from the
        recogniser a word at a time.
    """
    if not _CJK.search(text):
        return None
    edges, offset = {0}, 0
    for word in _segmenter()(text):
        offset += len(word)
        edges.add(offset)
    return edges

def _word_lines(
    words: Sequence[Union[CueWord, TranscriptWord]],
    fits: Callable[[int, int], bool],
) -> List[Tuple[int, int]]:
    """Break timed words into lines where a phrase ends, as far as it can tell.

    Args:
        words: The words, with their timings.
        fits: Whether words `i` up to `j` (not included) fit on one line.

    Returns:
        Each line as `(first, past_last)` word indices, in order.
    """
    widths = [_text_width(escape_ass_inline(word.text)) for word in words]
    offsets = [0]
    for word in words:
        offsets.append(offsets[-1] + len(word.text))
    edges = word_edges("".join(word.text for word in words))

    def edge(index: int) -> float:
        before, after = words[index - 1].text.strip(), words[index].text.strip()
        pause = float(words[index].start - words[index - 1].end)
        score = PHRASE_PAUSE_WEIGHT * min(max(pause, 0.0), PHRASE_PAUSE_SECONDS) / PHRASE_PAUSE_SECONDS
        if before.endswith(PHRASE_CLOSERS):
            score += PHRASE_CLOSER_WEIGHT
        if after.startswith(PHRASE_OPENERS):
            score += PHRASE_OPENER_WEIGHT
        if before.endswith(PHRASE_NUMERALS):
            score -= PHRASE_NUMERAL_PENALTY
        if edges is not None and offsets[index] not in edges:
            score -= PHRASE_INSIDE_WORD_PENALTY
        return score

    return _phrased(len(words), fits, lambda first, past: sum(widths[first:past]), edge)

def _at_share(cue: PlacedCue, share: float) -> Decimal:
    """Find the moment a given share of the way through a caption.

    Args:
        cue: The caption.
        share: How far through it, from 0 to 1.

    Returns:
        The time on the timeline, to the millisecond.
    """
    return (cue.start + (cue.end - cue.start) * Decimal(str(share))).quantize(MILLISECOND)

def one_line_each(cue: PlacedCue, geometry: CaptionGeometry, style: CaptionStyle) -> List[PlacedCue]:
    """Split a caption too long for one line into lines shown one after another.

    A stack of lines climbs up into the picture, and what a viewer reads at a
    glance is one line. With word timings each line comes up as its first word
    is said and stays until the next one does; without them, the caption's time
    is shared out by how much of it each line holds. Done where the captions are
    drawn rather than where they are written, so a caption stored for a wide
    frame still comes out as single lines in a narrow one.

    Args:
        cue: The placed caption.
        geometry: The frame's caption geometry.
        style: How captions are drawn.

    Returns:
        The caption as one or more captions, in order. Unchanged when it fits,
        when the style allows stacking, or when it is bilingual: the second
        language belongs under the whole of the first, not under a piece of it.
    """
    if not style.single_line or cue.secondary:
        return [cue]
    room = _room(_speaker_prefix(cue, style), geometry)
    words = [word for word in cue.words if escape_ass_inline(word.text).strip()]
    if words:
        widths = [_text_width(escape_ass_inline(word.text)) for word in words]
        if sum(widths) <= room:
            return [cue]
        lines = _word_lines(words, lambda first, past: sum(widths[first:past]) <= room)
        groups = [list(words[first:past]) for first, past in lines]
        starts = [cue.start] + [group[0].start for group in groups[1:]]
        ends = starts[1:] + [cue.end]
        return [
            cue.model_copy(update={
                "start": start, "end": end, "words": group,
                "text": "".join(word.text for word in group).strip(),
            })
            for group, start, end in zip(groups, starts, ends)
            if end > start
        ]
    # Several lines in the stored text are one thing said, so they are read as one run.
    text = " ".join(part for part in escape_ass_text(cue.text).split("\\N") if part)
    if _text_width(text) <= room:
        return [cue]
    # As few lines as fit, filled evenly; with no timings there are no pauses to prefer,
    # and `wrap_caption` already knows where the spaces are.
    fewest = len(wrap_caption(text, room))
    limit = _text_width(text) / fewest
    while limit < room and len(wrap_caption(text, limit)) > fewest:
        limit += 0.5
    wrapped = wrap_caption(text, min(limit, room))
    total = sum(_text_width(line) for line in wrapped) or 1.0
    pieces: List[PlacedCue] = []
    done = 0.0
    for line in wrapped:
        start = _at_share(cue, done / total)
        done += _text_width(line)
        end = _at_share(cue, done / total)
        if end > start:
            pieces.append(cue.model_copy(update={"start": start, "end": end, "text": line}))
    return pieces or [cue]

def caption_text(cue: PlacedCue, geometry: CaptionGeometry, style: CaptionStyle) -> Tuple[str, int, int]:
    """Write one caption the way it is drawn, and count its lines.

    A speaker's name goes in front of the words, so it is measured before the
    words are broken into lines rather than stuck on afterwards: Chinese has
    no spaces for libass to break at, and a first line made longer by a name
    it was not measured with runs off the side of the picture.

    Args:
        cue: The placed caption.
        geometry: The frame's caption geometry.
        style: How captions are drawn.

    Returns:
        `(text, lines, secondary_lines)` — the event text, empty when there
        is nothing to show, and how many lines of each size it takes.
    """
    prefix = _speaker_prefix(cue, style)
    room = _room(prefix, geometry)
    if style.karaoke and cue.words:
        text = _karaoke_text(cue, room)
    else:
        text = _wrapped(cue.text, room)
    if not text:
        return "", 0, 0
    lines = text.count("\\N") + 1
    text = prefix + text
    below = 0
    if cue.secondary:
        # Measured against its own size: a smaller line fits more before it wraps.
        second = _wrapped(cue.secondary, geometry.max_units / style.secondary_scale)
        if second:
            below = second.count("\\N") + 1
            text = f"{text}\\N{{\\fs{max(8, round(geometry.font_size * style.secondary_scale))}}}{second}"
    return text, lines, below

def caption_overflow(
    cues: Sequence[PlacedCue],
    width: int,
    height: int,
    style: Optional[CaptionStyle] = None,
) -> List[PlacedCue]:
    """Find the captions too tall to fit in the frame.

    Width is already taken care of — every line is broken to fit — so what can
    still go wrong is height: a long line in a narrow frame breaks into so
    many lines that the block climbs past the top of the picture. That happens
    most when a cut made for landscape is rendered portrait. With the style's
    `single_line` on, only a bilingual caption can stack, so that is what is
    left to find.

    Args:
        cues: The placed captions.
        width: Frame width in pixels.
        height: Frame height in pixels.
        style: How they are drawn; the plain style when not given.

    Returns:
        The captions that do not fit, in timeline order.
    """
    style = style or CaptionStyle()
    geometry = caption_geometry(width, height, style)
    line = geometry.font_size * CAPTION_LINE_HEIGHT
    room = height - geometry.margin_v
    over: List[PlacedCue] = []
    for cue in cues:
        for piece in one_line_each(cue, geometry, style):
            _, lines, below = caption_text(piece, geometry, style)
            if lines * line + below * line * style.secondary_scale > room:
                over.append(cue)
                break
    return over

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
    geometry = caption_geometry(width, height, style)
    font_size, margin_h, margin_v, outline = (
        geometry.font_size, geometry.margin_h, geometry.margin_v, geometry.outline,
    )
    # With karaoke the secondary colour is what a word looks like before it is said, so it
    # has to differ from the primary or nothing appears to happen. Without it the two are
    # the same, which is what a caption that simply sits there wants.
    waiting = UNSPOKEN_COLOUR if style.karaoke else "&H00FFFFFF"
    colours = _speaker_colours(cues) if style.speaker_mark in (SpeakerMark.COLOUR, SpeakerMark.BOTH) else {}

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
        # The shadow goes with the outline: a caption asked for with no outline is plain white text.
        f"-1,0,0,0,100,100,0,0,1,{outline},{max(1, outline // 2) if outline else 0},2,{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in (piece for placed in cues for piece in one_line_each(placed, geometry, style)):
        text, _, _ = caption_text(cue, geometry, style)
        if not text:
            continue
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
    max_units: Optional[float] = None,
) -> Iterator[Tuple[float, float, str, List[CueWord]]]:
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
        max_units: Widest caption before it is broken up, as a multiple of the
            font size, or no limit. A width rather than a count, because a
            Chinese character takes twice the room of a Latin letter.

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

    def fits(first: int, past: int) -> bool:
        joined = "".join(word.text for word in words[first:past]).strip()
        return (
            len(joined) <= max_characters
            and words[past - 1].end - words[first].start <= max_seconds
            and (max_units is None or _text_width(joined) <= max_units)
        )

    chunks = [words[first:past] for first, past in _word_lines(words, fits)]

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
    max_units: Optional[float] = None,
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
        max_units: Widest caption before it is broken up, as a multiple of
            the font size: one line of the frame the captions are for, so each
            caption proofread is one line seen. No limit when not given.

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
                segment, start, end, max_characters, max_seconds, max_units,
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
