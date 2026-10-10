"""What changed between two versions of a cut, said the way a person would say it.

Clips are matched by what they show — the same file over overlapping seconds —
not by their IDs or their place, since a recompile gives every clip a new ID
and a trim at the start moves everything after it. Changes are grouped by the
cut's parts (its markers, one per beat of the plan it was compiled from), so
two versions can be merged a part at a time. Pure, like the rest of the
engine: the same two cuts always give the same list.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from app.models.timeline import Clip, Project, TrackType

# Provisional (roadmap §13): shorter than this, a change in where a clip starts or ends is
# rounding or a nudge nobody would notice, not an edit worth reading out.
NOTICED_SECONDS = 0.05
# Provisional (roadmap §13): how much of what is said a stretch is named by before 「…」.
QUOTE_CHARACTERS = 16
# What a part is called when the cut has no markers: the whole of it.
WHOLE_PART = "全片"

Said = Callable[[str, float, float], str]
Named = Callable[[str], str]


@dataclass(frozen=True)
class Part:
    """One part of a cut: a name and the stretch it runs over."""

    name: str
    start: float
    end: float


def parts_of(project: Project) -> List[Part]:
    """The parts of a cut, from its markers; one part for a cut without any.

    Args:
        project: The cut.

    Returns:
        Each part, in order, running to the next one or the end.
    """
    length = float(project.duration)
    markers = sorted(project.markers or [], key=lambda marker: marker.timeline_in)
    if not markers:
        return [Part(WHOLE_PART, 0.0, length)]
    starts = [0.0 if index == 0 else float(marker.timeline_in) for index, marker in enumerate(markers)]
    return [Part(marker.name, start, starts[index + 1] if index + 1 < len(starts) else length)
            for index, (marker, start) in enumerate(zip(markers, starts))]


def _shown(project: Project) -> List[Clip]:
    """The sequence's clips, in order."""
    base = project.base_video_track
    return sorted(base.clips, key=lambda clip: clip.timeline_in) if base else []


def clips_in(part: Part, clips: List[Clip], last: bool) -> List[Clip]:
    """The clips whose middle is inside a part.

    The middle rather than the start: a trim by hand moves the clips after
    it but not the markers, and a clip that slid a second earlier still
    belongs to the part it was in.

    Args:
        part: The part.
        clips: The sequence's clips.
        last: Whether it is the cut's last part, which also takes anything
            running past its end.

    Returns:
        The part's clips, in the order given.
    """
    def middle(clip: Clip) -> float:
        return float(clip.timeline_in) + float(clip.timeline_duration) / 2

    return [clip for clip in clips if part.start <= middle(clip) < part.end or (last and middle(clip) >= part.end)]


def _overlap(one: Clip, other: Clip) -> float:
    """How many seconds of the same file two clips share."""
    if one.asset_id != other.asset_id:
        return 0.0
    return max(0.0, min(float(one.source_range.end), float(other.source_range.end))
               - max(float(one.source_range.start), float(other.source_range.start)))


def match_clips(before: List[Clip], after: List[Clip]) -> List[Tuple[Clip, Clip]]:
    """Pair the clips of two versions that show the same footage.

    Args:
        before: One version's clips.
        after: The other's.

    Returns:
        `(before, after)` pairs, each clip in at most one, the most shared
        footage first.
    """
    candidates = sorted(
        ((_overlap(one, other), index, other_index) for index, one in enumerate(before)
         for other_index, other in enumerate(after)),
        key=lambda item: -item[0],
    )
    used_before, used_after, pairs = set(), set(), []
    for shared, index, other_index in candidates:
        if shared <= 0:
            break
        if index in used_before or other_index in used_after:
            continue
        used_before.add(index)
        used_after.add(other_index)
        pairs.append((before[index], after[other_index]))
    return sorted(pairs, key=lambda pair: pair[0].timeline_in)


def _quote(said: Said, clip: Clip, start: float, end: float) -> str:
    """What is heard over a stretch of a clip's file, short, for naming it; empty when nothing is."""
    text = said(clip.asset_id, start, end).strip()
    return f"「{text[:QUOTE_CHARACTERS]}{'…' if len(text) > QUOTE_CHARACTERS else ''}」" if text else ""


def _seconds(value: float) -> str:
    """Seconds as a person says them."""
    return f"{value:.1f} 秒"


def _clip_changes(one: Clip, other: Clip, said: Said, named: Named) -> List[str]:
    """Say how the same footage is used differently in two versions."""
    changes = []
    head = float(other.source_range.start) - float(one.source_range.start)
    tail = float(other.source_range.end) - float(one.source_range.end)
    where = _quote(said, one, float(one.source_range.start), float(one.source_range.end)) or f"{named(one.asset_id)} 那段"
    if head > NOTICED_SECONDS:
        lost = _quote(said, one, float(one.source_range.start), float(other.source_range.start))
        changes.append(f"{where}開頭剪掉 {_seconds(head)}" + (f"（{lost}）" if lost else ""))
    elif head < -NOTICED_SECONDS:
        changes.append(f"{where}開頭多留 {_seconds(-head)}")
    if tail < -NOTICED_SECONDS:
        lost = _quote(said, one, float(other.source_range.end), float(one.source_range.end))
        changes.append(f"{where}結尾剪掉 {_seconds(-tail)}" + (f"（{lost}）" if lost else ""))
    elif tail > NOTICED_SECONDS:
        changes.append(f"{where}結尾多留 {_seconds(tail)}")
    if abs((other.speed or 1) - (one.speed or 1)) > 0.001:
        changes.append(f"{where}速度 {one.speed:g}× → {other.speed:g}×")
    if abs(float(other.volume) - float(one.volume)) > 0.001:
        changes.append(f"{where}音量 {round(float(one.volume) * 100)}% → {round(float(other.volume) * 100)}%")
    before_join = one.transition_in.kind if one.transition_in else None
    after_join = other.transition_in.kind if other.transition_in else None
    if before_join != after_join:
        changes.append(f"{where}的轉場 {before_join or '直接切'} → {after_join or '直接切'}")
    return changes


def _named_stretch(clip: Clip, said: Said, named: Named) -> str:
    """A clip as a person names it: by what is said in it, or else by its file."""
    return _quote(said, clip, float(clip.source_range.start), float(clip.source_range.end)) or \
        f"{named(clip.asset_id)} 的一段"


def _part_changes(before: List[Clip], after: List[Clip], pairs: List[Tuple[Clip, Clip]], moved_to: Dict[int, str],
                  said: Said, named: Named) -> List[dict]:
    """Say what changed inside one part.

    Args:
        before: The part's clips on the left.
        after: Its clips on the right.
        pairs: Every pair of clips showing the same footage, across the whole cut.
        moved_to: For a right clip in another part than its left one, that part's name, by `id`.
        said: What is heard in a stretch of a file.
        named: A file's name.

    Returns:
        The changes, in the order they come on the left.
    """
    left_paired = {id(one) for one, _ in pairs}
    right_paired = {id(other) for _, other in pairs}
    mine = [(one, other) for one, other in pairs if any(one is clip for clip in before)]
    changes: List[dict] = []
    for clip in before:
        if id(clip) not in left_paired:
            changes.append({"what": f"拿掉 {_named_stretch(clip, said, named)}（{_seconds(float(clip.timeline_duration))}）",
                            "a_at": float(clip.timeline_in), "b_at": None})
    for clip in after:
        if id(clip) not in right_paired:
            changes.append({"what": f"加入 {_named_stretch(clip, said, named)}（{_seconds(float(clip.timeline_duration))}）",
                            "a_at": None, "b_at": float(clip.timeline_in)})
    for one, other in mine:
        at = {"a_at": float(one.timeline_in), "b_at": float(other.timeline_in)}
        if id(other) in moved_to:
            changes.append({"what": f"{_named_stretch(one, said, named)}移到「{moved_to[id(other)]}」", **at})
        for what in _clip_changes(one, other, said, named):
            changes.append({"what": what, **at})
    staying = [(one, other) for one, other in mine if id(other) not in moved_to]
    if [id(one) for one, _ in sorted(staying, key=lambda pair: pair[0].timeline_in)] != \
            [id(one) for one, _ in sorted(staying, key=lambda pair: pair[1].timeline_in)]:
        changes.append({"what": "順序換了", "a_at": float(before[0].timeline_in) if before else None,
                        "b_at": float(after[0].timeline_in) if after else None})
    changes.sort(key=lambda change: (change["a_at"] if change["a_at"] is not None else change["b_at"] or 0))
    return changes


def _aligned(pairs: List[Tuple[Clip, Clip]]) -> List[list]:
    """Where the same footage plays on each side: `[a_start, a_end, b_start, b_end]` per pair."""
    aligned = []
    for one, other in pairs:
        shared_start = max(float(one.source_range.start), float(other.source_range.start))
        shared_end = min(float(one.source_range.end), float(other.source_range.end))
        aligned.append([
            round(float(one.timeline_in) + (shared_start - float(one.source_range.start)) / (one.speed or 1), 3),
            round(float(one.timeline_in) + (shared_end - float(one.source_range.start)) / (one.speed or 1), 3),
            round(float(other.timeline_in) + (shared_start - float(other.source_range.start)) / (other.speed or 1), 3),
            round(float(other.timeline_in) + (shared_end - float(other.source_range.start)) / (other.speed or 1), 3),
        ])
    return sorted(aligned)


def _songs(project: Project) -> List[Tuple[str, float]]:
    """The songs on the audio tracks, by file, with where each comes in."""
    return [(clip.asset_id, float(clip.timeline_in)) for track in project.tracks
            if track.track_type == TrackType.AUDIO and not track.voice for clip in track.clips]


def compare(before: Project, after: Project, said: Said, named: Named) -> dict:
    """Say what changed from one version of a cut to another.

    Args:
        before: The version on the left.
        after: The version on the right.
        said: What is heard in a file between two of its seconds, for naming
            a stretch by its words.
        named: A file's name, for a stretch with nothing said in it.

    Returns:
        `length` of each; `parts`, each with its `name`, its stretch on each
        side (`a`, `b`, None where it is missing), `same` and its `changes`
        (`what`, and `a_at`/`b_at` to jump each side to it); `whole`, changes
        to the video as a whole — songs, captions — keyed for picking a side;
        and `aligned`, `[a_start, a_end, b_start, b_end]` for each stretch of
        the same footage, for playing the two side by side.
    """
    left_parts, right_parts = parts_of(before), parts_of(after)
    left_clips, right_clips = _shown(before), _shown(after)
    right_by_name: Dict[str, Part] = {}
    for part in right_parts:
        right_by_name.setdefault(part.name, part)
    # Matched across the whole cut first, so a shot that went from one part to another is one move.
    pairs = match_clips(left_clips, right_clips)
    left_in = {part.name: clips_in(part, left_clips, part is left_parts[-1]) for part in left_parts}
    right_in = {part.name: clips_in(part, right_clips, part is right_parts[-1]) for part in right_parts}
    left_part = {id(clip): name for name, clips in left_in.items() for clip in clips}
    right_part = {id(clip): name for name, clips in right_in.items() for clip in clips}
    moved_to = {id(other): right_part[id(other)] for one, other in pairs
                if id(other) in right_part and right_part[id(other)] != left_part.get(id(one))}
    parts: List[dict] = []
    seen = set()
    for part in left_parts:
        other = right_by_name.get(part.name) if part.name not in seen else None
        seen.add(part.name)
        if other is None:
            parts.append({"name": part.name, "a": [part.start, part.end], "b": None, "same": False,
                          "changes": [{"what": f"拿掉整段「{part.name}」", "a_at": part.start, "b_at": None}]})
            continue
        arriving = {id(clip) for clip in right_in[part.name] if id(clip) in moved_to}
        changes = _part_changes(left_in[part.name], [clip for clip in right_in[part.name] if id(clip) not in arriving],
                                pairs, moved_to, said, named)
        parts.append({"name": part.name, "a": [part.start, part.end], "b": [other.start, other.end],
                      "same": not changes, "changes": changes})
    for part in right_parts:
        if part.name not in seen:
            parts.append({"name": part.name, "a": None, "b": [part.start, part.end], "same": False,
                          "changes": [{"what": f"加入整段「{part.name}」", "a_at": None, "b_at": part.start}]})
    if [part.name for part in left_parts] != [part.name for part in right_parts] and \
            {part.name for part in left_parts} == {part.name for part in right_parts}:
        parts.insert(0, {"name": "段落順序", "a": None, "b": None, "same": False, "order": True,
                         "changes": [{"what": "段落順序換了：" + " → ".join(part.name for part in right_parts),
                                      "a_at": 0.0, "b_at": 0.0}]})

    whole: List[dict] = []
    songs_before, songs_after = _songs(before), _songs(after)
    if [song for song, _ in songs_before] != [song for song, _ in songs_after]:
        was = "、".join(f"「{named(song)}」" for song, _ in songs_before) or "沒有配樂"
        now = "、".join(f"「{named(song)}」" for song, _ in songs_after) or "沒有配樂"
        whole.append({"key": "music", "what": f"配樂 {was} → {now}",
                      "a_at": songs_before[0][1] if songs_before else None,
                      "b_at": songs_after[0][1] if songs_after else None})
    if len(before.subtitles) != len(after.subtitles) or [cue.text for cue in before.subtitles] != \
            [cue.text for cue in after.subtitles]:
        whole.append({"key": "captions", "what": f"字幕 {len(before.subtitles)} 句 → {len(after.subtitles)} 句",
                      "a_at": None, "b_at": None})
    if before.caption_style != after.caption_style:
        whole.append({"key": "caption_style", "what": "字幕樣式改了", "a_at": None, "b_at": None})
    if (before.width, before.height) != (after.width, after.height):
        whole.append({"key": "frame", "what": f"畫面 {before.width}×{before.height} → {after.width}×{after.height}",
                      "a_at": None, "b_at": None})
    return {
        "length": [round(float(before.duration), 3), round(float(after.duration), 3)],
        "parts": parts,
        "whole": whole,
        "aligned": _aligned(pairs),
    }
