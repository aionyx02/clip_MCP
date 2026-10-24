"""What people corrected after the AI's first cut, counted by rule.

Every version of a project is compared with the one before it, once the AI
has made a first cut that shows footage. A change is counted against the
piece of the cut it touched: a caption heard wrong, a line broken in another
place, a caption put at another time, a cut point nudged, or a different
stretch of footage chosen.

The rules are deliberately plain. Rhythm (節奏) is not told apart from a cut
point: an edge moved to tighten a pause and an edge moved because the take
was wrong look the same here, and which one it was is a judgement about the
footage. Nor is a cut that lands inside a word told apart from one that lands
in a pause: that needs the transcript, which these functions do not read. So
neither is guessed at; both are left to be read from the footage.

Only the cut is compared: the footage the base video track shows, and the
captions. Title cards, music, B-roll and anything else beside the cut are not
counted. Footage is compared by which stretches of each file are shown, never
by clip: a recompile names every clip after its place, and a split leaves two
clips where one was. So a split, a swap, a move along the timeline and a
change of speed are not counted, and footage shown twice counts once — only
which footage is shown. Captions are matched by what they say where, and a
caption that comes or goes with its footage is not a correction. A restore or
an undo is a new version like any other, so what it puts back is counted
again, under whoever restored it. A branch keeps its own history, so hand
edits made on a branch before the branch's first AI version with footage in
it are not counted.

A version that cannot be read as a project is not compared at all: the
comparison runs from the last version that could be read to the next one that
could, and the ones skipped are counted as `unreadable`. Nothing about them
is guessed at.

Reading the history lives in `main`, which imports the workspace only when it
is called, so importing this module stays light and pure.
"""

import argparse
import json
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from app.engine.compare import shown
from app.models.timeline.parts import TOUCHING_SECONDS, Clip, Project, SubtitleCue
from app.storage.history import Version

# Who the editor app writes as. Said here rather than imported from the UI, which
# would drag the web app into a command that only reads the history.
EDITOR_AUTHOR = "編輯器"
# Two stretches of a file this close are one: the server's own tolerance for touching.
TOUCHING = float(TOUCHING_SECONDS)
# Provisional: a source edge that moves no more than this is a nudge of the same cut
# point, not a different stretch of the file. It separates `cut_point` from
# `wrong_piece`; below it a trim looks like the same shot, above it like another take.
SMALL_EDGE_SECONDS = 1.0
# Provisional: a gap longer than this between two versions made by hand is a new
# sitting at the editor, not a pause inside one.
SITTING_GAP_SECONDS = 600
# Provisional: a sitting counts as at least this long, because even one change took
# somebody some time to make.
MINUTES_PER_SITTING_FLOOR = 1.0
# How many versions of a project to read: more than any project is likely to have, so
# that no correction is silently left out of the summary.
VERSION_LIMIT = 10000

# The causes a change is counted under, in the order they are reported when their
# counts are equal.
CAUSES = ("misheard", "line_breaks", "caption_timing", "cut_point", "wrong_piece")
CAUSE_LABELS = {
    "misheard": "聽錯",
    "line_breaks": "換行",
    "caption_timing": "字幕時間",
    "cut_point": "剪點微調",
    "wrong_piece": "選錯片段",
}

def _readable(project: Optional[Mapping]) -> Optional[Project]:
    """A stored project as a model, or None when it cannot be read as one."""
    if project is None:
        return None
    try:
        return Project.model_validate(project)
    except ValidationError:
        return None

def _read(versions: List[Tuple[Version, Optional[dict]]]) -> List[Tuple[Version, Optional[Project]]]:
    """Each version with its project read as a model, None where it cannot be."""
    return [(version, _readable(project)) for version, project in versions]

def _footage_clips(project: Project) -> List[Clip]:
    """The clips of the cut that show footage: the base video track's, title cards left out."""
    return [clip for clip in shown(project) if clip.card is None]

def _first(read: List[Tuple[Version, Optional[Project]]]) -> Optional[int]:
    """The index of the first AI version whose cut shows footage, or None."""
    for index, (version, project) in enumerate(read):
        if version.author.startswith("AI") and project is not None and _footage_clips(project):
            return index
    return None

def first_cut(versions: List[Tuple[Version, Optional[dict]]]) -> Optional[int]:
    """The first version the AI made a cut in.

    Args:
        versions: A project's versions, oldest first, each with the project as
            it was then.

    Returns:
        The index of the first version written by an AI author whose project
        can be read and whose cut (the base video track) shows footage — a
        title card alone is not a cut — or None when there is no such version.
    """
    return _first(_read(versions))

@dataclass
class Stretch:
    """A stretch of one file the cut shows, and the stretch of the timeline it is shown over.

    Attributes:
        start: Where it starts in the file, in seconds.
        end: Where it ends in the file.
        timeline_in: Where the first clip showing it starts on the timeline.
        timeline_out: Where the last clip showing it ends.
    """

    start: float
    end: float
    timeline_in: float
    timeline_out: float

def _footage(project: Project) -> Dict[str, List[Stretch]]:
    """What the cut shows of each file, as stretches that neither overlap nor touch.

    Clips that show touching or overlapping stretches of a file are one
    stretch, whatever their order, speed or place on the timeline.

    Args:
        project: The project.

    Returns:
        Each file's stretches, in the order they come in the file.
    """
    found: Dict[str, List[Stretch]] = {}
    for clip in _footage_clips(project):
        found.setdefault(clip.asset_id, []).append(Stretch(
            float(clip.source_range.start), float(clip.source_range.end),
            float(clip.timeline_in), float(clip.timeline_out),
        ))
    merged: Dict[str, List[Stretch]] = {}
    for asset_id, stretches in found.items():
        joined: List[Stretch] = []
        for stretch in sorted(stretches, key=lambda one: one.start):
            last = joined[-1] if joined else None
            if last is not None and stretch.start <= last.end + TOUCHING:
                joined[-1] = Stretch(last.start, max(last.end, stretch.end), min(last.timeline_in, stretch.timeline_in),
                                     max(last.timeline_out, stretch.timeline_out))
            else:
                joined.append(stretch)
        merged[asset_id] = joined
    return merged

def _left_over(stretch: Stretch, others: List[Stretch]) -> Optional[List[float]]:
    """The lengths of the pieces of a stretch that none of the others cover.

    Args:
        stretch: The stretch.
        others: The other version's stretches of the same file.

    Returns:
        Each uncovered piece's length, in the order they come; None when the
        others cover none of it, so the whole stretch is gone or new.
    """
    covering = [other for other in others if min(stretch.end, other.end) - max(stretch.start, other.start) > TOUCHING]
    if not covering:
        return None
    pieces, at = [], stretch.start
    for other in covering:
        if other.start - at > TOUCHING:
            pieces.append(round(other.start - at, 3))
        at = max(at, other.end)
    if stretch.end - at > TOUCHING:
        pieces.append(round(stretch.end - at, 3))
    return pieces

def _pairs(count: int, other_count: int, shared: Callable[[int, int], float]) -> List[Tuple[int, int]]:
    """Pair two lists greedily, most shared first, each item in at most one pair; nothing shared, no pair.

    Args:
        count: How many items the first list has.
        other_count: How many the second has.
        shared: How much an item of the first shares with one of the second.

    Returns:
        `(index, other_index)` pairs.
    """
    candidates = sorted(
        ((shared(index, other_index), index, other_index)
         for index in range(count) for other_index in range(other_count)),
        key=lambda item: -item[0],
    )
    used, other_used, pairs = set(), set(), []
    for amount, index, other_index in candidates:
        if amount <= 0:
            break
        if index in used or other_index in other_used:
            continue
        used.add(index)
        other_used.add(other_index)
        pairs.append((index, other_index))
    return pairs

def _on_timeline_together(one: Stretch, other: Stretch) -> float:
    """How many seconds of the timeline two stretches are both shown over."""
    return max(0.0, min(one.timeline_out, other.timeline_out) - max(one.timeline_in, other.timeline_in))

def _clip_changes(before: Project, after: Project) -> Dict[str, Any]:
    """How the footage the cut shows changed, by what is shown rather than by clip.

    Each version's footage is reduced to the stretches of each file it shows
    (`_footage`). A stretch shown in both versions, but longer or shorter in
    one, moved edges: each piece one version shows and the other does not,
    next to footage both show, is an edge moved by that piece's length — a
    trimmed end, or a pause cut out of the middle of a shot. A stretch the
    other version shows none of was removed or added; a removed one and an
    added one shown over the same stretch of the timeline count once, as
    replaced. Which clips show a stretch, in what order, at what speed and
    where on the timeline is not counted.

    Args:
        before: The project as it was.
        after: The project as it is.

    Returns:
        `removed`, `added`, `replaced` and `edges_moved`, the length of each
        piece added to or taken from a stretch both show, in seconds.
    """
    old, new = _footage(before), _footage(after)
    edges: List[float] = []
    gone: List[Stretch] = []
    come: List[Stretch] = []
    for ours, theirs, whole in ((old, new, gone), (new, old, come)):
        for asset_id, stretches in ours.items():
            for stretch in stretches:
                pieces = _left_over(stretch, theirs.get(asset_id, []))
                if pieces is None:
                    whole.append(stretch)
                else:
                    edges.extend(pieces)
    swapped = _pairs(len(gone), len(come), lambda index, other: _on_timeline_together(gone[index], come[other]))
    return {"removed": len(gone) - len(swapped), "added": len(come) - len(swapped), "replaced": len(swapped),
            "edges_moved": edges}

def _same_words(one: SubtitleCue, other: SubtitleCue) -> float:
    """How many seconds of the same file two captions both cover."""
    if one.asset_id != other.asset_id:
        return 0.0
    return max(0.0, float(min(one.source_end, other.source_end) - max(one.source_start, other.source_start)))

def _within(part: SubtitleCue, whole: SubtitleCue) -> bool:
    """Whether a caption's middle falls inside another's stretch of the same file."""
    middle = (part.source_start + part.source_end) / 2
    return part.asset_id == whole.asset_id and whole.source_start <= middle <= whole.source_end

def _in_cut(cue: SubtitleCue, footage: Dict[str, List[Stretch]]) -> bool:
    """Whether the cut shows any of the words a caption is for."""
    return any(min(float(cue.source_end), stretch.end) - max(float(cue.source_start), stretch.start) > TOUCHING
               for stretch in footage.get(cue.asset_id, []))

def _tight(text: str) -> str:
    """Text with every space, tab and line break taken out, for comparing what was said."""
    return "".join(text.split())

def _broken_differently(ones: List[SubtitleCue], others: List[SubtitleCue], left: set, other_left: set) -> int:
    """Find each caption split into several, or several joined into one, with the same words.

    Every caption found is taken out of `left` and `other_left`.

    Args:
        ones: The captions on one side.
        others: The captions on the other side.
        left: Indexes of `ones` not yet accounted for.
        other_left: Indexes of `others` not yet accounted for.

    Returns:
        How many captions of `ones` were broken into several of `others`.
    """
    found = 0
    for index in sorted(left):
        whole = ones[index]
        parts = sorted((other for other in other_left if _within(others[other], whole)),
                       key=lambda other: others[other].source_start)
        if len(parts) > 1 and "".join(_tight(others[part].text) for part in parts) == _tight(whole.text):
            found += 1
            left.discard(index)
            other_left.difference_update(parts)
    return found

def _caption_changes(before: Project, after: Project) -> Dict[str, int]:
    """How the captions changed, matched by what they say where rather than by ID.

    A caption split into several with the same words, or several joined into
    one, is re-broken, once. The rest are paired by the most shared seconds of
    the same file, each in at most one pair, which survives captions being
    made again and renumbered. A paired caption whose words changed is
    rewritten, or only re-broken when nothing but its spaces and line breaks
    changed; one whose words are the same but whose times moved is retimed.
    An unpaired caption with exactly the words of an unpaired one on the other
    side, from the same file, was moved: retimed.

    What is left over counts only where it corrects the captions of footage
    the cut kept: a caption removed while the cut still shows its words (a
    line that was never said), and a caption added for words the cut already
    showed while it already had captions (a line that was missed). A caption
    that comes or goes with its footage, and the first captions a cut is
    given, are not corrections and are not counted.

    Args:
        before: The project as it was.
        after: The project as it is.

    Returns:
        `retimed`, `rewritten`, `rebroken`, `added` and `removed`.
    """
    old, new = list(before.subtitles), list(after.subtitles)
    counts = {"retimed": 0, "rewritten": 0, "rebroken": 0, "added": 0, "removed": 0}
    old_left, new_left = set(range(len(old))), set(range(len(new)))
    counts["rebroken"] += _broken_differently(old, new, old_left, new_left)
    counts["rebroken"] += _broken_differently(new, old, new_left, old_left)
    olds, news = sorted(old_left), sorted(new_left)
    for index, other_index in _pairs(len(olds), len(news),
                                     lambda one, other: _same_words(old[olds[one]], new[news[other]])):
        cue, other = old[olds[index]], new[news[other_index]]
        old_left.discard(olds[index])
        new_left.discard(news[other_index])
        if cue.text != other.text:
            counts["rebroken" if _tight(cue.text) == _tight(other.text) else "rewritten"] += 1
        elif (cue.source_start, cue.source_end) != (other.source_start, other.source_end):
            counts["retimed"] += 1
    for index in sorted(old_left):
        moved = next((other for other in sorted(new_left) if new[other].asset_id == old[index].asset_id
                      and _tight(new[other].text) == _tight(old[index].text)), None)
        if moved is not None:
            counts["retimed"] += 1
            old_left.discard(index)
            new_left.discard(moved)
    was_shown, now_shown = _footage(before), _footage(after)
    counts["removed"] = sum(1 for index in old_left if _in_cut(old[index], now_shown))
    if old:
        counts["added"] = sum(1 for index in new_left if _in_cut(new[index], was_shown))
    return counts

def compare(before: Project, after: Project) -> Dict[str, Any]:
    """What changed between two versions of a project, counted.

    Only the cut is compared: the footage the base video track shows, and the
    captions. Title cards, music beds, B-roll and anything else beside the cut
    are not counted. See `_clip_changes` and `_caption_changes` for how each
    is matched.

    Args:
        before: The project as it was.
        after: The project as it is.

    Returns:
        `clips` with `removed`, `added`, `replaced` and `edges_moved` (the
        lengths of footage added to or taken from a stretch both versions
        show, in seconds), and `captions` with `retimed`, `rewritten`,
        `rebroken`, `added` and `removed`.
    """
    return {"clips": _clip_changes(before, after), "captions": _caption_changes(before, after)}

def classify(change: Mapping) -> Dict[str, int]:
    """Count a change by cause, by rule.

    Args:
        change: What `compare` returned.

    Returns:
        `misheard`, `line_breaks`, `caption_timing`, `cut_point` and
        `wrong_piece`, each a count. `misheard` is a caption rewritten, or
        one removed or added while its footage stayed in the cut; `cut_point`
        is an edge moved by at most `SMALL_EDGE_SECONDS`; `wrong_piece` is
        footage replaced, removed or added, or an edge moved further than
        that.
    """
    clips = change.get("clips") or {}
    captions = change.get("captions") or {}
    moves = [float(move) for move in clips.get("edges_moved") or []]
    small = sum(1 for move in moves if move <= SMALL_EDGE_SECONDS)
    large = len(moves) - small
    return {
        "misheard": (int(captions.get("rewritten", 0)) + int(captions.get("removed", 0))
                     + int(captions.get("added", 0))),
        "line_breaks": int(captions.get("rebroken", 0)),
        "caption_timing": int(captions.get("retimed", 0)),
        "cut_point": small,
        "wrong_piece": (int(clips.get("replaced", 0)) + int(clips.get("removed", 0))
                        + int(clips.get("added", 0)) + large),
    }

def _summarise_changes(changes: Sequence[Mapping]) -> Dict[str, Any]:
    """Sum a run of compared changes into counts and edge-move statistics.

    Args:
        changes: Each a `compare` result.

    Returns:
        `versions`, the summed `clips` counts, the summed `captions` counts,
        the `causes` from `classify`, and `edge_moves` with `count`, `median`
        and `largest`.
    """
    clips: Dict[str, int] = {"removed": 0, "added": 0, "replaced": 0}
    captions: Dict[str, int] = {"retimed": 0, "rewritten": 0, "rebroken": 0, "added": 0, "removed": 0}
    causes: Dict[str, int] = {cause: 0 for cause in CAUSES}
    moves: List[float] = []
    for change in changes:
        for key in clips:
            clips[key] += int((change.get("clips") or {}).get(key, 0))
        for key in captions:
            captions[key] += int((change.get("captions") or {}).get(key, 0))
        for cause, count in classify(change).items():
            causes[cause] += count
        moves.extend(float(move) for move in (change.get("clips") or {}).get("edges_moved") or [])
    return {
        "versions": len(changes),
        "clips": clips,
        "captions": captions,
        "causes": causes,
        "edge_moves": {
            "count": len(moves),
            "median": round(float(statistics.median(moves)), 3) if moves else 0.0,
            "largest": round(max(moves), 3) if moves else 0.0,
        },
    }

def _hand_minutes(versions: List[Version]) -> float:
    """Estimate minutes spent editing by hand from the times of its versions.

    A sitting runs from its first version to its last, and counts at least
    `MINUTES_PER_SITTING_FLOOR`; a gap of more than `SITTING_GAP_SECONDS`
    starts a new one. This is an estimate from timestamps, not a measurement.

    Args:
        versions: The versions made by hand, oldest first.

    Returns:
        The estimated minutes, to one decimal place.
    """
    times = sorted(version.when for version in versions)
    if not times:
        return 0.0
    total = 0.0
    start = previous = times[0]
    for when in times[1:]:
        if (when - previous).total_seconds() > SITTING_GAP_SECONDS:
            total += (previous - start).total_seconds() / 60 + MINUTES_PER_SITTING_FLOOR
            start = when
        previous = when
    total += (previous - start).total_seconds() / 60 + MINUTES_PER_SITTING_FLOOR
    return round(total, 1)

def _changes_since_first_cut(
    versions: List[Tuple[Version, Optional[dict]]],
) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Version], int]]:
    """Compare every version after the AI's first cut with the one before it.

    A version that cannot be read as a project is skipped: the next one that
    can is compared with the last one that could.

    Args:
        versions: A project's versions, oldest first.

    Returns:
        The compared changes made by hand, those made by the AI, the versions
        made by hand, and how many versions after the first cut could not be
        read — or None when the project has no first cut. The AI's are every
        AI version after the first cut: those the person asked for, and those
        the AI went on to make by itself (captions, fixes the checks asked
        for), which the history does not tell apart. Versions written by the
        server itself are neither, but the next version is compared with them.
    """
    read = _read(versions)
    start = _first(read)
    if start is None:
        return None
    by_hand: List[Dict[str, Any]] = []
    by_ai: List[Dict[str, Any]] = []
    hand_versions: List[Version] = []
    unreadable = 0
    previous = read[start][1]
    for version, project in read[start + 1:]:
        if project is None:
            unreadable += 1
            continue
        change = compare(previous, project)
        previous = project
        if version.author == EDITOR_AUTHOR:
            by_hand.append(change)
            hand_versions.append(version)
        elif version.author.startswith("AI"):
            by_ai.append(change)
    return by_hand, by_ai, hand_versions, unreadable

def _name_of(versions: List[Tuple[Version, Optional[dict]]]) -> str:
    """The last name a project was given, or an empty string."""
    for _, project in reversed(versions):
        if project and project.get("name"):
            return str(project["name"])
    return ""

def _project_result(
    versions: List[Tuple[Version, Optional[dict]]],
    by_hand: List[Dict[str, Any]],
    by_ai: List[Dict[str, Any]],
    hand_versions: List[Version],
    unreadable: int,
) -> Dict[str, Any]:
    """One project's summary from its compared changes."""
    combined = _summarise_changes([*by_hand, *by_ai])
    return {
        "name": _name_of(versions),
        "unreadable": unreadable,
        "by_hand": _summarise_changes(by_hand),
        "by_ai": _summarise_changes(by_ai),
        "causes": combined["causes"],
        "edge_moves": combined["edge_moves"],
        "hand_minutes": _hand_minutes(hand_versions),
    }

def summarise_project(versions: List[Tuple[Version, Optional[dict]]]) -> Optional[Dict[str, Any]]:
    """Summarise the corrections made to one project after the AI's first cut.

    Args:
        versions: The project's versions, oldest first, each with the project
            as it was then.

    Returns:
        `name`, `by_hand` and `by_ai` (each the counts from
        `_summarise_changes`), the two together as `causes` and `edge_moves`,
        `hand_minutes` — an estimate from the versions' timestamps, not a
        measurement — and `unreadable`, how many versions after the first cut
        could not be read and were skipped; None when the project has no
        first cut.
    """
    changes = _changes_since_first_cut(versions)
    if changes is None:
        return None
    return _project_result(versions, *changes)

def _kinds_of(versions: List[Tuple[Version, Optional[dict]]], kinds: Mapping[str, List[str]]) -> List[str]:
    """The footage kinds a project was cut from, by the files its cut shows."""
    assets = {clip.asset_id for _, project in _read(versions) if project is not None
              for clip in _footage_clips(project)}
    found = {kind for asset_id in assets for kind in kinds.get(asset_id, [])}
    return sorted(found)

def _totals(
    by_hand: List[Dict[str, Any]],
    by_ai: List[Dict[str, Any]],
    hand_versions: List[Version],
) -> Dict[str, Any]:
    """The counts for a set of projects together."""
    combined = _summarise_changes([*by_hand, *by_ai])
    return {
        "by_hand": _summarise_changes(by_hand),
        "by_ai": _summarise_changes(by_ai),
        "causes": combined["causes"],
        "edge_moves": combined["edge_moves"],
        "hand_minutes": _hand_minutes(hand_versions),
    }

def summarise(
    projects: Dict[str, List[Tuple[Version, Optional[dict]]]],
    kinds: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Summarise the corrections made after the AI's first cut, per project and in total.

    Args:
        projects: Each project's ID, with its versions oldest first.
        kinds: Each asset's footage kinds, such as `{"<asset>": ["訪談"]}`.
            When given, the totals are also grouped by kind; when left out,
            `by_kind` is None, since no kinds can be invented.

    Returns:
        `projects` (each ID with its summary, or None when it has no first
        cut), `totals`, and `by_kind` — the totals per footage kind, or None.
    """
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    all_by_hand: List[Dict[str, Any]] = []
    all_by_ai: List[Dict[str, Any]] = []
    all_hands: List[Version] = []
    grouped: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Version]]] = {}
    for project_id, versions in projects.items():
        changes = _changes_since_first_cut(versions)
        if changes is None:
            results[project_id] = None
            continue
        by_hand, by_ai, hands, unreadable = changes
        results[project_id] = _project_result(versions, by_hand, by_ai, hands, unreadable)
        all_by_hand.extend(by_hand)
        all_by_ai.extend(by_ai)
        all_hands.extend(hands)
        if kinds is not None:
            for kind in _kinds_of(versions, kinds):
                gathered_by_hand, gathered_by_ai, gathered_hands = grouped.setdefault(kind, ([], [], []))
                gathered_by_hand.extend(by_hand)
                gathered_by_ai.extend(by_ai)
                gathered_hands.extend(hands)
    by_kind = None
    if kinds is not None:
        by_kind = {kind: _totals(*grouped[kind]) for kind in sorted(grouped)}
    return {
        "projects": results,
        "totals": _totals(all_by_hand, all_by_ai, all_hands),
        "by_kind": by_kind,
    }

def _display_width(text: str) -> int:
    """How many columns a string takes, counting a wide character as two."""
    return sum(2 if unicodedata.east_asian_width(character) in "WF" else 1 for character in text)

def _pad(text: str, width: int) -> str:
    """Pad a string on the right to a column width."""
    return text + " " * max(0, width - _display_width(text))

def _table(headers: List[str], rows: List[List[str]]) -> str:
    """Lay out a plain-text table with columns wide enough for their contents."""
    widths = [max(_display_width(row[column]) for row in [headers, *rows]) for column in range(len(headers))]
    lines = ["  ".join(_pad(cell, widths[column]) for column, cell in enumerate(row)).rstrip()
             for row in [headers, *rows]]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)

def _row(label: str, summary: Mapping[str, Any], causes: List[str]) -> List[str]:
    """One project's or the total's cells in the report."""
    by_hand = summary.get("by_hand") or {}
    by_ai = summary.get("by_ai") or {}
    counts = summary.get("causes") or {}
    moves = summary.get("edge_moves") or {}
    return [
        label,
        str(by_hand.get("versions", 0)),
        str(by_ai.get("versions", 0)),
        *(str(counts.get(cause, 0)) for cause in causes),
        f"{float(moves.get('median', 0)):.2f}",
        f"{float(moves.get('largest', 0)):.2f}",
        f"{float(summary.get('hand_minutes', 0)):.1f}",
    ]

def report(summary: Mapping[str, Any]) -> str:
    """Say a summary as a plain-text table in Chinese.

    Args:
        summary: What `summarise` returned.

    Returns:
        The table: per project and in total, versions by hand and by
        the AI, the counts by cause, edge-move median and largest, and the
        estimated minutes spent editing by hand. Causes are ordered by their
        total count, largest first. A line follows for each project that had
        versions that could not be read.
    """
    totals = summary.get("totals") or {}
    causes = sorted(CAUSES, key=lambda cause: -int((totals.get("causes") or {}).get(cause, 0)))
    headers = ["專案", "手動版本", "AI 版本", *(CAUSE_LABELS[cause] for cause in causes),
               "移動中位數", "最大移動", "手動分鐘"]
    rows: List[List[str]] = []
    projects = summary.get("projects") or {}
    for project_id, one in projects.items():
        if one is None:
            continue
        rows.append(_row(one.get("name") or project_id, one, causes))
    rows.append(_row("合計", totals, causes))
    skipped = [f"{one.get('name') or project_id}：{one['unreadable']} 個版本讀不出來，跳過沒有比較。"
               for project_id, one in projects.items() if one is not None and one.get("unreadable")]
    return "\n".join([
        "AI 第一次剪完之後改了什麼：手動的與 AI 的版本都算（依規則計數，沒有模型在裡面）",
        _table(headers, rows),
        "手動分鐘是從版本的時間戳推算的估計，不是量出來的；節奏與剪在字中間無法用這些規則分辨。",
        "只比較主軌畫面用了檔案的哪幾段與字幕：標題卡、音樂、B-roll 不算；字幕跟著畫面一起進出不算修正。",
        *skipped,
    ])

def main(argv: Optional[List[str]] = None) -> None:
    """Read the workspace's projects and print the corrections summary.

    Reads only: nothing is written and no job is started.

    Args:
        argv: Arguments after the module name; the process's own when not given.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.engine.corrections",
        description="Summarise what people changed after the AI's first cut, by rule.",
    )
    parser.add_argument("--project", action="append", dest="project_ids", metavar="ID",
                        help="only this project; may be given more than once (every project when omitted)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON instead of a table")
    arguments = parser.parse_args(argv)

    from app import runtime
    from app.tools.versioning import project_json_at

    project_ids = arguments.project_ids or [project.id for project in runtime.repo.list_projects()]
    projects: Dict[str, List[Tuple[Version, Optional[dict]]]] = {}
    for project_id in project_ids:
        newest_first = runtime.history.versions_of("projects/", f"{project_id}.json", VERSION_LIMIT)
        projects[project_id] = []
        for version in reversed(newest_first):
            try:
                stored = project_json_at(project_id, version.commit)
            except ValueError:
                stored = None  # Not even JSON: counted as unreadable, like a project that fails to validate.
            projects[project_id].append((version, stored))
    summary = summarise(projects)
    if arguments.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(report(summary))

if __name__ == "__main__":
    main()
