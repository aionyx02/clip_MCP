"""What people corrected after the AI's first cut, counted by rule.

Every version of a project is compared with the one before it, once the AI
has made a first cut that has clips in it. A change is counted against the
piece of the cut it touched: a caption heard wrong, a line broken in another
place, a caption put at another time, a cut point nudged, or a different
stretch of footage chosen.

The rules are deliberately plain. Rhythm (節奏) is not told apart from a cut
point: an edge moved to tighten a pause and an edge moved because the take
was wrong look the same here, and which one it was is a judgement about the
footage. Nor is a cut that lands inside a word told apart from one that lands
in a pause: that needs the transcript, which these functions do not read. So
neither is guessed at; both are left to be read from the footage.

Only the cut is compared: the clips on the base video track, and the
captions. Music, B-roll and anything else beside the cut are not counted.
Footage is matched by what it shows, not by its clip ID: a recompile names
every clip after its place, and a split leaves two clips where one was. So a
move along the timeline and a change of speed are not counted — only which
footage is shown. A restore or an undo is a new version like any other, so
what it puts back is counted again, under whoever restored it. A branch keeps
its own history, so hand edits made on a branch before the branch's first AI
version with clips in it are not counted.

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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from app.engine.compare import match_clips, shown
from app.models.timeline.parts import TOUCHING_SECONDS, Clip, Project, SubtitleCue, TimeRange
from app.storage.history import Version

# Who the editor app writes as. Said here rather than imported from the UI, which
# would drag the web app into a command that only reads the history.
EDITOR_AUTHOR = "編輯器"
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

def _first(read: List[Tuple[Version, Optional[Project]]]) -> Optional[int]:
    """The index of the first AI version whose cut has clips in it, or None."""
    for index, (version, project) in enumerate(read):
        if version.author.startswith("AI") and project is not None and shown(project):
            return index
    return None

def first_cut(versions: List[Tuple[Version, Optional[dict]]]) -> Optional[int]:
    """The first version the AI made a cut in.

    Args:
        versions: A project's versions, oldest first, each with the project as
            it was then.

    Returns:
        The index of the first version written by an AI author whose project
        can be read and has at least one clip in its cut (the base video
        track), or None when there is no such version.
    """
    return _first(_read(versions))

def _continues(one: Clip, other: Clip) -> bool:
    """Whether a clip picks up exactly where the one before it left off, in its file and on the timeline."""
    return (one.card is None and other.card is None and one.asset_id == other.asset_id
            and one.speed == other.speed
            and abs(other.source_range.start - one.source_range.end) <= TOUCHING_SECONDS
            and abs(other.timeline_in - one.timeline_out) <= TOUCHING_SECONDS)

def _joined(clips: List[Clip]) -> List[Clip]:
    """The cut's clips with each unbroken run of one file joined into one clip.

    A split leaves two clips that together show what the one did; joined, a
    split compares as no change at all.

    Args:
        clips: The base track's clips, in timeline order.

    Returns:
        The clips, each run that `_continues` joined made into one.
    """
    runs: List[Clip] = []
    for clip in clips:
        if runs and _continues(runs[-1], clip):
            last = runs[-1]
            runs[-1] = last.model_copy(update={
                "source_range": TimeRange(start=last.source_range.start, end=clip.source_range.end),
            })
        else:
            runs.append(clip)
    return runs

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

def _on_timeline_together(one: Clip, other: Clip) -> float:
    """How many seconds of the timeline two clips both cover."""
    return max(0.0, float(min(one.timeline_out, other.timeline_out) - max(one.timeline_in, other.timeline_in)))

def _same_words(one: SubtitleCue, other: SubtitleCue) -> float:
    """How many seconds of the same file two captions both cover."""
    if one.asset_id != other.asset_id:
        return 0.0
    return max(0.0, float(min(one.source_end, other.source_end) - max(one.source_start, other.source_start)))

def _tight(text: str) -> str:
    """Text with every space, tab and line break taken out, for comparing what was said."""
    return "".join(text.split())

def compare(before: Project, after: Project) -> Dict[str, Any]:
    """What changed between two versions of a project, counted.

    Only the cut is compared: the clips on the base video track. Music beds,
    B-roll and anything else above or beside it are not corrections to the
    cut and are not counted. Clips are matched by the footage they show, not
    by their ID, which a recompile renames: first each unbroken run of one
    file is joined (see `_joined`), so a split counts as nothing; then the
    clips are paired by `compare.match_clips`. A pair's source edges that
    moved are listed by how far each moved. A clip left unpaired before and
    one left unpaired after that cover the same stretch of the timeline count
    once, as replaced; the rest were removed or added. Moving a clip along the
    timeline or changing its speed is not counted: only which footage is shown.

    Captions are paired by what they say where: the most shared seconds of
    the same file first, each caption in at most one pair, which survives
    captions being made again and renumbered. A paired caption whose words
    changed is rewritten, or only re-broken when nothing but its whitespace
    and line breaks changed; one whose words are the same but whose times
    moved is retimed. Unpaired ones were removed or added.

    Args:
        before: The project as it was.
        after: The project as it is.

    Returns:
        `clips` with `removed`, `added`, `replaced` and `edges_moved` (the
        absolute moves of changed source edges, in seconds), and `captions`
        with `retimed`, `rewritten`, `rebroken`, `added` and `removed`.
    """
    old, new = _joined(shown(before)), _joined(shown(after))
    clips: Dict[str, Any] = {"removed": 0, "added": 0, "replaced": 0, "edges_moved": []}
    paired = match_clips(old, new)
    for clip, other in paired:
        for was, now in ((clip.source_range.start, other.source_range.start),
                         (clip.source_range.end, other.source_range.end)):
            if was != now:
                clips["edges_moved"].append(round(abs(float(now - was)), 3))
    gone = [clip for clip in old if all(clip is not one for one, _ in paired)]
    come = [clip for clip in new if all(clip is not other for _, other in paired)]
    swapped = _pairs(len(gone), len(come), lambda index, other: _on_timeline_together(gone[index], come[other]))
    clips["replaced"] = len(swapped)
    clips["removed"] = len(gone) - len(swapped)
    clips["added"] = len(come) - len(swapped)

    captions: Dict[str, int] = {"retimed": 0, "rewritten": 0, "rebroken": 0, "added": 0, "removed": 0}
    was_said, now_said = before.subtitles, after.subtitles
    matched = _pairs(len(was_said), len(now_said),
                     lambda index, other: _same_words(was_said[index], now_said[other]))
    for index, other_index in matched:
        cue, other = was_said[index], now_said[other_index]
        if cue.text != other.text:
            if _tight(cue.text) == _tight(other.text):
                captions["rebroken"] += 1
            else:
                captions["rewritten"] += 1
        elif (cue.source_start, cue.source_end) != (other.source_start, other.source_end):
            captions["retimed"] += 1
    captions["removed"] = len(was_said) - len(matched)
    captions["added"] = len(now_said) - len(matched)

    return {"clips": clips, "captions": captions}

def classify(change: Mapping) -> Dict[str, int]:
    """Count a change by cause, by rule.

    Args:
        change: What `compare` returned.

    Returns:
        `misheard`, `line_breaks`, `caption_timing`, `cut_point` and
        `wrong_piece`, each a count. `cut_point` is an edge moved by at most
        `SMALL_EDGE_SECONDS`; `wrong_piece` is a clip replaced, removed or
        added, or an edge moved further than that.
    """
    clips = change.get("clips") or {}
    captions = change.get("captions") or {}
    moves = [float(move) for move in clips.get("edges_moved") or []]
    small = sum(1 for move in moves if move <= SMALL_EDGE_SECONDS)
    large = len(moves) - small
    return {
        "misheard": int(captions.get("rewritten", 0)),
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
    """The footage kinds a project was cut from, by the files of the clips in its cut."""
    assets = {clip.asset_id for _, project in _read(versions) if project is not None for clip in shown(project)}
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
