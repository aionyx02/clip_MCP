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

Reading the history lives in `main`, which imports the workspace only when it
is called, so importing this module stays light and pure.
"""

import argparse
import json
import statistics
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

def _clips_of(project: Optional[Mapping]) -> List[Mapping]:
    """Every clip in a project, from every track.

    Args:
        project: The project as its file holds it, or None.

    Returns:
        The clips, in the order the tracks and their clips are stored.
    """
    return [clip for track in (project or {}).get("tracks", []) for clip in track.get("clips", [])]

def first_cut(versions: List[Tuple[Version, Optional[dict]]]) -> Optional[int]:
    """The first version the AI made a cut in.

    Args:
        versions: A project's versions, oldest first, each with the project as
            it was then.

    Returns:
        The index of the first version written by an AI author whose project
        has at least one clip, or None when there is no such version.
    """
    for index, (version, project) in enumerate(versions):
        if version.author.startswith("AI") and _clips_of(project):
            return index
    return None

def _seconds(value: Any) -> float:
    """A stored time as a float, or 0 when it is missing or unreadable."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def _edges(clip: Mapping) -> Tuple[float, float]:
    """The two source edges of a clip, start and end."""
    source = clip.get("source_range") or {}
    return _seconds(source.get("start")), _seconds(source.get("end"))

def _clip_map(project: Optional[Mapping]) -> Dict[str, Mapping]:
    """A project's clips by their ID."""
    return {str(clip.get("id")): clip for clip in _clips_of(project)}

def _cue_map(project: Optional[Mapping]) -> Dict[str, Mapping]:
    """A project's captions by their ID."""
    return {str(cue.get("id", "")): cue for cue in (project or {}).get("subtitles", [])}

def _tight(text: Any) -> str:
    """Text with every space, tab and line break taken out, for comparing what was said."""
    return "".join(str(text or "").split())

def _times(cue: Mapping) -> Tuple[float, float]:
    """The source stretch a caption covers."""
    return _seconds(cue.get("source_start")), _seconds(cue.get("source_end"))

def compare(before: Optional[Mapping], after: Optional[Mapping]) -> Dict[str, Any]:
    """What changed between two versions of a project, counted.

    Clips are matched by their ID. A clip with the same ID but another file is
    replaced; one that is only in one version was removed or added; for the
    rest, every source edge that moved is listed by how far it moved.

    Captions are matched by their ID. A caption whose words changed is
    rewritten, or only re-broken when nothing but its whitespace and line
    breaks changed; one whose words are the same but whose times moved is
    retimed.

    Args:
        before: The project as it was.
        after: The project as it is.

    Returns:
        `clips` with `removed`, `added`, `replaced` and `edges_moved` (the
        absolute moves of changed source edges, in seconds), and `captions`
        with `retimed`, `rewritten`, `rebroken`, `added` and `removed`.
    """
    clips: Dict[str, Any] = {"removed": 0, "added": 0, "replaced": 0, "edges_moved": []}
    old, new = _clip_map(before), _clip_map(after)
    for clip_id, clip in old.items():
        if clip_id not in new:
            clips["removed"] += 1
            continue
        other = new[clip_id]
        if clip.get("asset_id") != other.get("asset_id"):
            clips["replaced"] += 1
            continue
        for was, now in zip(_edges(clip), _edges(other)):
            if abs(now - was) > 1e-9:
                clips["edges_moved"].append(round(abs(now - was), 3))
    for clip_id in new:
        if clip_id not in old:
            clips["added"] += 1

    captions: Dict[str, int] = {"retimed": 0, "rewritten": 0, "rebroken": 0, "added": 0, "removed": 0}
    before_cues, after_cues = _cue_map(before), _cue_map(after)
    for cue_id, cue in before_cues.items():
        if cue_id not in after_cues:
            captions["removed"] += 1
            continue
        other = after_cues[cue_id]
        was, now = str(cue.get("text") or ""), str(other.get("text") or "")
        if was != now:
            if _tight(was) == _tight(now):
                captions["rebroken"] += 1
            else:
                captions["rewritten"] += 1
        elif _times(cue) != _times(other):
            captions["retimed"] += 1
    for cue_id in after_cues:
        if cue_id not in before_cues:
            captions["added"] += 1

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
) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Version]]]:
    """Compare every version after the AI's first cut with the one before it.

    Args:
        versions: A project's versions, oldest first.

    Returns:
        The compared changes made by hand, those made by the AI, and the
        versions made by hand — or None when the project has no first cut.
        The AI's are every AI version after the first cut: those the person
        asked for, and those the AI went on to make by itself (captions,
        fixes the checks asked for), which the history does not tell apart.
        Versions written by the server itself are neither.
    """
    start = first_cut(versions)
    if start is None:
        return None
    by_hand: List[Dict[str, Any]] = []
    by_ai: List[Dict[str, Any]] = []
    hand_versions: List[Version] = []
    for index in range(start + 1, len(versions)):
        version, project = versions[index]
        change = compare(versions[index - 1][1] or {}, project or {})
        if version.author == EDITOR_AUTHOR:
            by_hand.append(change)
            hand_versions.append(version)
        elif version.author.startswith("AI"):
            by_ai.append(change)
    return by_hand, by_ai, hand_versions

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
) -> Dict[str, Any]:
    """One project's summary from its compared changes."""
    combined = _summarise_changes([*by_hand, *by_ai])
    return {
        "name": _name_of(versions),
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
        and `hand_minutes` — an estimate from the versions' timestamps, not a
        measurement; None when the project has no first cut.
    """
    changes = _changes_since_first_cut(versions)
    if changes is None:
        return None
    return _project_result(versions, *changes)

def _kinds_of(versions: List[Tuple[Version, Optional[dict]]], kinds: Mapping[str, List[str]]) -> List[str]:
    """The footage kinds a project was cut from, by its clips' assets."""
    assets = {str(clip.get("asset_id")) for _, project in versions for clip in _clips_of(project)
              if clip.get("asset_id")}
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
        by_hand, by_ai, hands = changes
        results[project_id] = _project_result(versions, by_hand, by_ai, hands)
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
        total count, largest first.
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
    return "\n".join([
        "AI 第一次剪完之後改了什麼：手動的與 AI 的版本都算（依規則計數，沒有模型在裡面）",
        _table(headers, rows),
        "手動分鐘是從版本的時間戳推算的估計，不是量出來的；節奏與剪在字中間無法用這些規則分辨。",
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
        projects[project_id] = [
            (version, project_json_at(project_id, version.commit)) for version in reversed(newest_first)
        ]
    summary = summarise(projects)
    if arguments.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(report(summary))

if __name__ == "__main__":
    main()
