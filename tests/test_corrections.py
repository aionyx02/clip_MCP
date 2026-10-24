"""The corrections people made after the AI's first cut are counted by rule, with no model in it."""

import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.engine import corrections
from app.storage.history import Version

# Distinct commit IDs for the versions built in memory; the history only needs them to differ.
_COMMITS = itertools.count(1)
BASE = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)


def at(minutes: float) -> datetime:
    """A moment some minutes after the base one."""
    return BASE + timedelta(minutes=minutes)


def clip(clip_id: str, asset: str = "a", start: float = 0.0, end: float = 1.0,
         timeline_in: float = 0.0) -> dict:
    """A clip as a project's file holds it."""
    return {"id": clip_id, "asset_id": asset, "source_range": {"start": start, "end": end},
            "timeline_in": timeline_in}


def cue(cue_id: str, text: str, start: float = 0.0, end: float = 1.0, asset: str = "a") -> dict:
    """A caption as a project's file holds it."""
    return {"id": cue_id, "asset_id": asset, "source_start": start, "source_end": end, "text": text}


def project(*clips: dict, cues: tuple = ()) -> dict:
    """A project with one video track and the given clips and captions."""
    return {"name": "測試", "tracks": [{"id": "main", "track_type": "video", "clips": list(clips)}],
            "subtitles": list(cues)}


def version(author: str, when: datetime, project_dict: dict | None) -> tuple:
    """One version of a project, as a history reports it, with the project as it was then."""
    record = Version(commit=f"{next(_COMMITS):040x}", when=when, author=author, subject="", body="",
                     parents=())
    return record, project_dict


def test_first_cut_is_the_first_ai_version_that_has_a_clip() -> None:
    versions = [
        version("編輯器", at(0), project()),
        version("AI（Claude Code）", at(1), project()),
        version("自動", at(2), project(clip("a"))),
        version("AI（Claude Code）", at(3), project(clip("b"))),
    ]
    assert corrections.first_cut(versions) == 3
    assert corrections.first_cut(versions[:3]) is None
    assert corrections.first_cut([]) is None


def test_compare_counts_removed_added_replaced_and_both_edge_moves() -> None:
    before = project(clip("a", asset="a"), clip("b", start=1.0, end=2.0), clip("c"))
    after = project(clip("a", asset="b"), clip("b", start=1.5, end=2.25), clip("d"))
    changed = corrections.compare(before, after)
    assert changed["clips"] == {"removed": 1, "added": 1, "replaced": 1, "edges_moved": [0.5, 0.25]}


def test_compare_counts_each_kind_of_caption_change() -> None:
    before = project(cues=(
        cue("c1", "你好世界", 0.0, 1.0),
        cue("c2", "再見", 1.0, 2.0),
        cue("c3", "刪掉", 2.0, 3.0),
        cue("c4", "換行", 3.0, 4.0),
        cue("c5", "沒變", 4.0, 5.0),
    ))
    after = project(cues=(
        cue("c1", "你好世界", 0.5, 1.5),
        cue("c2", "珍重", 1.0, 2.0),
        cue("c4", "換\n行", 3.0, 4.0),
        cue("c5", "沒變", 4.0, 5.0),
        cue("c6", "新加的", 5.0, 6.0),
    ))
    assert corrections.compare(before, after)["captions"] == {
        "retimed": 1, "rewritten": 1, "rebroken": 1, "added": 1, "removed": 1,
    }


def test_classify_splits_edge_moves_on_the_small_edge_threshold() -> None:
    changed = {
        "clips": {"removed": 1, "added": 2, "replaced": 3, "edges_moved": [
            corrections.SMALL_EDGE_SECONDS, 0.2, corrections.SMALL_EDGE_SECONDS + 0.01,
        ]},
        "captions": {"rewritten": 4, "rebroken": 5, "retimed": 6, "added": 0, "removed": 0},
    }
    assert corrections.classify(changed) == {
        "misheard": 4, "line_breaks": 5, "caption_timing": 6, "cut_point": 2, "wrong_piece": 7,
    }


def test_summarise_project_splits_by_author_skips_the_server_and_counts_sittings() -> None:
    versions = [
        version("編輯器", at(0), project()),
        version("AI（Claude Code）", at(1), project(clip("a", end=1.0))),
        version("AI（Claude Code）", at(2), project(clip("a", start=0.5, end=1.0))),
        version("自動", at(3), project(clip("a", start=0.5, end=1.0), clip("z"))),
        version("編輯器", at(10), project(clip("a", start=0.5, end=1.5))),
        version("編輯器", at(13), project(clip("a", start=0.5, end=1.5), clip("b"))),
        version("編輯器", at(40), project(clip("a", start=0.5, end=1.5), clip("b"), clip("c"))),
    ]
    summary = corrections.summarise_project(versions)
    assert summary is not None
    # The AI's second change is 0.5s; the server's version is neither person nor request.
    assert summary["by_ai"]["versions"] == 1
    assert summary["by_ai"]["causes"]["cut_point"] == 1
    # Three changes by hand: a nudge (which also took back the clip the server added),
    # then one clip put in and another.
    assert summary["by_hand"]["versions"] == 3
    assert summary["by_hand"]["causes"] == {
        "misheard": 0, "line_breaks": 0, "caption_timing": 0, "cut_point": 1, "wrong_piece": 3,
    }
    assert summary["by_hand"]["edge_moves"] == {"count": 1, "median": 0.5, "largest": 0.5}
    assert summary["causes"]["wrong_piece"] == 3
    # Two sittings: 10 to 13 minutes is 3 + 1, the change at 40 is 1 on its own.
    assert summary["hand_minutes"] == 5.0


def test_summarise_project_is_none_without_a_first_cut() -> None:
    versions = [version("編輯器", at(0), project(clip("a")))]
    assert corrections.summarise_project(versions) is None


def _asset_project(asset: str) -> list:
    """A project cut from one file, with one change made by hand."""
    return [
        version("AI（Claude Code）", at(1), project(clip("a", asset=asset))),
        version("編輯器", at(2), project(clip("a", asset=asset), clip("b", asset=asset))),
    ]


def test_summarise_groups_totals_by_kind_only_when_kinds_are_given() -> None:
    projects = {"p1": _asset_project("a"), "p2": _asset_project("b")}
    without = corrections.summarise(projects)
    assert without["by_kind"] is None
    assert without["totals"]["by_hand"]["versions"] == 2
    assert without["totals"]["causes"]["wrong_piece"] == 2

    kinds = {"a": ["訪談"], "b": ["風景"]}
    grouped = corrections.summarise(projects, kinds=kinds)
    assert set(grouped["by_kind"]) == {"訪談", "風景"}
    assert grouped["by_kind"]["訪談"]["by_hand"]["versions"] == 1
    assert grouped["by_kind"]["訪談"]["causes"]["wrong_piece"] == 1
    assert grouped["by_kind"]["風景"]["causes"]["wrong_piece"] == 1


def test_summarise_keeps_a_project_with_no_first_cut_as_none() -> None:
    summary = corrections.summarise({"untouched": [version("編輯器", at(0), project(clip("a")))]})
    assert summary["projects"] == {"untouched": None}
    assert summary["totals"]["by_hand"]["versions"] == 0


def test_report_has_the_causes_and_the_total() -> None:
    summary = corrections.summarise({"p1": _asset_project("a")})
    text = corrections.report(summary)
    assert "合計" in text
    assert "選錯片段" in text
    assert "手動分鐘" in text
    assert "1" in text


def test_main_reads_a_real_workspace(kept_history, media: Path, capsys) -> None:
    from app.tools import catalog
    from helpers import build_project, edit, insert, video_track

    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    with kept_history.step("AI（Claude Code）"):
        project_id = build_project([video_track(), insert("a", asset, 0, 3)])
    with kept_history.step("AI（Claude Code）"):
        edit(project_id, [{"action": "trim_clip", "track_id": "main", "clip_id": "a",
                           "new_source_range": {"start": 0.4, "end": 2.6}}])

    corrections.main(["--project", project_id, "--json"])
    summary = json.loads(capsys.readouterr().out)

    assert project_id in summary["projects"]
    one = summary["projects"][project_id]
    assert one["by_ai"]["versions"] == 1
    assert one["by_ai"]["causes"]["cut_point"] == 2
    assert one["by_hand"]["versions"] == 0
