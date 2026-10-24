"""The corrections people made after the AI's first cut are counted by rule, with no model in it."""

import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.engine import corrections
from app.models.media import TranscriptSegment
from app.models.plan import Beat, EditPlan, Selection
from app.models.timeline.parts import Project
from app.runtime import repo
from app.storage.history import Version
from app.tools import catalog, editing, planning
from app.tools.timelines import build_semantic_timeline, query_clips
from edit_plan.shared import FOOTAGE
from helpers import build_project, caption, edit, insert, video_track
from test_semantic_timeline import analysis
from test_subtitles import transcribe, words

# Distinct commit IDs for the versions built in memory; the history only needs them to differ.
_COMMITS = itertools.count(1)
BASE = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
AI = "AI（Claude Code）"


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
    return {"id": "p", "name": "測試", "tracks": [{"id": "main", "track_type": "video", "clips": list(clips)}],
            "subtitles": list(cues)}


def compared(before: dict, after: dict) -> dict:
    """What `compare` counts between two projects as their files hold them."""
    return corrections.compare(Project.model_validate(before), Project.model_validate(after))


def version(author: str, when: datetime, project_dict: dict | None) -> tuple:
    """One version of a project, as a history reports it, with the project as it was then."""
    record = Version(commit=f"{next(_COMMITS):040x}", when=when, author=author, subject="", body="",
                     parents=())
    return record, project_dict


def summary_of(capsys, *arguments: str) -> dict:
    """What the command prints as JSON for the workspace, read back."""
    corrections.main([*arguments, "--json"])
    return json.loads(capsys.readouterr().out)


def test_first_cut_is_the_first_ai_version_that_has_a_clip() -> None:
    versions = [
        version("編輯器", at(0), project()),
        version(AI, at(1), project()),
        version("自動", at(2), project(clip("a"))),
        version(AI, at(3), project(clip("b"))),
    ]
    assert corrections.first_cut(versions) == 3
    assert corrections.first_cut(versions[:3]) is None
    assert corrections.first_cut([]) is None


def test_compare_counts_removed_added_replaced_and_both_edge_moves() -> None:
    before = project(clip("a", asset="a", timeline_in=0), clip("b", start=5.0, end=6.0, timeline_in=1),
                     clip("c", start=10.0, end=11.0, timeline_in=2))
    after = project(clip("a", asset="b", timeline_in=0), clip("b", start=5.5, end=6.25, timeline_in=1),
                    clip("d", start=20.0, end=21.0, timeline_in=3))
    changed = compared(before, after)
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
    assert compared(before, after)["captions"] == {
        "retimed": 1, "rewritten": 1, "rebroken": 1, "added": 1, "removed": 1,
    }


def test_captions_are_paired_by_what_they_say_where_not_by_their_id() -> None:
    before = project(cues=(cue("c1", "一", 0.0, 1.0), cue("c2", "二", 1.0, 2.0), cue("c3", "三", 2.0, 3.0)))
    after = project(cues=(cue("c1", "二", 1.0, 2.0), cue("c2", "三", 2.0, 3.0)))
    assert compared(before, after)["captions"] == {
        "retimed": 0, "rewritten": 0, "rebroken": 0, "added": 0, "removed": 1,
    }


def test_a_caption_whose_words_and_times_both_change_counts_once_as_rewritten() -> None:
    before = project(cues=(cue("c1", "你好", 0.0, 1.0),))
    after = project(cues=(cue("c1", "妳好", 0.2, 1.2),))
    assert compared(before, after)["captions"] == {
        "retimed": 0, "rewritten": 1, "rebroken": 0, "added": 0, "removed": 0,
    }


def test_a_clip_swapped_for_another_file_in_the_same_place_counts_once_as_replaced() -> None:
    before = project(clip("a", asset="a", end=2.0), clip("b", asset="a", start=5.0, end=6.0, timeline_in=2))
    after = project(clip("a", asset="other", end=2.0), clip("b", asset="a", start=5.0, end=6.0, timeline_in=2))
    assert compared(before, after)["clips"] == {"removed": 0, "added": 0, "replaced": 1, "edges_moved": []}


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
        version(AI, at(1), project(clip("a", end=1.0))),
        version(AI, at(2), project(clip("a", start=0.5, end=1.0))),
        version("自動", at(3), project(clip("a", start=0.5, end=1.0), clip("z", asset="z", timeline_in=0.5))),
        version("編輯器", at(10), project(clip("a", start=0.5, end=1.5))),
        version("編輯器", at(13), project(clip("a", start=0.5, end=1.5), clip("b", start=5, end=6, timeline_in=1))),
        version("編輯器", at(40), project(clip("a", start=0.5, end=1.5), clip("b", start=5, end=6, timeline_in=1),
                                         clip("c", start=10, end=11, timeline_in=2))),
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
    assert summary["unreadable"] == 0
    # Two sittings: 10 to 13 minutes is 3 + 1, the change at 40 is 1 on its own.
    assert summary["hand_minutes"] == 5.0


def test_summarise_project_is_none_without_a_first_cut() -> None:
    versions = [version("編輯器", at(0), project(clip("a")))]
    assert corrections.summarise_project(versions) is None


def test_a_version_that_cannot_be_read_is_skipped_and_counted() -> None:
    versions = [
        version(AI, at(1), project(clip("a"))),
        version("編輯器", at(2), {"id": "p", "tracks": "not a list"}),
        version("編輯器", at(3), None),
        version("編輯器", at(4), project(clip("a"), clip("b", start=5, end=6, timeline_in=1))),
    ]
    summary = corrections.summarise_project(versions)
    assert summary is not None
    assert summary["unreadable"] == 2
    # The last version is compared with the first, the last one that could be read.
    assert summary["by_hand"]["versions"] == 1
    assert summary["by_hand"]["causes"]["wrong_piece"] == 1
    text = corrections.report(corrections.summarise({"p": versions}))
    assert "測試：2 個版本讀不出來" in text


def _asset_project(asset: str) -> list:
    """A project cut from one file, with one change made by hand."""
    return [
        version(AI, at(1), project(clip("a", asset=asset))),
        version("編輯器", at(2), project(clip("a", asset=asset), clip("b", asset=asset, start=5, end=6,
                                                                      timeline_in=1))),
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
    assert "讀不出來" not in text


def test_main_reads_a_real_workspace(kept_history, media: Path, capsys) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    with kept_history.step(AI):
        project_id = build_project([video_track(), insert("a", asset, 0, 3)])
    with kept_history.step(AI):
        edit(project_id, [{"action": "trim_clip", "track_id": "main", "clip_id": "a",
                           "new_source_range": {"start": 0.4, "end": 2.6}}])

    one = summary_of(capsys, "--project", project_id)["projects"][project_id]
    assert one["by_ai"]["versions"] == 1
    assert one["by_ai"]["causes"]["cut_point"] == 2
    assert one["by_hand"]["versions"] == 0


def test_without_a_project_named_every_project_is_read(kept_history, media: Path, capsys) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    with kept_history.step(AI):
        first = build_project([video_track(), insert("a", asset, 0, 3)])
    with kept_history.step(AI):
        second = build_project([video_track(), insert("a", asset, 4, 6)])
    assert {first, second} <= set(summary_of(capsys)["projects"])


def test_a_hand_edit_after_the_ai_counts_by_hand(kept_history, media: Path, capsys) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    with kept_history.step(AI):
        project_id = build_project([video_track(), insert("a", asset, 0, 3), insert("b", asset, 5, 8)])
    with kept_history.step(corrections.EDITOR_AUTHOR):
        edit(project_id, [{"action": "delete_clip", "track_id": "main", "clip_id": "b"}])

    one = summary_of(capsys, "--project", project_id)["projects"][project_id]
    assert one["by_hand"]["causes"]["wrong_piece"] == 1
    assert one["by_ai"]["versions"] == 0


def test_a_split_counts_nothing(kept_history, media: Path, capsys) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    with kept_history.step(AI):
        project_id = build_project([video_track(), insert("a", asset, 0, 6)])
    with kept_history.step(corrections.EDITOR_AUTHOR):
        edit(project_id, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at": 2}])

    one = summary_of(capsys, "--project", project_id)["projects"][project_id]
    assert one["by_hand"]["versions"] == 1
    assert one["causes"] == {cause: 0 for cause in corrections.CAUSES}


def test_a_music_bed_lengthened_counts_nothing(kept_history, media: Path, capsys) -> None:
    footage = catalog.import_asset(str(media / "wide.mp4"))["id"]
    song = catalog.import_asset(str(media / "song.mp3"))["id"]
    with kept_history.step(AI):
        project_id = build_project([
            video_track(), insert("a", footage, 0, 6),
            {"action": "add_track", "track_id": "music", "track_type": "audio"}, insert("m001", song, 0, 3, "music"),
        ])
    with kept_history.step(corrections.EDITOR_AUTHOR):
        edit(project_id, [{"action": "trim_clip", "track_id": "music", "clip_id": "m001",
                           "new_source_range": {"start": 0, "end": 6}}])

    one = summary_of(capsys, "--project", project_id)["projects"][project_id]
    assert one["by_hand"]["versions"] == 1
    assert one["causes"] == {cause: 0 for cause in corrections.CAUSES}


def test_captions_made_again_after_a_line_left_the_cut_count_one_removal(
    kept_history, media: Path, capsys,
) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    transcribe(asset, [
        TranscriptSegment(start=start, end=start + 2, text=text, words=words((text, start, start + 2)))
        for start, text in ((1.0, "第一句"), (4.0, "第二句"), (7.0, "第三句"))
    ])
    with kept_history.step(AI):
        project_id = build_project([video_track(), insert("a", asset, 0, 9)])
    with kept_history.step(AI):
        caption(project_id)
    with kept_history.step(corrections.EDITOR_AUTHOR):
        edit(project_id, [{"action": "trim_clip", "track_id": "main", "clip_id": "a",
                           "new_source_range": {"start": 3.5, "end": 9}}])
    with kept_history.step(AI):
        caption(project_id)
    assert [cue.text for cue in repo.get_project(project_id).subtitles] == ["第二句", "第三句"]

    one = summary_of(capsys, "--project", project_id)["projects"][project_id]
    both = {key: one["by_hand"]["captions"][key] + one["by_ai"]["captions"][key]
            for key in one["by_ai"]["captions"]}
    assert both == {"retimed": 0, "rewritten": 0, "rebroken": 0, "added": 3, "removed": 1}


def _plan(media: Path, picks: tuple) -> tuple:
    """A plan choosing some of the spoken stretches of a real file.

    Args:
        media: The folder of test files.
        picks: Which stretches, by their place in the file.

    Returns:
        The plan's ID, and the plan as saved.
    """
    asset_id = catalog.import_asset(str(media / "wide.mp4"))["id"]
    repo.save_analysis(analysis(asset_id=asset_id, **FOOTAGE))
    built = build_semantic_timeline([asset_id], rebuild=True)
    speech = query_clips(timeline_id=built["timeline_id"], text="句話")["clips"]
    saved = planning.save_plan(EditPlan(
        timeline_id=built["timeline_id"], timeline_input_hash="", goal="做一支短的",
        beats=[Beat(id="b1", name="主體", intent="講完一件事")],
        selections=[Selection(clip_id=speech[pick]["clip_id"], beat_id="b1", rationale="要") for pick in picks],
    ))
    return saved["plan_id"], EditPlan.model_validate(planning.get_plan(saved["plan_id"])["plan"])


def test_a_recompile_that_drops_one_selection_counts_one_removal(kept_history, media: Path, capsys) -> None:
    # The first and third lines: two clips apart in the file, which the compiler does not join.
    plan_id, plan = _plan(media, (0, 2))
    project_id = editing.create_project("重編")["id"]
    with kept_history.step(AI):
        planning.compile_plan(project_id, editing.get_project(project_id)["version"], plan_id)
    with kept_history.step(AI):
        planning.save_plan(plan.model_copy(update={"selections": plan.selections[1:]}))
        planning.compile_plan(project_id, editing.get_project(project_id)["version"], plan_id)

    one = summary_of(capsys, "--project", project_id)["projects"][project_id]
    assert one["by_ai"]["clips"] == {"removed": 1, "added": 0, "replaced": 0}
    assert one["by_ai"]["edge_moves"]["count"] == 0


def test_a_stored_version_that_is_not_even_json_is_skipped(kept_history, media: Path, capsys) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    with kept_history.step(AI):
        project_id = build_project([video_track(), insert("a", asset, 0, 3), insert("b", asset, 5, 8)])
    with kept_history.step(corrections.EDITOR_AUTHOR):
        kept_history.write(kept_history.working_path("projects/", f"{project_id}.json"), b"{broken")
    with kept_history.step(corrections.EDITOR_AUTHOR):
        edit(project_id, [{"action": "delete_clip", "track_id": "main", "clip_id": "b"}])

    one = summary_of(capsys, "--project", project_id)["projects"][project_id]
    assert one["unreadable"] == 1
    assert one["by_hand"]["versions"] == 1
    assert one["by_hand"]["causes"]["wrong_piece"] == 1
