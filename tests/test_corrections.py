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


# A clip showing the first ten seconds of file `a`, so captions there are in the cut.
SHOWN = {"id": "shown", "asset_id": "a", "source_range": {"start": 0, "end": 10}, "timeline_in": 0}


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
    before = project(SHOWN, cues=(
        cue("c1", "你好世界", 0.0, 1.0),
        cue("c2", "再見", 1.0, 2.0),
        cue("c3", "刪掉", 2.0, 3.0),
        cue("c4", "換行", 3.0, 4.0),
        cue("c5", "沒變", 4.0, 5.0),
    ))
    after = project(SHOWN, cues=(
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
    before = project(SHOWN, cues=(cue("c1", "一", 0.0, 1.0), cue("c2", "二", 1.0, 2.0), cue("c3", "三", 2.0, 3.0)))
    after = project(SHOWN, cues=(cue("c1", "二", 1.0, 2.0), cue("c2", "三", 2.0, 3.0)))
    assert compared(before, after)["captions"] == {
        "retimed": 0, "rewritten": 0, "rebroken": 0, "added": 0, "removed": 1,
    }


def test_a_caption_whose_words_and_times_both_change_counts_once_as_rewritten() -> None:
    before = project(SHOWN, cues=(cue("c1", "你好", 0.0, 1.0),))
    after = project(SHOWN, cues=(cue("c1", "妳好", 0.2, 1.2),))
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


def test_captions_made_again_after_a_line_left_the_cut_count_nothing(
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
    # The first captions are not corrections, and the line that left went with its footage.
    assert one["by_ai"]["versions"] == 2
    assert one["causes"]["misheard"] == 0
    assert one["by_ai"]["captions"] == {"retimed": 0, "rewritten": 0, "rebroken": 0, "added": 0, "removed": 0}


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


NO_CAPTION_CHANGE = {"retimed": 0, "rewritten": 0, "rebroken": 0, "added": 0, "removed": 0}
NO_CLIP_CHANGE = {"removed": 0, "added": 0, "replaced": 0, "edges_moved": []}


def test_cutting_a_pause_out_of_a_shot_moves_one_cut_point(kept_history, media: Path, capsys) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    with kept_history.step(AI):
        project_id = build_project([video_track(), insert("a", asset, 0, 6)])
    with kept_history.step(corrections.EDITOR_AUTHOR):
        edit(project_id, [
            {"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at": 2},
            {"action": "trim_clip", "track_id": "main", "clip_id": "a2", "new_source_range": {"start": 2.5, "end": 6}},
        ])

    one = summary_of(capsys, "--project", project_id)["projects"][project_id]
    assert one["by_hand"]["causes"] == {**{cause: 0 for cause in corrections.CAUSES}, "cut_point": 1}
    assert one["by_hand"]["edge_moves"] == {"count": 1, "median": 0.5, "largest": 0.5}


def test_a_speed_change_after_a_split_counts_nothing(kept_history, media: Path, capsys) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    with kept_history.step(AI):
        project_id = build_project([video_track(), insert("a", asset, 0, 6)])
    with kept_history.step(corrections.EDITOR_AUTHOR):
        edit(project_id, [
            {"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at": 2},
            {"action": "set_clip_speed", "track_id": "main", "clip_id": "a2", "speed": 2.0},
        ])

    one = summary_of(capsys, "--project", project_id)["projects"][project_id]
    assert one["by_hand"]["versions"] == 1
    assert one["causes"] == {cause: 0 for cause in corrections.CAUSES}


def test_two_shots_of_one_file_swapped_count_nothing() -> None:
    before = project(clip("x", start=0, end=2, timeline_in=0), clip("y", start=2, end=4, timeline_in=2))
    after = project(clip("y", start=2, end=4, timeline_in=0), clip("x", start=0, end=2, timeline_in=2))
    assert compared(before, after)["clips"] == NO_CLIP_CHANGE


def test_shots_of_one_file_far_apart_on_the_timeline_are_compared_as_what_they_show() -> None:
    # Touching in the file, with another file between them on the timeline: still one stretch of `a`.
    before = project(clip("x", start=0, end=2, timeline_in=0), clip("o", asset="other", timeline_in=2),
                     clip("y", start=2, end=4, timeline_in=3))
    after = project(clip("x", start=0, end=2, timeline_in=0), clip("o", asset="other", timeline_in=2),
                    clip("y", start=2.4, end=4, timeline_in=3))
    assert compared(before, after)["clips"] == {**NO_CLIP_CHANGE, "edges_moved": [0.4]}


def test_a_title_card_renamed_by_a_recompile_is_not_counted() -> None:
    card = {"title": "第一天"}
    before = project(clip("t1", asset="a", end=2, timeline_in=0) | {"card": card}, clip("p001", start=5, end=6,
                                                                                         timeline_in=2))
    after = project(clip("p001", asset="a", end=2, timeline_in=0) | {"card": card}, clip("p002", start=5, end=6,
                                                                                           timeline_in=2))
    assert compared(before, after)["clips"] == NO_CLIP_CHANGE


def test_a_cut_of_only_a_title_card_is_not_the_first_cut() -> None:
    versions = [
        version(AI, at(1), project(clip("t1", end=2) | {"card": {"title": "片頭"}})),
        version("編輯器", at(2), project(clip("a", end=2))),
    ]
    assert corrections.first_cut(versions) is None


def test_a_caption_deleted_from_footage_still_in_the_cut_counts_as_misheard() -> None:
    before = project(SHOWN, cues=(cue("c1", "真的有說", 0, 1), cue("c2", "沒有人說過", 2, 3)))
    after = project(SHOWN, cues=(cue("c1", "真的有說", 0, 1),))
    changed = compared(before, after)
    assert changed["captions"] == {**NO_CAPTION_CHANGE, "removed": 1}
    assert corrections.classify(changed)["misheard"] == 1


def test_a_caption_that_leaves_with_its_footage_is_not_counted() -> None:
    before = project(SHOWN, clip("b", start=20, end=22, timeline_in=10),
                     cues=(cue("c1", "留下", 0, 1), cue("c2", "跟著走", 20, 21)))
    after = project(SHOWN, cues=(cue("c1", "留下", 0, 1),))
    assert compared(before, after)["captions"] == NO_CAPTION_CHANGE


def test_the_first_captions_and_captions_for_new_footage_are_not_corrections() -> None:
    first = compared(project(SHOWN), project(SHOWN, cues=(cue("c1", "第一句", 0, 1),)))
    assert first["captions"] == NO_CAPTION_CHANGE
    longer = compared(project(SHOWN, cues=(cue("c1", "第一句", 0, 1),)),
                      project(SHOWN, clip("b", start=20, end=22, timeline_in=10),
                              cues=(cue("c1", "第一句", 0, 1), cue("c2", "新的", 20, 21))))
    assert longer["captions"] == NO_CAPTION_CHANGE


def test_a_missed_line_added_to_captioned_footage_counts_as_misheard() -> None:
    before = project(SHOWN, cues=(cue("c1", "第一句", 0, 1),))
    after = project(SHOWN, cues=(cue("c1", "第一句", 0, 1), cue("c2", "漏掉的", 3, 4)))
    assert compared(before, after)["captions"] == {**NO_CAPTION_CHANGE, "added": 1}


def test_a_caption_moved_further_than_its_length_is_retimed() -> None:
    before = project(SHOWN, cues=(cue("c1", "晚一點", 1, 2),))
    after = project(SHOWN, cues=(cue("c1", "晚一點", 5, 6),))
    assert compared(before, after)["captions"] == {**NO_CAPTION_CHANGE, "retimed": 1}


def test_a_caption_split_in_two_or_two_joined_is_re_broken_once() -> None:
    whole = cue("c1", "我們今天去海邊", 0, 4)
    halves = (cue("c1", "我們今天", 0, 2), cue("c2", "去海邊", 2, 4))
    assert compared(project(SHOWN, cues=(whole,)), project(SHOWN, cues=halves))["captions"] == {
        **NO_CAPTION_CHANGE, "rebroken": 1}
    assert compared(project(SHOWN, cues=halves), project(SHOWN, cues=(whole,)))["captions"] == {
        **NO_CAPTION_CHANGE, "rebroken": 1}


def test_captions_of_different_files_at_the_same_seconds_are_not_paired() -> None:
    other = {"id": "o", "asset_id": "b", "source_range": {"start": 0, "end": 10}, "timeline_in": 10}
    before = project(SHOWN, other, cues=(cue("c1", "甲說的", 0, 1, asset="a"),))
    after = project(SHOWN, other, cues=(cue("c1", "乙說的", 0, 1, asset="b"),))
    assert compared(before, after)["captions"] == {**NO_CAPTION_CHANGE, "added": 1, "removed": 1}


def test_a_renamed_project_is_reported_by_its_newest_name() -> None:
    renamed = project(clip("a"))
    renamed["name"] = "新名字"
    summary = corrections.summarise_project([version(AI, at(1), project(clip("a"))),
                                             version("編輯器", at(2), renamed)])
    assert summary is not None and summary["name"] == "新名字"


def _recaptioned_after_a_trim(kept_history, media: Path, capsys, segments: list, start: float) -> dict:
    """Caption a real cut, trim its start by hand, caption it again, and read the summary back.

    Args:
        kept_history: The history fixture.
        media: The test media folder.
        capsys: The output capture fixture.
        segments: The transcript of the file.
        start: Where the trim makes the shot start.

    Returns:
        The project's summary.
    """
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    transcribe(asset, segments)
    with kept_history.step(AI):
        project_id = build_project([video_track(), insert("a", asset, 0, 9)])
    with kept_history.step(AI):
        caption(project_id)
    with kept_history.step(corrections.EDITOR_AUTHOR):
        edit(project_id, [{"action": "trim_clip", "track_id": "main", "clip_id": "a",
                           "new_source_range": {"start": start, "end": 9}}])
    with kept_history.step(AI):
        caption(project_id)
    return summary_of(capsys, "--project", project_id)["projects"][project_id]


def test_a_caption_that_shrinks_with_a_trim_is_not_counted(kept_history, media: Path, capsys) -> None:
    one = _recaptioned_after_a_trim(kept_history, media, capsys, [
        TranscriptSegment(start=begin, end=begin + 2, text=text, words=words((text, begin, begin + 2)))
        for begin, text in ((1.0, "第一句"), (4.0, "第二句"), (7.0, "第三句"))
    ], 2.0)
    assert one["causes"]["misheard"] == 0 and one["causes"]["caption_timing"] == 0


def test_a_caption_that_loses_a_word_cut_in_half_by_a_trim_is_not_counted(kept_history, media: Path, capsys) -> None:
    one = _recaptioned_after_a_trim(kept_history, media, capsys, [
        TranscriptSegment(start=1.0, end=3.0, text="大家好今天", words=words(("大家好", 1.0, 2.0), ("今天", 2.0, 3.0))),
        TranscriptSegment(start=4.0, end=6.0, text="第二句", words=words(("第二句", 4.0, 6.0))),
    ], 1.9)
    assert one["by_ai"]["captions"] == NO_CAPTION_CHANGE
    assert one["causes"]["misheard"] == 0 and one["causes"]["caption_timing"] == 0


def test_a_caption_rewritten_inside_the_footage_both_show_still_counts() -> None:
    before = project(SHOWN, cues=(cue("c1", "大家好今天", 1, 3) | {"words": [
        {"start": 1, "end": 2, "text": "大家好"}, {"start": 2, "end": 3, "text": "今天"}]},))
    after = project(SHOWN, cues=(cue("c1", "大家好明天", 1, 3) | {"words": [
        {"start": 1, "end": 2, "text": "大家好"}, {"start": 2, "end": 3, "text": "明天"}]},))
    assert compared(before, after)["captions"] == {**NO_CAPTION_CHANGE, "rewritten": 1}


def test_a_caption_whose_words_no_longer_spell_it_is_compared_by_its_text() -> None:
    # Words left from before the text was edited by hand do not hide the edit.
    stale = [{"start": 1, "end": 3, "text": "今天"}]
    before = project(SHOWN, cues=(cue("c1", "今天", 1, 3) | {"words": stale},))
    after = project(SHOWN, cues=(cue("c1", "明天", 1, 3) | {"words": stale},))
    assert compared(before, after)["captions"] == {**NO_CAPTION_CHANGE, "rewritten": 1}


def test_a_caption_with_the_same_words_in_another_file_is_not_a_move() -> None:
    other = {"id": "o", "asset_id": "b", "source_range": {"start": 0, "end": 10}, "timeline_in": 10}
    before = project(SHOWN, other, cues=(cue("c1", "一樣的話", 1, 2, asset="a"),))
    after = project(SHOWN, other, cues=(cue("c1", "一樣的話", 5, 6, asset="b"),))
    assert compared(before, after)["captions"] == {**NO_CAPTION_CHANGE, "added": 1, "removed": 1}


def test_a_part_that_starts_before_the_whole_but_centres_inside_it_is_re_broken() -> None:
    whole = cue("c1", "我們今天去海邊", 1, 4)
    parts = (cue("c1", "我們今天", 0.5, 2), cue("c2", "去海邊", 2, 4))
    assert compared(project(SHOWN, cues=(whole,)), project(SHOWN, cues=parts))["captions"] == {
        **NO_CAPTION_CHANGE, "rebroken": 1}


def test_an_edge_moved_by_no_more_than_touching_is_not_counted() -> None:
    before = project(clip("x", start=0, end=4))
    after = project(clip("x", start=0, end=3.9995))
    assert compared(before, after)["clips"] == NO_CLIP_CHANGE


def test_footage_replaced_is_matched_clip_by_clip_on_the_timeline() -> None:
    # One stretch of `a`, shown by two clips with `b` between them on the timeline. Footage of `c`
    # put where `b` was shown was not on the timeline at the same time as either clip of `a`.
    before = project(clip("x", start=0, end=2, timeline_in=0), clip("o", asset="b", start=0, end=8, timeline_in=2),
                     clip("y", start=2, end=4, timeline_in=10))
    after = project(clip("n", asset="c", start=0, end=2, timeline_in=4),
                    clip("o", asset="b", start=0, end=8, timeline_in=6))
    assert compared(before, after)["clips"] == {**NO_CLIP_CHANGE, "removed": 1, "added": 1}


# One long sentence a word a character, captioned as two lines; after a trim the words left fit on one.
SENTENCE = "今天我們要來聊一聊這個城市裡面最好吃的早餐店還有它們的故事"


def test_a_sentence_recaptioned_on_fewer_lines_after_a_trim_is_not_misheard(
    kept_history, media: Path, capsys,
) -> None:
    step = 8.0 / len(SENTENCE)
    one = _recaptioned_after_a_trim(kept_history, media, capsys, [TranscriptSegment(
        start=0.5, end=8.5, text=SENTENCE,
        words=words(*((character, 0.5 + index * step, 0.5 + (index + 1) * step)
                      for index, character in enumerate(SENTENCE))),
    )], 3.0)
    assert one["causes"]["misheard"] == 0
    # The lines are broken in other places than before; the history does not say who broke them.
    assert one["by_ai"]["captions"] == {**NO_CAPTION_CHANGE, "rebroken": 1}


def test_a_caption_with_words_retimed_by_hand_is_retimed(kept_history, media: Path, capsys) -> None:
    asset = catalog.import_asset(str(media / "wide.mp4"))["id"]
    transcribe(asset, [TranscriptSegment(start=1.0, end=3.0, text="大家好今天",
                                         words=words(("大家好", 1.0, 2.0), ("今天", 2.0, 3.0)))])
    with kept_history.step(AI):
        project_id = build_project([video_track(), insert("a", asset, 0, 9)])
    with kept_history.step(AI):
        cue_id = caption(project_id)["cues"][0]["id"]
    with kept_history.step(corrections.EDITOR_AUTHOR):
        edit(project_id, [{"action": "edit_subtitle", "cue_id": cue_id, "source_start": 1.5, "source_end": 3.5}])
    one = summary_of(capsys, "--project", project_id)["projects"][project_id]
    assert one["by_hand"]["captions"] == {**NO_CAPTION_CHANGE, "retimed": 1}


def test_the_same_short_word_deleted_in_one_place_and_added_far_from_it_is_not_a_move() -> None:
    before = project(SHOWN, cues=(cue("c1", "對", 1, 1.5),))
    after = project(SHOWN, cues=(cue("c1", "對", 8, 8.5),))
    assert compared(before, after)["captions"] == {**NO_CAPTION_CHANGE, "added": 1, "removed": 1}


def test_a_split_that_adds_punctuation_is_re_broken() -> None:
    before = project(SHOWN, cues=(cue("c1", "大家好今天", 1, 3),))
    after = project(SHOWN, cues=(cue("c1", "大家好，", 1, 2), cue("c2", "今天", 2, 3)))
    assert compared(before, after)["captions"] == {**NO_CAPTION_CHANGE, "rebroken": 1}


def test_lines_broken_in_other_places_on_as_many_captions_are_re_broken_once() -> None:
    before = project(SHOWN, cues=(cue("c1", "我們今天", 0, 2), cue("c2", "去海邊玩", 2, 4)))
    after = project(SHOWN, cues=(cue("c1", "我們今天去", 0, 2.5), cue("c2", "海邊玩", 2.5, 4)))
    assert compared(before, after)["captions"] == {**NO_CAPTION_CHANGE, "rebroken": 1}


def test_stretches_less_than_touching_apart_are_one() -> None:
    joined = corrections._footage(Project.model_validate(project(
        clip("x", start=0, end=2, timeline_in=0), clip("y", start=2.0005, end=4, timeline_in=2))))
    assert [(stretch.start, stretch.end) for stretch in joined["a"]] == [(0.0, 4.0)]


def test_a_sliver_no_longer_than_touching_is_not_footage_both_show() -> None:
    one = corrections._footage(Project.model_validate(project(clip("x", start=0, end=2))))
    other = corrections._footage(Project.model_validate(project(clip("x", start=1.9995, end=4))))
    assert corrections._common(one, other) == {}


def test_hand_versions_exactly_a_sitting_gap_apart_are_one_sitting() -> None:
    gap = corrections.SITTING_GAP_SECONDS / 60
    hands = [version("編輯器", at(0), None)[0], version("編輯器", at(gap), None)[0]]
    assert corrections._hand_minutes(hands) == round(gap + corrections.MINUTES_PER_SITTING_FLOOR, 1)
