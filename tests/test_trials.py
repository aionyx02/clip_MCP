"""A change to a cut tried blind: X and Y, a verdict, and only then which was the change."""

import os
from pathlib import Path

import pytest

from app import server
from helpers import build_project, edit, finished, insert, video_track


def test_a_change_is_judged_blind_and_kept_only_when_it_helped(media: Path) -> None:
    wide = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 4), insert("b", wide, 5, 9)], width=640, height=360)
    before = finished(server.render_project(project))
    started = server.start_trial(project, before["job_id"], "第二個鏡頭進得太突然", "把 b 的開頭往前多留半秒",
                                 "接點不再那麼跳", at=3.0, end=5.0, evidence="inspect_join #1")
    trial = started["trial_id"]
    with pytest.raises(ValueError, match="no change to judge yet"):
        server.judge_trial(trial, "X", "neither", "看不出來", "subagent")

    edit(project, [{"action": "trim_clip", "track_id": "main", "clip_id": "b",
                    "new_source_range": {"start": 4.5, "end": 9}}])
    after = finished(server.render_project(project))
    server.trial_candidate(trial, after["job_id"])

    pack = server.trial_pack(trial)
    text = pack.content[0].text
    # The judge reads the problem and what shows it, never what was changed or why.
    assert "第二個鏡頭進得太突然" in text and "inspect_join #1" in text
    assert "往前多留半秒" not in text and "不再那麼跳" not in text
    assert os.path.exists(text.split("X sound: ")[1].splitlines()[0])
    assert pack.content[1].type == "image"

    change = server.repo.trial(trial).candidate_is
    original = "Y" if change == "X" else "X"
    # The original breaking what is around the problem does not count against the change.
    judged = server.judge_trial(trial, change, original, "新的接點比較順，舊的後面那句被切掉", "subagent")
    assert judged["change_was"] == change and judged["keep"] == "change" and judged["blind"] is True
    assert not os.path.exists(server._trial_dir(trial))
    with pytest.raises(ValueError, match="was judged already"):
        server.judge_trial(trial, "same", "neither", "再看一次", "user")
    [kept] = server.get_trials(project)["trials"]
    assert kept["verdict"]["judged_by"] == "subagent" and kept["hypothesis"] == "把 b 的開頭往前多留半秒"


def test_a_change_that_broke_what_is_around_it_is_not_kept() -> None:
    from app.models.trial import Trial

    trial = Trial(id="tbroke1", project_id="p", problem="p", hypothesis="h", expected="e", at=0, end=1,
                  before_job="j1", before_version=3, after_job="j2", after_version=4, candidate_at=0,
                  candidate_end=1, candidate_is="Y")
    server.repo.save_trial(trial)
    judged = server.judge_trial("tbroke1", "Y", "Y", "Y 比較好，但後面那句被切掉", "same_ai")
    assert judged["keep"] == "original" and judged["blind"] is False and "version 3" in judged["next"]
