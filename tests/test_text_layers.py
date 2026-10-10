"""Words over footage: pinned to what is on screen, placed by kind, drawn the same in the render and the editor."""

from pathlib import Path

import pytest

from app import server
from app.engine import texts
from app.engine.frames import extract_frame
from app.engine.subtitles import build_ass
from helpers import build_project, clips_of, edit, insert, render, video_track


def words(clip_id: str = "a", text_id: str = "n", **fields) -> dict:
    return {"action": "add_text", "track_id": "main", "clip_id": clip_id, "text_id": text_id, "text": "王小明",
            **fields}


def test_words_stay_with_their_footage(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 2, 6), insert("b", asset, 0, 2)], name="文字")
    # Given in the cut's seconds, kept in the file's.
    edit(project, [words(timeline_start=1, timeline_end=3, second="店長")])
    [layer] = clips_of(project, "main")[0]["texts"]
    assert (float(layer["start"]), float(layer["end"])) == (3, 5) and layer["second"] == "店長"
    stored = server.repo.get_project(project)
    assert texts.shown(stored.base_video_track.clips[0], stored.base_video_track.clips[0].texts[0]) == (1, 3)
    # Moved to the end of the cut, the words go with the footage.
    edit(project, [{"action": "reorder_clip", "track_id": "main", "clip_id": "a"}])
    stored = server.repo.get_project(project)
    moved = next(clip for clip in stored.base_video_track.clips if clip.id == "a")
    assert texts.shown(moved, moved.texts[0]) == (3, 5)
    # Trimmed past part of it, it shows for what is left; split, each half shows its own part.
    edit(project, [{"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at_source": 4}])
    stored = server.repo.get_project(project)
    halves = {clip.id: texts.shown(clip, clip.texts[0]) for clip in stored.base_video_track.clips if clip.texts}
    assert halves == {"a": (3, 4), "a2": (4, 5)}


def test_words_are_changed_and_taken_off(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 4), words()], name="改文字")
    edit(project, [{"action": "set_text", "track_id": "main", "clip_id": "a", "text_id": "n", "style": "free",
                    "x": 0.3, "y": 0.2, "text": "注意", "second": ""}])
    [layer] = clips_of(project, "main")[0]["texts"]
    assert (layer["style"], layer["x"], layer["y"], layer["text"], layer["second"]) == ("free", 0.3, 0.2, "注意", None)
    with pytest.raises(ValueError, match="not in the clip"):
        edit(project, [{"action": "set_text", "track_id": "main", "clip_id": "a", "text_id": "n", "start": 8, "end": 9}])
    edit(project, [{"action": "remove_text", "track_id": "main", "clip_id": "a", "text_id": "n"}])
    assert clips_of(project, "main")[0]["texts"] == []
    with pytest.raises(ValueError, match="no text n"):
        edit(project, [{"action": "remove_text", "track_id": "main", "clip_id": "a", "text_id": "n"}])


def test_each_kind_keeps_to_its_place(media: Path) -> None:
    asset = server.import_asset(str(media / "tall.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 4)], width=1080, height=1920, name="位置")
    style = server.repo.get_project(project).caption_style
    layer = lambda kind: texts.placed(  # noqa: E731
        texts.TextLayer(id="t", text="地方", second="說明", style=kind, start=0, end=4), 1080, 1920, style)
    name, place, headline = layer("name"), layer("place"), layer("headline")
    # A name sits low, above the captions; a place high, below the app's bar; a headline in the middle.
    assert 1100 < name[-1]["y"] < 1700 and name[0]["box"] and name[0]["align"] == "left"
    assert 1920 * texts.TOP_SAFE_PORTRAIT <= place[0]["y"] < 500
    assert abs((headline[0]["y"] + headline[-1]["y"]) / 2 - 960) < 60 and headline[0]["align"] == "center"


def test_a_name_bar_renders_on_its_box(media: Path, tmp_path: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2), words(second="店長")],
                            width=640, height=360, name="輸出文字")
    stored = server.repo.get_project(project)
    lines = texts.ass_events(stored, stored.caption_style)
    assert len(lines) == 2 and "TextBox" in lines[0] and "王小明" in lines[0]
    ass = tmp_path / "texts.ass"
    ass.write_text(build_ass([], 640, 360, stored.caption_style, lines, texts.ass_styles(stored)), encoding="utf-8")
    out = tmp_path / "out.mp4"
    render(project, out, loudness_target=None, subtitle_path=str(ass))
    picture = extract_frame(str(out), 1.0, max_size=640).convert("L")
    row = texts.placed(stored.base_video_track.clips[0].texts[0], 640, 360, stored.caption_style)[0]
    # Inside the box, left of the words' middle: darkened, with white strokes of the name in it.
    strip = picture.crop((row["x"], row["y"] - row["size"] // 2, row["x"] + row["size"] * 3, row["y"] + row["size"] // 2))
    values = list(strip.getdata())
    assert max(values) > 200 and sorted(values)[len(values) // 4] < 120


def test_the_editor_draws_words_as_the_render_does(media: Path) -> None:
    from app.engine import playback

    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 1, 3), words(style="headline")],
                            width=640, height=360, name="播文字")
    stored = server.repo.get_project(project)
    shown = playback.describe(stored, server._referenced_assets(stored), playback.Prepared({}, {}, None), None, {}, {})
    [event] = shown["texts"]["events"]
    assert (event["start"], event["end"], event["text_id"]) == (0.0, 2.0, "n")
    assert event["rows"] == texts.placed(stored.base_video_track.clips[0].texts[0], 640, 360, stored.caption_style)


def test_a_title_card_keeps_its_own_words(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2),
                             {"action": "add_title_card", "track_id": "main", "clip_id": "t", "title": "標題"}],
                            name="卡片沒文字")
    with pytest.raises(ValueError, match="title card has its own"):
        edit(project, [words("t")])
