"""Title cards: a title over a frame of its own, taking up time in the cut."""

from pathlib import Path

import pytest

from app import server
from app.engine import cards
from app.engine.frames import extract_frame
from app.engine.subtitles import build_ass
from helpers import build_project, clips_of, edit, insert, level, render, video_track


def card(clip_id: str = "t", **fields) -> dict:
    return {"action": "add_title_card", "track_id": "main", "clip_id": clip_id, "title": "第一章", **fields}


def test_a_card_borrows_the_shot_it_leads_into(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 2, 5), insert("b", asset, 6, 8)], name="卡片")
    edit(project, [card(before_clip_id="b", subtitle="開始")])
    clips = clips_of(project, "main")
    made = next(clip for clip in clips if clip["id"] == "t")
    assert made["card"]["at"] == 6 and made["card"]["background"] == "blur" and made["volume"] == 0
    # It sits where the shot was, and the shot moves along by its length.
    assert float(made["timeline_in"]) == 3 and float(next(c for c in clips if c["id"] == "b")["timeline_in"]) == 5.5
    # At the end, it borrows the last frame of the shot before it.
    edit(project, [card("end")])
    end = next(clip for clip in clips_of(project, "main") if clip["id"] == "end")
    assert 7.8 < end["card"]["at"] < 8


def test_a_card_is_changed_and_stays_silent(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3), card(before_clip_id="a")], name="改卡片")
    edit(project, [{"action": "set_title_card", "track_id": "main", "clip_id": "t", "title": "新標題",
                    "subtitle": "", "background": "colour", "colour": "#203040", "seconds": 4}])
    made, after = clips_of(project, "main")
    assert made["card"]["title"] == "新標題" and made["card"]["subtitle"] is None
    assert made["card"]["colour"] == "#203040" and float(after["timeline_in"]) == 4
    with pytest.raises(ValueError, match="silent"):
        edit(project, [{"action": "set_clip_audio", "track_id": "main", "clip_id": "t", "volume": 1}])
    with pytest.raises(ValueError, match="not a title card"):
        edit(project, [{"action": "set_title_card", "track_id": "main", "clip_id": "a", "title": "x"}])


def test_a_card_renders_its_words_over_a_dark_blur(media: Path, tmp_path: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2), card(before_clip_id="a", subtitle="副標")],
                            width=640, height=360, name="輸出卡片")
    stored = server.repo.get_project(project)
    titles = cards.ass_events(stored, stored.caption_style)
    assert len(titles) == 2 and "第一章" in titles[0] and "\\fad(" in titles[0]
    ass = tmp_path / "cards.ass"
    ass.write_text(build_ass([], 640, 360, stored.caption_style, titles), encoding="utf-8")
    out = tmp_path / "out.mp4"
    render(project, out, loudness_target=None, subtitle_path=str(ass))
    picture = extract_frame(str(out), 1.2, max_size=640).convert("L")
    rows = cards.placed(stored.base_video_track.clips[0] if stored.base_video_track.clips[0].card else
                        stored.base_video_track.clips[1], 640, 360)["rows"]
    # The title's row is bright where the words are; the frame around it is the darkened blur.
    title = picture.crop((200, rows[0]["y"] - 10, 440, rows[0]["y"] + 10))
    corner = picture.crop((0, 0, 60, 40))
    assert max(title.getdata()) > 200 and sum(corner.getdata()) / (60 * 40) < 160
    # The card says nothing, though the shot behind it has sound.
    assert level(out, 0.2, 2.0) < -60


def test_a_card_needs_something_behind_it(media: Path) -> None:
    project = build_project([video_track()], name="空的")
    with pytest.raises(ValueError, match="photo_asset_id"):
        edit(project, [card()])


def test_the_editor_draws_a_card_as_the_render_does(media: Path) -> None:
    from app.engine import playback

    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2), card(before_clip_id="a")],
                            width=640, height=360, name="播卡片")
    stored = server.repo.get_project(project)
    shown = playback.describe(stored, server._referenced_assets(stored), playback.Prepared({}, {}, None), None, {}, {})
    clip = shown["tracks"][0]["clips"][0]
    assert clip["card"] == {"background": "blur", "colour": "#111114", "at": 0.0}
    [event] = shown["cards"]["events"]
    assert event["rows"][0]["text"] == "第一章" and event["end"] == 2.5


def test_a_card_goes_to_another_editor_as_a_gap(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2), card(before_clip_id="a")], name="交換")
    result = server.export_timeline(project, "otio")
    text = Path(result["output_path"]).read_text(encoding="utf-8")
    # The shot's picture and its sound; where the card was, a gap on both.
    assert text.count('"Gap.1"') == 2 and text.count('"Clip.2"') == 2
    assert any("title cards" in line and "第一章" in line for line in result["left_behind"])


def test_a_card_can_stand_on_a_photo(media: Path, tmp_path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (400, 300), (20, 160, 60)).save(tmp_path / "green.png")
    photo = server.import_asset(str(tmp_path / "green.png"))["id"]
    project = build_project([video_track(), card(photo_asset_id=photo)], width=320, height=240, name="照片底")
    [made] = clips_of(project, "main")
    assert made["asset_id"] == photo and made["card"]["background"] == "picture"
    out = tmp_path / "out.mp4"
    render(project, out, loudness_target=None)
    corner = extract_frame(str(out), 1.0, max_size=320).getpixel((5, 5))
    # Green, a little darkened so white words read on it.
    assert corner[1] > corner[0] + 60 and corner[1] < 160
