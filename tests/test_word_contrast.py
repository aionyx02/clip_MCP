"""White words on a bright background: check_render measures how they stand out, and says where they do not (#237)."""

import subprocess
from pathlib import Path

from PIL import Image

from app import runtime
from app.engine import cards, legibility
from app.engine.frames import extract_frame
from app.engine.subtitles import build_ass
from app.tools import catalog
from app.tools.rendering import check_render
from helpers import build_project, edit, insert, render, video_track


def _white(folder: Path) -> Path:
    path = folder / "white.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=white:size=640x360:rate=30:duration=4",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def card(**fields) -> dict:
    return {"action": "add_title_card", "track_id": "main", "clip_id": "t", "title": "8/31 週一",
            "subtitle": "台南古著大豐收", **fields}


def test_white_on_black_reads_and_white_on_white_does_not() -> None:
    rows = [{"text": "8/31 週一", "size": 36, "y": 180}]
    assert legibility.words_contrast(Image.new("RGB", (640, 360), "black"), rows) == 21.0
    assert legibility.words_contrast(Image.new("RGB", (640, 360), "white"), rows) == 1.0
    # A box behind the words carries them over white.
    boxed = [{"text": "王小明", "size": 20, "y": 300, "x": 40, "align": "left", "box": 6}]
    assert legibility.words_contrast(Image.new("RGB", (640, 360), "white"), boxed, 0.62) > legibility.LOW_CONTRAST
    # Only what is under the words counts: white elsewhere in the frame does not.
    half = Image.new("RGB", (640, 360), "black")
    half.paste((255, 255, 255), (0, 0, 640, 60))
    assert legibility.words_contrast(half, rows) == 21.0


def test_words_in_a_brand_colour_are_measured_in_that_colour() -> None:
    white = Image.new("RGB", (640, 360), "white")
    # Black words on white read as well as white ones on black; a pale yellow on white does not.
    assert legibility.words_contrast(white, [{"text": "追蹤", "size": 36, "y": 180, "colour": "#000000"}]) == 21.0
    assert legibility.words_contrast(white, [{"text": "追蹤", "size": 36, "y": 180, "colour": "#ffee88"}]) < 1.3
    # Each colour on its own: a line that reads does not carry one that does not.
    rows = [{"text": "追蹤", "size": 36, "y": 150, "colour": "#000000"},
            {"text": "每週", "size": 18, "y": 200, "colour": "#ffffff"}]
    assert legibility.words_contrast(white, rows) == 1.0


def test_a_card_read_off_a_dark_frame_is_not_flagged(media: Path) -> None:
    """The issue's own case: white words on a near-black frame stand out as far as they can."""
    asset = catalog.import_asset(str(media / "black.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3), card(before_clip_id="a")],
                            width=640, height=360, name="黑底字卡")
    assert "contrast" not in check_render(project)


def test_a_card_read_off_a_bright_frame_is_flagged_as_the_render_shows_it(media: Path, tmp_path: Path) -> None:
    asset = catalog.import_asset(str(_white(tmp_path)))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3), card(before_clip_id="a", background="picture")],
                            width=640, height=360, name="白底字卡")
    checked = check_render(project)
    [note] = checked["contrast"]
    assert note.startswith("title card t 「8/31 週一」 at 00:00.0") and "1.4:1" in note and "`at`" in note
    # A note, not a finding: it never stops a render.
    assert checked["ok"]

    # What was measured is what the render shows: the frame behind the title, around its words.
    stored = runtime.repo.get_project(project)
    made = next(clip for clip in stored.base_video_track.clips if clip.card is not None)
    ass = tmp_path / "cards.ass"
    ass.write_text(build_ass([], 640, 360, stored.caption_style, cards.ass_events(stored, stored.caption_style)),
                   encoding="utf-8")
    out = tmp_path / "out.mp4"
    render(project, out, loudness_target=None, subtitle_path=str(ass))
    rendered = extract_frame(str(out), 1.0, max_size=640)
    title = cards.placed(made, 640, 360)["rows"][0]
    left, top, right, bottom = legibility.row_box(title, 640)
    beside = rendered.crop((right + 4, top, right + 40, bottom))
    assert legibility.words_contrast(beside, [{"text": "x", "size": 10, "y": 5, "x": 0, "align": "left"}]) < 2

    # A dark colour behind it instead, and the note goes.
    edit(project, [{"action": "set_title_card", "track_id": "main", "clip_id": "t", "background": "colour"}])
    assert "contrast" not in check_render(project)


def test_a_headline_over_a_bright_shot_is_flagged_and_a_name_on_its_box_is_not(tmp_path: Path) -> None:
    asset = catalog.import_asset(str(_white(tmp_path)))["id"]
    words = {"action": "add_text", "track_id": "main", "clip_id": "a", "text_id": "h", "text": "出發台南"}
    project = build_project([video_track(), insert("a", asset, 0, 3), {**words, "style": "headline"}],
                            width=640, height=360, name="白底文字")
    [note] = check_render(project)["contrast"]
    assert note.startswith("words h 「出發台南」 on clip a at 00:00.0") and "1.0:1" in note

    edit(project, [{"action": "set_text", "track_id": "main", "clip_id": "a", "text_id": "h", "style": "name"}])
    assert "contrast" not in check_render(project)


def _plain(folder: Path, colour: str) -> Path:
    path = folder / f"{colour}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:size=640x360:rate=30:duration=4",
                    "-f", "lavfi", "-i", "sine=f=300:d=4", "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def captioned(asset: str, name: str) -> str:
    return build_project([video_track(), insert("a", asset, 0, 3),
                          {"action": "set_subtitles", "cues": [
                              {"id": "q1", "asset_id": asset, "source_start": 0.5, "source_end": 2.0, "text": "我們出發吧"}]}],
                         width=640, height=360, name=name)


def test_plain_captions_over_a_bright_shot_are_flagged_and_outlined_ones_are_not(tmp_path: Path) -> None:
    project = captioned(catalog.import_asset(str(_plain(tmp_path, "white")))["id"], "白底字幕")
    [note] = check_render(project)["contrast"]
    assert note.startswith("captions: 1 of 1 measured") and "「我們出發吧」 1.0:1" in note
    # Left out of a render that does not burn them in, they are not measured.
    assert "contrast" not in check_render(project, burn_subtitles=False)
    edit(project, [{"action": "set_caption_style", "style": {"outline_fraction": 0.08}}])
    assert "contrast" not in check_render(project)


def test_plain_captions_over_a_dark_shot_are_not_flagged(tmp_path: Path) -> None:
    project = captioned(catalog.import_asset(str(_plain(tmp_path, "black")))["id"], "黑底字幕")
    assert "contrast" not in check_render(project)
