"""Brand kits: kept across projects, named by a plan, and compiled into the cut where the length counts them."""

from pathlib import Path
from typing import List

import pytest
from pydantic import TypeAdapter

from app.engine import cards, texts
from app.engine.frames import extract_frame
from app.engine.plan.checks import check_plan
from app.engine.plan.compile import CLOSING_TEXT_ID, compile_operations
from app.engine.probe import probe_file
from app.models.brand import BrandKit, EndCard
from app.models.plan import Beat, EditPlan, PlanTarget, PlanTitleCard, Selection
from app.models.semantic import SemanticTimeline
from app.models.timeline.apply import apply_operation
from app.models.timeline.operations import EditOperation
from app.models.timeline.parts import CaptionStyle, Project
from app.runtime import repo as runtime_repo
from app.tools.catalog import import_asset
from app.tools.editing import apply_edits, create_project, get_project, list_projects
from app.tools.feedback import forget_brand_kit, save_brand_kit
from app.tools.planning import compile_plan, get_plan, save_plan, validate_plan
from helpers import render

from edit_plan.shared import ASSET_ID, beats_plan, checked, footage, plan_for, sourced, speech_of

OPERATIONS = TypeAdapter(List[EditOperation])
LOGO = "logo"
# A colour the test picture has nowhere near the corner it is drawn in.
LOGO_RGB = (250, 20, 200)


def kit(**fields) -> BrandKit:
    """A kit closing on a card of its own, three seconds long, with its logo in a corner."""
    return BrandKit(**{"name": "小店", "logo_asset_id": LOGO, "colour": "#203040", "font": "Noto Sans TC",
                       "corner_logo": True, "end_card": EndCard(title="追蹤看更多", mode="card", seconds=3.0),
                       **fields})


def overlaid(**fields) -> BrandKit:
    """A kit as it comes by default: closing words over the last shot, no corner logo."""
    return BrandKit(**{"name": "小店", "logo_asset_id": LOGO, "colour": "#203040", "accent": "#ffcc00",
                       "text_colour": "#f0f0f0", "end_card": EndCard(title="追蹤看更多", subtitle="每週更新", seconds=1.0),
                       **fields})


def render_with_words(project_id: str, out: Path) -> None:
    """Render a cut with its cards' words and the words over it burned in, as `render_project` does."""
    from app.engine.subtitles import build_ass

    cut = runtime_repo.get_project(project_id)
    words = out.with_suffix(".ass")
    words.write_text(build_ass([], cut.width, cut.height, cut.caption_style,
                               cards.ass_events(cut, cut.caption_style) + texts.ass_events(cut, cut.caption_style),
                               texts.ass_styles(cut, cut.caption_style)), encoding="utf-8")
    render(project_id, out, loudness_target=None, subtitle_path=str(words))


def logo_asset():
    return sourced(LOGO, seconds=1.0).model_copy(update={"still": True, "has_audio": False})


def compiled(plan, by_id, children, assets, brand=None, project: Project = None) -> Project:
    operations, provenance = compile_operations(plan, by_id, children, assets, project, None, brand)
    project = (project or Project(id="p", width=1920, height=1080)).model_copy(deep=True)
    for op in OPERATIONS.validate_python(operations):
        apply_operation(project, op, {**assets, LOGO: logo_asset()})
    for track in project.tracks:
        for clip in track.clips:
            origin = provenance.get(clip.id)
            if origin is not None:
                clip.from_plan_id, clip.from_clip_ids, clip.pinned = (
                    origin["from_plan_id"], list(origin["from_clip_ids"]), origin["pinned"])
    return project


def one_part(clips, **fields) -> EditPlan:
    speech = speech_of(clips)
    plan = beats_plan(
        [Selection(clip_id=speech[0].id, beat_id="open"), Selection(clip_id=speech[2].id, beat_id="body")],
        [Beat(id="open", name="開場"), Beat(id="body", name="主體")],
    )
    return plan.model_copy(update={"brand_kit": "小店", **fields})


def test_the_closing_card_ends_the_cut_and_is_counted_in_its_length() -> None:
    by_id, children, assets, clips = footage()
    plain = compiled(one_part(clips, brand_kit=None), by_id, children, assets)
    project = compiled(one_part(clips), by_id, children, assets, kit())
    last = max(project.base_video_track.clips, key=lambda clip: clip.timeline_in)
    assert last.card is not None and last.card.end and last.card.title == "追蹤看更多"
    assert last.card.background == "colour" and last.card.colour == "#203040" and last.card.logo_asset_id == LOGO
    assert float(project.duration) == pytest.approx(float(plain.duration) + 3.0)
    # The logo is over the cut, the font on its words, and both say where they came from.
    assert project.logo is not None and project.logo.asset_id == LOGO and project.logo.brand_kit == "小店"
    assert project.caption_style.font == "Noto Sans TC"
    # It belongs to no part: no marker is made for it, and the parts start where they did.
    assert [marker.timeline_in for marker in project.markers] == [marker.timeline_in for marker in plain.markers]


def worn_check(plan: EditPlan, by_id: dict, assets: dict, brand: BrandKit) -> list:
    timeline = SemanticTimeline(id=plan.timeline_id, asset_ids=[ASSET_ID], input_hash=plan.timeline_input_hash,
                                derivation_version=1)
    return check_plan(plan, timeline, by_id, {}, assets, None, kit=brand)[1]


def single(clips, **fields) -> EditPlan:
    speech = speech_of(clips)
    plan = plan_for([Selection(clip_id=speech[0].id, beat_id="b1"), Selection(clip_id=speech[2].id, beat_id="b1")])
    return plan.model_copy(update={"brand_kit": "小店", **fields})


def test_the_length_checked_before_compiling_has_the_closing_card_in_it() -> None:
    by_id, children, assets, clips = footage()
    plain = compiled(single(clips, brand_kit=None), by_id, children, assets)
    # Asked for exactly the cut without the card: three seconds more is over the line.
    target = PlanTarget(seconds=float(plain.duration))
    problems, without = checked(single(clips, brand_kit=None, target=target), by_id, assets)
    worn = worn_check(single(clips, target=target), by_id, assets, kit())
    assert problems == [] and not any(note.kind == "length_off_target" for note in without)
    off = [note for note in worn if note.kind == "length_off_target" and note.beat_id is None]
    assert off and off[0].seconds == pytest.approx(float(plain.duration) + 3.0)
    # A part's own length is not the brand's: the card is not counted against the last part.
    parted = single(clips, beats=[Beat(id="b1", name="主體", target_seconds=float(plain.duration))])
    notes = worn_check(parted, by_id, assets, kit())
    assert not any(note.kind == "length_off_target" and note.beat_id == "b1" for note in notes)


def test_a_plan_naming_a_kit_that_is_not_kept_does_not_compile() -> None:
    by_id, _, assets, clips = footage()
    problems, _ = checked(one_part(clips), by_id, assets)
    assert any("小店" in problem and "not kept" in problem for problem in problems)


def test_a_card_asked_for_on_a_colour_takes_the_brand_colour() -> None:
    by_id, children, assets, clips = footage()
    plan = one_part(clips)
    plan = plan.model_copy(update={"beats": [
        Beat(id="open", name="開場", title_card=PlanTitleCard(title="開場", background="colour")),
        Beat(id="body", name="主體", title_card=PlanTitleCard(title="主體", background="colour", colour="white")),
    ]})
    project = compiled(plan, by_id, children, assets, kit())
    colours = [clip.card.colour for clip in sorted(project.base_video_track.clips, key=lambda c: c.timeline_in)
               if clip.card is not None and not clip.card.end]
    assert colours == ["#203040", "white"]


def test_a_recompile_rebuilds_the_closing_card_once_and_keeps_one_changed_by_hand() -> None:
    by_id, children, assets, clips = footage()
    plan = one_part(clips)
    project = compiled(plan, by_id, children, assets, kit())
    again = compiled(plan, by_id, children, assets, kit(), project)
    assert sum(1 for clip in again.base_video_track.clips if clip.card is not None) == 1
    assert float(again.duration) == pytest.approx(float(project.duration))

    closing = next(clip for clip in again.base_video_track.clips if clip.card is not None)
    for op in OPERATIONS.validate_python([{"action": "set_title_card", "track_id": "main", "clip_id": closing.id,
                                           "title": "按讚訂閱"}]):
        apply_operation(again, op, assets)
    kept = compiled(plan, by_id, children, assets, kit(), again)
    ends = [clip for clip in kept.base_video_track.clips if clip.card is not None]
    assert [clip.card.title for clip in ends] == ["按讚訂閱"]
    assert max(kept.base_video_track.clips, key=lambda clip: clip.timeline_in).card.end


def test_taking_the_kit_off_takes_its_logo_and_card_off_but_not_a_logo_set_by_hand() -> None:
    by_id, children, assets, clips = footage()
    worn = compiled(one_part(clips), by_id, children, assets, kit())
    bare = compiled(one_part(clips, brand_kit=None), by_id, children, assets, None, worn)
    assert bare.logo is None and not any(clip.card for clip in bare.base_video_track.clips)

    by_hand = bare.model_copy(deep=True)
    for op in OPERATIONS.validate_python([{"action": "set_logo", "logo": {"asset_id": LOGO}}]):
        apply_operation(by_hand, op, {LOGO: logo_asset()})
    assert compiled(one_part(clips, brand_kit=None), by_id, children, assets, None, by_hand).logo is not None


def test_a_logo_has_to_be_a_picture() -> None:
    project = Project(id="p")
    for op in OPERATIONS.validate_python([{"action": "set_logo", "logo": {"asset_id": ASSET_ID}}]):
        with pytest.raises(ValueError, match="not a still picture"):
            apply_operation(project, op, {ASSET_ID: sourced()})


def test_a_card_with_a_logo_puts_it_above_the_words() -> None:
    project = compiled(one_part(footage()[3]), *footage()[:3], kit())
    closing = next(clip for clip in project.base_video_track.clips if clip.card is not None)
    placed = cards.placed(closing, 1920, 1080)
    box = placed["logo"]
    assert box["y"] + box["height"] < placed["rows"][0]["y"]
    # The box and the words together sit in the middle of the frame.
    assert abs((box["y"] + placed["rows"][-1]["y"]) / 2 - 540) < 60


def test_a_kit_is_kept_for_every_project_and_its_logo_has_to_be_in_the_library(media: Path, tmp_path: Path) -> None:
    from PIL import Image

    Image.new("RGBA", (200, 100), (*LOGO_RGB, 255)).save(tmp_path / "logo.png")
    logo = import_asset(str(tmp_path / "logo.png"))["id"]
    with pytest.raises(ValueError, match="not in the library"):
        save_brand_kit(BrandKit(name="別家", logo_asset_id="nope.png"))
    with pytest.raises(ValueError, match="not a still picture"):
        save_brand_kit(BrandKit(name="別家", logo_asset_id=import_asset(str(media / "wide.mp4"))["id"]))
    # By its file name, as anywhere else an asset is named.
    kept = save_brand_kit(BrandKit(name="小店", logo_asset_id="logo.png"))
    assert kept["kept"]["logo_asset_id"] == logo
    assert any(brand["name"] == "小店" for brand in list_projects()["brand_kits"])
    forget_brand_kit("小店")
    assert "小店" not in [brand["name"] for brand in list_projects().get("brand_kits", [])]


def test_closing_words_go_over_the_last_shot_and_add_no_length() -> None:
    by_id, children, assets, clips = footage()
    plain = compiled(one_part(clips, brand_kit=None), by_id, children, assets)
    project = compiled(one_part(clips), by_id, children, assets, overlaid())
    assert float(project.duration) == pytest.approx(float(plain.duration))
    assert not any(clip.card for clip in project.base_video_track.clips)
    last = max(project.base_video_track.clips, key=lambda clip: clip.timeline_in)
    [words] = [text for text in last.texts if text.id == CLOSING_TEXT_ID]
    assert (words.text, words.second, words.style, words.by_plan) == ("追蹤看更多", "每週更新", "cta", True)
    # The accent on the call to action, the words' colour under it and on the captions.
    assert (words.colour, words.second_colour) == ("#ffcc00", "#f0f0f0")
    assert project.caption_style.colour == "#f0f0f0"
    # The last second of the cut, as the render and the editor both draw it.
    [event] = [event for event in texts.drawn(project, project.caption_style)["events"]
               if event["text_id"] == CLOSING_TEXT_ID]
    assert event["end"] == pytest.approx(float(project.duration))
    assert event["start"] == pytest.approx(float(project.duration) - 1.0)
    assert [row["colour"] for row in event["rows"]] == ["#ffcc00", "#f0f0f0"]
    assert all("&H0000CCFF&" in line for line in texts.ass_events(project, project.caption_style)[:1])
    # No corner logo unless the kit asks for one.
    assert project.logo is None


def test_closing_words_are_rebuilt_once_and_kept_when_changed_by_hand() -> None:
    by_id, children, assets, clips = footage()
    plan = one_part(clips)
    project = compiled(plan, by_id, children, assets, overlaid())
    again = compiled(plan, by_id, children, assets, overlaid(), project)

    def closing_words(cut: Project) -> list:
        return [text for track in cut.video_tracks for clip in track.clips for text in clip.texts
                if text.id.startswith(CLOSING_TEXT_ID)]

    assert len(closing_words(again)) == 1
    last = max(again.base_video_track.clips, key=lambda clip: clip.timeline_in)
    for op in OPERATIONS.validate_python([{"action": "set_text", "track_id": "main", "clip_id": last.id,
                                           "text_id": CLOSING_TEXT_ID, "text": "按讚訂閱"}]):
        apply_operation(again, op, assets)
    kept = compiled(plan, by_id, children, assets, overlaid(), again)
    assert [(text.text, text.colour) for text in closing_words(kept)] == [("按讚訂閱", "#ffcc00")]
    # Without the kit the words the kit put there go.
    assert closing_words(compiled(one_part(clips, brand_kit=None), by_id, children, assets, None, project)) == []


def brand_checked(plan: EditPlan, by_id: dict, assets: dict, brand: BrandKit) -> tuple:
    timeline = SemanticTimeline(id=plan.timeline_id, asset_ids=[ASSET_ID], input_hash=plan.timeline_input_hash,
                                derivation_version=1)
    return check_plan(plan, timeline, by_id, {}, assets, None, kit=brand)


def test_closing_words_longer_than_the_last_shot_are_refused_rather_than_stretched() -> None:
    by_id, children, assets, clips = footage()
    long = overlaid(end_card=EndCard(title="追蹤看更多", seconds=10.0))
    problems, _ = brand_checked(single(clips), by_id, assets, long)
    assert any("closes on 「追蹤看更多」" in problem and "plays only" in problem for problem in problems)
    with pytest.raises(ValueError, match="plays only"):
        compile_operations(one_part(clips), by_id, children, assets, None, None, long)


def test_the_length_checked_before_compiling_has_no_closing_words_in_it() -> None:
    by_id, children, assets, clips = footage()
    plain = compiled(single(clips, brand_kit=None), by_id, children, assets)
    target = PlanTarget(seconds=float(plain.duration))
    problems, notes = brand_checked(single(clips, target=target), by_id, assets, overlaid())
    assert problems == [] and not any(note.kind == "length_off_target" for note in notes)


def test_the_corner_logo_is_only_there_when_the_kit_asks_for_it() -> None:
    by_id, children, assets, clips = footage()
    worn = compiled(one_part(clips), by_id, children, assets, kit())
    assert worn.logo is not None and worn.logo.asset_id == LOGO
    off = compiled(one_part(clips), by_id, children, assets, kit(corner_logo=False), worn)
    assert off.logo is None
    # Only the dark version kept: that one goes in the corner.
    dark = BrandKit(name="小店", logo_dark_asset_id=LOGO, corner_logo=True)
    assert compiled(one_part(clips), by_id, children, assets, dark).logo.asset_id == LOGO


def test_a_card_takes_the_logo_that_stands_out_from_its_colour() -> None:
    both = BrandKit(name="小店", logo_asset_id="light", logo_dark_asset_id="dark")
    assert both.logo_on("#203040") == "light" and both.logo_on("black") == "light"
    assert both.logo_on("#f5f0e0") == "dark" and both.logo_on("white") == "dark"
    # Footage behind, blurred and darkened: the light one.
    assert both.logo_on(None) == "light"
    assert BrandKit(name="小店", logo_dark_asset_id="dark").logo_on("#203040") == "dark"
    assert BrandKit(name="小店").logo_on("white") is None

    by_id, children, assets, clips = footage()
    light_card = both.model_copy(update={"colour": "#f5f0e0", "end_card": EndCard(title="追蹤", mode="card")})
    operations, _ = compile_operations(one_part(clips), by_id, children, assets, None, None, light_card)
    [closing] = [op["card"] for op in operations if op.get("card", {}).get("end")]
    assert closing["logo_asset_id"] == "dark"
    # A card's default length: 1.5 seconds.
    [placed] = [op for op in operations if op.get("card", {}).get("end")]
    assert placed["source_range"]["end"] == pytest.approx(1.5)


def test_words_on_a_plain_colour_have_no_shadow_and_wear_the_brand_colours() -> None:
    by_id, children, assets, clips = footage()
    worded = kit(accent="#ffcc00", text_colour="#f0f0f0",
                 end_card=EndCard(title="追蹤看更多", subtitle="每週更新", mode="card", seconds=3.0))
    plan = one_part(clips)
    plan = plan.model_copy(update={"beats": [
        Beat(id="open", name="開場", title_card=PlanTitleCard(title="開場")),
        Beat(id="body", name="主體"),
    ]})
    project = compiled(plan, by_id, children, assets, worded)
    opening, closing = [clip for clip in sorted(project.base_video_track.clips, key=lambda clip: clip.timeline_in)
                        if clip.card is not None]
    on_colour = cards.placed(closing, 1920, 1080)
    assert on_colour["shadow"] is False
    assert [row["colour"] for row in on_colour["rows"]] == ["#ffcc00", "#f0f0f0"]
    # Over the shot, blurred, the words keep their shadow to read.
    assert opening.card.background == "blur" and cards.placed(opening, 1920, 1080)["shadow"] is True
    lines = cards.ass_events(project)
    opened, closed = lines[0], lines[1:]
    assert "\\shad0" not in opened and all("\\shad0" in line for line in closed)
    assert "\\c&H0000CCFF&" in closed[0] and "\\c&H00F0F0F0&" in closed[-1]


def test_vertical_captions_keep_clear_of_the_buttons_up_the_right() -> None:
    from app.engine.subtitles import PORTRAIT_RIGHT_FRACTION, build_ass, caption_geometry

    tall = caption_geometry(1080, 1920, CaptionStyle())
    assert tall.margin_r == round(1080 * PORTRAIT_RIGHT_FRACTION) > tall.margin_h
    assert tall.centre < 540
    wide = caption_geometry(1920, 1080, CaptionStyle())
    assert wide.margin_r == wide.margin_h and wide.centre == 960
    written = build_ass([], 1080, 1920, CaptionStyle(colour="#f0f0f0"))
    assert f",&H00F0F0F0,&H00F0F0F0,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{tall.outline}," in written
    assert f",2,{tall.margin_h},{tall.margin_r},{tall.margin_v},1" in written


def test_the_closing_words_and_no_corner_logo_are_rendered_without_adding_length(
    planned: dict, media: Path, tmp_path: Path,
) -> None:
    from PIL import Image

    from app.tools.rendering import playback_of

    Image.new("RGBA", (200, 100), (*LOGO_RGB, 255)).save(tmp_path / "brand.png")
    logo = import_asset(str(tmp_path / "brand.png"))["id"]
    accent = (0, 255, 0)
    save_brand_kit(BrandKit(name="疊字品牌", logo_asset_id=logo, logo_opacity=1.0, logo_width=0.25,
                            accent="#00ff00", end_card=EndCard(title="追蹤追蹤", seconds=1.0)))
    plan = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    before = validate_plan(planned["plan_id"])["duration"]
    saved = save_plan(plan.model_copy(update={"brand_kit": "疊字品牌"}))
    assert validate_plan(saved["plan_id"])["duration"] == pytest.approx(before)

    project = create_project(name="疊字", width=640, height=360)["id"]
    result = compile_plan(project_id=project, expected_version=1, plan_id=saved["plan_id"])
    assert result["duration"] == pytest.approx(before)
    assert get_project(project).get("logo") is None

    out = tmp_path / "overlaid.mp4"
    render_with_words(project, out)
    assert float(probe_file(str(out))["format"]["duration"]) == pytest.approx(before, abs=0.1)

    # Where the call to action's first line is, by the arithmetic both the render and the editor follow.
    cut = runtime_repo.get_project(project)
    [event] = [event for event in playback_of(cut)["texts"]["events"] if event["text_id"] == CLOSING_TEXT_ID]
    row = event["rows"][0]
    assert row["colour"] == "#00ff00" and event["start"] == pytest.approx(before - 1.0)
    box = (row["x"] - 2 * row["size"], row["y"] - row["size"] // 2, row["x"] + 2 * row["size"], row["y"] + row["size"] // 2)

    def lit(seconds: float) -> int:
        picture = extract_frame(str(out), seconds, max_size=640).convert("RGB").crop(box)
        return sum(1 for pixel in picture.get_flattened_data() if all(abs(a - b) < 70 for a, b in zip(pixel, accent)))

    assert lit(before - 0.4) > lit(before - 1.6) + 30
    # No logo in the corner by default.
    corner = extract_frame(str(out), 1.0, max_size=640).convert("RGB").crop((640 - 14 - 140, 24, 640 - 34, 84))
    assert sum(1 for pixel in corner.get_flattened_data() if all(abs(a - b) < 40 for a, b in zip(pixel, LOGO_RGB))) < 50
    forget_brand_kit("疊字品牌")


def test_the_corner_logo_and_the_closing_card_are_drawn_and_shown_before_a_render(
    planned: dict, media: Path, tmp_path: Path,
) -> None:
    from PIL import Image, ImageStat

    from app.tools.rendering import playback_of, preview_project

    Image.new("RGBA", (200, 100), (*LOGO_RGB, 255)).save(tmp_path / "brand.png")
    logo = import_asset(str(tmp_path / "brand.png"))["id"]
    save_brand_kit(BrandKit(name="測試品牌", logo_asset_id=logo, logo_opacity=1.0, logo_width=0.25, corner_logo=True,
                            colour="#203040", end_card=EndCard(title="追蹤", mode="card", seconds=2.0)))
    plan = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    before = validate_plan(planned["plan_id"])["duration"]
    saved = save_plan(plan.model_copy(update={"brand_kit": "測試品牌"}))
    assert validate_plan(saved["plan_id"])["duration"] == pytest.approx(before + 2.0)

    project = create_project(name="品牌", width=640, height=360)["id"]
    result = compile_plan(project_id=project, expected_version=1, plan_id=saved["plan_id"])
    assert result["duration"] == pytest.approx(before + 2.0)
    assert get_project(project)["logo"]["brand_kit"] == "測試品牌"

    listing = preview_project(project).content[0].text
    assert f"logo {logo} in the top-right corner" in listing and "closing card 「追蹤」" in listing

    # The editor is told where the render puts both logos.
    cut = runtime_repo.get_project(project)
    described = playback_of(cut)
    assert described["logo"]["asset_id"] == logo and described["logo"]["width"] == 160
    assert described["logo"]["margin"] == 14 and described["logo"]["off"] == [[pytest.approx(before),
                                                                               pytest.approx(before + 2.0)]]
    [card] = described["cards"]["events"]
    assert card["logo"]["asset_id"] == logo and card["shadow"] is False

    out = tmp_path / "branded.mp4"
    render_with_words(project, out)

    def mean(picture, box) -> tuple:
        return tuple(ImageStat.Stat(picture.crop(box)).mean)

    def near(colour, wanted, slack=40) -> bool:
        return all(abs(a - b) < slack for a, b in zip(colour, wanted))

    # 160 px wide in the top right, 14 px in from the edges; looked at inside its edges.
    picture = extract_frame(str(out), 1.0, max_size=640).convert("RGB")
    assert near(mean(picture, (640 - 14 - 140, 24, 640 - 34, 84)), LOGO_RGB)
    # On the closing card the corner is the brand colour, and the logo is large above the words.
    picture = extract_frame(str(out), result["duration"] - 0.5, max_size=640).convert("RGB")
    assert near(mean(picture, (640 - 14 - 140, 24, 640 - 34, 84)), (0x20, 0x30, 0x40))
    closing = cut.base_video_track.clips[-1]
    placed = cards.placed(closing, 640, 360)
    box = placed["logo"]
    middle = (box["x"] + box["width"] // 2, box["y"] + box["height"] // 2)
    assert near(mean(picture, (middle[0] - 20, middle[1] - 10, middle[0] + 20, middle[1] + 10)), LOGO_RGB)
    # The words are drawn on it. Whether they cast a shadow is checked on the subtitle events: at this
    # size a one-pixel shadow under white words cannot be told apart from the encoder's own blur.
    row = placed["rows"][0]
    words = picture.convert("L").crop((200, row["y"] - row["size"], 440, row["y"] + row["size"]))
    assert max(words.get_flattened_data()) > 200

    # Set by hand, with no kit, it is the same layer.
    apply_edits(project, get_project(project)["version"],
                OPERATIONS.validate_python([{"action": "set_logo", "logo": None}]))
    assert get_project(project).get("logo") is None
    forget_brand_kit("測試品牌")
