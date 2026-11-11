"""Brand kits: kept across projects, named by a plan, and compiled into the cut where the length counts them."""

from pathlib import Path
from typing import List

import pytest
from pydantic import TypeAdapter

from app.engine import cards
from app.engine.frames import extract_frame
from app.engine.plan.checks import check_plan
from app.engine.plan.compile import compile_operations
from app.models.brand import BrandKit, EndCard
from app.models.plan import Beat, EditPlan, PlanTarget, PlanTitleCard, Selection
from app.models.semantic import SemanticTimeline
from app.models.timeline.apply import apply_operation
from app.models.timeline.operations import EditOperation
from app.models.timeline.parts import Project
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
    return BrandKit(name="小店", logo_asset_id=LOGO, colour="#203040", font="Noto Sans TC",
                    end_card=EndCard(title="追蹤看更多", seconds=3.0), **fields)


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


def test_the_logo_and_the_closing_card_are_drawn_and_shown_before_a_render(
    planned: dict, media: Path, tmp_path: Path,
) -> None:
    from PIL import Image, ImageStat

    from app.tools.rendering import preview_project

    Image.new("RGBA", (200, 100), (*LOGO_RGB, 255)).save(tmp_path / "brand.png")
    logo = import_asset(str(tmp_path / "brand.png"))["id"]
    save_brand_kit(BrandKit(name="測試品牌", logo_asset_id=logo, logo_opacity=1.0, logo_width=0.25,
                            colour="#203040", end_card=EndCard(title="追蹤", seconds=2.0)))
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

    out = tmp_path / "branded.mp4"
    render(project, out, loudness_target=None)

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
    closing = runtime_repo.get_project(project).base_video_track.clips[-1]
    box = cards.placed(closing, 640, 360)["logo"]
    middle = (box["x"] + box["width"] // 2, box["y"] + box["height"] // 2)
    assert near(mean(picture, (middle[0] - 20, middle[1] - 10, middle[0] + 20, middle[1] + 10)), LOGO_RGB)

    # Set by hand, with no kit, it is the same layer.
    apply_edits(project, get_project(project)["version"],
                OPERATIONS.validate_python([{"action": "set_logo", "logo": None}]))
    assert get_project(project).get("logo") is None
    forget_brand_kit("測試品牌")
