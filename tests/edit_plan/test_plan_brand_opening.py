"""The note `check_render` gives when a brand's cut opens without the product or the brand on screen (#273)."""

from decimal import Decimal

from app.engine.brand_opening import OPENING_SECONDS, brand_opening
from app.models.plan import Structure
from app.models.semantic import AdSlot, SemanticTimeline, SlotChoice
from app.models.timeline.parts import PlacedCue

from edit_plan.shared import footage
from edit_plan.test_plan_brand import compiled, kit, one_part, overlaid


def cue(text: str, start: float) -> PlacedCue:
    return PlacedCue(cue_id="c", start=Decimal(str(start)), end=Decimal(str(start + 1.5)), text=text)


def timeline(**fields) -> SemanticTimeline:
    return SemanticTimeline(id="tl", asset_ids=["a"], input_hash="h", derivation_version=1, **fields)


def made(brand=None, **plan_fields):
    by_id, children, assets, clips = footage()
    plan = one_part(clips, **plan_fields)
    return compiled(plan, by_id, children, assets, brand if brand is not None else overlaid()), plan


def test_an_opening_with_neither_product_nor_brand_is_noted() -> None:
    project, plan = made()
    note = brand_opening(project, plan, timeline())
    assert note is not None and f"first {OPENING_SECONDS:g} s" in note and "「小店」" in note


def test_the_brand_named_on_screen_in_the_opening_is_enough() -> None:
    project, plan = made()
    assert brand_opening(project, plan, timeline(), [cue("歡迎來到 小店", 0.5)]) is None
    # Named only after the opening is too late.
    assert brand_opening(project, plan, timeline(), [cue("歡迎來到小店", OPENING_SECONDS + 1)]) is not None


def test_the_product_named_by_another_name_counts() -> None:
    project, plan = made()
    named = timeline(ad_product="抽杯架", ad_aliases=["杯架"])
    assert brand_opening(project, plan, named, [cue("這個杯架終於好了", 0.2)]) is None


def test_a_product_slot_clip_in_the_opening_is_enough() -> None:
    project, plan = made()
    first = min(project.base_video_track.clips, key=lambda clip: clip.timeline_in)
    shown = timeline(ad_slots=[SlotChoice(slot=AdSlot.PRODUCT, clip_ids=[first.from_clip_ids[0]])])
    assert brand_opening(project, plan, shown) is None
    # A product slot whose clips this cut does not play leaves the opening without it.
    elsewhere = timeline(ad_slots=[SlotChoice(slot=AdSlot.PRODUCT, clip_ids=["tl:u9999"])])
    assert "product slot" in brand_opening(project, plan, elsewhere)


def test_the_corner_logo_shows_the_brand() -> None:
    project, plan = made(kit())
    assert project.logo is not None
    assert brand_opening(project, plan, timeline()) is None


def test_a_cut_with_no_brand_and_no_commercial_is_not_checked() -> None:
    project, plan = made(brand_kit=None)
    assert brand_opening(project, plan, timeline()) is None
    assert brand_opening(project, None, None) is None
    commercial = plan.model_copy(update={"structure": Structure.AD})
    assert brand_opening(project, commercial, timeline()) is not None


def test_without_a_product_name_the_note_says_the_product_was_not_checked() -> None:
    project, plan = made()
    unknown = brand_opening(project, plan, timeline())
    assert "the brand nowhere" in unknown and "could not be checked" in unknown
    known = brand_opening(project, plan, timeline(ad_product="抽杯架"))
    assert "neither the product nor the brand" in known and "could not be checked" not in known
