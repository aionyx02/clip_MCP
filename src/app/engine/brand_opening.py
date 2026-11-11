"""Whether a brand's cut shows what it is for in its first seconds.

A viewer scrolling past decides in the opening seconds; a commercial that
gets there without the product or the brand on screen has spent them. This
only says so: what opens the cut stays the model's and the user's choice.
"""

from typing import Iterable, List, Optional, Sequence

from app.engine import texts
from app.engine.ad_slots import normalized, product_names
from app.models.plan import EditPlan, Structure
from app.models.semantic import AdSlot, SemanticTimeline
from app.models.timeline.parts import PlacedCue, Project

# Provisional: how long the opening is that the product or brand should show in. The
# usual advice for short-form ads, not measured against any viewer.
OPENING_SECONDS = 3.0


def _unit(clip_id: str) -> str:
    """A semantic clip ID without its timeline, so whole and short forms compare equal."""
    return clip_id.rsplit(":", 1)[-1]


def _named(words: Iterable[str], names: Sequence[str]) -> Optional[str]:
    """The first of `names` that any of `words` contains, compared as the slot rules compare them."""
    wanted = [(name, normalized(name)) for name in names if normalized(name)]
    for said in words:
        flat = normalized(said)
        for name, key in wanted:
            if key in flat:
                return name
    return None


def brand_opening(
    project: Project,
    plan: Optional[EditPlan],
    timeline: Optional[SemanticTimeline],
    cues: Sequence[PlacedCue] = (),
) -> Optional[str]:
    """Say when a brand's or a commercial's cut opens without the product or the brand on screen.

    It applies to a cut compiled from a plan that names a brand kit or is a
    commercial. The opening counts as showing what it is for when, within
    `OPENING_SECONDS`, a clip of the product slot is on screen, the corner
    logo is up, or words on screen — a title card, words over the footage or
    a caption — name the product, one of its other names, or the brand kit.

    Args:
        project: The cut.
        plan: The plan it was compiled from, if any.
        timeline: That plan's semantic timeline, for the product slot and names.
        cues: The captions as placed on the cut.

    Returns:
        One line saying what the opening lacks and what would give it, or None
        when the opening shows it or the check does not apply.
    """
    if plan is None or (plan.brand_kit is None and plan.structure != Structure.AD):
        return None
    if project.logo is not None:
        return None
    end = OPENING_SECONDS
    product_units = {_unit(clip_id) for choice in (timeline.ad_slots if timeline else [])
                     if choice.slot == AdSlot.PRODUCT for clip_id in choice.clip_ids}
    names = product_names(timeline.ad_product, timeline.ad_aliases) if timeline and timeline.ad_product else []
    names = [*names, *([plan.brand_kit] if plan.brand_kit else [])]
    words: List[str] = []
    for track in project.video_tracks:
        for clip in track.clips:
            start = float(clip.timeline_in)
            if start >= end or float(clip.timeline_out) <= 0:
                continue
            if product_units and product_units & {_unit(clip_id) for clip_id in clip.from_clip_ids}:
                return None
            if clip.card is not None:
                words += [clip.card.title, clip.card.subtitle or ""]
            for text in clip.shown_texts:
                span = texts.shown(clip, text)
                if span is not None and span[0] < end:
                    words += [text.text, text.second or ""]
    words += [cue.text for cue in cues if float(cue.start) < end]
    if names and _named(words, names):
        return None
    showing = []
    if product_units:
        showing.append("a shot of the product slot")
    if names:
        showing.append("words naming " + " or ".join(f"「{name}」" for name in names))
    showing.append("the brand kit's corner logo" if plan.brand_kit else "the product on screen")
    known = bool(product_units) or bool(timeline and timeline.ad_product)
    lacking = "neither the product nor the brand" if known else "the brand nowhere"
    note = (f"the first {OPENING_SECONDS:g} s show {lacking}; a viewer deciding whether to keep watching has not "
            f"seen what this is for. Opening on {', or '.join(showing)} would; or leave it, when holding it back "
            "is the point")
    if not known:
        # Without a product name or slot there is nothing to look for, which is not the same as it being absent.
        note += (". Whether the product shows could not be checked: this cut's timeline has no product name or "
                 "product slot — propose_sections with the product, and set_sections with its slots, gives one")
    return note
