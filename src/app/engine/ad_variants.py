"""A commercial's variants: the same plan with another hook, or cut down to a platform length.

Ad platforms are bought by the slot length: a 6 second bumper, a 15 second
spot, a 30 second one. These are not the long cut squeezed. A bumper is one
idea, the product on screen and the brand; fifteen seconds is a hook, the
product and the ask; thirty can carry the problem and the proof. So which
slots survive at a length is a rule over the slots' roles, and the rule
over-produces: every combination of whole selections that keeps a hook and
the product and lands near the length is offered, and the model or the user
picks among them. Nothing is cut inside a selection, since a selection of an
ad plan is one sentence, and a length nothing reaches is refused with what
the shortest or longest combination runs instead of being met by cutting
into a sentence.

A hook test is the other axis: variants identical after their first
sentence, so that a platform's A/B test measures the opening and nothing
else. The other hooks are the hook slot's candidates.

Picks are pinned with a fingerprint of the plan, the slots and the
candidates they were chosen from, as sections and slots are.
"""

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.models.plan import AdVariant, EditPlan, PlanTarget, Rejection, Selection, SoundEffects
from app.models.semantic import AdSlot, SlotChoice

# Bumped whenever the rules below change in a way the constants do not show.
VARIANT_VERSION = 1
# Provisional: a cut-down may run this share under its length. Platforms cap a slot from
# above and a spot a little short still fills it; not measured against any campaign.
SHORT_SHARE = 0.25
# Provisional: and this many seconds over, the rounding of the last frame and the fade;
# a platform's length is a ceiling.
OVER_SECONDS = 0.5
# Provisional: lengths up to these take the bumper's and the short spot's rules; longer ones
# keep every role. The usual 6/15/30 s buys, with room for a 20 s spot to be a short one.
BUMPER_SECONDS = 6.0
SHORT_SPOT_SECONDS = 20.0
# Provisional: how far the sum of the pieces' own lengths may be from the length before the
# exact compiled length is worked out. Pieces that join lose a little air; this is generous.
ESTIMATE_SLACK = 1.0
# Provisional: most selections a plan may have for its cut-downs to be listed. Every
# combination is tried, so this keeps a call to a few thousand.
MOST_SELECTIONS = 12
# Provisional: most cut-downs listed for one hook at one length, so a reading stays short.
MOST_CUTS = 4

@dataclass(frozen=True)
class Format:
    """What a commercial of one length keeps.

    Attributes:
        label: What it is called.
        roles: The slots it may play.
        one_each: Whether it plays at most one clip of each.
    """

    label: str
    roles: Tuple[AdSlot, ...]
    one_each: bool

def format_for(seconds: float) -> Format:
    """The rules a commercial cut down to this length keeps to."""
    if seconds <= BUMPER_SECONDS:
        return Format("bumper", (AdSlot.HOOK, AdSlot.PRODUCT, AdSlot.CTA), True)
    if seconds <= SHORT_SPOT_SECONDS:
        return Format("short spot", (AdSlot.HOOK, AdSlot.PRODUCT, AdSlot.CTA), False)
    return Format("full spot", tuple(AdSlot), False)

def window(seconds: float) -> Tuple[float, float]:
    """How short and how long a cut-down to this length may run."""
    return round(seconds * (1 - SHORT_SHARE), 3), round(seconds + OVER_SECONDS, 3)

@dataclass(frozen=True)
class Unit:
    """One selection of the plan, as cutting down sees it.

    Attributes:
        clip_id: Its clip.
        slot: The slot its beat plays.
        seconds: How long it runs alone, as compiled.
        text: What is said in it.
    """

    clip_id: str
    slot: AdSlot
    seconds: float
    text: str

@dataclass(frozen=True)
class Cut:
    """One cut-down that lands in its window: the selections it keeps and how long it runs."""

    keeps: Tuple[str, ...]
    seconds: float

def admissible(kept: Sequence[Unit], rules: Format, asks: bool = False) -> bool:
    """Say whether a set of selections makes a commercial of these rules: a hook, the product and the ask.

    The ask is required only of a plan that says it in its own words, `asks`:
    one with no CTA line leaves it to the brand kit's closing words or card,
    which every cut-down keeps over its last shot.
    """
    slots = [unit.slot for unit in kept]
    if AdSlot.HOOK not in slots or AdSlot.PRODUCT not in slots:
        return False
    if asks and AdSlot.CTA not in slots:
        return False
    if any(slot not in rules.roles for slot in slots):
        return False
    return not rules.one_each or len(slots) == len(set(slots))

def cut_downs(
    units: Sequence[Unit],
    seconds: float,
    overhead: float,
    measure: Callable[[Tuple[str, ...]], float],
) -> Tuple[List[Cut], Optional[Cut], Optional[Cut]]:
    """Find every combination of whole selections that cuts the plan down to a length.

    Args:
        units: The plan's selections in the order they play.
        seconds: The length.
        overhead: What the cut runs beyond its pieces' own lengths: a closing
            card, less the air two joined pieces share.
        measure: The exact compiled length of the cut that keeps these clips.

    Returns:
        `(fits, shortest, longest)`: the cut-downs in the window, in the order
        they were found; and the shortest and the longest admissible ones,
        measured, to say what can be had when nothing fits.
    """
    rules = format_for(seconds)
    low, high = window(seconds)
    admitted: List[Tuple[Tuple[Unit, ...], float]] = []
    asks = any(unit.slot == AdSlot.CTA for unit in units)
    for size in range(len(units), 0, -1):
        for kept in combinations(units, size):
            if admissible(kept, rules, asks):
                admitted.append((kept, sum(unit.seconds for unit in kept) + overhead))
    fits: List[Cut] = []
    for kept, estimate in admitted:
        if low - ESTIMATE_SLACK <= estimate <= high + ESTIMATE_SLACK:
            keeps = tuple(unit.clip_id for unit in kept)
            exact = measure(keeps)
            if low <= exact <= high:
                fits.append(Cut(keeps, exact))
    if not admitted:
        return fits, None, None
    ends = []
    for kept, _ in (min(admitted, key=lambda item: item[1]), max(admitted, key=lambda item: item[1])):
        keeps = tuple(unit.clip_id for unit in kept)
        ends.append(Cut(keeps, measure(keeps)))
    return fits, ends[0], ends[1]

def refusal(seconds: float, units: Sequence[Unit], shortest: Optional[Cut], longest: Optional[Cut]) -> str:
    """Say why nothing cuts the plan down to a length, with what can be had instead.

    Args:
        seconds: The length asked for.
        units: The plan's selections, to name what the cuts keep.
        shortest: The shortest admissible cut, measured; None when none is.
        longest: The longest, likewise.

    Returns:
        One sentence for the user.
    """
    rules = format_for(seconds)
    low, high = window(seconds)
    said = {unit.clip_id: unit for unit in units}

    def listed(cut: Cut) -> str:
        return " + ".join(f"{said[clip_id].slot.value} 「{said[clip_id].text}」 ({said[clip_id].seconds:.1f}s)"
                          for clip_id in cut.keeps)

    if shortest is None or longest is None:
        return (f"{seconds:g}s: a {rules.label} keeps {', '.join(slot.value for slot in rules.roles)}, and this plan "
                "has no hook and product among them to keep")
    if shortest.seconds > high:
        return (f"{seconds:g}s: the shortest cut that keeps a whole hook and the product is {listed(shortest)}, "
                f"which runs {shortest.seconds:.1f}s as compiled — over the {high:g}s a {seconds:g}s "
                "spot may run, and nothing shorter can be had without cutting into a sentence. A shorter product "
                "line or a shorter hook would make it")
    if longest.seconds < low:
        return (f"{seconds:g}s: the longest cut a {rules.label} keeps ({', '.join(slot.value for slot in rules.roles)}) "
                f"is {listed(longest)}, which runs {longest.seconds:.1f}s — under the {low:g}s a {seconds:g}s spot "
                "should at least run")
    return (f"{seconds:g}s: no combination of whole selections lands between {low:g}s and {high:g}s; they run from "
            f"{shortest.seconds:.1f}s to {longest.seconds:.1f}s with nothing in between that fits")

def derive(
    plan: EditPlan,
    name: str,
    hook: Optional[Tuple[str, str]],
    keeps: Optional[Sequence[str]],
    target: Optional[float],
    pinned: str,
) -> Tuple[EditPlan, List[str], Dict[str, list]]:
    """Make a variant's plan from the original's.

    The hook's selections give way to the one swapped in, and everything
    after it stays exactly as it was; a cut-down leaves out the selections it
    does not keep, with the beats, music cues and covering picture that
    only they had, and the facts only they established. Each is said in
    what comes back, and filed under the plan's rejections.

    Args:
        plan: The original.
        name: The variant's name.
        hook: `(clip_id, why it was offered)` of the hook swapped in, or None.
        keeps: The clips a cut-down keeps, or None for all of them.
        target: The length it is cut down to, or None.
        pinned: The `variant_hash` it was picked with.

    Returns:
        `(plan, problems, dropped)`. Problems are what stop it being made,
        such as a kept piece relying on a fact only a dropped one
        establishes. `dropped` lists the `selections`, `beats`, `cues`,
        `broll` and `facts` left out.
    """
    roles = {beat.id: beat.role for beat in plan.beats}
    hook_beats = [beat.id for beat in plan.beats if roles[beat.id] is not None and roles[beat.id].value == "hook"]
    selections: List[Selection] = []
    swapped = False
    for selection in plan.selections:
        if hook is not None and selection.beat_id in hook_beats:
            if not swapped:
                selections.append(Selection(clip_id=hook[0], beat_id=hook_beats[0],
                                            rationale=f"{name}: the hook slot's candidate ({hook[1]})"))
                swapped = True
            continue
        selections.append(selection.model_copy(deep=True))
    kept = set(keeps) if keeps is not None else {selection.clip_id for selection in selections}
    if hook is not None:
        kept.add(hook[0])
    gone = [selection for selection in selections if selection.clip_id not in kept]
    replaced = [selection for selection in plan.selections
                if hook is not None and selection.beat_id in hook_beats and selection.clip_id != hook[0]]
    selections = [selection for selection in selections if selection.clip_id in kept]
    played_beats = {selection.beat_id for selection in selections}
    beats = [beat.model_copy(deep=True) for beat in plan.beats if beat.id in played_beats]
    established = {item.fact for selection in selections for item in selection.establishes}
    facts = [fact for fact in plan.facts if fact.id in established]
    lost = [fact.id for fact in plan.facts if fact.id not in established]
    problems = [f"{selection.clip_id} relies on {fact}, which only a selection this variant leaves out establishes"
                for selection in selections for fact in [*selection.requires, *selection.ahead] if fact in lost]
    if plan.facts and not facts:
        problems.append(
            f"it keeps none of the selections that establish what the viewer has to come to know "
            f"({', '.join(lost)}), so it says nothing the plan set out to say; mark in the original which of its "
            "kept selections establish them, or pick a cut that keeps one that does"
        )
    music = None
    cues_gone: List[str] = []
    if plan.music is not None:
        cues = [cue for cue in plan.music.cues if cue.beat_id is None or cue.beat_id in played_beats]
        cues_gone = [cue.beat_id for cue in plan.music.cues if cue not in cues]
        music = plan.music.model_copy(update={"cues": cues}) if cues else None
    kept_clips = {selection.clip_id for selection in selections}
    broll = [shot for shot in plan.broll if shot.over_clip_id in kept_clips]
    reason = f"left out of {name}"
    kept_clips_now = {selection.clip_id for selection in selections}
    rejected = [*(item for item in plan.rejected if item.clip_id not in kept_clips_now), *(Rejection(clip_id=selection.clip_id, reason=reason) for selection in [*gone, *replaced])]
    slots: List[SlotChoice] = []
    for selection in selections:
        slot = AdSlot(roles[selection.beat_id].value) if roles.get(selection.beat_id) is not None else None
        if slot is None:
            continue
        if slots and slots[-1].slot == slot:
            slots[-1].clip_ids.append(selection.clip_id)
        else:
            slots.append(SlotChoice(slot=slot, clip_ids=[selection.clip_id]))
    variant = AdVariant(
        name=name, of_plan_id=plan.id, of_version=plan.version, hook=hook[0] if hook else None,
        target_seconds=target, dropped=[selection.clip_id for selection in [*replaced, *gone]],
        slots=slots, variant_hash=pinned,
    )
    derived = plan.model_copy(deep=True, update={
        "selections": selections, "beats": beats, "facts": facts, "music": music, "broll": broll,
        "rejected": rejected, "sound_effects": None, "variant": variant,
        "target": PlanTarget(seconds=target, platform=plan.target.platform, note=plan.target.note)
        if target is not None else plan.target.model_copy(),
    })
    dropped = {
        "selections": [selection.clip_id for selection in [*replaced, *gone]],
        "beats": [beat.id for beat in plan.beats if beat.id not in played_beats],
        "cues": cues_gone,
        "broll": [shot.over_clip_id for shot in plan.broll if shot not in broll],
        "facts": lost,
    }
    return derived, problems, dropped

def kept_effects(chosen: Optional[SoundEffects], moments: Sequence, current: str) -> Tuple[Optional[SoundEffects], List[dict]]:
    """Carry the original's sound effects onto a variant's cut, the ones whose moment is still in it.

    A moment is named by what makes it, so the cut into the same clip is the
    same moment in the variant, and its pick is the same decision. A pick
    whose moment is not in the variant is dropped with that reason, never
    moved to the nearest moment; what is kept is pinned again to the
    variant's own moments.

    Args:
        chosen: The original's effects, or None.
        moments: The variant's moments, as `effect_moments` finds them.
        current: Their `moments_hash`.

    Returns:
        `(effects, dropped)`: the effects for the variant, None when none is
        left, and one `{candidate, why}` per pick left out.
    """
    if chosen is None or not chosen.picks:
        return None, []
    there = {moment.id for moment in moments}
    kept = [pick for pick in chosen.picks if pick.candidate in there]
    dropped = [{"candidate": pick.candidate, "why": "that moment is not in this variant's cut"}
               for pick in chosen.picks if pick.candidate not in there]
    return (SoundEffects(candidate_hash=current, picks=kept) if kept else None), dropped

def variant_hash(plan: EditPlan, slot_hash: Optional[str], offered: Dict[str, dict]) -> str:
    """Fingerprint what variants are picked from: the plan as it is, its slots, the rules and the list offered.

    Args:
        plan: The original plan.
        slot_hash: The fingerprint its timeline's slots were set with.
        offered: Each variant's `id` mapped to its hook and what it keeps.

    Returns:
        The fingerprint as a hex digest.
    """
    payload = {
        "version": VARIANT_VERSION,
        "rules": [SHORT_SHARE, OVER_SECONDS, BUMPER_SECONDS, SHORT_SPOT_SECONDS, ESTIMATE_SLACK, MOST_CUTS],
        "plan": [plan.id, plan.version],
        "slots": slot_hash,
        "offered": offered,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
