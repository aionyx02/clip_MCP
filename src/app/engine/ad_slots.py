"""Which clips could fill each place in a commercial's structure.

A commercial is usually built the same way: a hook in the first seconds, the
problem the viewer has, the product, proof that it works, and a call to
action. Which sentence is the hook is a judgement, so it is the model's to
make. What can be done by rule is narrowing the field: a sentence that fits in
three seconds, one that says 「每次」 or 「麻煩」, one that names the product, one
with a number or 「實測」 in it, one with a price or 「下單」. This module
over-produces candidates for every slot from those measurements, the model
assigns among them, and the assignment is pinned with a fingerprint of the
candidates it was chosen from, as sections are.

Nothing here scores how good a candidate is. A slot either has candidates or
it does not, and one that does not is reported as footage that is missing
rather than filled with whatever is nearest.

Speech recognition on Chinese often hands back phrases cut where the speaker
breathed, without punctuation, so 「現在下」 and 「單只要三九九」 can be two
utterances. Markers are therefore matched across the join with each
neighbouring utterance, and a match that reaches into a clip counts for it.
"""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.models.plan import EditPlan, Structure
from app.models.semantic import AdSlot, ClipKind, SemanticClip, SemanticTimeline, SlotChoice

# Bumped whenever the rules below change in a way the constants do not show.
# 2: the product goes by several names.
SLOT_VERSION = 2
# Provisional: how long a sentence may run and still open a commercial on its own. Short
# video platforms are commonly said to decide in the first three seconds; this is that
# figure, not one measured on this project's footage.
HOOK_SECONDS = 3.0
# Provisional: how many spoken units (Chinese characters or Latin words) a hook needs, so
# that 「嗯」 and 「然後」 are not offered as one.
HOOK_MIN_UNITS = 4
# Provisional: how far into each neighbouring utterance a marker may reach and still be
# found, for phrases the transcriber split in two. Long enough for the longest marker.
NEIGHBOUR_CHARACTERS = 6
# Provisional, all of these lists: phrases that place a sentence in a slot. They are
# meant to over-produce; 「難」 in 「難得」 is offered too, and the model says no to it.
PROBLEM_MARKERS = (
    "困擾", "麻煩", "問題", "煩", "受不了", "不方便", "不好用", "難", "痛", "怕", "擔心",
    "浪費", "每次", "總是", "老是", "常常", "是不是", "有沒有", "你是否",
    "缺點", "缺陷", "痛點", "卡住", "卡卡", "太慢", "很慢", "不順", "費力", "吃力", "費時",
    "弄髒", "打翻", "翻倒", "壞掉", "抱怨",
    "tired of", "problem", "struggle", "annoying", "hate",
)
PRODUCT_MARKERS = (
    "這款", "這個產品", "這支", "這台", "這組", "這瓶", "這件", "這套", "我們的", "功能",
    "設計", "材質", "尺寸", "用法", "使用方式", "安裝", "一鍵", "開箱", "介紹", "推出", "新品",
    "introducing", "feature", "design", "made of",
)
PROOF_MARKERS = (
    "實測", "測試", "試用", "親身", "用了", "用過", "效果", "結果", "見證", "評價", "好評",
    "回購", "推薦", "數據", "證明", "認證", "比較", "對比", "前後", "滿意", "銷量", "賣出",
    "顧客", "客人", "網友", "百分之",
    "tested", "results", "review", "customers", "proven", "before and after",
)
CTA_MARKERS = (
    "價格", "價錢", "售價", "原價", "特價", "優惠", "折扣", "打折", "限時", "限量", "下單",
    "購買", "買", "連結", "網址", "官網", "留言", "私訊", "訂閱", "追蹤", "按讚", "點擊",
    "趕快", "快去", "搶", "免運", "現在就",
    "link", "buy", "order", "subscribe", "follow", "discount", "code", "shop", "sale",
)
# What a description or tag written by somebody who looked says when the product is on screen.
PRODUCT_ON_SCREEN_WORDS = ("產品", "商品", "product")
_NUMBER = "[0-9零〇一二兩三四五六七八九十百千萬]+"
# A figure: a share, a multiple, a count of people or days.
PROOF_PATTERNS = (rf"{_NUMBER}\s*(%|％|倍|萬|人|次|天|小時)",)
# A price: 399 元, 三百九十九塊, NT$399, $399, 打八折, 79 折.
CTA_PATTERNS = (
    rf"{_NUMBER}\s*(元|塊|块|圓)",
    r"(nt\$|ntd|\$)\s*[0-9]",
    rf"打{_NUMBER}折",
    r"[0-9]+\s*折",
)
# What to shoot when a slot has nothing to choose from.
MISSING_FOOTAGE = {
    AdSlot.HOOK: f"a complete sentence of at least {HOOK_MIN_UNITS} words said in {HOOK_SECONDS:g} seconds or less",
    AdSlot.PROBLEM: "a sentence about what bothers the viewer, such as 「每次…都很麻煩」",
    AdSlot.PRODUCT: "a sentence naming or showing the product, or a shot described as showing it",
    AdSlot.PROOF: "a test, a result, a number or somebody else's word that it works",
    AdSlot.CTA: "a price, an offer, or telling the viewer what to do next, such as 「連結在下面」",
}
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_LATIN_WORD = re.compile(r"[a-z0-9]+")
_SPACE_BY_CJK = re.compile(r"(?<=[㐀-䶿一-鿿豈-﫿])\s+|\s+(?=[㐀-䶿一-鿿豈-﫿])")

@dataclass(frozen=True)
class SlotCandidate:
    """One clip that could fill one place in a commercial.

    Attributes:
        slot: The place it could fill.
        clip_id: The clip.
        asset_id: The file it is in.
        start: Where it starts in the file, in seconds.
        end: Where it ends in the file, in seconds.
        text: What is said in it, or what it shows when nothing is.
        why: What the rules found, one short line each.
    """

    slot: AdSlot
    clip_id: str
    asset_id: str
    start: float
    end: float
    text: str
    why: Tuple[str, ...]

def normalized(text: str) -> str:
    """Bring text to one form for matching markers against.

    Full-width digits and signs become their usual form, Latin is lowered,
    and the spaces a transcriber may put between Chinese characters go.
    Spaces between Latin words stay, so `buy now` is still two words.

    Args:
        text: Text as transcribed or written.

    Returns:
        The text to match against.
    """
    return _SPACE_BY_CJK.sub("", unicodedata.normalize("NFKC", text).lower()).strip()

def spoken_units(text: str) -> int:
    """Count what is said: one per Chinese character, one per Latin word.

    Args:
        text: Normalized text.

    Returns:
        The count.
    """
    return len(_CJK.findall(text)) + len(_LATIN_WORD.findall(_CJK.sub(" ", text)))

def _pattern(marker: str) -> str:
    """Turn a marker into a pattern, whole words only where it is Latin.

    Latin markers may not run into other Latin letters, so `code` is not
    found in `barcode`, but may touch Chinese, so `iphone` is in 「這支iphone」.

    Args:
        marker: The marker.

    Returns:
        A regular expression.
    """
    escaped = re.escape(marker)
    return rf"(?<![a-z0-9]){escaped}(?![a-z0-9])" if marker.isascii() else escaped

def _join(left: str, right: str) -> str:
    """Put two pieces of speech back together the way they were said.

    Args:
        left: The earlier piece.
        right: The later piece.

    Returns:
        Them joined, with a space only between two Latin words.
    """
    if not left or not right:
        return left + right
    if _CJK.match(left[-1]) or _CJK.match(right[0]):
        return left + right
    return f"{left} {right}"

def _found(patterns: Sequence[str], before: str, own: str, after: str) -> List[str]:
    """Find what matches in a clip's text, including across its edges.

    Args:
        patterns: Regular expressions to look for.
        before: The end of the utterance before, normalized.
        own: The clip's text, normalized.
        after: The start of the utterance after, normalized.

    Returns:
        The matched text of each pattern that reaches into the clip, in the
        order the patterns were given, each once.
    """
    left = _join(before, own)
    joined = _join(left, after)
    start, stop = len(left) - len(own), len(left)
    found: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, joined):
            if match.start() < stop and match.end() > start:
                if match.group(0) not in found:
                    found.append(match.group(0))
                break
    return found

def _said(clip: Optional[SemanticClip]) -> str:
    """Read what is said in a clip, normalized, or nothing if it is not speech."""
    return normalized(clip.text) if clip is not None and clip.kind == ClipKind.SPEECH else ""

def product_names(product: Optional[str], aliases: Sequence[str] = ()) -> List[str]:
    """Gather the names a product goes by, normalized, each once.

    Args:
        product: What it is called.
        aliases: What else the speaker calls it, such as 「杯子」 for a cup
            holder whose name is never said.

    Returns:
        The names in the order given, blanks and repeats left out.
    """
    names: List[str] = []
    for name in (product, *aliases):
        name = normalized(name) if name else ""
        if name and name not in names:
            names.append(name)
    return names

def slot_candidates(
    clips: Sequence[SemanticClip],
    product: Optional[str] = None,
    aliases: Sequence[str] = (),
) -> Dict[AdSlot, List[SlotCandidate]]:
    """Find every clip that could fill each place in a commercial.

    Args:
        clips: The timeline's `utterance` clips, ordered by file and then time.
        product: What the product is called, to find the sentences that name
            it and the shots described as showing it. Without it, only the
            phrases that usually introduce a product are found.
        aliases: Other names the speaker uses for it, found the same way.
            Only names somebody gave; none is guessed here.

    Returns:
        Every slot, in the order a commercial plays them, with its candidates
        in the order of the footage. A slot with nothing is an empty list.
    """
    names = product_names(product, aliases)
    marker_patterns = {
        AdSlot.PROBLEM: [_pattern(marker) for marker in PROBLEM_MARKERS],
        AdSlot.PRODUCT: [_pattern(marker) for marker in (*names, *PRODUCT_MARKERS)],
        AdSlot.PROOF: [_pattern(marker) for marker in PROOF_MARKERS] + list(PROOF_PATTERNS),
        AdSlot.CTA: [_pattern(marker) for marker in CTA_MARKERS] + list(CTA_PATTERNS),
    }
    shown_words = [*names, *PRODUCT_ON_SCREEN_WORDS]
    found: Dict[AdSlot, List[SlotCandidate]] = {slot: [] for slot in AdSlot}

    for index, clip in enumerate(clips):
        before_clip = clips[index - 1] if index > 0 and clips[index - 1].asset_id == clip.asset_id else None
        after_clip = clips[index + 1] if index + 1 < len(clips) and clips[index + 1].asset_id == clip.asset_id else None
        own = _said(clip)
        before = _said(before_clip)[-NEIGHBOUR_CHARACTERS:]
        after = _said(after_clip)[:NEIGHBOUR_CHARACTERS]
        reasons: Dict[AdSlot, List[str]] = {slot: [] for slot in AdSlot}

        if own and clip.duration <= HOOK_SECONDS and spoken_units(own) >= HOOK_MIN_UNITS:
            reasons[AdSlot.HOOK].append(f"said in {clip.duration:.1f}s")
        if own:
            for slot, patterns in marker_patterns.items():
                reasons[slot].extend(f"says 「{match}」" for match in _found(patterns, before, own, after))
        if clip.kind not in (ClipKind.SILENCE, ClipKind.UNUSABLE):
            written = [normalized(clip.description or "")] + [normalized(tag.value) for tag in clip.tags]
            for word in shown_words:
                if any(word in text for text in written):
                    reasons[AdSlot.PRODUCT].append(f"described as showing 「{word}」")
                    break

        for slot, why in reasons.items():
            if why:
                found[slot].append(SlotCandidate(
                    slot=slot,
                    clip_id=clip.id,
                    asset_id=clip.asset_id,
                    start=clip.source_range.start,
                    end=clip.source_range.end,
                    text=clip.text or (clip.description or ""),
                    why=tuple(why),
                ))
    return found

def missing_slots(candidates: Dict[AdSlot, List[SlotCandidate]]) -> List[dict]:
    """Say which places have nothing to fill them, and what footage would.

    Args:
        candidates: The candidates, as `slot_candidates` returns them.

    Returns:
        One `{"slot", "needs"}` per empty slot, in the order they play.
    """
    return [{"slot": slot.value, "needs": MISSING_FOOTAGE[slot]} for slot in AdSlot if not candidates[slot]]

def slot_hash(
    candidates: Dict[AdSlot, List[SlotCandidate]],
    product: Optional[str],
    aliases: Sequence[str] = (),
) -> str:
    """Fingerprint the candidates an assignment was chosen from, and the rules that found them.

    Args:
        candidates: The candidates offered.
        product: The product name they were found with.
        aliases: The other names they were found with.

    Returns:
        The fingerprint as a hex digest.
    """
    payload = {
        "slot_version": SLOT_VERSION,
        "product": sorted(product_names(product, aliases)),
        "hook": [HOOK_SECONDS, HOOK_MIN_UNITS],
        "neighbour": NEIGHBOUR_CHARACTERS,
        "markers": [PROBLEM_MARKERS, PRODUCT_MARKERS, PROOF_MARKERS, CTA_MARKERS, PRODUCT_ON_SCREEN_WORDS],
        "patterns": [PROOF_PATTERNS, CTA_PATTERNS],
        "candidates": {slot.value: [candidate.clip_id for candidate in found] for slot, found in candidates.items()},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()

def check_slots(chosen: Sequence[SlotChoice], candidates: Dict[AdSlot, List[SlotCandidate]]) -> List[str]:
    """Check that an assignment only uses what each slot was offered, in the order they play.

    Nothing is corrected here: an assignment that does not hold up is handed
    back with every reason, so that a clip moved to another slot is a decision
    the model made rather than one the server made for it.

    Args:
        chosen: The slots, in the order given.
        candidates: The candidates that were offered.

    Returns:
        One message per problem, empty when the assignment is sound.
    """
    order = list(AdSlot)
    offered = {slot: {candidate.clip_id for candidate in found} for slot, found in candidates.items()}
    problems: List[str] = []
    seen_slots: List[AdSlot] = []
    used: Dict[str, AdSlot] = {}
    for choice in chosen:
        slot = choice.slot
        if slot in seen_slots:
            problems.append(f"{slot.value}: given twice; put all of its clips in one entry")
            continue
        if seen_slots and order.index(slot) < order.index(seen_slots[-1]):
            problems.append(
                f"{slot.value}: comes after {seen_slots[-1].value}, but a commercial plays "
                + " → ".join(item.value for item in order)
            )
        seen_slots.append(slot)
        if not offered[slot]:
            problems.append(f"{slot.value}: nothing in this footage was offered for it; report it missing ({MISSING_FOOTAGE[slot]})")
            continue
        for clip_id in choice.clip_ids:
            if clip_id in used:
                problems.append(f"{slot.value}: {clip_id} is already in {used[clip_id].value}; a clip fills one slot")
            elif clip_id not in offered[slot]:
                problems.append(f"{slot.value}: {clip_id} is not one of its candidates")
            else:
                used[clip_id] = slot
    return problems

def stored_slots_current(timeline: SemanticTimeline, utterances: Sequence[SemanticClip]) -> bool:
    """Say whether the slots stored on a timeline were chosen from the candidates offered now.

    Args:
        timeline: The timeline, with its stored slots.
        utterances: Its `utterance` clips, ordered by file and then time.

    Returns:
        False when the rules or the footage have changed what is offered since.
    """
    offered = slot_candidates(utterances, timeline.ad_product, timeline.ad_aliases)
    return slot_hash(offered, timeline.ad_product, timeline.ad_aliases) == timeline.slot_hash

def plan_slot_problems(plan: EditPlan, timeline: SemanticTimeline, utterances: Sequence[SemanticClip]) -> List[str]:
    """Check that a commercial's plan plays the slots stored for its timeline, and nothing else in them.

    Every role of the ad structure is a slot, so every beat is checked: the
    selections of all the beats of one role, in the order they play, have to
    be that slot's stored clips in their stored order. A plan that strays is
    refused with what was stored, never brought into line: moving a clip from
    one slot to another is done with `set_sections(slots=...)`, where it is
    checked against the candidates and pinned again.

    Args:
        plan: The plan; nothing is checked unless its structure is `ad`.
        timeline: The timeline it is written against, with its stored slots.
        utterances: That timeline's `utterance` clips, ordered by file and
            then time, to tell whether the stored slots are still current.

    Returns:
        One message per problem, empty when the plan follows the slots.
    """
    if plan.structure != Structure.AD:
        return []
    if not timeline.ad_slots:
        return [
            "slots: an `ad` plan plays the clips chosen for each slot, and none are stored for this timeline. "
            'Call propose_sections(structure="ad") and assign them with set_sections(slots=...) first'
        ]
    problems: List[str] = []
    if not stored_slots_current(timeline, utterances):
        problems.append(
            "slots: the candidates have changed since the slots were set; the footage, the product's names or "
            'the rules are not what they were chosen from. Call propose_sections(structure="ad") again and set '
            "the slots again before planning on them"
        )
    stored = {choice.slot.value: list(choice.clip_ids) for choice in timeline.ad_slots}
    roles = {beat.id: beat.role for beat in plan.beats}
    order = {beat.id: index for index, beat in enumerate(plan.beats)}
    played: Dict[str, List[str]] = {}
    for selection in sorted(plan.selections, key=lambda item: order.get(item.beat_id, len(order))):
        role = roles.get(selection.beat_id)
        if role is not None:
            played.setdefault(role.value, []).append(selection.clip_id)
    for slot in AdSlot:
        want = stored.get(slot.value)
        got = played.get(slot.value)
        if want is None and got is None:
            continue
        if want is None:
            problems.append(
                f"slots: the plan has {slot.value} beats, but no {slot.value} was assigned; set it with "
                "set_sections(slots=...) from its candidates, or give those beats another role"
            )
        elif got is None:
            problems.append(
                f"slots: {slot.value} was assigned {', '.join(want)}, but no beat plays it; add a "
                f"`{slot.value}` beat with them, or set the slots again without it"
            )
        elif got != want:
            problems.append(
                f"slots: the {slot.value} beats play {', '.join(got)}, but {slot.value} was assigned "
                f"{', '.join(want)}, in that order. Play those, or change the assignment with "
                "set_sections(slots=...)"
            )
    return problems
