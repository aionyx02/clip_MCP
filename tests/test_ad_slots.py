"""Tests for the candidates offered to each place in a commercial, and for the slots chosen from them.

The footage is shaped like what speech recognition hands back for Chinese:
one word per character, phrases cut where the speaker breathed rather than
where the sentence ends, and no punctuation. What is tested is that the rules
find what they say they find, that a slot with nothing is reported rather
than filled, and that an assignment outside the candidates is refused with
the reason.
"""

from typing import Dict, List, Sequence, Tuple

import pytest

from app.engine.ad_slots import (
    HOOK_SECONDS,
    check_slots,
    missing_slots,
    slot_candidates,
    slot_hash,
)
from app.engine.semantic import build_timeline
from app.models.media import MediaAnalysis, Span, Transcript, TranscriptSegment, TranscriptWord
from app.models.semantic import AdSlot, ClipDescription, ClipKind, ClipLevel, SemanticClip, SlotChoice, SectionChoice
from app.runtime import repo as runtime_repo
from app.tools.catalog import import_asset
from app.tools.timelines import build_semantic_timeline, propose_sections, query_clips, set_clip_tags, set_sections
from test_semantic_timeline import ASSET_ID, asset

# A cup holder ad as it was said: the CTA 「現在下單」 is split across two phrases.
CUP_HOLDER = [
    (0.0, 2.4, "你是不是每次出門"),
    (2.4, 4.0, "杯子都沒地方放"),
    (4.5, 9.0, "這款杯架夾在桌邊就能用"),
    (9.5, 13.0, "我自己實測放了兩公斤都不會掉"),
    (13.5, 15.0, "現在下"),
    (15.0, 19.0, "單只要三百九十九元"),
]

def chinese_analysis(
    segments: Sequence[Tuple[float, float, str]],
    duration: float = 20.0,
    asset_id: str = ASSET_ID,
) -> MediaAnalysis:
    """Build an analysis the way the transcriber hands back Chinese: a word per character.

    Args:
        segments: `(start, end, text)` per phrase, without punctuation.
        duration: Length of the file.
        asset_id: Asset the analysis belongs to.

    Returns:
        The analysis, with silences filling every gap between phrases.
    """
    built = []
    for start, end, text in segments:
        step = (end - start) / len(text)
        words = [TranscriptWord(start=start + index * step, end=start + (index + 1) * step, text=character)
                 for index, character in enumerate(text)]
        built.append(TranscriptSegment(start=start, end=end, text=text, words=words))
    silences, cursor = [], 0.0
    for start, end, _ in segments:
        if start - cursor > 0.3:
            silences.append(Span(start=cursor, end=start))
        cursor = end
    if duration - cursor > 0.3:
        silences.append(Span(start=cursor, end=duration))
    return MediaAnalysis(
        asset_id=asset_id, duration=duration, silences=silences,
        transcript=Transcript(language="zh", model="test", segments=built),
    )

def clips_of(segments: Sequence[Tuple[float, float, str]], duration: float = 20.0) -> List[SemanticClip]:
    """Derive the utterance clips of one file.

    Args:
        segments: Phrases as `chinese_analysis` takes them.
        duration: Length of the file.

    Returns:
        The clips.
    """
    return build_timeline({ASSET_ID: asset()}, {ASSET_ID: chinese_analysis(segments, duration)})[1]

def said(candidates: Dict[AdSlot, list], slot: AdSlot) -> List[str]:
    """Read what the candidates of one slot say.

    Args:
        candidates: Candidates by slot.
        slot: The slot.

    Returns:
        Their texts, in order.
    """
    return [candidate.text for candidate in candidates[slot]]

def ids_saying(clips: Sequence[SemanticClip], *texts: str) -> List[str]:
    """Find the clips that say each text.

    Args:
        clips: Clips to search.
        *texts: What they say.

    Returns:
        Their IDs, in the order of the texts.
    """
    return [next(clip.id for clip in clips if clip.text == text) for text in texts]

def test_every_slot_of_a_complete_ad_has_candidates() -> None:
    candidates = slot_candidates(clips_of(CUP_HOLDER), product="杯架")
    # 「現在下」 fits in three seconds but is three characters, too little to open on.
    assert said(candidates, AdSlot.HOOK) == ["你是不是每次出門", "杯子都沒地方放"]
    assert said(candidates, AdSlot.PROBLEM) == ["你是不是每次出門"]
    assert said(candidates, AdSlot.PRODUCT) == ["這款杯架夾在桌邊就能用"]
    assert said(candidates, AdSlot.PROOF) == ["我自己實測放了兩公斤都不會掉"]
    assert missing_slots(candidates) == []

def test_a_call_to_action_split_across_two_phrases_is_found_in_both() -> None:
    candidates = slot_candidates(clips_of(CUP_HOLDER))
    cta = {candidate.text: candidate.why for candidate in candidates[AdSlot.CTA]}
    # 「現在下」 alone says nothing; joined with the next phrase it is 「現在下單」.
    assert "says 「下單」" in cta["現在下"]
    assert "says 「下單」" in cta["單只要三百九十九元"]
    assert "says 「三百九十九元」" in cta["單只要三百九十九元"]

def test_a_marker_wholly_in_the_neighbouring_phrase_does_not_count() -> None:
    candidates = slot_candidates(clips_of(CUP_HOLDER))
    # 「實測」 is said in the phrase before 「現在下」 and does not reach into it.
    assert "現在下" not in said(candidates, AdSlot.PROOF)

def test_spaces_between_characters_do_not_hide_a_marker() -> None:
    candidates = slot_candidates(clips_of([(0.0, 4.0, "這 款 杯 架 真 的 好 用"), (5.0, 9.0, "連 結 在 下 面")]), "杯架")
    assert len(candidates[AdSlot.PRODUCT]) == 1
    assert "says 「杯架」" in candidates[AdSlot.PRODUCT][0].why
    assert len(candidates[AdSlot.CTA]) == 1

def test_a_hook_is_a_whole_sentence_said_within_three_seconds() -> None:
    candidates = slot_candidates(clips_of([
        (0.0, 0.8, "嗯"),
        (1.5, 2.3, "然後"),
        (3.0, 3.0 + HOOK_SECONDS, "你絕對沒看過這種"),
        (7.0, 7.0 + HOOK_SECONDS + 1.0, "今天要跟大家分享一個東西"),
    ]))
    # A filler is too little to open on, and a sentence over three seconds too much.
    assert said(candidates, AdSlot.HOOK) == ["你絕對沒看過這種"]

def test_a_price_is_found_in_any_way_it_is_written() -> None:
    for text in ("只要３９９元", "特價NT$399", "現在打八折", "全館79折"):
        candidates = slot_candidates(clips_of([(0.0, 4.0, text)]))
        assert said(candidates, AdSlot.CTA) == [text], text

def test_latin_markers_are_whole_words() -> None:
    candidates = slot_candidates(clips_of([(0.0, 4.0, "scan the barcode"), (5.0, 9.0, "buy now")]))
    assert said(candidates, AdSlot.CTA) == ["buy now"]

def test_a_product_name_in_latin_is_found_between_chinese() -> None:
    candidates = slot_candidates(clips_of([(0.0, 4.0, "這支iPhone殼很薄")]), product="iPhone")
    assert "says 「iphone」" in candidates[AdSlot.PRODUCT][0].why

def test_a_shot_described_as_showing_the_product_is_a_product_candidate() -> None:
    clips = clips_of(CUP_HOLDER)
    quiet = SemanticClip(
        id="shot", timeline_id=clips[0].timeline_id, asset_id=ASSET_ID, level=ClipLevel.UTTERANCE,
        source_range=Span(start=19.0, end=20.0), kind=ClipKind.AMBIENT, description="杯架夾在桌邊的特寫",
    )
    candidates = slot_candidates([*clips, quiet], product="杯架")
    assert candidates[AdSlot.PRODUCT][-1].clip_id == "shot"
    assert candidates[AdSlot.PRODUCT][-1].why == ("described as showing 「杯架」",)
    # Without a name, a description that says it shows a product still counts.
    tagged = quiet.model_copy(update={"description": "商品特寫"})
    assert slot_candidates([*clips, tagged])[AdSlot.PRODUCT][-1].clip_id == "shot"

def test_a_slot_with_nothing_to_fill_it_is_reported_with_what_it_needs() -> None:
    candidates = slot_candidates(clips_of(CUP_HOLDER[:4]))
    assert candidates[AdSlot.CTA] == []
    assert [entry["slot"] for entry in missing_slots(candidates)] == ["cta"]
    assert "price" in missing_slots(candidates)[0]["needs"]

def test_an_assignment_from_the_candidates_in_order_is_accepted() -> None:
    clips = clips_of(CUP_HOLDER)
    candidates = slot_candidates(clips, "杯架")
    hook, product, proof, now, price = ids_saying(
        clips, "杯子都沒地方放", "這款杯架夾在桌邊就能用", "我自己實測放了兩公斤都不會掉", "現在下", "單只要三百九十九元",
    )
    chosen = [
        SlotChoice(slot=AdSlot.HOOK, clip_ids=[hook]),
        SlotChoice(slot=AdSlot.PRODUCT, clip_ids=[product]),
        SlotChoice(slot=AdSlot.PROOF, clip_ids=[proof]),
        SlotChoice(slot=AdSlot.CTA, clip_ids=[now, price]),
    ]
    assert check_slots(chosen, candidates) == []

def test_a_call_to_action_before_the_proof_is_refused() -> None:
    clips = clips_of(CUP_HOLDER)
    candidates = slot_candidates(clips)
    proof, price = ids_saying(clips, "我自己實測放了兩公斤都不會掉", "單只要三百九十九元")
    problems = check_slots([SlotChoice(slot=AdSlot.CTA, clip_ids=[price]),
                            SlotChoice(slot=AdSlot.PROOF, clip_ids=[proof])], candidates)
    assert problems == ["proof: comes after cta, but a commercial plays hook → problem → product → proof → cta"]

def test_a_clip_that_was_not_offered_for_a_slot_is_refused() -> None:
    clips = clips_of(CUP_HOLDER)
    candidates = slot_candidates(clips)
    (price,) = ids_saying(clips, "單只要三百九十九元")
    problems = check_slots([SlotChoice(slot=AdSlot.HOOK, clip_ids=[price])], candidates)
    assert problems == [f"hook: {price} is not one of its candidates"]

def test_one_clip_in_two_slots_and_a_slot_given_twice_are_refused() -> None:
    clips = clips_of(CUP_HOLDER)
    candidates = slot_candidates(clips)
    (opening,) = ids_saying(clips, "你是不是每次出門")
    problems = check_slots([
        SlotChoice(slot=AdSlot.HOOK, clip_ids=[opening]),
        SlotChoice(slot=AdSlot.PROBLEM, clip_ids=[opening]),
        SlotChoice(slot=AdSlot.PROBLEM, clip_ids=[opening]),
    ], candidates)
    assert problems == [
        f"problem: {opening} is already in hook; a clip fills one slot",
        "problem: given twice; put all of its clips in one entry",
    ]

def test_a_slot_with_no_candidates_cannot_be_filled() -> None:
    clips = clips_of(CUP_HOLDER[:4])
    candidates = slot_candidates(clips)
    (last,) = ids_saying(clips, "我自己實測放了兩公斤都不會掉")
    (problem,) = check_slots([SlotChoice(slot=AdSlot.CTA, clip_ids=[last])], candidates)
    assert problem.startswith("cta: nothing in this footage was offered for it; report it missing")

def test_the_fingerprint_is_stable_and_covers_the_product() -> None:
    clips = clips_of(CUP_HOLDER)
    first = slot_hash(slot_candidates(clips, "杯架"), "杯架")
    assert first == slot_hash(slot_candidates(clips, "杯架"), "杯架")
    assert first != slot_hash(slot_candidates(clips), None)

@pytest.fixture
def ad_timeline(media) -> dict:
    """Build a timeline of the cup holder ad through the tools.

    Returns:
        A dictionary with the `timeline_id` and the `ids` of its clips by what they say.
    """
    asset_id = import_asset(str(media / "wide.mp4"))["id"]
    runtime_repo.save_analysis(chinese_analysis(CUP_HOLDER, asset_id=asset_id))
    built = build_semantic_timeline([asset_id], rebuild=True)
    clips = runtime_repo.query_semantic_clips(built["timeline_id"], level=ClipLevel.UTTERANCE)
    return {"timeline_id": built["timeline_id"], "ids": {clip.text: clip.id for clip in clips if clip.text}}

def test_proposing_an_ad_lists_every_slot_in_order(ad_timeline: dict) -> None:
    proposed = propose_sections(timeline_id=ad_timeline["timeline_id"], structure="ad", product="杯架")
    assert [entry["slot"] for entry in proposed["slots"]] == ["hook", "problem", "product", "proof", "cta"]
    product = proposed["slots"][2]["candidates"]
    assert [entry["text"] for entry in product] == ["這款杯架夾在桌邊就能用"]
    assert set(product[0]) == {"clip_id", "asset_id", "start", "end", "text", "why"}
    assert proposed["missing"] == []
    assert proposed["slot_hash"]
    assert "chosen" not in proposed
    # Without the structure, nothing about slots is said.
    assert "slots" not in propose_sections(timeline_id=ad_timeline["timeline_id"])

def test_a_product_without_a_structure_is_refused(ad_timeline: dict) -> None:
    with pytest.raises(ValueError, match='structure="ad"'):
        propose_sections(timeline_id=ad_timeline["timeline_id"], product="杯架")

def test_assigned_slots_are_pinned_and_leave_the_sections_alone(ad_timeline: dict) -> None:
    ids = ad_timeline["ids"]
    stored = set_sections(
        slots=[
            # Short IDs, as the AI is shown them, are accepted.
            SlotChoice(slot=AdSlot.HOOK, clip_ids=[ids["你是不是每次出門"].split(":", 1)[1]]),
            SlotChoice(slot=AdSlot.PRODUCT, clip_ids=[ids["這款杯架夾在桌邊就能用"]]),
            SlotChoice(slot=AdSlot.CTA, clip_ids=[ids["現在下"], ids["單只要三百九十九元"]]),
        ],
        timeline_id=ad_timeline["timeline_id"], chosen_by="test-model/1", product="杯架",
    )
    assert stored["slots"] == {"assigned": ["hook", "product", "cta"], "left_out": ["problem", "proof"], "missing": []}
    assert stored["sections"] == 0

    timeline = runtime_repo.get_semantic_timeline(ad_timeline["timeline_id"])
    assert timeline.ad_slots[0].clip_ids == [ids["你是不是每次出門"]]
    assert timeline.ad_product == "杯架"
    assert timeline.slots_by == "test-model/1"
    assert timeline.slot_hash
    assert ClipLevel.SECTION not in timeline.levels
    assert len(query_clips(timeline_id=ad_timeline["timeline_id"], level=ClipLevel.UTTERANCE)["clips"]) > 0

    proposed = propose_sections(timeline_id=ad_timeline["timeline_id"], structure="ad")
    assert proposed["chosen"]["current"] is True
    assert proposed["chosen"]["slots"][2] == {"slot": "cta", "clip_ids": [ids["現在下"], ids["單只要三百九十九元"]]}

def test_a_changed_candidate_list_marks_the_assignment_out_of_date(ad_timeline: dict) -> None:
    ids = ad_timeline["ids"]
    set_sections(slots=[SlotChoice(slot=AdSlot.HOOK, clip_ids=[ids["你是不是每次出門"]])],
                 timeline_id=ad_timeline["timeline_id"], product="杯架")
    # Somebody looked and wrote that the proof shot shows the product: it is now a product candidate.
    set_clip_tags([ClipDescription(clip_id=ids["我自己實測放了兩公斤都不會掉"], description="手拿杯架")],
                  written_by="test", reviewed=True, timeline_id=ad_timeline["timeline_id"])
    proposed = propose_sections(timeline_id=ad_timeline["timeline_id"], structure="ad")
    assert proposed["chosen"]["current"] is False

def test_slots_that_do_not_hold_up_store_nothing(ad_timeline: dict) -> None:
    ids = ad_timeline["ids"]
    with pytest.raises(ValueError, match="nothing was stored") as raised:
        set_sections(slots=[SlotChoice(slot=AdSlot.CTA, clip_ids=[ids["單只要三百九十九元"]]),
                            SlotChoice(slot=AdSlot.HOOK, clip_ids=[ids["這款杯架夾在桌邊就能用"]])],
                     timeline_id=ad_timeline["timeline_id"])
    assert "hook: comes after cta" in str(raised.value)
    assert "is not one of its candidates" in str(raised.value)
    assert runtime_repo.get_semantic_timeline(ad_timeline["timeline_id"]).ad_slots == []

def test_sections_and_slots_go_in_together_or_not_at_all(ad_timeline: dict) -> None:
    proposed = propose_sections(timeline_id=ad_timeline["timeline_id"])
    utterances = [clip["clip_id"] for clip in proposed["utterances"]]
    whole = [SectionChoice(first_clip_id=utterances[0], last_clip_id=utterances[-1], name="整支廣告")]
    half = [SectionChoice(first_clip_id=utterances[0], last_clip_id=utterances[1], name="一半")]
    hook = [SlotChoice(slot=AdSlot.HOOK, clip_ids=[ad_timeline["ids"]["你是不是每次出門"]])]

    with pytest.raises(ValueError, match="are in no section"):
        set_sections(half, timeline_id=ad_timeline["timeline_id"], slots=hook)
    assert runtime_repo.get_semantic_timeline(ad_timeline["timeline_id"]).ad_slots == []

    stored = set_sections(whole, timeline_id=ad_timeline["timeline_id"], slots=hook)
    assert stored["sections"] == 1
    timeline = runtime_repo.get_semantic_timeline(ad_timeline["timeline_id"])
    assert timeline.ad_slots == hook
    assert ClipLevel.SECTION in timeline.levels

def test_a_product_without_slots_is_refused(ad_timeline: dict) -> None:
    proposed = propose_sections(timeline_id=ad_timeline["timeline_id"])
    utterances = [clip["clip_id"] for clip in proposed["utterances"]]
    whole = [SectionChoice(first_clip_id=utterances[0], last_clip_id=utterances[-1], name="整支廣告")]
    with pytest.raises(ValueError, match="send it with slots"):
        set_sections(whole, timeline_id=ad_timeline["timeline_id"], product="杯架")
