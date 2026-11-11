"""Tests for a commercial's variants: which slots survive at each length, refusing what cannot be had, hook tests.

The rules are tested on their own with lengths made up for the purpose, so
that which combinations are offered at 6, 15 and 30 seconds can be read off
the test. The tools are tested on the cup rack talk of `test_ad_slots`: a
variant is a branch of the confirmed cut, with its own plan, captions made
again on its cut, and everything after a swapped hook left as it was.
"""

from typing import Dict, Tuple

import pytest

from app.engine.ad_slots import slot_candidates, variant_slot_problems
from app.engine.ad_variants import Unit, cut_downs, derive, format_for, kept_effects, refusal, window
from app.engine.plan.effects import Moment
from app.engine.subtitles import place_cues
from app.models.plan import Beat, BeatRole, EditPlan, Fact, Selection, SoundEffectPick, SoundEffects, Structure
from app.models.semantic import AdSlot, SlotChoice
from app.runtime import repo as runtime_repo
from app.tools.editing import create_project
from app.tools.planning import compile_plan, save_plan
from app.tools.subtitles import generate_subtitles
from app.tools.variants import compare_variants, make_ad_variants, propose_ad_variants
from test_ad_slots import RACK_BEATS, clips_of, rack_timeline, set_rack_slots  # noqa: F401 — the fixture

# A commercial of every slot, each line as long as it runs alone.
LINES = [
    Unit("hook", AdSlot.HOOK, 2.5, "你是不是每次出門"),
    Unit("problem", AdSlot.PROBLEM, 4.0, "杯子都沒地方放"),
    Unit("product1", AdSlot.PRODUCT, 3.0, "這款杯架夾在桌邊就能用"),
    Unit("product2", AdSlot.PRODUCT, 6.0, "不同的杯子都放得穩"),
    Unit("proof", AdSlot.PROOF, 7.0, "我自己實測放了兩公斤都不會掉"),
    Unit("cta", AdSlot.CTA, 2.0, "連結在下面"),
]

def added(keeps: Tuple[str, ...]) -> float:
    """The length of a cut as the sum of its lines, as a compile with no air would lay it."""
    return round(sum(unit.seconds for unit in LINES if unit.clip_id in keeps), 3)

def offered(seconds: float) -> Dict[Tuple[str, ...], float]:
    """Every cut-down of `LINES` to a length, by what it keeps."""
    fits, _, _ = cut_downs(LINES, seconds, 0.0, added)
    return {cut.keeps: cut.seconds for cut in fits}

def test_a_bumper_is_one_hook_one_product_line_and_the_ask_never_the_problem_or_proof() -> None:
    assert format_for(6).label == "bumper"
    # 2.5 + 3.0 = 5.5 and with the ask 7.5; only the first is under the 6.5 s ceiling.
    assert offered(6) == {("hook", "product1"): 5.5}

def test_a_fifteen_second_spot_drops_the_problem_and_the_proof_and_keeps_whole_lines() -> None:
    cuts = offered(15)
    assert ("hook", "product1", "product2", "cta") in cuts
    assert all(not {"problem", "proof"} & set(keeps) for keeps in cuts)
    assert all(window(15)[0] <= seconds <= window(15)[1] for seconds in cuts.values())

def test_a_thirty_second_spot_may_carry_the_problem_and_the_proof() -> None:
    cuts = offered(30)
    assert ("hook", "problem", "product1", "product2", "proof", "cta") in cuts
    assert any("proof" in keeps for keeps in cuts)

def test_every_cut_keeps_a_hook_and_the_product() -> None:
    for seconds in (6, 15, 30):
        for keeps in offered(seconds):
            assert "hook" in keeps and ({"product1", "product2"} & set(keeps))

def test_a_length_no_whole_sentences_reach_is_refused_with_what_the_shortest_runs() -> None:
    long_product = [LINES[0], Unit("product", AdSlot.PRODUCT, 7.8, "各位這是現在的樣子")]
    fits, shortest, longest = cut_downs(long_product, 6, 0.0, lambda keeps: 10.3)
    assert fits == []
    said = refusal(6, long_product, shortest, longest)
    assert said.startswith("6s: the shortest cut that keeps a whole hook and the product is hook 「你是不是每次出門」")
    assert "runs 10.3s as compiled — over the 6.5s a 6s spot may run" in said
    assert "without cutting into a sentence" in said

def test_a_length_longer_than_everything_is_refused_as_too_short_not_padded() -> None:
    fits, shortest, longest = cut_downs(LINES[:3], 30, 0.0, added)
    assert fits == []
    assert "under the 22.5s a 30s spot should at least run" in refusal(30, LINES[:3], shortest, longest)

def plan_of_lines(timeline_id: str = "tl_00000000") -> EditPlan:
    """An ad plan of `LINES`, one beat per slot, each line establishing a fact of its own."""
    beats = [Beat(id=unit.clip_id, name=unit.clip_id, role=BeatRole(unit.slot.value)) for unit in LINES]
    return EditPlan(
        timeline_id=timeline_id, structure=Structure.AD,
        facts=[Fact(id=f"f_{unit.clip_id}", says=unit.text) for unit in LINES],
        beats=beats,
        selections=[Selection(clip_id=unit.clip_id, beat_id=unit.clip_id, rationale=unit.text,
                              establishes=[{"fact": f"f_{unit.clip_id}", "evidence": (0, 1)}]) for unit in LINES],
    )

def test_a_hook_swap_leaves_everything_after_the_hook_exactly_as_it_was() -> None:
    plan = plan_of_lines()
    variant, problems, dropped = derive(plan, "hook-B · full", ("other", "said in 2.0s"), None, None, "pinned")
    assert problems == []
    assert variant.selections[0].clip_id == "other"
    assert variant.selections[1:] == plan.selections[1:]
    assert variant.beats == plan.beats
    assert dropped["selections"] == ["hook"]
    assert variant.variant.slots[0] == SlotChoice(slot=AdSlot.HOOK, clip_ids=["other"])
    assert [rejection.clip_id for rejection in variant.rejected] == ["hook"]

def test_a_cut_down_drops_the_beats_and_facts_only_the_dropped_lines_had() -> None:
    plan = plan_of_lines()
    variant, problems, dropped = derive(plan, "hook-A · 15s", None, ["hook", "product1"], 15, "p")
    assert problems == []
    assert [beat.id for beat in variant.beats] == ["hook", "product1"]
    assert dropped["beats"] == ["problem", "product2", "proof", "cta"]
    assert [fact.id for fact in variant.facts] == ["f_hook", "f_product1"]
    assert variant.target.seconds == 15 and variant.variant.target_seconds == 15

def test_a_cut_down_whose_kept_line_relies_on_a_dropped_fact_is_not_made() -> None:
    plan = plan_of_lines()
    plan.selections[2].requires = ["f_problem"]
    _, problems, _ = derive(plan, "hook-A · 15s", None, ["hook", "product1"], 15, "p")
    assert problems == ["product1 relies on f_problem, which only a selection this variant leaves out establishes"]

def test_sound_effects_on_moments_still_in_the_cut_are_kept_and_the_rest_dropped_with_the_reason() -> None:
    chosen = SoundEffects(candidate_hash="old", picks=[
        SoundEffectPick(candidate="cut:u0010", asset_id="whoosh"),
        SoundEffectPick(candidate="cta:brand-end", asset_id="impact"),
    ])
    moments = [Moment("cut:u0017", "cut", 3.0, (), ""), Moment("cta:brand-end", "cta", 10.0, (), "")]
    kept, dropped = kept_effects(chosen, moments, "new")
    assert kept == SoundEffects(candidate_hash="new", picks=[chosen.picks[1]])
    assert dropped == [{"candidate": "cut:u0010", "why": "that moment is not in this variant's cut"}]
    assert kept_effects(chosen, moments[:1], "new") == (None, [
        {"candidate": "cut:u0010", "why": "that moment is not in this variant's cut"},
        {"candidate": "cta:brand-end", "why": "that moment is not in this variant's cut"},
    ])

def test_a_variant_may_leave_slot_clips_out_but_not_add_or_reorder_them() -> None:
    clips = clips_of([(0.0, 1.8, "你看這個杯子"), (2.2, 4.0, "這個機構往下一壓就出來"), (4.4, 6.2, "這個機構再壓一次")],
                     duration=10.0)
    offered_now = slot_candidates(clips, "抽杯架", ["機構"])
    hook, first, second = (clip.id for clip in clips if clip.text)
    stored = [SlotChoice(slot=AdSlot.HOOK, clip_ids=[hook]), SlotChoice(slot=AdSlot.PRODUCT, clip_ids=[first, second])]
    assert variant_slot_problems([SlotChoice(slot=AdSlot.PRODUCT, clip_ids=[second])], stored, offered_now) == []
    (problem,) = variant_slot_problems([SlotChoice(slot=AdSlot.PRODUCT, clip_ids=[second, first])], stored, offered_now)
    assert "a variant leaves clips out, it does not add or reorder them" in problem
    # The second product line is also a hook candidate, so opening on it is a hook test.
    assert variant_slot_problems([SlotChoice(slot=AdSlot.HOOK, clip_ids=[second])], stored, offered_now) == []

@pytest.fixture
def confirmed_rack(kept_history, rack_timeline: dict) -> dict:  # noqa: F811 — the fixture imported above
    """The cup rack's commercial compiled and captioned, as a user would confirm it.

    The problem line is left out of the slots, so it stays free as another hook.
    """
    ids = rack_timeline["ids"]
    spans = {clip.id: (clip.source_range.start, clip.source_range.end)
             for clip in runtime_repo.query_semantic_clips(rack_timeline["timeline_id"])}
    beats = [beat for beat in RACK_BEATS if beat[0] != "problem"]
    set_rack_slots(rack_timeline, [SlotChoice(slot=AdSlot(role), clip_ids=[ids[text]]) for role, text in beats])
    saved = save_plan(EditPlan(
        timeline_id=rack_timeline["timeline_id"], structure=Structure.AD, goal="抽杯架的短廣告",
        facts=[Fact(id=f"f{index}", says=text) for index, (_, text) in enumerate(beats)],
        beats=[Beat(id=f"b{index}", name=role, role=BeatRole(role)) for index, (role, _) in enumerate(beats)],
        selections=[Selection(clip_id=ids[text], beat_id=f"b{index}",
                              establishes=[{"fact": f"f{index}", "evidence": spans[ids[text]]}])
                    for index, (_, text) in enumerate(beats)],
    ))
    assert saved["problems"] == [], saved["problems"]
    project = create_project(name="抽杯架 廣告")
    compiled = compile_plan(project["id"], project["version"], saved["plan_id"])
    generate_subtitles(project["id"], compiled["new_version"])
    return {**rack_timeline, "plan_id": saved["plan_id"], "project_id": project["id"]}

def test_a_hook_test_variant_is_a_branch_that_opens_on_the_other_hook_with_its_caption(confirmed_rack: dict) -> None:
    ids = confirmed_rack["ids"]
    proposed = propose_ad_variants(confirmed_rack["project_id"], confirmed_rack["plan_id"], targets=[30])
    names = {variant["name"]: variant for variant in proposed["variants"]}
    assert names["hook-B · full"]["hook_line"] == "以前的缺點就是會卡住"
    made = make_ad_variants(confirmed_rack["project_id"], confirmed_rack["plan_id"], proposed["variant_hash"],
                            ["hook-B · full"], targets=[30])["made"][0]
    branch = runtime_repo.get_project(made["project_id"])
    assert branch.branched_from.project_id == confirmed_rack["project_id"]
    original, variant = runtime_repo.get_plan(confirmed_rack["plan_id"]), runtime_repo.get_plan(made["plan_id"])
    assert variant.variant.of_plan_id == original.id
    assert variant.selections[0].clip_id == ids["以前的缺點就是會卡住"]
    assert variant.selections[1:] == original.selections[1:]
    assert place_cues(branch, branch.subtitles)[0].text == "以前的缺點就是會卡住"
    rows = compare_variants(made["project_id"], check=False)["rows"]
    assert [row["name"] for row in rows] == ["original", "hook-B · full"]
    # What is said in the first 3 s: a hook of 1.8 s runs into the line after it.
    assert rows[1]["hook_line"].startswith("以前的缺點就是會卡住")

def test_a_length_the_footage_cannot_reach_is_refused_and_cannot_be_made(confirmed_rack: dict) -> None:
    proposed = propose_ad_variants(confirmed_rack["project_id"], confirmed_rack["plan_id"], targets=[2])
    refused = [entry for entry in proposed["refused"] if entry["target"] == 2]
    assert refused and all(entry["why"].startswith("2s: the shortest cut that keeps a whole hook and the product")
                           for entry in refused)
    with pytest.raises(ValueError, match="not offered: hook-A · 2s"):
        make_ad_variants(confirmed_rack["project_id"], confirmed_rack["plan_id"], proposed["variant_hash"],
                         ["hook-A · 2s"], targets=[2])

def test_variants_picked_from_a_plan_that_changed_since_are_refused(confirmed_rack: dict) -> None:
    proposed = propose_ad_variants(confirmed_rack["project_id"], confirmed_rack["plan_id"], targets=[30])
    plan = runtime_repo.get_plan(confirmed_rack["plan_id"])
    runtime_repo.save_plan(plan.model_copy(update={"goal": "另一個目標"}), "changed")
    project = runtime_repo.get_project(confirmed_rack["project_id"])
    compile_plan(project.id, project.version, plan.id)
    with pytest.raises(ValueError, match="have changed since these were picked"):
        make_ad_variants(confirmed_rack["project_id"], confirmed_rack["plan_id"], proposed["variant_hash"],
                         ["hook-B · full"], targets=[30])
