"""Tests for candidate section boundaries and for the sections chosen from them.

Candidate generation is deterministic, so it is checked against footage laid
out to carry one signal at a time. What a model does with the candidates is
not testable here; what is testable, and what these cover, is that nothing
outside the candidates can become a boundary and that a set of sections which
does not cover the footage is refused rather than repaired.
"""

from typing import List

import pytest

from app.engine.sections import (
    CANDIDATES_PER_MINUTE,
    MIN_CANDIDATES,
    SIGNAL_WEIGHTS,
    Candidate,
    build_sections,
    candidate_hash,
    check_sections,
    propose_candidates,
    section_kind,
)
from app.engine.semantic import build_timeline, combined_from_clips
from app.models.media import Span
from app.models.semantic import ClipKind, ClipLevel, SectionChoice, SemanticClip
from app.server import (
    build_semantic_timeline,
    import_asset,
    propose_sections,
    query_clips,
    set_sections,
)
from app.server import repo as server_repo
from test_semantic_timeline import ASSET_ID, analysis, asset

TWO_SUBJECTS = dict(
    duration=20.0,
    segments=[
        (1.0, 3.0, "我們在設計這個系統"),
        (3.0, 5.0, "系統的設計花了很久"),
        (5.0, 7.0, "設計上有很多取捨"),
        (10.0, 12.0, "今天午餐吃什麼"),
        (12.0, 14.0, "午餐我想吃拉麵"),
        (14.0, 16.0, "拉麵店就在附近"),
    ],
    silences=[(0.0, 1.0), (7.0, 10.0), (16.0, 20.0)],
)

def timeline_of(**kwargs) -> tuple:
    """Build a timeline and its candidates from one asset's analysis.

    Args:
        **kwargs: Passed to the analysis builder.

    Returns:
        `(clips, candidates)`.
    """
    made = analysis(**kwargs)
    _, clips = build_timeline({ASSET_ID: asset()}, {ASSET_ID: made})
    return clips, propose_candidates(clips, {ASSET_ID: made})

def score_at(candidates: List[Candidate], seconds: float) -> float:
    """Read the score of the candidate at a time.

    Args:
        candidates: Candidates to search.
        seconds: Time of the boundary.

    Returns:
        The score, or 0.0 if nothing was proposed there.
    """
    return next((candidate.score for candidate in candidates if candidate.at == pytest.approx(seconds)), 0.0)

def test_a_long_pause_is_the_strongest_boundary() -> None:
    _, candidates = timeline_of(**TWO_SUBJECTS)
    # The three-second gap where the subject changes, against a place mid-flow.
    assert score_at(candidates, 10.0) > score_at(candidates, 3.0)
    assert score_at(candidates, 10.0) >= SIGNAL_WEIGHTS["pause"]

def test_a_turn_in_the_vocabulary_is_noticed_without_a_pause() -> None:
    # No silence anywhere, so the pause signal cannot be what finds the boundary.
    _, candidates = timeline_of(
        duration=12.0,
        segments=[
            (0.0, 2.0, "我們在設計這個系統"),
            (2.0, 4.0, "系統的設計花了很久"),
            (4.0, 6.0, "設計上有很多取捨"),
            (6.0, 8.0, "今天午餐吃什麼"),
            (8.0, 10.0, "午餐我想吃拉麵"),
            (10.0, 12.0, "拉麵店就在附近"),
        ],
    )
    assert score_at(candidates, 6.0) > 0
    assert score_at(candidates, 6.0) == max(candidate.score for candidate in candidates)

def test_a_phrase_that_changes_the_subject_counts_towards_a_boundary() -> None:
    _, plain = timeline_of(duration=8.0, segments=[(0.0, 4.0, "這裡講一件事"), (4.0, 8.0, "這裡講一件事")])
    _, marked = timeline_of(duration=8.0, segments=[(0.0, 4.0, "這裡講一件事"), (4.0, 8.0, "那我們接下來看別的")])
    assert score_at(marked, 4.0) > score_at(plain, 4.0)

def test_a_boundary_with_nothing_to_say_for_it_is_not_offered() -> None:
    # One shot, no speech, no silences: nothing anywhere suggests a section starts.
    _, candidates = timeline_of(duration=8.0, scenes=[(0.0, 8.0)])
    assert candidates == []

def test_the_list_of_candidates_stays_readable_on_long_footage() -> None:
    segments = [(index * 2.0, index * 2.0 + 2.0, f"第{index}句話講的是某件事情") for index in range(120)]
    _, candidates = timeline_of(duration=240.0, segments=segments)
    assert len(candidates) <= max(MIN_CANDIDATES, round(240 / 60 * CANDIDATES_PER_MINUTE))

def test_candidates_are_the_same_every_time() -> None:
    _, first = timeline_of(**TWO_SUBJECTS)
    _, second = timeline_of(**TWO_SUBJECTS)
    assert [candidate.after_clip_id for candidate in first] == [candidate.after_clip_id for candidate in second]
    assert candidate_hash(first) == candidate_hash(second)

def test_changing_the_boundaries_changes_their_fingerprint() -> None:
    _, candidates = timeline_of(**TWO_SUBJECTS)
    assert candidate_hash(candidates) != candidate_hash(candidates[:-1])

def speech_ids(clips) -> List[str]:
    """List the clip IDs in order.

    Args:
        clips: Clips to list.

    Returns:
        Their IDs.
    """
    return [clip.id for clip in clips]

def test_sections_that_cover_the_footage_are_accepted() -> None:
    clips, candidates = timeline_of(**TWO_SUBJECTS)
    split = candidates[0].index
    ids = speech_ids(clips)
    assert check_sections([(ids[0], ids[split - 1]), (ids[split], ids[-1])], clips, candidates) == []

def test_a_boundary_that_was_never_offered_is_refused() -> None:
    clips, candidates = timeline_of(**TWO_SUBJECTS)
    ids = speech_ids(clips)
    offered = {candidate.index for candidate in candidates}
    never = next(index for index in range(1, len(ids)) if index not in offered)
    problems = check_sections([(ids[0], ids[never - 1]), (ids[never], ids[-1])], clips, candidates)
    assert any("not one of the candidate boundaries" in problem for problem in problems)

def test_a_gap_between_two_sections_is_refused() -> None:
    clips, candidates = timeline_of(**TWO_SUBJECTS)
    ids = speech_ids(clips)
    # A split far enough in that a section can be left short of it.
    split = next(candidate.index for candidate in candidates if candidate.index >= 2)
    problems = check_sections([(ids[0], ids[split - 2]), (ids[split], ids[-1])], clips, candidates)
    assert any("nothing in between" in problem for problem in problems)

def test_a_section_reaching_past_the_end_of_the_file_is_refused() -> None:
    clips, candidates = timeline_of(**TWO_SUBJECTS)
    ids = speech_ids(clips)
    # The first section already swallowed the file, so the second has nothing to start on.
    problems = check_sections([(ids[0], ids[-1]), (ids[1], ids[-1])], clips, candidates)
    assert any("already in an earlier section" in problem for problem in problems)

def test_footage_left_out_of_every_section_is_refused() -> None:
    clips, candidates = timeline_of(**TWO_SUBJECTS)
    ids = speech_ids(clips)
    split = candidates[0].index
    problems = check_sections([(ids[0], ids[split - 1])], clips, candidates)
    assert any("are in no section" in problem for problem in problems)

def test_a_section_that_ends_before_it_starts_is_refused() -> None:
    clips, candidates = timeline_of(**TWO_SUBJECTS)
    ids = speech_ids(clips)
    assert any("ends before it starts" in problem for problem in check_sections([(ids[3], ids[1])], clips, candidates))

def test_a_section_cannot_run_across_two_files() -> None:
    assets = {"a": asset("a"), "b": asset("b")}
    analyses = {
        "a": analysis(4.0, segments=[(0.0, 4.0, "甲檔說的話")], asset_id="a"),
        "b": analysis(4.0, segments=[(0.0, 4.0, "乙檔說的話")], asset_id="b"),
    }
    _, clips = build_timeline(assets, analyses)
    candidates = propose_candidates(clips, analyses)
    problems = check_sections([(clips[0].id, clips[-1].id)], clips, candidates)
    assert any("starts in one file and ends in another" in problem for problem in problems)

def test_a_built_section_spans_its_members_and_claims_them() -> None:
    clips, candidates = timeline_of(**TWO_SUBJECTS)
    split = candidates[0].index
    ids = speech_ids(clips)
    sections, utterances = build_sections("tl_test", clips, [
        SectionChoice(first_clip_id=ids[0], last_clip_id=ids[split - 1], name="系統設計", topic="工作", summary="講設計"),
        SectionChoice(first_clip_id=ids[split], last_clip_id=ids[-1], name="午餐", topic="閒聊", summary="講吃的"),
    ])
    assert [section.level for section in sections] == [ClipLevel.SECTION, ClipLevel.SECTION]
    assert sections[0].source_range.start == clips[0].source_range.start
    assert sections[-1].source_range.end == clips[-1].source_range.end
    assert sections[0].safe_in == clips[0].safe_in
    assert [section.name for section in sections] == ["系統設計", "午餐"]
    assert {utterance.parent_id for utterance in utterances} == {section.id for section in sections}

def test_a_section_scores_as_much_as_its_members_weigh() -> None:
    clips, candidates = timeline_of(
        duration=8.0,
        segments=[(0.0, 2.0, "亮的"), (2.0, 8.0, "暗的")],
        black=[(2.0, 8.0)],
    )
    ids = speech_ids(clips)
    sections, _ = build_sections("tl_test", clips, [
        SectionChoice(first_clip_id=ids[0], last_clip_id=ids[-1], name="整段", summary=""),
    ])
    # Six of the eight seconds are black, and the score says so rather than rounding to all or nothing.
    assert sections[0].scores["black"] == pytest.approx(0.75, abs=0.01)

def test_a_section_is_speech_when_anyone_talks_in_it() -> None:
    clips, _ = timeline_of(**TWO_SUBJECTS)
    assert section_kind(clips) == ClipKind.SPEECH
    assert section_kind([clip for clip in clips if clip.kind == ClipKind.SILENCE]) == ClipKind.SILENCE

@pytest.fixture
def sectioned(media) -> dict:
    """Build a timeline through the tools and section it in two.

    Returns:
        A dictionary with the `timeline_id` and the `sections` that were sent.
    """
    asset_id = import_asset(str(media / "wide.mp4"))["id"]
    server_repo.save_analysis(analysis(asset_id=asset_id, **TWO_SUBJECTS))
    built = build_semantic_timeline([asset_id], rebuild=True)
    proposed = propose_sections(timeline_id=built["timeline_id"])

    utterances = [clip["clip_id"] for clip in proposed["utterances"]]
    opening = proposed["candidates"][0]["after_clip_id"]
    split = utterances.index(opening)
    sections = [
        SectionChoice(first_clip_id=utterances[0], last_clip_id=utterances[split - 1],
                      name="系統設計", topic="工作", summary="講當初怎麼設計"),
        SectionChoice(first_clip_id=opening, last_clip_id=utterances[-1],
                      name="午餐", topic="閒聊", summary="講中午要吃什麼"),
    ]
    set_sections(sections, timeline_id=built["timeline_id"], chosen_by="test-model/1")
    return {"timeline_id": built["timeline_id"], "sections": sections}

def test_sections_come_back_from_a_search_with_their_names(sectioned: dict) -> None:
    found = query_clips(timeline_id=sectioned["timeline_id"], level=ClipLevel.SECTION)
    assert [clip["name"] for clip in found["clips"]] == ["系統設計", "午餐"]
    assert [clip["topic"] for clip in found["clips"]] == ["工作", "閒聊"]

def test_a_topic_gathers_the_sections_that_share_it(sectioned: dict) -> None:
    found = query_clips(timeline_id=sectioned["timeline_id"], level=ClipLevel.SECTION, topic="閒聊")
    assert [clip["name"] for clip in found["clips"]] == ["午餐"]

def test_what_judged_the_sections_is_recorded(sectioned: dict) -> None:
    timeline = server_repo.get_semantic_timeline(sectioned["timeline_id"])
    assert timeline.sections_by == "test-model/1"
    assert timeline.candidate_hash
    assert ClipLevel.SECTION in timeline.levels

def test_sections_that_do_not_hold_up_are_handed_back_with_the_reasons(sectioned: dict) -> None:
    broken = [SectionChoice(first_clip_id=sectioned["sections"][0].first_clip_id,
                            last_clip_id=sectioned["sections"][0].last_clip_id, name="只有一半")]
    with pytest.raises(ValueError, match="are in no section"):
        set_sections(broken, timeline_id=sectioned["timeline_id"])
    # Nothing was written: the sections already stored are untouched.
    assert len(query_clips(timeline_id=sectioned["timeline_id"], level=ClipLevel.SECTION)["clips"]) == 2

def test_sending_sections_again_replaces_them(sectioned: dict) -> None:
    revised = [section.model_copy(update={"name": section.name + "（修）"}) for section in sectioned["sections"]]
    set_sections(revised, timeline_id=sectioned["timeline_id"])
    found = query_clips(timeline_id=sectioned["timeline_id"], level=ClipLevel.SECTION)
    assert [clip["name"] for clip in found["clips"]] == ["系統設計（修）", "午餐（修）"]
    assert len(found["clips"]) == 2

def test_setting_sections_needs_some(sectioned: dict) -> None:
    with pytest.raises(ValueError, match="at least one section"):
        set_sections([], timeline_id=sectioned["timeline_id"])

def test_a_section_averages_a_measurement_over_only_the_parts_it_was_taken_in() -> None:
    """A stretch nobody measured must not read as a stretch that measured well."""
    members = [
        SemanticClip(
            id="c1", timeline_id="tl", asset_id=ASSET_ID, level=ClipLevel.UTTERANCE,
            source_range=Span(start=0, end=1), kind=ClipKind.SPEECH,
            scores={"speech": 1.0, "blur": 6.0, "shake": 0.04},
        ),
        SemanticClip(
            id="c2", timeline_id="tl", asset_id=ASSET_ID, level=ClipLevel.UTTERANCE,
            source_range=Span(start=1, end=4), kind=ClipKind.AMBIENT,
            scores={"speech": 0.0},
        ),
    ]
    scores = combined_from_clips(members, whole=4.0)
    # Three of the four seconds carry no speech, and that is what makes the section quiet.
    assert scores["speech"] == 0.25
    # Those same three seconds were never measured for blur, so they say nothing about it.
    assert scores["blur"] == 6.0
    # Three decimals would round an unsteady section down to nothing.
    assert scores["shake"] == 0.04

def test_a_long_member_is_not_outvoted_by_a_crowd_of_short_ones() -> None:
    """A section made mostly of one long shot sounds like that shot, not like the lines."""
    lines = [
        SemanticClip(
            id=f"s{index}", timeline_id="tl", asset_id=ASSET_ID, level=ClipLevel.UTTERANCE,
            source_range=Span(start=index * 0.5, end=index * 0.5 + 0.5), kind=ClipKind.SPEECH,
            scores={"noise_floor": -50.0},
        )
        for index in range(20)
    ]
    long_shot = SemanticClip(
        id="long", timeline_id="tl", asset_id=ASSET_ID, level=ClipLevel.UTTERANCE,
        source_range=Span(start=10, end=70), kind=ClipKind.AMBIENT,
        scores={"noise_floor": -30.0},
    )
    scores = combined_from_clips([*lines, long_shot], whole=70.0)
    # Ten seconds of lines against a minute of hiss: the minute is what the section is.
    assert scores["noise_floor"] == -30.0
