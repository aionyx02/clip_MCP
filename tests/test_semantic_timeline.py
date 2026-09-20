"""Tests for the semantic timeline's `utterance` level and the store behind it.

The derivation is a pure function of a stored analysis, so these build the
analysis by hand rather than decoding anything: every rule about what becomes
a clip, what kind it is, and how much room its edges have is checked against
numbers chosen to exercise it.
"""

from datetime import timedelta
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pytest

from app.engine.semantic import (
    MIN_UTTERANCE_SECONDS,
    build_timeline,
    timeline_input_hash,
)
from app.models.media import (
    Asset,
    MediaAnalysis,
    Span,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)
from app.models.semantic import ClipDescription, ClipKind, ClipLevel
from app.server import (
    CLIP_SUMMARY_CHARACTERS,
    MAX_STORYBOARD_TILES,
    build_semantic_timeline,
    frames_for_clips,
    get_semantic_clip,
    import_asset,
    query_clips,
    set_clip_tags,
)
from app.server import repo as server_repo
from app.storage import repo as repo_module
from app.storage.repo import Repository

ASSET_ID = "asset-1"

def spans(pairs: Sequence[Tuple[float, float]]) -> list:
    """Build spans from start and end pairs.

    Args:
        pairs: `(start, end)` in seconds.

    Returns:
        The spans.
    """
    return [Span(start=start, end=end) for start, end in pairs]

def analysis(
    duration: float,
    segments: Sequence[Tuple[float, float, str]] = (),
    silences: Sequence[Tuple[float, float]] = (),
    scenes: Sequence[Tuple[float, float]] = (),
    black: Sequence[Tuple[float, float]] = (),
    frozen: Sequence[Tuple[float, float]] = (),
    asset_id: str = ASSET_ID,
) -> MediaAnalysis:
    """Build an analysis with exactly the detections a test needs.

    Args:
        duration: Length of the asset in seconds.
        segments: `(start, end, text)` per transcribed sentence. Each gets one
            word spanning it, which is what the speech score measures.
        silences: Detected silences.
        scenes: Detected shots.
        black: Detected black picture.
        frozen: Detected frozen picture.
        asset_id: Asset the analysis belongs to.

    Returns:
        The analysis.
    """
    transcript = None
    if segments:
        transcript = Transcript(language="zh", model="test", segments=[
            TranscriptSegment(
                start=start, end=end, text=text,
                words=[TranscriptWord(start=start, end=end, text=text)],
            )
            for start, end, text in segments
        ])
    return MediaAnalysis(
        asset_id=asset_id,
        duration=duration,
        scenes=spans(scenes),
        black_frames=spans(black),
        frozen_frames=spans(frozen),
        silences=spans(silences),
        transcript=transcript,
    )

def asset(asset_id: str = ASSET_ID, has_video: bool = True, has_audio: bool = True) -> Asset:
    """Build an asset record.

    Args:
        asset_id: ID for the asset.
        has_video: Whether it has a video stream.
        has_audio: Whether it has an audio stream.

    Returns:
        The asset.
    """
    return Asset(id=asset_id, path=f"/footage/{asset_id}.mp4", has_video=has_video, has_audio=has_audio)

def build(one_analysis: MediaAnalysis, one_asset: Optional[Asset] = None) -> list:
    """Derive one asset's clips.

    Args:
        one_analysis: Analysis to derive from.
        one_asset: The asset it describes; defaults to one with both streams.

    Returns:
        The clips.
    """
    one_asset = one_asset or asset(one_analysis.asset_id)
    return build_timeline({one_asset.id: one_asset}, {one_analysis.asset_id: one_analysis})[1]

def ranges(clips: Sequence) -> list:
    """Summarize clips as kind and rounded range.

    Args:
        clips: Clips to summarize.

    Returns:
        One `(kind, start, end)` per clip.
    """
    return [(clip.kind.value, clip.source_range.start, clip.source_range.end) for clip in clips]

def test_each_transcribed_sentence_becomes_one_clip() -> None:
    clips = build(analysis(
        10.0,
        segments=[(1.0, 3.0, "第一句"), (4.0, 6.0, "第二句")],
        silences=[(0.0, 1.0), (3.0, 4.0), (6.0, 10.0)],
    ))
    speech = [clip for clip in clips if clip.kind == ClipKind.SPEECH]
    assert [clip.text for clip in speech] == ["第一句", "第二句"]
    assert all(clip.level == ClipLevel.UTTERANCE for clip in clips)

def test_headroom_reaches_to_the_far_edge_of_the_surrounding_silence() -> None:
    clips = build(analysis(
        10.0,
        segments=[(1.0, 3.0, "一句話")],
        silences=[(0.2, 1.0), (3.0, 4.5), (5.0, 10.0)],
    ))
    speech = next(clip for clip in clips if clip.kind == ClipKind.SPEECH)
    # The silence before it starts at 0.2, the one after it ends at 4.5.
    assert speech.safe_in == pytest.approx(0.8)
    assert speech.safe_out == pytest.approx(1.5)

def test_there_is_no_headroom_where_speech_runs_straight_into_speech() -> None:
    clips = build(analysis(
        6.0,
        segments=[(0.0, 3.0, "前半"), (3.0, 6.0, "後半")],
        silences=[],
    ))
    assert [(clip.safe_in, clip.safe_out) for clip in clips] == [(0.0, 0.0), (0.0, 0.0)]

def test_headroom_never_reaches_outside_the_asset() -> None:
    clips = build(analysis(
        5.0,
        segments=[(0.0, 2.0, "從頭開始")],
        silences=[(0.0, 0.0), (2.0, 5.0)],
    ))
    speech = next(clip for clip in clips if clip.kind == ClipKind.SPEECH)
    assert speech.safe_in == 0.0
    assert speech.source_range.end + speech.safe_out <= 5.0

def test_a_long_pause_becomes_a_clip_of_its_own() -> None:
    clips = build(analysis(
        10.0,
        segments=[(1.0, 3.0, "之前"), (6.0, 8.0, "之後")],
        silences=[(0.0, 1.0), (3.0, 6.0), (8.0, 10.0)],
    ))
    assert ("silence", 3.0, 6.0) in ranges(clips)

def test_a_pause_between_two_sentences_is_left_to_their_headroom() -> None:
    gap = MIN_UTTERANCE_SECONDS / 2
    clips = build(analysis(
        10.0,
        segments=[(1.0, 3.0, "之前"), (3.0 + gap, 8.0, "之後")],
        silences=[(0.0, 1.0), (3.0, 3.0 + gap), (8.0, 10.0)],
    ))
    # Too short to be a thing on its own, and both neighbours can already reach across it.
    assert all(clip.kind != ClipKind.SILENCE or clip.duration >= MIN_UTTERANCE_SECONDS for clip in clips)
    first = next(clip for clip in clips if clip.text == "之前")
    assert first.safe_out == pytest.approx(gap)

def test_a_stretch_without_speech_is_cut_at_the_shot_changes_in_it() -> None:
    clips = build(analysis(
        9.0,
        scenes=[(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)],
    ))
    assert [(clip.source_range.start, clip.source_range.end) for clip in clips] == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]

def test_a_shot_change_does_not_cut_off_a_fragment() -> None:
    # The cut at 8.9 would leave a tenth of a second on its own, so it is not taken.
    clips = build(analysis(9.0, scenes=[(0.0, 8.9), (8.9, 9.0)]))
    assert [(clip.source_range.start, clip.source_range.end) for clip in clips] == [(0.0, 9.0)]

def test_a_file_with_no_transcript_gives_one_clip_per_shot() -> None:
    clips = build(
        analysis(8.0, scenes=[(0.0, 4.0), (4.0, 8.0)]),
        asset(has_audio=False),
    )
    # Nothing is said and nothing is wrong with the picture, so these are just shots.
    assert ranges(clips) == [("ambient", 0.0, 4.0), ("ambient", 4.0, 8.0)]

def test_a_black_stretch_with_nobody_talking_is_marked_unusable() -> None:
    clips = build(analysis(
        8.0,
        segments=[(4.0, 8.0, "終於開口")],
        silences=[(0.0, 4.0)],
        black=[(0.0, 3.5)],
    ))
    assert ranges(clips)[0] == ("unusable", 0.0, 4.0)

def test_a_black_stretch_someone_is_talking_over_stays_speech() -> None:
    # The picture is unusable but the sound is not, and the two are cut separately.
    clips = build(analysis(8.0, segments=[(0.0, 8.0, "畫面黑掉但還在講")], black=[(0.0, 8.0)]))
    assert clips[0].kind == ClipKind.SPEECH
    assert clips[0].scores["black"] == 1.0

def test_scores_measure_what_each_detector_covered() -> None:
    clips = build(analysis(
        10.0,
        segments=[(0.0, 4.0, "說了四秒")],
        silences=[(4.0, 10.0)],
        frozen=[(0.0, 2.0)],
    ))
    speech = clips[0]
    assert speech.scores["speech"] == 1.0
    assert speech.scores["frozen"] == 0.5
    assert speech.scores["silence"] == 0.0
    assert speech.scores["words_per_second"] == pytest.approx(0.25)

def test_the_same_analysis_always_builds_the_same_timeline() -> None:
    made = analysis(10.0, segments=[(1.0, 3.0, "一"), (5.0, 8.0, "二")], silences=[(0.0, 1.0), (3.0, 5.0)])
    first_timeline, first_clips = build_timeline({ASSET_ID: asset()}, {ASSET_ID: made})
    second_timeline, second_clips = build_timeline({ASSET_ID: asset()}, {ASSET_ID: made})
    assert first_timeline.id == second_timeline.id
    assert [clip.id for clip in first_clips] == [clip.id for clip in second_clips]

def test_re_analyzing_the_same_footage_does_not_invalidate_the_timeline() -> None:
    first = analysis(10.0, segments=[(1.0, 3.0, "一樣的內容")])
    second = analysis(10.0, segments=[(1.0, 3.0, "一樣的內容")])
    second.analyzed_at = first.analyzed_at + timedelta(days=1)
    # Analyzing again hands back the same footage; only the moment differs, and
    # rebuilding on that alone would point every plan at a new set of IDs for nothing.
    assert timeline_input_hash({ASSET_ID: first}) == timeline_input_hash({ASSET_ID: second})

def test_a_changed_analysis_builds_a_different_timeline() -> None:
    before = timeline_input_hash({ASSET_ID: analysis(10.0, segments=[(1.0, 3.0, "原本")])})
    after = timeline_input_hash({ASSET_ID: analysis(10.0, segments=[(1.0, 3.0, "改過")])})
    assert before != after

def test_clips_from_several_assets_share_one_timeline() -> None:
    assets = {"a": asset("a"), "b": asset("b")}
    analyses = {
        "a": analysis(4.0, segments=[(0.0, 4.0, "甲檔")], asset_id="a"),
        "b": analysis(4.0, segments=[(0.0, 4.0, "乙檔")], asset_id="b"),
    }
    timeline, clips = build_timeline(assets, analyses)
    assert timeline.asset_ids == ["a", "b"]
    assert len({clip.id for clip in clips}) == len(clips)
    assert {clip.asset_id for clip in clips} == {"a", "b"}

@pytest.fixture
def stored(tmp_path: Path) -> Repository:
    """A repository holding one built timeline.

    Returns:
        The repository; the timeline covers `ASSET_ID` and has speech,
        silence, and an unusable stretch in it.
    """
    repo = Repository(str(tmp_path / "semantic.db"))
    made = analysis(
        20.0,
        segments=[(1.0, 4.0, "我們當初在設計這個系統的時候踩了很多坑"), (6.0, 7.0, "後來重寫")],
        silences=[(0.0, 1.0), (4.0, 6.0), (7.0, 20.0)],
        black=[(10.0, 20.0)],
        scenes=[(0.0, 10.0), (10.0, 20.0)],
    )
    timeline, clips = build_timeline({ASSET_ID: asset()}, {ASSET_ID: made})
    repo.save_semantic_timeline(timeline, clips)
    return repo

def timeline_id(repo: Repository) -> str:
    """Read the stored timeline's ID.

    Args:
        repo: Repository to read.

    Returns:
        The ID of the newest timeline.
    """
    return repo.get_semantic_timeline(None).id

def test_clips_survive_being_stored_and_read_back(stored: Repository) -> None:
    clips = stored.query_semantic_clips(timeline_id(stored))
    assert clips
    assert all(clip.level == ClipLevel.UTTERANCE for clip in clips)
    speech = [clip for clip in clips if clip.kind == ClipKind.SPEECH]
    assert speech[0].safe_in == pytest.approx(1.0) and speech[0].safe_out == pytest.approx(2.0)

def test_a_query_narrows_by_kind_and_by_text(stored: Repository) -> None:
    found = stored.query_semantic_clips(timeline_id(stored), kinds=[ClipKind.SPEECH], text="踩了很多坑")
    assert [clip.text for clip in found] == ["我們當初在設計這個系統的時候踩了很多坑"]

def test_a_two_character_chinese_search_finds_its_clip(stored: Repository) -> None:
    # The reason this store uses LIKE rather than full-text search: a trigram
    # index cannot match a word this short, and most Chinese words are.
    assert stored.query_semantic_clips(timeline_id(stored), text="設計")
    assert stored.query_semantic_clips(timeline_id(stored), text="重寫")

def test_a_search_for_a_wildcard_looks_for_that_character(stored: Repository) -> None:
    assert stored.query_semantic_clips(timeline_id(stored), text="%") == []
    assert stored.query_semantic_clips(timeline_id(stored), text="_") == []

def test_a_query_narrows_by_duration_and_by_score(stored: Repository) -> None:
    found = stored.query_semantic_clips(timeline_id(stored), min_duration=2.0, max_scores={"black": 0.0})
    assert all(clip.duration >= 2.0 and clip.scores["black"] == 0.0 for clip in found)
    assert any(clip.kind == ClipKind.SPEECH for clip in found)

def test_a_query_narrows_by_time(stored: Repository) -> None:
    # Overlap, not containment: a clip straddling the window is part of what is there.
    found = stored.query_semantic_clips(timeline_id(stored), start=6.0, end=8.0)
    assert ranges(found) == [("speech", 6.0, 7.0), ("silence", 7.0, 10.0)]
    assert ranges(stored.query_semantic_clips(timeline_id(stored), start=0.0, end=1.0)) == [("silence", 0.0, 1.0)]

def test_a_query_refuses_a_score_name_that_is_not_one(stored: Repository) -> None:
    with pytest.raises(ValueError, match="is not a score name"):
        stored.query_semantic_clips(timeline_id(stored), min_scores={"speech') = 1 OR ('1": 0.5})

def test_rebuilding_replaces_the_clips_rather_than_adding_to_them(stored: Repository) -> None:
    before = stored.query_semantic_clips(timeline_id(stored), limit=200)
    made = analysis(20.0, segments=[(1.0, 4.0, "我們當初在設計這個系統的時候踩了很多坑"), (6.0, 7.0, "後來重寫")],
                    silences=[(0.0, 1.0), (4.0, 6.0), (7.0, 20.0)], black=[(10.0, 20.0)],
                    scenes=[(0.0, 10.0), (10.0, 20.0)])
    timeline, clips = build_timeline({ASSET_ID: asset()}, {ASSET_ID: made})
    stored.save_semantic_timeline(timeline, clips)
    after = stored.query_semantic_clips(timeline.id, limit=200)
    assert [clip.id for clip in before] == [clip.id for clip in after]

def test_a_timeline_is_found_again_by_what_it_was_built_from(stored: Repository) -> None:
    timeline = stored.get_semantic_timeline(None)
    assert stored.find_semantic_timeline(timeline.input_hash).id == timeline.id
    assert stored.find_semantic_timeline("not a hash") is None

def test_counts_are_reported_by_level_and_kind(stored: Repository) -> None:
    counts = stored.count_semantic_clips(timeline_id(stored))
    assert counts["utterance/speech"] == 2
    assert sum(counts.values()) == len(stored.query_semantic_clips(timeline_id(stored), limit=200))

def test_a_workspace_from_before_the_semantic_timeline_upgrades_in_place(tmp_path: Path, monkeypatch) -> None:
    database = str(tmp_path / "old.db")
    shipped = repo_module._MIGRATIONS
    # A database as the version before semantic clips existed would have left it.
    monkeypatch.setattr(repo_module, "_MIGRATIONS", shipped[:-1])
    monkeypatch.setattr(repo_module, "SCHEMA_VERSION", len(shipped) - 1)
    older = Repository(database)
    older.save_asset(asset())

    monkeypatch.undo()
    upgraded = Repository(database)
    assert upgraded.get_assets([ASSET_ID])[ASSET_ID].path == asset().path
    assert upgraded.get_semantic_timeline(None) is None
    timeline, clips = build_timeline({ASSET_ID: asset()}, {ASSET_ID: analysis(4.0, segments=[(0.0, 4.0, "一句話")])})
    upgraded.save_semantic_timeline(timeline, clips)
    assert upgraded.query_semantic_clips(timeline.id)

@pytest.fixture
def analyzed(media: Path) -> str:
    """Import a real file and give it an analysis, without running speech recognition.

    Returns:
        The asset's ID.
    """
    asset_id = import_asset(str(media / "wide.mp4"))["id"]
    server_repo.save_analysis(analysis(
        10.0,
        segments=[(1.0, 4.0, "我們當初在設計這個系統的時候踩了很多坑"), (6.0, 7.0, "後來重寫")],
        silences=[(0.0, 1.0), (4.0, 6.0), (7.0, 10.0)],
        asset_id=asset_id,
    ))
    return asset_id

def test_building_over_an_unanalyzed_asset_says_which_one(media: Path, analyzed: str) -> None:
    other = import_asset(str(media / "tall.mp4"))["id"]
    with pytest.raises(ValueError, match="have not been analyzed"):
        build_semantic_timeline([analyzed, other])

def test_building_needs_something_to_build_from() -> None:
    with pytest.raises(ValueError, match="at least one asset"):
        build_semantic_timeline([])

def test_building_again_over_unchanged_analyses_returns_the_same_timeline(analyzed: str) -> None:
    first = build_semantic_timeline([analyzed])
    second = build_semantic_timeline([analyzed])
    assert first["reused"] is False and second["reused"] is True
    assert first["timeline_id"] == second["timeline_id"]
    assert second["counts"]["utterance/speech"] == 2

def test_a_search_comes_back_as_summaries_that_carry_the_headroom(analyzed: str) -> None:
    built = build_semantic_timeline([analyzed])
    found = query_clips(timeline_id=built["timeline_id"], kinds=[ClipKind.SPEECH], min_duration=2.0)
    assert [clip["clip_id"] for clip in found["clips"]]
    first = found["clips"][0]
    assert first["kind"] == "speech"
    assert (first["start"], first["end"]) == (1.0, 4.0)
    # A cut anywhere inside the headroom cannot clip a word; the model is not asked to work it out.
    assert first["safe_in"] == pytest.approx(1.0) and first["safe_out"] == pytest.approx(2.0)

def test_a_search_defaults_to_the_timeline_built_most_recently(analyzed: str) -> None:
    built = build_semantic_timeline([analyzed])
    assert query_clips(text="踩了很多坑")["timeline_id"] == built["timeline_id"]

def test_a_long_text_is_cut_in_the_summary_and_whole_in_the_clip(analyzed: str) -> None:
    built = build_semantic_timeline([analyzed])
    found = query_clips(timeline_id=built["timeline_id"], text="設計")
    summary = found["clips"][0]
    whole = get_semantic_clip(summary["clip_id"])
    assert whole["text"] == "我們當初在設計這個系統的時候踩了很多坑"
    assert len(summary["text"]) <= CLIP_SUMMARY_CHARACTERS + 1

def test_word_timings_come_only_when_they_are_asked_for(analyzed: str) -> None:
    built = build_semantic_timeline([analyzed])
    clip_id = query_clips(timeline_id=built["timeline_id"], text="重寫")["clips"][0]["clip_id"]
    assert "words" not in get_semantic_clip(clip_id)
    words = get_semantic_clip(clip_id, include_words=True)["words"]
    assert [word["text"] for word in words] == ["後來重寫"]

def test_reading_a_clip_that_does_not_exist_says_where_to_look() -> None:
    with pytest.raises(ValueError, match="query_clips"):
        get_semantic_clip("tl_00000000:u9999")

def first_clip(analyzed: str) -> str:
    """Build the timeline and return its first clip's ID.

    Args:
        analyzed: The analyzed asset.

    Returns:
        The clip ID.
    """
    built = build_semantic_timeline([analyzed])
    return query_clips(timeline_id=built["timeline_id"])["clips"][0]["clip_id"]

def test_a_description_is_not_stored_until_the_user_has_seen_it(analyzed: str) -> None:
    clip_id = first_clip(analyzed)
    with pytest.raises(ValueError, match="agreement first"):
        set_clip_tags([ClipDescription(clip_id=clip_id, description="海邊的空景")], written_by="test-model/1")
    assert get_semantic_clip(clip_id)["description"] is None

def test_a_description_is_stored_with_who_wrote_it(analyzed: str) -> None:
    clip_id = first_clip(analyzed)
    set_clip_tags(
        [ClipDescription(clip_id=clip_id, description="海邊的空景", tags=["海邊", "空景"])],
        written_by="test-model/1", reviewed=True,
    )
    stored = get_semantic_clip(clip_id)
    assert stored["description"] == "海邊的空景"
    assert stored["described_by"] == "test-model/1"
    # What a model read off a picture is not worth what a detector measured, so it says which it is.
    assert {tag["source"] for tag in stored["tags"]} == {"model"}
    assert {tag["value"] for tag in stored["tags"]} == {"海邊", "空景"}

def test_a_correction_replaces_its_own_authors_tags_and_leaves_the_rest(analyzed: str) -> None:
    clip_id = first_clip(analyzed)
    set_clip_tags([ClipDescription(clip_id=clip_id, tags=["海邊"])], written_by="test-model/1", reviewed=True)
    set_clip_tags([ClipDescription(clip_id=clip_id, tags=["碼頭"])], written_by="user", reviewed=True)
    tags = {tag["value"]: tag["source"] for tag in get_semantic_clip(clip_id)["tags"]}
    assert tags == {"海邊": "model", "碼頭": "user"}

    set_clip_tags([ClipDescription(clip_id=clip_id, tags=["漁港"])], written_by="user", reviewed=True)
    tags = {tag["value"]: tag["source"] for tag in get_semantic_clip(clip_id)["tags"]}
    assert tags == {"海邊": "model", "漁港": "user"}

def test_a_search_finds_a_clip_by_its_tag_and_by_what_is_on_screen(analyzed: str) -> None:
    built = build_semantic_timeline([analyzed])
    clip_id = query_clips(timeline_id=built["timeline_id"])["clips"][0]["clip_id"]
    set_clip_tags(
        [ClipDescription(clip_id=clip_id, description="黃昏的漁港，遠處有船", tags=["漁港"])],
        written_by="test-model/1", reviewed=True,
    )
    assert [clip["clip_id"] for clip in query_clips(timeline_id=built["timeline_id"], tag="漁港")["clips"]] == [clip_id]
    # The description is searched alongside the words: what is shown is as much a reason to pick a clip.
    assert [clip["clip_id"] for clip in query_clips(timeline_id=built["timeline_id"], text="黃昏")["clips"]] == [clip_id]

def test_writing_to_a_clip_that_does_not_exist_is_refused(analyzed: str) -> None:
    with pytest.raises(ValueError, match="not found"):
        set_clip_tags([ClipDescription(clip_id="tl_00000000:u9999", tags=["x"])], written_by="user", reviewed=True)

def test_frames_come_back_as_one_sheet_naming_every_clip(analyzed: str) -> None:
    built = build_semantic_timeline([analyzed])
    clip_ids = [clip["clip_id"] for clip in query_clips(timeline_id=built["timeline_id"])["clips"][:3]]
    result = frames_for_clips(clip_ids)
    assert all(clip_id in result.content[0].text for clip_id in clip_ids)
    assert len(result.content) == 2

def test_asking_for_more_frames_than_a_sheet_holds_is_refused(analyzed: str) -> None:
    built = build_semantic_timeline([analyzed])
    clip_id = query_clips(timeline_id=built["timeline_id"])["clips"][0]["clip_id"]
    with pytest.raises(ValueError, match="more than one sheet holds"):
        frames_for_clips([clip_id] * (MAX_STORYBOARD_TILES + 1))
    with pytest.raises(ValueError, match="at least one clip"):
        frames_for_clips([])
