"""Tests for finding clips by what they mean.

The real model is a 130 MB download, so these put a stand-in in its place that
treats texts sharing characters as close. What is checked is everything around
the model: which clips are read, what is kept, what is made again, and in what
order the answer comes back.
"""

import numpy as np
import pytest

from app import server
from app.engine import meaning
from app.models.media import Span
from app.models.semantic import ClipKind, ClipLevel, SemanticClip, SemanticTimeline

def _stand_in(texts, query=False):
    """Vectors from which characters a text holds: texts sharing characters come out close."""
    rows = np.zeros((len(texts), 64), dtype=np.float32)
    for row, text in enumerate(texts):
        for character in text.replace("passage: ", "").replace("query: ", ""):
            rows[row, ord(character) % 64] += 1
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)

@pytest.fixture
def embedded(monkeypatch: pytest.MonkeyPatch) -> list:
    """Every batch of texts the search embeds, with the stand-in model in place."""
    calls = []

    def embed(texts, query=False):
        if not query:
            calls.append(list(texts))
        return _stand_in(texts, query)

    monkeypatch.setattr(meaning, "embed", embed)
    return calls

def _timeline(timeline_id: str, said: dict) -> str:
    clips = [
        SemanticClip(id=f"{timeline_id}:u{index:04d}", timeline_id=timeline_id, asset_id="a", level=ClipLevel.UTTERANCE,
                     source_range=Span(start=index * 5.0, end=index * 5.0 + 4), kind=ClipKind.SPEECH, text=text)
        for index, text in enumerate(said.values())
    ]
    server.repo.save_semantic_timeline(
        SemanticTimeline(id=timeline_id, asset_ids=["a"], input_hash=f"hash-{timeline_id}", derivation_version=1), clips)
    return timeline_id

SAID = {
    "battery": "電池續航力一次充電三天",
    "hello": "大家好歡迎來到我的頻道",
    "food": "牛肉湯真的很好喝",
    "silent": "",
}

def test_clips_come_back_closest_first_with_how_close(embedded: list) -> None:
    timeline = _timeline("meaning-order", SAID)
    found = server.query_clips(timeline_id=timeline, about="電池可以充電多久")
    assert [clip["text"] for clip in found["clips"]][0] == SAID["battery"]
    scores = [clip["relevance"] for clip in found["clips"]]
    assert scores == sorted(scores, reverse=True)
    # A clip with nothing to read cannot be found by meaning.
    assert len(found["clips"]) == 3

def test_each_clip_is_embedded_once_and_again_only_when_its_words_change(embedded: list) -> None:
    timeline = _timeline("meaning-cache", SAID)
    server.query_clips(timeline_id=timeline, about="好喝")
    server.query_clips(timeline_id=timeline, about="歡迎")
    assert [len(batch) for batch in embedded] == [3]
    changed = server.repo.get_semantic_clip(f"{timeline}:u0002").model_copy(update={"text": "夜市雞排"})
    server.repo.update_semantic_clips([changed])
    server.query_clips(timeline_id=timeline, about="雞排")
    # The changed clip, and the one beside it, which is read with it.
    assert sorted(text.split("\n", 1)[0] for text in embedded[-1]) == sorted([changed.text, SAID["hello"]])

def test_a_fragment_too_short_to_mean_anything_is_left_to_a_search_by_words(embedded: list) -> None:
    timeline = _timeline("meaning-short", {**SAID, "fragment": "去"})
    found = server.query_clips(timeline_id=timeline, about="去哪裡")
    assert "去" not in [clip["text"] for clip in found["clips"]]
    assert server.query_clips(timeline_id=timeline, text="去")["clips"]

def test_each_clip_is_read_with_the_sentences_beside_it(embedded: list) -> None:
    timeline = _timeline("meaning-context", SAID)
    server.query_clips(timeline_id=timeline, about="好喝")
    read = {text.split("\n", 1)[0]: text for text in embedded[0]}
    assert read[SAID["hello"]] == "\n".join([SAID["hello"], SAID["battery"], SAID["food"]])

def test_a_rebuild_drops_the_vectors_of_clips_it_no_longer_has(embedded: list) -> None:
    timeline = _timeline("meaning-rebuild", SAID)
    server.query_clips(timeline_id=timeline, about="好喝")
    _timeline(timeline, {"food": SAID["food"]})
    assert set(server.repo.clip_vectors([f"{timeline}:u{index:04d}" for index in range(4)])) == {f"{timeline}:u0000"}

def test_meaning_combines_with_the_other_conditions_and_pages(embedded: list) -> None:
    timeline = _timeline("meaning-pages", SAID)
    first = server.query_clips(timeline_id=timeline, about="頻道", limit=1)
    second = server.query_clips(timeline_id=timeline, about="頻道", limit=1, offset=1)
    assert first["truncated"] and first["clips"][0]["clip_id"] != second["clips"][0]["clip_id"]
    late = server.query_clips(timeline_id=timeline, about="頻道", start=10)
    assert {clip["clip_id"] for clip in late["clips"]} == {f"{timeline}:u0002"}
    lines = server.query_clips(timeline_id=timeline, about="電池", brief=True)["lines"]
    assert lines[0].split(" ", 1)[0].replace(".", "").isdigit() and SAID["battery"] in lines[0]

def test_the_model_is_pinned_and_kept_with_the_workspace() -> None:
    for model in (meaning.SEARCH_MODEL, meaning.SEARCH_TOKENIZER):
        assert "/resolve/" in model.url and "/main/" not in model.url and len(model.sha256) == 64
        assert model.path.startswith("search/")

def test_a_model_that_cannot_be_fetched_says_so_and_points_at_the_word_search(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(texts, query=False):
        raise RuntimeError("could not download the meaning search model")

    monkeypatch.setattr(meaning, "embed", offline)
    timeline = _timeline("meaning-offline", SAID)
    with pytest.raises(ValueError, match="Search with `text`"):
        server.query_clips(timeline_id=timeline, about="電池")
