"""Tests for listening to one stretch again, with nothing the first transcript said."""

import wave
from pathlib import Path

import pytest

from app import server
from app.engine import analysis
from app.models.media import MediaAnalysis, Transcript, TranscriptSegment, TranscriptWord

@pytest.fixture
def heard(media: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A file with a transcript of its own, and a speech model that hears what the test says."""
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    server.repo.save_analysis(MediaAnalysis(asset_id=asset, duration=10.0, transcript=Transcript(
        language="zh", model="test", segments=[TranscriptSegment(start=2.0, end=4.0, text="怎麼那麼順", words=[
            TranscriptWord(start=2.0, end=4.0, text="怎麼那麼順", probability=0.4)])],
    )))
    asked = {}

    def model(device, path, duration, language, prompt, on_progress, is_cancelled, on_measured=None,
              model_name=None):
        asked.update(path=path, prompt=prompt, duration=duration)
        return Transcript(language="zh", model="large-v3-turbo", segments=[TranscriptSegment(
            start=0.5, end=1.5, text="沒那麼順",
            words=[TranscriptWord(start=0.5, end=1.5, text="沒那麼順", probability=0.93)])])

    monkeypatch.setattr(analysis, "_run_whisper", model)
    return {"asset": asset, "asked": asked}

def test_a_stretch_is_heard_again_without_the_transcript_and_kept_to_listen_to(heard: dict) -> None:
    found = server.listen_again(asset_id=heard["asset"], start=1.5, end=4.5)
    # Nothing of the first pass is handed to the model: no prompt, only this stretch of sound.
    assert heard["asked"]["prompt"] is None and heard["asked"]["duration"] == pytest.approx(3.0)
    assert found["heard"] == "沒那麼順" and found["stored"] == "怎麼那麼順"
    assert found["agrees"] is False
    # Times are the file's, not the clip of sound's.
    assert found["words"][0]["start"] == pytest.approx(2.0) and found["words"][0]["probability"] == 0.93
    with wave.open(found["audio_path"]) as sound:
        assert sound.getnframes() / sound.getframerate() == pytest.approx(3.0, abs=0.05)

def test_only_a_short_stretch_is_heard_again(heard: dict) -> None:
    with pytest.raises(ValueError, match="30"):
        server.listen_again(asset_id=heard["asset"], start=0, end=45)
    with pytest.raises(ValueError, match="after"):
        server.listen_again(asset_id=heard["asset"], start=4, end=3)

def hears(said: str, probability: float = 0.93):
    """A speech model that hears `said` from 0.5s to 1.5s into whatever it is given."""
    def model(device, path, duration, language, prompt, on_progress, is_cancelled, on_measured=None,
              model_name=None):
        return Transcript(language="zh", model="large-v3-turbo", segments=[TranscriptSegment(
            start=0.5, end=1.5, text=said, words=[TranscriptWord(start=0.5, end=1.5, text=said,
                                                                 probability=probability)])])
    return model

def test_a_hearing_that_disagrees_with_the_transcript_is_not_kept(heard: dict) -> None:
    found = server.listen_again(asset_id=heard["asset"], start=1.5, end=4.5, keep=True)
    # 「沒那麼順」 against 「怎麼那麼順」: neither is believed, so the user is asked.
    assert "different words" in found["not_kept"] and "kept" not in found
    assert server.repo.get_analysis(heard["asset"]).heard_again == []

def test_an_unsure_hearing_is_not_kept(heard: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis, "_run_whisper", hears("怎麼那麼順", probability=0.2))
    found = server.listen_again(asset_id=heard["asset"], start=1.5, end=4.5, keep=True)
    assert "unsure" in found["not_kept"]

def test_a_stretch_heard_again_and_kept_is_what_cuts_are_judged_by(heard: dict,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine.semantic import timeline_input_hash

    monkeypatch.setattr(analysis, "_run_whisper", hears("怎麼那麼順"))
    before = server.repo.get_analysis(heard["asset"])
    found = server.listen_again(asset_id=heard["asset"], start=1.5, end=4.5, keep=True)
    assert "check the cut again" in found["kept"]
    after = server.repo.get_analysis(heard["asset"])
    assert [(item.start, item.end) for item in after.heard_again] == [(1.5, 4.5)]
    # The plans written against the footage still stand.
    assert timeline_input_hash({heard["asset"]: after}) == timeline_input_hash({heard["asset"]: before})
    # The transcript's 2.0–4.0s gives way to the 2.0–3.0s heard again, so a cut at 3.5s is clear of it.
    words = [(word.start, word.end, word.text) for segment in analysis.with_words_heard_again(after).transcript.segments
             for word in segment.words]
    assert words == [(pytest.approx(2.0), pytest.approx(3.0), "怎麼那麼順")]
    with pytest.raises(ValueError, match="only what a file says is kept"):
        server.listen_again(job_id="any", start=0, end=1, keep=True)

def test_a_stretch_where_nobody_was_heard_again_keeps_no_words() -> None:
    made = MediaAnalysis(asset_id="x", duration=10.0, transcript=Transcript(language="zh", model="t", segments=[
        TranscriptSegment(start=1.0, end=6.0, text="前面中間後面", words=[
            TranscriptWord(start=1.0, end=2.0, text="前面"), TranscriptWord(start=3.0, end=4.0, text="中間"),
            TranscriptWord(start=5.0, end=6.0, text="後面")]),
    ]), heard_again=[{"start": 2.5, "end": 4.5, "sentences": []}])
    (segment,) = analysis.with_words_heard_again(made).transcript.segments
    assert [word.text for word in segment.words] == ["前面", "後面"]

def test_a_word_across_the_edge_of_a_kept_stretch_keeps_the_part_outside_it() -> None:
    made = MediaAnalysis(asset_id="x", duration=10.0, transcript=Transcript(language="zh", model="t", segments=[
        TranscriptSegment(start=1.0, end=4.0, text="前面中間", words=[
            TranscriptWord(start=1.0, end=3.0, text="前面"), TranscriptWord(start=3.2, end=4.0, text="中間")]),
    ]), heard_again=[{"start": 2.5, "end": 4.5, "sentences": [
        {"start": 3.0, "end": 3.6, "text": "中間", "words": [{"start": 3.0, "end": 3.6, "text": "中間"}]}]}])
    (segment,) = analysis.with_words_heard_again(made).transcript.segments
    # The hearing began at 2.5s and never heard the 1.0–2.5s of 「前面」, so that part still stops a cut.
    assert [(word.start, word.end, word.text) for word in segment.words] == [(1.0, 2.5, "前面"), (3.0, 3.6, "中間")]

def test_a_sentence_that_ends_inside_a_kept_stretch_still_ends_there() -> None:
    made = MediaAnalysis(asset_id="x", duration=10.0, transcript=Transcript(language="zh", model="t", segments=[
        TranscriptSegment(start=1.0, end=3.0, text="第一句", words=[TranscriptWord(start=1.0, end=3.0, text="第一句")]),
        TranscriptSegment(start=4.0, end=6.0, text="第二句", words=[TranscriptWord(start=4.0, end=6.0, text="第二句")]),
    ]), second_listen={"model": "t", "words": [{"start": 4.1, "end": 4.9, "text": "第二", "probability": 0.9}]},
        heard_again=[{"start": 0.5, "end": 6.5, "sentences": [
            {"start": 1.0, "end": 2.6, "text": "第一句", "words": [{"start": 1.0, "end": 2.6, "text": "第一句"}]},
            {"start": 4.3, "end": 5.8, "text": "第二句", "words": [{"start": 4.3, "end": 5.8, "text": "第二句"}]}]}])
    after = analysis.with_words_heard_again(made)
    # Two sentences, not one: the pause between them is still a place a cut may go.
    assert [(segment.start, segment.end) for segment in after.transcript.segments] == [(1.0, 2.6), (4.3, 5.8)]
    # And the second listen's own word there gives way, or it would argue with what was kept.
    assert after.second_listen.words == []
