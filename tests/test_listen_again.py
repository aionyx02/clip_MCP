"""Tests for listening to one stretch again, with nothing the first transcript said."""

import wave
from pathlib import Path

import pytest

from app import runtime, server
from app.engine import analysis
from app.models.media import MediaAnalysis, Transcript, TranscriptSegment, TranscriptWord

@pytest.fixture
def heard(media: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A file with a transcript of its own, and a speech model that hears what the test says."""
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    runtime.repo.save_analysis(MediaAnalysis(asset_id=asset, duration=10.0, transcript=Transcript(
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
    assert runtime.repo.get_analysis(heard["asset"]).heard_again == []

def test_an_unsure_hearing_is_not_kept(heard: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis, "_run_whisper", hears("怎麼那麼順", probability=0.2))
    found = server.listen_again(asset_id=heard["asset"], start=1.5, end=4.5, keep=True)
    assert "unsure" in found["not_kept"]

def test_a_stretch_heard_again_and_kept_is_what_cuts_are_judged_by(heard: dict,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine.semantic import timeline_input_hash

    monkeypatch.setattr(analysis, "_run_whisper", hears("怎麼那麼順"))
    before = runtime.repo.get_analysis(heard["asset"])
    found = server.listen_again(asset_id=heard["asset"], start=1.5, end=4.5, keep=True)
    assert "check the cut again" in found["kept"]
    after = runtime.repo.get_analysis(heard["asset"])
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
    ]), heard_again=[{"start": 2.5, "end": 4.5, "words": []}])
    (segment,) = analysis.with_words_heard_again(made).transcript.segments
    assert [word.text for word in segment.words] == ["前面", "後面"]

def test_a_stretch_with_a_word_across_its_edge_is_not_kept(heard: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis, "_run_whisper", hears("怎麼那麼順"))
    found = server.listen_again(asset_id=heard["asset"], start=3.0, end=4.5, keep=True)
    # 「怎麼那麼順」 runs 2.0–4.0s: kept, it would be there twice, cut back to 3.0s and as heard.
    assert found["not_kept"].endswith("hear 2.000s to 4.500s again with keep")
    assert runtime.repo.get_analysis(heard["asset"]).heard_again == []

def test_a_sentence_that_ends_inside_a_kept_stretch_still_ends_there() -> None:
    made = MediaAnalysis(asset_id="x", duration=10.0, transcript=Transcript(language="zh", model="t", segments=[
        TranscriptSegment(start=1.0, end=3.0, text="第一句", words=[TranscriptWord(start=1.0, end=3.0, text="第一句")]),
        TranscriptSegment(start=4.0, end=6.0, text="第二句", words=[TranscriptWord(start=4.0, end=6.0, text="第二句")]),
    ]), second_listen={"model": "t", "words": [{"start": 4.1, "end": 4.9, "text": "第二", "probability": 0.9}]},
        heard_again=[{"start": 0.5, "end": 6.5, "words": [
            {"start": 1.0, "end": 2.6, "text": "第一句"}, {"start": 4.3, "end": 5.8, "text": "第二句"}]}])
    after = analysis.with_words_heard_again(made)
    # Two sentences, not one: the pause between them is still a place a cut may go.
    assert [(segment.start, segment.end) for segment in after.transcript.segments] == [(1.0, 2.6), (4.3, 5.8)]
    # And the second listen's own word there gives way to the hearing, or the two would argue.
    assert [word.text for word in after.second_listen.words] == ["第一句", "第二句"]

def test_a_render_is_heard_a_shot_at_a_time_across_its_cuts(heard: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from helpers import build_project, finished, insert, video_track

    # Two shots meeting at 2s of the render; each is heard on its own, and both are said.
    project = build_project([video_track(), insert("a", heard["asset"], 0, 2), insert("b", heard["asset"], 4, 6)],
                            width=640, height=360)
    job = finished(server.render_project(project))
    lengths = []

    def model(device, path, duration, language, prompt, on_progress, is_cancelled, on_measured=None,
              model_name=None):
        lengths.append(round(duration, 2))
        return Transcript(language="zh", model="large-v3-turbo", segments=[TranscriptSegment(
            start=0.2, end=0.6, text="對", words=[TranscriptWord(start=0.2, end=0.6, text="對", probability=0.9)])])

    monkeypatch.setattr(analysis, "_run_whisper", model)
    found = server.listen_again(job_id=job["job_id"], start=1.0, end=3.5)
    assert lengths == [1.0, 1.5]
    assert [word["start"] for word in found["words"]] == pytest.approx([1.2, 2.2])
    with wave.open(found["audio_path"]) as sound:
        assert sound.getnframes() / sound.getframerate() == pytest.approx(2.5, abs=0.05)

def test_two_hearings_of_a_made_up_line_do_not_agree(heard: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime.repo.save_analysis(runtime.repo.get_analysis(heard["asset"]).model_copy(update={"transcript": Transcript(
        language="zh", model="test", segments=[TranscriptSegment(start=2.0, end=3.0, text="謝謝觀看", words=[
            TranscriptWord(start=2.0, end=3.0, text="謝謝觀看", probability=0.9)])])}))
    monkeypatch.setattr(analysis, "_run_whisper", hears("謝謝觀看"))
    found = server.listen_again(asset_id=heard["asset"], start=1.5, end=3.5)
    assert found["made_up"] is True and found["agrees"] is False

def test_a_render_loud_where_nothing_was_heard_says_so() -> None:
    from app.server import UNHEARD_LOUD_SECONDS, _loud_without_words

    said = Transcript(language="zh", model="t", segments=[TranscriptSegment(start=10.0, end=11.0, text="對", words=[
        TranscriptWord(start=10.0, end=11.0, text="對", probability=0.9)])])
    # Loud from 10s to 14s, a word only in the first second of it; quiet after.
    levels = [-20.0] * 8 + [-60.0] * 4
    found = _loud_without_words(levels, 10.0, said)
    assert found == [(11.0, 14.0)] and found[0][1] - found[0][0] >= UNHEARD_LOUD_SECONDS
    assert _loud_without_words([-60.0] * 8, 10.0, said) == []

def test_where_two_hearings_differ_the_times_of_the_words_they_agree_on_are_kept(
    media: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#87: 「四件」 against 「十件」 refused the whole stretch, and with it where 「褲子」 really ends."""
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    runtime.repo.save_analysis(MediaAnalysis(asset_id=asset, duration=10.0, transcript=Transcript(
        language="zh", model="test", segments=[TranscriptSegment(start=2.0, end=4.4, text="四件褲子然後", words=[
            TranscriptWord(start=2.0, end=2.8, text="四件", probability=0.9),
            TranscriptWord(start=2.8, end=3.6, text="褲子", probability=0.9),
            TranscriptWord(start=3.6, end=4.0, text="然後", probability=0.9)])],
    )))

    def model(device, path, duration, language, prompt, on_progress, is_cancelled, on_measured=None,
              model_name=None):
        # Heard from 1.5s of the file: 「褲子」 runs on to 3.78, and 「然後」 starts after it.
        return Transcript(language="zh", model="large-v3-turbo", segments=[TranscriptSegment(
            start=0.5, end=2.6, text="十件褲子然後", words=[
                TranscriptWord(start=0.5, end=1.3, text="十件", probability=0.9),
                TranscriptWord(start=1.3, end=2.28, text="褲子", probability=0.9),
                TranscriptWord(start=2.3, end=2.6, text="然後", probability=0.9)])])

    monkeypatch.setattr(analysis, "_run_whisper", model)
    found = server.listen_again(asset_id=asset, start=1.5, end=4.5, keep=True)
    assert "2 word(s) the two heard alike" in found["kept"] and "四件" in found["kept"]
    [kept] = runtime.repo.get_analysis(asset).heard_again
    assert [(word.text, word.start, word.end) for word in kept.words] == [
        ("四件", 2.0, 2.8), ("褲子", 2.8, 3.78), ("然後", 3.8, 4.1)]

def test_one_word_heard_unsure_keeps_its_own_time_and_not_the_whole_stretch_s(
    media: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#98: 「錯了⟦是⟧晚餐」, heard alike, was refused whole for the one word in doubt."""
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    runtime.repo.save_analysis(MediaAnalysis(asset_id=asset, duration=10.0, transcript=Transcript(
        language="zh", model="test", segments=[TranscriptSegment(start=2.0, end=4.0, text="錯了是晚餐", words=[
            TranscriptWord(start=2.0, end=2.8, text="錯了", probability=0.9),
            TranscriptWord(start=2.8, end=3.1, text="是", probability=0.9),
            TranscriptWord(start=3.1, end=4.0, text="晚餐", probability=0.9)])],
    )))

    def model(device, path, duration, language, prompt, on_progress, is_cancelled, on_measured=None,
              model_name=None):
        return Transcript(language="zh", model="large-v3-turbo", segments=[TranscriptSegment(
            start=0.5, end=2.4, text="錯了是晚餐", words=[
                TranscriptWord(start=0.5, end=1.2, text="錯了", probability=0.9),
                TranscriptWord(start=1.2, end=1.5, text="是", probability=0.2),
                TranscriptWord(start=1.6, end=2.4, text="晚餐", probability=0.9)])])

    monkeypatch.setattr(analysis, "_run_whisper", model)
    found = server.listen_again(asset_id=asset, start=1.5, end=4.5, keep=True)
    assert found["agrees"] and found["kept_words"] == ["錯了", "晚餐"] and found["left_as_transcribed"] == ["是"]
    [kept] = runtime.repo.get_analysis(asset).heard_again
    assert [(word.text, word.start, word.end) for word in kept.words] == [
        ("錯了", 2.0, 2.7), ("是", 2.8, 3.1), ("晚餐", 3.1, 3.9)]
