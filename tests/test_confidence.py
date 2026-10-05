"""Tests for keeping how sure the speech model was of each word, and pointing the AI at the doubtful ones."""

from pathlib import Path

import pytest

from app import server
from app.engine import analysis
from app.engine.diarize import speaker_model_name
from app.engine.rhythm import rhythm_model_name
from app.models.media import Asset, MediaAnalysis, Transcript, TranscriptSegment, TranscriptWord

def word(text: str, start: float, probability) -> TranscriptWord:
    return TranscriptWord(start=start, end=start + 0.2, text=text, probability=probability)

def segment(*words: TranscriptWord) -> TranscriptSegment:
    return TranscriptSegment(start=words[0].start, end=words[-1].end,
                             text="".join(w.text for w in words).strip(), words=list(words))

SURE, UNSURE = 0.95, 0.2

def test_doubtful_words_in_a_row_are_marked_once() -> None:
    line = segment(word("今天", 0, SURE), word("要", 0.2, SURE), word("賽博", 0.4, UNSURE), word("朋克", 0.6, UNSURE),
                   word("的", 0.8, SURE), word("剪法", 1.0, UNSURE))
    assert analysis.marked_text(line) == "今天要⟦賽博朋克⟧的⟦剪法⟧"
    assert analysis.doubtful_spots([line]) == 2

def test_english_keeps_its_spaces_outside_the_marks() -> None:
    line = segment(word(" Hello", 0, SURE), word(" Kubernetes", 0.2, UNSURE), word(" folks", 0.4, SURE))
    assert analysis.marked_text(line) == "Hello ⟦Kubernetes⟧ folks"

def test_a_transcript_made_before_probabilities_were_kept_is_left_alone() -> None:
    line = segment(word("今天", 0, None), word("好", 0.2, None))
    assert analysis.marked_text(line) == "今天好"
    assert analysis.doubtful_spots([line]) is None

def test_the_threshold_is_where_doubt_begins() -> None:
    line = segment(word("剛好", 0, analysis.LOW_CONFIDENCE), word("不夠", 0.2, analysis.LOW_CONFIDENCE - 0.01))
    assert analysis.marked_text(line) == "剛好⟦不夠⟧"

@pytest.fixture
def analysed(tmp_path: Path) -> str:
    asset = Asset(id="confidence-talk", path=str(tmp_path / "talk.mp4"), duration=10, has_video=True, has_audio=True)
    server.repo.save_asset(asset)
    model = analysis.FAST_WHISPER_MODEL
    server.repo.save_analysis(MediaAnalysis(
        asset_id=asset.id, duration=10.0,
        transcript=Transcript(language="zh", model=model, segments=[
            segment(word("大家好", 0, SURE)),
            segment(word("我是", 5, SURE), word("阿尼克斯", 5.2, UNSURE)),
        ]),
        recipe=analysis.current_recipe(model, speaker_model=speaker_model_name(), rhythm_model=rhythm_model_name()),
    ))
    return asset.id

def test_the_ai_reads_the_marks_and_how_many_there_are(analysed: str) -> None:
    found = server.get_analysis(analysed)
    assert [line["text"] for line in found["transcript"]["segments"]] == ["大家好", "我是⟦阿尼克斯⟧"]
    assert found["doubtful_spots"] == 1

def test_the_count_covers_the_whole_file_whatever_range_is_read(analysed: str) -> None:
    found = server.get_analysis(analysed, start=0, end=2)
    assert [line["text"] for line in found["transcript"]["segments"]] == ["大家好"]
    assert found["doubtful_spots"] == 1

def test_words_carry_their_probability_when_asked_for(analysed: str) -> None:
    words = server.get_analysis(analysed, include_words=True)["transcript"]["segments"][1]["words"]
    assert words[1]["probability"] == UNSURE

def test_the_stored_transcript_never_holds_the_marks(analysed: str) -> None:
    server.get_analysis(analysed)
    assert "⟦" not in server.repo.get_analysis(analysed).transcript.segments[1].text
