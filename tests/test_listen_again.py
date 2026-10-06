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
