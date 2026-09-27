"""What a user is told to take care of before publishing, and speaking a narration in their voice."""

from decimal import Decimal
from pathlib import Path

import pytest

from app.engine import speak
from app.engine.delivery import rights_facts
from app.models.media import (
    Asset, FaceMeasurement, MediaAnalysis, SyntheticVoice, Transcript, TranscriptSegment,
)
from app.models.timeline import Clip, Project, TimeRange, Track, TrackType
from app.server import export_notices, import_asset, narrate, repo
from helpers import build_project, insert, video_track

def clip(clip_id: str, asset_id: str, start: float, end: float, at: float) -> Clip:
    return Clip(id=clip_id, asset_id=asset_id, source_range=TimeRange(start=Decimal(str(start)), end=Decimal(str(end))),
                timeline_in=Decimal(str(at)))

def test_everything_that_needs_a_word_before_publishing_is_listed() -> None:
    project = Project(id="p", width=1080, height=1920, tracks=[
        Track(id="main", track_type=TrackType.VIDEO, clips=[clip("a", "v", 0, 10, 0)]),
        Track(id="music", track_type=TrackType.AUDIO, clips=[clip("m", "song", 0, 10, 0)]),
        Track(id="voice", track_type=TrackType.AUDIO, clips=[clip("n", "said", 0, 3, 4)]),
    ])
    assets = {
        "v": Asset(id="v", path="/footage/a.mp4", has_video=True, has_audio=True),
        "song": Asset(id="song", path="/music/Carefree.mp3", has_video=False, has_audio=True,
                      license="Kevin MacLeod, CC BY 4.0"),
        "said": Asset(id="said", path="/jobs/narration.wav", has_video=False, has_audio=True, synthetic=SyntheticVoice(
            model="F5TTS_v1_Base", license=speak.DEFAULT_LICENSE, commercial_use=False, reference_asset_id="v",
            reference_start=1.0, reference_end=6.0, text="原本的杯子區")),
    }
    analyses = {"v": MediaAnalysis(
        asset_id="v", duration=10.0,
        transcript=Transcript(language="zh", model="t", segments=[TranscriptSegment(start=1, end=6, text="各位")]),
        faces=[FaceMeasurement(start=2.0, end=3.0, faces=2, face_share=0.1, face_x=0.5, face_y=0.5)],
    )}
    facts = {fact.kind: fact for fact in rights_facts(project, analyses, assets, {"song": {"artist": "Kevin MacLeod"}})}
    assert "artist: Kevin MacLeod" in facts["music"].detail and "licence: Kevin MacLeod, CC BY 4.0" in facts["music"].detail
    assert facts["synthetic_voice"].spans == ((4.0, 7.0),) and "commercial use: not allowed" in facts["synthetic_voice"].detail
    # Nobody told the voices apart, which is said rather than taken to mean nobody else talks.
    assert facts["unchecked_voices"].spans == ((0.0, 10.0),)
    assert facts["faces"].spans == ((2.0, 3.0),)

def test_notices_are_written_beside_the_video_with_every_fact_under_them(media: Path, tmp_path: Path) -> None:
    song = import_asset(str(media / "song.mp3"))["id"]
    video = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([
        video_track(), insert("v", video, 0, 5),
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        insert("song", song, 0, 5, track_id="music"),
    ])
    asked = export_notices(project, licenses={song: "Kevin MacLeod, CC BY 4.0"})
    assert not asked["written"] and any("CC BY 4.0" in fact for fact in asked["facts"])
    written = export_notices(project, "配樂要標註作者。", next_to=str(tmp_path / "cut.mp4"))
    saved = Path(written["output_path"])
    assert saved == tmp_path / "cut_notices.txt"
    text = saved.read_text(encoding="utf-8-sig")
    assert text.startswith("配樂要標註作者。") and "CC BY 4.0" in text
    # What the user said about the file is kept with it, even when it is imported again.
    import_asset(str(media / "song.mp3"))
    assert repo.get_assets([song])[song].license == "Kevin MacLeod, CC BY 4.0"

def test_a_narration_is_not_spoken_until_the_user_chooses_it(media: Path) -> None:
    project = build_project([video_track(), insert("v", import_asset(str(media / "wide.mp4"))["id"], 0, 5)])
    with pytest.raises(ValueError, match="ask the user first.*CC BY-NC"):
        narrate(project, "原本的杯子區", user_chose_voice=False)

def test_a_voice_sample_has_to_be_long_enough_to_copy_and_short_enough_to_use(media: Path) -> None:
    video = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("v", video, 0, 5)])
    with pytest.raises(ValueError, match="3 to 10 seconds"):
        narrate(project, "原本的杯子區", user_chose_voice=True, reference_asset_id=video,
                reference_start=0.0, reference_end=1.0)

def test_what_a_narration_said_is_compared_letter_for_letter_in_either_script() -> None:
    assert speak.said_it_all("原本的杯子區，長這樣。", "原本的杯子区长这样") == 1.0
    assert speak.said_it_all("原本的杯子區，長這樣。", "本的杯子區長這樣") < speak.SAID_IT_ALL + 0.1

def test_a_voice_sample_is_the_longest_sentence_that_fits() -> None:
    chosen = speak.reference_candidates([(0, 2, "太短"), (3, 9, "剛好"), (10, 25, "太長"), (30, 34, "也可以")])
    assert [text for _, _, text in chosen] == ["剛好", "也可以"]
