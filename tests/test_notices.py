"""What a user is told to take care of before publishing a cut."""

from decimal import Decimal
from pathlib import Path

from app.engine.delivery import rights_facts
from app.models.media import Asset, FaceMeasurement, MediaAnalysis, Source, Transcript, TranscriptSegment
from app.models.timeline import Clip, Project, TimeRange, Track, TrackType
from app.server import export_notices, import_asset, repo, set_asset_library
from helpers import build_project, insert, video_track


def clip(clip_id: str, asset_id: str, start: float, end: float, at: float) -> Clip:
    return Clip(id=clip_id, asset_id=asset_id, source_range=TimeRange(start=Decimal(str(start)), end=Decimal(str(end))),
                timeline_in=Decimal(str(at)))


def test_everything_that_needs_a_word_before_publishing_is_listed() -> None:
    project = Project(id="p", width=1080, height=1920, tracks=[
        Track(id="main", track_type=TrackType.VIDEO, clips=[clip("a", "v", 0, 10, 0)]),
        Track(id="music", track_type=TrackType.AUDIO, clips=[clip("m", "song", 0, 10, 0), clip("w", "web", 0, 4, 10)]),
    ])
    assets = {
        "v": Asset(id="v", path="/footage/a.mp4", has_video=True, has_audio=True),
        "song": Asset(id="song", path="/music/Carefree.mp3", has_video=False, has_audio=True, library="music",
                      licence="Kevin MacLeod, CC BY 4.0"),
        "web": Asset(id="web", path="/music/Wepa.mp3", has_video=False, has_audio=True, library="music",
                     source=Source(site="YouTube 音效庫", page="https://example.com/wepa", url="https://example.com/w.mp3",
                                   title="Wepa", licence="YouTube Audio Library", credit="Wepa by Gloria")),
    }
    analyses = {"v": MediaAnalysis(
        asset_id="v", duration=10.0,
        transcript=Transcript(language="zh", model="t", segments=[TranscriptSegment(start=1, end=6, text="各位")]),
        faces=[FaceMeasurement(start=2.0, end=3.0, faces=2, face_share=0.1, face_x=0.5, face_y=0.5)],
    )}
    facts = rights_facts(project, analyses, assets, {"song": {"artist": "Kevin MacLeod"}})
    music = {fact.subject: fact for fact in facts if fact.kind == "music"}
    assert "artist: Kevin MacLeod" in music["Carefree.mp3"].detail
    assert "the user said: Kevin MacLeod, CC BY 4.0" in music["Carefree.mp3"].detail
    assert "credit it asks for: Wepa by Gloria" in music["Wepa.mp3"].detail
    # The footage's own sound on a track of its own is the user's recording, not one to license.
    project.tracks[1].clips.append(clip("j", "v", 10, 12, 14))
    assert [fact.subject for fact in rights_facts(project, analyses, assets, {}) if fact.kind == "recording"] == []
    kinds = {fact.kind: fact for fact in facts}
    # Nobody told the voices apart, which is said rather than taken to mean nobody else talks.
    assert kinds["unchecked_voices"].spans == ((0.0, 10.0),)
    assert kinds["faces"].spans == ((2.0, 3.0),)


def test_notices_are_written_beside_the_video_with_every_fact_under_them(media: Path, tmp_path: Path) -> None:
    song = import_asset(str(media / "song.mp3"))["id"]
    # Other tests move the shared song between the libraries; here it is music.
    set_asset_library([song], "music")
    video = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([
        video_track(), insert("v", video, 0, 5),
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        insert("song", song, 0, 5, track_id="music"),
    ])
    asked = export_notices(project, licences={song: "Kevin MacLeod, CC BY 4.0"})
    assert not asked["written"] and any("CC BY 4.0" in fact for fact in asked["facts"])
    written = export_notices(project, "配樂要標註作者。", next_to=str(tmp_path / "cut.mp4"))
    saved = Path(written["output_path"])
    assert saved == tmp_path / "cut_notices.txt"
    text = saved.read_text(encoding="utf-8-sig")
    assert text.startswith("配樂要標註作者。") and "CC BY 4.0" in text
    # What the user said about the file is kept with it, even when it is imported again.
    import_asset(str(media / "song.mp3"))
    assert repo.get_assets([song])[song].licence == "Kevin MacLeod, CC BY 4.0"
