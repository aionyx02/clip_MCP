"""Footage and music are two libraries: a file goes where it belongs, and is analyzed for what it is."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app import server
from app.engine import library
from app.engine.analysis import analyze_media
from app.models.media import Asset


def test_a_file_without_a_picture_is_music_unless_it_is_mostly_talk(monkeypatch: pytest.MonkeyPatch) -> None:
    song = Asset(id="s", path="song.mp3", duration=60, has_video=False, has_audio=True)
    monkeypatch.setattr(library, "speech_share", lambda *_: 0.03)
    assert library.classify(song) == library.MUSIC
    monkeypatch.setattr(library, "speech_share", lambda *_: 0.9)
    assert library.classify(song) == library.FOOTAGE
    shot = Asset(id="v", path="shot.mp4", duration=60, has_video=True, has_audio=True)
    assert library.classify(shot) == library.FOOTAGE


def test_a_tone_is_not_heard_as_speech(media: Path) -> None:
    assert library.speech_share(str(media / "song.mp3"), 20.0) < library.SPEECH_SHARE


def test_music_is_never_transcribed_and_a_voice_recording_never_beat_tracked(media: Path, tmp_path: Path) -> None:
    song = Asset(id="s", path=str(media / "song.mp3"), duration=20, has_video=False, has_audio=True,
                 library=library.MUSIC)
    # Asked to transcribe, a song is still only listened to for its beat.
    heard = analyze_media(song, str(tmp_path), True, None, None, None, lambda *_: None, lambda: False, "ffmpeg")
    assert heard.transcript is None and heard.recipe.rhythm_model is not None
    voice = song.model_copy(update={"id": "v", "library": library.FOOTAGE})
    heard = analyze_media(voice, str(tmp_path), False, None, None, None, lambda *_: None, lambda: False, "ffmpeg")
    assert heard.rhythm is None and heard.recipe.rhythm_model is None


def test_each_library_lists_its_own_files_and_a_file_can_move(media: Path) -> None:
    song = server.import_asset(str(media / "jingle.wav"))
    shot = server.import_asset(str(media / "wide.mp4"))
    assert song["library"] == "music" and shot["library"] == "footage"
    music = {item["id"] for item in server.list_assets(kind="music", limit=200)["assets"]}
    footage = {item["id"] for item in server.list_assets(kind="footage", limit=200)["assets"]}
    assert song["id"] in music and song["id"] not in footage and shot["id"] in footage

    moved = server.set_asset_library([song["id"]], "footage")
    assert moved["moved"] == ["jingle.wav"]
    assert song["id"] in {item["id"] for item in server.list_assets(kind="footage", limit=200)["assets"]}
    # Footage and music are filed apart: footage does not go in a music folder.
    with pytest.raises(ValueError, match="separate steps"):
        server.organize_library([server.CreateFolder(action="create_folder", name="分開放", library="music",
                                                     asset_ids=[song["id"]])])
    server.set_asset_library([song["id"]], "music")


def test_a_song_added_is_analyzed_behind_other_work(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = []
    monkeypatch.setattr(server, "ANALYZE_MUSIC_ON_ADD", True)
    monkeypatch.setattr(server, "_start_analysis", lambda asset, *args, **kwargs: started.append(
        (asset.id, kwargs.get("priority"))) or type("Job", (), {"job_id": "j"})())
    song = server.import_asset(str(media / "song.mp3"))
    assert started == [(song["id"], 1)]
    server.import_asset(str(media / "tall.mp4"))
    assert len(started) == 1


def test_the_editor_shows_and_moves_each_library(media: Path) -> None:
    from app.ui import app as editor

    client = TestClient(editor.create_app())
    song = server.import_asset(str(media / "song.mp3"))["id"]
    listed = client.get("/api/assets?library=music").json()["assets"]
    assert any(item["id"] == song and item["library"] == "music" for item in listed)
    assert not any(item["id"] == song for item in client.get("/api/assets?library=footage").json()["assets"])
    assert client.post("/api/assets/library", json={"ids": [song], "library": "footage"}).status_code == 200
    assert any(item["id"] == song for item in client.get("/api/assets?library=footage").json()["assets"])
    client.post("/api/assets/library", json={"ids": [song], "library": "music"})


def test_a_file_with_a_picture_is_never_music(media: Path) -> None:
    shot = server.import_asset(str(media / "wide.mp4"))["id"]
    with pytest.raises(ValueError, match="has a picture"):
        server.set_asset_library([shot], "music")


def test_a_song_moved_to_footage_is_listened_to_for_what_is_said(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.models.media import MediaAnalysis, Rhythm

    song = server.import_asset(str(media / "jingle.wav"))["id"]
    server.repo.save_analysis(MediaAnalysis(asset_id=song, duration=4.0, rhythm=Rhythm(tempo=None, beats=[])))
    started = []
    monkeypatch.setattr(server, "ANALYZE_MUSIC_ON_ADD", True)
    monkeypatch.setattr(server, "_start_analysis", lambda asset, transcribe, *args, **kwargs: started.append(
        (asset.id, transcribe)) or type("Job", (), {"job_id": "j"})())
    server.set_asset_library([song], "footage")
    assert started == [(song, True)]
    server.set_asset_library([song], "music")


def test_music_laid_by_hand_that_stops_early_is_said(media: Path) -> None:
    from helpers import build_project, insert, video_track

    shot = server.import_asset(str(media / "wide.mp4"))["id"]
    song = server.import_asset(str(media / "jingle.wav"))["id"]
    project = build_project([video_track(), insert("a", shot, 0, 10),
                             {"action": "add_track", "track_id": "music", "track_type": "audio",
                              "duck_under_speech": True},
                             {"action": "insert_clip", "track_id": "music", "clip_id": "m", "asset_id": song,
                              "source_range": {"start": 0, "end": 4}}], name="音樂太短")
    assert "stops at 4.0s" in server._music_ends_early(server.repo.get_project(project))


def test_analyzing_again_measures_only_what_would_come_out_different(media: Path, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import analysis

    shot = Asset(id="w", path=str(media / "wide.mp4"), duration=4, has_video=True, has_audio=True)
    first = analyze_media(shot, str(tmp_path), False, None, None, None, lambda *_: None, lambda: False, "ffmpeg")
    monkeypatch.setattr(analysis, "scan_media", lambda *_: pytest.fail("measured the file again"))
    monkeypatch.setattr(analysis.faces, "detect_faces", lambda *_: pytest.fail("looked for faces again"))
    again = analyze_media(shot, str(tmp_path), False, None, None, None, lambda *_: None, lambda: False, "ffmpeg",
                          earlier=first)
    assert (again.silences, again.sound, again.faces) == (first.silences, first.sound, first.faces)
    # Measured some other way, it is measured again.
    stale = first.model_copy(update={"recipe": first.recipe.model_copy(update={"detectors": "older"})})
    with pytest.raises(pytest.fail.Exception, match="measured the file again"):
        analyze_media(shot, str(tmp_path), False, None, None, None, lambda *_: None, lambda: False, "ffmpeg",
                      earlier=stale)
