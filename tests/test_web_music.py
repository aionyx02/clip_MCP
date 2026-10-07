"""Music from the web: only from the allowed sites, on terms the user agreed to, and only a song."""

import shutil
from pathlib import Path

import pytest

from app import server
from app.engine import fetch


def test_only_the_allowed_sites_and_never_plain_http() -> None:
    assert fetch.site_of("https://cdn.pixabay.com/download/audio/2022/song.mp3") == "pixabay"
    assert fetch.site_of("https://incompetech.com/music/royalty-free/mp3-royaltyfree/Wholesome.mp3") == "incompetech"
    assert fetch.site_of("http://cdn.pixabay.com/a.mp3") is None
    assert fetch.site_of("https://www.youtube.com/watch?v=x") is None
    # A look-alike is not the site.
    assert fetch.site_of("https://pixabay.com.evil.example/a.mp3") is None
    assert fetch.site_of("https://notpixabay.com/a.mp3") is None


def test_a_track_s_own_licence_decides_what_it_may_be_used_for() -> None:
    assert fetch.licence_problem("pixabay", None, False) is None
    assert "licence" in fetch.licence_problem("fma", None, False)
    assert "commercial use" in fetch.licence_problem("fma", "CC BY-NC", False)
    assert fetch.licence_problem("fma", "CC BY-NC", True) is None
    assert fetch.licence_problem("commons", "CC BY-SA", False) is None


def test_the_credit_is_written_the_way_the_licence_asks() -> None:
    assert fetch.credit_line("incompetech", "Wholesome", "", "CC BY 4.0", "p").startswith('"Wholesome" Kevin MacLeod')
    assert fetch.credit_line("pixabay", "A", "B", "Pixabay Content License", "p") is None
    assert fetch.credit_line("fma", "Song", "Band", "CC0", "p") is None
    assert fetch.credit_line("fma", "Song", "Band", "CC BY", "https://x") == '"Song" by Band (https://x), licensed under CC BY'


def test_a_redirect_stays_on_the_site_it_started_on() -> None:
    handler = fetch._StayOnSite("pixabay")
    with pytest.raises(ValueError, match="not the site it is on"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://elsewhere.example/a.mp3")
    # Another allowed site is still another site: its licence was not the one read.
    with pytest.raises(ValueError, match="not the site it is on"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://upload.wikimedia.org/a.mp3")


def test_a_song_is_taken_only_by_the_plan_the_user_heard(media: Path, monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    def fake_download(url, folder, title):
        Path(folder).mkdir(parents=True, exist_ok=True)
        return shutil.copy(media / "song.mp3", Path(folder) / f"{title}.mp3")

    monkeypatch.setattr(server, "output_dir", lambda: tmp_path)
    monkeypatch.setattr(server.fetch, "download", fake_download)
    url, page = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Tone.mp3", "https://incompetech.com/x"
    shown = server.add_music_from_url(url, page, "Tone")
    assert shown["licence"] == "CC BY 4.0" and shown["credit"].startswith('"Tone" Kevin MacLeod')
    assert not (tmp_path / "音樂").exists()
    with pytest.raises(ValueError):
        server.add_music_from_url(url, page, "Tone", confirm_plan="nope")
    taken = server.add_music_from_url(url, page, "Tone", confirm_plan=shown["plan"])
    asset = server.repo.get_assets([taken["id"]])[taken["id"]]
    assert asset.library == "music" and asset.source.licence == "CC BY 4.0"
    assert Path(taken["path"]).parent == tmp_path / "音樂"

    from helpers import build_project, video_track
    clip = {"action": "insert_clip", "track_id": "music", "clip_id": "m", "asset_id": asset.id,
            "source_range": {"start": 0, "end": 5}}
    project = build_project([video_track(), {"action": "add_track", "track_id": "music", "track_type": "audio"},
                             clip], name="有配樂")
    assert server.music_credits(project)["credits"] == [taken["credit"]]


def test_a_page_on_another_site_is_refused() -> None:
    with pytest.raises(ValueError, match="only taken from"):
        server.add_music_from_url("https://cdn.pixabay.com/a.mp3", "https://example.com/a", "A")
