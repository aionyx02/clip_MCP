"""Photos in a video: shown for as long as their clip runs, and moving slowly while they are."""

from pathlib import Path

import pytest

from app import server
from app.engine import stills
from app.engine.frames import extract_frame
from helpers import build_project, clips_of, edit, insert, render, video_track


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    """A landscape photo: red on the left half, blue on the right."""
    from PIL import Image

    picture = Image.new("RGB", (800, 600), (220, 30, 30))
    picture.paste((30, 30, 220), (400, 0, 800, 600))
    path = tmp_path / "photo.png"
    picture.save(path)
    return path


def test_a_photo_moves_the_way_it_was_told() -> None:
    assert stills.motion_at(None, 0) == (1.0, 0.5)
    assert stills.motion_at("push", 1)[0] == pytest.approx(1 + stills.MOTION_AMOUNT)
    assert stills.motion_at("pull", 1)[0] == pytest.approx(1.0)
    assert stills.motion_at("pan_left", 0.25) == (pytest.approx(1 + stills.MOTION_AMOUNT), 0.25)
    assert stills.motion_at("none", 0.7) == (1.0, 0.5)
    assert "d=120:s=1920x1080:fps=30" in stills.zoompan("push", 120, 1920, 1080, "30")


def test_a_photo_is_added_as_a_photo(photo: Path) -> None:
    added = server.import_asset(str(photo))
    listed = next(asset for asset in server.list_assets()["assets"] if asset["id"] == added["id"])
    assert listed["photo"] is True and listed["duration"] is None
    assert server.repo.get_assets([added["id"]])[added["id"]].duration == stills.STILL_LONGEST


def test_a_photo_renders_for_as_long_as_its_clip(photo: Path, tmp_path: Path) -> None:
    asset = server.import_asset(str(photo))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], width=320, height=240, name="照片")
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "a", "motion": "pan_right"}])
    assert clips_of(project, "main")[0]["motion"] == "pan_right"
    out = tmp_path / "out.mp4"
    render(project, out, loudness_target=None)
    from app.engine.probe import probe_file

    assert float(probe_file(str(out))["format"]["duration"]) == pytest.approx(3, abs=0.1)
    # Panning right starts on the photo's right side, which is blue, and ends nearer its middle.
    first = extract_frame(str(out), 0.0, max_size=320).getpixel((310, 120))
    assert first[2] > first[0]


def test_a_photo_is_looked_at_for_faces_only(photo: Path, tmp_path: Path) -> None:
    from app.engine.analysis import analyze_media

    added = server.import_asset(str(photo))["id"]
    asset = server.repo.get_assets([added])[added]
    analysis = analyze_media(asset, str(tmp_path), True, None, None, None, lambda *_: None, lambda: False, "ffmpeg")
    assert analysis.duration == stills.STILL_SECONDS
    assert [(span.start, span.end) for span in analysis.scenes] == [(0.0, stills.STILL_SECONDS)]
    assert analysis.transcript is None
    # Red and blue at full strength: a middling exposure, measured over as long as the photo is shown.
    assert len(analysis.shots) == 1 and 0.1 < analysis.shots[0].exposure < 0.5
    assert analysis.shots[0].end == float(stills.STILL_LONGEST) and analysis.shots[0].motion is None


def test_a_landscape_photo_in_a_vertical_video_sits_whole_over_its_blur(photo: Path, tmp_path: Path) -> None:
    asset = server.import_asset(str(photo))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2)], width=360, height=640, name="直式照片")
    stored = server.repo.get_project(project)
    framing = server._framing(stored, server._referenced_assets(stored))
    assert framing["a"].whole
    out = tmp_path / "out.mp4"
    render(project, out, loudness_target=None, framing=framing)
    picture = extract_frame(str(out), 1.0, max_size=640).convert("L")
    top = picture.crop((0, 0, picture.width, picture.height // 8))
    assert sum(top.getdata()) / (top.width * top.height) > 20


def test_the_editor_plays_a_photo_from_the_file_itself(photo: Path) -> None:
    from app.engine import playback

    asset = server.import_asset(str(photo))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 4)], width=320, height=240, name="播照片")
    stored = server.repo.get_project(project)
    assets = server._referenced_assets(stored)
    shown = playback.describe(stored, assets, playback.Prepared({}, {}, None), None, {}, {})
    clip = shown["tracks"][0]["clips"][0]
    assert clip["still"] and clip["proxy"] == f"/media/{asset}" and clip["motion"] == stills.DEFAULT_MOTION
    # Nothing to wait for and nothing to copy: a photo is drawn as it is.
    assert shown["waiting"] == [] and playback.missing(stored, assets, []) == []
