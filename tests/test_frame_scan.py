"""Every frame of a finished render is read for black, a stopped picture and a flash."""

from pathlib import Path

from app.engine import scan
from app.server import import_asset, render_project
from helpers import build_project, edit, finished, insert, video_track


def test_a_few_frames_that_jump_in_brightness_and_come_back_are_a_flash() -> None:
    steady = [(index / 30, 0.5) for index in range(10)]
    flashed = steady[:4] + [(4 / 30, 0.95), (5 / 30, 0.95)] + steady[6:]
    assert scan._flashes(flashed) == [(4 / 30, 6 / 30)]
    # A cut to a brighter shot that stays there is a cut, not a flash.
    assert scan._flashes(steady[:4] + [(index / 30, 0.95) for index in range(4, 10)]) == []
    # Longer than a few frames, it is a shot.
    assert scan._flashes(steady[:2] + [(index / 30, 0.95) for index in range(2, 7)] + steady[7:]) == []


def test_the_scan_is_read_from_its_log(tmp_path: Path) -> None:
    (tmp_path / scan.LOG_FILE_NAME).write_text(
        "[blackdetect @ 0] black_start:2.0 black_end:2.1 black_duration:0.1\n"
        "[freezedetect @ 0] lavfi.freezedetect.freeze_start: 5.5\n"
        "[freezedetect @ 0] lavfi.freezedetect.freeze_end: 6\n"
        "[freezedetect @ 0] lavfi.freezedetect.freeze_start: 9\n", encoding="utf-8")
    assert scan.spots(str(tmp_path), 10.0) == [
        {"kind": "black", "start": 2.0, "end": 2.1}, {"kind": "frozen", "start": 5.5, "end": 6.0},
        {"kind": "frozen", "start": 9.0, "end": 10.0},
    ]


def test_black_nobody_asked_for_is_found_in_a_render_and_a_fade_is_not(media: Path) -> None:
    wide, black = import_asset(str(media / "wide.mp4"))["id"], import_asset(str(media / "black.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", black, 0, 0.5),
                             insert("c", wide, 2, 4)], width=640, height=360)
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "c", "video_fade_out": 0.5}])
    frames = finished(render_project(project))["frames"]
    assert len(frames["found"]) == 1 and frames["found"][0].startswith("2.00–2.50s black (0.50s) in clip b")
    assert any("fade out of c" in line for line in frames["meant"])
