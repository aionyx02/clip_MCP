"""Tests for the storyboard that shows an edit before it is rendered."""

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from app.server import get_project, import_asset, preview_project
from helpers import build_project, insert, video_track

def storyboard(project_id: str, **kwargs) -> tuple:
    """Call `preview_project` and split its result.

    Args:
        project_id: Project to preview.
        **kwargs: Extra arguments for `preview_project`.

    Returns:
        The text lines and the decoded sheet image.
    """
    text, image = preview_project(project_id, **kwargs).content
    return text.text.splitlines(), Image.open(io.BytesIO(base64.b64decode(image.data)))

def tiles(lines: list) -> list:
    """Select the per-tile lines from a storyboard's text.

    Args:
        lines: All lines of the text block.

    Returns:
        The lines describing a tile.
    """
    return [line for line in lines if line.startswith("#")]

@pytest.fixture
def assets(media: Path) -> dict:
    """Import the shared media files.

    Returns:
        The assets keyed by file stem.
    """
    return {name: import_asset(str(media / f"{name}.mp4"))["id"] for name in ("wide", "tall", "silent")}

def test_shows_every_clip_even_when_asked_for_fewer_frames(assets: dict) -> None:
    project = build_project([
        video_track(),
        insert("a", assets["wide"], 0, 8),
        insert("b", assets["tall"], 0, 0.5),
        insert("c", assets["silent"], 0, 3),
    ])
    lines, _ = storyboard(project, count=1)
    assert len(tiles(lines)) == 3
    assert {"a", "b", "c"} == {line.split("clip ")[1].split(" |")[0] for line in tiles(lines)}

def test_gives_longer_clips_more_frames(assets: dict) -> None:
    project = build_project([
        video_track(),
        insert("long", assets["wide"], 0, 9),
        insert("short", assets["tall"], 0, 0.5),
    ])
    lines, _ = storyboard(project, count=8)
    counts = {"long": 0, "short": 0}
    for line in tiles(lines):
        counts[line.split("clip ")[1].split(" |")[0]] += 1
    assert counts["long"] > counts["short"]
    assert sum(counts.values()) == 8

def test_lists_tiles_in_timeline_order_with_both_times(assets: dict) -> None:
    project = build_project([video_track(), insert("a", assets["wide"], 4, 8)])
    lines, _ = storyboard(project, count=2)
    assert "clip a | edit 00:01.0 | source 5.000s (00:05.0)" in lines[1]
    assert "clip a | edit 00:03.0 | source 7.000s (00:07.0)" in lines[2]

def test_reports_black_gaps_between_clips(assets: dict) -> None:
    project = build_project([
        video_track(),
        insert("a", assets["wide"], 0, 2),
        {"action": "add_clip", "track_id": "main", "clip_id": "b", "asset_id": assets["tall"],
         "source_range": {"start": 0, "end": 2}, "timeline_in": 5},
    ])
    lines, _ = storyboard(project)
    gaps = [line for line in lines if "black gap" in line]
    assert gaps == ["-- black gap 00:02.0 to 00:05.0 (3.000s)"]

def test_reports_audio_tracks_that_the_frames_cannot_show(assets: dict, media: Path) -> None:
    song = import_asset(str(media / "song.mp3"))["id"]
    project = build_project([
        video_track(),
        insert("a", assets["wide"], 0, 6),
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        insert("m", song, 0, 6, track_id="music", volume=0.25),
    ])
    lines, _ = storyboard(project)
    assert "-- audio track music: 1 clip(s), 00:00.0 to 00:06.0, volume 0.25" in lines

def test_summarizes_the_output_format_and_length(assets: dict) -> None:
    project = build_project([video_track(), insert("a", assets["wide"], 0, 3)], width=1080, height=1920)
    lines, _ = storyboard(project)
    assert "1080x1920" in lines[0] and "30/1 fps" in lines[0] and "00:03.0 long" in lines[0]

def test_caps_the_number_of_tiles_and_says_what_it_skipped(assets: dict) -> None:
    operations = [video_track()]
    operations += [insert(f"c{index}", assets["wide"], index * 0.2, index * 0.2 + 0.2) for index in range(40)]
    lines, _ = storyboard(build_project(operations))
    assert len(tiles(lines)) == 36
    assert "4 clip(s) were skipped" in lines[0]

def long_sequence(asset_id: str, clips: int = 40) -> list:
    """Build a run of short clips, end to end, long enough to overflow the sheet.

    Args:
        asset_id: Asset every clip comes from.
        clips: How many clips to lay down.

    Returns:
        The operations, starting with the track.
    """
    return [video_track()] + [
        insert(f"c{index}", asset_id, index * 0.2, index * 0.2 + 0.2) for index in range(clips)
    ]

def test_a_skipped_clip_is_not_mistaken_for_a_black_gap(assets: dict) -> None:
    # Forty clips end to end, so four have to be left out of the sheet. The gaps used to be
    # read off the sampled list, which printed every skipped clip as a hole in the edit.
    lines, _ = storyboard(build_project(long_sequence(assets["wide"])))
    assert [line for line in lines if "black gap" in line] == []
    # They are still accounted for, in the places where they were dropped.
    notes = [line for line in lines if "not shown" in line]
    assert sum(int(line.split()[1]) for line in notes) == 4

def test_a_real_gap_survives_the_clips_that_are_skipped(assets: dict) -> None:
    operations = long_sequence(assets["wide"])
    operations.append({
        "action": "add_clip", "track_id": "main", "clip_id": "late", "asset_id": assets["tall"],
        "source_range": {"start": 0, "end": 2}, "timeline_in": 10,
    })
    lines, _ = storyboard(build_project(operations))
    assert [line for line in lines if "black gap" in line] == ["-- black gap 00:08.0 to 00:10.0 (2.000s)"]

def test_crops_tiles_to_the_project_shape(assets: dict) -> None:
    operations = [video_track(), insert("a", assets["wide"], 0, 4)]
    _, portrait = storyboard(build_project(operations, width=1080, height=1920), count=1)
    _, landscape = storyboard(build_project(operations, width=1920, height=1080), count=1)
    # The same landscape source must be shown portrait in a portrait project.
    assert portrait.height > portrait.width
    assert landscape.width > landscape.height

def test_samples_inside_a_clip_that_ends_at_the_assets_last_frame(assets: dict) -> None:
    project = build_project([video_track(), insert("last", assets["wide"], 9.9, 10.0)])
    lines, sheet = storyboard(project, count=3)
    assert len(tiles(lines)) == 3
    assert sheet.width > 0

def test_missing_project_is_reported_clearly() -> None:
    with pytest.raises(ValueError, match="project nope not found"):
        preview_project("nope")

def test_project_without_video_clips_is_reported_clearly() -> None:
    project = build_project([{"action": "add_track", "track_id": "music", "track_type": "audio"}])
    with pytest.raises(ValueError, match="no clips on a video track to preview"):
        preview_project(project)

def test_get_project_keeps_the_settings_the_caller_always_needs(assets: dict) -> None:
    project = build_project([video_track(), insert("a", assets["wide"], 0, 3)], width=1080, height=1920)
    state = get_project(project)
    assert state["version"] == 2
    assert (state["width"], state["height"], state["fps_num"], state["fps_den"]) == (1080, 1920, 30, 1)
    assert float(state["duration"]) == 3.0
    assert state["tracks"][0]["id"] == "main" and state["tracks"][0]["track_type"] == "video"

def test_get_project_leaves_out_clip_fields_still_at_their_default(assets: dict) -> None:
    project = build_project([
        video_track(),
        insert("plain", assets["wide"], 0, 3),
        insert("quiet", assets["wide"], 3, 5, volume=0.4),
    ])
    clips = {clip["id"]: clip for clip in get_project(project)["tracks"][0]["clips"]}
    assert "volume" not in clips["plain"] and "color" not in clips["plain"]
    assert set(clips["plain"]) == {"id", "asset_id", "source_range", "timeline_in"}
    # Anything actually set is still reported.
    assert float(clips["quiet"]["volume"]) == 0.4

def test_get_project_reports_an_empty_track(assets: dict) -> None:
    project = build_project([video_track(), insert("a", assets["wide"], 0, 3), video_track("spare")])
    spare = next(track for track in get_project(project)["tracks"] if track["id"] == "spare")
    assert spare["clips"] == []
