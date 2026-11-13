"""Tests for drawing one video track on top of another: a picture in a corner, and a split screen."""

import subprocess
from pathlib import Path

import pytest

from app.engine import insets, playback
from app.engine.builder import DUCK_DEPTH_DB, Talking, overlay_box
from app.engine.frames import extract_frame
from app.models.timeline.parts import CaptionStyle, ClipLayout
from app.runtime import repo
from app.tools import rendering, shared
from app.tools.catalog import import_asset
from app.tools.rendering import preview_project
from helpers import build_project, clips_of, edit, insert, layout, level, render, video_track

def brightness(path: Path, seconds: float, box: tuple) -> float:
    """Measure the average brightness of a rectangle of one frame.

    Args:
        path: Video to sample.
        seconds: Time of the frame.
        box: Region as `(left, top, right, bottom)` fractions of the frame.

    Returns:
        Mean luminance from 0 to 255.
    """
    frame = extract_frame(str(path), seconds, max_size=640).convert("L")
    width, height = frame.size
    left, top, right, bottom = box
    region = frame.crop((int(width * left), int(height * top), int(width * right), int(height * bottom)))
    return sum(region.get_flattened_data()) / (region.width * region.height)

def colour(path: Path, seconds: float, box: tuple) -> tuple:
    """Average the colour of a rectangle of one frame, as `(red, green, blue)`."""
    frame = extract_frame(str(path), seconds, max_size=640).convert("RGB")
    width, height = frame.size
    left, top, right, bottom = box
    region = frame.crop((int(width * left), int(height * top), int(width * right), int(height * bottom)))
    return region.resize((1, 1)).getpixel((0, 0))

def box(place: str, size=None) -> ClipLayout:
    """A layout."""
    return ClipLayout(place=place, size=size)

# --- where the box goes ------------------------------------------------------------------------

STYLE = CaptionStyle()
WIDE = (1920, 1080)

def test_a_medium_corner_on_a_landscape_frame_is_a_third_of_its_width() -> None:
    # 1080 short side: a 43 px margin; captions 68 px, two lines reach up to 845, so a bottom
    # corner ends 43 px above that, at 802. A third of the short side, in the shot's 16:9.
    assert insets.inset_box(box("top_left", "medium"), 1920, 1080, STYLE, WIDE) == (43, 43, 634, 356)
    assert insets.inset_box(box("bottom_right", "medium"), 1920, 1080, STYLE, WIDE) == (1243, 446, 634, 356)
    assert 634 / 1920 == pytest.approx(0.33, abs=0.005)

@pytest.mark.parametrize("size, wide", [("small", 480), ("medium", 634), ("large", 768)])
def test_the_three_sizes_are_a_quarter_a_third_and_two_fifths_of_a_landscape_frame(size: str, wide: int) -> None:
    _, _, width, height = insets.inset_box(box("top_right", size), 1920, 1080, STYLE, WIDE)
    assert width == wide and abs(width / height - 16 / 9) < 0.01

@pytest.mark.parametrize("frame", [(1920, 1080), (1080, 1920), (1080, 1080)])
@pytest.mark.parametrize("corner", ["top_left", "top_right", "bottom_left", "bottom_right"])
@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_a_corner_keeps_its_corner_its_size_and_clear_of_the_captions(frame, corner, size) -> None:
    width, height = frame
    x, y, w, h = insets.inset_box(box(corner, size), width, height, STYLE, WIDE)
    margin = round(min(width, height) * insets.INSET_MARGIN)
    # The same size in every shape, worked out from the short side.
    assert (w, h) == insets.inset_box(box(corner, size), 1920, 1080, STYLE, WIDE)[2:]
    assert w % 2 == 0 and h % 2 == 0
    assert (x == margin) if corner.endswith("left") else (x + w == width - margin)
    if corner.startswith("top"):
        assert y == margin
    else:
        assert y + h == int(insets.captions_top(width, height, STYLE) - margin)
    assert 0 <= x and x + w <= width and 0 <= y and y + h <= height

def test_a_corner_on_a_portrait_frame_stays_in_its_corner_above_the_phone_s_captions() -> None:
    # Captions on a vertical frame sit higher, clear of the phone's own controls: 250 px up.
    assert insets.inset_box(box("top_right", "medium"), 1080, 1920, STYLE, WIDE) == (403, 43, 634, 356)
    assert insets.inset_box(box("bottom_left", "medium"), 1080, 1920, STYLE, WIDE) == (43, 1101, 634, 356)

def test_a_corner_keeps_the_shape_of_its_own_picture() -> None:
    # A phone held upright, in the corner of a landscape frame: tall, its short edge the same.
    assert insets.inset_box(box("top_left", "small"), 1920, 1080, STYLE, (1080, 1920))[2:] == (270, 480)

def test_a_picture_too_big_for_the_room_is_brought_down_its_shape_kept() -> None:
    # A tall phone shot, large, on a landscape frame: 432 x 768 would run into the captions.
    _, y, w, h = insets.inset_box(box("bottom_left", "large"), 1920, 1080, STYLE, (1080, 1920))
    assert 43 <= y and y + h <= 802 and abs(h / w - 16 / 9) < 0.02

@pytest.mark.parametrize("frame, place, expected", [
    ((1920, 1080), "left", (0, 0, 960, 1080)), ((1920, 1080), "right", (960, 0, 960, 1080)),
    ((1920, 1080), "top", (0, 0, 1920, 540)), ((1920, 1080), "bottom", (0, 540, 1920, 540)),
    # Side by side on a phone would be two slivers: a left half is laid as the top one.
    ((1080, 1920), "left", (0, 0, 1080, 960)), ((1080, 1920), "right", (0, 960, 1080, 960)),
    ((1080, 1920), "top", (0, 0, 1080, 960)), ((1080, 1920), "bottom", (0, 960, 1080, 960)),
    ((1080, 1080), "left", (0, 0, 540, 1080)), ((1080, 1080), "bottom", (0, 540, 1080, 540)),
])
def test_a_half_is_half_the_frame(frame, place, expected) -> None:
    assert insets.inset_box(box(place), *frame, STYLE, WIDE) == expected

def test_a_corner_has_a_line_all_round_and_a_half_only_between_the_halves() -> None:
    corner = insets.edges(box("top_left", "small"), (43, 43, 480, 270), 1920, 1080)
    assert sorted(corner) == sorted([(43, 43, 480, 2), (43, 311, 480, 2), (43, 43, 2, 270), (521, 43, 2, 270)])
    assert insets.edges(box("right"), (960, 0, 960, 1080), 1920, 1080) == [(960, 0, 2, 1080)]
    assert insets.edges(box("left"), (0, 0, 1080, 960), 1080, 1920) == [(0, 958, 1080, 2)]

def test_a_corner_needs_a_size_and_a_half_has_none() -> None:
    with pytest.raises(ValueError, match="needs a size"):
        ClipLayout(place="top_left")
    with pytest.raises(ValueError, match="has no size"):
        ClipLayout(place="left", size="small")
    with pytest.raises(ValueError):
        ClipLayout(place="middle", size="small")

# --- drawing it ------------------------------------------------------------------------------

@pytest.fixture
def sources(media: Path) -> dict:
    """Import a black base clip and a bright clip to lay over it.

    Returns:
        The asset IDs keyed by role.
    """
    return {
        "base": import_asset(str(media / "black.mp4"))["id"],
        "top": import_asset(str(media / "tall.mp4"))["id"],
        "wide": import_asset(str(media / "wide.mp4"))["id"],
    }

@pytest.fixture
def inset(sources: dict) -> str:
    """Build a 6 s black base with a bright inset in the bottom right from 2 s to 4 s.

    Returns:
        The project ID.
    """
    return build_project([
        video_track("base"),
        insert("b", sources["base"], 0, 6, track_id="base"),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "pip", "asset_id": sources["top"],
         "source_range": {"start": 0, "end": 2}, "timeline_in": 2,
         "layout": {"place": "bottom_right", "size": "small"}},
    ], width=640, height=360)

def test_the_inset_is_where_the_arithmetic_says(inset: str) -> None:
    project = repo.get_project(inset)
    clip = project.tracks[1].clips[0]
    # 360 short side: a 14 px margin, two lines of 22 px captions up to 283; the phone shot 90 x 160.
    assert overlay_box(clip, project, (360, 640)) == (536, 109, 90, 160)

def test_the_inset_is_drawn_only_while_it_is_on_the_timeline(inset: str, tmp_path: Path) -> None:
    render(inset, tmp_path / "out.mp4", loudness_target=None)
    inside = (540 / 640, 115 / 360, 620 / 640, 262 / 360)
    assert brightness(tmp_path / "out.mp4", 1.0, inside) < 10
    assert brightness(tmp_path / "out.mp4", 3.0, inside) > 40
    assert brightness(tmp_path / "out.mp4", 5.0, inside) < 10

def test_the_inset_stays_inside_its_box(inset: str, tmp_path: Path) -> None:
    render(inset, tmp_path / "out.mp4", loudness_target=None)
    # The base is black, so anything bright outside the box would be the inset spilling out.
    assert brightness(tmp_path / "out.mp4", 3.0, (0.0, 0.0, 530 / 640, 1.0)) < 10
    assert brightness(tmp_path / "out.mp4", 3.0, (0.0, 0.0, 1.0, 105 / 360)) < 10
    assert brightness(tmp_path / "out.mp4", 3.0, (0.0, 275 / 360, 1.0, 1.0)) < 10

def test_a_dark_inset_on_a_busy_picture_keeps_a_light_edge(sources: dict, tmp_path: Path) -> None:
    project = build_project([
        video_track("base"),
        insert("b", sources["wide"], 0, 3, track_id="base"),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "pip", "asset_id": sources["base"],
         "source_range": {"start": 0, "end": 2}, "timeline_in": 0, "layout": {"place": "top_left", "size": "large"}},
    ], width=640, height=360)
    render(project, tmp_path / "out.mp4", loudness_target=None)
    frame = extract_frame(str(tmp_path / "out.mp4"), 1.0, max_size=640).convert("L")
    x, y, w, h = overlay_box(repo.get_project(project).tracks[1].clips[0], repo.get_project(project), (640, 360))
    assert frame.getpixel((x + w // 2, y)) > 180 and frame.getpixel((x, y + h // 2)) > 180
    assert frame.getpixel((x + w // 2, y + h // 2)) < 20

def test_a_clip_without_a_layout_covers_the_whole_frame(sources: dict, tmp_path: Path) -> None:
    project = build_project([
        video_track("base"),
        insert("b", sources["base"], 0, 6, track_id="base"),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "cover", "asset_id": sources["top"],
         "source_range": {"start": 0, "end": 2}, "timeline_in": 2},
    ], width=640, height=360)
    render(project, tmp_path / "out.mp4", loudness_target=None)
    assert brightness(tmp_path / "out.mp4", 3.0, (0.1, 0.1, 0.9, 0.9)) > 40

def test_the_base_track_still_decides_the_length(sources: dict) -> None:
    project = build_project([
        video_track("base"),
        insert("b", sources["base"], 0, 3, track_id="base"),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "long", "asset_id": sources["top"],
         "source_range": {"start": 0, "end": 5}, "timeline_in": 0},
    ], width=640, height=360)
    assert layout(project, "base")[-1][2] == 3.0

# --- a split screen ----------------------------------------------------------------------------

@pytest.fixture
def thirds(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A landscape shot in three bands, red, green and blue, so where it is cropped can be read."""
    path = tmp_path_factory.mktemp("thirds") / "thirds.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:s=214x360:r=30:d=4", "-f", "lavfi", "-i", "color=c=lime:s=212x360:r=30:d=4",
        "-f", "lavfi", "-i", "color=c=blue:s=214x360:r=30:d=4",
        "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return import_asset(str(path))["id"]

def split_over(base: str, top: str, place: str, width: int, height: int) -> str:
    """A 4 s cut of `base` with `top` in one half of it from 1 s to 3 s."""
    return build_project([
        video_track("base"),
        insert("a", base, 0, 2, track_id="base"), insert("b", base, 2, 4, track_id="base"),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "half", "asset_id": top,
         "source_range": {"start": 0, "end": 2}, "timeline_in": 1, "volume": 0.0, "layout": {"place": place}},
    ], width=width, height=height)

def test_a_split_moves_the_sequence_into_the_other_half_cropped_on_its_middle(
        sources: dict, thirds: str, tmp_path: Path) -> None:
    project = split_over(thirds, sources["base"], "right", 640, 360)
    render(project, tmp_path / "out.mp4", loudness_target=None)
    # The left half shows the middle of the shot, not its left half: green, not red.
    red, green, blue = colour(tmp_path / "out.mp4", 2.0, (0.05, 0.1, 0.45, 0.9))
    assert green > 150 and red < 80 and blue < 80
    assert brightness(tmp_path / "out.mp4", 2.0, (0.52, 0.0, 1.0, 1.0)) < 10
    # The line between the halves.
    assert brightness(tmp_path / "out.mp4", 2.0, (320 / 640, 0.1, 322 / 640, 0.9)) > 150
    # Before the split the shot is whole again, its left band red.
    red, green, _ = colour(tmp_path / "out.mp4", 0.5, (0.02, 0.1, 0.25, 0.9))
    assert red > 150 and green < 80

def test_a_right_half_on_a_portrait_frame_is_the_bottom_one(sources: dict, thirds: str, tmp_path: Path) -> None:
    project = split_over(thirds, sources["base"], "right", 360, 640)
    render(project, tmp_path / "out.mp4", loudness_target=None)
    # The shot is in the top half, cropped on its middle; the split's picture, black, below.
    _, green, _ = colour(tmp_path / "out.mp4", 2.0, (0.3, 0.05, 0.7, 0.45))
    assert green > 150
    assert brightness(tmp_path / "out.mp4", 2.0, (0.0, 0.52, 1.0, 1.0)) < 10

def test_the_sequence_in_the_other_half_follows_a_trim_and_is_silent(sources: dict, thirds: str) -> None:
    project = split_over(thirds, sources["base"], "right", 640, 360)
    partners = insets.split_partners(repo.get_project(project))
    # One copy per clip of the sequence under the split, each over only the part under it.
    assert [(clip.id, float(clip.timeline_in), float(clip.timeline_out), float(clip.source_range.start))
            for clip in partners] == [("half~a", 1.0, 2.0, 1.0), ("half~b", 2.0, 3.0, 2.0)]
    assert all(clip.volume == 0 and clip.layout.place == "left" for clip in partners)
    edit(project, [{"action": "trim_clip", "track_id": "base", "clip_id": "a", "new_source_range": {"start": 0.5, "end": 2}}])
    trimmed = insets.split_partners(repo.get_project(project))
    assert [(clip.id, float(clip.source_range.start)) for clip in trimmed][0] == ("half~a", 1.5)

def test_the_editor_draws_the_split_as_the_render_does(sources: dict, thirds: str) -> None:
    project = repo.get_project(split_over(thirds, sources["base"], "right", 640, 360))
    assets = shared.referenced_assets(project)
    described = playback.describe(project, assets, playback.Prepared({}, {}, None), None, {},
                                  rendering.framing_of(project, assets))
    drawn = [(track["id"], [clip["id"] for clip in track["clips"]]) for track in described["tracks"]
             if track["type"] == "video"]
    # The sequence, the sequence again in the left half, the split over both.
    assert drawn == [("base", ["a", "b"]), (insets.SPLIT_TRACK_ID, ["half~a", "half~b"]), ("top", ["half"])]
    clips = {clip["id"]: clip for track in described["tracks"] for clip in track["clips"]}
    assert clips["half~a"]["box"] == [0, 0, 320, 360] and clips["half"]["box"] == [320, 0, 320, 360]
    assert clips["half"]["edges"] == [[320, 0, 2, 360]] and "edges" not in clips["half~a"]

def test_two_splits_at_once_are_refused(sources: dict) -> None:
    with pytest.raises(ValueError, match="only be moved into one other half"):
        build_project([
            video_track("base"),
            insert("b", sources["base"], 0, 4, track_id="base"),
            {"action": "add_track", "track_id": "top", "track_type": "video"},
            {"action": "add_track", "track_id": "top2", "track_type": "video"},
            {"action": "add_clip", "track_id": "top", "clip_id": "one", "asset_id": sources["top"],
             "source_range": {"start": 0, "end": 2}, "timeline_in": 0, "layout": {"place": "left"}},
            {"action": "add_clip", "track_id": "top2", "clip_id": "two", "asset_id": sources["top"],
             "source_range": {"start": 0, "end": 2}, "timeline_in": 1, "layout": {"place": "right"}},
        ], width=640, height=360)

# --- sound -------------------------------------------------------------------------------------

def test_an_inset_turned_down_leaves_only_the_sequence_s_sound(audio_media: Path, tmp_path: Path) -> None:
    talker = import_asset(str(audio_media / "quiet.mp4"))["id"]
    other = import_asset(str(audio_media / "loud.mp4"))["id"]
    alone = build_project([video_track("base"), insert("b", talker, 0, 4, track_id="base")], width=320, height=240)
    laid = build_project([
        video_track("base"), insert("b", talker, 0, 4, track_id="base"),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "pip", "asset_id": other, "volume": 0.0,
         "source_range": {"start": 0, "end": 2}, "timeline_in": 1, "layout": {"place": "top_left", "size": "small"}},
    ], width=320, height=240)
    render(alone, tmp_path / "alone.mp4", loudness_target=None)
    render(laid, tmp_path / "laid.mp4", loudness_target=None)
    assert abs(level(tmp_path / "laid.mp4", 1.5, 1.0) - level(tmp_path / "alone.mp4", 1.5, 1.0)) < 0.2

def test_an_inset_heard_drops_while_the_sequence_talks(audio_media: Path, media: Path, tmp_path: Path) -> None:
    # The sequence talks from three seconds on, as its transcript would say, in a 440 Hz tone far
    # from the inset's 800 Hz one, which plays throughout.
    talking = import_asset(str(media / "wide.mp4"))["id"]
    said = Talking(spans=((3.0, 6.0),), footage_spans=((3.0, 6.0),))
    tone = import_asset(str(audio_media / "loud.mp4"))["id"]

    def with_inset(ducked: bool) -> Path:
        project = build_project([
            video_track("base"), insert("b", talking, 0, 6, track_id="base"),
            {"action": "add_track", "track_id": "top", "track_type": "video", "duck_under_speech": ducked},
            {"action": "add_clip", "track_id": "top", "clip_id": "pip", "asset_id": tone,
             "source_range": {"start": 0, "end": 5}, "timeline_in": 0.5,
             "layout": {"place": "top_left", "size": "small"}},
        ], width=320, height=240)
        out = tmp_path / f"ducked_{ducked}.mp4"
        render(project, out, loudness_target=None, talking=said)
        return out

    plain, ducked = with_inset(False), with_inset(True)
    # Measured in the inset's own band.
    before = level(ducked, 1.5, 1.0, freq=800)
    under = level(ducked, 4.0, 1.0, freq=800)
    assert abs(before - level(plain, 1.5, 1.0, freq=800)) < 1.0
    # The same drop as music under the talking.
    assert before - under == pytest.approx(DUCK_DEPTH_DB, abs=1.0)
    assert abs(level(plain, 4.0, 1.0, freq=800) - before) < 1.0

# --- the rest of the timeline ------------------------------------------------------------------

def test_a_layout_on_the_base_track_is_refused(sources: dict) -> None:
    # The base track is not drawn on top of anything, so a box there would be silently ignored.
    with pytest.raises(ValueError, match="above the base one"):
        build_project([
            video_track("base"),
            insert("b", sources["base"], 0, 3, track_id="base", layout={"place": "top_left", "size": "small"}),
        ], width=640, height=360)

def test_the_layout_can_be_changed_and_cleared_afterwards(inset: str) -> None:
    edit(inset, [{"action": "set_clip_look", "track_id": "top", "clip_id": "pip",
                  "layout": {"place": "top_left", "size": "large"}}])
    assert clips_of(inset, "top")[0]["layout"] == {"place": "top_left", "size": "large"}
    edit(inset, [{"action": "set_clip_look", "track_id": "top", "clip_id": "pip", "clear_layout": True}])
    assert clips_of(inset, "top")[0]["layout"] is None

def test_the_storyboard_shows_the_sequence_and_names_the_insets(inset: str) -> None:
    lines = preview_project(inset).content[0].text.splitlines()
    # The inset must not be mixed in as if it were a shot in the sequence.
    assert all("clip pip" not in line for line in lines if line.startswith("#"))
    assert any(line.startswith("-- track top: clip pip") and "inset in the bottom-right corner, small" in line
               for line in lines)

def test_the_storyboard_says_when_the_tiles_are_not_what_will_be_seen(inset: str) -> None:
    # Without a layout the clip fills the frame, so it replaces the tiles rather than
    # sitting in a corner of them. A storyboard of a B-roll pass that did not say so
    # would look exactly like a pass that never happened.
    edit(inset, [{"action": "set_clip_look", "track_id": "top", "clip_id": "pip", "clear_layout": True}])
    lines = preview_project(inset).content[0].text.splitlines()
    covering = next(line for line in lines if line.startswith("-- track top: clip pip"))
    assert "covering picture, over the whole frame" in covering
    assert "the sequence underneath" in covering

def test_splitting_an_inset_keeps_the_box_on_both_halves(inset: str) -> None:
    edit(inset, [{"action": "split_clip", "track_id": "top", "clip_id": "pip", "new_clip_id": "pip2", "at": 3}])
    assert [clip["layout"]["place"] for clip in clips_of(inset, "top")] == ["bottom_right", "bottom_right"]

def test_a_clip_drawn_over_the_sequence_plays_at_its_speed(media: Path, tmp_path: Path) -> None:
    """It used to be read at its speed and played at normal speed, showing half of what it read."""
    changing = tmp_path / "red_then_green.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:r=30:d=1", "-f", "lavfi", "-i", "color=c=green:s=320x240:r=30:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(changing),
    ], check=True, capture_output=True)
    over = import_asset(str(changing))["id"]
    base = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([
        video_track(), insert("a", base, 0, 2),
        video_track("broll"), {"action": "add_clip", "track_id": "broll", "clip_id": "b", "asset_id": over,
                               "source_range": {"start": 0, "end": 2}, "timeline_in": 0, "speed": 2.0},
    ], width=320, height=240)
    out = tmp_path / "fast_over.mp4"
    render(project, out, loudness_target=None)
    # Three quarters of a second in at double speed is a second and a half into the source.
    red, green, _ = extract_frame(str(out), 0.75).resize((1, 1)).getpixel((0, 0))
    assert green > red
