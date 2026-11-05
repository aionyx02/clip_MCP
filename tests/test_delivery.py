"""Tests for what a cut needs on its way out: chapters, covers, a last check, and export.

Most of it is arithmetic on the project and the analyses, checked directly.
The rest goes through the tools, and two renders check that what is decided
reaches the file: chapters inside the MP4, and a render refused until the
user has said to go ahead.
"""

import json
import subprocess
import xml.etree.ElementTree as ElementTree
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image as PILImage

from app.engine.delivery import (
    YOUTUBE_MIN_CHAPTER_SECONDS,
    chapter_metadata,
    chapters,
    check_delivery,
    clock,
    cover_candidates,
    frames_around,
    half_sentences,
    light_steps,
    look_candidates,
    run_on_cuts,
)
from app.engine.interchange import left_behind, write_edl, write_fcpxml, write_otio, write_srt
from app.engine.subtitles import build_ass, caption_overflow
from app.models.media import Asset, FaceMeasurement, MediaAnalysis, ShotMeasurement, SoundMeasurement, Span
from app.models.timeline.parts import (
    CaptionStyle, Clip, ClipLayout, Dissolve, Marker, PlacedCue, Project, SpeakerMark, TimeRange, Track, TrackType,
)
from app.runtime import repo
from app.tools.catalog import import_asset
from app.tools.delivery import export_cover, export_timeline, get_chapters
from app.tools.rendering import check_render, render_project
from helpers import build_project, edit, finished, insert, video_track

def clip(clip_id: str, asset_id: str, start: float, end: float, at: float, **fields) -> Clip:
    """Build a clip.

    Args:
        clip_id: Its ID.
        asset_id: Its file.
        start: In point, in seconds.
        end: Out point.
        at: Where it sits on the timeline.
        **fields: Anything else.

    Returns:
        The clip.
    """
    return Clip(id=clip_id, asset_id=asset_id, timeline_in=Decimal(str(at)),
                source_range=TimeRange(start=Decimal(str(start)), end=Decimal(str(end))), **fields)

def project_of(*clips: Clip, markers=(), over=(), music=(), **fields) -> Project:
    """Build a project around a sequence.

    Args:
        *clips: The sequence.
        markers: `(time, name)` per part.
        over: Clips on a track above the sequence.
        music: Clips on an audio track.
        **fields: Project settings.

    Returns:
        The project.
    """
    tracks = [Track(id="main", track_type=TrackType.VIDEO, clips=list(clips))]
    if over:
        tracks.append(Track(id="broll", track_type=TrackType.VIDEO, clips=list(over)))
    if music:
        tracks.append(Track(id="music", track_type=TrackType.AUDIO, clips=list(music)))
    return Project(id="p", name="測試", tracks=tracks, markers=[
        Marker(id=f"m{index}", name=name, timeline_in=Decimal(str(at))) for index, (at, name) in enumerate(markers)
    ], **fields)

# --- chapters ----------------------------------------------------------------------------

def test_the_parts_of_a_cut_are_its_chapters() -> None:
    project = project_of(clip("a", "x", 0, 60, 0), markers=[(0, "開場"), (15, "中段"), (40, "結尾")])
    listed, problems = chapters(project)
    assert listed == [(0.0, 15.0, "開場"), (15.0, 40.0, "中段"), (40.0, 60.0, "結尾")]
    assert problems == []

def test_what_youtube_would_refuse_is_said_rather_than_changed() -> None:
    project = project_of(clip("a", "x", 0, 30, 0), markers=[(2, "開場"), (25, "結尾")])
    listed, problems = chapters(project)
    assert [name for _, _, name in listed] == ["開場", "結尾"]
    assert any("0:00" in problem for problem in problems)
    assert any("at least 3" in problem for problem in problems)
    assert any(f"{YOUTUBE_MIN_CHAPTER_SECONDS:g}s" in problem for problem in problems)

def test_chapter_names_cannot_break_the_file_they_are_carried_in() -> None:
    text = chapter_metadata([(0.0, 10.0, "a=b; #c")])
    assert "title=a\\=b\\; \\#c" in text
    assert "START=0" in text and "END=10000" in text

def test_the_description_lines_are_ready_to_paste(media: Path) -> None:
    asset = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 10)], width=640, height=360)
    edit(project, [{"action": "set_markers", "markers": [
        {"id": "m1", "name": "開場", "timeline_in": 0}, {"id": "m2", "name": "結尾", "timeline_in": 5},
    ]}])
    result = get_chapters(project)
    assert result["description"] == "0:00 開場\n0:05 結尾"

# --- the check before a render -----------------------------------------------------------

def black_at(start: float, end: float, asset_id: str = "x") -> MediaAnalysis:
    """An analysis with a stretch of black in it.

    Args:
        start: Where the black starts.
        end: Where it ends.
        asset_id: The file.

    Returns:
        The analysis.
    """
    return MediaAnalysis(asset_id=asset_id, duration=60.0, black_frames=[Span(start=start, end=end)])

def test_black_that_reaches_the_screen_is_found_where_it_lands() -> None:
    project = project_of(clip("a", "x", 10, 20, 0))
    findings = check_delivery(project, {"x": black_at(15, 16)})
    assert [finding.check for finding in findings] == ["bad_picture"]
    assert "0:05" in findings[0].message and "clip a" in findings[0].message

def test_black_the_cut_does_not_use_is_nobody_s_business() -> None:
    assert check_delivery(project_of(clip("a", "x", 10, 20, 0)), {"x": black_at(30, 31)}) == []

def test_black_under_full_frame_covering_picture_is_not_seen() -> None:
    project = project_of(clip("a", "x", 10, 20, 0), over=[clip("b", "y", 0, 3, 4, volume=0.0)])
    assert check_delivery(project, {"x": black_at(15, 16)}) == []
    # An inset does not hide what is under the rest of the frame.
    inset = clip("b", "y", 0, 3, 4, volume=0.0, layout=ClipLayout(x=0.6, y=0.6, width=0.3, height=0.3))
    assert check_delivery(project_of(clip("a", "x", 10, 20, 0), over=[inset]), {"x": black_at(15, 16)})

def test_a_voice_recorded_clipping_is_found() -> None:
    clipped = MediaAnalysis(asset_id="x", duration=60.0, sound=[
        SoundMeasurement(start=12.0, end=13.0, loudness=-3.0, peak=0.0, noise_floor=-60.0, flatness=20.0),
        SoundMeasurement(start=13.0, end=14.0, loudness=-3.0, peak=-0.2, noise_floor=-60.0, flatness=0.0),
    ])
    findings = check_delivery(project_of(clip("a", "x", 10, 20, 0)), {"x": clipped})
    assert [finding.check for finding in findings] == ["clipping"]
    assert "1 second(s)" in findings[0].message and "0:02" in findings[0].message

def test_a_clip_turned_all_the_way_down_cannot_be_heard_clipping() -> None:
    clipped = MediaAnalysis(asset_id="x", duration=60.0, sound=[
        SoundMeasurement(start=12.0, end=13.0, loudness=-3.0, peak=0.0, noise_floor=-60.0, flatness=20.0),
    ])
    assert check_delivery(project_of(clip("a", "x", 10, 20, 0, volume=0.0)), {"x": clipped}) == []

def test_a_cut_far_from_its_target_length_is_found() -> None:
    project = project_of(clip("a", "x", 0, 20, 0))
    assert [finding.check for finding in check_delivery(project, {}, target_seconds=60.0)] == ["length"]
    assert check_delivery(project, {}, target_seconds=21.0) == []

def placed(text: str, **fields) -> PlacedCue:
    """A caption on screen from one second to three.

    Args:
        text: What it says.
        **fields: Anything else.

    Returns:
        The caption.
    """
    return PlacedCue(cue_id="c", start=Decimal(1), end=Decimal(3), text=text, **fields)

def test_a_caption_too_tall_for_the_frame_is_found() -> None:
    # Only a caption allowed to stack can climb out of the frame.
    style = CaptionStyle(size_fraction=1 / 6, single_line=False)
    long = placed("這是一句非常非常長的字幕" * 8)
    assert caption_overflow([long], 1080, 1920, CaptionStyle(size_fraction=1 / 6)) == []
    assert caption_overflow([long], 1080, 1920, style) == [long]
    assert caption_overflow([placed("短")], 1080, 1920, style) == []
    findings = check_delivery(project_of(clip("a", "x", 0, 10, 0), width=1080, height=1920), {}, captions=[long],
                              style=style)
    assert [finding.check for finding in findings] == ["captions"]

def test_a_speaker_s_name_is_measured_with_the_line_it_goes_in_front_of() -> None:
    """Chinese has no spaces to break at: a name stuck on after the line was measured
    pushes the line off the side of the picture."""
    from app.engine.subtitles import _display_width, caption_geometry

    style = CaptionStyle(speaker_mark=SpeakerMark.NAME, speaker_names={"V1": "陳大文先生"})
    geometry = caption_geometry(1080, 1920, style)
    text = build_ass([placed("字" * 60, speaker="V1")], 1080, 1920, style).splitlines()[-1].split(",,")[-1]
    for line in text.split("\\N"):
        assert sum(_display_width(character) for character in line) <= geometry.max_units

def test_a_replaced_preview_says_which_render_took_its_place(media: Path) -> None:
    """The old job still read completed, with nothing to say its file was gone (#195)."""
    from app.runtime import job_manager
    from app.tools.job_queue import get_job
    from app.tools.rendering import finished_render

    project = build_project([video_track(), insert("a", import_asset(str(media / "wide.mp4"))["id"], 0, 2)],
                            width=640, height=360)
    first, second = render_project(project), None
    finished(first)
    assert "file_gone" not in get_job([first["job_id"]])["jobs"][0]
    second = render_project(project, loudness_target=-20.0)
    finished(second)

    # The second preview is what took the first one's file away.
    assert not Path(job_manager.get_job(first["job_id"]).output_path).exists()
    said = get_job([first["job_id"]])["jobs"][0]
    assert said["status"] == "completed" and second["job_id"] in said["file_gone"]
    with pytest.raises(ValueError, match=second["job_id"]):
        finished_render(first["job_id"])


def test_a_preview_in_another_shape_does_not_stand_in_for_one_that_was_replaced(media: Path) -> None:
    from app.runtime import job_manager
    from app.tools.job_queue import get_job

    project = build_project([video_track(), insert("a", import_asset(str(media / "wide.mp4"))["id"], 0, 2)],
                            width=640, height=360)
    job = render_project(project)
    finished(job)
    other_shape = render_project(project, frame="portrait")
    finished(other_shape)

    assert not Path(job_manager.get_job(job["job_id"]).output_path).exists()
    said = get_job([job["job_id"]])["jobs"][0]["file_gone"]
    assert other_shape["job_id"] not in said and "render again" in said


def test_a_replacement_made_from_a_later_edit_says_so(media: Path) -> None:
    from app.tools.job_queue import get_job

    project = build_project([video_track(), insert("a", import_asset(str(media / "wide.mp4"))["id"], 0, 2)],
                            width=640, height=360)
    job = render_project(project)
    finished(job)
    edit(project, [{"action": "set_clip_audio", "track_id": "main", "clip_id": "a", "volume": 0.5}])
    later = render_project(project)
    finished(later)

    said = get_job([job["job_id"]])["jobs"][0]["file_gone"]
    assert later["job_id"] in said and "later edit" in said


def test_a_render_with_nothing_after_it_is_told_to_render_again(media: Path) -> None:
    from app.runtime import job_manager
    from app.tools.job_queue import get_job

    project = build_project([video_track(), insert("a", import_asset(str(media / "wide.mp4"))["id"], 0, 2)],
                            width=640, height=360)
    job = render_project(project)
    finished(job)
    Path(job_manager.get_job(job["job_id"]).output_path).unlink()

    assert "render again" in get_job([job["job_id"]])["jobs"][0]["file_gone"]


def test_render_is_refused_until_the_user_says_to_go_ahead(media: Path) -> None:
    asset = import_asset(str(media / "black.mp4"))["id"]
    repo.save_analysis(MediaAnalysis(asset_id=asset, duration=8.0, black_frames=[Span(start=0.0, end=8.0)]))
    project = build_project([video_track(), insert("a", asset, 0, 4)], width=640, height=360)
    assert check_render(project)["findings"][0]["check"] == "bad_picture"
    # A preview is how you look, so it is never refused; it says what it found.
    preview = render_project(project)
    assert preview["findings"][0]["check"] == "bad_picture"
    finished(preview)
    with pytest.raises(ValueError, match="allow="):
        render_project(project, is_preview=False)
    assert render_project(project, is_preview=False, allow=["bad_picture"])["findings"][0]["check"] == "bad_picture"

def test_a_final_render_waits_for_a_preview_of_the_cut_as_it_is_now(media: Path) -> None:
    asset = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2)], width=640, height=360)
    anything = ["bad_picture", "clipping", "music", "mid_speech", "repeated", "look_alike", "continuity", "unplanned",
                "captions", "length"]
    with pytest.raises(ValueError, match="no preview of the cut as it is now has finished"):
        render_project(project, is_preview=False, allow=anything)
    finished(render_project(project))
    with pytest.raises(ValueError, match="no preview of the cut as it is now, portrait, has finished"):
        render_project(project, is_preview=False, frame="portrait", allow=anything)
    edit(project, [{"action": "rename_project", "name": "改過"}])
    with pytest.raises(ValueError, match="no preview"):
        render_project(project, is_preview=False, allow=anything)
    finished(render_project(project))
    assert finished(render_project(project, is_preview=False, allow=anything))["status"] == "completed"

def test_the_chapters_are_carried_inside_the_rendered_file(media: Path, tmp_path: Path) -> None:
    from app.engine.builder import FFmpegRenderer
    from app.tools.shared import referenced_assets

    asset = import_asset(str(media / "wide.mp4"))["id"]
    project_id = build_project([video_track(), insert("a", asset, 0, 6)], width=320, height=180)
    edit(project_id, [{"action": "set_markers", "markers": [
        {"id": "m1", "name": "開場", "timeline_in": 0}, {"id": "m2", "name": "結尾", "timeline_in": 3},
    ]}])
    project = repo.get_project(project_id)
    listing = tmp_path / "chapters.txt"
    listing.write_text(chapter_metadata(chapters(project)[0]), encoding="utf-8")
    out = tmp_path / "out.mp4"
    command = FFmpegRenderer().build_command(project, referenced_assets(project), str(out), chapters_path=str(listing),
                                            loudness_target=None)
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")[-600:]
    probed = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_chapters", str(out)], capture_output=True,
    ).stdout)
    assert [(round(float(item["start_time"])), item["tags"]["title"]) for item in probed["chapters"]] == [
        (0, "開場"), (3, "結尾"),
    ]

# --- covers ------------------------------------------------------------------------------

def test_a_cover_candidate_is_the_second_with_the_largest_face() -> None:
    faces = [FaceMeasurement(start=float(second), end=float(second + 1), faces=1, face_share=share, face_x=0.5,
                             face_y=0.5) for second, share in enumerate([0.02, 0.2, 0.05, 0.01])]
    offered = cover_candidates(project_of(clip("a", "x", 0, 4, 0)), {"x": MediaAnalysis(
        asset_id="x", duration=4.0, faces=faces)})
    assert [(item.id, seconds) for item, seconds in offered] == [("a", 1.5)]

def test_without_a_face_the_sharpest_stretch_is_offered_and_never_black() -> None:
    analysis = MediaAnalysis(asset_id="x", duration=10.0, black_frames=[Span(start=0.0, end=5.0)], shots=[
        ShotMeasurement(start=0.0, end=5.0, exposure=0.0, contrast=0.0, blur=1.0),
        ShotMeasurement(start=5.0, end=10.0, exposure=0.5, contrast=0.5, blur=6.0),
    ])
    offered = cover_candidates(project_of(clip("a", "x", 0, 10, 0)), {"x": analysis})
    # The sharpest shot is black, so the next one is offered instead.
    assert offered == [(offered[0][0], 7.5)]

def test_a_cover_is_saved_at_the_size_the_cut_is_rendered_at(media: Path) -> None:
    asset = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 6)], width=640, height=360)
    saved = export_cover(project, 2.0, "test", frame="portrait").structured_content
    with PILImage.open(saved["output_path"]) as picture:
        assert picture.size == (360, 640) == (saved["width"], saved["height"])

def test_a_cover_comes_back_at_feed_size_and_as_a_profile_grid_crops_it(media: Path) -> None:
    import base64
    import io

    asset = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 6)], width=1080, height=1920)
    shown = export_cover(project, 2.0, "the product, large and sharp").content[1]
    with PILImage.open(io.BytesIO(base64.b64decode(shown.data))) as preview:
        # 360 wide in the feed, and beside it the 3:4 middle at 360 by 480.
        assert preview.size == (360 + 6 + 360, 640)

def test_more_than_one_moment_of_a_shot_can_be_offered_and_never_black() -> None:
    analysis = MediaAnalysis(asset_id="x", duration=10.0, black_frames=[Span(start=0.0, end=3.0)])
    offered = cover_candidates(project_of(clip("a", "x", 0, 10, 0)), {"x": analysis}, per_shot=3)
    assert len(offered) == 3 and all(seconds >= 3.0 for _, seconds in offered)

def test_a_close_look_shows_what_is_on_screen_either_side_of_one_moment() -> None:
    project = project_of(clip("a", "x", 0, 4, 0), clip("b", "y", 10, 14, 4))
    found = frames_around(project, {}, 4.0, reach=0.5, step=0.25)
    assert [(item.id, source, at) for item, source, at in found] == [
        ("a", 3.5, 3.5), ("a", 3.75, 3.75), ("b", 10.0, 4.0), ("b", 10.25, 4.25), ("b", 10.5, 4.5),
    ]

def test_covers_are_offered_one_part_at_a_time_with_what_the_video_is_for(media: Path) -> None:
    from app.tools.delivery import propose_covers

    asset = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3), insert("b", asset, 4, 8)], width=640, height=360)
    edit(project, [{"action": "set_markers", "markers": [{"id": "m1", "name": "開場", "timeline_in": 0},
                                                         {"id": "m2", "name": "結果", "timeline_in": 3}]}])
    listing = propose_covers(project, part="結果", per_shot=2).content[0].text
    assert "ask the user what it is about" in listing and "clip b" in listing and "clip a" not in listing
    with pytest.raises(ValueError, match="no part called '中段'"):
        propose_covers(project, part="中段")

# --- export ------------------------------------------------------------------------------

ASSETS = {
    "x": Asset(id="x", path="/footage/台北.mp4", duration=Decimal(60), has_video=True, has_audio=True),
    "y": Asset(id="y", path="/footage/broll.mp4", duration=Decimal(20), has_video=True, has_audio=False),
    "m": Asset(id="m", path="/footage/song.mp3", duration=Decimal(120), has_video=False, has_audio=True),
}

def exported() -> Project:
    """A cut with a gap, covering picture, music and two parts, at 30fps.

    Returns:
        The project.
    """
    return project_of(
        clip("p1", "x", 1, 3, 0), clip("p2", "x", 8, 10.5, 2), clip("p3", "x", 20, 25, 6),
        over=[clip("b1", "y", 0, 1.5, 3, volume=0.0)],
        music=[clip("m1", "m", 0, 11, 0)],
        markers=[(0, "開場"), (6, "結尾")],
    )

def test_an_edl_lists_the_sequence_in_timecode() -> None:
    text, behind = write_edl(exported(), ASSETS)
    events = [line for line in text.splitlines() if line[:3].isdigit()]
    assert events[0].split()[-4:] == ["00:00:01:00", "00:00:03:00", "00:00:00:00", "00:00:02:00"]
    # A gap is where the record timecode jumps, not an event.
    assert events[2].split()[-2] == "00:00:06:00"
    assert "* FROM CLIP NAME: 台北.mp4" in text
    assert "* LOC: 00:00:06:00 WHITE 結尾" in text
    assert any("one picture track" in line for line in behind)

def test_otio_carries_every_track_and_the_parts() -> None:
    timeline = json.loads(write_otio(exported(), ASSETS)[0])
    tracks = timeline["tracks"]["children"]
    assert [(track["name"], track["kind"]) for track in tracks] == [
        ("V1", "Video"), ("V2", "Video"), ("A1", "Audio"), ("A2", "Audio"),
    ]
    kinds = [item["OTIO_SCHEMA"] for item in tracks[0]["children"]]
    assert kinds == ["Clip.2", "Clip.2", "Gap.1", "Clip.2"]
    first = tracks[0]["children"][0]
    assert first["source_range"]["start_time"]["value"] == 30.0
    assert first["media_references"]["DEFAULT_MEDIA"]["target_url"].startswith("file:")
    assert [marker["name"] for marker in timeline["tracks"]["markers"]] == ["開場", "結尾"]

def test_fcpxml_hangs_everything_off_the_spine_where_it_plays() -> None:
    root = ElementTree.fromstring(write_fcpxml(exported(), ASSETS)[0].split("\n", 2)[2])
    spine = root.find(".//spine")
    assert [element.tag for element in spine] == ["asset-clip", "asset-clip", "gap", "asset-clip"]
    second = spine[1]
    assert (second.get("offset"), second.get("start"), second.get("duration")) == ("2s", "8s", "5/2s")
    # The covering picture starts three seconds in, one second into the second clip,
    # whose own time starts at eight: so it sits at nine in that clip's time.
    broll = second.find("asset-clip[@lane='1']")
    assert broll.get("offset") == "9s"
    music = spine[0].find("asset-clip[@lane='-1']")
    assert music is not None and music.get("duration") == "11s"
    assert [marker.get("value") for marker in root.iter("marker")] == ["開場", "結尾"]

def test_an_export_says_what_it_could_not_carry() -> None:
    project = project_of(
        clip("p1", "x", 1, 3, 0, speed=2.0),
        clip("p2", "x", 8, 10, 1, transition_in=Dissolve(kind="dissolve", seconds=Decimal("0.5"))),
    )
    behind = left_behind(project)
    assert any(line.startswith("speed changes") and "p1" in line for line in behind)
    assert any(line.startswith("transitions") and "p2" in line for line in behind)
    assert left_behind(project_of(clip("p1", "x", 1, 3, 0))) == []

def test_captions_go_out_as_srt() -> None:
    text = write_srt([placed("你好", secondary="hello")])
    assert text == "1\n00:00:01,000 --> 00:00:03,000\n你好\nhello\n"

def test_the_export_tool_writes_the_file(media: Path) -> None:
    asset = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 6)], width=640, height=360)
    written = export_timeline(project, "fcpxml")
    assert Path(written["output_path"]).suffix == ".fcpxml"
    assert "<spine>" in Path(written["output_path"]).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="no captions"):
        export_timeline(project, "srt")

def with_speed_and_a_dissolve() -> Project:
    """A cut with a clip at double speed and one that dissolves in, and quieter music.

    Returns:
        The project.
    """
    return project_of(
        clip("p1", "x", 1, 3, 0),
        clip("p2", "x", 10, 14, 2, speed=2.0),
        clip("p3", "x", 20, 23, 4, transition_in=Dissolve(kind="dissolve", seconds=Decimal("0.5"))),
        music=[clip("m1", "m", 0, 7, 0, volume=0.5)],
        markers=[(0, "開場"), (4, "結尾")],
    )

def test_an_edl_carries_speed_and_places_what_follows_by_the_time_it_plays() -> None:
    text, behind = write_edl(with_speed_and_a_dissolve(), ASSETS)
    events = [line.split() for line in text.splitlines() if line[:3].isdigit()]
    # Four seconds of source at double speed is two on the timeline, so the next event starts at 4.
    assert events[1][-4:] == ["00:00:10:00", "00:00:14:00", "00:00:02:00", "00:00:04:00"]
    assert events[2][-2] == "00:00:04:00"
    assert "M2   AX       060.0                00:00:10:00" in text
    assert not any(line.startswith("speed changes") for line in behind)

def test_otio_carries_speed_transitions_and_the_files_own_timecode() -> None:
    timeline = json.loads(write_otio(with_speed_and_a_dissolve(), ASSETS, {"x": 36000.0})[0])
    items = timeline["tracks"]["children"][0]["children"]
    assert [item["OTIO_SCHEMA"] for item in items] == ["Clip.2", "Clip.2", "Transition.1", "Clip.2"]
    fast = items[1]
    assert fast["effects"][0]["time_scalar"] == 2.0
    # Its range is where it starts in the file and how long it runs on the track.
    assert (fast["source_range"]["start_time"]["value"], fast["source_range"]["duration"]["value"]) == (36010 * 30, 60)
    dissolve = items[2]
    assert dissolve["transition_type"] == "SMPTE_Dissolve"
    # Ours ends on the cut, so all of it overlaps the clip before.
    assert (dissolve["in_offset"]["value"], dissolve["out_offset"]["value"]) == (15, 0)

def test_otio_says_what_a_wipe_was_when_it_cannot_say_it_as_one() -> None:
    from app.models.timeline.parts import Wipe

    project = project_of(clip("p1", "x", 1, 3, 0), clip("p2", "x", 10, 12, 2, transition_in=Wipe(
        kind="wipe", seconds=Decimal("0.6"), direction="left")))
    text, behind = write_otio(project, ASSETS)
    wipe = json.loads(text)["tracks"]["children"][0]["children"][1]
    assert wipe["transition_type"] == "Custom_Transition"
    assert wipe["metadata"]["clip_mcp"]["direction"] == "left"
    assert any(line.startswith("wipes and dips") for line in behind)

def test_fcpxml_carries_speed_level_and_timecode() -> None:
    text, behind = write_fcpxml(with_speed_and_a_dissolve(), ASSETS, {"x": 36000.0})
    root = ElementTree.fromstring(text.split("\n", 2)[2])
    assert {asset.get("id"): asset.get("start") for asset in root.iter("asset")}["r3"] == "36000s"
    spine = root.find(".//spine")
    fast = spine[1]
    assert (fast.get("offset"), fast.get("start"), fast.get("duration")) == ("2s", "36010s", "2s")
    points = [(point.get("time"), point.get("value")) for point in fast.find("timeMap")]
    assert points == [("36010s", "36010s"), ("36012s", "36014s")]
    music = spine[0].find("asset-clip[@lane='-1']")
    assert music.find("adjust-volume").get("amount") == "-6.0dB"
    assert not any(line.startswith(("speed changes", "volume")) for line in behind)
    assert any(line.startswith("transitions") for line in behind)

@pytest.mark.parametrize("written, rate, expected", [
    ("10:00:00:00", "25", 36000.0),
    ("01:00:00;02", "30000/1001", 107894 * 1001 / 30000),
    (None, "30", None),
], ids=["non-drop", "drop-frame", "none"])
def test_a_files_own_timecode_is_read(tmp_path: Path, written, rate: str, expected) -> None:
    from app.engine.probe import probe_file, timecode_start

    path = tmp_path / "stamped.mov"
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate={rate}:duration=1",
               "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    subprocess.run(command + (["-timecode", written] if written else []) + [str(path)], check=True)
    found = timecode_start(probe_file(str(path)))
    assert found == (None if expected is None else pytest.approx(expected, abs=1e-6))

# --- cuts that ignore what is said -----------------------------------------------------

def talking(asset_id: str = "x") -> MediaAnalysis:
    """Analyze a ten-second file of two sentences, with a real pause inside the first.

    The first runs 1–5s: 今天 天氣 (a half-second pause) 很好. The second runs 6–9s.

    Args:
        asset_id: The file.

    Returns:
        The analysis.
    """
    from app.models.media import Transcript, TranscriptSegment, TranscriptWord

    def said(*words):
        return [TranscriptWord(text=text, start=start, end=end) for text, start, end in words]

    return MediaAnalysis(asset_id=asset_id, duration=10.0, transcript=Transcript(language="zh", model="t", segments=[
        TranscriptSegment(start=1.0, end=5.0, text="今天天氣很好",
                          words=said(("今天", 1.0, 2.0), ("天氣", 2.05, 3.0), ("很好", 3.5, 5.0))),
        TranscriptSegment(start=6.0, end=9.0, text="我們出發", words=said(("我們", 6.0, 7.0), ("出發", 7.05, 9.0))),
    ]))

def checks(project: Project, analyses: dict) -> list:
    """Run the check before a render and keep only what it found.

    Args:
        project: The cut.
        analyses: Its analyses.

    Returns:
        `(check, message)` per finding.
    """
    return [(finding.check, finding.message) for finding in check_delivery(project, analyses)]

@pytest.mark.parametrize("out_point", [1.5, 2.02, 7.02])
def test_a_cut_in_the_middle_of_somebody_talking_is_found(out_point: float) -> None:
    found = checks(project_of(clip("a", "x", 0, out_point, 0)), {"x": talking()})
    assert [check for check, _ in found] == ["mid_speech"]
    assert "ends in the middle" in found[0][1]

@pytest.mark.parametrize("out_point", [3.25, 5.5, 9.5])
def test_a_cut_in_a_pause_or_between_sentences_is_clean(out_point: float) -> None:
    assert checks(project_of(clip("a", "x", 0, out_point, 0)), {"x": talking()}) == []

def test_the_nearest_pause_is_offered_instead() -> None:
    found = checks(project_of(clip("a", "x", 0, 2.5, 0)), {"x": talking()})
    assert "nearest pause is at 3.25s" in found[0][1]

def test_where_the_camera_started_is_not_a_cut_anybody_made() -> None:
    analysis = talking()
    analysis.transcript.segments[0].start = 0.0
    analysis.transcript.segments[0].words[0].start = 0.0
    assert checks(project_of(clip("a", "x", 0, 5.5, 0)), {"x": analysis}) == []

def test_a_muted_clip_is_not_cut_in_the_middle_of_anything() -> None:
    assert checks(project_of(clip("a", "x", 0, 1.5, 0, volume=0.0)), {"x": talking()}) == []

def test_a_cut_that_takes_the_edge_off_a_measured_sound_is_found_without_a_transcript() -> None:
    # Nobody transcribed it, but the silences say a sound rises at 4.0s and falls at 6.0s.
    heard = MediaAnalysis(asset_id="x", duration=10.0, silences=[Span(start=0.0, end=4.0), Span(start=6.0, end=10.0)])
    found = checks(project_of(clip("a", "x", 4.3, 9.0, 0)), {"x": heard})
    assert [check for check, _ in found] == ["mid_speech"] and "Start at 4.00s" in found[0][1]
    found = checks(project_of(clip("a", "x", 1.0, 5.8, 0)), {"x": heard})
    assert "End at 6.00s" in found[0][1]
    # In a silence, or in the middle of a sound that runs on, nothing is cut off at its edge.
    assert checks(project_of(clip("a", "x", 3.5, 6.5, 0)), {"x": heard}) == []
    assert checks(project_of(clip("a", "x", 5.0, 9.0, 0)), {"x": heard}) == []

def test_a_sound_running_into_black_picture_is_not_advice_to_cut_there() -> None:
    """A hand over the lens: told to keep the sound, the cut then showed the black (#192)."""
    heard = MediaAnalysis(asset_id="x", duration=10.0, silences=[Span(start=0.0, end=4.0), Span(start=6.0, end=10.0)],
                          black_frames=[Span(start=5.92, end=10.0)])

    found = checks(project_of(clip("a", "x", 4.0, 5.9, 0)), {"x": heard})

    assert [check for check, _ in found] == ["mid_speech"]
    assert "End at" not in found[0][1] and "black or frozen from 5.92s" in found[0][1]
    moved_as_told = checks(project_of(clip("a", "x", 4.0, 6.0, 0)), {"x": heard})
    assert "bad_picture" in [check for check, _ in moved_as_told]


def test_a_sound_at_the_start_behind_frozen_picture_is_not_advice_to_cut_there() -> None:
    heard = MediaAnalysis(asset_id="x", duration=10.0, silences=[Span(start=0.0, end=4.0), Span(start=6.0, end=10.0)],
                          frozen_frames=[Span(start=3.0, end=4.2)])

    found = checks(project_of(clip("a", "x", 4.3, 9.0, 0)), {"x": heard})

    assert "Start at" not in found[0][1] and "black or frozen until 4.20s" in found[0][1]


def test_the_same_shot_shown_twice_is_found() -> None:
    project = project_of(clip("a", "y", 0, 36, 0), clip("b", "y", 2, 16, 36))
    found = checks(project, {})
    assert [check for check, _ in found] == ["repeated"]
    assert "14.0s of the same file" in found[0][1]

def test_neighbouring_moments_of_one_file_are_not_a_repeat() -> None:
    assert checks(project_of(clip("a", "y", 0, 10, 0), clip("b", "y", 10.5, 20, 10)), {}) == []

def test_an_edit_put_together_by_hand_is_found() -> None:
    hand = project_of(clip("a", "y", 0, 5, 0), clip("b", "z", 0, 5, 5), clip("c", "w", 0, 5, 10))
    assert [check for check, _ in checks(hand, {})] == ["unplanned"]
    planned = project_of(*(item.model_copy(update={"from_plan_id": "plan"}) for item in hand.base_video_track.clips))
    assert checks(planned, {}) == []

def test_a_line_the_recogniser_wrote_over_silence_is_not_cut_into() -> None:
    analysis = talking()
    analysis.silences = [Span(start=0.5, end=5.5)]
    assert checks(project_of(clip("a", "x", 0, 1.5, 0)), {"x": analysis}) == []

def test_a_line_the_recogniser_made_up_over_other_sound_is_not_speech() -> None:
    from app.engine.semantic import was_made_up

    # 「3D列印、炒麵」 over a drone taking off: sound, but every word a guess near zero.
    analysis = talking()
    for word in analysis.transcript.segments[0].words:
        word.probability = 0.02
    assert was_made_up(analysis.transcript.segments[0])
    assert checks(project_of(clip("a", "x", 0, 1.5, 0)), {"x": analysis}) == []
    # One word heard clearly is enough for it to have been said.
    analysis.transcript.segments[0].words[1].probability = 0.9
    assert not was_made_up(analysis.transcript.segments[0])
    # A transcript made before probabilities were kept is never judged.
    assert not was_made_up(talking().transcript.segments[0])

def test_a_cut_in_a_real_silence_is_clean_whatever_the_word_timings_say() -> None:
    # The transcript runs 天氣 on to 很好 with 0.05s between; the microphone heard half a
    # second of nothing there.
    analysis = talking()
    analysis.silences = [Span(start=2.1, end=2.6)]
    analysis.transcript.segments[0].words[2].start = 2.05
    assert checks(project_of(clip("a", "x", 0, 2.4, 0)), {"x": analysis}) == []

def running_on(*lines: str, start: float = 1.0) -> MediaAnalysis:
    """Analyze talking that never stops: one long transcribed sentence, a character a word.

    The shape Whisper gives continuous Chinese: half a minute in one segment, a
    word per character laid end to end, and as often as not no punctuation.
    Each character takes 0.2s, the next one starting as it ends.

    Args:
        *lines: What is said, in order; a line may carry its own punctuation,
            which stays on the character before it as the transcriber puts it.
        start: Where the talking starts.

    Returns:
        The analysis.
    """
    from app.models.media import Transcript, TranscriptSegment, TranscriptWord

    words, at = [], start
    for character in "".join(lines):
        if character in "。，！？":
            words[-1].text += character
            continue
        words.append(TranscriptWord(text=character, start=round(at, 2), end=round(at + 0.2, 2)))
        at += 0.2
    text = "".join(word.text for word in words)
    return MediaAnalysis(asset_id="x", duration=at + 1.0, transcript=Transcript(language="zh", model="t", segments=[
        TranscriptSegment(start=start, end=words[-1].end, text=text, words=words)]))

def end_of(analysis: MediaAnalysis, said: str) -> float:
    """Where a stretch of the talking ends, in seconds of the file."""
    text = "".join(word.text.rstrip("。，！？") for word in analysis.transcript.segments[0].words)
    return analysis.transcript.segments[0].words[text.index(said) + len(said) - 1].end

@pytest.mark.parametrize("said, sentence", [
    ("今天的進度差不多結案了吧", "那我們明天再來處理剩下的部分"),
    ("這趟真的收穫滿滿啊", "掰掰下次見"),
])
def test_a_cut_at_a_sentence_end_with_no_pause_after_it_is_not_mid_speech(said: str, sentence: str) -> None:
    """#230: a sentence ending on a particle, the next running straight on, was refused for no pause."""
    analysis = running_on(said, sentence)
    at = end_of(analysis, said)
    ends = project_of(clip("a", "x", 0, at, 0))
    assert checks(ends, {"x": analysis}) == []
    [note] = run_on_cuts(ends.base_video_track.clips, {"x": analysis})
    assert f"ends at 0:0{int(at)}" in note and f"「{said[-8:]}」 ends a sentence" in note
    # A clip opening on the next sentence there takes whole words off the front, as it always could.
    starts = project_of(clip("a", "x", at, analysis.duration - 0.5, 0))
    assert checks(starts, {"x": analysis}) == []

def test_a_sentence_ended_by_its_punctuation_is_an_end_whatever_its_last_word() -> None:
    analysis = running_on("我們今天就先拍到這裡。", "下禮拜再來")
    at = end_of(analysis, "這裡")
    assert checks(project_of(clip("a", "x", 0, at, 0)), {"x": analysis}) == []

@pytest.mark.parametrize("said, sentence, cut_after", [
    # Nothing marks an end: 了 sits inside a sentence as often as at its end.
    ("差不多結案了", "吧那我們明天再來", "結案了"),
    # A comma parts a sentence without ending it.
    ("我們先去吃飯，", "然後再去看電影", "吃飯"),
    # 吧 the end of a word, not a particle.
    ("我們晚上去酒吧", "喝一杯好不好", "酒吧"),
    ("先坐在吧", "台那邊等一下", "坐在吧"),
])
def test_a_cut_between_words_with_no_sentence_end_marked_is_still_mid_speech(
        said: str, sentence: str, cut_after: str) -> None:
    analysis = running_on(said, sentence)
    project = project_of(clip("a", "x", 0, end_of(analysis, cut_after), 0))
    found = checks(project, {"x": analysis})
    assert [check for check, _ in found] == ["mid_speech"]
    assert "marks no sentence end between these words" in found[0][1] and "`listen_again`" in found[0][1]
    assert run_on_cuts(project.base_video_track.clips, {"x": analysis}) == []

def test_a_cut_inside_the_particle_that_ends_a_sentence_is_still_mid_speech() -> None:
    analysis = running_on("今天的進度差不多結案了吧", "那我們明天再來處理")
    at = end_of(analysis, "結案了吧") - 0.1
    found = checks(project_of(clip("a", "x", 0, at, 0)), {"x": analysis})
    assert [check for check, _ in found] == ["mid_speech"]
    assert "marks no sentence end" not in found[0][1]

def test_a_sentence_end_does_not_clear_a_word_the_second_listen_heard_cut_off() -> None:
    from app.models.media import HeardWord, SecondListen

    # The transcript ends a sentence on 吧, but the second listen, which never read it,
    # heard 那我們 run on from it, and the transcript has no word for that stretch.
    analysis = running_on("今天的進度差不多結案了吧", "那我們明天再來處理")
    at = end_of(analysis, "結案了吧")
    analysis.transcript.segments[0].words = [
        word for word in analysis.transcript.segments[0].words if word.end <= at + 0.01 or word.start >= at + 0.6]
    analysis.second_listen = SecondListen(model="t", voiced=[Span(start=1.0, end=at + 0.4)], words=[
        HeardWord(text="吧", start=at - 0.2, end=at, probability=0.9),
        HeardWord(text="那我們", start=at + 0.02, end=at + 0.4, probability=0.9)])
    found = checks(project_of(clip("a", "x", 0, at, 0)), {"x": analysis})
    assert [check for check, _ in found] == ["mid_speech"]
    # The second listen is the reason; the transcript does end a sentence here, so it is not said that it does not.
    assert "heard on a second listen" in found[0][1] and "marks no sentence end" not in found[0][1]

def test_a_cut_in_talking_with_no_word_timings_does_not_say_what_the_words_mark() -> None:
    analysis = running_on("今天的進度差不多結案了", "那我們明天再來處理")
    at = end_of(analysis, "結案了")
    for segment in analysis.transcript.segments:
        segment.words = []
    found = checks(project_of(clip("a", "x", 0, at, 0)), {"x": analysis})
    assert [check for check, _ in found] == ["mid_speech"]
    assert "marks no sentence end" not in found[0][1]

def spoken(*parts) -> MediaAnalysis:
    """Analyze talking with pauses in it, split into sentences the way a transcription is.

    A character a word, 0.2s each, as Whisper gives spoken Chinese; a number
    between two lines is a pause of that many seconds. The sentences are cut
    by `split_sentences`, the transcriber's own rule: at 。！？ or a pause of
    `SENTENCE_PAUSE_SECONDS` or more, so a shorter pause stays inside one.

    Args:
        *parts: Lines and pauses, in order.

    Returns:
        The analysis.
    """
    from app.engine.analysis import split_sentences
    from app.models.media import Transcript, TranscriptWord

    words, at = [], 1.0
    for part in parts:
        if not isinstance(part, str):
            at += part
            continue
        for character in part:
            if character in "。，！？":
                words[-1].text += character
                continue
            words.append(TranscriptWord(text=character, start=round(at, 2), end=round(at + 0.2, 2)))
            at += 0.2
    return MediaAnalysis(asset_id="x", duration=round(at + 1.0, 2), transcript=Transcript(
        language="zh", model="t", segments=split_sentences(words)))

def said_until(analysis: MediaAnalysis, said: str) -> float:
    """Where the first time something is said ends, in seconds of the file."""
    words = [word for segment in analysis.transcript.segments for word in segment.words]
    text = "".join(word.text.rstrip("。，！？") for word in words)
    return words[text.index(said) + len(said) - 1].end

def test_a_clip_that_stops_on_a_word_leading_into_more_is_a_half_sentence() -> None:
    # #238, as transcribed: the speaker trailed off on a drawn-out 「大概」 and paused, so the
    # transcript ends the sentence there and the cut is clean, but the line says half a thing.
    analysis = spoken("各位這是現在的樣子然後就是堪用，但還可以更好可以看一下它的機構這樣抽下來大概", 1.9,
                      "就這樣那好，另外一個稍等一下")
    end = said_until(analysis, "抽下來大概")
    opened = said_until(analysis, "可以看一下")
    project = project_of(clip("a", "x", 1.0, end, 0), clip("b", "y", 0, 3, end - 1.0))
    assert checks(project, {"x": analysis}) == []
    caption = PlacedCue(cue_id="c4", start=Decimal(str(round(opened - 1.0, 2))), end=Decimal(str(round(end - 1.0, 2))),
                        text="它的機構這樣抽下來大概")
    [note] = half_sentences(project.base_video_track.clips, {"x": analysis}, [caption])
    assert note.startswith(f"clip a ends at {clock(end - 1.0)} ({end:.2f}s of the file)")
    assert "caption 「它的機構這樣抽下來大概」" in note and "stops on 「大概」" in note
    # Nothing earlier in the clip ends a sentence — no 。！？, no closing particle, no pause the
    # transcriber ended one at — and that is said rather than left out.
    assert "No sentence ends earlier in this clip" in note
    # The end of the video is the same as a cut away from the speaker.
    alone = project_of(clip("a", "x", 1.0, end, 0))
    assert len(half_sentences(alone.base_video_track.clips, {"x": analysis})) == 1

def test_a_clip_that_starts_partway_through_the_word_it_stops_on_names_that_word() -> None:
    # The clip keeps only 「概」 of 「大概」: the sentence still stops on 大概, and saying so
    # must not depend on how much of it the clip kept.
    analysis = spoken("可以看一下它的機構這樣抽下來大概", 1.9, "就這樣")
    end = said_until(analysis, "大概")
    project = project_of(clip("a", "x", end - 0.2, end, 0))
    [note] = half_sentences(project.base_video_track.clips, {"x": analysis})
    assert "stops on 「大概」" in note

def test_a_clip_that_stops_where_the_transcribed_sentence_goes_on_is_a_half_sentence() -> None:
    # A breath inside one sentence is a clean place to cut the sound, but the sentence goes on.
    analysis = spoken("早安今天天氣很好。", 0.8, "我們今天要去", 0.3, "台南吃牛肉湯", 1.0, "出發")
    end = said_until(analysis, "要去")
    project = project_of(clip("a", "x", 1.0, end + 0.15, 0), clip("b", "y", 0, 3, end - 0.85))
    assert checks(project, {"x": analysis}) == []
    [note] = half_sentences(project.base_video_track.clips, {"x": analysis})
    assert "the words heard last, 「我們今天要去」" in note and "「台南吃牛肉湯」" in note
    # Where it could end instead: the sentence before, in the file's own seconds, or where this one ends.
    assert f"after 「早安今天天氣很好」 at {said_until(analysis, '天氣很好'):.2f}s of the file" in note
    assert f"runs on to {said_until(analysis, '牛肉湯'):.2f}s" in note

@pytest.mark.parametrize("parts, cut_after", [
    # Marked by its punctuation, with only a breath before the next.
    (("我們今天就先拍到這裡。", 0.3, "下禮拜再來"), "這裡"),
    # Marked by a particle that closes it.
    (("差不多結案了吧", 0.3, "那我們明天再來"), "結案了吧"),
    # Ended by the transcriber at a pause, on a word that says all it needs to.
    (("今天天氣很好", 1.0, "我們出門吧"), "很好"),
])
def test_a_clip_that_stops_at_a_sentence_end_is_not_a_half_sentence(parts, cut_after: str) -> None:
    analysis = spoken(*parts)
    end = said_until(analysis, cut_after)
    project = project_of(clip("a", "x", 1.0, end + 0.1, 0), clip("b", "y", 0, 3, end - 0.9))
    assert half_sentences(project.base_video_track.clips, {"x": analysis}) == []

def test_a_cut_the_same_talking_picks_up_after_is_not_a_half_sentence() -> None:
    analysis = spoken("可以看一下它的機構這樣抽下來大概", 1.9, "就是這樣一抽就出來了", 0.3, "很順")
    end, resumes = said_until(analysis, "大概"), said_until(analysis, "大概") + 1.9
    # The pause taken out of the middle: what comes next is what was said next.
    tightened = project_of(clip("a", "x", 1.0, end, 0), clip("b", "x", resumes, resumes + 4.0, end - 1.0))
    assert half_sentences(tightened.base_video_track.clips, {"x": analysis}) == []
    # A clip split at a breath, both halves kept.
    breath = said_until(analysis, "一抽就出來了")
    halves = breath + 0.15 - resumes
    split = project_of(clip("a", "x", resumes, breath + 0.15, 0), clip("b", "x", breath + 0.15, breath + 1.5, halves))
    assert half_sentences(split.base_video_track.clips, {"x": analysis}) == []

def test_a_cut_inside_a_word_is_left_to_mid_speech() -> None:
    analysis = spoken("我們今天要去台南吃牛肉湯", 1.0, "出發")
    end = said_until(analysis, "台南") - 0.1
    project = project_of(clip("a", "x", 1.0, end, 0))
    assert [check for check, _ in checks(project, {"x": analysis})] == ["mid_speech"]
    assert half_sentences(project.base_video_track.clips, {"x": analysis}) == []

# What a vlogger says, sentence by sentence, as a transcription of continuous Chinese hands it
# back: no punctuation, a breath inside a sentence now and then, a pause between them.
VLOG = (
    "早安各位", 0.8, "今天要帶大家去台南", 0.3, "吃一間很有名的牛肉湯", 0.9, "我四點就起床了", 0.7,
    "外面還是暗的", 1.1, "好我們到了", 0.8, "排隊的人真的很多", 0.3, "大概排了半小時", 0.9,
    "這碗是招牌", 0.7, "湯頭很清", 0.3, "肉也很嫩", 1.2, "老闆說每天現切", 0.8, "所以要早點來", 0.9,
    "接下來去市場", 0.7, "買一些水果回家", 1.0, "芒果現在正好吃", 0.8, "一斤一百", 0.9, "我買了兩斤",
    0.7, "今天收穫滿滿", 1.0, "我們下次見",
)

def test_on_continuous_talking_only_a_cut_inside_a_sentence_is_a_half_sentence() -> None:
    """The rate this fires at on every cut a vlog's sound lets through, as each clip's end.

    None of the sentences ends on anything that marks an end, so a rule that
    asked for 。！？ or a particle would call all 16 cuts between them half a
    sentence. This one goes by where the transcriber ended the sentence: it is
    quiet on all 16, and fires on the 3 breaths inside one, of which
    「排隊的人真的很多 / 大概排了半小時」 and 「湯頭很清 / 肉也很嫩」 could be
    heard as two sentences. That is the price: 2 wrong notes in 19 cuts here.
    """
    analysis = spoken(*VLOG)
    segments = analysis.transcript.segments
    breaths = [(before.end + after.start) / 2 for segment in segments
               for before, after in zip(segment.words, segment.words[1:]) if after.start - before.end >= 0.2]
    ends = [segment.end + 0.1 for segment in segments[:-1]]
    unmarked = [segment for segment in segments if not segment.text.endswith(tuple("。！？吧啊"))]
    assert len(segments) == 17 and len(unmarked) == 17 and len(breaths) == 3
    fired = {at for at in [*breaths, *ends]
             if half_sentences(project_of(clip("a", "x", 0.5, at, 0)).base_video_track.clips, {"x": analysis})}
    # Every breath inside a sentence, which leaves the rest of it out; no sentence end at all.
    assert fired == set(breaths)

def test_a_cut_that_takes_off_a_start_the_transcript_missed_is_found() -> None:
    # The second sentence really starts at 5.6, where the silence ends; the transcript says
    # 6.0. Starting the clip at 6.0 takes the first syllable off.
    analysis = talking()
    analysis.silences = [Span(start=5.0, end=5.6)]
    found = checks(project_of(clip("a", "x", 6.0, 9.5, 0)), {"x": analysis})
    assert [check for check, _ in found] == ["mid_speech"] and "starts in the middle" in found[0][1]

def test_a_clip_from_the_afternoon_between_two_from_the_morning_is_found() -> None:
    from datetime import datetime, timezone

    project = project_of(clip("a", "m", 0, 4, 0), clip("b", "p", 0, 4, 4), clip("c", "m", 10, 14, 8))
    recorded = {"m": datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc),
                "p": datetime(2026, 8, 31, 6, 0, tzinfo=timezone.utc)}
    found = [finding for finding in check_delivery(project, {}, recorded=recorded) if finding.check == "continuity"]
    assert len(found) == 1 and "clip b" in found[0].message
    # Where a new part begins, the jump is the part beginning.
    marked = project_of(clip("a", "m", 0, 4, 0), clip("b", "p", 0, 4, 4), clip("c", "m", 10, 14, 8),
                        markers=[(4, "下午")])
    assert not [f for f in check_delivery(marked, {}, recorded=recorded) if f.check == "continuity"]

# --- captions long enough to read --------------------------------------------------------

def cue(cue_id: str, start: float, end: float, text: str = "晚安"):
    """A caption on file `x`, as stored."""
    from app.models.timeline.parts import SubtitleCue

    return SubtitleCue(id=cue_id, asset_id="x", source_start=start, source_end=end, text=text)

def test_a_short_caption_is_held_on_screen_while_its_shot_has_room() -> None:
    from app.engine.subtitles import place_cues

    placed = place_cues(project_of(clip("a", "x", 10, 20, 0)), [cue("c1", 12.0, 12.36)])
    assert float(placed[0].end - placed[0].start) == pytest.approx(0.7)

def test_a_short_caption_is_held_no_further_than_the_next_one() -> None:
    from app.engine.subtitles import place_cues

    placed = place_cues(project_of(clip("a", "x", 10, 20, 0)), [cue("c1", 12.0, 12.3), cue("c2", 12.5, 13.5, "下一句")])
    assert float(placed[0].end) == pytest.approx(2.5)

def test_a_caption_that_cannot_be_held_long_enough_is_found_with_the_ways_out() -> None:
    from app.engine.subtitles import place_cues

    # The last words of the video: the cut ends 0.36s after they start, so there is no room.
    project = project_of(clip("a", "x", 10, 12.36, 0))
    placed = place_cues(project, [cue("c1", 12.0, 12.36)])
    findings = check_delivery(project, {}, captions=placed, style=CaptionStyle())
    short = [finding for finding in findings if finding.check == "captions"]
    assert len(short) == 1
    assert "0.36s" in short[0].message and "0.7s" in short[0].message and "min_seconds" in short[0].message

def test_a_caption_held_short_by_the_next_one_is_found_with_that_way_out() -> None:
    from app.engine.subtitles import place_cues

    project = project_of(clip("a", "x", 10, 20, 0))
    placed = place_cues(project, [cue("c1", 12.0, 12.3), cue("c2", 12.3, 13.5, "下一句")])
    found = [f for f in check_delivery(project, {}, captions=placed, style=CaptionStyle(min_seconds=0.1))
             if f.check == "captions"]
    # Measured against the minimum it was placed with, the project's own — not the render's style.
    assert len(found) == 1 and "next caption" in found[0].message and "shot ends" not in found[0].message

def test_how_long_a_caption_has_to_stay_is_the_users_to_change() -> None:
    from app.engine.subtitles import place_cues

    project = project_of(clip("a", "x", 10, 20, 0), caption_style=CaptionStyle(min_seconds=1.2))
    placed = place_cues(project, [cue("c1", 12.0, 12.36)])
    assert float(placed[0].end - placed[0].start) == pytest.approx(1.2)
    relaxed = project_of(clip("a", "x", 10, 12.36, 0), caption_style=CaptionStyle(min_seconds=0.3))
    placed = place_cues(relaxed, [cue("c1", 12.0, 12.36)])
    assert not [f for f in check_delivery(relaxed, {}, captions=placed, style=relaxed.caption_style)
                if f.check == "captions"]


def test_check_render_checks_the_captions_without_being_told(media: Path) -> None:
    """The render refused what the check had passed, because the check left captions out unless asked (#191)."""
    asset = import_asset(str(media / "wide.mp4"))["id"]
    project_id = build_project([video_track(), insert("a", asset, 0, 4)], width=640, height=360)
    project = repo.get_project(project_id)
    words = cue("c1", 3.64, 4.0).model_copy(update={"asset_id": asset})
    repo.update_project(project.model_copy(update={"subtitles": [words]}), project.version)

    assert [finding["check"] for finding in check_render(project_id)["findings"]] == ["captions"]
    assert check_render(project_id, burn_subtitles=False)["ok"]
    finished(render_project(project_id, burn_subtitles=True))
    with pytest.raises(ValueError, match="under the 0.7s"):
        render_project(project_id, is_preview=False, burn_subtitles=True)


def test_two_files_that_look_alike_apart_in_the_cut_are_found() -> None:
    project = project_of(clip("a", "x", 0, 4, 0), clip("b", "y", 0, 4, 4), clip("c", "z", 0, 4, 8),
                         clip("d", "x", 5, 9, 12))
    same, other = (0x0F0F0F0F0F0F0F0F,) * 3, (0xF0F0F0F0F0F0F0F0,) * 3
    def alike(looks: dict) -> list:
        return [finding.message for finding in check_delivery(project, {}, looks=looks)
                if finding.check == "look_alike"]

    # a and c are different files with b between, and so are b and d; a and b play side by side,
    # and a and d are one file.
    found = alike({"a": same, "b": same, "c": same, "d": same})
    assert len(found) == 2 and found[0].startswith("clips a (at 0:00) and c (at 0:08)")
    assert found[1].startswith("clips b (at 0:04) and d (at 0:12)")
    assert alike({}) == [] and alike({"a": same, "c": other}) == []

    # Marked off by a transition into it, the second is told apart; so would a card just before it be.
    from app.models.timeline.parts import Dissolve
    project.tracks[0].clips[2].transition_in = Dissolve(kind="dissolve", seconds=Decimal("0.5"))
    assert [message[:15] for message in alike({"a": same, "b": same, "c": same, "d": same})] == ["clips b (at 0:0"]

def test_only_a_clip_nothing_is_said_or_written_over_is_told_apart_by_its_look() -> None:
    from app.models.media import Transcript, TranscriptSegment
    from app.models.timeline.parts import TextLayer

    spoken = MediaAnalysis(asset_id="x", duration=10.0, transcript=Transcript(language="zh", model="test", segments=[
        TranscriptSegment(start=1.0, end=2.0, text="成功了")]))
    words = clip("c", "z", 0, 4, 8)
    words.texts.append(TextLayer(id="t1", text="第二次", start=0, end=4))
    project = project_of(clip("a", "x", 0, 4, 0), clip("b", "y", 0, 4, 4), words)
    assert [found.id for found in look_candidates(project, {"x": spoken})] == ["b"]

def test_a_render_is_checked_for_two_files_that_look_alike(media: Path, tmp_path: Path) -> None:
    from app.engine import looks

    # The same picture at half the size is another file that looks the same.
    small = tmp_path / "small.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(media / "wide.mp4"), "-vf", "scale=320:180",
                    "-an", "-c:v", "libx264", "-preset", "ultrafast", str(small)], check=True)
    first, second = import_asset(str(media / "wide.mp4"))["id"], import_asset(str(small))["id"]
    black = import_asset(str(media / "black.mp4"))["id"]
    project = build_project([video_track(), insert("a", first, 0, 2), insert("b", black, 0, 2),
                             insert("c", second, 0, 2)], width=640, height=360)
    found = [finding for finding in check_render(project)["findings"] if finding["check"] == "look_alike"]
    assert len(found) == 1 and "clips a" in found[0]["message"] and "and c" in found[0]["message"]
    # Read once: the frames are kept against their files.
    assert set(repo.looks(second)) == {round(2 * share, 1) for share in looks.SAMPLES}


def test_a_shot_darker_than_the_ones_either_side_is_offered_the_brightness_that_evens_it_out() -> None:
    from app.models.timeline.parts import ColorAdjust

    def lit(asset_id: str, exposure: float, warmth=None) -> MediaAnalysis:
        return MediaAnalysis(asset_id=asset_id, duration=10.0, shots=[
            ShotMeasurement(start=0, end=10, exposure=exposure, contrast=0.5, warmth=warmth)])

    analyses = {"x": lit("x", 0.5), "y": lit("y", 0.2), "z": lit("z", 0.52)}
    project = project_of(clip("a", "x", 0, 4, 0), clip("b", "y", 0, 4, 4), clip("c", "z", 0, 4, 8))
    found = light_steps(project, analyses)
    assert len(found) == 2 and all("set_clip_look on clip b with color brightness +0.31" in line for line in found)
    # Set, it is measured again and nothing is left.
    project.tracks[0].clips[1].color = ColorAdjust(brightness=0.31)
    assert light_steps(project, analyses) == []
    # A new part, or a dip, is a change made on purpose.
    project.tracks[0].clips[1].color = None
    assert light_steps(project_of(clip("a", "x", 0, 4, 0), clip("b", "y", 0, 4, 4), markers=[(0, "一"), (4, "二")]),
                       analyses) == []
    # A part beginning inside a shot puts the shots either side of it in different parts.
    assert light_steps(project_of(clip("a", "x", 0, 4, 0), clip("b", "y", 0, 4, 4), markers=[(0, "一"), (2, "二")]),
                       analyses) == []
    # A colour cast is said with the way to set it.
    warm = {"x": lit("x", 0.5, warmth=0.0), "y": lit("y", 0.5, warmth=0.06)}
    [step] = light_steps(project_of(clip("a", "x", 0, 4, 0), clip("b", "y", 0, 4, 4)), warm)
    assert "clip b is warmer by 0.06; a color temperature above 6500 on it cools it" in step
    warmed = project_of(clip("a", "x", 0, 4, 0), clip("b", "y", 0, 4, 4))
    warmed.tracks[0].clips[1].color = ColorAdjust(temperature=7500)
    assert light_steps(warmed, warm) == []


def test_checking_a_project_with_nothing_on_it_is_not_an_all_clear() -> None:
    """#96: a project whose compile was refused checked out `ok` with nothing in it."""
    from app.tools.editing import create_project
    from app.tools.rendering import check_render

    project = create_project(name="空的", width=640, height=360)["id"]
    with pytest.raises(ValueError, match="nothing on its sequence to check"):
        check_render(project)
