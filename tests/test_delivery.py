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
    cover_candidates,
)
from app.engine.interchange import left_behind, write_edl, write_fcpxml, write_otio, write_srt
from app.engine.subtitles import build_ass, caption_overflow
from app.models.media import Asset, FaceMeasurement, MediaAnalysis, ShotMeasurement, SoundMeasurement, Span
from app.models.timeline import (
    CaptionStyle, Clip, ClipLayout, Dissolve, Marker, PlacedCue, Project, SpeakerMark, TimeRange, Track, TrackType,
)
from app.server import (
    check_render, export_cover, export_timeline, get_chapters, import_asset, render_project, repo,
)
from helpers import build_project, edit, insert, video_track

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

def test_render_is_refused_until_the_user_says_to_go_ahead(media: Path) -> None:
    asset = import_asset(str(media / "black.mp4"))["id"]
    repo.save_analysis(MediaAnalysis(asset_id=asset, duration=8.0, black_frames=[Span(start=0.0, end=8.0)]))
    project = build_project([video_track(), insert("a", asset, 0, 4)], width=640, height=360)
    assert check_render(project)["findings"][0]["check"] == "bad_picture"
    with pytest.raises(ValueError, match="allow="):
        render_project(project)
    # A preview is how you look, so it is never refused; it says what it found.
    assert render_project(project, is_preview=True)["findings"][0]["check"] == "bad_picture"
    assert render_project(project, allow=["bad_picture"])["findings"][0]["check"] == "bad_picture"

def test_the_chapters_are_carried_inside_the_rendered_file(media: Path, tmp_path: Path) -> None:
    from app.engine.builder import FFmpegRenderer
    from app.server import _referenced_assets

    asset = import_asset(str(media / "wide.mp4"))["id"]
    project_id = build_project([video_track(), insert("a", asset, 0, 6)], width=320, height=180)
    edit(project_id, [{"action": "set_markers", "markers": [
        {"id": "m1", "name": "開場", "timeline_in": 0}, {"id": "m2", "name": "結尾", "timeline_in": 3},
    ]}])
    project = repo.get_project(project_id)
    listing = tmp_path / "chapters.txt"
    listing.write_text(chapter_metadata(chapters(project)[0]), encoding="utf-8")
    out = tmp_path / "out.mp4"
    command = FFmpegRenderer().build_command(project, _referenced_assets(project), str(out), chapters_path=str(listing))
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
    saved = export_cover(project, 2.0, frame="portrait")
    with PILImage.open(saved["output_path"]) as picture:
        assert picture.size == (360, 640) == (saved["width"], saved["height"])

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
    from app.models.timeline import Wipe

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
