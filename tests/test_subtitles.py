"""Tests for proposing captions from transcripts and burning them in."""

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.engine.frames import extract_frame
from app.engine.subtitles import build_ass, escape_ass_text, format_ass_time, wrap_caption
from app.models.media import MediaAnalysis, Span, Transcript, TranscriptSegment, TranscriptWord
from app.models.timeline import SubtitleCue
from app.server import generate_subtitles, get_project, get_subtitles, import_asset, render_project, repo
from helpers import build_project, edit, insert, render, video_track

def words(*pairs) -> list:
    """Build transcript words from `(text, start, end)` triples.

    Args:
        *pairs: One triple per word.

    Returns:
        The words as model objects.
    """
    return [TranscriptWord(text=text, start=start, end=end) for text, start, end in pairs]

def transcribe(asset_id: str, segments: list) -> None:
    """Store a transcript for an asset as if `analyze_asset` had run.

    Args:
        asset_id: Asset the transcript belongs to.
        segments: Transcript segments to store.
    """
    repo.save_analysis(MediaAnalysis(
        asset_id=asset_id,
        duration=10.0,
        transcript=Transcript(language="zh", model="test", segments=segments),
    ))

@pytest.fixture
def spoken(media: Path) -> str:
    """Import a video and give it a transcript covering its whole length.

    Returns:
        The asset ID.
    """
    asset = import_asset(str(media / "wide.mp4"))["id"]
    transcribe(asset, [
        TranscriptSegment(start=1.0, end=3.0, text="第一句",
                          words=words(("第一句", 1.0, 3.0))),
        TranscriptSegment(start=4.0, end=6.0, text="第二句",
                          words=words(("第二句", 4.0, 6.0))),
        TranscriptSegment(start=7.0, end=9.0, text="第三句",
                          words=words(("第三句", 7.0, 9.0))),
    ])
    return asset

def test_format_ass_time_uses_centiseconds() -> None:
    assert format_ass_time(0) == "0:00:00.00"
    assert format_ass_time(3723.456) == "1:02:03.46"

def test_escape_ass_text_neutralizes_braces_and_newlines() -> None:
    assert escape_ass_text("a {b}\nc") == "a (b)" + chr(92) + "Nc"

def test_build_ass_sizes_the_style_from_the_output_format() -> None:
    portrait = build_ass([], 1080, 1920)
    landscape = build_ass([], 1920, 1080)
    assert "PlayResX: 1080" in portrait and "PlayResY: 1920" in portrait
    # A phone covers the bottom of a vertical video, so its captions sit higher.
    portrait_margin = int(portrait.split("Style: Default,")[1].splitlines()[0].split(",")[-2])
    landscape_margin = int(landscape.split("Style: Default,")[1].splitlines()[0].split(",")[-2])
    assert portrait_margin > landscape_margin

def test_only_the_speech_that_survived_the_edit_is_captioned(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 3.5, 6.5)])
    cues = generate_subtitles(project)["cues"]
    assert [cue["text"] for cue in cues] == ["第二句"]

def test_captions_move_to_where_the_speech_lands_in_the_result(spoken: str) -> None:
    project = build_project([
        video_track(),
        insert("intro", spoken, 0, 2),
        insert("a", spoken, 3.5, 6.5),
    ])
    cue = next(item for item in generate_subtitles(project)["cues"] if item["text"] == "第二句")
    # The line was at 4.0 in a clip starting at 3.5, placed after a 2 s intro.
    assert float(cue["start"]) == pytest.approx(2.5)
    assert float(cue["end"]) == pytest.approx(4.5)

def test_captions_are_clamped_so_they_do_not_run_past_a_cut(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 2.0)])
    cue = generate_subtitles(project)["cues"][0]
    assert float(cue["end"]) == pytest.approx(2.0)

def test_reordering_the_clips_reorders_the_captions(spoken: str) -> None:
    project = build_project([
        video_track(),
        insert("first", spoken, 0.5, 3.5),
        insert("second", spoken, 6.5, 9.5),
    ])
    edit(project, [{"action": "reorder_clip", "track_id": "main", "clip_id": "second", "before_clip_id": "first"}])
    assert [cue["text"] for cue in generate_subtitles(project)["cues"]] == ["第三句", "第一句"]

def test_long_sentences_are_broken_between_words(media: Path) -> None:
    asset = import_asset(str(media / "silent.mp4"))["id"]
    transcribe(asset, [TranscriptSegment(
        start=0.0, end=4.0, text="one two three four",
        words=words(("one ", 0.0, 1.0), ("two ", 1.0, 2.0), ("three ", 2.0, 3.0), ("four", 3.0, 4.0)),
    )])
    project = build_project([video_track(), insert("a", asset, 0, 4)])
    cues = generate_subtitles(project, max_characters=8)["cues"]
    assert len(cues) > 1
    assert all(len(cue["text"]) <= 10 for cue in cues)
    assert "".join(cue["text"] for cue in cues).replace(" ", "") == "onetwothreefour"

def test_a_segment_without_word_timings_stays_in_one_piece(media: Path) -> None:
    asset = import_asset(str(media / "silent.mp4"))["id"]
    transcribe(asset, [TranscriptSegment(start=0.0, end=3.0, text="a whole sentence with no word timings")])
    project = build_project([video_track(), insert("a", asset, 0, 3)])
    cues = generate_subtitles(project, max_characters=8)["cues"]
    assert len(cues) == 1

def test_assets_without_a_transcript_are_reported(spoken: str, media: Path) -> None:
    other = import_asset(str(media / "tall.mp4"))["id"]
    project = build_project([video_track(), insert("a", spoken, 0, 3), insert("b", other, 0, 2)])
    assert generate_subtitles(project)["assets_without_transcript"] == [other]

def test_a_narration_on_an_audio_track_is_captioned(media: Path) -> None:
    # A voice-over recorded separately: the picture under it says nothing, and the
    # words the viewer actually hears used to come back with no captions at all.
    quiet = import_asset(str(media / "silent.mp4"))["id"]
    voice = import_asset(str(media / "song.mp3"))["id"]
    transcribe(voice, [
        TranscriptSegment(start=0.5, end=2.0, text="這一天從早上開始",
                          words=words(("這一天從早上開始", 0.5, 2.0))),
    ])
    project = build_project([
        video_track(), insert("shot", quiet, 0, 4),
        {"action": "add_track", "track_id": "voice", "track_type": "audio"},
        insert("v", voice, 0, 4, track_id="voice"),
    ])
    spoken_lines = [cue for cue in generate_subtitles(project)["cues"] if cue["text"] == "這一天從早上開始"]
    assert [(cue["start"], cue["end"]) for cue in spoken_lines] == [(Decimal("0.5"), Decimal("2.0"))]

def test_a_music_bed_is_not_reported_as_missing_a_transcript(spoken: str, media: Path) -> None:
    song = import_asset(str(media / "jingle.wav"))["id"]
    project = build_project([
        video_track(), insert("a", spoken, 0, 3),
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        insert("m", song, 0, 3, track_id="music"),
    ])
    # Nobody transcribes a song, so it is not a gap to go and fill.
    assert generate_subtitles(project)["assets_without_transcript"] == []

def test_a_clip_turned_all_the_way_down_is_not_captioned(spoken: str) -> None:
    # Silenced footage is a picture laid over someone else's sound, so it has no lines.
    project = build_project([video_track(), insert("a", spoken, 0, 4, volume=0)])
    assert generate_subtitles(project)["cues"] == []

def test_captions_that_talk_over_each_other_are_counted(spoken: str, media: Path) -> None:
    voice = import_asset(str(media / "song.mp3"))["id"]
    transcribe(voice, [
        TranscriptSegment(start=1.5, end=3.0, text="旁白蓋過去",
                          words=words(("旁白蓋過去", 1.5, 3.0))),
    ])
    project = build_project([
        video_track(), insert("a", spoken, 0, 4),
        {"action": "add_track", "track_id": "voice", "track_type": "audio"},
        insert("v", voice, 0, 4, track_id="voice"),
    ])
    result = generate_subtitles(project)
    # 第一句 runs 1.0 to 3.0 under a narration line starting at 1.5.
    assert result["overlapping"] == 1

def test_a_sentence_transcribed_over_silence_is_not_captioned(media: Path) -> None:
    # The same invention the semantic timeline refuses to call speech: it must not
    # reach the picture either, or it is burned in before anyone reads it.
    asset = import_asset(str(media / "muted.mp4"))["id"]
    repo.save_analysis(MediaAnalysis(
        asset_id=asset,
        duration=4.0,
        silences=[Span(start=1.8, end=4.0)],
        transcript=Transcript(language="zh", model="test", segments=[
            TranscriptSegment(start=0.0, end=1.5, text="真的講了這句",
                              words=words(("真的講了這句", 0.0, 1.5))),
            TranscriptSegment(start=2.0, end=3.8, text="中文字幕——YK",
                              words=words(("中文字幕——YK", 2.0, 3.8))),
        ]),
    ))
    project = build_project([video_track(), insert("a", asset, 0, 4)])
    assert [cue["text"] for cue in generate_subtitles(project)["cues"]] == ["真的講了這句"]

def test_a_project_with_no_transcripts_is_reported_clearly(media: Path) -> None:
    asset = import_asset(str(media / "tall.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2)])
    with pytest.raises(ValueError, match="no transcribed clips"):
        generate_subtitles(project)

def test_burning_without_stored_captions_is_refused(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 3)])
    with pytest.raises(ValueError, match="no captions to burn"):
        render_project(project, is_preview=True, burn_subtitles=True)

def test_captions_are_drawn_low_in_the_picture_when_they_are_due(media: Path, tmp_path: Path) -> None:
    asset = import_asset(str(media / "black.mp4"))["id"]
    transcribe(asset, [TranscriptSegment(
        start=1.0, end=3.0, text="字幕測試", words=words(("字幕測試", 1.0, 3.0)),
    )])
    project = build_project([video_track(), insert("a", asset, 0, 5)], width=640, height=360)
    edit(project, [{"action": "set_subtitles", "cues": generate_subtitles(project)["cues"]}])
    stored = repo.get_project(project)

    subtitle_file = tmp_path / "subs.ass"
    subtitle_file.write_text(build_ass(stored.subtitles, 640, 360), encoding="utf-8")
    render(project, tmp_path / "burned.mp4", loudness_target=None, subtitle_path=str(subtitle_file))

    def bright_pixels(seconds: float, top: float, bottom: float) -> int:
        """Count light pixels in a horizontal band of one frame."""
        frame = extract_frame(str(tmp_path / "burned.mp4"), seconds, max_size=640).convert("L")
        width, height = frame.size
        rows = range(int(height * top), int(height * bottom))
        return sum(1 for y in rows for x in range(width) if frame.getpixel((x, y)) > 160)

    # The source is black, so anything bright was drawn by libass.
    assert bright_pixels(2.0, 0.6, 1.0) > 150
    assert bright_pixels(2.0, 0.0, 0.5) == 0
    assert bright_pixels(4.0, 0.6, 1.0) == 0

@pytest.mark.parametrize(("text", "units", "expected"), [
    ("這是一句比較長的中文字幕", 6.0,
     ["這是一句比較", "長的中文字幕"]),
    ("the quick brown fox jumps", 8.0, ["the quick brown", "fox jumps"]),
    ("short", 20.0, ["short"]),
])
def test_wrap_caption_breaks_lines_to_fit(text: str, units: float, expected: list) -> None:
    assert wrap_caption(text, units) == expected

def test_wrap_caption_breaks_chinese_without_spaces(media: Path) -> None:
    # libass only breaks at spaces, so a Chinese caption has to be broken here or it runs off the frame.
    long_line = "字" * 40
    lines = wrap_caption(long_line, 12.0)
    assert len(lines) == 4 and all(len(line) <= 12 for line in lines)

def test_build_ass_wraps_long_captions_into_several_lines() -> None:
    cue = SubtitleCue(start=0, end=2, text="字" * 30)
    dialogue = [line for line in build_ass([cue], 1080, 1920).splitlines() if line.startswith("Dialogue")][0]
    assert dialogue.count(chr(92) + "N") >= 1

def test_captions_come_only_from_the_base_track(spoken: str, media: Path) -> None:
    # An inset is a picture over the sequence's sound, not a second voice to caption.
    inset = import_asset(str(media / "tall.mp4"))["id"]
    transcribe(inset, [TranscriptSegment(start=0.0, end=2.0, text="插入畫面", words=words(("插入畫面", 0.0, 2.0)))])
    project = build_project([
        video_track(),
        insert("a", spoken, 0, 10),
        {"action": "add_track", "track_id": "top", "track_type": "video"},
        {"action": "add_clip", "track_id": "top", "clip_id": "pip", "asset_id": inset,
         "source_range": {"start": 0, "end": 2}, "timeline_in": 0,
         "layout": {"x": 0.6, "y": 0.6, "width": 0.3, "height": 0.3}},
    ])
    result = generate_subtitles(project)
    assert all(cue["text"] != "插入畫面" for cue in result["cues"])
    assert inset not in result["assets_without_transcript"]

def test_captions_burn_from_a_path_with_awkward_characters(media: Path, tmp_path: Path) -> None:
    # An apostrophe, a bracket or a comma in the workspace path must not break the filter graph.
    asset = import_asset(str(media / "black.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3)], width=320, height=240)
    folder = tmp_path / "o'brien [take 1], final"
    folder.mkdir()
    subtitle_file = folder / "subs.ass"
    subtitle_file.write_text(build_ass([SubtitleCue(start=0, end=2, text="OK")], 320, 240), encoding="utf-8")
    render(project, tmp_path / "burned.mp4", loudness_target=None, subtitle_path=str(subtitle_file))
    assert (tmp_path / "burned.mp4").exists()

def test_generated_captions_carry_ids(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 10)])
    ids = [cue["id"] for cue in generate_subtitles(project)["cues"]]
    assert ids == ["c1", "c2", "c3"]

def test_one_caption_can_be_corrected_without_resending_the_rest(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 10)])
    edit(project, [{"action": "set_subtitles", "cues": generate_subtitles(project)["cues"]}])
    before = get_subtitles(project)["cues"]
    edit(project, [{"action": "edit_subtitle", "cue_id": "c2", "text": "改過的句子"}])
    after = get_subtitles(project)["cues"]
    assert [cue["text"] for cue in after] == [before[0]["text"], "改過的句子", before[2]["text"]]
    # The untouched captions keep their exact timing as well as their text.
    assert [(cue["start"], cue["end"]) for cue in after] == [(cue["start"], cue["end"]) for cue in before]

def test_a_caption_can_be_retimed_and_deleted(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 10)])
    edit(project, [{"action": "set_subtitles", "cues": generate_subtitles(project)["cues"]}])
    edit(project, [{"action": "edit_subtitle", "cue_id": "c1", "end": 3.8}])
    assert float(get_subtitles(project)["cues"][0]["end"]) == 3.8
    edit(project, [{"action": "edit_subtitle", "cue_id": "c2", "delete": True}])
    assert [cue["id"] for cue in get_subtitles(project)["cues"]] == ["c1", "c3"]

def test_editing_an_unknown_caption_is_reported_clearly(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 10)])
    edit(project, [{"action": "set_subtitles", "cues": generate_subtitles(project)["cues"]}])
    with pytest.raises(ValueError, match="caption c99 not found"):
        edit(project, [{"action": "edit_subtitle", "cue_id": "c99", "text": "x"}])

def test_a_caption_cannot_be_retimed_to_end_before_it_starts(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 10)])
    edit(project, [{"action": "set_subtitles", "cues": generate_subtitles(project)["cues"]}])
    with pytest.raises(ValueError, match="must end after it starts"):
        edit(project, [{"action": "edit_subtitle", "cue_id": "c1", "end": 0.5}])

def test_captions_can_be_read_a_window_at_a_time(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 10)])
    edit(project, [{"action": "set_subtitles", "cues": generate_subtitles(project)["cues"]}])
    window = get_subtitles(project, start=3.5, end=6.5)
    assert [cue["id"] for cue in window["cues"]] == ["c2"]
    assert window["total"] == 3

def test_get_project_leaves_the_captions_out_by_default(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 10)])
    edit(project, [{"action": "set_subtitles", "cues": generate_subtitles(project)["cues"]}])
    assert "subtitles" not in get_project(project)
    assert get_project(project)["subtitle_count"] == 3
    assert len(get_project(project, include_subtitles=True)["subtitles"]) == 3

def test_get_subtitles_shows_captions_stranded_past_the_end(spoken: str) -> None:
    # A later edit can leave a caption past the end of the video; it has to stay findable.
    project = build_project([video_track(), insert("a", spoken, 0, 10)])
    edit(project, [{"action": "set_subtitles", "cues": generate_subtitles(project)["cues"]}])
    edit(project, [{"action": "trim_clip", "track_id": "main", "clip_id": "a",
                    "new_source_range": {"start": 0, "end": 4}}])
    assert float(get_project(project)["duration"]) == 4.0
    visible = get_subtitles(project)
    assert len(visible["cues"]) == visible["total"] == 3

def test_get_subtitles_refuses_a_window_that_ends_before_it_starts(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 10)])
    with pytest.raises(ValueError, match="not after its start"):
        get_subtitles(project, start=5, end=2)

def test_a_caption_cannot_be_blanked_instead_of_deleted(spoken: str) -> None:
    project = build_project([video_track(), insert("a", spoken, 0, 10)])
    edit(project, [{"action": "set_subtitles", "cues": generate_subtitles(project)["cues"]}])
    with pytest.raises(ValidationError, match="set delete to remove it"):
        edit(project, [{"action": "edit_subtitle", "cue_id": "c1", "text": "   "}])
