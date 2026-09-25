"""Tests for who is on screen and who is talking.

Both of these run a model, which makes them awkward to test honestly: a
detector that has been wired up wrongly finds nothing, and a test that only
ever shows it footage with nobody in it would pass either way. So both halves
have a positive case made out of nothing but code. The face tests draw a face
with Pillow, which YuNet does detect, and check that where it was drawn is
where it is reported; the speaker tests synthesize two voices with FFmpeg's
`flite` filter and check that the two are told apart at the right moments.

What is not tested here is how good the models are. That is their business,
and this suite has no ground truth to hold them to.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from PIL import Image, ImageDraw, ImageFilter

from app.engine import models
from app.engine.analysis import (
    CAMERA_FILE_NAME,
    DETECT_LOG_FILE_NAME,
    PICTURE_FILE_NAME,
    analyze_media,
    scan_media,
)
from app.engine.ffmpeg import OperationCancelled
from app.engine.diarize import find_speakers
from app.engine.faces import _per_second, detect_faces, framing_note, frames_dir
from app.engine.semantic import SPEAKER_MAJORITY, build_timeline, join_voices, speaker_at
from app.models.media import (
    Asset,
    FaceMeasurement,
    MediaAnalysis,
    SpeakerTurn,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    Voice,
)

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
# Voices far enough apart that telling them apart is a fair thing to ask.
VOICES = ("slt", "kal")
LINES = (
    "good morning everyone and welcome back to the show today",
    "thanks for having me it is really a pleasure to be here",
    "let us start with the question everybody keeps asking about",
    "that is a fair question and the answer is rather complicated",
)

def run(args: List[str]) -> None:
    """Run FFmpeg, failing the test with its error output.

    Args:
        args: Arguments after the executable name.
    """
    result = subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")

def draw_face(path: Path, width: int = 640, height: int = 360, cx: int = 320, face_h: int = 210) -> None:
    """Draw a face the detector recognises, at a known place in the frame.

    Crude as it is, YuNet finds this one, which is what makes it useful: the
    frame is built here, so where the face is is known exactly and the
    reported position can be held to it.

    Args:
        path: Image file to write.
        width: Frame width.
        height: Frame height.
        cx: Where to put the middle of the face, in pixels from the left.
        face_h: How tall to draw it.
    """
    image = Image.new("RGB", (width, height), (150, 155, 160))
    pen = ImageDraw.Draw(image)
    cy = height // 2
    face_w = face_h * 0.74
    left, top, right, bottom = cx - face_w / 2, cy - face_h / 2, cx + face_w / 2, cy + face_h / 2
    pen.ellipse([left - 10, top - 18, right + 10, top + face_h * 0.55], fill=(52, 40, 34))
    pen.rectangle([cx - face_w * 0.22, cy + face_h * 0.3, cx + face_w * 0.22, bottom + 40], fill=(198, 158, 128))
    pen.ellipse([left, top, right, bottom], fill=(224, 186, 156))
    pen.chord([left - 6, top - 10, right + 6, top + face_h * 0.5], 180, 360, fill=(52, 40, 34))
    eye_y, eye_dx = cy - face_h * 0.06, face_w * 0.21
    eye_w, eye_h = face_w * 0.17, face_h * 0.055
    for side in (-1, 1):
        ex = cx + side * eye_dx
        pen.ellipse([ex - eye_w, eye_y - eye_h, ex + eye_w, eye_y + eye_h], fill=(248, 246, 242))
        pen.ellipse([ex - eye_h * 0.9, eye_y - eye_h * 0.9, ex + eye_h * 0.9, eye_y + eye_h * 0.9], fill=(60, 45, 35))
        pen.ellipse([ex - eye_h * 0.4, eye_y - eye_h * 0.4, ex + eye_h * 0.4, eye_y + eye_h * 0.4], fill=(12, 10, 10))
        pen.arc([ex - eye_w * 1.1, eye_y - eye_h * 4.2, ex + eye_w * 1.1, eye_y - eye_h * 0.4], 200, 340,
                fill=(58, 44, 36), width=max(2, int(face_h * 0.017)))
    pen.polygon([(cx, eye_y + face_h * 0.04), (cx - face_w * 0.08, cy + face_h * 0.14),
                 (cx + face_w * 0.08, cy + face_h * 0.14)], fill=(206, 168, 138))
    pen.arc([cx - face_w * 0.19, cy + face_h * 0.17, cx + face_w * 0.19, cy + face_h * 0.30],
            0, 180, fill=(150, 88, 82), width=max(2, int(face_h * 0.022)))
    image.filter(ImageFilter.GaussianBlur(1.5)).save(path)

def scan(asset: Asset, work_dir: Path, **options) -> MediaAnalysis:
    """Analyze one asset.

    Args:
        asset: Asset to analyze.
        work_dir: Directory for the pass's files.
        **options: Passed on to `analyze_media`.

    Returns:
        The analysis.
    """
    return analyze_media(
        asset, str(work_dir), options.pop("transcribe", False), options.pop("language", None),
        None, None, lambda fraction, stage: None, lambda: False, FFMPEG, **options,
    )

def face_clip(tmp_path: Path, name: str, cx: int, seconds: int = 3) -> Asset:
    """Build a clip that holds one drawn face at a known place.

    Args:
        tmp_path: Directory to write to.
        name: File name for the clip.
        cx: Where the face sits, in pixels from the left of a 640-wide frame.
        seconds: How long the clip runs.

    Returns:
        The asset.
    """
    still = tmp_path / f"{name}.png"
    draw_face(still, cx=cx)
    path = tmp_path / name
    run(["-loop", "1", "-i", str(still), "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-t", str(seconds), "-r", "30", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)])
    return Asset(id=f"a-{name}", path=str(path), duration=seconds, has_video=True, has_audio=True)

@pytest.fixture(scope="module")
def face_model() -> None:
    """Fetch the face model once, skipping the tests that need it if it cannot be had."""
    try:
        models.ensure(models.FACE_DETECTION)
    except RuntimeError as error:
        pytest.skip(f"the face model could not be downloaded: {error}")

@pytest.fixture(scope="module")
def speaker_models() -> None:
    """Fetch the speaker models once, skipping the tests that need them if they cannot be had."""
    try:
        models.ensure(models.SPEAKER_SEGMENTATION)
        models.ensure(models.SPEAKER_EMBEDDING)
    except RuntimeError as error:
        pytest.skip(f"the speaker models could not be downloaded: {error}")

def has_flite() -> bool:
    """Say whether this FFmpeg can synthesize speech.

    Returns:
        True when the `flite` filter is built in. Without it there is no way
        to make two voices out of nothing, and the tests that need them say so
        rather than failing.
    """
    help_text = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "quiet", "-h", "filter=flite"],
        capture_output=True, text=True,
    ).stdout
    return "flite AVOptions" in help_text

@pytest.fixture(scope="module")
def conversation(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Asset, List[Tuple[str, float, float]]]:
    """Synthesize two voices taking four turns, and say where the turns really are.

    Returns:
        The asset, and `(voice, start, end)` for each turn as it was built.
    """
    if not has_flite():
        pytest.skip("this FFmpeg has no flite filter, so two voices cannot be synthesized")
    folder = tmp_path_factory.mktemp("conversation")
    parts, expected, cursor = [], [], 0.0
    for index, line in enumerate(LINES):
        voice = VOICES[index % len(VOICES)]
        part = folder / f"turn{index}.wav"
        # A little silence after each turn, so the handovers are where a handover would be.
        run(["-f", "lavfi", "-i", f"flite=text='{line}':voice={voice}",
             "-af", "apad=pad_dur=0.8", "-ar", "16000", "-ac", "1", str(part)])
        length = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(part)],
            capture_output=True, text=True,
        ).stdout)
        parts.append(part)
        expected.append((voice, round(cursor, 2), round(cursor + length, 2)))
        cursor += length

    listing = folder / "parts.txt"
    listing.write_text("\n".join(f"file '{part.as_posix()}'" for part in parts), encoding="utf-8")
    sound = folder / "talk.wav"
    run(["-f", "concat", "-safe", "0", "-i", str(listing), "-ar", "16000", "-ac", "1", str(sound)])
    path = folder / "talk.mp4"
    run(["-f", "lavfi", "-i", f"smptebars=size=320x180:rate=30:duration={cursor}", "-i", str(sound),
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)])
    return Asset(id="talk", path=str(path), duration=cursor, has_video=True, has_audio=True), expected

# --- faces ---------------------------------------------------------------------------------

def test_a_face_is_found_where_it_was_drawn(tmp_path: Path, face_model: None) -> None:
    """Wired up wrongly, a detector finds nothing and every negative test still passes."""
    analysis = scan(face_clip(tmp_path, "left.mp4", cx=160), tmp_path)
    assert analysis.faces, "no face records at all"
    assert all(record.faces == 1 for record in analysis.faces)
    # Drawn a quarter of the way across a 640-wide frame.
    assert all(record.face_x == pytest.approx(0.25, abs=0.05) for record in analysis.faces)
    assert all(record.face_y == pytest.approx(0.5, abs=0.1) for record in analysis.faces)
    assert all(0.0 < record.face_share < 0.5 for record in analysis.faces)

def test_moving_the_face_moves_the_measurement(tmp_path: Path, face_model: None) -> None:
    """A position that does not follow the face is a position nothing can reframe on."""
    left = scan(face_clip(tmp_path, "l.mp4", cx=160), tmp_path / "l")
    right = scan(face_clip(tmp_path, "r.mp4", cx=480), tmp_path / "r")
    assert left.faces[0].face_x < 0.4 < 0.6 < right.faces[0].face_x

def test_footage_with_nobody_in_it_reports_no_faces(tmp_path: Path, face_model: None) -> None:
    """Colour bars are the case a detector must not invent somebody in."""
    path = tmp_path / "bars.mp4"
    run(["-f", "lavfi", "-i", "smptebars=size=640x360:rate=30:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)])
    asset = Asset(id="bars", path=str(path), duration=3, has_video=True, has_audio=True)
    analysis = scan(asset, tmp_path)
    # Measured, and found empty — which is not the same as never looked at.
    assert len(analysis.faces) == 3
    assert all(record.faces == 0 for record in analysis.faces)
    assert all(record.face_x is None for record in analysis.faces)

def test_a_file_with_no_picture_has_no_stills_to_look_at(tmp_path: Path) -> None:
    """Nothing to detect, and no directory of frames left behind either."""
    path = tmp_path / "sound.m4a"
    run(["-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(path)])
    asset = Asset(id="sound", path=str(path), duration=2, has_video=False, has_audio=True)
    analysis = scan(asset, tmp_path)
    assert analysis.faces == []
    assert not os.path.isdir(frames_dir(str(tmp_path)))

def test_a_second_takes_the_most_faces_it_saw_and_the_biggest_ones_place() -> None:
    """A face missed in one frame of two did not leave the room, and averaging two
    positions puts a crop between two people and on neither."""
    frames = [
        (0.0, [(0.2, 0.8, 0.5), (0.05, 0.2, 0.5)]),
        (0.5, [(0.1, 0.3, 0.5)]),
    ]
    second = _per_second(frames, duration=10.0)[0]
    assert second.faces == 2
    assert (second.face_share, second.face_x) == (0.2, 0.8)

def test_a_second_with_nobody_in_it_has_no_position() -> None:
    """Absent is the honest answer: the middle of the frame would be a made-up one."""
    second = _per_second([(0.0, []), (0.5, [])], duration=10.0)[0]
    assert second.faces == 0
    assert (second.face_share, second.face_x, second.face_y) == (None, None, None)

def test_the_framing_note_says_how_far_the_face_moved() -> None:
    """One crop can hold a stretch the face barely moves in; it cannot hold one it crosses."""
    still = [FaceMeasurement(start=s, end=s + 1, faces=1, face_share=0.1, face_x=0.5, face_y=0.5) for s in range(4)]
    crossing = [
        FaceMeasurement(start=s, end=s + 1, faces=1, face_share=0.1, face_x=0.1 + 0.25 * s, face_y=0.5)
        for s in range(4)
    ]
    assert framing_note(still)["face_x_range"] == [0.5, 0.5]
    assert framing_note(crossing)["face_x_range"] == [0.1, 0.85]
    assert framing_note([])is None

def test_a_stretch_with_nobody_in_it_still_reports_how_long_it_was() -> None:
    """`with_a_face` of 0 is a measurement. No note at all would not be."""
    empty = [FaceMeasurement(start=s, end=s + 1, faces=0) for s in range(3)]
    note = framing_note(empty)
    assert note["seconds"] == 3
    assert note["with_a_face"] == 0.0
    assert "face_x_median" not in note

def test_detecting_faces_needs_no_second_pass_over_the_footage(tmp_path: Path, face_model: None) -> None:
    """The stills the detector reads are written by the pass that was decoding anyway."""
    asset = face_clip(tmp_path, "stills.mp4", cx=320, seconds=2)
    # scan_media is that pass, and on its own it leaves the stills behind for the detector.
    scan_media(asset, str(tmp_path), lambda fraction: None, lambda: False, FFMPEG)
    written = os.listdir(frames_dir(str(tmp_path)))
    assert len(written) == 4, written
    found = detect_faces(str(tmp_path), 2.0, lambda fraction: None, lambda: False)
    assert [record.faces for record in found] == [1, 1]

def test_the_stills_do_not_outlive_the_analysis(tmp_path: Path, face_model: None) -> None:
    """Two a second for a whole file is a couple of hundred megabytes an hour of scaffolding."""
    asset = face_clip(tmp_path, "tidy.mp4", cx=320, seconds=2)
    analysis = scan(asset, tmp_path)
    assert analysis.faces, "the detector should still have run"
    assert not os.path.isdir(frames_dir(str(tmp_path)))

def test_nothing_the_pass_wrote_for_itself_outlives_a_finished_analysis(
    tmp_path: Path, face_model: None
) -> None:
    """The stills, the frame metadata and the camera transforms are all scaffolding."""
    asset = face_clip(tmp_path, "tidy2.mp4", cx=320, seconds=2)
    analysis = scan(asset, tmp_path)
    assert analysis.faces and analysis.shots, "the measurements should still have been taken"
    left = sorted(os.listdir(tmp_path))
    # The log stays: it is kilobytes, and it is what is worth having when one fails.
    assert [name for name in left if name.endswith((".txt", ".trf"))] == []
    assert not os.path.isdir(frames_dir(str(tmp_path)))
    assert DETECT_LOG_FILE_NAME in left

def test_the_scratch_goes_even_when_the_scan_is_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, face_model: None
) -> None:
    """Cancelling during the scan is the likeliest moment, and the stills are already there.

    Driven by making the scan itself raise, rather than by cancelling a real
    one: a clip short enough for a fast test finishes scanning before the
    first cancellation poll, so timing it would only look like a test of this.
    """
    folder = Path(frames_dir(str(tmp_path)))
    folder.mkdir(parents=True)
    (folder / "face_000001.jpg").write_bytes(b"not really a frame")
    (tmp_path / PICTURE_FILE_NAME).write_text("frame:0    pts:0       pts_time:0\n", encoding="utf-8")
    (tmp_path / CAMERA_FILE_NAME).write_text("VID.STAB 1\n", encoding="utf-8")

    def stopped(*args, **kwargs):
        raise OperationCancelled()

    monkeypatch.setattr("app.engine.analysis.scan_media", stopped)
    asset = Asset(id="stopped", path=str(tmp_path / "absent.mp4"), duration=2, has_video=True, has_audio=False)
    with pytest.raises(OperationCancelled):
        analyze_media(asset, str(tmp_path), False, None, None, None,
                      lambda fraction, stage: None, lambda: False, FFMPEG)
    assert not folder.exists()
    assert not (tmp_path / PICTURE_FILE_NAME).exists()
    assert not (tmp_path / CAMERA_FILE_NAME).exists()

# --- speakers -------------------------------------------------------------------------------

def test_two_voices_are_told_apart(conversation, speaker_models: None) -> None:
    """The positive case: two people really do come back as two."""
    asset, expected = conversation
    turns, voices = find_speakers(asset.path, lambda fraction: None, lambda: False, speakers=2, ffmpeg_bin=FFMPEG)
    assert len({turn.speaker for turn in turns}) == 2
    assert len(turns) == len(expected)
    # And each of them was measured, so it can be recognised in another file.
    assert sorted(voice.speaker for voice in voices) == sorted({turn.speaker for turn in turns})
    assert all(voice.embedding and voice.seconds > 0 for voice in voices)
    # The same voice took the first and third turns, and the other took the second and fourth.
    labels = [turn.speaker for turn in turns]
    assert labels[0] == labels[2] and labels[1] == labels[3] and labels[0] != labels[1]

def test_the_turns_land_where_the_voices_actually_change(conversation, speaker_models: None) -> None:
    """Turns that drift are turns no clip can be labelled from."""
    asset, expected = conversation
    turns, _ = find_speakers(asset.path, lambda fraction: None, lambda: False, speakers=2, ffmpeg_bin=FFMPEG)
    for turn, (_, start, end) in zip(turns, expected):
        assert turn.start == pytest.approx(start, abs=0.6)
        # Each turn is padded with silence, so its end is reported before the padding.
        assert start < turn.end <= end + 0.6

def test_progress_is_reported_and_cancellation_is_honoured(conversation, speaker_models: None) -> None:
    """A stage with no progress on a long file looks like a hang."""
    asset, _ = conversation
    seen: List[float] = []
    find_speakers(asset.path, seen.append, lambda: False, speakers=2, ffmpeg_bin=FFMPEG)
    assert seen and seen == sorted(seen) and seen[-1] == pytest.approx(1.0)

def test_a_clip_is_labelled_with_whoever_said_it(conversation, speaker_models: None) -> None:
    """The point of all of it: `SemanticClip.speaker` stops being empty."""
    asset, _ = conversation
    turns, voices = find_speakers(asset.path, lambda fraction: None, lambda: False, speakers=2, ffmpeg_bin=FFMPEG)
    # A sentence per turn, which is what a transcript of this conversation looks like.
    segments = [
        TranscriptSegment(
            start=turn.start, end=turn.end, text=f"turn {index}",
            words=[TranscriptWord(start=turn.start, end=turn.end, text=f"turn {index}")],
        )
        for index, turn in enumerate(turns)
    ]
    analysis = MediaAnalysis(
        asset_id=asset.id,
        duration=asset.duration,
        transcript=Transcript(language="en", model="test", segments=segments),
        speakers=turns,
        voices=voices,
    )
    _, clips = build_timeline({asset.id: asset}, {asset.id: analysis})
    spoken = [clip for clip in clips if clip.text]
    # Labelled with the joined names, which are what mean the same thing across files.
    joined = {turn.speaker: label for turn, label in zip(turns, (clip.speaker for clip in spoken))}
    assert [clip.speaker for clip in spoken] == [joined[turn.speaker] for turn in turns]
    assert len(set(joined.values())) == 2 and all(label.startswith("V") for label in joined.values())

def test_a_clip_that_straddles_a_handover_is_left_unlabelled(conversation, speaker_models: None) -> None:
    """A clip holding two people is not one of them, and saying so beats picking one."""
    asset, _ = conversation
    turns, voices = find_speakers(asset.path, lambda fraction: None, lambda: False, speakers=2, ffmpeg_bin=FFMPEG)
    # One clip over the whole conversation: everybody is in it, so nobody is.
    analysis = MediaAnalysis(asset_id=asset.id, duration=asset.duration, speakers=turns, voices=voices)
    _, clips = build_timeline({asset.id: asset}, {asset.id: analysis})
    assert [clip.speaker for clip in clips] == [None]

# --- attributing a stretch to a voice ----------------------------------------------------------

def speakers() -> List[SpeakerTurn]:
    """Two voices, one after the other.

    Returns:
        A turn each, five seconds apiece.
    """
    return [SpeakerTurn(start=0, end=5, speaker="S1"), SpeakerTurn(start=5, end=10, speaker="S2")]

@pytest.mark.parametrize(
    "start, end, expected",
    [
        (1.0, 4.0, "S1"),
        (4.8, 9.0, "S2"),
        (4.0, 6.0, None),
        (11.0, 12.0, None),
    ],
)
def test_a_stretch_belongs_to_a_voice_only_when_it_clearly_does(
    start: float, end: float, expected: Optional[str]
) -> None:
    """A label goes on screen and into an edit. A wrong one is worse than none."""
    assert speaker_at(speakers(), start, end) == expected

def test_silence_inside_a_stretch_is_not_held_against_anybody() -> None:
    """A sentence with a pause in it still belongs to whoever said both halves."""
    turns = [SpeakerTurn(start=0, end=2, speaker="S1"), SpeakerTurn(start=8, end=10, speaker="S1")]
    assert speaker_at(turns, 0, 10) == "S1"

def test_the_majority_a_label_needs_is_the_one_that_is_documented() -> None:
    """The threshold is provisional, so the tests read it rather than repeating it."""
    turns = [SpeakerTurn(start=0, end=SPEAKER_MAJORITY * 10, speaker="S1"),
             SpeakerTurn(start=SPEAKER_MAJORITY * 10, end=10, speaker="S2")]
    assert speaker_at(turns, 0, 10) == "S1"
    assert speaker_at(turns, 0, 10, majority=1.0) is None

# --- the same person in two files ---------------------------------------------------------

@pytest.fixture(scope="module")
def two_recordings(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Asset, Asset]:
    """Two files of the same two voices, each starting with a different one.

    Returns:
        The two assets. Nothing in either file says the voices are shared: the
        labels are worked out per file, so one of them calls a voice `S1` that
        the other calls `S2` — which is the whole problem being solved.
    """
    if not has_flite():
        pytest.skip("this FFmpeg has no flite filter, so two voices cannot be synthesized")
    folder = tmp_path_factory.mktemp("recordings")

    def recording(name: str, order: Tuple[str, ...]) -> Asset:
        """Build one file in which the voices speak in the given order."""
        parts, seconds = [], 0.0
        for index, voice in enumerate(order):
            part = folder / f"{name}{index}.wav"
            run(["-f", "lavfi", "-i", f"flite=text='{LINES[index % len(LINES)]}':voice={voice}",
                 "-af", "apad=pad_dur=0.8", "-ar", "16000", "-ac", "1", str(part)])
            parts.append(part)
            seconds += float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(part)],
                capture_output=True, text=True,
            ).stdout)
        listing = folder / f"{name}.txt"
        listing.write_text("\n".join(f"file '{part.as_posix()}'" for part in parts), encoding="utf-8")
        sound = folder / f"{name}.wav"
        run(["-f", "concat", "-safe", "0", "-i", str(listing), "-ar", "16000", "-ac", "1", str(sound)])
        path = folder / f"{name}.mp4"
        run(["-f", "lavfi", "-i", f"smptebars=size=320x180:rate=30:duration={seconds}", "-i", str(sound),
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(path)])
        return Asset(id=name, path=str(path), duration=seconds, has_video=True, has_audio=True)

    # The second file opens on the other voice, so the per-file labels disagree.
    return recording("first", VOICES * 2), recording("second", tuple(reversed(VOICES)) * 2)

def test_the_same_voice_in_two_files_is_joined_up(two_recordings, speaker_models: None) -> None:
    """What `V1` is for: a two-camera interview is two people, not four."""
    analyses = {}
    for asset in two_recordings:
        turns, voices = find_speakers(asset.path, lambda fraction: None, lambda: False,
                                      speakers=2, ffmpeg_bin=FFMPEG)
        analyses[asset.id] = MediaAnalysis(
            asset_id=asset.id, duration=asset.duration, speakers=turns, voices=voices,
        )
    joined = join_voices(analyses)
    assert len(joined) == 4, joined
    # Two people across both files, not four.
    assert len(set(joined.values())) == 2, joined
    # And each file heard both of them.
    for asset in two_recordings:
        assert len({label for (asset_id, _), label in joined.items() if asset_id == asset.id}) == 2

def test_a_file_nobody_measured_the_voices_of_is_absent_rather_than_nobody() -> None:
    """Not knowing is not the same as there being nobody, so it is left out entirely."""
    quiet = MediaAnalysis(asset_id="a", duration=10.0, speakers=[
        SpeakerTurn(start=0.0, end=5.0, speaker="S1"),
    ])
    assert join_voices({"a": quiet}) == {}

def test_voices_are_joined_in_a_fixed_order() -> None:
    """Assigned by file and label, not by who speaks first, so a rebuild cannot rename them."""
    def heard(asset_id: str, *labels: str) -> MediaAnalysis:
        """One analysis whose voices sit at fixed points."""
        points = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]}
        return MediaAnalysis(asset_id=asset_id, duration=10.0, voices=[
            Voice(speaker=f"S{index + 1}", embedding=points[label], seconds=5.0)
            for index, label in enumerate(labels)
        ])

    # The same two voices, met in the opposite order in the second file.
    joined = join_voices({"one": heard("one", "a", "b"), "two": heard("two", "b", "a")})
    assert joined[("one", "S1")] == "V1" and joined[("one", "S2")] == "V2"
    assert joined[("two", "S1")] == "V2" and joined[("two", "S2")] == "V1"

def test_two_voices_far_enough_apart_stay_apart() -> None:
    """The threshold has to do something, or everybody is one person."""
    def alone(asset_id: str, embedding: List[float]) -> MediaAnalysis:
        """One analysis holding a single voice."""
        return MediaAnalysis(asset_id=asset_id, duration=10.0, voices=[
            Voice(speaker="S1", embedding=embedding, seconds=5.0),
        ])

    apart = join_voices({"one": alone("one", [1.0, 0.0]), "two": alone("two", [0.0, 1.0])})
    assert len(set(apart.values())) == 2
    together = join_voices({"one": alone("one", [1.0, 0.0]), "two": alone("two", [1.0, 0.05])})
    assert len(set(together.values())) == 1
