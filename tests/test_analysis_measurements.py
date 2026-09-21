"""Tests for what the analysis pass measures about the picture and the sound.

Two halves. The parsers and the folding of samples into shots and seconds are
pure functions, so those are checked against text written by hand. The
measurements themselves are not: whether `blur` really rises when a picture
goes soft can only be answered by decoding footage that is soft, so the second
half synthesizes clips that differ in one way each and checks that the number
meant to catch that difference is the one that moves.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

import pytest

from app.engine.analysis import (
    ANALYSIS_VERSION,
    CAMERA_FPS,
    SILENCE_DB,
    _camera_movement,
    _frame_records,
    _scan_graph,
    _shot_measurements,
    _sound_measurements,
    analyze_media,
    can_measure_shake,
    current_recipe,
)
from app.engine.faces import FRAME_PATTERN, frames_dir
from app.models.media import (
    AnalysisRecipe,
    Asset,
    MediaAnalysis,
    ShotMeasurement,
    SoundMeasurement,
    Span,
)
from app.engine.semantic import build_timeline, combined_over

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

def picture_file(path: Path, blocks) -> str:
    """Write a file in the shape the `metadata` filter prints.

    Args:
        path: File to write.
        blocks: `(pts_time, {key: text})` per frame.

    Returns:
        The path as a string.
    """
    lines = []
    for index, (seconds, values) in enumerate(blocks):
        lines.append(f"frame:{index}    pts:{index}       pts_time:{seconds}")
        lines.extend(f"{key}={value}" for key, value in values.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)

def scan(asset: Asset, work_dir: Path) -> MediaAnalysis:
    """Analyze one asset without transcribing it.

    Args:
        asset: Asset to analyze.
        work_dir: Directory for the log and the measurement files.

    Returns:
        The analysis.
    """
    return analyze_media(
        asset, str(work_dir), False, None, None, None, lambda fraction, stage: None, lambda: False, FFMPEG
    )

def clip(
    tmp_path: Path,
    name: str,
    video_filter: Optional[str] = None,
    audio: str = "sine=frequency=440",
    audio_filter: Optional[str] = None,
    audio_codec: str = "aac",
    seconds: int = 4,
) -> Asset:
    """Synthesize one clip that differs from the plain one in a single way.

    Args:
        tmp_path: Directory to write it to.
        name: File name, whose extension decides the container.
        video_filter: Filter that spoils or moves the picture, if any.
        audio: lavfi audio source.
        audio_filter: Filter applied to the sound, if any.
        audio_codec: Encoder for the sound; lossless where the test measures
            levels at the ceiling, since an encoder would round them off.
        seconds: Length of the clip.

    Returns:
        The asset, registered against the file.
    """
    path = tmp_path / name
    args = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"smptebars=size=640x360:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"{audio}:duration={seconds}",
    ]
    if video_filter:
        args += ["-vf", video_filter]
    if audio_filter:
        args += ["-af", audio_filter]
    args += ["-c:a", audio_codec, "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    result = subprocess.run(args, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return Asset(id=f"a-{name}", path=str(path), duration=seconds, has_video=True, has_audio=True)

@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> MediaAnalysis:
    """Analyze a clip with nothing wrong with it, to compare the spoiled ones against.

    Returns:
        The analysis of a sharp, well-exposed, locked-off clip with a clean tone on it.
    """
    folder = tmp_path_factory.mktemp("plain")
    return scan(clip(folder, "plain.mp4"), folder)

# --- the parsers -------------------------------------------------------------------------

def test_frame_records_drop_values_that_are_not_numbers(tmp_path: Path) -> None:
    """A filter reporting nothing measurable leaves the key out, not a nan in."""
    path = picture_file(tmp_path / "picture.txt", [
        (0.0, {"lavfi.signalstats.YAVG": "100", "lavfi.blur": "nan"}),
        (0.25, {"lavfi.signalstats.YAVG": "120", "lavfi.blur": "7.5"}),
    ])
    records = _frame_records(path)
    assert [seconds for seconds, _ in records] == [0.0, 0.25]
    assert "lavfi.blur" not in records[0][1]
    assert records[1][1]["lavfi.blur"] == 7.5

def test_frame_records_keep_a_frame_whose_every_value_was_dropped(tmp_path: Path) -> None:
    """Digital silence measures as minus infinity, and still has to be a measurement."""
    path = picture_file(tmp_path / "sound.txt", [
        (0.0, {"lavfi.astats.Overall.RMS_level": "-inf", "lavfi.astats.Overall.Peak_level": "-inf"}),
        (1.0, {"lavfi.astats.Overall.RMS_level": "-20.5", "lavfi.astats.Overall.Peak_level": "-12.0"}),
    ])
    records = _frame_records(path)
    assert len(records) == 2
    assert records[0][1] == {}

def test_frame_records_of_a_stream_that_was_not_measured_are_empty(tmp_path: Path) -> None:
    """A file the graph never wrote reads as nothing measured rather than as an error."""
    assert _frame_records(str(tmp_path / "never-written.txt")) == []

def test_sound_measurements_write_silence_rather_than_leaving_a_gap(tmp_path: Path) -> None:
    """A window with no level at all is silence, and says so in decibels."""
    records = [(0.0, {}), (1.0, {"lavfi.astats.Overall.RMS_level": -20.5})]
    windows = _sound_measurements(records, duration=1.4)
    assert [window.loudness for window in windows] == [SILENCE_DB, -20.5]
    # The last window is cut to the file, not left running past its end.
    assert windows[1].end == 1.4

def test_camera_movement_takes_the_median_of_the_measurement_fields(tmp_path: Path) -> None:
    """One field on a moving subject must not become the camera's own movement."""
    path = tmp_path / "camera.trf"
    path.write_text(
        "VID.STAB 1\n"
        "#      accuracy = 9\n"
        "Frame 1 (List 3 [(LM 0 0 10 10 32 0.5 0.0),(LM 0 0 20 20 32 0.5 0.0),(LM 0 0 30 30 32 0.5 0.0)])\n"
        "Frame 2 (List 3 [(LM 0 0 10 10 32 0.5 0.0),(LM 64 0 20 20 32 0.5 0.0),(LM 0 0 30 30 32 0.5 0.0)])\n",
        encoding="utf-8",
    )
    movement = _camera_movement(str(path), fps=10, width=320)
    assert [seconds for seconds, _, _ in movement] == [0.0, 0.1]
    # Two of three fields say nothing moved, so nothing moved.
    assert movement[1][1] == 0.0

def test_camera_movement_of_a_file_that_was_not_written_is_empty(tmp_path: Path) -> None:
    """A build without libvidstab leaves no transforms behind, and that is not a failure."""
    assert _camera_movement(str(tmp_path / "absent.trf")) == []

def test_a_shot_with_no_sample_in_it_gets_no_measurement(tmp_path: Path) -> None:
    """Not measured has to stay distinguishable from measured and unremarkable."""
    records = _frame_records(picture_file(tmp_path / "picture.txt", [
        (0.0, {"lavfi.signalstats.YAVG": "128", "lavfi.signalstats.YHIGH": "200",
               "lavfi.signalstats.YLOW": "50", "lavfi.signalstats.YDIF": "0", "lavfi.blur": "6"}),
    ]))
    scenes = [Span(start=0.0, end=1.0), Span(start=1.0, end=1.05)]
    shots = _shot_measurements(scenes, records, movement=[])
    assert [(shot.start, shot.end) for shot in shots] == [(0.0, 1.0)]
    # Nothing steady enough to compare, so shake is null rather than zero.
    assert shots[0].shake is None

def test_shake_separates_a_hand_from_an_even_pan() -> None:
    """A pan moves as much as a hand does. What it does not do is keep changing direction."""
    scenes = [Span(start=0.0, end=1.0)]
    records = [
        (seconds, {"lavfi.signalstats.YAVG": 128.0, "lavfi.signalstats.YHIGH": 200.0,
                   "lavfi.signalstats.YLOW": 50.0, "lavfi.signalstats.YDIF": 3.0})
        for seconds in (0.0, 0.25, 0.5, 0.75)
    ]
    pan = [(index / CAMERA_FPS, 0.01, 0.0) for index in range(15)]
    hand = [(index / CAMERA_FPS, 0.01 if index % 2 else -0.01, 0.0) for index in range(15)]
    steady = _shot_measurements(scenes, records, pan)[0].shake
    wobbly = _shot_measurements(scenes, records, hand)[0].shake
    assert steady == 0.0
    assert wobbly > 0.01

# --- the filtergraph ---------------------------------------------------------------------

def test_the_graph_leaves_out_what_a_file_has_no_stream_for(tmp_path: Path) -> None:
    """A file with no picture must not be asked for picture measurements."""
    audio_only = Asset(id="a", path="x.mp3", duration=1, has_video=False, has_audio=True)
    graph, outputs = _scan_graph(audio_only, str(tmp_path), with_shake=True)
    assert "signalstats" not in graph and "vidstabdetect" not in graph
    # No picture, so no stills to write either.
    assert outputs == [["-map", "[heard]", "-f", "null", "-"]]

    silent = Asset(id="b", path="x.mp4", duration=1, has_video=True, has_audio=False)
    graph, outputs = _scan_graph(silent, str(tmp_path), with_shake=False)
    assert "astats" not in graph and "vidstabdetect" not in graph
    assert "split=3" in graph
    assert [output[1] for output in outputs] == ["[seen]", "[still]"]

def test_the_stills_the_face_detector_reads_come_out_of_the_same_pass(tmp_path: Path) -> None:
    """A second decode of the footage just to look for faces is what this avoids."""
    asset = Asset(id="a", path="x.mp4", duration=1, has_video=True, has_audio=True)
    graph, outputs = _scan_graph(asset, str(tmp_path), with_shake=True)
    # One split feeds the detectors, the picture measurements, the stills and the camera.
    assert "split=4" in graph
    stills = next(output for output in outputs if output[1] == "[still]")
    assert stills[-1] == os.path.join(frames_dir(str(tmp_path)), FRAME_PATTERN)
    assert stills[-2] == "image2"

def test_the_graph_quotes_the_paths_its_filters_write_to(tmp_path: Path) -> None:
    """An ordinary Windows path is filtergraph syntax, and breaks the whole pass."""
    asset = Asset(id="a", path="x.mp4", duration=1, has_video=True, has_audio=False)
    graph, _ = _scan_graph(asset, r"C:\Users\someone\jobs\j1", with_shake=True)
    assert r"file='C\:/Users/someone/jobs/j1/picture.txt'" in graph
    assert r"result='C\:/Users/someone/jobs/j1/camera.trf'" in graph

# --- the recipe --------------------------------------------------------------------------

def test_an_analysis_from_before_the_measurements_reads_as_stale() -> None:
    """Anything already on disk was measured by a pass that took fewer numbers."""
    assert AnalysisRecipe().differs_from(current_recipe(None, FFMPEG))
    assert AnalysisRecipe().version == 0 < ANALYSIS_VERSION

def test_changing_the_speech_model_only_ages_analyses_that_used_one() -> None:
    """There is nothing in a picture-only analysis that a speech model wrote."""
    current = current_recipe("large-v3-turbo", FFMPEG)
    spoken = current_recipe("small", FFMPEG)
    silent = current_recipe(None, FFMPEG)
    assert spoken.differs_from(current)
    assert not silent.differs_from(current)
    assert not current.differs_from(current)

def test_the_recipe_covers_the_thresholds_diarization_runs_with(monkeypatch: pytest.MonkeyPatch) -> None:
    """These are the provisional numbers a corpus is expected to move. Moving one has to show."""
    before = current_recipe(None, FFMPEG).detectors
    monkeypatch.setattr("app.engine.analysis.SPEAKER_DISTANCE", 0.9)
    assert current_recipe(None, FFMPEG).detectors != before

def test_the_recipe_covers_whether_shake_could_be_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine that cannot measure shake produces a different analysis, not the same one."""
    here = can_measure_shake(FFMPEG)
    with_shake = current_recipe(None, FFMPEG)
    monkeypatch.setattr("app.engine.analysis.can_measure_shake", lambda _bin="ffmpeg": False)
    without = current_recipe(None, FFMPEG)
    assert (with_shake.detectors != without.detectors) == here

# --- what the numbers actually catch, measured off real footage ---------------------------

def test_a_plain_clip_measures_as_sharp_still_and_well_exposed(plain: MediaAnalysis) -> None:
    """The baseline the spoiled clips are compared against has to be unremarkable itself."""
    shot = plain.shots[0]
    assert 0.2 < shot.exposure < 0.8
    assert shot.contrast > 0.3
    assert shot.motion == pytest.approx(0.0, abs=0.001)
    assert len(plain.sound) == 4
    assert plain.sound[0].loudness > SILENCE_DB

def test_blur_rises_when_the_picture_goes_soft(tmp_path: Path, plain: MediaAnalysis) -> None:
    """Softness is the one thing that changed, so blur is the one number that may move."""
    soft = scan(clip(tmp_path, "soft.mp4", video_filter="gblur=sigma=8"), tmp_path)
    assert soft.shots[0].blur > plain.shots[0].blur * 2
    assert soft.shots[0].exposure == pytest.approx(plain.shots[0].exposure, abs=0.01)

def test_exposure_and_contrast_fall_on_an_underexposed_clip(tmp_path: Path, plain: MediaAnalysis) -> None:
    """A clip shot in the dark reads as dark without anyone looking at it."""
    dark = scan(clip(tmp_path, "dark.mp4", video_filter="lutyuv=y=val*0.08"), tmp_path)
    assert dark.shots[0].exposure < plain.shots[0].exposure / 4
    assert dark.shots[0].contrast < plain.shots[0].contrast / 4

def test_a_handheld_clip_measures_as_moving_and_unsteady(tmp_path: Path, plain: MediaAnalysis) -> None:
    """Motion says something changed; shake says it was the camera, and erratically."""
    shaky = scan(clip(tmp_path, "shaky.mp4", video_filter="crop=600:340:10+9*sin(n*2.3):10+9*cos(n*3.1)"), tmp_path)
    assert shaky.shots[0].motion > 0.005
    assert plain.shots[0].motion == pytest.approx(0.0, abs=0.001)
    if can_measure_shake(FFMPEG):
        assert shaky.shots[0].shake > 0.005
        assert plain.shots[0].shake == pytest.approx(0.0, abs=0.001)
    else:
        assert shaky.shots[0].shake is None

def test_a_muted_recording_measures_as_silence_in_every_window(tmp_path: Path) -> None:
    """A microphone left off is measured and found empty, not left unmeasured."""
    muted = scan(clip(tmp_path, "muted.mp4", audio="anullsrc=r=48000:cl=mono"), tmp_path)
    assert len(muted.sound) == 4
    assert all(window.loudness == SILENCE_DB for window in muted.sound)
    assert all(window.noise_floor == SILENCE_DB for window in muted.sound)

def test_a_clipped_recording_peaks_at_the_ceiling_and_is_flat_there(tmp_path: Path, plain: MediaAnalysis) -> None:
    """Clipping is a peak with nowhere left to go and a waveform squared off against it."""
    clipped = scan(clip(tmp_path, "clipped.mov", audio_filter="volume=40dB", audio_codec="pcm_s16le"), tmp_path)
    assert clipped.sound[1].peak > -0.5
    assert clipped.sound[1].flatness > 10
    assert plain.sound[1].flatness == pytest.approx(0.0, abs=0.01)

def test_a_cut_into_a_shot_is_not_movement_inside_it(tmp_path: Path) -> None:
    """A shot's first sample measured its change against the last frame of the shot before."""
    path = tmp_path / "cut.mp4"
    result = subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:size=640x360:rate=30:duration=2",
        "-f", "lavfi", "-i", "color=c=white:size=640x360:rate=30:duration=2",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
    ], capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    asset = Asset(id="cut", path=str(path), duration=4, has_video=True, has_audio=False)
    analysis = scan(asset, tmp_path)

    assert [(shot.start, shot.end) for shot in analysis.shots] == [(0.0, 2.0), (2.0, 4.0)]
    # Nothing happens in either shot. The second one only follows a very large cut.
    assert [shot.motion for shot in analysis.shots] == [0.0, 0.0]

def test_exposure_is_measured_on_the_full_range_whatever_the_file_carries(tmp_path: Path) -> None:
    """Most footage puts black at 16 and white at 235, and a raw reading is neither 0 nor 1."""
    path = tmp_path / "ends.mp4"
    result = subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:size=640x360:rate=30:duration=2",
        "-f", "lavfi", "-i", "color=c=white:size=640x360:rate=30:duration=2",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
    ], capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    asset = Asset(id="ends", path=str(path), duration=4, has_video=True, has_audio=False)
    analysis = scan(asset, tmp_path)
    assert [shot.exposure for shot in analysis.shots] == [0.0, 1.0]

def test_an_analysis_records_the_recipe_that_produced_it(plain: MediaAnalysis) -> None:
    """A measurement that does not say what took it cannot be told from a newer one."""
    assert plain.recipe.version == ANALYSIS_VERSION
    assert not plain.recipe.differs_from(current_recipe(None, FFMPEG))

# --- and how they reach a clip -----------------------------------------------------------

def test_a_clip_carries_the_measurements_of_the_shot_and_seconds_it_spans(
    tmp_path: Path, plain: MediaAnalysis
) -> None:
    """Choosing material is done over clips, so the numbers have to arrive there."""
    asset = Asset(id=plain.asset_id, path="plain.mp4", duration=plain.duration, has_video=True, has_audio=True)
    _, clips = build_timeline({asset.id: asset}, {asset.id: plain})
    assert clips
    scores = clips[0].scores
    for name in ("exposure", "contrast", "blur", "motion", "loudness", "peak", "noise_floor", "flatness"):
        assert name in scores, name

def test_combining_weights_each_measurement_by_how_much_of_the_clip_it_covers() -> None:
    """A shot that a clip barely touches must not count as much as the one it sits in."""
    shots = [
        ShotMeasurement(start=0, end=2, exposure=0.5, contrast=0.4, blur=5.0, motion=0.01),
        ShotMeasurement(start=2, end=4, exposure=0.2, contrast=0.4, blur=None, motion=0.01),
    ]
    combined = combined_over(1, 4, shots)
    assert combined["exposure"] == pytest.approx((0.5 * 1 + 0.2 * 2) / 3, abs=5e-5)
    # Only one of the two shots has a blur, so the answer is that one's, not half of it.
    assert combined["blur"] == 5.0
    # Nothing overlapping was measured, so nothing is claimed.
    assert combined_over(9, 10, shots) == {}

def test_a_silent_second_does_not_make_a_whole_clip_look_quiet() -> None:
    """Decibels are logarithmic. Averaging them is how a fine clip reads as unusable."""
    speech = [SoundMeasurement(start=s, end=s + 1, loudness=-20.0, peak=-6.0, noise_floor=-50.0, flatness=0.0)
              for s in range(9)]
    silence = SoundMeasurement(start=9, end=10, loudness=SILENCE_DB, peak=SILENCE_DB,
                               noise_floor=SILENCE_DB, flatness=0.0)
    combined = combined_over(0, 10, [*speech, silence])
    # Nine seconds of speech and one of nothing is still a clip recorded at about -20 dB.
    assert combined["loudness"] == pytest.approx(-20.0, abs=0.6)
    # A peak is the loudest moment there was, not the average of moments.
    assert combined["peak"] == -6.0
    # And the hiss under the clip is the hiss under most of it, not the absence of hiss
    # under the one second that had nothing in it at all.
    assert combined["noise_floor"] == -50.0

def test_one_clipped_second_is_not_averaged_out_of_a_clip() -> None:
    """Whether it ever happened is the whole question for a peak."""
    windows = [SoundMeasurement(start=s, end=s + 1, loudness=-20.0, peak=-6.0, noise_floor=-50.0, flatness=0.0)
               for s in range(9)]
    windows.append(SoundMeasurement(start=9, end=10, loudness=-1.0, peak=0.0, noise_floor=-50.0, flatness=24.0))
    combined = combined_over(0, 10, windows)
    assert combined["peak"] == 0.0
    assert combined["flatness"] == 24.0
