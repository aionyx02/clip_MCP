"""A finished render's sound, checked against where the cut says the talking is."""

import json
import os
import subprocess
import types
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pytest

from app import runtime
from app.engine import renderer, sync
from app.models.job import Job, JobKind, JobStatus
from app.models.timeline.parts import Clip, TimeRange, TitleCard
from app.runtime import repo
from app.storage.repo import Repository
from app.tools import rendering
from app.tools.catalog import import_asset
from app.tools.job_queue import get_job
from helpers import build_project, insert, video_track

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FRAME = 1 / 30

def make_clip(clip_id: str = "a", asset_id: str = "asset", source_start: float = 0, source_end: float = 2,
              timeline_in: float = 0, volume: float = 1.0, speed: float = 1.0, audio_lead: float = 0,
              card: Optional[TitleCard] = None) -> Clip:
    """Build a clip of the base video track.

    Args:
        clip_id: The clip's ID.
        asset_id: The file it comes from.
        source_start: Where in the file its picture begins.
        source_end: Where in the file its picture ends.
        timeline_in: Where it begins on the timeline.
        volume: How loud its sound plays.
        speed: How fast it plays.
        audio_lead: How far its sound starts before its picture.
        card: The title card it is, if any.

    Returns:
        The clip.
    """
    return Clip(
        id=clip_id,
        asset_id=asset_id,
        source_range=TimeRange(start=Decimal(str(source_start)), end=Decimal(str(source_end))),
        timeline_in=Decimal(str(timeline_in)),
        volume=volume,
        speed=speed,
        audio_lead=Decimal(str(audio_lead)),
        card=card,
    )

def test_streams_that_line_up_are_not_reported() -> None:
    probed = {"streams": [
        {"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": "0.000000", "duration": "10.000000"},
    ], "format": {"duration": "10.000000"}}
    assert sync.stream_mismatch(probed, FRAME) is None

def test_an_audio_stream_that_starts_late_is_reported() -> None:
    probed = {"streams": [
        {"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": "0.200000", "duration": "10.000000"},
    ], "format": {"duration": "10.000000"}}
    found = sync.stream_mismatch(probed, FRAME)
    assert found == {"video_start": 0.0, "audio_start": 0.2, "video_length": 10.0, "audio_length": 10.0}

def test_a_file_without_sound_is_not_compared() -> None:
    probed = {"streams": [{"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"}],
              "format": {"duration": "10.000000"}}
    assert sync.stream_mismatch(probed, FRAME) is None

def test_a_stream_without_a_duration_is_measured_by_the_file() -> None:
    probed = {"streams": [
        {"codec_type": "video", "start_time": "0.000000", "duration": "10.000000"},
        {"codec_type": "audio", "start_time": "0.000000"},
    ], "format": {"duration": "10.200000"}}
    found = sync.stream_mismatch(probed, FRAME)
    assert found is not None and found["audio_length"] == 10.2

def test_streams_whose_length_nothing_says_are_not_compared() -> None:
    probed = {"streams": [{"codec_type": "video", "start_time": "0.000000"},
                          {"codec_type": "audio", "start_time": "0.300000"}], "format": {}}
    assert sync.stream_mismatch(probed, FRAME) is None

def test_a_clip_at_double_speed_has_its_sound_squeezed_with_it() -> None:
    clip = make_clip(source_start=2, source_end=6, timeline_in=10, speed=2.0)
    assert sync.expected_voicing([clip], {"asset": [(2.0, 3.0)]}) == [(10.0, 10.5)]

def test_a_lead_puts_the_sound_before_the_picture_and_takes_it_from_earlier() -> None:
    clip = make_clip(source_start=2, source_end=4, timeline_in=10, audio_lead=0.5)
    assert sync.expected_voicing([clip], {"asset": [(2.0, 3.0)]}) == [(10.0, 11.0)]

def test_a_muted_clip_contributes_no_voicing() -> None:
    clip = make_clip(source_start=0, source_end=2, timeline_in=0, volume=0)
    assert sync.expected_voicing([clip], {"asset": [(0.0, 2.0)]}) == []

def test_a_title_card_contributes_no_voicing() -> None:
    clip = make_clip(source_start=0, source_end=2, timeline_in=0, card=TitleCard(title="開場"))
    assert sync.expected_voicing([clip], {"asset": [(0.0, 2.0)]}) == []

def test_a_voiced_stretch_reaching_past_the_clip_is_cut_to_it() -> None:
    clip = make_clip(source_start=2, source_end=4, timeline_in=10)
    assert sync.expected_voicing([clip], {"asset": [(1.0, 3.0)]}) == [(10.0, 11.0)]

def test_touching_voicing_from_two_clips_is_merged() -> None:
    first = make_clip("a", "one", source_start=0, source_end=2, timeline_in=0)
    second = make_clip("b", "two", source_start=0, source_end=2, timeline_in=2)
    assert sync.expected_voicing([first, second], {"one": [(0.0, 2.0)], "two": [(0.0, 2.0)]}) == [(0.0, 4.0)]

def test_a_join_next_to_a_muted_clip_is_left_out() -> None:
    first = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    muted = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, volume=0)
    last = make_clip("c", "asset", source_start=4, source_end=6, timeline_in=4)
    assert sync.joins([first, muted, last], {"asset": [(0.0, 6.0)]}) == []

def test_a_join_into_a_title_card_is_left_out() -> None:
    first = make_clip("a", "asset", source_start=0, source_end=2, timeline_in=0)
    card = make_clip("b", "asset", source_start=2, source_end=4, timeline_in=2, card=TitleCard(title="開場"))
    assert sync.joins([first, card], {"asset": [(0.0, 4.0)]}) == []

def test_a_join_near_a_file_nobody_listened_to_for_voicing_is_left_out() -> None:
    # Its talking would be silence to the cut and read as a voice out of place.
    first = make_clip("a", "heard", source_start=0, source_end=2, timeline_in=0)
    second = make_clip("b", "heard", source_start=2, source_end=4, timeline_in=2)
    third = make_clip("c", "unheard", source_start=0, source_end=2, timeline_in=4.6)
    fourth = make_clip("d", "heard", source_start=4, source_end=10, timeline_in=6.6)
    # The joins into and out of it are left out; the one before it, out of reach, is not.
    assert sync.joins([first, second, third, fourth], {"heard": [(0.0, 10.0)]}) == [2.0]

def test_a_join_is_left_out_when_an_unlistened_clip_starts_just_after_it() -> None:
    first = make_clip("a", "heard", source_start=0, source_end=2, timeline_in=0)
    second = make_clip("b", "heard", source_start=0, source_end=1, timeline_in=2)
    later = make_clip("c", "unheard", source_start=0, source_end=2, timeline_in=4.3)
    assert sync.joins([first, second, later], {"heard": [(0.0, 10.0)]}) == []

def test_identical_voicing_has_no_lag() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(1.0, 2.0)], 2.0) == 0.0

def test_sound_heard_later_than_the_cut_says_has_a_positive_lag() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(1.12, 2.12)], 2.0) == 0.12

def test_sound_heard_sooner_than_the_cut_says_has_a_negative_lag() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(0.88, 1.88)], 2.0) == -0.12

def test_a_window_with_nothing_changing_has_nothing_to_line_up() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(0.0, 4.0)], 8.0) is None

def test_unrelated_voicing_does_not_agree_enough_to_be_believed() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(3.0, 4.0)], 2.0) is None

def test_a_short_sound_beside_a_short_word_does_not_give_a_lag() -> None:
    assert sync.lag_at([(1.0, 1.5)], [(2.0, 2.4)], 2.0) is None

def test_a_lag_past_the_edge_of_the_search_is_not_reported() -> None:
    assert sync.lag_at([(1.0, 2.0)], [(1.6, 2.6)], 2.0) is None

def test_agreeing_equally_well_at_several_lags_takes_the_smallest_move() -> None:
    assert sync.lag_at([(0.0, 3.5)], [(-1.0, 10.0)], 2.0) == 0.0

def test_agreeing_equally_either_way_at_one_size_takes_the_negative_lag() -> None:
    # Each shift covers the same stretch of the short blocks either side of the join.
    expected = [(0.0, 0.2), (0.4, 3.6), (3.8, 4.0)]
    measured = [(0.2, 3.8)]
    assert sync.lag_at(expected, measured, 2.0) == -0.2

def test_check_compares_a_real_file_and_reports_its_streams(tmp_path: Path) -> None:
    path = tmp_path / "late.mp4"
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=3",
        "-itsoffset", "0.3", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True)

    def refuse(*args, **kwargs):
        raise AssertionError("the sound was decoded though there was nothing to compare it with")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(sync, "read_samples", refuse)
        found = sync.check(str(path), [], [], FRAME, FFMPEG, "ffprobe")
    assert found["streams"] is not None
    assert found["joins_asked"] == 0
    assert found["frame"] == round(FRAME, 4)

def test_check_does_not_decode_when_there_are_no_joins(tmp_path: Path) -> None:
    path = tmp_path / "plain.mp4"
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True)

    def refuse(*args, **kwargs):
        raise AssertionError("the sound was decoded though no join was asked about")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(sync, "read_samples", refuse)
        found = sync.check(str(path), [[1.0, 2.0]], [], FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == []
    assert found["joins_asked"] == 0

def test_check_lines_the_heard_voice_up_with_the_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "plain.mp4"
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    monkeypatch.setattr(sync, "voiced_stretches", lambda samples: [(1.12, 2.12)])
    found = sync.check(str(path), [[1.0, 2.0]], [2.0], FRAME, FFMPEG, "ffprobe")
    assert found["joins"] == [{"at": 2.0, "lag": 0.12}]
    assert found["joins_asked"] == 1

def test_a_finished_render_keeps_what_its_sound_check_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = Repository(str(tmp_path / "jobs.db"))
    found = {"streams": None, "joins": [{"at": 2.0, "lag": 0.12}], "frame": round(FRAME, 4)}
    (tmp_path / renderer.SPEC_FILE_NAME).write_text(json.dumps({"render_path": str(tmp_path / "out.mp4")}))
    job = Job(kind=JobKind.RENDER, work_dir=str(tmp_path))
    store.add_job(job)
    monkeypatch.setattr(renderer, "_run_render", lambda job, spec, context, repo: sync.keep(job.work_dir, found))
    monkeypatch.setattr(renderer.JobManager, "launch_ready", lambda self: None)
    renderer.run_worker(store, job.job_id)
    assert store.get_job(job.job_id).sync == found

def make_finished_job(project_id: str, sync_data: Optional[dict]) -> Job:
    """Store a completed render with what its sound check found.

    Args:
        project_id: The project it was rendered from.
        sync_data: The check's result, or None when it was not checked.

    Returns:
        The stored job.
    """
    job = Job(kind=JobKind.RENDER, project_id=project_id, status=JobStatus.COMPLETED, sync=sync_data)
    repo.add_job(job)
    return job

def test_get_job_names_the_clip_whose_voice_is_heard_late(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {"streams": None, "joins": [{"at": 2.0, "lag": 0.12}], "frame": FRAME})
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {"out_of_step": ["at 2.00s, clip b: the voice is heard 0.12s late"]}

def test_get_job_says_nothing_about_sound_within_a_frame(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {"streams": None, "joins": [{"at": 2.0, "lag": 0.02}], "frame": FRAME})
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_get_job_says_where_each_stream_starts_and_ends(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    streams = {"video_start": 0.0, "audio_start": 0.3, "video_length": 3.0, "audio_length": 2.7}
    job = make_finished_job(project, {"streams": streams, "joins": [], "frame": FRAME})
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {"out_of_step": ["the sound runs 0.30s to 3.00s while the picture runs 0.00s to 3.00s"]}

def test_get_job_ignores_a_lag_smaller_than_the_voice_detectors_step(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    job = make_finished_job(project, {"streams": None, "joins": [{"at": 2.0, "lag": 0.03}],
                                      "joins_asked": 1, "frame": 1 / 60})
    entry = get_job([job.job_id])["jobs"][0]
    assert "sound" not in entry

def test_get_job_says_when_the_sound_was_not_checked(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, None)
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_checked": "the sound was not checked against the picture — the check failed or was stopped — "
                       "so whether it lines up is not known"
    }

def test_get_job_says_when_no_join_could_be_compared(media: Path) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2)], width=640, height=360)
    job = make_finished_job(project, {"streams": None, "joins": [], "joins_asked": 3, "frame": FRAME})
    entry = get_job([job.job_id])["jobs"][0]
    assert entry["sound"] == {
        "not_compared": "none of the 3 joins could be lined up — the voice does not start or stop near them, "
                        "or did not agree well enough — so whether they are in step is not known"
    }

def test_render_project_asks_for_the_sound_check(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 2), insert("b", wide, 2, 4)],
                            width=640, height=360)
    started = []
    voices = {wide: types.SimpleNamespace(transcript=None, second_listen=types.SimpleNamespace(
        voiced=[types.SimpleNamespace(start=0.0, end=2.0)]))}
    monkeypatch.setattr(runtime.job_manager, "start_job", lambda job, spec: started.append(spec) or job)
    monkeypatch.setattr(rendering, "analyses_of", lambda project, listened=False: voices)
    monkeypatch.setattr(rendering, "talking_of", lambda project, voices: None)
    monkeypatch.setattr(rendering, "check_findings", lambda project, burn: [])
    rendering.render_project(project)
    assert started[-1]["sync"] == {"expected": [[0.0, 2.0]], "joins": [2.0], "frame": FRAME}

def sync_spec(work_dir: Path) -> dict:
    """Build the job specification a sound-check test needs, with no pieces to render.

    Args:
        work_dir: The job's work folder.

    Returns:
        The specification.
    """
    return {
        "pieces": [],
        "duration_seconds": 5.0,
        "command": ["ffmpeg"],
        "render_path": str(work_dir / "out.mp4"),
        "sync": {"expected": [[1.0, 2.0]], "joins": [2.0], "frame": FRAME},
    }

def run_render(monkeypatch: pytest.MonkeyPatch, work_dir: Path, spec: dict) -> None:
    """Run a render past its sound check with FFmpeg stubbed out.

    Args:
        monkeypatch: The test's monkeypatch.
        work_dir: Where the render's files go.
        spec: The job's specification.
    """
    class Reported:
        def report(self, *_):
            pass

        def is_cancelled(self):
            return False

    monkeypatch.setattr(renderer, "run_ffmpeg", lambda *args: None)
    renderer._run_render(Job(work_dir=str(work_dir)), spec, Reported())

def test_a_sound_check_that_breaks_leaves_the_render_unchecked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Broke(Exception):
        pass

    def broke(*args, **kwargs):
        raise Broke("the check broke")

    monkeypatch.setattr(sync, "check", broke)
    run_render(monkeypatch, tmp_path, sync_spec(tmp_path))
    assert sync.kept(str(tmp_path)) is None

def test_a_render_runs_the_sound_check_the_spec_asked_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    found = {"streams": None, "joins": [], "joins_asked": 1, "frame": round(FRAME, 4)}
    asked = []
    monkeypatch.setattr(sync, "check", lambda *args: asked.append(args) or found)
    spec = sync_spec(tmp_path)
    run_render(monkeypatch, tmp_path, spec)
    assert asked == [(spec["render_path"], [[1.0, 2.0]], [2.0], FRAME, "ffmpeg", "ffprobe")]
    assert sync.kept(str(tmp_path)) == found
