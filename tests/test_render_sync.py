"""A finished render's sound, checked against where the cut says the talking is."""

import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from app.engine import renderer, sync
from app.models.job import Job, JobKind, JobStatus
from app.models.timeline import Clip, TimeRange
from app.runtime import repo
from app.storage.repo import Repository
from app.tools.catalog import import_asset
from app.tools.job_queue import get_job
from helpers import build_project, insert, video_track

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FRAME = 1 / 30

def make_clip(clip_id: str = "a", asset_id: str = "asset", source_start: float = 0, source_end: float = 2,
              timeline_in: float = 0, volume: float = 1.0, speed: float = 1.0, audio_lead: float = 0) -> Clip:
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

def test_a_join_near_a_file_nobody_listened_to_for_voicing_is_left_out() -> None:
    # Its talking would be silence to the cut and read as a voice out of place.
    first = make_clip("a", "heard", source_start=0, source_end=2, timeline_in=0)
    second = make_clip("b", "heard", source_start=2, source_end=4, timeline_in=2)
    third = make_clip("c", "unheard", source_start=0, source_end=2, timeline_in=4)
    fourth = make_clip("d", "heard", source_start=4, source_end=10, timeline_in=6)
    # The joins into and out of it are left out; the one two seconds before it is not.
    assert sync.joins([first, second, third, fourth], {"heard": [(0.0, 10.0)]}) == [2.0]

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

def test_agreeing_equally_well_at_several_lags_takes_the_smallest_move() -> None:
    assert sync.lag_at([(0.0, 3.5)], [(-1.0, 10.0)], 2.0) == 0.0

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
    assert found["frame"] == round(FRAME, 4)

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

def make_finished_job(project_id: str, sync_data: dict) -> Job:
    """Store a completed render with what its sound check found.

    Args:
        project_id: The project it was rendered from.
        sync_data: The check's result.

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
