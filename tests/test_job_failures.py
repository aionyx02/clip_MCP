"""Tests for what a job that did not complete says ended it (#107).

The kind is decided where the cause is known — the worker that went, the
file that cannot be read, FFmpeg's own failure, the cancellation — and never
read back out of the error's words, so each test makes a job end one way and
checks the kind recorded, with the message left as it was.
"""

import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app import runtime
from app.engine import renderer
from app.engine.ffmpeg import FFmpegFailed, OperationCancelled, run_ffmpeg
from app.engine.renderer import SPEC_FILE_NAME, JobManager, run_worker
from app.models.job import FailureKind, Job, JobKind, JobStatus
from app.models.media import Asset
from app.storage.repo import Repository
from app.tools import job_queue, rendering
from app.tools.catalog import import_asset
from app.tools.rendering import _read_from, render_project
from helpers import build_project, finished, insert, video_track

@pytest.fixture
def store(tmp_path: Path) -> Repository:
    """A repository of its own, so no other job is admitted when a worker finishes."""
    return Repository(str(tmp_path / "jobs.db"))

def queued(store: Repository, tmp_path: Path, spec: Dict[str, Any], **fields) -> Job:
    """Store a queued job with its specification written, as `start_job` leaves it, without a worker.

    Args:
        store: Repository receiving the job.
        tmp_path: Folder the job's work directory goes in.
        spec: The job's specification.
        **fields: Other job fields, such as `kind` or `asset_id`.

    Returns:
        The stored job.
    """
    job = Job(work_dir=str(tmp_path / "job"), **fields)
    os.makedirs(job.work_dir, exist_ok=True)
    with open(os.path.join(job.work_dir, SPEC_FILE_NAME), "w", encoding="utf-8") as handle:
        json.dump(spec, handle)
    store.add_job(job)
    return job

def render_spec(tmp_path: Path, source: Path) -> Dict[str, Any]:
    """A render whose one FFmpeg run reads `source` and nothing else."""
    return {
        "command": ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(source), "-f", "null", "-"],
        "duration_seconds": 1.0, "pieces": [], "render_path": str(tmp_path / "out.mp4"), "inputs": [str(source)],
    }

def ended(store: Repository, job: Job) -> Job:
    """Run a queued job's worker in this process and read back how it ended."""
    run_worker(store, job.job_id)
    return store.get_job(job.job_id)

def without_the_address(message: str) -> str:
    """FFmpeg's message with the input's address taken out, which changes on every run."""
    return re.sub(r"@ [0-9a-fA-F]+", "@ <address>", message)

def read_the_voices(path: str, folder: Path) -> None:
    """Read a file's sound for the voices, the way an analysis does."""
    from app.engine import diarize
    diarize.read_samples(path)

def read_the_beat(path: str, folder: Path) -> None:
    """Read a file's sound for its beat, the way an analysis does."""
    from app.engine import rhythm
    rhythm._decode(path, "ffmpeg")

def cut_the_sound(path: str, folder: Path) -> None:
    """Cut a stretch of a file's sound, the way an analysis does."""
    from app.engine import listen
    listen.cut_sound(path, 0.0, 1.0, str(folder / "out.wav"))

def read_a_frame(path: str, folder: Path) -> None:
    """Read a frame of a file, the way an analysis does."""
    from app.engine import frames
    frames.extract_frame(path, 0.0)

def read_the_metadata(path: str, folder: Path) -> None:
    """Read a file's metadata, the way an analysis does."""
    from app.engine import probe
    probe.probe_file(path)

def test_ffmpeg_failing_with_its_file_there_is_a_render_failure(store: Repository, tmp_path: Path) -> None:
    source = tmp_path / "not-a-video.mp4"
    source.write_text("these are words, not a picture", encoding="utf-8")
    job = ended(store, queued(store, tmp_path, render_spec(tmp_path, source)))
    assert (job.status, job.failure_kind) == (JobStatus.FAILED, FailureKind.RENDER)
    assert job.error_message

def test_ffmpeg_failing_on_a_file_that_is_gone_is_an_input_failure(store: Repository, tmp_path: Path) -> None:
    job = ended(store, queued(store, tmp_path, render_spec(tmp_path, tmp_path / "moved away.mp4")))
    assert (job.status, job.failure_kind) == (JobStatus.FAILED, FailureKind.INPUT)
    # The message is FFmpeg's own, as it always was: only the kind is new.
    assert job.error_message and "could not read" not in job.error_message

def test_a_file_taken_away_while_the_job_reads_it_is_put_down_to_the_file(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "talk.mp4"
    source.write_bytes(b"\0")

    def taken(job, spec, context, repo) -> None:
        source.unlink()
        raise RuntimeError("the program reading it stopped")

    monkeypatch.setattr(renderer, "_run_render", taken)
    job = ended(store, queued(store, tmp_path, render_spec(tmp_path, source)))
    assert (job.failure_kind, job.error_message) == (FailureKind.INPUT, "the program reading it stopped")

def test_an_analysis_whose_file_is_taken_away_while_it_runs_is_put_down_to_the_file(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "talk.mp4"
    source.write_bytes(b"\0")
    store.save_asset(Asset(id="talk", path=str(source), duration=10, has_video=True, has_audio=True))

    def taken(*args, **kwargs):
        source.unlink()
        raise RuntimeError("the program reading it stopped")

    monkeypatch.setattr(renderer, "analyze_media", taken)
    job = ended(store, queued(store, tmp_path, {"transcribe": False}, kind=JobKind.ANALYZE, asset_id="talk"))
    # It was there when the analysis began; what the library holds now is what was looked at.
    assert (job.failure_kind, job.error_message) == (FailureKind.INPUT, "the program reading it stopped")

def test_ffmpeg_failing_to_read_a_photo_a_mix_or_a_stretch_of_sound_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.engine import analysis, loudness
    from app.models.media import Transcript

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a picture")
    with pytest.raises(FFmpegFailed):
        loudness.measure(str(broken))
    asset = Asset(id="photo", path=str(broken), has_video=True, has_audio=False)
    with pytest.raises(FFmpegFailed):
        analysis.scan_photo(asset, str(tmp_path / "work"), lambda *_: None, lambda: False)
    monkeypatch.setattr(analysis, "_unsure_stretches", lambda segments, duration: [(0.0, 1.0)])
    with pytest.raises(FFmpegFailed):
        analysis.refine_unsure_lines(str(broken), Transcript(language="zh", model="small"), 1.0, None, [], "large",
                                     lambda *_: None, lambda: False, str(tmp_path / "refine"))

def test_ffmpeg_failing_to_decode_sound_for_the_voices_says_so(tmp_path: Path) -> None:
    from app.engine import diarize

    broken = tmp_path / "broken.wav"
    broken.write_text("not sound", encoding="utf-8")
    with pytest.raises(FFmpegFailed):
        diarize.read_samples(str(broken))

def test_ffmpeg_failing_to_decode_sound_for_the_beat_says_so(tmp_path: Path) -> None:
    from app.engine import rhythm

    broken = tmp_path / "broken.mp3"
    broken.write_text("not sound", encoding="utf-8")
    with pytest.raises(FFmpegFailed):
        rhythm._decode(str(broken), "ffmpeg")

def test_ffmpeg_failing_to_cut_a_stretch_of_sound_says_so(tmp_path: Path) -> None:
    from app.engine import listen

    broken = tmp_path / "broken.mp4"
    broken.write_text("not sound", encoding="utf-8")
    with pytest.raises(FFmpegFailed):
        listen.cut_sound(str(broken), 0.0, 1.0, str(tmp_path / "out.wav"))

def test_ffmpeg_failing_to_read_a_frame_says_so(tmp_path: Path) -> None:
    from app.engine import frames

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a picture")
    with pytest.raises(FFmpegFailed):
        frames.extract_frame(str(broken), 0.0)

def test_ffprobe_failing_to_read_a_file_says_so(tmp_path: Path) -> None:
    from app.engine import probe

    broken = tmp_path / "broken.mp4"
    broken.write_text("not media", encoding="utf-8")
    with pytest.raises(FFmpegFailed):
        probe.probe_file(str(broken))

def test_a_frame_with_no_picture_behind_a_clean_exit_is_not_ffmpeg_failing(tmp_path: Path) -> None:
    from app.engine import frames

    source = tmp_path / "tiny.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=32x32:rate=5",
         "-pix_fmt", "yuv420p", str(source)],
        stdin=subprocess.DEVNULL, capture_output=True, check=True,
    )
    # Past the end FFmpeg exits 0 and writes nothing: a clean run that found no picture.
    with pytest.raises(RuntimeError) as raised:
        frames.extract_frame(str(source), 5.0)
    assert not isinstance(raised.value, FFmpegFailed)

def test_a_frame_ffmpeg_failed_on_is_not_kept_for_what_it_wrote_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.engine import frames

    def failed_halfway(*args, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args, 1, stdout=b"\x89PNG half a picture", stderr=b"broken frame")

    monkeypatch.setattr(frames.subprocess, "run", failed_halfway)
    # Whatever reached the pipe before FFmpeg failed is not a picture to keep.
    with pytest.raises(FFmpegFailed):
        frames.extract_frame(str(tmp_path / "talk.mp4"), 1.0)

@pytest.mark.parametrize(
    "read_the_file",
    [
        pytest.param(read_the_voices, id="diarize.read_samples"),
        pytest.param(read_the_beat, id="rhythm._decode"),
        pytest.param(cut_the_sound, id="listen.cut_sound"),
        pytest.param(read_a_frame, id="frames.extract_frame"),
        pytest.param(read_the_metadata, id="probe.probe_file"),
    ],
)
def test_an_analysis_ffmpeg_cannot_read_is_a_render_failure_wherever_it_reads(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_the_file,
) -> None:
    source = tmp_path / "talk.mp4"
    source.write_text("these are words, not sound", encoding="utf-8")
    # What the tool says when called on its own is what the job has to record.
    with pytest.raises(FFmpegFailed) as raised:
        read_the_file(str(source), tmp_path)
    store.save_asset(Asset(id="talk", path=str(source), duration=10, has_video=True, has_audio=True))

    def analyze_by_reading(asset, *args, **kwargs) -> None:
        read_the_file(asset.path, tmp_path)

    monkeypatch.setattr(renderer, "analyze_media", analyze_by_reading)
    job = ended(store, queued(store, tmp_path, {"transcribe": False}, kind=JobKind.ANALYZE, asset_id="talk"))
    # The file was still there when it ended, so the failure is FFmpeg's and not the input's.
    assert (job.status, job.failure_kind) == (JobStatus.FAILED, FailureKind.RENDER)
    assert job.error_message
    assert without_the_address(job.error_message) == without_the_address(str(raised.value))

def test_an_analysis_of_a_file_that_is_gone_is_an_input_failure(store: Repository, tmp_path: Path) -> None:
    store.save_asset(Asset(id="gone", path=str(tmp_path / "gone.mp4"), duration=10, has_video=True, has_audio=True))
    job = ended(store, queued(store, tmp_path, {"transcribe": False}, kind=JobKind.ANALYZE, asset_id="gone"))
    assert (job.status, job.failure_kind) == (JobStatus.FAILED, FailureKind.INPUT)
    assert job.error_message.startswith(f"could not read {tmp_path / 'gone.mp4'}")

def test_an_analysis_of_an_asset_no_longer_in_the_library_is_an_input_failure(
    store: Repository, tmp_path: Path,
) -> None:
    job = ended(store, queued(store, tmp_path, {"transcribe": False}, kind=JobKind.ANALYZE, asset_id="forgotten"))
    assert (job.failure_kind, job.error_message) == (FailureKind.INPUT, "asset forgotten not found")

def test_an_error_of_no_known_kind_is_left_without_one(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "talk.mp4"
    source.write_bytes(b"\0")

    def broken(job, spec, context, repo) -> None:
        raise RuntimeError("something nobody foresaw")

    monkeypatch.setattr(renderer, "_run_render", broken)
    job = ended(store, queued(store, tmp_path, render_spec(tmp_path, source)))
    assert (job.status, job.failure_kind, job.error_message) == (
        JobStatus.FAILED, None, "something nobody foresaw",
    )

def test_a_job_stopped_while_running_is_cancelled(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stopped(job, spec, context, repo) -> None:
        raise OperationCancelled()

    monkeypatch.setattr(renderer, "_run_render", stopped)
    job = ended(store, queued(store, tmp_path, render_spec(tmp_path, tmp_path / "talk.mp4")))
    assert (job.status, job.failure_kind, job.error_message) == (JobStatus.CANCELLED, FailureKind.CANCELLED, None)

def test_a_job_cancelled_while_waiting_is_cancelled(store: Repository, tmp_path: Path) -> None:
    job = queued(store, tmp_path, {})
    cancelled = JobManager(store).cancel_job(job.job_id, wait_seconds=0)
    assert (cancelled.status, cancelled.failure_kind) == (JobStatus.CANCELLED, FailureKind.CANCELLED)

def test_a_worker_that_stopped_reporting_is_a_worker_failure(store: Repository, tmp_path: Path) -> None:
    job = queued(store, tmp_path, {}, status=JobStatus.RUNNING)
    when = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    store._conn.execute("UPDATE jobs SET data = json_set(data, '$.updated_at', ?) WHERE id = ?", (when, job.job_id))
    failed = JobManager(store).get_job(job.job_id)
    assert (failed.failure_kind, failed.error_message) == (FailureKind.WORKER, renderer.ABANDONED)

def test_a_job_asked_to_stop_whose_worker_then_went_is_cancelled(store: Repository, tmp_path: Path) -> None:
    job = queued(store, tmp_path, {}, status=JobStatus.RUNNING)
    JobManager(store).cancel_job(job.job_id, wait_seconds=0)
    when = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    store._conn.execute("UPDATE jobs SET data = json_set(data, '$.updated_at', ?) WHERE id = ?", (when, job.job_id))
    stopped = JobManager(store).get_job(job.job_id)
    assert (stopped.status, stopped.failure_kind, stopped.error_message) == (
        JobStatus.CANCELLED, FailureKind.CANCELLED, None)

def test_a_worker_that_could_not_start_is_a_worker_failure(
    store: Repository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused(*args, **kwargs) -> None:
        raise OSError("no more processes")

    monkeypatch.setattr(subprocess, "Popen", refused)
    job = Job(kind=JobKind.ANALYZE, work_dir=str(tmp_path / "job"))
    JobManager(store).start_job(job, {})
    failed = store.get_job(job.job_id)
    assert failed.failure_kind == FailureKind.WORKER
    assert failed.error_message.startswith("could not start worker")

def test_a_worker_that_crashed_is_a_worker_failure(store: Repository, tmp_path: Path) -> None:
    job = queued(store, tmp_path, {})
    os.remove(os.path.join(job.work_dir, SPEC_FILE_NAME))
    with pytest.raises(FileNotFoundError):
        renderer.main([store.db_path, job.job_id])
    failed = store.get_job(job.job_id)
    assert failed.failure_kind == FailureKind.WORKER
    assert failed.error_message.startswith("worker crashed")

def test_ffmpeg_failing_raises_its_own_error(tmp_path: Path) -> None:
    progress: List[float] = []
    with pytest.raises(FFmpegFailed):
        run_ffmpeg(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(tmp_path / "nothing.mp4"), "-f", "null", "-"],
                   str(tmp_path / "ffmpeg.log"), 1.0, progress.append, lambda: False)

def test_get_job_says_what_ended_a_job_and_nothing_for_one_that_finished() -> None:
    gone = Job(kind=JobKind.RENDER, status=JobStatus.FAILED, error_message="No such file or directory",
               failure_kind=FailureKind.INPUT)
    done = Job(kind=JobKind.ANALYZE, status=JobStatus.COMPLETED, progress=1.0)
    for job in (gone, done):
        runtime.repo.add_job(job)
    listed = job_queue.get_job([gone.job_id, done.job_id])["jobs"]
    assert (listed[0]["failure_kind"], listed[0]["error_message"]) == ("input", "No such file or directory")
    assert "failure_kind" not in listed[1]

def test_a_job_reads_footage_and_made_files_apart_and_not_what_it_writes_itself() -> None:
    commands = [["ffmpeg", "-i", "talk.mp4", "-i", "cache/picture.mp4", "-i", "cache/new.mp4", "out.mp4"],
                ["ffmpeg", "-i", "talk.mp4", "cache/new.mp4.part"]]
    # A picture already in the cache is read; one this job makes is not there to be read before it runs.
    assert _read_from(commands, ["talk.mp4", "unused.mp4"], ["cache/new.mp4.part", "cache/new.mp4"]) == (
        ["talk.mp4"], ["cache/picture.mp4"])

def test_ffmpeg_failing_on_a_made_file_that_is_gone_is_put_down_to_it(store: Repository, tmp_path: Path) -> None:
    source, picture = tmp_path / "talk.mp4", tmp_path / "picture.mp4"
    source.write_bytes(b"footage")
    spec = {**render_spec(tmp_path, picture), "inputs": [str(source)], "made": [str(picture)]}
    job = ended(store, queued(store, tmp_path, spec, kind=JobKind.RENDER))
    # The footage is there; the picture the cache held is not, and rendering again makes it.
    assert (job.status, job.failure_kind) == (JobStatus.FAILED, FailureKind.MADE)

def test_footage_that_is_gone_outranks_a_made_file_that_is_gone(store: Repository, tmp_path: Path) -> None:
    source, picture = tmp_path / "talk.mp4", tmp_path / "picture.mp4"
    spec = {**render_spec(tmp_path, source), "made": [str(picture)]}
    job = ended(store, queued(store, tmp_path, spec, kind=JobKind.RENDER))
    # Starting it again would make the picture and still not find the footage.
    assert (job.status, job.failure_kind) == (JobStatus.FAILED, FailureKind.INPUT)

def test_a_render_is_queued_knowing_which_files_it_reads(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 1)], width=640, height=360)
    specs: List[Dict[str, Any]] = []

    def queue(job: Job, spec: Dict[str, Any]) -> Job:
        specs.append(spec)
        return job

    footage = runtime.repo.get_assets([wide])[wide].path
    # A preview made once leaves its pictures in the cache for the next one to read.
    assert finished(render_project(project, loudness_target=None))["status"] == "completed"
    monkeypatch.setattr(runtime.job_manager, "start_job", queue)
    render_project(project, loudness_target=None)
    # Without loudness there is no mix: the final command alone reads the cached picture.
    assert specs[0]["inputs"] == [footage] and not specs[0]["pieces"]
    assert specs[0]["made"] and all(path.startswith(runtime.WORKSPACE_DIR) for path in specs[0]["made"])
    # A finished video has no pieces: its one command reads the footage itself.
    render_project(project, is_preview=False, loudness_target=None)
    assert (specs[1]["inputs"], specs[1]["made"]) == ([footage], [])

def test_the_words_drawn_on_are_a_made_file_the_render_reads(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    card = {"action": "add_title_card", "track_id": "main", "clip_id": "t", "title": "第一章", "before_clip_id": "a"}
    project = build_project([video_track(), insert("a", wide, 0, 1), card], width=640, height=360)
    specs: List[Dict[str, Any]] = []
    monkeypatch.setattr(runtime.job_manager, "start_job", lambda job, spec: specs.append(spec) or job)
    render_project(project, loudness_target=None)
    # Read through a filter rather than `-i`, so only the render knows to list it.
    assert "subtitles.ass" in [os.path.basename(path) for path in specs[0]["made"]]

def test_a_photo_only_the_preview_s_own_picture_reads_is_one_of_its_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    path = tmp_path / "photo.png"
    Image.new("RGB", (320, 240), (220, 30, 30)).save(path)
    photo = import_asset(str(path))["id"]
    project = build_project([video_track(), insert("a", photo, 0, 2)], width=320, height=240)
    specs: List[Dict[str, Any]] = []
    monkeypatch.setattr(runtime.job_manager, "start_job", lambda job, spec: specs.append(spec) or job)
    render_project(project, loudness_target=None)
    # A photo has no sound for the final command to read: only the command making its picture opens it.
    assert specs[0]["pieces"] and specs[0]["inputs"] == [runtime.repo.get_assets([photo])[photo].path]
    # And the picture it is about to make is not a made file it reads: that one is not there yet.
    assert specs[0]["made"] == []

def test_playback_is_prepared_knowing_which_files_it_reads(media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wide = import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", wide, 0, 1)], width=640, height=360)
    specs: List[Dict[str, Any]] = []

    def queue(job: Job, spec: Dict[str, Any]) -> Job:
        specs.append(spec)
        return job

    monkeypatch.setattr(rendering, "PREPARE_PLAYBACK", True)
    monkeypatch.setattr(rendering, "_make_room", lambda: None)
    monkeypatch.setattr(rendering, "_being_prepared", set)
    # Whatever an earlier test left in the shared folder, the copy to play from is made again.
    monkeypatch.setattr(rendering.playback, "missing", lambda project, assets, present: [wide])
    monkeypatch.setattr(runtime.job_manager, "start_job", queue)
    rendering.prepare_playback(runtime.repo.get_project(project))
    # The copy to play from, and the loudness measured from a mix the job writes and reads back itself.
    assert [spec["steps"][-1]["stage"] for spec in specs] == ["making a copy to play from", "measuring loudness"]
    footage = runtime.repo.get_assets([wide])[wide].path
    assert [(spec["inputs"], spec["made"]) for spec in specs] == [([footage], [])] * 2
