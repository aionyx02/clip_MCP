"""What the browser plays a cut from: proxies, repaired sound, a measured loudness, and a description."""

import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import server
from app.engine import playback, renderer
from app.models.job import Job, JobKind, JobStatus
from helpers import build_project, edit, insert, video_track


def wait_for(job_ids) -> None:
    """Wait for jobs to end, and fail if any did not complete."""
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        states = server.get_job(job_ids)["jobs"]
        if all(state["status"] in {"completed", "failed", "cancelled"} for state in states):
            assert all(state["status"] == "completed" for state in states), states
            return
        time.sleep(0.3)
    raise AssertionError(f"jobs {job_ids} did not finish")


@pytest.fixture
def preparing():
    """Make what the browser needs after every edit, as the server does outside tests."""
    server.PREPARE_PLAYBACK = True
    yield
    server.PREPARE_PLAYBACK = False


def stream(path: str) -> dict:
    """The first picture stream of a file, as ffprobe sees it."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", path], capture_output=True)
    streams = json.loads(probe.stdout)["streams"]
    return next(item for item in streams if item["codec_type"] == "video")


def test_a_cut_is_described_for_the_browser_and_its_copies_are_made(preparing, media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 3), insert("b", asset, 4, 7)], name="播放")
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "b",
                    "transition_in": {"kind": "dissolve", "seconds": 0.5}},
                   {"action": "set_clip_audio", "track_id": "main", "clip_id": "a", "cleanup": {"hiss": True}}])
    first = server.playback_of(server.repo.get_project(project))
    assert first["width"] == 1920 and first["duration"] == pytest.approx(6.0)
    clips = first["tracks"][0]["clips"]
    assert [clip["id"] for clip in clips] == ["a", "b"]
    assert clips[1]["transition"]["kind"] == "dissolve" and clips[1]["box"] == [0, 0, 1920, 1080]
    assert first["waiting"]  # nothing made yet, and asking started making it

    jobs = [job.job_id for job in server.repo.jobs_to_show(datetime.now(timezone.utc) - timedelta(minutes=5))[0]
            if job.kind == JobKind.PREPARE]
    wait_for(jobs)
    ready = server.playback_of(server.repo.get_project(project))
    assert ready["waiting"] == []
    proxy = server.WORKSPACE_DIR + "/cache/proxies/" + ready["tracks"][0]["clips"][0]["proxy"].split("/")[-1]
    # Never bigger than the file: this one is 640x360, so it is copied at its own size.
    assert (stream(proxy)["width"], stream(proxy)["height"]) == (640, 360)
    assert ready["tracks"][0]["clips"][0]["repaired"].startswith("/sound/")
    assert ready["gain_db"] is not None


def test_the_editor_serves_the_description_and_the_copies(preparing, media: Path) -> None:
    from starlette.testclient import TestClient
    from app.ui import app as editor

    asset = server.import_asset(str(media / "tall.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 2)], name="播放編輯器")
    client = TestClient(editor.create_app())
    described = client.get(f"/api/projects/{project}/playback").json()
    jobs = [job.job_id for job in server.repo.jobs_to_show(datetime.now(timezone.utc) - timedelta(minutes=5))[0]
            if job.kind == JobKind.PREPARE]
    wait_for(jobs)
    described = client.get(f"/api/projects/{project}/playback").json()
    url = described["tracks"][0]["clips"][0]["proxy"]
    part = client.get(url, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    assert client.get("/proxy/..%2Fclip_mcp.db").status_code == 404


def test_somebody_waiting_to_watch_goes_ahead_of_the_other_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(renderer.resources, "max_concurrent_jobs", lambda: 1)
    monkeypatch.setattr(renderer.resources, "spare_bytes", lambda: None)
    now = datetime.now(timezone.utc)
    older = Job(kind=JobKind.ANALYZE, created_at=now - timedelta(seconds=10))
    behind = Job(kind=JobKind.PREPARE, priority=1, created_at=now - timedelta(seconds=20))
    watched = Job(kind=JobKind.PREPARE, priority=-1, created_at=now)
    renderer.JobManager(server.repo)._plan([behind, older, watched])
    assert watched.admitted_at is not None
    assert older.admitted_at is None and behind.admitted_at is None
    assert all(job.status == JobStatus.QUEUED for job in (older, behind, watched))


def test_a_version_is_got_ready_in_the_background_but_never_while_exporting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(renderer.resources, "max_concurrent_jobs", lambda: 4)
    monkeypatch.setattr(renderer.resources, "spare_bytes", lambda: None)
    now = datetime.now(timezone.utc)
    export = Job(kind=JobKind.RENDER, status=JobStatus.RUNNING, admitted_at=now)
    background = Job(kind=JobKind.PREPARE, priority=1, created_at=now - timedelta(seconds=20))
    song = Job(kind=JobKind.ANALYZE, priority=1, created_at=now)
    watched = Job(kind=JobKind.PREPARE, priority=-1, created_at=now)
    renderer.JobManager(server.repo)._plan([export, background, song, watched])
    assert background.admitted_at is None and background.stage == renderer.EXPORT_FIRST
    # Nothing waits behind it, and somebody waiting to watch still goes ahead.
    assert song.admitted_at is not None and watched.admitted_at is not None
    renderer.JobManager(server.repo)._plan([background])
    assert background.admitted_at is not None


def test_going_back_to_a_version_gets_it_ready_to_play(media: Path, monkeypatch: pytest.MonkeyPatch,
                                                         kept_history) -> None:
    from helpers import build_project, edit, insert, video_track

    asked = []
    project = build_project([video_track(), insert("a", server.import_asset(str(media / "wide.mp4"))["id"], 0, 4)],
                            name="準備好")
    first = server.project_history(project)["versions"][0]["commit"]
    edit(project, [{"action": "delete_clip", "track_id": "main", "clip_id": "a"}])
    monkeypatch.setattr(server, "prepare_playback", lambda cut, urgent=False: asked.append((cut.id, urgent)))
    server.restore_version(project, first, server.repo.get_project(project).version)
    assert asked == [(project, False)]


def test_a_proxy_is_named_after_the_file_as_it_is(media: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "會變的檔.mp4"
    shutil.copy(media / "wide.mp4", copy)
    asset = server.repo.get_assets([server.import_asset(str(copy))["id"]])
    asset = next(iter(asset.values()))
    first = playback.proxy_name(asset)
    shutil.copy(media / "tall.mp4", copy)
    assert playback.proxy_name(asset) != first


def test_captions_break_in_the_editor_where_they_break_in_a_render() -> None:
    from decimal import Decimal

    from app.engine.subtitles import build_ass, drawn_captions
    from app.models.timeline import CaptionStyle, CueWord, PlacedCue

    words = [CueWord(start=Decimal(index) / 2, end=Decimal(index) / 2 + Decimal("0.4"), text=text)
             for index, text in enumerate(["今天", "我們", "來看", "這台", "抽杯架", "怎麼", "裝"])]
    cue = PlacedCue(cue_id="c1", start=Decimal(0), end=Decimal("3.5"), text="".join(word.text for word in words),
                    words=words, speaker="S2")
    style = CaptionStyle(karaoke=True, single_line=False, speaker_mark="both", size_fraction=0.2)
    drawn = drawn_captions([cue], 1080, 1920, style)
    event = drawn["events"][0]
    # The same lines the render's subtitle file breaks it into, each word with its time.
    dialogue = build_ass([cue], 1080, 1920, style).splitlines()[-1]
    assert len(event["lines"]) == dialogue.count(r"\N") + 1 > 1
    assert event["lines"][0][0] == ["今天", 0.0, 0.4]
    assert event["prefix"] == "S2：" and event["colour"] == "#FFFFFF"
