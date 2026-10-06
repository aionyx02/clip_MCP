"""Looking at and listening to what a render made, not only what the plan meant it to make."""

import time
from pathlib import Path

import pytest

from app import server
from app.engine import analysis
from app.models.media import MediaAnalysis, Transcript, TranscriptSegment, TranscriptWord
from helpers import build_project, caption, edit, insert, video_track


def finished(job: dict) -> str:
    """Wait for a render to end, and give back its job ID."""
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = server.get_job([job["job_id"]])["jobs"][0]
        if state["status"] in {"completed", "failed", "cancelled"}:
            assert state["status"] == "completed", state
            return job["job_id"]
        time.sleep(0.3)
    raise AssertionError(f"render {job['job_id']} did not finish")


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory, media: Path) -> dict:
    """A two-shot cut with a dissolve between them and captions burned in, rendered as a preview."""
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    server.repo.save_analysis(MediaAnalysis(asset_id=asset, duration=10.0, transcript=Transcript(
        language="zh", model="test", segments=[
            TranscriptSegment(start=1.0, end=2.5, text="第一句", words=[TranscriptWord(start=1.0, end=2.5, text="第一句")]),
            TranscriptSegment(start=6.0, end=7.5, text="第二句", words=[TranscriptWord(start=6.0, end=7.5, text="第二句")]),
        ],
    )))
    project = build_project([video_track(), insert("a", asset, 0, 4), insert("b", asset, 5, 9)])
    edit(project, [{"action": "set_clip_look", "track_id": "main", "clip_id": "b",
                    "transition_in": {"kind": "dissolve", "seconds": 1.0}}])
    caption(project)
    job = server.render_project(project, is_preview=True, burn_subtitles=True)
    return {"project": project, "job_id": finished(job)}


def test_a_render_is_looked_at_around_every_transition(rendered: dict) -> None:
    seen = server.view_render(rendered["job_id"], at="transitions")
    listing = seen.content[0].text
    # The cut is at 4s and the dissolve ends on it: before, inside and after.
    assert "before the dissolve" in listing and "middle of the dissolve" in listing and "after the dissolve" in listing
    assert "3.50s" in listing and "4.15s" in listing
    assert seen.content[1].type == "image"


def test_a_render_is_looked_at_on_every_caption(rendered: dict) -> None:
    listing = server.view_render(rendered["job_id"], at="captions").content[0].text
    assert "c1 第一句" in listing and "c2 第二句" in listing
    assert "no captions burned in" not in listing


def test_a_render_says_when_the_project_has_moved_on_since(rendered: dict) -> None:
    edit(rendered["project"], [{"action": "rename_project", "name": "改過名字"}])
    listing = server.view_render(rendered["job_id"], times=[1.0]).content[0].text
    assert "edited since this render" in listing


def test_a_render_that_has_not_finished_is_not_looked_at() -> None:
    with pytest.raises(ValueError, match="not found"):
        server.view_render("no-such-job")


def test_a_render_is_heard_with_its_levels(rendered: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    def model(device, path, duration, language, prompt, on_progress, is_cancelled, on_measured=None,
              model_name=None):
        return Transcript(language="zh", model="test", segments=[])

    monkeypatch.setattr(analysis, "_run_whisper", model)
    heard = server.listen_again(start=0.0, end=3.0, job_id=rendered["job_id"])
    assert heard["levels"]["every_seconds"] == 0.5 and len(heard["levels"]["db"]) == 6
    # The footage's tone is there: well above a gap.
    assert min(heard["levels"]["db"]) > -40
    assert heard["captions"] and heard["captions"][0].endswith("第一句")
    with pytest.raises(ValueError, match="either"):
        server.listen_again(start=0.0, end=3.0)
