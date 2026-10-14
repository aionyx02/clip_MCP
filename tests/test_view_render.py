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


def test_a_render_lists_its_joins_without_frames(rendered: dict) -> None:
    seen = server.inspect_join(rendered["job_id"])
    listing = seen.content[0].text
    assert "#1 " in listing and "(4.00s) dissolve: clip a: wide.mp4 0.00–4.00s → clip b: wide.mp4 5.00–9.00s" in listing
    assert len(seen.content) == 1


def test_a_join_is_looked_at_frame_by_frame_with_its_words_and_sound(rendered: dict) -> None:
    seen = server.inspect_join(rendered["job_id"], joins=[1])
    text = seen.content[0].text
    assert "a row of 7 up to the cut, then 7 from it" in text
    # 「第一句」 ends at 2.5s, before the stretch; 「第二句」 starts a second into clip b.
    assert "words: ‖ [clip b] +1.00 第二句" in text
    levels = text.split("levels: ")[1].splitlines()[0]
    assert levels.startswith("dB every 0.1s from -1.50s") and " ‖ " in levels
    assert Path(text.split("sound: ")[1].splitlines()[0]).exists()
    assert seen.content[1].type == "image"


def test_a_join_is_found_by_time_and_a_number_not_in_the_list_is_refused(rendered: dict) -> None:
    assert server.inspect_join(rendered["job_id"], at=[3.2]).content[0].text.startswith("#1 ")
    with pytest.raises(ValueError, match="has 1 joins; there is no #2"):
        server.inspect_join(rendered["job_id"], joins=[2])
    with pytest.raises(ValueError, match="more than the 36 a sheet holds"):
        server.inspect_join(rendered["job_id"], joins=[1], seconds=3, fps=8)


def test_a_render_is_looked_at_on_every_caption(rendered: dict) -> None:
    listing = server.view_render(rendered["job_id"], at="captions").content[0].text
    assert "c1 第一句" in listing and "c2 第二句" in listing
    assert "no captions burned in" not in listing


def test_a_render_says_when_the_project_has_moved_on_since(rendered: dict) -> None:
    edit(rendered["project"], [{"action": "rename_project", "name": "改過名字"}])
    listing = server.view_render(rendered["job_id"], times=[1.0]).content[0].text
    assert "edited since this render" in listing


def test_a_render_lists_the_joins_it_was_made_with_after_the_cut_has_changed(rendered: dict) -> None:
    edit(rendered["project"], [{"action": "trim_clip", "track_id": "main", "clip_id": "a",
                                "new_source_range": {"start": 0, "end": 3}, "ripple": True}])
    listing = server.inspect_join(rendered["job_id"]).content[0].text
    assert "these are the joins of the version rendered" in listing and "(4.00s) dissolve" in listing


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


def test_a_preview_breaks_its_captions_where_the_finished_video_does(rendered: dict) -> None:
    # Laid out at the delivered size and shrunk by libass: laid out at the preview's own size,
    # rounded pixels gave a line room for 13.47 characters there and 13.35 in the finished video.
    project = server.repo.get_project(rendered["project"])
    job = server.repo.get_job(rendered["job_id"])
    script = (Path(job.work_dir) / "subtitles.ass").read_text(encoding="utf-8")
    assert f"PlayResX: {project.width}" in script and f"PlayResY: {project.height}" in script
    assert min(project.width, project.height) > server.PREVIEW_SHORT_SIDE


def test_a_cut_is_read_back_as_a_script(rendered: dict) -> None:
    script = server.read_cut(rendered["project"])["script"]
    assert len(script) == 2 and script[0].startswith("0:00–") and "| clip a | wide.mp4 0.0–" in script[0]
    assert "said: 「第一句」" in script[0] and "captions: 第一句" in script[0]
    assert "said: 「第二句」" in script[1] and "captions: 第二句" in script[1]


def test_a_join_shows_the_captions_on_screen_around_it(rendered: dict) -> None:
    text = server.inspect_join(rendered["job_id"], joins=[1]).content[0].text
    # The captions are what somebody read and could correct; they sit beside the words.
    assert "captions: +1.00 第二句" in text
