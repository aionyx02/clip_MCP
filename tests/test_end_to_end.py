"""One whole job, from a folder of footage to a finished file.

The individual features have their own tests; this one checks they still work
together, which is where a change to the shared frame and sample maths would
show up.
"""

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from app.models.media import MediaAnalysis, Transcript, TranscriptSegment, TranscriptWord
from app.server import (
    generate_subtitles, get_job, get_project, get_subtitles,
    import_folder, preview_project, render_project, repo,
)
from helpers import build_project, edit, loudness, render

@pytest.fixture
def footage(tmp_path_factory: pytest.TempPathFactory, media: Path, audio_media: Path) -> Path:
    """Lay out a folder of footage the way a user would hand it over.

    Returns:
        The folder path.
    """
    folder = tmp_path_factory.mktemp("job")
    shutil.copy(media / "wide.mp4", folder / "take1.mp4")
    shutil.copy(audio_media / "loud.mp4", folder / "take2.mp4")
    shutil.copy(media / "tall.mp4", folder / "take10.mp4")
    shutil.copy(media / "jingle.wav", folder / "bgm.wav")
    return folder

def test_a_whole_job_from_a_folder_to_a_finished_file(footage: Path, tmp_path: Path) -> None:
    imported = import_folder(str(footage))
    names = [Path(asset["path"]).stem for asset in imported["assets"]]
    assert names == ["bgm", "take1", "take2", "take10"]
    assets = {Path(asset["path"]).stem: asset["id"] for asset in imported["assets"]}

    repo.save_analysis(MediaAnalysis(
        asset_id=assets["take1"], duration=10.0,
        transcript=Transcript(language="zh", model="test", segments=[
            TranscriptSegment(start=1.0, end=3.0, text="開場白",
                              words=[TranscriptWord(text="開場白", start=1.0, end=3.0)]),
            TranscriptSegment(start=5.0, end=7.0, text="結尾話",
                              words=[TranscriptWord(text="結尾話", start=5.0, end=7.0)]),
        ]),
    ))

    project = build_project([
        {"action": "add_track", "track_id": "main", "track_type": "video"},
        {"action": "insert_clip", "track_id": "main", "clip_id": "a", "asset_id": assets["take1"],
         "source_range": {"start": 0, "end": 8}, "video_fade_in": 0.5},
        {"action": "insert_clip", "track_id": "main", "clip_id": "b", "asset_id": assets["take2"],
         "source_range": {"start": 0, "end": 4}, "color": {"saturation": 0.5}},
        {"action": "add_track", "track_id": "cam", "track_type": "video"},
        {"action": "add_clip", "track_id": "cam", "clip_id": "pip", "asset_id": assets["take10"],
         "source_range": {"start": 0, "end": 3}, "timeline_in": 2,
         "layout": {"x": 0.62, "y": 0.06, "width": 0.32, "height": 0.32}},
        {"action": "add_track", "track_id": "music", "track_type": "audio", "duck_under_speech": True},
        {"action": "insert_clip", "track_id": "music", "clip_id": "song", "asset_id": assets["bgm"],
         "source_range": {"start": 0, "end": 4}, "volume": 0.5},
        {"action": "fit_track", "track_id": "music", "fade_in": 1, "fade_out": 2},
    ], width=1080, height=1920)

    # Rearranged into a flashback, then the opening line cut in two.
    edit(project, [
        {"action": "split_clip", "track_id": "main", "clip_id": "a", "new_clip_id": "a2", "at_source": 4},
        {"action": "reorder_clip", "track_id": "main", "clip_id": "b", "before_clip_id": "a"},
        {"action": "fit_track", "track_id": "music", "fade_out": 2},
    ])

    state = get_project(project)
    assert [clip["id"] for clip in state["tracks"][0]["clips"]] == ["b", "a", "a2"]
    assert float(state["duration"]) == 12.0
    music = next(track for track in state["tracks"] if track["id"] == "music")
    assert sum(
        float(clip["source_range"]["end"]) - float(clip["source_range"]["start"])
        for clip in music["clips"]
    ) == 12.0

    cues = generate_subtitles(project)["cues"]
    assert [cue["text"] for cue in cues] == ["開場白", "結尾話"]
    edit(project, [{"action": "set_subtitles", "cues": cues}])
    edit(project, [{"action": "edit_subtitle", "cue_id": "c1", "text": "修正過的開場白"}])
    assert get_subtitles(project)["cues"][0]["text"] == "修正過的開場白"

    lines = preview_project(project).content[0].text.splitlines()
    assert "1080x1920" in lines[0] and "00:12.0 long" in lines[0]
    assert any("inset on track cam" in line for line in lines)
    assert any("audio track music" in line for line in lines)

    # The workflow's own last steps: render in the background and poll until it lands.
    job = render_project(project, is_preview=True, burn_subtitles=True)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = get_job(job["job_id"])
        if state["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.5)
    assert state["status"] == "completed", state
    assert Path(job["output_path"]).stat().st_size > 0

    render(project, tmp_path / "final.mp4")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(tmp_path / "final.mp4")],
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr.decode("utf-8", errors="replace")
    assert float(probe.stdout.decode()) == pytest.approx(12.0, abs=0.1)
    assert loudness(tmp_path / "final.mp4") == pytest.approx(-14.0, abs=1.5)
