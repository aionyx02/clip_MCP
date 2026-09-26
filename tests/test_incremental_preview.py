"""Tests for previews that only render what changed.

Driven through the builder and run the way the job worker runs them — the
cache pieces first, each moved into place once finished, then the render — so
that what is counted as reused really is read back from the cache.
"""

import os
import subprocess
from pathlib import Path

import pytest

from app.engine.builder import FFmpegRenderer
from app.server import _preview_sized, _referenced_assets, import_asset, render_project, repo
from helpers import build_project, edit, insert, level, video_track

def preview(project_id: str, cache: Path, out: Path) -> int:
    """Render a cached preview the way the worker does.

    Args:
        project_id: The project.
        cache: The picture cache.
        out: File to write.

    Returns:
        How many pictures had to be rendered into the cache first.
    """
    project = _preview_sized(repo.get_project(project_id))
    command, pieces = FFmpegRenderer().build_incremental(
        project, _referenced_assets(project), str(out), str(cache), loudness_target=None,
    )
    for piece in pieces:
        result = subprocess.run(piece["command"], capture_output=True)
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")[-600:]
        os.replace(piece["part"], piece["path"])
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")[-600:]
    return len(pieces)

def frames_in(path: Path) -> int:
    """Count a video's frames.

    Args:
        path: The video.

    Returns:
        The number of frames.
    """
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
    ).stdout
    return int(output.strip())

@pytest.fixture
def three_clips(media: Path, audio_media: Path) -> str:
    """A sequence of three clips with sound.

    Returns:
        The project ID.
    """
    wide = import_asset(str(media / "wide.mp4"))["id"]
    loud = import_asset(str(audio_media / "loud.mp4"))["id"]
    return build_project([
        video_track(), insert("a", wide, 0, 2), insert("b", loud, 0, 2), insert("c", wide, 4, 6),
    ], width=1280, height=720)

def test_a_second_preview_renders_nothing_it_rendered_before(three_clips: str, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    assert preview(three_clips, cache, tmp_path / "first.mp4") == 3
    assert preview(three_clips, cache, tmp_path / "second.mp4") == 0
    assert frames_in(tmp_path / "second.mp4") == frames_in(tmp_path / "first.mp4") == 180

def test_changing_one_cut_renders_only_the_clip_it_touched(three_clips: str, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    preview(three_clips, cache, tmp_path / "before.mp4")
    edit(three_clips, [{"action": "trim_clip", "track_id": "main", "clip_id": "c", "new_source_range": {"start": 4, "end": 5.5}}])
    assert preview(three_clips, cache, tmp_path / "after.mp4") == 1
    assert frames_in(tmp_path / "after.mp4") == 165

def test_the_sound_of_a_cached_preview_is_still_where_it_belongs(three_clips: str, tmp_path: Path) -> None:
    """A cached picture has no sound in it, so the sound is read from the source."""
    cache = tmp_path / "cache"
    cache.mkdir()
    out = tmp_path / "heard.mp4"
    preview(three_clips, cache, out)
    # The loud clip's 800Hz tone plays from two seconds to four, and only there.
    assert level(out, 2.5, 1.0, freq=800) > level(out, 4.5, 1.0, freq=800) + 20

def test_a_preview_is_rendered_small_from_the_start() -> None:
    from app.models.timeline import Project

    assert (_preview_sized(Project(id="p", width=1080, height=1920)).width,
            _preview_sized(Project(id="p", width=1080, height=1920)).height) == (480, 854)
    small = Project(id="p", width=320, height=240)
    assert _preview_sized(small) is small

def test_the_preview_tool_says_how_much_it_reused(three_clips: str) -> None:
    first = render_project(three_clips, is_preview=True)
    assert first["pictures"]["rendered"] + first["pictures"]["reused"] == 3
