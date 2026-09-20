"""Shared fixtures: an isolated workspace and small generated media files.

The workspace must be set before `app.server` is imported, because the server
opens its database at import time.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator, List

import pytest

WORKSPACE = tempfile.mkdtemp(prefix="clip-mcp-tests-")
os.environ["CLIP_MCP_WORKSPACE"] = WORKSPACE

FFMPEG = shutil.which("ffmpeg")

def _run_ffmpeg(args: List[str]) -> None:
    """Run FFmpeg, failing the test run with its error output.

    Args:
        args: Arguments after the executable name.

    Raises:
        AssertionError: If FFmpeg exits with a non-zero status.
    """
    result = subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")

def _make_video(path: Path, source: str, size: str, seconds: int, with_audio: bool = True) -> None:
    """Generate a small synthetic video file.

    Args:
        path: File to write.
        source: FFmpeg lavfi video source, such as `testsrc`.
        size: Frame size, such as `640x360`.
        seconds: Length of the file.
        with_audio: Whether to include an audio stream.
    """
    args = ["-f", "lavfi", "-i", f"{source}=size={size}:rate=30:duration={seconds}"]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac", "-shortest"]
    _run_ffmpeg([*args, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)])

@pytest.fixture(scope="session", autouse=True)
def require_ffmpeg() -> None:
    """Skip the whole suite when FFmpeg is not installed."""
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")

@pytest.fixture(scope="session", autouse=True)
def workspace() -> Iterator[str]:
    """Throw the temporary workspace away once the run is over.

    Yields:
        The workspace directory the server was pointed at.
    """
    yield WORKSPACE
    # The server holds its database open, so a locked file must not fail the run.
    shutil.rmtree(WORKSPACE, ignore_errors=True)

@pytest.fixture(scope="session")
def media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create the media files the tests import.

    Returns:
        A folder holding `wide.mp4` (landscape, 10s), `tall.mp4` (portrait,
        6s), `silent.mp4` (no audio, 4s), `muted.mp4` (an audio track of pure
        silence, 4s), `song.mp3` (20s), `jingle.wav` (only 4s), and `black.mp4`
        (an all-black 8s clip).
    """
    folder = tmp_path_factory.mktemp("media")
    _make_video(folder / "wide.mp4", "testsrc", "640x360", 10)
    _make_video(folder / "tall.mp4", "smptebars", "360x640", 6)
    _make_video(folder / "silent.mp4", "testsrc2", "320x240", 4, with_audio=False)
    # A clip recorded with the microphone muted: it has an audio stream, and every
    # sample in it is zero. Nothing about the file says so short of decoding it.
    _run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=4",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=4",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(folder / "muted.mp4"),
    ])
    _run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=220:duration=20", "-c:a", "libmp3lame", str(folder / "song.mp3")])
    # A song too short to cover a normal edit, so looping can be tested rather than plain extension.
    # Lossless, because fit_track measures how much of the source is left: an mp3 carries encoder
    # padding that some ffmpeg builds report as duration and others subtract, which moves every
    # loop boundary by ~0.05s. A wav is exactly as long as it says.
    _run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=330:duration=4", "-c:a", "pcm_s16le", str(folder / "jingle.wav")])
    # A black clip, so anything bright in a rendered frame can only have been drawn on top.
    _run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=black:size=640x360:rate=30:duration=8",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(folder / "black.mp4"),
    ])
    return folder

@pytest.fixture(scope="session")
def footage_folder(tmp_path_factory: pytest.TempPathFactory, media: Path) -> Path:
    """Create a folder that mixes media, a nested folder, and unreadable files.

    Returns:
        A folder holding `clip1.mp4`, `clip2.mp4`, `clip10.mp4`, `bgm.mp3`,
        `sub/nested.mp4`, plus `notes.txt` and an unreadable `broken.mp4`.
    """
    folder = tmp_path_factory.mktemp("footage")
    (folder / "sub").mkdir()
    shutil.copy(media / "tall.mp4", folder / "clip1.mp4")
    shutil.copy(media / "wide.mp4", folder / "clip2.mp4")
    shutil.copy(media / "wide.mp4", folder / "clip10.mp4")
    shutil.copy(media / "song.mp3", folder / "bgm.mp3")
    shutil.copy(media / "wide.mp4", folder / "sub" / "nested.mp4")
    (folder / "notes.txt").write_text("not media", encoding="utf-8")
    (folder / "broken.mp4").write_text("not really an mp4", encoding="utf-8")
    return folder

@pytest.fixture(scope="session")
def audio_media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create media with known, very different audio levels.

    Returns:
        A folder holding `quiet.mp4` (a faint tone), `loud.mp4` (a strong
        tone), `talk.mp4` (silent for three seconds, then a strong tone, which
        stands in for someone starting to speak), and `music.mp3` (a steady
        200 Hz bed that is easy to measure on its own).
    """
    folder = tmp_path_factory.mktemp("audio")
    for name, level in (("quiet.mp4", 0.05), ("loud.mp4", 0.6)):
        _run_ffmpeg([
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=5",
            "-f", "lavfi", "-i", "sine=frequency=800:duration=5", "-af", f"volume={level}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(folder / name),
        ])
    _run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=6",
        "-f", "lavfi", "-i", r"aevalsrc=if(gte(t\,3)\,0.7*sin(2*PI*1000*t)\,0):d=6",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(folder / "talk.mp4"),
    ])
    _run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=200:duration=20", "-c:a", "libmp3lame", str(folder / "music.mp3")])
    return folder
