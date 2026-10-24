"""Tests for where a clip's source is read from: a seek near the start of an AAC file, and the cut after it."""

import re
import subprocess
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Tuple

import numpy
import pytest

from app.engine import builder
from app.engine.builder import FFmpegRenderer
from app.engine.diarize import SAMPLE_RATE, read_samples
from app.models.media import Asset
from app.models.timeline.parts import Clip, Dissolve, TimeRange
from app.runtime import repo
from app.tools.catalog import import_asset
from app.tools.shared import referenced_assets
from helpers import build_project, insert, render, video_track

# Where the noise bursts of the test file start and stop, in seconds: irregular, so the
# cross-correlation that measures the lag has one sharp peak and no repeating pattern.
BURSTS = (
    (0.13, 0.31), (0.67, 0.79), (1.21, 1.6), (2.03, 2.17),
    (2.71, 3.05), (3.43, 3.59), (4.11, 4.5), (5.03, 5.21),
)
# How far either side of the expected place the lag is looked for: 100 ms.
WINDOW_SECONDS = 0.1
# Silence put in front of the source, so a render that begins a little late can still be
# lined up: a negative shift needs samples from before the file's start.
PAD = round(WINDOW_SECONDS * SAMPLE_RATE)
# The loudness chain's limiter looks 1 ms ahead, so every render is about that much late
# however its clips are read. A clip far into its file is read with a seek, which cuts
# exactly, so it is the reference; a clip near the start must land within a sample of
# 16 kHz either way of where that one lands.
TOLERANCE_SAMPLES = 1
# How far from where the source says the reference itself may land: the limiter's 1 ms and
# a little over.
REFERENCE_SAMPLES = 2 * SAMPLE_RATE // 1000


@pytest.fixture(scope="module")
def bursts(tmp_path_factory: pytest.TempPathFactory) -> Tuple[str, Path]:
    """A 22.05 kHz AAC file of noise bursts, with a picture, and its asset ID.

    Returns:
        `(asset_id, path)`. The sound is what makes a seek near the start of an
        AAC file visible; the picture is there so the file renders like footage.
    """
    path = tmp_path_factory.mktemp("bursts") / "bursts.mp4"
    gate = "+".join(f"between(t,{start},{end})" for start, end in BURSTS)
    result = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:size=320x240:rate=30:duration=6",
        "-f", "lavfi", "-i", "anoisesrc=d=6:c=white:r=22050:a=0.5",
        "-filter_complex", f"[1:a]volume='{gate}':eval=frame[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "22050", "-b:a", "96k", "-shortest", str(path),
    ], capture_output=True)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return import_asset(str(path))["id"], path


def lag_samples(heard: numpy.ndarray, source: numpy.ndarray, offset: float) -> int:
    """Find where a render's sound begins in its source, by raw-sample cross-correlation.

    Args:
        heard: The render's samples, at `SAMPLE_RATE`.
        source: The whole source file's samples, decoded without a seek.
        offset: Where in the source the render is expected to begin, in seconds.

    Returns:
        The shift in samples that lines the two up best, within
        `WINDOW_SECONDS` either side of the expected place. Positive means the
        render's sound is early: it begins with source that comes later in the
        file. Negative means it is late.
    """
    heard = numpy.asarray(heard, dtype=numpy.float64)
    heard = heard - heard.mean()
    source = numpy.concatenate([numpy.zeros(PAD), numpy.asarray(source, dtype=numpy.float64)])
    base = round(offset * SAMPLE_RATE) + PAD
    window = round(WINDOW_SECONDS * SAMPLE_RATE)
    best, best_score = 0, -numpy.inf
    for shift in range(-window, window + 1):
        start = base + shift
        if start < 0 or start + len(heard) > len(source):
            continue
        part = source[start: start + len(heard)]
        part = part - part.mean()
        score = float(numpy.dot(heard, part) / (numpy.linalg.norm(part) + 1e-9))
        if score > best_score:
            best, best_score = shift, score
    return best


def rendered_lag(asset: str, path: Path, start: float, out: Path) -> int:
    """Render two seconds of the file from `start` and say where its sound lands, in samples."""
    project = build_project([video_track(), insert("a", asset, start, start + 2)], width=320, height=240)
    render(project, out)
    return lag_samples(read_samples(str(out)), read_samples(str(path)), start)


def rendered_lead_lag(asset: str, path: Path, start: float, lead: float, out: Path) -> int:
    """Render a clip whose sound leads its picture, after a silent clip, and say where that sound lands.

    Args:
        asset: The source file's asset ID.
        path: The source file, decoded whole as the comparison.
        start: Where the leading clip begins in the source.
        lead: How far its sound starts before its picture.
        out: Where the render is written.

    Returns:
        The shift in samples that lines the render's sound up with the source,
        against the point in the source where the lead begins.
    """
    project = build_project([
        video_track(),
        insert("before", asset, 2.0, 3.0, volume=0),
        insert("led", asset, start, start + 2),
        {"action": "set_clip_audio", "track_id": "main", "clip_id": "led", "audio_lead": lead},
    ], width=320, height=240)
    render(project, out)
    heard = read_samples(str(out))
    begins = round((1.0 - lead) * SAMPLE_RATE)
    length = round((2.0 + lead) * SAMPLE_RATE)
    return lag_samples(heard[begins:begins + length], read_samples(str(path)), start - lead)


@pytest.fixture(scope="module")
def reference(bursts: Tuple[str, Path], tmp_path_factory: pytest.TempPathFactory) -> int:
    """Where a clip read with a seek, two seconds into the file, lands: the lag every render shares."""
    asset, path = bursts
    return rendered_lag(asset, path, 2.0, tmp_path_factory.mktemp("reference") / "far.mp4")


def source_clip(start: str, end: str) -> Clip:
    """Build a clip reading a stretch of a source file.

    Args:
        start: Source range start, in seconds.
        end: Source range end, in seconds.

    Returns:
        The clip, on the timeline from zero.
    """
    return Clip(
        id="c", asset_id="a",
        source_range=TimeRange(start=Decimal(start), end=Decimal(end)),
        timeline_in=Decimal(0),
    )


def test_a_clip_at_the_start_of_an_aac_file_keeps_its_sound(
    bursts: Tuple[str, Path], reference: int, tmp_path: Path,
) -> None:
    """A seek of zero used to drop one more AAC frame than reading from the start does.

    The sound then came out 46 ms early. Read from the beginning and cut by
    timestamp, it lines up with the source decoded without a seek.
    """
    asset, path = bursts
    lag = rendered_lag(asset, path, 0.0, tmp_path / "start.mp4")
    assert abs(lag - reference) <= TOLERANCE_SAMPLES, (lag, reference)


def test_a_clip_a_little_way_in_keeps_its_sound(bursts: Tuple[str, Path], reference: int, tmp_path: Path) -> None:
    """A clip starting inside `SEEK_FROM_SECONDS` is read from the beginning and cut by timestamp.

    Read that way it lands where a clip sought into the file does, within a
    sample of the reference.
    """
    asset, path = bursts
    lag = rendered_lag(asset, path, 0.3, tmp_path / "in.mp4")
    assert abs(lag - reference) <= TOLERANCE_SAMPLES, (lag, reference)


def test_a_clip_read_with_a_seek_lands_where_its_source_says(reference: int) -> None:
    """A clip past the first second is seeked to, and lands where the source says it does."""
    assert abs(reference) <= REFERENCE_SAMPLES, reference


def test_a_clip_far_from_the_start_still_seeks(bursts: Tuple[str, Path]) -> None:
    """Past the first second a seek is exact, and reading from the start would be wasteful."""
    asset, path = bursts
    args = builder._input_args(source_clip("2.0", "4.0"), Asset(
        id="a", path=str(path), duration=6, has_video=True, has_audio=True,
    ), 60, Fraction(30))
    assert args[:2] == ["-ss", "2.000000"]


def test_a_clip_near_the_start_is_read_from_the_start_and_cut() -> None:
    """The unit of the fix: no seek, a longer read, and a trim at the very front of the chain."""
    asset = Asset(id="a", path="x.mp4", duration=6, has_video=True, has_audio=True)
    clip = source_clip("0.3", "2.3")
    args = builder._input_args(clip, asset, 60, Fraction(30))
    assert "-ss" not in args
    # (60 + 1) frames at 30 fps, plus the 0.3 s now read before the clip begins.
    assert args[:2] == ["-t", "2.333333"]
    chain = builder._picture_chain(clip, 61, 320, 240, Fraction(30), None)
    assert chain.startswith("trim=start=0.300000,")


def test_a_clip_with_a_run_up_is_cut_where_its_picture_input_begins() -> None:
    """A 0.4 s dissolve in reaches back from 0.6 s to 0.2 s: the picture is read from the start and cut at 0.2 s."""
    asset = Asset(id="a", path="x.mp4", duration=6, has_video=True, has_audio=True)
    clip = source_clip("0.6", "2.6").model_copy(update={"transition_in": Dissolve(kind="dissolve", seconds=Decimal("0.4"))})
    assert "-ss" not in builder._input_args(clip, asset, 72, Fraction(30))
    chain = builder._picture_chain(clip, 73, 320, 240, Fraction(30), None)
    assert chain.startswith("trim=start=0.200000,"), chain


def test_a_sounds_own_input_reads_from_the_start_with_room_for_the_cut() -> None:
    """A sound starting at 0.3 s gets no seek, and is read long enough to cut 0.3 s off its front."""
    asset = Asset(id="a", path="x.mp4", duration=6, has_video=True, has_audio=True)
    args = builder._audio_input_args(source_clip("0.3", "2.3"), asset, Fraction(30))
    assert "-ss" not in args
    # The sound's 2.0 s, one frame of margin, and the 0.3 s now read before it begins.
    assert args[:2] == ["-t", "2.333333"]


def test_a_leading_sound_near_the_start_lines_up_with_its_source(
    bursts: Tuple[str, Path], reference: int, tmp_path: Path,
) -> None:
    """A 0.2 s lead read from the beginning of the file lands where its source says it does."""
    asset, path = bursts
    lag = rendered_lead_lag(asset, path, 0.3, 0.2, tmp_path / "lead.mp4")
    assert abs(lag - reference) <= TOLERANCE_SAMPLES, (lag, reference)


def test_a_leading_overlay_cuts_its_sound_where_its_input_was_read(
    bursts: Tuple[str, Path], tmp_path: Path,
) -> None:
    """A sound sharing the picture's input is cut where that input was read, not where the clip's own sound starts."""
    asset, _path = bursts
    project_id = build_project([
        video_track(),
        insert("base", asset, 0, 1),
        video_track("over"),
        {
            "action": "add_clip", "track_id": "over", "clip_id": "over", "asset_id": asset,
            "source_range": {"start": 0.5, "end": 1.5}, "timeline_in": 0.5,
        },
        {"action": "set_clip_audio", "track_id": "over", "clip_id": "over", "audio_lead": 0.3},
    ], width=320, height=240)
    project = repo.get_project(project_id)
    command = FFmpegRenderer().build_command(project, referenced_assets(project), str(tmp_path / "out.mp4"))
    chains = command[command.index("-filter_complex") + 1].split(";")
    sound = next(chain for chain in chains if "atrim=start=0.500000" in chain)
    # `(?<!a)` keeps `atrim=start=0.500000` from being mistaken for the picture's cut.
    picture_cut = re.compile(r"(?<!a)trim=start=0\.500000")
    picture = next(chain for chain in chains if picture_cut.search(chain))
    assert sound.split("]", 1)[1].startswith("atrim=start=0.500000,"), sound
    assert picture_cut.search(picture), picture


def test_the_music_is_cut_where_its_input_was_read(media: Path, tmp_path: Path) -> None:
    """A song whose clip starts 0.5 s in is cut at the front of its own chain, where its input began."""
    picture = import_asset(str(media / "wide.mp4"))["id"]
    song = import_asset(str(media / "song.mp3"))["id"]
    project_id = build_project([
        video_track(),
        insert("base", picture, 0, 2),
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        insert("song", song, 0.5, 2.5, track_id="music"),
    ], width=320, height=240)
    project = repo.get_project(project_id)
    command = FFmpegRenderer().build_command(project, referenced_assets(project), str(tmp_path / "out.mp4"))
    chains = command[command.index("-filter_complex") + 1].split(";")
    sound = next(chain for chain in chains if "atrim=start=0.500000" in chain)
    assert sound.split("]", 1)[1].startswith("atrim=start=0.500000,"), sound
