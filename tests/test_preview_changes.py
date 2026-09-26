"""Tests for seeing what a change did, and hearing what a cut sounds like.

The plan diff is arithmetic on two compiled cuts, checked through the tool on
a real file. The sound preview renders real sound and measures it, because a
chart drawn from a graph that is wrong in some quiet way still looks like a
chart.
"""

import base64
import io
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from app.engine.listen import levels, parts_of
from app.models.plan import Beat, EditPlan, Selection, Trim, TrimKind
from app.server import get_plan, import_asset, preview_plan_diff, preview_sound, save_plan
from helpers import build_project, edit, insert, video_track
from test_edit_plan import planned  # noqa: F401 — the fixture

def picture_of(result) -> Image.Image:
    """Decode the image a tool returned.

    Args:
        result: The tool's result.

    Returns:
        The image.
    """
    return Image.open(io.BytesIO(base64.b64decode(result.content[1].data)))

# --- what a change did -------------------------------------------------------------------

def test_a_change_is_shown_shot_by_shot(planned: dict) -> None:  # noqa: F811
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    first, second, third = (planned["clips"][index]["clip_id"] for index in range(3))
    changed = save_plan(stored.model_copy(update={
        "id": "changed", "version": 1,
        "selections": [
            Selection(clip_id=first, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=1.0)),
            Selection(clip_id=second, beat_id="b1"),
            Selection(clip_id=third, beat_id="b1"),
        ],
        "rejected": [],
    }))
    result = preview_plan_diff(planned["plan_id"], changed["plan_id"])
    text = result.content[0].text
    assert "1 added" in text and "1 retrimmed" in text
    assert "#1: retrimmed" in text and "#2: added" in text
    # Untouched shots are on the sheet but not in the list of changes.
    assert "#3:" not in text
    assert picture_of(result).width > 0

def test_a_shot_taken_out_is_shown_where_it_was(planned: dict) -> None:  # noqa: F811
    stored = EditPlan.model_validate(get_plan(planned["plan_id"])["plan"])
    shorter = save_plan(stored.model_copy(update={
        "id": "shorter", "version": 1, "selections": stored.selections[:1],
    }))
    text = preview_plan_diff(planned["plan_id"], shorter["plan_id"]).content[0].text
    assert "1 dropped" in text and "was " in text
    assert "s): " in text and "(-" in text

# --- what a cut sounds like --------------------------------------------------------------

@pytest.fixture
def talk_over_music(audio_media: Path) -> str:
    """Speech from three seconds in, over a ducked music bed, in two parts.

    Returns:
        The project ID.
    """
    talk = import_asset(str(audio_media / "talk.mp4"))["id"]
    music = import_asset(str(audio_media / "music.mp3"))["id"]
    project = build_project([
        video_track(), insert("t", talk, 0, 6),
        {"action": "add_track", "track_id": "music", "track_type": "audio", "duck_under_speech": True},
        insert("m", music, 0, 6, track_id="music", volume=0.5),
    ], width=320, height=240)
    edit(project, [{"action": "set_markers", "markers": [
        {"id": "a", "name": "安靜", "timeline_in": 0}, {"id": "b", "name": "講話", "timeline_in": 3},
    ]}])
    return project

def test_the_sound_is_described_part_by_part(talk_over_music: str) -> None:
    result = preview_sound(talk_over_music)
    lines = result.content[0].text.splitlines()
    quiet = next(line for line in lines if line.startswith("安靜"))
    spoken = next(line for line in lines if line.startswith("講話"))
    assert "nobody talks" in quiet and "in the gaps" in quiet
    # Ducking pulls the music down while the tone — standing in for a voice — plays.
    assert "under the talking" in spoken
    assert picture_of(result).format == "PNG"

def test_the_mix_is_saved_to_listen_to(talk_over_music: str) -> None:
    text = preview_sound(talk_over_music).content[0].text
    path = text.splitlines()[0].removeprefix("The mix is at ").rstrip(".")
    seconds = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout)
    assert seconds == pytest.approx(6.0, abs=0.1)

def test_ducking_shows_as_the_music_dropping_under_the_voice(talk_over_music: str, tmp_path: Path) -> None:
    """Measured off the stems themselves, so the sentence is known to come from the sound."""
    from app.engine.builder import FFmpegRenderer
    from app.server import _referenced_assets, repo

    project = repo.get_project(talk_over_music)
    voice, music = tmp_path / "voice.wav", tmp_path / "music.wav"
    command = FFmpegRenderer().build_sound(project, _referenced_assets(project), str(tmp_path / "mix.m4a"),
                                           str(voice), str(music))
    assert subprocess.run(command, capture_output=True).returncode == 0
    parts = parts_of(project, levels(str(voice)), levels(str(music)))
    quiet, spoken = parts
    assert quiet.voice is None
    assert spoken.voice is not None and spoken.music_under_voice is not None
    assert quiet.music_alone - spoken.music_under_voice > 6

def test_a_cut_with_no_music_says_so(audio_media: Path) -> None:
    loud = import_asset(str(audio_media / "loud.mp4"))["id"]
    project = build_project([video_track(), insert("l", loud, 0, 4)], width=320, height=240)
    assert "no music" in preview_sound(project).content[0].text
