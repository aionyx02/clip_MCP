"""A song is heard for its mood, style and energy, and a plan is told where it sits badly."""

from pathlib import Path

import pytest

from app.engine import clap, music
from app.engine.analysis import analyze_media
from app.models.media import Asset


def test_energy_is_the_song_on_its_own_scale_in_sections() -> None:
    quiet, loud = [-40.0] * 30, [-12.0] * 30
    energy = music.energy_curve(quiet + loud + quiet[:10])
    assert min(energy) == 0.0 and max(energy) == 1.0
    found = music.sections(energy)
    assert [section["level"] for section in found] == ["quiet", "full", "quiet"]
    assert music.fullest(energy)[0] >= 28
    fading = music.energy_curve([-12.0] * 30 + [-12.0 - 6 * step for step in range(8)])
    assert music.ending(fading) == "fades out"
    assert music.ending(music.energy_curve([-30.0] * 20 + [-10.0] * 10)) == "ends full"


def test_tags_are_what_sets_a_song_apart_from_the_library() -> None:
    # Everything scores high on `corporate`; what tells these songs apart is the rest.
    scores = {f"s{n}": {"corporate": 0.6, "jazz": 0.3 + (0.2 if n == 0 else 0), "piano": 0.3 + (0.2 if n == 1 else 0)}
              for n in range(10)}
    ranked = clap.ranked(scores)
    assert ranked["s0"][0] == "jazz" and ranked["s1"][0] == "piano"
    # With too few songs to know what is usual, the scores stand as they are.
    assert clap.ranked({"a": {"corporate": 0.6, "jazz": 0.4}})["a"][0] == "corporate"


def test_a_song_is_heard_and_can_be_found_by_description(media: Path, tmp_path: Path) -> None:
    song = Asset(id="s", path=str(media / "song.mp3"), duration=20, has_video=False, has_audio=True, library="music")
    heard = analyze_media(song, str(tmp_path), False, None, None, None, lambda *_: None, lambda: False, "ffmpeg")
    assert heard.music is not None and len(heard.music.vector) == 512
    assert set(heard.music.moods) == set(clap.MOODS) and set(heard.music.styles) == set(clap.STYLES)
    assert len(heard.music.energy) == 20 and heard.recipe.clap_model == clap.MODEL_ID
    # A description lies closer to a song like it: a steady sine is a tone, not a choir.
    closest = clap.closest("a single steady electronic tone", {"tone": heard.music.vector})
    assert closest[0][0] == "tone" and -1 <= closest[0][1] <= 1


def test_a_plan_is_told_when_its_song_loops_or_is_cut_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.engine import plan as compiler

    song = Asset(id="song", path="song.mp3", duration=30, has_video=False, has_audio=True, library="music")
    energy = tuple([0.2] * 10 + [1.0] * 10 + [0.6] * 10)
    cuts = {"song": compiler.CleanCuts(energy=energy, beats=None)}
    beds = [compiler.Bed(asset_id="song", start=0, end=30, timeline_in=0, volume=0.35, fade_in=1, fade_out=0,
                         key=("cue:^", "pass:1")),
            compiler.Bed(asset_id="song", start=0, end=12, timeline_in=30, volume=0.35, fade_in=0, fade_out=2,
                         key=("cue:^", "pass:2"))]
    monkeypatch.setattr(compiler, "music_beds", lambda *_: beds)
    monkeypatch.setattr(compiler, "compiled_duration", lambda *_: 42.0)
    plan = type("Plan", (), {"beats": []})()
    notes = compiler.music_fit_notes(plan, [], {"song": song}, cuts)
    kinds = [note.kind for note in notes]
    assert "music_loops" in kinds and "music_cut_off" in kinds
    assert all(note.look for note in notes)
