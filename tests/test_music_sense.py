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
    plan = type("Plan", (), {"beats": [], "music": None})()
    notes = compiler.music_fit_notes(plan, [], {"song": song}, cuts)
    kinds = [note.kind for note in notes]
    assert "music_loops" in kinds and "music_cut_off" in kinds
    assert all(note.look for note in notes)


def test_a_song_started_where_the_plan_is_told_ends_with_the_cut_rather_than_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import re

    from app.engine import plan as compiler

    # 58.74s of cut: rounded to 58.7 first, the start said was 191.3 and the song ran 0.04s short.
    song = Asset(id="song", path="song.mp3", duration=250, has_video=False, has_audio=True, library="music")
    cuts = {"song": compiler.CleanCuts(energy=tuple([1.0] * 250), beats=None)}
    beds = [compiler.Bed(asset_id="song", start=0, end=58.74, timeline_in=0, volume=0.35, fade_in=1, fade_out=2,
                         key=("cue:^", "pass:1"))]
    monkeypatch.setattr(compiler, "music_beds", lambda *_: beds)
    monkeypatch.setattr(compiler, "compiled_duration", lambda *_: 58.74)
    plan = type("Plan", (), {"beats": [], "music": None})()
    notes = compiler.music_fit_notes(plan, [], {"song": song}, cuts)
    [cut_off] = [note for note in notes if note.kind == "music_cut_off"]
    start = float(re.search(r"start the cue at ([\d.]+)s", cut_off).group(1))
    assert start + 58.74 <= 250


def test_only_the_song_playing_at_the_payoff_is_told_where_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from app.engine import plan as compiler
    from app.models.plan import Beat, BeatRole, MusicCue, MusicPlan

    # The first song hands over at the payoff, so its fullest part can never land there; told to
    # start at 27.0s, it was told 27.0s again for ever. The second cross-fades in 1.5s early.
    songs = {"a": Asset(id="a", path="a.mp3", duration=140, has_video=False, has_audio=True, library="music"),
             "b": Asset(id="b", path="b.mp3", duration=60, has_video=False, has_audio=True, library="music")}
    cuts = {"a": compiler.CleanCuts(energy=tuple([0.2] * 125 + [1.0] * 8 + [0.2] * 7), beats=None),
            "b": compiler.CleanCuts(energy=tuple([0.2] * 30 + [1.0] * 8 + [0.2] * 22), beats=None)}
    plan = SimpleNamespace(
        beats=[Beat(id="payoff", name="收尾", role=BeatRole.PAYOFF)],
        music=MusicPlan(cues=[MusicCue(asset_id="a", start=27.0), MusicCue(beat_id="payoff", asset_id="b", start=10.0)]),
    )
    pieces = [SimpleNamespace(beat_id="intro", played=98.0), SimpleNamespace(beat_id="payoff", played=20.0)]
    beds = [compiler.Bed(asset_id="a", start=27.0, end=125.0, timeline_in=0, volume=0.35, fade_in=0.5, fade_out=1.5,
                         key=("cue:^", "pass:1")),
            compiler.Bed(asset_id="b", start=8.5, end=30.0, timeline_in=96.5, volume=0.35, fade_in=1.5, fade_out=2,
                         key=("cue:payoff", "pass:1"), lane=1)]
    monkeypatch.setattr(compiler, "music_beds", lambda *_: beds)
    monkeypatch.setattr(compiler, "compiled_duration", lambda *_: 118.0)
    notes = [note for note in compiler.music_fit_notes(plan, pieces, songs, cuts) if note.kind == "music_peak_off"]
    assert [note.beat_id for note in notes] == ["payoff"]
    # Its fullest part begins 30s in; the cue takes over at the payoff, so that is where it starts.
    assert "start the cue at 30.0s" in notes[0]


def test_the_song_fading_out_as_the_payoff_begins_is_not_told_to_land_its_peak_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from app.engine import plan as compiler
    from app.models.plan import Beat, BeatRole, MusicCue, MusicPlan

    # 0.1 + 0.7 adds up to a hair under the 0.8 the cue's stretch is rounded to, so the payoff
    # looked to begin while the first song still held the cut.
    songs = {"a": Asset(id="a", path="a.mp3", duration=140, has_video=False, has_audio=True, library="music"),
             "b": Asset(id="b", path="b.mp3", duration=60, has_video=False, has_audio=True, library="music")}
    cuts = {"a": compiler.CleanCuts(energy=tuple([0.2] * 125 + [1.0] * 8 + [0.2] * 7), beats=None),
            "b": compiler.CleanCuts(energy=tuple([0.2] * 30 + [1.0] * 8 + [0.2] * 22), beats=None)}
    plan = SimpleNamespace(
        beats=[Beat(id="payoff", name="收尾", role=BeatRole.PAYOFF)],
        music=MusicPlan(cues=[MusicCue(asset_id="a"), MusicCue(beat_id="payoff", asset_id="b", start=10.0)]),
    )
    pieces = [SimpleNamespace(beat_id="intro", played=0.1), SimpleNamespace(beat_id="intro", played=0.7),
              SimpleNamespace(beat_id="payoff", played=20.0)]
    beds = [compiler.Bed(asset_id="a", start=0.0, end=2.3, timeline_in=0, volume=0.35, fade_in=0.5, fade_out=1.5,
                         key=("cue:^", "pass:1")),
            compiler.Bed(asset_id="b", start=8.5, end=30.0, timeline_in=0.8, volume=0.35, fade_in=1.5, fade_out=2,
                         key=("cue:payoff", "pass:1"), lane=1)]
    monkeypatch.setattr(compiler, "music_beds", lambda *_: beds)
    monkeypatch.setattr(compiler, "compiled_duration", lambda *_: 20.8)
    notes = [note for note in compiler.music_fit_notes(plan, pieces, songs, cuts) if note.kind == "music_peak_off"]
    assert [note.beat_id for note in notes] == ["payoff"]


def test_landing_the_peak_and_ending_with_the_song_are_said_to_be_a_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from app.engine import plan as compiler
    from app.models.plan import Beat, BeatRole, MusicCue, MusicPlan

    # Going all the way through, its fullest part 120s in: started to land that on the payoff at
    # 50s, the cut stops it 70s early; started to end with the cut, the fullest part is never heard.
    song = Asset(id="song", path="song.mp3", duration=200, has_video=False, has_audio=True, library="music")
    cuts = {"song": compiler.CleanCuts(energy=tuple([0.6] * 120 + [1.0] * 8 + [0.6] * 72), beats=None)}
    plan = SimpleNamespace(beats=[Beat(id="payoff", name="收尾", role=BeatRole.PAYOFF)],
                           music=MusicPlan(cues=[MusicCue(asset_id="song")]))
    pieces = [SimpleNamespace(beat_id="intro", played=50.0), SimpleNamespace(beat_id="payoff", played=10.0)]
    beds = [compiler.Bed(asset_id="song", start=0.0, end=60.0, timeline_in=0, volume=0.35, fade_in=1, fade_out=2,
                         key=("cue:^", "pass:1"))]
    monkeypatch.setattr(compiler, "music_beds", lambda *_: beds)
    monkeypatch.setattr(compiler, "compiled_duration", lambda *_: 60.0)
    notes = {note.kind: note for note in compiler.music_fit_notes(plan, pieces, {"song": song}, cuts)}
    assert "start the cue at 70.0s" in notes["music_peak_off"] and "choose one" in notes["music_peak_off"]
    assert "start the cue at 140.0s" in notes["music_cut_off"] and "choose one" in notes["music_cut_off"]
