"""Tests for fitting the cut to the music: cutting on the beat, beat order, transitions and pauses."""

from dataclasses import replace
from typing import Optional

import pytest

from app.engine.plan.beats import BEAT_SNAP_SECONDS, compile_pieces, plan_pieces
from app.engine.plan.music import music_beds
from app.engine.plan.pieces import Piece
from app.engine.semantic import CleanCuts, build_timeline, clean_cuts
from app.models.plan import Beat, BeatTransition, EditPlan, MusicCue, MusicPlan, Selection
from test_semantic_timeline import ASSET_ID, analysis

from edit_plan.shared import (
    FOOTAGE, SONG, checked, cut_points, every, footage, joined_plan, plan_for, song_of, sourced, speech_of,
)


def test_beats_decide_the_order_the_footage_plays_in() -> None:
    by_id, children, assets, clips = footage()
    first, third = speech_of(clips)[0], speech_of(clips)[2]
    plan = EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash",
        beats=[Beat(id="open", name="開場"), Beat(id="body", name="主體")],
        selections=[
            Selection(clip_id=first.id, beat_id="body"),
            Selection(clip_id=third.id, beat_id="open"),
        ],
    )
    # Written body-first, but the beats say the opening comes first.
    assert plan_pieces(plan, by_id, children, assets)[0].start == pytest.approx(third.source_range.start, abs=0.2)

def on_beat(*cues: MusicCue, beats=None, song_seconds: float = 30.0, footage_made=None) -> tuple:
    """Shots of talk in one beat, with music under them.

    Args:
        *cues: The music.
        beats: The song's beats.
        song_seconds: How long the song runs.
        footage_made: The footage's analysis, if not the usual one.

    Returns:
        `(plan, clips_by_id, assets, cuts)`. With the usual footage the plan
        uses its first, third and fourth lines, which sit in silence with room
        either side.
    """
    made = footage_made or analysis(**FOOTAGE)
    _, clips = build_timeline({ASSET_ID: sourced()}, {ASSET_ID: made})
    talk = speech_of(clips)
    song, heard = song_of(beats, song_seconds)
    chosen = [talk[0], talk[-2], talk[-1]] if len(talk) > 3 else talk
    plan = plan_for([Selection(clip_id=clip.id, beat_id="b1") for clip in chosen],
                    music=MusicPlan(cues=list(cues)))
    return (plan, {clip.id: clip for clip in clips}, {ASSET_ID: sourced(), SONG: song},
            {ASSET_ID: clean_cuts(made), SONG: heard})

def test_a_cut_moves_through_silence_onto_the_beat() -> None:
    plan, by_id, assets, cuts = on_beat(MusicCue(asset_id=SONG, cut_on_beat=True), beats=every(0.5))
    pieces, notes = compile_pieces(plan, by_id, {}, assets, cuts)
    assert cut_points(pieces) == [2.5, 5.0]
    # The first shot's cut was 0.2s past a beat and 0.3s short of the next. Going back to
    # the nearer one would have cut the breath after its last word, so it went forward.
    assert pieces[0].end == 3.4
    assert any("moved onto the beat" in note for note in notes)

def test_a_cue_that_does_not_ask_leaves_the_cuts_alone() -> None:
    plan, by_id, assets, cuts = on_beat(MusicCue(asset_id=SONG), beats=every(0.5))
    without = plan.model_copy(update={"music": None})
    assert compile_pieces(plan, by_id, {}, assets, cuts) == compile_pieces(without, by_id, {}, assets, cuts)

def test_a_cut_never_reaches_into_the_next_word() -> None:
    """A beat 0.15s on would cost nothing — except that the next line starts 0.2s on,
    and the cut would take its first breath in with it."""
    tight = analysis(
        duration=20.0,
        segments=[(1.0, 3.0, "第一句話"), (3.3, 5.0, "第二句話"), (8.0, 10.0, "第三句話")],
        silences=[(0.0, 1.0), (3.0, 3.3), (5.0, 8.0), (10.0, 20.0)],
    )
    plan, by_id, assets, cuts = on_beat(
        MusicCue(asset_id=SONG, cut_on_beat=True), beats=[0.0, 2.0, 2.35, 10.0], footage_made=tight,
    )
    plan = plan.model_copy(update={"selections": [plan.selections[0], plan.selections[-1]]})
    moved, notes = compile_pieces(plan, by_id, {}, assets, cuts)
    still = compile_pieces(plan.model_copy(update={"music": None}), by_id, {}, assets, cuts)[0]
    assert moved == still
    assert any("off the beat" in note for note in notes)

def test_a_cut_further_from_the_beat_than_it_may_move_stays_and_is_said() -> None:
    plan, by_id, assets, cuts = on_beat(
        MusicCue(asset_id=SONG, cut_on_beat=True), beats=[0.0, 2.2 + BEAT_SNAP_SECONDS + 0.2, 10.0],
    )
    moved, notes = compile_pieces(plan, by_id, {}, assets, cuts)
    assert cut_points(moved)[0] == 2.2
    assert any(f"further than {BEAT_SNAP_SECONDS:g}s" in note for note in notes)

def test_footage_nobody_transcribed_keeps_its_cuts() -> None:
    """Nobody knows where the words are, so nobody knows the move is through silence."""
    plan, by_id, assets, cuts = on_beat(MusicCue(asset_id=SONG, cut_on_beat=True), beats=every(0.5))
    cuts = {**cuts, ASSET_ID: replace(cuts[ASSET_ID], words=(), sentences=(), said=(), transcribed=False)}
    moved, notes = compile_pieces(plan, by_id, {}, assets, cuts)
    assert moved == compile_pieces(plan.model_copy(update={"music": None}), by_id, {}, assets, cuts)[0]
    assert any("never transcribed" in note for note in notes)

def test_the_song_comes_in_on_a_beat_and_counts_from_it() -> None:
    plan, by_id, assets, cuts = on_beat(
        MusicCue(asset_id=SONG, start=0.2, cut_on_beat=True), beats=every(0.5, first=0.45),
    )
    pieces, _ = compile_pieces(plan, by_id, {}, assets, cuts)
    assert music_beds(plan, pieces, assets, cuts)[0].start == 0.45
    # Counted from where the song came in, the beats land on the same half-seconds.
    assert cut_points(pieces) == [2.5, 5.0]
    _, notes = checked(plan, by_id, assets, cuts)
    assert any("comes in at 0.45s" in note for note in notes)

def test_the_beat_carries_on_when_the_song_loops() -> None:
    """A 1.8s loop with beats every 0.6s: after the loop they fall at 1.8, 2.4, 3.0."""
    plan, by_id, assets, cuts = on_beat(
        MusicCue(asset_id=SONG, cut_on_beat=True), beats=[0.0, 0.6, 1.2], song_seconds=1.8,
    )
    pieces, _ = compile_pieces(plan, by_id, {}, assets, cuts)
    assert cut_points(pieces)[0] == 2.4

def joined(transition: Optional[BeatTransition], lead: Optional[float], opening: float) -> tuple:
    """Compile two beats, the second opening on silent picture at `opening` seconds into its file.

    Returns:
        `(pieces, notes)`.
    """
    plan, by_id, children, assets = joined_plan(transition, lead, opening)
    return compile_pieces(plan, by_id, children, assets)

def test_a_beat_comes_in_on_its_transition_and_its_sound_early() -> None:
    pieces, notes = joined(BeatTransition(kind="dip", seconds=0.6, through="white"), 0.4, opening=5.0)
    assert pieces[0].transition is None
    assert pieces[1].transition == ("dip", 0.6, "white", "left")
    assert pieces[1].sound_lead == 0.4
    assert notes == []

def test_a_transition_is_shortened_to_the_picture_its_file_has() -> None:
    # A dissolve reaches back for its whole length, and this shot starts 0.3s into its file.
    pieces, notes = joined(BeatTransition(kind="dissolve", seconds=1.0), 1.0, opening=0.3)
    assert pieces[1].transition == ("dissolve", 0.3, "black", "left")
    assert pieces[1].sound_lead == 0.3
    assert any("shortened from 1s to 0.3s" in note for note in notes)
    assert any("from_seconds" in note for note in notes)

def test_a_beat_can_open_on_a_moment_of_quiet() -> None:
    plan, by_id, children, assets = joined_plan(None, None, opening=5.0)
    plan.beats[1].pause_before = 0.8
    pieces, notes = compile_pieces(plan, by_id, children, assets)
    assert pieces[1].start == pytest.approx(4.2)
    assert notes == []
    # With less of the shot before it, it opens on what there is, and says so.
    plan, by_id, children, assets = joined_plan(None, None, opening=0.3)
    plan.beats[1].pause_before = 0.8
    pieces, notes = compile_pieces(plan, by_id, children, assets)
    assert pieces[1].start == 0.0
    assert any("0.3s of quiet rather than 0.8s" in note for note in notes)

def test_a_pause_never_opens_a_shot_onto_the_line_before() -> None:
    from app.engine.plan.beats import _joined_beats

    plan = plan_for([]).model_copy(update={"beats": [Beat(id="b1", name="講"), Beat(id="b2", name="走", pause_before=1.0)]})
    pieces = [Piece(asset_id="a", start=0.0, end=2.0, from_clip_ids=("u1",), beat_id="b1"),
              Piece(asset_id="a", start=5.0, end=8.0, from_clip_ids=("u2",), beat_id="b2")]
    cuts = {"a": CleanCuts(words=((3.0, 4.6), (5.0, 6.0)), transcribed=True)}
    out, notes = _joined_beats(plan, pieces, {}, cuts)
    # The rest is made up from the quiet after the shot going out, up to its next word.
    assert out[1].start == pytest.approx(4.6) and out[0].end == pytest.approx(2.6) and notes == []

def test_a_pause_its_shot_cannot_give_is_borrowed_from_the_shot_going_out() -> None:
    """#76: a shot cut into mid-sentence has no quiet before it, but the one before it has
    seconds of room after its last word."""
    from app.engine.plan.beats import _joined_beats

    plan = plan_for([]).model_copy(update={"beats": [Beat(id="b1", name="講"), Beat(id="b2", name="走", pause_before=0.4)]})
    pieces = [Piece(asset_id="a", start=2.2, end=9.39, from_clip_ids=("u1",), beat_id="b1"),
              Piece(asset_id="b", start=13.0, end=20.1, from_clip_ids=("u2",), beat_id="b2")]
    cuts = {"a": CleanCuts(words=((2.3, 9.2),), transcribed=True, duration=12.5),
            "b": CleanCuts(words=((12.5, 13.4),), transcribed=True)}
    out, notes = _joined_beats(plan, pieces, {}, cuts)
    assert out[1].start == pytest.approx(13.0)
    assert out[0].end == pytest.approx(9.79)
    assert notes == []
    # Where neither side has it, what both gave is what is reported.
    cuts["a"] = CleanCuts(words=((2.3, 9.2), (9.5, 10.0)), transcribed=True)
    out, notes = _joined_beats(plan, pieces, {}, cuts)
    assert out[0].end == pytest.approx(9.5)
    assert any("0.11s of quiet rather than 0.4s" in note for note in notes)

def test_a_pause_a_millisecond_short_is_not_reported_as_shortened() -> None:
    """#89: 「0.399s of quiet rather than 0.4s」 was worth a look."""
    from app.engine.plan.beats import _joined_beats

    plan = plan_for([]).model_copy(update={"beats": [Beat(id="b1", name="講"), Beat(id="b2", name="走", pause_before=0.4)]})
    pieces = [Piece(asset_id="a", start=0.0, end=2.0, from_clip_ids=("u1",), beat_id="b1"),
              Piece(asset_id="b", start=5.0, end=8.0, from_clip_ids=("u2",), beat_id="b2")]
    out, notes = _joined_beats(plan, pieces, {}, {"b": CleanCuts(words=((4.601, 4.601), (5.0, 6.0)), transcribed=True)})
    assert out[1].start == pytest.approx(4.601) and notes == []

def test_a_pause_given_in_full_is_not_reported_as_shortened() -> None:
    """#76: a pause asked for at a speed that does not divide it evenly is still all there."""
    from app.engine.plan.beats import _joined_beats

    plan = plan_for([]).model_copy(update={"beats": [Beat(id="b1", name="講"), Beat(id="b2", name="走", pause_before=0.4)]})
    pieces = [Piece(asset_id="a", start=0.0, end=2.0, from_clip_ids=("u1",), beat_id="b1"),
              Piece(asset_id="b", start=5.0, end=8.0, from_clip_ids=("u2",), beat_id="b2", speed=1.1)]
    out, notes = _joined_beats(plan, pieces, {}, {"b": CleanCuts(words=((5.0, 6.0),), transcribed=True)})
    assert out[1].start == pytest.approx(4.56)
    assert notes == []

def test_a_pause_never_opens_a_shot_onto_what_the_cut_already_played_of_its_take() -> None:
    from app.engine.plan.beats import _joined_beats

    plan = plan_for([]).model_copy(update={"beats": [Beat(id="b1", name="講"), Beat(id="b2", name="走", pause_before=1.0)]})
    pieces = [Piece(asset_id="a", start=0.0, end=4.7, from_clip_ids=("u1",), beat_id="b1"),
              Piece(asset_id="a", start=5.0, end=8.0, from_clip_ids=("u2",), beat_id="b2")]
    out, _ = _joined_beats(plan, pieces, {}, {"a": CleanCuts(transcribed=True)})
    assert out[1].start == pytest.approx(4.7)

def test_a_pause_never_opens_onto_talking_only_the_second_listen_heard() -> None:
    from app.engine.plan.beats import _joined_beats

    plan = plan_for([]).model_copy(update={"beats": [Beat(id="b1", name="講"), Beat(id="b2", name="走", pause_before=1.0)]})
    pieces = [Piece(asset_id="a", start=0.0, end=2.0, from_clip_ids=("u1",), beat_id="b1"),
              Piece(asset_id="b", start=5.0, end=8.0, from_clip_ids=("u2",), beat_id="b2")]
    # The transcript has nothing before 5s; the voice detector heard somebody until 4.5s.
    cuts = {"b": CleanCuts(words=((5.0, 6.0),), voiced=((3.0, 4.5),), transcribed=True)}
    out, _ = _joined_beats(plan, pieces, {}, cuts)
    assert out[1].start == pytest.approx(4.5)

def test_a_beat_s_pause_leaves_its_transition_the_picture_it_runs_in_from() -> None:
    """#95: the pause opened the shot onto its file's first frame, and the dissolve was then
    refused for having no picture before it."""
    from app.engine.plan.beats import _joined_beats

    plan = plan_for([]).model_copy(update={"beats": [
        Beat(id="b1", name="講"),
        Beat(id="b2", name="走", pause_before=0.4, transition_in=BeatTransition(kind="dissolve", seconds=0.4)),
    ]})
    pieces = [Piece(asset_id="a", start=0.0, end=4.0, from_clip_ids=("u1",), beat_id="b1"),
              Piece(asset_id="b", start=0.5, end=8.0, from_clip_ids=("u2",), beat_id="b2")]
    cuts = {"a": CleanCuts(words=((0.2, 3.0),), transcribed=True, duration=10.0),
            "b": CleanCuts(words=((1.0, 2.0),), transcribed=True)}
    out, notes = _joined_beats(plan, pieces, {}, cuts)
    # 0.1s of pause from the shot itself, 0.4s kept for the dissolve, 0.3s from the shot going out.
    assert out[1].start == pytest.approx(0.4) and out[1].transition is not None
    assert out[0].end == pytest.approx(4.3)
    assert not any(note.kind == "join_left_out" for note in notes)
