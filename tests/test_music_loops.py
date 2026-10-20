"""A song too short for the video loops on a loop point: whole bars, inside its full part, never through its ending."""

from pathlib import Path

from app import runtime
from app.tools import catalog
from app.engine.music import BAR_BEATS, LOOP_SHORTEST_BARS, loop_span
from app.engine.plan import music_beds, plan_pieces, turnaround
from app.engine.semantic import CleanCuts
from app.models.media import MediaAnalysis, MusicSense, Rhythm
from app.models.plan import MusicCue, MusicPlan, Selection
from helpers import build_project, edit, insert, layout, video_track
from test_edit_plan import footage, plan_for, sourced, speech_of

# A 20-second song at 240 beats a minute — a bar a second — with a quiet intro and a fading end.
BEATS = tuple(round(index * 0.25, 3) for index in range(80))
ENERGY = tuple([0.1, 0.1] + [0.8] * 15 + [0.1] * 3)


def test_a_loop_is_whole_bars_inside_the_full_part() -> None:
    start, end = loop_span(BEATS, ENERGY, 20.0)
    bar = 0.25 * BAR_BEATS
    assert (end - start) / bar == round((end - start) / bar) >= LOOP_SHORTEST_BARS
    assert ENERGY[int(start)] == ENERGY[int(end) - 1] == 0.8
    # Too few bars, or nothing loud enough, and there is no loop to offer.
    assert loop_span(BEATS[:20], ENERGY, 20.0) is None
    assert loop_span(BEATS, tuple([0.1] * 20), 20.0) is None


def test_a_song_that_lasts_plays_through_and_one_that_runs_out_turns_round() -> None:
    song = CleanCuts(beats=BEATS, energy=ENERGY, duration=20.0)
    assert turnaround(song, 0.0, 20.0, span=15.0) is None
    assert turnaround(song, 0.0, 20.0, span=45.0) == song.loop


def test_the_plan_s_music_loops_on_the_loop_point() -> None:
    by_id, children, assets, clips = footage()
    assets = {**assets, "song": sourced("song", seconds=20.0).model_copy(update={"has_video": False})}
    speech = speech_of(clips)
    plan = plan_for([Selection(clip_id=clip.id, beat_id="b1") for clip in speech],
                    music=MusicPlan(cues=[MusicCue(asset_id="song", start=13.0)]))
    cuts = {"song": CleanCuts(beats=BEATS, energy=ENERGY, duration=20.0)}
    beds = music_beds(plan, plan_pieces(plan, by_id, children, assets), assets, cuts)
    start, end = cuts["song"].loop
    # Coming in 13s into a 20s song, it runs out before the cut does: it turns round at the
    # loop's end instead of playing its fade and jumping back to 13s.
    assert len(beds) >= 2 and beds[0].start == 13.0 and beds[0].end == end
    assert all(bed.start == start for bed in beds[1:])


def test_fitting_the_music_loops_it_on_the_beat(media: Path) -> None:
    video = catalog.import_asset(str(media / "wide.mp4"))["id"]
    song = catalog.import_asset(str(media / "song.mp3"))["id"]
    runtime.repo.save_analysis(MediaAnalysis(
        asset_id=song, duration=20.0, rhythm=Rhythm(tempo=240.0, beats=list(BEATS)),
        music=MusicSense(energy=list(ENERGY)),
    ))
    project = build_project([
        video_track(), insert("v1", video, 0, 10), insert("v2", video, 0, 10), insert("v3", video, 0, 10),
        {"action": "add_track", "track_id": "music", "track_type": "audio"},
        insert("song", song, 0, 20, track_id="music"),
    ], name="循環")
    edit(project, [{"action": "fit_track", "track_id": "music"}])
    start, end = loop_span(BEATS, ENERGY, 20.0)
    laid = layout(project, "music")
    # It plays to the loop's end rather than through its fading ending, then the loop, to the end of the video.
    assert laid[0][3:] == (0.0, end)
    assert all(row[3:] == (start, end) for row in laid[1:-1]) and laid[-1][3] == start
    assert laid[-1][2] == 30.0


def test_the_start_the_loop_note_says_plays_the_song_through_once() -> None:
    """#78: a start a little late makes the song turn round on its loop point, well before its
    end; the note says the latest start that does not, on the beat the cue would move to."""
    import re

    from app.engine.plan import compiled_duration, music_fit_notes

    by_id, children, assets, clips = footage()
    assets = {**assets, "song": sourced("song", seconds=20.0).model_copy(update={"has_video": False})}
    speech = speech_of(clips)
    cuts = {"song": CleanCuts(beats=BEATS, energy=ENERGY, duration=20.0)}

    def laid(start: float):
        plan = plan_for([Selection(clip_id=clip.id, beat_id="b1") for clip in speech],
                        music=MusicPlan(cues=[MusicCue(asset_id="song", start=start, cut_on_beat=True)]))
        pieces = plan_pieces(plan, by_id, children, assets)
        return compiled_duration(pieces), music_beds(plan, pieces, assets, cuts), music_fit_notes(plan, pieces, assets, cuts)

    total, _, _ = laid(0.0)
    assert total < 19.0
    late = round(20.0 - total + 0.04, 2)
    _, beds, notes = laid(late)
    assert len(beds) > 1
    [loops] = [note for note in notes if note.kind == "music_loops"]
    said = float(re.search(r"start the cue at ([\d.]+)s of the song or earlier", loops).group(1))
    _, beds, notes = laid(said)
    assert len(beds) == 1 and not any(note.kind == "music_loops" for note in notes)
