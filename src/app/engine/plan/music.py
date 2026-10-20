"""The music under a compiled cut, where a song sits badly under it, and the markers where each part begins."""

import math
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine.music import MIDDLE, energy_at, fullest
from app.engine.plan.beats import cue_entries, cue_stretches, song_entry, song_length, turnaround
from app.engine.plan.pieces import MUSIC_B_TRACK_ID, MUSIC_TRACK_ID, Note, Piece, compiled_duration
from app.engine.semantic import CleanCuts
from app.models.media import Asset
from app.models.plan import BeatRole, EditPlan, MusicCue


def plan_markers(plan: EditPlan, pieces: Sequence[Piece]) -> List[dict]:
    """Work out where each part of the video begins on the compiled timeline.

    A plan's beats are its shape, and until they are written onto the timeline
    that shape exists only in the plan: nothing downstream — where the music
    changes, how dense the B-roll is, which caption style applies — can read
    it. This is where they land.

    A beat whose first piece was merged into the one before it begins where
    that merged shot begins. Two pieces close enough to merge are one shot,
    and a marker partway through a shot would be claiming a cut that is not
    there.

    Args:
        plan: The plan being compiled.
        pieces: Its windows, in order, as `plan_pieces` merged them.

    Returns:
        One marker per beat that any piece belongs to, in timeline order,
        ready for a `set_markers` operation.
    """
    names = {beat.id: beat.name for beat in plan.beats}
    markers: List[dict] = []
    started: set = set()
    position = 0.0
    for piece in pieces:
        if piece.beat_id and piece.beat_id not in started and piece.beat_id in names:
            started.add(piece.beat_id)
            markers.append({
                "id": piece.beat_id,
                "name": names[piece.beat_id],
                "timeline_in": round(position, 3),
                "from_beat_id": piece.beat_id,
            })
        position += piece.played
    return markers

@dataclass(frozen=True)
class Bed:
    """One stretch of music worked out onto the compiled timeline.

    Attributes:
        asset_id: The song.
        start: Where it starts in the song, in seconds.
        end: Where it ends, in seconds.
        timeline_in: Where it lands on the timeline, in seconds.
        volume: Its gain.
        fade_in: Seconds of fade at its start.
        fade_out: Seconds of fade at its end.
        key: What it is, for matching a stretch adjusted by hand on the next
            compile: which cue it belongs to and which pass of that cue's song
            it is. Stable for as long as the cue comes in on the same beat.
        lane: 0 for the first music track, 1 for the second, which a song
            cross-fading in over another is laid on.
    """

    asset_id: str
    start: float
    end: float
    timeline_in: float
    volume: float
    fade_in: float
    fade_out: float
    key: Tuple[str, ...] = ()
    lane: int = 0

    @property
    def cue_beat(self) -> Optional[str]:
        """The beat of the cue this stretch belongs to; None for the cue that starts with the video."""
        beat = self.key[0].split(":", 1)[1] if self.key else "^"
        return None if beat == "^" else beat

    @property
    def track_id(self) -> str:
        """Say which music track this stretch goes on.

        Returns:
            The first music track, or the second for a song that comes in
            cross-fading over the one before.
        """
        return MUSIC_TRACK_ID if self.lane == 0 else MUSIC_B_TRACK_ID

def music_beds(
    plan: EditPlan,
    pieces: Sequence[Piece],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]] = None,
) -> List[Bed]:
    """Lay the plan's music out under its compiled cut.

    Each cue runs from where its beat begins to where the next cue's does, or
    to the end of the cut. A song shorter than its stretch loops back to where
    the cue came in, not to the top of the song: the cue chose that point, and
    the intro it skipped is no more welcome the second time round. A cue that
    cuts on the beat comes in on a beat, and loops back to the same one.

    The fades belong to the cue rather than to each loop: in at its start, out
    where the next cue takes over. Two cues meet end to end, the one going out
    fading as the one coming in fades up — a track holds one clip at a time,
    so the two never overlap.

    Args:
        plan: The plan.
        pieces: Its windows, in order, as `compile_pieces` settled them.
        assets: The files, songs included, keyed by asset ID.
        cuts: What is known about each file, keyed by asset ID; a song's beat
            is in here.

    Returns:
        The stretches of music, in timeline order. Empty when the plan has
        none, or none that plays.
    """
    if plan.music is None or not pieces:
        return []
    crossfade = plan.music.crossfade_seconds
    beds: List[Bed] = []
    lane, sounding = 0, False
    for cue, at, until in cue_stretches(plan, pieces):
        if cue.asset_id is None or until <= at:
            sounding = False
            continue
        entry = song_entry(cue, cuts.get(cue.asset_id) if cuts else None)
        # A song cross-fading in over the one before starts early, and from earlier in
        # itself by the same amount — so at the cut it is exactly where the cue said, on
        # its beat when it cuts on the beat. Never from before the top of the song: what
        # that leaves short is made up after the cut instead, by the song going out running
        # on past it, so a song coming in from its top still cross-fades.
        lead = min(crossfade, entry, at) if crossfade and sounding else 0.0
        tail = 0.0
        if crossfade and sounding and lead < crossfade:
            going = beds[-1]
            room = song_length(assets.get(going.asset_id))
            tail = round(max(0.0, min(crossfade - lead, until - at,
                                      (room - going.end) if room is not None else crossfade - lead)), 3)
            if tail:
                beds[-1] = replace(going, end=round(going.end + tail, 3))
        if lead or tail:
            lane = 1 - lane
            # The song going out fades over exactly the stretch the one coming in rises.
            beds[-1] = replace(beds[-1], fade_out=round(min(lead + tail, beds[-1].end - beds[-1].start), 3))
        else:
            lane = 0
        # Everything settled to the millisecond the timeline keeps before it is added up,
        # so a loop can neither run a hair past the file nor overlap the next one.
        last = song_length(assets.get(cue.asset_id))
        # A song that runs out turns round on its loop point when it has one, so it never plays
        # its ending and starts over; otherwise it comes back to where the cue came in.
        turn = turnaround(cuts.get(cue.asset_id) if cuts else None, entry, last, until - at)
        stop = last if turn is None else turn[1]
        parts: List[Tuple[float, float, float]] = []
        cursor, begin = round(at - lead, 3), round(entry - lead, 3)
        while until - cursor > 0.001:
            end = round(until - cursor + begin, 3) if stop is None else min(round(until - cursor + begin, 3), stop)
            if end - begin <= 0.001:
                break
            parts.append((begin, end, round(cursor, 3)))
            cursor = round(cursor + end - begin, 3)
            # Loops come back to where the cue came in, not to the lead-in before it.
            begin = entry if turn is None else turn[0]
        if not parts:
            sounding = False
            continue
        sounding = True
        lengths = [end - start for start, end, _ in parts]
        fade_in = lead + tail if lead or tail else min(cue.fade_in, lengths[0])
        fade_out = min(cue.fade_out, lengths[-1])
        if len(parts) == 1 and fade_in + fade_out > lengths[0]:
            # One short stretch with both fades on it: shrink them in proportion rather
            # than let one swallow the other.
            scale = lengths[0] / (fade_in + fade_out)
            fade_in, fade_out = fade_in * scale, fade_out * scale
        for number, (start, end, timeline_in) in enumerate(parts):
            beds.append(Bed(
                asset_id=cue.asset_id, start=start, end=end, timeline_in=timeline_in, volume=cue.volume,
                fade_in=round(fade_in, 3) if number == 0 else 0.0,
                fade_out=round(fade_out, 3) if number == len(parts) - 1 else 0.0,
                key=(f"cue:{cue.beat_id or '^'}", f"pass:{number + 1}"),
                lane=lane,
            ))
    return beds

# A song's fullest part landing this far from the start of the payoff does not land on it. Provisional.
PEAK_TOLERANCE_SECONDS = 4.0
# A cut ending this much before its song does, while the song is still going, cuts it off. Provisional.
CUT_OFF_SECONDS = 3.0

def _latest_start(cue: Optional[MusicCue], song: Optional[CleanCuts], length: float, span: float) -> Optional[float]:
    """Find the latest second of its song a cue can come in at and still play its stretch once through.

    As it is said: to the tenth, rounded down, so a start copied from it is
    never a hair late; and for a cue cut on the beat, at or before the beat
    `song_entry` would move it to, since a start moved later runs out.

    Args:
        cue: The cue, when known.
        song: What is known about the song.
        length: How long the song is, as `song_length` gives it.
        span: How long the cue holds the cut.

    Returns:
        The second, or None when the song is shorter than the stretch.
    """
    latest = length - span
    if latest < 0:
        return None
    if cue is not None and cue.cut_on_beat and song is not None and song.beats:
        on = [beat for beat in song.beats if beat <= latest + 0.001]
        if not on:
            return None
        latest = on[-1]
    return math.floor(round(latest * 10, 6)) / 10

def music_fit_notes(
    plan: EditPlan,
    pieces: Sequence[Piece],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]],
) -> List[Note]:
    """Say where a song sits badly under the cut, each with what would fix it.

    Three things, from the song's analysis: it runs out and starts again, its
    fullest part misses the payoff, the cut stops it while it is still going.
    These are for the AI to weigh and the user to hear about; none of them
    stops a compile, since a loop or an ending faded out can be just right.

    Args:
        plan: The plan.
        pieces: Its windows, in order.
        assets: The files, songs included.
        cuts: What is known about each file; a song's energy is in here.

    Returns:
        The notes, worth a look each.
    """
    beds = music_beds(plan, pieces, assets, cuts)
    if not beds:
        return []
    total = compiled_duration(pieces)
    payoff = next((beat.id for beat in plan.beats if beat.role == BeatRole.PAYOFF), None)
    climax, position, opens = None, 0.0, None
    for index, piece in enumerate(pieces):
        if piece.beat_id == payoff and climax is None:
            climax, opens = position, index
        position += piece.played
    # The cue the payoff opens under, told by the window it came in on rather than by seconds:
    # one cross-fading out as the payoff begins runs past it, but is leaving, not playing there.
    entries = cue_entries(plan, pieces)
    under = max((index for index in entries if opens is not None and index <= opens), default=None)
    playing = f"cue:{entries[under].beat_id or '^'}" if under is not None else None
    # Where each cue holds the cut, and the second of its song it is at when it takes over: a
    # cross-fade lends it a little either side, but this is the stretch it was asked to play under.
    spans = {f"cue:{cue.beat_id or '^'}": (at, until, song_entry(cue, cuts.get(cue.asset_id) if cuts else None))
             for cue, at, until in cue_stretches(plan, pieces)}
    cues = {f"cue:{cue.beat_id or '^'}": cue for cue, _, _ in cue_stretches(plan, pieces)}
    by_cue: Dict[str, List[Bed]] = {}
    for bed in beds:
        by_cue.setdefault(bed.key[0], []).append(bed)
    notes: List[Note] = []
    for passes in by_cue.values():
        song = assets.get(passes[0].asset_id)
        name = os.path.basename(song.path) if song is not None else passes[0].asset_id
        beat_id = passes[0].cue_beat
        first, last = passes[0], passes[-1]
        heard = round(last.timeline_in + last.end - last.start - first.timeline_in, 1)
        known = (cuts or {}).get(first.asset_id)
        length = song_length(song)
        span = spans.get(first.key[0])
        # The one start both notes below say, worked out the way the music is laid (#78).
        latest = (_latest_start(cues.get(first.key[0]), known, length, span[1] - span[0])
                  if span is not None and length is not None else None)
        if len(passes) > 1:
            earlier = (f"start the cue at {latest:.1f}s of the song or earlier (set_music_cue) — one that runs "
                       "out turns round on its loop point, before its ending — or " if latest is not None else "")
            notes.append(Note(
                f"{name} runs out at {passes[1].timeline_in:.1f}s of the cut and starts again from "
                f"{passes[1].start:g}s of the song, since the part it plays under is {heard:g}s. To have it play "
                f"once, {earlier}choose a song of at least {heard:.0f}s (find_music with min_seconds), or let "
                f"another cue take over before {passes[1].timeline_in:.1f}s",
                "music_loops", beat_id=beat_id, seconds=passes[1].timeline_in, look=True,
            ))
        energy = known.energy if known is not None else None
        if not energy or length is None:
            continue
        peak = fullest(energy)
        ends_with_cut = abs(last.timeline_in + last.end - last.start - total) < 0.01
        # How much of the song the part plays, from its own length rather than the rounded one said above.
        played = last.timeline_in + last.end - last.start - first.timeline_in

        def cut_off_from(start: float) -> Optional[float]:
            """How long before the song's end the cut stops it, still going, when the cue starts here."""
            stops = start + played
            going = energy_at(energy, stops)
            if ends_with_cut and length - stops > CUT_OFF_SECONDS and going is not None and going >= MIDDLE:
                return length - stops
            return None

        def lands_at(start: float) -> Optional[float]:
            """Where in the cut the fullest part is heard when the cue starts here, or None."""
            at, until, _ = span
            return at + peak[0] - start if start <= peak[0] < start + until - at else None

        # Only the song the payoff opens under: one another song takes over from before then, or
        # one fading out as it begins, cannot land its fullest part there, wherever it starts.
        aims = (len(passes) == 1 and peak is not None and climax is not None and span is not None
                and first.key[0] == playing)
        if aims:
            lands = lands_at(span[2])
            if lands is None or abs(lands - climax) > PEAK_TOLERANCE_SECONDS:
                start = peak[0] - (climax - span[0])
                fix = (f"start the cue at {start:.1f}s of the song (set_music_cue) to land it there"
                       if 0 <= start < length else "a song that builds later would land it there")
                short = cut_off_from(start) if 0 <= start < length else None
                if short is not None:
                    fix += (f" — but then the cut ends {short:.0f}s before the song does, while it is still going, "
                            "and it is faded out: with this song, landing its fullest part on the payoff and ending "
                            "with the video cannot both be had, so choose one, or another song")
                notes.append(Note(
                    f"the fullest part of {name} ({peak[0]}–{peak[1]}s of the song) "
                    + (f"lands at {lands:.1f}s of the cut" if lands is not None else "is never heard")
                    + f", and the payoff ({payoff}) starts at {climax:.1f}s; {fix}",
                    "music_peak_off", beat_id=beat_id, seconds=climax, look=True,
                ))
        short = cut_off_from(last.end - played)
        if short is not None:
            # Rounded down as it is said: a start a hundredth late leaves the song short of the part, and it
            # loops. The same start the loop note says, so following either one never brings the other on.
            start = latest if latest is not None else math.floor((length - played) * 10) / 10
            cost = ""
            if aims and start >= 0:
                moved = lands_at(start)
                if moved is None or abs(moved - climax) > PEAK_TOLERANCE_SECONDS:
                    cost = (" — but then its fullest part "
                            + (f"lands at {moved:.1f}s of the cut" if moved is not None else "is never heard")
                            + f", not on the payoff at {climax:.1f}s: the two cannot both be had with this song, "
                              "so choose one")
            notes.append(Note(
                f"the cut ends {short:.0f}s before {name} does, while it is still going; it is faded "
                f"out over {last.fade_out:g}s. For the song to end with the video, start the cue at "
                f"{max(0.0, start):.1f}s of the song (set_music_cue)" + ("" if start >= 0 else " — it is shorter than the part, so "
                                                         "it loops first") + cost + ", or give it a longer fade_out",
                "music_cut_off", beat_id=beat_id, seconds=total, look=True,
            ))
    return notes
