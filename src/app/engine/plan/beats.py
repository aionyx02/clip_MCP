"""Fitting the windows to the music, cutting on the beat and joining them: compile_pieces and plan_pieces."""

import bisect
import math
from dataclasses import replace
from typing import AbstractSet, Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine.plan.pieces import (
    BREATH_SECONDS, MERGE_GAP_SECONDS, MIN_TRANSITION_SECONDS, Note, Piece, edge_of, first_clips, merge_pieces,
    off_bad_frames, on_the_sound, pace_of, selected_pieces, trimmed_ends, without_pauses, without_retakes,
)
from app.engine.semantic import EDGE_TOLERANCE_SECONDS, CleanCuts
from app.models.media import Asset
from app.models.plan import EditPlan, MusicCue
from app.models.semantic import SemanticClip


# How far a cut may move to land on the beat of the music. At 120 beats a minute no cut
# is ever more than a quarter of a second from one, so this reaches every beat of most
# music and leaves the slow ones alone rather than let the song rewrite the pacing.
# Provisional until there is a corpus to tune it against.
BEAT_SNAP_SECONDS = 0.3
# A cut this close to a beat is already on it — well inside one frame at 60fps, about 17ms.
ON_BEAT_SECONDS = 0.005

# A join given this much less than it asked for is given all of it: a frame at 30 fps, less
# than anybody sees, and more than the milliseconds rounding takes off (#89). Provisional.
JOIN_TOLERANCE_SECONDS = 1 / 30

def beginnings(pieces: Sequence[Piece]) -> Dict[str, int]:
    """Say which window each beat of the plan begins with.

    The same rule `plan_markers` follows, because a cue comes in where the
    marker is: a beat begins at the first window carrying it, and a beat whose
    first window was merged into the one before it has no window of its own
    and so does not begin anywhere.

    Args:
        pieces: The windows, in order.

    Returns:
        The index of each beat's first window, keyed by beat ID.
    """
    first: Dict[str, int] = {}
    for index, piece in enumerate(pieces):
        if piece.beat_id and piece.beat_id not in first:
            first[piece.beat_id] = index
    return first

def cue_entries(plan: EditPlan, pieces: Sequence[Piece]) -> Dict[int, MusicCue]:
    """Say which window each music cue comes in on.

    Args:
        plan: The plan.
        pieces: Its windows, in order.

    Returns:
        The cue coming in at each window that has one, keyed by the window's
        index. A cue whose beat never begins is left out; `check_plan` refuses
        a plan with one.
    """
    if plan.music is None:
        return {}
    first = beginnings(pieces)
    entries: Dict[int, MusicCue] = {}
    for cue in plan.music.cues:
        at = 0 if cue.beat_id is None else first.get(cue.beat_id)
        if at is not None:
            entries[at] = cue
    return entries

def cue_stretches(plan: EditPlan, pieces: Sequence[Piece]) -> List[Tuple[MusicCue, float, float]]:
    """Say where in the cut each music cue holds: from where its beat begins to where the next cue's does.

    Args:
        plan: The plan.
        pieces: Its windows, in order.

    Returns:
        `(cue, at, until)` in seconds of the cut, in the order they come in;
        the last runs to the end of the cut.
    """
    starts: List[float] = []
    position = 0.0
    for piece in pieces:
        starts.append(round(position, 3))
        position += piece.played
    entries = sorted(cue_entries(plan, pieces).items())
    return [(cue, starts[index], starts[entries[order + 1][0]] if order + 1 < len(entries) else round(position, 3))
            for order, (index, cue) in enumerate(entries)]

def song_length(source: Optional[Asset]) -> Optional[float]:
    """Say how long a song runs, to the millisecond the timeline keeps.

    Rounded down, so a clip cut to it never runs past the file. The beat grid
    and the music laid on the track both loop on this one number: a loop a
    millisecond out would be thirty milliseconds out after thirty of them.

    Args:
        source: The song's file.

    Returns:
        The length in seconds, or None when it is not known.
    """
    if source is None or source.duration is None:
        return None
    return math.floor(float(source.duration) * 1000) / 1000

def song_entry(cue: MusicCue, song: Optional[CleanCuts]) -> float:
    """Work out where in its song a cue comes in.

    Where it was asked to — unless the cue cuts on the beat, in which case the
    song comes in on the first beat at or after that point. A cut is only on
    the beat if the beat is counted from somewhere that is one, and the cut
    the song comes in on is the first of them.

    Args:
        cue: The cue.
        song: What is known about its song, if anything.

    Returns:
        The second of the song it starts playing from, to the millisecond.
    """
    if cue.cut_on_beat and song is not None and song.beats:
        at = bisect.bisect_left(song.beats, cue.start - ON_BEAT_SECONDS)
        if at < len(song.beats):
            return round(song.beats[at], 3)
    return round(cue.start, 3)

def turnaround(
    song: Optional[CleanCuts], entry: float, length: Optional[float], span: float,
) -> Optional[Tuple[float, float]]:
    """Say whether a cue's song loops on its loop point, and where.

    Only when it has to: a song that lasts the cue from where it comes in
    plays straight through, outro and all. One that runs out turns round at
    the end of its loop, on the beat, and carries on from its start, rather
    than playing its ending and starting over from where the cue came in.

    Args:
        song: What is known about the song.
        entry: Where in the song the cue comes in.
        length: How long the song is, or None when not known.
        span: How long the cue plays on the timeline.

    Returns:
        `(start, end)` of the loop in song seconds, or None to loop the old
        way, back to `entry` from the song's end.
    """
    if song is None or length is None or span <= length - entry + 0.001:
        return None
    loop = song.loop
    if loop is None or entry >= loop[1] - 1.0:
        return None
    return loop

def _beats_near(
    beats: Sequence[float],
    entry: float,
    length: Optional[float],
    entered_at: float,
    seconds: float,
    turn: Optional[Tuple[float, float]] = None,
) -> List[float]:
    """Find the beats of a playing song either side of a moment of the timeline.

    The song plays from `entry` and loops the way `music_beds` lays it: back
    to `entry` when it runs out, or, given a `turn`, from its end to its start
    once it gets there. The beats repeat with the length of one pass.

    Args:
        beats: The song's beats, in song seconds.
        entry: Where in the song it came in.
        length: How long the song is, or None when that is not known and it is
            taken never to loop.
        entered_at: Where on the timeline it came in.
        seconds: The moment of the timeline to look near.
        turn: The loop the song turns round on, from `turnaround`, if any.

    Returns:
        The beat just before and the beat just after, on the timeline, nearer
        one first — both, because the nearer one may be out of reach where the
        other is not. Empty when the song has no beat after where it came in.
    """
    if turn is not None:
        start, end = turn
        first = [entered_at + beat - entry for beat in beats if entry <= beat < end]
        period = end - start
        looped = [beat - start for beat in beats if start <= beat < end]
        since = seconds - (entered_at + end - entry)
        around = list(first)
        if looped and period > 0:
            for n in range(max(0, math.floor(since / period) - 1), max(0, math.floor(since / period) + 2)):
                around += [entered_at + end - entry + n * period + beat for beat in looped]
        before = [beat for beat in around if beat <= seconds]
        after = [beat for beat in around if beat > seconds]
        near = ([max(before)] if before else []) + ([min(after)] if after else [])
        return sorted(near, key=lambda beat: abs(beat - seconds))
    heard = [beat - entry for beat in beats if beat >= entry and (length is None or beat < length)]
    if not heard:
        return []
    period = length - entry if length is not None and length > entry else None
    since = seconds - entered_at
    passes = [0] if period is None else [
        n for n in range(math.floor(since / period) - 1, math.floor(since / period) + 2) if n >= 0
    ]
    around: List[float] = []
    for n in passes:
        began = n * period if period is not None else 0.0
        at = bisect.bisect_left(heard, since - began)
        around += [entered_at + began + heard[index] for index in (at - 1, at) if 0 <= index < len(heard)]
    before = [beat for beat in around if beat <= seconds]
    after = [beat for beat in around if beat > seconds]
    near = ([max(before)] if before else []) + ([min(after)] if after else [])
    return sorted(near, key=lambda beat: abs(beat - seconds))

def _may_close_at(
    piece: Piece,
    following: Piece,
    end: float,
    known: CleanCuts,
    asset: Optional[Asset],
    breath: float = BREATH_SECONDS,
) -> bool:
    """Say whether a window may end somewhere else to put its cut on the beat.

    Moving a cut for the music is a matter of taste, so it gets none of the
    leeway moving one off a word does: it may only move through silence. It
    may not end before the last word has had its breath, may not reach into
    the next word, may not show a bad frame, and may not run into the window
    that follows when that plays on from the same file.

    Args:
        piece: The window.
        following: The window after it.
        end: Where it would end instead, in source seconds.
        known: What is known about its file, which has to have been
            transcribed: otherwise nobody knows the move is through silence.
        asset: The file.

    Returns:
        True when the window may end there.
    """
    if end <= piece.start or not known.is_clean(end, opens=False):
        return False
    if asset is not None and asset.duration is not None and end > float(asset.duration):
        return False
    if end < piece.end:
        last = known.last_word_in(piece.start, piece.end)
        return last is None or end >= min(piece.end, last + breath)
    if any(piece.end - EDGE_TOLERANCE_SECONDS < opened < end + breath for opened, _ in known.words):
        return False
    if any(start < end and stop > piece.end for start, stop in known.bad_picture):
        return False
    if following.asset_id == piece.asset_id and following.start >= piece.start:
        return end <= following.start - MERGE_GAP_SECONDS
    return True

def on_the_beat(
    plan: EditPlan,
    pieces: Sequence[Piece],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]],
    held: AbstractSet[Tuple[str, ...]] = frozenset(),
) -> Tuple[List[Piece], List[str]]:
    """Move each picture cut onto the beat of the music playing under it, where it can go.

    Only under a cue that asked for it: cutting on the beat is a style, right
    for a montage and wrong for an interview, so it is the plan's call. A cut
    moves by ending the window before it somewhere else, which shifts every
    cut after it by the same amount — so they are settled in order, each one
    against where the ones before it finally landed.

    The cut a new cue comes in on is settled against the song going out, not
    the one coming in: the new song starts on that cut, and starts on a beat
    of its own, so it is on both.

    Args:
        plan: The plan.
        pieces: The windows, in order, merged.
        assets: The files they play from, keyed by asset ID.
        cuts: What is known about each file, songs included, keyed by asset ID.
        held: What the windows that may not move were made from: clips pinned
            by hand, which play as they were left, so every cut after them is
            settled against where they really end.

    Returns:
        `(pieces, notes)`.
    """
    entries = cue_entries(plan, pieces)
    if not cuts or not any(cue.cut_on_beat for cue in entries.values()):
        return list(pieces), []
    out = list(pieces)
    moved: List[float] = []
    stuck, unknown = 0, 0
    moved_in: List[Piece] = []
    stuck_in: List[Piece] = []
    unknown_in: List[Piece] = []
    playing: Optional[MusicCue] = None
    song: Optional[CleanCuts] = None
    length: Optional[float] = None
    turn: Optional[Tuple[float, float]] = None
    entry, entered_at, position = 0.0, 0.0, 0.0
    starts = sorted(entries)
    for index in range(len(out)):
        if index in entries:
            playing = entries[index]
            song = cuts.get(playing.asset_id) if playing.asset_id else None
            length = song_length(assets.get(playing.asset_id)) if playing.asset_id else None
            entry, entered_at = song_entry(playing, song), position
            # How long it plays, as the windows stand: ends moved for the beat move it by a fraction.
            following = next((at for at in starts if at > index), len(out))
            span = sum(piece.played for piece in out[index:following])
            turn = turnaround(song, entry, length, span)
        piece = out[index]
        on_beat = playing is not None and playing.cut_on_beat and song is not None and song.beats
        # A title card is as long as it was asked to be; the cut out of it stays where it falls.
        if index + 1 < len(out) and on_beat and piece.card is None and piece.from_clip_ids not in held:
            closing_at = position + piece.played
            shifts = [
                round(beat - closing_at, 3)
                for beat in _beats_near(song.beats, entry, length, entered_at, closing_at, turn)
            ]
            known = cuts.get(piece.asset_id)
            # A shift is timeline seconds; the window's end moves by that much of its file.
            if not shifts or abs(shifts[0]) <= ON_BEAT_SECONDS:
                pass
            elif known is None or not known.transcribed:
                unknown += 1
                unknown_in.append(piece)
            else:
                reachable = [
                    shift for shift in shifts
                    if abs(shift) <= BEAT_SNAP_SECONDS and _may_close_at(
                        piece, out[index + 1], round(piece.end + shift * piece.speed, 3), known,
                        assets.get(piece.asset_id), pace_of(plan).breath,
                    )
                ]
                if reachable:
                    end = round(piece.end + reachable[0] * piece.speed, 3)
                    out[index] = replace(piece, end=end)
                    moved.append(abs(reachable[0]))
                    moved_in.append(piece)
                else:
                    stuck += 1
                    stuck_in.append(piece)
        position += out[index].played

    notes: List[str] = []
    if moved:
        notes.append(Note(f"{len(moved)} cut(s) moved onto the beat of the music, {max(moved):.2f}s at most",
                          "cuts_moved_onto_the_beat", clip_ids=first_clips(moved_in), seconds=max(moved)))
    if stuck:
        notes.append(Note(
            f"{stuck} cut(s) are off the beat: reaching it would cut into a word, show a bad frame, or move "
            f"further than {BEAT_SNAP_SECONDS:g}s, so they were left where they were",
            "cuts_off_the_beat", clip_ids=first_clips(stuck_in), look=True,
        ))
    if unknown:
        notes.append(Note(
            f"{unknown} cut(s) are off the beat because their footage was never transcribed, so there is no "
            "knowing whether moving them would cut into a word",
            "cuts_off_the_beat", clip_ids=first_clips(unknown_in), look=True,
        ))
    return out, notes

def compile_pieces(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]] = None,
) -> Tuple[List[Piece], List[str]]:
    """Work out every window the plan puts on the timeline, cleaned up, in order.

    The cleaning runs in this order for a reason. Edges the transcript placed
    inside somebody's sound are put back on it first, since every later stage
    trusts where a window starts and ends. Retakes go next: there is
    no sense taking the pauses out of a take about to be dropped. Pauses next,
    which is the stage that splits a window in two. Then the ends, which want
    the first and last window as they will finally be. Bad frames after that,
    so they have the final say over every edge the others made. The beat comes
    after the merge, because it moves cuts, and a cut is only a cut once two
    windows have been left apart.

    All of it depends on `cuts`. Without them the compiler knows nothing about
    the footage, and a stage that cleaned anyway would be guessing.

    Args:
        plan: Plan to lay out.
        clips: The timeline's clips, keyed by ID.
        children: Each section's own clips in time order, keyed by section ID.
        assets: The assets they play from, keyed by asset ID.
        cuts: What is known about each file, keyed by asset ID. Without them a
            `head` or `tail` lands at exactly the second it was given, which is
            how a cut ends mid-syllable, and nothing is cleaned.

    Returns:
        `(pieces, notes)` — the windows, merged where they nearly touch, and a
        line for each cleaning stage that did something.
    """
    pieces = selected_pieces(plan, clips, children, assets, cuts)
    chosen = list(pieces)
    notes: List[str] = []
    pace = pace_of(plan)
    for stage in (on_the_sound, without_retakes, without_pauses, trimmed_ends, off_bad_frames):
        pieces, said = stage(pieces, cuts, pace)
        notes.extend(said)
    merged = merge_pieces(pieces)
    if pace.speed != 1.0:
        # Set once the windows are settled: every stage before this works in the file's own
        # seconds, and only where things land on the timeline cares how fast they play.
        merged = [piece if piece.own_speed else replace(piece, speed=round(piece.speed * pace.speed, 4))
                  for piece in merged]
    pieces, said = on_the_beat(plan, merged, assets, cuts)
    notes += said
    # Last, because the beat moves cuts, and how much a transition can reach back into
    # depends on exactly where its window starts.
    pieces, said = _joined_beats(plan, pieces, assets, cuts)
    notes += said
    return pieces, notes + _left_out(chosen, pieces, notes)

def _left_out(chosen: Sequence[Piece], pieces: Sequence[Piece], notes: Sequence[str]) -> List[str]:
    """Say which windows the plan chose play nowhere in the cut, when no stage said it took them out.

    The plan says what is in the cut; a window lost on the way without a word
    is the cut quietly disagreeing with it (#91).

    Args:
        chosen: The windows the selections made, before cleaning.
        pieces: The windows after it.
        notes: What the cleaning stages said, naming the windows they dropped.

    Returns:
        A note per window lost unsaid.
    """
    dropped = {clip_id for note in notes if isinstance(note, Note) and note.kind == "retakes_dropped"
               for clip_id in note.clip_ids}
    lost = [piece for piece in chosen
            if not any(other.asset_id == piece.asset_id and min(other.end, piece.end) > max(other.start, piece.start)
                       for other in pieces)
            and not set(first_clips([piece])) & dropped]
    if not lost:
        return []
    return [Note(
        "these windows the plan chose play nowhere in the cut: "
        + "; ".join(f"{edge_of(piece, 'starts', piece.start)} to {piece.end:.2f}s" for piece in lost)
        + ". Compile again and say what was asked if it stays",
        "selections_lost", clip_ids=first_clips(lost), look=True,
    )]

def _reach(kind: str) -> float:
    """How much of its length a transition reaches back before the cut: a dip only half,
    since the colour covers its first half; a dissolve or a wipe all of it."""
    return 0.5 if kind == "dip" else 1.0

def picture_needed(wanted, piece: Piece) -> float:
    """Seconds of the file a transition needs before the shot it brings in, at that shot's speed."""
    return wanted.seconds * _reach(wanted.kind) * piece.speed

def _milliseconds(seconds: float) -> float:
    """Floor a length to whole milliseconds, rounding first so one already whole is not floored short."""
    return math.floor(round(seconds * 1000, 6)) / 1000

def _lent_tail(
    out: List[Piece], at: int, assets: Mapping[str, Asset], cuts: Optional[Mapping[str, CleanCuts]], short: float,
) -> float:
    """Let the shot before a beat run on into its own quiet, to make up a pause its next shot cannot give.

    Only into quiet the file is known to have: up to the next thing said or
    heard in it, never into what the cut plays later of the same take, never
    past the end of the file. A card or a file nobody transcribed has nothing
    known there, so it lends nothing.

    Args:
        out: The windows, changed in place.
        at: Index of the beat's first window.
        assets: The files, keyed by asset ID, for how long each runs.
        cuts: What is known about each file, keyed by asset ID.
        short: Seconds of the timeline still wanted.

    Returns:
        The seconds of the timeline lent.
    """
    if short <= 0 or at <= 0:
        return 0.0
    before = out[at - 1]
    known = (cuts or {}).get(before.asset_id)
    if before.card is not None or known is None or not known.transcribed:
        return 0.0
    # Anywhere else in the cut, before or after: running on into footage it plays elsewhere is
    # hearing it twice.
    later = [other.start for index, other in enumerate(out)
             if index != at - 1 and other.asset_id == before.asset_id and other.start >= before.end]
    source = assets.get(before.asset_id)
    ends = [float(source.duration)] if source is not None and source.duration is not None else []
    room = min([known.first_sound_after(before.end), *later, *ends]) - before.end
    lent = _milliseconds(min(short * before.speed, max(0.0, room)))
    if lent <= 0:
        return 0.0
    out[at - 1] = replace(before, end=round(before.end + lent, 3))
    return lent / before.speed

def _joined_beats(
    plan: EditPlan,
    pieces: Sequence[Piece],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]],
) -> Tuple[List[Piece], List[str]]:
    """Put each beat's pause, transition and sound lead on the window it begins with.

    The pause first: it opens the window earlier, into the quiet of its shot
    before the first word, and the transition then runs in from where the
    window now starts.

    Both are fed from before the window's start in its file, and a shot that
    begins at the very start of its file has nothing there. Refusing the plan
    for that would make every transition a guess about where each file starts,
    so the transition is shortened to what the file has, and the notes say by
    how much. Shortening is arithmetic on a number the plan gave, not a
    judgement about the cut, which is why it is done rather than handed back.

    Args:
        plan: The plan.
        pieces: Its windows, settled.
        assets: The assets they play from, keyed by asset ID.
        cuts: What is known about each file, keyed by asset ID: where the
            words are, so a pause never opens a window onto the line before.

    Returns:
        `(pieces, notes)`.
    """
    first = beginnings(pieces)
    out = list(pieces)
    notes: List[str] = []
    for position, beat in enumerate(plan.beats):
        at = first.get(beat.id)
        if position > 0 and beat.pause_before is not None and at:
            piece = out[at]
            known = (cuts or {}).get(piece.asset_id)
            # Back no further than the last word before it, nor into what the cut already plays of
            # the same take: opening onto either would hear it twice or start mid-line.
            said = [known.last_sound_before(piece.start)] if known is not None else []
            played = [other.end for other in out[:at] if other.asset_id == piece.asset_id and other.end <= piece.start]
            quiet = piece.start - max(said + played, default=0.0)
            # The picture the beat's transition runs in from is kept for it: opening the shot onto
            # it left the transition nothing before the shot, and it was refused (#95). Whatever
            # of the pause that costs is made up from the shot going out, below.
            source = assets.get(piece.asset_id)
            if beat.transition_in is not None and not (source is not None and source.still):
                quiet = min(quiet, piece.start - picture_needed(beat.transition_in, piece))
            # Rounded first, so a time already in whole milliseconds is not floored a millisecond short,
            # and the pause asked for is held to the same milliseconds, so getting all of it is not
            # reported as falling short of it by a rounding error (#76).
            wanted = _milliseconds(beat.pause_before * piece.speed)
            fits = min(wanted, _milliseconds(max(0.0, quiet)))
            if fits > 0:
                out[at] = replace(piece, start=round(piece.start - fits, 3))
            # Seconds of the timeline still missing, made up from the quiet at the end of the shot
            # going out — the other side of the same join — before any of it is reported (#76).
            lent = _lent_tail(out, at, assets, cuts, _milliseconds(beat.pause_before - fits / piece.speed))
            given = round(fits / piece.speed + lent, 3)
            if given < beat.pause_before - JOIN_TOLERANCE_SECONDS:
                notes.append(Note(
                    f"beat {beat.id}: it opens on {given:g}s of quiet rather than {beat.pause_before:g}s, "
                    "all its first shot has before somebody speaks or the cut has already played — less any its "
                    "transition runs in from — and all the shot before it has after its last word",
                    "join_shortened", beat_id=beat.id, clip_ids=first_clips([piece]),
                    seconds=given, look=True,
                ))
        if position == 0 or (beat.transition_in is None and beat.sound_lead is None):
            continue
        if at is None or at == 0:
            notes.append(Note(
                f"beat {beat.id} has no shot of its own to begin with — its first one was joined to the shot "
                "before it — so it comes in on a straight cut",
                "join_left_out", beat_id=beat.id, look=True,
            ))
            continue
        piece, before = out[at], out[at - 1]
        # Seconds of the timeline the file can supply before this window, at its speed. A photo
        # is the same picture before its window as in it, so it has all the picture asked for.
        room = piece.start / piece.speed
        source = assets.get(piece.asset_id)
        picture_room = math.inf if source is not None and source.still else room
        changes: Dict[str, object] = {}
        wanted = beat.transition_in
        if wanted is not None:
            reach = _reach(wanted.kind)
            fits = math.floor(min(wanted.seconds, picture_room / reach, before.played) * 1000) / 1000
            if fits < MIN_TRANSITION_SECONDS:
                # `check_plan` refuses this with the ways to fix it; a preview built
                # without checking still says what happened.
                notes.append(Note(
                    f"beat {beat.id}: its {wanted.kind} was left out — its first shot starts "
                    f"{piece.start:.2f}s into its file, with no picture before it to run in from",
                    "join_left_out", beat_id=beat.id, clip_ids=first_clips([piece]), look=True,
                ))
            else:
                if fits < wanted.seconds - JOIN_TOLERANCE_SECONDS:
                    short = picture_needed(wanted, piece) - piece.start
                    notes.append(Note(
                        f"beat {beat.id}: its {wanted.kind} was shortened from {wanted.seconds:g}s to {fits:g}s, "
                        f"all the picture its file has before the shot; for the whole {wanted.seconds:g}s, start "
                        f"that shot {short:.2f}s later (`set_trim` with a later `from_seconds`)",
                        "join_shortened", beat_id=beat.id, clip_ids=first_clips([piece]), seconds=fits, look=True,
                    ))
                changes["transition"] = (wanted.kind, fits, wanted.through, wanted.direction)
        if beat.sound_lead is not None:
            asset = assets.get(piece.asset_id)
            if asset is not None and not asset.has_audio:
                notes.append(Note(f"beat {beat.id}: its first shot has no sound to lead with, so it leads with none",
                                  "join_left_out", beat_id=beat.id, clip_ids=first_clips([piece]), look=True))
            else:
                fits = math.floor(min(beat.sound_lead, room, before.played) * 1000) / 1000
                if fits < beat.sound_lead - JOIN_TOLERANCE_SECONDS:
                    notes.append(Note(
                        f"beat {beat.id}: its sound comes in {fits:g}s early rather than {beat.sound_lead:g}s, "
                        "all the sound its file has before the shot",
                        "join_shortened", beat_id=beat.id, clip_ids=first_clips([piece]), seconds=fits, look=True,
                    ))
                if fits > 0:
                    changes["sound_lead"] = fits
        if changes:
            out[at] = replace(piece, **changes)
    return out, notes

def plan_pieces(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]] = None,
) -> List[Piece]:
    """Work out every window the plan puts on the timeline, in order.

    Args:
        plan: Plan to lay out.
        clips: The timeline's clips, keyed by ID.
        children: Each section's own clips in time order, keyed by section ID.
        assets: The assets they play from, keyed by asset ID.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        The windows, merged where they nearly touch. Use `compile_pieces`
        where what the compiler did to them also matters.
    """
    return compile_pieces(plan, clips, children, assets, cuts)[0]
