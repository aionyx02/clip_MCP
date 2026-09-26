"""Turning an edit plan into timeline operations, and checking one before that happens.

This is the deterministic half of the split the whole design rests on. A model
decides which footage is used and what it is for; everything from there — where
each cut lands, how much air it gets, what gets merged, how long the result
runs — happens here, as a pure function of the plan and the semantic timeline.

Two things follow. The same plan always compiles to the same cut, so an edit
can be reproduced and two plans can be compared rather than two videos. And
every rule in here is a unit test rather than a sentence in a guide hoping to
be followed.

Nothing is repaired on the way through. A plan that does not hold up is handed
back with the reasons, because a compiler that quietly fixed things would be
making the decisions it was built to stay out of.
"""

import bisect
import math
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.models.media import Asset
from app.models.plan import EditPlan, MusicCue, Selection, TrimKind
from app.engine.semantic import EDGE_TOLERANCE_SECONDS, CleanCuts, voice_gains
from app.models.semantic import ClipKind, SemanticClip, SemanticTimeline
from app.models.timeline import Clip, Project

# Air left around a cut, taken from the clip's measured headroom, so a line does not
# begin the instant the picture does. Never more than the headroom allows. The default
# for a plan's `pacing.breath_seconds`. Provisional.
BREATH_SECONDS = 0.1
# Two pieces this close together become one clip; a gap this short reads as a stumble.
MERGE_GAP_SECONDS = 0.3
# How far a `head` or `tail` cut may move to stop landing in the middle of a word.
# Measured on real footage, finishing the word in progress costs about a fifth of a
# second and never more than about two, so this buys a clean cut for almost nothing.
# Provisional until there is a corpus to tune it against.
SNAP_SECONDS = 1.0
# How long a silence has to run before it is dead air rather than a breath. It has to
# exceed `2 * BREATH_SECONDS + MERGE_GAP_SECONDS`: taking a pause out leaves a breath on
# each side, and if what is left between the two halves fits inside the merge gap they
# are joined straight back together and the whole stage is a no-op nobody can see.
# Provisional until there is a corpus to tune it against.
PAUSE_SECONDS = 0.6
# How alike two neighbouring windows from one file have to sound before they are taken
# for two goes at the same line. 1.0 is word for word. Provisional.
RETAKE_SIMILARITY = 0.85
# How much of the opening a lead-in may cost before it is left alone, and the same for a
# trailing reach towards the camera. Enough to lose someone gathering themselves, small
# enough that an establishing shot held on purpose survives it. Provisional.
HEAD_TRIM_SECONDS = 2.0
TAIL_TRIM_SECONDS = 2.0
VIDEO_TRACK_ID = "main"
MUSIC_TRACK_ID = "music"
# Where the incoming song goes while two cross-fade: a track holds one clip at a time, so
# the songs alternate between the two.
MUSIC_B_TRACK_ID = "music_b"
BROLL_TRACK_ID = "broll"
# Every track a compile owns and rebuilds; any other track belongs to whoever made it.
COMPILED_TRACKS = (VIDEO_TRACK_ID, MUSIC_TRACK_ID, MUSIC_B_TRACK_ID, BROLL_TRACK_ID)
# How long one shot of covering picture may run. Under the first it reads as a flicker
# rather than a shot; over the second the viewer has lost the thread of what is being
# said underneath. Both provisional until there is a corpus to tune them against.
BROLL_MIN_SECONDS = 1.0
BROLL_MAX_SECONDS = 6.0
# How much of one beat may be under covering picture. A beat that is mostly B-roll is not
# a beat with B-roll on it, it is a beat made of B-roll with a voice over it — which is a
# different thing and should be selected as such. Provisional.
BROLL_MAX_SHARE = 0.4
# How long one shot has to hold before it is worth offering something to cover it with.
# Provisional.
BROLL_LONG_SHOT_SECONDS = 8.0
# How far apart two voices have to be before it is worth saying they do not match. Three
# decibels is about where a difference stops being something you have to listen for.
# Provisional.
VOICE_SPREAD_DB = 3.0
# How far a cut may move to land on the beat of the music. At 120 beats a minute no cut
# is ever more than a quarter of a second from one, so this reaches every beat of most
# music and leaves the slow ones alone rather than let the song rewrite the pacing.
# Provisional until there is a corpus to tune it against.
BEAT_SNAP_SECONDS = 0.3
# A cut this close to a beat is already on it — well inside one frame at 60fps, about 17ms.
ON_BEAT_SECONDS = 0.005
# How far the compiled length may sit from what the plan asked for before it is worth saying.
LENGTH_TOLERANCE = 0.1

@dataclass(frozen=True)
class Pace:
    """How tight a cut is, as the compiler applies it.

    Attributes:
        breath: Air left before and after every cut, in seconds.
        pause: How long a silence inside a shot has to run before it is taken
            out, in seconds.
        speed: How fast the whole sequence plays.
    """

    breath: float = BREATH_SECONDS
    pause: float = PAUSE_SECONDS
    speed: float = 1.0

def pace_of(plan: EditPlan) -> Pace:
    """Read how tight a plan asks its cut to be.

    Args:
        plan: The plan.

    Returns:
        Its pacing, with the defaults where it names none.
    """
    return Pace(
        breath=BREATH_SECONDS if plan.pacing.breath_seconds is None else plan.pacing.breath_seconds,
        pause=PAUSE_SECONDS if plan.pacing.pause_seconds is None else plan.pacing.pause_seconds,
        speed=1.0 if plan.pacing.speed is None else plan.pacing.speed,
    )

@dataclass(frozen=True)
class Piece:
    """One window of source footage on the compiled timeline.

    Attributes:
        asset_id: File it plays from.
        start: Where it starts in that file, in seconds.
        end: Where it ends, in seconds.
        from_clip_ids: The semantic clips it was made from, in order. This is
            what a second compile matches a hand-adjusted clip on, so it is
            the piece's identity rather than a note about it.
        beat_id: The part of the video it belongs to, which is what the
            timeline's markers are made from.
        speed: How fast it plays: the plan's pacing speed, set once the
            windows are settled.
    """

    asset_id: str
    start: float
    end: float
    from_clip_ids: Tuple[str, ...]
    beat_id: str = ""
    speed: float = 1.0

    @property
    def duration(self) -> float:
        """Length of the window in its file, in seconds.

        Returns:
            The seconds of source it covers.
        """
        return self.end - self.start

    @property
    def played(self) -> float:
        """Length of the window on the timeline, at its speed.

        Everything about where things land in the cut — markers, covers,
        music, beats, the cut's length — is counted in this, not in
        `duration`.

        Returns:
            The seconds it takes up on the timeline.
        """
        return self.duration / self.speed

def _breath(clip: SemanticClip, breath: float) -> Tuple[float, float]:
    """Work out how much air a clip can be given at each end.

    Args:
        clip: Clip to pad.
        breath: How much air the plan asks for.

    Returns:
        `(lead, trail)` in seconds, never more than the clip's headroom.
    """
    return min(breath, clip.safe_in), min(breath, clip.safe_out)

def _piece(
    clip: SemanticClip,
    asset: Asset,
    start: Optional[float] = None,
    end: Optional[float] = None,
    pad_head: bool = True,
    pad_tail: bool = True,
    cuts: Optional[CleanCuts] = None,
    beat_id: str = "",
    breath: float = BREATH_SECONDS,
) -> Piece:
    """Cut one window out of a clip, with air around it where there is room.

    Args:
        clip: Clip the window comes from.
        asset: The asset it plays from, for its length.
        start: Window start in the source; the clip's own start by default.
        end: Window end in the source; the clip's own end by default.
        pad_head: Whether to reach back into the headroom before the window.
        pad_tail: Whether to reach on into the headroom after it.
        cuts: Where this file may be cut without splitting a word. Given them,
            an edge that is not padded — which is to say one a `head` or
            `tail` put at an arbitrary second — moves to the nearest such
            place within `SNAP_SECONDS`. Padded edges are already sitting in
            measured silence and are left alone.

    Returns:
        The window, in source seconds and inside the asset.
    """
    lead, trail = _breath(clip, breath)
    opened = clip.source_range.start if start is None else start
    closed = clip.source_range.end if end is None else end
    opening = opened - (lead if pad_head else 0.0)
    closing = closed + (trail if pad_tail else 0.0)
    if cuts is not None:
        # A padded edge is meant to be sitting in silence, but the breath is measured
        # from the silence detector while words come from the transcriber, and the two
        # disagree: a breath can reach back into a word the detector called quiet. So
        # every edge is checked, and a padded one that lands in a word gives the breath
        # back rather than clipping a syllable.
        opening = _off_a_word(opening, opened if pad_head else None, clip, cuts)
        closing = _off_a_word(closing, closed if pad_tail else None, clip, cuts)
    # Round to milliseconds before clamping, never after: rounding 36.266667 up to 36.267
    # would put the window past the end of a file that is only 36.266667s long.
    closing = round(closing, 3)
    if asset.duration is not None:
        closing = min(closing, float(asset.duration))
    return Piece(asset.id, round(max(0.0, opening), 3), closing, (clip.id,), beat_id)

def _off_a_word(
    seconds: float,
    unpadded: Optional[float],
    clip: SemanticClip,
    cuts: CleanCuts,
) -> float:
    """Move a cut off the middle of a word, if there is somewhere near to put it.

    Words only. Bad frames are weighed later, by the stage that owns them, so
    that moving off one cannot undo the breath this gives back.

    Args:
        seconds: Where the cut would fall.
        unpadded: The same edge without its breath, for an edge that has one;
            None for an edge that never had one. Giving the breath back is
            tried first, because it is the smallest move available and it
            returns the edge to the sentence boundary it came from.
        clip: The clip being cut, whose own range bounds the answer.
        cuts: Where this file may be cut without splitting a word.

    Returns:
        A second that does not fall inside a word, as near as possible to the
        one asked for. The time as asked when nothing near enough qualifies —
        which happens where a word runs longer than `SNAP_SECONDS` allows the
        cut to travel, and which `check_plan` reports rather than papers over.
    """
    if not cuts.splits_a_word(seconds):
        return seconds
    if unpadded is not None and not cuts.splits_a_word(unpadded):
        return unpadded
    landed = cuts.nearest_word_edge(seconds, SNAP_SECONDS)
    if landed is None:
        return seconds
    # Clamped back into the clip, the edge can land in a word all over again, which
    # would undo the move without anybody noticing.
    inside = min(max(landed, clip.source_range.start), clip.source_range.end)
    return seconds if cuts.splits_a_word(inside) else inside

def _selection_pieces(
    selection: Selection,
    clip: SemanticClip,
    children: Sequence[SemanticClip],
    asset: Asset,
    cuts: Optional[CleanCuts],
    breath: float = BREATH_SECONDS,
) -> List[Piece]:
    """Work out what one selection contributes to the cut.

    Args:
        selection: The selection, carrying its trim.
        clip: The semantic clip it names.
        children: That clip's own clips, in time order, empty for an utterance.
        asset: The asset they play from.
        cuts: Where this file may be cut without splitting a word.
        breath: How much air the plan asks for around each cut.

    Returns:
        The windows this selection puts on the timeline, in order.
    """
    trim = selection.trim
    beat = selection.beat_id
    if trim.kind == TrimKind.KEEP:
        wanted = set(trim.keep_clip_ids)
        return [_piece(child, asset, beat_id=beat, breath=breath) for child in children if child.id in wanted]
    if trim.kind == TrimKind.TIGHTEN:
        # Drop what nobody would keep — the pauses and the unusable picture — and leave the rest.
        kept = [child for child in children if child.kind not in (ClipKind.SILENCE, ClipKind.UNUSABLE)]
        return (
            [_piece(child, asset, beat_id=beat, breath=breath) for child in kept]
            if kept else [_piece(clip, asset, beat_id=beat, breath=breath)]
        )
    if trim.kind == TrimKind.HEAD and trim.seconds is not None:
        return [_piece(
            clip, asset, end=min(clip.source_range.start + trim.seconds, clip.source_range.end),
            pad_tail=False, cuts=cuts, beat_id=beat, breath=breath,
        )]
    if trim.kind == TrimKind.TAIL and trim.seconds is not None:
        return [_piece(
            clip, asset, start=max(clip.source_range.end - trim.seconds, clip.source_range.start),
            pad_head=False, cuts=cuts, beat_id=beat, breath=breath,
        )]
    return [_piece(clip, asset, beat_id=beat, breath=breath)]

def _merge(pieces: Sequence[Piece]) -> List[Piece]:
    """Join pieces that follow one another closely enough to be one shot.

    Args:
        pieces: Windows in the order they are used.

    Returns:
        The windows, with neighbours from the same file that nearly touch
        joined together.
    """
    merged: List[Piece] = []
    for piece in pieces:
        last = merged[-1] if merged else None
        if last is not None and last.asset_id == piece.asset_id and piece.start - last.end <= MERGE_GAP_SECONDS:
            # The first piece's beat wins. Two pieces this close are one shot, and a
            # marker inside a shot would say a new part starts partway through it.
            merged[-1] = Piece(
                last.asset_id, last.start, max(last.end, piece.end),
                (*last.from_clip_ids, *piece.from_clip_ids), last.beat_id,
            )
        else:
            merged.append(piece)
    return merged

def _ordered_selections(plan: EditPlan) -> List[Selection]:
    """Put the selections in the order the finished video plays them.

    Beats run in the order the plan lists them, and within a beat the
    selections keep the order they were written in.

    Args:
        plan: Plan to order.

    Returns:
        The selections, in playing order. Selections whose beat is not in the
        plan come last, so a broken plan still compiles to something the
        checks can describe.
    """
    position = {beat.id: index for index, beat in enumerate(plan.beats)}
    return sorted(plan.selections, key=lambda selection: position.get(selection.beat_id, len(position)))

def _selected(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]],
) -> List[Piece]:
    """Work out what the selections alone put on the timeline, before any cleaning.

    One window per selection, or per kept child, and in that order — which is
    what lets the snapping be reported piece for piece. The cleaning stages
    after this one split and drop windows, so they count what they did
    themselves rather than being counted by comparison.

    Args:
        plan: Plan to lay out.
        clips: The timeline's clips, keyed by ID.
        children: Each section's own clips in time order, keyed by section ID.
        assets: The assets they play from, keyed by asset ID.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        The windows, in order, unmerged.
    """
    pieces: List[Piece] = []
    for selection in _ordered_selections(plan):
        clip = clips.get(selection.clip_id)
        asset = assets.get(clip.asset_id) if clip is not None else None
        if clip is None or asset is None:
            continue
        pieces.extend(_selection_pieces(
            selection, clip, children.get(clip.id, []), asset,
            None if cuts is None else cuts.get(clip.asset_id), pace_of(plan).breath,
        ))
    return pieces

def _bare(text: str) -> str:
    """Strip a line down to what was said, for comparing two goes at it.

    Args:
        text: Transcribed text.

    Returns:
        The letters and digits, punctuation and spacing gone. Written with
        `isalnum` rather than a list of marks because these transcripts are
        Chinese as often as not, and every CJK character answers to it.
    """
    return "".join(character for character in text if character.isalnum())

def _same_line(first: Piece, second: Piece, cuts: Mapping[str, CleanCuts]) -> bool:
    """Decide whether two neighbouring windows are two goes at the same line.

    Args:
        first: The earlier window.
        second: The one straight after it.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        True when both come from one file, both have somebody talking in them,
        and what is said is near enough identical. Two windows with nobody
        talking are never a retake: they would compare as identical for having
        nothing to compare, and silent B-roll is not a stammer.
    """
    if first.asset_id != second.asset_id:
        return False
    known = cuts.get(first.asset_id)
    if known is None:
        return False
    said = _bare(known.spoken_between(first.start, first.end))
    again = _bare(known.spoken_between(second.start, second.end))
    if not said or not again:
        return False
    return SequenceMatcher(None, said, again).ratio() >= RETAKE_SIMILARITY

def _without_retakes(
    pieces: Sequence[Piece],
    cuts: Optional[Mapping[str, CleanCuts]],
    pace: Pace = Pace(),
) -> Tuple[List[Piece], List[str]]:
    """Keep one go at each line, where the same one was said more than once.

    The last complete one, because a retake happens when somebody stumbles and
    starts again, so the finished attempt is usually the last. Complete means
    the window does not stop partway through a sentence; where none of them is
    complete the last is kept anyway, since something has to be.

    This stage drops footage the plan chose, so every window it takes out is
    named. A cleanup nobody can see is a cleanup nobody can argue with.

    Args:
        pieces: The windows, in order.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        `(pieces, notes)`.
    """
    if not cuts:
        return list(pieces), []
    kept: List[Piece] = []
    dropped: List[Piece] = []

    def settle(run: List[Piece]) -> None:
        """Choose which window of one run of retakes survives.

        Args:
            run: The windows saying the same line, in order.
        """
        if len(run) == 1:
            kept.append(run[0])
            return
        known = cuts[run[0].asset_id]
        finished = [piece for piece in run if not known.splits_a_sentence(piece.end)]
        survives = finished[-1] if finished else run[-1]
        kept.append(survives)
        dropped.extend(piece for piece in run if piece is not survives)

    run: List[Piece] = []
    for piece in pieces:
        if run and _same_line(run[-1], piece, cuts):
            run.append(piece)
            continue
        if run:
            settle(run)
        run = [piece]
    if run:
        settle(run)

    notes: List[str] = []
    if dropped:
        seconds = sum(piece.duration for piece in dropped)
        notes.append(
            f"{len(dropped)} window(s) say a line that is said again straight after, and were dropped, "
            f"{seconds:.1f}s in all: {', '.join(piece.from_clip_ids[0] for piece in dropped)}. "
            "The last complete go at each was kept"
        )
    return kept, notes

def _somewhere_clean(seconds: float, known: CleanCuts, opens: bool) -> float:
    """Move a time onto somewhere a cut may land, if anywhere near enough is.

    Args:
        seconds: The time as worked out.
        known: What is known about the file.
        opens: Whether the point opens a window rather than closing it.

    Returns:
        The nearest second that splits no word and shows no bad frame, or the
        time as asked when nothing within `SNAP_SECONDS` qualifies. Callers
        check the answer rather than assume it moved: leaving it is the honest
        result and has to stay visible.
    """
    if known.is_clean(seconds, opens):
        return seconds
    landed = known.nearest_clean_point(seconds, SNAP_SECONDS, opens)
    return seconds if landed is None else round(landed, 3)

def _without_pauses(
    pieces: Sequence[Piece],
    cuts: Optional[Mapping[str, CleanCuts]],
    pace: Pace = Pace(),
) -> Tuple[List[Piece], List[str]]:
    """Take the dead air out of the middle of a window.

    A window covering a whole section covers the gaps inside it too, and a gap
    this long is not a breath, it is the edit waiting. The window is split
    around it, keeping a breath on each side so neither half sounds clipped.
    Both halves keep the clip IDs the window had, because they do still come
    from the same footage.

    Args:
        pieces: The windows, in order.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        `(pieces, notes)`.
    """
    if not cuts:
        return list(pieces), []
    out: List[Piece] = []
    taken: List[float] = []
    talked_through = 0
    for piece in pieces:
        known = cuts.get(piece.asset_id)
        dead = known.pauses_inside(piece.start, piece.end, pace.pause) if known is not None else ()
        opened = piece.start
        for quiet, loud in dead:
            # The silence came from the detector and the words came from the
            # transcriber, and on real footage the two disagree all the time: a word's
            # reported end runs on into a stretch measured as quiet. So neither edge of
            # the gap is trusted where it falls — each is settled onto somewhere a cut
            # may actually land, which for an edge inside a word is that word's own
            # boundary.
            closing = _somewhere_clean(round(quiet + pace.breath, 3), known, opens=False)
            opening = _somewhere_clean(round(loud - pace.breath, 3), known, opens=True)
            if closing <= opened or opening <= closing or opening >= piece.end:
                # One side would be left with no length at all, so the gap stays
                # rather than a window of nothing being put on the timeline.
                continue
            if opening - closing <= MERGE_GAP_SECONDS:
                # `_merge` runs after every stage and joins two halves this close back
                # together. Splitting here would leave the cut exactly as it was while
                # the note claimed seconds had been taken out of it, and a report that
                # is wrong is worse than a pause that stayed. The constant is chosen so
                # an unsettled gap always clears this; settling each edge inward by as
                # much as a second is what can still close it.
                continue
            if not known.is_clean(closing, opens=False) or not known.is_clean(opening, opens=True):
                # Nowhere near enough to settle onto. Cutting here would go through a
                # word, which is worse than the pause it saves, so it stays and the
                # disagreement is reported rather than papered over.
                talked_through += 1
                continue
            out.append(Piece(piece.asset_id, opened, closing, piece.from_clip_ids, piece.beat_id))
            taken.append(round(opening - closing, 3))
            opened = opening
        out.append(Piece(piece.asset_id, opened, piece.end, piece.from_clip_ids, piece.beat_id))

    notes: List[str] = []
    if taken:
        notes.append(
            f"{len(taken)} pause(s) longer than {pace.pause:g}s were taken out of the middle of a shot, "
            f"{sum(taken):.1f}s in all"
        )
    if talked_through:
        notes.append(
            f"{talked_through} pause(s) were left in: they were measured as quiet, but the transcript "
            "has somebody talking across them, and cutting there would go through a word"
        )
    return out, notes

def _trimmed_ends(
    pieces: Sequence[Piece],
    cuts: Optional[Mapping[str, CleanCuts]],
    pace: Pace = Pace(),
) -> Tuple[List[Piece], List[str]]:
    """Take the run-up off the first window and the wind-down off the last.

    Nobody wants to open on somebody gathering themselves to speak, or close
    on them reaching for the camera. Both ends are pulled in to a breath
    either side of the talking — but only where there is talking to measure
    against, and never by more than the trims allow, so a shot held silent on
    purpose survives.

    Args:
        pieces: The windows, in order.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        `(pieces, notes)`.
    """
    if not cuts or not pieces:
        return list(pieces), []
    out = list(pieces)
    notes: List[str] = []

    def say(where: str, taken: float, left: float, limit: float) -> None:
        """Say what was trimmed, and what the limit would not let go of.

        Args:
            where: Which end of the cut, for the sentence.
            taken: Seconds removed.
            left: Seconds of the run-up or wind-down still there.
            limit: The trim that stopped it going further.
        """
        said = f"{taken:.1f}s {where} was trimmed off"
        if left > 0:
            said += (
                f", and {left:.1f}s of it is still there: the trim goes no further than "
                f"{limit:g}s in case the shot is being held on purpose"
            )
        notes.append(said)

    head = out[0]
    known = cuts.get(head.asset_id)
    first_word = known.first_word_in(head.start, head.end) if known is not None else None
    if first_word is not None:
        wanted = round(first_word - pace.breath, 3)
        opening = round(min(wanted, head.start + HEAD_TRIM_SECONDS), 3)
        if opening > head.start:
            say("before the first word", opening - head.start, round(wanted - opening, 3), HEAD_TRIM_SECONDS)
            out[0] = Piece(head.asset_id, opening, head.end, head.from_clip_ids, head.beat_id)

    tail = out[-1]
    known = cuts.get(tail.asset_id)
    last_word = known.last_word_in(tail.start, tail.end) if known is not None else None
    if last_word is not None:
        wanted = round(last_word + pace.breath, 3)
        closing = round(max(wanted, tail.end - TAIL_TRIM_SECONDS), 3)
        if closing < tail.end:
            say("after the last word", tail.end - closing, round(closing - wanted, 3), TAIL_TRIM_SECONDS)
            out[-1] = Piece(tail.asset_id, tail.start, closing, tail.from_clip_ids, tail.beat_id)

    return out, notes

def _off_bad_frames(
    pieces: Sequence[Piece],
    cuts: Optional[Mapping[str, CleanCuts]],
    pace: Pace = Pace(),
) -> Tuple[List[Piece], List[str]]:
    """Move any cut point that shows black or frozen picture.

    Run last, so it has the final say over every edge the stages before it
    made. It travels no further than the word snapping does, and every
    candidate it weighs has to clear both objections, so getting off a bad
    frame can never land in the middle of a word.

    Args:
        pieces: The windows, in order.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        `(pieces, notes)`.
    """
    if not cuts:
        return list(pieces), []
    out: List[Piece] = []
    moved, stuck = 0, 0
    for piece in pieces:
        known = cuts.get(piece.asset_id)
        if known is None:
            out.append(piece)
            continue
        settled: List[float] = []
        objected = 0
        for seconds, opens in ((piece.start, True), (piece.end, False)):
            if not known.shows_bad_picture(seconds, opens):
                settled.append(seconds)
                continue
            objected += 1
            settled.append(_somewhere_clean(seconds, known, opens))
        opening, closing = settled
        if closing <= opening:
            # Moving both ends has closed the window. A bad frame is a smaller loss
            # than a shot, so it goes back as it was and is counted as stuck.
            stuck += objected
            out.append(piece)
            continue
        shifted = sum(1 for now, before in zip(settled, (piece.start, piece.end)) if now != before)
        moved += shifted
        stuck += objected - shifted
        out.append(Piece(piece.asset_id, opening, closing, piece.from_clip_ids, piece.beat_id))

    notes: List[str] = []
    if moved:
        notes.append(f"{moved} cut(s) moved off black or frozen picture")
    if stuck:
        notes.append(
            f"{stuck} cut(s) still land on black or frozen picture: no clean frame is within "
            f"{SNAP_SECONDS:g}s of them, so they were left as asked"
        )
    return out, notes

def _beginnings(pieces: Sequence[Piece]) -> Dict[str, int]:
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

def _cue_entries(plan: EditPlan, pieces: Sequence[Piece]) -> Dict[int, MusicCue]:
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
    first = _beginnings(pieces)
    entries: Dict[int, MusicCue] = {}
    for cue in plan.music.cues:
        at = 0 if cue.beat_id is None else first.get(cue.beat_id)
        if at is not None:
            entries[at] = cue
    return entries

def _song_length(source: Optional[Asset]) -> Optional[float]:
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

def _beats_near(
    beats: Sequence[float],
    entry: float,
    length: Optional[float],
    entered_at: float,
    seconds: float,
) -> List[float]:
    """Find the beats of a playing song either side of a moment of the timeline.

    The song plays from `entry` and loops back there when it runs out, the way
    `music_beds` lays it, so the beats repeat with the length of one pass.

    Args:
        beats: The song's beats, in song seconds.
        entry: Where in the song it came in.
        length: How long the song is, or None when that is not known and it is
            taken never to loop.
        entered_at: Where on the timeline it came in.
        seconds: The moment of the timeline to look near.

    Returns:
        The beat just before and the beat just after, on the timeline, nearer
        one first — both, because the nearer one may be out of reach where the
        other is not. Empty when the song has no beat after where it came in.
    """
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

def _on_the_beat(
    plan: EditPlan,
    pieces: Sequence[Piece],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]],
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

    Returns:
        `(pieces, notes)`.
    """
    entries = _cue_entries(plan, pieces)
    if not cuts or not any(cue.cut_on_beat for cue in entries.values()):
        return list(pieces), []
    out = list(pieces)
    moved: List[float] = []
    stuck, unknown = 0, 0
    playing: Optional[MusicCue] = None
    song: Optional[CleanCuts] = None
    length: Optional[float] = None
    entry, entered_at, position = 0.0, 0.0, 0.0
    for index in range(len(out)):
        if index in entries:
            playing = entries[index]
            song = cuts.get(playing.asset_id) if playing.asset_id else None
            length = _song_length(assets.get(playing.asset_id)) if playing.asset_id else None
            entry, entered_at = song_entry(playing, song), position
        piece = out[index]
        on_beat = playing is not None and playing.cut_on_beat and song is not None and song.beats
        if index + 1 < len(out) and on_beat:
            closing_at = position + piece.played
            shifts = [
                round(beat - closing_at, 3)
                for beat in _beats_near(song.beats, entry, length, entered_at, closing_at)
            ]
            known = cuts.get(piece.asset_id)
            # A shift is timeline seconds; the window's end moves by that much of its file.
            if not shifts or abs(shifts[0]) <= ON_BEAT_SECONDS:
                pass
            elif known is None or not known.transcribed:
                unknown += 1
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
                else:
                    stuck += 1
        position += out[index].played

    notes: List[str] = []
    if moved:
        notes.append(f"{len(moved)} cut(s) moved onto the beat of the music, {max(moved):.2f}s at most")
    if stuck:
        notes.append(
            f"{stuck} cut(s) are off the beat: reaching it would cut into a word, show a bad frame, or move "
            f"further than {BEAT_SNAP_SECONDS:g}s, so they were left where they were"
        )
    if unknown:
        notes.append(
            f"{unknown} cut(s) are off the beat because their footage was never transcribed, so there is no "
            "knowing whether moving them would cut into a word"
        )
    return out, notes

def compile_pieces(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]] = None,
) -> Tuple[List[Piece], List[str]]:
    """Work out every window the plan puts on the timeline, cleaned up, in order.

    The cleaning runs in this order for a reason. Retakes go first: there is
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
    pieces = _selected(plan, clips, children, assets, cuts)
    notes: List[str] = []
    pace = pace_of(plan)
    for stage in (_without_retakes, _without_pauses, _trimmed_ends, _off_bad_frames):
        pieces, said = stage(pieces, cuts, pace)
        notes.extend(said)
    merged = _merge(pieces)
    if pace.speed != 1.0:
        # Set once the windows are settled: every stage before this works in the file's own
        # seconds, and only where things land on the timeline cares how fast they play.
        merged = [replace(piece, speed=pace.speed) for piece in merged]
    pieces, said = _on_the_beat(plan, merged, assets, cuts)
    return pieces, notes + said

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

@dataclass(frozen=True)
class Cover:
    """One shot of B-roll worked out onto the compiled timeline.

    Attributes:
        clip_id: Semantic clip the covering picture comes from.
        asset_id: File that picture plays from.
        start: Where it starts in that file, in seconds.
        end: Where it ends, in seconds.
        timeline_in: Where it lands on the timeline, in seconds.
        over_clip_id: The clip in the cut it starts over.
    """

    clip_id: str
    asset_id: str
    start: float
    end: float
    timeline_in: float
    over_clip_id: str

    @property
    def duration(self) -> float:
        """Length of the shot in seconds.

        Returns:
            The seconds it covers.
        """
        return self.end - self.start

    @property
    def timeline_out(self) -> float:
        """Where the shot stops covering, in timeline seconds.

        Returns:
            The second the picture underneath comes back.
        """
        return round(self.timeline_in + self.duration, 3)

def piece_gain(
    piece: Piece,
    clips: Mapping[str, SemanticClip],
    gains: Optional[Mapping[str, float]],
) -> float:
    """Work out what to turn one window by so its voice matches the others.

    Only a window that is one person's. A window merged across a handover
    belongs to neither of them, so turning it by either one's gain would be
    picking a side — the same reason `speaker_at` leaves such a stretch
    unlabelled rather than guessing.

    Args:
        piece: The compiled window.
        clips: The timeline's clips, keyed by ID.
        gains: A multiplier per joined voice, or None to leave every window
            at the level it was recorded.

    Returns:
        The multiplier, 1.0 when the window is nobody's in particular.
    """
    if not gains:
        return 1.0
    said_by = {
        clips[clip_id].speaker for clip_id in piece.from_clip_ids
        if clip_id in clips and clips[clip_id].speaker
    }
    return gains.get(next(iter(said_by)), 1.0) if len(said_by) == 1 else 1.0

def _placed(pieces: Sequence[Piece]) -> List[Tuple[Piece, float]]:
    """Say where each window lands on the timeline.

    Args:
        pieces: The compiled windows, in order.

    Returns:
        Each window with the second it starts at on the timeline.
    """
    placed, at = [], 0.0
    for piece in pieces:
        placed.append((piece, round(at, 3)))
        at += piece.played
    return placed

def _in_cut(clip: SemanticClip, placed: Sequence[Tuple[Piece, float]]) -> Optional[Tuple[float, float]]:
    """Find where a semantic clip ended up on the compiled timeline.

    The first window that shows any of it, because that is where it cuts in.
    A clip split across two windows by the pause stage is therefore reported
    from its first half, which is where it starts on screen.

    Args:
        clip: The semantic clip to locate.
        placed: The windows with their timeline positions.

    Returns:
        `(start, end)` in timeline seconds, or None when the cut does not use
        this clip at all.
    """
    for piece, at in placed:
        if piece.asset_id != clip.asset_id:
            continue
        opened = max(piece.start, clip.source_range.start)
        closed = min(piece.end, clip.source_range.end)
        if closed > opened:
            return (round(at + (opened - piece.start) / piece.speed, 3),
                    round(at + (closed - piece.start) / piece.speed, 3))
    return None

def compiled_duration(pieces: Sequence[Piece]) -> float:
    """Measure how long the compiled cut runs.

    Args:
        pieces: The windows on the timeline.

    Returns:
        The total in seconds.
    """
    return round(sum(piece.played for piece in pieces), 3)

def _under(placed: Sequence[Tuple[Piece, float]], seconds: float) -> Optional[Tuple[Piece, float]]:
    """Find the window playing at one second of the timeline.

    Args:
        placed: The windows with their timeline positions.
        seconds: The second to look at.

    Returns:
        The window and where it starts, or None past the end of the cut.
    """
    for piece, at in placed:
        if at <= seconds < round(at + piece.played, 3):
            return piece, at
    return None

def _splits_a_sentence(
    under: Tuple[Piece, float],
    seconds: float,
    cuts: Optional[Mapping[str, CleanCuts]],
) -> bool:
    """Say whether a moment of the timeline falls partway through a spoken sentence.

    Args:
        under: The window playing there, with the second it starts at.
        seconds: The moment on the timeline.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        True when somebody is midway through a sentence at that moment. False
        when nothing is known about the file, which is not the same answer as
        nobody talking, but is the only honest one available.
    """
    piece, at = under
    known = cuts.get(piece.asset_id) if cuts else None
    return known is not None and known.splits_a_sentence(piece.start + (seconds - at) * piece.speed)

def broll_covers(
    plan: EditPlan,
    pieces: Sequence[Piece],
    clips: Mapping[str, SemanticClip],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]] = None,
) -> Tuple[List[Cover], List[str], List[str]]:
    """Work out where each shot of B-roll lands, and what is wrong with the ones that do not.

    The coverage rules live here rather than in a guide, because none of them
    needs judgement once the shot has been chosen: how long a shot may run, how
    much of a beat may be under one, and which shots may not be covered at all
    are arithmetic against the plan. What needs judgement — whether this
    picture belongs over these words — is the plan's, and nothing here second
    guesses it.

    Args:
        plan: Plan whose B-roll is being laid out.
        pieces: The compiled windows of the main sequence, in order.
        clips: The timeline's clips, keyed by ID.
        assets: The assets they play from, keyed by asset ID.
        cuts: What is known about each file, keyed by asset ID, for saying when
            a shot cuts back partway through a sentence.

    Returns:
        `(covers, problems, notes)`. A problem stops the plan compiling; a
        note is worth saying to whoever wrote it. Only shots with no problem
        of their own come back as covers.
    """
    placed = _placed(pieces)
    total = compiled_duration(pieces)
    problems: List[str] = []
    notes: List[str] = []

    held: List[Tuple[str, float, float]] = []
    for selection in plan.selections:
        clip = clips.get(selection.clip_id)
        span = None if clip is None else _in_cut(clip, placed)
        if selection.hold_picture and span is not None:
            held.append((selection.clip_id, *span))

    in_beat: Dict[str, float] = {}
    for piece in pieces:
        in_beat[piece.beat_id] = in_beat.get(piece.beat_id, 0.0) + piece.played

    covers: List[Cover] = []
    covered_in_beat: Dict[str, float] = {}
    for position, shot in enumerate(plan.broll, start=1):
        where = f"B-roll {position} (over {shot.over_clip_id})"
        source = clips.get(shot.clip_id)
        over = clips.get(shot.over_clip_id)
        if source is None:
            problems.append(f"{where}: {shot.clip_id} is not a clip in this timeline")
            continue
        if over is None:
            problems.append(f"{where}: {shot.over_clip_id} is not a clip in this timeline")
            continue
        asset = assets.get(source.asset_id)
        if asset is None:
            problems.append(f"{where}: asset {source.asset_id} is no longer registered")
            continue
        if not asset.has_video:
            problems.append(f"{where}: {asset.id} has no picture, so there is nothing to lay over anything")
            continue
        span = _in_cut(over, placed)
        if span is None:
            problems.append(
                f"{where}: the cut does not use {shot.over_clip_id}, so there is nowhere for this to start"
            )
            continue
        if not BROLL_MIN_SECONDS <= shot.seconds <= BROLL_MAX_SECONDS:
            problems.append(
                f"{where}: {shot.seconds:g}s is outside the {BROLL_MIN_SECONDS:g}–{BROLL_MAX_SECONDS:g}s a shot "
                "may run; under that it reads as a flicker, over it the viewer loses what is being said"
            )
            continue
        if shot.seconds > source.duration:
            problems.append(
                f"{where}: {shot.clip_id} runs {source.duration:.1f}s, which is not the {shot.seconds:g}s asked for"
            )
            continue

        opened = span[0]
        closed = round(opened + shot.seconds, 3)
        if closed > total:
            problems.append(
                f"{where}: would run to {closed:.1f}s of a cut that is {total:.1f}s long"
            )
            continue
        clashes = [
            clip_id for clip_id, starts, ends in held
            if starts < closed and opened < ends
        ]
        if clashes:
            problems.append(
                f"{where}: covers {', '.join(clashes)}, which the plan holds for what it shows"
            )
            continue
        over_another = [
            other for other in covers if other.timeline_in < closed and opened < other.timeline_out
        ]
        if over_another:
            problems.append(
                f"{where}: overlaps the shot over {over_another[0].over_clip_id}; "
                "two pictures cannot be on screen at once"
            )
            continue

        beat = _under(placed, opened)
        if beat is not None and _splits_a_sentence(beat, opened, cuts):
            # Cutting away mid-sentence is the half of the boundary rule that can be
            # fixed for nothing: the shot moves to another clip and the cut keeps its
            # length. The other half — cutting back — cannot, so that one is reported.
            problems.append(
                f"{where}: cuts in partway through a sentence. Start it over a clip that opens one, "
                "or the viewer loses the line as the picture changes"
            )
            continue
        beat_id = beat[0].beat_id if beat is not None else ""
        running = covered_in_beat.get(beat_id, 0.0) + shot.seconds
        allowed = in_beat.get(beat_id, 0.0) * BROLL_MAX_SHARE
        if running > allowed:
            problems.append(
                f"{where}: would put {running:.1f}s of beat {beat_id or '(none)'} under B-roll, more than the "
                f"{BROLL_MAX_SHARE:.0%} of its {in_beat.get(beat_id, 0.0):.1f}s that may be covered"
            )
            continue
        covered_in_beat[beat_id] = running

        covers.append(Cover(
            clip_id=shot.clip_id,
            asset_id=source.asset_id,
            start=round(source.source_range.start, 3),
            end=round(source.source_range.start + shot.seconds, 3),
            timeline_in=opened,
            over_clip_id=shot.over_clip_id,
        ))
        if not over.text:
            notes.append(
                f"{where}: nobody is talking under it, so this swaps one picture for another rather than "
                "showing what is being said"
            )

    if cuts:
        interrupted = []
        for cover in covers:
            under = _under(placed, cover.timeline_out)
            if under is not None and _splits_a_sentence(under, cover.timeline_out, cuts):
                interrupted.append(cover.over_clip_id)
        if interrupted:
            notes.append(
                f"{len(interrupted)} B-roll shot(s) cut back partway through a sentence: "
                f"{', '.join(interrupted)}. Lengthening them would change how long the video runs, "
                "so it is left to whoever chose the length"
            )
    return covers, problems, notes

def broll_slots(
    plan: EditPlan,
    pieces: Sequence[Piece],
    clips: Mapping[str, SemanticClip],
    cuts: Optional[Mapping[str, CleanCuts]] = None,
) -> List[dict]:
    """Offer the places in a cut where covering picture would help.

    Two signals, both of them lengths rather than judgements about how a shot
    looks. One shot held a long time is the clearest case: the sound carries
    on and the picture has stopped saying anything new. A run of shots from
    one file is the same problem wearing a different hat — the camera never
    moved, so cutting between them changed nothing.

    Deliberately not offered: "the sound is good but the picture is weak".
    Weak needs a threshold on the exposure and blur measurements, and the
    roadmap refuses to pick those against synthetic footage. The measurements
    are on every clip's `scores` for whoever is choosing to read.

    Args:
        plan: Plan to read, for which shots are held and which beats exist.
        pieces: The compiled windows, in order.
        clips: The timeline's clips, keyed by ID.
        cuts: What is known about each file, keyed by asset ID. Given them, a
            place where a shot would cut in partway through a sentence is not
            offered — the compiler refuses those, and offering a position it
            will not accept is worse than offering nothing.

    Returns:
        One slot per stretch worth covering, worst first, each saying where a
        shot would start, how long it may run there, how long the stretch
        holds, and which signal offered it. A cut made of long static shots
        will have most of itself offered back — that is the answer, not noise,
        which is why the order is by how long each stretch holds rather than
        by where it falls.
    """
    placed = _placed(pieces)
    holding = {
        selection.clip_id for selection in plan.selections if selection.hold_picture
    }
    # A run is a stretch of unchanged picture: one file, and the source moving forward
    # through it. Going backwards in the same file is somebody putting a later moment
    # first, which is a real cut whatever the framing was.
    runs: List[List[Tuple[Piece, float]]] = []
    for piece, at in placed:
        last = runs[-1][-1][0] if runs else None
        if last is not None and last.asset_id == piece.asset_id and piece.start >= last.end:
            runs[-1].append((piece, at))
        else:
            runs.append([(piece, at)])

    slots: List[dict] = []
    for run in runs:
        opened = run[0][1]
        closed = round(run[-1][1] + run[-1][0].played, 3)
        seconds = round(closed - opened, 3)
        if seconds < BROLL_LONG_SHOT_SECONDS:
            continue
        why = (
            f"one shot holds for {seconds:.1f}s"
            if len(run) == 1 else
            f"{len(run)} windows in a row hold the same picture, {seconds:.1f}s of it"
        )
        # A slot at the start, then one at every clip boundary far enough past the last
        # offer that two shots taken from here could not overlap. Over-generating is the
        # point: a shot held half a minute wants covering in more than one place.
        offered = -BROLL_MAX_SECONDS
        for piece, _ in run:
            for clip_id in piece.from_clip_ids:
                clip = clips.get(clip_id)
                span = None if clip is None else _in_cut(clip, placed)
                if span is None or clip_id in holding:
                    continue
                if span[0] - offered < BROLL_MAX_SECONDS:
                    continue
                under = _under(placed, span[0])
                if under is not None and _splits_a_sentence(under, span[0], cuts):
                    continue
                room = min(BROLL_MAX_SECONDS, round(closed - span[0], 3))
                if room < BROLL_MIN_SECONDS:
                    continue
                offered = span[0]
                slots.append({
                    "over_clip_id": clip_id,
                    "seconds": room,
                    "held_seconds": seconds,
                    "why": why,
                })
    # Stable, so the slots inside one stretch keep the order they play in.
    return sorted(slots, key=lambda slot: -slot["held_seconds"])

def _cut_notes(
    pieces: Sequence[Piece],
    selected: Sequence[Piece],
    unsnapped: Sequence[Piece],
    cuts: Optional[Mapping[str, CleanCuts]],
    duration: float,
) -> List[str]:
    """Say what the compiler did to the cuts, and what it left for somebody to decide.

    Three things, each said once however many cuts it covers. Fifteen near
    identical lines is not a report, it is something to scroll past.

    Moving an edge off the middle of a word is arithmetic and is done without
    asking, but it is still a change to somebody's edit, so it is said.
    Letting a sentence finish is not arithmetic: measured on real footage it
    runs to a third of the length of the video, and how long the video runs is
    not the compiler's to decide. Leaving a cut inside a word because no clean
    second is near enough is a failure to say plainly rather than to hide.

    Args:
        pieces: The finished windows, cleaning and all, for the faults that
            survived to the end.
        selected: The windows the selections alone produced, for counting what
            the snapping moved. One per selection, so they pair off exactly
            with `unsnapped`, which the cleaned windows would not.
        unsnapped: The same windows as they would be without the cut points.
        cuts: Where each file may be cut without splitting a word, keyed by
            asset ID.
        duration: How long the compiled cut runs, for costing what is owed.

    Returns:
        The notes, empty when nothing was transcribed and so nothing is known.
    """
    if not cuts:
        return []
    said: List[str] = []

    moved = [
        abs(new_edge - old_edge)
        for new, old in zip(selected, unsnapped)
        for new_edge, old_edge in ((new.start, old.start), (new.end, old.end))
        if new_edge != old_edge
    ]
    if moved:
        said.append(
            f"{len(moved)} cut(s) moved off the middle of a word, by up to {max(moved):.2f}s"
        )

    stuck = [
        piece for piece in pieces
        if (known := cuts.get(piece.asset_id))
        and any(known.splits_a_word(edge) for edge in (piece.start, piece.end))
    ]
    if stuck:
        said.append(
            f"{len(stuck)} cut(s) still land inside a word: {', '.join(p.from_clip_ids[0] for p in stuck)}. "
            f"No clean point is within {SNAP_SECONDS:g}s of them, so they were left as asked"
        )

    owed = [
        (known.rest_of_sentence(piece.end), piece)
        for piece in pieces
        if (known := cuts.get(piece.asset_id)) and known.splits_a_sentence(piece.end)
    ]
    if owed:
        total = sum(seconds for seconds, _ in owed)
        worst, piece = max(owed, key=lambda item: item[0])
        share = f", {total / duration:.0%} of the cut" if duration else ""
        said.append(
            f"{len(owed)} window(s) stop before the sentence ends. Finishing them all would add "
            f"{total:.1f}s{share}; the longest is {piece.from_clip_ids[0]}, {worst:.1f}s short. "
            "Give those seconds to the ones that sound abrupt, or leave them"
        )
    return said

def _cue_problems(plan: EditPlan, pieces: Sequence[Piece]) -> List[str]:
    """Check that every music cue has somewhere to come in, in the order given.

    Only answerable once the cut is laid out: a beat can be in the plan and
    still never begin, when its first shot was merged into the one before it.

    Args:
        plan: The plan.
        pieces: Its windows, in order, as `compile_pieces` settled them.

    Returns:
        One message per problem.
    """
    if plan.music is None:
        return []
    first = _beginnings(pieces)
    problems: List[str] = []
    crossfade = plan.music.crossfade_seconds
    if crossfade:
        starts = _starts(pieces)
        total = compiled_duration(pieces)
        entered = sorted(
            (0 if cue.beat_id is None else first[cue.beat_id], cue) for cue in plan.music.cues
            if cue.beat_id is None or cue.beat_id in first
        )
        for (index, cue), (following, incoming) in zip(entered, entered[1:]):
            part = starts[following] - starts[index]
            if cue.asset_id and incoming.asset_id and crossfade > part:
                problems.append(
                    f"music: a {crossfade:g}s cross-fade into the cue on {incoming.beat_id} is longer than the "
                    f"{part:.1f}s part before it; shorten `crossfade_seconds`"
                )
    previous = -1
    for position, cue in enumerate(plan.music.cues, start=1):
        if cue.beat_id is None:
            at = 0
        elif cue.beat_id in first:
            at = first[cue.beat_id]
        else:
            problems.append(
                f"music cue {position}: beat {cue.beat_id} never begins in the cut — it has no footage, or its "
                "first shot runs on from the one before it — so there is no cut for the music to come in on"
            )
            continue
        if at <= previous:
            problems.append(
                f"music cue {position}: comes in on beat {cue.beat_id}, which begins before the cue listed "
                "ahead of it; list the cues in the order their beats come in"
            )
        previous = max(previous, at)
    return problems

def check_plan(
    plan: EditPlan,
    timeline: SemanticTimeline,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]] = None,
    levels: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], List[str]]:
    """Check a plan against the footage it claims to be made of.

    Args:
        plan: Plan to check.
        timeline: The semantic timeline it names.
        clips: That timeline's clips, keyed by ID.
        children: Each section's own clips, keyed by section ID.
        assets: The assets they play from, keyed by asset ID.
        cuts: Where each file may be cut without splitting a word, keyed by
            asset ID.
        levels: How loudly each joined voice speaks. Given them, how far apart
            the voices are is reported, so whoever is planning can decide
            whether to ask for them matched.

    Returns:
        `(problems, notes)`. A problem stops the plan compiling; a note is
        something worth saying to whoever wrote it, such as the cut coming out
        longer than the length they asked for.
    """
    problems: List[str] = []
    notes: List[str] = []

    if plan.timeline_id != timeline.id:
        problems.append(f"the plan is written against timeline {plan.timeline_id}, not {timeline.id}")
    if plan.timeline_input_hash != timeline.input_hash:
        problems.append(
            "the footage has been analyzed again since this plan was written, so its clip IDs no longer "
            "mean the same moments; build the timeline again and rewrite the plan against it"
        )
    if not plan.selections:
        problems.append("the plan chooses no footage")

    beats = {beat.id for beat in plan.beats}
    if len(beats) != len(plan.beats):
        problems.append("two beats share an id")

    for position, selection in enumerate(plan.selections, start=1):
        where = f"selection {position} ({selection.clip_id})"
        clip = clips.get(selection.clip_id)
        if clip is None:
            problems.append(f"{where}: no such clip in this timeline")
            continue
        if selection.beat_id not in beats:
            problems.append(f"{where}: belongs to beat {selection.beat_id}, which the plan does not have")
        if clip.kind == ClipKind.UNUSABLE:
            problems.append(f"{where}: this clip is unusable — nobody is talking and the picture is black or frozen")
        asset = assets.get(clip.asset_id)
        if asset is None:
            problems.append(f"{where}: asset {clip.asset_id} is no longer registered")
        elif not asset.has_video:
            problems.append(f"{where}: {asset.id} has no picture, so it cannot carry the sequence; put it under `music`")
        elif asset.duration is not None and clip.source_range.start >= float(asset.duration):
            # The timeline is derived from the analysis, so this only happens when the
            # analysis is describing a different file from the one on disk. Said here
            # because the alternative is an operation with an end before its start.
            problems.append(
                f"{where}: this clip starts at {clip.source_range.start:.1f}s of {asset.id}, which runs "
                f"{float(asset.duration):.1f}s. The file and its analysis disagree; analyze it again"
            )

        trim = selection.trim
        if trim.kind == TrimKind.KEEP:
            inside = {child.id for child in children.get(clip.id, [])}
            if not inside:
                problems.append(f"{where}: `keep` needs a section; this clip has nothing inside it to keep")
            stray = sorted(set(trim.keep_clip_ids) - inside)
            if stray:
                problems.append(f"{where}: {', '.join(stray)} is not inside this clip")
            if not trim.keep_clip_ids:
                problems.append(f"{where}: `keep` was asked for without saying what to keep")
        if trim.kind in (TrimKind.HEAD, TrimKind.TAIL):
            if trim.seconds is None:
                problems.append(f"{where}: `{trim.kind.value}` needs how many seconds to take")
            elif trim.seconds > clip.duration:
                problems.append(
                    f"{where}: asks for {trim.seconds:g}s of a clip that runs {clip.duration:.3f}s"
                )

    for rejection in plan.rejected:
        if rejection.clip_id not in clips:
            notes.append(f"the rejected clip {rejection.clip_id} is not in this timeline")

    pace = pace_of(plan)
    if pace.pause <= 2 * pace.breath + MERGE_GAP_SECONDS:
        problems.append(
            f"pacing: pauses over {pace.pause:g}s cannot be taken out while {pace.breath:g}s of air is left either "
            f"side of each cut — what would be left between the two halves is under {MERGE_GAP_SECONDS:g}s, "
            "so they would be joined straight back together. Raise `pause_seconds` above "
            f"{2 * pace.breath + MERGE_GAP_SECONDS:g}s or lower `breath_seconds`"
        )

    cues = plan.music.cues if plan.music is not None else []
    placed_cues: set = set()
    for position, cue in enumerate(cues, start=1):
        where = f"music cue {position}"
        if cue.beat_id is None and position > 1:
            problems.append(f"{where}: only the first cue may leave out `beat_id`; say which beat this one comes in on")
        elif cue.beat_id is not None and cue.beat_id not in beats:
            problems.append(f"{where}: comes in on beat {cue.beat_id}, which the plan does not have")
        if cue.beat_id is not None and cue.beat_id in placed_cues:
            problems.append(f"{where}: another cue already comes in on beat {cue.beat_id}")
        placed_cues.add(cue.beat_id)
        if cue.asset_id is not None:
            song = assets.get(cue.asset_id)
            if song is None:
                problems.append(f"{where}: the music asset {cue.asset_id} is not registered")
            elif not song.has_audio:
                problems.append(f"{where}: the music asset {cue.asset_id} has no sound")
            elif song.duration is not None and cue.start >= float(song.duration):
                problems.append(
                    f"{where}: comes in at {cue.start:g}s of {cue.asset_id}, which runs {float(song.duration):.1f}s"
                )
        if cue.cut_on_beat:
            known = (cuts or {}).get(cue.asset_id) if cue.asset_id is not None else None
            if cue.asset_id is None:
                problems.append(f"{where}: is silence, so there is no beat to cut on")
            elif known is None or known.beats is None:
                problems.append(
                    f"{where}: cutting on the beat needs the beat of {cue.asset_id} measured; analyze it "
                    "with analyze_asset first, or again if it was analyzed before beats were measured"
                )
            elif not known.beats:
                problems.append(
                    f"{where}: {cue.asset_id} has no steady beat to cut on — it was measured, and nothing "
                    "in it pulses regularly enough to call one"
                )

    if not problems:
        pieces, cleaned = compile_pieces(plan, clips, children, assets, cuts)
        duration = compiled_duration(pieces)
        notes.extend(_cut_notes(
            pieces,
            _selected(plan, clips, children, assets, cuts),
            _selected(plan, clips, children, assets, None),
            cuts, duration,
        ))
        notes.extend(cleaned)
        problems.extend(_cue_problems(plan, pieces))
        for position, cue in enumerate(cues, start=1):
            entry = song_entry(cue, (cuts or {}).get(cue.asset_id) if cue.asset_id is not None else None)
            if entry != cue.start:
                notes.append(
                    f"music cue {position} comes in at {entry:g}s of {cue.asset_id}, the first beat after "
                    f"the {cue.start:g}s asked for, so the cuts it sets the beat for are counted from a beat"
                )
        _, covering, said = broll_covers(plan, pieces, clips, assets, cuts)
        problems.extend(covering)
        notes.extend(said)
        heard = {
            level for clip_id, level in (levels or {}).items()
            if any(clip.speaker == clip_id for clip in clips.values())
        }
        if len(heard) > 1 and max(heard) - min(heard) >= VOICE_SPREAD_DB and not plan.level_voices:
            notes.append(
                f"the voices in this cut are {max(heard) - min(heard):.0f}dB apart, so one of them will sound "
                "much louder than the other. Set `level_voices` to bring them together"
            )
        if plan.target.seconds:
            drift = abs(duration - plan.target.seconds) / plan.target.seconds
            if drift > LENGTH_TOLERANCE:
                notes.append(
                    f"the cut comes out {duration:.1f}s against a target of {plan.target.seconds:g}s, "
                    f"{drift:.0%} out"
                )
        for beat in plan.beats:
            if beat.target_seconds is None:
                continue
            chosen = [selection for selection in plan.selections if selection.beat_id == beat.id]
            in_beat = compiled_duration(plan_pieces(
                plan.model_copy(update={"selections": chosen}), clips, children, assets, cuts,
            ))
            if abs(in_beat - beat.target_seconds) / beat.target_seconds > LENGTH_TOLERANCE:
                notes.append(f"beat {beat.id} ({beat.name}) runs {in_beat:.1f}s against {beat.target_seconds:g}s")
    return problems, notes

def _compiled_clips(project: Optional[Project]) -> List[Tuple[str, Clip]]:
    """List the clips sitting on the tracks a compile owns.

    Args:
        project: Project to read, or `None` for an empty one.

    Returns:
        One `(track_id, clip)` per clip on the sequence, music and B-roll
        tracks the compiler builds. Clips on any other track belong to whoever
        put them there and are not listed.
    """
    if project is None:
        return []
    return [
        (track.id, clip)
        for track in project.tracks if track.id in COMPILED_TRACKS
        for clip in track.clips
    ]

def check_recompile(
    plan: EditPlan,
    project: Project,
    pieces: Sequence[Piece],
    covers: Sequence[Cover] = (),
    beds: Sequence["Bed"] = (),
) -> List[str]:
    """Check that compiling over a project would not destroy work done by hand.

    The compiler owns the sequence and the music bed, and rebuilds them both.
    What it must not do is quietly rebuild over a change somebody made
    themselves, because that turns every round of feedback into a round of
    lost work. So a clip on those tracks has to be either something this plan
    compiled — which may be replaced — or something pinned, which is kept as
    it stands and has to still have a place in the new cut.

    Args:
        plan: Plan about to be compiled.
        project: Project it would be compiled onto.
        pieces: The windows the plan compiles to.
        covers: The B-roll the plan lays over them, which is identified by its
            own pair of clips rather than by a window.
        beds: The music the plan lays under them, each stretch identified by
            its cue and its pass through the song.

    Returns:
        One message per problem, empty when the project is safe to compile
        over.
    """
    problems: List[str] = []
    available = {piece.from_clip_ids for piece in pieces}
    available |= {(cover.clip_id, cover.over_clip_id) for cover in covers}
    available |= {bed.key for bed in beds}
    if beds:
        # A music clip compiled before stretches had names of their own carries none; it
        # was the first stretch, and is taken as the first stretch now.
        available.add(())
    for track_id, clip in _compiled_clips(project):
        if clip.pinned and track_id in (MUSIC_TRACK_ID, MUSIC_B_TRACK_ID) and beds:
            ordered = sorted((bed for bed in beds if bed.track_id == track_id), key=lambda bed: bed.timeline_in)
            match = next((index for index, bed in enumerate(ordered) if bed.key == tuple(clip.from_clip_ids)),
                         0 if not clip.from_clip_ids else None)
            if match is not None and match + 1 < len(ordered):
                room = ordered[match + 1].timeline_in - ordered[match].timeline_in
                if float(clip.timeline_duration) > room + 0.001:
                    problems.append(
                        f"music clip {clip.id} was lengthened by hand to {float(clip.timeline_duration):.1f}s, "
                        f"and the next stretch of music now starts {room:.1f}s after it; shorten it, or unpin "
                        "it with set_clip_pinned to let the plan lay it again"
                    )
        if clip.pinned:
            if tuple(clip.from_clip_ids) not in available:
                problems.append(
                    f"clip {clip.id} on track {track_id} was adjusted by hand, and the footage it was made "
                    "from is no longer in the plan; put it back in the plan, or unpin the clip with "
                    "set_clip_pinned to let it go"
                )
            continue
        if clip.from_plan_id != plan.id:
            came_from = f"plan {clip.from_plan_id}" if clip.from_plan_id else "nowhere this plan knows about"
            problems.append(
                f"clip {clip.id} on track {track_id} came from {came_from}; compiling would remove it. "
                "Pin it with set_clip_pinned to keep it, or take it off the track"
            )
    return problems

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
    starts: List[float] = []
    position = 0.0
    for piece in pieces:
        starts.append(round(position, 3))
        position += piece.played
    total = round(position, 3)
    entries = sorted(_cue_entries(plan, pieces).items())
    crossfade = plan.music.crossfade_seconds
    beds: List[Bed] = []
    lane, sounding = 0, False
    for order, (index, cue) in enumerate(entries):
        at = starts[index]
        until = starts[entries[order + 1][0]] if order + 1 < len(entries) else total
        if cue.asset_id is None or until <= at:
            sounding = False
            continue
        entry = song_entry(cue, cuts.get(cue.asset_id) if cuts else None)
        # A song cross-fading in over the one before starts early, and from earlier in
        # itself by the same amount — so at the cut it is exactly where the cue said, on
        # its beat when it cuts on the beat. Never from before the top of the song.
        lead = min(crossfade, entry, at) if crossfade and sounding else 0.0
        if lead:
            lane = 1 - lane
            # The song going out fades over exactly the stretch the one coming in rises.
            beds[-1] = replace(beds[-1], fade_out=round(min(lead, beds[-1].end - beds[-1].start), 3))
        else:
            lane = 0
        # Everything settled to the millisecond the timeline keeps before it is added up,
        # so a loop can neither run a hair past the file nor overlap the next one.
        last = _song_length(assets.get(cue.asset_id))
        parts: List[Tuple[float, float, float]] = []
        cursor, begin = round(at - lead, 3), round(entry - lead, 3)
        while until - cursor > 0.001:
            end = round(until - cursor + begin, 3) if last is None else min(round(until - cursor + begin, 3), last)
            if end - begin <= 0.001:
                break
            parts.append((begin, end, round(cursor, 3)))
            cursor = round(cursor + end - begin, 3)
            # Loops come back to where the cue came in, not to the lead-in before it.
            begin = entry
        if not parts:
            sounding = False
            continue
        sounding = True
        lengths = [end - start for start, end, _ in parts]
        fade_in = lead if lead else min(cue.fade_in, lengths[0])
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

def _as_made(pinned: Clip) -> dict:
    """Describe a hand-adjusted clip so it can be put back exactly as it is.

    Every field a person can change by hand, because a recompile rebuilds a
    pinned clip by inserting it again and anything left out here is quietly
    lost. Where it sits in the order is the one thing the plan still decides.

    Args:
        pinned: The clip as somebody left it.

    Returns:
        The fields, ready to go on the operation that recreates it.
    """
    return {
        "source_range": {
            "start": float(pinned.source_range.start), "end": float(pinned.source_range.end),
        },
        "volume": pinned.volume,
        "speed": pinned.speed,
        "audio_fade_in": float(pinned.audio_fade_in),
        "audio_fade_out": float(pinned.audio_fade_out),
        "audio_lead": float(pinned.audio_lead),
        "audio_lag": float(pinned.audio_lag),
        "cleanup": pinned.cleanup.model_dump(),
        "preserve_pitch": pinned.preserve_pitch,
        "transition_in": (
            None if pinned.transition_in is None else pinned.transition_in.model_dump(mode="json")
        ),
        "video_fade_in": float(pinned.video_fade_in),
        "video_fade_out": float(pinned.video_fade_out),
        "color": pinned.color.model_dump() if pinned.color else None,
        "layout": pinned.layout.model_dump() if pinned.layout else None,
    }

def compile_operations(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
    project: Optional[Project] = None,
    cuts: Optional[Mapping[str, CleanCuts]] = None,
    levels: Optional[Mapping[str, float]] = None,
) -> Tuple[List[dict], Dict[str, dict]]:
    """Turn a plan into the edit operations that build its cut.

    Compiling a second time rebuilds the sequence from scratch, except for
    the clips somebody pinned: those keep the range and the settings they were
    given by hand, and only their place in the order comes from the plan.

    Args:
        plan: Plan to compile. Check it first, with `check_plan` and, when
            compiling over existing work, `check_recompile`.
        clips: The timeline's clips, keyed by ID.
        children: Each section's own clips, keyed by section ID.
        assets: The assets they play from, keyed by asset ID.
        project: What is already on the timeline, if anything.
        cuts: Where each file may be cut without splitting a word, keyed by
            asset ID.
        levels: How loudly each joined voice speaks, as
            `semantic.voice_levels` measures it. Used only when the plan asked
            for its voices matched; without that, or without them, every window
            stays at the level it was recorded.

    Returns:
        `(operations, provenance)` — operations for `apply_edits`, and, keyed
        by the clip ID each one creates, where it came from and whether it
        stays pinned. The provenance is stamped onto the clips after the
        operations are applied, so it never has to travel through the tool
        surface.
    """
    pieces = plan_pieces(plan, clips, children, assets, cuts)
    # Only when the plan asked for it: matching two people's levels is a judgement about
    # a conversation, and on one speaker there is nothing to match.
    gains = voice_gains(levels) if plan.level_voices and levels else None
    # Keyed by track as well as by what it was made of. The three kinds of clip the
    # compiler owns mean different things by `from_clip_ids` — merged semantic clips on
    # the sequence, a (cover, covered) pair on the B-roll track, nothing at all for the
    # music — and one namespace for all three lets a sequence clip made of exactly those
    # two semantic clips take the B-roll's hand-made settings.
    kept = {
        (track_id, tuple(clip.from_clip_ids)): clip
        for track_id, clip in _compiled_clips(project) if clip.pinned
    }

    operations: List[dict] = [
        # Everything the compiler owns comes down first, so the cut is rebuilt rather than
        # added to. Rippling is off: the whole track goes, so there is nothing to close up.
        {"action": "delete_clip", "track_id": track_id, "clip_id": clip.id, "ripple": False}
        for track_id, clip in _compiled_clips(project)
    ]
    present = {track.id for track in project.tracks} if project is not None else set()
    if VIDEO_TRACK_ID not in present:
        operations.append({"action": "add_track", "track_id": VIDEO_TRACK_ID, "track_type": "video"})

    provenance: Dict[str, dict] = {}
    for position, piece in enumerate(pieces, start=1):
        clip_id = f"p{position:03d}"
        # Taken rather than read: taking a pause out of the middle of a shot leaves two
        # windows carrying the clip IDs one of them had, and one hand-made range must
        # not be stamped onto both of them. The earlier window keeps it.
        pinned = kept.pop((VIDEO_TRACK_ID, piece.from_clip_ids), None)
        placed = {
            "action": "insert_clip",
            "track_id": VIDEO_TRACK_ID,
            "clip_id": clip_id,
            "asset_id": piece.asset_id,
            "source_range": {"start": piece.start, "end": piece.end},
        }
        level = piece_gain(piece, clips, gains)
        if level != 1.0:
            placed["volume"] = level
        if piece.speed != 1.0:
            placed["speed"] = piece.speed
        if pinned is not None:
            placed.update(_as_made(pinned))
        operations.append(placed)
        provenance[clip_id] = {
            "from_plan_id": plan.id,
            "from_clip_ids": list(piece.from_clip_ids),
            "pinned": pinned is not None,
        }

    # B-roll goes on a track of its own above the sequence, at absolute positions rather
    # than in a row: a cover sits where the words it belongs to are, and the gaps between
    # covers are where the sequence shows through. Silent, because the point is the sound
    # underneath carrying on.
    covers, refused, _ = broll_covers(plan, pieces, clips, assets, cuts)
    if refused:
        # `check_plan` reports these, and `compile_plan` stops on them — but the engine
        # is meant to stand on its own, and a compiler that silently left a shot out of
        # the cut would be repairing a plan instead of handing it back.
        raise ValueError("this plan's B-roll does not hold up: " + "; ".join(refused))
    if covers:
        if BROLL_TRACK_ID not in present:
            operations.append({"action": "add_track", "track_id": BROLL_TRACK_ID, "track_type": "video"})
        for position, cover in enumerate(covers, start=1):
            clip_id = f"b{position:03d}"
            identity = (cover.clip_id, cover.over_clip_id)
            pinned = kept.pop((BROLL_TRACK_ID, identity), None)
            laid = {
                "action": "add_clip",
                "track_id": BROLL_TRACK_ID,
                "clip_id": clip_id,
                "asset_id": cover.asset_id,
                "source_range": {"start": cover.start, "end": cover.end},
                "timeline_in": cover.timeline_in,
                "volume": 0.0,
            }
            if pinned is not None:
                laid.update(_as_made(pinned))
            operations.append(laid)
            provenance[clip_id] = {
                "from_plan_id": plan.id,
                "from_clip_ids": list(identity),
                "pinned": pinned is not None,
            }

    # The compiler owns the markers it made and nothing else: one somebody added by hand
    # has no beat behind it, so it is carried across rather than rebuilt.
    by_hand = [
        marker.model_dump(mode="json")
        for marker in (project.markers if project is not None else [])
        if marker.from_beat_id is None
    ]
    markers = plan_markers(plan, pieces)
    if markers or by_hand:
        operations.append({"action": "set_markers", "markers": [*markers, *by_hand]})

    # Laid at absolute positions, like the B-roll, because a cue begins where its beat
    # does and a part with no music is a gap on the track rather than a clip of silence.
    beds = music_beds(plan, pieces, assets, cuts)
    if beds:
        for track_id in dict.fromkeys(bed.track_id for bed in sorted(beds, key=lambda bed: bed.lane)):
            if track_id not in present:
                operations.append({
                    "action": "add_track", "track_id": track_id, "track_type": "audio",
                    "duck_under_speech": plan.music.duck_under_speech,
                })
            else:
                operations.append({
                    "action": "set_track_audio", "track_id": track_id,
                    "duck_under_speech": plan.music.duck_under_speech,
                })
        for position, bed in enumerate(beds, start=1):
            clip_id = f"m{position:03d}"
            laid = {
                "action": "add_clip", "track_id": bed.track_id, "clip_id": clip_id,
                "asset_id": bed.asset_id, "source_range": {"start": bed.start, "end": bed.end},
                "timeline_in": bed.timeline_in, "volume": bed.volume,
                "audio_fade_in": bed.fade_in, "audio_fade_out": bed.fade_out,
            }
            # A stretch turned down or refaded by hand comes back as it was left; only where
            # it sits is the plan's. One compiled before stretches had names is the first.
            pinned = kept.pop((bed.track_id, bed.key), None)
            if pinned is None and position == 1:
                pinned = kept.pop((MUSIC_TRACK_ID, ()), None)
            if pinned is not None:
                laid.update(_as_made(pinned))
            operations.append(laid)
            provenance[clip_id] = {
                "from_plan_id": plan.id, "from_clip_ids": list(bed.key), "pinned": pinned is not None,
            }
    return operations, provenance

@dataclass(frozen=True)
class PieceChange:
    """What happened to one window of the cut between two versions of a plan.

    Attributes:
        status: `kept`, `retrimmed` (the same footage, cut at different
            points), `moved` (the same footage, somewhere else in the order),
            `added`, or `dropped`.
        piece: The window as it is now; for a dropped one, as it was.
        was: The window as it was, when there was one.
        at: Where it starts in its cut, in seconds: the new cut, or for a
            dropped window the old one.
    """

    status: str
    piece: Piece
    was: Optional[Piece]
    at: float

def piece_changes(before: Sequence[Piece], after: Sequence[Piece]) -> List[PieceChange]:
    """Line up two compiled cuts and say what happened to each window.

    Windows are matched on what they were made of, the same identity a
    recompile matches hand-adjusted clips on. A window taken apart by a pause
    carries its clips twice, so the matches are made in order. Among the
    windows in both cuts, the longest run still in its old order counts as
    staying put, and anything outside it has moved — which is how a person
    would describe a reorder: one clip moved, not everything after it.

    Args:
        before: The old cut's windows, in order.
        after: The new cut's windows, in order.

    Returns:
        One change per window of the new cut, in its order, followed by one
        per window that was dropped, in the old cut's order.
    """
    waiting: Dict[Tuple[str, ...], List[int]] = {}
    for index, piece in enumerate(before):
        waiting.setdefault(piece.from_clip_ids, []).append(index)
    matched: List[Optional[int]] = [
        waiting[piece.from_clip_ids].pop(0) if waiting.get(piece.from_clip_ids) else None for piece in after
    ]
    # The longest run of matched windows still in their old order stays put.
    pairs = [(position, index) for position, index in enumerate(matched) if index is not None]
    chains: List[List[int]] = []
    for k, (position, index) in enumerate(pairs):
        prior = [chains[j] for j in range(k) if pairs[j][1] < index]
        chains.append(max(prior, key=len, default=[]) + [position])
    staying = set(max(chains, key=len, default=[]))

    changes: List[PieceChange] = []
    for (position, piece), index, at in zip(enumerate(after), matched, _starts(after)):
        if index is None:
            changes.append(PieceChange("added", piece, None, at))
            continue
        was = before[index]
        status = "moved" if position not in staying else (
            "kept" if (was.asset_id, was.start, was.end) == (piece.asset_id, piece.start, piece.end) else "retrimmed"
        )
        changes.append(PieceChange(status, piece, was, at))
    used = {index for index in matched if index is not None}
    changes += [
        PieceChange("dropped", piece, piece, at)
        for index, (piece, at) in enumerate(zip(before, _starts(before))) if index not in used
    ]
    return changes

def _starts(pieces: Sequence[Piece]) -> List[float]:
    """Say where each window starts in its cut.

    Args:
        pieces: The windows, in order.

    Returns:
        Their start times, in seconds.
    """
    starts, position = [], 0.0
    for piece in pieces:
        starts.append(round(position, 3))
        position += piece.played
    return starts

def diff_plans(before: EditPlan, after: EditPlan) -> Dict[str, List[str]]:
    """Say what changed between two plans.

    Args:
        before: The earlier plan.
        after: The later one.

    Returns:
        Changes grouped as `goal`, `beats`, `selections`, `broll` and
        `music`, each a list of lines, with empty groups left out.
    """
    changes: Dict[str, List[str]] = {}

    top: List[str] = []
    if before.goal != after.goal:
        top.append(f"goal: {before.goal!r} → {after.goal!r}")
    if before.target.seconds != after.target.seconds:
        top.append(f"length: {before.target.seconds} → {after.target.seconds}")
    if before.target.platform != after.target.platform:
        top.append(f"platform: {before.target.platform} → {after.target.platform}")
    if before.timeline_id != after.timeline_id:
        top.append(f"footage: {before.timeline_id} → {after.timeline_id}")
    if pace_of(before) != pace_of(after):
        was, now = pace_of(before), pace_of(after)
        top.append(
            f"pacing: pauses over {was.pause:g}s → {now.pause:g}s taken out, {was.breath:g}s → {now.breath:g}s of air, "
            f"x{was.speed:g} → x{now.speed:g}"
        )
    if top:
        changes["goal"] = top

    was = {beat.id: beat for beat in before.beats}
    now = {beat.id: beat for beat in after.beats}
    beats = [f"added beat {beat_id} ({now[beat_id].name})" for beat_id in now.keys() - was.keys()]
    beats += [f"removed beat {beat_id} ({was[beat_id].name})" for beat_id in was.keys() - now.keys()]
    beats += [
        f"beat {beat_id}: {was[beat_id].name} → {now[beat_id].name}"
        for beat_id in was.keys() & now.keys() if was[beat_id].name != now[beat_id].name
    ]
    if [beat.id for beat in before.beats if beat.id in now] != [beat.id for beat in after.beats if beat.id in was]:
        beats.append("the beats were reordered")
    if beats:
        changes["beats"] = sorted(beats)

    chosen_before = {selection.clip_id: selection for selection in before.selections}
    chosen_after = {selection.clip_id: selection for selection in after.selections}
    selections = [f"added {clip_id}" for clip_id in chosen_after.keys() - chosen_before.keys()]
    selections += [f"dropped {clip_id}" for clip_id in chosen_before.keys() - chosen_after.keys()]
    for clip_id in chosen_before.keys() & chosen_after.keys():
        was_one, now_one = chosen_before[clip_id], chosen_after[clip_id]
        if was_one.beat_id != now_one.beat_id:
            selections.append(f"moved {clip_id}: {was_one.beat_id} → {now_one.beat_id}")
        if was_one.trim != now_one.trim:
            selections.append(f"retrimmed {clip_id}: {was_one.trim.kind.value} → {now_one.trim.kind.value}")
        if was_one.hold_picture != now_one.hold_picture:
            selections.append(
                f"{'held' if now_one.hold_picture else 'released'} the picture of {clip_id}"
            )
    if selections:
        changes["selections"] = sorted(selections)

    laid_before = {shot.over_clip_id: shot for shot in before.broll}
    laid_after = {shot.over_clip_id: shot for shot in after.broll}
    broll = [
        f"covered {over} with {laid_after[over].clip_id} for {laid_after[over].seconds:g}s"
        for over in laid_after.keys() - laid_before.keys()
    ]
    broll += [f"uncovered {over}" for over in laid_before.keys() - laid_after.keys()]
    broll += [
        f"changed the cover over {over}: {laid_before[over].clip_id} for "
        f"{laid_before[over].seconds:g}s → {laid_after[over].clip_id} for {laid_after[over].seconds:g}s"
        for over in laid_before.keys() & laid_after.keys() if laid_before[over] != laid_after[over]
    ]
    if broll:
        changes["broll"] = sorted(broll)

    def cues(plan: EditPlan) -> Dict[Optional[str], MusicCue]:
        """Key a plan's music cues by the beat each comes in on.

        Args:
            plan: The plan.

        Returns:
            The cues, keyed by beat ID, None for one that starts with the video.
        """
        return {cue.beat_id: cue for cue in plan.music.cues} if plan.music is not None else {}

    def named(beat_id: Optional[str]) -> str:
        """Name where a cue comes in, for a line of the diff.

        Args:
            beat_id: The beat, or None for the start.

        Returns:
            The phrase.
        """
        return f"beat {beat_id}" if beat_id is not None else "the start"

    heard_before, heard_after = cues(before), cues(after)
    music = [
        f"music from {named(beat_id)}: {heard_after[beat_id].asset_id or 'silence'}"
        for beat_id in heard_after.keys() - heard_before.keys()
    ]
    music += [f"no longer a music cue at {named(beat_id)}" for beat_id in heard_before.keys() - heard_after.keys()]
    music += [
        f"music from {named(beat_id)}: {heard_before[beat_id].asset_id or 'silence'} → "
        f"{heard_after[beat_id].asset_id or 'silence'}"
        + ("" if heard_before[beat_id].cut_on_beat == heard_after[beat_id].cut_on_beat else
           f", {'now' if heard_after[beat_id].cut_on_beat else 'no longer'} cutting on the beat")
        for beat_id in heard_before.keys() & heard_after.keys() if heard_before[beat_id] != heard_after[beat_id]
    ]
    if (before.music is not None and after.music is not None
            and before.music.duck_under_speech != after.music.duck_under_speech):
        music.append(f"{'now' if after.music.duck_under_speech else 'no longer'} ducking under speech")
    if music:
        changes["music"] = sorted(music, key=str)
    return changes
