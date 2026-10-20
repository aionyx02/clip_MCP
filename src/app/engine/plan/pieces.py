"""The windows a plan's selections become, and the passes that clean them: retakes, pauses, trimmed ends, bad frames."""

import math
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from app.models.media import Asset
from app.models.plan import EditPlan, Selection, TrimKind
from app.engine.semantic import EDGE_TOLERANCE_SECONDS, CleanCuts
from app.models.semantic import ClipKind, SemanticClip
from app.models.timeline.parts import TitleCard


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

# A transition shortened below this is a few frames of flicker rather than a transition,
# so it is left out instead. Three frames at 30fps, two at 24.
MIN_TRANSITION_SECONDS = 0.1

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

class Note(str):
    """Something worth knowing about a plan, said and labelled.

    Still a sentence, so everything that reads notes as text reads these. The
    labels are for the AI: which clips and which part a note is about without
    reading the sentence for them, and whether it only says what was done or
    asks for a look.

    Attributes:
        kind: What sort of note, such as `pauses_taken_out`.
        beat_id: The part of the video it is about, when it is about one.
        clip_ids: The clips it is about.
        seconds: The one number the note is about, as its message says:
            seconds taken out or short, a length, or a point in a song.
        look: Whether it is worth weighing — something the plan did not ask
            for, or something left that may sound wrong — rather than only
            said. What to do about it is in the sentence.
    """

    kind: str
    beat_id: Optional[str]
    clip_ids: Tuple[str, ...]
    seconds: Optional[float]
    look: bool

    def __new__(
        cls, message: str, kind: str, *, beat_id: Optional[str] = None, clip_ids: Iterable[str] = (),
        seconds: Optional[float] = None, look: bool = False,
    ) -> "Note":
        note = super().__new__(cls, message)
        note.kind, note.beat_id, note.look = kind, beat_id, look
        note.clip_ids = tuple(dict.fromkeys(clip_ids))
        note.seconds = None if seconds is None else round(seconds, 3)
        return note

def note_record(note: str) -> dict:
    """Write a note out for the AI.

    Args:
        note: A `Note`, or a plain sentence from somewhere that has no labels
            to give, which is taken as worth a look.

    Returns:
        `kind`, `message` and `look`, with `beat_id`, `clip_ids` and `seconds`
        where the note has them.
    """
    if not isinstance(note, Note):
        return {"kind": "note", "message": str(note), "look": True}
    return {
        "kind": note.kind,
        "message": str(note),
        **({"beat_id": note.beat_id} if note.beat_id else {}),
        **({"clip_ids": list(note.clip_ids)} if note.clip_ids else {}),
        **({"seconds": note.seconds} if note.seconds is not None else {}),
        "look": note.look,
    }

def first_clips(pieces: Iterable["Piece"]) -> List[str]:
    """Name windows by the first clip each is made of, as the plan names them."""
    return [piece.from_clip_ids[0] for piece in pieces if piece.from_clip_ids]

def edge_of(piece: "Piece", side: str, seconds: float) -> str:
    """Say where one edge of a window is: `u0410 starts at 2.23s of its file`.

    A window merged from several clips starts on its first and ends on its last,
    so that is the one named, and the one whose selection is to be changed.
    """
    named = piece.from_clip_ids or (piece.asset_id,)
    return f"{named[-1] if side == 'ends' else named[0]} {side} at {seconds:.2f}s of its file"

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
        speed: How fast it plays: its selection's own speed when it has one,
            otherwise the plan's pacing speed, set once the windows are settled.
        own_speed: Whether its selection said how fast it plays: a timelapse
            or a slow motion is that piece's decision, and the pacing speed is
            about the talking, so it is not put on top.
        gain: How loud its own sound plays, from its selection's volume.
        keep_level: Whether its talking stays at the level it was recorded,
            from its selection.
        transition: How it comes in over the window before, as `(kind,
            seconds, through, direction)`, already shortened to the picture
            the file has; None is a straight cut. Only the first window of a
            beat has one.
        sound_lead: Seconds its sound starts before its picture, already
            shortened to the sound the file has.
        card: Set on a title card: the card. Its window is then only its
            length, from 0, and the file is the one behind it.
        card_from: For a card that was on the cut before this compile —
            made by hand, or the plan's and changed by hand — the plan it came
            from, if any, and whether it is pinned; None for one the plan
            makes now.
        excused: Per clip in `from_clip_ids`, whether its selection gave an
            `edge_reason`, so a reason on one of two selections of the same
            clip stays with that one. Left out of comparing windows: it says
            what may be heard at an edge, not what plays.
    """

    asset_id: str
    start: float
    end: float
    from_clip_ids: Tuple[str, ...]
    beat_id: str = ""
    speed: float = 1.0
    own_speed: bool = False
    gain: float = 1.0
    keep_level: bool = False
    transition: Optional[Tuple[str, float, str, str]] = None
    sound_lead: float = 0.0
    card: Optional[TitleCard] = None
    card_from: Optional[Tuple[Optional[str], bool]] = None
    excused: Tuple[bool, ...] = field(default=(), compare=False)

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
    if trim.kind == TrimKind.RANGE and trim.from_seconds is not None and trim.to_seconds is not None:
        return [_piece(
            clip, asset,
            start=min(clip.source_range.start + trim.from_seconds, clip.source_range.end),
            end=min(clip.source_range.start + trim.to_seconds, clip.source_range.end),
            pad_head=False, pad_tail=False, cuts=cuts, beat_id=beat, breath=breath,
        )]
    return [_piece(clip, asset, beat_id=beat, breath=breath)]

def merge_pieces(pieces: Sequence[Piece]) -> List[Piece]:
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
        if (
            last is not None and last.asset_id == piece.asset_id and piece.start - last.end <= MERGE_GAP_SECONDS
            # Only a window that plays on forward in its file: one from earlier in it, placed after,
            # is a flashback, and joining it on kept one of the two and lost the other (#91).
            and piece.start >= last.start
            # A window sped up or turned down on purpose is its own clip, however close it sits.
            and (last.speed, last.own_speed, last.gain, last.keep_level)
            == (piece.speed, piece.own_speed, piece.gain, piece.keep_level)
        ):
            # The first piece's beat wins. Two pieces this close are one shot, and a
            # marker inside a shot would say a new part starts partway through it.
            merged[-1] = replace(
                last, end=max(last.end, piece.end), from_clip_ids=(*last.from_clip_ids, *piece.from_clip_ids),
                excused=(*last.excused, *piece.excused),
            )
        else:
            merged.append(piece)
    return merged

def ordered_selections(plan: EditPlan) -> List[Selection]:
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

def selected_pieces(
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
    for selection in ordered_selections(plan):
        clip = clips.get(selection.clip_id)
        asset = assets.get(clip.asset_id) if clip is not None else None
        if clip is None or asset is None:
            continue
        pieces.extend(
            replace(
                piece,
                speed=1.0 if selection.speed is None else selection.speed,
                own_speed=selection.speed is not None,
                gain=1.0 if selection.volume is None else selection.volume,
                keep_level=selection.keep_level,
                excused=(bool(selection.edge_reason),),
            )
            for piece in _selection_pieces(
                selection, clip, children.get(clip.id, []), asset,
                None if cuts is None else cuts.get(clip.asset_id), pace_of(plan).breath,
            )
        )
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

# A window's edge moved further than this to reach the sound is more than a word placed a
# little late: it is said, with where it was asked for and where it went. Provisional.
MOVED_FAR_SECONDS = 0.3

def on_the_sound(
    pieces: Sequence[Piece],
    cuts: Optional[Mapping[str, CleanCuts]],
    pace: Pace = Pace(),
) -> Tuple[List[Piece], List[str]]:
    """Open and close each window where its sound does, not where the transcript says.

    A window's edges come from word timings, and the transcriber now and then
    places a first word late or a last one early. The measured silences, and
    the second listen where the room was too loud to measure one, say where
    the sound actually rises and falls, so an edge with talking the transcript
    missed right up against it moves out to where the talking starts or stops,
    and keeps the breath the plan asks for without taking in the words either
    side.

    Args:
        pieces: The windows, in order.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        `(pieces, notes)`.
    """
    if not cuts:
        return list(pieces), []
    out: List[Piece] = []
    moved = 0
    shifted: List[Piece] = []
    # Moves too long to be a slip of the transcript, said one by one so none is made unseen.
    far: List[str] = []
    far_in: List[Piece] = []
    for piece in pieces:
        known = cuts.get(piece.asset_id)
        if known is None:
            out.append(piece)
            continue
        start, end = piece.start, piece.end
        # Never out past the edge of a word the transcript has on the far side: sound running on
        # into it is that word, which the window was cut to leave out, and moving into it would
        # land the cut inside it (#77). Where it starts right at the cut, the cut stays. A word
        # run on across a silence is not where it was put, and is what moving out is for.
        placed = [(opened, closed) for opened, closed in known.words if known.placed(opened, closed)]
        before = max((closed for _, closed in placed if closed <= start + EDGE_TOLERANCE_SECONDS), default=0.0)
        after = min((opened for opened, _ in placed if opened >= end - EDGE_TOLERANCE_SECONDS), default=math.inf)
        onset = known.sound_before(start)
        if onset is not None:
            start = min(piece.start, round(max(onset - pace.breath, known.room_before(onset), before, 0.0), 3))
        fall = known.sound_after(end)
        if fall is not None:
            end = max(piece.end, round(min(fall + pace.breath, known.room_after(fall), after), 3))
        if (start, end) != (piece.start, piece.end):
            moved += (start != piece.start) + (end != piece.end)
            shifted.append(piece)
            pulled = [(side, asked, made) for side, asked, made in (("starts", piece.start, start), ("ends", piece.end, end))
                      if abs(made - asked) > MOVED_FAR_SECONDS]
            if pulled:
                far_in.append(piece)
                far += [f"{edge_of(piece, side, made)}, not the {asked:.2f}s asked" for side, asked, made in pulled]
            out.append(replace(piece, start=start, end=end))
        else:
            out.append(piece)
    notes = []
    if moved:
        notes.append(Note(
            f"{moved} cut(s) moved out to where the sound really starts or stops: the transcript placed a word "
            "inside the talking, and cutting there would take the start or end of it off",
            "cuts_moved_to_the_sound", clip_ids=first_clips(shifted),
        ))
    if far:
        notes.append(Note(
            "moved further than a slip of the transcript: " + "; ".join(far) + ". If the words there were meant "
            "to go, start or end the window on the edge of the first or last word kept, and it stays",
            "cuts_moved_far", clip_ids=first_clips(far_in), look=True,
        ))
    return out, notes

def without_retakes(
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
        notes.append(Note(
            f"{len(dropped)} window(s) say a line that is said again straight after, and were dropped, "
            f"{seconds:.1f}s in all: {', '.join(first_clips(dropped))}. "
            "The last complete go at each was kept",
            "retakes_dropped", clip_ids=first_clips(dropped), seconds=seconds, look=True,
        ))
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

def without_pauses(
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
    split: List[Piece] = []
    spoken_over: List[Piece] = []
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
                # `merge_pieces` runs after every stage and joins two halves this close back
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
                spoken_over.append(piece)
                continue
            out.append(replace(piece, start=opened, end=closing))
            taken.append(round(opening - closing, 3))
            split.append(piece)
            opened = opening
        out.append(replace(piece, start=opened))

    notes: List[str] = []
    if taken:
        notes.append(Note(
            f"{len(taken)} pause(s) longer than {pace.pause:g}s were taken out of the middle of a shot, "
            f"{sum(taken):.1f}s in all",
            "pauses_taken_out", clip_ids=first_clips(split), seconds=sum(taken),
        ))
    if talked_through:
        notes.append(Note(
            f"{talked_through} pause(s) were left in: they were measured as quiet, but the transcript "
            "has somebody talking across them, and cutting there would go through a word",
            "pauses_left_in", clip_ids=first_clips(spoken_over),
        ))
    return out, notes

def trimmed_ends(
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

    def say(where: str, taken: float, left: float, limit: float, piece: Piece) -> None:
        """Say what was trimmed, and what the limit would not let go of.

        Args:
            where: Which end of the cut, for the sentence.
            taken: Seconds removed.
            left: Seconds of the run-up or wind-down still there.
            limit: The trim that stopped it going further.
            piece: The window it was trimmed off.
        """
        said = f"{taken:.1f}s {where} was trimmed off"
        if left > 0:
            said += (
                f", and {left:.1f}s of it is still there: the trim goes no further than "
                f"{limit:g}s in case the shot is being held on purpose"
            )
        notes.append(Note(said, "ends_trimmed", beat_id=piece.beat_id or None, clip_ids=first_clips([piece]),
                          seconds=taken))

    head = out[0]
    known = cuts.get(head.asset_id)
    first_word = known.first_word_in(head.start, head.end) if known is not None else None
    if first_word is not None:
        # Where the talking really starts, when the transcript put its first word late.
        first_word = known.sound_before(first_word) or first_word
        wanted = round(first_word - pace.breath, 3)
        opening = round(min(wanted, head.start + HEAD_TRIM_SECONDS), 3)
        if opening > head.start:
            say("before the first word", opening - head.start, round(wanted - opening, 3), HEAD_TRIM_SECONDS, head)
            out[0] = replace(head, start=opening)

    tail = out[-1]
    known = cuts.get(tail.asset_id)
    last_word = known.last_word_in(tail.start, tail.end) if known is not None else None
    if last_word is not None:
        last_word = known.sound_after(last_word) or last_word
        wanted = round(last_word + pace.breath, 3)
        closing = round(max(wanted, tail.end - TAIL_TRIM_SECONDS), 3)
        if closing < tail.end:
            say("after the last word", tail.end - closing, round(closing - wanted, 3), TAIL_TRIM_SECONDS, tail)
            out[-1] = replace(tail, end=closing)

    return out, notes

def off_bad_frames(
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
    moved_in: List[Piece] = []
    stuck_in: List[Piece] = []
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
            if objected:
                stuck_in.append(piece)
            out.append(piece)
            continue
        shifted = sum(1 for now, before in zip(settled, (piece.start, piece.end)) if now != before)
        moved += shifted
        stuck += objected - shifted
        if shifted:
            moved_in.append(piece)
        if objected > shifted:
            stuck_in.append(piece)
        out.append(replace(piece, start=opening, end=closing))

    notes: List[str] = []
    if moved:
        notes.append(Note(f"{moved} cut(s) moved off black or frozen picture", "cuts_moved_off_bad_picture",
                          clip_ids=first_clips(moved_in)))
    if stuck:
        notes.append(Note(
            f"{stuck} cut(s) still land on black or frozen picture: no clean frame is within "
            f"{SNAP_SECONDS:g}s of them, so they were left as asked",
            "cuts_on_bad_picture", clip_ids=first_clips(stuck_in), look=True,
        ))
    return out, notes

def compiled_duration(pieces: Sequence[Piece]) -> float:
    """Measure how long the compiled cut runs.

    Args:
        pieces: The windows on the timeline.

    Returns:
        The total in seconds.
    """
    return round(sum(piece.played for piece in pieces), 3)

def piece_starts(pieces: Sequence[Piece]) -> List[float]:
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
