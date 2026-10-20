"""Where B-roll goes: the shots a plan lays over its cut, and the places that would take one."""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine.plan.pieces import Note, Piece, compiled_duration
from app.engine.semantic import CleanCuts
from app.models.media import Asset
from app.models.plan import EditPlan
from app.models.semantic import SemanticClip


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

def _covers_bad_picture(
    under: Tuple[Piece, float],
    seconds: float,
    length: float,
    cuts: Optional[Mapping[str, CleanCuts]],
) -> bool:
    """Say whether covering picture laid from a moment of the timeline hides a frame nobody should see.

    The one reason to cut away partway through a sentence: the shot went black
    or froze while somebody talked, and the sound cannot be cut (#99).

    Args:
        under: The window playing there, with the second it starts at.
        seconds: Where the covering picture cuts in on the timeline.
        length: How long it runs.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        True when the window's own picture under it is black or frozen
        somewhere in that stretch.
    """
    piece, at = under
    known = cuts.get(piece.asset_id) if cuts else None
    if known is None:
        return False
    start = piece.start + (seconds - at) * piece.speed
    end = min(piece.end, start + length * piece.speed)
    return any(min(end, stop) > max(start, begin) for begin, stop in known.bad_picture)

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

        opened = round(span[0] + shot.after_seconds, 3)
        if opened >= span[1]:
            problems.append(
                f"{where}: {shot.after_seconds:g}s into {shot.over_clip_id} is past where it ends in the cut, "
                f"{span[1] - span[0]:.2f}s in"
            )
            continue
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
        if beat is not None and _splits_a_sentence(beat, opened, cuts) and not _covers_bad_picture(
                beat, opened, shot.seconds, cuts):
            # Cutting away mid-sentence is the half of the boundary rule that can be
            # fixed for nothing: the shot moves to another clip and the cut keeps its
            # length. The other half — cutting back — cannot, so that one is reported.
            problems.append(
                f"{where}: cuts in partway through a sentence. Start it over a clip that opens one, "
                "or the viewer loses the line as the picture changes — partway through one only to cover a "
                "frame gone black or frozen"
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
            notes.append(Note(
                f"{where}: nobody is talking under it, so this swaps one picture for another rather than "
                "showing what is being said",
                "broll_over_silence", clip_ids=(shot.over_clip_id, shot.clip_id), look=True,
            ))

    if cuts:
        interrupted = []
        for cover in covers:
            under = _under(placed, cover.timeline_out)
            if under is not None and _splits_a_sentence(under, cover.timeline_out, cuts):
                interrupted.append(cover.over_clip_id)
        if interrupted:
            notes.append(Note(
                f"{len(interrupted)} B-roll shot(s) cut back partway through a sentence: "
                f"{', '.join(interrupted)}. Lengthening them would change how long the video runs, "
                "so it is left to whoever chose the length",
                "broll_ends_mid_sentence", clip_ids=interrupted, look=True,
            ))
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
