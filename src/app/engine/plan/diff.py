"""What changed between two plans, and between two compiled cuts."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.engine.plan.pieces import Piece, pace_of, piece_starts
from app.models.plan import EditPlan, MusicCue


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
    for (position, piece), index, at in zip(enumerate(after), matched, piece_starts(after)):
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
        for index, (piece, at) in enumerate(zip(before, piece_starts(before))) if index not in used
    ]
    return changes

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
