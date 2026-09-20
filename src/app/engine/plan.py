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

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.models.media import Asset
from app.models.plan import EditPlan, Selection, TrimKind
from app.models.semantic import ClipKind, SemanticClip, SemanticTimeline
from app.models.timeline import Clip, Project

# Air left around a cut, taken from the clip's measured headroom, so a line does not
# begin the instant the picture does. Never more than the headroom allows.
BREATH_SECONDS = 0.1
# Two pieces this close together become one clip; a gap this short reads as a stumble.
MERGE_GAP_SECONDS = 0.3
VIDEO_TRACK_ID = "main"
MUSIC_TRACK_ID = "music"
# How far the compiled length may sit from what the plan asked for before it is worth saying.
LENGTH_TOLERANCE = 0.1

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
    """

    asset_id: str
    start: float
    end: float
    from_clip_ids: Tuple[str, ...]

    @property
    def duration(self) -> float:
        """Length of the window in seconds.

        Returns:
            The seconds it covers.
        """
        return self.end - self.start

def _breath(clip: SemanticClip) -> Tuple[float, float]:
    """Work out how much air a clip can be given at each end.

    Args:
        clip: Clip to pad.

    Returns:
        `(lead, trail)` in seconds, never more than the clip's headroom.
    """
    return min(BREATH_SECONDS, clip.safe_in), min(BREATH_SECONDS, clip.safe_out)

def _piece(
    clip: SemanticClip,
    asset: Asset,
    start: Optional[float] = None,
    end: Optional[float] = None,
    pad_head: bool = True,
    pad_tail: bool = True,
) -> Piece:
    """Cut one window out of a clip, with air around it where there is room.

    Args:
        clip: Clip the window comes from.
        asset: The asset it plays from, for its length.
        start: Window start in the source; the clip's own start by default.
        end: Window end in the source; the clip's own end by default.
        pad_head: Whether to reach back into the headroom before the window.
        pad_tail: Whether to reach on into the headroom after it.

    Returns:
        The window, in source seconds and inside the asset.
    """
    lead, trail = _breath(clip)
    opening = (clip.source_range.start if start is None else start) - (lead if pad_head else 0.0)
    closing = (clip.source_range.end if end is None else end) + (trail if pad_tail else 0.0)
    if asset.duration is not None:
        closing = min(closing, float(asset.duration))
    return Piece(asset.id, round(max(0.0, opening), 3), round(closing, 3), (clip.id,))

def _selection_pieces(
    selection: Selection,
    clip: SemanticClip,
    children: Sequence[SemanticClip],
    asset: Asset,
) -> List[Piece]:
    """Work out what one selection contributes to the cut.

    Args:
        selection: The selection, carrying its trim.
        clip: The semantic clip it names.
        children: That clip's own clips, in time order, empty for an utterance.
        asset: The asset they play from.

    Returns:
        The windows this selection puts on the timeline, in order.
    """
    trim = selection.trim
    if trim.kind == TrimKind.KEEP:
        wanted = set(trim.keep_clip_ids)
        return [_piece(child, asset) for child in children if child.id in wanted]
    if trim.kind == TrimKind.TIGHTEN:
        # Drop what nobody would keep — the pauses and the unusable picture — and leave the rest.
        kept = [child for child in children if child.kind not in (ClipKind.SILENCE, ClipKind.UNUSABLE)]
        return [_piece(child, asset) for child in kept] if kept else [_piece(clip, asset)]
    if trim.kind == TrimKind.HEAD and trim.seconds is not None:
        return [_piece(clip, asset, end=min(clip.source_range.start + trim.seconds, clip.source_range.end), pad_tail=False)]
    if trim.kind == TrimKind.TAIL and trim.seconds is not None:
        return [_piece(clip, asset, start=max(clip.source_range.end - trim.seconds, clip.source_range.start), pad_head=False)]
    return [_piece(clip, asset)]

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
            merged[-1] = Piece(
                last.asset_id, last.start, max(last.end, piece.end),
                (*last.from_clip_ids, *piece.from_clip_ids),
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

def plan_pieces(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
) -> List[Piece]:
    """Work out every window the plan puts on the timeline, in order.

    Args:
        plan: Plan to lay out.
        clips: The timeline's clips, keyed by ID.
        children: Each section's own clips in time order, keyed by section ID.
        assets: The assets they play from, keyed by asset ID.

    Returns:
        The windows, merged where they nearly touch.
    """
    pieces: List[Piece] = []
    for selection in _ordered_selections(plan):
        clip = clips.get(selection.clip_id)
        asset = assets.get(clip.asset_id) if clip is not None else None
        if clip is None or asset is None:
            continue
        pieces.extend(_selection_pieces(selection, clip, children.get(clip.id, []), asset))
    return _merge(pieces)

def compiled_duration(pieces: Sequence[Piece]) -> float:
    """Measure how long the compiled cut runs.

    Args:
        pieces: The windows on the timeline.

    Returns:
        The total in seconds.
    """
    return round(sum(piece.duration for piece in pieces), 3)

def check_plan(
    plan: EditPlan,
    timeline: SemanticTimeline,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
) -> Tuple[List[str], List[str]]:
    """Check a plan against the footage it claims to be made of.

    Args:
        plan: Plan to check.
        timeline: The semantic timeline it names.
        clips: That timeline's clips, keyed by ID.
        children: Each section's own clips, keyed by section ID.
        assets: The assets they play from, keyed by asset ID.

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

    if plan.music is not None:
        music = assets.get(plan.music.asset_id)
        if music is None:
            problems.append(f"the music asset {plan.music.asset_id} is not registered")
        elif not music.has_audio:
            problems.append(f"the music asset {plan.music.asset_id} has no sound")

    if not problems:
        pieces = plan_pieces(plan, clips, children, assets)
        duration = compiled_duration(pieces)
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
                plan.model_copy(update={"selections": chosen}), clips, children, assets,
            ))
            if abs(in_beat - beat.target_seconds) / beat.target_seconds > LENGTH_TOLERANCE:
                notes.append(f"beat {beat.id} ({beat.name}) runs {in_beat:.1f}s against {beat.target_seconds:g}s")
    return problems, notes

def _compiled_clips(project: Optional[Project]) -> List[Tuple[str, Clip]]:
    """List the clips sitting on the tracks a compile owns.

    Args:
        project: Project to read, or `None` for an empty one.

    Returns:
        One `(track_id, clip)` per clip on the video and music tracks the
        compiler builds. Clips on any other track belong to whoever put them
        there and are not listed.
    """
    if project is None:
        return []
    return [
        (track.id, clip)
        for track in project.tracks if track.id in (VIDEO_TRACK_ID, MUSIC_TRACK_ID)
        for clip in track.clips
    ]

def check_recompile(plan: EditPlan, project: Project, pieces: Sequence[Piece]) -> List[str]:
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

    Returns:
        One message per problem, empty when the project is safe to compile
        over.
    """
    problems: List[str] = []
    available = [piece.from_clip_ids for piece in pieces]
    for track_id, clip in _compiled_clips(project):
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

def compile_operations(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
    project: Optional[Project] = None,
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

    Returns:
        `(operations, provenance)` — operations for `apply_edits`, and, keyed
        by the clip ID each one creates, where it came from and whether it
        stays pinned. The provenance is stamped onto the clips after the
        operations are applied, so it never has to travel through the tool
        surface.
    """
    pieces = plan_pieces(plan, clips, children, assets)
    kept = {
        tuple(clip.from_clip_ids): clip
        for _, clip in _compiled_clips(project) if clip.pinned
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
        pinned = kept.get(piece.from_clip_ids)
        placed = {
            "action": "insert_clip",
            "track_id": VIDEO_TRACK_ID,
            "clip_id": clip_id,
            "asset_id": piece.asset_id,
            "source_range": {"start": piece.start, "end": piece.end},
        }
        if pinned is not None:
            # The hand-made version wins on everything but where it sits in the order.
            placed["source_range"] = {
                "start": float(pinned.source_range.start), "end": float(pinned.source_range.end),
            }
            placed.update({
                "volume": pinned.volume,
                "audio_fade_in": float(pinned.audio_fade_in),
                "audio_fade_out": float(pinned.audio_fade_out),
                "video_fade_in": float(pinned.video_fade_in),
                "video_fade_out": float(pinned.video_fade_out),
                "color": pinned.color.model_dump() if pinned.color else None,
                "layout": pinned.layout.model_dump() if pinned.layout else None,
            })
        operations.append(placed)
        provenance[clip_id] = {
            "from_plan_id": plan.id,
            "from_clip_ids": list(piece.from_clip_ids),
            "pinned": pinned is not None,
        }

    if plan.music is not None and pieces:
        music = plan.music
        if MUSIC_TRACK_ID not in present:
            operations.append({
                "action": "add_track", "track_id": MUSIC_TRACK_ID, "track_type": "audio",
                "duck_under_speech": music.duck_under_speech,
            })
        else:
            operations.append({
                "action": "set_track_audio", "track_id": MUSIC_TRACK_ID,
                "duck_under_speech": music.duck_under_speech,
            })
        source = assets.get(music.asset_id)
        length = float(source.duration) if source is not None and source.duration is not None else compiled_duration(pieces)
        operations.append({
            "action": "insert_clip", "track_id": MUSIC_TRACK_ID, "clip_id": "music",
            "asset_id": music.asset_id, "source_range": {"start": 0, "end": length},
            "volume": music.volume,
        })
        # fit_track trims, extends or loops the bed to the length of the picture.
        operations.append({
            "action": "fit_track", "track_id": MUSIC_TRACK_ID,
            "fade_in": music.fade_in, "fade_out": music.fade_out,
        })
        provenance["music"] = {"from_plan_id": plan.id, "from_clip_ids": [], "pinned": False}
    return operations, provenance

def diff_plans(before: EditPlan, after: EditPlan) -> Dict[str, List[str]]:
    """Say what changed between two plans.

    Args:
        before: The earlier plan.
        after: The later one.

    Returns:
        Changes grouped as `goal`, `beats`, `selections` and `music`, each a
        list of lines, with empty groups left out.
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
    if selections:
        changes["selections"] = sorted(selections)

    if before.music != after.music:
        changes["music"] = [
            f"music: {before.music.asset_id if before.music else 'none'} → "
            f"{after.music.asset_id if after.music else 'none'}"
        ]
    return changes
