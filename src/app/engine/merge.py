"""Two versions of a cut made into one, a part from each as the user picked.

The parts are the ones the comparison lists — the cut's markers — and each is
taken whole from one side: its clips in the sequence, and whatever is laid
over them (covering picture, a narration). The video as a whole — its songs,
its captions and how they look — is taken from one side or the other too.
Captions belong to the footage rather than to a second of the cut, so they
follow whichever clips end up in it.
"""

from decimal import Decimal
from typing import Dict, List, Mapping, Optional, Tuple

from app.engine.compare import Part, clips_in, match_clips, parts_of
from app.models.plan import EditPlan
from app.models.timeline import Clip, Marker, Project, Track, TrackType

# What can be picked for the video as a whole, besides its parts.
WHOLE = ("music", "captions", "caption_style")


def _middle(clip: Clip) -> float:
    """Where the middle of a clip is on the cut."""
    return float(clip.timeline_in) + float(clip.timeline_duration) / 2


def _part_at(parts: List[Part], seconds: float) -> Optional[Part]:
    """The part a moment of the cut is in."""
    for index, part in enumerate(parts):
        if part.start <= seconds < part.end or (index == len(parts) - 1 and seconds >= part.start):
            return part
    return None


def _order(left: List[Part], right: List[Part]) -> List[Tuple[str, Optional[Part], Optional[Part]]]:
    """Every part of either side once, in the right side's order, a part only the left has after its neighbour there."""
    by_name = {part.name: part for part in right}
    order: List[Tuple[str, Optional[Part], Optional[Part]]] = []
    for part in right:
        match = next((other for other in left if other.name == part.name), None)
        order.append((part.name, match, part))
    for index, part in enumerate(left):
        if part.name in by_name:
            continue
        before = left[index - 1].name if index else None
        at = next((place + 1 for place, (name, _, _) in enumerate(order) if name == before), 0)
        order.insert(at, (part.name, part, None))
    return order


def merge(before: Project, after: Project, picks: Mapping[str, str]) -> Project:
    """Make one cut of two, a part from each side and the video's songs and captions from one.

    Args:
        before: The left version.
        after: The right version, which the result is built on: its ID,
            name, size and everything not picked stay.
        picks: `a` or `b` for each part name and for `music`, `captions`
            and `caption_style`; anything left out is the right side's.

    Returns:
        The merged cut, laid end to end in the right side's order of parts,
        with a marker at the start of each.
    """
    def side(key: str) -> str:
        """Which side a part, or the video's songs or captions, is taken from."""
        return "a" if picks.get(key) == "a" else "b"

    left, right = parts_of(before), parts_of(after)
    base_left, base_right = before.base_video_track, after.base_video_track
    taken: List[Tuple[Clip, float]] = []  # each clip with how far it moves
    over: Dict[str, List[Tuple[Clip, float]]] = {}
    markers: List[Marker] = []
    used_ids = set()
    cursor = 0.0
    # Each sequence clip's part. On the left, a clip showing the same footage as one on the right is
    # in that one's part: the result follows the right side's parts, and a shot that slid across a
    # stale marker would otherwise be in neither part taken.
    left_clips = list(base_left.clips) if base_left else []
    right_clips = list(base_right.clips) if base_right else []
    part_of: Dict[int, str] = {}
    for parts, clips in ((left, left_clips), (right, right_clips)):
        for part in parts:
            for clip in clips_in(part, clips, part is parts[-1]):
                part_of[id(clip)] = part.name
    for one, other in match_clips(left_clips, right_clips):
        if id(other) in part_of:
            part_of[id(one)] = part_of[id(other)]
    for name, a_part, b_part in _order(left, right):
        chosen_side = side(name)
        part = a_part if chosen_side == "a" else b_part
        if part is None:
            continue
        project, base = (before, base_left) if chosen_side == "a" else (after, base_right)
        parts = left if chosen_side == "a" else right
        clips = [clip for clip in (left_clips if chosen_side == "a" else right_clips) if part_of.get(id(clip)) == name]
        # A trim by hand moves the clips but not the markers: a part runs over its own clips.
        start = min(float(clip.timeline_in) for clip in clips) if clips else part.start
        end = max(float(clip.timeline_out) for clip in clips) if clips else part.end
        shift = cursor - start
        old_marker = next((marker for marker in project.markers or [] if marker.name == name), None)
        markers.append(Marker(id=old_marker.id if old_marker else f"m{len(markers) + 1}", name=name,
                              timeline_in=Decimal(str(round(cursor, 3))),
                              from_beat_id=old_marker.from_beat_id if old_marker else None))
        for clip in clips:
            taken.append((clip, shift))
        for track in project.tracks:
            if track is base or (track.track_type == TrackType.AUDIO and not track.voice):
                continue
            for clip in track.clips:
                if _part_at(parts, _middle(clip)) is part:
                    over.setdefault(track.id, []).append((clip, shift))
        cursor += end - start

    def placed(clip: Clip, shift: float) -> Clip:
        """A clip moved to its place in the merged cut, under an ID nothing else there has."""
        new_id = clip.id
        while new_id in used_ids:
            new_id = f"{new_id}x"
        used_ids.add(new_id)
        return clip.model_copy(update={"id": new_id,
                                       "timeline_in": Decimal(str(round(float(clip.timeline_in) + shift, 3)))})

    sequence = [placed(clip, shift) for clip, shift in sorted(taken, key=lambda item: float(item[0].timeline_in) + item[1])]
    if sequence and sequence[0].transition_in is not None:
        # Nothing comes before the first clip now for it to dissolve from.
        sequence[0] = sequence[0].model_copy(update={"transition_in": None})
    for index in range(1, len(sequence)):
        previous, clip = sequence[index - 1], sequence[index]
        if clip.transition_in is not None and abs(float(previous.timeline_out) - float(clip.timeline_in)) > 0.001:
            sequence[index] = clip.model_copy(update={"transition_in": None})

    music_from = before if side("music") == "a" else after
    tracks: List[Track] = []
    seen = set()
    for track in after.tracks + [track for track in before.tracks if track.id not in {item.id for item in after.tracks}]:
        if track.id in seen:
            continue
        seen.add(track.id)
        if base_right is not None and track.id == base_right.id:
            tracks.append(track.model_copy(update={"clips": sequence}))
        elif track.track_type == TrackType.AUDIO and not track.voice:
            kept = next((item for item in music_from.tracks if item.id == track.id), None)
            if kept is not None:
                tracks.append(kept.model_copy(deep=True))
            elif track in after.tracks:
                tracks.append(track.model_copy(update={"clips": []}))
        else:
            clips = sorted((placed(clip, shift) for clip, shift in over.get(track.id, [])),
                           key=lambda clip: clip.timeline_in)
            if clips or track in after.tracks:
                tracks.append(track.model_copy(update={"clips": clips}))
    captions_from = before if side("captions") == "a" else after
    style_from = before if side("caption_style") == "a" else after
    return after.model_copy(update={
        "tracks": tracks,
        "markers": markers if (before.markers or after.markers) else [],
        "subtitles": [cue.model_copy() for cue in captions_from.subtitles],
        "caption_style": style_from.caption_style.model_copy(),
    })


def merge_plans(left: EditPlan, right: EditPlan, beats_from_left: List[str]) -> EditPlan:
    """Make one plan of two, the named beats' footage, reasons and trims from the left.

    Both plans must be over the same timeline, so their clip IDs mean the
    same moments.

    Args:
        left: The left version's plan.
        right: The right version's, which the result is built on.
        beats_from_left: Names of the beats to take from the left.

    Returns:
        The right plan with those beats' selections, covering shots and
        rejections replaced by the left's.
    """
    chosen = set(beats_from_left)
    right_ids = {beat.name: beat.id for beat in right.beats}
    left_ids = {beat.id: beat.name for beat in left.beats}
    replaced = {right_ids[name] for name in chosen if name in right_ids}
    taken = [selection.model_copy(update={"beat_id": right_ids[left_ids[selection.beat_id]]})
             for selection in left.selections
             if left_ids.get(selection.beat_id) in chosen and left_ids[selection.beat_id] in right_ids]
    kept = [selection for selection in right.selections if selection.beat_id not in replaced]
    # In the order the right plan's beats come in.
    order = {beat.id: index for index, beat in enumerate(right.beats)}
    selections = sorted(kept + taken, key=lambda selection: order.get(selection.beat_id, len(order)))
    used = {selection.clip_id for selection in selections}
    left_over = {selection.clip_id for selection in taken}
    broll = [shot for shot in right.broll if shot.over_clip_id in used and shot.over_clip_id not in left_over] + \
        [shot for shot in left.broll if shot.over_clip_id in left_over]
    return right.model_copy(update={"selections": selections, "broll": broll})
