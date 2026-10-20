"""Turning a plan into the edit operations that build its cut."""

import hashlib
from dataclasses import replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine.plan.beats import on_the_beat, plan_pieces
from app.engine.plan.broll import broll_covers
from app.engine.plan.checks import as_left, compiled_clips
from app.engine.plan.music import music_beds, plan_markers
from app.engine.plan.pieces import BROLL_TRACK_ID, MUSIC_TRACK_ID, VIDEO_TRACK_ID, Piece
from app.engine.semantic import CleanCuts
from app.models.media import Asset
from app.models.plan import EditPlan, short_clip_ids
from app.models.semantic import SemanticClip
from app.models.timeline.parts import Clip, Project, TitleCard, card_background


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
        "keep_level": pinned.keep_level,
        "preserve_pitch": pinned.preserve_pitch,
        "transition_in": (
            None if pinned.transition_in is None else pinned.transition_in.model_dump(mode="json")
        ),
        "video_fade_in": float(pinned.video_fade_in),
        "video_fade_out": float(pinned.video_fade_out),
        "color": pinned.color.model_dump() if pinned.color else None,
        "layout": pinned.layout.model_dump() if pinned.layout else None,
        "fit": pinned.fit.model_dump() if pinned.fit else None,
        "motion": pinned.motion,
    }

def _with_cards(plan: EditPlan, pieces: List[Piece], project: Optional[Project]) -> List[Piece]:
    """Put the title cards into a compile's windows, where they will play.

    A card is a window of its own, so everything laid out by where windows
    fall — markers, B-roll, music — makes room for it. A part the plan gives
    a card opens on it, and the part's transition comes in on the card. A
    card that was on the cut already and made by hand, or made by a plan and
    changed by hand since, comes back as it was: the plan's in front of its
    part, one made by hand next to the shot whose frame it shows — in front
    of it, or after it when it shows the shot's end — or failing that
    wherever it was nearest. Another plan's cards nobody touched go with
    that plan.

    Args:
        plan: The plan being compiled.
        pieces: Its windows, as laid out.
        project: What is on the timeline now.

    Returns:
        The windows with the cards among them.
    """
    beats = {beat.id: beat for beat in plan.beats}
    old = sorted((clip for track_id, clip in compiled_clips(project)
                  if track_id == VIDEO_TRACK_ID and clip.card is not None), key=lambda clip: clip.timeline_in)
    held = {clip.card.beat_id: clip for clip in old
            if clip.from_plan_id == plan.id and clip.pinned and clip.card.beat_id in beats}
    # Made by hand, or a plan's card somebody changed: kept. Another plan's untouched cards are
    # that plan's work, rebuilt by nobody now, and go with it.
    by_hand = [clip for clip in old if clip not in held.values() and (clip.from_plan_id is None or clip.pinned)]

    def from_clip(clip: Clip, beat_id: str) -> Piece:
        length = float(clip.timeline_duration)
        return Piece(asset_id=clip.asset_id, start=0.0, end=length, from_clip_ids=(), beat_id=beat_id,
                     card=clip.card, card_from=(clip.from_plan_id, clip.pinned))

    out: List[Piece] = []
    opened: set = set()
    for piece in pieces:
        beat = beats.get(piece.beat_id)
        if beat is not None and piece.beat_id not in opened:
            opened.add(piece.beat_id)
            card: Optional[Piece] = None
            if piece.beat_id in held:
                card = from_clip(held[piece.beat_id], piece.beat_id)
            elif beat.title_card is not None:
                wanted = beat.title_card
                photo = wanted.photo_asset_id
                card = Piece(
                    asset_id=photo or piece.asset_id, start=0.0, end=wanted.seconds, from_clip_ids=(),
                    beat_id=piece.beat_id,
                    card=TitleCard(
                        title=wanted.title, subtitle=wanted.subtitle,
                        background=card_background(wanted.background, photo),
                        at=0 if photo else piece.start, beat_id=piece.beat_id,
                        **({"colour": wanted.colour} if wanted.colour else {}),
                    ),
                )
            if card is not None:
                # The part comes in on its card, and its first shot follows the card on a cut.
                out.append(replace(card, transition=piece.transition))
                piece = replace(piece, transition=None)
        out.append(piece)

    for clip in by_hand:
        at = float(clip.card.at)
        shows = next((index for index, piece in enumerate(out) if piece.card is None
                      and piece.asset_id == clip.asset_id and piece.start - 0.05 <= at <= piece.end + 0.05), None)
        found = shows is not None
        if shows is None:
            starts, position = [], 0.0
            for piece in out:
                starts.append(position)
                position += piece.played
            was = float(clip.timeline_in)
            shows = min(range(len(out)), key=lambda index: abs(starts[index] - was)) if out else 0
        beat_id = out[shows].beat_id if shows < len(out) else ""
        # A card showing the end of its shot came after it — the closing card of a part — and goes back there.
        # Only for the shot it shows: `at` is a second of that file, and means nothing against another's.
        if found and abs(at - out[shows].end) < abs(at - out[shows].start):
            shows += 1
        out.insert(shows, from_clip(clip, beat_id))
    return out

def _card_operation(clip_id: str, piece: Piece) -> dict:
    """Write the operation that puts one title card on the sequence."""
    placed = {
        "action": "insert_clip", "track_id": VIDEO_TRACK_ID, "clip_id": clip_id, "asset_id": piece.asset_id,
        "source_range": {"start": 0.0, "end": round(piece.end, 3)}, "volume": 0.0,
        "card": piece.card.model_dump(mode="json"),
    }
    if piece.transition is not None:
        kind, seconds, through, direction = piece.transition
        placed["transition_in"] = {"kind": kind, "seconds": seconds}
        if kind == "dip":
            placed["transition_in"]["through"] = through
        elif kind == "wipe":
            placed["transition_in"]["direction"] = direction
    return placed

def _text_operations(
    plan: EditPlan,
    operations: Sequence[dict],
    provenance: Mapping[str, dict],
    project: Optional[Project],
    children: Mapping[str, List[SemanticClip]],
) -> List[dict]:
    """Write the words over a compiled cut: the plan's, and every other set kept from before.

    Words are pinned to footage, so they find their clips by it: the plan's
    over the windows its selections became, and words put on by hand — or the
    plan's, changed by hand since — over whichever new clip plays the footage
    they were over.

    Args:
        plan: The plan being compiled.
        operations: The compile's operations so far, which say what each new
            clip plays.
        provenance: What each new clip was made from, by its ID.
        project: What is on the timeline now.
        children: Each section's own clips, so a selection of a section
            finds the windows made from its parts.

    Returns:
        `add_text` operations.
    """
    made = [
        (op["track_id"], op["clip_id"], op["asset_id"], float(op["source_range"]["start"]),
         float(op["source_range"]["end"]), set(provenance.get(op["clip_id"], {}).get("from_clip_ids", ())))
        for op in operations
        if op["action"] in ("insert_clip", "add_clip") and op["track_id"] in (VIDEO_TRACK_ID, BROLL_TRACK_ID)
        and "card" not in op
    ]
    texts: List[dict] = []
    taken: set = set()

    def put(track_id: str, clip_id: str, text_id: str, start: float, end: float, fields: dict) -> None:
        name, number = text_id, 2
        while (clip_id, name) in taken:
            name, number = f"{text_id}-{number}", number + 1
        taken.add((clip_id, name))
        texts.append({"action": "add_text", "track_id": track_id, "clip_id": clip_id, "text_id": name,
                      "start": round(start, 3), "end": round(end, 3), **fields})

    # Kept first, so the plan's own words give way to a changed or taken-off copy of themselves.
    # Each once: a split clip's halves both carry the words of the whole, and a new clip
    # spanning both halves would otherwise get them twice.
    old = [clip for track_id, clip in compiled_clips(project) if track_id in (VIDEO_TRACK_ID, BROLL_TRACK_ID)]
    kept, carried = set(), set()
    for clip in old:
        for text in clip.texts:
            key = (clip.asset_id, text.id, text.start, text.end)
            if text.by_plan or key in carried:
                continue
            carried.add(key)
            # Named with the whole clip ID before they were named as the AI is shown them.
            kept.add(short_clip_ids(text.id))
            fields = {"text": text.text, "second": text.second, "style": text.style, "x": text.x, "y": text.y,
                      "hidden": text.hidden}
            for track_id, clip_id, asset_id, start, end, _ in made:
                low, high = max(start, float(text.start)), min(end, float(text.end))
                if asset_id == clip.asset_id and high > low:
                    put(track_id, clip_id, short_clip_ids(text.id), low, high, fields)

    for selection in plan.selections:
        if not selection.texts:
            continue
        own = {selection.clip_id} | {child.id for child in children.get(selection.clip_id, [])}
        windows = [entry for entry in made if entry[0] == VIDEO_TRACK_ID and entry[5] & own]
        for text in selection.texts:
            # Named by its words, not its place in the list: a copy changed or taken off by hand
            # stands in for exactly these words, and new words the plan is given come in beside it.
            words = hashlib.sha1(f"{text.text}\n{text.second or ''}".encode("utf-8")).hexdigest()[:8]
            # The clip as the AI is shown it: everything said back has the timeline taken off clip
            # IDs, so a text named with the whole one could never be named back.
            text_id = f"plan-{short_clip_ids(selection.clip_id)}-{words}"
            if text_id in kept:
                continue
            fields = {"text": text.text, "second": text.second, "style": text.style, "x": text.x, "y": text.y,
                      "by_plan": True}
            for track_id, clip_id, _, start, end, _ in windows:
                low = max(start, text.start if text.start is not None else start)
                high = min(end, text.end if text.end is not None else end)
                if high > low:
                    put(track_id, clip_id, text_id, low, high, fields)
    return texts

def compile_operations(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
    project: Optional[Project] = None,
    cuts: Optional[Mapping[str, CleanCuts]] = None,
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

    Returns:
        `(operations, provenance)` — operations for `apply_edits`, and, keyed
        by the clip ID each one creates, where it came from and whether it
        stays pinned. The provenance is stamped onto the clips after the
        operations are applied, so it never has to travel through the tool
        surface.
    """
    pieces, _ = as_left(plan_pieces(plan, clips, children, assets, cuts), project)
    # Keyed by track as well as by what it was made of. The three kinds of clip the
    # compiler owns mean different things by `from_clip_ids` — merged semantic clips on
    # the sequence, a (cover, covered) pair on the B-roll track, nothing at all for the
    # music — and one namespace for all three lets a sequence clip made of exactly those
    # two semantic clips take the B-roll's hand-made settings.
    kept = {
        (track_id, tuple(clip.from_clip_ids)): clip
        for track_id, clip in compiled_clips(project) if clip.pinned and clip.card is None
    }
    pieces = _with_cards(plan, pieces, project)
    if any(piece.card is not None for piece in pieces):
        # The beat was found for the cut without its cards; with them in, everything after a
        # card is that much later against the music, so the cuts are put on the beat again.
        held = {from_ids for track_id, from_ids in kept if track_id == VIDEO_TRACK_ID}
        pieces, _ = on_the_beat(plan, pieces, assets, cuts, held)

    operations: List[dict] = [
        # Everything the compiler owns comes down first, so the cut is rebuilt rather than
        # added to. Rippling is off: the whole track goes, so there is nothing to close up.
        {"action": "delete_clip", "track_id": track_id, "clip_id": clip.id, "ripple": False}
        for track_id, clip in compiled_clips(project)
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
        if piece.card is not None:
            operations.append(_card_operation(clip_id, piece))
            made_by, held = piece.card_from or (plan.id, False)
            provenance[clip_id] = {"from_plan_id": made_by, "from_clip_ids": [], "pinned": held}
            continue
        pinned = kept.pop((VIDEO_TRACK_ID, piece.from_clip_ids), None)
        placed = {
            "action": "insert_clip",
            "track_id": VIDEO_TRACK_ID,
            "clip_id": clip_id,
            "asset_id": piece.asset_id,
            "source_range": {"start": piece.start, "end": piece.end},
        }
        level = piece.gain
        if level != 1.0:
            placed["volume"] = level
        if piece.speed != 1.0:
            placed["speed"] = piece.speed
        if piece.keep_level:
            placed["keep_level"] = True
        if piece.transition is not None:
            kind, seconds, through, direction = piece.transition
            placed["transition_in"] = {"kind": kind, "seconds": seconds}
            if kind == "dip":
                placed["transition_in"]["through"] = through
            elif kind == "wipe":
                placed["transition_in"]["direction"] = direction
        if piece.sound_lead:
            placed["audio_lead"] = piece.sound_lead
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

    operations += _text_operations(plan, operations, provenance, project, children)

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
