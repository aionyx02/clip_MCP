"""Turning a plan into the edit operations that build its cut."""

import hashlib
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine.plan.beats import plan_pieces
from app.engine.plan.broll import broll_covers
from app.engine.plan.checks import as_left, compiled_clips, with_cards
from app.engine.plan.music import music_beds, plan_markers
from app.engine.plan.pieces import BROLL_TRACK_ID, MUSIC_TRACK_ID, VIDEO_TRACK_ID, Piece
from app.engine.semantic import CleanCuts
from app.models.brand import BrandKit
from app.models.media import Asset
from app.models.plan import EditPlan, short_clip_ids
from app.models.semantic import SemanticClip
from app.models.timeline.parts import CaptionStyle, Clip, Project

# The words a brand kit closes on over the last shot, by the one name a recompile knows them by.
CLOSING_TEXT_ID = "brand-end"


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
        "stabilize": pinned.stabilize,
    }

def _plain(value: Any) -> Any:
    """Put a value into the one shape a plan's ask and a clip's setting are compared in."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (Decimal, float)):
        return round(float(value), 3)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value

def _transition(piece: Piece) -> Optional[dict]:
    """Write how a window comes in over the one before, as the operation that places it takes it."""
    if piece.transition is None:
        return None
    kind, seconds, through, direction = piece.transition
    made = {"kind": kind, "seconds": seconds}
    if kind == "dip":
        made["through"] = through
    elif kind == "wipe":
        made["direction"] = direction
    return made

def _piece_asks(piece: Piece, raw: Piece) -> dict:
    """Write down what the plan asks of one shot in the sequence.

    Args:
        piece: The window as the compile places it.
        raw: The same window as the plan alone makes it, before a pinned
            clip's range is laid over it or a cut is moved onto the beat:
            the range has to come from the plan, or a hand trim would read as
            the plan changing.

    Returns:
        The fields a hand adjustment can also hold, in `_plain` shape.
    """
    return _plain({
        "source_range": {"start": raw.start, "end": raw.end},
        "speed": raw.speed,
        "volume": piece.gain,
        "keep_level": piece.keep_level,
        "stabilize": piece.stabilize,
        "motion": piece.motion,
        "transition_in": _transition(piece),
        "audio_lead": piece.sound_lead or 0.0,
    })

def _clip_holds(clip: Clip) -> dict:
    """Read the same fields off a clip as it stands, in `_plain` shape."""
    held = clip.transition_in
    return _plain({
        "source_range": {"start": clip.source_range.start, "end": clip.source_range.end},
        "speed": clip.speed,
        "volume": clip.volume,
        "keep_level": clip.keep_level,
        "stabilize": clip.stabilize,
        "motion": clip.motion,
        "transition_in": None if held is None else held.model_dump(),
        "audio_lead": clip.audio_lead,
        "layout": clip.layout.model_dump() if clip.layout else None,
    })

def _overridden(pinned: Clip, asked: dict) -> List[Tuple[str, Any, Any]]:
    """Find what the plan changed about a pinned clip that keeping it would throw away.

    A pinned clip is meant to differ from the plan — that is what it is
    pinned for — so its own settings are not the comparison. What is compared
    is what the plan asked of this clip last time against what it asks now: a
    field only the plan changed is a change the person made to the plan and
    expects to see. It is reported only when the clip does not already have
    what the plan now asks.

    Args:
        pinned: The pinned clip, carrying the plan's asks from the compile
            that made it. One compiled before those were recorded carries
            none, and nothing can be said about it.
        asked: What the plan asks of the same clip now.

    Returns:
        `(field, asked before, asked now)` for each such field.
    """
    before = pinned.plan_asked
    if not before:
        return []
    holds = _clip_holds(pinned)
    return [
        (field, before[field], now)
        for field, now in asked.items()
        if field in before and before[field] != now and holds.get(field) != now
    ]

def _say(field: str, value: Any) -> str:
    """Say one asked-for setting in words."""
    if field == "layout":
        if not value:
            return "the whole frame"
        return " ".join(str(part) for part in (value.get("place"), value.get("size")) if part)
    if field == "source_range":
        return f"{value['start']:.2f}s-{value['end']:.2f}s of its file"
    if field == "transition_in" and not value:
        return "a straight cut"
    if field == "motion" and value is None:
        return "no motion"
    if isinstance(value, dict):
        return ", ".join(f"{key} {item}" for key, item in value.items())
    return str(value)

def pinned_differences(provenance: Mapping[str, dict]) -> List[str]:
    """Say which pinned clips the plan has changed since they were pinned, and how.

    A recompile keeps a pinned clip exactly as it is, so a change the plan
    makes to that same clip would be dropped without a word. `compile_plan`
    stops on these instead of choosing between the hand adjustment and the
    plan for the person.

    Args:
        provenance: What `compile_operations` returned.

    Returns:
        One line per pinned clip that differs, naming every field.
    """
    lines = []
    for origin in provenance.values():
        changed = origin.get("overridden")
        if not changed:
            continue
        what = "; ".join(
            f"{field} was {_say(field, was)}, the plan now asks for {_say(field, now)}"
            for field, was, now in changed
        )
        lines.append(
            f"clip {origin['kept_clip_id']} on track {origin['track_id']} is pinned to keep a hand adjustment, "
            f"and the plan has changed it since ({what}); compiling would keep the clip as it is and drop that. "
            "Unpin it with set_clip_pinned to take the plan's version, which gives up the hand adjustment, "
            "or change the plan back to keep the clip as it is"
        )
    return lines

def _kept(provenance: Dict[str, dict], clip_id: str, track_id: str, pinned: Optional[Clip], asked: dict) -> None:
    """Record a compiled clip's asks, and, for one kept from a pin, what keeping it drops."""
    origin = provenance[clip_id]
    origin["asked"] = asked
    if pinned is not None:
        origin.update(track_id=track_id, kept_clip_id=pinned.id, overridden=_overridden(pinned, asked))

def _card_operation(clip_id: str, piece: Piece) -> dict:
    """Write the operation that puts one title card on the sequence."""
    placed = {
        "action": "insert_clip", "track_id": VIDEO_TRACK_ID, "clip_id": clip_id, "asset_id": piece.asset_id,
        "source_range": {"start": 0.0, "end": round(piece.end, 3)}, "volume": 0.0,
        "card": piece.card.model_dump(mode="json"),
    }
    if piece.transition is not None:
        placed["transition_in"] = _transition(piece)
    return placed

def _text_operations(
    plan: EditPlan,
    operations: Sequence[dict],
    provenance: Mapping[str, dict],
    project: Optional[Project],
    children: Mapping[str, List[SemanticClip]],
    kit: Optional[BrandKit] = None,
) -> List[dict]:
    """Write the words over a compiled cut: the plan's, its brand kit's, and every other set kept from before.

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
        kit: The brand kit the plan wears, if any: words it closes on go
            over the last seconds of the last shot, unless a copy of them
            changed or taken off by hand is kept.

    Returns:
        `add_text` operations.

    Raises:
        ValueError: If the last shot plays for less time than the kit's
            closing words stay up; `check_plan` says so first, but a shot
            adjusted by hand can come out shorter than the plan's.
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
                      "hidden": text.hidden, "colour": text.colour, "second_colour": text.second_colour}
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

    closing = kit.end_card if kit is not None else None
    shots = [op for op in operations if op["action"] == "insert_clip" and op["track_id"] == VIDEO_TRACK_ID
             and "card" not in op]
    if closing is not None and closing.mode == "overlay" and CLOSING_TEXT_ID not in kept and shots:
        last = shots[-1]
        start, end = float(last["source_range"]["start"]), float(last["source_range"]["end"])
        # Seconds of the file, so as many more as the shot is sped up.
        reach = closing.held * float(last.get("speed", 1.0))
        if end - reach < start - 1e-6:
            raise ValueError(
                f"brand kit 「{kit.name}」 closes on words over the last {closing.held:g}s of the last shot, which "
                f"plays only {(end - start) / float(last.get('speed', 1.0)):.2f}s"
            )
        put(VIDEO_TRACK_ID, last["clip_id"], CLOSING_TEXT_ID, max(start, end - reach), end, {
            "text": closing.title, "second": closing.subtitle, "style": "cta", "by_plan": True,
            "colour": kit.accent or kit.text_colour, "second_colour": kit.text_colour,
        })
    return texts

def _brand_operations(kit: Optional[BrandKit], project: Optional[Project]) -> List[dict]:
    """Write the brand kit's corner logo, font and words' colour onto the cut, or take a kit's logo off.

    Args:
        kit: The brand kit the plan names, or None.
        project: What is on the timeline now, if anything.

    Returns:
        `set_logo` and `set_caption_style` operations, as needed. The corner
        logo only when the kit asks for one; a kit's logo already there goes
        when it no longer does, or when there is no kit.
    """
    operations: List[dict] = []
    had = project.logo if project is not None else None
    corner = kit.corner_logo_asset_id if kit is not None and kit.corner_logo else None
    if corner is not None:
        operations.append({"action": "set_logo", "logo": {
            "asset_id": corner, "corner": kit.logo_corner, "opacity": kit.logo_opacity,
            "width": kit.logo_width, "brand_kit": kit.name,
        }})
    elif had is not None and had.brand_kit is not None:
        operations.append({"action": "set_logo", "logo": None})
    style = project.caption_style if project is not None else CaptionStyle()
    changed = {}
    if kit is not None and kit.font and style.font != kit.font:
        changed["font"] = kit.font
    if kit is not None and style.colour != kit.text_colour:
        changed["colour"] = kit.text_colour
    if changed:
        operations.append({"action": "set_caption_style", "style": changed})
    return operations

def compile_operations(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
    project: Optional[Project] = None,
    cuts: Optional[Mapping[str, CleanCuts]] = None,
    kit: Optional[BrandKit] = None,
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
        kit: The brand kit the plan names, if any: its closing words over
            the end of the last shot or its closing card after it, its logo
            in a corner when it asks for one, its font and colours on the
            words. A logo a kit put there goes when the plan stops naming
            one; one set by hand stays.

    Returns:
        `(operations, provenance)` — operations for `apply_edits`, and, keyed
        by the clip ID each one creates, where it came from and whether it
        stays pinned. The provenance is stamped onto the clips after the
        operations are applied, so it never has to travel through the tool
        surface.
    """
    raw = plan_pieces(plan, clips, children, assets, cuts)
    # What the plan alone makes of each window, by what it is made of. The first one wins,
    # as it does when a pinned clip is matched: two windows can share their clip IDs.
    as_planned: Dict[Tuple[str, ...], Piece] = {}
    for piece in raw:
        as_planned.setdefault(piece.from_clip_ids, piece)
    pieces, _ = as_left(raw, project)
    # Keyed by track as well as by what it was made of. The three kinds of clip the
    # compiler owns mean different things by `from_clip_ids` — merged semantic clips on
    # the sequence, a (cover, covered) pair on the B-roll track, nothing at all for the
    # music — and one namespace for all three lets a sequence clip made of exactly those
    # two semantic clips take the B-roll's hand-made settings.
    kept = {
        (track_id, tuple(clip.from_clip_ids)): clip
        for track_id, clip in compiled_clips(project) if clip.pinned and clip.card is None
    }
    pieces = with_cards(plan, pieces, assets, cuts, project, kit)

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
        if piece.stabilize:
            placed["stabilize"] = True
        if piece.motion is not None:
            placed["motion"] = piece.motion
        if piece.transition is not None:
            placed["transition_in"] = _transition(piece)
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
        asked = _piece_asks(piece, as_planned.get(piece.from_clip_ids, piece))
        _kept(provenance, clip_id, VIDEO_TRACK_ID, pinned, asked)

    # B-roll goes on a track of its own above the sequence, at absolute positions rather
    # than in a row: a cover sits where the words it belongs to are, and the gaps between
    # covers are where the sequence shows through. Silent, because the point is the sound
    # underneath carrying on — unless a picture in a corner or a half asked to be heard.
    covers, refused, _ = broll_covers(plan, pieces, clips, assets, cuts)
    if refused:
        # `check_plan` reports these, and `compile_plan` stops on them — but the engine
        # is meant to stand on its own, and a compiler that silently left a shot out of
        # the cut would be repairing a plan instead of handing it back.
        raise ValueError("this plan's B-roll does not hold up: " + "; ".join(refused))
    if covers:
        # A picture whose own sound was turned on drops under the talking; nothing else on the track is heard.
        ducked = any(cover.sound for cover in covers)
        if BROLL_TRACK_ID not in present:
            operations.append({"action": "add_track", "track_id": BROLL_TRACK_ID, "track_type": "video",
                               "duck_under_speech": ducked})
        else:
            operations.append({"action": "set_track_audio", "track_id": BROLL_TRACK_ID, "duck_under_speech": ducked})
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
                "volume": 1.0 if cover.sound else 0.0,
                "layout": cover.layout.model_dump() if cover.layout is not None else None,
            }
            if pinned is not None:
                laid.update(_as_made(pinned))
            operations.append(laid)
            provenance[clip_id] = {
                "from_plan_id": plan.id,
                "from_clip_ids": list(identity),
                "pinned": pinned is not None,
            }
            asked = _plain({
                "source_range": {"start": cover.start, "end": cover.end},
                "volume": laid["volume"],
                "layout": cover.layout.model_dump() if cover.layout is not None else None,
            })
            _kept(provenance, clip_id, BROLL_TRACK_ID, pinned, asked)

    operations += _text_operations(plan, operations, provenance, project, children, kit)
    operations += _brand_operations(kit, project)

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
            # Only its level: where a stretch starts and ends in the song follows the cut, so a
            # shot trimmed by hand would read as the plan moving the music.
            _kept(provenance, clip_id, bed.track_id, pinned, _plain({"volume": bed.volume}))
    return operations, provenance
