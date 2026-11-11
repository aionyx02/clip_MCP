"""Sound effects: the moments a cut could take one, and laying the ones a plan picked.

Where an effect could go is found by rule: every cut on the sequence, every
title card, words coming up over the footage, covering picture cutting in,
and the closing call to action. All of them are offered, deliberately too
many, and none is ranked: which moments get an effect, and which effect, is
the model's choice and the user's. The picks are pinned to a fingerprint of
the moments they were chosen from, so a pick made against a cut that has
changed since is handed back rather than placed somewhere near.

A moment is named by what makes it, not by when it falls: the cut into the
clip made of u0012, the card that opens a beat, the words by their own ID. A
retrim earlier in the cut moves a moment without changing it, and its effect
follows; a moment added or taken away changes the fingerprint.

The level is the mix's: an effect is brought to the level of the talking the
way a song is (`levels.talking_in`), then sits `SFX_UNDER_TALK_DB` under it
at volume 1. It never ducks: it is a moment meant to be heard, not a bed.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from app.engine.library import SFX, library_of
from app.engine.plan.pieces import SFX_B_TRACK_ID, SFX_TRACK_ID
from app.models.media import Asset, MediaAnalysis
from app.models.plan import SoundEffects, short_clip_ids
from app.models.timeline.parts import Clip, Project, TrackType

# Bumped whenever the rules below change in a way the moments themselves do not show.
MOMENT_VERSION = 1
# How far under the talking an effect at volume 1 sits, in dB, once it has been brought to
# the talking's level. Under the words, so a whoosh on a cut lands without covering what is
# said across it. Provisional: a starting point, not measured on any cut.
SFX_UNDER_TALK_DB = 6.0
# The tracks effects are laid on, in the order they are filled.
SFX_TRACKS = (SFX_TRACK_ID, SFX_B_TRACK_ID)
# Closer than this, two times are the same moment. One frame at 1000 fps.
_SAME = 0.001
_SAME_D = Decimal("0.001")
# The kinds of moment, in the order they are listed when two fall together.
KINDS = ("cut", "title_card", "words", "broll", "cta")


@dataclass(frozen=True)
class Moment:
    """One place in a cut a sound effect could go.

    Attributes:
        id: What it is called, made from what makes it, so it survives a
            retrim elsewhere.
        kind: `cut`, `title_card`, `words`, `broll` or `cta`.
        at: Where it falls, in seconds of the cut.
        marks: What else is true of it: a cut that comes in on a
            `transition`, changes `speed`, follows a `gap`, or opens a
            `card`; covering picture that is an `inset`.
        what: It in words, for reading back.
    """

    id: str
    kind: str
    at: float
    marks: Tuple[str, ...]
    what: str

    def record(self) -> dict:
        """The moment as a tool hands it back."""
        return {"id": self.id, "kind": self.kind, "at": round(self.at, 3), "marks": list(self.marks),
                "what": self.what}


def _made_of(clip: Clip) -> str:
    """What a clip is made of, as a moment is named by it."""
    if clip.card is not None:
        if clip.card.end:
            return "end-card"
        if clip.card.beat_id:
            return f"card-{clip.card.beat_id}"
        return "card-" + hashlib.sha1(clip.card.title.encode("utf-8")).hexdigest()[:6]
    if clip.from_clip_ids:
        return "+".join(short_clip_ids(clip_id) for clip_id in clip.from_clip_ids)
    return f"{clip.asset_id[:8]}@{float(clip.source_range.start):.2f}"


def _said(clip: Clip) -> str:
    """A clip in a few words."""
    if clip.card is not None:
        return f"the {'closing' if clip.card.end else 'title'} card 「{clip.card.title}」"
    return f"clip {clip.id} ({_made_of(clip)})"


def effect_moments(project: Project) -> List[Moment]:
    """Every moment of a cut a sound effect could go at, by rule and on purpose too many.

    Args:
        project: The compiled cut.

    Returns:
        The moments in the order they play: every cut on the sequence, every
        card, the first second of every set of words over the footage (the
        closing words of a brand kit as `cta`), and every clip of covering
        picture. The start of the video is not a cut.
    """
    taken: Set[str] = set()

    def name(kind: str, made: str) -> str:
        """A moment's name, numbered when two are made of the same thing."""
        named, number = f"{kind}:{made}", 2
        while named in taken:
            named, number = f"{kind}:{made}#{number}", number + 1
        taken.add(named)
        return named

    moments: List[Moment] = []
    base = project.base_video_track
    shots = sorted(base.clips, key=lambda clip: clip.timeline_in) if base else []
    for before, after in zip(shots, shots[1:]):
        marks = []
        if after.transition_in is not None:
            marks.append("transition")
        if after.speed != before.speed:
            marks.append("speed")
        if float(after.timeline_in) > float(before.timeline_out) + _SAME:
            marks.append("gap")
        if after.card is not None or before.card is not None:
            marks.append("card")
        joined = f" on a {after.transition_in.kind}" if after.transition_in is not None else ""
        moments.append(Moment(name("cut", _made_of(after)), "cut", float(after.timeline_in), tuple(marks),
                              f"the cut into {_said(after)}{joined}"))
    for clip in shots:
        if clip.card is not None:
            kind = "cta" if clip.card.end else "title_card"
            moments.append(Moment(name(kind, _made_of(clip)), kind, float(clip.timeline_in), (), _said(clip)))
    seen: Set[str] = set()
    for track in project.video_tracks:
        for clip in sorted(track.clips, key=lambda item: item.timeline_in):
            for text in clip.shown_texts:
                # Words carried on from the clip before, as when a pause was taken out under them,
                # are not coming up here: they came up where they were first shown.
                if text.id in seen or not clip.source_range.start - _SAME_D <= text.start < clip.source_range.end:
                    continue
                seen.add(text.id)
                at = float(clip.timeline_in) + float(text.start - clip.source_range.start) / clip.speed
                kind = "cta" if text.style == "cta" else "words"
                moments.append(Moment(name(kind, short_clip_ids(text.id)), kind, at, (),
                                      f"「{text.text}」 coming up over {_said(clip)}"))
    for track in project.video_tracks[1:]:
        for clip in sorted(track.clips, key=lambda item: item.timeline_in):
            marks = ("inset",) if clip.layout is not None else ()
            moments.append(Moment(name("broll", _made_of(clip)), "broll", float(clip.timeline_in), marks,
                                  f"{_said(clip)} cutting in on track {track.id}"))
    return sorted(moments, key=lambda moment: (round(moment.at, 3), KINDS.index(moment.kind), moment.id))


def moments_hash(moments: Sequence[Moment]) -> str:
    """Fingerprint what the moments are, not when they fall, so a pick can tell they have changed.

    Args:
        moments: The moments offered.

    Returns:
        Twelve hex characters.
    """
    named = sorted([moment.id, moment.kind, list(moment.marks)] for moment in moments)
    blob = json.dumps({"version": MOMENT_VERSION, "moments": named}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def effect_volume(volume: float) -> float:
    """The clip volume an effect is laid at: the pick's volume, `SFX_UNDER_TALK_DB` under the talking."""
    return round(volume * 10 ** (-SFX_UNDER_TALK_DB / 20), 4)


def effect_operations(
    chosen: Optional[SoundEffects],
    moments: Sequence[Moment],
    assets: Mapping[str, Asset],
    present: Set[str],
    plan_id: str,
    kept: Optional[Mapping[Tuple[str, ...], Clip]] = None,
) -> Tuple[List[dict], Dict[str, dict], List[str]]:
    """Lay a plan's sound effects at their moments, or say why they cannot be laid.

    Nothing is moved to make a pick fit: a pick that names a moment not
    offered, a file that is not a sound effect, or moments that have changed
    since it was made, comes back as a reason.

    Args:
        chosen: The plan's sound effects, or None.
        moments: The moments of the cut as compiled now.
        assets: The effects' files, keyed by asset ID.
        present: The tracks the project has already.
        plan_id: The plan, for where each clip came from.
        kept: Effect clips adjusted by hand, keyed by what they were made of:
            each comes back with its own range, level and fades.

    Returns:
        `(operations, provenance, problems)`. When there are problems the
        operations are empty.
    """
    if chosen is None or not chosen.picks:
        return [], {}, []
    problems: List[str] = []
    current = moments_hash(moments)
    by_id = {moment.id: moment for moment in moments}
    if chosen.candidate_hash != current:
        gone = sorted({pick.candidate for pick in chosen.picks} - set(by_id))
        problems.append(
            f"the cut's moments have changed since these sound effects were picked (hash {chosen.candidate_hash}, "
            f"now {current})" + (f"; {', '.join(gone)} is no longer one of them" if gone else "")
            + ". Call propose_sound_effects on the compiled cut and pick again from what it lists"
        )
        return [], {}, problems
    laid: List[Tuple[float, float, int, Moment, Asset, Optional[Clip]]] = []
    for number, pick in enumerate(chosen.picks, start=1):
        where = f"sound effect {number} ({pick.candidate})"
        moment = by_id.get(pick.candidate)
        if moment is None:
            problems.append(f"{where}: not one of the moments propose_sound_effects offered; pick one of its `id`s")
        asset = assets.get(pick.asset_id)
        if asset is None:
            problems.append(f"{where}: no file {pick.asset_id} in the library")
            continue
        library = library_of(asset)
        if library != SFX:
            problems.append(f"{where}: {os.path.basename(asset.path)} is in the {library} "
                            "library, not sound effects; move it with set_asset_library if it is one")
            continue
        if not asset.has_audio or not asset.duration:
            problems.append(f"{where}: {asset.id} has no sound to play")
            continue
        if moment is None:
            continue
        if pick.lead > moment.at + _SAME:
            problems.append(f"{where}: starts {pick.lead:g}s before a moment at {moment.at:.2f}s, which is before "
                            "the video begins; take the lead down")
            continue
        pinned = (kept or {}).get((pick.candidate, pick.asset_id))
        length = float(pinned.timeline_duration) if pinned is not None else float(asset.duration)
        start = max(0.0, moment.at - pick.lead)
        laid.append((start, start + length, number, moment, asset, pinned))
    if problems:
        return [], {}, problems

    laid.sort(key=lambda item: (item[0], item[2]))
    ends = [float("-inf")] * len(SFX_TRACKS)
    lanes: List[str] = []
    for start, end, number, moment, _, _ in laid:
        lane = next((index for index, free in enumerate(ends) if free <= start + _SAME), None)
        if lane is None:
            ringing = [str(other) for other_start, other_end, other, _, _, _ in laid
                       if other != number and other_start <= start + _SAME < other_end]
            problems.append(f"sound effect {number} ({moment.id}) starts at {start:.2f}s while effects "
                            f"{' and '.join(ringing)} still play; at most {len(SFX_TRACKS)} play at once — "
                            "leave one out, or pick a shorter one")
            continue
        ends[lane] = end
        lanes.append(SFX_TRACKS[lane])
    if problems:
        return [], {}, problems

    operations: List[dict] = []
    for track_id in dict.fromkeys(lanes):
        if track_id not in present:
            operations.append({"action": "add_track", "track_id": track_id, "track_type": "audio",
                               "duck_under_speech": False, "voice": False})
        else:
            operations.append({"action": "set_track_audio", "track_id": track_id, "duck_under_speech": False,
                               "voice": False})
    provenance: Dict[str, dict] = {}
    for position, ((start, _, number, moment, asset, pinned), track_id) in enumerate(zip(laid, lanes), start=1):
        pick = chosen.picks[number - 1]
        clip_id = f"s{position:03d}"
        placed = {
            "action": "add_clip", "track_id": track_id, "clip_id": clip_id, "asset_id": asset.id,
            "source_range": {"start": 0.0, "end": float(asset.duration)},
            "timeline_in": round(start, 3), "volume": effect_volume(pick.volume),
        }
        if pinned is not None:
            from app.engine.plan.compile import _as_made

            placed.update(_as_made(pinned))
        operations.append(placed)
        provenance[clip_id] = {"from_plan_id": plan_id, "from_clip_ids": [moment.id, asset.id],
                               "pinned": pinned is not None}
    return operations, provenance, []


def effects_on_words(project: Project, analyses: Mapping[str, MediaAnalysis],
                     assets: Mapping[str, Asset]) -> List[dict]:
    """Find the sound effects that play over somebody's words.

    An effect on a word can be meant — a pop on the word it punctuates — so
    this is said, never refused.

    Args:
        project: The cut.
        analyses: Its files' analyses, keyed by asset ID.
        assets: Its files, keyed by asset ID.

    Returns:
        One entry per effect over words: its `clip`, the `effect` file, `at`
        and `until` in seconds of the cut, and the `words` it covers.
    """
    from app.engine.delivery import _on_timeline

    spoken: List[Tuple[float, float, str]] = []
    base = project.base_video_track
    for clip in base.clips if base else []:
        analysis = analyses.get(clip.asset_id)
        if clip.volume == 0 or clip.card is not None or analysis is None or analysis.transcript is None:
            continue
        for segment in analysis.transcript.segments:
            for word in segment.words:
                landed = _on_timeline(clip, word.start, word.end)
                if landed is not None:
                    spoken.append((landed[0], landed[1], word.text))
    found = []
    for track in project.tracks:
        if track.track_type != TrackType.AUDIO:
            continue
        for clip in track.clips:
            asset = assets.get(clip.asset_id)
            if asset is None or library_of(asset) != SFX or clip.volume == 0:
                continue
            opened, closed = float(clip.timeline_in), float(clip.timeline_out)
            covered = [text for start, end, text in spoken if start < closed and end > opened]
            if covered:
                found.append({"clip": clip.id, "effect": os.path.basename(asset.path),
                              "at": round(opened, 3), "until": round(closed, 3), "words": "".join(covered).strip()})
    return found
