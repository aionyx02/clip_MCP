"""What could cover each stretch: B-roll candidates found by rule, and the picks pinned to them.

`propose_broll` says where a cut wants covering picture; this says with what.
An editor choosing a cutaway asks the same few things of every shot: does it
show what is being said right then, is it long enough, is it steady and lit,
is it already on screen nearby, is it the person talking, does anything move,
and did the user bring it for this video. The ones that are arithmetic are
answered here. Whether a shot belongs over these words is not: that is the
model's to judge and the user's to overrule, so the rules over-produce
candidates, show each with the facts it was offered on, and leave the choice.

Two rules leave a shot out, because no editor would cut to it: the same
camera as the picture it would cover (cutting to it is a jump, not a
cutaway), and somebody talking to camera, which is what B-roll is laid over
in the first place. A shot with no picture, black or frozen, or too short to
be a shot is not one either. Everything else is offered and described.

The order is what is said against what each shot is known to show — its
description, its tags, its file's name and notes — by the
meaning search `query_clips` uses for `about`. That is an order, not a score
that decides: the facts are beside every candidate, and nothing is blended.
A shot nothing describes cannot be placed by meaning, so a few of those are
offered too, the ones recorded nearest the talk they would cover, since a
shot taken minutes before is usually the same place.

The picks are pinned. Each offer is kept with what it was made from — the
words under each stretch, the files on screen there, and what every clip of
the timeline is known to show — and a pick names its offer by its
`candidate_hash`. A pick that was not offered, or one whose stretch or
footage has changed since, is handed back with the reason.
"""

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine.delivery import LIGHT_STEP, REPEAT_SECONDS, clock
from app.engine.plan.broll import BROLL_MIN_SECONDS, Stretch, _placed, stretch_over
from app.engine.plan.checks import SHAKY
from app.engine.plan.pieces import Piece
from app.engine.semantic import CleanCuts
from app.models.media import Asset
from app.models.plan import EditPlan
from app.models.semantic import ClipKind, ClipLevel, SemanticClip, SemanticTimeline

# Bumped whenever the rules below change in a way the constants do not show.
CANDIDATE_VERSION = 1
# How many shots are offered for one stretch. Enough that the right one is usually among
# them and the model has something to turn down, few enough to read. Provisional.
CANDIDATES_PER_STRETCH = 8
# Of those, how many are kept for shots nothing describes, so footage nobody labelled is
# still put in front of the model. Provisional.
UNREAD_PER_STRETCH = 2
# At most this many from one file, so one long take does not fill the list. Provisional.
PER_FILE = 2
# A clip of speech with at least this many faces up, on average, is somebody talking to
# camera: what B-roll covers, not what covers it. Provisional, against the faces score.
TALKING_FACES = 0.5
# Less frame-to-frame change than this and the shot is all but a still. Provisional:
# below the quietest shot of the footage this was looked at on, not tuned on viewers.
STILL_MOTION = 0.005
# A file on screen this close to a stretch, before or after it, is still in the viewer's
# eye. Provisional.
NEARBY_SECONDS = 10.0
# A candidate this high in another stretch's list is that stretch's likely pick too, and
# is said so, for variety across the cut. Provisional.
TOP_ELSEWHERE = 3
# Files in the library but not this timeline offered per stretch, as a second tier.
# Provisional.
LIBRARY_PER_STRETCH = 3
# Two words this common say nothing about what is on screen. Provisional, and short on
# purpose: it only has to keep filler out of `matched`.
_FILLER = frozenset((
    "這個", "那個", "一下", "可以", "就是", "然後", "我們", "現在", "因為", "所以", "還是", "什麼",
    "怎麼", "這樣", "那樣", "一個", "沒有", "覺得", "他們", "你們", "大家", "知道", "已經", "真的",
    "有點", "這裡", "那裡", "the", "and", "this", "that", "with",
))
# A file named by the camera says nothing about what it shows.
_CAMERA_NAME = re.compile(r"^(img|dji|dsc|dscf|mvi|gopr|gx|gh|vid|video|pxl|mov|movie|clip|mah|sam|rec|record|screen|"
                          r"screenshot|p|c)?[\W_]*\d[\d\W_]*[a-z]?$", re.IGNORECASE)
_WORDISH = re.compile(r"[\w]", re.UNICODE)


def file_label(asset: Asset) -> str:
    """What a file's name says it is, or "" when the camera named it.

    The number a folder of footage is put in order by is taken off the front:
    `13-機構測試4.mp4` reads `機構測試4`.
    """
    stem = os.path.splitext(os.path.basename(asset.path))[0]
    if _CAMERA_NAME.match(stem):
        return ""
    return re.sub(r"^\d+[\s_\-.]+", "", stem).strip()


def known_as(clip: SemanticClip, asset: Asset) -> List[Tuple[str, str]]:
    """What a clip is known to show, part by part, each with where it was read from.

    Args:
        clip: The clip.
        asset: The file it plays from.

    Returns:
        `(source, words)` pairs: its description, tags, file name, file notes,
        and section and topic. Empty when nothing says. What is said in it
        is not among them: words spoken in a shot say what was talked about
        there, not what the picture shows, and matching one talk's words
        against another's offers whatever was said alike — 「等一下」, 「掰掰」.
    """
    parts = [
        ("description", clip.description or ""),
        ("tags", "、".join(tag.value for tag in clip.tags)),
        ("file name", file_label(asset)),
        ("file notes", asset.notes or ""),
        ("section", " ".join(part for part in (clip.name, clip.topic) if part)),
    ]
    return [(source, words.strip()) for source, words in parts if words and words.strip()]


def reading(clip: SemanticClip, asset: Asset) -> str:
    """A clip's known parts as one text, for the meaning search."""
    return "\n".join(words for _, words in known_as(clip, asset))


def matched_words(said: str, text: str) -> List[str]:
    """The words a candidate's text shares with what is said, longest first.

    Runs of at least two characters of Chinese, or three of a Latin word, that
    appear in both, less filler.
    """
    said, text = said.lower(), text.lower()
    found: List[str] = []
    index = 0
    while index < len(text):
        length = 0
        while index + length < len(text) and _WORDISH.match(text[index + length]) \
                and text[index:index + length + 1] in said:
            length += 1
        run = text[index:index + length]
        latin = run.isascii()
        if length >= (3 if latin else 2) and not run.isdigit() and run not in _FILLER:
            found.append(run)
        index += max(length, 1)
    kept = [run for run in dict.fromkeys(found) if not any(run != other and run in other for other in found)]
    return sorted(kept, key=lambda run: -len(run))


def pool_signature(clips: Sequence[SemanticClip], assets: Mapping[str, Asset]) -> str:
    """Fingerprint what every clip that could cover something is known to show and measures.

    Retagging a clip, describing it, or writing a note on its file changes
    which shots are offered, so a pick made before is out of date.
    """
    rows = sorted(
        [clip.id, reading(clip, assets[clip.asset_id]), round(clip.duration, 3), clip.kind.value,
         round(clip.scores.get("faces", 0.0), 2)]
        for clip in clips if clip.asset_id in assets
    )
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def stretch_inputs(stretch: Stretch, timeline: SemanticTimeline, pool: str) -> Dict[str, str]:
    """What one stretch's candidates were worked out from, part by part, so a change can be named."""
    return {
        "version": str(CANDIDATE_VERSION),
        "timeline": timeline.input_hash,
        "said": hashlib.sha256(stretch.said.encode("utf-8")).hexdigest()[:12],
        "under": ",".join(sorted(stretch.under)),
        "footage": pool,
    }


_CHANGED = {
    "version": "the rules that offer candidates",
    "timeline": "the footage's analysis",
    "said": "the words under it",
    "under": "the picture under it",
    "footage": "what the footage's clips are known to show (descriptions, tags, file notes)",
}


def candidate_hash(offered: Mapping[str, Mapping]) -> str:
    """Fingerprint an offer: every stretch, what it was made from, and what was offered over it.

    Args:
        offered: Per stretch, by the clip it starts over, its `inputs` and `candidates`.

    Returns:
        Twelve hex characters.
    """
    blob = json.dumps({over: [dict(entry["inputs"]), list(entry["candidates"])] for over, entry in offered.items()},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def eligible(clips: Sequence[SemanticClip], assets: Mapping[str, Asset]) -> List[SemanticClip]:
    """The timeline's clips that could cover anything at all, before any stretch is looked at."""
    return [clip for clip in clips if clip.level == ClipLevel.UTTERANCE and clip.asset_id in assets]


def _left_out(clip: SemanticClip, asset: Asset, stretch: Stretch) -> Optional[str]:
    """Why a clip is no cutaway for a stretch, or None when it is one."""
    if not asset.has_video:
        return "no picture"
    if clip.kind == ClipKind.UNUSABLE:
        return "black or frozen"
    if clip.duration < BROLL_MIN_SECONDS:
        return f"shorter than {BROLL_MIN_SECONDS:g}s"
    if clip.kind == ClipKind.SPEECH and clip.scores.get("faces", 0.0) >= TALKING_FACES:
        return "somebody talking to camera"
    if clip.asset_id in stretch.under:
        return "the same camera as the picture it would cover"
    return None


def _apart(first: Optional[datetime], second: Optional[datetime]) -> Optional[float]:
    """Seconds from one recording to another, signed, or None when either time is unknown."""
    if first is None or second is None:
        return None
    try:
        return (first - second).total_seconds()
    except TypeError:
        return None


def _when(seconds: float) -> str:
    """A gap between two recordings in words."""
    size = abs(seconds)
    said = f"{size / 3600:.1f}h" if size >= 3600 else f"{size / 60:.0f}m" if size >= 60 else f"{size:.0f}s"
    return f"recorded {said} {'after' if seconds > 0 else 'before'} the talk under it"


def offer_candidates(
    stretch: Stretch,
    plan: EditPlan,
    pieces: Sequence[Piece],
    clips: Sequence[SemanticClip],
    assets: Mapping[str, Asset],
    meaning: Optional[Mapping[str, float]],
    talk_recorded: Optional[datetime],
) -> Tuple[List[dict], Dict[str, int], Optional[dict]]:
    """Offer the shots that could cover one stretch, each with what it was offered on.

    Args:
        stretch: Where the shot would go.
        plan: The plan, for the shots it already covers with.
        pieces: The compiled windows, in order, cards among them.
        clips: The clips that could cover something, from `eligible`.
        assets: Their files, keyed by asset ID, each knowing when it was recorded where anything says.
        meaning: How close each clip's text is in meaning to what is said, by
            clip ID; None when nothing is said under the stretch.
        talk_recorded: When the picture under the stretch was recorded.

    Returns:
        `(candidates, left_out, spread)`: the candidates in the order offered,
        how many clips each rule left out, and the highest, median and lowest
        meaning among those that could be read.
    """
    placed = _placed(pieces)
    left: Dict[str, int] = {}
    readable: List[Tuple[SemanticClip, str]] = []
    unread: List[SemanticClip] = []
    seen_texts: Dict[Tuple[str, str], SemanticClip] = {}
    for clip in clips:
        asset = assets[clip.asset_id]
        why = _left_out(clip, asset, stretch)
        if why is not None:
            left[why] = left.get(why, 0) + 1
            continue
        text = reading(clip, asset)
        # Two clips of one file that read the same are the same candidate to anybody choosing by
        # what they show; the longer is offered, as it leaves the most room to choose from.
        key = (clip.asset_id, text)
        other = seen_texts.get(key)
        if other is not None:
            left["another clip of the same file reads the same; the longest is offered"] = \
                left.get("another clip of the same file reads the same; the longest is offered", 0) + 1
            if clip.duration <= other.duration:
                continue
        seen_texts[key] = clip
    for clip in seen_texts.values():
        text = reading(clip, assets[clip.asset_id])
        if text and meaning is not None and clip.id in meaning:
            readable.append((clip, text))
        else:
            unread.append(clip)
    # A word said under the stretch that is also in what a shot is known to show is the
    # plainest evidence a rule can see, and the meaning model packs short labels a few
    # hundredths apart; so those come first, each group in order of meaning.
    shares = {clip.id: matched_words(stretch.said, text) for clip, text in readable}
    readable.sort(key=lambda item: (not shares[item[0].id], -meaning[item[0].id], item[0].asset_id,
                                    item[0].source_range.start))

    def distance(clip: SemanticClip) -> float:
        gap = _apart(assets[clip.asset_id].recorded_at, talk_recorded)
        return abs(gap) if gap is not None else float("inf")

    unread.sort(key=lambda clip: (distance(clip), clip.asset_id, clip.source_range.start))

    chosen: List[SemanticClip] = []
    per_file: Dict[str, int] = {}

    def take(pool: Sequence[SemanticClip], room: int) -> None:
        for clip in pool:
            if room <= 0:
                return
            if clip in chosen:
                continue
            if per_file.get(clip.asset_id, 0) >= PER_FILE:
                continue
            chosen.append(clip)
            per_file[clip.asset_id] = per_file.get(clip.asset_id, 0) + 1
            room -= 1

    keep_unread = min(UNREAD_PER_STRETCH, len(unread))
    take([clip for clip, _ in readable], CANDIDATES_PER_STRETCH - keep_unread)
    take(unread, CANDIDATES_PER_STRETCH - len(chosen))
    take([clip for clip, _ in readable], CANDIDATES_PER_STRETCH - len(chosen))
    ranked = [clip for clip, _ in readable]
    chosen.sort(key=lambda clip: (0, ranked.index(clip)) if clip in ranked else (1, unread.index(clip)))
    passed = len(seen_texts) - len(chosen)
    if passed:
        left[f"beyond the {CANDIDATES_PER_STRETCH} offered, or past {PER_FILE} from one file"] = passed

    scores = sorted((meaning[clip.id] for clip, _ in readable), reverse=True) if meaning else []
    spread = ({"highest": round(scores[0], 3), "median": round(scores[len(scores) // 2], 3),
               "lowest": round(scores[-1], 3)} if scores else None)

    offered: List[dict] = []
    for clip in chosen:
        asset = assets[clip.asset_id]
        parts = known_as(clip, asset)
        text = reading(clip, asset)
        record: dict = {
            "clip_id": clip.id,
            "file": os.path.basename(asset.path),
            "kind": clip.kind.value,
            "runs": round(clip.duration, 2),
            "fits": clip.duration >= stretch.seconds,
            "from_seconds_up_to": round(max(0.0, clip.duration - min(stretch.seconds, clip.duration)), 2),
            **{name: round(clip.scores[name], 3) for name in ("shake", "exposure", "motion", "faces")
               if name in clip.scores},
        }
        notes: List[str] = []
        if clip in ranked:
            words = shares[clip.id]
            position = ranked.index(clip) + 1
            read_from = [f"{source} 「{said[:40]}」" for source, said in parts]
            record.update({"meaning": round(meaning[clip.id], 3), "rank": position,
                           "read_from": read_from, "matched": words})
            shared = (f"shares 「{'」「'.join(words)}」 with what is said, so it comes before those that share nothing; "
                      if words else "shares no word with what is said; ")
            record["why"] = (f"{shared}{position} of {len(ranked)} read ({meaning[clip.id]:.3f} by meaning), "
                             f"from its {', '.join(source for source, _ in parts)}")
        else:
            record["why"] = ("nothing says what is in it — no description, tags or telling file name — so it was "
                             "not placed by meaning; look at it before picking")
            gap = _apart(asset.recorded_at, talk_recorded)
            if gap is not None:
                record["why"] += f"; offered as {_when(gap)}"
        if not record["fits"]:
            notes.append(f"runs {clip.duration:.1f}s, short of the {stretch.seconds:.1f}s the stretch has room for")
        shake = clip.scores.get("shake")
        if shake is not None and shake > SHAKY:
            notes.append(f"handheld: shake {shake:.3f}, against {SHAKY:g} for a steady shot")
        motion = clip.scores.get("motion")
        if motion is not None and motion < STILL_MOTION:
            notes.append(f"barely moves: motion {motion:.3f}")
        exposure = clip.scores.get("exposure")
        if exposure is not None and stretch.exposure is not None and abs(exposure - stretch.exposure) >= LIGHT_STEP:
            notes.append(f"{'brighter' if exposure > stretch.exposure else 'darker'} than the picture it covers by "
                         f"{abs(exposure - stretch.exposure):.2f}, a step a cut shows (at {LIGHT_STEP:g})")
        if asset.still:
            notes.append("a photo: it holds still for as long as the shot runs")
        if clip.kind != ClipKind.SPEECH and clip.scores.get("faces", 0.0) >= TALKING_FACES:
            notes.append("a face is on screen in it")
        nearest: Optional[float] = None
        for piece, at in placed:
            if piece.card is not None or piece.asset_id != clip.asset_id:
                continue
            overlap = min(piece.end, clip.source_range.end) - max(piece.start, clip.source_range.start)
            if overlap >= REPEAT_SECONDS:
                notes.append(f"the cut already shows {overlap:.1f}s of this stretch of the file at {clock(at)}; "
                             "check_render would call it repeated")
                nearest = None
                break
            ends = round(at + piece.played, 3)
            gap = stretch.opened - ends if ends <= stretch.opened else at - (stretch.opened + stretch.seconds)
            if gap <= NEARBY_SECONDS and (nearest is None or gap < nearest):
                nearest = max(gap, 0.0)
        if nearest is not None:
            notes.append(f"its file is on screen {nearest:.1f}s from this stretch")
        elsewhere = [shot.over_clip_id for shot in plan.broll
                     if shot.clip_id == clip.id and shot.over_clip_id != stretch.over_clip_id]
        if elsewhere:
            notes.append(f"already covers {', '.join(elsewhere)} in this plan; twice in one cut reads as a repeat")
        if notes:
            record["notes"] = notes
        offered.append(record)
    return offered, left, spread


def mark_variety(slots: Sequence[dict]) -> None:
    """Say on each candidate which other stretches have it near the top too, in place.

    The same best shot offered for every stretch is the same shot over and
    over in the cut; the model is told rather than kept from choosing it.
    """
    tops: Dict[str, List[str]] = {}
    for slot in slots:
        for candidate in slot["candidates"][:TOP_ELSEWHERE]:
            tops.setdefault(candidate["clip_id"], []).append(slot["over_clip_id"])
    for slot in slots:
        for candidate in slot["candidates"]:
            others = [over for over in tops.get(candidate["clip_id"], []) if over != slot["over_clip_id"]]
            if others:
                candidate["also_near_the_top_over"] = others


def broll_pick_problems(
    plan: EditPlan,
    pieces: Sequence[Piece],
    clips: Mapping[str, SemanticClip],
    assets: Mapping[str, Asset],
    timeline: SemanticTimeline,
    offers: Mapping[str, Mapping],
    cuts: Optional[Mapping[str, CleanCuts]] = None,
) -> List[str]:
    """Say which of a plan's B-roll shots were not picked from what was offered, or were picked from a stale offer.

    Nothing is moved to make a pick fit: a shot over a stretch nobody asked
    about, of a clip not among that stretch's candidates, or from an offer
    whose stretch or footage has changed since, comes back as a reason.

    Args:
        plan: The plan.
        pieces: The cut as it plays, cards among them.
        clips: The timeline's clips, keyed by ID.
        assets: Their files, keyed by asset ID.
        timeline: The plan's timeline.
        offers: The offers the shots name, keyed by `candidate_hash`.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        One problem per shot that does not hold up.
    """
    if not plan.broll:
        return []
    problems: List[str] = []
    pool: Optional[str] = None
    for position, shot in enumerate(plan.broll, start=1):
        where = f"B-roll {position} (over {shot.over_clip_id})"
        if not shot.candidate_hash:
            problems.append(f"{where}: was not picked from propose_broll's candidates; call propose_broll and pick one "
                            "of the clips it lists for this stretch, with its candidate_hash")
            continue
        offer = offers.get(shot.candidate_hash)
        if offer is None or offer.get("timeline_id") != timeline.id:
            problems.append(f"{where}: no candidates were offered on this timeline under hash {shot.candidate_hash}; "
                            "call propose_broll and pick again from what it lists")
            continue
        stretches = offer.get("stretches", {})
        entry = stretches.get(shot.over_clip_id)
        if entry is None:
            problems.append(
                f"{where}: propose_broll offered nothing over {shot.over_clip_id} under hash {shot.candidate_hash} "
                f"(it offered over {', '.join(stretches) or 'nothing'}); call it with over_clip_ids "
                f"[{shot.over_clip_id}] for candidates there"
            )
            continue
        if shot.clip_id not in entry["candidates"]:
            problems.append(
                f"{where}: {shot.clip_id} is not one of the candidates offered over {shot.over_clip_id} "
                f"({', '.join(entry['candidates']) or 'none'}); pick one of them, or tell the user none fits"
            )
            continue
        stretch, _ = stretch_over(plan, pieces, clips, shot.over_clip_id, cuts)
        if stretch is None:
            # Where it lands is broll_covers' to refuse, with its own reason.
            continue
        if pool is None:
            pool = pool_signature(eligible(list(clips.values()), assets), assets)
        now = stretch_inputs(stretch, timeline, pool)
        changed = [_CHANGED[part] for part, value in entry["inputs"].items() if now.get(part) != value]
        if changed:
            problems.append(
                f"{where}: {', '.join(changed)} changed since the candidates were offered (hash "
                f"{shot.candidate_hash}); call propose_broll again and pick from what it lists"
            )
    return problems


def stretch_for_offer(stretch: Stretch, candidates: Sequence[dict], timeline: SemanticTimeline, pool: str) -> dict:
    """What an offer keeps for one stretch: what it was made from and the clips offered."""
    return {"inputs": stretch_inputs(stretch, timeline, pool), "candidates": [item["clip_id"] for item in candidates]}
