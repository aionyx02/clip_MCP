"""The story's nominations: where a cut may go wrong, read off its sequence, and the model's rulings on them.

`story_doubts` lists them; `rule_doubts` keeps a ruling on each, with the
sequence it was ruled on, on the project's unfinished production. While an
`auto` production is unfinished, a final render is refused until every
nomination is ruled `keep` with a reason on the sequence as it is now
(`undecided`, which `render_project` asks). The editor's own renders go round
it, as they go round every check of a production: the user is there.
"""

from datetime import timedelta
from typing import List, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, Field

from app.engine import narrative
from app.engine.semantic import was_made_up
from app.models.media import Asset
from app.models.production import Production, Ruling
from app.models.timeline.parts import Clip, Project
from app import runtime
from app.runtime import mcp
from app.tools import production, shared

# A word only grazing a clip's edge is the next clip's, as `read_cut` reads it.
WORD_INSIDE_SECONDS = 0.05


class DoubtRuling(BaseModel):
    """The model's ruling on one nomination."""

    id: str = Field(..., description="The nomination's `id`, as `story_doubts` gave it")
    verdict: Literal["keep", "change"] = Field(
        ..., description="`keep`: it stays as it is, on purpose — say why in `reason`. `change`: the cut is to "
                         "change here; only a registration, the render stays refused until the nomination is gone",
    )
    reason: str = Field(default="", description="Why it stays; required for `keep`. For `change`, what will change")


# ------------------------------------------------------------------ reading the sequence

def _words(clip: Clip, asset: Optional[Asset]) -> Tuple[Tuple[Tuple[float, str], ...], bool]:
    """The words heard over a clip, placed in the cut, and whether its file could be read for them.

    Returns:
        `(words, heard)`: each word as `(second of the cut, text)`; `heard` is
        False only when the file has sound and no transcript.
    """
    if clip.card is not None:
        return (), True
    analysis = runtime.repo.get_analysis(clip.asset_id)
    if analysis is None or analysis.transcript is None:
        return (), asset is not None and not asset.has_audio
    if clip.volume == 0:
        return (), True
    start, end = float(clip.source_range.start), float(clip.source_range.end)
    speed, placed = float(clip.speed), float(clip.timeline_in)
    found = []
    for segment in analysis.transcript.segments:
        if was_made_up(segment):
            continue
        for word in segment.words or [segment]:
            if word.start < end - WORD_INSIDE_SECONDS and word.end > start + WORD_INSIDE_SECONDS:
                found.append((placed + (max(word.start, start) - start) / speed, word.text))
    return tuple(sorted(found)), True


def read_sequence(project: Project) -> Tuple[List[narrative.Shot], str]:
    """Read a project's sequence as the story check reads it.

    Args:
        project: The project.

    Returns:
        The base track's clips as shots, in play order, and the sequence's hash.
    """
    base = project.base_video_track
    clips = sorted(base.clips, key=lambda clip: clip.timeline_in) if base is not None else []
    assets = shared.referenced_assets(project)
    shots, order = [], []
    for clip in clips:
        start, end = float(clip.timeline_in), float(clip.timeline_out)
        order.append((clip.asset_id, round(float(clip.source_range.start), 3),
                      round(float(clip.source_range.end), 3), clip.card.title if clip.card is not None else None))
        if clip.card is not None:
            shots.append(narrative.Shot(clip_id=clip.id, start=start, end=end, card=True))
            continue
        asset = assets.get(clip.asset_id)
        words, heard = _words(clip, asset)
        found = [runtime.repo.get_semantic_clip(clip_id) for clip_id in clip.from_clip_ids]
        found = sorted((semantic for semantic in found if semantic is not None),
                       key=lambda semantic: semantic.source_range.start)
        topics = []
        for semantic in found:
            parent = runtime.repo.get_semantic_clip(semantic.parent_id) if semantic.parent_id else None
            topic = semantic.topic or (parent.topic if parent is not None else None)
            if topic and (not topics or topics[-1] != topic):
                topics.append(topic)
        moment = (asset.recorded_at + timedelta(seconds=float(clip.source_range.start))
                  if asset is not None and asset.recorded_at is not None else None)
        shots.append(narrative.Shot(
            clip_id=clip.id, start=start, end=end, words=words, heard=heard, moment=moment, topics=tuple(topics),
            sourced=bool(found),
        ))
    return shots, narrative.sequence_hash(order)


def _project(project_id: str) -> Project:
    """Read a project with something on its sequence."""
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    base = project.base_video_track
    if base is None or not base.clips:
        raise ValueError(f"project {project_id} has nothing on its sequence to read")
    return project


def _standing(doubt: narrative.Doubt, rulings: Mapping[str, Ruling], sequence: str) -> Optional[dict]:
    """The ruling a nomination has, and whether it still holds on this sequence."""
    ruling = rulings.get(doubt.id)
    if ruling is None:
        return None
    holds = ruling.sequence == sequence
    return {"verdict": ruling.verdict, "reason": ruling.reason, "on_this_sequence": holds,
            "settled": holds and ruling.verdict == "keep" and bool(ruling.reason.strip())}


def _read(project_id: str) -> Tuple[narrative.Doubts, str, float, Optional[Production]]:
    """Nominate over a project as it is now, with its sequence hash, length and unfinished production."""
    project = _project(project_id)
    shots, sequence = read_sequence(project)
    return narrative.nominate(shots), sequence, float(project.duration), production.of_project(project_id)


def _said(project_id: str) -> dict:
    """The nominations over a project as `story_doubts` gives them."""
    found, sequence, length, owner = _read(project_id)
    rulings = {ruling.doubt_id: ruling for ruling in owner.rulings} if owner is not None else {}
    doubts = []
    for doubt in found.doubts:
        doubts.append({"id": doubt.id, "kind": doubt.kind, "at": round(doubt.at, 1), "clip_ids": list(doubt.clip_ids),
                       "says": doubt.says, "ruling": _standing(doubt, rulings, sequence)})
    open_ = [doubt["id"] for doubt in doubts if not (doubt["ruling"] or {}).get("settled")]
    return {
        "project_id": project_id, "sequence": sequence, "length": round(length, 1),
        "doubts": doubts,
        "not_checked": found.unchecked,
        "checked": [kind for kind in narrative.KINDS if kind not in found.unchecked],
        "production_id": owner.id if owner is not None else None,
        "unruled": open_,
    }


def undecided(project_id: str) -> Optional[str]:
    """Say why a final render waits on the story, for a project an unfinished `auto` production is finishing.

    Args:
        project_id: The project.

    Returns:
        The nominations not ruled `keep` with a reason on the sequence as it is
        now, and what to do; None when there are none, the project has no
        unfinished `auto` production, or it does not exist or is empty, which
        rendering says in its own words.
    """
    owner = production.of_project(project_id)
    if owner is None or owner.mode != "auto":
        return None
    project = runtime.repo.get_project(project_id)
    base = project.base_video_track if project is not None else None
    if base is None or not base.clips:
        return None
    shots, sequence = read_sequence(project)
    rulings = {ruling.doubt_id: ruling for ruling in owner.rulings}
    waiting = []
    for doubt in narrative.nominate(shots).doubts:
        standing = _standing(doubt, rulings, sequence)
        if standing is not None and standing["settled"]:
            continue
        if standing is None:
            how = "not ruled"
        elif not standing["on_this_sequence"]:
            how = "ruled on the cut before it changed"
        else:
            how = "ruled change, and still there"
        waiting.append(f"{doubt.id} ({how}): {doubt.says}")
    if not waiting:
        return None
    return (f"production {owner.id} is finishing this on its own, and the story check has {len(waiting)} "
            "nomination(s) not ruled keep on the cut as it is now: " + "; ".join(waiting) + ". Read the cut with "
            "read_cut, then change it or rule_doubts keep with a reason each; what you cannot judge, ask the user")


# ------------------------------------------------------------------ the tools

@mcp.tool()
def story_doubts(project_id: str) -> dict:
    """List where the cut's story may go wrong, read off the sequence as it plays, for you to rule on.

    Nominated by rule, generously, and never judged: a goodbye with more of
    the cut after it, a step back in recording time between neighbouring
    shots, a topic the cut leaves and comes back to, and the first words
    landing late. A hook or a flashback is
    nominated too; that is yours to rule `keep`, with a reason, in
    `rule_doubts`. Read each against `read_cut`.

    A kind that could not be checked is named in `not_checked` with why — no
    topics until the footage is sectioned with `set_sections`, no recording
    times, clips put in by hand. An empty `doubts` with something in
    `not_checked` is not a clean bill.

    Args:
        project_id: The project.

    Returns:
        `doubts`: each with `id`, `kind`, `at` (second of the cut),
        `clip_ids`, `says`, and `ruling` (its verdict and reason,
        `on_this_sequence`, and `settled` when it is a `keep` with a reason on
        the cut as it is now). `not_checked`: per kind, why not. `sequence`:
        the hash rulings are tied to. `unruled`: the IDs not settled.

    Raises:
        ValueError: If the project does not exist or has nothing on its sequence.
    """
    return _said(project_id)


@mcp.tool()
def rule_doubts(project_id: str, rulings: List[DoubtRuling]) -> dict:
    """Rule on `story_doubts`' nominations: `keep` with a reason, or `change`.

    A ruling is tied to the sequence it was made on: moving, trimming or
    swapping a shot — in the editor too — undoes it; captions and music do
    not. Only a `keep` with a reason lets a final render of an `auto`
    production past; `change` registers that the cut will change, and the
    nomination has to be gone before it passes. Kept on the project's
    unfinished production. All or nothing: one ruling refused saves none.

    Args:
        project_id: The project.
        rulings: One per nomination ruled on, by its `id`.

    Returns:
        The nominations as `story_doubts` gives them, with the rulings in.

    Raises:
        ValueError: If the project has no unfinished production, an `id` is
            not among the nominations now, one is ruled twice, or a `keep`
            has no reason.
    """
    found, sequence, _, owner = _read(project_id)
    if owner is None:
        raise ValueError(f"project {project_id} has no unfinished production to keep rulings on: start_production "
                         "with its project_id first")
    if not rulings:
        raise ValueError("no rulings given")
    known = {doubt.id for doubt in found.doubts}
    problems = []
    unknown = [ruling.id for ruling in rulings if ruling.id not in known]
    if unknown:
        problems.append(f"not among the nominations on the cut as it is now: {', '.join(unknown)} (they are: "
                        f"{', '.join(sorted(known)) or 'none'}); call story_doubts again")
    twice = sorted({ruling.id for ruling in rulings if [other.id for other in rulings].count(ruling.id) > 1})
    if twice:
        problems.append(f"ruled more than once: {', '.join(twice)}")
    bare = [ruling.id for ruling in rulings if ruling.verdict == "keep" and not ruling.reason.strip()]
    if bare:
        problems.append(f"keep without a reason: {', '.join(bare)}; say why it stays as it is")
    if problems:
        raise ValueError("; ".join(problems) + ". Nothing was saved")
    ruled = {ruling.id for ruling in rulings}
    kept = [ruling for ruling in owner.rulings if ruling.doubt_id not in ruled]
    kept += [Ruling(doubt_id=ruling.id, verdict=ruling.verdict, reason=ruling.reason.strip(), sequence=sequence)
             for ruling in rulings]
    runtime.repo.save_production(owner.model_copy(update={"rulings": kept}))
    return _said(project_id)
