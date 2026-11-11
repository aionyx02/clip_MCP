"""A commercial's variants: other hooks and platform lengths of one confirmed plan, each a branch of its own."""

from typing import Annotated, Dict, List, Optional, Tuple

from pydantic import Field

from app.engine.ad_slots import slot_candidates
from app.engine.ad_variants import (
    MOST_CUTS, MOST_SELECTIONS, Unit, cut_downs, derive, format_for, kept_effects, refusal, variant_hash, window,
)
from app.engine.plan.beats import plan_pieces
from app.engine.plan.checks import check_plan, with_cards
from app.engine.plan.compile import compile_operations
from app.engine.plan.effects import effect_moments, moments_hash
from app.engine.plan.pieces import MUSIC_B_TRACK_ID, MUSIC_TRACK_ID, compiled_duration
from app.models.plan import EditPlan, Structure
from app.models.semantic import AdSlot, ClipLevel
from app.models.timeline.parts import Project
from app import runtime
from app.runtime import mcp
from app.tools import planning, rendering, subtitles, versioning

# The lengths ad platforms sell, when none are asked for.
DEFAULT_TARGETS = (6.0, 15.0, 30.0)
# Provisional: the opening a hook line is read from, as `brand_opening` reads it.
OPENING_SECONDS = 3.0
# Provisional: a song whose last fade is shorter than this is said to stop rather than end.
CLEAN_FADE_SECONDS = 1.0

def _letter(number: int) -> str:
    """`hook-A` for the plan's own hook, `hook-B` and on for the others."""
    return f"hook-{chr(ord('A') + number)}"

def _offer(plan: EditPlan, project: Project, targets: List[float]) -> dict:
    """Work out every variant of a commercial's plan that can be had, and the ones that cannot.

    Returns:
        `variants` by name, each with what `derive` needs to make it again
        and what is said about it; `refused`; the `units` of each hook; and
        `notes`.

    Raises:
        ValueError: If the plan is not a commercial, does not check out, or
            has too many selections to try every combination of.
    """
    if plan.structure != Structure.AD:
        raise ValueError('variants are a commercial\'s: the plan\'s structure is not "ad"')
    timeline, clips, children, assets, cuts = planning._plan_context(plan)
    kit = planning._brand(plan)
    problems, _ = check_plan(plan, timeline, clips, children, assets, cuts, kit=kit)
    if problems:
        raise ValueError("the plan variants are made from has to check out first:\n- " + "\n- ".join(problems))
    if len(plan.selections) > MOST_SELECTIONS:
        raise ValueError(f"the plan has {len(plan.selections)} selections; variants try every combination of at most "
                         f"{MOST_SELECTIONS}. Cut it down by hand with amend_plan first")

    def measure(candidate: EditPlan) -> float:
        pieces = plan_pieces(candidate, clips, children, assets, cuts)
        return compiled_duration(with_cards(candidate, pieces, assets, cuts, project, kit))

    utterances = sorted((clip for clip in clips.values() if clip.level == ClipLevel.UTTERANCE),
                        key=lambda clip: (clip.asset_id, clip.source_range.start))
    offered = slot_candidates(utterances, timeline.ad_product, timeline.ad_aliases)
    roles = {beat.id: beat.role for beat in plan.beats}
    own = [selection.clip_id for selection in plan.selections
           if roles.get(selection.beat_id) is not None and roles[selection.beat_id].value == "hook"]
    used = {selection.clip_id for selection in plan.selections}
    hooks: List[Tuple[str, Optional[Tuple[str, str]]]] = [(_letter(0), None)]
    for candidate in offered[AdSlot.HOOK]:
        if candidate.clip_id not in used:
            hooks.append((_letter(len(hooks)), (candidate.clip_id, "; ".join(candidate.why))))

    def text(clip_id: str) -> str:
        clip = clips.get(clip_id)
        return (clip.text or clip.description or "") if clip else ""

    variants: Dict[str, dict] = {}
    refused: List[dict] = []
    notes: List[str] = []
    for letter, hook in hooks:
        full, stops, _ = derive(plan, f"{letter} · full", hook, None, None, "offered")
        seconds = measure(full)
        found, _ = check_plan(full, timeline, clips, children, assets, cuts, kit=kit)
        found += stops
        line = text(hook[0]) if hook else " ".join(text(clip_id) for clip_id in own)
        if found:
            refused.append({"hook": letter, "line": line, "target": None, "why": found[0]})
            continue
        if hook is not None:
            variants[f"{letter} · full"] = {"hook": letter, "hook_clip": hook, "line": line, "target": None,
                                            "keeps": None, "seconds": seconds}
        units = []
        for selection in full.selections:
            alone = measure(full.model_copy(update={"selections": [selection]}))
            role = roles.get(selection.beat_id)
            units.append(Unit(selection.clip_id, AdSlot(role.value), alone, text(selection.clip_id)))
        overhead = seconds - sum(unit.seconds for unit in units)
        everything = tuple(unit.clip_id for unit in units)
        for target in targets:
            def exact(keeps: Tuple[str, ...]) -> float:
                return measure(derive(full, "measured", None, keeps, target, "offered")[0])

            fits, shortest, longest = cut_downs(units, target, overhead, exact)
            listed, failed = 0, []
            for cut in fits:
                if cut.keeps == everything:
                    notes.append(f"{letter}: the whole cut runs {cut.seconds:.1f}s, already a {target:g}s spot")
                    continue
                if listed >= MOST_CUTS:
                    notes.append(f"{letter} · {target:g}s: {len(fits) - MOST_CUTS} more fit and are not listed")
                    break
                derived, stops, _ = derive(full, "measured", None, cut.keeps, target, "offered")
                found, _ = check_plan(derived, timeline, clips, children, assets, cuts, kit=kit)
                found += stops
                if found:
                    failed.append(found[0])
                    continue
                listed += 1
                name = f"{letter} · {target:g}s" + (f" #{listed}" if listed > 1 else "")
                variants[name] = {"hook": letter, "hook_clip": hook, "line": line, "target": target,
                                  "keeps": list(cut.keeps), "seconds": cut.seconds}
            if not listed and not (fits and fits[0].keeps == everything and len(fits) == 1):
                why = refusal(target, units, shortest, longest) if not failed else (
                    f"{target:g}s: the cuts that fit do not check out; the first: {failed[0]}")
                refused.append({"hook": letter, "line": line, "target": target, "why": why})
    pinned = variant_hash(plan, timeline.slot_hash, {
        name: {"hook": entry["hook_clip"][0] if entry["hook_clip"] else None, "keeps": entry["keeps"],
               "target": entry["target"]} for name, entry in variants.items()})
    return {"variants": variants, "refused": refused, "notes": notes, "hash": pinned, "text": text}

def _require_compiled(project_id: str, plan_id: str) -> Tuple[Project, EditPlan]:
    """The project and the plan its cut was compiled from, which has to be this one."""
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    plan = planning._require_plan(plan_id)
    compiled = rendering.compiled_plan(project)
    if compiled is None or compiled.id != plan.id:
        raise ValueError(f"project {project_id} was not compiled from plan {plan_id}; variants are made from the cut "
                         "the user confirmed, so compile that plan onto it first")
    return project, plan

@mcp.tool()
def propose_ad_variants(
    project_id: str,
    plan_id: str,
    targets: Optional[List[Annotated[float, Field(gt=0)]]] = None,
) -> dict:
    """List the variants a confirmed commercial can be made into: other hooks, and cut down to platform lengths.

    For 「做幾版不同開頭」 or 「剪 6／15／30 秒版」. A hook test keeps everything
    after the first sentence the same, so a platform's A/B test measures the
    opening alone; the other hooks are the hook slot's candidates. A cut-down
    keeps whole selections only, by rule: 6s is a bumper (one hook, one
    product line, the ask), up to 20s a short spot (hook, product, ask), and
    longer ones may keep the problem and the proof. Every combination that
    lands near the length is listed, none ranked; pick with the user. A
    length nothing reaches is in `refused` with what the shortest or longest
    cut runs: say it, do not cut into a sentence.

    Args:
        project_id: The cut the user confirmed.
        plan_id: The plan it was compiled from.
        targets: The lengths, in seconds; omit for 6, 15 and 30.

    Returns:
        `variants`, each with its `name` (such as `hook-B · 15s`), `hook`,
        `hook_line`, `target`, `seconds`, and the `keeps` and `drops` by
        slot and words; `refused`, each with `why`; `notes`; and
        `variant_hash`, to give `make_ad_variants` with the names picked.

    Raises:
        ValueError: If the project was not compiled from the plan, or the
            plan is not a commercial that checks out.
    """
    project, plan = _require_compiled(project_id, plan_id)
    found = _offer(plan, project, sorted(set(targets or DEFAULT_TARGETS)))
    text = found["text"]
    roles = {beat.id: beat.role.value for beat in plan.beats if beat.role is not None}
    slot_of = {selection.clip_id: roles.get(selection.beat_id) for selection in plan.selections}
    listed = []
    for name, entry in found["variants"].items():
        keeps = entry["keeps"]
        entry_target = entry["target"]
        low_high = window(entry_target) if entry_target else None
        listed.append({
            "name": name, "hook": entry["hook"], "hook_line": entry["line"], "target": entry_target,
            "seconds": round(entry["seconds"], 2),
            **({"format": format_for(entry_target).label, "window": list(low_high)} if entry_target else {}),
            "drops": [{"slot": slot_of[clip_id], "clip_id": clip_id, "text": text(clip_id)}
                      for clip_id in slot_of if keeps is not None and clip_id not in keeps
                      and not (entry["hook_clip"] and slot_of[clip_id] == "hook")],
        })
    return {"plan_id": plan.id, "plan_version": plan.version, "original_seconds":
            round(float(project.duration), 2), "variants": listed, "refused": found["refused"],
            "notes": found["notes"], "variant_hash": found["hash"]}

@mcp.tool()
def make_ad_variants(
    project_id: str,
    plan_id: str,
    variant_hash: str,
    names: List[str],
    targets: Optional[List[Annotated[float, Field(gt=0)]]] = None,
) -> dict:
    """Make the variants picked from `propose_ad_variants`, each a branch of the confirmed cut.

    Each is a project of its own beside the original, opened in the editor
    like any branch, with its own copy of the plan: the analyses, timeline
    and slots are shared, nothing is analysed again. Its music is laid to its
    own length with the cue's fades; its captions are made again on its cut,
    corrected ones reused; the sound effects whose moment is still in its cut
    are kept and the others dropped with the reason. Then show them side by
    side with `compare_variants`.

    Args:
        project_id: The confirmed cut, as given to `propose_ad_variants`.
        plan_id: Its plan.
        variant_hash: The `variant_hash` the names were picked from.
        names: The variants' `name`s, as listed.
        targets: The lengths given to `propose_ad_variants`.

    Returns:
        `made`, one per name: its `name`, `project_id`, `plan_id`, `seconds`,
        `target`, `dropped` (selections, beats, cues, B-roll, facts),
        `sound_effects_dropped` and `captions`.

    Raises:
        ValueError: If the hash is not what the variants are now, a name was
            not offered, or a variant does not check out. Nothing is made then.
    """
    project, plan = _require_compiled(project_id, plan_id)
    found = _offer(plan, project, sorted(set(targets or DEFAULT_TARGETS)))
    if variant_hash != found["hash"]:
        raise ValueError("the plan, its slots or the variants offered have changed since these were picked; call "
                         "propose_ad_variants again and pick from what it lists")
    unknown = [name for name in names if name not in found["variants"]]
    if unknown:
        raise ValueError(f"not offered: {', '.join(unknown)}; the variants are "
                         f"{', '.join(found['variants']) or 'none'}")
    timeline, clips, children, assets, cuts = planning._plan_context(plan)
    kit = planning._brand(plan)
    ready = []
    for name in dict.fromkeys(names):
        entry = found["variants"][name]
        derived, stops, dropped = derive(plan, name, entry["hook_clip"], entry["keeps"], entry["target"], variant_hash)
        problems, _ = check_plan(derived, timeline, clips, children, assets, cuts, kit=kit)
        if stops or problems:
            raise ValueError(f"{name} does not check out, so nothing was made:\n- " + "\n- ".join(stops + problems))
        built, provenance = compile_operations(derived, clips, children, assets, project, cuts, kit)
        moments = effect_moments(planning.scratch_cut(project, built, provenance))
        effects, effects_dropped = kept_effects(plan.sound_effects, moments, moments_hash(moments))
        ready.append((name, entry, derived.model_copy(update={"sound_effects": effects}), dropped, effects_dropped))
    made = []
    for name, entry, derived, dropped, effects_dropped in ready:
        branch = versioning.branch_project(project_id, f"{project.name or project_id} {name}",
                                           note=f"廣告變體 {name}")
        copied = rendering.compiled_plan(runtime.repo.get_project(branch["project_id"]))
        stored = runtime.repo.save_plan(derived.model_copy(update={
            "id": copied.id, "version": copied.version, "timeline_input_hash": timeline.input_hash,
        }), f"廣告變體 {name}")
        current = runtime.repo.get_project(branch["project_id"])
        compiled = planning.compile_plan(current.id, current.version, stored.id, note=f"廣告變體 {name}")
        captions = None
        if project.subtitles:
            captioned = subtitles.generate_subtitles(current.id, compiled["new_version"], note=f"廣告變體 {name}")
            captions = {"captions": len(captioned["captions"]), "to_check": len(captioned["to_check"])}
        made.append({"name": name, "project_id": current.id, "plan_id": stored.id,
                     "seconds": round(compiled["duration"], 2), "target": entry["target"], "dropped": dropped,
                     "sound_effects_dropped": effects_dropped, "captions": captions})
    return {"made": made}

def _hook_line(project: Project) -> str:
    """What is said in a cut's first seconds, as its transcript has it."""
    base = project.base_video_track
    said = []
    for clip in sorted(base.clips, key=lambda item: item.timeline_in) if base else []:
        start = float(clip.timeline_in)
        if start >= OPENING_SECONDS or clip.card is not None:
            continue
        until = min(OPENING_SECONDS, float(clip.timeline_out)) - start
        source = float(clip.source_range.start)
        said.append(versioning.said_between(clip.asset_id, source, source + until * float(clip.speed)))
    return "".join(said).strip()

def _music_end(project: Project) -> Optional[dict]:
    """How the music ends: where, with how long a fade, and whether that is with the video."""
    music = [clip for track in project.tracks if track.id in (MUSIC_TRACK_ID, MUSIC_B_TRACK_ID)
             for clip in track.clips]
    if not music:
        return None
    last = max(music, key=lambda clip: clip.timeline_out)
    fade = float(last.audio_fade_out)
    ends = float(last.timeline_out)
    with_video = abs(ends - float(project.duration)) < 0.05
    said = (f"fades out over {fade:g}s" if fade >= CLEAN_FADE_SECONDS else
            f"stops with only a {fade:g}s fade") + (" with the video" if with_video else f" at {ends:.1f}s")
    return {"ends_at": round(ends, 2), "fade_out": fade, "with_video": with_video, "said": said}

@mcp.tool()
def compare_variants(project_id: str, check: bool = True) -> dict:
    """Show a commercial and its variants side by side, one row each, for picking which go to the platform.

    Args:
        project_id: The original, or any of its variants.
        check: Run `check_render` on each, which reads frames and takes a few
            seconds a row; false for lengths and hooks alone.

    Returns:
        `rows`, the original first: `name`, `project_id`, `seconds`,
        `target`, `hook_line` (what is said in the first 3 s), `dropped` by
        slot and words, `music_end`, and with `check` `check_render`'s `ok`,
        the `findings` checks, `half_sentence` count and `brand_opening`.

    Raises:
        ValueError: If the project does not exist or was not compiled from a plan.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    plan = rendering.compiled_plan(project)
    if plan is None:
        raise ValueError(f"project {project_id} was not compiled from one plan")
    root = project
    if plan.variant is not None and project.branched_from is not None:
        root = runtime.repo.get_project(project.branched_from.project_id) or project
    rows = []
    for each in [root, *versioning.branches_of(root.id)]:
        compiled = rendering.compiled_plan(each)
        if compiled is None or (each.id != root.id and compiled.variant is None):
            continue
        clips = {clip.id: clip for clip in runtime.repo.query_semantic_clips(compiled.timeline_id, limit=100000)}
        roles = {}
        if compiled.variant is not None:
            original = runtime.repo.get_plan_version(compiled.variant.of_plan_id, compiled.variant.of_version)
            if original is not None:
                beats = {beat.id: beat.role.value for beat in original.beats if beat.role is not None}
                roles = {selection.clip_id: beats.get(selection.beat_id) for selection in original.selections}
        row = {
            "name": compiled.variant.name if compiled.variant else "original",
            "project_id": each.id, "seconds": round(float(each.duration), 2),
            "target": compiled.target.seconds, "hook_line": _hook_line(each),
            "dropped": [{"slot": roles.get(clip_id), "text": clips[clip_id].text if clip_id in clips else ""}
                        for clip_id in (compiled.variant.dropped if compiled.variant else [])],
            "music_end": _music_end(each),
        }
        if check:
            checked = rendering.check_render(each.id)
            row.update({"ok": checked["ok"], "findings": [finding["check"] for finding in checked["findings"]],
                        "half_sentence": len(checked.get("half_sentence", [])),
                        "brand_opening": checked.get("brand_opening")})
        rows.append(row)
    return {"rows": rows}
