"""Edit plans: saving, amending and reverting them, checking them, and compiling them onto a project's timeline."""

import os
import subprocess
import uuid
from typing import Annotated, List, Literal, Optional

from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field, TypeAdapter

from app.models.plan import (
    EditPlan, PlanAmendment, Trim, TrimKind, apply_amendment, describe_amendment, with_whole_clip_ids,
)
from app.models.semantic import SemanticClip
from app.models.timeline.operations import EditOperation
from app.models.timeline.parts import Project
from app.engine import loudness
from app.engine.plan import (
    COMPILED_TRACKS, Note, as_left, broll_covers, broll_slots, check_plan, check_recompile, compile_operations,
    compiled_duration, diff_plans, music_beds, note_record, piece_changes, plan_pieces,
)
from app.engine.semantic import clean_cuts
from app.engine.ffmpeg import graph_from_file, hidden_window_flags
from app.engine.builder import DEFAULT_LOUDNESS_TARGET
from app.engine.frames import format_timestamp, storyboard_sheet
from app.engine.listen import chart, describe, levels, parts_of
from app.storage import housekeeping
from app import runtime
from app.runtime import mcp
from app.tools import editing, examining, rendering, shared

# Compiled operations are validated the same way a client's are, so a plan cannot reach
# the timeline through a door the tool surface does not have.
OPERATIONS = TypeAdapter(List[EditOperation])

def _plan_context(plan: EditPlan) -> tuple:
    """Gather everything a plan has to be judged and compiled against.

    Args:
        plan: Plan to read.

    Returns:
        `(timeline, clips_by_id, children_by_parent, assets_by_id,
        cuts_by_asset_id)`.

    Raises:
        ValueError: If the timeline the plan names is gone.
    """
    timeline = runtime.repo.get_semantic_timeline(plan.timeline_id)
    if timeline is None:
        raise ValueError(f"the plan is written against timeline {plan.timeline_id}, which no longer exists")
    clips = runtime.repo.query_semantic_clips(timeline.id, limit=shared.MAX_TIMELINE_CLIPS)
    children: dict[str, List[SemanticClip]] = {}
    for clip in clips:
        if clip.parent_id:
            children.setdefault(clip.parent_id, []).append(clip)
    for inside in children.values():
        inside.sort(key=lambda clip: clip.source_range.start)
    footage = {clip.asset_id for clip in clips}
    songs = {cue.asset_id for cue in plan.music.cues if cue.asset_id} if plan.music else set()
    # Where each file may be cut without splitting a word, and where each song's beat
    # falls. The compiler is handed these numbers rather than the transcripts and the
    # sound they come from: reading those is interpretation, and the compiler has to stay
    # a pure function of what it is given.
    cuts = {}
    budget = [rendering.BACKFILL_SECONDS]
    for asset_id in sorted(footage | songs):
        analysis = runtime.repo.get_analysis(asset_id)
        if analysis is not None:
            cuts[asset_id] = clean_cuts(rendering.listened_to(analysis, budget) if asset_id in footage else analysis)
    return timeline, {clip.id: clip for clip in clips}, children, shared.dated(runtime.repo.get_assets(footage | songs)), cuts

def _require_plan(plan_id: Optional[str]) -> EditPlan:
    """Fetch a plan, or the one saved most recently.

    Args:
        plan_id: ID of the plan, or `None` for the newest.

    Returns:
        The plan.

    Raises:
        ValueError: If it does not exist, or nothing has been planned yet.
    """
    plan = runtime.repo.get_plan(plan_id)
    if plan is None:
        raise ValueError(f"plan {plan_id} not found" if plan_id else "no plan has been saved yet; write one with save_plan")
    return plan

@mcp.tool()
def save_plan(plan: EditPlan, note: str = "") -> dict:
    """Store a plan for a cut: what it is for, its parts, and which footage fills them.

    Before choosing a length, an opening or how fast to cut, read
    skill://clip-editing/pacing-and-structure.md.

    A plan is written in terms of semantic clips rather than seconds. You
    decide what goes in and why; `compile_plan` works out where every cut
    lands. Write down why each piece is there and why the ones you passed over
    were passed over — that is what makes the next round of changes possible
    rather than a fresh start.

    Each selection carries a `trim` saying how much of its clip to use:
    `full`, `keep` with the clips inside a section to keep, `head` or `tail`
    with a number of seconds, `range` for a stretch out of the middle, or
    `tighten` to drop the pauses and unusable picture inside a section. A
    selection's `speed` and `volume`, and a beat's `transition_in` and
    `sound_lead`, say how it plays and how one part hands over to the next. There is no free-text trim on purpose: the
    compiler has to produce the same cut from the same plan every time, and it
    cannot do that if it has to interpret a sentence.

    Saving a plan that already exists needs its current `version`, and raises
    it by one. Saving under a fresh id keeps both, which is how two ways of
    cutting the same footage are compared with `diff_plan`.

    Every version saved is kept, so a round of feedback that made the cut
    worse can be undone with `revert_plan`; `list_plan_versions` shows them.

    Args:
        plan: The plan. `timeline_id` says which footage it is about;
            `timeline_input_hash` is filled in for you.
        note: What this version is, kept in its history — the user's own
            words for what they wanted changed are the best thing to put here.

    Returns:
        A dictionary with the `plan_id`, its new `version`, the `timeline_id`,
        and `problems` and `notes` from checking it. A plan is stored whether
        or not it checks out, so it can be fixed rather than retyped. Each
        note has a `kind`, a `message` saying what to do about it, `look` —
        true when it is worth weighing rather than only said — and the
        `beat_id`, `clip_ids` and `seconds` it is about where it has them.

    Raises:
        ValueError: If the timeline does not exist, or the plan was saved at a
            version other than the stored one.
    """
    timeline = runtime.repo.get_semantic_timeline(plan.timeline_id)
    if timeline is None:
        raise ValueError(f"semantic timeline {plan.timeline_id} not found; build one before planning against it")

    stamped = plan.model_copy(update={"timeline_input_hash": timeline.input_hash})
    stored = runtime.repo.save_plan(stamped, note)
    problems, notes = check_plan(stored, *_plan_context(stored))
    newest = runtime.repo.get_semantic_timeline(None)
    if newest is not None and newest.id != timeline.id:
        notes.append(Note(f"a newer timeline ({newest.id}) has been built since; this plan is about {timeline.id}",
                          "newer_timeline"))
    return {
        "plan_id": stored.id,
        "version": stored.version,
        "timeline_id": stored.timeline_id,
        "problems": problems,
        "notes": [note_record(note) for note in notes],
    }

@mcp.tool()
def copy_plan(plan_id: str, timeline_id: Optional[str] = None, version: Optional[int] = None, note: str = "") -> dict:
    """Store a copy of a plan under a new ID, optionally moved onto another timeline.

    Use it to try a second way of cutting without touching the first, or to
    carry a plan onto a timeline built over a different set of files — a
    timeline over three days of footage and one over only the second day give
    the same moment two different clip IDs, and this matches them up.

    A clip is matched to the one in the other timeline that comes from the
    same file, at the same level, over the same seconds; sections match only
    if the same sections were set on both. Every reason, trim, cue and beat
    is carried as it is.

    After the footage is transcribed again, sentences start and end a little
    elsewhere, so a clip with no exact match takes the one from the same file
    and level that covers most of it, and its trim is moved to take the same
    seconds of the file as before. Those are listed in `matched_nearby` with
    both stretches and any trim moved: read them back, since a sentence that
    was split or joined may now say a little more or less.

    Args:
        plan_id: The plan to copy.
        timeline_id: Timeline to move the copy onto; omit to keep the plan's
            own.
        version: Which version to copy; omit for the current one.
        note: What the copy is for, kept in its history.

    Returns:
        As `save_plan`, for the new plan.

    Raises:
        ValueError: If the plan or the timeline does not exist, or a clip the
            copy uses has no match in the other timeline — named, so it can be
            taken out of the plan first — or its trim's seconds are no longer
            in the clip matched to it. A rejected clip with no match is
            simply not carried over, since there is nothing to reject.
    """
    plan = runtime.repo.get_plan(plan_id) if version is None else runtime.repo.get_plan_version(plan_id, version)
    if plan is None:
        raise ValueError(f"plan {plan_id} not found" + ("" if version is None else f" at version {version}"))
    copy = plan.model_copy(deep=True, update={"id": str(uuid.uuid4()), "version": 1})
    target = timeline_id or plan.timeline_id
    if target != plan.timeline_id:
        if runtime.repo.get_semantic_timeline(target) is None:
            raise ValueError(f"semantic timeline {target} not found")
        everything = 1_000_000
        mapping, nearby, shifts = _matched_clips(
            runtime.repo.query_semantic_clips(plan.timeline_id, limit=everything),
            runtime.repo.query_semantic_clips(target, limit=everything),
        )
        used = [
            *(item.clip_id for item in copy.selections),
            *(kept for item in copy.selections for kept in item.trim.keep_clip_ids),
            *(clip for shot in copy.broll for clip in (shot.clip_id, shot.over_clip_id)),
        ]
        missing = sorted({clip for clip in used if mapping.get(clip) is None})
        if missing:
            raise ValueError(
                f"timeline {target} has nothing matching {', '.join(missing)}: its file is not in that timeline, "
                "or it is a section that timeline does not have. Take them out of the plan, or copy it onto a "
                "timeline that has them"
            )
        moved, outside = {}, []
        for item in copy.selections:
            if item.clip_id in shifts:
                said_trim = _trim_moved(item.trim, *shifts[item.clip_id])
                if said_trim == "outside":
                    outside.append(item.clip_id)
                elif said_trim:
                    moved[item.clip_id] = said_trim
        if outside:
            raise ValueError(
                f"{', '.join(outside)}: after the footage was transcribed again, the sentence each is matched to "
                "no longer holds the seconds its trim takes. Set its trim to `full` on this plan, copy it, and "
                "trim it again on the copy"
            )
        for item in copy.selections:
            item.clip_id = mapping[item.clip_id]
            item.trim.keep_clip_ids = [mapping[kept] for kept in item.trim.keep_clip_ids]
        for shot in copy.broll:
            shot.clip_id, shot.over_clip_id = mapping[shot.clip_id], mapping[shot.over_clip_id]
        copy.rejected = [
            item.model_copy(update={"clip_id": mapping[item.clip_id]})
            for item in copy.rejected if mapping.get(item.clip_id)
        ]
        copy.timeline_id = target
    else:
        nearby, moved = [], {}
    said = f"copied from {plan.id} v{plan.version}" + (f" onto {target}" if target != plan.timeline_id else "")
    saved = save_plan(copy, f"{said}: {note}" if note else said)
    used = {item.clip_id for item in plan.selections} | {clip for shot in plan.broll for clip in
                                                         (shot.clip_id, shot.over_clip_id)}
    nearby = [{**match, **({"trim": moved[match["clip_id"]]} if match["clip_id"] in moved else {})}
              for match in nearby if match["clip_id"] in used]
    if nearby:
        saved["matched_nearby"] = nearby
    return saved

# Provisional (roadmap §13): how much of a clip the one in another timeline must cover to
# stand in for it, when the same seconds are not there — a sentence a second transcription moved.
NEARBY_SHARE = 0.5

def _matched_clips(here: List[SemanticClip], there: List[SemanticClip]) -> tuple[dict, list, dict]:
    """Match each clip to the same moment of the same file in another timeline.

    Args:
        here: The clips the plan uses IDs from.
        there: The clips of the timeline it moves onto.

    Returns:
        The new ID for each old one (None when nothing matches), the matches
        that are not exact, and for each of those how far its start and its
        end moved back, and how long the clip standing in for it is.
    """
    def place(clip: SemanticClip) -> tuple:
        """Say which moment of which file a clip is, whatever timeline it sits in."""
        return (clip.asset_id, clip.level, round(clip.source_range.start, 3), round(clip.source_range.end, 3))

    exact = {place(clip): clip for clip in there}
    alike: dict[tuple, List[SemanticClip]] = {}
    for clip in there:
        alike.setdefault((clip.asset_id, clip.level), []).append(clip)
    mapping, nearby, shifts = {}, [], {}
    for clip in here:
        found = exact.get(place(clip))
        if found is None:
            start, end = clip.source_range.start, clip.source_range.end

            def covered(other: SemanticClip) -> float:
                """Say how many of this clip's seconds the other one covers."""
                return max(0.0, min(end, other.source_range.end) - max(start, other.source_range.start))

            best = max(alike.get((clip.asset_id, clip.level), []), key=covered, default=None)
            if best is not None and covered(best) >= NEARBY_SHARE * max(end - start, 0.001):
                found = best
                shifts[clip.id] = (round(start - best.source_range.start, 3), round(end - best.source_range.end, 3),
                                   round(best.source_range.end - best.source_range.start, 3))
                nearby.append({
                    "clip_id": clip.id, "now": best.id,
                    "was": [round(start, 2), round(end, 2)],
                    "is": [round(best.source_range.start, 2), round(best.source_range.end, 2)],
                    "text": best.text,
                })
        mapping[clip.id] = found.id if found else None
    return mapping, nearby, shifts

def _trim_moved(trim: Trim, start_moved: float, end_moved: float, length: float) -> str:
    """Keep a trim on the same seconds of its file when its clip now starts or ends elsewhere.

    A trim is counted from its clip's own start, or for `tail` its end, so a
    sentence that a second transcription moved would otherwise carry it onto
    other words. Moving it back is arithmetic, so it is done; it is said.

    Args:
        trim: The trim, changed in place.
        start_moved: How far back the clip's start moved, in seconds.
        end_moved: How far back its end moved.
        length: How long the clip is now.

    Returns:
        What changed, for whoever reads the copy back; empty when nothing
        did; `outside` when the same seconds are no longer in the clip, and
        then nothing is changed.
    """
    if trim.kind == TrimKind.HEAD and start_moved:
        now = round(trim.seconds + start_moved, 3)
        if not 0 < now <= length:
            return "outside"
        said, trim.seconds = f"head {trim.seconds:g}s → {now:g}s", now
        return said
    if trim.kind == TrimKind.TAIL and end_moved:
        now = round(trim.seconds - end_moved, 3)
        if not 0 < now <= length:
            return "outside"
        said, trim.seconds = f"tail {trim.seconds:g}s → {now:g}s", now
        return said
    if trim.kind == TrimKind.RANGE and start_moved:
        opened = round((trim.from_seconds or 0.0) + start_moved, 3)
        closed = round(trim.to_seconds + start_moved, 3) if trim.to_seconds is not None else None
        if opened < 0 or opened >= (closed if closed is not None else length) or (closed or 0) > length:
            return "outside"
        said = (f"range {trim.from_seconds or 0:g}–{trim.to_seconds if trim.to_seconds is not None else 'end'}s → "
                f"{opened:g}–{closed if closed is not None else 'end'}s")
        trim.from_seconds, trim.to_seconds = opened, closed
        return said
    return ""

@mcp.tool()
def amend_plan(plan_id: str, expected_version: int, amendments: list[PlanAmendment], note: str = "") -> dict:
    """Change part of a stored plan without sending the whole thing again.

    A cut is argued with one piece at a time — this beat runs long, that shot
    does not work, one more thing belongs in the middle — and every selection
    carries a written reason that sending the plan again puts at risk. Retrim
    with `set_trim`, rewrite a reason or move a piece to another beat with
    `set_rationale`, put something in with `add_selection`, and take something
    out with `drop_selection`, which files it under the plan's rejections with
    the reason so the question can be answered later.

    Some feedback is about the whole video, not one piece of it, and has its
    own amendments rather than a trim to every selection: `set_pacing` for
    「整體節奏太慢」 (shorter pauses, less air around every cut),
    `set_music_level` for 「音樂太大聲」 (everywhere, or from one beat),
    `set_music_cue` to change one cue's start, volume, fades or cutting on
    its beat without sending the plan again,
    `set_target` for a new length or platform, and `drop_beat` for
    「這整段不要」, which files every selection in it under the rejections.
    What the viewer has to know is `set_facts` for the plan and
    `set_knowledge` for what one piece establishes and relies on.

    Every version is kept: `revert_plan` goes back to any of them.

    Nothing is compiled here. Call `compile_plan` when the plan reads right.

    Args:
        plan_id: Plan to amend.
        expected_version: The plan's current `version`, as `get_plan` reported
            it. The amendment is refused if the stored plan has moved on.
        amendments: What to change, applied in order.
        note: What the user asked for, in their words, kept with this
            version in its history alongside what the amendments did.

    Returns:
        A dictionary with the `plan_id`, its new `version`, the `timeline_id`,
        and `problems` and `notes` from checking the amended plan. As with
        `save_plan`, it is stored whether or not it checks out.

    Raises:
        ValueError: If the plan or its timeline is gone, the version does not
            match, or an amendment names a selection the plan does not have.
    """
    plan = _require_plan(plan_id)
    if plan.version != expected_version:
        raise ValueError(f"version conflict: plan {plan.id} is at version {plan.version}, not {expected_version}")
    amendments = [type(amendment).model_validate(with_whole_clip_ids(amendment.model_dump(), plan.timeline_id))
                  for amendment in amendments]
    amended = plan.model_copy(deep=True)
    for amendment in amendments:
        apply_amendment(amended, amendment)

    timeline = runtime.repo.get_semantic_timeline(amended.timeline_id)
    if timeline is None:
        raise ValueError(f"the plan is written against timeline {amended.timeline_id}, which no longer exists")
    done = "; ".join(describe_amendment(amendment) for amendment in amendments)
    stored = runtime.repo.save_plan(
        amended.model_copy(update={"timeline_input_hash": timeline.input_hash}),
        f"{note} — {done}" if note and done else note or done,
    )
    problems, notes = check_plan(stored, *_plan_context(stored))
    return {
        "plan_id": stored.id,
        "version": stored.version,
        "timeline_id": stored.timeline_id,
        "problems": problems,
        "notes": [note_record(note) for note in notes],
    }

def _plan_at(plan_id: str, version: Optional[int]) -> EditPlan:
    """Fetch a plan as it is now, or as it was at one version.

    Args:
        plan_id: ID of the plan.
        version: Which version, or None for the current one.

    Returns:
        The plan.

    Raises:
        ValueError: If the plan or that version does not exist.
    """
    plan = _require_plan(plan_id)
    if version is None or version == plan.version:
        return plan
    earlier = runtime.repo.get_plan_version(plan_id, version)
    if earlier is None:
        kept = ", ".join(str(item["version"]) for item in runtime.repo.list_plan_versions(plan_id)) or "none"
        raise ValueError(f"plan {plan_id} has no version {version}; the versions kept are {kept}")
    return earlier

@mcp.tool()
def list_plan_versions(plan_id: Optional[str] = None) -> dict:
    """List every version of a plan, with what each one changed.

    For 「剛剛那樣比較好」 and 「回到前兩版」: every save and every amendment is
    kept as a version, with the note it was saved with and a count of what
    changed from the version before.

    Args:
        plan_id: Plan to read; omit for the one saved most recently.

    Returns:
        A dictionary with the `plan_id`, its `current` version, and
        `versions`, oldest first, each with its `version`, `saved_at`, `note`
        and `changed`, the kinds of change from the version before it.

    Raises:
        ValueError: If the plan does not exist.
    """
    plan = _require_plan(plan_id)
    versions = runtime.repo.list_plan_versions(plan.id)
    listed, before = [], None
    for item in versions:
        now = runtime.repo.get_plan_version(plan.id, int(item["version"]))
        changed = diff_plans(before, now) if before is not None and now is not None else {}
        listed.append({**item, "changed": {group: len(lines) for group, lines in changed.items()}})
        before = now
    return {"plan_id": plan.id, "current": plan.version, "versions": listed}

@mcp.tool()
def revert_plan(plan_id: str, to_version: int, expected_version: int, note: str = "") -> dict:
    """Go back to an earlier version of a plan.

    Nothing is thrown away: the earlier version's content is saved as a new
    version on top, so going back can itself be undone, and the history
    always says what happened. Compile the plan again afterwards; pieces
    adjusted by hand on the project stay pinned as before.

    Args:
        plan_id: Plan to take back.
        to_version: The version to go back to, from `list_plan_versions`.
        expected_version: The plan's current version.
        note: Why, kept with the new version.

    Returns:
        As `save_plan`: the `plan_id`, its new `version`, the `timeline_id`,
        and `problems` and `notes` from checking it — the footage may have
        been analyzed again since that version was written.

    Raises:
        ValueError: If the plan or the version does not exist, or the current
            version does not match.
    """
    plan = _require_plan(plan_id)
    if plan.version != expected_version:
        raise ValueError(f"version conflict: plan {plan.id} is at version {plan.version}, not {expected_version}")
    earlier = _plan_at(plan_id, to_version)
    stored = runtime.repo.save_plan(
        earlier.model_copy(update={"version": plan.version}),
        f"back to version {to_version}" + (f" — {note}" if note else ""),
    )
    problems, notes = check_plan(stored, *_plan_context(stored))
    return {
        "plan_id": stored.id,
        "version": stored.version,
        "timeline_id": stored.timeline_id,
        "problems": problems,
        "notes": [note_record(note) for note in notes],
    }

@mcp.tool()
def get_plan(plan_id: Optional[str] = None, version: Optional[int] = None) -> dict:
    """Read a stored plan back, with the other plans there are to compare it with.

    Args:
        plan_id: Plan to read; omit for the one saved most recently.
        version: An earlier version to read instead of the current one.

    Returns:
        A dictionary with the `plan` as it was stored — its goal, target,
        beats, selections with their reasons, what was rejected and why, and
        any music — plus `other_plans`, each with its `plan_id`, `goal` and
        `timeline_id`, for `diff_plan` to compare against.

    Raises:
        ValueError: If the plan does not exist, or none have been saved.
    """
    plan = _require_plan(plan_id)
    if version is not None:
        plan = _plan_at(plan.id, version)
    return {
        "plan": plan.model_dump(),
        "other_plans": [
            {"plan_id": other.id, "goal": other.goal, "timeline_id": other.timeline_id, "version": other.version}
            for other in runtime.repo.list_plans() if other.id != plan.id
        ],
    }

@mcp.tool()
def validate_plan(plan_id: Optional[str] = None) -> dict:
    """Check a plan against the footage before anything is rendered from it.

    This is the cheap way to find out whether an edit works: it needs no
    decoding and no rendering, only arithmetic. It catches a clip that is not
    in the timeline, a beat nothing belongs to, a trim asking for more seconds
    than a clip has, and footage marked unusable being selected anyway.

    Nothing is corrected. A problem stops the plan compiling and is yours to
    resolve; a note is something to weigh, such as the cut coming out a third
    longer than the length that was asked for.

    Args:
        plan_id: Plan to check; omit for the one saved most recently.

    Returns:
        A dictionary with `ok`, the `problems` that stop it compiling, the
        `notes` worth reading, the `duration` the cut would run, and `clips`,
        how many pieces it would place.

    Raises:
        ValueError: If the plan or its timeline does not exist.
    """
    plan = _require_plan(plan_id)
    timeline, clips, children, assets, cuts = _plan_context(plan)
    problems, notes = check_plan(plan, timeline, clips, children, assets, cuts)
    pieces = [] if problems else plan_pieces(plan, clips, children, assets, cuts)
    return {
        "ok": not problems,
        "problems": problems,
        "notes": [note_record(note) for note in notes],
        "duration": compiled_duration(pieces),
        "clips": len(pieces),
    }

@mcp.tool()
def compile_plan(project_id: str, expected_version: int, plan_id: Optional[str] = None, note: str = "") -> dict:
    """Build a plan's cut on a project's timeline.

    Every second is worked out here, the same way every time: each selection
    becomes the piece its trim asks for, widened into the measured silence
    around it so no line starts abruptly, and pieces that nearly touch become
    one clip. Music, if the plan has any, goes underneath: each cue from where
    its beat begins to where the next one takes over, looped if the song runs
    short. Under a cue that cuts on the beat, each picture cut is moved onto
    the nearest beat where it can get there through silence, and the notes say
    how many moved and how many could not.

    Compiling the same plan again after it has been changed rebuilds the cut,
    which is how a round of feedback lands: change the plan, compile again.
    Anything adjusted by hand in the meantime is kept exactly as it was — a
    clip that was trimmed, moved, recoloured or split is pinned by that edit
    alone, and comes back with those changes, only in the place the new plan
    gives it. If the new plan no longer uses the footage a pinned clip was
    made from, nothing is compiled and you are told which clips are in the
    way, because losing somebody's work to a recompile is worse than stopping.

    The sequence, the music bed and the covering picture belong to the plan,
    each on its own track, and all three are rebuilt. Other tracks — an
    inset, a second music bed — are left exactly as they are.

    Args:
        project_id: Project to build the cut on.
        expected_version: Version you last read from `get_project`.
        plan_id: Plan to compile; omit for the one saved most recently.
        note: The user's own words for what this round changes, kept as what
            this version is in the project's history.

    Returns:
        A dictionary with the `plan_id`, the `project_id`, its `new_version`,
        how many `clips` were placed, how many of those were `kept` from a
        hand adjustment, the `duration` they run, and any `notes` from
        checking the plan.

    Raises:
        ValueError: If the plan does not check out, compiling would destroy
            work done by hand, the project does not exist, or the version does
            not match. Nothing is compiled in part.
    """
    plan = _require_plan(plan_id)
    timeline, clips, children, assets, cuts = _plan_context(plan)
    problems, notes = check_plan(plan, timeline, clips, children, assets, cuts)
    if problems:
        raise ValueError("this plan cannot be compiled yet:\n- " + "\n- ".join(problems))

    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    pieces, left_out = as_left(plan_pieces(plan, clips, children, assets, cuts), project)
    notes = notes + left_out
    covers, _, _ = broll_covers(plan, pieces, clips, assets, cuts)
    blocked = check_recompile(plan, project, pieces, covers, music_beds(plan, pieces, assets, cuts))
    if blocked:
        raise ValueError("compiling would undo work already on this project:\n- " + "\n- ".join(blocked))

    built, provenance = compile_operations(plan, clips, children, assets, project, cuts)
    operations = OPERATIONS.validate_python(built)

    def record(edited: Project) -> None:
        """Write onto each compiled clip where it came from."""
        for track in edited.tracks:
            for clip in track.clips:
                origin = provenance.get(clip.id)
                if origin is not None and track.id in COMPILED_TRACKS:
                    clip.from_plan_id = origin["from_plan_id"]
                    clip.from_clip_ids = origin["from_clip_ids"]
                    clip.pinned = origin["pinned"]

    saved = editing.apply(project_id, expected_version, operations, stamp=record)
    rendering.prepare_playback(saved)
    return {
        "plan_id": plan.id,
        "project_id": project_id,
        "new_version": saved.version,
        "clips": sum(1 for operation in operations if operation.action == "insert_clip"),
        "kept": sum(1 for origin in provenance.values() if origin["pinned"]),
        "duration": float(saved.duration),
        "notes": [note_record(note) for note in notes],
        **({"songs": songs} if (songs := _songs_in(plan)) else {}),
    }

def _songs_in(plan: EditPlan) -> List[dict]:
    """The songs a plan plays, each with how it sounds, to say when the plan is read back."""
    heard = examining.song_tags()
    found = []
    for cue in plan.music.cues if plan.music is not None else []:
        if cue.asset_id is None or any(song["id"] == cue.asset_id for song in found):
            continue
        asset = runtime.repo.get_assets([cue.asset_id]).get(cue.asset_id)
        if asset is not None:
            found.append({"id": asset.id, "name": os.path.basename(asset.path), **heard.get(asset.id, {})})
    return found

@mcp.tool()
def diff_plan(
    before_plan_id: str,
    after_plan_id: str,
    before_version: Optional[int] = None,
    after_version: Optional[int] = None,
) -> dict:
    """Say what changed between two plans.

    Use this when there are two ways of cutting the same footage, or when a
    plan has been reworked after feedback and the user asks what is actually
    different.

    To see what one round of feedback changed, pass the same plan twice with
    the two versions.

    Args:
        before_plan_id: The earlier plan.
        after_plan_id: The later one.
        before_version: A version of the earlier plan; its current one when
            left out.
        after_version: A version of the later plan; its current one when left
            out.

    Returns:
        A dictionary with the two plan ids and `changes`, grouped as `goal`,
        `beats`, `selections`, `broll` and `music`. A group with nothing in it
        is left out, so an empty `changes` means the two plans are the same.

    Raises:
        ValueError: If either plan does not exist.
    """
    return {
        "before": before_plan_id,
        "after": after_plan_id,
        "changes": diff_plans(_plan_at(before_plan_id, before_version), _plan_at(after_plan_id, after_version)),
    }

# How each kind of change is framed on the sheet: the colours people already read these
# in — green for new, yellow for changed, blue for moved, red for gone.
CHANGE_COLOURS = {
    "added": (60, 200, 90), "retrimmed": (240, 200, 40), "moved": (70, 140, 240), "dropped": (230, 60, 60),
}

@mcp.tool()
def preview_sound(
    project_id: str,
    loudness_target: Annotated[Optional[float], Field(ge=-40, le=-5)] = DEFAULT_LOUDNESS_TARGET,
) -> ToolResult:
    """Hear the cut without rendering it: the mix as a file, and a picture of it for you.

    A storyboard shows the rhythm and none of the sound, and you cannot
    listen at all — so this renders the sound alone, exactly as the full
    render will mix it, and measures it. The chart draws the voice (green)
    and the music after ducking (orange) along the cut, with the cuts and the
    parts marked; the text gives, part by part, how loud the voice is, how
    far under it the music sits while somebody talks, and how far the music
    rises in the gaps. Use it to check that music ducks where it should,
    changes where the part changes, and does not drown anybody. The levels
    are measured before the render's loudness normalization, so read them
    against each other rather than as what the viewer hears.

    It takes a fraction of a render, because no picture is decoded. The mix
    is saved as an audio file the user can play.

    Args:
        project_id: ID of the project.
        loudness_target: The loudness the mix file is normalized to, as for
            `render_project`.

    Returns:
        The sound part by part and the `output_path` of the mix, followed by
        the chart.

    Raises:
        ValueError: If the project does not exist or has nothing on its
            sequence.
        RuntimeError: If FFmpeg cannot render the sound.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if project.base_video_track is None or not project.base_video_track.clips:
        raise ValueError(f"project {project_id} has nothing on its sequence to listen to")
    checks = housekeeping.sound_dir(runtime.WORKSPACE_DIR, project_id)
    folder = os.path.join(checks, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)
    # A sound check is looked at once; only the newest of a project's is kept.
    housekeeping.prune(checks, {os.path.basename(folder)})
    mix = os.path.join(folder, rendering.output_name(project, "sound").replace(".mp4", ".m4a"))
    voice, music = os.path.join(folder, "voice.wav"), os.path.join(folder, "music.wav")
    assets, voices = shared.referenced_assets(project), rendering.voices_of(project)
    talking = rendering.talking_of(project, voices)
    command = runtime.renderer.build_sound(
        project, assets, mix, voice, music, loudness_target, voices=voices, talking=talking,
    )
    graph = os.path.join(folder, "graph.txt")
    if loudness_target is not None:
        raw = os.path.join(folder, "raw.wav")
        measured = subprocess.run(
            graph_from_file(runtime.renderer.build_mix(project, assets, raw, voices=voices, talking=talking), graph),
            capture_output=True, creationflags=hidden_window_flags(),
        )
        if measured.returncode != 0:
            raise RuntimeError(measured.stderr.decode("utf-8", errors="replace").strip()[-500:] or "ffmpeg failed")
        command = loudness.with_gain(command, loudness.settle_gain(raw, loudness_target))
        os.remove(raw)
    command = graph_from_file(command, graph)
    result = subprocess.run(command, capture_output=True, creationflags=hidden_window_flags())
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip()[-500:] or "ffmpeg failed")
    heard, played = levels(voice), levels(music)
    for leftover in (voice, music, graph):
        os.remove(leftover)
    lines = [f"The mix is at {mix}.", *describe(parts_of(project, heard, played))]
    return ToolResult(content=["\n".join(lines), Image(data=chart(project, heard, played), format="png")])

@mcp.tool()
def preview_plan_diff(
    before_plan_id: str,
    after_plan_id: str,
    aspect: Optional[Literal["landscape", "portrait", "square"]] = None,
    before_version: Optional[int] = None,
    after_version: Optional[int] = None,
) -> ToolResult:
    """Show what changed between two versions of a cut, on one storyboard.

    Use it after a round of feedback, before anything is rendered — the same
    plan at two versions, or two plans — instead of
    two storyboards to compare by eye, one sheet of the new cut with every
    shot marked — green for added, yellow for retrimmed (the same footage,
    cut at different points), blue for moved, unmarked for untouched — and
    the shots that were taken out at the end in red. The text says how the
    length changed, part by part, and lists every change with its time.

    Nothing is compiled onto a project; both plans are laid out the way
    `compile_plan` would lay them out, so this is safe to call as often as
    you like.

    Args:
        before_plan_id: The earlier plan.
        after_plan_id: The later one.
        aspect: Crop the tiles to the shape the cut will be delivered in;
            omit to show the footage uncropped.
        before_version: A version of the earlier plan. To show what one round
            of feedback did, pass the same plan twice with the version before
            the round and the one after.
        after_version: A version of the later plan.

    Returns:
        A listing of the changes and lengths, followed by the sheet.

    Raises:
        ValueError: If either plan, or its timeline, does not exist, or
            either does not check out well enough to lay out.
    """
    cuts = []
    for plan_id, version in ((before_plan_id, before_version), (after_plan_id, after_version)):
        plan = _plan_at(plan_id, version)
        timeline, clips, children, assets, known = _plan_context(plan)
        problems, _ = check_plan(plan, timeline, clips, children, assets, known)
        if problems:
            raise ValueError(f"plan {plan_id} does not lay out yet:\n- " + "\n- ".join(problems))
        cuts.append((plan, plan_pieces(plan, clips, children, assets, known), assets))
    (before, old, _), (after, new, assets) = cuts
    assets = {**cuts[0][2], **assets}
    changes = piece_changes(old, new)

    old_length, new_length = compiled_duration(old), compiled_duration(new)
    counts = {status: sum(1 for change in changes if change.status == status)
              for status in ("added", "retrimmed", "moved", "dropped", "kept")}
    lines = [
        f"The new cut runs {new_length:.1f}s against {old_length:.1f}s before ({new_length - old_length:+.1f}s): "
        + ", ".join(f"{count} {status}" for status, count in counts.items() if count) + "."
    ]
    names = {beat.id: beat.name for beat in [*before.beats, *after.beats]}
    for beat_id in dict.fromkeys([piece.beat_id for piece in [*old, *new] if piece.beat_id]):
        was = sum(piece.played for piece in old if piece.beat_id == beat_id)
        now = sum(piece.played for piece in new if piece.beat_id == beat_id)
        if round(was, 1) != round(now, 1):
            lines.append(f"part {names.get(beat_id, beat_id)}: {was:.1f}s -> {now:.1f}s")

    shots, borders = [], []
    for number, change in enumerate(changes, start=1):
        piece = change.piece
        middle = round((piece.start + piece.end) / 2, 3)
        tag = change.status.upper() if change.status != "kept" else ""
        where = f"was {format_timestamp(change.at)}" if change.status == "dropped" else format_timestamp(change.at)
        shots.append((assets[piece.asset_id].path, middle, f"#{number} {where} {tag}".strip()))
        borders.append(CHANGE_COLOURS.get(change.status))
        length = f"{piece.played:.1f}s"
        if change.status == "retrimmed" and change.was is not None:
            length = f"{change.was.played:.1f}s -> {piece.played:.1f}s"
        if change.status != "kept":
            lines.append(f"#{number}: {change.status} | {where} | {length} | {', '.join(piece.from_clip_ids)}")
    if len(shots) > shared.MAX_STORYBOARD_TILES:
        lines.append(f"the sheet shows the first {shared.MAX_STORYBOARD_TILES} of {len(shots)} shots")
    shape = rendering.FRAME_SHAPES.get(aspect) if aspect else None
    image = storyboard_sheet(
        shots[:shared.MAX_STORYBOARD_TILES], aspect=shape[0] / shape[1] if shape else None,
        borders=borders[:shared.MAX_STORYBOARD_TILES],
    )
    return ToolResult(content=["\n".join(lines), Image(data=image, format="jpeg")])

@mcp.tool()
def propose_broll(plan_id: Optional[str] = None) -> dict:
    """List the places in a cut where covering picture would help.

    B-roll is a second pass over a rough cut, not something to write at the
    same time as the selections: you have to be able to see where the picture
    runs out of things to say before you can cover it. So plan the cut, then
    come back here.

    What comes back is where, not what. Each slot is a stretch the picture
    holds too long — one shot held on, or several in a row from the same file,
    which is the same problem since the camera never moved. Finding footage
    that belongs over those words is yours: search for it with `query_clips`
    by `tag` or `text`, then put each shot in the plan with an `add_broll`
    amendment, or by saving a plan with `broll` on it.

    What is deliberately not offered is "the sound is good but the picture is
    weak". Weak means a threshold on the exposure and blur measurements, and
    picking one against footage nobody has measured properly would be making a
    number up. Those measurements are on every clip's `scores` if you want to
    read them and decide yourself.

    Args:
        plan_id: Plan to read; omit for the one saved most recently.

    Returns:
        A dictionary with the `plan_id` and `slots`, worst first, each saying
        the `over_clip_id` a shot would start over, the `seconds` it may run
        there, how long the stretch `held_seconds`, and `why` it was offered.
        Empty `slots` means nothing in the cut holds long enough to be worth
        covering. A cut made of long static shots will have most of itself
        offered back, which is the answer rather than a list to work through:
        take the ones at the top.

    Raises:
        ValueError: If the plan does not exist, or its timeline is gone.
    """
    plan = _require_plan(plan_id)
    _, clips, children, assets, cuts = _plan_context(plan)
    pieces = plan_pieces(plan, clips, children, assets, cuts)
    return {"plan_id": plan.id, "slots": broll_slots(plan, pieces, clips, cuts)}
