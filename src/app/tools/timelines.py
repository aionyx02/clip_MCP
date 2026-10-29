"""The semantic timeline: building it from analysed files, searching its clips, and the sections it is cut into."""

import os
from datetime import datetime, timezone
from typing import Annotated, Dict, Iterable, List, Optional, Tuple

import numpy as np
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field

from app.models.media import Asset
from app.models.plan import whole_clip_id, with_whole_clip_ids
from app.models.semantic import (
    ClipDescription, ClipKind, ClipLevel, SectionChoice, SemanticClip, SemanticTimeline, Tag, TagSource,
)
from app.engine import meaning
from app.engine.analysis import unsure_words
from app.engine.sections import build_sections, candidate_hash, check_sections, propose_candidates
from app.engine.semantic import build_timeline, content_scores, maybe_one_person, timeline_input_hash, voice_seconds
from app.engine.frames import format_timestamp, storyboard_sheet
from app import runtime
from app.runtime import mcp
from app.tools import examining, shared

MAX_CLIP_RESULTS = 200
# A summary says enough to judge a clip by; the whole text is one call away.
CLIP_SUMMARY_CHARACTERS = 80
# Which scores a search result carries: the ones describing what is in a clip, which is
# what a search asks about, while how well it was shot is a question for the few clips
# that survive it and `get_semantic_clip` carries those. Taken from where the scores are
# defined rather than listed again here, so a new one lands on the right side by saying
# what it is. Every score can be filtered on either way: `query_clips` matches
# `min_scores` and `max_scores` against all of them, so narrowing by one that is not
# shown here works and returns a shorter list.
SUMMARY_SCORES = content_scores()

def _clip_summary(clip: SemanticClip) -> dict:
    """Describe a semantic clip in one short record.

    Args:
        clip: Clip to describe.

    Returns:
        The clip's identity, placement, headroom, the `SUMMARY_SCORES`, and
        its text cut to `CLIP_SUMMARY_CHARACTERS`; `get_semantic_clip` has the
        rest of both.
    """
    text = clip.text if len(clip.text) <= CLIP_SUMMARY_CHARACTERS else clip.text[:CLIP_SUMMARY_CHARACTERS] + "…"
    return {
        "clip_id": clip.id,
        "asset_id": clip.asset_id,
        **({"name": clip.name} if clip.name else {}),
        **({"topic": clip.topic} if clip.topic else {}),
        **({"speaker": clip.speaker} if clip.speaker else {}),
        "kind": clip.kind.value,
        "start": clip.source_range.start,
        "end": clip.source_range.end,
        "duration": round(clip.duration, 3),
        "safe_in": clip.safe_in,
        "safe_out": clip.safe_out,
        "text": text,
        **({"description": clip.description} if clip.description else {}),
        **({"tags": [tag.value for tag in clip.tags]} if clip.tags else {}),
        "scores": {name: clip.scores[name] for name in SUMMARY_SCORES if name in clip.scores},
    }

def _semantic_clip(clip_id: str, timeline_id: Optional[str]) -> SemanticClip:
    """Fetch a semantic clip by the ID the AI was shown.

    Args:
        clip_id: `u0034`, or a whole ID.
        timeline_id: The timeline a short one is in. Left out, the newest
            build that has such a clip, as a search uses the newest build; the
            clip says which timeline it is from.

    Returns:
        The clip.

    Raises:
        ValueError: If there is no such clip.
    """
    if ":" in clip_id:
        whole = clip_id
    elif timeline_id is not None:
        whole = whole_clip_id(clip_id, _require_timeline(timeline_id).id)
    else:
        holding = runtime.repo.timelines_holding(clip_id)
        whole = whole_clip_id(clip_id, holding[-1]) if holding else clip_id
    clip = runtime.repo.get_semantic_clip(whole)
    if clip is None:
        raise ValueError(f"semantic clip {clip_id} not found; query_clips lists the ones that exist")
    return clip

def _require_timeline(timeline_id: Optional[str]) -> SemanticTimeline:
    """Fetch a semantic timeline, or the newest one.

    Args:
        timeline_id: ID of the timeline, or `None` for the newest build.

    Returns:
        The timeline.

    Raises:
        ValueError: If it does not exist, or if nothing has been built yet.
    """
    timeline = runtime.repo.get_semantic_timeline(timeline_id)
    if timeline is None:
        raise ValueError(
            f"semantic timeline {timeline_id} not found" if timeline_id
            else "no semantic timeline has been built; call build_semantic_timeline first"
        )
    return timeline

@mcp.tool()
def build_semantic_timeline(asset_ids: List[str], rebuild: bool = False) -> dict:
    """Turn the analyses of a set of assets into semantic clips that can be searched.

    A semantic clip is one thing that stands on its own: a sentence someone
    said, a shot, a pause. Each one carries how much room its edges have
    before they would run into sound, so a cut can be placed without having to
    work out from the transcript whether it would clip a word.

    Build this once the assets have been analyzed, then work through
    `query_clips` rather than reading transcripts with `get_analysis`. Every
    asset must have been analyzed first; transcribed assets give sentences,
    and assets without a transcript give shots.

    Building again over unchanged analyses returns the same timeline with the
    same clip IDs rather than making a new one, so an ID stays pointing at the
    same moment.

    Args:
        asset_ids: IDs of the assets to cover. They can be built together, so
            that one timeline covers a whole shoot.
        rebuild: Build again even when nothing has changed. Only useful after
            re-analyzing an asset.

    Returns:
        A dictionary with the `timeline_id`, the `asset_ids` it covers,
        `reused` saying whether an existing build was returned untouched,
        `total_clips`, `counts` of clips by level and kind, `voices`, the
        seconds each voice talks, most first — one voice holding nearly all of
        it and the rest a few seconds each is one person heard several ways,
        so do not tell the user there are many; `maybe_one_person`, each
        voice talking under a tenth of the time with `like`, the main voice,
        and `distance`, how far apart the two sound (0 alike, 1 unrelated) —
        ask the user whether those are one person on another microphone
        before marking speakers in captions or telling them apart; and
        `stale`.

        `stale` lists assets analyzed with detection settings or a speech
        model the server no longer uses. The build goes ahead with them,
        because re-analyzing an hour of footage to change nothing is worse
        than a number measured slightly differently — but those assets are
        being compared with the rest on an uneven footing, so analyze them
        again when the comparison matters.

    Raises:
        ValueError: If no assets were given, an asset does not exist, or an
            asset has not been analyzed yet.
    """
    if not asset_ids:
        raise ValueError("give at least one asset to build a semantic timeline from")
    assets = {asset_id: shared.get_asset(asset_id) for asset_id in dict.fromkeys(asset_ids)}
    analyses = {asset_id: runtime.repo.get_analysis(asset_id) for asset_id in assets}
    missing = sorted(asset_id for asset_id, analysis in analyses.items() if analysis is None)
    if missing:
        raise ValueError(f"these assets have not been analyzed yet, call analyze_asset on them first: {', '.join(missing)}")

    stale = sorted(asset_id for asset_id, analysis in analyses.items() if examining.is_stale(analysis))
    existing = runtime.repo.find_semantic_timeline(timeline_input_hash(analyses))
    if existing is not None and not rebuild:
        voices = voice_seconds(runtime.repo.query_semantic_clips(existing.id, kinds=[ClipKind.SPEECH], limit=-1))
        return {
            "timeline_id": existing.id,
            "asset_ids": existing.asset_ids,
            "reused": True,
            "total_clips": sum(runtime.repo.count_semantic_clips(existing.id).values()),
            "counts": runtime.repo.count_semantic_clips(existing.id),
            "voices": voices,
            "maybe_one_person": maybe_one_person(analyses, voices),
            "stale": stale,
        }

    timeline, clips = build_timeline(assets, analyses)
    runtime.repo.save_semantic_timeline(timeline, clips)
    voices = voice_seconds(clips)
    return {
        "timeline_id": timeline.id,
        "asset_ids": timeline.asset_ids,
        "reused": False,
        "total_clips": len(clips),
        "counts": runtime.repo.count_semantic_clips(timeline.id),
        "voices": voices,
        "maybe_one_person": maybe_one_person(analyses, voices),
        "stale": stale,
    }

@mcp.tool()
def list_timelines() -> dict:
    """List every timeline built, newest first, to find the old ones to clear away.

    A timeline is built for each set of files, so the same footage built with
    one file more or less is another timeline, numbered afresh. They pile up;
    a clip's short ID then means a clip in each of them.

    Returns:
        `timelines`, each with its ID, when it was built, its files, how many
        clips it has, `plans` built on it now and `past`, older versions of
        plans that were. One with neither can go with `delete_timelines`.
    """
    names = {asset.id: os.path.basename(asset.path) for asset in runtime.repo.list_assets()}
    return {"timelines": [
        {
            "timeline_id": timeline.id,
            "built_at": timeline.built_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            "files": [names.get(asset_id, asset_id) for asset_id in timeline.asset_ids],
            "clips": clips,
            "plans": plans,
            "past": [f"{plan_id} v{version}" for plan_id, version in past],
        }
        for timeline, clips, plans, past in runtime.repo.list_semantic_timelines()
    ]}

@mcp.tool()
def delete_timelines(timeline_ids: List[str], with_past_versions: bool = False) -> dict:
    """Delete timelines no plan is built on, with their clips and search vectors.

    The files and their analyses stay: building a timeline over the same files
    again makes it again. A cut already made keeps its clips; what it loses is
    only the footage's descriptions `read_cut` reads.

    Args:
        timeline_ids: The timelines, as `list_timelines` gives them.
        with_past_versions: Delete one older versions of a plan were built on
            too; going back to those versions then needs `copy_plan` onto a
            timeline that is still there. Only once the user has said so.

    Returns:
        `deleted`, the timelines deleted.

    Raises:
        ValueError: If a timeline does not exist, a plan is built on it now,
            or older versions of one were and `with_past_versions` is off.
            Nothing is deleted then.
    """
    known = {timeline.id: (plans, past) for timeline, _, plans, past in runtime.repo.list_semantic_timelines()}
    problems = []
    for timeline_id in timeline_ids:
        if timeline_id not in known:
            problems.append(f"there is no timeline {timeline_id}")
            continue
        plans, past = known[timeline_id]
        if plans:
            problems.append(f"{timeline_id} has {', '.join(plans)} built on it; copy_plan it onto another timeline first")
        elif past and not with_past_versions:
            problems.append(
                f"{timeline_id} has older versions built on it ({', '.join(f'{plan} v{version}' for plan, version in past)}); "
                "going back to them would need copy_plan. Delete it anyway with with_past_versions once the user says so"
            )
    if problems:
        raise ValueError("nothing deleted: " + "; ".join(problems))
    for timeline_id in dict.fromkeys(timeline_ids):
        runtime.repo.delete_semantic_timeline(timeline_id)
    return {"deleted": list(dict.fromkeys(timeline_ids))}

@mcp.tool()
def query_clips(
    timeline_id: Optional[str] = None,
    level: ClipLevel = ClipLevel.UTTERANCE,
    kinds: Optional[List[ClipKind]] = None,
    asset_ids: Optional[List[str]] = None,
    topic: Optional[str] = None,
    tag: Optional[str] = None,
    text: Optional[str] = None,
    start: Annotated[Optional[float], Field(ge=0)] = None,
    end: Optional[float] = None,
    min_duration: Annotated[Optional[float], Field(ge=0)] = None,
    max_duration: Annotated[Optional[float], Field(ge=0)] = None,
    min_scores: Optional[dict[str, float]] = None,
    max_scores: Optional[dict[str, float]] = None,
    limit: Annotated[int, Field(ge=1, le=MAX_CLIP_RESULTS)] = 50,
    offset: Annotated[int, Field(ge=0)] = 0,
    brief: bool = False,
    about: Optional[str] = None,
) -> dict:
    """Search the semantic timeline for the clips worth looking at.

    This is how to find material: ask for what is needed rather than reading a
    transcript end to end. Conditions combine, so "the parts of this file
    where someone speaks for more than four seconds and the picture is not
    black" is one call.

    `about` finds clips by what they mean rather than the words in them:
    "where the product is introduced" finds the sentence that says what it
    does, though nobody said "introduce". Prefer it to reading everything with
    `brief` when looking for something in particular; use `text` when the
    exact word is known.

    Results are summaries. Each carries enough to decide whether a clip is
    wanted — what is said, how long it runs, how much room its edges have —
    and `get_semantic_clip` has the full text and word timings for the few
    that matter. To read through footage — what was said in a whole folder,
    in order — ask for `brief`: one line per clip, a few hundred of them in
    a single result, with `offset` for the next page.

    A result shows the scores that say whether there is usable content:
    `speech` and `silence` are the shares of the clip covered by words and by
    detected silence, `black` and `frozen` the shares the picture detectors
    marked, and `words_per_second` its pace (characters, for Chinese). Every
    clip carries more — how well it was shot, who was on screen, what the
    sound was like — and `min_scores` says what each one means. They are
    measurements with no threshold behind them: filter on them to shorten a
    list, then read `get_semantic_clip` for the few that matter.

    A clip from a file whose voices were told apart carries `speaker`, a
    label like `V1` that is the same person in every file of the timeline; a
    sentence that straddles a handover carries none rather than a guess.

    Clip IDs are shown without their timeline, as `u0034`: write them that
    way in a plan.

    Args:
        timeline_id: Timeline to search; omit for the one built most recently.
        level: `utterance` for single sentences and shots, `section` for the
            parts a whole subject was covered in. Sections exist once
            `set_sections` has stored them; plan from those and drop to
            `utterance` when placing the cuts.
        kinds: Keep only these kinds: `speech` (someone is talking), `silence`
            (nothing is audible), `ambient` (sound but no speech, and shots
            from files without sound), `unusable` (nobody talking over a black
            or frozen picture). Omit for all of them.
        asset_ids: Keep only clips from these assets. Omit for every file
            in the timeline.
        topic: Keep only sections about this subject, as `set_sections`
            labelled them. Topics are shared between files, so this is how to
            gather everything shot about one thing.
        tag: Keep only clips carrying this tag, as `set_clip_tags` wrote it.
        text: Keep only clips whose words or on-screen description contain
            this.
        start: Keep only clips reaching past this second of their source file.
        end: Keep only clips beginning before this second of their source file.
        min_duration: Keep only clips at least this many seconds long.
        max_duration: Keep only clips at most this many seconds long.
        min_scores: Lower bounds on scores, such as `{"speech": 0.5}`. A clip
            without that score is left out, since a measurement nobody took
            cannot be said to pass. Besides the ones a result shows:
            `exposure` is mean brightness from 0 to 1 and `contrast` the
            spread between the dark and bright ends; `blur` rises as the
            picture goes soft, a sharp shot near 5; `motion` is how much
            changes from frame to frame, 0 locked off; `shake` is how unsteady
            the camera was, near 0 on a tripod or an even pan and far higher
            handheld. Sound: `loudness`, `peak` and `noise_floor` (the hiss
            under everything) in dBFS, and `flatness`, which is not a level:
            high where the waveform is squared off, so a `peak` near 0 with a
            high `flatness` is a clipped recording. Faces: `faces` is how
            many were up, averaged over the clip; `face_share` how much of the frame the largest
            filled (close-up against wide); `face_x` and `face_y` where it
            sat, 0 to 1 across and down. Faces are found, not people: 0 means
            no face was seen, not that the shot is empty.
        max_scores: Upper bounds on scores, such as `{"black": 0.1}` or
            `{"shake": 0.01}` for shots steady enough to hold on screen.
        limit: Largest number of clips to return.
        offset: How many matching clips to skip first, for the next page.
        brief: Return `lines` instead of `clips`, in the order the footage
            was shot: one line per clip, reading
            `<id> <file> <start>-<end> <kind> [<speaker>] [<topic>] <text>`,
            with the whole of what is said. With `about`,
            the lines follow relevance rather than the order of shooting.
        about: Rank the matching clips by how close they are in meaning to
            this, in any language, closest first, each with its `relevance`.
            Answered on this machine by a small model; each sentence is read
            with the ones either side of it, and a fragment of a few
            characters or a clip with neither words nor a description is left
            out. `relevance` orders one search's results and is not comparable
            between searches; `spread` gives its highest, median and lowest,
            and results bunched close together mean nothing stood out.
            `outdated` and `stale` say when the timeline or its files'
            analyses are behind and the ranking may hold lines since dropped.
            The first search downloads the model, about 130 MB, and embeds
            every clip once, about a second per thousand.

    Returns:
        A dictionary with the `timeline_id` searched and the matching `clips`,
        each with its `clip_id`, `asset_id`, `kind`, `start` and `end` in the
        source file, `duration`, `safe_in` / `safe_out`, its `text`, and its
        `scores` — or with `brief`, their `lines`, and `notes`: each file's
        notes on how it may be used, by file name. `truncated` is true when the
        limit cut the results short; the next page starts at `offset` plus
        `limit`.

    Raises:
        ValueError: If the timeline does not exist, or a score name is not a
            plain identifier.
    """
    timeline = _require_timeline(timeline_id)
    clips = runtime.repo.query_semantic_clips(
        timeline.id,
        level=level,
        kinds=kinds,
        asset_ids=asset_ids,
        topic=topic,
        tag=tag,
        text=text,
        start=start,
        end=end,
        min_duration=min_duration,
        max_duration=max_duration,
        min_scores=min_scores,
        max_scores=max_scores,
        # Read through in the order it was shot, or ranked by meaning: either needs every match to sort.
        limit=shared.MAX_TIMELINE_CLIPS if brief or about else offset + limit,
    )
    relevance: Dict[str, float] = {}
    if about and about.strip():
        ranked = _rank_by_meaning(clips, about.strip())
        by_id = {clip.id: clip for clip in clips}
        clips = [by_id[clip_id] for clip_id, _ in ranked]
        relevance = {clip_id: round(score, 3) for clip_id, score in ranked}
        page = clips[offset:offset + limit]
        # How the scores are spread, so a page of results all a few hundredths apart reads as
        # nothing standing out rather than as sixty good matches (#83); and whether the
        # timeline was built over what the footage now says.
        scores = sorted(relevance.values(), reverse=True)
        about_how = {
            **({"spread": {"highest": scores[0], "median": scores[len(scores) // 2], "lowest": scores[-1]}}
               if scores else {}),
            **_outdated(timeline),
        }
        if brief:
            assets = runtime.repo.get_assets({clip.asset_id for clip in page})
            return {
                "timeline_id": timeline.id,
                "lines": [
                    f"{relevance[clip.id]:.3f} " + _clip_line(
                        clip, os.path.basename(assets[clip.asset_id].path) if clip.asset_id in assets else clip.asset_id)
                    for clip in page
                ],
                **_file_notes(assets.values()),
                **about_how,
                "truncated": offset + limit < len(clips),
            }
        return {
            "timeline_id": timeline.id,
            "clips": [{**_clip_summary(clip), "relevance": relevance[clip.id]} for clip in page],
            **about_how,
            "truncated": offset + limit < len(clips),
        }
    if brief:
        assets = shared.dated(runtime.repo.get_assets({clip.asset_id for clip in clips}))
        never = datetime.max.replace(tzinfo=timezone.utc)

        def shot(clip: SemanticClip) -> tuple:
            """Sort key: when its file was recorded, then its name, then where in it."""
            asset = assets.get(clip.asset_id)
            return (asset.recorded_at or never if asset else never,
                    os.path.basename(asset.path) if asset else "", clip.source_range.start)

        ordered = sorted(clips, key=shot)
        page = ordered[offset:offset + limit]
        return {
            "timeline_id": timeline.id,
            "lines": [
                _clip_line(clip, os.path.basename(assets[clip.asset_id].path) if clip.asset_id in assets
                           else clip.asset_id)
                for clip in page
            ],
            **_file_notes(assets[clip.asset_id] for clip in page if clip.asset_id in assets),
            "truncated": offset + limit < len(ordered),
        }
    truncated = len(clips) == offset + limit
    clips = clips[offset:]
    return {
        "timeline_id": timeline.id,
        "clips": [_clip_summary(clip) for clip in clips],
        "truncated": truncated,
    }

def _outdated(timeline: SemanticTimeline) -> dict:
    """Say whether a timeline was built over what its files' analyses now say, and by the rules now used.

    Args:
        timeline: The timeline.

    Returns:
        `outdated`, what to do about it, when a file was analyzed again or the
        rules changed since it was built, and `stale`, the files analyzed in
        a way the server no longer uses; empty when neither.
    """
    analyses = {asset_id: runtime.repo.get_analysis(asset_id) for asset_id in timeline.asset_ids}
    if any(analysis is None for analysis in analyses.values()):
        return {}
    found: dict = {}
    # A second listen saved since the build changes the analysis and nothing the timeline was made
    # from, so a build from before it still counts as current.
    unheard = {asset_id: analysis.model_copy(update={"second_listen": None}) for asset_id, analysis in analyses.items()}
    if timeline.input_hash not in (timeline_input_hash(analyses), timeline_input_hash(unheard)):
        found["outdated"] = ("this timeline was built before its files were analyzed again or the rules changed — "
                             "lines since found to be made up may still be in it; build_semantic_timeline over the "
                             "same files for a fresh one")
    stale = sorted(asset_id for asset_id, analysis in analyses.items() if examining.is_stale(analysis))
    if stale:
        found["stale"] = stale
    return found

def _meaning_text(clip: SemanticClip) -> str:
    """What a clip is about, in words: its section name and topic, what is said, and what is seen."""
    return "\n".join(part for part in (clip.name, clip.topic, clip.text, clip.description) if part and part.strip())

# Fewer characters than this say too little to be placed by meaning ("各位我", "去"); left
# to search by `text`, they would otherwise sit close to every question.
MIN_MEANING_CHARACTERS = 4
# Sentences either side read with each one: what is said on camera comes in fragments,
# and "放在這裡" means something only next to what was being put there.
MEANING_NEIGHBOURS = 1

def _meaning_context(clips: List[SemanticClip]) -> Dict[str, str]:
    """What each clip is read as for a search by meaning: its own words, then its neighbours'.

    Neighbours are the clips beside it in its file at the same level, from the
    whole timeline rather than from what this search matched, so the text, and
    the stored vector made from it, do not change with the search's conditions.

    Args:
        clips: The clips to read.

    Returns:
        The text by clip ID, for clips with enough of their own to read.
    """
    found: Dict[str, str] = {}
    for timeline_id, level in {(clip.timeline_id, clip.level) for clip in clips}:
        everything = [clip for clip in runtime.repo.query_semantic_clips(timeline_id, level=level, limit=shared.MAX_TIMELINE_CLIPS)
                      if _meaning_text(clip)]
        everything.sort(key=lambda clip: (clip.asset_id, clip.source_range.start))
        for index, clip in enumerate(everything):
            own = _meaning_text(clip)
            if len("".join(own.split())) < MIN_MEANING_CHARACTERS:
                continue
            around = [other for other in everything[max(index - MEANING_NEIGHBOURS, 0):index + MEANING_NEIGHBOURS + 1]
                      if other is not clip and other.asset_id == clip.asset_id]
            found[clip.id] = "\n".join([own] + [_meaning_text(other) for other in around])
    wanted = {clip.id for clip in clips}
    return {clip_id: text for clip_id, text in found.items() if clip_id in wanted}

def _rank_by_meaning(clips: List[SemanticClip], question: str) -> List[Tuple[str, float]]:
    """Rank clips by how close they are in meaning to a question, embedding any not yet embedded.

    Args:
        clips: The clips to rank.
        question: What is being looked for.

    Returns:
        `(clip_id, similarity)` pairs, closest first. Clips with nothing to
        read are left out.
    """
    context = _meaning_context(clips)
    wanted = {clip.id: (clip, context[clip.id]) for clip in clips if clip.id in context}
    stored = runtime.repo.clip_vectors(wanted)
    vectors: Dict[str, np.ndarray] = {}
    missing = []
    for clip_id, (clip, words) in wanted.items():
        made_from = meaning.digest(words)
        kept = stored.get(clip_id)
        if kept and kept[0] == made_from:
            vectors[clip_id] = np.frombuffer(kept[1], dtype=np.float32)
        else:
            missing.append((clip, words, made_from))
    if missing:
        needed = meaning.download_size()
        try:
            made = meaning.embed([words for _, words, _ in missing])
        except RuntimeError as error:
            # The first search fetches the model; without it, say so and point at the search that needs none.
            raise ValueError(
                f"search by meaning needs its model ({needed // meaning.models.MEGABYTE} MB to download once), "
                f"which could not be fetched: {error}. Search with `text` meanwhile, and try `about` again later."
            ) from error
        runtime.repo.save_clip_vectors(
            (clip.id, clip.timeline_id, made_from, vector.tobytes())
            for (clip, _, made_from), vector in zip(missing, made)
        )
        vectors.update({clip.id: vector for (clip, _, _), vector in zip(missing, made)})
    return meaning.rank(question, vectors)

def _file_notes(assets: Iterable[Asset]) -> dict:
    """The notes of the files a page of search results comes from, once per file.

    Args:
        assets: The files.

    Returns:
        `{"notes": {name: notes}}`, or nothing when none of them has notes.
    """
    noted = {os.path.basename(asset.path): asset.notes for asset in assets if asset.notes}
    return {"notes": noted} if noted else {}

def _clip_line(clip: SemanticClip, file_name: str) -> str:
    """Describe a semantic clip in one line, for reading footage through.

    Args:
        clip: Clip to describe.
        file_name: The name of the file it comes from.

    Returns:
        Its short ID, file, source range, kind, speaker and topic where it
        has them, and all of what is said or its description.
    """
    parts = [
        clip.id.split(":", 1)[-1], file_name,
        f"{clip.source_range.start:.2f}-{clip.source_range.end:.2f}", clip.kind.value,
    ]
    if clip.speaker:
        parts.append(clip.speaker)
    if clip.topic:
        parts.append(f"[{clip.topic}]")
    said = clip.text or (clip.description or "")
    return " ".join(parts + ([said] if said else []))

def _timeline_utterances(timeline: SemanticTimeline) -> List[SemanticClip]:
    """Read every `utterance` clip of a timeline, in order.

    Args:
        timeline: Timeline to read.

    Returns:
        The clips, ordered by asset and then by time.
    """
    return runtime.repo.query_semantic_clips(timeline.id, level=ClipLevel.UTTERANCE, limit=shared.MAX_TIMELINE_CLIPS)

def _timeline_analyses(timeline: SemanticTimeline) -> dict:
    """Read the analyses a timeline was built from.

    Args:
        timeline: Timeline whose assets to read.

    Returns:
        The analyses, keyed by asset ID.

    Raises:
        ValueError: If an asset's analysis has gone missing.
    """
    analyses = {asset_id: runtime.repo.get_analysis(asset_id) for asset_id in timeline.asset_ids}
    missing = sorted(asset_id for asset_id, found in analyses.items() if found is None)
    if missing:
        raise ValueError(f"the analysis of {', '.join(missing)} is gone; analyze those assets and build the timeline again")
    return analyses

@mcp.tool()
def propose_sections(timeline_id: Optional[str] = None, asset_id: Optional[str] = None) -> dict:
    """List the places a section could begin, for you to choose between.

    Sentences are too fine to plan from: an hour of talk is hundreds of them.
    Sections are the level to think in — one per thing the speaker gets
    through — and deciding where one ends is a judgement, so it is yours to
    make. What the server does is narrow the field, by measuring where a
    boundary is plausible: how long the pause is, how far the vocabulary turns
    over, whether the shots change, whether the sentence opens with a phrase
    like 「那我們接下來」.

    Read the utterances, pick the candidates that are real boundaries, name
    each section, and send them back with `set_sections`. You can only choose
    among these candidates; that is deliberate, because each one already sits
    on a boundary measured from the audio, so a section can never begin in the
    middle of a word.

    Give every section a `topic` as well when several of them are about the
    same subject. Topics are labels, not spans, so the same one can be used in
    different files and it is what answers "what is in this footage".

    Args:
        timeline_id: Timeline to propose for; omit for the newest build.
        asset_id: Show only one file's candidates and utterances. Candidates
            are always worked out over the whole timeline, so this changes
            what you read, not what you may choose.

    Returns:
        A dictionary with the `timeline_id`, the `candidates` — each with the
        `after_clip_id` that would open the next section, its time, its
        `signals` and its `score` — and the `utterances` to read them against,
        each with its `clip_id`, times, `kind` and text.

    Raises:
        ValueError: If the timeline does not exist or its analyses are gone.
    """
    timeline = _require_timeline(timeline_id)
    clips = _timeline_utterances(timeline)
    candidates = propose_candidates(clips, _timeline_analyses(timeline))

    shown = [clip for clip in clips if asset_id is None or clip.asset_id == asset_id]
    return {
        "timeline_id": timeline.id,
        "candidates": [
            {
                "after_clip_id": candidate.after_clip_id,
                "asset_id": candidate.asset_id,
                "at": round(candidate.at, 3),
                "signals": candidate.signals,
                "score": candidate.score,
            }
            for candidate in candidates
            if asset_id is None or candidate.asset_id == asset_id
        ],
        "utterances": [
            {
                "clip_id": clip.id,
                "asset_id": clip.asset_id,
                "start": clip.source_range.start,
                "end": clip.source_range.end,
                "kind": clip.kind.value,
                "text": clip.text,
            }
            for clip in shown
        ],
    }

@mcp.tool()
def set_sections(
    sections: List[SectionChoice],
    timeline_id: Optional[str] = None,
    chosen_by: Optional[str] = None,
) -> dict:
    """Store the sections you chose from `propose_sections`.

    Every utterance belongs to exactly one section, so the sections of a file
    follow one another with nothing left between them, and each one after the
    first begins at a candidate boundary. Anything that does not hold up is
    handed back with the reason rather than quietly corrected: a section moved
    without your say-so is a decision nobody made.

    Sections replace whatever was stored for this timeline before, so sending
    a revised set is how you change your mind.

    Args:
        sections: The sections, in order, each with the utterance it opens on
            and the one it ends on, a `name`, an optional `topic` shared with
            related sections, and a one-line `summary`.
        timeline_id: Timeline they belong to; omit for the newest build.
        chosen_by: What decided them, such as the model and version doing the
            reading. Stored alongside, so it is clear later what judged this.

    Returns:
        A dictionary with the `timeline_id`, how many `sections` were stored,
        and `topics`, each with how many sections carry it and how many
        seconds they cover between them.

    Raises:
        ValueError: If the timeline does not exist, no sections were given, or
            the sections do not cover the footage or start where they may. The
            message lists every problem at once.
    """
    if not sections:
        raise ValueError("give at least one section; to clear them, build the timeline again with rebuild")
    timeline = _require_timeline(timeline_id)
    sections = [SectionChoice.model_validate(with_whole_clip_ids(choice.model_dump(), timeline.id))
                for choice in sections]
    clips = _timeline_utterances(timeline)
    candidates = propose_candidates(clips, _timeline_analyses(timeline))

    problems = check_sections([(choice.first_clip_id, choice.last_clip_id) for choice in sections], clips, candidates)
    if problems:
        raise ValueError("these sections were not stored:\n- " + "\n- ".join(problems))

    built, utterances = build_sections(timeline.id, clips, sections)
    timeline = timeline.model_copy(update={
        "levels": [ClipLevel.UTTERANCE, ClipLevel.SECTION],
        "sections_by": chosen_by,
        "candidate_hash": candidate_hash(candidates),
    })
    runtime.repo.save_semantic_timeline(timeline, [*utterances, *built])

    topics: dict[str, dict] = {}
    for section in built:
        if section.topic:
            entry = topics.setdefault(section.topic, {"sections": 0, "seconds": 0.0})
            entry["sections"] += 1
            entry["seconds"] = round(entry["seconds"] + section.duration, 3)
    return {"timeline_id": timeline.id, "sections": len(built), "topics": topics}

@mcp.tool()
def get_semantic_clip(clip_id: str, include_words: bool = False, timeline_id: Optional[str] = None) -> dict:
    """Read one semantic clip in full.

    Use this on the few clips a `query_clips` search turned up that are worth
    a closer look: the whole text rather than the summary's opening, and, when
    a cut has to land mid-sentence, the timing of every word.

    `safe_in` and `safe_out` are the point of this record. They say how far
    the clip's start may move earlier and its end may move later while staying
    inside the silence around it, measured from the audio rather than guessed.
    A cut placed within them does not clip a word; one placed past them does.

    Args:
        clip_id: ID of the clip, as `query_clips` returned it.
        include_words: Whether to include each word's timing. Leave it off
            unless cutting inside a sentence.
        timeline_id: The timeline the clip is in; left out, the newest
            build that has the clip.

    Returns:
        The clip: its `clip_id`, `timeline_id`, `asset_id`, `level`, `kind`,
        `source_range`, `safe_in` / `safe_out`, full `text`, `speaker`,
        `scores` (measured as `query_clips`' `min_scores` describes: sound
        levels in dBFS), and `tags` with where each came from. With `include_words`,
        also `words`, each with `start`, `end`, `text` and `probability`, how
        sure the speech model was of it; `unsure`, the words it was guessing
        at, worth `listen_again`; and `unsure_unknown` when the file was
        transcribed before that was kept.

    Raises:
        ValueError: If no clip has that ID.
    """
    clip = _semantic_clip(clip_id, timeline_id)
    record = clip.model_dump(exclude={"id"})
    record["clip_id"] = clip.id
    if include_words:
        analysis = runtime.repo.get_analysis(clip.asset_id)
        transcript = analysis.transcript if analysis is not None else None
        words = [
            word
            for segment in (transcript.segments if transcript else [])
            for word in segment.words
            if word.end > clip.source_range.start and word.start < clip.source_range.end
        ]
        record["words"] = [word.model_dump() for word in words]
        # What `get_analysis` marks with ⟦ ⟧, so deciding what to hear again needs only this (#83).
        record["unsure"] = unsure_words(words)
        if words and all(word.probability is None for word in words):
            record["unsure_unknown"] = ("this file was transcribed before how sure the speech model was of each "
                                        "word was kept; analyze_asset again to have it")
    return shared.plain(record)

@mcp.tool()
def frames_for_clips(
    clip_ids: List[str],
    columns: Annotated[int, Field(ge=1, le=8)] = 4,
    timeline_id: Optional[str] = None,
) -> ToolResult:
    """Look at one frame from each of several semantic clips, as a labeled grid.

    This server has no model that can see, so describing what is on screen is
    yours to do: read the sheet, then write what you saw back with
    `set_clip_tags`, and from then on those descriptions can be searched like
    anything else.

    The frames are returned to you as a tool result, which means they leave
    this machine if you are not running on it. Say so before using this on
    footage the user may consider private.

    Args:
        clip_ids: Clips to sample, in the order to show them. One frame is
            taken from the middle of each.
        columns: Most tiles per row.
        timeline_id: The timeline the clips are in; left out, the newest
            build that has the clips.

    Returns:
        A text block naming each tile's clip and time, followed by the grid as
        a JPEG image.

    Raises:
        ValueError: If no clips were given, there are too many, a clip does
            not exist, or its asset has no picture.
        RuntimeError: If a frame cannot be decoded.
    """
    if not clip_ids:
        raise ValueError("give at least one clip to look at")
    if len(clip_ids) > shared.MAX_STORYBOARD_TILES:
        raise ValueError(f"{len(clip_ids)} clips is more than one sheet holds; ask for at most {shared.MAX_STORYBOARD_TILES}")

    shots: List[tuple[str, float, str]] = []
    listing: List[str] = []
    for position, clip_id in enumerate(clip_ids):
        clip = _semantic_clip(clip_id, timeline_id)
        asset = shared.get_asset(clip.asset_id)
        if not asset.has_video:
            raise ValueError(f"clip {clip_id} comes from {asset.id}, which has no picture to show")
        middle = (clip.source_range.start + clip.source_range.end) / 2
        if asset.duration is not None:
            middle = min(middle, float(asset.duration) - 0.05)
        seconds = round(max(middle, 0.0), 3)
        shots.append((asset.path, seconds, f"#{position + 1}  {format_timestamp(seconds)}"))
        listing.append(f"#{position + 1}: {clip_id} | {clip.kind.value} | {seconds:.3f}s ({format_timestamp(seconds)})")

    return ToolResult(content=[
        "One frame from the middle of each clip, in reading order:\n" + "\n".join(listing),
        Image(data=storyboard_sheet(shots, columns=columns), format="jpeg"),
    ])

@mcp.tool()
def set_clip_tags(
    descriptions: List[ClipDescription],
    written_by: str,
    reviewed: bool = False,
    timeline_id: Optional[str] = None,
) -> dict:
    """Write back what you saw in a clip, so it can be searched later.

    These descriptions become facts that later choices are made from — which
    B-roll covers which line, what the footage is said to contain. A model
    looking at a thumbnail gets things wrong, so show the user what you are
    about to write and let them correct it before calling this. `reviewed`
    says you did.

    Where each claim came from is stored with it, because a label somebody
    read off a picture and a number a detector measured are not worth the
    same later.

    Writing again replaces what the same author wrote before and leaves other
    authors' tags alone, so a correction from the user does not wipe out what
    a detector measured.

    Args:
        descriptions: One entry per clip, each with what is on screen and any
            short tags to search on.
        written_by: Who or what is writing, such as the model and version that
            read the frames, or the user's name when they dictated a fix.
        reviewed: True once the user has seen these and agreed to them.
        timeline_id: The timeline the clips are in; left out, the newest
            build that has the clips.

    Returns:
        A dictionary with how many clips were `updated` and the total `tags`
        now on them.

    Raises:
        ValueError: If nothing was given, a clip does not exist, or the
            descriptions have not been reviewed.
    """
    if not descriptions:
        raise ValueError("give at least one clip to describe")
    if not reviewed:
        raise ValueError(
            "show these descriptions to the user and get their agreement first, then call again with reviewed: true; "
            "they are stored as facts about the footage and later choices are made from them"
        )

    written: List[SemanticClip] = []
    for entry in descriptions:
        clip = _semantic_clip(entry.clip_id, timeline_id)
        source = TagSource.USER if written_by == "user" else TagSource.MODEL
        kept = [tag for tag in clip.tags if not (tag.source == source and tag.model_version == written_by)]
        written.append(clip.model_copy(update={
            "description": entry.description or clip.description,
            "described_by": written_by if entry.description else clip.described_by,
            "tags": [*kept, *(Tag(value=value, source=source, model_version=written_by) for value in entry.tags)],
        }))

    runtime.repo.update_semantic_clips(written)
    return {"updated": len(written), "tags": sum(len(clip.tags) for clip in written)}
