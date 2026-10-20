"""Captions: making them from what was said in the cut, and reading them back."""

from typing import Annotated, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import Field

from app.models.media import MediaAnalysis, Span, SpeakerTurn
from app.models.timeline.apply import mostly_over, retimed_words
from app.models.timeline.operations import SetSubtitlesOp
from app.models.timeline.parts import Project, SubtitleCue
from app.engine.analysis import marked_words, unsure_words
from app.engine.semantic import join_voices
from app.engine.frames import format_timestamp
from app.engine.subtitles import (
    DEFAULT_MAX_CHARACTERS, DEFAULT_MAX_SECONDS, caption_geometry, captioned_clips, place_cues, timeline_cues,
)
from app import runtime
from app.runtime import mcp
from app.tools import editing, examining, shared

def _joined_people(project: Project, heard: Mapping[str, Sequence[SpeakerTurn]]) -> Dict[Tuple[str, str], str]:
    """Work out which person each file's speaker label is, across the files of a project.

    Joined over the same files the clips' own labels were joined over, so a
    caption and the clip it sits on cannot call one person by two names: the
    joined labels are numbered in the order the files are walked, and joining
    a different set of files can number the same person differently. For a
    cut compiled from a plan, that set is the plan's timeline; for one put
    together by hand, it is the files being captioned.

    Args:
        project: The project being captioned.
        heard: The speaker turns of each file being captioned, keyed by asset
            ID.

    Returns:
        The joined label, `V1` and so on, for each `(asset_id, label)`. A file
        outside that set, or one analyzed before voices were kept, is absent,
        and its captions keep the file's own labels.
    """
    compiled_from = {clip.from_plan_id for track in project.tracks for clip in track.clips if clip.from_plan_id}
    files: set = set()
    for plan_id in sorted(compiled_from):
        plan = runtime.repo.get_plan(plan_id)
        timeline = runtime.repo.get_semantic_timeline(plan.timeline_id) if plan is not None else None
        if timeline is not None:
            files |= set(timeline.asset_ids)
    analyses = {
        asset_id: analysis
        for asset_id in (files or set(heard))
        if (analysis := runtime.repo.get_analysis(asset_id)) is not None
    }
    return join_voices(analyses)

@mcp.tool()
def generate_subtitles(
    project_id: str,
    expected_version: int,
    max_characters: Annotated[int, Field(ge=8, le=120)] = DEFAULT_MAX_CHARACTERS,
    max_seconds: Annotated[float, Field(gt=0.2, le=15)] = DEFAULT_MAX_SECONDS,
    fix_words: Optional[Dict[str, str]] = None,
    note: str = "",
) -> dict:
    """Caption the edited sequence from transcripts already made, and store the captions.

    Reads the transcript of every clip on the base video track and on the
    audio tracks, keeps only the speech that survived the edit, and moves each
    line to where it now falls in the result. A narration recorded separately
    and laid on an audio track is captioned like any other speech; music is
    not, because nobody transcribes a song. Insets on video tracks above the
    base are pictures over that sound, so they are not captioned, and neither
    is a clip turned all the way down. Long sentences are broken between
    words, and while the caption style keeps to one line (its `single_line`,
    on unless turned off) never wider than one line of this project's frame
    at that style's size, so each caption checked is one line on screen. Set
    the caption style first for that to be the right width.

    Wherever somebody has already written or corrected captions for the same
    stretch of a file — in this project or any other — those are used
    instead of the transcript, so a correction is made once. Every word in
    the workspace glossary is corrected as well; `fix_words` adds to it.

    The captions replace any the project had, and come back one line each
    for proofreading: correct a line with `edit_subtitle`, add one the
    transcript missed with `add_subtitle`. Render them into the picture with
    `render_project` and `burn_subtitles`.

    Args:
        project_id: ID of the project to caption.
        expected_version: Version you last read from `get_project`.
        max_characters: Longest caption text before it is broken in two.
        max_seconds: Longest caption before it is broken in two.
        fix_words: Words the transcriber gets wrong, mapped to what was
            meant — {"Packet": "Pocket", "裁風寺": "裁縫師"}. Kept for every
            later caption in this workspace; an empty meaning takes a word
            out of the glossary.
        note: The user's own words for this, kept in the project's history.

    Returns:
        A dictionary with the project's `new_version`; `captions`, each as
        one line — its ID, where it lands in the cut, and what it says;
        `reused`, how many came from captions already reviewed, and
        `from_other_projects`, each of those as a line with the project it was
        written in — read them back, since words put to another cut are wrong
        in this one; `sources`,
        how many came from each of `reviewed`, `this_project` and
        `transcript`; `to_check`, the IDs of the ones straight from the
        transcript, those with the most words the speech model was unsure of
        first — proofread those, the others somebody has read; `glossary`,
        the words corrected; `assets_without_transcript`, the video sources
        that still need `analyze_asset` before they can be captioned;
        `left_out_as_deleted`, the lines left out because a caption over
        the same words was deleted with `edit_subtitle` (`add_subtitle`
        there puts one back); and `overlapping`, how many captions land on
        top of the one before them once the cut puts them on screen, which is
        worth a look when a narration track talks over footage that speaks
        for itself.

    Raises:
        ValueError: If the project does not exist, or nothing it plays has
            been transcribed.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")

    transcripts = {}
    silences: dict[str, List[Span]] = {}
    speakers: dict[str, List[SpeakerTurn]] = {}
    untranscribed: List[str] = []
    base = project.base_video_track
    # Reported as a gap only for the sequence itself: a music bed has no transcript and
    # never needs one. Read off the same walk the captions come from, so a clip that is
    # turned all the way down cannot be called a gap in one place and skipped in the other.
    sequence = {clip.asset_id for clip in base.clips if clip.volume != 0} if base else set()
    for clip in captioned_clips(project):
        if clip.asset_id in transcripts or clip.asset_id in untranscribed:
            continue
        analysis = runtime.repo.get_analysis(clip.asset_id)
        if analysis is not None and analysis.transcript is not None:
            transcripts[clip.asset_id] = analysis.transcript
            silences[clip.asset_id] = analysis.silences
            speakers[clip.asset_id] = analysis.speakers
        elif clip.asset_id in sequence:
            untranscribed.append(clip.asset_id)

    if not transcripts:
        # Two different setups end up here and they need different advice: footage that
        # was never put through speech recognition, and footage that was but is silent in
        # the cut. Telling someone to analyze a file they already analyzed sends them off
        # to redo work that was never the problem.
        if not captioned_clips(project):
            raise ValueError(
                f"project {project_id} has nothing audible to caption; every clip that could "
                "carry words is turned all the way down, or there are none"
            )
        raise ValueError(
            f"project {project_id} has no transcribed clips to caption; "
            "call analyze_asset on its sources first"
        )
    people = _joined_people(project, speakers)
    speakers = {
        asset_id: [
            turn.model_copy(update={"speaker": people.get((asset_id, turn.speaker), turn.speaker)})
            for turn in turns
        ]
        for asset_id, turns in speakers.items()
    }
    style = project.caption_style
    glossary = runtime.repo.update_glossary(fix_words) if fix_words else runtime.repo.glossary()
    proposed = [
        _corrected(cue, glossary) for cue in timeline_cues(
            project, transcripts, max_characters=max_characters, max_seconds=max_seconds,
            silences=silences, speakers=speakers,
            max_units=caption_geometry(project.width, project.height, style).max_units if style.single_line else None,
            min_seconds=style.min_seconds,
        )
    ]
    reviewed = {asset_id: runtime.repo.reviewed_captions(asset_id) for asset_id in {cue.asset_id for cue in proposed}}
    reviewed.update({
        clip.asset_id: runtime.repo.reviewed_captions(clip.asset_id)
        for clip in (base.clips if base else []) if clip.asset_id not in reviewed
    })
    # What this project kept to itself comes first, then what was corrected on the footage
    # for every project, then the transcript: each fills only what the ones before left.
    # A correction stored before captions had to be said to be for every project is no longer
    # handed on, but one this project already shows is its own: kept, not lost to the transcript.
    unclaimed = {asset_id: runtime.repo.unclaimed_captions(asset_id) for asset_id in {cue.asset_id for cue in project.subtitles}}
    local = [
        cue.model_copy(update={"id": "", "local": True}) for cue in project.subtitles
        if cue.local or (cue.remembered_in is None and any(
            _overlaps(cue, old) and old.text == cue.text for old in unclaimed[cue.asset_id]))
    ]

    def taken(cue: SubtitleCue, over: Sequence[SubtitleCue]) -> bool:
        return any(cue.asset_id == known.asset_id and _overlaps(cue, known) for known in over)

    def deleted(cue: SubtitleCue) -> bool:
        # Deleted by hand in this project: kept out however the lines now break (#79).
        return any(mostly_over(cue, gone) for gone in project.deleted_subtitles)

    kept = [
        cue for cue in proposed
        if not any(_overlaps(cue, known) for known in reviewed.get(cue.asset_id, ())) and not taken(cue, local)
    ]
    reused = [known.model_copy(update={"id": ""}) for known_list in reviewed.values() for known in known_list]
    reused = [cue for cue in reused if place_cues(project, [cue]) and not taken(cue, local)]
    left_out = [cue for cue in [*kept, *reused] if deleted(cue)]
    kept, reused = [cue for cue in kept if not deleted(cue)], [cue for cue in reused if not deleted(cue)]
    # Each caption's own `local` stands: the transcript's and the remembered ones are the
    # footage's, for every project, and this project's stay its own. Nothing is remembered
    # from here; only `apply_edits` writes to what every project starts from.
    saved = editing.apply(project_id, expected_version, [SetSubtitlesOp(cues=[*kept, *reused, *local], for_all_projects=True)])
    # Counted where the captions land, not where the words were said: two lines from
    # different files overlap only once the cut puts them on screen together.
    placed = place_cues(saved, saved.subtitles)
    overlapping = sum(1 for earlier, later in zip(placed, placed[1:]) if later.start < earlier.end)
    heard = _caption_sources(saved)
    from_transcript = [cue for cue in saved.subtitles if heard[cue.id]["from"] == "transcript"]
    first_shown = {}
    for placed_cue in placed:
        first_shown.setdefault(placed_cue.cue_id, placed_cue.start)
    to_check = sorted(
        (cue for cue in from_transcript if cue.id in first_shown),
        key=lambda cue: (-len(heard[cue.id].get("unsure", [])), first_shown[cue.id]),
    )
    return {
        "new_version": saved.version,
        "captions": [
            f"{cue.cue_id} {format_timestamp(float(cue.start))} {cue.text}"
            + (f" / {cue.secondary}" if cue.secondary else "")
            for cue in placed
        ],
        "reused": len(reused),
        "from_other_projects": [
            f"{cue.id} {cue.text} — 「{_project_name(cue.remembered_in)}」"
            for cue in saved.subtitles if heard[cue.id]["from"] == "reviewed" and cue.remembered_in != project_id
        ],
        "sources": {source: sum(1 for cue in saved.subtitles if heard[cue.id]["from"] == source)
                    for source in ("reviewed", "this_project", "transcript")},
        "to_check": [cue.id for cue in to_check],
        "glossary": glossary,
        "assets_without_transcript": untranscribed,
        "overlapping": overlapping,
        "left_out_as_deleted": [cue.text for cue in left_out],
    }

def _project_name(project_id: Optional[str]) -> str:
    """A project's name to show the user, or what to say when it is gone."""
    found = runtime.repo.get_project(project_id) if project_id else None
    return found.name if found is not None else "a project since deleted"

# How much of a file two captions, or a caption and a word, may share or miss by and still be
# told apart: the rounding their times went through, not a judgement.
CAPTION_SLIVER = 0.05

def _caption_sources(project: Project) -> Dict[str, dict]:
    """Say where each of a project's captions came from, and what in it is worth checking.

    Worked out each time rather than stored: a caption kept to this project is
    this project's; one that matches what is remembered against its footage
    was written or corrected by somebody; anything else is the transcript as
    it was heard, and only those carry the words the speech model was unsure
    of.

    Args:
        project: The project.

    Returns:
        For each caption ID, `from` — `this_project`, `reviewed` or
        `transcript` — and for a transcript one with unsure words, `unsure`,
        those words, and `marked`, its text with them in ⟦ ⟧ when the text is
        still the transcript's own.
    """
    remembered: Dict[str, List[SubtitleCue]] = {}
    analyses: Dict[str, Optional[MediaAnalysis]] = {}
    sources: Dict[str, dict] = {}
    for cue in project.subtitles:
        if cue.local:
            sources[cue.id] = {"from": "this_project"}
            continue
        if cue.asset_id not in remembered:
            remembered[cue.asset_id] = runtime.repo.reviewed_captions(cue.asset_id)
        if any(_overlaps(cue, known) and known.text == cue.text for known in remembered[cue.asset_id]):
            sources[cue.id] = {"from": "reviewed"}
            continue
        if cue.asset_id not in analyses:
            analyses[cue.asset_id] = runtime.repo.get_analysis(cue.asset_id)
        analysis = analyses[cue.asset_id]
        # The same sliver `_overlaps` allows: a caption's edges were cut from these words' times.
        words = [
            word for segment in (analysis.transcript.segments if analysis and analysis.transcript else [])
            for word in segment.words
            if word.start >= float(cue.source_start) - CAPTION_SLIVER and word.end <= float(cue.source_end) + CAPTION_SLIVER
        ]
        found: dict = {"from": "transcript"}
        doubted = unsure_words(words)
        if doubted:
            found["unsure"] = doubted
            if examining.bare_words("".join(word.text for word in words)) == examining.bare_words(cue.text):
                found["marked"] = marked_words(words)
        sources[cue.id] = found
    return sources

def _overlaps(first: SubtitleCue, second: SubtitleCue) -> bool:
    """Say whether two captions of one file cover some of the same words.

    Args:
        first: One caption.
        second: The other.

    Returns:
        True when they share more than a sliver of the file.
    """
    overlap = min(first.source_end, second.source_end) - max(first.source_start, second.source_start)
    return float(overlap) > CAPTION_SLIVER

def _corrected(cue: SubtitleCue, glossary: Mapping[str, str]) -> SubtitleCue:
    """Correct a proposed caption's text with the workspace glossary.

    Args:
        cue: The caption as proposed.
        glossary: Heard words mapped to meant ones.

    Returns:
        The caption with every glossary word replaced, and its word timings
        kept where the correction left them fitting.
    """
    text = shared.glossed(cue.text, glossary)
    if text == cue.text:
        return cue
    return cue.model_copy(update={"text": text, "words": retimed_words(cue.words, text)})

@mcp.tool()
def get_subtitles(
    project_id: str,
    start: Annotated[float, Field(ge=0)] = 0,
    end: Annotated[Optional[float], Field(ge=0)] = None,
    words: bool = False,
) -> dict:
    """Read the captions stored on a project, a window at a time.

    A long video has hundreds of captions, so read the stretch you need rather
    than all of them. Each caption's `cue_id` is what `edit_subtitle` takes, so
    one wrong word can be corrected on its own.

    Captions are stored against the footage they transcribe, so this works out
    where each one falls in the cut as it stands. One whose words the edit cut
    out is not here, because it does not appear; one whose words survived in
    two places is here twice, under the same `cue_id`, and correcting it
    corrects both.

    Args:
        project_id: ID of the project to read.
        start: Start of the window in seconds on the timeline.
        end: End of the window in seconds; omit for the rest of the video.
        words: Include each caption's word timings, which are long and only
            needed to check how a caption lights up word by word.

    Returns:
        A dictionary with `cues` in the window — each with its `cue_id`,
        `start` and `end` on the timeline, `text`, and `from`: `reviewed`
        (somebody wrote or corrected it), `this_project` (kept to this
        project) or `transcript` (as heard, not yet read by anyone); a
        transcript one the speech model was unsure of also has `unsure`, those
        words, and `marked`, its text with them in ⟦ ⟧ — for reading only,
        never to write back — and `placed`, how many
        land anywhere in the cut, `stored`, how many captions the project
        holds, and the `window` that was read. `stored` above `placed` means
        some captions belong to footage the edit dropped.

    Raises:
        ValueError: If the project does not exist, or `end` is not after
            `start`.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    placed = place_cues(project, project.subtitles)
    if end is None:
        limit = float(project.duration)
    else:
        limit = end
        if limit <= start:
            raise ValueError(f"the window ends at {limit}s, which is not after its start at {start}s")
    heard = _caption_sources(project)
    window = [
        shared.plain({**cue.model_dump(exclude=None if words else {"words"}), **heard.get(cue.cue_id, {})})
        for cue in placed
        if float(cue.end) > start and float(cue.start) < limit
    ]
    return {
        "cues": window,
        "placed": len(placed),
        "stored": len(project.subtitles),
        "window": {"start": start, "end": limit},
    }
