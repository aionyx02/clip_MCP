import os
import re
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Callable, List, Literal, Optional, Sequence

from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp.server.transforms import ResourcesAsTools
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field, TypeAdapter
from app.models.media import Asset, Measured, MediaAnalysis, Span, SpeakerTurn
from app.models.plan import EditPlan, PlanAmendment, apply_amendment
from app.models.semantic import (
    ClipDescription, ClipKind, ClipLevel, SectionChoice, SemanticClip, SemanticTimeline, Tag, TagSource,
)
from app.models.timeline import (
    EditOperation,
    Project,
    SetSubtitlesOp,
    TrackType,
    apply_operation,
    validate_project,
)
from app.models.job import Job, JobKind, JobStatus
from app.engine import resources
from app.engine.analysis import current_recipe, sound_note, whisper_model_name
from app.engine.diarize import speaker_model_name
from app.engine.faces import framing_note
from app.engine.plan import (
    BROLL_TRACK_ID, MUSIC_TRACK_ID, VIDEO_TRACK_ID, broll_covers, broll_slots, check_plan, check_recompile,
    compile_operations, compiled_duration, diff_plans, plan_pieces,
)
from app.engine.sections import build_sections, candidate_hash, check_sections, propose_candidates
from app.engine.semantic import (
    build_timeline, clean_cuts, content_scores, join_voices, timeline_input_hash, voice_levels,
)
from app.engine.probe import probe_file
from app.engine.builder import DEFAULT_LOUDNESS_TARGET, FFmpegRenderer
from app.engine.frames import format_timestamp, storyboard_sheet
from app.engine.subtitles import (
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_MAX_SECONDS,
    build_ass,
    captioned_clips,
    place_cues,
    timeline_cues,
)
from app.engine.renderer import JobManager
from app.storage.repo import Repository

WORKSPACE_DIR = os.path.abspath(os.environ.get("CLIP_MCP_WORKSPACE", "workspace"))
SKILLS_DIR = Path(__file__).parent / "skills"
MAX_STORYBOARD_TILES = 36
MAX_CLIP_RESULTS = 200
# Reading a whole timeline at once: an hour of talk is a few hundred utterances.
MAX_TIMELINE_CLIPS = 5000
# Characters a Windows or POSIX filesystem will not take in a name, plus the control
# range. A project is named by whoever is editing, so a render cannot assume anything
# about what is in that name.
FORBIDDEN_IN_NAMES = frozenset('<>:"/\\|?*') | {chr(code) for code in range(32)}
MAX_OUTPUT_NAME = 60
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
MEDIA_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".mts", ".m2ts", ".wmv", ".flv", ".3gp",
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff",
})

mcp = FastMCP(
    "VideoEditingCore",
    instructions=(
        "Timeline-based video editing backed by FFmpeg. Before planning edits, "
        "read the clip-editing skill at skill://clip-editing/SKILL.md (clients "
        "without resource support can call read_resource with that URI). It "
        "explains how to turn editing requests into tool calls, how to read "
        "footage with analyze_asset, get_analysis, and view_frames, how to "
        "check an edit with preview_project before rendering, how to work "
        "through a folder of unedited footage when the user has no plan, "
        "what to ask when a request is incomplete, how music ducks under "
        "speech and how every render is normalized to one consistent "
        "loudness, how to caption an edit with generate_subtitles and "
        "burn_subtitles, and the current limits: no overlapping picture on a "
        "track, no still images, and no graphics beyond captions. "
        "Transitions end on the cut rather than straddling it, so adding one "
        "never changes how long the video runs. The person using this server is "
        "editing their own video, not writing code: reply in plain language "
        "with no tool names, IDs, or JSON, and read the plan back in one "
        "sentence for confirmation before the first render. Projects, assets, "
        "analyses, and jobs are saved on disk and survive server restarts."
    ),
)
# "resources" lists the supporting files alongside SKILL.md, so a client sees them
# without having to read the manifest first.
mcp.add_provider(SkillsDirectoryProvider(roots=SKILLS_DIR, supporting_files="resources"))
mcp.add_transform(ResourcesAsTools(mcp))

# Compiled operations are validated the same way a client's are, so a plan cannot reach
# the timeline through a door the tool surface does not have.
_OPERATIONS = TypeAdapter(List[EditOperation])

repo = Repository(os.path.join(WORKSPACE_DIR, "clip_mcp.db"))
job_manager = JobManager(repo)
renderer = FFmpegRenderer()

def _referenced_assets(project: Project) -> dict[str, Asset]:
    """Fetch the assets referenced by a project's clips.

    Args:
        project: Project whose clips are inspected.

    Returns:
        The registered assets used by the project, keyed by asset ID. IDs that
        are not registered are omitted.
    """
    return repo.get_assets(clip.asset_id for track in project.tracks for clip in track.clips)

def _get_asset(asset_id: str) -> Asset:
    """Fetch a registered asset.

    Args:
        asset_id: ID of the asset.

    Returns:
        The asset.

    Raises:
        ValueError: If no asset with `asset_id` exists.
    """
    asset = repo.get_assets([asset_id]).get(asset_id)
    if asset is None:
        raise ValueError(f"asset {asset_id} not found")
    return asset

def _covering(records: Sequence[Measured], start: float, end: float) -> list:
    """Select the records that reach into a time range.

    Args:
        records: Records to filter; anything with `start` and `end`.
        start: Range start in seconds.
        end: Range end in seconds.

    Returns:
        The overlapping records themselves, for a caller that has to read them.
    """
    return [record for record in records if record.end > start and record.start < end]

def _overlapping(spans: Sequence[Measured], start: float, end: float) -> List[dict]:
    """Select the spans that overlap a time range.

    Args:
        spans: Spans to filter; anything with `start` and `end` attributes.
        start: Range start in seconds.
        end: Range end in seconds.

    Returns:
        The overlapping spans, serialized as dictionaries.
    """
    return [span.model_dump() for span in _covering(spans, start, end)]

@mcp.tool()
def inspect_media(filepath: str) -> dict:
    """Inspect a local media file and return its technical metadata.

    Use this to check duration, resolution, frame rate, codecs, and audio
    streams before adding a file to a project.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.

    Returns:
        ffprobe output with a `format` object and a `streams` list.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    return probe_file(filepath)

def _natural_key(name: str) -> List[object]:
    """Build a sort key that orders embedded numbers by value.

    Camera and phone exports are often numbered without padding, so plain
    text order would put `clip10` before `clip2`.

    Args:
        name: File name to build a key for.

    Returns:
        The name split into text and integer parts, lowercased.
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

def _register_asset(filepath: str) -> Asset:
    """Probe a media file and save it as an asset.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.

    Returns:
        The saved asset. A path that was imported before keeps its asset ID.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    path = os.path.abspath(filepath)
    info = probe_file(path)

    existing = repo.find_asset_by_path(path)
    streams = info.get("streams", [])
    duration = info.get("format", {}).get("duration")
    asset = Asset(
        id=existing.id if existing else str(uuid.uuid4()),
        path=path,
        duration=Decimal(duration) if duration is not None else None,
        has_video=any(
            stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")
            for stream in streams
        ),
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )
    repo.save_asset(asset)
    return asset

@mcp.tool()
def import_asset(filepath: str) -> dict:
    """Register a local media file as an asset that clips can reference.

    Importing the same path again re-probes the file and keeps its asset ID.
    Video-track clips need `has_video`, and audio-track clips, such as
    background music, need `has_audio`. Cover art embedded in audio files is
    not counted as video. Video assets without audio contribute silence.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.

    Returns:
        The asset serialized as a dictionary: its `id`, absolute `path`,
        `duration` in seconds (or null if unknown), and `has_video` /
        `has_audio` flags.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    return _register_asset(filepath).model_dump()

@mcp.tool()
def list_assets() -> dict:
    """List all imported assets, and say which of them have been analyzed.

    Returns:
        A dictionary with an `assets` list, each item in the format
        `import_asset` returns plus `analyzed` and `stale`. `stale` is true
        for an asset analyzed with detection settings or a speech model the
        server no longer uses: its analysis still works, but it was measured
        with different instruments from a fresh one, so analyze those assets
        again before comparing them with each other.
    """
    assets = []
    for asset in repo.list_assets():
        analysis = repo.get_analysis(asset.id)
        assets.append({
            **asset.model_dump(),
            "analyzed": analysis is not None,
            "stale": analysis is not None and _is_stale(analysis),
        })
    return {"assets": assets}

@mcp.tool()
def import_folder(folderpath: str, recursive: bool = False) -> dict:
    """Register every media file in a folder as an asset, in one call.

    Use this when the user points at a folder of footage instead of naming
    files. Files are returned in natural name order, so `clip2` comes before
    `clip10`, which is also the order to put them on the timeline unless the
    user says otherwise. A file that cannot be read is reported in `skipped`
    rather than failing the whole folder, and importing a folder again keeps
    the asset IDs it already handed out.

    Args:
        folderpath: Path to the folder, absolute or relative to the server's
            working directory.
        recursive: Whether to include media in sub-folders.

    Returns:
        A dictionary with `assets` (each in the format `import_asset` returns),
        `total_duration` in seconds, and `skipped`, a list of `path` and
        `reason` for files that could not be read.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    folder = os.path.abspath(folderpath)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"folder {folder} not found")

    paths: List[str] = []
    if recursive:
        for directory, _, names in os.walk(folder):
            paths.extend(os.path.join(directory, name) for name in names)
    else:
        paths = [os.path.join(folder, name) for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]
    paths = sorted(
        (path for path in paths if os.path.splitext(path)[1].lower() in MEDIA_EXTENSIONS),
        key=lambda path: (_natural_key(os.path.dirname(path)), _natural_key(os.path.basename(path))),
    )

    assets: List[Asset] = []
    skipped: List[dict] = []
    for path in paths:
        try:
            assets.append(_register_asset(path))
        except (OSError, RuntimeError, ValueError) as error:
            skipped.append({"path": path, "reason": str(error)})

    total = sum((asset.duration for asset in assets if asset.duration is not None), Decimal(0))
    return {"assets": [asset.model_dump() for asset in assets], "total_duration": total, "skipped": skipped}

def _is_stale(analysis: MediaAnalysis) -> bool:
    """Say whether an analysis was taken in a way the server no longer uses.

    Args:
        analysis: Analysis to check.

    Returns:
        True when the detection thresholds, the measurements taken, or the
        speech model have changed since it was made. Nothing in it is wrong;
        it just answers a slightly different question from a fresh one, which
        matters most when two assets are being compared with each other.
    """
    return analysis.recipe.differs_from(current_recipe(whisper_model_name(), speaker_model=speaker_model_name()))

@mcp.tool()
def analyze_asset(
    asset_id: str,
    transcribe: bool = True,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    chinese_variant: Optional[Literal["zh-TW", "zh-HK", "zh-Hant", "zh-Hans"]] = None,
    diarize: bool = True,
    speakers: Annotated[Optional[int], Field(ge=1, le=20)] = None,
) -> dict:
    """Start analyzing what an asset contains, in the background.

    One decoding pass detects scene changes, black and frozen picture, and
    silences, and alongside them measures each shot — exposure, contrast,
    blur, motion, camera shake — each second of sound (level, peak, noise
    floor, and how squared off the waveform is where it peaks), and each
    second of picture for faces: how many, how big the largest is, and where
    it sits in frame. If `transcribe` is true and the asset has audio, speech
    is then transcribed locally with word-level timestamps that stay accurate
    on long recordings, which is the slow part by far, and the voices are told
    apart so that each clip knows who was speaking.

    Returns immediately: poll `get_job` until the job completes, then read the
    results with `get_analysis`, or search them through `query_clips` once a
    semantic timeline is built. Analyzing again replaces earlier results.

    Everything runs on this machine. The models are downloaded the first time
    they are needed — the speech model is the large one, the rest are tens of
    megabytes — and they are kept inside the workspace, so deleting the
    workspace deletes them too. `stage` says when a download is what a job is
    waiting on.

    Args:
        asset_id: ID of the asset to analyze.
        transcribe: Whether to transcribe speech.
        language: Spoken language code, such as "zh" or "en"; omit to detect it.
        prompt: Text that guides transcription, such as names and terms that
            appear in the recording.
        diarize: Whether to tell the voices apart. Only done alongside a
            transcript, since a speaker label with nothing said under it has
            nowhere to go. Turn it off for footage with one person in it if
            the extra minute matters.
        speakers: How many people are talking, when that is known. Worth
            giving for an interview or a two-hander: told the number, the
            server has to find exactly that many, and cannot split one person
            who leaned closer to the microphone into two. Omit it to let the
            number be worked out.
        chinese_variant: Convert a Chinese transcript to this script and
            regional variant: "zh-TW" (Traditional, Taiwan characters and
            vocabulary, e.g. 視頻 becomes 影片), "zh-HK" (Traditional, Hong
            Kong), "zh-Hant" (Traditional, no vocabulary changes), or "zh-Hans"
            (Simplified). Omit to keep the speech model's output, which may mix
            scripts.

    Returns:
        A dictionary with the `job_id`, the initial `status`, and `stage`.
        Analyses run one or two at a time, so that several of them cannot
        exhaust the machine's memory between them; `stage` says what a job
        that has not started yet is waiting for, and is null when it started
        immediately. Waiting costs nothing and needs no action: keep polling
        `get_job` as usual.

    Raises:
        ValueError: If the asset does not exist or has no known duration.
    """
    asset = _get_asset(asset_id)
    if asset.duration is None:
        raise ValueError(f"asset {asset_id} has no known duration and cannot be analyzed")

    job = Job(kind=JobKind.ANALYZE, asset_id=asset_id)
    job.work_dir = os.path.join(WORKSPACE_DIR, "jobs", job.job_id)
    with_speech = transcribe and asset.has_audio
    job.memory_estimate = resources.analysis_memory_bytes(
        with_speech,
        whisper_model_name(),
        duration=float(asset.duration),
        diarize=with_speech and diarize,
        detect_faces=asset.has_video,
    )
    job = job_manager.start_job(job, {
        "transcribe": transcribe,
        "language": language,
        "prompt": prompt,
        "chinese_variant": chinese_variant,
        "diarize": diarize,
        "speakers": speakers,
    })
    return {"job_id": job.job_id, "status": job.status.value, "stage": job.stage}

@mcp.tool()
def get_analysis(
    asset_id: str,
    start: Annotated[float, Field(ge=0)] = 0,
    end: Optional[float] = None,
    include_words: bool = False,
) -> dict:
    """Return the analysis of an asset for a time range.

    Only items overlapping the range are returned. Transcripts of long
    recordings are large, so read them in windows of about 10 minutes. Times
    are seconds in the source file and can be used directly as a clip's
    `source_range`.

    Args:
        asset_id: ID of the analyzed asset.
        start: Range start in seconds.
        end: Range end in seconds; omit for the end of the asset.
        include_words: Whether to include word-level timings in transcript
            segments. Use them to cut exactly at a word; leave them out to
            save space.

    Returns:
        A dictionary with the asset `duration`, `analyzed_at`, and the
        overlapping `scenes`, `black_frames`, `frozen_frames`, `silences`,
        `shots`, and `transcript` (`language`, `model`, `chinese_variant`, and
        `segments` with `start`, `end`, and `text`), or null for `transcript`
        if speech was not transcribed.

        `shots` carries one record per shot in the range — `exposure`,
        `contrast`, `blur`, `motion` and `shake` — while `sound` and `faces`
        summarise the range's per-second records rather than listing them,
        since an hour of those is thousands of rows. `faces` says how much of
        the range had anybody on screen, how many there usually were, and
        where the largest face sat across the frame and how far it moved,
        which is what a vertical reframe needs. `speakers` lists the stretches
        each voice held; the labels are this file's own and mean nothing
        outside it.

        All of these are measurements and none of them is a verdict: whether a
        shot is too dark or too wobbly depends on what it is for.

        `recipe` says what measured it, and `stale` is true when the server
        has since changed how it measures. To pick material out of this, use
        `query_clips`: every measurement here is also averaged onto the
        semantic clips, where it can be filtered on.

    Raises:
        ValueError: If the asset has not been analyzed yet.
    """
    analysis = repo.get_analysis(asset_id)
    if analysis is None:
        raise ValueError(f"asset {asset_id} has not been analyzed; call analyze_asset first")
    end = analysis.duration if end is None else end

    transcript = None
    if analysis.transcript is not None:
        exclude = None if include_words else {"words"}
        transcript = {
            "language": analysis.transcript.language,
            "model": analysis.transcript.model,
            "chinese_variant": analysis.transcript.chinese_variant,
            "segments": [
                segment.model_dump(exclude=exclude)
                for segment in analysis.transcript.segments
                if segment.end > start and segment.start < end
            ],
        }
    return {
        "asset_id": asset_id,
        "duration": analysis.duration,
        "analyzed_at": analysis.analyzed_at.isoformat(),
        "recipe": analysis.recipe.model_dump(),
        "stale": _is_stale(analysis),
        "range": {"start": start, "end": end},
        "scenes": _overlapping(analysis.scenes, start, end),
        "black_frames": _overlapping(analysis.black_frames, start, end),
        "frozen_frames": _overlapping(analysis.frozen_frames, start, end),
        "silences": _overlapping(analysis.silences, start, end),
        "shots": _overlapping(analysis.shots, start, end),
        "sound": sound_note(_covering(analysis.sound, start, end)),
        "faces": framing_note(_covering(analysis.faces, start, end)),
        "speakers": _overlapping(analysis.speakers, start, end),
        "transcript": transcript,
    }

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

def _require_timeline(timeline_id: Optional[str]) -> SemanticTimeline:
    """Fetch a semantic timeline, or the newest one.

    Args:
        timeline_id: ID of the timeline, or `None` for the newest build.

    Returns:
        The timeline.

    Raises:
        ValueError: If it does not exist, or if nothing has been built yet.
    """
    timeline = repo.get_semantic_timeline(timeline_id)
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
        `total_clips`, `counts` of clips by level and kind, and `stale`.

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
    assets = {asset_id: _get_asset(asset_id) for asset_id in dict.fromkeys(asset_ids)}
    analyses = {asset_id: repo.get_analysis(asset_id) for asset_id in assets}
    missing = sorted(asset_id for asset_id, analysis in analyses.items() if analysis is None)
    if missing:
        raise ValueError(f"these assets have not been analyzed yet, call analyze_asset on them first: {', '.join(missing)}")

    stale = sorted(asset_id for asset_id, analysis in analyses.items() if _is_stale(analysis))
    existing = repo.find_semantic_timeline(timeline_input_hash(analyses))
    if existing is not None and not rebuild:
        return {
            "timeline_id": existing.id,
            "asset_ids": existing.asset_ids,
            "reused": True,
            "total_clips": sum(repo.count_semantic_clips(existing.id).values()),
            "counts": repo.count_semantic_clips(existing.id),
            "stale": stale,
        }

    timeline, clips = build_timeline(assets, analyses)
    repo.save_semantic_timeline(timeline, clips)
    return {
        "timeline_id": timeline.id,
        "asset_ids": timeline.asset_ids,
        "reused": False,
        "total_clips": len(clips),
        "counts": repo.count_semantic_clips(timeline.id),
    }

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
) -> dict:
    """Search the semantic timeline for the clips worth looking at.

    This is how to find material: ask for what is needed rather than reading a
    transcript end to end. Conditions combine, so "the parts of this file
    where someone speaks for more than four seconds and the picture is not
    black" is one call.

    Results are summaries. Each carries enough to decide whether a clip is
    wanted — what is said, how long it runs, how much room its edges have —
    and `get_semantic_clip` has the full text and word timings for the few
    that matter.

    The scores each clip carries are measurements, not opinions. A result
    shows the ones that say whether there is usable content here: `speech` and
    `silence` are the shares of the clip covered by words and by detected
    silence, `black` and `frozen` the shares the picture detectors marked, and
    `words_per_second` its pace. Pace counts whatever the transcript counts as
    a word, which for Chinese is characters, so compare it within one language
    rather than across two.

    A clip from a file whose voices were told apart also carries `speaker`,
    a label like `S1`. The labels are that file's own — `S1` in one recording
    is not `S1` in another — and a sentence that straddles a handover carries
    none at all rather than a guess.

    Each clip carries more than a result shows — how well it was shot, who was
    on screen, and what the sound was like — and `min_scores` and `max_scores`
    filter on those too. `exposure` is mean brightness from 0 to 1 and `contrast` the
    spread between the dark and bright ends; `blur` rises as the picture goes
    soft, with a sharp shot near 5; `motion` is how much changes from frame to
    frame, 0 for a locked-off shot; `shake` is how unsteady the camera itself
    was, near 0 on a tripod or an even pan and far higher handheld. For the
    sound, in decibels relative to full scale: `loudness`, `peak`,
    `noise_floor` — the hiss under everything — and `flatness`, which is high
    where the waveform is squared off at its peaks, so a `peak` near 0 with a
    high `flatness` is what a clipped recording looks like. For who is on
    screen: `faces` is how many were up, averaged over the clip, so 0.5 means
    somebody was there half the time; `face_share` is how much of the frame
    the largest one filled, which is the difference between a close-up and a
    wide; and `face_x` and `face_y` are where it sat, 0 to 1 across and down.
    Faces are found, not people, so `faces` of 0 means no face was seen and
    not that the shot is empty.

    None of these has a threshold behind it, because what counts as too dark
    or too wobbly depends on the shot. Filter on them to shorten a list, then
    read `get_semantic_clip` for the few that matter — it carries every score.

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
            cannot be said to pass.
        max_scores: Upper bounds on scores, such as `{"black": 0.1}` or
            `{"shake": 0.01}` for shots steady enough to hold on screen.
        limit: Largest number of clips to return.

    Returns:
        A dictionary with the `timeline_id` searched and the matching `clips`,
        each with its `clip_id`, `asset_id`, `kind`, `start` and `end` in the
        source file, `duration`, `safe_in` / `safe_out`, its `text`, and its
        `scores`. `truncated` is true when the limit cut the results short.

    Raises:
        ValueError: If the timeline does not exist, or a score name is not a
            plain identifier.
    """
    timeline = _require_timeline(timeline_id)
    clips = repo.query_semantic_clips(
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
        limit=limit,
    )
    return {
        "timeline_id": timeline.id,
        "clips": [_clip_summary(clip) for clip in clips],
        "truncated": len(clips) == limit,
    }

def _timeline_utterances(timeline: SemanticTimeline) -> List[SemanticClip]:
    """Read every `utterance` clip of a timeline, in order.

    Args:
        timeline: Timeline to read.

    Returns:
        The clips, ordered by asset and then by time.
    """
    return repo.query_semantic_clips(timeline.id, level=ClipLevel.UTTERANCE, limit=MAX_TIMELINE_CLIPS)

def _timeline_analyses(timeline: SemanticTimeline) -> dict:
    """Read the analyses a timeline was built from.

    Args:
        timeline: Timeline whose assets to read.

    Returns:
        The analyses, keyed by asset ID.

    Raises:
        ValueError: If an asset's analysis has gone missing.
    """
    analyses = {asset_id: repo.get_analysis(asset_id) for asset_id in timeline.asset_ids}
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
    repo.save_semantic_timeline(timeline, [*utterances, *built])

    topics: dict[str, dict] = {}
    for section in built:
        if section.topic:
            entry = topics.setdefault(section.topic, {"sections": 0, "seconds": 0.0})
            entry["sections"] += 1
            entry["seconds"] = round(entry["seconds"] + section.duration, 3)
    return {"timeline_id": timeline.id, "sections": len(built), "topics": topics}

@mcp.tool()
def get_semantic_clip(clip_id: str, include_words: bool = False) -> dict:
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

    Returns:
        The clip: its `clip_id`, `timeline_id`, `asset_id`, `level`, `kind`,
        `source_range`, `safe_in` / `safe_out`, full `text`, `speaker`,
        `scores`, and `tags` with where each came from. With `include_words`,
        also `words`, each with `start`, `end`, and `text`.

    Raises:
        ValueError: If no clip has that ID.
    """
    clip = repo.get_semantic_clip(clip_id)
    if clip is None:
        raise ValueError(f"semantic clip {clip_id} not found; query_clips lists the ones that exist")

    record = clip.model_dump(exclude={"id"})
    record["clip_id"] = clip.id
    if include_words:
        analysis = repo.get_analysis(clip.asset_id)
        transcript = analysis.transcript if analysis is not None else None
        record["words"] = [
            word.model_dump()
            for segment in (transcript.segments if transcript else [])
            for word in segment.words
            if word.end > clip.source_range.start and word.start < clip.source_range.end
        ]
    return record

@mcp.tool()
def frames_for_clips(
    clip_ids: List[str],
    columns: Annotated[int, Field(ge=1, le=8)] = 4,
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
    if len(clip_ids) > MAX_STORYBOARD_TILES:
        raise ValueError(f"{len(clip_ids)} clips is more than one sheet holds; ask for at most {MAX_STORYBOARD_TILES}")

    shots: List[tuple[str, float, str]] = []
    listing: List[str] = []
    for position, clip_id in enumerate(clip_ids):
        clip = repo.get_semantic_clip(clip_id)
        if clip is None:
            raise ValueError(f"semantic clip {clip_id} not found; query_clips lists the ones that exist")
        asset = _get_asset(clip.asset_id)
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
        clip = repo.get_semantic_clip(entry.clip_id)
        if clip is None:
            raise ValueError(f"semantic clip {entry.clip_id} not found; query_clips lists the ones that exist")
        source = TagSource.USER if written_by == "user" else TagSource.MODEL
        kept = [tag for tag in clip.tags if not (tag.source == source and tag.model_version == written_by)]
        written.append(clip.model_copy(update={
            "description": entry.description or clip.description,
            "described_by": written_by if entry.description else clip.described_by,
            "tags": [*kept, *(Tag(value=value, source=source, model_version=written_by) for value in entry.tags)],
        }))

    repo.update_semantic_clips(written)
    return {"updated": len(written), "tags": sum(len(clip.tags) for clip in written)}

@mcp.tool()
def view_frames(
    asset_ids: list[str],
    start: Annotated[float, Field(ge=0)] = 0,
    end: Optional[float] = None,
    count: Annotated[int, Field(ge=1, le=MAX_STORYBOARD_TILES)] = 12,
) -> ToolResult:
    """Look at frames from one or more assets as one labeled contact sheet image.

    With a single asset, frames are sampled at evenly spaced times between
    `start` and `end`: use a wide range for an overview and a narrow range to
    inspect a moment before choosing a cut point. With several, `count` frames
    are taken across the whole of each one and they follow each other on the
    sheet, which is how a folder of footage is surveyed without a call per
    file. A range means nothing across files of different lengths, so `start`
    and `end` are only accepted for a single asset.

    Every tile is labeled with its number, its source time, and which file it
    came from. Works without `analyze_asset`.

    Args:
        asset_ids: IDs of the assets to sample, in the order to show them.
        start: Range start in seconds; one asset only.
        end: Range end in seconds; one asset only, and the end of the asset
            when omitted.
        count: Number of frames per asset.

    Returns:
        A text block listing each tile's asset and source time, followed by
        the contact sheet as a JPEG image.

    Raises:
        ValueError: If an asset does not exist or has no video, the range is
            empty, a range is given for more than one asset, or the frames
            asked for do not fit on one sheet.
        RuntimeError: If frames cannot be decoded.
    """
    if not asset_ids:
        raise ValueError("name at least one asset to show frames from")
    if len(asset_ids) * count > MAX_STORYBOARD_TILES:
        raise ValueError(
            f"{len(asset_ids)} assets at {count} frames each is more than one sheet holds; "
            f"ask for at most {MAX_STORYBOARD_TILES} frames in total"
        )
    if len(asset_ids) > 1 and (start or end is not None):
        raise ValueError("a time range only means something for one asset; drop start and end, or ask about one file")

    shots: List[tuple[str, float, str]] = []
    listing: List[str] = []
    for asset_id in asset_ids:
        asset = _get_asset(asset_id)
        if not asset.has_video or asset.duration is None:
            raise ValueError(f"asset {asset_id} has no video frames to show")
        duration = float(asset.duration)
        last = duration if end is None else min(end, duration)
        if last <= start:
            raise ValueError(f"empty range: start {start}s must be before end {last}s (asset is {duration}s long)")
        step = (last - start) / count
        for index in range(count):
            # Sample the middle of each interval, and stay clear of the very last frame, which may not decode.
            seconds = round(min(start + step * (index + 0.5), duration - 0.05), 3)
            number = len(shots) + 1
            name = Path(asset.path).name
            shots.append((asset.path, seconds, f"#{number}  {format_timestamp(seconds)}  {name[:14]}"))
            listing.append(f"#{number}: {name} | {seconds:.3f}s ({format_timestamp(seconds)})")

    where = f"asset {asset_ids[0]}" if len(asset_ids) == 1 else f"{len(asset_ids)} assets, {count} frames each"
    return ToolResult(content=[
        f"Contact sheet of {where}, in reading order:\n" + "\n".join(listing),
        Image(data=storyboard_sheet(shots), format="jpeg"),
    ])

@mcp.tool()
def create_project(
    name: Annotated[Optional[str], Field(max_length=120)] = None,
    width: Annotated[int, Field(gt=0, multiple_of=2)] = 1920,
    height: Annotated[int, Field(gt=0, multiple_of=2)] = 1080,
    fps_num: Annotated[int, Field(gt=0)] = 30,
    fps_den: Annotated[int, Field(gt=0)] = 1,
) -> dict:
    """Create an empty project.

    Add tracks and clips to the new project with `apply_edits`. When rendered,
    every source is scaled to fill the output size, center-cropped, and
    converted to the output frame rate.

    Args:
        name: What to call the project, such as 'EP1 台北'. Worth giving: it is
            what `list_projects` shows, and several projects at once are hard
            to tell apart by ID. Change it later with a `rename_project`
            operation.
        width: Output width in pixels; must be even.
        height: Output height in pixels; must be even.
        fps_num: Frame rate numerator, e.g. 30000 for 29.97 fps.
        fps_den: Frame rate denominator, e.g. 1001 for 29.97 fps.

    Returns:
        The new project serialized as a dictionary, including its generated
        `id` and initial `version`.
    """
    project = Project(
        id=str(uuid.uuid4()), name=(name.strip() or None) if name else None,
        width=width, height=height, fps_num=fps_num, fps_den=fps_den,
    )
    repo.add_project(project)
    return project.model_dump()

@mcp.tool()
def list_projects() -> dict:
    """List all projects.

    Returns:
        A dictionary with a `projects` list, each item holding a project's
        `id`, its `name` if it was given one, and its current `version`.
    """
    return {"projects": [
        {"id": project.id, "name": project.name, "version": project.version}
        for project in repo.list_projects()
    ]}

@mcp.tool()
def get_project(project_id: str, include_subtitles: bool = False) -> dict:
    """Return the state of a project: its tracks, clips, and version.

    The returned `version` is required as `expected_version` when calling
    `apply_edits`. Captions are left out unless asked for, because a long
    video has hundreds of them and they are not needed to edit the timeline;
    read them with `get_subtitles` instead. Clip fields that are still at
    their default are left out too, so a clip showing no `volume` is at 1.0
    and one showing no `color` is as shot.

    Args:
        project_id: ID of the project to retrieve.
        include_subtitles: Whether to include every caption in the result.

    Returns:
        The project serialized as a dictionary, including `duration`, the
        length of the edited video in seconds, and `subtitle_count`.

    Raises:
        ValueError: If no project with `project_id` exists.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    # Clips are dumped without the fields that are still at their defaults, which is most of them
    # on a normal cut. The project's own settings are put back, because the caller needs them every time.
    state = project.model_dump(exclude_defaults=True)
    state.update({
        "id": project.id,
        "version": project.version,
        "width": project.width,
        "height": project.height,
        "fps_num": project.fps_num,
        "fps_den": project.fps_den,
        "duration": project.duration,
        "subtitle_count": len(project.subtitles),
    })
    state["tracks"] = []
    for track in project.tracks:
        summary = track.model_dump(exclude_defaults=True)
        summary.update({"id": track.id, "track_type": track.track_type.value})
        summary["clips"] = [clip.model_dump(exclude_defaults=True) for clip in track.clips]
        state["tracks"].append(summary)
    if include_subtitles:
        state["subtitles"] = [cue.model_dump() for cue in project.subtitles]
    else:
        state.pop("subtitles", None)
    return state

@mcp.tool()
def apply_edits(project_id: str, expected_version: int, operations: list[EditOperation]) -> dict:
    """Apply a batch of edit operations to a project with optimistic locking.

    Operations are applied in order and atomically: if any operation fails,
    or the resulting timeline is invalid, the project is left unchanged. On
    success the project version is incremented by one. Call `get_project`
    first to obtain the current version.

    Editing behaves like a magnetic timeline: `insert_clip` shifts later
    clips on the same track to make room, and `trim_clip` and `delete_clip`
    shift them to close or open space unless `ripple` is false.
    `reorder_clip` moves a clip elsewhere in the sequence, closing the space
    it leaves and opening space where it lands, and it keeps the clip's cut
    and audio settings. `split_clip` cuts one clip into two halves that
    together occupy exactly what the original did, so nothing else moves.
    `add_clip` and `move_clip` use absolute positions and never move other
    clips. Clips on other tracks never move.

    Background music goes on an audio track. The first video track is the
    base and defines the output length: gaps are rendered as black frames
    with silence, and audio past its last clip is cut. Further video tracks
    are drawn on top of the base, each clip inside the box its `layout`
    gives, or over the whole frame without one. `set_clip_audio` changes a
    clip's `volume`, its `audio_fade_in` / `audio_fade_out`, and how far its
    sound runs outside its picture: `audio_lead` brings the sound in early,
    so the next scene is heard under the end of the shot still on screen (a J
    cut), and `audio_lag` lets it run on after the picture has gone, so a line
    finishes over the shot that follows (an L cut). Both are seconds, both
    take their extra sound from outside the clip's `source_range`, and 0 puts
    the sound back with its picture. `set_clip_look` sets its `transition_in`
    — a dissolve, a wipe, or a dip through a colour, running it in over the end
    of the clip before it — along with its fades through black and its `color`.
    A transition ends on the cut rather than straddling it, so it never changes
    how long the sequence runs; `clear_transition` puts a straight cut back.
    `set_clip_speed` changes how fast a clip plays, and with it how long it
    runs, and `preserve_pitch` decides whether the sound keeps its pitch or
    rises and falls with the speed. `set_track_audio` sets a whole track's `duck_under_speech`.
    `set_markers` replaces the timeline's structure markers — where each part
    of the video begins. `compile_plan` writes one per beat, so a cut compiled
    from a plan arrives with its shape on it; markers added by hand have no
    beat behind them and survive the next compile. `set_subtitles` replaces the captions
    `render_project` burns in, and `set_caption_style` says how they are drawn:
    a platform preset for the safe area its own buttons take up, whether each
    word lights as it is said, and whether a caption says who is talking.

    The resulting timeline must satisfy these rules:
    - Clips stay within their asset's duration and do not overlap on a track.
      Sound is the exception and deliberately so: a lead or a lag overlaps the
      neighbouring clip's sound, which is what a J or an L cut is. It still
      has to come from inside the file and start after the timeline does.
    - Audio and video fades fit within their clip.
    - Only clips above the base video track take a `layout`; base-track clips
      always fill the frame.
    - Only audio tracks duck under speech.
    - Video-track clips use assets with video; audio-track clips use assets
      with audio.
    - Clips use speed 1.0.

    Args:
        project_id: ID of the project to edit.
        expected_version: Project version the caller last read. The request is
            rejected if the project has been modified since then.
        operations: Edit operations to apply, each identified by its `action`:
            `add_track`, `add_clip`, `insert_clip`, `trim_clip`, `move_clip`,
            `delete_clip`, `split_clip`, `reorder_clip`, `fit_track`,
            `set_clip_audio`, `set_clip_look`, `set_track_audio`,
            `set_subtitles`, or `edit_subtitle`.

    Returns:
        A dictionary with the `status` and the project's `new_version`.

    Raises:
        ValueError: If the project does not exist, the version does not match,
            an operation references a missing or duplicate track or clip, or
            the resulting timeline breaks a rule above.
    """
    return {"status": "success", "new_version": _apply(project_id, expected_version, operations).version}

def _apply(
    project_id: str,
    expected_version: int,
    operations: list,
    stamp: Optional[Callable[[Project], None]] = None,
) -> Project:
    """Apply a batch of operations to a project, atomically.

    Args:
        project_id: Project to edit.
        expected_version: Version the caller last read.
        operations: Operations to apply, in order.
        stamp: Called with the edited project before it is checked, for
            whatever the operations themselves cannot carry. `compile_plan`
            uses it to record where each clip came from, which keeps
            provenance off the tool surface and out of a caller's reach.

    Returns:
        The saved project, at its new version.

    Raises:
        ValueError: If the project does not exist, the version does not match,
            or the result breaks a timeline rule. Nothing is saved in that
            case.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if project.version != expected_version:
        raise ValueError(f"version conflict: current version is {project.version}, expected version is {expected_version}")

    # A caption names the file its words were spoken in. One naming a file nobody
    # imported would store cleanly and then never appear, which reads as a captioning
    # bug rather than a typo, so it is refused here where the typo still has a name.
    named = {cue.asset_id for op in operations if isinstance(op, SetSubtitlesOp) for cue in op.cues}
    unknown = sorted(named - set(repo.get_assets(named)))
    if unknown:
        raise ValueError(
            f"caption(s) name assets that are not imported: {', '.join(unknown)}; "
            "import the footage the words were spoken in first"
        )

    # Assets are looked up first so operations that need source lengths, such as fit_track, can use them.
    known = repo.get_assets({clip.asset_id for track in project.tracks for clip in track.clips})
    for op in operations:
        apply_operation(project, op, known)
    if stamp is not None:
        stamp(project)
    assets = _referenced_assets(project)
    validate_project(project, assets)
    renderer.check_supported(project, assets)

    project.version += 1
    repo.update_project(project, expected_version)
    return project

def _plan_context(plan: EditPlan) -> tuple:
    """Gather everything a plan has to be judged and compiled against.

    Args:
        plan: Plan to read.

    Returns:
        `(timeline, clips_by_id, children_by_parent, assets_by_id,
        cuts_by_asset_id, levels_by_voice)`.

    Raises:
        ValueError: If the timeline the plan names is gone.
    """
    timeline = repo.get_semantic_timeline(plan.timeline_id)
    if timeline is None:
        raise ValueError(f"the plan is written against timeline {plan.timeline_id}, which no longer exists")
    clips = repo.query_semantic_clips(timeline.id, limit=MAX_TIMELINE_CLIPS)
    children: dict[str, List[SemanticClip]] = {}
    for clip in clips:
        if clip.parent_id:
            children.setdefault(clip.parent_id, []).append(clip)
    for inside in children.values():
        inside.sort(key=lambda clip: clip.source_range.start)
    wanted = {clip.asset_id for clip in clips} | ({plan.music.asset_id} if plan.music else set())
    # Where each file may be cut without splitting a word. The compiler is handed these
    # numbers rather than the transcripts they come from: reading a transcript is
    # interpretation, and the compiler has to stay a pure function of what it is given.
    cuts, analyses = {}, {}
    for asset_id in {clip.asset_id for clip in clips}:
        analysis = repo.get_analysis(asset_id)
        if analysis is not None:
            cuts[asset_id] = clean_cuts(analysis)
            analyses[asset_id] = analysis
    # How loudly each voice speaks, worked out over the same joined labels the timeline
    # put on its clips — so a gain and the clip it applies to cannot mean different people.
    levels = voice_levels(analyses, join_voices(analyses))
    return timeline, {clip.id: clip for clip in clips}, children, repo.get_assets(wanted), cuts, levels

def _require_plan(plan_id: Optional[str]) -> EditPlan:
    """Fetch a plan, or the one saved most recently.

    Args:
        plan_id: ID of the plan, or `None` for the newest.

    Returns:
        The plan.

    Raises:
        ValueError: If it does not exist, or nothing has been planned yet.
    """
    plan = repo.get_plan(plan_id)
    if plan is None:
        raise ValueError(f"plan {plan_id} not found" if plan_id else "no plan has been saved yet; write one with save_plan")
    return plan

@mcp.tool()
def save_plan(plan: EditPlan) -> dict:
    """Store a plan for a cut: what it is for, its parts, and which footage fills them.

    A plan is written in terms of semantic clips rather than seconds. You
    decide what goes in and why; `compile_plan` works out where every cut
    lands. Write down why each piece is there and why the ones you passed over
    were passed over — that is what makes the next round of changes possible
    rather than a fresh start.

    Each selection carries a `trim` saying how much of its clip to use:
    `full`, `keep` with the clips inside a section to keep, `head` or `tail`
    with a number of seconds, or `tighten` to drop the pauses and unusable
    picture inside a section. There is no free-text trim on purpose: the
    compiler has to produce the same cut from the same plan every time, and it
    cannot do that if it has to interpret a sentence.

    Saving a plan that already exists needs its current `version`, and raises
    it by one. Saving under a fresh id keeps both, which is how two ways of
    cutting the same footage are compared with `diff_plan`.

    Args:
        plan: The plan. `timeline_id` says which footage it is about;
            `timeline_input_hash` is filled in for you.

    Returns:
        A dictionary with the `plan_id`, its new `version`, the `timeline_id`,
        and `problems` and `notes` from checking it. A plan is stored whether
        or not it checks out, so it can be fixed rather than retyped.

    Raises:
        ValueError: If the timeline does not exist, or the plan was saved at a
            version other than the stored one.
    """
    timeline = repo.get_semantic_timeline(plan.timeline_id)
    if timeline is None:
        raise ValueError(f"semantic timeline {plan.timeline_id} not found; build one before planning against it")

    stamped = plan.model_copy(update={"timeline_input_hash": timeline.input_hash})
    stored = repo.save_plan(stamped)
    problems, notes = check_plan(stored, *_plan_context(stored))
    newest = repo.get_semantic_timeline(None)
    if newest is not None and newest.id != timeline.id:
        notes.append(f"a newer timeline ({newest.id}) has been built since; this plan is about {timeline.id}")
    return {
        "plan_id": stored.id,
        "version": stored.version,
        "timeline_id": stored.timeline_id,
        "problems": problems,
        "notes": notes,
    }

@mcp.tool()
def amend_plan(plan_id: str, expected_version: int, amendments: list[PlanAmendment]) -> dict:
    """Change part of a stored plan without sending the whole thing again.

    A cut is argued with one piece at a time — this beat runs long, that shot
    does not work, one more thing belongs in the middle — and every selection
    carries a written reason that sending the plan again puts at risk. Retrim
    with `set_trim`, rewrite a reason or move a piece to another beat with
    `set_rationale`, put something in with `add_selection`, and take something
    out with `drop_selection`, which files it under the plan's rejections with
    the reason so the question can be answered later.

    Nothing is compiled here. Call `compile_plan` when the plan reads right.

    Args:
        plan_id: Plan to amend.
        expected_version: The plan's current `version`, as `get_plan` reported
            it. The amendment is refused if the stored plan has moved on.
        amendments: What to change, applied in order.

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
    amended = plan.model_copy(deep=True)
    for amendment in amendments:
        apply_amendment(amended, amendment)

    timeline = repo.get_semantic_timeline(amended.timeline_id)
    if timeline is None:
        raise ValueError(f"the plan is written against timeline {amended.timeline_id}, which no longer exists")
    stored = repo.save_plan(amended.model_copy(update={"timeline_input_hash": timeline.input_hash}))
    problems, notes = check_plan(stored, *_plan_context(stored))
    return {
        "plan_id": stored.id,
        "version": stored.version,
        "timeline_id": stored.timeline_id,
        "problems": problems,
        "notes": notes,
    }

@mcp.tool()
def get_plan(plan_id: Optional[str] = None) -> dict:
    """Read a stored plan back, with the other plans there are to compare it with.

    Args:
        plan_id: Plan to read; omit for the one saved most recently.

    Returns:
        A dictionary with the `plan` as it was stored — its goal, target,
        beats, selections with their reasons, what was rejected and why, and
        any music — plus `other_plans`, each with its `plan_id`, `goal` and
        `timeline_id`, for `diff_plan` to compare against.

    Raises:
        ValueError: If the plan does not exist, or none have been saved.
    """
    plan = _require_plan(plan_id)
    return {
        "plan": plan.model_dump(),
        "other_plans": [
            {"plan_id": other.id, "goal": other.goal, "timeline_id": other.timeline_id, "version": other.version}
            for other in repo.list_plans() if other.id != plan.id
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
    timeline, clips, children, assets, cuts, levels = _plan_context(plan)
    problems, notes = check_plan(plan, timeline, clips, children, assets, cuts, levels)
    pieces = [] if problems else plan_pieces(plan, clips, children, assets, cuts)
    return {
        "ok": not problems,
        "problems": problems,
        "notes": notes,
        "duration": compiled_duration(pieces),
        "clips": len(pieces),
    }

@mcp.tool()
def compile_plan(project_id: str, expected_version: int, plan_id: Optional[str] = None) -> dict:
    """Build a plan's cut on a project's timeline.

    Every second is worked out here, the same way every time: each selection
    becomes the piece its trim asks for, widened into the measured silence
    around it so no line starts abruptly, and pieces that nearly touch become
    one clip. Music, if the plan has any, goes underneath and is fitted to the
    picture.

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
    timeline, clips, children, assets, cuts, levels = _plan_context(plan)
    problems, notes = check_plan(plan, timeline, clips, children, assets, cuts, levels)
    if problems:
        raise ValueError("this plan cannot be compiled yet:\n- " + "\n- ".join(problems))

    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    pieces = plan_pieces(plan, clips, children, assets, cuts)
    covers, _, _ = broll_covers(plan, pieces, clips, assets, cuts)
    blocked = check_recompile(plan, project, pieces, covers)
    if blocked:
        raise ValueError("compiling would undo work already on this project:\n- " + "\n- ".join(blocked))

    built, provenance = compile_operations(plan, clips, children, assets, project, cuts, levels)
    operations = _OPERATIONS.validate_python(built)

    def record(edited: Project) -> None:
        """Write onto each compiled clip where it came from."""
        for track in edited.tracks:
            for clip in track.clips:
                origin = provenance.get(clip.id)
                if origin is not None and track.id in (VIDEO_TRACK_ID, MUSIC_TRACK_ID, BROLL_TRACK_ID):
                    clip.from_plan_id = origin["from_plan_id"]
                    clip.from_clip_ids = origin["from_clip_ids"]
                    clip.pinned = origin["pinned"]

    saved = _apply(project_id, expected_version, operations, stamp=record)
    return {
        "plan_id": plan.id,
        "project_id": project_id,
        "new_version": saved.version,
        "clips": sum(1 for operation in operations if operation.action == "insert_clip"),
        "kept": sum(1 for origin in provenance.values() if origin["pinned"]),
        "duration": float(saved.duration),
        "notes": notes,
    }

@mcp.tool()
def diff_plan(before_plan_id: str, after_plan_id: str) -> dict:
    """Say what changed between two plans.

    Use this when there are two ways of cutting the same footage, or when a
    plan has been reworked after feedback and the user asks what is actually
    different.

    Args:
        before_plan_id: The earlier plan.
        after_plan_id: The later one.

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
        "changes": diff_plans(_require_plan(before_plan_id), _require_plan(after_plan_id)),
    }

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
    _, clips, children, assets, cuts, _ = _plan_context(plan)
    pieces = plan_pieces(plan, clips, children, assets, cuts)
    return {"plan_id": plan.id, "slots": broll_slots(plan, pieces, clips, cuts)}

@mcp.tool()
def preview_project(
    project_id: str,
    count: Annotated[int, Field(ge=1, le=MAX_STORYBOARD_TILES)] = 12,
) -> ToolResult:
    """Look at the edited sequence as one labeled storyboard image.

    A storyboard takes seconds, while `render_project` takes minutes, so use
    it to check an edit before rendering: the order of the clips, what each
    one opens on, and whether a cut lands somewhere wrong. Every clip gets at
    least one tile and longer clips get more, so a short clip is never missed;
    only a project with more clips than the sheet holds is thinned out, and
    the text says how many were skipped.
    Tiles are labeled with their time in the edited result and the clip they
    come from, and cropped the way the render crops, so a vertical project
    shows vertical tiles. Tiles come from the base video track, which is the
    sequence itself. Black gaps and audio tracks are listed in the text, since
    they cannot be seen in the frames, and so is anything on a higher video
    track — an inset, or covering picture over the whole frame, in which case
    the tiles for that stretch show the sequence underneath rather than what
    the viewer will see.

    Args:
        project_id: ID of the project to look at.
        count: Number of frames to sample. Raised to one per clip when the
            project has more clips than this.

    Returns:
        A text block describing the output format, every tile, the black gaps,
        and the audio tracks, followed by the storyboard as a JPEG image.

    Raises:
        ValueError: If the project does not exist, has no clips on a video
            track, or a video clip uses an asset without video.
        RuntimeError: If frames cannot be decoded.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")

    # Tiles come from the base track, which is the sequence; tracks above it are insets, noted in the text.
    base = project.base_video_track
    clips = sorted(base.clips, key=lambda clip: clip.timeline_in) if base else []
    if not clips:
        raise ValueError(f"project {project_id} has no clips on a video track to preview")

    assets = _referenced_assets(project)
    for clip in clips:
        asset = assets.get(clip.asset_id)
        if asset is None:
            raise ValueError(f"clip {clip.id}: asset {clip.asset_id} not found")
        if not asset.has_video:
            raise ValueError(
                f"clip {clip.id}: asset {asset.id} has no video stream; "
                "put audio-only assets on an audio track"
            )

    # Where the picture really goes black, found over the whole sequence rather than over
    # the tiles chosen below. A clip left out of the storyboard is not a hole in the edit,
    # and reading the gaps off the sampled list turned every skipped clip into one.
    total_clips = len(clips)
    gaps: List[List[tuple[Decimal, Decimal]]] = [[] for _ in clips]
    running_out = Decimal(0)
    for index, clip in enumerate(clips):
        if clip.timeline_in > running_out:
            gaps[index].append((running_out, clip.timeline_in))
        running_out = clip.timeline_out

    def gap_line(opened: Decimal, closed: Decimal) -> str:
        """Describe one stretch of black picture for the listing."""
        return (
            f"-- black gap {format_timestamp(float(opened))} to "
            f"{format_timestamp(float(closed))} ({float(closed - opened):.3f}s)"
        )

    # One tile per clip is the floor, so a one-second clip in a long edit is still shown.
    omitted = max(0, total_clips - MAX_STORYBOARD_TILES)
    if omitted:
        stride = total_clips / MAX_STORYBOARD_TILES
        chosen = sorted({min(int(stride * index), total_clips - 1) for index in range(MAX_STORYBOARD_TILES)})
    else:
        chosen = list(range(total_clips))
    # Every tile carries what fell between the tile before it and itself, so a clip that is
    # not drawn is still counted and a real gap behind it is still reported.
    spans = list(zip([-1, *chosen], chosen))
    skipped = [later - earlier - 1 for earlier, later in spans]
    carried = [
        [gap for index in range(earlier + 1, later + 1) for gap in gaps[index]]
        for earlier, later in spans
    ]
    clips = [clips[index] for index in chosen]
    per_clip = [1] * len(clips)
    for _ in range(max(count, len(clips)) - len(clips)):
        # Give the next frame to whichever clip currently covers the most seconds per frame.
        longest = max(range(len(clips)), key=lambda index: float(clips[index].timeline_duration) / per_clip[index])
        per_clip[longest] += 1

    shots: List[tuple[str, float, str]] = []
    listing: List[str] = []
    for position, (clip, frames) in enumerate(zip(clips, per_clip)):
        if skipped[position]:
            listing.append(f"-- {skipped[position]} clip(s) not shown")
        listing.extend(gap_line(opened, closed) for opened, closed in carried[position])
        asset = assets[clip.asset_id]
        start, end = float(clip.source_range.start), float(clip.source_range.end)
        step = (end - start) / frames
        for offset in range(frames):
            # Sample the middle of each interval, and stay clear of the very last frame, which may not decode.
            seconds = start + step * (offset + 0.5)
            if asset.duration is not None:
                seconds = min(seconds, float(asset.duration) - 0.05)
            seconds = round(max(seconds, 0.0), 3)
            at = float(clip.timeline_in) + (seconds - start) / clip.speed
            number = len(shots) + 1
            label = clip.id if len(clip.id) <= 10 else f"{clip.id[:9]}~"
            shots.append((asset.path, seconds, f"#{number}  {format_timestamp(at)}  {label}"))
            listing.append(
                f"#{number}: clip {clip.id} | edit {format_timestamp(at)} | "
                f"source {seconds:.3f}s ({format_timestamp(seconds)})"
            )

    # Sampling can stop short of the last clip, so say what runs on past the final tile.
    trailing = total_clips - 1 - chosen[-1]
    if trailing:
        listing.append(f"-- {trailing} clip(s) not shown")
        for index in range(chosen[-1] + 1, total_clips):
            listing.extend(gap_line(opened, closed) for opened, closed in gaps[index])

    for track in project.video_tracks[1:]:
        for clip in sorted(track.clips, key=lambda item: item.timeline_in):
            box = clip.layout
            # A clip with no layout fills the frame, which means it replaces what the
            # tiles show rather than sitting in a corner of it. Said plainly, because a
            # storyboard of a B-roll pass otherwise looks like the pass never happened.
            what = (
                "covering picture, over the whole frame — the tiles in this stretch show "
                "the sequence underneath, not this" if box is None
                else f"inset at {box.x:.0%} across and {box.y:.0%} down, "
                     f"{box.width:.0%} x {box.height:.0%} of the frame"
            )
            listing.append(
                f"-- track {track.id}: clip {clip.id}, "
                f"{format_timestamp(float(clip.timeline_in))} to {format_timestamp(float(clip.timeline_out))}, {what}"
            )

    for track in project.tracks:
        if track.track_type != TrackType.AUDIO or not track.clips:
            continue
        covered_from = min(clip.timeline_in for clip in track.clips)
        covered_to = max(clip.timeline_out for clip in track.clips)
        volumes = sorted({clip.volume for clip in track.clips})
        level = f"volume {volumes[0]}" if len(volumes) == 1 else f"volumes {', '.join(str(v) for v in volumes)}"
        listing.append(
            f"-- audio track {track.id}: {len(track.clips)} clip(s), "
            f"{format_timestamp(float(covered_from))} to {format_timestamp(float(covered_to))}, {level}"
        )

    header = (
        f"Storyboard of project {project_id} in timeline order: {project.width}x{project.height}, "
        f"{project.fps_num}/{project.fps_den} fps, {format_timestamp(float(project.duration))} long, "
        f"{len(shots)} frames from {total_clips} clip(s)."
    )
    if omitted:
        header += f" {omitted} clip(s) were skipped: the project has more clips than tiles."
    return ToolResult(content=[
        "\n".join([header, *listing]),
        Image(data=storyboard_sheet(shots, aspect=project.width / project.height), format="jpeg"),
    ])

@mcp.tool()
def generate_subtitles(
    project_id: str,
    max_characters: Annotated[int, Field(ge=8, le=120)] = DEFAULT_MAX_CHARACTERS,
    max_seconds: Annotated[float, Field(gt=0.2, le=15)] = DEFAULT_MAX_SECONDS,
) -> dict:
    """Propose captions for the edited sequence, from transcripts already made.

    Reads the transcript of every clip on the base video track and on the
    audio tracks, keeps only the speech that survived the edit, and moves each
    line to where it now falls in the result. A narration recorded separately
    and laid on an audio track is captioned like any other speech; music is
    not, because nobody transcribes a song. Insets on video tracks above the
    base are pictures over that sound, so they are not captioned, and neither
    is a clip turned all the way down. Long sentences are broken between
    words. Nothing is saved: check the wording, fix any name the transcript
    misheard, and then store the captions with a `set_subtitles` operation in
    `apply_edits`. Render them into the picture with `render_project` and
    `burn_subtitles`.

    Args:
        project_id: ID of the project to caption.
        max_characters: Longest caption text before it is broken in two.
        max_seconds: Longest caption before it is broken in two.

    Returns:
        A dictionary with `cues`, each holding its `id`, the `asset_id` the
        words were spoken in, `source_start` and `source_end` in seconds
        within that file, its `text`, the `speaker` the speaker split credits
        it to where one voice clearly holds it, and the `words` inside it with
        their own timings, which is what lets a caption light up as it is
        said; `assets_without_transcript`, the
        video sources that still need `analyze_asset` before they can be
        captioned; and `overlapping`, how many captions land on top of the one
        before them once the cut puts them on screen, which is worth a look
        when a narration track talks over footage that speaks for itself.

    Raises:
        ValueError: If the project does not exist, or nothing it plays has
            been transcribed.
    """
    project = repo.get_project(project_id)
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
        analysis = repo.get_analysis(clip.asset_id)
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
    cues = timeline_cues(
        project, transcripts, max_characters=max_characters, max_seconds=max_seconds,
        silences=silences, speakers=speakers,
    )
    # Counted where the captions land, not where the words were said: two lines from
    # different files overlap only once the cut puts them on screen together.
    placed = place_cues(project, cues)
    overlapping = sum(1 for earlier, later in zip(placed, placed[1:]) if later.start < earlier.end)
    return {
        "cues": [cue.model_dump() for cue in cues],
        "assets_without_transcript": untranscribed,
        "overlapping": overlapping,
    }

@mcp.tool()
def get_subtitles(
    project_id: str,
    start: Annotated[float, Field(ge=0)] = 0,
    end: Annotated[Optional[float], Field(ge=0)] = None,
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

    Returns:
        A dictionary with `cues` in the window — each with its `cue_id`,
        `start` and `end` on the timeline, and `text` — `placed`, how many
        land anywhere in the cut, `stored`, how many captions the project
        holds, and the `window` that was read. `stored` above `placed` means
        some captions belong to footage the edit dropped.

    Raises:
        ValueError: If the project does not exist, or `end` is not after
            `start`.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    placed = place_cues(project, project.subtitles)
    if end is None:
        limit = float(project.duration)
    else:
        limit = end
        if limit <= start:
            raise ValueError(f"the window ends at {limit}s, which is not after its start at {start}s")
    window = [
        cue.model_dump() for cue in placed
        if float(cue.end) > start and float(cue.start) < limit
    ]
    return {
        "cues": window,
        "placed": len(placed),
        "stored": len(project.subtitles),
        "window": {"start": start, "end": limit},
    }

def _output_name(project: Project, kind: str) -> str:
    """Name a rendered file after the project it came from.

    A render used to be `<job uuid>/<project uuid>_output.mp4`, which is two
    identifiers and nothing a person can pick out of a folder. That is why
    finished cuts end up copied somewhere else under a name somebody made up
    — and next to the footage is the worst of the places they end up, because
    a cut sitting among its own sources is indistinguishable from source.

    Args:
        project: The project being rendered.
        kind: `output` or `preview`.

    Returns:
        A file name. The project's name where it has one, with anything a
        filesystem would refuse taken out, and its ID where the name is
        missing or survives sanitising as nothing. Each render has its own
        directory, so two of the same project do not collide.
    """
    stem = "".join(" " if character in FORBIDDEN_IN_NAMES else character for character in (project.name or ""))
    stem = " ".join(stem.split())[:MAX_OUTPUT_NAME].strip(" .")
    return f"{stem or project.id}_{kind}.mp4"

@mcp.tool()
def render_project(
    project_id: str,
    is_preview: bool = False,
    loudness_target: Annotated[Optional[float], Field(ge=-40, le=-5)] = DEFAULT_LOUDNESS_TARGET,
    burn_subtitles: bool = False,
) -> dict:
    """Start rendering a project to an MP4 file in the background.

    Returns immediately. Poll `get_job` with the returned `job_id` until the
    job reaches a terminal status, or stop it with `cancel_job`. Rendering
    runs in a separate worker process and continues even if this server
    restarts.

    Args:
        project_id: ID of the project to render.
        is_preview: If true, render a fast 480p preview instead of the
            full-quality output.
        loudness_target: Loudness of the finished file in LUFS. The default,
            -14, is what streaming platforms normalize to, so clips recorded
            on different devices come out at one consistent level instead of
            jumping. Pass null to leave the mix exactly as the clip volumes
            set it.
        burn_subtitles: If true, burn the project's stored captions into the
            picture. Store them first with a `set_subtitles` operation;
            `generate_subtitles` proposes them from the transcripts.

    Returns:
        A dictionary with the `job_id`, the initial `status`, `stage`, and the
        absolute `output_path` the file will be written to, named after the
        project so the outputs folder can be read at a glance. Each job writes to
        its own directory, so repeated renders never overwrite each other.
        Renders and analyses run one or two at a time, so that several of them
        cannot exhaust the machine's memory between them; `stage` says what a
        job that has not started yet is waiting for, and is null when it
        started immediately.

    Raises:
        ValueError: If the project does not exist, contains no clips, or uses
            an unsupported feature.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")

    kind = "preview" if is_preview else "output"
    job = Job(kind=JobKind.RENDER, project_id=project_id)
    job.work_dir = os.path.join(WORKSPACE_DIR, "outputs", job.job_id)
    job.output_path = os.path.join(job.work_dir, _output_name(project, kind))

    subtitle_path = None
    if burn_subtitles:
        if not project.subtitles:
            raise ValueError(
                f"project {project_id} has no captions to burn; "
                "propose them with generate_subtitles and store them with a set_subtitles operation"
            )
        placed = place_cues(project, project.subtitles)
        if not placed:
            raise ValueError(
                f"project {project_id} has captions, but none of the footage they transcribe is in the cut; "
                "run generate_subtitles again against the sequence as it stands"
            )
        os.makedirs(job.work_dir, exist_ok=True)
        subtitle_path = os.path.join(job.work_dir, "subtitles.ass")
        with open(subtitle_path, "w", encoding="utf-8") as handle:
            handle.write(build_ass(placed, project.width, project.height, project.caption_style))

    command = renderer.build_command(
        project, _referenced_assets(project), job.output_path,
        is_preview=is_preview, loudness_target=loudness_target, subtitle_path=subtitle_path,
    )
    # One input per clip, all opened at once, is what a render's memory use is made of.
    job.memory_estimate = resources.render_memory_bytes(command.count("-i"))
    job = job_manager.start_job(job, {"command": command, "duration_seconds": float(renderer.output_duration(project))})

    return {"job_id": job.job_id, "status": job.status.value, "stage": job.stage, "output_path": job.output_path}

@mcp.tool()
def get_job(job_id: str) -> dict:
    """Return the current state of a background job.

    Args:
        job_id: ID of the job, as returned by `render_project` or
            `analyze_asset`.

    Returns:
        The job serialized as a dictionary, including `kind`, `status`,
        `progress` (0.0 to 1.0), the current `stage`, `output_path` for
        renders, and, for failed jobs, `error_message`.

    Raises:
        ValueError: If no job with `job_id` exists.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise ValueError(f"job {job_id} not found")
    return job.model_dump()

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Cancel a queued or running background job.

    Waits up to a few seconds for a running job to stop.

    Args:
        job_id: ID of the job, as returned by `render_project` or
            `analyze_asset`.

    Returns:
        A dictionary with the `job_id`, a `cancelled` flag, and the job's
        current `status` (null if the job does not exist). `cancelled` is
        false if the job does not exist or had already finished.
    """
    job = job_manager.cancel_job(job_id)
    return {
        "job_id": job_id,
        "cancelled": job is not None and job.status == JobStatus.CANCELLED,
        "status": job.status.value if job else None,
    }

def main():
    """Run the MCP server over the stdio transport."""
    mcp.run()

if __name__ == "__main__":
    main()
