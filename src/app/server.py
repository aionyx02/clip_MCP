import hashlib
import json
import os
from datetime import datetime, timezone
import re
import shutil
import subprocess
import threading
import time
import uuid
from contextvars import ContextVar
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp.server.transforms import ResourcesAsTools
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent
from pydantic import BaseModel, Field, TypeAdapter
from app.models.media import Asset, Measured, MediaAnalysis, Source, Span, SpeakerTurn, Transcript
from app.models.plan import (
    EditPlan, PlanAmendment, TrimKind, apply_amendment, describe_amendment, short_clip_ids, whole_clip_id,
    with_whole_clip_ids,
)
from app.models.semantic import (
    ClipDescription, ClipKind, ClipLevel, SectionChoice, SemanticClip, SemanticTimeline, Tag, TagSource,
)
from app.models.timeline import (
    AddSubtitleOp,
    BranchPoint,
    Clip,
    EditOperation,
    EditSubtitleOp,
    Project,
    SetSubtitlesOp,
    SubtitleCue,
    TrackType,
    retimed_words,
    apply_operation,
    validate_project,
)
from app.models.comment import Comment, Reply
from app.models.job import Job, JobKind, JobStatus
from app.engine import loudness, machine, meaning, resources, speed
from app.engine.analysis import listen_again as listen_again_in_file
from app.engine.analysis import (
    current_recipe, doubtful_spots, marked_text, marked_words, sound_note, transcription_of, unsure_words,
    whisper_model_for, whisper_model_name,
)
from app.engine.diarize import speaker_model_name
from app.engine.rhythm import rhythm_model_name
from app.engine.faces import framing_note
from app.engine.plan import (
    BROLL_TRACK_ID, COMPILED_TRACKS, MUSIC_TRACK_ID, VIDEO_TRACK_ID, as_left, broll_covers, broll_slots, check_plan,
    check_recompile, compile_operations, compiled_duration, diff_plans, music_beds, Note, note_record, piece_changes,
    plan_pieces,
)
from app.engine.sections import build_sections, candidate_hash, check_sections, propose_candidates
from app.engine.semantic import (
    build_timeline, clean_cuts, content_scores, join_voices, timeline_input_hash, was_made_up,
)
from app.engine.ffmpeg import graph_from_file, hidden_window_flags
from app.engine.probe import picture_size, probe_file, recorded_at, speech_loudness, timecode_start
from app.engine.reframe import Framing, frame_project, placement_at
from app.engine import playback
from app.engine.library import FOLDER_KINDS, FOOTAGE, LIBRARIES, MUSIC, classify, library_of
from app.engine import clap
from app.engine import fetch
from app.engine import music as music_energy
from app.engine.builder import DEFAULT_LOUDNESS_TARGET, FFmpegRenderer, Talking, voice_keys
from app.engine.levels import song_level, talking_in
from app.engine.comments import anchor_at, where_now
from app.engine.compare import compare
from app.engine.compare import parts_of as cut_parts
from app.engine.merge import WHOLE_PICKS, merge, merge_plans
from app.engine.delivery import CHECKS, chapter_metadata, chapters, check_delivery, clock, cover_candidates
from app.engine.frames import format_timestamp, still, storyboard_sheet
from app.engine.interchange import write_edl, write_fcpxml, write_otio, write_srt
from app.engine.listen import chart, describe, levels, levels_every, parts_of
from app.engine.subtitles import (
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_MAX_SECONDS,
    build_ass,
    caption_geometry,
    captioned_clips,
    place_cues,
    timeline_cues,
)
from app.engine.renderer import JobManager
from app.storage import housekeeping
from app.storage.history import Version, current_batch
from app.storage.repo import Repository
from app.workspace import output_dir, workspace_dir

WORKSPACE_DIR = workspace_dir()
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

# Opened before the server is described: what it says about this computer's speed comes from runs kept here.
repo = Repository(os.path.join(WORKSPACE_DIR, "clip_mcp.db"))
history = repo.keep_history(os.path.join(WORKSPACE_DIR, "history"))

# The tools an ordinary edit goes through, in the order they come up. Named in the opening
# instructions because a client that loads tools on demand has to guess which ones it will
# need before it has read anything; a test keeps every name here a real tool.
CORE_TOOLS = (
    "list_assets", "analyze_asset", "get_job", "build_semantic_timeline", "query_clips", "view_frames",
    "save_plan", "amend_plan", "create_project", "compile_plan", "preview_project", "render_project",
    "view_render", "listen_again", "check_render", "apply_edits", "generate_subtitles",
)

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
        "burn_subtitles, how a render is checked first and refused until the "
        "user has heard what the check found, how one cut is delivered in "
        "several shapes and handed to other editing programs, "
        "and the current limits: no overlapping picture on a "
        "track, no still images, and no graphics beyond captions. "
        "Transitions end on the cut rather than straddling it, so adding one "
        "never changes how long the video runs. The person using this server is "
        "editing their own video, not writing code: reply in plain language "
        "with no tool names, IDs, or JSON, and read the plan back in one "
        "sentence for confirmation before the first render — unless they asked "
        "you to finish on your own, and then the skill says what still has to "
        "happen. Only files in the "
        "library can go into a project: a file the user names is added to it "
        "with import_asset or import_folder first, and you tell them it was "
        "added, since they see the library in the editor app and can take "
        "files out of it there; a file taken out is no longer usable until it "
        "is added again. Projects, assets, "
        "analyses, and jobs are saved on disk and survive server restarts. "
        f"The tools most edits go through, in order: {', '.join(CORE_TOOLS)}. "
        "Wherever a tool takes an asset ID, the file's exact name works too. "
        # Looked at once, at start: the AI has to know before it sends footage off how long that will take here.
        + machine.describe(machine.profile(), speed.estimate(repo, machine.profile(), {"accurate": 3600}, False))
    ),
)
# "resources" lists the supporting files alongside SKILL.md, so a client sees them
# without having to read the manifest first.
mcp.add_provider(SkillsDirectoryProvider(roots=SKILLS_DIR, supporting_files="resources"))
mcp.add_transform(ResourcesAsTools(mcp))

def _shortened(value: object) -> object:
    """Take the timeline off every clip ID in a result, however deep it sits.

    Args:
        value: A result, or any part of one.

    Returns:
        The same shape, with every string's clip IDs short.
    """
    if isinstance(value, str):
        return short_clip_ids(value)
    if isinstance(value, list):
        return [_shortened(item) for item in value]
    if isinstance(value, dict):
        return {key: _shortened(item) for key, item in value.items()}
    return value

class ShortClipIds(Middleware):
    """Show the AI every clip ID one way: `u0034`, without the timeline in front.

    A clip's whole ID is its timeline's and its own, and the AI used to be shown the
    whole one in some places and the short one in others, then had to write the whole
    one into a plan. Everything said back passes through here, errors included, and
    every tool that takes a clip knows which timeline it means.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> ToolResult:
        """Run the tool, then shorten every clip ID in what it said or raised.

        Args:
            context: The call.
            call_next: The rest of the chain, ending in the tool.

        Returns:
            The tool's result with short clip IDs.

        Raises:
            ToolError: The tool's own error, its clip IDs short.
        """
        try:
            result = await call_next(context)
        except Exception as exc:
            said = str(exc)
            if short_clip_ids(said) != said:
                raise ToolError(short_clip_ids(said)) from exc
            raise
        if not isinstance(result, ToolResult):
            return result
        return ToolResult(
            content=[
                TextContent(type="text", text=short_clip_ids(block.text)) if isinstance(block, TextContent) else block
                for block in result.content
            ],
            structured_content=_shortened(result.structured_content),
            meta=result.meta,
        )

mcp.add_middleware(ShortClipIds())

def asset_id_for(name: str) -> str:
    """Find the asset a file name means, for a tool given a name where it wants an ID.

    Only an exact name: guessing which of `0828.mp4` and `0828(1).mp4` was
    meant is worse than one more look at the list.

    Args:
        name: An asset ID, or a file's name as `list_assets` shows it.

    Returns:
        The asset's ID, or `name` as it was when it is an ID or no file has
        that name, so the tool says what it says about an asset it cannot find.

    Raises:
        ValueError: If more than one file has that name.
    """
    if not isinstance(name, str) or repo.get_assets([name]):
        return name
    named = [asset for asset in repo.list_assets() if os.path.basename(asset.path) == name]
    if len(named) > 1:
        raise ValueError(
            f"{len(named)} files are called {name}: "
            + "; ".join(f"{asset.id} ({asset.path})" for asset in named)
            + ". Give the ID of the one meant"
        )
    return named[0].id if named else name

def _with_asset_ids(value: object) -> object:
    """Put an ID in place of every file name given as an asset, however deep it sits.

    Args:
        value: A tool's arguments, or any part of them.

    Returns:
        The same shape, with each `asset_id` and `asset_ids` entry an ID
        where it named exactly one file.

    Raises:
        ValueError: If a name is shared by more than one file.
    """
    if isinstance(value, list):
        return [_with_asset_ids(item) for item in value]
    if not isinstance(value, dict):
        return value
    named = {}
    for key, item in value.items():
        if key == "asset_id":
            named[key] = asset_id_for(item)
        elif key == "asset_ids" and isinstance(item, list):
            named[key] = [asset_id_for(name) for name in item]
        else:
            named[key] = _with_asset_ids(item)
    return named

class AssetNames(Middleware):
    """Take a file's exact name wherever a tool takes an asset ID.

    The AI reads footage by file name — in a search's lines, in what the user
    says — and used to look each one up to get the ID a tool wanted. A name
    that is one file's becomes its ID; anything else goes on as it came.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> ToolResult:
        """Swap file names for IDs in the arguments, then run the tool.

        Args:
            context: The call.
            call_next: The rest of the chain, ending in the tool.

        Returns:
            What the tool returned.

        Raises:
            ToolError: If a name is shared by more than one file.
        """
        arguments = context.message.arguments
        if arguments:
            try:
                named = _with_asset_ids(arguments)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            context = context.copy(message=context.message.model_copy(update={"arguments": named}))
        return await call_next(context)

mcp.add_middleware(AssetNames())

# What each tool that changes things is doing, for a version saved without the user's words.
DOING = {
    "apply_edits": "修改時間軸", "compile_plan": "依計畫剪接", "save_plan": "存計畫", "amend_plan": "修改計畫",
    "revert_plan": "計畫回到舊版", "copy_plan": "複製計畫", "generate_subtitles": "上字幕",
    "create_project": "建立專案", "import_asset": "加入素材", "import_folder": "加入資料夾", "edit_asset": "寫素材備註",
    "restore_version": "回到舊版本", "branch_project": "開分支", "mark_version": "標記版本",
    "organize_library": "整理素材庫", "move_files": "搬動檔案", "set_asset_library": "移動分類",
    "find_music": "找歌", "compare_versions": "比較版本", "merge_version_parts": "合成兩版", "add_music_from_url": "從網路加入音樂", "resolve_comment": "回覆留言",
}
# What the AI programs call themselves when they connect, as a person would name them.
CLIENT_NAMES = {
    "claude-code": "Claude Code", "claude-ai": "Claude Desktop", "codex-mcp-client": "Codex",
    "gemini-cli-mcp-client": "Gemini CLI", "opencode": "opencode", "cherry-studio": "Cherry Studio",
    "lm-studio": "LM Studio",
}

def _client_name(context: MiddlewareContext) -> str:
    """The AI program on the other end of a call, by the name it gave when it connected."""
    try:
        params = context.fastmcp_context.session.client_params
        name = params.client_info.name if params is not None and params.client_info is not None else ""
    except (AttributeError, RuntimeError):
        name = ""
    return CLIENT_NAMES.get(name.lower(), name) or "AI 程式"

# The AI program making the current call, for what it says under its own name.
_calling_client: ContextVar[str] = ContextVar("calling_client", default="AI 程式")

class HistorySteps(Middleware):
    """Keep what one tool call changed as one version, said in words, under the AI's name."""

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> ToolResult:
        """Run the tool inside one step of the history.

        Args:
            context: The call; its `note` argument, when there is one, is the
                version's description.
            call_next: The rest of the chain, ending in the tool.

        Returns:
            What the tool returned.
        """
        arguments = context.message.arguments or {}
        note = arguments.get("note") if isinstance(arguments.get("note"), str) else ""
        name = context.message.name
        client = _client_name(context)
        token = _calling_client.set(client)
        try:
            with history.step(f"AI（{client}）", note=note, doing=DOING.get(name, name)):
                return await call_next(context)
        finally:
            _calling_client.reset(token)

mcp.add_middleware(HistorySteps())

# Tools that already are about the comments, so are not reminded of them.
ABOUT_COMMENTS = {"get_comments", "resolve_comment"}

class OpenComments(Middleware):
    """Remind the AI, on anything it does to a video, of what the user said about it and is still waiting.

    The editor cannot call the AI: a comment written there waits until the
    AI next looks. So every tool that names a project says how many are
    waiting, whether or not the user remembered to mention them.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> ToolResult:
        """Run the tool, then add the count of open comments on the project it was about.

        Args:
            context: The call.
            call_next: The rest of the chain, ending in the tool.

        Returns:
            The tool's result, with `open_comments` when there are any.
        """
        result = await call_next(context)
        project_id = (context.message.arguments or {}).get("project_id")
        if (context.message.name in ABOUT_COMMENTS or not isinstance(project_id, str)
                or not isinstance(result, ToolResult)):
            return result
        try:
            waiting = open_comments(project_id)
        except Exception:
            # A reminder must never cost the tool its answer: whatever went wrong reading the
            # comments, the tool's own result still goes back as it was.
            return result
        if not waiting:
            return result
        said = (f"the user left {waiting} comment(s) on this video in the editor that are not dealt with yet: "
                "read them with `get_comments`, and close each with `resolve_comment` once done")
        structured = result.structured_content
        if isinstance(structured, dict):
            structured = {**structured, "open_comments": {"count": waiting, "say": said}}
        return ToolResult(content=[*result.content, TextContent(type="text", text=said)],
                          structured_content=structured, meta=result.meta)

mcp.add_middleware(OpenComments())

# Compiled operations are validated the same way a client's are, so a plan cannot reach
# the timeline through a door the tool surface does not have.
_OPERATIONS = TypeAdapter(List[EditOperation])

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

def register_asset(filepath: str, library: Optional[str] = None) -> Asset:
    """Probe a media file and save it as an asset, in the footage or the music library.

    A new file without a picture is listened to: mostly somebody talking, it
    is footage; otherwise music, and a song is analyzed in the background
    straight away.

    Args:
        filepath: Path to the media file, absolute or relative to the server's
            working directory.
        library: Put it in this library rather than the one it sounds like,
            as when it is added on the editor's music page.

    Returns:
        The saved asset. A path that was imported before keeps its asset ID
        and its library.

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
    size = picture_size(info)
    if size is not None:
        asset.width, asset.height = size
    asset.recorded_at = recorded_at(info, path)
    if existing is not None:
        asset.notes = existing.notes
        asset.library = existing.library
        asset.source = existing.source
    if library in LIBRARIES:
        asset.library = library
    elif asset.library is None:
        asset.library = classify(asset)
    repo.save_asset(asset)
    _queue_music([asset])
    return asset

# Whether a song added to the music library starts being analyzed. Off where nothing should
# run in the background unasked, such as a test run.
ANALYZE_MUSIC_ON_ADD = True

def _queue_music(assets: Iterable[Asset]) -> List[str]:
    """Start analyzing songs of the music library that have no current analysis, behind other work.

    Args:
        assets: Files just added, moved to the music library, or found
            waiting; anything else among them is left alone.

    Returns:
        The IDs of the jobs started.
    """
    if not ANALYZE_MUSIC_ON_ADD:
        return []
    busy = {job.asset_id for job in repo.jobs_to_show(datetime.now(timezone.utc))[0] if job.kind == JobKind.ANALYZE}
    started = []
    for asset in assets:
        if library_of(asset) != MUSIC or asset.duration is None or asset.id in busy:
            continue
        existing = repo.get_analysis(asset.id)
        if existing is not None and existing.rhythm is not None and existing.music is not None \
                and not _is_stale(existing):
            continue
        # Behind the footage somebody asked about: a song can wait until it is wanted.
        started.append(_start_analysis(asset, False, "accurate", priority=1).job_id)
    return started

def tend_library() -> None:
    """Sort the files added before there were two libraries, and queue the songs not yet analyzed.

    Run once when the server or the editor starts, in the background: listening
    takes a moment per file. Whichever starts first does it; the other finds
    it taken and leaves it.
    """
    from filelock import FileLock, Timeout

    try:
        with FileLock(os.path.join(WORKSPACE_DIR, "library.lock"), timeout=0):
            for asset in repo.list_assets():
                if asset.library is None and os.path.exists(asset.path):
                    try:
                        asset.library = classify(asset)
                    except (OSError, RuntimeError, ValueError):
                        # Left where its picture says until it can be listened to; the next start tries again.
                        continue
                    repo.save_asset(asset)
            _queue_music(repo.list_assets())
    except Timeout:
        return

def _dated(assets: Mapping[str, Asset]) -> Dict[str, Asset]:
    """Make sure every asset knows when it was recorded, where anything says.

    Assets imported before the time was kept do not, so it is read off the
    file the first time it is asked for and saved, the same way `_sized` does
    the picture size.

    Args:
        assets: The assets, keyed by ID.

    Returns:
        The same assets, each with `recorded_at` filled in where it can be.
    """
    dated = {}
    for asset_id, asset in assets.items():
        if asset.recorded_at is None and os.path.isfile(asset.path):
            try:
                moment = recorded_at(probe_file(asset.path), asset.path)
            except (OSError, RuntimeError):
                moment = None
            if moment is not None:
                asset = asset.model_copy(update={"recorded_at": moment})
                repo.save_asset(asset)
        dated[asset_id] = asset
    return dated

def _sized(assets: Mapping[str, Asset]) -> Dict[str, Asset]:
    """Make sure every asset with a picture knows how big it is.

    Assets imported before the size was kept do not. Reframing needs it, so it
    is read off the file the first time it is asked for and saved, rather than
    making anybody import everything again.

    Args:
        assets: The assets, keyed by ID.

    Returns:
        The same assets, sized where the file could still be read. One whose
        file has gone is returned as it was; the render reports the missing
        file in its own words.
    """
    sized: Dict[str, Asset] = {}
    for asset_id, asset in assets.items():
        if asset.has_video and asset.width is None:
            try:
                size = picture_size(probe_file(asset.path))
            except (FileNotFoundError, RuntimeError):
                size = None
            if size is not None:
                asset = asset.model_copy(update={"width": size[0], "height": size[1]})
                repo.save_asset(asset)
        sized[asset_id] = asset
    return sized

@mcp.tool()
def import_asset(filepath: str) -> dict:
    """Register a local media file as an asset that clips can reference.

    This is what puts a file in the library, the only files a project can use;
    the user sees the library in the editor app, files are never copied, and
    a file the user takes out of the library there loses its asset ID.
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
    return _plain(register_asset(filepath).model_dump())

def _plain(value):
    """Turn the Decimals a model keeps its seconds in into plain numbers, all the way down.

    Times are kept as Decimal so the timeline adds up to the millisecond, but a
    client reading them gets strings — `"12.5"` — and compares or adds them
    wrongly. What leaves the server is numbers; what is stored stays exact.

    Args:
        value: What a tool is about to return.

    Returns:
        The same, with every Decimal a float.
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value

@mcp.tool()
def list_assets(
    kind: Optional[Literal["footage", "music"]] = None,
    folder_id: Optional[str] = None,
    text: Optional[str] = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> dict:
    """List imported assets a page at a time, and say which of them have been analyzed.

    A library grows into the hundreds, so read it the way the user sorted it:
    by the folders they made in the editor, or by part of a file name.

    Args:
        kind: `footage` — what was shot, and recordings of people talking —
            or `music`; omit for both. A song is listed with its `tempo` in
            beats per minute, the `moods` and `styles` that set it apart in
            this library, how it `ends` and where it is `fullest`, once
            analyzed; `analyzing` while it is. The mood is how it sounds to a
            model, right about half the time: say it as 「聽起來偏…」, never as
            a verdict.
        folder_id: Only the files filed in this folder, from `folders`. Its
            sub-folders are their own; ask for them by their own IDs.
        text: Only files whose name contains this.
        limit: Largest number of files to return.
        offset: How many matching files to skip, for the next page.

    Returns:
        A dictionary with `assets`, one short record per file — `id`, `name`,
        `duration` in seconds, `has_video`, `has_audio`, `analyzed`,
        `transcription` (`fast`, `accurate`, or null when speech was not
        transcribed), `stale`, `library`, `folder_id` and, where the file has
        them, its `notes` — how it may be used — in name order; `total`, how
        many match; `offset`; and `folders`, every folder of that library (the
        footage's when `kind` is omitted) with its `id`, `name` and
        `parent_id`. `stale` is true for an asset analyzed
        with detection settings or a speech model the server no longer uses:
        its analysis still works, but analyze those assets again before
        comparing them with each other. `inspect_media` has a file's full path,
        size and frame rate.
    """
    filed = {**repo.folder_of("assets"), **repo.folder_of("music")}
    matching = [
        asset for asset in repo.list_assets()
        if (kind is None or library_of(asset) == kind)
        and (folder_id is None or filed.get(asset.id) == folder_id)
        and (not text or text.lower() in os.path.basename(asset.path).lower())
    ]
    analyzing = analyzing_assets()
    heard = song_tags() if kind == MUSIC or any(library_of(asset) == MUSIC for asset in matching) else {}
    matching.sort(key=lambda asset: _natural_key(os.path.basename(asset.path)))
    assets = []
    for asset in matching[offset:offset + limit]:
        analysis = repo.get_analysis(asset.id)
        assets.append({
            "id": asset.id,
            "name": os.path.basename(asset.path),
            "duration": None if asset.duration is None else float(asset.duration),
            "has_video": asset.has_video,
            "has_audio": asset.has_audio,
            "analyzed": analysis is not None,
            "transcription": _transcription(analysis),
            "stale": analysis is not None and _is_stale(analysis),
            "library": library_of(asset),
            "folder_id": filed.get(asset.id),
            **({"notes": asset.notes} if asset.notes else {}),
            **(song_details(asset, analysis, heard, analyzing) if library_of(asset) == MUSIC
               else {"analyzing": True} if asset.id in analyzing else {}),
        })
    return {"assets": assets, "total": len(matching), "offset": offset,
            "folders": repo.list_folders(_folder_kind(kind or FOOTAGE))}

def song_tags() -> Dict[str, Dict[str, List[str]]]:
    """The moods and styles that set each analyzed song apart from the rest of the library.

    Returns:
        `moods` and `styles` by song.
    """
    sensed = {}
    for asset in repo.list_assets():
        if library_of(asset) != MUSIC:
            continue
        analysis = repo.get_analysis(asset.id)
        if analysis is not None and analysis.music is not None:
            sensed[asset.id] = analysis.music
    moods = clap.ranked({song: sense.moods for song, sense in sensed.items()})
    styles = clap.ranked({song: sense.styles for song, sense in sensed.items()})
    return {song: {"moods": moods.get(song, []), "styles": styles.get(song, [])} for song in sensed}

def _sounds_like(asset_id: str, analysis: MediaAnalysis, heard: Mapping[str, Mapping[str, List[str]]]) -> dict:
    """A song in a line: its moods and styles, how it ends and where it is fullest."""
    summary = music_energy.summary(analysis.music.energy)
    return {**heard.get(asset_id, {}), "ends": summary["ends"], "fullest": summary["fullest"],
            # The speech detector put it with music; the model hears talking. Asked about, not moved.
            **({"sounds_like_speech": True} if analysis.music.sounds_like_speech else {})}

def analyzing_assets() -> set:
    """The files an analysis is queued or running for."""
    return {job.asset_id for job in repo.jobs_to_show(datetime.now(timezone.utc))[0] if job.kind == JobKind.ANALYZE}

def song_details(asset: Asset, analysis: Optional[MediaAnalysis], heard: Mapping[str, Mapping[str, List[str]]],
                 analyzing: set) -> dict:
    """What is known about a song, for the AI's list and the editor's page alike.

    Args:
        asset: The song.
        analysis: Its analysis, if any.
        heard: From `song_tags`.
        analyzing: From `analyzing_assets`.

    Returns:
        Its `tempo`, its moods, styles, ending and fullest part once heard,
        and `analyzing` until then — queued or running, since every song
        added is analyzed by itself.
    """
    found: dict = {}
    if analysis is not None and analysis.rhythm is not None:
        found["tempo"] = analysis.rhythm.tempo
    if analysis is not None and analysis.music is not None:
        found.update(_sounds_like(asset.id, analysis, heard))
    if asset.id in analyzing or analysis is None or analysis.music is None:
        found["analyzing"] = True
    return found

@mcp.tool()
def find_music(description: str, min_seconds: Annotated[float, Field(ge=0)] = 0,
               limit: Annotated[int, Field(ge=1, le=30)] = 8) -> dict:
    """Find songs in the music library that sound like a description, closest first.

    Describe the music in English, the way the model was taught — "calm
    acoustic guitar music", "upbeat electronic music with a strong beat",
    "tense cinematic strings" — however the user put it. Give `min_seconds`
    as the length of the part it plays under, so a song that would loop is
    left out. A song not analyzed yet cannot be found; `analyzing` says how
    many are waiting.

    The match is a model's ear: read the top few back with their moods and
    styles as 「聽起來偏…」, and let the user listen before you build on one.

    Args:
        description: The music wanted, in English.
        min_seconds: Shortest song to consider, in seconds.
        limit: Most songs to return.

    Returns:
        `songs`, each with its `id`, `name`, `duration`, `tempo`, `match`
        (higher is closer), `moods`, `styles`, how it `ends` and where it is
        `fullest`; and `analyzing`, songs not analyzed yet.
    """
    heard = song_tags()
    vectors, found, waiting = {}, {}, 0
    for asset in repo.list_assets():
        if library_of(asset) != MUSIC or not os.path.exists(asset.path):
            continue
        analysis = repo.get_analysis(asset.id)
        if analysis is None or analysis.music is None or not analysis.music.vector:
            waiting += 1
            continue
        if float(asset.duration or 0) < min_seconds:
            continue
        vectors[asset.id] = analysis.music.vector
        found[asset.id] = (asset, analysis)
    songs = []
    for asset_id, match in clap.closest(description, vectors)[:limit]:
        asset, analysis = found[asset_id]
        songs.append({
            "id": asset_id, "name": os.path.basename(asset.path), "duration": float(asset.duration or 0),
            "tempo": analysis.rhythm.tempo if analysis.rhythm is not None else None, "match": match,
            **_sounds_like(asset_id, analysis, heard),
        })
    return {"songs": songs, "analyzing": waiting}

def _music_folder() -> str:
    """Where songs taken from the web are kept: beside the user's finished videos, where they can find them."""
    return os.path.join(str(output_dir()), "音樂")

def _web_song_plan(url: str, page: str, title: str, artist: str, licence: Optional[str], personal_use: bool) -> dict:
    """What taking a song would do, or why it cannot be taken."""
    site = fetch.site_of(url)
    if site is None or fetch.site_of(page) != site:
        raise ValueError("music is only taken from " + ", ".join(item.name for item in fetch.SITES.values())
                         + ", with the file's link and its page both on the same one")
    problem = fetch.licence_problem(site, licence, personal_use)
    if problem:
        raise ValueError(problem)
    terms = fetch.SITES[site]
    chosen = terms.licence or licence
    return {
        "site": terms.name, "title": title, "artist": artist, "licence": chosen, "terms": terms.terms,
        "credit": fetch.credit_line(site, title, artist, chosen, page), "personal_only": personal_use,
        "saved_in": _music_folder(), "url": url, "page": page,
    }

@mcp.tool()
def add_music_from_url(url: str, page: str, title: str, artist: str = "", licence: Optional[str] = None,
                       personal_use: bool = False, confirm_plan: Optional[str] = None) -> dict:
    """Take a song from one of the allowed music sites into the music library.

    Allowed: Pixabay Music, Mixkit, Incompetech (Kevin MacLeod), Free Music
    Archive and Wikimedia Commons; anything else is refused. Call without
    `confirm_plan` first: it downloads nothing and says the site, the
    licence, what it allows and the credit it asks for, and names that
    `plan`. Tell the user the title, the site and those terms, and only once
    they agree call again with `confirm_plan` set to that `plan`. On Free
    Music Archive and Wikimedia Commons each track has its own licence: read
    it off the page and pass it as `licence`. NC or ND is refused unless the
    user said the video is only for themselves (`personal_use`).

    The song is saved in the videos folder's 音樂, checked to be a song, and
    analyzed in the background like any other. `music_credits` gives the
    credit lines a video owes.

    Args:
        url: The audio file's own link, not its page.
        page: The track's page, where its licence is shown.
        title: The track's title.
        artist: Who made it, for the credit.
        licence: For a per-track site, as the page shows it: CC0, Public
            domain, CC BY, CC BY-SA, CC BY-NC, CC BY-ND, CC BY-NC-SA, CC BY-NC-ND.
        personal_use: The user said the video is only for themselves.
        confirm_plan: The `plan` the user agreed to; downloads it.

    Returns:
        Without `confirm_plan`: `plan` and what would be taken. With it: the
        song's `id`, `name`, `path` and `credit`.

    Raises:
        ValueError: If the site is not allowed, the licence does not allow
            the use, the plan is not the one shown, or the file is not a song.
    """
    planned = _web_song_plan(url, page, title, artist, licence, personal_use)
    key = hashlib.sha256(json.dumps(planned, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    if confirm_plan is None:
        return {"plan": key, **planned}
    if confirm_plan.strip() != key:
        raise ValueError("this is not the plan the user was shown: show them the new one and ask again")
    path = fetch.download(url, _music_folder(), title)
    try:
        streams = probe_file(path).get("streams", [])
    except RuntimeError:
        streams = []
    pictures = any(stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")
                   for stream in streams)
    if pictures or not any(stream.get("codec_type") == "audio" for stream in streams):
        # Ours, just downloaded, and not what was agreed to: it does not stay.
        os.remove(path)
        raise ValueError("what was downloaded is not a song: it has no sound, or it has a picture")
    _say_in_history(f"從 {planned['site']} 加入「{title}」")
    asset = register_asset(path, MUSIC).model_copy(update={"source": Source(
        site=planned["site"], page=page, url=url, title=title, artist=artist, licence=planned["licence"],
        credit=planned["credit"], personal_only=personal_use,
    )})
    repo.save_asset(asset)
    return {"id": asset.id, "name": os.path.basename(path), "path": path, "credit": planned["credit"]}

@mcp.tool()
def music_credits(project_id: str) -> dict:
    """The credit lines a video owes for the songs in it, ready for its description.

    Args:
        project_id: The video.

    Returns:
        `credits`, one line per song that asks for one; `personal_only`, the
        songs taken for a video only the user watches, which must not go in
        one that is published; and `unknown`, songs with no record of where
        they came from — ask the user how they are licensed.

    Raises:
        ValueError: If there is no such project.
    """
    project = repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    used = repo.get_assets({clip.asset_id for track in project.tracks if track.track_type == TrackType.AUDIO
                            for clip in track.clips})
    # A voice-over is on an audio track too, but owes nobody a credit.
    used = {asset_id: asset for asset_id, asset in used.items() if library_of(asset) == MUSIC}
    credits, personal, unknown = [], [], []
    for asset in used.values():
        if asset.source is None:
            unknown.append(os.path.basename(asset.path))
            continue
        if asset.source.credit and asset.source.credit not in credits:
            credits.append(asset.source.credit)
        if asset.source.personal_only:
            personal.append(asset.source.title)
    return {"credits": credits, "personal_only": personal, "unknown": unknown}

def _folder_kind(library: str) -> str:
    """The kind of folder a library's files are filed in."""
    return FOLDER_KINDS[library]

@mcp.tool()
def set_asset_library(asset_ids: List[str], library: Literal["footage", "music"], note: str = "") -> dict:
    """Move files between the footage and the music library.

    A file without a picture is put in one when it is added, by whether it is
    mostly somebody talking; that can be wrong — a song that is mostly a
    spoken intro, a recording of somebody singing. Move it when the user says
    so or when what you hear says otherwise, and say in your reply that you
    did. A song moved to music is analyzed for its beat in the background;
    footage is transcribed when analyzed, music never is.

    Args:
        asset_ids: The files.
        library: `footage` or `music`.
        note: The user's words for why, kept in the library's history.

    Returns:
        `moved`, the names of the files moved; a file already there is not
        among them. Each now sits at the top of that library's folders. A
        recording moved to footage is transcribed in the background.

    Raises:
        ValueError: If a file is not in the library, or a file with a
            picture is moved to music.
    """
    assets = [_get_asset(asset_id) for asset_id in asset_ids]
    pictured = [os.path.basename(asset.path) for asset in assets if asset.has_video]
    if library == MUSIC and pictured:
        raise ValueError(f"{', '.join(pictured)} has a picture, so it is footage: music is sound alone")
    moved = repo.move_to_library([asset.id for asset in assets], library)
    _queue_music(moved)
    if library == FOOTAGE and ANALYZE_MUSIC_ON_ADD:
        # Heard as a song, it was never transcribed: now it is listened to for what is said.
        for asset in moved:
            existing = repo.get_analysis(asset.id)
            if existing is not None and existing.transcript is None and asset.has_audio:
                _start_analysis(asset, True, "accurate", priority=1)
    return {"moved": [os.path.basename(asset.path) for asset in moved]}

@mcp.tool()
def edit_asset(asset_id: str, notes: str, note: str = "") -> dict:
    """Write down how a file may be used, so it is read wherever the file is.

    For an instruction about the footage itself — 「請消音」, 「可不放」,
    「0:12 以後手入鏡」. Users often put these in the file's name; when you see
    one there, follow it, and ask whether to keep it here too. The editor
    shows the same notes and the user can change them there.

    Args:
        asset_id: The file.
        notes: The notes, replacing what was there; empty clears them.
        note: Why, in the user's words, kept in the library's history.

    Returns:
        The file's `id`, `name` and `notes`.

    Raises:
        ValueError: If the file is not in the library.
    """
    asset = _get_asset(asset_id).model_copy(update={"notes": notes.strip()})
    repo.save_asset(asset)
    return {"id": asset.id, "name": os.path.basename(asset.path), "notes": asset.notes}

class CreateFolder(BaseModel):
    """Make a folder in the library, and put files in it."""

    action: Literal["create_folder"]
    name: str = Field(..., description="What to call it, in the user's words")
    library: Literal["footage", "music"] = Field(default="footage", description="Which library it is a folder of")
    parent_id: Optional[str] = Field(default=None, description="The folder it goes in; omit for the top")
    asset_ids: List[str] = Field(default_factory=list, description="Files to put in it straight away")

class RenameFolder(BaseModel):
    """Rename a folder of the library."""

    action: Literal["rename_folder"]
    folder_id: str
    name: str

class MoveFolder(BaseModel):
    """Put a folder inside another, or at the top."""

    action: Literal["move_folder"]
    folder_id: str
    parent_id: Optional[str] = Field(default=None, description="The folder to put it in; omit for the top")

class FileAssets(BaseModel):
    """Put files in a folder, or back at the top."""

    action: Literal["file_assets"]
    asset_ids: List[str]
    folder_id: Optional[str] = Field(default=None, description="The folder; omit for the top")

LibraryStep = Annotated[Union[CreateFolder, RenameFolder, MoveFolder, FileAssets], Field(discriminator="action")]

@mcp.tool()
def organize_library(steps: List[LibraryStep], note: str = "") -> dict:
    """Sort the library into folders: make, rename and move folders, and file the footage in them.

    Folders live only in clip-mcp — no file on disk moves — so this is yours
    to do when it helps, without asking first; then tell the user in a
    sentence what you made and filed. Nothing is ever deleted: footage the
    user does not want goes in a folder called 歸檔. The steps run in order
    and are kept as one version of the library. To move the files themselves
    on disk, see `move_files`.

    Args:
        steps: What to do, in order. `create_folder` can file footage in the
            new folder at once, since its ID is not known before.
        note: The user's words for why, kept in the library's history.

    Returns:
        `done`, each step in words, and `folders` and `music_folders`, every
        folder of each library now.

    Raises:
        ValueError: If a folder or file is unknown, or a name is taken.
    """
    names = {folder["id"]: folder["name"] for kind in ("assets", "music") for folder in repo.list_folders(kind)}
    done = []
    for step in steps:
        try:
            if isinstance(step, CreateFolder):
                made = repo.create_folder(uuid.uuid4().hex, _folder_kind(step.library), step.name, step.parent_id)
                names[made["id"]] = made["name"]
                if step.asset_ids:
                    _file_in(step.asset_ids, made["id"], step.library)
                done.append(f"新增資料夾「{made['name']}」" + (f"，放進 {len(step.asset_ids)} 個檔案" if step.asset_ids else ""))
            elif isinstance(step, RenameFolder):
                old = names.get(step.folder_id, step.folder_id)
                repo.rename_folder(step.folder_id, step.name)
                names[step.folder_id] = step.name
                done.append(f"資料夾「{old}」改名為「{step.name}」")
            elif isinstance(step, MoveFolder):
                repo.move_folder(step.folder_id, step.parent_id)
                done.append(f"把資料夾「{names.get(step.folder_id, step.folder_id)}」移到"
                            + (f"「{names.get(step.parent_id, step.parent_id)}」裡" if step.parent_id else "最上層"))
            else:
                _file_in(step.asset_ids, step.folder_id)
                done.append(f"把 {len(step.asset_ids)} 個檔案放進"
                            + (f"「{names.get(step.folder_id, step.folder_id)}」" if step.folder_id else "最上層"))
        except LookupError as exc:
            raise ValueError(str(exc).strip("'\"")) from exc
    _say_in_history("；".join(done))
    return {"done": done, "folders": repo.list_folders("assets"), "music_folders": repo.list_folders("music")}

def _file_in(asset_ids: List[str], folder_id: Optional[str], library: Optional[str] = None) -> None:
    """File footage or songs in a folder of their own library, or at its top.

    Args:
        asset_ids: The files.
        folder_id: The folder; its library's top when None.
        library: The library the folder belongs to, when it is known apart
            from the files.

    Raises:
        ValueError: If the files are of both libraries, or the folder is of the other one.
    """
    assets = [_get_asset(asset_id) for asset_id in asset_ids]
    kinds = {library_of(asset) for asset in assets} | ({library} if library else set())
    if len(kinds) > 1:
        raise ValueError("footage and music are filed in folders of their own: file them in separate steps, or "
                         "move a file to the other library first with set_asset_library")
    repo.file_items(_folder_kind(kinds.pop() if kinds else FOOTAGE), [asset.id for asset in assets], folder_id)

class FileMove(BaseModel):
    """One file to move on disk."""

    asset_id: str
    to_folder: str = Field(..., description="The folder on disk to move it to, as a full path; made if missing")
    name: Optional[str] = Field(default=None, description="A new file name; omit to keep its own. The extension "
                                                          "is kept when left off")

def _free_target(folder: str, name: str, taken: set) -> str:
    """Where a file can go without replacing anything: its name, or its name with a number added."""
    stem, extension = os.path.splitext(name)
    candidate, number = os.path.join(folder, name), 2
    while os.path.exists(candidate) or os.path.normcase(candidate) in taken:
        candidate = os.path.join(folder, f"{stem} ({number}){extension}")
        number += 1
    return candidate

def _inside(path: str, folder: str) -> bool:
    """Whether a path is a folder or inside it, whole names only: `D:\\work2` is not inside `D:\\work`."""
    path, folder = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(folder))
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)

def _planned_moves(moves: List[FileMove]) -> List[dict]:
    """Where each file would go, and what that touches."""
    users: Dict[str, set] = {}
    using: Dict[str, set] = {}
    for project in repo.list_projects():
        for track in project.tracks:
            for clip in track.clips:
                users.setdefault(clip.asset_id, set()).add(project.name or project.id)
                using.setdefault(clip.asset_id, set()).add(project.id)
    working = repo.jobs_to_show(datetime.now(timezone.utc))[0]
    taken: set = set()
    planned = []
    for move in moves:
        asset = _get_asset(move.asset_id)
        if not os.path.isabs(move.to_folder):
            raise ValueError(f"{move.to_folder} is not a full folder path")
        folder = os.path.abspath(move.to_folder)
        if any(_inside(folder, own) for own in (WORKSPACE_DIR, str(output_dir()))):
            raise ValueError("footage is not moved into clip-mcp's own folders")
        name = (move.name or os.path.basename(asset.path)).strip()
        if not name or os.path.basename(name) != name:
            raise ValueError(f"{name!r} is not a file name")
        if not os.path.splitext(name)[1]:
            name += os.path.splitext(asset.path)[1]
        wanted = os.path.join(folder, name)
        same = os.path.normcase(wanted) == os.path.normcase(asset.path)
        target = asset.path if same else _free_target(folder, name, taken)
        taken.add(os.path.normcase(target))
        planned.append({
            "asset_id": asset.id, "from": asset.path, "to": target, "missing": not os.path.exists(asset.path),
            "already_there": same, "numbered": not same and target != wanted,
            "new_folder": not os.path.isdir(folder),
            "megabytes": round(os.path.getsize(asset.path) / 1e6, 1) if os.path.exists(asset.path) else 0,
            "projects": sorted(users.get(asset.id, ())),
            # A render or an analysis has the file open: Windows will not let it move.
            "busy": any(job.asset_id == asset.id or job.project_id in using.get(asset.id, ()) for job in working),
        })
    return planned

def _plan_key(planned: List[dict]) -> str:
    """A short name for exactly this plan, so what is carried out is what the user was shown."""
    moves = [[plan["asset_id"], plan["from"], plan["to"]] for plan in planned]
    return hashlib.sha256(json.dumps(moves, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]

@mcp.tool()
def move_files(moves: List[FileMove], confirm_plan: Optional[str] = None, note: str = "") -> dict:
    """Move or rename footage on disk, keeping the library and every project pointing at it.

    Moving a user's own files is theirs to agree to. Call without
    `confirm_plan` first: it moves nothing, says where each file would go, and
    names that `plan`. Read it back — how many files, from where to where, any
    that get a number added — and call again with the same moves and
    `confirm_plan` set to that `plan` only once they agree; a plan that has
    changed since is refused. Nothing is overwritten: a name already taken
    gets ` (2)` added. Nothing is deleted: footage they do not want goes to an
    歸檔 folder like any other move. Projects keep working, since the library
    follows each file to where it went.

    Args:
        moves: Each file and the folder to move it to, with a new name if wanted.
        confirm_plan: The `plan` the user agreed to; carries it out.
        note: The user's words for why, kept in the library's history.

    Returns:
        Without `confirm_plan`: `plan`, and `planned`, each with `from`, `to`,
        `numbered` (a number was added), `new_folder`, `megabytes`, the
        `projects` using it, `busy` for a file a render or analysis has open,
        and `missing` for a file not found where the library has it. With it:
        `moved`, and `not_moved` with the reason.

    Raises:
        ValueError: If a file is not in the library, a folder is not a full
            path or is one of clip-mcp's own, or the plan is not the one shown.
    """
    planned = _planned_moves(moves)
    if confirm_plan is None:
        return {"plan": _plan_key(planned), "planned": planned}
    if confirm_plan.strip() != _plan_key(planned):
        raise ValueError("this is not the plan the user was shown: things have changed since, so show them the new "
                         "plan and ask again")
    _say_in_history(note or f"搬動了 {len(planned)} 個檔案")
    moved, not_moved = [], []
    for plan in planned:
        why = ("already there" if plan["already_there"] else "file not found" if plan["missing"]
               else "in use by a render or an analysis" if plan["busy"] else "")
        if why:
            not_moved.append({"from": plan["from"], "why": why})
            continue
        # Worked out again at the moment of moving: a file may have appeared there since the plan.
        target = _free_target(os.path.dirname(plan["to"]), os.path.basename(plan["to"]), set())
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.move(plan["from"], target)
        except OSError as exc:
            # Across drives a move is a copy first: one that stopped halfway is ours to take back.
            if os.path.exists(plan["from"]) and os.path.exists(target):
                try:
                    os.remove(target)
                except OSError:
                    pass
            not_moved.append({"from": plan["from"], "why": str(exc)})
            continue
        asset = _get_asset(plan["asset_id"])
        repo.save_asset(asset.model_copy(update={"path": target}))
        moved.append({"from": plan["from"], "to": target})
    return {"moved": moved, "not_moved": not_moved}

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
            assets.append(register_asset(path))
        except (OSError, RuntimeError, ValueError) as error:
            skipped.append({"path": path, "reason": str(error)})

    total = sum((asset.duration for asset in assets if asset.duration is not None), Decimal(0))
    return _plain({"assets": [asset.model_dump() for asset in assets], "total_duration": total, "skipped": skipped})

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
    # A fast transcript is one the user chose, not one made with a model since replaced.
    speech_model = analysis.recipe.speech_model
    current = speech_model if transcription_of(speech_model) == "fast" else whisper_model_name()
    return analysis.recipe.differs_from(current_recipe(
        current, speaker_model=speaker_model_name(), rhythm_model=rhythm_model_name(), clap_model=clap.MODEL_ID,
    ))

def _transcription(analysis: Optional[MediaAnalysis]) -> Optional[str]:
    """Say how an asset's speech was transcribed: `fast`, `accurate`, or None when it was not."""
    return transcription_of(None if analysis is None or analysis.transcript is None else analysis.transcript.model)

def _transcribed(asset: Asset, transcribe: bool) -> bool:
    """Whether an analysis of this file transcribes it: never a song, whatever was asked."""
    return transcribe and asset.has_audio and library_of(asset) != MUSIC

def _start_analysis(asset: Asset, transcribe: bool, transcription: str, language: Optional[str] = None,
                    prompt: Optional[str] = None, chinese_variant: Optional[str] = None, diarize: bool = True,
                    speakers: Optional[int] = None, priority: int = 0) -> Job:
    """Queue one file's analysis.

    Args:
        asset: The file.
        transcribe: Whether speech is transcribed; never for a song.
        transcription: `accurate` or `fast`.
        language: Spoken language code, or None to detect it.
        prompt: Text that guides transcription.
        chinese_variant: Script and regional variant for a Chinese transcript.
        diarize: Whether to tell the voices apart.
        speakers: How many people are talking, when known.
        priority: Lower goes first; a song added by itself waits behind footage.

    Returns:
        The job.
    """
    job = Job(kind=JobKind.ANALYZE, asset_id=asset.id, priority=priority)
    job.work_dir = os.path.join(WORKSPACE_DIR, "jobs", job.job_id)
    with_speech = _transcribed(asset, transcribe)
    job.memory_estimate = resources.analysis_memory_bytes(
        with_speech,
        whisper_model_for(transcription),
        duration=float(asset.duration),
        diarize=with_speech and diarize,
        detect_faces=asset.has_video,
    )
    return job_manager.start_job(job, {
        "transcribe": transcribe,
        "transcription": transcription,
        "language": language,
        "prompt": prompt,
        "chinese_variant": chinese_variant,
        "diarize": diarize,
        "speakers": speakers,
    })

def _plan_analyses(
    assets: Sequence[Asset], transcribe: bool, transcription: str, again: bool,
) -> Tuple[List[Tuple[Asset, str]], List[str], List[str]]:
    """Decide which files an analysis starts, and with which transcription.

    A file whose transcript is accurate is never made fast: asked for fast,
    it is skipped while current, and analyzed accurately again when it is
    out of date or `again` is given. A fast transcript asked for accurate is
    redone without `again`.

    Args:
        assets: The files asked for.
        transcribe: Whether speech is transcribed.
        transcription: `accurate` or `fast`, as asked.
        again: Redo files whose analysis is current.

    Returns:
        `(to_start, skipped, upgrading)`: each file to start with the
        transcription it gets, the files left alone, and the files whose fast
        transcript is being replaced.
    """
    to_start: List[Tuple[Asset, str]] = []
    skipped: List[str] = []
    upgrading: List[str] = []
    for asset in assets:
        existing = repo.get_analysis(asset.id)
        had = _transcription(existing)
        upgrade = _transcribed(asset, transcribe) and transcription == "accurate" and had == "fast"
        # Heard as a song, never transcribed; footage now, so it is listened to again for what is said.
        heard_as_song = (existing is not None and existing.transcript is None and existing.rhythm is not None
                         and _transcribed(asset, transcribe))
        if not again and not upgrade and not heard_as_song and existing is not None and not _is_stale(existing):
            skipped.append(asset.id)
            continue
        if upgrade:
            upgrading.append(asset.id)
        to_start.append((asset, "accurate" if had == "accurate" else transcription))
    return to_start, skipped, upgrading

def _estimate(to_start: Sequence[Tuple[Asset, str]], transcribe: bool, diarize: bool,
              upgrading: Sequence[str]) -> Optional[dict]:
    """How long the files about to be analyzed take here, or None when nothing in them is transcribed."""
    footage: Dict[str, float] = {"accurate": 0.0, "fast": 0.0}
    for asset, transcription in to_start:
        if _transcribed(asset, transcribe):
            footage[transcription] += float(asset.duration)
    if not any(footage.values()):
        return None
    here = machine.profile()
    found = {"tier": here.tier, "footage_minutes": round(sum(footage.values()) / 60, 1),
             **speed.estimate(repo, here, footage, diarize)}
    if upgrading:
        found["note"] = ("files being upgraded to an accurate transcript are analyzed whole again, so reading"
                         " their picture and sound adds a little on top")
    return found

@mcp.tool()
def analyze_asset(
    asset_ids: List[str],
    transcribe: bool = True,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    chinese_variant: Optional[Literal["zh-TW", "zh-HK", "zh-Hant", "zh-Hans"]] = None,
    diarize: bool = True,
    speakers: Annotated[Optional[int], Field(ge=1, le=20)] = None,
    again: bool = False,
    transcription: Literal["accurate", "fast"] = "accurate",
    dry_run: bool = False,
) -> dict:
    """Start analyzing what assets contain, in the background.

    Takes every file to analyze in one call — a whole imported folder at
    once — and starts one job per file. A file that already has a current
    analysis is left alone unless `again` is true, so the same call can be
    repeated over a folder after a few files are added.

    One decoding pass detects scene changes, black and frozen picture, and
    silences, and alongside them measures each shot — exposure, contrast,
    blur, motion, camera shake — each second of sound (level, peak, noise
    floor, and how squared off the waveform is where it peaks), and each
    second of picture for faces: how many, how big the largest is, and where
    it sits in frame. If `transcribe` is true and the asset has audio, speech
    is then transcribed locally with word-level timestamps that stay accurate
    on long recordings, which is the slow part by far, and the voices are told
    apart so that each clip knows who was speaking.

    Returns immediately: poll `get_job` with all the job IDs at once until
    they are done, then read the results with `get_analysis`, or search them
    through `query_clips` once a semantic timeline is built.

    Everything runs on this machine. The models are downloaded the first time
    they are needed — the speech model is the large one, the rest are tens of
    megabytes — and they are kept inside the workspace, so deleting the
    workspace deletes them too. `stage` says when a download is what a job is
    waiting on.

    Args:
        asset_ids: IDs of the assets to analyze. The same settings apply to
            all of them; analyze files that need different ones in separate
            calls.
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
        again: Analyze files that already have a current analysis too,
            replacing it — for example with a `prompt` or `speakers` the first
            run did not have. Only what would come out different is redone:
            the transcript, the voices when `speakers` is given; the scan and
            the faces are kept.
        transcription: "accurate", the default, or "fast": about twice as
            quick on a computer without an NVIDIA card, but with noticeably
            more misheard words and looser word timings, so cuts land less
            exactly. It is the user's trade to make: on a low-tier computer,
            or a mid-tier one with more than about an hour of footage, call
            with `dry_run` first and ask them with both times; use it unasked
            only when they said they want it quick. A file with a fast
            transcript asked for "accurate" is transcribed again without
            `again` — the whole analysis is redone — and an accurate
            transcript is never replaced with a fast one.
        dry_run: Start nothing; say what would start and how long it would
            take, as `estimates` with one estimate for "accurate" and one for
            "fast", so the user can choose before waiting on either.

    Returns:
        A dictionary with `jobs`, one `asset_id`, `job_id`, `status` and
        `stage` per job started, and `skipped`, the assets left alone because
        their analysis is current. Analyses run one or two at a time, so that
        several of them cannot exhaust the machine's memory between them;
        `stage` says what a job that has not started yet is waiting for.
        Waiting costs nothing and needs no action: keep polling `get_job`.
        `upgrading` lists the files whose fast transcript is being replaced
        with an accurate one. With `dry_run`, `would_start` lists the files
        in place of `jobs`.

        When anything started is transcribed, `estimate` says how long that
        takes here for all of it together: `footage_minutes`,
        `transcription_minutes` and `speakers_minutes` as `[low, high]` (high
        null when there is no telling; `speakers_minutes` null when voices are
        not told apart), the computer's `tier`, and `basis` — this computer's
        own past runs, or a rough figure from its hardware — and a `note`
        when upgraded files are analyzed whole. Tell the user in a sentence
        when it runs past a few minutes.

    Raises:
        ValueError: If an asset does not exist or has no known duration;
            nothing is started then.
    """
    assets = [_get_asset(asset_id) for asset_id in dict.fromkeys(asset_ids)]
    for asset in assets:
        if asset.duration is None:
            raise ValueError(f"asset {asset.id} has no known duration and cannot be analyzed")

    if dry_run:
        choices = {}
        for choice in ("accurate", "fast"):
            to_start, skipped, upgrading = _plan_analyses(assets, transcribe, choice, again)
            choices[choice] = _estimate(to_start, transcribe, diarize, upgrading)
        to_start, skipped, upgrading = _plan_analyses(assets, transcribe, transcription, again)
        return {"would_start": [asset.id for asset, _ in to_start], "skipped": skipped, "upgrading": upgrading,
                "estimates": choices}

    started: List[dict] = []
    to_start, skipped, upgrading = _plan_analyses(assets, transcribe, transcription, again)
    for asset, chosen in to_start:
        job = _start_analysis(asset, transcribe, chosen, language, prompt, chinese_variant, diarize, speakers)
        started.append({"asset_id": asset.id, "job_id": job.job_id, "status": job.status.value, "stage": job.stage})
    result: dict = {"jobs": started, "skipped": skipped, "upgrading": upgrading}
    found = _estimate(to_start, transcribe, diarize, upgrading)
    if found is not None:
        result["estimate"] = found
    return result

# Provisional (roadmap §13): the longest stretch heard again at once, short enough to come back
# inside the time most AI clients allow a tool call.
LISTEN_AGAIN_SECONDS = 30.0
# Provisional (roadmap §13): the longest `get_job` waits, under the same allowance.
MAX_JOB_WAIT_SECONDS = 50.0

def _bare_words(text: str) -> str:
    """Text reduced to what is said, for comparing two hearings: no spaces or punctuation."""
    return re.sub(r"[\s\W_]+", "", text)

@mcp.tool()
def listen_again(
    start: float,
    end: float,
    asset_id: Optional[str] = None,
    language: Optional[str] = None,
    job_id: Optional[str] = None,
) -> dict:
    """Hear one stretch of a file or of a finished render again, to check a word, a line or the mix.

    The transcript is one long pass, and a word it misheard — 「沒那麼順」
    written as 「怎麼那麼順」 — is misheard for everything that reads it. This
    transcribes only that stretch of sound, from scratch: no prompt, no text
    before it. Where the two disagree, or the words come back unsure, believe
    neither — give the user `audio_path` to listen to and ask.

    Given a render's `job_id` instead of a file, it hears what the viewer
    will: whether a word was cut into at an edit, and, from `levels`, whether
    the music dips under speech and two songs really cross-fade where the
    plan says.

    Args:
        start: Where the stretch starts, in seconds of the file or the render.
        end: Where it ends; at most 30 seconds after `start`.
        asset_id: The file to hear. Give this or `job_id`.
        language: Spoken language code; omit to detect it.
        job_id: A finished render to hear, preview or full.

    Returns:
        A dictionary with `heard`, the new hearing with unsure words in ⟦ ⟧;
        `words`, each with `start`, `end` and its `probability`; `levels`,
        the sound's level in dBFS every `every_seconds` from `start` — speech
        sits well above a ducked bed, a gap is near -60; `audio_path`, a WAV of
        the stretch for the user to play; for a file, `stored`, what the
        transcript says there, or null, and `agrees`, whether the two say the
        same words; for a render, `captions`, the ones on screen in the
        stretch, and `changed_since`, true when the project has been edited
        since it was rendered.

    Raises:
        ValueError: If neither or both of a file and a render are given, it
            has no sound, or the stretch is empty, too long or outside it.
    """
    if (asset_id is None) == (job_id is None):
        raise ValueError("give either asset_id, to hear a file, or job_id, to hear a render")
    if job_id is not None:
        return _listen_to_render(job_id, start, end, language)
    asset = _get_asset(asset_id)
    if not asset.has_audio:
        raise ValueError(f"asset {asset_id} has no sound to hear again")
    _hearable(start, end)
    if asset.duration is not None and start >= float(asset.duration):
        raise ValueError(f"the file is only {float(asset.duration):g}s long")
    known = repo.get_analysis(asset_id)
    transcript = known.transcript if known is not None else None
    keep_as = os.path.join(WORKSPACE_DIR, "temp", "listen", f"{asset_id[:8]}-{start:.2f}-{end:.2f}.wav")
    fresh = listen_again_in_file(
        asset.path, start, end, language or (transcript.language if transcript else None), keep_as,
        chinese_variant=transcript.chinese_variant if transcript else None,
    )
    stored = None
    if transcript is not None:
        said = [word.text for segment in transcript.segments for word in segment.words
                if word.end > start and word.start < end]
        stored = "".join(said).strip() or None
    plain_heard = "".join(segment.text for segment in fresh.segments)
    return {
        **_heard(fresh, keep_as, start),
        "stored": stored,
        "agrees": stored is not None and _bare_words(plain_heard) == _bare_words(stored),
    }

# Provisional (roadmap §13): how long each level `listen_again` reports covers — fine
# enough to see a fade or a dip under a word, coarse enough that thirty seconds is sixty numbers.
LEVEL_EVERY_SECONDS = 0.5

def _heard(fresh: Transcript, keep_as: str, start: float) -> dict:
    """What `listen_again` says about any stretch it heard.

    Args:
        fresh: The new hearing, in seconds of the file or render.
        keep_as: The stretch's sound, kept for the user to play.
        start: Where the stretch starts, which the levels count from.

    Returns:
        `heard`, `words`, `levels` and `audio_path`.
    """
    return {
        "heard": "".join(marked_text(segment) for segment in fresh.segments),
        "words": [
            {"text": word.text.strip(), "start": word.start, "end": word.end, "probability": word.probability}
            for segment in fresh.segments for word in segment.words
        ],
        "levels": {"start": start, "every_seconds": LEVEL_EVERY_SECONDS,
                   "db": levels_every(keep_as, LEVEL_EVERY_SECONDS)},
        "audio_path": keep_as,
    }

def _finished_render(job_id: str) -> Tuple[Job, str]:
    """Find a render that has finished, and the file it made.

    Args:
        job_id: The render's job.

    Returns:
        `(job, path)`.

    Raises:
        ValueError: If it is not a render, has not finished, or its file is gone.
    """
    job = job_manager.get_job(job_id)
    if job is None or job.kind != JobKind.RENDER:
        raise ValueError(f"render {job_id} not found; `render_project` gives the job ID of one")
    if job.status != JobStatus.COMPLETED:
        raise ValueError(f"render {job_id} is {job.status.value}; wait for it with `get_job` first")
    if not job.output_path or not os.path.exists(job.output_path):
        raise ValueError(f"the file render {job_id} made is no longer there; render again")
    return job, job.output_path

def _rendered_from(job: Job) -> Optional[Project]:
    """The project a render was made from, as it is now; None when it is gone."""
    return repo.get_project(job.project_id) if job.project_id else None

def _changed_since(job: Job) -> bool:
    """Whether the project a render was made from has been edited since.

    Args:
        job: The render.

    Returns:
        True when its project's version has moved on from the one rendered.
    """
    project = _rendered_from(job)
    return project is not None and job.project_version is not None and project.version != job.project_version

def _hearable(start: float, end: float) -> None:
    """Refuse a stretch `listen_again` cannot take: empty, or longer than it hears at once.

    Raises:
        ValueError: Saying which.
    """
    if end <= start:
        raise ValueError(f"the stretch has to end after it starts ({start}s to {end}s)")
    if end - start > LISTEN_AGAIN_SECONDS:
        raise ValueError(f"hear a stretch of at most {LISTEN_AGAIN_SECONDS:g} seconds at a time; this one is "
                         f"{end - start:g}s")

def _listen_to_render(job_id: str, start: float, end: float, language: Optional[str]) -> dict:
    """`listen_again` over a finished render rather than a file.

    Args:
        job_id: The render.
        start: Where the stretch starts, in seconds of the render.
        end: Where it ends.
        language: Spoken language code, or None for the footage's own.

    Returns:
        What `_heard` says, with the `captions` on screen in the stretch and
        `changed_since`.
    """
    job, path = _finished_render(job_id)
    _hearable(start, end)
    project = _rendered_from(job)
    if project is not None and not _changed_since(job) and start >= float(project.duration):
        raise ValueError(f"the render is only {float(project.duration):g}s long")
    keep_as = os.path.join(WORKSPACE_DIR, "temp", "listen", f"render-{job_id[:8]}-{start:.2f}-{end:.2f}.wav")
    known = None
    for clip in (project.base_video_track.clips if project and project.base_video_track else []):
        analysis = repo.get_analysis(clip.asset_id)
        if analysis is not None and analysis.transcript is not None:
            known = analysis.transcript
            break
    fresh = listen_again_in_file(
        path, start, end, language or (known.language if known else None), keep_as,
        chinese_variant=known.chinese_variant if known else None,
    )
    shown = [cue for cue in (place_cues(project, project.subtitles) if project and job.captioned else [])
             if float(cue.end) > start and float(cue.start) < end]
    return {
        **_heard(fresh, keep_as, start),
        "captions": [f"{format_timestamp(float(cue.start))} {cue.text}" for cue in shown],
        "changed_since": _changed_since(job),
    }

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
        A dictionary with the asset's `notes` where it has them — how it may
        be used, which every cut from it has to respect — its `duration`, `analyzed_at`,
        `transcription` (`fast` — expect misheard words, check them before
        trusting a cut to a word — `accurate`, or null), `doubtful_spots`
        (how many stretches in the whole file the speech model was unsure
        of; null for a transcript made before that was kept), and the
        overlapping `scenes`, `black_frames`, `frozen_frames`, `silences`,
        `shots`, and `transcript` (`language`, `model`, `chinese_variant`, and
        `segments` with `start`, `end`, and `text`), or null for `transcript`
        if speech was not transcribed.

        In `text`, words the speech model was unsure of are wrapped in ⟦ ⟧.
        The marks are only here: captions and the video never carry them, so
        never copy them into anything the user will see. Read the marked
        stretches with what the footage is about in mind — names and terms
        are what it mishears — and fix what is wrong through
        `generate_subtitles`' `fix_words` or by correcting the caption; check
        by listening before trusting a cut to a marked word.

        `shots` carries one record per shot in the range — `exposure`,
        `contrast`, `blur`, `motion` and `shake` — while `sound` and `faces`
        summarise the range's per-second records rather than listing them,
        since an hour of those is thousands of rows. `faces` says how much of
        the range had anybody on screen, how many there usually were, and
        where the largest face sat across the frame and how far it moved,
        which is what a vertical reframe needs. `speakers` lists the stretches
        each voice held; the labels are this file's own and mean nothing
        outside it. `rhythm` is for a file in the music library, and gives
        the `tempo` in beats per minute and the `beats` in the range; its `tempo` is null when the music has no steady pulse, and
        the whole of it is null for footage, or for music analyzed before
        beats were measured. `music`, for a song, has the `moods` and
        `styles` that set it apart in the library (how it sounds to a model,
        right about half the time on mood), its `sections` of quiet, middle
        and full energy, where it is `fullest`, and how it `ends`.

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
                {**segment.model_dump(exclude=exclude), "text": marked_text(segment)}
                for segment in analysis.transcript.segments
                if segment.end > start and segment.start < end
            ],
        }
    notes = _get_asset(asset_id).notes
    return {
        "asset_id": asset_id,
        **({"notes": notes} if notes else {}),
        "duration": analysis.duration,
        "analyzed_at": analysis.analyzed_at.isoformat(),
        "recipe": analysis.recipe.model_dump(),
        "stale": _is_stale(analysis),
        "transcription": _transcription(analysis),
        "doubtful_spots": None if analysis.transcript is None else doubtful_spots(analysis.transcript.segments),
        "range": {"start": start, "end": end},
        "scenes": _overlapping(analysis.scenes, start, end),
        "black_frames": _overlapping(analysis.black_frames, start, end),
        "frozen_frames": _overlapping(analysis.frozen_frames, start, end),
        "silences": _overlapping(analysis.silences, start, end),
        "shots": _overlapping(analysis.shots, start, end),
        "sound": sound_note(_covering(analysis.sound, start, end)),
        "faces": framing_note(_covering(analysis.faces, start, end)),
        "speakers": _overlapping(analysis.speakers, start, end),
        "rhythm": None if analysis.rhythm is None else {
            "tempo": analysis.rhythm.tempo,
            "beats": [beat for beat in analysis.rhythm.beats if start <= beat < end],
        },
        "music": None if analysis.music is None or library_of(_get_asset(asset_id)) != MUSIC else {
            **song_tags().get(asset_id, {}),
            **music_energy.summary(analysis.music.energy),
        },
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

def _semantic_clip(clip_id: str, timeline_id: Optional[str]) -> SemanticClip:
    """Fetch a semantic clip by the ID the AI was shown.

    Args:
        clip_id: `u0034`, or a whole ID.
        timeline_id: The timeline a short one is in. Left out, the one
            timeline that has such a clip; never a guess between several,
            since the same short ID is a different moment in each.

    Returns:
        The clip.

    Raises:
        ValueError: If there is no such clip, or a short ID is in more than
            one timeline and none was named.
    """
    if ":" in clip_id:
        whole = clip_id
    elif timeline_id is not None:
        whole = whole_clip_id(clip_id, _require_timeline(timeline_id).id)
    else:
        holding = repo.timelines_holding(clip_id)
        if len(holding) > 1:
            raise ValueError(f"{clip_id} is in {len(holding)} timelines ({', '.join(holding)}); "
                             "give the timeline_id of the one meant")
        whole = whole_clip_id(clip_id, holding[0]) if holding else clip_id
    clip = repo.get_semantic_clip(whole)
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
            handheld. Sound, in dBFS: `loudness`, `peak`, `noise_floor` (the
            hiss under everything) and `flatness`, high where the waveform is
            squared off, so a `peak` near 0 with a high `flatness` is a
            clipped recording. Faces: `faces` is how many were up, averaged
            over the clip; `face_share` how much of the frame the largest
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
            between searches. The first search downloads the model, about
            130 MB, and embeds every clip once, about a second per thousand.

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
        # Read through in the order it was shot, or ranked by meaning: either needs every match to sort.
        limit=MAX_TIMELINE_CLIPS if brief or about else offset + limit,
    )
    relevance: Dict[str, float] = {}
    if about and about.strip():
        ranked = _rank_by_meaning(clips, about.strip())
        by_id = {clip.id: clip for clip in clips}
        clips = [by_id[clip_id] for clip_id, _ in ranked]
        relevance = {clip_id: round(score, 3) for clip_id, score in ranked}
        page = clips[offset:offset + limit]
        if brief:
            assets = repo.get_assets({clip.asset_id for clip in page})
            return {
                "timeline_id": timeline.id,
                "lines": [
                    f"{relevance[clip.id]:.3f} " + _clip_line(
                        clip, os.path.basename(assets[clip.asset_id].path) if clip.asset_id in assets else clip.asset_id)
                    for clip in page
                ],
                **_file_notes(assets.values()),
                "truncated": offset + limit < len(clips),
            }
        return {
            "timeline_id": timeline.id,
            "clips": [{**_clip_summary(clip), "relevance": relevance[clip.id]} for clip in page],
            "truncated": offset + limit < len(clips),
        }
    if brief:
        assets = _dated(repo.get_assets({clip.asset_id for clip in clips}))
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
        everything = [clip for clip in repo.query_semantic_clips(timeline_id, level=level, limit=MAX_TIMELINE_CLIPS)
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
    stored = repo.clip_vectors(wanted)
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
        repo.save_clip_vectors(
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
        f"{clip.source_range.start:.1f}-{clip.source_range.end:.1f}", clip.kind.value,
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
    repo.save_semantic_timeline(timeline, [*utterances, *built])

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
        timeline_id: The timeline the clip is in; needed only when more than
            one timeline has been built.

    Returns:
        The clip: its `clip_id`, `timeline_id`, `asset_id`, `level`, `kind`,
        `source_range`, `safe_in` / `safe_out`, full `text`, `speaker`,
        `scores`, and `tags` with where each came from. With `include_words`,
        also `words`, each with `start`, `end`, and `text`.

    Raises:
        ValueError: If no clip has that ID.
    """
    clip = _semantic_clip(clip_id, timeline_id)
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
    return _plain(record)

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
        timeline_id: The timeline the clips are in; needed only when more
            than one timeline has been built.

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
        clip = _semantic_clip(clip_id, timeline_id)
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
        timeline_id: The timeline the clips are in; needed only when more
            than one timeline has been built.

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

# Provisional (roadmap §13): how far either side of a transition `view_render` looks —
# clear of the mix, near enough to be the same shot.
AROUND_TRANSITION_SECONDS = 0.15

@mcp.tool()
def view_render(
    job_id: str,
    at: Literal["even", "transitions", "captions"] = "even",
    times: Optional[List[Annotated[float, Field(ge=0)]]] = None,
    start: Annotated[float, Field(ge=0)] = 0,
    end: Optional[float] = None,
    count: Annotated[int, Field(ge=1, le=MAX_STORYBOARD_TILES)] = 12,
) -> ToolResult:
    """Look at frames of a finished render: what the viewer will see, captions and transitions included.

    `preview_project` shows the footage a cut is made of; this shows the file
    that came out, so it is how to check what only rendering makes — captions
    burned in where they belong and readable, transitions that really happen,
    the framing of another shape. Render a preview, look here, fix what is
    wrong, and only then render the final cut.

    The frames are returned to you as a tool result, which means they leave
    this machine if you are not running on it.

    Args:
        job_id: The render, preview or full, from `render_project`.
        at: `even` spreads `count` frames over the stretch; `transitions`
            takes one just before, one in the middle and one just after each
            transition; `captions` one from the middle of each caption on
            screen.
        times: Exact seconds of the render to look at instead.
        start: Only look from here, in seconds of the render.
        end: Only look up to here; omit for the end.
        count: How many frames `even` takes.

    Returns:
        A text block naming each tile's time and what it is, then the frames
        as one JPEG. It says so when the project has changed since the render,
        when a render has no captions burned in, or when more frames were
        wanted than one sheet holds.

    Raises:
        ValueError: If the job is not a finished render or its file is gone.
    """
    job, path = _finished_render(job_id)
    project = _rendered_from(job)
    length = float(project.duration) if project is not None else None
    last = length if end is None else (end if length is None else min(end, length))
    said: List[str] = []
    wanted: List[tuple] = []
    if times:
        wanted = [(seconds, "") for seconds in times]
    elif at == "transitions":
        base = project.base_video_track if project else None
        for clip in (base.clips if base else []):
            if clip.transition_in is None:
                continue
            cut, seconds = float(clip.timeline_in), float(clip.transition_in.seconds)
            if cut < start or (last is not None and cut > last):
                continue
            kind = clip.transition_in.kind
            wanted += [(max(0.0, cut - seconds - AROUND_TRANSITION_SECONDS), f"before the {kind}"),
                       (cut - seconds / 2, f"middle of the {kind}"),
                       (cut + AROUND_TRANSITION_SECONDS, f"after the {kind}")]
        if not wanted:
            said.append("there is no transition in this stretch of the cut")
    elif at == "captions":
        if not job.captioned:
            said.append("this render has no captions burned in; render with burn_subtitles to look at them")
        placed = place_cues(project, project.subtitles) if project else []
        wanted = [((float(cue.start) + float(cue.end)) / 2, f"{cue.cue_id} {cue.text}") for cue in placed
                  if float(cue.end) > start and (last is None or float(cue.start) < last)]
    else:
        stop = last if last is not None else start + 1
        step = (stop - start) / count
        wanted = [(start + step * (index + 0.5), "") for index in range(count)]
    if len(wanted) > MAX_STORYBOARD_TILES:
        said.append(f"{len(wanted)} frames wanted, the first {MAX_STORYBOARD_TILES} shown; look at the rest with "
                    "a later `start`")
        wanted = wanted[:MAX_STORYBOARD_TILES]
    if _changed_since(job):
        said.append("the project has been edited since this render, so it may no longer show what the cut is now")
    if length is not None:
        wanted = [(min(seconds, length - 0.05), what) for seconds, what in wanted]
    if not wanted:
        return ToolResult(content=["Nothing to show: " + "; ".join(said)])
    shots = [(path, round(seconds, 3), f"#{number}  {format_timestamp(seconds)}")
             for number, (seconds, _) in enumerate(wanted, start=1)]
    listing = [f"#{number}: {seconds:.2f}s ({format_timestamp(seconds)})" + (f" | {what}" if what else "")
               for number, (seconds, what) in enumerate(wanted, start=1)]
    return ToolResult(content=[
        "\n".join([*said, "Frames of the render, in reading order:", *listing]),
        Image(data=storyboard_sheet(shots), format="jpeg"),
    ])

@mcp.tool()
def create_project(
    name: Annotated[str, Field(min_length=1, max_length=120)],
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
        name: What to call the project, such as 'EP1 台北' — required. It is
            what the editor app and `list_projects` show, and what the
            finished video is called, so name it for what the video is about,
            not 'test' or 'project'. If the user has not said, propose one in
            your read-back. Change it later with a `rename_project` operation.
        width: Output width in pixels; must be even.
        height: Output height in pixels; must be even.
        fps_num: Frame rate numerator, e.g. 30000 for 29.97 fps.
        fps_den: Frame rate denominator, e.g. 1001 for 29.97 fps.

    Returns:
        The new project serialized as a dictionary, including its generated
        `id` and initial `version`.
    """
    if not name.strip():
        raise ValueError("a project needs a name: what the video is about, such as 'EP1 台北'")
    project = Project(
        id=str(uuid.uuid4()), name=name.strip(),
        width=width, height=height, fps_num=fps_num, fps_den=fps_den,
    )
    repo.add_project(project)
    return _plain(project.model_dump())

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
    return _plain(state)

@mcp.tool()
def apply_edits(project_id: str, expected_version: int, operations: list[EditOperation], note: str = "") -> dict:
    """Apply a batch of edit operations to a project with optimistic locking.

    Operations are applied in order and atomically: if any operation fails,
    or the resulting timeline is invalid, the project is left unchanged. On
    success the project version is incremented by one. Call `get_project`
    first to obtain the current version.

    Editing behaves like a magnetic timeline: `insert_clip` shifts later
    clips on the same track to make room, and `trim_clip` and `delete_clip`
    shift them to close or open space unless `ripple` is false.
    `reorder_clip` moves a clip elsewhere in the sequence, keeping its cut
    and audio settings. `split_clip` cuts one clip into two halves that
    together occupy exactly what the original did, so nothing else moves.
    `add_clip` and `move_clip` use absolute positions and never move other
    clips. Clips on other tracks never move.

    The first video track is the base and defines the output length: gaps
    are rendered as black frames with silence, and audio past its last clip
    is cut. Further video tracks are drawn on top of the base, each clip
    inside the box its `layout` gives. Background music goes on an audio
    track. Each operation's own description says what it changes: a clip's
    sound running outside its picture (J and L cuts), its transition, colour
    and speed, a track's ducking, the structure markers, the captions and how
    they are drawn.

    The resulting timeline must satisfy these rules:
    - Clips stay within their asset's duration and do not overlap on a track.
      Sound is the exception: a lead or a lag overlaps the neighbouring
      clip's sound, which is what a J or an L cut is, but still comes from
      inside the file.
    - Audio and video fades fit within their clip.
    - Only clips above the base video track take a `layout`.
    - Only audio tracks duck under speech.
    - Video-track clips use assets with video; audio-track clips use assets
      with audio.

    Args:
        project_id: ID of the project to edit.
        expected_version: Project version the caller last read. The request is
            rejected if the project has been modified since then.
        note: The user's own words for this change — 「結尾的晚安多留一下」 —
            kept as what this version is in the project's history.
        operations: Edit operations to apply, each identified by its `action`:
            `add_track`, `add_clip`, `insert_clip`, `trim_clip`, `move_clip`,
            `delete_clip`, `split_clip`, `reorder_clip`, `fit_track`,
            `set_clip_audio`, `set_clip_look`, `set_track_audio`,
            `set_subtitles`, `edit_subtitle`, or `add_subtitle`. Captions stored
            or corrected here are also remembered against the file they belong
            to, and the next `generate_subtitles` over that file, in any
            project, starts from them — unless `this_project_only` keeps one
            to this project. Caption times can be given in the cut's own
            seconds with `timeline_start` and `timeline_end`.

    Returns:
        A dictionary with the `status` and the project's `new_version`, and,
        when captions were stored or corrected, `captions`: each one touched,
        as it now reads, one line apiece, marked `(this project only)` where it
        is — read them back rather than assume — and `captions_kept`, which says
        where the rest are now remembered.

    Raises:
        ValueError: If the project does not exist, the version does not match,
            an operation references a missing or duplicate track or clip, or
            the resulting timeline breaks a rule above.
    """
    before = repo.get_project(project_id)
    saved = _apply(project_id, expected_version, operations)
    prepare_playback(saved)
    result = {"status": "success", "new_version": saved.version}
    gap = _music_ends_early(saved)
    if gap:
        result["music"] = gap
    touched = _touched_captions(before, saved, operations)
    if touched:
        shared = [cue for cue in touched if not cue.local]
        if shared:
            repo.remember_captions(shared)
        result["captions"] = [_caption_line(saved, cue) for cue in touched]
        if not shared:
            result["captions_kept"] = "in this project only"
        elif len(shared) < len(touched):
            result["captions_kept"] = ("remembered against their footage, so every project that captions it starts "
                                       "from them — except the ones marked (this project only)")
        else:
            result["captions_kept"] = "remembered against their footage: every project that captions it starts from them"
    return result

# Music stopping this much before the video does leaves an ending without it. Provisional.
MUSIC_GAP_SECONDS = 3.0

def _music_ends_early(project: Project) -> Optional[str]:
    """Say when the music laid by hand stops well before the video ends, with what would fix it."""
    music = [track for track in project.tracks if track.track_type == TrackType.AUDIO and track.duck_under_speech]
    clips = [clip for track in music for clip in track.clips]
    if not clips:
        return None
    ends = max(float(clip.timeline_out) for clip in clips)
    left = float(project.duration) - ends
    if left <= MUSIC_GAP_SECONDS:
        return None
    return (f"the music stops at {ends:.1f}s and the last {left:.0f}s have none: fit_track with loop to repeat it, "
            f"a longer song (find_music with min_seconds), or say that is wanted")

def _touched_captions(before: Optional[Project], project: Project, operations: Sequence) -> List[SubtitleCue]:
    """Find the captions a batch of edits stored or corrected, as they now stand.

    By comparing the project's captions before and after rather than reading
    the operations: a caption added at a time in the cut does not say which
    file or second it lands on until it has been applied.

    Args:
        before: The project before the edits.
        project: The project after the edits.
        operations: The edits.

    Returns:
        Every caption set, added or corrected, in source order; a deleted one
        is left out.
    """
    if before is None or any(isinstance(op, SetSubtitlesOp) for op in operations):
        return list(project.subtitles)
    if not any(isinstance(op, (EditSubtitleOp, AddSubtitleOp)) for op in operations):
        return []
    was = {cue.id: cue for cue in before.subtitles}
    return [cue for cue in project.subtitles if was.get(cue.id) != cue]

def _caption_line(project: Project, cue: SubtitleCue) -> str:
    """Write one caption as a line to read back: where it lands and what it says.

    Args:
        project: The project it is on.
        cue: The caption.

    Returns:
        Its ID, its time in the cut (or that it is not in the cut), and its text.
    """
    placed = [item for item in place_cues(project, [cue])]
    at = f"{format_timestamp(float(placed[0].start))}" if placed else "not in the cut"
    return (f"{cue.id} {at} {cue.text}" + (f" / {cue.secondary}" if cue.secondary else "")
            + (" (this project only)" if cue.local else ""))

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
        cuts_by_asset_id)`.

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
    footage = {clip.asset_id for clip in clips}
    songs = {cue.asset_id for cue in plan.music.cues if cue.asset_id} if plan.music else set()
    # Where each file may be cut without splitting a word, and where each song's beat
    # falls. The compiler is handed these numbers rather than the transcripts and the
    # sound they come from: reading those is interpretation, and the compiler has to stay
    # a pure function of what it is given.
    cuts = {}
    for asset_id in footage | songs:
        analysis = repo.get_analysis(asset_id)
        if analysis is not None:
            cuts[asset_id] = clean_cuts(analysis)
    return timeline, {clip.id: clip for clip in clips}, children, _dated(repo.get_assets(footage | songs)), cuts

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
    timeline = repo.get_semantic_timeline(plan.timeline_id)
    if timeline is None:
        raise ValueError(f"semantic timeline {plan.timeline_id} not found; build one before planning against it")

    stamped = plan.model_copy(update={"timeline_input_hash": timeline.input_hash})
    stored = repo.save_plan(stamped, note)
    problems, notes = check_plan(stored, *_plan_context(stored))
    newest = repo.get_semantic_timeline(None)
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
    and level that covers most of it. Those are listed in `matched_nearby`
    with both stretches: read them back, since a sentence that was split or
    joined may now say a little more or less.

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
            taken out of the plan first. A rejected clip with no match is
            simply not carried over, since there is nothing to reject.
    """
    plan = repo.get_plan(plan_id) if version is None else repo.get_plan_version(plan_id, version)
    if plan is None:
        raise ValueError(f"plan {plan_id} not found" + ("" if version is None else f" at version {version}"))
    copy = plan.model_copy(deep=True, update={"id": str(uuid.uuid4()), "version": 1})
    target = timeline_id or plan.timeline_id
    if target != plan.timeline_id:
        if repo.get_semantic_timeline(target) is None:
            raise ValueError(f"semantic timeline {target} not found")
        everything = 1_000_000
        mapping, nearby, shifts = _matched_clips(
            repo.query_semantic_clips(plan.timeline_id, limit=everything),
            repo.query_semantic_clips(target, limit=everything),
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
        for item in copy.selections:
            shift = shifts.get(item.clip_id, 0.0)
            if shift and item.trim.kind == TrimKind.RANGE:
                # A range is counted from the clip's own start, which moved.
                item.trim.from_seconds = max(0.0, round((item.trim.from_seconds or 0.0) + shift, 3))
                if item.trim.to_seconds is not None:
                    item.trim.to_seconds = max(0.001, round(item.trim.to_seconds + shift, 3))
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
        nearby = []
    said = f"copied from {plan.id} v{plan.version}" + (f" onto {target}" if target != plan.timeline_id else "")
    saved = save_plan(copy, f"{said}: {note}" if note else said)
    used = {item.clip_id for item in plan.selections} | {clip for shot in plan.broll for clip in
                                                         (shot.clip_id, shot.over_clip_id)}
    nearby = [match for match in nearby if match["clip_id"] in used]
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
        that are not exact, and how far each of those clips' start moved.
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
                shifts[clip.id] = round(start - best.source_range.start, 3)
                nearby.append({
                    "clip_id": clip.id, "now": best.id,
                    "was": [round(start, 2), round(end, 2)],
                    "is": [round(best.source_range.start, 2), round(best.source_range.end, 2)],
                    "text": best.text,
                })
        mapping[clip.id] = found.id if found else None
    return mapping, nearby, shifts

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
    `set_target` for a new length or platform, and `drop_beat` for
    「這整段不要」, which files every selection in it under the rejections.

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

    timeline = repo.get_semantic_timeline(amended.timeline_id)
    if timeline is None:
        raise ValueError(f"the plan is written against timeline {amended.timeline_id}, which no longer exists")
    done = "; ".join(describe_amendment(amendment) for amendment in amendments)
    stored = repo.save_plan(
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
    earlier = repo.get_plan_version(plan_id, version)
    if earlier is None:
        kept = ", ".join(str(item["version"]) for item in repo.list_plan_versions(plan_id)) or "none"
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
    versions = repo.list_plan_versions(plan.id)
    listed, before = [], None
    for item in versions:
        now = repo.get_plan_version(plan.id, int(item["version"]))
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
    stored = repo.save_plan(
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

    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    pieces, left_out = as_left(plan_pieces(plan, clips, children, assets, cuts), project)
    notes = notes + left_out
    covers, _, _ = broll_covers(plan, pieces, clips, assets, cuts)
    blocked = check_recompile(plan, project, pieces, covers, music_beds(plan, pieces, assets, cuts))
    if blocked:
        raise ValueError("compiling would undo work already on this project:\n- " + "\n- ".join(blocked))

    built, provenance = compile_operations(plan, clips, children, assets, project, cuts)
    operations = _OPERATIONS.validate_python(built)

    def record(edited: Project) -> None:
        """Write onto each compiled clip where it came from."""
        for track in edited.tracks:
            for clip in track.clips:
                origin = provenance.get(clip.id)
                if origin is not None and track.id in COMPILED_TRACKS:
                    clip.from_plan_id = origin["from_plan_id"]
                    clip.from_clip_ids = origin["from_clip_ids"]
                    clip.pinned = origin["pinned"]

    saved = _apply(project_id, expected_version, operations, stamp=record)
    prepare_playback(saved)
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
    heard = song_tags()
    found = []
    for cue in plan.music.cues if plan.music is not None else []:
        if cue.asset_id is None or any(song["id"] == cue.asset_id for song in found):
            continue
        asset = repo.get_assets([cue.asset_id]).get(cue.asset_id)
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
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if project.base_video_track is None or not project.base_video_track.clips:
        raise ValueError(f"project {project_id} has nothing on its sequence to listen to")
    checks = housekeeping.sound_dir(WORKSPACE_DIR, project_id)
    folder = os.path.join(checks, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)
    # A sound check is looked at once; only the newest of a project's is kept.
    housekeeping.prune(checks, {os.path.basename(folder)})
    mix = os.path.join(folder, _output_name(project, "sound").replace(".mp4", ".m4a"))
    voice, music = os.path.join(folder, "voice.wav"), os.path.join(folder, "music.wav")
    assets, voices = _referenced_assets(project), _voices(project)
    talking = _talking(project, voices)
    command = renderer.build_sound(
        project, assets, mix, voice, music, loudness_target, voices=voices, talking=talking,
    )
    graph = os.path.join(folder, "graph.txt")
    if loudness_target is not None:
        raw = os.path.join(folder, "raw.wav")
        measured = subprocess.run(
            graph_from_file(renderer.build_mix(project, assets, raw, voices=voices, talking=talking), graph),
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
    if len(shots) > MAX_STORYBOARD_TILES:
        lines.append(f"the sheet shows the first {MAX_STORYBOARD_TILES} of {len(shots)} shots")
    shape = FRAME_SHAPES.get(aspect) if aspect else None
    image = storyboard_sheet(
        shots[:MAX_STORYBOARD_TILES], aspect=shape[0] / shape[1] if shape else None,
        borders=borders[:MAX_STORYBOARD_TILES],
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

@mcp.tool()
def preview_project(
    project_id: str,
    count: Annotated[int, Field(ge=1, le=MAX_STORYBOARD_TILES)] = 12,
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    follow_faces: bool = True,
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
        frame: Show the cut in another shape, the way `render_project` with
            the same `frame` would render it; omit for the project's own.
        follow_faces: Crop around the face the way the render does; the
            same setting as `render_project`'s.

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
    project = _reshaped(project, frame)

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

    framing = _framing(project, assets, follow_faces)
    fps = Fraction(project.fps_num, project.fps_den)

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

    shots: List[tuple] = []
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
            framed = framing.get(clip.id)
            shots.append((
                asset.path, seconds, f"#{number}  {format_timestamp(at)}  {label}",
                placement_at(framed, clip, seconds, fps),
            ))
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
        plan = repo.get_plan(plan_id)
        timeline = repo.get_semantic_timeline(plan.timeline_id) if plan is not None else None
        if timeline is not None:
            files |= set(timeline.asset_ids)
    analyses = {
        asset_id: analysis
        for asset_id in (files or set(heard))
        if (analysis := repo.get_analysis(asset_id)) is not None
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
        `reused`, how many came from captions already reviewed; `sources`,
        how many came from each of `reviewed`, `this_project` and
        `transcript`; `to_check`, the IDs of the ones straight from the
        transcript, those with the most words the speech model was unsure of
        first — proofread those, the others somebody has read; `glossary`,
        the words corrected; `assets_without_transcript`, the video sources
        that still need `analyze_asset` before they can be captioned; and
        `overlapping`, how many captions land on top of the one before them
        once the cut puts them on screen, which is worth a look when a
        narration track talks over footage that speaks for itself.

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
    people = _joined_people(project, speakers)
    speakers = {
        asset_id: [
            turn.model_copy(update={"speaker": people.get((asset_id, turn.speaker), turn.speaker)})
            for turn in turns
        ]
        for asset_id, turns in speakers.items()
    }
    style = project.caption_style
    glossary = repo.update_glossary(fix_words) if fix_words else repo.glossary()
    proposed = [
        _corrected(cue, glossary) for cue in timeline_cues(
            project, transcripts, max_characters=max_characters, max_seconds=max_seconds,
            silences=silences, speakers=speakers,
            max_units=caption_geometry(project.width, project.height, style).max_units if style.single_line else None,
        )
    ]
    reviewed = {asset_id: repo.reviewed_captions(asset_id) for asset_id in {cue.asset_id for cue in proposed}}
    reviewed.update({
        clip.asset_id: repo.reviewed_captions(clip.asset_id)
        for clip in (base.clips if base else []) if clip.asset_id not in reviewed
    })
    # What this project kept to itself comes first, then what was corrected on the footage
    # for every project, then the transcript: each fills only what the ones before left.
    local = [cue.model_copy(update={"id": ""}) for cue in project.subtitles if cue.local]

    def taken(cue: SubtitleCue, over: Sequence[SubtitleCue]) -> bool:
        return any(cue.asset_id == known.asset_id and _overlaps(cue, known) for known in over)

    kept = [
        cue for cue in proposed
        if not any(_overlaps(cue, known) for known in reviewed.get(cue.asset_id, ())) and not taken(cue, local)
    ]
    reused = [known.model_copy(update={"id": ""}) for known_list in reviewed.values() for known in known_list]
    reused = [cue for cue in reused if place_cues(project, [cue]) and not taken(cue, local)]
    saved = _apply(project_id, expected_version, [SetSubtitlesOp(cues=[*kept, *reused, *local])])
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
        "sources": {source: sum(1 for cue in saved.subtitles if heard[cue.id]["from"] == source)
                    for source in ("reviewed", "this_project", "transcript")},
        "to_check": [cue.id for cue in to_check],
        "glossary": glossary,
        "assets_without_transcript": untranscribed,
        "overlapping": overlapping,
    }

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
            remembered[cue.asset_id] = repo.reviewed_captions(cue.asset_id)
        if any(_overlaps(cue, known) and known.text == cue.text for known in remembered[cue.asset_id]):
            sources[cue.id] = {"from": "reviewed"}
            continue
        if cue.asset_id not in analyses:
            analyses[cue.asset_id] = repo.get_analysis(cue.asset_id)
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
            if _bare_words("".join(word.text for word in words)) == _bare_words(cue.text):
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
    shared = min(first.source_end, second.source_end) - max(first.source_start, second.source_start)
    return float(shared) > CAPTION_SLIVER

def _corrected(cue: SubtitleCue, glossary: Mapping[str, str]) -> SubtitleCue:
    """Correct a proposed caption's text with the workspace glossary.

    Args:
        cue: The caption as proposed.
        glossary: Heard words mapped to meant ones.

    Returns:
        The caption with every glossary word replaced, and its word timings
        kept where the correction left them fitting.
    """
    text = cue.text
    for heard, meant in glossary.items():
        text = text.replace(heard, meant)
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
    heard = _caption_sources(project)
    window = [
        _plain({**cue.model_dump(exclude=None if words else {"words"}), **heard.get(cue.cue_id, {})})
        for cue in placed
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

def _delivered(job_id: str) -> bool:
    """Whether a full render finished and its file was saved, so its working folder can go."""
    job = repo.get_job(job_id)
    return job is not None and job.status == JobStatus.COMPLETED

def _delivery_name(project: Project, kind: str, extension: str, captioned: bool = False) -> str:
    """Name a file saved for the user after the project: `EP1 台北.mp4`, `EP1 台北 封面.jpg`.

    Args:
        project: The project it came from.
        kind: `output`, `cover` or `timeline`, with `-portrait` and the like
            for another shape.
        extension: With its dot.
        captioned: Whether its captions are burned in.

    Returns:
        A file name; the caller makes it unique in its folder.
    """
    stem = _output_name(project, "x")[: -len("_x.mp4")]
    return housekeeping.delivery_name(stem, kind, extension, captioned)

# A preview is rendered with its short side this long, from pictures cached at that size.
PREVIEW_SHORT_SIDE = 480
# Cached preview pictures nobody has used for this long are thrown away. Housekeeping,
# not a judgement: a picture is re-rendered in seconds if it is wanted again.
PICTURE_CACHE_DAYS = 14

def _preview_sized(project: Project, side: int = PREVIEW_SHORT_SIDE) -> Project:
    """Shrink a project to the size its preview is rendered at.

    Args:
        project: The project.
        side: The preview's short side.

    Returns:
        A copy with its short side `side` long, or the project itself when it
        is already that small.
    """
    short = min(project.width, project.height)
    if short <= side:
        return project
    scale = side / short
    even = lambda value: max(2, int(round(value * scale / 2)) * 2)
    return project.model_copy(update={"width": even(project.width), "height": even(project.height)})

def _picture_cache() -> str:
    """Find the cache of preview pictures, clearing out what has not been used lately.

    Returns:
        The cache directory.
    """
    folder = os.path.join(WORKSPACE_DIR, "cache", "pictures")
    os.makedirs(folder, exist_ok=True)
    stale = time.time() - PICTURE_CACHE_DAYS * 86400
    for entry in os.scandir(folder):
        try:
            # A part file is a picture some render never finished; a day is long enough
            # for any render still writing one to be done with it.
            limit = time.time() - 86400 if entry.name.endswith(".part.mp4") else stale
            if entry.stat().st_mtime < limit:
                os.remove(entry.path)
        except OSError:
            pass
    return folder

# The shapes a cut is delivered in, as width to height.
FRAME_SHAPES = {"landscape": (16, 9), "portrait": (9, 16), "square": (1, 1)}

def _reshaped(project: Project, frame: Optional[str]) -> Project:
    """Give a project another shape for one render, leaving the stored one alone.

    The short side is kept, because that is the side a platform's resolution
    is quoted by: 1080p is 1080 across the short side whichever way up it is.

    Args:
        project: The project.
        frame: `landscape`, `portrait`, `square`, or None for as it is.

    Returns:
        A copy at the new size, or the project itself when no shape was asked for.
    """
    if frame is None:
        return project
    across, down = FRAME_SHAPES[frame]
    short = min(project.width, project.height)
    even = lambda value: max(2, int(round(value / 2)) * 2)
    if across >= down:
        width, height = even(short * across / down), short
    else:
        width, height = short, even(short * down / across)
    return project.model_copy(update={"width": width, "height": height})

def _framing(project: Project, assets: Mapping[str, Asset], follow_faces: bool = True) -> Dict[str, Framing]:
    """Work out how each clip sits in the frame: where its crop goes, or whether it is shown whole.

    Args:
        project: The project, at the size it is being rendered at.
        assets: Its files.
        follow_faces: Whether a crop may follow the face in a shot.

    Returns:
        A framing per clip that is not simply cropped from the middle, keyed
        by clip ID.
    """
    faces = {}
    for asset_id in assets:
        analysis = repo.get_analysis(asset_id)
        if analysis is not None and analysis.faces:
            faces[asset_id] = analysis.faces
    return frame_project(project, _sized(assets), faces, follow_faces)

@mcp.tool()
def render_project(
    project_id: str,
    is_preview: bool = False,
    loudness_target: Annotated[Optional[float], Field(ge=-40, le=-5)] = DEFAULT_LOUDNESS_TARGET,
    burn_subtitles: bool = False,
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    follow_faces: bool = True,
    allow: Optional[List[Literal[
        "bad_picture", "clipping", "music", "mid_speech", "repeated", "continuity", "unplanned", "captions",
        "length",
    ]]] = None,
) -> dict:
    """Start rendering a project to an MP4 file in the background.

    Before a final render, read skill://clip-editing/platform-conventions.md;
    after any render, look at it with `view_render`.

    Before anything is rendered the cut is checked the way `check_render`
    checks it, and a full render is refused while it finds anything: black
    or frozen picture on screen, a voice recorded clipping, a caption too
    tall for the frame, or a length far from what the plan asked for. Fix
    the cut, or name the kinds of finding the user has decided to live with
    in `allow`. Previews are never refused — they are how you look — but
    say what they found.

    Returns immediately. Poll `get_job` with the returned `job_id` until the
    job reaches a terminal status, or stop it with `cancel_job`. Rendering
    runs in a separate worker process and continues even if this server
    restarts.

    Args:
        project_id: ID of the project to render.
        is_preview: If true, render a fast preview, 480 pixels on its short
            side, instead of the full-quality output. A preview renders each
            clip's picture once and keeps it: after a change, only the clips
            the change touched are rendered again, so a second preview of a
            long cut takes a fraction of the first.
        loudness_target: Loudness of the finished file in LUFS. The default,
            -14, is what streaming platforms normalize to, so clips recorded
            on different devices come out at one consistent level instead of
            jumping. Pass null to leave the mix exactly as the clip volumes
            set it.
        burn_subtitles: If true, burn the project's stored captions into the
            picture. Store them first with a `set_subtitles` operation;
            `generate_subtitles` proposes them from the transcripts.
        frame: Render the same cut in another shape — `landscape` (16:9),
            `portrait` (9:16) or `square` — without touching the project.
            The short side stays the project's, so a 1920x1080 project comes
            out at 1080x1920 portrait and 1080x1080 square. Call once per
            shape to deliver one cut to several platforms. Omit it for the
            project's own size.
        follow_faces: Where a shot is wider (or taller) than the frame it is
            shown in, crop it around the face the analysis found rather than
            from the middle. The crop holds still while the face stays inside
            it and cuts to a new framing once the face has sat near the edge
            for a couple of seconds — it never pans. Shots with no face seen
            are cropped from the middle either way.
        allow: Kinds of finding to render in spite of, after the user has
            heard about them and said so. Never fill this in on your own.

    Returns:
        A dictionary with the `job_id`, the initial `status`, `stage`, and the
        absolute `output_path` the file will be written to. A full render is
        saved in the user's videos folder under the project's name, such as
        `Videos\\clip-mcp\\EP1 台北.mp4`, with what sets it apart from the
        project's other videos in brackets — `EP1 台北（直式、字幕）.mp4` — and
        `(2)` is added rather than overwrite one already there; `get_job` gives the final path once it is
        done. A preview is a working file in the workspace, and only the
        newest preview of each project is kept.
        Renders and analyses run one or two at a time, so that several of them
        cannot exhaust the machine's memory between them; `stage` says what a
        job that has not started yet is waiting for, and is null when it
        started immediately. `findings` lists what the check found that this
        render went ahead in spite of. The parts of the video, where it has
        markers, are carried in the file as chapters. A preview also says how
        many clip `pictures` it `rendered` and how many it `reused`.

    Raises:
        ValueError: If the project does not exist, contains no clips, uses an
            unsupported feature, or the check found something not allowed.
    """
    return _start_render(
        project_id, is_preview, loudness_target, burn_subtitles, frame, follow_faces, allow, PREVIEW_SHORT_SIDE,
    )

def _start_render(
    project_id: str,
    is_preview: bool,
    loudness_target: Optional[float],
    burn_subtitles: bool,
    frame: Optional[str],
    follow_faces: bool,
    allow: Optional[List[str]],
    preview_side: int,
) -> dict:
    """Start a render; what `render_project` does, with the preview's size open.

    The editor app previews at 720 pixels so captions and mouths can be read;
    a client previews at 480, which is quicker and plenty to judge a cut by.
    The size is not on the tool, so a client is never asked to choose it.

    Args:
        project_id: As for `render_project`, and the rest likewise.
        preview_side: The short side of a preview, in pixels.

    Returns:
        As `render_project`.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")

    project = _reshaped(project, frame)
    findings = _findings(project, burn_subtitles)
    if is_preview:
        project = _preview_sized(project, preview_side)
    blocking = [finding for finding in findings if finding.check not in (allow or [])]
    if blocking and not is_preview:
        raise ValueError(
            "the cut is not ready to render:\n- " + "\n- ".join(finding.message for finding in blocking)
            + "\nFix these, or once the user has said to go ahead anyway, render again with allow="
            + str(sorted({finding.check for finding in blocking}))
        )
    kind = ("preview" if is_preview else "output") + (f"-{frame}" if frame else "")
    if is_preview:
        _make_room()
    housekeeping.sweep_renders(WORKSPACE_DIR, _delivered)
    job = Job(kind=JobKind.RENDER, project_id=project_id, project_version=project.version, captioned=burn_subtitles,
              preview=is_preview)
    delivered: Optional[str] = None
    if is_preview:
        job.work_dir = os.path.join(housekeeping.preview_dir(WORKSPACE_DIR, project_id), job.job_id)
        job.output_path = os.path.join(job.work_dir, _output_name(project, kind))
    else:
        # Rendered in the workspace and moved out whole once finished, so the videos
        # folder never holds half a file.
        job.work_dir = os.path.join(housekeeping.render_dir(WORKSPACE_DIR), job.job_id)
        delivered = housekeeping.unique_path(str(output_dir()), _delivery_name(project, kind, ".mp4", burn_subtitles))
    render_path = os.path.join(job.work_dir, _output_name(project, kind))

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

    chapters_path = None
    listed, _ = chapters(project)
    if listed:
        os.makedirs(job.work_dir, exist_ok=True)
        chapters_path = os.path.join(job.work_dir, "chapters.txt")
        with open(chapters_path, "w", encoding="utf-8") as handle:
            handle.write(chapter_metadata(listed))

    assets = _referenced_assets(project)
    framing = _framing(project, assets, follow_faces)
    voices = _voices(project)
    talking = _talking(project, voices)
    pieces: list = []
    if is_preview:
        cache = _picture_cache()
        command, pieces = renderer.build_incremental(
            project, assets, render_path, cache,
            loudness_target=loudness_target, subtitle_path=subtitle_path, framing=framing,
            chapters_path=chapters_path, voices=voices, talking=talking,
        )
        # What this preview reads from the cache counts as used, so it is kept.
        for argument in command:
            if argument.startswith(cache) and os.path.exists(argument):
                os.utime(argument)
    else:
        command = renderer.build_command(
            project, assets, render_path, loudness_target=loudness_target, subtitle_path=subtitle_path,
            framing=framing, chapters_path=chapters_path, voices=voices, talking=talking,
        )
    # One input per clip, all opened at once, is what a render's memory use is made of.
    job.memory_estimate = resources.render_memory_bytes(command.count("-i"))
    spec = {
        "command": command, "duration_seconds": float(renderer.output_duration(project)), "pieces": pieces,
        "render_path": render_path,
    }
    if delivered:
        job.output_path = delivered
        spec["deliver"] = delivered
    else:
        spec["prune"] = os.path.dirname(job.work_dir)
    if loudness_target is not None:
        mix_path = os.path.join(job.work_dir, "mix.wav")
        spec["loudness"] = {
            "mix_command": renderer.build_mix(project, assets, mix_path, voices=voices, talking=talking),
            "mix_path": mix_path, "target": loudness_target,
        }
    job = job_manager.start_job(job, spec)
    # Every clip on a picture track is one picture, the sequence's and the ones drawn over it.
    pictures = sum(len(track.clips) for track in project.video_tracks)

    result = {
        "job_id": job.job_id, "status": job.status.value, "stage": job.stage, "output_path": job.output_path,
        "findings": [{"check": finding.check, "message": finding.message} for finding in findings],
    }
    if is_preview:
        result["pictures"] = {"rendered": len(pieces), "reused": pictures - len(pieces)}
    return result

def _talking(project: Project, voices: Mapping[str, float]) -> Optional[Talking]:
    """Work out from the transcripts where a cut has somebody talking, and how loud.

    Args:
        project: The project about to be rendered.
        voices: Its narration tracks, from `_voices`, with the gain each is
            heard at.

    Returns:
        Where somebody is heard talking, and the gain that brings each music
        track's songs to that talking, as `levels.talking` works them out.
        None when any clip whose words are heard was never transcribed: then
        nobody knows where all the talking is, and the music ducks by level.
    """
    assets = _referenced_assets(project)
    return talking_in(
        project, assets, _analyses(project), voices, lambda clip: song_level(assets[clip.asset_id].path, clip),
    )

def _voices(project: Project) -> Dict[str, float]:
    """Find the tracks a render treats as a voice recorded apart from the picture.

    Args:
        project: The project about to be rendered.

    Returns:
        As `voice_keys`, with each voice clip's talking measured from its file.
    """
    assets = _referenced_assets(project)

    def loudness(clip: Clip) -> Optional[float]:
        asset = assets.get(clip.asset_id)
        if asset is None:
            return None
        start, end = float(clip.source_range.start), float(clip.source_range.end)
        return speech_loudness(asset.path, start, end - start)

    return voice_keys(project, _analyses(project), loudness)

def _analyses(project: Project) -> Dict[str, MediaAnalysis]:
    """Read the analysis of every file a project plays.

    Args:
        project: The project.

    Returns:
        The analyses that exist, keyed by asset ID.
    """
    found: Dict[str, MediaAnalysis] = {}
    for asset_id in {clip.asset_id for track in project.tracks for clip in track.clips}:
        analysis = repo.get_analysis(asset_id)
        if analysis is not None:
            found[asset_id] = analysis
    return found

def _plan_target(project: Project) -> Optional[float]:
    """Find how long the plan a project was compiled from asked it to run.

    Args:
        project: The project.

    Returns:
        The plan's target length, or None when the sequence did not all come
        from one plan or that plan named no length.
    """
    base = project.base_video_track
    compiled_from = {clip.from_plan_id for clip in base.clips} if base else set()
    if len(compiled_from) != 1 or None in compiled_from:
        return None
    plan = repo.get_plan(next(iter(compiled_from)))
    return plan.target.seconds if plan is not None else None

def _findings(project: Project, burn_subtitles: bool) -> list:
    """Check a cut before it is rendered.

    Args:
        project: The project, at the size it will be rendered at.
        burn_subtitles: Whether its captions will be burned in.

    Returns:
        What `delivery.check_delivery` finds.
    """
    placed = place_cues(project, project.subtitles) if burn_subtitles and project.subtitles else None
    dated = _dated(_referenced_assets(project))
    return check_delivery(
        project, _analyses(project), _plan_target(project), placed, project.caption_style,
        _talking(project, _voices(project)), {asset_id: asset.recorded_at for asset_id, asset in dated.items()},
    )

@mcp.tool()
def check_render(
    project_id: str,
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    burn_subtitles: bool = False,
) -> dict:
    """Check a cut for what would be noticed in the finished file, without rendering it.

    The same check `render_project` runs first and refuses a render over. It
    looks for nine things: `bad_picture`, black or frozen source picture that
    reaches the screen; `clipping`, a recording squared off at the ceiling,
    which no amount of turning down undoes; `music`, music heard too close
    under somebody talking; `mid_speech`, a cut that lands
    inside a word or between two words of one phrase, or takes the start or
    the end off a sound the microphone measured, with where to move it; `repeated`, the same stretch of a file shown twice;
    `continuity`, a clip put on by hand between two shot at another
    moment; `unplanned`, a sequence of three or more clips put together by hand
    rather than compiled from a plan, so nothing says how it opens, turns
    and ends; `captions`, a caption too tall for
    the frame — most likely when a landscape cut is rendered portrait — or on
    screen too briefly to read, with no room left in its shot to hold it; and
    `length`, a cut far from the length its plan asked for. Each is a fact
    about the cut, not a verdict on it: black may be meant, and the user may
    prefer the longer cut. Tell them, and let them decide.

    Args:
        project_id: ID of the project to check.
        frame: Check it in another shape, as `render_project` would render it.
        burn_subtitles: Check the captions as they would be burned in.

    Returns:
        A dictionary with `ok` and `findings`, each with its `check` and a
        `message` saying what and where.

    Raises:
        ValueError: If the project does not exist.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    findings = _findings(_reshaped(project, frame), burn_subtitles)
    return {
        "ok": not findings,
        "findings": [{"check": finding.check, "message": finding.message} for finding in findings],
    }

@mcp.tool()
def get_chapters(project_id: str) -> dict:
    """Read the parts of a cut back as chapters, ready for a video description.

    The parts are the project's markers — compiling a plan writes one per
    beat — so a chapter is a part and runs to the next. Rendering carries the
    same chapters inside the file. YouTube only shows chapters that start at
    0:00, number at least three and run at least ten seconds each; anything
    that breaks those is reported rather than changed, because merging or
    renaming parts changes the video.

    Args:
        project_id: ID of the project.

    Returns:
        A dictionary with `chapters` (`start`, `end`, `name`), the
        `description` text to paste — one `m:ss name` line each — and the
        `problems` that would stop YouTube showing them.

    Raises:
        ValueError: If the project does not exist.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    listed, problems = chapters(project)
    return {
        "chapters": [{"start": round(start, 3), "end": round(end, 3), "name": name} for start, end, name in listed],
        "description": "\n".join(f"{clock(start)} {name}" for start, _, name in listed),
        "problems": problems,
    }

@mcp.tool()
def propose_covers(
    project_id: str,
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    follow_faces: bool = True,
) -> ToolResult:
    """Offer frames of the cut as cover or thumbnail candidates, one per shot.

    Which frame makes the cover is a judgement, so this only narrows the
    field: a frame per shot, never one of black or frozen picture, each the
    moment its shot's measurements favour — the largest face, or where nobody
    is seen, the sharpest stretch. Look at the sheet, pick with the user, and
    save the one they choose with `export_cover`.

    Args:
        project_id: ID of the project.
        frame: Crop the candidates to another shape, as it would be rendered.
        follow_faces: Crop around the face, as the render does.

    Returns:
        A listing — each candidate's number, time in the cut and clip — and
        the candidates as one labeled image.

    Raises:
        ValueError: If the project does not exist or has nothing to offer.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    project = _reshaped(project, frame)
    offered = cover_candidates(project, _analyses(project))[:MAX_STORYBOARD_TILES]
    if not offered:
        raise ValueError(f"project {project_id} has no picture to take a cover from")
    assets = _referenced_assets(project)
    framing = _framing(project, assets, follow_faces)
    fps = Fraction(project.fps_num, project.fps_den)
    shots, listing = [], []
    for number, (clip, seconds) in enumerate(offered, start=1):
        at = float(clip.timeline_in) + (seconds - float(clip.source_range.start)) / clip.speed
        framed = framing.get(clip.id)
        shots.append((assets[clip.asset_id].path, seconds, f"#{number}  {format_timestamp(at)}",
                      placement_at(framed, clip, seconds, fps)))
        listing.append(f"#{number}: edit {at:.3f}s ({format_timestamp(at)}) | clip {clip.id}")
    header = f"{len(offered)} cover candidate(s) for project {project_id}, {project.width}x{project.height}:"
    return ToolResult(content=[
        "\n".join([header, *listing]),
        Image(data=storyboard_sheet(shots, aspect=project.width / project.height), format="jpeg"),
    ])

@mcp.tool()
def export_cover(
    project_id: str,
    seconds: Annotated[float, Field(ge=0)],
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    follow_faces: bool = True,
) -> dict:
    """Save one frame of the cut as a full-size JPEG, for a cover or thumbnail.

    The frame is the one the viewer sees at that moment — covering picture
    where there is some — at the size the cut is rendered at and cropped the
    way the render crops it.

    Args:
        project_id: ID of the project.
        seconds: The moment in the cut, as `propose_covers` lists it.
        frame: Save it in another shape.
        follow_faces: Crop around the face, as the render does.

    Returns:
        A dictionary with the `output_path` of the JPEG and its `width` and
        `height`.

    Raises:
        ValueError: If the project does not exist or nothing is on screen then.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    project = _reshaped(project, frame)
    on_screen = [
        clip for track in project.video_tracks for clip in track.clips
        if (track is project.base_video_track or clip.layout is None)
        and float(clip.timeline_in) <= seconds < float(clip.timeline_out)
    ]
    if not on_screen:
        raise ValueError(f"nothing is on screen at {seconds:g}s of project {project_id}")
    # The topmost full-frame picture is what the viewer sees.
    clip = on_screen[-1]
    assets = _referenced_assets(project)
    source = float(clip.source_range.start) + (seconds - float(clip.timeline_in)) * clip.speed
    framed = _framing(project, assets, follow_faces).get(clip.id)
    fps = Fraction(project.fps_num, project.fps_den)
    picture = still(assets[clip.asset_id].path, source, project.width, project.height,
                    placement_at(framed, clip, source, fps))
    folder = str(output_dir())
    os.makedirs(folder, exist_ok=True)
    path = housekeeping.unique_path(folder, _delivery_name(project, "cover" + (f"-{frame}" if frame else ""), ".jpg"))
    picture.save(path, "JPEG", quality=92)
    return {"output_path": path, "width": project.width, "height": project.height}

def _timecodes(assets: Mapping[str, Asset]) -> Dict[str, float]:
    """Read the timecode each file starts at, for an editing program to line it up by.

    Read at export rather than kept on the asset: it is only wanted here, and a
    file that cannot be read now simply starts at zero.

    Args:
        assets: The files, keyed by asset ID.

    Returns:
        The second each file's timecode starts at, for the files that carry one.
    """
    found: Dict[str, float] = {}
    for asset_id, asset in assets.items():
        try:
            start = timecode_start(probe_file(asset.path))
        except (FileNotFoundError, RuntimeError):
            start = None
        if start:
            found[asset_id] = start
    return found

@mcp.tool()
def export_timeline(
    project_id: str,
    format: Literal["edl", "otio", "fcpxml", "srt"],
) -> dict:
    """Write the cut out for a professional editing program, pointed at the original files.

    For a user who wants to finish the edit in Premiere, DaVinci Resolve or
    Final Cut rather than start again from the rendered file. `fcpxml` is
    the fullest and opens in all three; `otio` is OpenTimelineIO, for Resolve
    and anything built on it; `edl` is the oldest and plainest — the sequence
    alone, one picture track with its sound. `srt` writes the stored captions
    as they fall in the cut, which every one of those programs imports.

    The edit comes across — every clip's in and out, its place and track,
    speed changes, and the parts of the video as markers, on each file's own
    timecode where it has one. `otio` also carries transitions and `fcpxml`
    each clip's level. What stays behind is what this server draws itself:
    colour, fades, voice repair, where an inset sits. `left_behind` says which of those this cut
    uses; tell the user, so they are not surprised in the other program.

    Args:
        project_id: ID of the project.
        format: `fcpxml`, `otio`, `edl`, or `srt`.

    Returns:
        A dictionary with the `output_path` of the file and `left_behind`,
        one line per kind of thing the export could not carry.

    Raises:
        ValueError: If the project does not exist, has nothing on its
            sequence, or has no captions to write as SRT.
    """
    project = repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if format == "srt":
        placed = place_cues(project, project.subtitles) if project.subtitles else []
        if not placed:
            raise ValueError(
                f"project {project_id} has no captions in the cut; propose them with generate_subtitles "
                "and store them with a set_subtitles operation"
            )
        text, behind = write_srt(placed), []
    else:
        if project.base_video_track is None or not project.base_video_track.clips:
            raise ValueError(f"project {project_id} has nothing on its sequence to export")
        writer = {"edl": write_edl, "otio": write_otio, "fcpxml": write_fcpxml}[format]
        assets = _referenced_assets(project)
        text, behind = writer(project, assets, _timecodes(assets))
        # Decided at render time from the recordings, so the writers, which never see an
        # analysis, cannot say it. Only which tracks, not how loud: nothing is measured.
        if voice_keys(project, _analyses(project), lambda clip: None):
            behind.append("the footage's sound ducking under a narration, and the narration brought up to level")
    folder = str(output_dir())
    os.makedirs(folder, exist_ok=True)
    path = housekeeping.unique_path(folder, _delivery_name(project, "timeline", f".{format}"))
    # UTF-8 with a byte-order mark only for the EDL: the programs that read EDLs guess the
    # encoding, and a Chinese file name in a FROM CLIP NAME comes out garbled without it.
    with open(path, "w", encoding="utf-8-sig" if format == "edl" else "utf-8", newline="\n") as handle:
        handle.write(text)
    return {"output_path": path, "left_behind": behind}

# Whether edits start making what the browser needs to play them. Off where nobody will
# watch, such as a test run.
PREPARE_PLAYBACK = True
# What a preparation job is expected to hold in memory: one FFmpeg encoding a small picture.
# Provisional: an estimate, not measured.
PREPARE_MEMORY_BYTES = 300 * 1024 * 1024
# Giving cache room back looks at the disk at most this often, in seconds. Provisional.
ROOM_CHECK_SECONDS = 60

def _being_prepared() -> set:
    """The files preparation jobs still waiting or running will write, so none is asked for twice."""
    paths = set()
    for job in repo.jobs_to_show(datetime.now(timezone.utc))[0]:
        if job.kind != JobKind.PREPARE or not job.work_dir:
            continue
        try:
            with open(os.path.join(job.work_dir, "job.json"), encoding="utf-8") as spec:
                paths.update(step["path"] for step in json.load(spec).get("steps", []))
        except (OSError, ValueError):
            continue
    return paths

def _frame_rate(asset: Asset) -> float:
    """A file's frame rate as its first picture stream says, or 0 when it does not."""
    try:
        streams = probe_file(asset.path).get("streams", [])
    except (OSError, RuntimeError):
        return 0.0
    for stream in streams:
        if stream.get("codec_type") == "video":
            try:
                return float(Fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0"))
            except (ValueError, ZeroDivisionError):
                return 0.0
    return 0.0

def _start_preparing(steps: List[dict], urgent: bool, asset_id: Optional[str] = None,
                     project_id: Optional[str] = None) -> Optional[str]:
    """Queue one preparation job over its steps; returns its ID, or None when there was nothing to do."""
    if not steps:
        return None
    job = Job(kind=JobKind.PREPARE, asset_id=asset_id, project_id=project_id, priority=-1 if urgent else 1,
              memory_estimate=PREPARE_MEMORY_BYTES)
    job.work_dir = os.path.join(WORKSPACE_DIR, "jobs", job.job_id)
    return job_manager.start_job(job, {"steps": steps}).job_id

def prepare_playback(project: Project, urgent: bool = False) -> List[str]:
    """Start making what the browser still needs to play a cut: proxies, repaired sound, its loudness.

    Each is made once and kept by what went into it, so this costs nothing
    for what is already there or already being made.

    Args:
        project: The cut.
        urgent: Somebody is waiting to watch it, so it goes ahead of the
            other work waiting; otherwise it waits behind it.

    Returns:
        The IDs of the jobs started.
    """
    if not PREPARE_PLAYBACK:
        return []
    _make_room()
    assets = _referenced_assets(project)
    busy = _being_prepared()
    proxies = playback.proxy_dir(WORKSPACE_DIR)
    started = []
    for asset_id in playback.missing(project, assets, os.listdir(proxies)):
        asset = assets[asset_id]
        path = os.path.join(proxies, playback.proxy_name(asset))
        if path in busy:
            continue
        step = {"command": playback.proxy_command(asset, path + ".part", _frame_rate(asset)), "part": path + ".part",
                "path": path, "seconds": float(asset.duration or 1), "stage": "making a copy to play from"}
        started.append(_start_preparing([step], urgent, asset_id=asset_id))
    sound = playback.sound_dir(WORKSPACE_DIR)
    steps = []
    for track in project.tracks:
        for clip in track.clips:
            asset = assets.get(clip.asset_id)
            name = playback.repair_name(clip, asset) if asset is not None and playback.needs_repair(clip) else None
            path = os.path.join(sound, name) if name else None
            if path and not os.path.exists(path) and path not in busy:
                steps.append({"command": playback.repair_command(clip, asset, path + ".part"), "part": path + ".part",
                              "path": path, "seconds": float(clip.source_range.end - clip.source_range.start),
                              "stage": "repairing a voice"})
    gain = os.path.join(sound, playback.mix_key(project) + ".json")
    if project.tracks and float(project.duration) > 0 and not os.path.exists(gain) and gain not in busy:
        voices = _voices(project)
        mix = gain + ".wav"
        steps.append({"command": renderer.build_mix(project, assets, mix, voices, _talking(project, voices)),
                      "part": mix, "path": gain, "seconds": float(project.duration), "measure": True,
                      "target": playback.LOUDNESS_TARGET, "stage": "measuring loudness"})
    started.append(_start_preparing(steps, urgent, project_id=project.id))
    return [job_id for job_id in started if job_id]

def playback_of(project: Project, urgent: bool = True) -> dict:
    """Describe a cut for the browser to play, and start making whatever it still lacks.

    Args:
        project: The cut, as it is now or as it was at a version.
        urgent: Whether somebody is waiting to watch it.

    Returns:
        What `playback.describe` returns.
    """
    assets = _referenced_assets(project)
    proxies = playback.proxy_dir(WORKSPACE_DIR)
    sound = playback.sound_dir(WORKSPACE_DIR)
    have = {}
    for asset_id, asset in assets.items():
        name = playback.proxy_name(asset)
        if name and os.path.exists(os.path.join(proxies, name)):
            have[asset_id] = f"/proxy/{name}"
    repaired = {}
    for track in project.tracks:
        for clip in track.clips:
            asset = assets.get(clip.asset_id)
            if asset is None or not playback.needs_repair(clip):
                continue
            name = playback.repair_name(clip, asset)
            if name and os.path.exists(os.path.join(sound, name)):
                repaired[clip.id] = f"/sound/{name}"
    gain = None
    measured = os.path.join(sound, playback.mix_key(project) + ".json")
    if os.path.exists(measured):
        with open(measured, encoding="utf-8") as kept:
            gain = json.load(kept).get("gain_db")
    voices = _voices(project)
    described = playback.describe(
        project, assets, playback.Prepared(have, repaired, gain), _talking(project, voices), voices,
        _framing(project, assets),
    )
    if described["waiting"] or gain is None:
        prepare_playback(project, urgent=urgent)
    return described

# What the user can mark a version as. `exported` is a third mark, never set by hand: a
# version that was rendered out, not as a preview, wears it by itself.
VERSION_MARKS = ("starred", "published")

def _say_in_history(words: str) -> None:
    """Give the version being made these words, unless the user's own were passed on."""
    batch = current_batch()
    if batch is not None and not batch.note and words:
        batch.note = words

def _project_json_at(project_id: str, commit: str) -> Optional[dict]:
    """A project as its file held it at one version, or None when it was not there then."""
    path = history.find("projects/", f"{project_id}.json", commit)
    content = history.read(path, commit) if path else None
    return json.loads(content) if content else None

def _branches_of(project_id: str) -> List[Project]:
    """The projects started as branches of this one."""
    return [other for other in repo.list_projects()
            if other.branched_from is not None and other.branched_from.project_id == project_id]

def _version_record(version: Version, marks: Iterable[str] = ()) -> dict:
    """One version of a project as the AI reads it."""
    return {
        "commit": version.commit[:10],
        "when": version.when.astimezone().strftime("%Y-%m-%d %H:%M"),
        "by": version.author,
        "what": version.subject,
        "marks": sorted(marks),
    }

def _project_versions(project_id: str, limit: int) -> List[Tuple[Version, Optional[dict], set]]:
    """A project's versions, newest first, each with the project as it was then and its marks.

    Returns:
        `(version, project, marks)`, the project None where the version took
        it away. A version whose finished video is still there is `exported`;
        one whose finished video has gone since, `removed`.
    """
    marks = repo.version_marks(project_id)
    delivered = _delivered_renders(project_id)
    there = {job.project_version for job in delivered if os.path.exists(job.output_path)}
    gone = {job.project_version for job in delivered} - there
    found = []
    for version in history.versions_of("projects/", f"{project_id}.json", limit):
        data = _project_json_at(project_id, version.commit)
        marked = set(marks.get(version.commit, ()))
        if data is not None and data.get("version") in there:
            marked.add("exported")
        elif data is not None and data.get("version") in gone:
            marked.add("removed")
        found.append((version, data, marked))
    return found

def _delivered_renders(project_id: str) -> List[Job]:
    """Every full render of a project that finished, whether its file is still there or not."""
    previews = os.path.abspath(housekeeping.preview_dir(WORKSPACE_DIR, project_id))
    return [job for job in repo.renders_of(project_id)
            if job.output_path and not os.path.abspath(job.output_path).startswith(previews)]

def _outputs_of(project_id: str) -> List[Job]:
    """The finished videos of a project that are still where they were saved; previews are not among them."""
    return [job for job in _delivered_renders(project_id) if os.path.exists(job.output_path)]

def _project_caches(project_id: str) -> List[str]:
    """The folders of a project's previews and sound checks: made again when wanted."""
    return [housekeeping.preview_dir(WORKSPACE_DIR, project_id), housekeeping.sound_dir(WORKSPACE_DIR, project_id)]

def project_storage(project_id: str) -> dict:
    """What a video takes on disk, and what deleting its other versions would give back.

    Its other versions are the finished videos of every version but the one
    it is now and those the user starred or marked published; with them go
    its previews. The versions themselves stay in the history, marked as no
    longer exported.

    Args:
        project_id: The video.

    Returns:
        `versions`, `branches`, `exported` (versions with a finished video
        still there), `outputs_megabytes`, `cache_megabytes`,
        `freeable_megabytes`, and `other_outputs`, the files that would go.

    Raises:
        ValueError: If there is no such project.
    """
    project = repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    # Only the marked versions are read: the storage page asks this of every video at once.
    kept = {project.version}
    for commit, marks in repo.version_marks(project_id).items():
        if marks & set(VERSION_MARKS):
            data = _project_json_at(project_id, commit)
            if data:
                kept.add(data.get("version"))
    outputs = _outputs_of(project_id)
    others = [job for job in outputs if job.project_version not in kept]
    size = {job.job_id: housekeeping.size_of(job.output_path) for job in outputs}
    cache = _cache_bytes(project_id)
    return {
        "versions": history.counts("projects/").get(project_id, 0), "branches": len(_branches_of(project_id)),
        "exported": len({job.project_version for job in outputs}),
        "outputs_megabytes": round(sum(size.values()) / 1e6, 1),
        "cache_megabytes": round(cache / 1e6, 1),
        "freeable_megabytes": round((sum(size[job.job_id] for job in others) + cache) / 1e6, 1),
        "other_outputs": [{"path": job.output_path, "megabytes": round(size[job.job_id] / 1e6, 1)} for job in others],
    }

def _cache_bytes(project_id: str) -> int:
    """What a project's previews and sound checks take."""
    return sum(housekeeping.size_of(folder) for folder in _project_caches(project_id))

def disk_of(project_id: str) -> float:
    """What a video takes on disk in megabytes, its finished videos and previews together, for its card."""
    outputs = sum(housekeeping.size_of(job.output_path) for job in _outputs_of(project_id))
    return round((outputs + _cache_bytes(project_id)) / 1e6, 1)

def trim_project(project_id: str, typed_name: str) -> dict:
    """Delete a video's other versions: their finished videos go to the trash, its previews go.

    Only the editor calls this, once the user has typed the video's name: the
    AI has no tool that deletes anything.

    Args:
        project_id: The video.
        typed_name: The name the user typed to confirm.

    Returns:
        `trashed` (finished videos put in the trash) and `freed_megabytes`.

    Raises:
        ValueError: If there is no such project, or the name typed is not its name.
    """
    project = repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    if not (project.name or "").strip():
        raise ValueError("name the video first: its name is what is typed to confirm")
    if typed_name.strip() != project.name.strip():
        raise ValueError("the name typed is not the video's name")
    found = project_storage(project_id)
    paths = [item["path"] for item in found["other_outputs"]]
    freed = 0
    if paths:
        trashed = housekeeping.put_in_trash(str(output_dir()), "outputs", project_id, project.name or "", {}, paths)
        freed += trashed.bytes
    busy = repo.active_job_ids()
    for folder in _project_caches(project_id):
        if not os.path.isdir(folder):
            continue
        for entry in os.scandir(folder):
            if entry.name in busy:
                continue
            size = housekeeping.size_of(entry.path)
            shutil.rmtree(entry.path, ignore_errors=True)
            freed += size
    return {"trashed": len(paths), "freed_megabytes": round(freed / 1e6, 1)}

def trash_projects(project_ids: List[str]) -> dict:
    """Move projects to the trash, unless one of them is being rendered.

    Their finished videos stay where they were saved; their previews go,
    since they are made again. A project waits in the trash for
    `TRASH_KEEP_DAYS` days and can be put back until then. Each is in the
    trash before it leaves the database, so nothing is lost on the way.

    Args:
        project_ids: The projects.

    Returns:
        `deleted`, the IDs moved, and `busy`, the names of those being
        rendered; when `busy` is not empty nothing was moved.
    """
    job_manager.forget_abandoned()
    before = {project_id: project for project_id in project_ids if (project := repo.get_project(project_id))}
    filed = repo.folder_of("projects")
    named = [project.name for project in before.values() if project.name]
    _say_in_history(f"把「{'」「'.join(named[:3])}」移到垃圾桶" + (f"等 {len(named)} 支" if len(named) > 3 else ""))
    outputs = str(output_dir())
    trashed = {
        project_id: housekeeping.put_in_trash(outputs, "project", project_id, project.name or "", {
            "project.json": project.model_dump_json().encode("utf-8"),
            "place.json": json.dumps({"folder_id": filed.get(project_id)}).encode("utf-8"),
        }, [])
        for project_id, project in before.items()
    }
    deleted, busy = repo.delete_projects(list(before))
    for project_id, kept in trashed.items():
        if project_id not in deleted:
            housekeeping.let_go(outputs, kept.key)
    for project_id in deleted:
        for folder in _project_caches(project_id):
            shutil.rmtree(folder, ignore_errors=True)
    housekeeping.empty_old_trash(outputs)
    return {"deleted": deleted, "busy": busy}

def list_trash(expire: bool = False) -> dict:
    """What is in the trash.

    Args:
        expire: First remove for good what has waited longer than the trash
            keeps things. Only the editor asks for that: the AI reads the
            trash and never deletes anything.

    Returns:
        `items`, newest first, each with its `key`, `kind` (`project` or
        `outputs`), `name`, `days_left`, `megabytes` and `files`; and
        `keep_days`.
    """
    outputs = str(output_dir())
    if expire:
        housekeeping.empty_old_trash(outputs)
    now = datetime.now(timezone.utc)
    return {
        "items": [{"key": item.key, "kind": item.kind, "name": item.name, "project_id": item.project_id,
                   "days_left": item.days_left(now), "megabytes": round(item.bytes / 1e6, 1),
                   "files": len(item.files)} for item in housekeeping.in_trash(outputs)],
        "keep_days": housekeeping.TRASH_KEEP_DAYS,
    }

def restore_from_trash(key: str) -> dict:
    """Put something back from the trash: a project as it was, or finished videos where they were.

    The trash lets go of it only once the project is back in the database.

    Args:
        key: Its `key` from `list_trash`.

    Returns:
        Its `kind` and `name`, the `project_id` it is back as, and the `files`
        moved back.

    Raises:
        ValueError: If there is no such thing in the trash.
    """
    outputs = str(output_dir())
    try:
        what, kept = housekeeping.look_in_trash(outputs, key)
    except KeyError as exc:
        raise ValueError(f"{key} is not in the trash") from exc
    project_id = what.project_id
    if what.kind == "project" and "project.json" in kept:
        _say_in_history(f"從垃圾桶放回「{what.name}」")
        project = Project.model_validate_json(kept["project.json"])
        if repo.get_project(project.id) is not None:
            project = project.model_copy(update={"id": str(uuid.uuid4())})
        repo.add_project(project)
        project_id = project.id
        folder = json.loads(kept.get("place.json", b"{}")).get("folder_id")
        if folder and any(item["id"] == folder for item in repo.list_folders("projects")):
            repo.file_items("projects", [project.id], folder)
    back = housekeeping.take_out_of_trash(outputs, key)
    return {"kind": what.kind, "name": what.name, "project_id": project_id, "files": back}

_room_checked = 0.0

def _make_room() -> None:
    """Give cache room back if the disk is running low, at most once a minute."""
    global _room_checked
    if time.monotonic() - _room_checked < ROOM_CHECK_SECONDS:
        return
    _room_checked = time.monotonic()
    housekeeping.give_cache_back(WORKSPACE_DIR, repo.active_job_ids())

def _base_clips(data: Optional[dict]) -> List[dict]:
    """A project's sequence, as its file holds it, in the order it plays."""
    base = next((track for track in (data or {}).get("tracks", []) if track.get("track_type") == "video"), None)
    return sorted(base.get("clips", []) if base else [], key=lambda clip: float(clip["timeline_in"]))

def _thumb_of(data: Optional[dict]) -> Optional[dict]:
    """Which frame stands for a version: its first shot, a second in, as the projects page shows it."""
    clips = _base_clips(data)
    if not clips:
        return None
    first = clips[0]
    start, end = float(first["source_range"]["start"]), float(first["source_range"]["end"])
    return {"asset_id": first["asset_id"], "t": round(min(start + 1.0, (start + end) / 2), 2)}

def _length_of(data: Optional[dict]) -> float:
    """How long a version runs: where its sequence ends."""
    return round(max((float(clip["timeline_in"])
                      + (float(clip["source_range"]["end"]) - float(clip["source_range"]["start"]))
                      / float(clip.get("speed") or 1) for clip in _base_clips(data)), default=0.0), 2)

def _version_nodes(entries: List[Tuple[Version, Optional[dict], set]], apart: set,
                   outputs: Mapping[int, List[Job]]) -> List[dict]:
    """Group versions into what the version panel shows as one dot each.

    A run of changes by one author with nothing rendered out between them is
    one dot: twenty nudges in the editor are one sitting, not twenty
    versions. A render ends a sitting, so a rendered version heads its own
    dot; a version the user starred or marked published, or one a branch
    started from, stands alone.

    Args:
        entries: From `_project_versions`, newest first.
        apart: Commits that must not be folded into another's dot.
        outputs: The finished videos still there, by the project version
            they were rendered from.

    Returns:
        The dots, newest first, each named by its newest version.
    """
    alone = set(VERSION_MARKS)
    nodes: List[dict] = []
    for version, data, marked in entries:
        record = {"commit": version.commit, "what": version.subject,
                  "when": version.when.astimezone().strftime("%m/%d %H:%M")}
        last = nodes[-1] if nodes else None
        if (last is not None and last["by"] == version.author and not marked and not alone & set(last["marks"])
                and version.commit not in apart and last["versions"][-1]["commit"] not in apart):
            last["versions"].append(record)
            continue
        number = data.get("version") if data else None
        nodes.append({
            "commit": version.commit, "when": record["when"], "by": version.author, "what": version.subject,
            "body": version.body, "marks": sorted(marked), "gone": data is None, "project_version": number,
            "thumb": _thumb_of(data), "seconds": _length_of(data),
            "outputs": [{"job_id": job.job_id, "name": os.path.basename(job.output_path)}
                        for job in outputs.get(number, [])],
            "versions": [record],
        })
    return nodes

def _nodes_of(project_id: str, limit: int, apart: set) -> List[dict]:
    """A project's dots for the version panel."""
    outputs: Dict[int, List[Job]] = {}
    for job in _outputs_of(project_id):
        outputs.setdefault(job.project_version, []).append(job)
    return _version_nodes(_project_versions(project_id, limit), apart, outputs)

def version_graph(project_id: str, limit: int = 200) -> dict:
    """Everything the editor's version panel draws for one video.

    Args:
        project_id: The video.
        limit: Most versions to read.

    Returns:
        The `project`; its `nodes`, newest first; the `branches` started from
        it, each with the commit it started `from` and its own nodes; and
        `branched_from` when it is itself a branch.

    Raises:
        ValueError: If there is no such project.
    """
    project = repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    branches = _branches_of(project_id)
    starts = {branch.branched_from.commit for branch in branches}
    parent = None
    if project.branched_from is not None:
        source = repo.get_project(project.branched_from.project_id)
        start = next((version for version in history.versions_of(
            "projects/", f"{project.branched_from.project_id}.json", 500)
            if version.commit == project.branched_from.commit), None)
        parent = {"project_id": project.branched_from.project_id, "commit": project.branched_from.commit,
                  "name": source.name if source else None, "what": start.subject if start else None}
    return {
        "project": {"id": project.id, "name": project.name, "version": project.version},
        "nodes": _nodes_of(project_id, limit, starts),
        "branches": [
            {"project_id": branch.id, "name": branch.name, "from": branch.branched_from.commit,
             "nodes": _nodes_of(branch.id, 30, set())}
            for branch in sorted(branches, key=lambda item: item.name or "")
        ],
        "branched_from": parent,
    }

@mcp.tool()
def project_history(project_id: str, limit: Annotated[int, Field(ge=1, le=200)] = 30) -> dict:
    """List a video's versions: every change to it, newest first, with who made it and what it was.

    Every change to a project — a compile, an edit, captions — is kept as a
    version, and nothing is ever taken out of the history: going back is one
    more version on top. Use it for 「剛剛那樣比較好」, 「回到昨天那版」 or
    「誰改了結尾」, then `restore_version` or `branch_project`. A client with a
    shell can also read the same history with git in the workspace's
    `history` folder.

    Args:
        project_id: The video.
        limit: Most versions to list.

    Returns:
        `versions`, each with its `commit`, `when`, `by` (the AI program or
        the editor), `what` changed in words, and its `marks`: `starred`
        (the user's favourite), `published` (the one that went out) and
        `exported` (rendered out); and `branched_from`, the project and
        version a branch started as, or null.

    Raises:
        ValueError: If there is no such project.
    """
    project = repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    return {
        "versions": [_version_record(version, marks) for version, _, marks in _project_versions(project_id, limit)],
        "branched_from": project.branched_from.model_dump() if project.branched_from else None,
    }

def _project_at(project_id: str, commit: str) -> Project:
    """A project as it was at one version.

    Raises:
        ValueError: If there is no such version, or the project was not there then.
    """
    try:
        path = history.find("projects/", f"{project_id}.json", commit)
    except KeyError as exc:
        raise ValueError(str(exc).strip("'\"")) from exc
    content = history.read(path, commit) if path else None
    if content is None:
        raise ValueError(f"project {project_id} did not exist at version {commit}")
    return Project.model_validate_json(content)

def _plans_at(project: Project, commit: str) -> Dict[str, EditPlan]:
    """The plans a project's clips were compiled from, as they were at one version."""
    found: Dict[str, EditPlan] = {}
    for plan_id in {clip.from_plan_id for track in project.tracks for clip in track.clips if clip.from_plan_id}:
        content = history.read(f"plans/{plan_id}.json", commit)
        if content is not None:
            found[plan_id] = EditPlan.model_validate_json(content)
    return found

def _with_plans(project: Project, renamed: Mapping[str, str]) -> Project:
    """A project whose compiled clips name their plans by new IDs."""
    copy = project.model_copy(deep=True)
    for track in copy.tracks:
        for clip in track.clips:
            if clip.from_plan_id in renamed:
                clip.from_plan_id = renamed[clip.from_plan_id]
    return copy

def _copied_plan(plan: EditPlan, why: str) -> str:
    """Save a plan under a new ID, as the starting point of its own history; returns the ID."""
    stored = repo.save_plan(plan.model_copy(update={"id": str(uuid.uuid4()), "version": 1}), why)
    return stored.id

@mcp.tool()
def restore_version(project_id: str, commit: str, expected_version: int, note: str = "") -> dict:
    """Make a video what it was at one of its versions, as a new version on top.

    Nothing after that version is lost: it all stays in the history, and
    going back can itself be undone the same way. The plan the video was
    compiled from goes back with it, so the next compile does not undo this
    — and when another video was compiled from that same plan, this one gets
    a copy of it instead, so the other is left exactly as it is.

    Args:
        project_id: The video.
        commit: The version to go back to, from `project_history`.
        expected_version: The project's current `version`.
        note: The user's own words for why, kept in the history.

    Returns:
        The project's `new_version`, the `restored` version's `commit` and
        `what`, and `plan_copied` — true when the plan was shared and this
        video now has a copy of its own, which is worth telling the user.

    Raises:
        ValueError: If the project or version does not exist, or the project
            has moved on from `expected_version`.
    """
    current = repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    if current.version != expected_version:
        raise ValueError(f"version conflict: project {project_id} is at version {current.version}, not {expected_version}")
    earlier = _project_at(project_id, commit)
    chosen = next((version for version in history.versions_of("projects/", f"{project_id}.json", 500)
                   if version.commit.startswith(commit.strip().lower())), None)
    if chosen is not None:
        _say_in_history(f"回到「{chosen.subject}」那一版")
    renamed: Dict[str, str] = {}
    copied = False
    others = [other for other in repo.list_projects() if other.id != project_id]
    for plan_id, plan in _plans_at(earlier, commit).items():
        shared = any(clip.from_plan_id == plan_id for other in others for track in other.tracks for clip in track.clips)
        if shared:
            renamed[plan_id] = _copied_plan(plan, f"「{current.name or project_id}」回到舊版時另存的計畫")
            copied = True
            continue
        now = repo.get_plan(plan_id)
        if now is None:
            repo.save_plan(plan.model_copy(update={"version": 1}), "回到舊版時放回的計畫")
        elif now.model_dump(exclude={"version", "updated_at"}) != plan.model_dump(exclude={"version", "updated_at"}):
            repo.save_plan(plan.model_copy(update={"version": now.version}), "跟著影片回到舊版")
    restored = _with_plans(earlier, renamed).model_copy(update={
        "id": project_id, "version": current.version + 1, "branched_from": current.branched_from,
    })
    repo.update_project(restored, current.version)
    prepare_playback(repo.get_project(project_id))
    return {
        "new_version": restored.version,
        "restored": {"commit": chosen.commit[:10], "what": chosen.subject} if chosen else {"commit": commit},
        "plan_copied": copied,
    }

@mcp.tool()
def branch_project(project_id: str, name: str, commit: Optional[str] = None, note: str = "") -> dict:
    """Start another way of cutting a video from one of its versions, beside it rather than over it.

    For 「做一個 Reels 60 秒版」 or 「試試看沒有配樂的版本」: the branch is a project
    of its own, with its own copy of the plan, so it can be cut, captioned and
    rendered while the original is left alone. The editor draws it as a line
    running beside the original's from the version it started at.

    Args:
        project_id: The video to branch from.
        name: What to call the branch, in words: 「抽杯架 Reels 60 秒版」.
        commit: The version to start from, from `project_history`; omit for
            the video as it is now.
        note: The user's own words for why, kept in the history.

    Returns:
        The new `project_id`, its `name`, and `branched_from`.

    Raises:
        ValueError: If the project or version does not exist.
    """
    current = repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    if commit is None:
        versions = history.versions_of("projects/", f"{project_id}.json", 1)
        if not versions:
            raise ValueError(f"project {project_id} has no version in the history yet")
        commit = versions[0].commit
    start = _project_at(project_id, commit)
    found = next((version.commit for version in history.versions_of("projects/", f"{project_id}.json", 500)
                  if version.commit.startswith(commit.strip().lower())), commit)
    renamed = {plan_id: _copied_plan(plan, f"「{name}」分支的計畫")
               for plan_id, plan in _plans_at(start, commit).items()}
    branch = _with_plans(start, renamed).model_copy(update={
        "id": str(uuid.uuid4()), "name": name.strip(), "version": 1,
        "branched_from": BranchPoint(project_id=project_id, commit=found),
    })
    _say_in_history(f"從「{current.name or project_id}」開分支「{branch.name}」")
    repo.add_project(branch)
    prepare_playback(branch)
    folder = repo.folder_of("projects").get(project_id)
    if folder:
        repo.file_items("projects", [branch.id], folder)
    return {"project_id": branch.id, "name": branch.name, "branched_from": branch.branched_from.model_dump()}

# Provisional (roadmap §13): a word counts as said in a stretch only when this much of it is
# inside, so a word the cut just grazes does not name the stretch.
WORD_INSIDE_SECONDS = 0.05
# Provisional (roadmap §13): a comment dragged over less than this is a moment, not a stretch.
SHORTEST_STRETCH_SECONDS = 0.05

def _said_between(asset_id: str, start: float, end: float) -> str:
    """What the transcript has said in a file between two of its seconds; empty when nothing was."""
    analysis = repo.get_analysis(asset_id)
    if analysis is None or analysis.transcript is None:
        return ""
    return "".join(
        word.text for segment in analysis.transcript.segments if not was_made_up(segment) for word in segment.words
        if word.start < end - WORD_INSIDE_SECONDS and word.end > start + WORD_INSIDE_SECONDS
    ).strip()

def _version_at(project_id: str, commit: Optional[str]) -> Tuple[Project, dict]:
    """A project at one of its versions, or as it is now, with that version as the AI reads it.

    Args:
        project_id: The video.
        commit: The version, or a prefix of it; empty for the video as it is.

    Returns:
        The project as it was, and the version as `project_history` shows it.

    Raises:
        ValueError: If the project or the version does not exist.
    """
    current = repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    versions = history.versions_of("projects/", f"{project_id}.json", 500)
    if not commit:
        return current, (_version_record(versions[0]) if versions else {"commit": None, "what": "目前"})
    chosen = next((version for version in versions if version.commit.startswith(commit.strip().lower())), None)
    if chosen is None:
        raise ValueError(f"project {project_id} has no version {commit}")
    return _project_at(project_id, chosen.commit), _version_record(chosen)

def compare_projects(before: Project, after: Project) -> dict:
    """What changed from one cut to another, files named by name and stretches by their words.

    Args:
        before: The left cut.
        after: The right cut.

    Returns:
        What `compare.compare` returns.
    """
    names = {asset_id: os.path.basename(asset.path) for asset_id, asset in repo.get_assets(
        {clip.asset_id for project in (before, after) for track in project.tracks for clip in track.clips}).items()}
    return compare(before, after, _said_between, lambda asset_id: names.get(asset_id, asset_id))

@mcp.tool()
def compare_versions(project_id: str, commit_a: str, commit_b: Optional[str] = None) -> dict:
    """Say what changed between two versions of a video, part by part, in words the user reads.

    For 「第 3 版跟現在差在哪」 or before restoring or merging. Footage is matched
    by what it shows, not by clip IDs, so a recompile is not reported as
    everything changing. The editor's comparison view shows the same list.

    Args:
        project_id: The video.
        commit_a: The earlier version, from `project_history`.
        commit_b: The other one; omit for the video as it is now.

    Returns:
        `a` and `b` (each version's `commit`, `when`, `what`), `length` of
        each in seconds, `parts` — each part of the cut (its markers, one per
        beat) with `same` and `changes`, each a sentence with `a_at`/`b_at`,
        where it is in each version — and `whole`, changes to the video as a
        whole: songs, captions, caption style, frame.

    Raises:
        ValueError: If the project or a version does not exist.
    """
    before, a = _version_at(project_id, commit_a)
    after, b = _version_at(project_id, commit_b)
    compared = compare_projects(before, after)
    compared.pop("aligned")
    return {"a": a, "b": b, **compared}

def _merged_plan(before: Project, after: Project, commits: Tuple[Optional[str], Optional[str]], merged: Project,
                 from_left: List[str], name: str) -> Tuple[Project, str]:
    """Merge the plans the two sides were compiled from, beat by beat, when they can be.

    Each side's plan is read as it was at that side's version, so merging two
    old versions does not mix in the plan as it is today.

    Args:
        before: The left version.
        after: The right version.
        commits: The two versions, left and right; None for the video as it is.
        merged: The merged cut.
        from_left: Names of the parts taken from the left.
        name: The video's name, for a plan saved under a new ID.

    Returns:
        The merged cut with its compiled clips naming the plan they now come
        from, and what happened to the plans, in words; empty words when no
        plan was involved.
    """
    def plan_ids(project: Project) -> set:
        """The plans a cut's clips were compiled from."""
        return {clip.from_plan_id for track in project.tracks for clip in track.clips if clip.from_plan_id}

    def plan_at(project: Project, plan_id: str, commit: Optional[str]) -> Optional[EditPlan]:
        """A plan as it was at a version, or as it is."""
        return _plans_at(project, commit).get(plan_id) if commit else repo.get_plan(plan_id)

    left_ids, right_ids = plan_ids(before), plan_ids(after)
    if not from_left or not left_ids:
        return merged, ""
    if len(left_ids) != 1 or len(right_ids) != 1:
        return merged, "沒有合併剪輯計畫：兩邊不是各從一份計畫剪出來的"
    left_id, right_id = next(iter(left_ids)), next(iter(right_ids))
    left, right = plan_at(before, left_id, commits[0]), plan_at(after, right_id, commits[1])
    if left is None or right is None:
        return merged, "沒有合併剪輯計畫：找不到其中一份"
    if left.timeline_id != right.timeline_id:
        return merged, "沒有合併剪輯計畫：兩份計畫用的是不同的素材時間軸"
    combined = merge_plans(left, right, from_left)
    stored = repo.get_plan(right_id)
    shared = any(clip.from_plan_id == right_id for other in repo.list_projects() if other.id != merged.id
                 for track in other.tracks for clip in track.clips)
    if shared or stored is None:
        plan_id = _copied_plan(combined, f"「{name}」合成兩版時另存的計畫")
        said = "剪輯計畫也合併了；原本的計畫有別支影片在用，所以另存了一份" if shared else "剪輯計畫也依段落合併了"
    else:
        # Saved on top of the plan as it is now, whichever version the right side was.
        plan_id = repo.save_plan(combined.model_copy(update={"version": stored.version}), "合成兩版").id
        said = "剪輯計畫也依段落合併了"
    relinked = merged.model_copy(update={"tracks": [
        track.model_copy(update={"clips": [
            clip.model_copy(update={"from_plan_id": plan_id}) if clip.from_plan_id in (left_id, right_id) else clip
            for clip in track.clips]})
        for track in merged.tracks]})
    return relinked, said

def merge_versions(project_id: str, commit_a: str, commit_b: Optional[str], picks: Mapping[str, str],
                   expected_version: int) -> dict:
    """Make the video one cut of two of its versions, a part from each, as a new version on top.

    The editor's comparison view and `merge_version_parts` both come here.

    Args:
        project_id: The video.
        commit_a: The left version.
        commit_b: The right version, which everything not picked comes
            from; None for the video as it is.
        picks: `a` or `b` for part names and the whole-video picks in
            `WHOLE_PICKS`; anything left out is `b`.
        expected_version: The version the video is at now.

    Returns:
        `new_version`, the merged cut's length in `seconds`, `from_a` — what
        was taken from the left — and `plan` when a plan was merged or could
        not be.

    Raises:
        ValueError: If the project or a version does not exist, a pick is
            not `a` or `b`, or the project moved on from `expected_version`.
    """
    current = repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    if current.version != expected_version:
        raise ValueError(f"version conflict: project {project_id} is at version {current.version}, "
                         f"not {expected_version}")
    before, a = _version_at(project_id, commit_a)
    after, b = _version_at(project_id, commit_b)
    names = {part.name for part in cut_parts(before) + cut_parts(after)} | set(WHOLE_PICKS)
    unknown = sorted(set(picks) - names)
    if unknown:
        raise ValueError(f"nothing called {', '.join(unknown)} to pick: the parts are "
                         f"{', '.join(sorted(names - set(WHOLE_PICKS)))}, and the whole video's "
                         f"{', '.join(WHOLE_PICKS)}")
    if any(side not in ("a", "b") for side in picks.values()):
        raise ValueError("each pick is a (the first version) or b (the second)")
    merged = merge(before, after, picks)
    from_left = [name for name, side in picks.items() if side == "a" and name not in WHOLE_PICKS]
    merged, plan_said = _merged_plan(before, after, (commit_a or None, commit_b), merged, from_left,
                                     current.name or project_id)
    taken = [name for name, side in picks.items() if side == "a"]
    _say_in_history(f"合成兩版：{'、'.join(taken) or '全部'}用「{a.get('what')}」" if taken else "合成兩版")
    result = merged.model_copy(update={"id": project_id, "version": current.version + 1,
                                       "name": current.name, "branched_from": current.branched_from})
    validate_project(result, repo.get_assets({clip.asset_id for track in result.tracks for clip in track.clips}))
    repo.update_project(result, current.version)
    prepare_playback(repo.get_project(project_id))
    said = {"new_version": result.version, "seconds": round(float(result.duration), 3), "from_a": taken}
    if plan_said:
        said["plan"] = plan_said
    return said

@mcp.tool()
def merge_version_parts(project_id: str, commit_a: str, picks: Dict[str, Literal["a", "b"]], expected_version: int,
                        commit_b: Optional[str] = None, note: str = "") -> dict:
    """Make a video one cut of two of its versions, taking each part from one or the other.

    For 「開頭用第 3 版的，其他用現在的」. The parts are the ones
    `compare_versions` lists; each is taken whole, and `music`, `captions`
    and `caption_style` can be picked for the video as a whole. The result is
    a new version on top of the video — both versions it came from stay in
    the history. When both were compiled from one plan, the plan is merged
    beat by beat too, with the reasons and trims of the beats taken.

    Args:
        project_id: The video.
        commit_a: The other version, from `project_history`.
        picks: `a` for each part, or `music`/`captions`/`caption_style`/`frame`, to
            take from `commit_a`; anything left out comes from `commit_b`.
        expected_version: The project's current version.
        commit_b: The version the rest comes from; omit for the video as it is.
        note: The user's own words, kept in the history.

    Returns:
        `new_version`, its length in `seconds`, `from_a` — what was taken
        from `commit_a` — and `plan`, what happened to the plan, when it did.

    Raises:
        ValueError: If a version does not exist, a pick names no part, or the
            project moved on from `expected_version`.
    """
    return merge_versions(project_id, commit_a, commit_b, picks, expected_version)

@mcp.tool()
def mark_version(project_id: str, commit: str, mark: Literal["starred", "published"], on: bool = True) -> dict:
    """Star a version of a video, or mark it as the one that was published; or take the mark off.

    For 「把這版標起來」 or 「這版已經發到 IG 了」. The marks show in the editor's
    version panel and in `project_history`, so 「回到我標星號那版」 can be
    found later.

    Args:
        project_id: The video.
        commit: The version, from `project_history`.
        mark: `starred` or `published`.
        on: False takes the mark off.

    Returns:
        The version's `commit` and its `marks` now.

    Raises:
        ValueError: If the project or version does not exist.
    """
    if repo.get_project(project_id) is None:
        raise ValueError(f"project {project_id} not found")
    chosen = next((version for version in history.versions_of("projects/", f"{project_id}.json", 500)
                   if version.commit.startswith(commit.strip().lower())), None)
    if chosen is None:
        raise ValueError(f"project {project_id} has no version {commit}")
    repo.set_version_mark(project_id, chosen.commit, mark, on)
    marks = next((marked for version, _, marked in _project_versions(project_id, 500)
                  if version.commit == chosen.commit), set(repo.version_marks(project_id).get(chosen.commit, ())))
    return {"commit": chosen.commit[:10], "marks": sorted(marks)}

# ------------------------------------------------------------------ what the user said about a moment

def _current_commit(project_id: str) -> Optional[str]:
    """The version a project is at now, as its latest commit; None before it has one."""
    found = history.versions_of("projects/", f"{project_id}.json", 1)
    return found[0].commit if found else None

def _comment_said(project: Project, comment: Comment) -> dict:
    """A comment as the AI and the editor read it: where it is on the cut now, and what was said back."""
    now = where_now(project, comment)
    asset = repo.get_assets([comment.start_anchor.asset_id]).get(comment.start_anchor.asset_id) \
        if comment.start_anchor else None
    return {
        "id": comment.id, "text": comment.text, "by": comment.by, "status": comment.status,
        "at": clock(now["start"]) + (f"–{clock(now['end'])}" if now["end"] is not None else ""),
        "start": now["start"], "end": now["end"], "clip_id": now["clip_id"], "gone": now["gone"],
        "file": os.path.basename(asset.path) if asset else None,
        "file_seconds": comment.start_anchor.source if comment.start_anchor else None,
        "written": comment.created_at.isoformat(timespec="seconds"),
        "replies": [{"by": reply.by, "text": reply.text, "at": reply.at.isoformat(timespec="seconds"),
                     "commit": reply.commit[:10] if reply.commit else None}
                    for reply in comment.replies],
        # The version it was dealt with in, to watch what was done about it.
        "resolved_in": comment.resolved_commit[:10] if comment.resolved_commit else None,
    }

def open_comments(project_id: str) -> int:
    """How many things the user said about a project are still waiting.

    Args:
        project_id: The video.

    Returns:
        The number of open comments.
    """
    return sum(comment.status == "open" for comment in repo.comments(project_id))

def add_comment(project_id: str, text: str, start: float, end: Optional[float] = None) -> dict:
    """Write down what the user said about a moment of the cut, pinned to the footage under it.

    The editor's, never the AI's: a comment is the user speaking.

    Args:
        project_id: The video.
        text: What they said.
        start: Where on the cut, in seconds.
        end: Where the stretch it is about ends; None for a moment.

    Returns:
        The comment as `get_comments` shows it.

    Raises:
        ValueError: If the project does not exist or the text is empty.
    """
    project = repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    if not text.strip():
        raise ValueError("a comment needs something said")
    if end is not None and end < start:
        start, end = end, start
    if end is not None and end - start < SHORTEST_STRETCH_SECONDS:
        end = None
    comment = Comment(
        id="c" + uuid.uuid4().hex[:7], project_id=project_id, text=text.strip(), commit=_current_commit(project_id),
        start=round(max(0.0, start), 3), end=round(end, 3) if end is not None else None,
        start_anchor=anchor_at(project, start), end_anchor=anchor_at(project, end) if end is not None else None,
    )
    repo.save_comment(comment, f"留言：{comment.text[:30]}")
    return _comment_said(project, comment)

def change_comment(project_id: str, comment_id: str, by: str, reply: str = "",
                   status: Optional[Literal["open", "resolved"]] = None) -> dict:
    """Say something back on a comment, close it, or open it again.

    Args:
        project_id: The video.
        comment_id: The comment.
        by: Who is speaking: the AI program, or 你.
        reply: What is said; empty to only change its status.
        status: `resolved` to close it, `open` to open it again; None leaves it.

    Returns:
        The comment as `get_comments` shows it.

    Raises:
        ValueError: If the project or comment does not exist.
    """
    project = repo.get_project(project_id)
    comment = next((item for item in repo.comments(project_id) if item.id == comment_id), None)
    if project is None or comment is None:
        raise ValueError(f"project {project_id} has no comment {comment_id}")
    commit = _current_commit(project_id)
    if reply.strip():
        comment.replies.append(Reply(by=by, text=reply.strip(), commit=commit))
    if status == "resolved":
        comment.status, comment.resolved_commit = "resolved", commit
    elif status == "open":
        comment.status, comment.resolved_commit = "open", None
    said = {"resolved": "處理了一則留言", "open": "重開一則留言"}.get(status or "", "回覆一則留言")
    repo.save_comment(comment, f"{said}：{comment.text[:30]}")
    return _comment_said(project, comment)

@mcp.tool()
def get_comments(project_id: str, include_resolved: bool = False) -> dict:
    """Read what the user said about moments of a video while watching it in the editor.

    Each is a request written on the cut — 「這裡太拖」, 「這段配樂太大聲」 —
    pinned to the footage that was on screen, so it is shown where that
    footage is now even after the cut changed: `start` (and `end` for a
    stretch) in seconds, `at` as a timecode, the `clip_id` under it, and the
    `file` and `file_seconds`. `gone` means that footage has since been cut
    out. Read it, do what it asks — or ask the user if it can be read two
    ways — then close it with `resolve_comment`, saying what you did.

    Args:
        project_id: The video.
        include_resolved: Also list the ones already dealt with.

    Returns:
        `comments`, oldest first, each with its `id`, `text`, `status`,
        `replies` and, once closed, `resolved_in` — the version it was dealt
        with in; and `open`, how many are waiting.

    Raises:
        ValueError: If the project does not exist.
    """
    project = repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    every = repo.comments(project_id)
    return {
        "comments": [_comment_said(project, comment) for comment in every
                     if include_resolved or comment.status == "open"],
        "open": sum(comment.status == "open" for comment in every),
    }

@mcp.tool()
def resolve_comment(project_id: str, comment_id: str, reply: str, resolved: bool = True) -> dict:
    """Close a comment the user wrote, saying what was done about it.

    The reply is shown under the comment in the editor, in plain words the
    user reads: 「把開頭縮短 2 秒」, not the edits. The comment stays, greyed,
    and the user can open it again. With `resolved` false the reply is only
    said — to ask which of two things they meant, or why it was not done.
    Comments are the user's: there is no way to delete one from here.

    Args:
        project_id: The video.
        comment_id: The comment, from `get_comments`.
        reply: What was done, or what you need to know.
        resolved: False to reply and leave it open.

    Returns:
        The comment as `get_comments` shows it.

    Raises:
        ValueError: If the project or comment does not exist, or the reply
            is empty.
    """
    if not reply.strip():
        raise ValueError("say what was done, or what you need to know")
    return change_comment(project_id, comment_id, f"AI（{_calling_client.get()}）", reply,
                          "resolved" if resolved else None)

@mcp.tool()
def get_job(
    job_ids: List[str], wait_seconds: Annotated[float, Field(ge=0, le=MAX_JOB_WAIT_SECONDS)] = 0,
) -> dict:
    """Return the current state of background jobs, waiting for one to end if asked.

    Pass every job of a batch at once — all the analyses of a folder, or a
    render — rather than polling them one by one. With `wait_seconds`, the
    answer comes back as soon as any of them finishes, fails or is cancelled,
    or when that long has passed: call it again with the same wait until the
    batch is done, instead of asking every few seconds.

    Args:
        job_ids: IDs of the jobs, as returned by `render_project` or
            `analyze_asset`.
        wait_seconds: How long to wait for one of them to end, at most 50 —
            under the time most clients allow a tool call. 0 answers at once.

    Returns:
        A dictionary with `jobs`, each with its `job_id`, `kind`, `status`,
        `progress` (0.0 to 1.0), the current `stage`, the `asset_id` of an
        analysis or the `output_path` of a render, `remaining_seconds` once a
        running job has gone far enough for its pace to say, `ahead` — how
        many queued jobs go before a queued one — for a failed job its
        `error_message`, and for a finished analysis worth a second look a
        `note` to pass on; and a summary over all of them: `finished` and
        `total` counts, `failed` counting those that failed or were
        cancelled, and `progress`, the average.

    Raises:
        ValueError: If a job does not exist.
    """
    ended = {JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}
    listed = _job_states(job_ids)
    deadline = time.monotonic() + wait_seconds
    waiting = {entry["job_id"] for entry in listed if entry["status"] not in ended}
    while waiting and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        listed = _job_states(job_ids)
        if any(entry["job_id"] in waiting and entry["status"] in ended for entry in listed):
            break
    return {
        "jobs": listed,
        "finished": sum(entry["status"] in ended for entry in listed),
        "failed": sum(entry["status"] in ended - {JobStatus.COMPLETED.value} for entry in listed),
        "total": len(listed),
        "progress": round(sum(entry["progress"] for entry in listed) / len(listed), 3) if listed else 1.0,
    }

def queue_positions(active: Sequence[Job]) -> Dict[str, int]:
    """Say how many queued jobs go before each queued one.

    Args:
        active: The jobs still queued or running, in the order they will run.

    Returns:
        Each queued job's place, 0 for the next to start.
    """
    waiting = sorted((job for job in active
                      if job.status == JobStatus.QUEUED and job.admitted_at is None and not job.cancel_requested),
                     key=lambda job: job.priority)
    return {job.job_id: place for place, job in enumerate(waiting)}

def _job_states(job_ids: List[str]) -> List[dict]:
    """Read each job's state, one record apiece, in the order asked.

    Args:
        job_ids: The jobs.

    Returns:
        Their records.

    Raises:
        ValueError: If a job does not exist.
    """
    listed: List[dict] = []
    now = datetime.now(timezone.utc)
    ahead = queue_positions(repo.jobs_to_show(now)[0])
    for job_id in dict.fromkeys(job_ids):
        job = job_manager.get_job(job_id)
        if not job:
            raise ValueError(f"job {job_id} not found")
        entry = {
            "job_id": job.job_id, "kind": job.kind.value, "status": job.status.value,
            "progress": round(job.progress, 3), "stage": job.stage,
        }
        if job.asset_id:
            entry["asset_id"] = job.asset_id
        if job.output_path:
            entry["output_path"] = job.output_path
        remaining = job.remaining_seconds(now)
        if remaining is not None:
            entry["remaining_seconds"] = remaining
        if job.job_id in ahead:
            entry["ahead"] = ahead[job.job_id]
        if job.error_message:
            entry["error_message"] = job.error_message
        if job.kind == JobKind.ANALYZE and job.status == JobStatus.COMPLETED and job.asset_id:
            heard = repo.get_analysis(job.asset_id)
            voices = len({turn.speaker for turn in heard.speakers}) if heard else 0
            if voices > MANY_VOICES:
                entry["note"] = (
                    f"heard {voices} different voices, more than a video usually has — noisy talk splits one "
                    "person into many. If you know how many people talk, ask the user and analyze it again "
                    "with `speakers` and `again`"
                )
        listed.append(entry)
    return listed

# Provisional (roadmap §13): more voices than this in one file is far more likely a noisy
# recording split apart than that many people — 40 for two friends talking over traffic.
MANY_VOICES = 8

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

@mcp.tool()
def storage_usage() -> dict:
    """Say how much room clip-mcp takes on this computer, and what can be cleared.

    Use it when the user asks how much space this takes or wants room back.
    Read the sizes out in plain words; never clear anything they did not ask
    to clear.

    Returns:
        `items`, each with a `key`, the `label` the user knows it by, its size
        in `megabytes`, the `path` of its folder, and whether it is
        `clearable` with `clean_storage` — only what can be made again is.
        `legacy` is the old outputs folder from before finished videos were
        saved to the videos folder: sort it out with `tidy_old_outputs`.
        `outputs` is where finished videos, covers and timeline exports are
        saved now, and is never cleared. `trash` is the app's trash: deleted
        projects and finished videos, kept for 30 days.

        `videos`, most room to give back first: each with its `versions`,
        how many are `exported`, the megabytes its finished videos and its
        previews take, and `freeable_megabytes` — what deleting its other
        versions would give back. That, emptying the trash and deleting a
        project are the user's to do in the editor (儲存空間, and each
        video's 版本 tab); suggest them, never claim to have done them.
        `trash` lists what is in the trash.
    """
    items = housekeeping.usage(WORKSPACE_DIR, str(output_dir()))
    videos = []
    for project in repo.list_projects():
        found = project_storage(project.id)
        videos.append({"project_id": project.id, "name": project.name, **{
            key: found[key] for key in ("versions", "exported", "outputs_megabytes", "cache_megabytes",
                                        "freeable_megabytes")}})
    return {"items": [
        {"key": item.key, "label": item.label, "megabytes": round(item.bytes / 1e6, 1), "path": item.path,
         "clearable": item.clearable}
        for item in items
    ], "videos": sorted(videos, key=lambda video: -video["freeable_megabytes"]),
        "trash": list_trash()["items"]}

@mcp.tool()
def clean_storage(kinds: List[Literal["previews", "thumbnails", "playback", "work"]]) -> dict:
    """Clear files that can be made again, to give the user disk space back.

    Nothing the user made is touched: projects, analyses and finished videos
    stay. A preview, a thumbnail or a working file still in use by a render is
    left. The next preview of a project renders its pictures again, which
    takes longer once.

    Args:
        kinds: `previews` (preview renders, sound checks and the pictures they
            are made from), `thumbnails` (the editor's thumbnails and
            waveforms), `playback` (the small copies the editor plays a cut
            from, made again when a project is opened), `work` (what analyses
            and renders leave behind).

    Returns:
        `freed_megabytes`.
    """
    job_manager.forget_abandoned()
    freed = housekeeping.clean(WORKSPACE_DIR, kinds, repo.active_job_ids())
    return {"freed_megabytes": round(freed / 1e6, 1)}

@mcp.tool()
def tidy_old_outputs(confirm: bool = False) -> dict:
    """Sort out the old outputs folder, where every render used to be kept.

    Without `confirm`, only says what would happen: the newest finished video,
    cover and export of each project move to the videos folder, and the older
    versions of them go to the recycle bin, from where they can still be
    restored. Sound checks and render logs stay where they are; `clean_storage`
    is not how they go either, so say the folder is left with them. Show the
    user that list and the sizes, and call again with `confirm` only once they
    have agreed to it.

    Args:
        confirm: Carry it out.

    Returns:
        Without `confirm`: `keep` (each file kept and where it goes),
        `recycle` (how many duplicates and megabytes go to the recycle bin) and
        `left_in_place` (what stays in the old folder). With
        it: `kept` (where each went), `recycled`, and `left` — files that
        could not be moved, such as one open in a player.
    """
    planned = housekeeping.legacy_plan(WORKSPACE_DIR, str(output_dir()))
    if not confirm:
        recycled = [item for item in planned if item.duplicate]
        stays = [item for item in planned if not item.keep_as and not item.duplicate]
        return {
            "keep": [{"from": item.path, "to": item.keep_as, "megabytes": round(item.bytes / 1e6, 1)}
                     for item in planned if item.keep_as],
            "recycle": {"files": len(recycled), "megabytes": round(sum(item.bytes for item in recycled) / 1e6, 1)},
            "left_in_place": {"files": len(stays), "megabytes": round(sum(item.bytes for item in stays) / 1e6, 1)},
        }
    return housekeeping.legacy_apply(WORKSPACE_DIR, planned)

def main():
    """Run the MCP server over the stdio transport."""
    threading.Thread(target=tend_library, name="tend-library", daemon=True).start()
    # The banner goes to stderr, which every client keeps as its server log; a box of
    # ASCII art in it on every start is noise, and on a console that is not UTF-8 it
    # arrives mangled.
    mcp.run(show_banner=False)

if __name__ == "__main__":
    main()
