import hashlib
import json
import os
from datetime import datetime, timezone
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent
from pydantic import BaseModel, Field, TypeAdapter
from app.models.media import Asset, MediaAnalysis, Source, Span, SpeakerTurn
from app.models.plan import (
    EditPlan, MusicCue, PlanAmendment, SetMusicCueOp, Trim, TrimKind, apply_amendment, describe_amendment,
    short_clip_ids, whole_clip_id, with_whole_clip_ids,
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
    FitTrackOp,
    Project,
    SetCaptionStyleOp,
    SetSubtitlesOp,
    SubtitleCue,
    TimeRange,
    Track,
    TrackType,
    retimed_words,
    mostly_over,
    apply_operation,
    validate_project,
)
from app.models.comment import Comment, Preference, Question, Reply
from app.models.offer import TRIAL_VOLUME, OfferedSong, SongOffer
from app.models.trial import Trial, Verdict
from app.models.job import Job, JobKind, JobStatus
from app.engine import cards, loudness, meaning, stills
from app.engine.analysis import marked_words, unsure_words
from app.engine.plan import (
    BROLL_TRACK_ID, COMPILED_TRACKS, MUSIC_TRACK_ID, VIDEO_TRACK_ID, as_left, broll_covers, broll_slots, check_plan,
    check_recompile, compile_operations, compiled_duration, diff_plans, music_beds, Note, note_record, piece_changes,
    plan_pieces,
)
from app.engine.sections import build_sections, candidate_hash, check_sections, propose_candidates
from app.engine.semantic import (
    build_timeline, clean_cuts, content_scores, join_voices, maybe_one_person, timeline_input_hash, voice_seconds,
    was_made_up,
)
from app.engine.ffmpeg import graph_from_file, hidden_window_flags
from app.engine.probe import picture_size, probe_file, recorded_at, rights_tags, timecode_start
from app.engine.reframe import placement_at
from app.engine.library import FOLDER_KINDS, FOOTAGE, LIBRARIES, MUSIC, classify, library_of
from app.engine import clap
from app.engine import fetch
from app.engine.builder import DEFAULT_LOUDNESS_TARGET, voice_keys
from app.engine.comments import anchor_at, where_now
from app.engine.compare import compare
from app.engine.compare import parts_of as cut_parts
from app.engine.merge import WHOLE_PICKS, merge, merge_plans
from app.engine.delivery import (
    Finding, chapters, clock, cover_candidates, frames_around, marked_parts, on_screen, rights_facts,
)
from app.engine.joins import HeardWord, Join, joins_in, pause_across, words_heard
from app.engine.frames import feed_preview, format_timestamp, still, storyboard_sheet
from app.engine.interchange import write_edl, write_fcpxml, write_otio, write_srt
from app.engine.listen import WINDOW_SECONDS, chart, cut_sound, describe, levels, levels_every, parts_of
from app.engine.subtitles import (
    DEFAULT_MAX_CHARACTERS, DEFAULT_MAX_SECONDS, caption_geometry, captioned_clips, place_cues, timeline_cues,
)
from app.storage import housekeeping
from app.storage.history import Version
from app.workspace import output_dir
from app import runtime
# By name, unlike everything else from runtime: every tool is registered with it, and it is never swapped out.
from app.runtime import mcp
# Importing a tool module is what registers its tools, so every one is imported here.
from app.tools import examining, rendering, shared  # noqa: F401

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
MEDIA_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".mts", ".m2ts", ".wmv", ".flv", ".3gp",
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff",
}) | stills.IMAGE_EXTENSIONS

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
            named[key] = shared.asset_id_for(item)
        elif key == "asset_ids" and isinstance(item, list):
            named[key] = [shared.asset_id_for(name) for name in item]
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
        token = runtime.calling_client.set(client)
        try:
            with runtime.history.step(f"AI（{client}）", note=note, doing=DOING.get(name, name)):
                return await call_next(context)
        finally:
            runtime.calling_client.reset(token)

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

    existing = runtime.repo.find_asset_by_path(path)
    streams = info.get("streams", [])
    duration = info.get("format", {}).get("duration")
    photo = stills.is_image(path)
    if photo:
        # A photo is as long as it is shown.
        duration = stills.STILL_LONGEST
    asset = Asset(
        id=existing.id if existing else str(uuid.uuid4()),
        path=path,
        still=photo,
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
        asset.licence = existing.licence
        asset.library = existing.library
        asset.source = existing.source
    if library in LIBRARIES:
        asset.library = library
    elif asset.library is None:
        asset.library = classify(asset)
    runtime.repo.save_asset(asset)
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
    busy = {job.asset_id for job in runtime.repo.jobs_to_show(datetime.now(timezone.utc))[0] if job.kind == JobKind.ANALYZE}
    started = []
    for asset in assets:
        if library_of(asset) != MUSIC or asset.duration is None or asset.id in busy:
            continue
        existing = runtime.repo.get_analysis(asset.id)
        if existing is not None and existing.rhythm is not None and existing.music is not None \
                and not examining.is_stale(existing):
            continue
        # Behind the footage somebody asked about: a song can wait until it is wanted.
        started.append(examining.start_analysis(asset, False, "accurate", priority=1).job_id)
    return started

def tend_library() -> None:
    """Sort the files added before there were two libraries, and queue the songs not yet analyzed.

    Run once when the server or the editor starts, in the background: listening
    takes a moment per file. Whichever starts first does it; the other finds
    it taken and leaves it.
    """
    from filelock import FileLock, Timeout

    try:
        with FileLock(os.path.join(runtime.WORKSPACE_DIR, "library.lock"), timeout=0):
            for asset in runtime.repo.list_assets():
                if asset.library is None and os.path.exists(asset.path):
                    try:
                        asset.library = classify(asset)
                    except (OSError, RuntimeError, ValueError):
                        # Left where its picture says until it can be listened to; the next start tries again.
                        continue
                    runtime.repo.save_asset(asset)
            _queue_music(runtime.repo.list_assets())
    except Timeout:
        return

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
        `duration` in seconds (or null if unknown, and for a photo, which
        lasts as long as its clip; `still` is true for one), and
        `has_video` / `has_audio` flags.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe cannot read the file.
    """
    return shared.plain(_asset_record(register_asset(filepath)))

def _asset_record(asset: Asset) -> dict:
    """An asset as a tool hands it back: a photo with no length, since it has none of its own."""
    record = asset.model_dump()
    if asset.still:
        record["duration"] = None
    return record

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
        `duration` in seconds (null for a photo, which lasts as long as its
        clip; `photo` is true for one), `has_video`, `has_audio`, `analyzed`,
        `transcription` (`fast`, `balanced`, `accurate`, or null when speech was not
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
    filed = {**runtime.repo.folder_of("assets"), **runtime.repo.folder_of("music")}
    matching = [
        asset for asset in runtime.repo.list_assets()
        if (kind is None or library_of(asset) == kind)
        and (folder_id is None or filed.get(asset.id) == folder_id)
        and (not text or text.lower() in os.path.basename(asset.path).lower())
    ]
    analyzing = examining.analyzing_assets()
    heard = examining.song_tags() if kind == MUSIC or any(library_of(asset) == MUSIC for asset in matching) else {}
    matching.sort(key=lambda asset: _natural_key(os.path.basename(asset.path)))
    assets = []
    for asset in matching[offset:offset + limit]:
        analysis = runtime.repo.get_analysis(asset.id)
        assets.append({
            "id": asset.id,
            "name": os.path.basename(asset.path),
            # A photo has no length of its own: it lasts as long as a clip of it is made to.
            "duration": None if asset.duration is None or asset.still else float(asset.duration),
            "has_video": asset.has_video,
            "has_audio": asset.has_audio,
            **({"photo": True} if asset.still else {}),
            "analyzed": analysis is not None,
            "transcription": examining.transcription_kind(analysis),
            "stale": analysis is not None and examining.is_stale(analysis),
            "library": library_of(asset),
            "folder_id": filed.get(asset.id),
            **({"notes": asset.notes} if asset.notes else {}),
            **(examining.song_details(asset, analysis, heard, analyzing) if library_of(asset) == MUSIC
               else {"analyzing": True} if asset.id in analyzing else {}),
        })
    return {"assets": assets, "total": len(matching), "offset": offset,
            "folders": runtime.repo.list_folders(_folder_kind(kind or FOOTAGE))}

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
    heard = examining.song_tags()
    vectors, found, waiting = {}, {}, 0
    for asset in runtime.repo.list_assets():
        if library_of(asset) != MUSIC or not os.path.exists(asset.path):
            continue
        analysis = runtime.repo.get_analysis(asset.id)
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
            **examining.sounds_like(asset_id, analysis, heard),
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
    shared.say_in_history(f"從 {planned['site']} 加入「{title}」")
    asset = register_asset(path, MUSIC).model_copy(update={"source": Source(
        site=planned["site"], page=page, url=url, title=title, artist=artist, licence=planned["licence"],
        credit=planned["credit"], personal_only=personal_use,
    )})
    runtime.repo.save_asset(asset)
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
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    used = runtime.repo.get_assets({clip.asset_id for track in project.tracks if track.track_type == TrackType.AUDIO
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

@mcp.tool()
def export_notices(
    project_id: str,
    text: Optional[str] = None,
    licences: Optional[Dict[str, str]] = None,
    next_to: Optional[str] = None,
) -> dict:
    """Write what the user has to take care of before publishing a cut, as a text file.

    Call it after the final render, twice. The first time, without `text`, it
    says what is in the cut that may need permission or has to be disclosed —
    the music under it, where each song came from and what each file says
    about itself, other voices, more than one face on screen — and nothing is
    written. Write the notices from that, in the user's language, and call
    again with `text`: for each thing, what the concern is and how to resolve
    it (the credit line a CC BY song needs, a source to confirm for music
    whose licence nobody gave, the consent of somebody heard or seen). Add
    what you see in the frames and hear in the footage that no measurement
    finds: a brand or a logo, music playing in the room, a screen, somebody's
    work shown. Say plainly where a question is the user's to answer rather
    than yours. `music_credits` gives the credit lines alone.

    Args:
        project_id: ID of the project.
        text: The notices, as the user should read them. Leave it out to be
            told what they have to cover.
        licences: Where each file came from and what it may be used for, as
            the user told you, keyed by asset ID — such as "Kevin MacLeod,
            incompetech.com, CC BY 4.0". Kept with the file, so the next
            video that uses it already knows.
        next_to: The rendered file the notices go with; they are written
            beside it. Left out, they go to the videos folder, named after
            the project.

    Returns:
        `facts`, what the notices have to cover; and once `text` is given,
        `output_path`, the file written — the text followed by those facts,
        so none of them can be left out.

    Raises:
        ValueError: If the project, or a file named in `licences`, does not exist.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    assets = shared.referenced_assets(project)
    # Every file looked up before any is written, so a refusal changes nothing.
    named = {asset_id: assets.get(asset_id) or runtime.repo.get_assets([asset_id]).get(asset_id) for asset_id in licences or {}}
    unknown = [asset_id for asset_id, asset in named.items() if asset is None]
    if unknown:
        raise ValueError(f"asset {', '.join(unknown)} not found; no licence was kept")
    for asset_id, licence in (licences or {}).items():
        asset = named[asset_id].model_copy(update={"licence": licence.strip() or None})
        runtime.repo.save_asset(asset)
        assets[asset_id] = asset
    tags: Dict[str, Dict[str, str]] = {}
    for asset_id, asset in assets.items():
        try:
            tags[asset_id] = rights_tags(probe_file(asset.path))
        except (OSError, RuntimeError, ValueError):
            tags[asset_id] = {}
    facts = rights_facts(project, rendering.analyses_of(project), assets, tags)
    listed = [
        " | ".join([fact.kind, fact.subject, ", ".join(f"{clock(start)}–{clock(end)}" for start, end in fact.spans),
                    *fact.detail])
        for fact in facts
    ]
    if text is None:
        return {"facts": listed, "written": False, "next": (
            "Write the notices from these facts and what you saw and heard in the cut, in the user's language, "
            "and call again with `text`" + ("" if next_to else " and `next_to` the rendered file")
        )}
    if next_to:
        if not os.path.isdir(os.path.dirname(os.path.abspath(next_to))):
            raise ValueError(f"there is no folder for {next_to}; give the rendered file's path")
        stem, _ = os.path.splitext(next_to)
        path = housekeeping.unique_path(os.path.dirname(os.path.abspath(next_to)), f"{os.path.basename(stem)}_notices.txt")
    else:
        path = housekeeping.unique_path(str(output_dir()), rendering.delivery_name(project, "notices", ".txt"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # With a byte-order mark and Windows line ends, so Notepad opens it right.
    with open(path, "w", encoding="utf-8-sig", newline="\r\n") as out:
        out.write(text.rstrip() + "\n\n---\n" + "\n".join(listed or ["(nothing found)"]) + "\n")
    return {"facts": listed, "written": True, "output_path": path}

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
    assets = [shared.get_asset(asset_id) for asset_id in asset_ids]
    pictured = [os.path.basename(asset.path) for asset in assets if asset.has_video]
    if library == MUSIC and pictured:
        raise ValueError(f"{', '.join(pictured)} has a picture, so it is footage: music is sound alone")
    moved = runtime.repo.move_to_library([asset.id for asset in assets], library)
    _queue_music(moved)
    if library == FOOTAGE and ANALYZE_MUSIC_ON_ADD:
        # Heard as a song, it was never transcribed: now it is listened to for what is said.
        for asset in moved:
            existing = runtime.repo.get_analysis(asset.id)
            if existing is not None and existing.transcript is None and asset.has_audio:
                examining.start_analysis(asset, True, "accurate", priority=1)
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
    asset = shared.get_asset(asset_id).model_copy(update={"notes": notes.strip()})
    runtime.repo.save_asset(asset)
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
    names = {folder["id"]: folder["name"] for kind in ("assets", "music") for folder in runtime.repo.list_folders(kind)}
    done = []
    for step in steps:
        try:
            if isinstance(step, CreateFolder):
                made = runtime.repo.create_folder(uuid.uuid4().hex, _folder_kind(step.library), step.name, step.parent_id)
                names[made["id"]] = made["name"]
                if step.asset_ids:
                    _file_in(step.asset_ids, made["id"], step.library)
                done.append(f"新增資料夾「{made['name']}」" + (f"，放進 {len(step.asset_ids)} 個檔案" if step.asset_ids else ""))
            elif isinstance(step, RenameFolder):
                old = names.get(step.folder_id, step.folder_id)
                runtime.repo.rename_folder(step.folder_id, step.name)
                names[step.folder_id] = step.name
                done.append(f"資料夾「{old}」改名為「{step.name}」")
            elif isinstance(step, MoveFolder):
                runtime.repo.move_folder(step.folder_id, step.parent_id)
                done.append(f"把資料夾「{names.get(step.folder_id, step.folder_id)}」移到"
                            + (f"「{names.get(step.parent_id, step.parent_id)}」裡" if step.parent_id else "最上層"))
            else:
                _file_in(step.asset_ids, step.folder_id)
                done.append(f"把 {len(step.asset_ids)} 個檔案放進"
                            + (f"「{names.get(step.folder_id, step.folder_id)}」" if step.folder_id else "最上層"))
        except LookupError as exc:
            raise ValueError(str(exc).strip("'\"")) from exc
    shared.say_in_history("；".join(done))
    return {"done": done, "folders": runtime.repo.list_folders("assets"), "music_folders": runtime.repo.list_folders("music")}

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
    assets = [shared.get_asset(asset_id) for asset_id in asset_ids]
    kinds = {library_of(asset) for asset in assets} | ({library} if library else set())
    if len(kinds) > 1:
        raise ValueError("footage and music are filed in folders of their own: file them in separate steps, or "
                         "move a file to the other library first with set_asset_library")
    runtime.repo.file_items(_folder_kind(kinds.pop() if kinds else FOOTAGE), [asset.id for asset in assets], folder_id)

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
    for project in runtime.repo.list_projects():
        for track in project.tracks:
            for clip in track.clips:
                users.setdefault(clip.asset_id, set()).add(project.name or project.id)
                using.setdefault(clip.asset_id, set()).add(project.id)
    working = runtime.repo.jobs_to_show(datetime.now(timezone.utc))[0]
    taken: set = set()
    planned = []
    for move in moves:
        asset = shared.get_asset(move.asset_id)
        if not os.path.isabs(move.to_folder):
            raise ValueError(f"{move.to_folder} is not a full folder path")
        folder = os.path.abspath(move.to_folder)
        if any(_inside(folder, own) for own in (runtime.WORKSPACE_DIR, str(output_dir()))):
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
    shared.say_in_history(note or f"搬動了 {len(planned)} 個檔案")
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
        asset = shared.get_asset(plan["asset_id"])
        runtime.repo.save_asset(asset.model_copy(update={"path": target}))
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

    # Photos have no length of their own: counted, an album would read as days of footage.
    total = sum((asset.duration for asset in assets if asset.duration is not None and not asset.still), Decimal(0))
    return shared.plain({"assets": [_asset_record(asset) for asset in assets], "total_duration": total, "skipped": skipped})

# Provisional (roadmap §13): the longest `get_job` waits, under the same allowance.
MAX_JOB_WAIT_SECONDS = 50.0

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
        `scores`, and `tags` with where each came from. With `include_words`,
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

@mcp.tool()
def view_frames(
    asset_ids: list[str],
    start: Annotated[float, Field(ge=0)] = 0,
    end: Optional[float] = None,
    count: Annotated[int, Field(ge=1, le=shared.MAX_STORYBOARD_TILES)] = 12,
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
    if len(asset_ids) * count > shared.MAX_STORYBOARD_TILES:
        raise ValueError(
            f"{len(asset_ids)} assets at {count} frames each is more than one sheet holds; "
            f"ask for at most {shared.MAX_STORYBOARD_TILES} frames in total"
        )
    if len(asset_ids) > 1 and (start or end is not None):
        raise ValueError("a time range only means something for one asset; drop start and end, or ask about one file")

    shots: List[tuple[str, float, str]] = []
    listing: List[str] = []
    for asset_id in asset_ids:
        asset = shared.get_asset(asset_id)
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
def view_render(
    job_id: str,
    at: Literal["even", "captions", "spots"] = "even",
    times: Optional[List[Annotated[float, Field(ge=0)]]] = None,
    start: Annotated[float, Field(ge=0)] = 0,
    end: Optional[float] = None,
    count: Annotated[int, Field(ge=1, le=shared.MAX_STORYBOARD_TILES)] = 12,
) -> ToolResult:
    """Look at frames of a finished render: what the viewer will see, captions and transitions included.

    `preview_project` shows the footage a cut is made of; this shows the file
    that came out, so it is how to check what only rendering makes — captions
    burned in where they belong and readable, transitions that really happen,
    the framing of another shape. Render a preview, look here, fix what is
    wrong, and only then render the final cut. How one shot hands over to the
    next — every cut, transition and picture laid over — is `inspect_join`.

    The frames are returned to you as a tool result, which means they leave
    this machine if you are not running on it.

    Args:
        job_id: The render, preview or full, from `render_project`.
        at: `even` spreads `count` frames over the stretch; `captions` takes
            one from the middle of each caption on screen; `spots` takes, for
            each stretch reading every frame found black, frozen or a flash,
            the frame before it, its middle and the frame after.
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
    job, path = rendering.finished_render(job_id)
    project = rendering.rendered_from(job)
    length = float(project.duration) if project is not None else None
    last = length if end is None else (end if length is None else min(end, length))
    said: List[str] = []
    wanted: List[tuple] = []
    if times:
        wanted = [(seconds, "") for seconds in times]
    elif at == "spots":
        if job.spots is None:
            said.append("this render was not read frame by frame — the check failed or was stopped — so nothing "
                        "is known about black, frozen or flashing frames in it; render again to have it read")
        elif not job.spots:
            said.append("every frame of this render was read and nothing was found black, frozen or a flash")
        step = 1 / (project.fps_num / project.fps_den) if project is not None else 0.04
        for spot in job.spots or []:
            begin, finish = float(spot["start"]), float(spot["end"])
            if finish <= start or (last is not None and begin >= last):
                continue
            name = f"{spot['kind']} {begin:.2f}–{finish:.2f}s"
            wanted += [(max(0.0, begin - step), f"{name}, the frame before"), ((begin + finish) / 2, f"{name}, middle"),
                       (finish + step, f"{name}, the frame after")]
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
    if len(wanted) > shared.MAX_STORYBOARD_TILES:
        said.append(f"{len(wanted)} frames wanted, the first {shared.MAX_STORYBOARD_TILES} shown; look at the rest with "
                    "a later `start`")
        wanted = wanted[:shared.MAX_STORYBOARD_TILES]
    if rendering.changed_since(job):
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

# How much of a join `inspect_join` looks at unless asked for more: a second and a half
# either side, four frames a second. Enough for a movement to carry through or not, and
# one join is a sheet of fourteen frames.
JOIN_SECONDS = 1.5
JOIN_FPS = 4.0
# How the two frames either side of a cut are framed, so the cut can be found in the sheet.
CUT_BORDER = (230, 60, 60)

def _project_as_rendered(job: Job) -> Tuple[Optional[Project], bool]:
    """The project a render was made from, as it was when it was rendered.

    Args:
        job: The render.

    Returns:
        `(project, exact)`: the project at the version rendered and True; or,
        when that version is no longer in its history, the project as it is
        now and False. `(None, False)` when the project is gone.
    """
    project = rendering.rendered_from(job)
    if project is None or not rendering.changed_since(job):
        return project, project is not None
    for version in runtime.history.versions_of("projects/", f"{project.id}.json", 50):
        data = _project_json_at(project.id, version.commit)
        if data is not None and data.get("version") == job.project_version:
            return Project.model_validate(data), True
    return project, False

def _found_at(join: Join, findings: Sequence[Finding]) -> List[Finding]:
    """The findings about the stretch of the cut that looking at a join shows."""
    opened, closed = join.at - JOIN_SECONDS, join.at + JOIN_SECONDS
    return [finding for finding in findings
            if any(start <= closed and end >= opened for start, end in finding.spans)]

def _join_line(number: int, join: Join, found: Sequence[Finding]) -> str:
    """Say where a join is, what kind, between what, and what was found there."""
    line = (f"#{number} {format_timestamp(join.at)} ({join.at:.2f}s) {join.kind}: {join.before} → {join.after}"
            + (f"; part 「{join.part}」 begins" if join.part else "")
            + (f"; the sound changes over at {join.sound_at:.2f}s" if join.sound_at is not None else ""))
    return "\n".join([line, *(f"    found: {finding.check}: {finding.message}" for finding in found)])

def _words_across(join: Join, heard: Sequence[HeardWord]) -> str:
    """Write the words heard around a join, in order, with ‖ where the picture cuts."""
    words: List[str] = []
    side, marked = None, False
    for word in heard:
        if not marked and word.start >= join.at:
            words.append("‖")
            marked = True
        if side != word.clip_id:
            words.append(f"[clip {word.clip_id}]")
            side = word.clip_id
        words.append(f"{word.start - join.at:+.2f} {word.text}" + ("✂" if word.cut else ""))
    if not heard:
        return "none transcribed"
    return " ".join(words + ([] if marked else ["‖"]))

def _levels_across(path: str, join: Join, opened: float, closed: float, keep_as: str) -> str:
    """Measure a render's sound around a join, keeping the stretch to play."""
    try:
        cut_sound(path, opened, closed, keep_as)
    except RuntimeError:
        return "the render has no sound here"
    measured = levels_every(keep_as, WINDOW_SECONDS)
    split = round((join.at - opened) / WINDOW_SECONDS)
    return (f"dB every {WINDOW_SECONDS:g}s from {opened - join.at:+.2f}s: "
            + " ".join(f"{level:g}" for level in measured[:split]) + " ‖ "
            + " ".join(f"{level:g}" for level in measured[split:]) + f"\n  sound: {keep_as}")

@mcp.tool()
def inspect_join(
    job_id: str,
    joins: Optional[List[Annotated[int, Field(ge=1)]]] = None,
    at: Optional[List[Annotated[float, Field(ge=0)]]] = None,
    seconds: Annotated[float, Field(gt=0, le=3)] = JOIN_SECONDS,
    fps: Annotated[float, Field(gt=0, le=8)] = JOIN_FPS,
) -> ToolResult:
    """Look at how a finished render hands over from one shot to the next.

    A storyboard shows each shot; whether a movement carries through a cut,
    or a sentence ends before the picture moves on, is in the second either
    side. Every join counts: plain cuts, transitions, B-roll coming in and
    going off, the start of a part.

    With neither `joins` nor `at`, it lists the render's joins and no frames:
    number, time, kind, what is either side, and what `check_render` found
    there. Then look at the ones that matter, by number or by time.

    The frames are returned to you as a tool result, which means they leave
    this machine if you are not running on it.

    Args:
        job_id: The render, preview or full, from `render_project`.
        joins: Numbers from the list; they belong to this render only.
        at: Seconds of the render; each looks at the join nearest it.
        seconds: How far either side of each join to look.
        fps: Frames a second to take.

    Returns:
        The list; or for each join its frames, a row before the cut and a row
        from it with the two either side framed red; the transcribed words
        laid onto the render's time, ✂ on any a cut takes into, and what
        they read as, corrected with the captions' glossary; the captions on
        screen there, as written or corrected by hand, which are what was
        said where the two differ; where a line
        ends and another begins across the cut, the pause between them beside
        the pauses the speaker makes between sentences; the render's
        sound level every 0.1s with ‖ at the cut; and a WAV of the stretch to
        play or hear again with `listen_again`. It says when the project has
        changed since the render, and which joins did not fit on the sheet.

    Raises:
        ValueError: If the job is not a finished render, a number is not in
            the list, or one join is more frames than a sheet holds.
    """
    job, path = rendering.finished_render(job_id)
    project, exact = _project_as_rendered(job)
    if project is None:
        raise ValueError(f"the project render {job_id} was made from is gone, so its joins are not known")
    said: List[str] = []
    if not exact:
        said.append("the project has been edited since this render and the version rendered is no longer kept, "
                    "so these are the joins of the cut as it is now; render again to be sure")
    elif rendering.changed_since(job):
        said.append("the project has been edited since this render; these are the joins of the version rendered")
    listed = joins_in(project, {asset_id: os.path.basename(asset.path)
                                for asset_id, asset in shared.referenced_assets(project).items()})
    if not listed:
        return ToolResult(content=["\n".join([*said, f"render {job_id} has no joins: it is one shot"])])
    findings = [finding for finding in rendering.check_findings(project, job.captioned) if finding.spans]
    if not joins and not at:
        return ToolResult(content=["\n".join([
            *said, f"Joins of render {job_id}; look at any with `joins` or `at`:",
            *(_join_line(number, join, _found_at(join, findings)) for number, join in enumerate(listed, start=1)),
        ])])

    wanted: List[int] = []
    for number in joins or []:
        if number > len(listed):
            raise ValueError(f"render {job_id} has {len(listed)} joins; there is no #{number}")
        wanted.append(number)
    for seconds_in in at or []:
        wanted.append(1 + min(range(len(listed)), key=lambda index: abs(listed[index].at - seconds_in)))
    wanted = list(dict.fromkeys(wanted))
    steps = round(seconds * fps)
    per_join = 2 * (steps + 1)
    if per_join > shared.MAX_STORYBOARD_TILES:
        raise ValueError(f"{seconds:g}s at {fps:g} frames a second is {per_join} frames a join, more than the "
                         f"{shared.MAX_STORYBOARD_TILES} a sheet holds; look less far, or more thinly")
    fits = shared.MAX_STORYBOARD_TILES // per_join
    if len(wanted) > fits:
        said.append(f"{', '.join(f'#{number}' for number in wanted[fits:])} did not fit on the sheet; "
                    "look at them in another call")
        wanted = wanted[:fits]

    length = float(project.duration)
    frame = project.fps_den / project.fps_num
    analyses = rendering.analyses_of(project)
    placed = place_cues(project, project.subtitles) if project.subtitles else []
    shots: list = []
    borders: list = []
    text: List[str] = []
    for number in wanted:
        join = listed[number - 1]
        before = [join.at - step / fps for step in range(steps, 0, -1)] + [join.at - frame]
        after = [join.at] + [join.at + step / fps for step in range(1, steps + 1)]
        for times, edge in ((before, len(before) - 1), (after, 0)):
            for index, moment in enumerate(times):
                shots.append((path, round(min(max(0.0, moment), length - 0.05), 3),
                              f"#{number} {moment - join.at:+.2f}s"))
                borders.append(CUT_BORDER if index == edge else None)
        opened, closed = max(0.0, join.at - seconds), min(length, join.at + seconds)
        keep_as = os.path.join(runtime.WORKSPACE_DIR, "temp", "joins", f"render-{job_id[:8]}-{join.at:.2f}.wav")
        heard = words_heard(project, analyses, opened, closed)
        text += [
            _join_line(number, join, _found_at(join, findings)),
            f"  frames: a row of {steps + 1} up to the cut, then {steps + 1} from it",
            "  words: " + _words_across(join, heard),
        ]
        if heard:
            # The words above are as the transcriber timed them; this is what they say, corrected
            # with the glossary the captions use, so a misheard word fixed once reads fixed here.
            sides = [shared.glossed("".join(word.text for word in heard if (word.start >= join.at) == later))
                     for later in (False, True)]
            text.append(f"  reads: 「{sides[0]}」 ‖ 「{sides[1]}」")
        # The captions are what somebody read and corrected, word for word, where the transcript
        # and its glossary only fix what was misheard the same way every time.
        shown = [f"{float(cue.start) - join.at:+.2f} {cue.text}" for cue in placed
                 if float(cue.end) > opened and float(cue.start) < closed]
        if shown:
            text.append("  captions: " + " / ".join(shown))
        pause = pause_across(join.at, heard, analyses, join.part)
        if pause:
            text.append("  pause: " + pause)
        text += [
            "  levels: " + _levels_across(path, join, opened, closed, keep_as),
        ]
    return ToolResult(content=[
        "\n".join([*said, *text]),
        Image(data=storyboard_sheet(shots, columns=steps + 1, borders=borders), format="jpeg"),
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
        `id` and initial `version`, and `preferences`: how the user likes
        their videos, as they said it on earlier ones — cut it that way.
    """
    if not name.strip():
        raise ValueError("a project needs a name: what the video is about, such as 'EP1 台北'")
    project = Project(
        id=str(uuid.uuid4()), name=name.strip(),
        width=width, height=height, fps_num=fps_num, fps_den=fps_den,
    )
    runtime.repo.add_project(project)
    return _with_preferences(shared.plain(project.model_dump()))

def _with_preferences(result: dict) -> dict:
    """Add how the user likes their videos to what a tool says, when anything is kept."""
    kept = runtime.repo.preferences()
    if kept:
        result["preferences"] = [{"id": preference.id, "text": preference.text} for preference in kept]
    return result

@mcp.tool()
def list_projects() -> dict:
    """List all projects.

    Returns:
        A dictionary with a `projects` list, each item holding a project's
        `id`, its `name` if it was given one, and its current `version`; and
        `preferences`, how the user likes their videos, when any are kept.
    """
    return _with_preferences({"projects": [
        {"id": project.id, "name": project.name, "version": project.version}
        for project in runtime.repo.list_projects()
    ]})

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
        length of the edited video in seconds, and `subtitle_count`; and
        `preferences`, how the user likes their videos, when any are kept.

    Raises:
        ValueError: If no project with `project_id` exists.
    """
    project = runtime.repo.get_project(project_id)
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
    return _with_preferences(shared.plain(state))

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
            `add_title_card`, `set_title_card`, `add_text`, `set_text`,
            `remove_text`,
            `set_subtitles`, `edit_subtitle`, or `add_subtitle`. Captions stored
            or corrected here are also remembered against the file they belong
            to, and the next `generate_subtitles` over that file, in any
            project, starts from them — but only those given
            `for_all_projects`; the rest stay in this project. Caption times can be given in the cut's own
            seconds with `timeline_start` and `timeline_end`.

    Returns:
        A dictionary with the `status` and the project's `new_version`, and,
        when captions were stored or corrected, `captions`: each one touched,
        as it now reads, one line apiece, marked `(this project only)` where it
        is — read them back rather than assume — and `captions_kept`, which says
        where the rest are now remembered; and, when the caption style was
        set, `caption_style`: the whole of it as it now stands.

    Raises:
        ValueError: If the project does not exist, the version does not match,
            an operation references a missing or duplicate track or clip, or
            the resulting timeline breaks a rule above.
    """
    before = runtime.repo.get_project(project_id)
    saved = _apply(project_id, expected_version, operations)
    rendering.prepare_playback(saved)
    result = {"status": "success", "new_version": saved.version}
    gap = _music_ends_early(saved)
    if gap:
        result["music"] = gap
    if any(isinstance(op, SetCaptionStyleOp) for op in operations):
        result["caption_style"] = saved.caption_style.model_dump(mode="json")
    touched = _touched_captions(before, saved, operations)
    if touched:
        shared_cues = [cue for cue in touched if not cue.local]
        if shared_cues:
            runtime.repo.remember_captions([cue.model_copy(update={"remembered_in": saved.id}) for cue in shared_cues])
        result["captions"] = [_caption_line(saved, cue) for cue in touched]
        if not shared_cues:
            result["captions_kept"] = "in this project only"
        elif len(shared_cues) < len(touched):
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
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    if project.version != expected_version:
        raise ValueError(f"version conflict: current version is {project.version}, expected version is {expected_version}")

    # A caption names the file its words were spoken in. One naming a file nobody
    # imported would store cleanly and then never appear, which reads as a captioning
    # bug rather than a typo, so it is refused here where the typo still has a name.
    named = {cue.asset_id for op in operations if isinstance(op, SetSubtitlesOp) for cue in op.cues}
    unknown = sorted(named - set(runtime.repo.get_assets(named)))
    if unknown:
        raise ValueError(
            f"caption(s) name assets that are not imported: {', '.join(unknown)}; "
            "import the footage the words were spoken in first"
        )

    # Assets are looked up first so operations that need source lengths, such as fit_track, can use them.
    # Where each song on a track being fitted can loop on the beat: the timeline cannot read analyses.
    for op in operations:
        if isinstance(op, FitTrackOp):
            track = next((item for item in project.tracks if item.id == op.track_id), None)
            for asset_id in {clip.asset_id for clip in (track.clips if track else [])}:
                analysis = runtime.repo.get_analysis(asset_id)
                span = clean_cuts(analysis).loop if analysis is not None else None
                if span is not None:
                    op.loops[asset_id] = (Decimal(str(span[0])), Decimal(str(span[1])))
    # A photo a title card is to stand on is not in the cut until the card puts it there.
    known = runtime.repo.get_assets({clip.asset_id for track in project.tracks for clip in track.clips}
                                    | {op.photo_asset_id for op in operations if getattr(op, "photo_asset_id", None)})
    for op in operations:
        apply_operation(project, op, known)
    if stamp is not None:
        stamp(project)
    assets = shared.referenced_assets(project)
    validate_project(project, assets)
    runtime.renderer.check_supported(project, assets)

    project.version += 1
    runtime.repo.update_project(project, expected_version)
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
    saved = _apply(project_id, expected_version, [SetSubtitlesOp(cues=[*kept, *reused, *local], for_all_projects=True)])
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

@mcp.tool()
def read_cut(project_id: str) -> dict:
    """Read a cut back as a script: every shot in the order it plays, as a viewer meets it.

    The checks find what is wrong with a cut mechanically — a black frame, a
    cut inside a word. Whether the story holds is read, not measured: a line
    that says why somebody did something, landing after what they did, passes
    every check. Read this through before a final render and tell the user the
    story in a few sentences, as it plays, so they can say where it goes wrong.

    Args:
        project_id: The project.

    Returns:
        `script`: one line per shot of the sequence — when it plays, the part
        of the video it is in, the clip and what is on screen (the footage's
        description, or its file), what is said over it with the captions' glossary
        words corrected, the captions shown
        and the words on screen; what plays over it on a track above is under
        it, indented.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    base = project.base_video_track
    if base is None or not base.clips:
        raise ValueError(f"project {project_id} has nothing on its sequence to read")
    assets = shared.referenced_assets(project)
    parts = marked_parts(project)
    placed = place_cues(project, project.subtitles) if project.subtitles else []

    def seen(clip: Clip) -> str:
        if clip.card is not None:
            return f"card 「{clip.card.title}」" + (f" 「{clip.card.subtitle}」" if clip.card.subtitle else "")
        described = [runtime.repo.get_semantic_clip(clip_id) for clip_id in clip.from_clip_ids]
        words = "; ".join(found.description for found in described if found is not None and found.description)
        asset = assets.get(clip.asset_id)
        name = os.path.basename(asset.path) if asset is not None else clip.asset_id
        return f"{name} {float(clip.source_range.start):.1f}–{float(clip.source_range.end):.1f}s" + (
            f": {words}" if words else "")

    def line(clip: Clip) -> str:
        start, end = float(clip.timeline_in), float(clip.timeline_out)
        parts = [f"{clock(start)}–{clock(end)}", f"clip {clip.id}", seen(clip)]
        said = "" if clip.card is not None or clip.volume == 0 else _said_between(
            clip.asset_id, float(clip.source_range.start), float(clip.source_range.end))
        if said:
            parts.append(f"said: 「{said}」")
        # More than a sliver: a caption starting where this shot ends is the next shot's (#80).
        shown = [cue.text for cue in placed if min(float(cue.end), end) - max(float(cue.start), start) > CAPTION_SLIVER]
        if shown:
            parts.append("captions: " + " / ".join(dict.fromkeys(shown)))
        if clip.shown_texts:
            parts.append("on screen: " + " / ".join(text.text for text in clip.shown_texts))
        return " | ".join(parts)

    script: List[str] = []
    part = None
    above = [clip for track in project.video_tracks[1:] for clip in track.clips]
    for clip in sorted(base.clips, key=lambda clip: clip.timeline_in):
        # By its middle, as a cut's parts are read everywhere else: a shot coming in on a
        # transition can start a hair before its part's marker and is still that part's (#80).
        middle = float(clip.timeline_in) + float(clip.timeline_duration) / 2
        now = next((marker.name for opens, _, marker in reversed(parts) if opens <= middle), None)
        if now != part and now is not None:
            script.append(f"== {now}")
            part = now
        script.append(line(clip))
        for over in sorted((other for other in above
                            if other.timeline_in < clip.timeline_out and other.timeline_out > clip.timeline_in
                            and clip.timeline_in <= other.timeline_in),
                           key=lambda other: other.timeline_in):
            script.append("    over it: " + line(over))
    return {"project_id": project_id, "version": project.version, "script": script}

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
    project = runtime.repo.get_project(project_id)
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
    per_shot: Annotated[int, Field(ge=1, le=4)] = 1,
    around: Optional[Annotated[float, Field(ge=0)]] = None,
    part: Optional[str] = None,
) -> ToolResult:
    """Lay out frames of the cut for you to choose a cover or thumbnail from.

    Choosing is yours, not this tool's: the cover has to show what the video
    is about, and nothing measured here knows that. So the listing opens with
    what the plan says the video is for and its parts, and every frame comes
    with what is being said over it. Pick the frame that shows the video's
    point — the thing it is about, the result it leads to — and that reads at
    a glance at the size a feed shows it: one subject, large in the frame,
    sharp, uncluttered, a face with its eyes open if there is one. Then look
    closer: call again with `around` set to that moment to see the frames
    either side of it, and choose the exact one. Offer the user your best one
    or two with the reason for each, and save the one they choose with
    `export_cover`.

    Args:
        project_id: ID of the project.
        frame: Crop the candidates to another shape, as it would be rendered.
        follow_faces: Crop around the face, as the render does.
        per_shot: How many moments to offer from each shot: the one its
            measurements favour and others spread through it.
        around: A moment of the cut, in seconds, to look at closely instead:
            every quarter second for two seconds either side of it.
        part: Only the frames of one part of the cut, by its name or the ID
            of the beat it came from — the payoff, where the point of the
            video usually is, or any part a long cut's sheet had no room for.

    Returns:
        What the video is for, a listing — each candidate's number, time in
        the cut, clip and what is said then — and the candidates as one
        labeled image.

    Raises:
        ValueError: If the project or the part does not exist, or there is
            nothing to offer.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    project = rendering.reshaped(project, frame)
    analyses = rendering.analyses_of(project)
    if around is None:
        offered = [
            (clip, seconds, float(clip.timeline_in) + (seconds - float(clip.source_range.start)) / clip.speed)
            for clip, seconds in cover_candidates(project, analyses, per_shot)
        ]
    else:
        offered = frames_around(project, analyses, around)
    if part is not None:
        parts = marked_parts(project)
        span = next(((opens, closes) for opens, closes, marker in parts
                     if part in (marker.name, marker.id, marker.from_beat_id)), None)
        if span is None:
            raise ValueError(f"no part called {part!r}; the parts are: "
                             f"{', '.join(marker.name for _, _, marker in parts) or 'none'}")
        opens, closes = span
        offered = [item for item in offered if opens <= item[2] < closes]
    if not offered:
        raise ValueError(f"project {project_id} has no picture to take a cover from")
    left_out = max(0, len(offered) - shared.MAX_STORYBOARD_TILES)
    offered = offered[:shared.MAX_STORYBOARD_TILES]
    assets = shared.referenced_assets(project)
    framing = rendering.framing_of(project, assets, follow_faces)
    fps = Fraction(project.fps_num, project.fps_den)
    said = place_cues(project, project.subtitles) if project.subtitles else []
    shots, listing = [], []
    for number, (clip, seconds, at) in enumerate(offered, start=1):
        shots.append((assets[clip.asset_id].path, seconds, f"#{number}  {format_timestamp(at)}",
                      placement_at(framing.get(clip.id), clip, seconds, fps)))
        words = next((cue.text for cue in said if float(cue.start) <= at < float(cue.end)), "")
        listing.append(f"#{number}: edit {at:.3f}s ({format_timestamp(at)}) | clip {clip.id}"
                       + (f" | 「{words}」" if words else ""))
    header = [f"{len(offered)} cover candidate(s) for project {project_id}, {project.width}x{project.height}"
              + (f", around {around:g}s" if around is not None else "") + ":"]
    plan = rendering.compiled_plan(project)
    if plan is not None and (plan.goal or plan.beats):
        parts = " → ".join(beat.name or beat.id for beat in plan.beats)
        header.insert(0, "What the video is for: " + (plan.goal or "(no goal written)")
                      + (f"\nIts parts: {parts}" if parts else ""))
    else:
        header.insert(0, "No plan says what this video is for; ask the user what it is about before choosing.")
    if left_out:
        header.append(f"{left_out} more were left out, from the end of the cut; ask for one `part` at a time.")
    return ToolResult(content=[
        "\n".join([*header, *listing]),
        Image(data=storyboard_sheet(shots, aspect=project.width / project.height), format="jpeg"),
    ])

@mcp.tool()
def export_cover(
    project_id: str,
    seconds: Annotated[float, Field(ge=0)],
    reason: Annotated[str, Field(min_length=1)],
    frame: Optional[Literal["landscape", "portrait", "square"]] = None,
    follow_faces: bool = True,
) -> ToolResult:
    """Save one frame of the cut as a full-size JPEG, for a cover or thumbnail.

    The frame is the one the viewer sees at that moment — covering picture
    where there is some — at the size the cut is rendered at and cropped the
    way the render crops it. It comes back at the size a feed shows it too,
    and for a vertical cut cropped to the 3:4 middle a profile grid shows:
    look at both before telling the user it is done, and choose again if the
    subject is lost at that size or falls outside the middle.

    Args:
        project_id: ID of the project.
        seconds: The moment in the cut, as `propose_covers` lists it.
        reason: Why this frame: what of the video's point it shows, and why
            it reads at a glance. Written for the user, who decides.
        frame: Save it in another shape.
        follow_faces: Crop around the face, as the render does.

    Returns:
        The `output_path` of the JPEG, its `width` and `height`, the reason,
        and the cover at feed size.

    Raises:
        ValueError: If the project does not exist, nothing is on screen then,
            or what is on screen is a title card.
    """
    project = runtime.repo.get_project(project_id)
    if not project:
        raise ValueError(f"project {project_id} not found")
    project = rendering.reshaped(project, frame)
    clip = on_screen(project, seconds)
    if clip is None:
        raise ValueError(f"nothing is on screen at {seconds:g}s of project {project_id}")
    if clip.card is not None:
        raise ValueError(f"{seconds:g}s of project {project_id} is a title card; take the cover from a shot")
    assets = shared.referenced_assets(project)
    source = float(clip.source_range.start) + (seconds - float(clip.timeline_in)) * clip.speed
    framed = rendering.framing_of(project, assets, follow_faces).get(clip.id)
    fps = Fraction(project.fps_num, project.fps_den)
    picture = still(assets[clip.asset_id].path, source, project.width, project.height,
                    placement_at(framed, clip, source, fps))
    folder = str(output_dir())
    os.makedirs(folder, exist_ok=True)
    path = housekeeping.unique_path(folder, rendering.delivery_name(project, "cover" + (f"-{frame}" if frame else ""), ".jpg"))
    picture.save(path, "JPEG", quality=92)
    saved = {"output_path": path, "width": project.width, "height": project.height, "reason": reason}
    grid = ", and the 3:4 middle a profile grid crops a vertical video to" if project.height > project.width else ""
    return ToolResult(
        content=[
            f"Saved {path} ({project.width}x{project.height}). Why: {reason}\n"
            f"Below: the cover at the size a feed shows it{grid}.",
            Image(data=feed_preview(picture), format="jpeg"),
        ],
        structured_content=saved,
    )

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
    project = runtime.repo.get_project(project_id)
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
        # A title card is a frame and words this server draws; another program would read it as
        # footage of the file behind it. It goes across as the gap it fills, and is said to.
        titled = cards.cards_of(project)
        shown = project.model_copy(deep=True)
        if titled:
            base = shown.base_video_track
            base.clips = [clip for clip in base.clips if clip.card is None]
        assets = shared.referenced_assets(shown)
        text, behind = writer(shown, assets, _timecodes(assets))
        if titled:
            behind.append("title cards (they come across as gaps; add them as titles there): "
                          + ", ".join(clip.card.title for clip in titled[:6]) + (" …" if len(titled) > 6 else ""))
        # Decided at render time from the recordings, so the writers, which never see an
        # analysis, cannot say it. Only which tracks, not how loud: nothing is measured.
        if voice_keys(project, rendering.analyses_of(project), lambda clip: None):
            behind.append("the footage's sound ducking under a narration, and the narration brought up to level")
    folder = str(output_dir())
    os.makedirs(folder, exist_ok=True)
    path = housekeeping.unique_path(folder, rendering.delivery_name(project, "timeline", f".{format}"))
    # UTF-8 with a byte-order mark only for the EDL: the programs that read EDLs guess the
    # encoding, and a Chinese file name in a FROM CLIP NAME comes out garbled without it.
    with open(path, "w", encoding="utf-8-sig" if format == "edl" else "utf-8", newline="\n") as handle:
        handle.write(text)
    return {"output_path": path, "left_behind": behind}

# What the user can mark a version as. `exported` is a third mark, never set by hand: a
# version that was rendered out, not as a preview, wears it by itself.
VERSION_MARKS = ("starred", "published")

def _project_json_at(project_id: str, commit: str) -> Optional[dict]:
    """A project as its file held it at one version, or None when it was not there then."""
    path = runtime.history.find("projects/", f"{project_id}.json", commit)
    content = runtime.history.read(path, commit) if path else None
    return json.loads(content) if content else None

def _branches_of(project_id: str) -> List[Project]:
    """The projects started as branches of this one."""
    return [other for other in runtime.repo.list_projects()
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
    marks = runtime.repo.version_marks(project_id)
    delivered = _delivered_renders(project_id)
    there = {job.project_version for job in delivered if os.path.exists(job.output_path)}
    gone = {job.project_version for job in delivered} - there
    found = []
    for version in runtime.history.versions_of("projects/", f"{project_id}.json", limit):
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
    previews = os.path.abspath(housekeeping.preview_dir(runtime.WORKSPACE_DIR, project_id))
    return [job for job in runtime.repo.renders_of(project_id)
            if job.output_path and not os.path.abspath(job.output_path).startswith(previews)]

def _outputs_of(project_id: str) -> List[Job]:
    """The finished videos of a project that are still where they were saved; previews are not among them."""
    return [job for job in _delivered_renders(project_id) if os.path.exists(job.output_path)]

def _project_caches(project_id: str) -> List[str]:
    """The folders of a project's previews and sound checks: made again when wanted."""
    return [housekeeping.preview_dir(runtime.WORKSPACE_DIR, project_id), housekeeping.sound_dir(runtime.WORKSPACE_DIR, project_id)]

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
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    # Only the marked versions are read: the storage page asks this of every video at once.
    kept = {project.version}
    for commit, marks in runtime.repo.version_marks(project_id).items():
        if marks & set(VERSION_MARKS):
            data = _project_json_at(project_id, commit)
            if data:
                kept.add(data.get("version"))
    outputs = _outputs_of(project_id)
    others = [job for job in outputs if job.project_version not in kept]
    size = {job.job_id: housekeeping.size_of(job.output_path) for job in outputs}
    cache = _cache_bytes(project_id)
    return {
        "versions": runtime.history.counts("projects/").get(project_id, 0), "branches": len(_branches_of(project_id)),
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
    project = runtime.repo.get_project(project_id)
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
    busy = runtime.repo.active_job_ids()
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
    runtime.job_manager.forget_abandoned()
    before = {project_id: project for project_id in project_ids if (project := runtime.repo.get_project(project_id))}
    filed = runtime.repo.folder_of("projects")
    named = [project.name for project in before.values() if project.name]
    shared.say_in_history(f"把「{'」「'.join(named[:3])}」移到垃圾桶" + (f"等 {len(named)} 支" if len(named) > 3 else ""))
    outputs = str(output_dir())
    trashed = {
        project_id: housekeeping.put_in_trash(outputs, "project", project_id, project.name or "", {
            "project.json": project.model_dump_json().encode("utf-8"),
            "place.json": json.dumps({"folder_id": filed.get(project_id)}).encode("utf-8"),
        }, [])
        for project_id, project in before.items()
    }
    deleted, busy = runtime.repo.delete_projects(list(before))
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
        shared.say_in_history(f"從垃圾桶放回「{what.name}」")
        project = Project.model_validate_json(kept["project.json"])
        if runtime.repo.get_project(project.id) is not None:
            project = project.model_copy(update={"id": str(uuid.uuid4())})
        runtime.repo.add_project(project)
        project_id = project.id
        folder = json.loads(kept.get("place.json", b"{}")).get("folder_id")
        if folder and any(item["id"] == folder for item in runtime.repo.list_folders("projects")):
            runtime.repo.file_items("projects", [project.id], folder)
    back = housekeeping.take_out_of_trash(outputs, key)
    return {"kind": what.kind, "name": what.name, "project_id": project_id, "files": back}

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
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    branches = _branches_of(project_id)
    starts = {branch.branched_from.commit for branch in branches}
    parent = None
    if project.branched_from is not None:
        source = runtime.repo.get_project(project.branched_from.project_id)
        start = next((version for version in runtime.history.versions_of(
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
    project = runtime.repo.get_project(project_id)
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
        path = runtime.history.find("projects/", f"{project_id}.json", commit)
    except KeyError as exc:
        raise ValueError(str(exc).strip("'\"")) from exc
    content = runtime.history.read(path, commit) if path else None
    if content is None:
        raise ValueError(f"project {project_id} did not exist at version {commit}")
    return Project.model_validate_json(content)

def _plans_at(project: Project, commit: str) -> Dict[str, EditPlan]:
    """The plans a project's clips were compiled from, as they were at one version."""
    found: Dict[str, EditPlan] = {}
    for plan_id in {clip.from_plan_id for track in project.tracks for clip in track.clips if clip.from_plan_id}:
        content = runtime.history.read(f"plans/{plan_id}.json", commit)
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
    stored = runtime.repo.save_plan(plan.model_copy(update={"id": str(uuid.uuid4()), "version": 1}), why)
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
    current = runtime.repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    if current.version != expected_version:
        raise ValueError(f"version conflict: project {project_id} is at version {current.version}, not {expected_version}")
    earlier = _project_at(project_id, commit)
    chosen = next((version for version in runtime.history.versions_of("projects/", f"{project_id}.json", 500)
                   if version.commit.startswith(commit.strip().lower())), None)
    if chosen is not None:
        shared.say_in_history(f"回到「{chosen.subject}」那一版")
    renamed: Dict[str, str] = {}
    copied = False
    others = [other for other in runtime.repo.list_projects() if other.id != project_id]
    for plan_id, plan in _plans_at(earlier, commit).items():
        used_elsewhere = any(clip.from_plan_id == plan_id
                             for other in others for track in other.tracks for clip in track.clips)
        if used_elsewhere:
            renamed[plan_id] = _copied_plan(plan, f"「{current.name or project_id}」回到舊版時另存的計畫")
            copied = True
            continue
        now = runtime.repo.get_plan(plan_id)
        if now is None:
            runtime.repo.save_plan(plan.model_copy(update={"version": 1}), "回到舊版時放回的計畫")
        elif now.model_dump(exclude={"version", "updated_at"}) != plan.model_dump(exclude={"version", "updated_at"}):
            runtime.repo.save_plan(plan.model_copy(update={"version": now.version}), "跟著影片回到舊版")
    restored = _with_plans(earlier, renamed).model_copy(update={
        "id": project_id, "version": current.version + 1, "branched_from": current.branched_from,
    })
    runtime.repo.update_project(restored, current.version)
    rendering.prepare_playback(runtime.repo.get_project(project_id))
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
    current = runtime.repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    if commit is None:
        versions = runtime.history.versions_of("projects/", f"{project_id}.json", 1)
        if not versions:
            raise ValueError(f"project {project_id} has no version in the history yet")
        commit = versions[0].commit
    start = _project_at(project_id, commit)
    found = next((version.commit for version in runtime.history.versions_of("projects/", f"{project_id}.json", 500)
                  if version.commit.startswith(commit.strip().lower())), commit)
    renamed = {plan_id: _copied_plan(plan, f"「{name}」分支的計畫")
               for plan_id, plan in _plans_at(start, commit).items()}
    branch = _with_plans(start, renamed).model_copy(update={
        "id": str(uuid.uuid4()), "name": name.strip(), "version": 1,
        "branched_from": BranchPoint(project_id=project_id, commit=found),
    })
    shared.say_in_history(f"從「{current.name or project_id}」開分支「{branch.name}」")
    runtime.repo.add_project(branch)
    rendering.prepare_playback(branch)
    folder = runtime.repo.folder_of("projects").get(project_id)
    if folder:
        runtime.repo.file_items("projects", [branch.id], folder)
    return {"project_id": branch.id, "name": branch.name, "branched_from": branch.branched_from.model_dump()}

# Provisional (roadmap §13): a word counts as said in a stretch only when this much of it is
# inside, so a word the cut just grazes does not name the stretch.
WORD_INSIDE_SECONDS = 0.05
# Provisional (roadmap §13): a comment dragged over less than this is a moment, not a stretch.
SHORTEST_STRETCH_SECONDS = 0.05

def _said_between(asset_id: str, start: float, end: float) -> str:
    """What the transcript has said in a file between two of its seconds, glossary words corrected; empty when nothing was."""
    analysis = runtime.repo.get_analysis(asset_id)
    if analysis is None or analysis.transcript is None:
        return ""
    return shared.glossed("".join(
        word.text for segment in analysis.transcript.segments if not was_made_up(segment) for word in segment.words
        if word.start < end - WORD_INSIDE_SECONDS and word.end > start + WORD_INSIDE_SECONDS
    ).strip())

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
    current = runtime.repo.get_project(project_id)
    if current is None:
        raise ValueError(f"project {project_id} not found")
    versions = runtime.history.versions_of("projects/", f"{project_id}.json", 500)
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
    names = {asset_id: os.path.basename(asset.path) for asset_id, asset in runtime.repo.get_assets(
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
        return _plans_at(project, commit).get(plan_id) if commit else runtime.repo.get_plan(plan_id)

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
    stored = runtime.repo.get_plan(right_id)
    used_elsewhere = any(clip.from_plan_id == right_id for other in runtime.repo.list_projects() if other.id != merged.id
                         for track in other.tracks for clip in track.clips)
    if used_elsewhere or stored is None:
        plan_id = _copied_plan(combined, f"「{name}」合成兩版時另存的計畫")
        said = "剪輯計畫也合併了；原本的計畫有別支影片在用，所以另存了一份" if used_elsewhere else "剪輯計畫也依段落合併了"
    else:
        # Saved on top of the plan as it is now, whichever version the right side was.
        plan_id = runtime.repo.save_plan(combined.model_copy(update={"version": stored.version}), "合成兩版").id
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
    current = runtime.repo.get_project(project_id)
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
    shared.say_in_history(f"合成兩版：{'、'.join(taken) or '全部'}用「{a.get('what')}」" if taken else "合成兩版")
    result = merged.model_copy(update={"id": project_id, "version": current.version + 1,
                                       "name": current.name, "branched_from": current.branched_from})
    validate_project(result, runtime.repo.get_assets({clip.asset_id for track in result.tracks for clip in track.clips}))
    runtime.repo.update_project(result, current.version)
    rendering.prepare_playback(runtime.repo.get_project(project_id))
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
    if runtime.repo.get_project(project_id) is None:
        raise ValueError(f"project {project_id} not found")
    chosen = next((version for version in runtime.history.versions_of("projects/", f"{project_id}.json", 500)
                   if version.commit.startswith(commit.strip().lower())), None)
    if chosen is None:
        raise ValueError(f"project {project_id} has no version {commit}")
    runtime.repo.set_version_mark(project_id, chosen.commit, mark, on)
    marks = next((marked for version, _, marked in _project_versions(project_id, 500)
                  if version.commit == chosen.commit), set(runtime.repo.version_marks(project_id).get(chosen.commit, ())))
    return {"commit": chosen.commit[:10], "marks": sorted(marks)}

# ------------------------------------------------------------------ what the user said about a moment

def _current_commit(project_id: str) -> Optional[str]:
    """The version a project is at now, as its latest commit; None before it has one."""
    found = runtime.history.versions_of("projects/", f"{project_id}.json", 1)
    return found[0].commit if found else None

def _comment_said(project: Project, comment: Comment) -> dict:
    """A comment as the AI and the editor read it: where it is on the cut now, and what was said back."""
    now = where_now(project, comment)
    asset = runtime.repo.get_assets([comment.start_anchor.asset_id]).get(comment.start_anchor.asset_id) \
        if comment.start_anchor else None
    return {
        "id": comment.id, "text": comment.text, "by": comment.by, "status": comment.status,
        **({"question": True, "answered": comment.answered}
           if comment.question else {}),
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

    A question the AI asked waits on the user until they answer it, so it is
    not counted before then.

    Args:
        project_id: The video.

    Returns:
        The number of open comments.
    """
    return sum(comment.waiting for comment in runtime.repo.comments(project_id))

# How far either side of the problem a judge sees, for whether the change broke what is
# around it. Provisional.
TRIAL_MARGIN_SECONDS = 2.0

def _render_length(path: str) -> Optional[float]:
    """How long a rendered file runs, or None when it cannot be read."""
    try:
        return float(probe_file(path)["format"]["duration"])
    except (OSError, RuntimeError, KeyError, ValueError):
        return None

def _trial_dir(trial_id: str) -> str:
    """Where a trial keeps its own copies of the two renders."""
    return os.path.join(runtime.WORKSPACE_DIR, "trials", trial_id)

@mcp.tool()
def start_trial(
    project_id: str,
    before_job: str,
    problem: Annotated[str, Field(min_length=1)],
    hypothesis: Annotated[str, Field(min_length=1)],
    expected: Annotated[str, Field(min_length=1)],
    at: Annotated[float, Field(ge=0)],
    end: Annotated[float, Field(gt=0)],
    evidence: str = "",
) -> dict:
    """Put a change you are about to make to a blind test, before you make it.

    Whoever makes a change reads the result through their own reason for it,
    and says it worked. So when you change a cut to fix something — your own
    check found it, or the user's feedback — write the problem down first:
    where it is, what shows it, what you will change and what should be better.
    Then change the cut, render a preview and give it with `trial_candidate`;
    the two are handed over blind as X and Y with `trial_pack`; and only the
    verdict in `judge_trial` says whether the change is kept.

    Args:
        project_id: The video.
        before_job: A finished render of the cut before the change. A copy is
            kept, since a newer preview replaces it.
        problem: What is wrong and where, as it was seen or heard.
        hypothesis: What you will change and why it should help. Never shown
            to whoever judges.
        expected: What should be better afterwards. Never shown either.
        at: Where the problem starts in that render, in seconds.
        end: Where it ends.
        evidence: What shows it — a join's words and levels from
            `inspect_join`, a comment, a finding.

    Returns:
        The trial's `trial_id`, and what to do next.

    Raises:
        ValueError: If the render is not a finished one of this project, or
            the stretch is outside it.
    """
    job, path = rendering.finished_render(before_job)
    if job.project_id != project_id:
        raise ValueError(f"render {before_job} is not of project {project_id}")
    length = _render_length(path)
    if end <= at or (length is not None and end > length + 0.05):
        raise ValueError(f"the stretch {at:g}–{end:g}s is not in the {length or 0:.1f}s render")
    trial = Trial(id="t" + uuid.uuid4().hex[:7], project_id=project_id, problem=problem.strip(),
                  evidence=evidence.strip(), hypothesis=hypothesis.strip(), expected=expected.strip(),
                  at=at, end=end, before_job=before_job, before_version=job.project_version)
    os.makedirs(_trial_dir(trial.id), exist_ok=True)
    shutil.copyfile(path, os.path.join(_trial_dir(trial.id), "before.mp4"))
    runtime.repo.save_trial(trial)
    return {"trial_id": trial.id, "next": (
        "make the change, render a preview of it, and give it with `trial_candidate`")}

@mcp.tool()
def trial_candidate(
    trial_id: str,
    after_job: str,
    candidate_at: Optional[Annotated[float, Field(ge=0)]] = None,
    candidate_end: Optional[Annotated[float, Field(gt=0)]] = None,
) -> dict:
    """Give a trial the render of the cut after the change.

    Args:
        trial_id: The trial, from `start_trial`.
        after_job: A finished render of the cut after the change.
        candidate_at: Where the same stretch starts in it, when the change
            moved it; the same seconds as before when not given.
        candidate_end: Where it ends.

    Returns:
        What to do next: who judges it.

    Raises:
        ValueError: If the trial does not exist or has been judged, or the
            render is not a finished one of the same project after the first.
    """
    trial = runtime.repo.trial(trial_id)
    if trial is None:
        raise ValueError(f"there is no trial {trial_id}")
    if trial.verdict is not None:
        raise ValueError(f"trial {trial_id} has been judged; start another")
    job, path = rendering.finished_render(after_job)
    if (job.project_id != trial.project_id or after_job == trial.before_job
            or (job.project_version or 0) <= (trial.before_version or 0)):
        raise ValueError(f"render {after_job} is not a render of project {trial.project_id} made after the change; "
                         f"the cut before was version {trial.before_version}")
    shutil.copyfile(path, os.path.join(_trial_dir(trial.id), "after.mp4"))
    trial.after_job, trial.after_version = after_job, job.project_version
    trial.candidate_at = trial.at if candidate_at is None else candidate_at
    trial.candidate_end = trial.end if candidate_end is None else candidate_end
    trial.candidate_is = secrets.choice(["X", "Y"])
    runtime.repo.save_trial(trial)
    return {"trial_id": trial.id, "next": (
        "have it judged by someone who has not seen the change: start a fresh subagent with no part in this "
        "conversation, give it only `trial_pack` and ask which of X and Y handles the problem better and which, "
        "if either, breaks what is around it; or show the pack to the user. Then record it with `judge_trial`. "
        "Judging it yourself is not blind — you know which you made — and the record says so")}

@mcp.tool()
def trial_pack(trial_id: str) -> ToolResult:
    """Hand over the two versions of a trial blind, as X and Y, for someone to judge.

    It holds the problem and its evidence, and for each of X and Y the frames
    of the stretch and a little either side, and its sound to hear — nothing
    that says which is the change or why it was made. Give it to whoever
    judges, as it is.

    Args:
        trial_id: The trial.

    Returns:
        The problem and evidence, where each sound file is, and the frames of
        X above Y.

    Raises:
        ValueError: If the trial does not exist or has no candidate yet.
    """
    trial = runtime.repo.trial(trial_id)
    if trial is None:
        raise ValueError(f"there is no trial {trial_id}")
    if trial.candidate_is is None:
        raise ValueError(f"trial {trial_id} has no change to compare yet; `trial_candidate` gives it one")
    folder = _trial_dir(trial.id)
    files = {"before": (os.path.join(folder, "before.mp4"), trial.at, trial.end),
             "after": (os.path.join(folder, "after.mp4"), trial.candidate_at, trial.candidate_end)}
    labels = {trial.candidate_is: "after", ("Y" if trial.candidate_is == "X" else "X"): "before"}
    per_side = shared.MAX_STORYBOARD_TILES // 2
    shots, lines = [], [f"The problem: {trial.problem}"]
    if trial.evidence:
        lines.append(f"What shows it: {trial.evidence}")
    lines.append(f"Below, X above Y: the same stretch in two versions of the cut, with {TRIAL_MARGIN_SECONDS:g}s "
                 "either side. Say which handles the problem better, or that neither does, and which, if either, "
                 "makes what is around it worse.")
    for label in ("X", "Y"):
        path, start, finish = files[labels[label]]
        length = _render_length(path) or finish
        opened, closed = max(0.0, start - TRIAL_MARGIN_SECONDS), min(length, finish + TRIAL_MARGIN_SECONDS)
        step = (closed - opened) / (per_side - 1)
        for index in range(per_side):
            moment = round(min(opened + index * step, length - 0.05), 3)
            shots.append((path, moment, f"{label} {moment - start:+.1f}s"))
        heard = os.path.join(folder, f"{label}.wav")
        try:
            cut_sound(path, opened, closed, heard)
            lines.append(f"{label} sound: {heard}")
        except RuntimeError:
            lines.append(f"{label} has no sound here")
    return ToolResult(content=["\n".join(lines), Image(data=storyboard_sheet(shots, columns=per_side // 2),
                                                         format="jpeg")])

@mcp.tool()
def judge_trial(
    trial_id: str,
    better: Literal["X", "Y", "same"],
    broke: Literal["neither", "X", "Y", "both"],
    why: Annotated[str, Field(min_length=1)],
    judged_by: Literal["subagent", "user", "same_ai"],
) -> dict:
    """Write down the verdict on a trial, and only then hear which was the change.

    Args:
        trial_id: The trial.
        better: Which handles the problem better, as whoever judged said; `same`
            when neither does.
        broke: Which, if either, they said makes what is around the problem
            worse. Only the change breaking something counts against it.
        why: What they saw and heard, in their words.
        judged_by: `subagent` for a fresh one that saw none of the change,
            `user`, or `same_ai` when you judged it yourself — which is not
            blind, and the record says so.

    Returns:
        `change_was`, the change's label; `keep`, `change` only when it made
        the problem better and broke nothing, otherwise `original` — then put
        the cut back with `restore_version` to the version before; and the
        trial as recorded.

    Raises:
        ValueError: If the trial does not exist, has no change yet, or was
            judged already.
    """
    trial = runtime.repo.trial(trial_id)
    if trial is None:
        raise ValueError(f"there is no trial {trial_id}")
    if trial.candidate_is is None:
        raise ValueError(f"trial {trial_id} has no change to judge yet")
    if trial.verdict is not None:
        raise ValueError(f"trial {trial_id} was judged already: {trial.verdict.better}")
    trial.verdict = Verdict(better=better, broke=broke, why=why.strip(), judged_by=judged_by)
    runtime.repo.save_trial(trial)
    shutil.rmtree(_trial_dir(trial.id), ignore_errors=True)
    keep = "change" if better == trial.candidate_is and broke not in (trial.candidate_is, "both") else "original"
    said = {
        "change_was": trial.candidate_is, "keep": keep, "blind": judged_by != "same_ai",
        "trial": _trial_said(trial),
    }
    if keep == "original":
        said["next"] = (f"the change is not kept: put the cut back as it was in version {trial.before_version} with "
                        "`project_history` and `restore_version`, and tell the user what was tried and why it was not "
                        "kept")
    return said

def _trial_said(trial: Trial) -> dict:
    """A trial as it is read back."""
    said = {"trial_id": trial.id, "problem": trial.problem, "hypothesis": trial.hypothesis,
            "expected": trial.expected, "before_version": trial.before_version, "after_version": trial.after_version}
    if trial.verdict is not None:
        said["verdict"] = {"better": trial.verdict.better, "change_was": trial.candidate_is,
                           "broke": trial.verdict.broke, "why": trial.verdict.why,
                           "judged_by": trial.verdict.judged_by}
    return said

@mcp.tool()
def get_trials(project_id: str) -> dict:
    """Read back the changes to a video that were tried blind, and what came of each.

    Args:
        project_id: The video.

    Returns:
        `trials`, oldest first: the problem, what was changed and expected,
        and the verdict with who gave it.
    """
    return {"trials": [_trial_said(trial) for trial in runtime.repo.trials(project_id)]}

@mcp.tool()
def ask_user(project_id: str, questions: Annotated[List[Question], Field(min_length=1, max_length=8)]) -> dict:
    """Put what you are not sure of about a cut to the user, each on the moment it is about.

    The checks say whether something is wrong; none says whether it is good —
    whether four seconds of opening hook anybody, whether a 57-second test
    loses them. Do not settle those by yourself: after a preview, ask about
    the few you weighed and could not decide, with why. Each shows in the
    editor as a question on its moment of the cut, and the answer comes back
    as a reply to it in `get_comments`. When an answer says how the user
    likes their videos in general, not only this one, `remember_preference`.

    Args:
        project_id: The video.
        questions: Each with where it is (`start`, and `end` for a stretch),
            the `question` in the user's language and `why` it is in doubt.

    Returns:
        The questions as `get_comments` shows them.

    Raises:
        ValueError: If the project does not exist.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    asked = []
    length = float(project.duration)
    for question in questions:
        if question.start > length or (question.end is not None and not (
                question.start + SHORTEST_STRETCH_SECONDS <= question.end <= length + 0.001)):
            raise ValueError(f"「{question.question[:20]}」 is asked at {question.start:g}"
                             + (f"–{question.end:g}" if question.end is not None else "")
                             + f"s, which is not a moment or a stretch of the {length:.1f}s cut; nothing was asked")
    for question in questions:
        end = question.end
        comment = Comment(
            id="c" + uuid.uuid4().hex[:7], project_id=project_id, by=f"AI（{runtime.calling_client.get()}）",
            text=question.question.strip() + (f"\n（{question.why.strip()}）" if question.why.strip() else ""),
            commit=_current_commit(project_id), question=True,
            start=round(question.start, 3), end=round(end, 3) if end is not None else None,
            start_anchor=anchor_at(project, question.start),
            end_anchor=anchor_at(project, end) if end is not None else None,
        )
        runtime.repo.save_comment(comment, f"AI 問：{question.question[:30]}")
        asked.append(_comment_said(project, comment))
    return {"asked": asked}

@mcp.tool()
def remember_preference(text: Annotated[str, Field(min_length=1, max_length=300)],
                        project_id: Optional[str] = None) -> dict:
    """Keep how the user likes their videos, for every project after this one.

    Only what the user said, in their words — 「開場不要超過三秒」, 「不要用
    轉場」 — and only when it is about their videos in general rather than
    this cut. What is kept comes back with `list_projects`, `create_project`
    and `get_project`, so the next video starts from it. Read it back to them
    when you keep it.

    Args:
        text: The preference, as they said it.
        project_id: The video it was said about, if one.

    Returns:
        `kept`, the preference with its `id`; and `preferences`, all of them.
    """
    preference = Preference(id="p" + uuid.uuid4().hex[:7], text=text.strip(), project_id=project_id)
    runtime.repo.save_preference(preference)
    return _with_preferences({"kept": {"id": preference.id, "text": preference.text}})

@mcp.tool()
def forget_preference(preference_id: str) -> dict:
    """Stop keeping a preference, when the user says it no longer holds.

    Args:
        preference_id: Its ID, from `preferences` in `list_projects`.

    Returns:
        `forgotten`, its ID; and `preferences`, what is still kept.

    Raises:
        ValueError: If no preference has that ID.
    """
    if not runtime.repo.delete_preference(preference_id):
        raise ValueError(f"there is no preference {preference_id}")
    return _with_preferences({"forgotten": preference_id})

# More songs than this to hear under the same stretch is a library, not a choice. Provisional.
MOST_OFFERED_SONGS = 4

@mcp.tool()
def offer_songs(
    project_id: str,
    songs: Annotated[List[OfferedSong], Field(min_length=1, max_length=MOST_OFFERED_SONGS)],
    beat_id: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> dict:
    """Put a few songs to the user to hear under the same stretch of the cut, and let them pick.

    You cannot hear whether a song suits the pictures; its tags are a guess at
    its mood, about half the time right. Narrow the music down to two or three
    with `find_music`, then offer them here: the editor plays the stretch with
    each in turn in place of its music — at the level and fades of the plan's
    cue there, or else as loud as the music there is — and the user picks by
    ear. Their pick comes back as a comment on the stretch: which song, from
    which second. When the stretch is a music cue of the plan the cut was
    compiled from (`beat_id`, or the start of the cut), the pick is already
    written into that cue, and the comment says so — compile the plan to hear
    it; otherwise set it with `fit_track`. A new offer replaces the last.

    Args:
        project_id: The video.
        songs: The songs, each with the second to hear it from and, in
            `why`, what it is meant to do there.
        beat_id: The part of the plan to hear them under: from where it
            begins to where the next begins.
        start: Otherwise, where the stretch begins in seconds of the cut; the
            start when neither is given.
        end: Where it ends; the end of the cut when not given.

    Returns:
        `offered`, the songs by name, and what to tell the user.

    Raises:
        ValueError: If the project, a song or the part does not exist, a song
            has no sound or is shorter than the second it is heard from, or
            the stretch is empty.
    """
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    length = float(project.duration)
    if beat_id is not None and (start is not None or end is not None):
        raise ValueError("give the part as `beat_id`, or the stretch as `start` and `end`, not both")
    if beat_id is not None:
        parts = marked_parts(project)
        span = next(((opens, closes) for opens, closes, marker in parts if marker.from_beat_id == beat_id), None)
        if span is None:
            raise ValueError(f"the cut has no part {beat_id}; its parts: "
                             + (", ".join(marker.from_beat_id or marker.name for _, _, marker in parts) or "none"))
        start, end = span
    start = 0.0 if start is None else start
    end = length if end is None else end
    if not 0 <= start < end <= length + 0.001:
        raise ValueError(f"the stretch {start:g}–{end:g}s is not inside the {length:.1f}s cut")
    names = []
    for song in songs:
        asset = runtime.repo.get_assets([shared.asset_id_for(song.asset_id)]).get(shared.asset_id_for(song.asset_id))
        if asset is None:
            raise ValueError(f"there is no song {song.asset_id}")
        if not asset.has_audio:
            raise ValueError(f"{os.path.basename(asset.path)} has no sound to hear")
        if asset.duration is not None and song.start >= float(asset.duration):
            raise ValueError(f"{os.path.basename(asset.path)} runs {float(asset.duration):.1f}s, "
                             f"so it cannot be heard from {song.start:g}s")
        song.asset_id = asset.id
        names.append(os.path.basename(asset.path))
    offer = SongOffer(project_id=project_id, start=round(start, 3), end=round(end, 3), songs=songs, beat_id=beat_id)
    # Heard as the plan's cue there would play it; else as loud as the music the cut has there.
    found = _offered_cue(project, offer)
    if found is not None:
        cue = found[1]
        offer = offer.model_copy(update={"volume": cue.volume, "fade_in": cue.fade_in, "fade_out": cue.fade_out})
    else:
        playing = [clip for track in project.tracks if track.track_type == TrackType.AUDIO
                   and track.duck_under_speech for clip in track.clips
                   if float(clip.timeline_in) < end and float(clip.timeline_out) > start]
        if playing:
            offer = offer.model_copy(update={"volume": playing[0].volume})
    runtime.repo.save_song_offer(offer, project_id)
    return {"offered": names, "say": (
        f"tell the user {len(names)} songs are waiting in the editor to hear under {clock(start)}–{clock(end)}; "
        "their pick comes back as a comment")}

def song_trial(project: Project, offer: SongOffer, number: int) -> Project:
    """The cut with one offered song in place of its music over the stretch, for the editor to play.

    Args:
        project: The cut.
        offer: The songs offered.
        number: Which of them, from 1.

    Returns:
        A copy: every music clip that plays in the stretch taken off, and the
        song laid over the stretch from its second, faded in and out.
    """
    song = offer.songs[number - 1]
    asset = runtime.repo.get_assets([song.asset_id]).get(song.asset_id)
    trial = project.model_copy(deep=True)
    music = [track for track in trial.tracks if track.track_type == TrackType.AUDIO and track.duck_under_speech]
    for track in music:
        track.clips = [clip for clip in track.clips
                       if float(clip.timeline_out) <= offer.start or float(clip.timeline_in) >= offer.end]
    if not music:
        music = [Track(id="music", track_type=TrackType.AUDIO, duck_under_speech=True)]
        trial.tracks.append(music[0])
    heard = offer.end - offer.start
    if asset is not None and asset.duration is not None:
        heard = min(heard, float(asset.duration) - song.start)
    music[0].clips.append(Clip(
        id="trial", asset_id=song.asset_id, timeline_in=Decimal(str(offer.start)),
        source_range=TimeRange(start=Decimal(str(song.start)), end=Decimal(str(round(song.start + heard, 3)))),
        volume=offer.volume, audio_fade_in=Decimal(str(offer.fade_in)), audio_fade_out=Decimal(str(offer.fade_out)),
    ))
    music[0].clips.sort(key=lambda clip: clip.timeline_in)
    return trial

def _offered_cue(project: Project, offer: SongOffer) -> Optional[Tuple[EditPlan, MusicCue]]:
    """The music cue of the plan the cut was compiled from that the offered stretch is, if it is one."""
    plan = rendering.compiled_plan(project)
    if plan is None or plan.music is None or (offer.beat_id is None and offer.start > 0):
        return None
    cue = next((cue for cue in plan.music.cues if cue.beat_id == offer.beat_id), None)
    return (plan, cue) if cue is not None else None

def pick_song(project_id: str, number: int) -> dict:
    """Take the user's pick of the offered songs to the AI, as a comment on the stretch.

    Where the stretch is a music cue of the plan the cut was compiled from,
    the pick is written into that cue too, so compiling the plan again plays
    it; the comment says so.

    The editor's, never the AI's: the pick is the user speaking.

    Args:
        project_id: The video.
        number: Which song, from 1.

    Returns:
        The comment.

    Raises:
        ValueError: If nothing is offered, or there is no such song.
    """
    offer = runtime.repo.song_offer(project_id)
    if offer is None or not 1 <= number <= len(offer.songs):
        raise ValueError("no such song is offered")
    song = offer.songs[number - 1]
    asset = runtime.repo.get_assets([song.asset_id]).get(song.asset_id)
    name = os.path.basename(asset.path) if asset is not None else song.asset_id
    said = f"配樂選「{name}」，從歌的 {song.start:g} 秒開始（{song.asset_id}）"
    found = _offered_cue(runtime.repo.get_project(project_id), offer)
    if found is not None:
        plan, cue = found
        amended = amend_plan(plan.id, plan.version, [SetMusicCueOp(beat_id=cue.beat_id, asset_id=song.asset_id,
                                                                     start=song.start)], note=said)
        said += f"；已寫進計畫 {plan.id} 第 {amended['version']} 版，重新 compile_plan 後生效"
    made = add_comment(project_id, said, offer.start, offer.end)
    runtime.repo.save_song_offer(None, project_id)
    return made

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
    project = runtime.repo.get_project(project_id)
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
    runtime.repo.save_comment(comment, f"留言：{comment.text[:30]}")
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
    project = runtime.repo.get_project(project_id)
    comment = next((item for item in runtime.repo.comments(project_id) if item.id == comment_id), None)
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
    runtime.repo.save_comment(comment, f"{said}：{comment.text[:30]}")
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
    ways — then close it with `resolve_comment`, saying what you did. One
    marked `question` is one you asked with `ask_user`: it waits on the user
    until it is `answered`, in its replies.

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
    project = runtime.repo.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id} not found")
    every = runtime.repo.comments(project_id)
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
    return change_comment(project_id, comment_id, f"AI（{runtime.calling_client.get()}）", reply,
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
        `error_message`, for a finished analysis worth a second look a
        `note` to pass on, and for a finished render whose every frame was
        found black, frozen or a flash somewhere, `frames`: `found`, what the
        cut did not ask for — look at them with `view_render` `at: "spots"` —
        and `meant`, inside a fade, a dip, a card or a photo; or `not_read`
        when the check could not read it, which says nothing about its frames; and a summary
        over all of them: `finished` and
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
    ahead = queue_positions(runtime.repo.jobs_to_show(now)[0])
    for job_id in dict.fromkeys(job_ids):
        job = runtime.job_manager.get_job(job_id)
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
        if job.kind == JobKind.RENDER and job.status == JobStatus.COMPLETED:
            if job.spots is None:
                entry["frames"] = {"not_read": "every frame was not read — the check failed or was stopped — so "
                                               "black, frozen or flashing frames in it are not known"}
            elif job.spots:
                entry["frames"] = _frame_spots(job)
        if job.kind == JobKind.ANALYZE and job.status == JobStatus.COMPLETED and job.asset_id:
            heard = runtime.repo.get_analysis(job.asset_id)
            voices = len({turn.speaker for turn in heard.speakers}) if heard else 0
            if voices > MANY_VOICES:
                entry["note"] = (
                    f"heard {voices} different voices, more than a video usually has — noisy talk splits one "
                    "person into many. If you know how many people talk, ask the user and analyze it again "
                    "with `speakers` and `again`"
                )
        listed.append(entry)
    return listed

def _frame_spots(job: Job) -> dict:
    """Say where every frame of a render was found black, frozen or a flash, and which of it the cut meant.

    Black inside a fade or a dip through black, and a card or a photo held
    still, are what the cut asked for; the rest is what nobody asked for — two
    pieces that do not quite meet, a frame of the wrong shot — and is said
    first, as `found`.

    Args:
        job: A finished render.

    Returns:
        `found` and `meant`, a line each: when, what, and the clips there.
    """
    project, _ = _project_as_rendered(job)
    clips = sorted(project.base_video_track.clips, key=lambda clip: clip.timeline_in) if (
        project is not None and project.base_video_track is not None) else []
    assets = shared.referenced_assets(project) if project is not None else {}
    slack = 1 / (project.fps_num / project.fps_den) if project is not None else 0.04

    def meant(kind: str, start: float, end: float, here: List[Clip]) -> Optional[str]:
        for clip in here:
            begin, finish = float(clip.timeline_in), float(clip.timeline_out)
            if clip.card is not None:
                return f"card {clip.id}"
            asset = assets.get(clip.asset_id)
            if kind == "frozen" and asset is not None and stills.is_image(asset.path):
                return f"photo {clip.id}"
            if kind == "black" and clip.video_fade_in and start >= begin - slack and end <= begin + float(
                    clip.video_fade_in) + slack:
                return f"fade in of {clip.id}"
            if kind == "black" and clip.video_fade_out and end <= finish + slack and start >= finish - float(
                    clip.video_fade_out) - slack:
                return f"fade out of {clip.id}"
        # A dip ends on the cut, so its black lies wholly inside the clip before the one it leads
        # into; every clip's dip is looked at, not only the clips the black falls in.
        for clip in clips:
            begin, dip = float(clip.timeline_in), clip.transition_in
            if (kind in ("black", "flash") and dip is not None and dip.kind == "dip"
                    and start >= begin - float(dip.seconds) - slack and end <= begin + slack):
                part = next((marker.name for marker in project.markers
                             if abs(float(marker.timeline_in) - begin) <= slack), None) if project is not None else None
                return f"dip into {clip.id}" + (f", where 「{part}」 begins" if part else "")
        return None

    found, kept = [], []
    for spot in job.spots:
        start, end = float(spot["start"]), float(spot["end"])
        here = [clip for clip in clips if float(clip.timeline_in) < max(end, start + slack)
                and float(clip.timeline_out) > start]
        where = " → ".join(f"clip {clip.id}" for clip in here) or "past the cut"
        line = f"{start:.2f}–{end:.2f}s {spot['kind']} ({end - start:.2f}s) in {where}"
        reason = meant(str(spot["kind"]), start, end, here)
        (kept if reason else found).append(line + (f": the {reason}" if reason else ""))
    return {"found": found, "meant": kept}

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
    job = runtime.job_manager.cancel_job(job_id)
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
    items = housekeeping.usage(runtime.WORKSPACE_DIR, str(output_dir()))
    videos = []
    for project in runtime.repo.list_projects():
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
    runtime.job_manager.forget_abandoned()
    freed = housekeeping.clean(runtime.WORKSPACE_DIR, kinds, runtime.repo.active_job_ids())
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
    planned = housekeeping.legacy_plan(runtime.WORKSPACE_DIR, str(output_dir()))
    if not confirm:
        recycled = [item for item in planned if item.duplicate]
        stays = [item for item in planned if not item.keep_as and not item.duplicate]
        return {
            "keep": [{"from": item.path, "to": item.keep_as, "megabytes": round(item.bytes / 1e6, 1)}
                     for item in planned if item.keep_as],
            "recycle": {"files": len(recycled), "megabytes": round(sum(item.bytes for item in recycled) / 1e6, 1)},
            "left_in_place": {"files": len(stays), "megabytes": round(sum(item.bytes for item in stays) / 1e6, 1)},
        }
    return housekeeping.legacy_apply(runtime.WORKSPACE_DIR, planned)

# How often a running server moves the job queue along. Provisional.
TEND_JOBS_SECONDS = 15.0

def tend_jobs() -> None:
    """Keep the job queue moving while the server runs.

    A worker starts the next queued job when it finishes, but one killed
    halfway never does, and what was queued behind it waited for some later
    call to look (#88). Run at start and then every `TEND_JOBS_SECONDS`: a
    job whose worker is gone is failed once it has been quiet long enough,
    and the queue behind it starts.
    """
    while True:
        try:
            runtime.job_manager.launch_ready()
        except Exception:
            # A failed look is tried again next time; it is never the end of the server.
            pass
        time.sleep(TEND_JOBS_SECONDS)

def main():
    """Run the MCP server over the stdio transport."""
    threading.Thread(target=tend_library, name="tend-library", daemon=True).start()
    threading.Thread(target=tend_jobs, name="tend-jobs", daemon=True).start()
    # The banner goes to stderr, which every client keeps as its server log; a box of
    # ASCII art in it on every start is noise, and on a console that is not UTF-8 it
    # arrives mangled.
    mcp.run(show_banner=False)

if __name__ == "__main__":
    main()
