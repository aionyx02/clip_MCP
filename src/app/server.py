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
from typing import Annotated, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent
from pydantic import BaseModel, Field, TypeAdapter
from app.models.media import Asset, Source
from app.models.plan import (
    EditPlan, MusicCue, PlanAmendment, SetMusicCueOp, Trim, TrimKind, apply_amendment, describe_amendment,
    short_clip_ids, with_whole_clip_ids,
)
from app.models.semantic import SemanticClip
from app.models.timeline import Clip, EditOperation, Project, TimeRange, Track, TrackType
from app.models.comment import Comment, Preference, Question, Reply
from app.models.offer import TRIAL_VOLUME, OfferedSong, SongOffer
from app.models.trial import Trial, Verdict
from app.models.job import Job, JobKind, JobStatus
from app.engine import cards, loudness, stills
from app.engine.plan import (
    BROLL_TRACK_ID, COMPILED_TRACKS, MUSIC_TRACK_ID, VIDEO_TRACK_ID, as_left, broll_covers, broll_slots, check_plan,
    check_recompile, compile_operations, compiled_duration, diff_plans, music_beds, Note, note_record, piece_changes,
    plan_pieces,
)
from app.engine.semantic import clean_cuts
from app.engine.ffmpeg import graph_from_file, hidden_window_flags
from app.engine.probe import picture_size, probe_file, recorded_at, rights_tags, timecode_start
from app.engine.reframe import placement_at
from app.engine.library import FOLDER_KINDS, FOOTAGE, LIBRARIES, MUSIC, classify, library_of
from app.engine import clap
from app.engine import fetch
from app.engine.builder import DEFAULT_LOUDNESS_TARGET, voice_keys
from app.engine.comments import anchor_at, where_now
from app.engine.delivery import (
    Finding, chapters, clock, cover_candidates, frames_around, marked_parts, on_screen, rights_facts,
)
from app.engine.joins import HeardWord, Join, joins_in, pause_across, words_heard
from app.engine.frames import feed_preview, format_timestamp, still, storyboard_sheet
from app.engine.interchange import write_edl, write_fcpxml, write_otio, write_srt
from app.engine.listen import WINDOW_SECONDS, chart, cut_sound, describe, levels, levels_every, parts_of
from app.engine.subtitles import place_cues
from app.storage import housekeeping
from app.workspace import output_dir
from app import runtime
# By name, unlike everything else from runtime: every tool is registered with it, and it is never swapped out.
from app.runtime import mcp
# Importing a tool module is what registers its tools, so every one is imported here.
from app.tools import editing, examining, rendering, shared, subtitles, timelines, upkeep, versioning  # noqa: F401

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
        data = versioning.project_json_at(project.id, version.commit)
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
        said = "" if clip.card is not None or clip.volume == 0 else versioning.said_between(
            clip.asset_id, float(clip.source_range.start), float(clip.source_range.end))
        if said:
            parts.append(f"said: 「{said}」")
        # More than a sliver: a caption starting where this shot ends is the next shot's (#80).
        shown = [cue.text for cue in placed if min(float(cue.end), end) - max(float(cue.start), start) > subtitles.CAPTION_SLIVER]
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

# Provisional (roadmap §13): a comment dragged over less than this is a moment, not a stretch.
SHORTEST_STRETCH_SECONDS = 0.05

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
    return editing.with_preferences({"kept": {"id": preference.id, "text": preference.text}})

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
    return editing.with_preferences({"forgotten": preference_id})

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
