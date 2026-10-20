"""Music: finding songs in the library, adding them from the web, credits and notices, and offering songs to try."""

import hashlib
import json
import os
from decimal import Decimal
from typing import Annotated, Dict, List, Optional, Tuple

from pydantic import Field

from app.models.media import Source
from app.models.plan import EditPlan, MusicCue, SetMusicCueOp
from app.models.timeline.parts import Clip, Project, TimeRange, Track, TrackType
from app.models.offer import OfferedSong, SongOffer
from app.engine import clap, fetch
from app.engine.probe import probe_file, rights_tags
from app.engine.library import MUSIC, library_of
from app.engine.delivery import clock, marked_parts, rights_facts
from app.storage import housekeeping
from app.workspace import output_dir
from app import runtime
from app.runtime import mcp
from app.tools import catalog, examining, feedback, planning, rendering, shared

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
    asset = catalog.register_asset(path, MUSIC).model_copy(update={"source": Source(
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
        amended = planning.amend_plan(plan.id, plan.version, [SetMusicCueOp(beat_id=cue.beat_id, asset_id=song.asset_id,
                                                                              start=song.start)], note=said)
        said += f"；已寫進計畫 {plan.id} 第 {amended['version']} 版，重新 compile_plan 後生效"
    made = feedback.add_comment(project_id, said, offer.start, offer.end)
    runtime.repo.save_song_offer(None, project_id)
    return made
