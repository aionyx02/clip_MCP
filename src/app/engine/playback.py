"""What the browser needs to play a cut as it is edited, without rendering it.

The editor and the version panel play a cut the way editing programs do:
the browser reads small copies of the footage — proxies — and puts the cut
together as it plays, picture and sound, from a description of the timeline.
Nothing is rendered to watch an edit, so a change plays at once and moving
between versions costs nothing.

What the browser cannot do the server prepares, once, and keeps by what went
into it: the proxies themselves, the sound of a clip whose voice is repaired,
and the gain that brings a whole mix to its loudness. This module writes the
commands for those, and the description the browser plays from.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine.builder import (
    DUCK_ATTACK_SECONDS, DUCK_DEPTH_DB, DUCK_HOLD_SECONDS, DUCK_RELEASE_SECONDS, Talking, _cleanup_filters,
    overlay_box,
)
from app.engine.reframe import Framing
from app.engine.subtitles import drawn_captions, place_cues
from app.models.media import Asset
from app.models.timeline import Clip, Project, TrackType
from app.storage.history import fingerprint

# Provisional (roadmap §13): a proxy's short side, in pixels. The editor's player is a few
# hundred pixels across, and decoding several of these at once is what keeps playback smooth.
PROXY_SHORT_SIDE = 480
# Provisional (roadmap §13): seconds between a proxy's keyframes. Jumping to a cut can only
# start at one, so this is the longest a seek waits; denser costs size.
PROXY_KEY_SECONDS = 0.5
# The fastest a proxy runs: more frames than this decode work and show nothing more at its size.
PROXY_MAX_FPS = 30
# How hard a proxy is compressed: plain enough to look at, small enough to keep many.
PROXY_CRF = 26
# The loudness every render is brought to, and so what the browser's mix is aimed at.
LOUDNESS_TARGET = -14.0


@dataclass(frozen=True)
class Prepared:
    """What the server has ready for the browser to play.

    Attributes:
        proxies: The proxy's address by asset ID, for each file that has one.
        repaired: The repaired sound's address by clip ID.
        gain_db: The gain that brings the mix to its loudness, or None while
            it has not been measured.
    """

    proxies: Mapping[str, str]
    repaired: Mapping[str, str]
    gain_db: Optional[float]


def proxy_name(asset: Asset) -> Optional[str]:
    """Name a file's proxy after the file as it is now, so a changed file gets a new one.

    Args:
        asset: The file.

    Returns:
        The proxy's file name, or None when the file is not there to copy.
    """
    found = fingerprint(asset.path)
    if found is None:
        return None
    return f"{asset.id}-{found}.{'mp4' if asset.has_video else 'm4a'}"


def proxy_command(asset: Asset, part: str, fps: float, ffmpeg_bin: str = "ffmpeg") -> List[str]:
    """Write the command that makes a file's proxy.

    The picture is shrunk to `PROXY_SHORT_SIDE` and given a keyframe every
    `PROXY_KEY_SECONDS`, so the browser can jump to any cut almost at once;
    the sound is kept whole, at 48 kHz, since it is what is mixed.

    Args:
        asset: The file.
        part: Where to write it, under a name of its own until it is whole.
        fps: The file's frame rate, if known; 0 when not.
        ffmpeg_bin: FFmpeg executable.

    Returns:
        The command.
    """
    command = [ffmpeg_bin, "-y", "-loglevel", "error", "-nostats", "-i", asset.path]
    if asset.has_video:
        rate = min(fps, PROXY_MAX_FPS) if fps > 0 else PROXY_MAX_FPS
        short = PROXY_SHORT_SIDE
        command += [
            "-map", "0:v:0", "-vf",
            # Shrunk to `short` on its short side, never enlarged: a file already that small is copied as it is.
            f"scale='if(gt(iw,ih),-2,min({short},iw))':'if(gt(iw,ih),min({short},ih),-2)',fps={rate:g},format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(PROXY_CRF),
            "-force_key_frames", f"expr:gte(t,n_forced*{PROXY_KEY_SECONDS})", "-sc_threshold", "0",
        ]
    else:
        command += ["-vn"]
    if asset.has_audio:
        command += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
    else:
        command += ["-an"]
    return command + ["-movflags", "+faststart", "-f", "mp4", part]


def needs_repair(clip: Clip) -> bool:
    """Whether a clip's sound has repairs the browser cannot make itself."""
    return bool(_cleanup_filters(clip.cleanup))


def repair_name(clip: Clip, asset: Asset) -> Optional[str]:
    """Name a clip's repaired sound after exactly what went into it.

    Args:
        clip: The clip.
        asset: Its file.

    Returns:
        The file name, or None when the file is not there.
    """
    found = fingerprint(asset.path)
    if found is None:
        return None
    start, end = _heard_span(clip)
    recipe = json.dumps([found, start, end, _cleanup_filters(clip.cleanup)])
    return hashlib.sha256(recipe.encode("utf-8")).hexdigest()[:32] + ".m4a"


def _heard_span(clip: Clip) -> Tuple[float, float]:
    """The stretch of its file a clip's sound comes from, J and L cuts included."""
    speed = float(clip.speed)
    start = max(0.0, float(clip.source_range.start) - float(clip.audio_lead) * speed)
    return round(start, 3), round(float(clip.source_range.end) + float(clip.audio_lag) * speed, 3)


def repair_command(clip: Clip, asset: Asset, part: str, ffmpeg_bin: str = "ffmpeg") -> List[str]:
    """Write the command that makes a clip's repaired sound, at its own speed.

    The repairs run on the sound as recorded, before any change of speed or
    level, just as they do in a render; the browser applies the rest.

    Args:
        clip: The clip.
        asset: Its file.
        part: Where to write it.
        ffmpeg_bin: FFmpeg executable.

    Returns:
        The command.
    """
    start, end = _heard_span(clip)
    return [
        ffmpeg_bin, "-y", "-loglevel", "error", "-nostats", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
        "-i", asset.path, "-vn", "-af", ",".join(["aresample=48000", *_cleanup_filters(clip.cleanup)]),
        "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-f", "mp4", part,
    ]


def mix_key(project: Project) -> str:
    """Name a cut's sound by everything that decides how it sounds, for keeping its measured gain."""
    sound = {
        "tracks": [
            {"id": track.id, "type": track.track_type.value, "duck": track.duck_under_speech, "voice": track.voice,
             "clips": [clip.model_dump(mode="json", include={
                 "asset_id", "source_range", "timeline_in", "speed", "volume", "audio_fade_in", "audio_fade_out",
                 "audio_lead", "audio_lag", "cleanup", "keep_level", "preserve_pitch",
             }) for clip in track.clips]}
            for track in project.tracks
        ],
    }
    return hashlib.sha256(json.dumps(sound, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def _seconds(value) -> float:
    return round(float(value), 3)


def _crop(framing: Optional[Framing], clip: Clip, fps: Fraction) -> Optional[dict]:
    """Where a clip's crop sits, as times into the clip and centres, for the browser to follow."""
    if framing is None:
        return None
    return {
        "axis": framing.axis,
        "steps": [[round(frame / float(fps), 3), centre] for frame, centre in framing.positions],
    }


def describe(
    project: Project,
    assets: Mapping[str, Asset],
    prepared: Prepared,
    talking: Optional[Talking],
    voices: Mapping[str, float],
    framing: Mapping[str, Framing],
) -> dict:
    """Describe a cut for the browser to play it.

    Every number the browser needs is worked out here, the way a render works
    it out, so the two cannot disagree about where a cut is or how loud a
    clip is; the browser only follows it.

    Args:
        project: The cut.
        assets: Its files, by asset ID.
        prepared: What the server has ready.
        talking: Where somebody is talking and how loud each clip sits, or
            None when the transcripts cannot say.
        voices: The narration tracks, with the gain each is heard at.
        framing: Each clip's crop, by clip ID.

    Returns:
        The description: the frame, the tracks and their clips, the music's
        ducking, the captions, the gain to the target loudness, and what is
        still being prepared.
    """
    fps = Fraction(project.fps_num, project.fps_den)
    music_gains = talking.music_gains if talking is not None else {}
    clip_gains = talking.clip_gains if talking is not None else {}
    base = project.base_video_track
    tracks = []
    waiting = set()
    for track in project.tracks:
        clips = []
        for clip in sorted(track.clips, key=lambda item: item.timeline_in):
            asset = assets.get(clip.asset_id)
            if asset is None:
                continue
            media = prepared.proxies.get(clip.asset_id)
            if media is None:
                # Until its copy is made the clip plays from the file itself: heavier, but it plays.
                waiting.add(clip.asset_id)
                media = f"/media/{clip.asset_id}"
            gain = (clip_gains.get(clip.id, 0.0) + music_gains.get(track.id, {}).get(clip.id, 0.0)
                    + voices.get(track.id, 0.0))
            entry = {
                "id": clip.id, "asset_id": clip.asset_id, "proxy": media, "original": clip.asset_id in waiting,
                "has_video": asset.has_video,
                "has_audio": asset.has_audio, "source_start": _seconds(clip.source_range.start),
                "source_end": _seconds(clip.source_range.end), "timeline_in": _seconds(clip.timeline_in),
                "timeline_out": _seconds(clip.timeline_out), "speed": float(clip.speed),
                "preserve_pitch": clip.preserve_pitch, "volume": clip.volume, "gain_db": round(gain, 2),
                "audio_fade_in": _seconds(clip.audio_fade_in), "audio_fade_out": _seconds(clip.audio_fade_out),
                "audio_lead": _seconds(clip.audio_lead), "audio_lag": _seconds(clip.audio_lag),
                "video_fade_in": _seconds(clip.video_fade_in), "video_fade_out": _seconds(clip.video_fade_out),
                "repaired": prepared.repaired.get(clip.id),
            }
            if needs_repair(clip) and entry["repaired"] is None:
                waiting.add(f"sound:{clip.id}")
            if track.track_type == TrackType.VIDEO:
                entry["transition"] = clip.transition_in.model_dump(mode="json") if clip.transition_in else None
                entry["color"] = clip.color.model_dump(mode="json") if clip.color else None
                left, top, width, height = overlay_box(clip, project.width, project.height)
                entry["box"] = [left, top, width, height]
                entry["source_size"] = [asset.width, asset.height] if asset.width and asset.height else None
                entry["crop"] = _crop(framing.get(clip.id), clip, fps)
            clips.append(entry)
        tracks.append({
            "id": track.id, "type": track.track_type.value, "base": base is not None and track.id == base.id,
            "duck": track.duck_under_speech and track.id not in voices, "voice": track.id in voices, "clips": clips,
        })
    return {
        "width": project.width, "height": project.height, "fps": float(fps),
        "duration": _seconds(project.duration), "tracks": tracks,
        "duck": {
            # Without transcripts saying where the talking is, a render ducks by how loud
            # the talking is, which the browser cannot follow; it plays without, and says so.
            "spans": [list(span) for span in talking.spans] if talking is not None else None,
            "depth_db": DUCK_DEPTH_DB, "attack": DUCK_ATTACK_SECONDS, "release": DUCK_RELEASE_SECONDS,
            "hold": DUCK_HOLD_SECONDS,
        },
        "captions": drawn_captions(place_cues(project, project.subtitles), project.width, project.height,
                                   project.caption_style),
        "gain_db": prepared.gain_db,
        "loudness_target": LOUDNESS_TARGET,
        "approximate": {
            "color": any(clip.color for track in project.video_tracks for clip in track.clips),
            "ducking": talking is None and any(track.duck_under_speech for track in project.tracks),
            "narration": bool(voices),
        },
        "waiting": sorted(waiting),
    }


def missing(project: Project, assets: Mapping[str, Asset], have: Sequence[str]) -> List[str]:
    """The files of a cut that have no proxy yet.

    Args:
        project: The cut.
        assets: Its files.
        have: The proxy file names that exist.

    Returns:
        Their asset IDs, in the order the cut plays them.
    """
    seen: Dict[str, None] = {}
    existing = set(have)
    for track in project.tracks:
        for clip in sorted(track.clips, key=lambda item: item.timeline_in):
            asset = assets.get(clip.asset_id)
            name = proxy_name(asset) if asset is not None else None
            if name is not None and name not in existing:
                seen.setdefault(clip.asset_id, None)
    return list(seen)


def proxy_dir(workspace: str) -> str:
    """Where proxies are kept: a cache, rebuilt whenever one is missing."""
    folder = os.path.join(workspace, "cache", "proxies")
    os.makedirs(folder, exist_ok=True)
    return folder


def sound_dir(workspace: str) -> str:
    """Where repaired sound and measured gains are kept: a cache, like the proxies."""
    folder = os.path.join(workspace, "cache", "sound")
    os.makedirs(folder, exist_ok=True)
    return folder
