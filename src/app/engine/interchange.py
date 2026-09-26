"""Handing a cut to a professional editor: EDL, OpenTimelineIO, FCPXML, and SRT.

Somebody who wants to finish a cut in Premiere, Resolve or Final Cut should
not have to rebuild it from the rendered file. These write the cut as those
programs read it — which footage, which part of it, where on the timeline, on
which track — so they open it pointed at the original files.

What they carry is the edit: every clip's in and out, its place, its track,
and the parts of the video as markers. What they leave behind is everything
this server draws itself: speed changes, transitions, colour, volume and
fades, voice repair, where an inset sits, and captions (those go out as SRT,
which every one of those programs imports). Each exporter says exactly which
of those the cut used, so nobody finds out by watching the timeline in the
other program and noticing it is wrong.

Written by hand rather than through a library: each format here is a small,
stable, documented text format, and the parts of them used are smaller still.
"""

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from xml.sax.saxutils import quoteattr

from app.models.media import Asset
from app.models.timeline import Clip, PlacedCue, Project, TrackType

FORMATS = ("edl", "otio", "fcpxml", "srt")

def _frames(seconds, fps: Fraction) -> int:
    """Count the frames up to a moment, rounded to the nearest.

    Args:
        seconds: The moment, as a number or a Decimal.
        fps: The frame rate.

    Returns:
        Whole frames.
    """
    return int(round(Fraction(str(seconds)) * fps))

def _uri(path: str) -> str:
    """Write a file's path the way these formats point at media.

    Args:
        path: The file, as the asset records it.

    Returns:
        A `file:` URI, made absolute first.
    """
    return Path(os.path.abspath(path)).as_uri()

def left_behind(project: Project) -> List[str]:
    """List what a cut uses that an exported timeline cannot carry.

    Args:
        project: The project.

    Returns:
        One line per kind of thing, naming the clips that use it. Empty when
        the export carries the whole cut.
    """
    uses: Dict[str, List[str]] = {}
    for track in project.tracks:
        for clip in track.clips:
            if clip.speed != 1.0:
                uses.setdefault("speed changes (they come across at normal speed)", []).append(clip.id)
            if clip.transition_in is not None:
                uses.setdefault("transitions (they come across as straight cuts)", []).append(clip.id)
            if clip.color is not None and not clip.color.is_neutral:
                uses.setdefault("colour adjustments", []).append(clip.id)
            if clip.volume not in (0.0, 1.0) or clip.audio_fade_in or clip.audio_fade_out:
                uses.setdefault("volume and audio fades", []).append(clip.id)
            if clip.video_fade_in or clip.video_fade_out:
                uses.setdefault("fades to and from black", []).append(clip.id)
            if clip.audio_lead or clip.audio_lag:
                uses.setdefault("J and L cuts (sound comes across cut with its picture)", []).append(clip.id)
            if clip.cleanup.rumble or clip.cleanup.hiss or clip.cleanup.sibilance:
                uses.setdefault("voice repair", []).append(clip.id)
            if clip.layout is not None:
                uses.setdefault("where an inset sits (it comes across full frame)", []).append(clip.id)
    lines = [f"{kind}: {', '.join(clips[:6])}{' …' if len(clips) > 6 else ''}" for kind, clips in uses.items()]
    if project.subtitles:
        lines.append("captions (export them separately as SRT)")
    if any(track.duck_under_speech for track in project.tracks):
        lines.append("music ducking under speech")
    return lines

# --- EDL ---------------------------------------------------------------------------------

def _timecode(frames: int, rate: int) -> str:
    """Write a frame count as non-drop timecode.

    Args:
        frames: Frames from zero.
        rate: Whole frames per second the timecode counts in.

    Returns:
        `HH:MM:SS:FF`.
    """
    seconds, frame = divmod(frames, rate)
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d}:{frame:02d}"

def write_edl(project: Project, assets: Mapping[str, Asset]) -> Tuple[str, List[str]]:
    """Write the sequence as a CMX 3600 edit decision list.

    An EDL is one picture track with its sound, so that is what it carries:
    the sequence, each event naming its file in a `FROM CLIP NAME` comment,
    which is how Premiere and Resolve find the media. Timecode is counted in
    whole frames at the nearest whole rate and never drop-frame, so it is
    frame-accurate but runs a little slow against the clock at 29.97.

    Args:
        project: The project.
        assets: Its files, keyed by asset ID.

    Returns:
        `(text, left_behind)` — the EDL, and what it could not carry, which
        for an EDL includes every track above the sequence and every audio
        track.
    """
    fps = Fraction(project.fps_num, project.fps_den)
    rate = int(round(fps))
    lines = [f"TITLE: {(project.name or project.id)[:70]}", "FCM: NON-DROP FRAME", ""]
    base = project.base_video_track
    # A marker is written under the event it falls in, which is where EDL readers look for one.
    marks = [(_frames(marker.timeline_in, fps), marker.name) for marker in project.markers]
    for number, clip in enumerate(sorted(base.clips if base else [], key=lambda item: item.timeline_in), start=1):
        asset = assets[clip.asset_id]
        channel = "B" if asset.has_audio and clip.volume > 0 else "V"
        source_in = _frames(clip.source_range.start, fps)
        source_out = _frames(clip.source_range.end, fps)
        record_in = _frames(clip.timeline_in, fps)
        record_out = record_in + (source_out - source_in)
        lines.append(
            f"{number:03d}  AX       {channel:<5} C        "
            f"{_timecode(source_in, rate)} {_timecode(source_out, rate)} "
            f"{_timecode(record_in, rate)} {_timecode(record_out, rate)}"
        )
        lines.append(f"* FROM CLIP NAME: {Path(asset.path).name}")
        lines += [
            f"* LOC: {_timecode(frame, rate)} WHITE {name}"
            for frame, name in sorted(marks) if record_in <= frame < record_out
        ]
        lines.append("")
    extra = [track.id for track in project.tracks if track is not base and track.clips]
    behind = left_behind(project)
    if extra:
        behind.insert(0, f"tracks other than the sequence ({', '.join(extra)}): an EDL holds one picture track")
    return "\n".join(lines).rstrip() + "\n", behind

# --- OpenTimelineIO ----------------------------------------------------------------------

def _time(frames: int, rate: float) -> dict:
    """Write a time the way OpenTimelineIO does.

    Args:
        frames: Frames from zero.
        rate: Frames per second.

    Returns:
        A `RationalTime`.
    """
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": float(frames)}

def _range(start: int, duration: int, rate: float) -> dict:
    """Write a stretch of time the way OpenTimelineIO does.

    Args:
        start: First frame.
        duration: Length in frames.
        rate: Frames per second.

    Returns:
        A `TimeRange`.
    """
    return {"OTIO_SCHEMA": "TimeRange.1", "start_time": _time(start, rate), "duration": _time(duration, rate)}

def _otio_track(name: str, kind: str, clips: Sequence[Clip], assets: Mapping[str, Asset], fps: Fraction) -> dict:
    """Write one track, with gaps where nothing plays.

    Args:
        name: The track's name.
        kind: `Video` or `Audio`.
        clips: Its clips.
        assets: Their files.
        fps: The frame rate.

    Returns:
        A `Track`.
    """
    rate = float(fps)
    children: List[dict] = []
    cursor = 0
    for clip in sorted(clips, key=lambda item: item.timeline_in):
        start = _frames(clip.timeline_in, fps)
        source_in = _frames(clip.source_range.start, fps)
        length = _frames(clip.source_range.end, fps) - source_in
        if start > cursor:
            children.append({
                "OTIO_SCHEMA": "Gap.1", "name": "", "metadata": {}, "effects": [], "markers": [],
                "enabled": True, "source_range": _range(0, start - cursor, rate),
            })
        asset = assets[clip.asset_id]
        children.append({
            "OTIO_SCHEMA": "Clip.2",
            "name": Path(asset.path).name,
            "metadata": {"clip_mcp": {"clip_id": clip.id, "asset_id": clip.asset_id}},
            "effects": [], "markers": [], "enabled": True,
            "source_range": _range(source_in, length, rate),
            "media_references": {"DEFAULT_MEDIA": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "name": Path(asset.path).name,
                "metadata": {},
                "target_url": _uri(asset.path),
                "available_range": _range(0, _frames(asset.duration, fps), rate) if asset.duration else None,
                "available_image_bounds": None,
            }},
            "active_media_reference_key": "DEFAULT_MEDIA",
        })
        cursor = start + length
    return {
        "OTIO_SCHEMA": "Track.1", "name": name, "kind": kind, "metadata": {}, "effects": [], "markers": [],
        "enabled": True, "source_range": None, "children": children,
    }

def write_otio(project: Project, assets: Mapping[str, Asset]) -> Tuple[str, List[str]]:
    """Write the whole cut as an OpenTimelineIO timeline.

    Every track comes across: the sequence and each track above it as video
    tracks, the sequence's own sound as an audio track beside it, and each
    audio track after that. The parts of the video are markers on the
    timeline.

    Args:
        project: The project.
        assets: Its files, keyed by asset ID.

    Returns:
        `(json_text, left_behind)`.
    """
    fps = Fraction(project.fps_num, project.fps_den)
    rate = float(fps)
    tracks: List[dict] = []
    for number, track in enumerate(project.video_tracks, start=1):
        tracks.append(_otio_track(f"V{number}", "Video", track.clips, assets, fps))
    base = project.base_video_track
    heard = [clip for clip in (base.clips if base else []) if assets[clip.asset_id].has_audio and clip.volume > 0]
    audio = [track for track in project.tracks if track.track_type == TrackType.AUDIO]
    if heard:
        tracks.append(_otio_track("A1", "Audio", heard, assets, fps))
    for number, track in enumerate(audio, start=2 if heard else 1):
        tracks.append(_otio_track(f"A{number}", "Audio", track.clips, assets, fps))
    markers = [
        {"OTIO_SCHEMA": "Marker.2", "name": marker.name, "metadata": {}, "color": "GREEN", "comment": "",
         "marked_range": _range(_frames(marker.timeline_in, fps), 0, rate)}
        for marker in sorted(project.markers, key=lambda item: item.timeline_in)
    ]
    timeline = {
        "OTIO_SCHEMA": "Timeline.1",
        "name": project.name or project.id,
        "metadata": {"clip_mcp": {"project_id": project.id, "width": project.width, "height": project.height}},
        "global_start_time": None,
        "tracks": {
            "OTIO_SCHEMA": "Stack.1", "name": "tracks", "metadata": {}, "effects": [], "markers": markers,
            "enabled": True, "source_range": None, "children": tracks,
        },
    }
    return json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", left_behind(project)

# --- FCPXML ------------------------------------------------------------------------------

def _rational(frames: int, fps: Fraction) -> str:
    """Write a frame count as an FCPXML time.

    Args:
        frames: Frames from zero.
        fps: The frame rate.

    Returns:
        A rational number of seconds, such as `1001/30000s`.
    """
    if frames == 0:
        return "0s"
    seconds = Fraction(frames) / fps
    return f"{seconds.numerator}/{seconds.denominator}s" if seconds.denominator != 1 else f"{seconds.numerator}s"

def write_fcpxml(project: Project, assets: Mapping[str, Asset]) -> Tuple[str, List[str]]:
    """Write the whole cut as Final Cut Pro XML, which Resolve and Premiere also read.

    The sequence is the spine, with gaps where it is empty. Everything else
    is connected to it the way Final Cut connects things: tracks above the
    sequence as lanes above it, audio tracks as lanes below it. The parts of
    the video are markers on whichever piece of the spine they fall in.

    Args:
        project: The project.
        assets: Its files, keyed by asset ID.

    Returns:
        `(xml_text, left_behind)`.
    """
    fps = Fraction(project.fps_num, project.fps_den)
    used = sorted({clip.asset_id for track in project.tracks for clip in track.clips})
    reference = {asset_id: f"r{index}" for index, asset_id in enumerate(used, start=2)}
    resources = [
        f'<format id="r1" frameDuration="{_rational(1, fps)}" width="{project.width}" height="{project.height}"/>'
    ]
    for asset_id in used:
        asset = assets[asset_id]
        duration = _rational(_frames(asset.duration, fps), fps) if asset.duration else "0s"
        resources.append(
            f'<asset id="{reference[asset_id]}" name={quoteattr(Path(asset.path).name)} start="0s" '
            f'duration="{duration}" hasVideo="{int(asset.has_video)}" hasAudio="{int(asset.has_audio)}"'
            f'{" audioSources=\"1\" audioChannels=\"2\"" if asset.has_audio else ""}>'
            f'<media-rep kind="original-media" src={quoteattr(_uri(asset.path))}/></asset>'
        )

    # The spine: (offset, start, duration, element head) per piece, in order.
    base = project.base_video_track
    spine: List[Tuple[int, int, int, str]] = []
    cursor = 0
    for clip in sorted(base.clips if base else [], key=lambda item: item.timeline_in):
        offset = _frames(clip.timeline_in, fps)
        start = _frames(clip.source_range.start, fps)
        length = _frames(clip.source_range.end, fps) - start
        if offset > cursor:
            spine.append((cursor, 0, offset - cursor, "gap"))
        spine.append((offset, start, length, clip.asset_id))
        cursor = offset + length
    children: Dict[int, List[str]] = {index: [] for index in range(len(spine))}

    def parent_of(frame: int) -> Optional[int]:
        """Find the piece of the spine a moment falls in."""
        for index, (offset, _, length, _) in enumerate(spine):
            if offset <= frame < offset + length:
                return index
        return len(spine) - 1 if spine else None

    def connect(clip: Clip, lane: int) -> None:
        """Hang one clip off the spine piece it starts over."""
        at = _frames(clip.timeline_in, fps)
        index = parent_of(at)
        if index is None:
            return
        offset, start, _, _ = spine[index]
        source_in = _frames(clip.source_range.start, fps)
        length = _frames(clip.source_range.end, fps) - source_in
        children[index].append(
            f'<asset-clip ref="{reference[clip.asset_id]}" lane="{lane}" name={quoteattr(clip.id)} '
            f'offset="{_rational(start + at - offset, fps)}" start="{_rational(source_in, fps)}" '
            f'duration="{_rational(length, fps)}"/>'
        )

    for lane, track in enumerate(project.video_tracks[1:], start=1):
        for clip in track.clips:
            connect(clip, lane)
    for lane, track in enumerate([track for track in project.tracks if track.track_type == TrackType.AUDIO], start=1):
        for clip in track.clips:
            connect(clip, -lane)
    for marker in project.markers:
        at = _frames(marker.timeline_in, fps)
        index = parent_of(at)
        if index is not None:
            offset, start, _, _ = spine[index]
            children[index].append(
                f'<marker start="{_rational(start + at - offset, fps)}" duration="{_rational(1, fps)}" '
                f'value={quoteattr(marker.name)}/>'
            )

    body: List[str] = []
    for index, (offset, start, length, what) in enumerate(spine):
        timing = f'offset="{_rational(offset, fps)}" start="{_rational(start, fps)}" duration="{_rational(length, fps)}"'
        head = (f'<gap name="Gap" {timing}>' if what == "gap"
                else f'<asset-clip ref="{reference[what]}" name={quoteattr(Path(assets[what].path).name)} {timing}>')
        close = "</gap>" if what == "gap" else "</asset-clip>"
        body.append(head + "".join(children[index]) + close)
    total = _rational(cursor, fps)
    name = project.name or project.id
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n<fcpxml version="1.9">\n'
        f'<resources>{"".join(resources)}</resources>\n'
        f'<library><event name={quoteattr(name)}><project name={quoteattr(name)}>'
        f'<sequence format="r1" duration="{total}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">'
        f'<spine>{"".join(body)}</spine></sequence></project></event></library>\n</fcpxml>\n'
    )
    return xml, left_behind(project)

# --- SRT ---------------------------------------------------------------------------------

def _srt_time(seconds) -> str:
    """Write a time the way SRT does.

    Args:
        seconds: The time.

    Returns:
        `HH:MM:SS,mmm`.
    """
    milliseconds = int(round(Fraction(str(seconds)) * 1000))
    whole, milli = divmod(milliseconds, 1000)
    minutes, second = divmod(whole, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d},{milli:03d}"

def write_srt(cues: Sequence[PlacedCue]) -> str:
    """Write placed captions as an SRT file, the one caption format everything imports.

    Args:
        cues: The captions as they fall in the cut.

    Returns:
        The file's contents; a bilingual caption carries its second line
        underneath.
    """
    blocks = []
    for number, cue in enumerate(cues, start=1):
        text = cue.text if not cue.secondary else f"{cue.text}\n{cue.secondary}"
        blocks.append(f"{number}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n{text}\n")
    return "\n".join(blocks)
