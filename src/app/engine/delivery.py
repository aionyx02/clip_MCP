"""What a finished cut needs on its way out: chapters, a cover, and a last look.

None of this changes the cut. Chapters are the plan's beats read back as
times; cover candidates are frames the footage's own measurements say are
worth a look; and the check before a render is a list of things somebody
would notice in the finished file — black on screen, a voice clipping, a
caption off the top of the picture, a length nobody asked for — found while
fixing them is still a change to a plan rather than a re-render.

All of it is a pure function of the project and the analyses, like the plan
compiler, so it can be asked before anything is rendered and gives the same
answer every time.
"""

import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine.builder import DUCK_DEPTH_DB, Talking
from app.engine.continuity import Shot, stray_runs
from app.engine.looks import alike
from app.engine.plan import LENGTH_TOLERANCE
from app.engine.library import MUSIC, library_of
from app.engine.semantic import clean_cuts, combined_over, join_voices, share_covered, was_audible, was_made_up
from app.engine.subtitles import caption_overflow, captioned_clips
from app.models.media import Asset, MediaAnalysis
from app.models.timeline import CaptionStyle, Clip, Marker, PlacedCue, Project, TrackType

# YouTube's rules for chapters written into a description: the first at 0:00, at least
# three of them, each at least ten seconds long. The platform's rules, not a taste, and
# like the caption presets they live in one place so a change to them is one edit.
YOUTUBE_MIN_CHAPTERS = 3
YOUTUBE_MIN_CHAPTER_SECONDS = 10.0
# A recording has clipped when a second of it peaks at the ceiling and its waveform is
# squared off there. Both numbers are provisional until there is a corpus: a recording
# that peaks this high without flattening is loud, not broken.
CLIPPED_PEAK_DB = -0.5
CLIPPED_FLATNESS = 5.0
# The kinds of finding a render can be told to go ahead in spite of.
CHECKS = (
    "bad_picture", "clipping", "music", "mid_speech", "repeated", "look_alike", "continuity", "unplanned", "captions",
    "length",
)
# How far under the talking music has to sit while somebody speaks, in dB. Under that it
# competes with the words: the first edits made with the server had music two decibels
# under the voice, and it was the first thing anybody listening mentioned. Provisional.
MUSIC_UNDER_TALKING_DB = 12.0
# A cut is clean where the speaker stopped: between two words at least this far apart.
# Shorter gaps are the spaces inside a phrase, and a cut there sounds like a word bitten
# off. Provisional.
CLEAN_GAP_SECONDS = 0.2
# How far either side of a cut to look for a clean one to offer instead. Provisional.
CLEAN_SEARCH_SECONDS = 3.0
# A cut inside a measured silence at least this long is clean whatever the word timings
# say: the transcriber runs a word on across a pause often enough that taking its word
# for it refused cuts made in the middle of real silence. Provisional.
CLEAN_SILENCE_SECONDS = 0.3
# How much of the same stretch of a file has to come back before it is the same shot shown
# twice rather than two neighbouring moments. Provisional.
REPEAT_SECONDS = 1.0
# A cut this close after a sound rises out of a measured silence, or this close before it
# falls back into one, takes the start or the end of that sound off — a word's first
# consonant or its tail — whatever the transcript says. The transcript's word times are
# a guess; the silences were measured. Provisional.
SOUND_EDGE_SECONDS = 0.5
# Under this, the sound cut off is too short to hear. Provisional.
SOUND_EDGE_MIN_SECONDS = 0.04
# A sequence of this many clips is an edit, and an edit with no plan has nothing that says
# how it begins, turns and ends. Fewer is a trim. Provisional.
UNPLANNED_CLIPS = 3

@dataclass(frozen=True)
class Finding:
    """One thing about a cut that would be noticed in the finished file.

    Attributes:
        check: Which check found it: one of `CHECKS`.
        message: What it is and where, for whoever decides.
        spans: Where in the cut it is, as `(start, end)` in seconds, a
            moment being a span that starts where it ends; empty for what
            is about the whole cut.
    """

    check: str
    message: str
    spans: Tuple[Tuple[float, float], ...] = ()

def clock(seconds: float) -> str:
    """Write a time the way a video description does.

    Args:
        seconds: The time.

    Returns:
        `m:ss`, or `h:mm:ss` from an hour on.
    """
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"

def marked_parts(project: Project) -> List[Tuple[float, float, Marker]]:
    """The parts of a cut: each marker, in timeline order, running to the next or to the end.

    Args:
        project: The cut.

    Returns:
        `(start, end, marker)` per marker that begins inside the cut.
    """
    duration = float(project.duration)
    marks = sorted((marker for marker in project.markers if float(marker.timeline_in) < duration),
                   key=lambda marker: marker.timeline_in)
    return [(float(marker.timeline_in), float(marks[index + 1].timeline_in) if index + 1 < len(marks) else duration,
             marker) for index, marker in enumerate(marks)]

def chapters(project: Project) -> Tuple[List[Tuple[float, float, str]], List[str]]:
    """Read a project's parts back as chapters.

    The markers are the plan's beats written onto the timeline, so a chapter is
    a marker and runs to the next one. Nothing is moved to satisfy a platform:
    a part too short to be a YouTube chapter is reported, because merging two
    parts or renaming one is a decision about the video, not about its
    description.

    Args:
        project: The project.

    Returns:
        `(chapters, problems)` — one `(start, end, name)` per marker in
        timeline order, and what stops YouTube from taking them as chapters.
    """
    listed = [(start, end, marker.name) for start, end, marker in marked_parts(project)]
    problems: List[str] = []
    if not listed:
        return [], ["the project has no markers; compiling a plan writes one per beat, or add them with set_markers"]
    if listed[0][0] > 0:
        problems.append(f"the first part starts at {clock(listed[0][0])}; YouTube needs a chapter at 0:00")
    if len(listed) < YOUTUBE_MIN_CHAPTERS:
        problems.append(f"{len(listed)} part(s); YouTube needs at least {YOUTUBE_MIN_CHAPTERS} to show chapters")
    for start, end, name in listed:
        if end - start < YOUTUBE_MIN_CHAPTER_SECONDS:
            problems.append(
                f"{name} runs {end - start:.1f}s; YouTube needs every chapter to be at least "
                f"{YOUTUBE_MIN_CHAPTER_SECONDS:g}s"
            )
    return listed, problems

def chapter_metadata(listed: Sequence[Tuple[float, float, str]]) -> str:
    """Write chapters as an FFmpeg metadata file, to be carried inside the MP4.

    Players that read chapters — VLC, QuickTime, most desktop players — show
    them from the file itself; the description text is for platforms that
    read them from there instead.

    Args:
        listed: `(start, end, name)` per chapter.

    Returns:
        The file's contents.
    """
    def escaped(text: str) -> str:
        """Escape what the metadata format reserves."""
        for reserved in ("\\", "=", ";", "#", "\n"):
            text = text.replace(reserved, "\\" + reserved)
        return text

    lines = [";FFMETADATA1"]
    for start, end, name in listed:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(start * 1000)}", f"END={round(end * 1000)}",
                  f"title={escaped(name)}"]
    return "\n".join(lines) + "\n"

def _on_timeline(clip: Clip, start: float, end: float) -> Optional[Tuple[float, float]]:
    """Say where a stretch of a clip's source lands on the timeline.

    Args:
        clip: The clip.
        start: Where the stretch starts in the source, in seconds.
        end: Where it ends.

    Returns:
        `(start, end)` in timeline seconds, or None when none of it is in the clip.
    """
    opened, closed = max(start, float(clip.source_range.start)), min(end, float(clip.source_range.end))
    if closed <= opened:
        return None
    into = lambda seconds: float(clip.timeline_in) + (seconds - float(clip.source_range.start)) / clip.speed
    return into(opened), into(closed)

def _uncovered(stretch: Tuple[float, float], covers: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Take out of a stretch whatever is hidden under full-frame picture.

    Args:
        stretch: `(start, end)` on the timeline.
        covers: Stretches drawn over the whole frame.

    Returns:
        What is left, as stretches.
    """
    left = [stretch]
    for opened, closed in covers:
        left = [
            piece for start, end in left
            for piece in ((start, min(end, opened)), (max(start, closed), end))
            if piece[1] - piece[0] > 0
        ]
    return left

def check_delivery(
    project: Project,
    analyses: Mapping[str, MediaAnalysis],
    target_seconds: Optional[float] = None,
    captions: Optional[Sequence[PlacedCue]] = None,
    style: Optional[CaptionStyle] = None,
    talking: Optional[Talking] = None,
    recorded: Optional[Mapping[str, datetime]] = None,
    looks: Optional[Mapping[str, Sequence[int]]] = None,
) -> List[Finding]:
    """Find what would be noticed in the finished file, before it is rendered.

    Args:
        project: The project, at the size it will be rendered at.
        analyses: The analyses of its files, keyed by asset ID. A file with no
            analysis has nothing known about it and nothing found in it.
        target_seconds: How long the plan it was compiled from asked for it to
            run, if it came from one that said.
        captions: The captions being burned in, placed; None when none are.
        style: How those captions are drawn.
        talking: Where the talking is and what brings each music track to
            it, from `levels.talking_in`. Without it, how the music sits
            under the voices is not known and not checked.
        recorded: When each file was recorded, keyed by asset ID, for
            finding a shot dropped between two from another moment. Only the
            clips put on by hand are checked: a compiled plan was already
            checked for the same thing, with its reasons to hand.
        looks: The fingerprints of the clips `look_candidates` names, keyed
            by clip ID, for finding two from different files a viewer could
            not tell apart. Without them that is not checked.

    Returns:
        The findings, grouped by check in the order of `CHECKS`.
    """
    fps = project.fps_num / project.fps_den
    findings: List[Finding] = []
    duration = float(project.duration)

    # Black or frozen source picture that reaches the screen: on the sequence, where
    # no full-frame picture is laid over it, and on that full-frame picture itself.
    covers = [
        (float(clip.timeline_in), float(clip.timeline_out))
        for track in project.video_tracks[1:] for clip in track.clips if clip.layout is None
    ]
    base = project.base_video_track
    shown: List[Tuple[Clip, List[Tuple[float, float]]]] = []
    for clip in (base.clips if base else []):
        shown.append((clip, covers))
    for track in project.video_tracks[1:]:
        shown += [(clip, []) for clip in track.clips if clip.layout is None]
    for clip, hidden_by in sorted(shown, key=lambda item: item[0].timeline_in):
        analysis = analyses.get(clip.asset_id)
        # A title card's source range is only its length, not a stretch of its file that plays.
        if analysis is None or clip.card is not None:
            continue
        for span in sorted([*analysis.black_frames, *analysis.frozen_frames], key=lambda item: item.start):
            landed = _on_timeline(clip, span.start, span.end)
            if landed is None:
                continue
            for start, end in _uncovered((max(0.0, landed[0]), min(duration, landed[1])), hidden_by):
                if end - start >= 1 / fps:
                    findings.append(Finding("bad_picture", (
                        f"black or frozen picture from {clock(start)} to {clock(end)} "
                        f"({end - start:.1f}s, clip {clip.id})"
                    ), ((start, end),)))

    # Everything heard: the sequence and the audio tracks, less anything turned all the way down.
    for clip in captioned_clips(project):
        analysis = analyses.get(clip.asset_id)
        if analysis is None:
            continue
        clipped = [
            landed for second in analysis.sound
            if second.peak >= CLIPPED_PEAK_DB and second.flatness >= CLIPPED_FLATNESS
            and (landed := _on_timeline(clip, second.start, second.end)) is not None
        ]
        if clipped:
            findings.append(Finding("clipping", (
                f"clip {clip.id} was recorded clipping — squared off at the ceiling — for {len(clipped)} "
                f"second(s), first at {clock(clipped[0][0])}. Turning it down does not undo it"
            )))

    findings.extend(_music_over_talking(project, talking))
    findings.extend(_speech_cuts(base.clips if base else [], analyses))
    findings.extend(_repeats(base.clips if base else []))
    findings.extend(_look_alikes(base.clips if base else [], looks or {}))
    findings.extend(_out_of_time(project, recorded or {}))
    if base and len(base.clips) >= UNPLANNED_CLIPS and any(clip.from_plan_id is None for clip in base.clips):
        findings.append(Finding("unplanned", (
            f"these {len(base.clips)} clips were put together by hand, not compiled from a plan, so nothing "
            "says what the video is for or how its parts hold together. Ask the user which structure it takes — "
            "起承轉合, 清單式, 教學步驟 or another — write a plan in it, and compile it"
        )))

    if captions:
        for cue in caption_overflow(captions, project.width, project.height, style):
            findings.append(Finding("captions", (
                f"the caption at {clock(float(cue.start))} ({cue.text[:16]}) is too tall for a "
                f"{project.width}x{project.height} frame and runs off the top"
            )))
        # Placing already held every short caption on as long as its shot allowed; one still
        # short is one there was no room for, which only a change to the cut can give.
        # The same minimum they were placed with, which is the project's own style.
        shortest = project.caption_style.min_seconds
        starts = sorted(float(cue.start) for cue in captions)
        for cue in captions:
            shown = float(cue.end - cue.start)
            if shown >= shortest - 0.0005:
                continue
            crowded = any(float(cue.end) - 0.001 <= start <= float(cue.end) + 0.001 for start in starts)
            why = ("the next caption follows straight on. Join the two with `edit_subtitle`" if crowded
                   else "its shot ends before it can be held longer. Let the shot run on (a later end on its trim)")
            findings.append(Finding("captions", (
                f"the caption at {clock(float(cue.start))} ({cue.text[:16]}) is on screen {shown:.2f}s, under the "
                f"{shortest:g}s it takes to read, and {why}, take the caption out with `edit_subtitle` and "
                "`delete`, or lower `min_seconds` in the caption style if the user reads that fast"
            )))

    if target_seconds and abs(duration - target_seconds) / target_seconds > LENGTH_TOLERANCE:
        findings.append(Finding("length", (
            f"the cut runs {duration:.1f}s against the {target_seconds:g}s its plan asked for"
        )))
    order = {check: index for index, check in enumerate(CHECKS)}
    return sorted(findings, key=lambda finding: order[finding.check])

def _music_over_talking(project: Project, talking: Optional[Talking]) -> List[Finding]:
    """Find music that plays too close to the level of somebody talking over it.

    Only music brought to the level of the talking can be judged against it;
    a track that was not, or a cut nothing says the talking of, is left out.

    Args:
        project: The project.
        talking: Where the talking is, and which music tracks were brought
            to it.

    Returns:
        One finding per music clip heard too close under the talking.
    """
    if talking is None or not talking.spans:
        return []
    findings = []
    for track in project.tracks:
        if track.id not in talking.music_gains:
            continue
        for clip in track.clips:
            opened, closed = float(clip.timeline_in), float(clip.timeline_out)
            if clip.id not in talking.music_gains[track.id] or clip.volume == 0 or not any(start < closed and end > opened for start, end in talking.spans):
                continue
            under = -20 * math.log10(clip.volume) + (DUCK_DEPTH_DB if track.duck_under_speech else 0.0)
            if under < MUSIC_UNDER_TALKING_DB:
                findings.append(Finding("music", (
                    f"the music on track {track.id} from {clock(opened)} plays {under:.0f} dB under the talking "
                    f"over it, where {MUSIC_UNDER_TALKING_DB:g} dB is about what keeps the words clear; "
                    "turn it down, or let it duck under speech"
                )))
    return findings

def _said(analysis: MediaAnalysis) -> list:
    """List the sentences of a file that were really said.

    Args:
        analysis: The file's analysis.

    Returns:
        Its transcribed sentences, less any the recogniser wrote over a
        stretch measured as silent — the same rule that keeps them out of the
        captions, so a cut is not said to split a line nobody spoke.
    """
    segments = [segment for segment in (analysis.transcript.segments if analysis.transcript else [])
                if not was_made_up(segment)]
    if not analysis.silences:
        return segments
    return [
        segment for segment in segments
        if was_audible(share_covered(segment.start, segment.end, analysis.silences), True)
    ]

def _clean_points(analysis: MediaAnalysis) -> List[float]:
    """List where a file's speech can be cut without biting into it.

    Args:
        analysis: The file's analysis.

    Returns:
        The middle of every gap between words long enough to be a pause, and
        the start and end of every sentence, in source seconds.
    """
    points: List[float] = []
    for segment in _said(analysis):
        points += [segment.start, segment.end]
        for before, after in zip(segment.words, segment.words[1:]):
            if after.start - before.end >= CLEAN_GAP_SECONDS:
                points.append((before.end + after.start) / 2)
    return sorted(points)

def _inside_speech(analysis: MediaAnalysis, at: float, opens: bool) -> bool:
    """Tell whether a cut at one moment of a file lands in the middle of somebody talking.

    Args:
        analysis: The file's analysis.
        at: The moment, in source seconds.
        opens: Whether the cut starts a clip rather than ending one.

    Returns:
        True inside a word, or between two words of one sentence closer
        together than a pause, or where sound the transcript missed runs
        right up to the cut on the side it keeps out. Never inside a measured
        silence of `CLEAN_SILENCE_SECONDS` or more, nor where the second
        listen hears nobody, whatever the word timings say.
    """
    if any(span.start <= at <= span.end and span.end - span.start >= CLEAN_SILENCE_SECONDS
           for span in analysis.silences):
        return False
    known = clean_cuts(analysis)
    if (known.sound_before(at) if opens else known.sound_after(at)) is not None:
        return True
    if known.quiet_on_second_listen(at):
        return False
    for segment in _said(analysis):
        if not segment.start < at < segment.end:
            continue
        words = segment.words
        if not words:
            # Nothing finer is known than the sentence, and the cut is inside it.
            return True
        for before, after in zip(words, words[1:]):
            if before.end <= at <= after.start:
                return after.start - before.end < CLEAN_GAP_SECONDS
        return any(word.start < at < word.end for word in words)
    return False

def _bitten_sound(analysis: MediaAnalysis, at: float, opens: bool) -> Optional[float]:
    """Find a sound a cut takes the start or the end off, from the measured silences alone.

    Args:
        analysis: The file's analysis.
        at: The moment, in source seconds.
        opens: Whether the cut starts a clip rather than ending one.

    Returns:
        Where the sound begins (for a cut that opens) or ends (for one that
        closes), when that is between `SOUND_EDGE_MIN_SECONDS` and
        `SOUND_EDGE_SECONDS` away on the side the cut leaves out. None when
        the cut is in a silence, nothing was measured, or the sound runs on
        further than that — the middle of a long sound is not its edge.
    """
    spans = sorted((span.start, span.end) for span in analysis.silences)
    if not spans or any(start <= at <= end for start, end in spans):
        return None
    if opens:
        rises = [end for _, end in spans if end < at]
        edge = max(rises) if rises else None
    else:
        falls = [start for start, _ in spans if start > at]
        edge = min(falls) if falls else None
    if edge is None or not SOUND_EDGE_MIN_SECONDS <= abs(at - edge) <= SOUND_EDGE_SECONDS:
        return None
    return edge

def _speech_cuts(clips: Sequence[Clip], analyses: Mapping[str, MediaAnalysis]) -> List[Finding]:
    """Find cuts on the sequence that land in the middle of somebody talking.

    Only cuts somebody made: the start and end of a recording are where the
    camera started and stopped, not a decision. And the cut is where the
    sound is cut, which a J or an L cut moves away from the picture's.

    Args:
        clips: The sequence's clips.
        analyses: Analyses by asset ID.

    Returns:
        One finding per cut, naming the nearest clean place instead when there
        is one close by.
    """
    found: List[Finding] = []
    for clip in sorted(clips, key=lambda item: item.timeline_in):
        analysis = analyses.get(clip.asset_id)
        if clip.volume == 0 or analysis is None:
            continue
        ends = (
            ("starts", float(clip.audio_source_start), float(clip.audio_timeline_in)),
            ("ends", float(clip.audio_source_end), float(clip.audio_timeline_out)),
        )
        for side, at, landed in ends:
            if at <= 0 or at >= analysis.duration:
                continue
            opens = side == "starts"
            if analysis.transcript is None or not _inside_speech(analysis, at, opens):
                edge = _bitten_sound(analysis, at, opens)
                if edge is not None:
                    where, part, move = ("after a sound begins", "start", "Start") if opens else (
                        "before a sound ends", "end", "End")
                    found.append(Finding("mid_speech", (
                        f"clip {clip.id} {side} {abs(at - edge):.2f}s {where}, at {clock(landed)} ({at:.2f}s of the "
                        f"file), so the {part} of it is cut off — the transcript may have no word there, but the "
                        f"sound was measured. {move} at {edge:.2f}s of the file to keep all of it"
                    ), ((landed, landed),)))
                continue
            known = clean_cuts(analysis)
            really = known.sound_before(at) if opens else known.sound_after(at)
            lost = known.heard_cut_off(at, opens)
            if really is not None:
                # What the sound says beats the nearest pause in a transcript that got this wrong.
                advice = f"; the talking really {'starts' if opens else 'stops'} at {really:.2f}s of the file"
            else:
                near = [point for point in _clean_points(analysis) if abs(point - at) <= CLEAN_SEARCH_SECONDS]
                advice = (
                    f"; the nearest pause is at {min(near, key=lambda point: abs(point - at)):.2f}s of the file"
                    if near else ""
                )
            found.append(Finding("mid_speech", (
                f"clip {clip.id} {side} in the middle of somebody talking, at {clock(landed)} "
                f"({at:.2f}s of the file)"
                + (f", and takes 「{lost}」 off, heard on a second listen" if lost else "")
                + advice
            ), ((landed, landed),)))
    return found

def _out_of_time(project: Project, recorded: Mapping[str, datetime]) -> List[Finding]:
    """Find a shot put on by hand between two from another moment.

    Args:
        project: The project.
        recorded: When each file was recorded, keyed by asset ID.

    Returns:
        One finding per stray run of clips.
    """
    base = project.base_video_track
    # A title card was shot at no moment: it borrows a frame, and stands between the shots either side.
    clips = sorted((clip for clip in (base.clips if base else []) if clip.card is None),
                   key=lambda clip: clip.timeline_in)
    parts = {float(marker.timeline_in) for marker in project.markers}
    shots = [
        Shot(
            label=f"clip {clip.id} ({clock(float(clip.timeline_in))})",
            moment=(recorded[clip.asset_id] + timedelta(seconds=float(clip.source_range.start))
                    if recorded.get(clip.asset_id) is not None else None),
            opens_part=float(clip.timeline_in) in parts,
            exempt=clip.from_plan_id is not None,
        )
        for clip in clips
    ]
    return [
        Finding("continuity", f"{stray}; it may be meant — a flash-forward — or dropped in by mistake",
                ((float(clips[first].timeline_in),) * 2, (float(clips[last].timeline_out),) * 2))
        for stray, (first, last) in stray_runs(shots)
    ]

def _repeats(clips: Sequence[Clip]) -> List[Finding]:
    """Find the same stretch of a file shown twice on the sequence.

    Args:
        clips: The sequence's clips.

    Returns:
        One finding per pair. Sometimes that is the point — a callback, a
        replay — which is why it is a finding and not a refusal.
    """
    found: List[Finding] = []
    # A title card's source range is only its length, so it shows nothing of its file twice.
    ordered = sorted((clip for clip in clips if clip.card is None), key=lambda item: item.timeline_in)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            if first.asset_id != second.asset_id:
                continue
            overlap = min(first.source_range.end, second.source_range.end) - max(
                first.source_range.start, second.source_range.start,
            )
            if float(overlap) >= REPEAT_SECONDS:
                found.append(Finding("repeated", (
                    f"clips {first.id} (at {clock(float(first.timeline_in))}) and {second.id} "
                    f"(at {clock(float(second.timeline_in))}) show the same {float(overlap):.1f}s of the same file"
                ), tuple((float(at),) * 2 for clip in (first, second) for at in (clip.timeline_in, clip.timeline_out))))
    return found

# Two shots side by side in one part this far apart in brightness, on 0 to 1, jump when one
# cuts to the other. Provisional.
LIGHT_STEP = 0.12
# And this far apart in colour cast. Provisional.
WARMTH_STEP = 0.03

def light_steps(project: Project, analyses: Mapping[str, MediaAnalysis]) -> List[str]:
    """Find where the picture jumps in brightness or colour from one shot to the next within a part.

    Brightness is what the footage measured plus what the clip's own
    adjustment adds, so a step already evened out by hand is not found again.
    For the shot further from the rest of its part, the brightness that would
    put it between its neighbours is worked out and offered; whether to use it
    is the caller's, and `check_render` measures again once it is set. A dip
    or a new part between two shots is a change the cut makes on purpose.

    Args:
        project: The cut.
        analyses: The analyses of its files, keyed by asset ID.

    Returns:
        One line per step.
    """
    base = project.base_video_track
    clips = sorted((clip for clip in (base.clips if base else []) if clip.card is None),
                   key=lambda clip: clip.timeline_in)
    parts = {float(marker.timeline_in) for marker in project.markers}
    measured: Dict[str, Tuple[float, Optional[float]]] = {}
    for clip in clips:
        analysis = analyses.get(clip.asset_id)
        if analysis is None:
            continue
        scores = combined_over(float(clip.source_range.start), float(clip.source_range.end), analysis.shots)
        if "exposure" in scores:
            lift = clip.color.brightness if clip.color is not None else 0.0
            measured[clip.id] = (scores["exposure"] + lift, scores.get("warmth"))
    starts = sorted(parts)

    def part_of(clip: Clip) -> int:
        return sum(1 for start in starts if start <= float(clip.timeline_in))

    found: List[str] = []
    for index in range(1, len(clips)):
        before, after = clips[index - 1], clips[index]
        # Measured against its own part only: a new part, wherever its marker falls, is a change made on purpose.
        if (before.id not in measured or after.id not in measured or part_of(before) != part_of(after)
                or float(after.timeline_in) in parts
                or (after.transition_in is not None and after.transition_in.kind == "dip")):
            continue
        (light_before, warm_before), (light_after, warm_after) = measured[before.id], measured[after.id]
        said = []
        if abs(light_after - light_before) >= LIGHT_STEP:
            # The one to change is the one further from how bright the rest of its part is, and it
            # is brought to between its neighbours in the part.
            part = [other for other in clips if other.id in measured and part_of(other) == part_of(after)]
            typical = statistics.median(measured[other.id][0] for other in part)
            odd_at = index - 1 if abs(light_before - typical) > abs(light_after - typical) else index
            odd = clips[odd_at]
            around = [measured[clips[other].id][0] for other in (odd_at - 1, odd_at + 1)
                      if 0 <= other < len(clips) and clips[other].id in measured
                      and part_of(clips[other]) == part_of(odd)]
            target = sum(around) / len(around)
            current = odd.color.brightness if odd.color is not None else 0.0
            lift = round(max(-1.0, min(1.0, current + target - measured[odd.id][0])), 2)
            said.append(f"brightness {light_before:.2f} → {light_after:.2f}; set_clip_look on clip {odd.id} with "
                        f"color brightness {lift:+.2f} puts it at {target:.2f}, between its neighbours")
        # A colour temperature set on either is the step dealt with: what kelvin does to the cast is
        # not measured here, so it is not checked again, and the next render is the one to look at.
        tempered = any(clip.color is not None and clip.color.temperature is not None for clip in (before, after))
        if (warm_before is not None and warm_after is not None and not tempered
                and abs(warm_after - warm_before) >= WARMTH_STEP):
            warmer = warm_after > warm_before
            said.append(f"clip {after.id} is {'warmer' if warmer else 'cooler'} by {abs(warm_after - warm_before):.2f}; "
                        f"a color temperature {'above' if warmer else 'below'} 6500 on it "
                        f"{'cools' if warmer else 'warms'} it towards clip {before.id}")
        if said:
            found.append(f"clip {before.id} → clip {after.id} at {clock(float(after.timeline_in))}: " + "; ".join(said))
    return found

def look_candidates(project: Project, analyses: Mapping[str, MediaAnalysis]) -> List[Clip]:
    """Name the clips of the sequence that only their picture tells apart.

    A clip somebody talks over is told apart by what they say, and one with
    words over the footage by those: neither can be mistaken for another that
    looks like it.

    Args:
        project: The project.
        analyses: The analyses of its files, keyed by asset ID.

    Returns:
        The clips worth fingerprinting, in timeline order.
    """
    base = project.base_video_track
    found = []
    for clip in sorted((clip for clip in (base.clips if base else []) if clip.card is None),
                       key=lambda clip: clip.timeline_in):
        analysis = analyses.get(clip.asset_id)
        said = clip.volume != 0 and analysis is not None and analysis.transcript is not None and any(
            segment.text.strip() and segment.start < clip.source_range.end and segment.end > clip.source_range.start
            for segment in analysis.transcript.segments
        )
        if not said and not clip.shown_texts:
            found.append(clip)
    return found

def _look_alikes(clips: Sequence[Clip], looks: Mapping[str, Sequence[int]]) -> List[Finding]:
    """Find two clips from different files that look so alike a viewer could not tell which is which.

    Two side by side play as one shot, which is not a question of telling them
    apart; anything between them that is not alike is what a viewer would go by.

    Args:
        clips: The sequence's clips.
        looks: Fingerprints by clip ID, of the clips only their picture tells apart.

    Returns:
        One finding per pair.
    """
    found: List[Finding] = []
    ordered = sorted((clip for clip in clips if clip.card is None), key=lambda item: item.timeline_in)
    # A card just before a clip, or a transition into it, marks it off from what came before.
    carded = {float(clip.timeline_out) for clip in clips if clip.card is not None}
    marked = {clip.id for clip in ordered
              if clip.transition_in is not None or any(abs(float(clip.timeline_in) - out) < 0.01 for out in carded)}
    for index, first in enumerate(ordered):
        for second in ordered[index + 2:]:
            if (first.asset_id == second.asset_id or first.id not in looks or second.id not in looks
                    or second.id in marked):
                continue
            same, bits = alike(looks[first.id], looks[second.id])
            if same:
                found.append(Finding("look_alike", (
                    f"clips {first.id} (at {clock(float(first.timeline_in))}) and {second.id} "
                    f"(at {clock(float(second.timeline_in))}) are different files that look alike ({bits} of 64 "
                    "bits apart), and nothing said or written over them tells them apart; give the second a line, "
                    "words over it, a card before it that says which it is, or a transition into it"
                ), tuple((float(at),) * 2 for clip in (first, second) for at in (clip.timeline_in, clip.timeline_out))))
    return found

# Moments offered from one shot are at least this far apart. Provisional.
COVER_APART_SECONDS = 0.5
# Looking closely at one moment: this far either side of it, a frame every this often. Provisional.
COVER_REACH_SECONDS = 2.0
COVER_STEP_SECONDS = 0.25

def _shows_picture(analysis: Optional[MediaAnalysis]):
    """Build a test for whether a moment of a file is worth showing: not black, not frozen."""
    bad = [*analysis.black_frames, *analysis.frozen_frames] if analysis else []
    return lambda seconds: not any(span.start <= seconds < span.end for span in bad)

def _favoured_moment(analysis: Optional[MediaAnalysis], inner: Tuple[float, float], clean) -> Optional[float]:
    """Pick the moment of a shot its measurements favour: the largest face, else the sharpest stretch."""
    if analysis is not None:
        faced = [
            (second.face_share or 0.0, -second.start, min(max((second.start + second.end) / 2, inner[0]), inner[1]))
            for second in analysis.faces
            if second.faces and second.end > inner[0] and second.start < inner[1]
        ]
        faced = [item for item in faced if clean(item[2])]
        if faced:
            return max(faced)[2]
        sharp = [
            (shot.blur, min(max((shot.start + shot.end) / 2, inner[0]), inner[1]))
            for shot in analysis.shots
            if shot.blur is not None and shot.end > inner[0] and shot.start < inner[1]
        ]
        sharp = [item for item in sharp if clean(item[1])]
        if sharp:
            return min(sharp)[1]
    middle = (inner[0] + inner[1]) / 2
    return middle if clean(middle) else None

def cover_candidates(
    project: Project,
    analyses: Mapping[str, MediaAnalysis],
    per_shot: int = 1,
) -> List[Tuple[Clip, float]]:
    """Offer frames of every shot as a cover, never one of black or frozen picture.

    Which frame makes the cover is a judgement about what the video is for,
    and no measurement here knows that: the second with the largest face in a
    shot is the right cover for a talking head and the wrong one for a video
    about the thing in the speaker's hands. So this only lays the field out —
    the moment the shot's measurements favour (the largest face, or where
    nobody is seen, the sharpest stretch), and, asked for more, moments spread
    evenly through the shot beside it — and whoever is choosing looks.

    Args:
        project: The project.
        analyses: The analyses of its files, keyed by asset ID.
        per_shot: How many moments to offer from each shot.

    Returns:
        `(clip, source_seconds)` per moment, in timeline order. Full-frame
        covering picture counts as a shot: a good cover is often B-roll.
    """
    # A title card is a frame borrowed from a shot, blurred: never what a cover should be.
    clips = [clip for clip in project.base_video_track.clips if clip.card is None] if project.base_video_track else []
    clips += [clip for track in project.video_tracks[1:] for clip in track.clips if clip.layout is None]
    offered: List[Tuple[Clip, float]] = []
    for clip in sorted(clips, key=lambda item: item.timeline_in):
        start, end = float(clip.source_range.start), float(clip.source_range.end)
        # Clear of the cut at either end, where a frame may still belong to the shot before.
        inner = (start + min(0.25, (end - start) / 4), end - min(0.25, (end - start) / 4))
        analysis = analyses.get(clip.asset_id)
        clean = _shows_picture(analysis)
        moments = [moment for moment in (_favoured_moment(analysis, inner, clean),) if moment is not None]
        count = max(1, per_shot)
        # Each further moment is the clean one farthest from those already offered, so they
        # spread through whatever of the shot is worth showing.
        tried = [inner[0] + (inner[1] - inner[0]) * (index + 0.5) / (4 * count) for index in range(4 * count)]
        while len(moments) < count:
            apart = [
                (min((abs(seconds - moment) for moment in moments), default=math.inf), seconds)
                for seconds in tried if clean(seconds)
            ]
            best = max(apart, default=None)
            if best is None or best[0] < COVER_APART_SECONDS:
                break
            moments.append(best[1])
        offered += [(clip, round(moment, 3)) for moment in sorted(moments)]
    return offered

def on_screen(project: Project, seconds: float) -> Optional[Clip]:
    """Find the picture the viewer sees at one moment of the cut: the topmost full-frame clip, or None."""
    showing = [
        clip for track in project.video_tracks for clip in track.clips
        if (track is project.base_video_track or clip.layout is None)
        and float(clip.timeline_in) <= seconds < float(clip.timeline_out)
    ]
    return showing[-1] if showing else None

def frames_around(
    project: Project,
    analyses: Mapping[str, MediaAnalysis],
    at: float,
    reach: float = COVER_REACH_SECONDS,
    step: float = COVER_STEP_SECONDS,
) -> List[Tuple[Clip, float, float]]:
    """Offer every few frames either side of one moment of the cut, for choosing the exact one.

    A shot that is the right subject still has frames where the hand is a
    blur or the eyes are shut, and which one reads at a glance is only seen
    by looking at them side by side.

    Args:
        project: The project.
        analyses: The analyses of its files, keyed by asset ID.
        at: The moment of the cut, in seconds.
        reach: How far either side of it to look.
        step: How far apart the frames are, in seconds of the cut.

    Returns:
        `(clip, source_seconds, cut_seconds)` for each frame shown then — the
        topmost full-frame picture, as `export_cover` saves it — never a card
        or a frame of black or frozen picture.
    """
    found: List[Tuple[Clip, float, float]] = []
    steps = int(round(reach / step))
    for index in range(-steps, steps + 1):
        seconds = round(at + index * step, 3)
        if not 0 <= seconds < float(project.duration):
            continue
        clip = on_screen(project, seconds)
        if clip is None or clip.card is not None:
            continue
        source = float(clip.source_range.start) + (seconds - float(clip.timeline_in)) * clip.speed
        if _shows_picture(analyses.get(clip.asset_id))(source):
            found.append((clip, round(source, 3), seconds))
    return found

# Stretches of the cut this close together are said as one. Provisional.
NOTICE_JOIN_SECONDS = 1.0

@dataclass(frozen=True)
class RightsFact:
    """Something in a cut its publisher may need permission for, or has to say.

    Attributes:
        kind: `music` for a song laid under the cut, `recording` for any
            other sound laid on a sound track — a song filed with the footage,
            a voice-over, a sound effect — `other_voice` for
            somebody heard who is not the main speaker, `unchecked_voices` for
            talking nobody told the voices apart in, and `faces` for more than
            one face on screen at once.
        subject: What it is — the file, the voice's label.
        spans: Where in the cut, in seconds, joined where close together.
        detail: What is known about it: where it came from and on what terms,
            what the user said about it, and what the file says about itself.
    """

    kind: str
    subject: str
    spans: Tuple[Tuple[float, float], ...]
    detail: Tuple[str, ...] = ()

def rights_facts(
    project: Project,
    analyses: Mapping[str, MediaAnalysis],
    assets: Mapping[str, Asset],
    tags: Mapping[str, Mapping[str, str]],
) -> List[RightsFact]:
    """List what in a cut its publisher may need permission for, or has to disclose.

    Only facts: which songs play under it and what is known of them, who else
    is heard, where more than one face is on screen. Whether any of it is a
    problem depends on where the video is going and what the user agreed with
    whom, and is theirs to judge — this is what they need in front of them to
    judge it.

    Args:
        project: The cut.
        analyses: The analyses of its files, keyed by asset ID.
        assets: Its files, keyed by asset ID.
        tags: What each file's own metadata says — title, artist, copyright,
            licence — keyed by asset ID.

    Returns:
        The facts, music first, in the order each first comes in.
    """
    facts: List[RightsFact] = []
    audio = [track for track in project.tracks if track.track_type == TrackType.AUDIO]
    for asset_id in dict.fromkeys(clip.asset_id for track in audio for clip in track.clips if clip.volume != 0):
        asset = assets.get(asset_id)
        if asset is None:
            continue
        spans = _joined_spans((float(clip.timeline_in), float(clip.timeline_out)) for track in audio
                              for clip in track.clips if clip.asset_id == asset_id and clip.volume != 0)
        known = []
        if asset.source is not None:
            known += [f"from: {asset.source.site} {asset.source.page}", f"licence: {asset.source.licence}"]
            known += [f"credit it asks for: {asset.source.credit}"] if asset.source.credit else []
            known += ["taken for a video only the user watches"] if asset.source.personal_only else []
        if asset.licence:
            known.append(f"the user said: {asset.licence}")
        if asset.source is None and not asset.licence:
            known.append("licence: nobody has said")
        known += [f"{key}: {value}" for key, value in tags.get(asset_id, {}).items() if value]
        # Every recording laid on the sound tracks: a song filed with the footage is still a song.
        kind = "music" if library_of(asset) == MUSIC else "recording"
        facts.append(RightsFact(kind, os.path.basename(asset.path), spans, tuple(known)))

    tracks = ([project.base_video_track] if project.base_video_track else []) + audio
    heard = [clip for track in tracks for clip in track.clips
             if clip.volume != 0 and clip.card is None and clip.asset_id in assets
             and library_of(assets[clip.asset_id]) != MUSIC]
    joined = join_voices(analyses)
    talk: dict = {}
    for clip in heard:
        analysis = analyses.get(clip.asset_id)
        for turn in analysis.speakers if analysis is not None else []:
            landed = _on_timeline(clip, turn.start, turn.end)
            if landed is not None:
                label = joined.get((clip.asset_id, turn.speaker), f"{clip.asset_id}:{turn.speaker}")
                talk.setdefault(label, []).append(landed)
    unchecked = [clip for clip in heard if (analysis := analyses.get(clip.asset_id)) is not None
                 and analysis.transcript is not None and analysis.transcript.segments and not analysis.speakers]
    if unchecked:
        # Not knowing who else talks is not the same as nobody else talking.
        facts.append(RightsFact("unchecked_voices", "talking whose voices were never told apart", _joined_spans(
            (float(clip.timeline_in), float(clip.timeline_out)) for clip in unchecked)))
    if len(talk) > 1:
        main = max(talk, key=lambda label: sum(end - start for start, end in talk[label]))
        for label, stretches in talk.items():
            if label != main:
                facts.append(RightsFact("other_voice", label, _joined_spans(stretches),
                                        (f"the main speaker is {main}",)))

    crowded = []
    base = project.base_video_track
    shown = list(base.clips) if base else []
    shown += [clip for track in project.video_tracks[1:] for clip in track.clips if clip.layout is None]
    for clip in shown:
        analysis = analyses.get(clip.asset_id)
        for second in analysis.faces if analysis else []:
            if second.faces > 1 and (landed := _on_timeline(clip, second.start, second.end)) is not None:
                crowded.append(landed)
    if crowded:
        facts.append(RightsFact("faces", "more than one face on screen", _joined_spans(crowded)))
    return facts

def _joined_spans(spans) -> Tuple[Tuple[float, float], ...]:
    """Join stretches of the cut that touch or nearly do, rounded to tenths."""
    joined: List[List[float]] = []
    for start, end in sorted(spans):
        if joined and start - joined[-1][1] <= NOTICE_JOIN_SECONDS:
            joined[-1][1] = max(joined[-1][1], end)
        else:
            joined.append([start, end])
    return tuple((round(start, 1), round(end, 1)) for start, end in joined)
