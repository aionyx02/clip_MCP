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
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from app.engine.builder import DUCK_DEPTH_DB, Talking
from app.engine.plan import LENGTH_TOLERANCE
from app.engine.semantic import share_covered, was_audible
from app.engine.subtitles import caption_overflow, captioned_clips
from app.models.media import MediaAnalysis
from app.models.timeline import CaptionStyle, Clip, PlacedCue, Project

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
CHECKS = ("bad_picture", "clipping", "music", "mid_speech", "repeated", "unplanned", "captions", "length")
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
# How much of the same stretch of a file has to come back before it is the same shot shown
# twice rather than two neighbouring moments. Provisional.
REPEAT_SECONDS = 1.0
# A sequence of this many clips is an edit, and an edit with no plan has nothing that says
# how it begins, turns and ends. Fewer is a trim. Provisional.
UNPLANNED_CLIPS = 3

@dataclass(frozen=True)
class Finding:
    """One thing about a cut that would be noticed in the finished file.

    Attributes:
        check: Which check found it: one of `CHECKS`.
        message: What it is and where, for whoever decides.
    """

    check: str
    message: str

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
    duration = float(project.duration)
    marks = sorted(
        (marker for marker in project.markers if float(marker.timeline_in) < duration),
        key=lambda marker: marker.timeline_in,
    )
    listed = [
        (float(marker.timeline_in), float(marks[index + 1].timeline_in) if index + 1 < len(marks) else duration,
         marker.name)
        for index, marker in enumerate(marks)
    ]
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
        if analysis is None:
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
                    )))

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
    if base and len(base.clips) >= UNPLANNED_CLIPS and any(clip.from_plan_id is None for clip in base.clips):
        findings.append(Finding("unplanned", (
            f"these {len(base.clips)} clips were put together by hand, not compiled from a plan, so nothing "
            "says what the video is for or how it opens, turns and ends. Write a plan whose parts are a hook, "
            "a setup, a turn and a payoff, and compile it"
        )))

    if captions:
        for cue in caption_overflow(captions, project.width, project.height, style):
            findings.append(Finding("captions", (
                f"the caption at {clock(float(cue.start))} ({cue.text[:16]}) is too tall for a "
                f"{project.width}x{project.height} frame and runs off the top"
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
            if clip.volume == 0 or not any(start < closed and end > opened for start, end in talking.spans):
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
    segments = analysis.transcript.segments if analysis.transcript else []
    if not analysis.silences:
        return list(segments)
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

def _inside_speech(analysis: MediaAnalysis, at: float) -> bool:
    """Tell whether a cut at one moment of a file lands in the middle of somebody talking.

    Args:
        analysis: The file's analysis.
        at: The moment, in source seconds.

    Returns:
        True inside a word, or between two words of one sentence closer
        together than a pause.
    """
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
        if clip.volume == 0 or analysis is None or analysis.transcript is None:
            continue
        ends = (
            ("starts", float(clip.audio_source_start), float(clip.audio_timeline_in)),
            ("ends", float(clip.audio_source_end), float(clip.audio_timeline_out)),
        )
        for side, at, landed in ends:
            if at <= 0 or at >= analysis.duration or not _inside_speech(analysis, at):
                continue
            near = [point for point in _clean_points(analysis) if abs(point - at) <= CLEAN_SEARCH_SECONDS]
            advice = (
                f"; the nearest pause is at {min(near, key=lambda point: abs(point - at)):.2f}s of the file"
                if near else ""
            )
            found.append(Finding("mid_speech", (
                f"clip {clip.id} {side} in the middle of somebody talking, at {clock(landed)} "
                f"({at:.2f}s of the file){advice}"
            )))
    return found

def _repeats(clips: Sequence[Clip]) -> List[Finding]:
    """Find the same stretch of a file shown twice on the sequence.

    Args:
        clips: The sequence's clips.

    Returns:
        One finding per pair. Sometimes that is the point — a callback, a
        replay — which is why it is a finding and not a refusal.
    """
    found: List[Finding] = []
    ordered = sorted(clips, key=lambda item: item.timeline_in)
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
                )))
    return found

def cover_candidates(
    project: Project,
    analyses: Mapping[str, MediaAnalysis],
) -> List[Tuple[Clip, float]]:
    """Offer one frame per shot as a cover, the one its measurements favour.

    Which frame makes the cover is a judgement, so this only narrows the
    field — one candidate per shot, never a frame of black or frozen picture —
    and whoever chooses looks at them. Within a shot the pick is an argmax,
    not a threshold: the second with the largest face in it, or where nobody
    is seen, the middle of the sharpest stretch.

    Args:
        project: The project.
        analyses: The analyses of its files, keyed by asset ID.

    Returns:
        `(clip, source_seconds)` per shot, in timeline order. Full-frame
        covering picture counts as a shot: a good cover is often B-roll.
    """
    clips = list(project.base_video_track.clips) if project.base_video_track else []
    clips += [clip for track in project.video_tracks[1:] for clip in track.clips if clip.layout is None]
    offered: List[Tuple[Clip, float]] = []
    for clip in sorted(clips, key=lambda item: item.timeline_in):
        start, end = float(clip.source_range.start), float(clip.source_range.end)
        # Clear of the cut at either end, where a frame may still belong to the shot before.
        inner = (start + min(0.25, (end - start) / 4), end - min(0.25, (end - start) / 4))
        analysis = analyses.get(clip.asset_id)
        bad = [*analysis.black_frames, *analysis.frozen_frames] if analysis else []
        clean = lambda seconds: not any(span.start <= seconds < span.end for span in bad)
        choice: Optional[float] = None
        if analysis is not None:
            faced = [
                (second.face_share or 0.0, -second.start, min(max((second.start + second.end) / 2, inner[0]), inner[1]))
                for second in analysis.faces
                if second.faces and second.end > inner[0] and second.start < inner[1]
            ]
            faced = [item for item in faced if clean(item[2])]
            if faced:
                choice = max(faced)[2]
            else:
                sharp = [
                    (shot.blur, min(max((shot.start + shot.end) / 2, inner[0]), inner[1]))
                    for shot in analysis.shots
                    if shot.blur is not None and shot.end > inner[0] and shot.start < inner[1]
                ]
                sharp = [item for item in sharp if clean(item[1])]
                if sharp:
                    choice = min(sharp)[1]
        if choice is None:
            middle = (inner[0] + inner[1]) / 2
            if not clean(middle):
                continue
            choice = middle
        offered.append((clip, round(choice, 3)))
    return offered
