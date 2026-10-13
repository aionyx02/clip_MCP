"""Where one picture gives way to another in a cut, and what is said across it.

A storyboard shows what each shot looks like; it cannot show how one shot
hands over to the next — whether a movement carries through, whether a
sentence is finished before the picture moves on. That happens in the second
either side of a join, so a join is what gets looked at, every kind of it: a
plain cut, a transition, a picture laid over the sequence and taken off
again, the start of a new part.

What is said there comes from the transcripts, laid onto the cut's time: the
words the viewer should hear, and which of them a cut takes into. It is what
the footage holds, not what the render sounds like; the render's own sound is
measured separately, by level, and heard again with `listen_again`.
"""

from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

from app.engine.delivery import _said
from app.engine.subtitles import captioned_clips
from app.models.media import MediaAnalysis
from app.models.timeline import Clip, Project


@dataclass(frozen=True)
class Join:
    """One place in a cut where the picture changes.

    Attributes:
        at: Where it happens, in seconds of the cut.
        kind: `cut`, the transition's kind (`dissolve`, `wipe`, `dip`),
            `b-roll in` or `b-roll out`.
        before: What is on screen up to it.
        after: What is on screen from it.
        part: The name of the part of the video that begins here, if one does.
        sound_at: Where the sound changes over, when a J or an L cut moves it
            off the picture's cut; None when the two change together.
    """

    at: float
    kind: str
    before: str
    after: str
    part: Optional[str] = None
    sound_at: Optional[float] = None


@dataclass(frozen=True)
class HeardWord:
    """A word of the footage, where the cut plays it.

    Attributes:
        start: Where it starts, in seconds of the cut.
        end: Where it ends.
        text: The word.
        clip_id: The clip it is heard in.
        cut: Whether the clip's sound starts or stops partway through it.
    """

    start: float
    end: float
    text: str
    clip_id: str
    cut: bool


def _shown(clip: Clip, names: Mapping[str, str]) -> str:
    """Name what a clip puts on screen: its file and the stretch of it, or its title."""
    if clip.card is not None:
        return f"clip {clip.id}: title card 「{clip.card.title}」"
    start, end = float(clip.source_range.start), float(clip.source_range.end)
    return f"clip {clip.id}: {names.get(clip.asset_id, clip.asset_id)} {start:.2f}–{end:.2f}s"


def joins_in(project: Project, names: Mapping[str, str]) -> List[Join]:
    """List every join of a cut, in the order they play.

    Args:
        project: The cut.
        names: What to call each file, keyed by asset ID.

    Returns:
        One join where each clip of the sequence follows another, and one
        where each picture laid over the sequence comes in and goes off.
        Where both happen at once they are listed apart, sequence first.
    """
    parts = {round(float(marker.timeline_in), 3): marker.name for marker in project.markers}
    found: List[Tuple[float, int, Join]] = []
    base = project.base_video_track
    sequence = sorted(base.clips if base else [], key=lambda clip: clip.timeline_in)
    for before, after in zip(sequence, sequence[1:]):
        at = float(after.timeline_in)
        sound = float(after.audio_timeline_in)
        found.append((at, 0, Join(
            at=at,
            kind=after.transition_in.kind if after.transition_in is not None else "cut",
            before=_shown(before, names),
            after=_shown(after, names),
            part=parts.get(round(at, 3)),
            sound_at=sound if abs(sound - at) > 1e-3 else None,
        )))

    def under(at: float) -> str:
        """Name what the sequence shows at one moment, where the laid-over picture is not."""
        showing = [clip for clip in sequence if float(clip.timeline_in) <= at < float(clip.timeline_out)]
        return _shown(showing[0], names) if showing else "nothing"

    for track in project.video_tracks[1:]:
        for clip in track.clips:
            # A picture inside the frame leaves the sequence on screen around it.
            laid = _shown(clip, names) + (" (inset)" if clip.layout is not None else "")
            opens, closes = float(clip.timeline_in), float(clip.timeline_out)
            found.append((opens, 1, Join(at=opens, kind="b-roll in", before=under(opens - 1e-3), after=laid)))
            found.append((closes, 1, Join(at=closes, kind="b-roll out", before=laid, after=under(closes))))
    return [join for _, _, join in sorted(found, key=lambda item: (item[0], item[1]))]


def words_heard(
    project: Project,
    analyses: Mapping[str, MediaAnalysis],
    start: float,
    end: float,
) -> List[HeardWord]:
    """Lay the transcribed words of every clip heard in a stretch of the cut onto its time.

    Args:
        project: The cut.
        analyses: The analyses of its files, keyed by asset ID.
        start: Where the stretch starts, in seconds of the cut.
        end: Where it ends.

    Returns:
        The words, in the order they are heard: from the sequence and the
        sound tracks, less any clip turned all the way down and any sentence
        nobody spoke, by the rules the captions follow. A word only partly
        inside its clip's sound is kept, marked as cut.
    """
    heard: List[HeardWord] = []
    for clip in captioned_clips(project):
        analysis = analyses.get(clip.asset_id)
        if analysis is None or clip.card is not None:
            continue
        opened, closed = float(clip.audio_source_start), float(clip.audio_source_end)
        into = lambda seconds: float(clip.audio_timeline_in) + (seconds - opened) / clip.speed
        for segment in _said(analysis):
            for word in segment.words:
                if word.end <= opened or word.start >= closed:
                    continue
                lands = (into(max(word.start, opened)), into(min(word.end, closed)))
                if lands[1] <= start or lands[0] >= end:
                    continue
                heard.append(HeardWord(
                    start=lands[0], end=lands[1], text=word.text.strip(), clip_id=clip.id,
                    cut=word.start < opened or word.end > closed,
                ))
    return sorted(heard, key=lambda word: word.start)
