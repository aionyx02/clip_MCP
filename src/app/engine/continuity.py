"""Shots that do not belong where they sit: from another time, or about something else.

Whether a cut tells its story well is a judgement, and the server does not
make it. What it can measure is the commonest way a story breaks: a shot
dropped between two that belong together — the next morning's coffee in the
middle of last night's dinner, the owner's meeting in the middle of the test
prints. Two things say a shot belongs with its neighbours: when it was shot,
and which topic the footage was sorted into. Where the neighbours agree with
each other on either one and the shot between them does not, it is named, and
whoever wrote the plan has to say why it is there or move it.

A jump where a new part of the video begins is how parts begin, so a shot
that opens a part is never named, and neither is anything in the hook, which
exists to show what comes later.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

# Two shots this close in time belong to the same moment. Provisional.
NEAR = timedelta(minutes=60)
# A shot this far from both neighbours is from another moment altogether. Provisional.
FAR = timedelta(hours=3)
# How many shots in a row can be the stray between two that belong together. Past this
# it is a sequence of its own, not something dropped in. Provisional.
MAX_STRAY_RUN = 2
# A part that goes back this far in time from one shot to the next has gone back to another
# moment, as a night opening a part and then the day before it does. Provisional.
BACKWARD = timedelta(hours=3)
# With no time to go by, a shot this dark followed in the same part by one this bright looks
# like night going back to day. Mean brightness, 0 to 1; indoors to outdoors looks the same,
# which is why it is only ever a note. Provisional.
NIGHT_EXPOSURE = 0.2
DAY_EXPOSURE = 0.4

@dataclass(frozen=True)
class Shot:
    """One shot in the order the video plays it.

    Attributes:
        label: What to call it in a message: its clip or selection.
        moment: When it was shot, or None when nothing says.
        topic: The topic its footage was sorted into, or None.
        opens_part: Whether a new part of the video begins with it.
        exempt: Whether it is allowed to be anywhere: in the hook, or with
            its reason for being there already given.
        exposure: Its mean brightness, 0 to 1, or None when not measured.
    """

    label: str
    moment: Optional[datetime] = None
    topic: Optional[str] = None
    opens_part: bool = False
    exempt: bool = False
    exposure: Optional[float] = None

def _clock(moment: datetime) -> str:
    """Write a moment the way a person reads it, in this machine's time zone."""
    return moment.astimezone().strftime("%m/%d %H:%M")

def _apart(moment: datetime, low: datetime, high: datetime) -> timedelta:
    """How far a moment is from a stretch of time; zero inside it."""
    if moment < low:
        return low - moment
    return moment - high if moment > high else timedelta(0)

def strays(shots: Sequence[Shot]) -> List[str]:
    """Name the shots dropped between two that belong together.

    Args:
        shots: The video's shots in the order they play.

    Returns:
        One message per stray run of shots, saying what its neighbours have
        in common and what it has instead.
    """
    return [message for message, _ in stray_runs(shots)]

def stray_runs(shots: Sequence[Shot]) -> List[Tuple[str, Tuple[int, int]]]:
    """Name the shots dropped between two that belong together, and say which they are.

    Args:
        shots: The video's shots in the order they play.

    Returns:
        One `(message, (first, last))` per stray run, `first` and `last`
        being the positions in `shots` of the run's first and last shot.
    """
    found: List[Tuple[str, Tuple[int, int]]] = []
    index = 1
    while index < len(shots) - 1:
        for run in range(1, MAX_STRAY_RUN + 1):
            if index + run >= len(shots):
                break
            before, middle, after = shots[index - 1], shots[index:index + run], shots[index + run]
            # A part that begins here, or begins right after, is a new part, not an insertion.
            if any(shot.opens_part or shot.exempt for shot in middle) or after.opens_part:
                continue
            reason = _by_time(before, middle, after) or _by_topic(before, middle, after)
            if reason:
                names = ", ".join(shot.label for shot in middle)
                found.append((f"{names} sits between {before.label} and {after.label}: {reason}",
                              (index, index + run - 1)))
                index += run
                break
        index += 1
    return found

def _by_time(before: Shot, middle: Sequence[Shot], after: Shot) -> Optional[str]:
    """Say how a run is out of time with its neighbours, if it is."""
    if before.moment is None or after.moment is None or any(shot.moment is None for shot in middle):
        return None
    low, high = sorted((before.moment, after.moment))
    if high - low > NEAR:
        return None
    if all(_apart(shot.moment, low, high) >= FAR for shot in middle):
        shot_at = ", ".join(_clock(shot.moment) for shot in middle)
        return (f"they were shot {_clock(low)}–{_clock(high)}, "
                f"and what is between them at {shot_at}")
    return None

def _by_topic(before: Shot, middle: Sequence[Shot], after: Shot) -> Optional[str]:
    """Say how a run is off the topic its neighbours share, if it is."""
    if before.topic is None or before.topic != after.topic:
        return None
    if all(shot.topic is not None and shot.topic != before.topic for shot in middle):
        return f"both are about 「{before.topic}」, and what is between them about 「{middle[0].topic}」"
    return None

def backward_steps(shots: Sequence[Shot]) -> Tuple[List[str], List[Tuple[str, Tuple[str, str]]]]:
    """Name the places a part goes back in time from one shot to the next.

    A part that opens on the night and then cuts to that afternoon tells it out
    of order, and no shot is a stray between two that agree, so `stray_runs`
    never sees it. The shot a part opens on is where it starts, not a step
    back; a shot that is exempt, or a structure told backwards, is left to the
    plan to have meant.

    Args:
        shots: The video's shots in the order they play.

    Returns:
        `(by_time, by_light)`: steps back the times say, and, where the times
        do not say, steps from a dark shot to a bright one that might be night
        going back to day, each with the labels of its two shots.
    """
    by_time: List[str] = []
    by_light: List[Tuple[str, Tuple[str, str]]] = []
    # A shot let be out of order, or already named as a stray, is not what the part is measured
    # against: coming back from it is coming back to the part's own time.
    aside = {index for _, (first, last) in stray_runs(shots) for index in range(first, last + 1)}
    before: Optional[Shot] = None
    for index, after in enumerate(shots):
        if after.opens_part:
            before = None
        if after.exempt or index in aside:
            continue
        if before is not None and before.moment is not None and after.moment is not None:
            if before.moment - after.moment >= BACKWARD:
                by_time.append(f"{after.label} was shot at {_clock(after.moment)}, before {before.label} "
                               f"({_clock(before.moment)}) that comes ahead of it in the same part")
        elif (before is not None and before.exposure is not None and after.exposure is not None
              and before.exposure <= NIGHT_EXPOSURE and after.exposure >= DAY_EXPOSURE):
            by_light.append((f"{before.label} is dark ({before.exposure:.2f}) and {after.label}, after it in the "
                             f"same part, bright ({after.exposure:.2f}); nothing says when they were shot",
                             (before.label, after.label)))
        before = after
    return by_time, by_light
