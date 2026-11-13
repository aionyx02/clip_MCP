"""Where a cut's story may go wrong, nominated from the order its shots actually play in.

A senior editor watching a rough cut flags a handful of things again and again:
you said goodnight and kept going; we jumped back to this morning; we left that
subject and came back to it; it was a minute before I knew what this was about.
Each of those can also be the point — a hook shows the ending first, a flashback
goes back on purpose — so nothing here judges. It nominates generously, by rule,
and the model rules on every nomination with a reason; the server's part is only
that nothing gets past unruled.

What it reads is the project's sequence as it plays, not the plan, so a clip put
in by hand is read as much as a compiled one, and `read_cut` reads the same
thing. Where what a kind needs is missing — no topics because the
timeline was never sectioned, no recording times, clips with no origin — the
kind says it was not checked and why, rather than coming back empty, which
would read as nothing wrong.

Pure: the tools gather the shots, this works them out.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

# Provisional: words a video ends on. After one of them, this many seconds more of the cut is
# a video that said goodbye and went on — the cup-rack cut said 晚安, then went back to a
# failure and succeeded a second time. An end card or a last look is shorter than this.
# Not measured against anyone watching.
AFTER_CLOSING = 10.0
# Provisional: the closing words listed, as transcripts write them; generous on purpose,
# since a nomination is only a question. Matched with spaces and punctuation taken out,
# letters lowered.
CLOSING_WORDS = (
    "晚安", "掰掰", "拜拜", "謝謝收看", "謝謝觀看", "謝謝大家", "今天就到這", "今天先到這", "今天的影片就到這",
    "我們下次見", "下次見", "下集見", "下支影片見", "下期見", "收工",
    "goodnight", "byebye", "seeyou",
)
# Provisional: two neighbouring shots whose recording goes back further than this are
# nominated, whatever part they are in. Stricter than the plan check's three hours
# (`checks.BACKWARD`), which refuses: this only asks, so an afternoon told out of order is
# asked about too.
NOMINATE_BACK = timedelta(minutes=20)
# Provisional: the first words landing later than this many seconds, or this share of the
# cut, whichever is sooner, leave the viewer waiting to hear what the video is about; the
# cup-rack cut said nothing for 27 s. Only words: a plan's first established fact was tried
# and measured the wrong thing (0 of 9 on real cuts), since it is a story fact, not the subject.
LATE_SECONDS = 15.0
LATE_SHARE = 0.15

KINDS = ("closing_too_early", "back_in_time", "topic_returns", "late_subject")


@dataclass(frozen=True)
class Shot:
    """One clip of the sequence as the story reads it.

    Attributes:
        clip_id: The clip's ID in the project.
        start: When it starts playing in the cut, in seconds.
        end: When it stops.
        words: Each word heard over it, as `(second of the cut, text)`, in order.
            Empty when nothing is said, the clip is muted or it is a card.
        heard: Whether its file has a transcript to read words from; False for
            a file never transcribed, True for a card or a muted clip, which
            are silent either way.
        moment: When it was recorded; None for a card or a file with no
            recording time.
        topics: The subjects its footage was sectioned under, in the order
            it plays them: one clip can run on over pieces of several
            sections. Empty when it has none, or the clip has no origin to
            look them up by.
        sourced: Whether the clip came from footage the semantic timeline
            knows, so its topics can be looked up.
        card: Whether it is a title card.
    """

    clip_id: str
    start: float
    end: float
    words: Tuple[Tuple[float, str], ...] = ()
    heard: bool = True
    moment: Optional[datetime] = None
    topics: Tuple[str, ...] = ()
    sourced: bool = False
    card: bool = False


@dataclass(frozen=True)
class Doubt:
    """One nomination: a place the story may go wrong, for the model to rule on.

    Attributes:
        id: Stable from its kind and the clips it is about, so the same
            nomination keeps its ID while the sequence does not change it.
        kind: One of `KINDS`.
        clip_ids: The clips it is about, in the order they play.
        at: The second of the cut it is about.
        says: What was found, in words, with the numbers.
    """

    id: str
    kind: str
    clip_ids: Tuple[str, ...]
    at: float
    says: str


@dataclass
class Doubts:
    """Everything nominated over a sequence, and what could not be checked.

    Attributes:
        doubts: The nominations, in the order they play.
        unchecked: Per kind, or part of a kind, why it was not checked or
            only partly: no topics, no recording times, clips with no origin.
            A kind not named here was checked over the whole cut.
    """

    doubts: List[Doubt] = field(default_factory=list)
    unchecked: Dict[str, str] = field(default_factory=dict)


def sequence_hash(order: Sequence[Tuple]) -> str:
    """Name a sequence by the order of its shots and the stretch of file each plays.

    Rulings are tied to it: the captions or the music changing leaves it as it
    is, moving, trimming or swapping a shot — in the editor too — changes it.

    Args:
        order: Per shot of the sequence, in play order, what it plays: such as
            `(asset ID, source start, source end)`.

    Returns:
        A short hash.
    """
    return hashlib.sha256(json.dumps([list(item) for item in order]).encode("utf-8")).hexdigest()[:16]


def _bare(text: str) -> str:
    """Text with spaces and punctuation taken out and letters lowered, as closing words are matched."""
    return re.sub(r"[\s\W_]+", "", text).lower()


def _clock(seconds: float) -> str:
    """A second of the cut as minutes and seconds."""
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def _closing(shots: Sequence[Shot], length: float) -> List[Doubt]:
    """Nominate every closing word with more than `AFTER_CLOSING` seconds of the cut after it."""
    found = []
    for shot in shots:
        said = ""
        ends: List[Tuple[int, float]] = []
        for at, text in shot.words:
            said += _bare(text)
            ends.append((len(said), at))
        for word in CLOSING_WORDS:
            place = said.find(word)
            if place < 0:
                continue
            # The word of the cut the closing word ends in.
            at = next((second for end, second in ends if end >= place + len(word)), shot.end)
            after = length - at
            if after > AFTER_CLOSING:
                found.append(Doubt(
                    id=f"closing_too_early:{shot.clip_id}", kind="closing_too_early", clip_ids=(shot.clip_id,),
                    at=at, says=f"「{word}」 at {_clock(at)} in clip {shot.clip_id}, and the cut goes on "
                                f"{after:.0f} s after it (more than {AFTER_CLOSING:g} s)",
                ))
                break
    return found


def _back(shots: Sequence[Shot]) -> Tuple[List[Doubt], Optional[str]]:
    """Nominate every step back in recording time of more than `NOMINATE_BACK` between neighbouring shots."""
    dated = [shot for shot in shots if shot.moment is not None]
    footage = [shot for shot in shots if not shot.card]
    undated = [shot.clip_id for shot in footage if shot.moment is None]
    why = None
    if len(footage) < 2:
        return [], None
    if len(dated) < 2:
        return [], (f"fewer than two shots have a recording time ({', '.join(undated[:8])} have none): their "
                    "files carry no date, so going back in time cannot be measured")
    if undated:
        why = (f"partly: {', '.join(undated[:8])}{' and more' if len(undated) > 8 else ''} have no recording time; "
               "the shots either side of them were compared instead")
    found = []
    for before, after in zip(dated, dated[1:]):
        back = before.moment - after.moment
        if back > NOMINATE_BACK:
            found.append(Doubt(
                id=f"back_in_time:{before.clip_id}>{after.clip_id}", kind="back_in_time",
                clip_ids=(before.clip_id, after.clip_id), at=after.start,
                says=f"clip {after.clip_id} at {_clock(after.start)} was recorded {_span(back)} before clip "
                     f"{before.clip_id}, which plays just before it ({_local(after.moment)} after "
                     f"{_local(before.moment)})",
            ))
    return found, why


def _local(moment: datetime) -> str:
    """A recording time as the clock on this computer read it, as the files' names have it."""
    return f"{moment.astimezone():%m-%d %H:%M}" if moment.tzinfo is not None else f"{moment:%m-%d %H:%M}"


def _span(gap: timedelta) -> str:
    """A length of time in the largest unit that reads well."""
    minutes = gap.total_seconds() / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} h"
    return f"{minutes / 1440:.1f} days"


def _topics(shots: Sequence[Shot]) -> Tuple[List[Doubt], Optional[str]]:
    """Nominate every topic that comes back after the cut has left it for another.

    Shots with no topic — a card, a clip put in by hand — neither start nor
    break a run.
    """
    footage = [shot for shot in shots if not shot.card]
    if not footage:
        return [], None
    topical = [(shot, topic) for shot in footage for topic in shot.topics]
    loose = [shot.clip_id for shot in footage if not shot.sourced]
    if not topical:
        if loose and len(loose) == len(footage):
            return [], ("no clip came from the semantic timeline (they were put in by hand), so none has a topic "
                        "to compare")
        return [], ("no clip has a topic: topics come from the sections set with set_sections, and the footage "
                    "was never sectioned. propose_sections, rule on them with set_sections, then ask again")
    why = None
    bare = [shot.clip_id for shot in footage if not shot.topics]
    if bare:
        why = (f"partly: {', '.join(bare[:8])}{' and more' if len(bare) > 8 else ''} have no topic "
               f"({'put in by hand' if loose else 'not in a section'}); they were passed over")
    runs: List[Tuple[str, List[Shot]]] = []
    for shot, topic in topical:
        if runs and runs[-1][0] == topic:
            if runs[-1][1][-1] is not shot:
                runs[-1][1].append(shot)
        else:
            runs.append((topic, [shot]))
    found = []
    for index, (topic, run) in enumerate(runs):
        earlier = next((place for place in range(index - 1, -1, -1) if runs[place][0] == topic), None)
        if earlier is None:
            continue
        left, between = runs[earlier][1][-1], [name for name, _ in runs[earlier + 1:index]]
        back = run[0]
        found.append(Doubt(
            id=f"topic_returns:{left.clip_id}>{back.clip_id}", kind="topic_returns",
            clip_ids=(left.clip_id, *[shot.clip_id for _, shots_ in runs[earlier + 1:index] for shot in shots_],
                      back.clip_id),
            at=back.start,
            says=f"「{topic}」 left after clip {left.clip_id} for 「{'」→「'.join(dict.fromkeys(between))}」 and comes "
                 f"back at {_clock(back.start)} with clip {back.clip_id}",
        ))
    return found, why


def late_after(length: float) -> float:
    """The second after which the subject is late in a cut this long."""
    return min(LATE_SECONDS, LATE_SHARE * length)


def _late(shots: Sequence[Shot], length: float) -> Tuple[List[Doubt], Dict[str, str]]:
    """Nominate the first words landing after `late_after`."""
    found: List[Doubt] = []
    why: Dict[str, str] = {}
    late = late_after(length)
    deaf = [shot.clip_id for shot in shots if not shot.heard]
    first = next(((at, shot) for shot in shots for at, _ in shot.words[:1]), None)
    if first is None and deaf:
        why["late_subject.words"] = (f"{', '.join(deaf[:8])} have no transcript, and nothing heard elsewhere: "
                                     "analyze them with transcription to read when the first words land")
    elif first is None:
        found.append(Doubt(
            id="late_subject:words:none", kind="late_subject", clip_ids=(), at=length,
            says=f"nothing is said in the whole cut ({length:.0f} s)",
        ))
    else:
        at, shot = first
        if deaf and any(other.start < at for other in shots if other.clip_id in deaf):
            why["late_subject.words"] = (f"partly: {', '.join(deaf[:8])} have no transcript and play before the "
                                         "first words heard")
        if at > late:
            found.append(Doubt(
                id=f"late_subject:words:{shot.clip_id}", kind="late_subject", clip_ids=(shot.clip_id,), at=at,
                says=f"the first words are at {_clock(at)}, in clip {shot.clip_id} 「{shot.words[0][1]}…」; later "
                     f"than {late:.1f} s ({LATE_SECONDS:g} s or {LATE_SHARE:.0%} of {length:.0f} s, whichever "
                     "is sooner)",
            ))
    return found, why


def nominate(shots: Sequence[Shot]) -> Doubts:
    """Nominate every place the story may go wrong, in the order the cut plays.

    Args:
        shots: The base track's clips, in play order.

    Returns:
        The nominations, and per kind what could not be checked and why.
    """
    shots = sorted(shots, key=lambda shot: shot.start)
    length = max((shot.end for shot in shots), default=0.0)
    result = Doubts()
    deaf = [shot.clip_id for shot in shots if not shot.heard]
    if shots and len(deaf) == len(shots):
        result.unchecked["closing_too_early"] = ("no clip's file has a transcript: analyze them with "
                                                 "transcription to read what is said")
    elif deaf:
        result.unchecked["closing_too_early"] = (f"partly: {', '.join(deaf[:8])} have no transcript, so what is "
                                                 "said over them was not read")
    result.doubts += _closing(shots, length)
    back, why = _back(shots)
    result.doubts += back
    if why:
        result.unchecked["back_in_time"] = why
    topics, why = _topics(shots)
    result.doubts += topics
    if why:
        result.unchecked["topic_returns"] = why
    late, whys = _late(shots, length)
    result.doubts += late
    result.unchecked.update(whys)
    result.doubts.sort(key=lambda doubt: (doubt.at, doubt.id))
    # One clip can be nominated twice for one kind — a topic left and come back to inside it twice.
    seen: Dict[str, int] = {}
    for place, doubt in enumerate(result.doubts):
        seen[doubt.id] = seen.get(doubt.id, 0) + 1
        if seen[doubt.id] > 1:
            result.doubts[place] = replace(doubt, id=f"{doubt.id}#{seen[doubt.id]}")
    return result
