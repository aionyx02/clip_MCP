"""Checking a plan against its footage, and that compiling over a project keeps the work done by hand."""

import os
import re
from dataclasses import replace
from datetime import timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from app.engine.continuity import Shot, backward_steps, strays
from app.engine.plan.beats import beginnings, compile_pieces, picture_needed, plan_pieces, song_entry
from app.engine.plan.broll import Cover, broll_covers
from app.engine.plan.music import Bed, music_fit_notes
from app.engine.plan.pieces import (
    BROLL_TRACK_ID, MERGE_GAP_SECONDS, MIN_TRANSITION_SECONDS, MOVED_FAR_SECONDS, MUSIC_B_TRACK_ID, MUSIC_TRACK_ID,
    SNAP_SECONDS, VIDEO_TRACK_ID, Note, Piece, compiled_duration, edge_of, first_clips, ordered_selections, pace_of,
    piece_starts, selected_pieces,
)
from app.engine.semantic import CLEAN_SILENCE_SECONDS, EDGE_TOLERANCE_SECONDS, CleanCuts
from app.models.media import Asset
from app.models.plan import STRUCTURES, Beat, BeatRole, EditPlan, Structure, TrimKind
from app.models.semantic import ClipKind, SemanticClip, SemanticTimeline
from app.models.timeline.parts import MAX_SPEED, MIN_SPEED, Clip, Project


# `query_clips` reads a clip's range back to the hundredth, so evidence copied from it can sit
# that far outside the range it was read from.
EVIDENCE_SLACK_SECONDS = 0.01

# Every track a compile owns and rebuilds; any other track belongs to whoever made it.
COMPILED_TRACKS = (VIDEO_TRACK_ID, MUSIC_TRACK_ID, MUSIC_B_TRACK_ID, BROLL_TRACK_ID)

# How far the compiled length may sit from what the plan asked for before it is worth saying.
LENGTH_TOLERANCE = 0.1

def _cut_notes(
    pieces: Sequence[Piece],
    selected: Sequence[Piece],
    unsnapped: Sequence[Piece],
    cuts: Optional[Mapping[str, CleanCuts]],
    duration: float,
    refused: Sequence[Piece] = (),
) -> List[str]:
    """Say what the compiler did to the cuts, and what it left for somebody to decide.

    Three things, each said once however many cuts it covers. Fifteen near
    identical lines is not a report, it is something to scroll past.

    Moving an edge off the middle of a word is arithmetic and is done without
    asking, but it is still a change to somebody's edit, so it is said.
    Letting a sentence finish is not arithmetic: measured on real footage it
    runs to a third of the length of the video, and how long the video runs is
    not the compiler's to decide. Leaving a cut inside a word because no clean
    second is near enough is a failure to say plainly rather than to hide.

    Args:
        pieces: The finished windows, cleaning and all, for the faults that
            survived to the end.
        selected: The windows the selections alone produced, for counting what
            the snapping moved. One per selection, so they pair off exactly
            with `unsnapped`, which the cleaned windows would not.
        unsnapped: The same windows as they would be without the cut points.
        cuts: Where each file may be cut without splitting a word, keyed by
            asset ID.
        duration: How long the compiled cut runs, for costing what is owed.
        refused: Windows whose stopping short is already refused, so it is
            not said again as something to weigh.

    Returns:
        The notes, empty when nothing was transcribed and so nothing is known.
    """
    if not cuts:
        return []
    said: List[str] = []

    # Each one named, with where it was asked for: a cut written into the plan and moved
    # without a word said about it reads as the plan being ignored (#87).
    shifted = [
        (new, side, new_edge, old_edge)
        for new, old in zip(selected, unsnapped)
        for side, new_edge, old_edge in (("starts", new.start, old.start), ("ends", new.end, old.end))
        if new_edge != old_edge
    ]
    if shifted:
        moved = [abs(new_edge - old_edge) for _, _, new_edge, old_edge in shifted]
        said.append(Note(
            f"{len(shifted)} cut(s) moved off the middle of a word, by up to {max(moved):.2f}s: "
            + "; ".join(f"{edge_of(piece, side, new_edge)}, not the {old_edge:.2f}s asked, inside a word"
                        for piece, side, new_edge, old_edge in shifted),
            "cuts_moved_off_words", clip_ids=first_clips(piece for piece, _, _, _ in shifted), seconds=max(moved),
            look=max(moved) > MOVED_FAR_SECONDS,
        ))

    stuck: List[str] = []
    where: List[str] = []
    # A window turned all the way down is there for its picture: nobody hears a word or a
    # sentence in it cut off, so neither is reported (#81).
    heard = [piece for piece in pieces if piece.gain > 0]
    for piece in heard:
        known = cuts.get(piece.asset_id)
        if known is None:
            continue
        # Which end, at which second of the file, inside which word: a window has two edges.
        edges = [(side, edge, word) for side, edge in (("starts", piece.start), ("ends", piece.end))
                 if (word := known.word_split_at(edge)) is not None]
        if edges:
            # The clip each edge names: a merged window starts on its first and ends on its last.
            for side, _, _ in edges:
                stuck += piece.from_clip_ids[-1:] if side == "ends" else piece.from_clip_ids[:1]
            where += [f"{edge_of(piece, side, edge)}, inside the word at {word[0]:.2f}–{word[1]:.2f}s"
                      for side, edge, word in edges]
    if stuck:
        said.append(Note(
            f"{len(where)} cut(s) still land inside a word: {'; '.join(where)}. "
            f"No clean point is within {SNAP_SECONDS:g}s of them, so they were left as asked",
            "cuts_inside_words", clip_ids=list(dict.fromkeys(stuck)), look=True,
        ))

    owed = [
        (known.rest_of_sentence(piece.end), piece)
        for piece in heard
        # By identity: the same cut heard earlier in the video is not the ending that was refused.
        if (known := cuts.get(piece.asset_id)) and known.splits_a_sentence(piece.end)
        and all(piece is not ending for ending in refused)
    ]
    if owed:
        total = sum(seconds for seconds, _ in owed)
        worst, piece = max(owed, key=lambda item: item[0])
        share = f", {total / duration:.0%} of the cut" if duration else ""
        # A window ends on the last clip it was merged from, so that is the one named.
        short = [f"{cut.from_clip_ids[-1]} {seconds:.1f}s" for seconds, cut in owed if cut.from_clip_ids]
        said.append(Note(
            f"{len(owed)} window(s) stop before the sentence ends: {', '.join(short)} short. Finishing them all "
            f"would add {total:.1f}s{share}; the longest is {piece.from_clip_ids[-1]}, {worst:.1f}s short. "
            "Give those seconds to the ones that sound abrupt, or leave them",
            "stops_before_the_sentence_ends", clip_ids=[cut.from_clip_ids[-1] for _, cut in owed if cut.from_clip_ids],
            seconds=total,
            look=True,
        ))
    return said

# What a sentence opens on when it starts the next subject rather than finishing this one;
# a cut whose last line opens on one ends on a beginning. Words that also close a subject
# (那我們、好那、說到) are left out, since 「那我們下次見」 ends a video well.
NEXT_SUBJECT_OPENERS = ("接下來", "那接下來", "另外", "再來", "然後呢", "對了")
# Marks that part a line without ending it: what follows one still belongs to the line it is in.
CLAUSE_PUNCTUATION = "，,、；;：:"
# Marks that end a line.
SENTENCE_END_PUNCTUATION = "。！？!?…"
# Words that only make sense after the clause they answer: a because, a so, a but. A line
# that opens on one where that clause is not heard leans on nothing.
LEANING_OPENERS = ("因為", "所以", "但是", "可是", "不過", "而且", "結果", "就是說")
# Words that carry a story on from the last thing said. Across a join that is what they
# are for, so they are only refused as the first words of the video, where nothing came first.
SEQUENCE_OPENERS = ("然後", "還有", "再來")
# Particles that end a phrase and so cannot begin one: a line that starts on one is the
# tail of a sentence somebody split. 了 is left out for 了解, and 的確 is a word of its own.
FRAGMENT_OPENERS = ("的", "呢", "吧", "嗎", "嘛", "囉", "啦", "呀")
# Words that start on one of those characters without being a particle.
NOT_PARTICLES = ("的確", "吧台", "嗎啡", "啦啦隊")
# The sounds somebody makes before they start. They say nothing, so a line is read past them:
# 「嗯，所以」 still opens on 「所以」.
FILLERS = "啊喔欸嗯哦呃"
# What may be taken off the front of a phrase without leaving a fragment: the particles that end
# what came before, and the fillers. Not 的 or 了: transcribers split words a character at a time,
# and a 「的」 or 「了」 on its own may be the first half of 的確 or 了解.
DROPPABLE = set("呢吧嗎嘛囉啦呀" + FILLERS)
# A clause this short, past its fillers, is said before a line rather than being one: 「好，」 or
# 「OK,」 before 「接下來」. Provisional: measured on nothing yet, it only has to tell 好 from 先把它關機.
INTERJECTION_CHARACTERS = 2

def _past_fillers(said: str) -> str:
    """What a line says once the sounds made before it, and the commas after them, are passed."""
    return said.lstrip(FILLERS + CLAUSE_PUNCTUATION + " ")

def _leans(opening: str, first: bool) -> Optional[str]:
    """Say whether a line, past its fillers, cannot open where nothing before it is heard.

    Args:
        opening: The line's words, past the fillers before it.
        first: Whether they are the first words of the video.

    Returns:
        "" when it opens on a particle, the word it leans with when it opens on
        one, None when it can open.
    """
    if opening.startswith(FRAGMENT_OPENERS) and not opening.startswith(NOT_PARTICLES):
        return ""
    return next((opener for opener in (*LEANING_OPENERS, *(SEQUENCE_OPENERS if first else ()))
                 if opening.startswith(opener)), None)

def _says_nothing(said: str) -> bool:
    """Whether a stretch of text is only particles, fillers, commas and spaces."""
    return all(character in DROPPABLE or character in CLAUSE_PUNCTUATION or character.isspace()
               for character in said)

def _quiet(known: CleanCuts, seconds: float) -> bool:
    """Say whether nobody is talking at a moment, whatever the word timings say."""
    return known.quiet_on_second_listen(seconds) or any(
        start <= seconds <= end and end - start >= CLEAN_SILENCE_SECONDS for start, end in known.pauses)

def _phrase(known: CleanCuts, first: int, last: Optional[int] = None) -> Tuple[int, int, str]:
    """Find the phrase a word belongs to, and what it says.

    Args:
        known: What is known about the file.
        first: A word, by its place in `known.words`.
        last: Where to stop instead of at the end of the phrase, for the part of
            one that is heard.

    Returns:
        `(opens, closes, said)`: the phrase's first and last word, and its text up
        to `last` when given.
    """
    opens = first
    while not known.begins_phrase(opens):
        opens -= 1
    closes = first
    while closes + 1 < len(known.words) and not known.begins_phrase(closes + 1):
        closes += 1
    stop = closes if last is None else last
    said = "".join(known.spoken[opens:stop + 1]).strip() if len(known.spoken) == len(known.words) else ""
    return opens, closes, said

def _opening_problem(piece: Piece, word: int, known: CleanCuts, first: bool) -> Optional[str]:
    """Say what is wrong with the first words a window opens on, if anything.

    Args:
        piece: The window.
        word: The first word of its file heard in it, by its place in `known.words`.
        known: What is known about that file.
        first: Whether these are the first words of the video.

    Returns:
        The problem, or None when the window opens on the start of a phrase that stands.
    """
    where = f"{edge_of(piece, 'starts', piece.start)}, the first words {'of the video' if first else 'after a join'}"
    opens, closes, phrase = _phrase(known, word)
    # Where the phrase starts before any particle comes off its front: the one before that is
    # the earlier start to offer, since starting on the particle would open on a tail.
    unstripped = opens
    worded = len(known.spoken) == len(known.words)
    if opens != word and worded and _says_nothing("".join(known.spoken[opens:word])):
        # Only a particle or a sound before speaking comes off the front, so the phrase starts here.
        opens, phrase = word, "".join(known.spoken[word:closes + 1]).strip()
    # A filler the transcriber gave its own comma is a phrase of its own and says nothing, so the
    # line is read on into the phrases after it, and back past such phrases for the start to offer.
    while worded and not _past_fillers(phrase) and closes + 1 < len(known.words):
        _, closes, more = _phrase(known, closes + 1)
        phrase += more
    def stands(start: int) -> bool:
        """Whether a phrase could be opened on without this check refusing it in turn."""
        said = _phrase(known, start)[2]
        return not _says_nothing(said) and _leans(_past_fillers(said), first) is None

    # Offered starts are word starts, the places the compiler cuts at as asked.
    earlier = unstripped
    while worded and earlier > 0:
        earlier = _phrase(known, earlier - 1)[0]
        if stands(earlier):
            break
    # Nothing said before it in the file that can be opened on means nothing to start from instead.
    instead = (f"start on the phrase before it in the file, at {known.words[earlier][0]:.2f}s"
               if worded and earlier < unstripped and stands(earlier)
               else "nothing is said before it in the file that can open, so open on another piece")
    bitten = known.words[word][0] < piece.start - EDGE_TOLERANCE_SECONDS and not _quiet(known, piece.start)
    if opens != word or bitten:
        # Where the phrase can be opened on: past any particle or filler on its front, which
        # would leave the opening on a tail; and when that start would lean too, the phrase before.
        clean = (next((index for index in range(opens, word) if not _says_nothing(known.spoken[index])), word)
                 if worded else opens)
        way = (instead if _leans(_past_fillers("".join(known.spoken[clean:closes + 1])), first) is not None
               else f"start on its first word, at {known.words[clean][0]:.2f}s of the file")
        return (f"{where}, starts partway through 「{phrase}」: {way}, or give `edge_reason` on that selection if it "
                "should open there")
    opening = _past_fillers(phrase)
    leaning = _leans(opening, first)
    if leaning == "":
        return (f"{where}, opens on 「{phrase}」, the tail of a sentence split in two: {instead}, or give "
                "`edge_reason` on that selection if it should open there")
    if leaning is not None:
        return (f"{where}, opens on 「{phrase}」, whose 「{leaning}」 leans on something before it that is not "
                f"heard: {instead}, or give `edge_reason` on that selection if it should open there")
    return None

def _ending_problem(
    last: Piece, opened: int, heard: Sequence[Tuple[str, int]], known: CleanCuts,
) -> Tuple[Optional[str], bool, Optional[Note]]:
    """Say why the last words heard do not end the video, if they do not.

    Args:
        last: The last window anything is said in.
        opened: The first word heard in it, by its place in `known.words`.
        heard: Every word heard, in order, as `(asset ID, place in its file's words)`.
        known: What is known about the last window's file.

    Returns:
        `(problem, stopped, note)`: the problem, or None when the video ends on
        a finished phrase; whether it is that the words stop partway; and a
        note when the last line may open the next subject, which is for whoever
        reads it to weigh.
    """
    word = heard[-1][1]
    worded = len(known.spoken) == len(known.words)
    # The words heard straight on in this file up to the last one, across windows a pause was
    # taken out between: the line the video ends on can start anywhere in them.
    run = len(heard) - 1
    while run > 0 and heard[run - 1] == (last.asset_id, heard[run][1] - 1):
        run -= 1
    floor = heard[run][1]
    opens, closes, phrase = _phrase(known, word, last=word)
    # Each pause or comma begins a phrase, and most pauses in a transcript carry no punctuation;
    # the line is still one until a sentence ends or the voice changes, so it is read back over
    # them, as far as the words run straight on.
    clauses = [opens]
    while clauses[-1] > floor and worded:
        previous = clauses[-1] - 1
        if known.spoken[previous].rstrip().endswith(tuple(SENTENCE_END_PUNCTUATION)):
            break
        if len(known.voices) == len(known.words) and None not in known.voices[previous:previous + 2] \
                and known.voices[previous] != known.voices[previous + 1]:
            break
        clauses.append(max(_phrase(known, previous)[0], floor))
    line = clauses[-1]
    starts = clauses[::-1]

    def next_subject(through: int) -> bool:
        """Whether the line, heard through a word, opens on the next subject: on its first
        clause, or on the one after an interjection, as 「好，接下來…」 does, while
        「先把它關機，再來把電池拔掉」 is one finished line."""
        for at, clause in enumerate(starts):
            own = _past_fillers("".join(known.spoken[clause:(starts + [through + 1])[at + 1]]).strip())
            if own.startswith(NEXT_SUBJECT_OPENERS):
                return True
            if len(own.rstrip(CLAUSE_PUNCTUATION + " ")) > INTERJECTION_CHARACTERS:
                return False
        return False

    bitten = known.words[word][1] > last.end + EDGE_TOLERANCE_SECONDS and not _quiet(known, last.end)
    if closes != word or bitten:
        finishes = known.words[closes][1]
        # The phrase before it in the window, if one finishes inside it, is the other place to stop.
        earlier = (f", end where the phrase before it finishes, at {known.words[opens - 1][1]:.2f}s of the file"
                   if opens > opened else "")
        return (f"{edge_of(last, 'ends', last.end)}, partway through 「{_phrase(known, word)[2]}」, and nothing "
                "after it is heard, so the video stops rather than ends: end where that phrase finishes, at "
                f"{finishes:.2f}s of the file, {(finishes - last.end) / last.speed:.1f}s more of the video{earlier}, "
                "or give `edge_reason` on that selection if it should stop there"), True, None
    said = "".join(known.spoken[line:word + 1]).strip() if line != opens else phrase
    # Whether a line opening on 「接下來」 starts something the video never shows or closes it, as
    # 「再來就睡覺吧…晚安拜拜」 does, is what it means, not what it says: pointed out, not refused.
    pointed = None
    if worded and next_subject(word):
        instead = (f"end on the phrase before it, at {known.words[line - 1][1]:.2f}s of the file" if line > floor
                   else "end on the piece before it")
        pointed = Note(
            f"{edge_of(last, 'ends', last.end)}, and the last thing heard is 「{said}」, which may open the next "
            f"subject rather than finish this one: if it does, {instead}; if it closes the video, leave it",
            "ends_on_the_next_subject", clip_ids=last.from_clip_ids[-1:], look=True,
        )
    # Voices are told apart within one file only, so a line from another file says nothing about who asks.
    before = next((heard_word for heard_word in reversed(heard)
                   if heard_word[0] != last.asset_id or heard_word[1] < line), None)
    if (phrase.endswith(("？", "?")) and len(known.voices) == len(known.words) and before is not None
            and before[0] == last.asset_id):
        asker, answerer = known.voices[line], known.voices[before[1]]
        if asker is not None and answerer is not None and asker != answerer:
            return (f"{edge_of(last, 'ends', last.end)}, on 「{said}」, a question from {asker} after {answerer} "
                    "spoke, which is the next question rather than an ending: end on the answer before it, or give "
                    "`edge_reason` on that selection if the video should end on the question"), False, pointed
    return None, False, pointed

def _edge_problems(
    plan: EditPlan, pieces: Sequence[Piece], cuts: Optional[Mapping[str, CleanCuts]],
) -> Tuple[List[str], List[Note], Optional[Piece]]:
    """Refuse a cut whose first words lean on nothing or whose last words stop short of an ending.

    A single selection is a trim, and where it starts and stops is what was
    asked for. Past that, a video that stops partway through what somebody is
    saying has not ended, and no length target makes it end; finishing it costs
    seconds, so it is handed back rather than done. Nor has one whose last line
    is somebody else's next question. And a video that opens on 「所以」 opens on
    the middle of a thought. A last line that opens on 「接下來」 may start the next
    subject or close this one, which only its meaning says, so it is pointed out.

    Everything is read by the phrase, between two places the speaker stopped,
    rather than by the transcriber's sentence, which can run half a minute.

    Args:
        plan: The plan.
        pieces: Its windows, as compiled.
        cuts: What is known about each file, keyed by asset ID.

    Returns:
        `(problems, notes, ending)`: one problem per edge that does not hold,
        saying the way out; a note for an ending worth a look; and the last
        window when it is refused for stopping partway through what is being
        said, which is then not said again as a note.
    """
    if len(plan.selections) < 2 or not cuts:
        return [], [], None
    # Every word heard, in the order it is heard, as (file, place in its words); and each window
    # with the word it opens on and whether the word heard before it is the one before it in the file.
    heard: List[Tuple[str, int]] = []
    opens: List[Tuple[Piece, int, bool]] = []
    for piece in pieces:
        known = cuts.get(piece.asset_id)
        if piece.gain <= 0 or piece.card is not None or known is None:
            continue
        words = [index for index, (start, end) in enumerate(known.words)
                 if start < piece.end - EDGE_TOLERANCE_SECONDS and end > piece.start + EDGE_TOLERANCE_SECONDS]
        if not words:
            continue
        before = heard[-1] if heard else None
        # The same phrase with a pause taken out of it runs straight on; the word heard last is heard
        # again when the cut between the two falls inside it.
        straight = before is not None and before[0] == piece.asset_id and words[0] - before[1] in (0, 1)
        opens.append((piece, words[0], straight))
        heard += [(piece.asset_id, index) for index in words if heard[-1:] != [(piece.asset_id, index)]]
    if not opens:
        return [], [], None
    found: List[str] = []
    for position, (piece, word, straight) in enumerate(opens):
        # A window merged from several selections opens on the first one's words and ends on the last one's.
        if straight or any(piece.excused[:1]):
            continue
        opening = _opening_problem(piece, word, cuts[piece.asset_id], first=position == 0)
        if opening is not None:
            found.append(opening)
    last, opened, _ = opens[-1]
    ending, stopped, pointed = ((None, False, None) if any(last.excused[-1:])
                                else _ending_problem(last, opened, heard, cuts[last.asset_id]))
    if ending is not None:
        found.append(ending)
    return found, [pointed] if pointed is not None else [], last if stopped else None

def _unfit_transitions(
    plan: EditPlan, pieces: Sequence[Piece], clips: Optional[Mapping[str, SemanticClip]] = None,
) -> Tuple[List[str], List[str]]:
    """Name each beat whose transition has no picture before its first shot to run in from.

    A transition ends on the cut, so the incoming shot plays under it from
    before its own start, and a shot that begins at the top of its file has
    nothing there. Left out with a note, it went unnoticed in a cut where every
    part began at the top of a file. Refused instead, with what each way out
    costs: starting the shot later cuts that much of its opening, which is the
    planner's call to make, not the compiler's.

    Args:
        plan: The plan.
        pieces: Its windows, as compiled.
        clips: The timeline's clips, keyed by ID, to say where the shot's own
            clip starts when the breath before it reached back past that.

    Returns:
        `(problems, beat_ids)`: one problem per beat whose transition could
        not be made, and those beats.
    """
    first = beginnings(pieces)
    problems: List[str] = []
    refused: List[str] = []
    for position, beat in enumerate(plan.beats):
        wanted = beat.transition_in
        at = first.get(beat.id)
        if position == 0 or wanted is None or not at or pieces[at].transition is not None:
            continue
        piece, before = pieces[at], pieces[at - 1]
        if before.played < MIN_TRANSITION_SECONDS:
            # The shot going out is what is too short to run it over; moving this one would not help.
            problems.append(
                f"beat {beat.id}: its {wanted.kind} has nothing to run over — the shot before it, "
                f"{before.from_clip_ids[0]}, plays only {before.played:.2f}s. Give that shot more of its footage, "
                "or take the transition off with `set_beat_join`"
            )
            refused.append(beat.id)
            continue
        need = picture_needed(wanted, piece)
        later = need - piece.start
        own = (clips or {}).get(piece.from_clip_ids[0]) if piece.from_clip_ids else None
        # Said apart, so a window opened earlier than its clip is not read as the plan's mistake (#95).
        reached = (f" (its clip starts at {own.source_range.start:.2f}s; the breath left before it reached back "
                   "the rest)" if own is not None and own.source_range.start - piece.start > 0.005 else "")
        other = (f"a shorter {wanted.kind}" if wanted.kind == "dip"
                 else f"a dip instead, which only needs half as much ({need / 2:.2f}s)")
        problems.append(
            f"beat {beat.id}: its {wanted.seconds:g}s {wanted.kind} has no picture to run in from — its first "
            f"shot, {piece.from_clip_ids[0]}, starts {piece.start:.2f}s into its file{reached} and the {wanted.kind} "
            f"needs {need:.2f}s before it. Start that shot at least {later:.2f}s later (`set_trim` with a "
            f"`from_seconds` that much later, which cuts that much of its opening), use {other}, or take the "
            "transition off with `set_beat_join`"
        )
        refused.append(beat.id)
    return problems, refused

def _cue_problems(plan: EditPlan, pieces: Sequence[Piece]) -> List[str]:
    """Check that every music cue has somewhere to come in, in the order given.

    Only answerable once the cut is laid out: a beat can be in the plan and
    still never begin, when its first shot was merged into the one before it.

    Args:
        plan: The plan.
        pieces: Its windows, in order, as `compile_pieces` settled them.

    Returns:
        One message per problem.
    """
    if plan.music is None:
        return []
    first = beginnings(pieces)
    problems: List[str] = []
    crossfade = plan.music.crossfade_seconds
    if crossfade:
        starts = piece_starts(pieces)
        total = compiled_duration(pieces)
        entered = sorted(
            (0 if cue.beat_id is None else first[cue.beat_id], cue) for cue in plan.music.cues
            if cue.beat_id is None or cue.beat_id in first
        )
        for (index, cue), (following, incoming) in zip(entered, entered[1:]):
            part = starts[following] - starts[index]
            if cue.asset_id and incoming.asset_id and crossfade > part:
                problems.append(
                    f"music: a {crossfade:g}s cross-fade into the cue on {incoming.beat_id} is longer than the "
                    f"{part:.1f}s part before it; shorten `crossfade_seconds`"
                )
    previous = -1
    for position, cue in enumerate(plan.music.cues, start=1):
        if cue.beat_id is None:
            at = 0
        elif cue.beat_id in first:
            at = first[cue.beat_id]
        else:
            problems.append(
                f"music cue {position}: beat {cue.beat_id} never begins in the cut — it has no footage, or its "
                "first shot runs on from the one before it — so there is no cut for the music to come in on"
            )
            continue
        if at <= previous:
            problems.append(
                f"music cue {position}: comes in on beat {cue.beat_id}, which begins before the cue listed "
                "ahead of it; list the cues in the order their beats come in"
            )
        previous = max(previous, at)
    return problems

# Why a structure cannot do without a role, said when a plan has none of it.
MISSING_ROLE = {
    BeatRole.HOOK: "倒敘 opens on the moment from later that the rest leads up to, so put it first",
    BeatRole.TURN: (
        "nothing changes, so the parts are a list rather than a story. Find where something goes wrong, "
        "surprises, or is decided, and build a beat around it"
    ),
    BeatRole.PAYOFF: "say what it all came to, and end on it",
    BeatRole.PROBLEM: "say what is wrong before the solution, or the solution answers nothing",
    BeatRole.SOLUTION: "the problem is never solved; show what fixes it",
    BeatRole.ITEM: "a list is its points; give each one a beat",
    BeatRole.STEP: "there is nothing to follow along with; give each step a beat",
    BeatRole.RESULT: "show what it came out as, and end on it",
    BeatRole.BEFORE: "the after means nothing without the before",
    BeatRole.AFTER: "show what it became, and end on it",
    BeatRole.DETAIL: "a review needs the closer look its verdict rests on",
    BeatRole.VERDICT: "say whether it is worth it, and end on it",
    BeatRole.PEAK: "a montage builds to something; find the moment it all comes to and give it a beat",
}

def _a(role: BeatRole) -> str:
    """Name a role with its article: a turn, an item."""
    return f"{'an' if role.value[0] in 'aeiou' else 'a'} {role.value}"

def _story_problems(beats: Sequence[Beat], structure: Structure = Structure.STORY) -> List[str]:
    """Find what stops a plan's parts from holding together in its structure.

    Asked of any plan with more than one part or more than one file. One
    part cut from one recording is a trim — the best thirty seconds of an
    interview — and a structure is not what was asked for; several files
    strung together as one part is exactly the list of moments this is here
    to stop.

    Args:
        beats: The plan's parts, in the order they play.
        structure: The shape the plan says it takes.

    Returns:
        What is missing or out of place: a part with no role or a role the
        structure has no place for, too few of a role it needs, a last part
        it cannot end on, a hook that is not first, or parts out of the order
        the structure plays them in. Empty for a plan that holds together.
    """
    if not beats:
        return []
    rules = STRUCTURES[structure]
    allowed = ", ".join(role.value for role in rules.roles)
    problems: List[str] = []
    unsaid = [beat.id for beat in beats if beat.role is None]
    if unsaid:
        problems.append(
            f"story: these beats do not say what they do in the story: {', '.join(unsaid)}. Give every beat a "
            f"`role` — for {rules.label}, {allowed}"
        )
        return problems
    foreign = [beat for beat in beats if beat.role not in rules.roles]
    if foreign:
        problems.append(
            f"story: {', '.join(f'{beat.id} is {_a(beat.role)}' for beat in foreign)}, which {rules.label} has "
            f"no place for; its roles are {allowed}. Use those, or change the plan's `structure` to the one the "
            "user chose"
        )
        return problems
    roles = [beat.role for beat in beats]
    for role, needed in rules.at_least.items():
        count = roles.count(role)
        if count == 0:
            problems.append(f"story: no beat is {_a(role)} — {MISSING_ROLE[role]}")
        elif count < needed:
            problems.append(f"story: only {count} beat is {_a(role)}; {rules.label} needs at least {needed}")
    if roles[-1] not in rules.ends_on:
        ending = " or ".join(f"the {role.value}" for role in rules.ends_on)
        problems.append(f"story: the video ends on {beats[-1].id}, {_a(roles[-1])}; end on {ending}")
    if BeatRole.HOOK in roles and roles[0] != BeatRole.HOOK:
        problems.append("story: the hook is not the first beat; open on it")
    if rules.in_order:
        rank = {role: index for index, role in enumerate(rules.roles)}
        latest = roles[0]
        for beat in beats[1:]:
            if rank[beat.role] < rank[latest]:
                problems.append(
                    f"story: {beat.id}, {_a(beat.role)}, comes after {_a(latest)}; {rules.label} plays "
                    f"{allowed} in that order"
                )
                break
            latest = beat.role
    return problems

def _knowledge_problems(plan: EditPlan, clips: Mapping[str, SemanticClip]) -> List[str]:
    """Check that every fact the viewer needs is established, and before what relies on it.

    Only names and order are checked here. Whether a piece really shows what
    it says it establishes is for whoever looks at its evidence.

    Args:
        plan: The plan.
        clips: The timeline's clips, keyed by ID.

    Returns:
        What is wrong, one line each.
    """
    problems: List[str] = []
    named = [fact.id for fact in plan.facts]
    if len(set(named)) != len(named):
        problems.append("two facts share an id")
    if len(plan.beats) > 1 and not plan.facts:
        problems.append(
            "facts: a plan of more than one part says what the viewer has to come to know — list it in `facts`, "
            "and in each selection's `establishes` where the footage shows or says it"
        )
    hook = {beat.id for beat in plan.beats if beat.role == BeatRole.HOOK}
    told: Dict[str, str] = {}
    owed: Dict[str, str] = {}
    for selection in ordered_selections(plan):
        where = f"selection {selection.clip_id}"
        for fact in [*selection.requires, *selection.ahead, *(item.fact for item in selection.establishes)]:
            if fact not in named:
                problems.append(f"{where}: names the fact {fact}, which is not in the plan's `facts`")
        for fact in selection.requires:
            if fact in told or fact not in named:
                continue
            if selection.beat_id in hook:
                owed.setdefault(fact, selection.clip_id)
            else:
                problems.append(
                    f"{where}: relies on {fact}, which nothing before it establishes. Put the piece that does "
                    "first, or if this comes before it on purpose, list it in `ahead`"
                )
        for fact in selection.ahead:
            if fact in told:
                problems.append(f"{where}: lists {fact} in `ahead`, but {told[fact]} has already established it")
            elif fact in named:
                owed.setdefault(fact, selection.clip_id)
        clip = clips.get(selection.clip_id)
        for item in selection.establishes:
            start, end = item.evidence
            # The slack is the rounding a clip's range is read back with, not a judgement.
            if clip is not None and not (clip.source_range.start - EVIDENCE_SLACK_SECONDS <= start
                                         and end <= clip.source_range.end + EVIDENCE_SLACK_SECONDS):
                problems.append(
                    f"{where}: the evidence for {item.fact}, {start:g}–{end:g}s, is not in the clip, which runs "
                    f"{clip.source_range.start:g}–{clip.source_range.end:g}s of its file"
                )
            told.setdefault(item.fact, selection.clip_id)
            owed.pop(item.fact, None)
    for fact, clip_id in owed.items():
        problems.append(f"selection {clip_id}: relies on {fact} coming later, and nothing after it establishes it")
    for fact in named:
        if fact not in told:
            problems.append(f"facts: nothing establishes {fact}; say which piece shows or says it, or take it out")
    return problems

def _unplayed_evidence(
    plan: EditPlan,
    pieces: Sequence[Piece],
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
) -> List[str]:
    """Find evidence for a fact that the cut does not play, because the piece was trimmed off it.

    A pause taken out in the middle of a piece leaves what is around it, so
    evidence counts as played when it lies between the first second the
    piece plays and the last.

    Args:
        plan: The plan.
        pieces: Its compiled windows.
        clips: The timeline's clips, keyed by ID.
        children: Each section's own clips, keyed by section ID.

    Returns:
        One line per piece of evidence that is not played.
    """
    problems: List[str] = []
    for selection in plan.selections:
        clip = clips.get(selection.clip_id)
        if clip is None or not selection.establishes:
            continue
        own = {selection.clip_id, *selection.trim.keep_clip_ids,
               *(child.id for child in children.get(selection.clip_id, []))}
        played = [(piece.start, piece.end) for piece in pieces
                  if piece.asset_id == clip.asset_id and own & set(piece.from_clip_ids)]
        if not played:
            continue
        first, last = min(start for start, _ in played), max(end for _, end in played)
        for item in selection.establishes:
            start, end = item.evidence
            if start < first - 1e-3 or end > last + 1e-3:
                problems.append(
                    f"selection {selection.clip_id}: the evidence for {item.fact}, {start:g}–{end:g}s, is not "
                    f"all played — the piece plays {first:.2f}–{last:.2f}s of its file once trimmed. Trim it to "
                    "keep what shows it, or say where in what it plays it does"
                )
    return problems

def check_plan(
    plan: EditPlan,
    timeline: SemanticTimeline,
    clips: Mapping[str, SemanticClip],
    children: Mapping[str, List[SemanticClip]],
    assets: Mapping[str, Asset],
    cuts: Optional[Mapping[str, CleanCuts]] = None,
) -> Tuple[List[str], List[str]]:
    """Check a plan against the footage it claims to be made of.

    Args:
        plan: Plan to check.
        timeline: The semantic timeline it names.
        clips: That timeline's clips, keyed by ID.
        children: Each section's own clips, keyed by section ID.
        assets: The assets they play from, keyed by asset ID.
        cuts: Where each file may be cut without splitting a word, keyed by
            asset ID.

    Returns:
        `(problems, notes)`. A problem stops the plan compiling; a note is
        something worth saying to whoever wrote it, such as the cut coming out
        longer than the length they asked for.
    """
    problems: List[str] = []
    notes: List[str] = []

    if plan.timeline_id != timeline.id:
        problems.append(f"the plan is written against timeline {plan.timeline_id}, not {timeline.id}")
    if plan.timeline_input_hash != timeline.input_hash:
        problems.append(
            "the footage has been analyzed again since this plan was written, so its clip IDs no longer "
            "mean the same moments; build the timeline again and rewrite the plan against it"
        )
    if not plan.selections:
        problems.append("the plan chooses no footage")

    beats = {beat.id for beat in plan.beats}
    if len(beats) != len(plan.beats):
        problems.append("two beats share an id")
    files = {clips[selection.clip_id].asset_id for selection in plan.selections if selection.clip_id in clips}
    if len(plan.beats) > 1 or len(files) > 1:
        problems.extend(_story_problems(plan.beats, plan.structure))

    for position, selection in enumerate(plan.selections, start=1):
        where = f"selection {position} ({selection.clip_id})"
        clip = clips.get(selection.clip_id)
        if clip is None:
            problems.append(f"{where}: no such clip in this timeline")
            continue
        if selection.beat_id not in beats:
            problems.append(f"{where}: belongs to beat {selection.beat_id}, which the plan does not have")
        if clip.kind == ClipKind.UNUSABLE:
            problems.append(f"{where}: this clip is unusable — nobody is talking and the picture is black or frozen")
        asset = assets.get(clip.asset_id)
        if asset is None:
            problems.append(f"{where}: asset {clip.asset_id} is no longer registered")
        elif not asset.has_video:
            problems.append(f"{where}: {asset.id} has no picture, so it cannot carry the sequence; put it under `music`")
        elif asset.duration is not None and clip.source_range.start >= float(asset.duration):
            # The timeline is derived from the analysis, so this only happens when the
            # analysis is describing a different file from the one on disk. Said here
            # because the alternative is an operation with an end before its start.
            problems.append(
                f"{where}: this clip starts at {clip.source_range.start:.1f}s of {asset.id}, which runs "
                f"{float(asset.duration):.1f}s. The file and its analysis disagree; analyze it again"
            )

        trim = selection.trim
        if trim.kind == TrimKind.KEEP:
            inside = {child.id for child in children.get(clip.id, [])}
            if not inside:
                problems.append(f"{where}: `keep` needs a section; this clip has nothing inside it to keep")
            stray = sorted(set(trim.keep_clip_ids) - inside)
            if stray:
                problems.append(f"{where}: {', '.join(stray)} is not inside this clip")
            if not trim.keep_clip_ids:
                problems.append(f"{where}: `keep` was asked for without saying what to keep")
        if trim.kind in (TrimKind.HEAD, TrimKind.TAIL):
            if trim.seconds is None:
                problems.append(f"{where}: `{trim.kind.value}` needs how many seconds to take")
            elif trim.seconds > clip.duration:
                problems.append(
                    f"{where}: asks for {trim.seconds:g}s of a clip that runs {clip.duration:.3f}s"
                )
        if trim.kind == TrimKind.TIGHTEN and not children.get(selection.clip_id):
            # It would play the clip whole and say nothing, as if it had tightened it (#99).
            problems.append(
                f"{where}: `tighten` drops the pauses and unusable picture between the clips inside a section, "
                f"and {clip.id} is one sentence with nothing inside it; use `range` to play part of it, or cover a "
                "frame gone black with B-roll and `after_seconds`"
            )
        if trim.kind == TrimKind.RANGE:
            if trim.from_seconds is None or trim.to_seconds is None:
                problems.append(f"{where}: `range` needs both `from_seconds` and `to_seconds`")
            elif trim.to_seconds <= trim.from_seconds:
                problems.append(f"{where}: `range` ends at {trim.to_seconds:g}s, before it starts")
            elif trim.from_seconds >= clip.duration:
                problems.append(
                    f"{where}: `range` starts {trim.from_seconds:g}s in, but the clip runs {clip.duration:.3f}s"
                )
        # The timeline holds a clip to what the renderer can stretch sound to. A selection's
        # own speed stands; the pacing speed is for the pieces that set none.
        speed = selection.speed if selection.speed is not None else (plan.pacing.speed or 1.0)
        if not MIN_SPEED <= speed <= MAX_SPEED:
            problems.append(
                f"{where}: plays at x{speed:g}, outside the x{MIN_SPEED:g} to "
                f"x{MAX_SPEED:g} a clip can play at"
            )

    for rejection in plan.rejected:
        if rejection.clip_id not in clips:
            notes.append(Note(f"the rejected clip {rejection.clip_id} is not in this timeline",
                              "rejected_clip_missing", clip_ids=[rejection.clip_id]))
    problems.extend(_knowledge_problems(plan, clips))
    notes.extend(_file_order_notes(plan, clips, assets))
    notes.extend(_shaky_notes(plan, clips))
    shots = _shots(plan, clips, assets)
    for stray in strays(shots):
        problems.append(
            f"continuity: {stray}. Move it next to what it belongs with, drop it, or say in its `jump_reason` "
            "why it belongs here — a flash-forward, a callback"
        )
    if plan.structure != Structure.FLASHBACK:
        by_time, by_light = backward_steps(shots)
        for step in by_time:
            problems.append(
                f"continuity: {step}. Put the part in the order it happened, or say in the later shot's "
                "`jump_reason` why it goes back — a memory, a callback"
            )
        for step, pair in by_light:
            notes.append(Note(f"{step}: if that is night going back to day, put them in order or give a "
                              "`jump_reason`", "dark_to_bright", clip_ids=list(pair), look=True))
    both = sorted({rejection.clip_id for rejection in plan.rejected} & {item.clip_id for item in plan.selections})
    if both:
        problems.append(
            f"{', '.join(both)} is both used and rejected; take it out of `rejected` if it belongs in the cut"
        )

    pace = pace_of(plan)
    # Only a silence longer than `pause` is taken out, so at exactly the floor what is left
    # between the halves still runs past the merge gap: the floor itself is allowed (#75).
    floor = round(2 * pace.breath + MERGE_GAP_SECONDS, 6)
    if round(pace.pause, 6) < floor:
        problems.append(
            f"pacing: pauses over {pace.pause:g}s cannot be taken out while {pace.breath:g}s of air is left either "
            f"side of each cut — what would be left between the two halves is under {MERGE_GAP_SECONDS:g}s, "
            "so they would be joined straight back together. Raise `pause_seconds` to at least "
            f"{floor:g}s or lower `breath_seconds`"
        )

    cues = plan.music.cues if plan.music is not None else []
    placed_cues: set = set()
    for position, cue in enumerate(cues, start=1):
        where = f"music cue {position}"
        if cue.beat_id is None and position > 1:
            problems.append(f"{where}: only the first cue may leave out `beat_id`; say which beat this one comes in on")
        elif cue.beat_id is not None and cue.beat_id not in beats:
            problems.append(f"{where}: comes in on beat {cue.beat_id}, which the plan does not have")
        if cue.beat_id is not None and cue.beat_id in placed_cues:
            problems.append(f"{where}: another cue already comes in on beat {cue.beat_id}")
        placed_cues.add(cue.beat_id)
        if cue.asset_id is not None:
            song = assets.get(cue.asset_id)
            if song is None:
                problems.append(f"{where}: the music asset {cue.asset_id} is not registered")
            elif not song.has_audio:
                problems.append(f"{where}: the music asset {cue.asset_id} has no sound")
            elif song.duration is not None and cue.start >= float(song.duration):
                problems.append(
                    f"{where}: comes in at {cue.start:g}s of {cue.asset_id}, which runs {float(song.duration):.1f}s"
                )
        if cue.cut_on_beat:
            known = (cuts or {}).get(cue.asset_id) if cue.asset_id is not None else None
            if cue.asset_id is None:
                problems.append(f"{where}: is silence, so there is no beat to cut on")
            elif known is None or known.beats is None:
                problems.append(
                    f"{where}: cutting on the beat needs the beat of {cue.asset_id} measured; analyze it "
                    "with analyze_asset first, or again if it was analyzed before beats were measured"
                )
            elif not known.beats:
                problems.append(
                    f"{where}: {cue.asset_id} has no steady beat to cut on — it was measured, and nothing "
                    "in it pulses regularly enough to call one"
                )

    if not problems:
        pieces, cleaned = compile_pieces(plan, clips, children, assets, cuts)
        duration = compiled_duration(pieces)
        edges, ending_notes, stopped = _edge_problems(plan, pieces, cuts)
        notes.extend(_cut_notes(
            pieces,
            selected_pieces(plan, clips, children, assets, cuts),
            selected_pieces(plan, clips, children, assets, None),
            cuts, duration, [stopped] if stopped is not None else [],
        ))
        notes.extend(cleaned)
        notes.extend(ending_notes)
        problems.extend(_cue_problems(plan, pieces))
        unfit, refused = _unfit_transitions(plan, pieces, clips)
        problems.extend(unfit)
        # Refused is said once, as the problem: the note that the compile left it out is the
        # same thing said again, as if it had been dealt with (#95).
        notes[:] = [note for note in notes if not (isinstance(note, Note) and note.kind == "join_left_out"
                                                   and note.beat_id in refused)]
        problems.extend(_unplayed_evidence(plan, pieces, clips, children))
        problems.extend(edges)
        for position, cue in enumerate(cues, start=1):
            entry = song_entry(cue, (cuts or {}).get(cue.asset_id) if cue.asset_id is not None else None)
            if entry != cue.start:
                notes.append(Note(
                    f"music cue {position} comes in at {entry:g}s of {cue.asset_id}, the first beat after "
                    f"the {cue.start:g}s asked for, so the cuts it sets the beat for are counted from a beat",
                    "music_enters_on_a_beat", beat_id=cue.beat_id, seconds=entry,
                ))
        notes.extend(music_fit_notes(plan, pieces, assets, cuts))
        _, covering, said = broll_covers(plan, pieces, clips, assets, cuts)
        problems.extend(covering)
        notes.extend(said)
        if plan.target.seconds:
            drift = abs(duration - plan.target.seconds) / plan.target.seconds
            if drift > LENGTH_TOLERANCE:
                notes.append(Note(
                    f"the cut comes out {duration:.1f}s against a target of {plan.target.seconds:g}s, "
                    f"{drift:.0%} out",
                    "length_off_target", seconds=duration, look=True,
                ))
        for beat in plan.beats:
            if beat.target_seconds is None:
                continue
            chosen = [selection for selection in plan.selections if selection.beat_id == beat.id]
            in_beat = compiled_duration(plan_pieces(
                plan.model_copy(update={"selections": chosen}), clips, children, assets, cuts,
            ))
            if abs(in_beat - beat.target_seconds) / beat.target_seconds > LENGTH_TOLERANCE:
                notes.append(Note(f"beat {beat.id} ({beat.name}) runs {in_beat:.1f}s against {beat.target_seconds:g}s",
                                  "length_off_target", beat_id=beat.id, seconds=in_beat, look=True))
    return problems, notes

def compiled_clips(project: Optional[Project]) -> List[Tuple[str, Clip]]:
    """List the clips sitting on the tracks a compile owns.

    Args:
        project: Project to read, or `None` for an empty one.

    Returns:
        One `(track_id, clip)` per clip on the sequence, music and B-roll
        tracks the compiler builds. Clips on any other track belong to whoever
        put them there and are not listed.
    """
    if project is None:
        return []
    return [
        (track.id, clip)
        for track in project.tracks if track.id in COMPILED_TRACKS
        for clip in track.clips
    ]

def as_left(pieces: Sequence[Piece], project: Optional[Project]) -> Tuple[List[Piece], List[str]]:
    """Lay the plan's windows out the way the clips pinned by hand actually play.

    A clip adjusted by hand comes back from a recompile with its own range
    and speed, so everything placed against the cut — music, covering
    picture, markers — has to be placed against those, or it lands where the
    plan alone would have put things. And a clip lengthened by hand over the
    footage of the window after it has taken that footage in: the window is
    not put back, or the same moment plays twice.

    The matching is the compiler's own: a pinned clip belongs to the first
    window made of exactly the semantic clips it was made from.

    Args:
        pieces: The plan's windows, in order.
        project: What is on the timeline now, or None.

    Returns:
        `(pieces, notes)` — the windows with each pinned one's range and speed,
        less those a pinned clip has taken in, and a line naming what was
        left out that way.
    """
    kept = {
        tuple(clip.from_clip_ids): clip
        for track_id, clip in compiled_clips(project)
        if track_id == VIDEO_TRACK_ID and clip.pinned and clip.card is None
    }
    laid: List[Piece] = []
    hand: List[bool] = []
    for piece in pieces:
        pinned = kept.pop(piece.from_clip_ids, None)
        if pinned is not None:
            piece = replace(
                piece, start=float(pinned.source_range.start), end=float(pinned.source_range.end),
                speed=pinned.speed,
            )
        laid.append(piece)
        hand.append(pinned is not None)

    def taken_in(piece: Piece) -> bool:
        """Say whether a pinned clip plays most of this window already."""
        return any(
            min(piece.end, other.end) - max(piece.start, other.start) > piece.duration / 2
            for other, by_hand in zip(laid, hand)
            if by_hand and other is not piece and other.asset_id == piece.asset_id
        )

    out = [piece for piece, by_hand in zip(laid, hand) if by_hand or not taken_in(piece)]
    gone = [piece for piece, by_hand in zip(laid, hand) if not by_hand and taken_in(piece)]
    notes = []
    if gone:
        notes.append(Note(
            f"{len(gone)} window(s) were left out because a clip lengthened by hand already plays them: "
            f"{', '.join(first_clips(gone))}",
            "left_out_for_a_hand_edit", clip_ids=first_clips(gone), look=True,
        ))
    return out, notes

def _shots(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    assets: Mapping[str, Asset],
) -> List[Shot]:
    """Line a plan's pieces up as shots, with when each was shot and what it is about.

    Args:
        plan: The plan.
        clips: The timeline's clips, keyed by ID.
        assets: The assets they play from, keyed by asset ID.

    Returns:
        One shot per selection, in the order the video plays them. A clip's
        topic is its section's; an utterance takes its section's.
    """
    hook = {beat.id for beat in plan.beats if beat.role == BeatRole.HOOK}
    begun: set = set()
    shots: List[Shot] = []
    for selection in ordered_selections(plan):
        clip = clips.get(selection.clip_id)
        if clip is None:
            continue
        asset = assets.get(clip.asset_id)
        moment = (
            asset.recorded_at + timedelta(seconds=clip.source_range.start)
            if asset is not None and asset.recorded_at is not None else None
        )
        parent = clips.get(clip.parent_id) if clip.parent_id else None
        shots.append(Shot(
            label=selection.clip_id,
            moment=moment,
            topic=clip.topic or (parent.topic if parent is not None else None),
            opens_part=selection.beat_id not in begun,
            exempt=selection.beat_id in hook or bool(selection.jump_reason),
            exposure=clip.scores.get("exposure"),
        ))
        begun.add(selection.beat_id)
    return shots

# A clip whose camera was this unsteady — the mean change in its movement, in frame widths —
# is handheld enough to be felt. Provisional: the steady side of what the search docs offer
# as a bound, not measured against anyone watching.
SHAKY = 0.01

def _shaky_notes(plan: EditPlan, clips: Mapping[str, SemanticClip]) -> List[Note]:
    """Say which chosen clips the camera shook in, with how much, so a steadier one can be looked for.

    Args:
        plan: The plan.
        clips: The timeline's clips, keyed by ID.

    Returns:
        One note naming them all, or none.
    """
    shaky = []
    for selection in plan.selections:
        clip = clips.get(selection.clip_id)
        shake = clip.scores.get("shake") if clip is not None else None
        if shake is not None and shake > SHAKY:
            shaky.append((selection.clip_id, shake))
    if not shaky:
        return []
    listed = ", ".join(f"{clip_id} ({shake:.3f})" for clip_id, shake in sorted(shaky, key=lambda item: -item[1]))
    return [Note(f"the camera shakes in {listed}, against {SHAKY:g} for a steady shot; if a shake is not wanted, "
                 "query_clips with max_scores {\"shake\": " + f"{SHAKY:g}" + "} finds steadier footage of the same",
                 "shaky", clip_ids=[clip_id for clip_id, _ in shaky], look=True)]

# A number a file's name starts with, before a separator: `1-`, `01_`, `1.`, `1 `, `1、`, `(1)`.
# Numbered by hand, it is the order somebody meant the files to be cut in.
FILE_NUMBER = re.compile(r"^\(?(\d{1,4})\s*[-_.、)\s]")

def file_number(path: str) -> Optional[int]:
    """Read the number a file's name starts with, if it starts with one before a separator."""
    found = FILE_NUMBER.match(os.path.basename(path))
    return int(found.group(1)) if found else None

def _file_order_notes(
    plan: EditPlan,
    clips: Mapping[str, SemanticClip],
    assets: Mapping[str, Asset],
) -> List[Note]:
    """Say where the plan plays numbered files out of the order their names give.

    Files numbered by hand in one folder are the order somebody meant them to
    be cut in; files in different folders were numbered apart. Only where a
    file first comes in counts, and the hook and a selection that says why it
    is out of place are left out: both are meant to be. A note, not a refusal:
    the numbers may be the order they were shot rather than the order wanted.

    Args:
        plan: The plan.
        clips: The timeline's clips, keyed by ID.
        assets: The files, keyed by asset ID.

    Returns:
        One note per folder played out of its order.
    """
    hook = {beat.id for beat in plan.beats if beat.role == BeatRole.HOOK}
    seen: set = set()
    highest: Dict[str, Tuple[int, str]] = {}
    behind: Dict[str, List[str]] = {}
    labels: Dict[str, List[str]] = {}
    for selection in ordered_selections(plan):
        clip = clips.get(selection.clip_id)
        asset = assets.get(clip.asset_id) if clip is not None else None
        if asset is None or asset.id in seen:
            continue
        seen.add(asset.id)
        number = file_number(asset.path)
        if number is None or selection.beat_id in hook or selection.jump_reason:
            continue
        folder, name = os.path.dirname(os.path.abspath(asset.path)), os.path.basename(asset.path)
        top = highest.get(folder)
        if top is not None and number < top[0]:
            behind.setdefault(folder, []).append(f"{name} after {top[1]}")
            labels.setdefault(folder, []).append(selection.clip_id)
        else:
            highest[folder] = (number, name)
    return [
        Note(f"the files are numbered in the order they were meant to be cut, and the plan plays "
             f"{'; '.join(steps[:3])}{f' and {len(steps) - 3} more' if len(steps) > 3 else ''}: put them in "
             "their order, or say in the selection's `jump_reason` why it comes there",
             "file_order", clip_ids=labels[folder], look=True)
        for folder, steps in behind.items()
    ]

def check_recompile(
    plan: EditPlan,
    project: Project,
    pieces: Sequence[Piece],
    covers: Sequence[Cover] = (),
    beds: Sequence["Bed"] = (),
) -> List[str]:
    """Check that compiling over a project would not destroy work done by hand.

    The compiler owns the sequence and the music bed, and rebuilds them both.
    What it must not do is quietly rebuild over a change somebody made
    themselves, because that turns every round of feedback into a round of
    lost work. So a clip on those tracks has to be either something this plan
    compiled — which may be replaced — or something pinned, which is kept as
    it stands and has to still have a place in the new cut.

    Args:
        plan: Plan about to be compiled.
        project: Project it would be compiled onto.
        pieces: The windows the plan compiles to.
        covers: The B-roll the plan lays over them, which is identified by its
            own pair of clips rather than by a window.
        beds: The music the plan lays under them, each stretch identified by
            its cue and its pass through the song.

    Returns:
        One message per problem, empty when the project is safe to compile
        over.
    """
    problems: List[str] = []
    available = {piece.from_clip_ids for piece in pieces}
    available |= {(cover.clip_id, cover.over_clip_id) for cover in covers}
    available |= {bed.key for bed in beds}
    if beds:
        # A music clip compiled before stretches had names of their own carries none; it
        # was the first stretch, and is taken as the first stretch now.
        available.add(())
    for track_id, clip in compiled_clips(project):
        if clip.card is not None:
            # Title cards come back on their own: the plan's are rebuilt, any other is put back.
            continue
        if clip.pinned and track_id in (MUSIC_TRACK_ID, MUSIC_B_TRACK_ID) and beds:
            ordered = sorted((bed for bed in beds if bed.track_id == track_id), key=lambda bed: bed.timeline_in)
            match = next((index for index, bed in enumerate(ordered) if bed.key == tuple(clip.from_clip_ids)),
                         0 if not clip.from_clip_ids else None)
            if match is not None and match + 1 < len(ordered):
                room = ordered[match + 1].timeline_in - ordered[match].timeline_in
                if float(clip.timeline_duration) > room + 0.001:
                    problems.append(
                        f"music clip {clip.id} was lengthened by hand to {float(clip.timeline_duration):.1f}s, "
                        f"and the next stretch of music now starts {room:.1f}s after it; shorten it, or unpin "
                        "it with set_clip_pinned to let the plan lay it again"
                    )
        if clip.pinned:
            if tuple(clip.from_clip_ids) not in available:
                problems.append(
                    f"clip {clip.id} on track {track_id} was adjusted by hand, and the footage it was made "
                    "from is no longer in the plan; put it back in the plan, or unpin the clip with "
                    "set_clip_pinned to let it go"
                )
            continue
        if clip.from_plan_id != plan.id:
            came_from = f"plan {clip.from_plan_id}" if clip.from_plan_id else "nowhere this plan knows about"
            problems.append(
                f"clip {clip.id} on track {track_id} came from {came_from}; compiling would remove it. "
                "Pin it with set_clip_pinned to keep it, or take it off the track"
            )
    return problems
