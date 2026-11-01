"""Names in a transcription prompt, set against the words the speech model was unsure of (#164).

A name written in `analyze_asset`'s prompt can still come out as a common word
that sounds the same. This offers the prompt's spelling where an unsure stretch
reads alike. It only ever offers: the transcript is never changed here, and
which offer is right is for the model to judge by what the footage is about.

The speech model's words are mostly single characters, so stretches are read
character by character, whatever the word boundaries were.
"""

import functools
import re
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Sequence

from app.models.media import TranscriptSegment, TranscriptWord

MIN_CJK_TERM = 2
MAX_CJK_TERM = 6
MIN_LATIN_TERM = 2
_CJK = r"㐀-鿿"
_TERM = re.compile(rf"[{_CJK}]+|[A-Za-z][A-Za-z0-9'-]*")
_IS_CJK = re.compile(rf"[{_CJK}]")


@dataclass(frozen=True)
class _Character:
    """One character of a line, and the word it came from."""

    text: str
    word: TranscriptWord


def prompt_terms(prompt: str) -> List[str]:
    """The names and terms a prompt spells out.

    Args:
        prompt: What was given to guide transcription.

    Returns:
        Its runs of Chinese characters of a name's length and its Latin words,
        in order, without repeats.
    """
    terms: List[str] = []
    for term in _TERM.findall(prompt):
        wanted = (MIN_CJK_TERM <= len(term) <= MAX_CJK_TERM if _IS_CJK.match(term)
                  else len(term) >= MIN_LATIN_TERM)
        if wanted and term not in terms:
            terms.append(term)
    return terms


@functools.lru_cache(maxsize=4096)
def _readings(character: str) -> FrozenSet[str]:
    """Every toneless reading a character can have."""
    from pypinyin import Style, pinyin

    return frozenset(pinyin(character, style=Style.NORMAL, heteronym=True)[0])


def _blurred(reading: str) -> str:
    """A reading with the sounds that are often not told apart made the same."""
    blurred = re.sub(r"^(z|c|s)h", r"\1", reading)
    blurred = re.sub(r"^l", "n", blurred)
    return re.sub(r"(e|i)ng$", r"\1n", blurred)


def _reads_like(one: str, other: str, how: Callable[[str], str]) -> bool:
    """Whether two characters share a reading once each is put through `how`."""
    return bool({how(reading) for reading in _readings(one)} & {how(reading) for reading in _readings(other)})


def _characters(words: Sequence[TranscriptWord]) -> List[List[_Character]]:
    """A line's Chinese characters, in runs that nothing else interrupts."""
    runs: List[List[_Character]] = [[]]
    for word in words:
        for character in word.text:
            if _IS_CJK.match(character):
                runs[-1].append(_Character(character, word))
            elif runs[-1] and not character.isspace():
                runs.append([])
    return [run for run in runs if run]


def offers(segments: Sequence[TranscriptSegment], prompt: str, is_unsure: Callable[[TranscriptWord], bool],
           ) -> List[Dict[str, object]]:
    """Where an unsure stretch reads like a name in the prompt.

    Args:
        segments: The lines to look in.
        prompt: What was given to guide transcription.
        is_unsure: Whether the speech model was guessing at a word.

    Returns:
        One offer per place, in time order: `heard`, `suggested`, its `start`
        and `end`, and `exact`, true when the readings are the same and false
        when they are only close. A stretch with nothing unsure in it is not
        offered, and neither is one that already says the name.
    """
    terms = [term for term in prompt_terms(prompt) if _IS_CJK.match(term)]
    found: List[Dict[str, object]] = []
    for segment in segments:
        for run in _characters(segment.words):
            for term in terms:
                for at in range(len(run) - len(term) + 1):
                    window = run[at:at + len(term)]
                    heard = "".join(character.text for character in window)
                    if heard == term or not any(is_unsure(character.word) for character in window):
                        continue
                    pairs = list(zip(heard, term))
                    if not all(_reads_like(one, other, _blurred) for one, other in pairs):
                        continue
                    found.append({
                        "heard": heard, "suggested": term,
                        "start": window[0].word.start, "end": window[-1].word.end,
                        "exact": all(_reads_like(one, other, str) for one, other in pairs),
                    })
    return sorted(found, key=lambda offer: offer["start"])


def missing(segments: Sequence[TranscriptSegment], prompt: str) -> List[str]:
    """The prompt's names and terms that appear nowhere in these lines.

    Args:
        segments: The lines to look in.
        prompt: What was given to guide transcription.

    Returns:
        The terms not found, in the prompt's order, ignoring Latin case.
    """
    said = "".join(segment.text for segment in segments).lower()
    return [term for term in prompt_terms(prompt) if term.lower() not in said]
