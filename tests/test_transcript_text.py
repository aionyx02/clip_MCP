"""Tests for what the transcript says: its script, and the lines the recogniser made up."""

import pytest

from app.engine.analysis import convert_chinese_variant
from app.models.media import TranscriptSegment, TranscriptWord

def segment_of(*texts: str) -> TranscriptSegment:
    """A segment with one word per text, a tenth of a second each."""
    words = [TranscriptWord(start=index / 10, end=(index + 1) / 10, text=text) for index, text in enumerate(texts)]
    return TranscriptSegment(start=0.0, end=len(texts) / 10, text="".join(texts), words=words)

@pytest.mark.parametrize(("heard", "written"), [
    ("列印了然後", "列印了然後"),
    ("到高鐵了然後", "到高鐵了然後"),
    ("裡面包含兩本", "裡面包含兩本"),
    ("里面包含", "裡面包含"),
    ("我头发很长", "我頭髮很長"),
    ("一目了然", "一目瞭然"),
    ("麵包很好吃", "麵包很好吃"),
])
def test_taiwan_traditional_never_reads_a_phrase_across_two_words(heard: str, written: str) -> None:
    [segment] = convert_chinese_variant([segment_of(*heard)], "zh-TW")
    assert segment.text == written
    assert [word.text for word in segment.words] == list(written)
