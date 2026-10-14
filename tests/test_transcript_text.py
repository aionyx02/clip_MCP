"""Tests for what the transcript says: its script, and the lines the recogniser made up."""

import pytest

from app.engine.analysis import convert_chinese_variant
from app.engine.semantic import was_made_up
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

def timed(*words: tuple[str, float, float, float]) -> TranscriptSegment:
    """A segment from (text, start, end, probability) words."""
    made = [TranscriptWord(text=text, start=start, end=end, probability=chance) for text, start, end, chance in words]
    return TranscriptSegment(start=made[0].start, end=made[-1].end, text="".join(w.text for w in made), words=made)

def test_a_subtitle_credit_over_cafe_noise_was_made_up() -> None:
    credit = timed(("中", 0.0, 1.0, 0.012), ("文", 1.0, 1.0, 0.6), ("字", 1.0, 1.0, 0.6),
                   ("幕", 1.0, 1.0, 0.6), ("——", 1.0, 1.0, 0.6), ("YK", 1.0, 1.0, 0.6))
    assert was_made_up(credit)

@pytest.mark.parametrize("line", ["請不吝點贊 訂閱 打賞支援明鏡與點點欄目", "謝謝觀看", "字幕由志願者提供"])
def test_a_channel_sign_off_was_made_up_whatever_its_confidence(line: str) -> None:
    assert was_made_up(timed(*((char, index * 0.2, index * 0.2 + 0.2, 0.9) for index, char in enumerate(line))))

def test_one_word_looping_was_made_up() -> None:
    assert was_made_up(timed(*(("沒興趣,", index * 0.5, index * 0.5 + 0.5, 0.8) for index in range(40))))

def test_an_ordinary_line_is_kept() -> None:
    line = timed(("我", 0.0, 0.2, 0.3), ("到", 0.2, 0.4, 0.9), ("高", 0.4, 0.6, 0.9), ("鐵", 0.6, 0.8, 0.9),
                 ("了", 0.8, 0.8, 0.9), ("對", 1.0, 1.2, 0.9), ("對", 1.2, 1.4, 0.9), ("謝謝大家", 1.4, 2.0, 0.9))
    assert not was_made_up(line)
