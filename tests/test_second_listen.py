"""The second listen: a voice detector and the speech model, neither reading the transcript, checking where talking starts and stops."""

import pytest

from app.engine.plan import BREATH_SECONDS
from app.engine.semantic import clean_cuts
from app.models.media import HeardWord, SecondListen, Span
from test_delivery import checks, clip, project_of, talking
from test_edit_plan import compiled
from test_semantic_timeline import analysis, spans


def listened(voiced=(), heard=()) -> SecondListen:
    """Build a second listen: where the voice detector heard talking, and the words heard.

    Args:
        voiced: `(start, end)` per voiced stretch.
        heard: `(start, end, text)` per word heard, each heard for certain.

    Returns:
        The second listen.
    """
    return SecondListen(
        voiced=spans(voiced),
        words=[HeardWord(start=start, end=end, text=text, probability=0.9) for start, end, text in heard],
        model="test",
    )


def test_a_soft_first_word_the_transcript_left_out_is_put_back() -> None:
    # 「我們可能要拆個30分鐘」 in a room too loud for any silence to be measured: the
    # transcript starts the line at 可能, the voice detector only from 10.56, and the second
    # listen hears 我們 from 10.16, straight on into 可能.
    made = analysis(duration=20.0, segments=[(10.56, 12.0, "可能要拆")])
    made.second_listen = listened(voiced=[(10.56, 11.84)], heard=[(10.16, 10.6, "我們"), (10.6, 10.8, "可能")])
    pieces, notes = compiled(made)
    assert pieces[0].start == pytest.approx(10.16 - BREATH_SECONDS)
    assert any("where the sound really starts" in note for note in notes)


def test_where_no_silence_was_measured_the_voice_detector_says_where_the_talking_ends() -> None:
    # 「我終於成功囉」: the transcript ends 囉 at 9.47, and it is drawn out until 9.82.
    made = analysis(duration=10.7, segments=[(8.32, 9.47, "我終於成功囉")])
    made.second_listen = listened(voiced=[(8.32, 9.82)])
    pieces, _ = compiled(made)
    assert pieces[-1].end == pytest.approx(9.82 + BREATH_SECONDS)


def test_the_voice_detector_holding_on_after_a_voice_stops_moves_nothing() -> None:
    # It ran 0.1s past 「一下」 on the first edit, which was over when the transcript said.
    made = analysis(duration=12.0, segments=[(6.5, 7.93, "給大家示範一下")])
    made.second_listen = listened(voiced=[(6.5, 8.03)])
    assert clean_cuts(made).sound_after(7.93) is None


def test_a_word_the_transcript_knows_is_not_put_back() -> None:
    # 然後 is in the transcript as a sentence of its own, so starting after it is a choice.
    made = analysis(duration=12.0, segments=[(5.7, 5.95, "然後"), (6.0, 8.0, "出發")])
    made.second_listen = listened(voiced=[(5.7, 8.0)], heard=[(5.7, 6.0, "然後"), (6.0, 6.5, "出發")])
    assert clean_cuts(made).sound_before(6.0) is None


def test_a_pause_the_voice_detector_hears_is_a_pause_whatever_the_heard_word_edges_say() -> None:
    # The speech model stretched 然後 back from 18.24 to 18.02, over the gap after 難受.
    made = analysis(duration=20.0, segments=[(17.2, 17.84, "真是難受"), (18.24, 19.0, "然後再來")])
    made.second_listen = listened(
        voiced=[(17.18, 18.05), (18.24, 20.0)], heard=[(17.54, 17.84, "受"), (18.02, 18.34, "然後")],
    )
    known = clean_cuts(made)
    assert known.sound_after(18.05) is None and known.quiet_on_second_listen(18.1)


def test_a_cut_the_second_listen_hears_as_quiet_is_clean() -> None:
    # The transcript runs 天氣 on to 3.45; the voice detector hears nobody from 3.0 to 3.5.
    made = talking()
    made.transcript.segments[0].words[1].end = 3.45
    made.second_listen = SecondListen(
        voiced=[Span(start=1.0, end=3.0), Span(start=3.5, end=5.0)], words=[], model="test",
    )
    assert checks(project_of(clip("a", "x", 0, 3.2, 0)), {"x": made}) == []


def test_a_cut_that_takes_off_a_word_heard_on_the_second_listen_names_it() -> None:
    # 我們 is really said from 5.7, softly enough that the voice detector only starts at 6.2.
    made = talking()
    made.second_listen = SecondListen(
        voiced=[Span(start=6.2, end=9.0)],
        words=[
            HeardWord(start=5.7, end=6.2, text="我們", probability=0.9),
            HeardWord(start=6.2, end=7.0, text="出發", probability=0.9),
        ],
        model="test",
    )
    found = checks(project_of(clip("a", "x", 6.0, 9.5, 0)), {"x": made})
    assert [check for check, _ in found] == ["mid_speech"]
    assert "「我們」" in found[0][1] and "really starts at 5.70s" in found[0][1]


def opening_with(*words) -> "MediaAnalysis":
    """One line 「各位我到高鐵了」 with a word per `(start, end, text)`, heard straight through."""
    from app.models.media import TranscriptWord

    made = analysis(duration=12.0, segments=[(words[0][0], words[-1][1], "".join(text for _, _, text in words))])
    made.transcript.segments[0].words = [TranscriptWord(start=start, end=end, text=text) for start, end, text in words]
    made.second_listen = listened(voiced=[(1.75, 4.0)], heard=[(1.8, 2.2, "各位"), (2.2, 2.4, "我")])
    return made


def test_a_word_taken_off_the_front_of_a_line_on_its_edge_stays_off() -> None:
    # Starting on 我 takes 各位 off; it is in the transcript, ending where 我 begins.
    made = opening_with((1.81, 2.23, "各位"), (2.23, 2.45, "我"), (2.45, 3.0, "到高鐵了"))
    known = clean_cuts(made)
    assert known.sound_before(2.23) is None and known.sound_before(2.2) is None
    assert known.heard_cut_off(2.2, opens=True) == ""
    assert checks(project_of(clip("a", "x", 2.2, 3.0, 0)), {"x": made}) == []


def test_starting_inside_a_word_is_still_starting_in_the_middle_of_it() -> None:
    made = opening_with((1.81, 2.23, "各位"), (2.23, 2.45, "我"), (2.45, 3.0, "到高鐵了"))
    assert [check for check, _ in checks(project_of(clip("a", "x", 2.0, 3.0, 0)), {"x": made})] == ["mid_speech"]


def test_a_cut_moved_far_to_the_sound_says_where_it_was_asked_for_and_where_it_went() -> None:
    from app.engine.plan import MOVED_FAR_SECONDS

    # The transcript missed 我們 altogether; the second listen hears it 0.4s before 可能.
    made = analysis(duration=20.0, segments=[(10.56, 12.0, "可能要拆")])
    made.second_listen = listened(voiced=[(10.56, 11.84)], heard=[(10.16, 10.6, "我們"), (10.6, 10.8, "可能")])
    pieces, notes = compiled(made)
    assert 10.56 - pieces[0].start > MOVED_FAR_SECONDS
    [far] = [note for note in notes if note.kind == "cuts_moved_far"]
    assert far.look and "not the 10.56s asked" in far
