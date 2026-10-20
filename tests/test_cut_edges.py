"""Tests for how a cut begins and ends: the first words heard, the last, and the words after a join.

Whether an opening grabs or an ending satisfies is judgement, and stays with
whoever writes the plan. What can be measured is narrower: a video that ends
partway through a sentence, or on somebody else's next question, has not
ended; a piece that opens on 「因為」 after the cause was cut out leans on
something the viewer never heard. Those are refused with the way out, and a
written reason gets past them when the writer means it. A last line that opens
on 「接下來」 may start the next subject or close this one, which only its
meaning says, so it is pointed out instead.
"""

import re
from typing import List, Optional, Sequence, Tuple, Union

from app.engine.plan import check_plan
from app.engine.semantic import CleanCuts, build_timeline, clean_cuts
from app.models.media import SecondListen, Span, SpeakerTurn, TranscriptWord
from app.models.plan import Selection, Trim, TrimKind
from app.models.semantic import SemanticClip
from test_edit_plan import plan_for, sourced
from test_semantic_timeline import ASSET_ID, analysis

TALK = [
    (1.0, 3.0, "今天我們去爬山。"),
    (3.5, 5.0, "山上風很大。"),
    (8.0, 10.0, "最後我們看到了日出。"),
    (12.0, 18.0, "真的很值得這樣一路走上來。"),
]

Words = Sequence[Tuple[str, float, float]]

def pointed(notes: Sequence[str]) -> List[str]:
    """The notes that say the last line may open the next subject."""
    return [note for note in notes if getattr(note, "kind", None) == "ends_on_the_next_subject"]

def checked(
    segments: Sequence[Union[Tuple[float, float, str], Words]], choose,
    voices: Optional[Sequence[Optional[str]]] = None, voiced: Optional[Sequence[Tuple[float, float]]] = None,
    quiet: Sequence[Tuple[float, float]] = (),
) -> Tuple[List[str], List[str]]:
    """Check a plan of the given talk, one file, one beat.

    Args:
        segments: Per stretch the transcriber wrote, either `(start, end,
            text)`, said as one word, or its words as `(text, start, end)`.
        choose: Turns the file's sentence clips, in order, into the plan's selections.
        voices: Who says each sentence, in the same order, None for one no turn
            covers; nobody told apart when None.
        voiced: Where a second listen heard somebody talking; not listened to again when None.
        quiet: Silences measured inside the stretches, besides the ones between them.

    Returns:
        `(problems, notes)` from `check_plan`.
    """
    worded = [list(segment) if isinstance(segment, list) else [(segment[2], segment[0], segment[1])]
              for segment in segments]
    segments = [(words[0][1], words[-1][2], "".join(text for text, _, _ in words)) for words in worded]
    silences, last = [], 0.0
    for start, end, _ in segments:
        silences.append((last, start))
        last = end
    silences.append((last, 20.0))
    silences = sorted([*silences, *quiet])
    made = analysis(20.0, segments=segments, silences=silences)
    made = made.model_copy(update={"transcript": made.transcript.model_copy(update={"segments": [
        segment.model_copy(update={"words": [TranscriptWord(text=text, start=start, end=end)
                                             for text, start, end in words]})
        for segment, words in zip(made.transcript.segments, worded)
    ]})})
    if voiced is not None:
        made = made.model_copy(update={"second_listen": SecondListen(
            voiced=[Span(start=start, end=end) for start, end in voiced], model="test")})
    if voices is not None:
        made = made.model_copy(update={"speakers": [
            SpeakerTurn(start=start, end=end, speaker=voice) for (start, end, _), voice in zip(segments, voices)
            if voice is not None
        ]})
    assets = {ASSET_ID: sourced()}
    timeline, clips = build_timeline(assets, {ASSET_ID: made})
    spoken: List[SemanticClip] = [clip for clip in clips if clip.text]
    plan = plan_for(choose(spoken)).model_copy(
        update={"timeline_id": timeline.id, "timeline_input_hash": timeline.input_hash})
    return check_plan(plan, timeline, {clip.id: clip for clip in clips}, {}, assets, {ASSET_ID: clean_cuts(made)})

def test_a_cut_that_ends_partway_through_its_last_sentence_is_refused() -> None:
    problems, _ = checked(TALK, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[3].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.0)),
    ])
    assert any("真的很值得" in problem and "partway" in problem for problem in problems)
    # The window opens on that phrase, so there is no earlier one in it to offer.
    assert not any("phrase before it" in problem for problem in problems)

def test_an_ending_that_stops_short_on_purpose_passes_with_its_reason() -> None:
    problems, _ = checked(TALK, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[3].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.0),
                  edge_reason="話講到一半被打斷，就是這支的笑點"),
    ])
    assert not any("partway" in problem for problem in problems)

def test_a_single_trim_stops_where_it_was_asked_to() -> None:
    # 「剪掉後面」 on one recording is the user's own cut, not an ending somebody chose.
    problems, _ = checked(TALK, lambda spoken: [
        Selection(clip_id=spoken[3].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.0)),
    ])
    assert problems == []

def test_the_refused_ending_is_not_also_reported_as_a_window_stopping_short() -> None:
    problems, notes = checked(TALK, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[3].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.0)),
    ])
    assert any("partway" in problem for problem in problems)
    assert not any("stop before the sentence ends" in note for note in notes)

NEXT_SUBJECT = [
    (1.0, 3.0, "今天我們去爬山。"),
    (3.5, 5.0, "山上風很大。"),
    (8.0, 10.0, "接下來我們去吃飯。"),
]

def test_a_cut_that_ends_on_the_first_line_of_the_next_subject_is_pointed_out_not_refused() -> None:
    problems, notes = checked(NEXT_SUBJECT, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ])
    assert problems == []
    assert any("接下來我們去吃飯" in note for note in pointed(notes))

def test_a_cut_that_ends_on_a_finished_line_passes() -> None:
    problems, _ = checked(NEXT_SUBJECT, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[1].id, beat_id="b1"),
    ])
    assert problems == []

INTERVIEW = [
    (1.0, 3.0, "我第一次爬山就迷路了。"),
    (3.5, 5.0, "最後是靠星星找到路的。"),
    (8.0, 10.0, "那你後來還敢再去嗎？"),
]

def test_a_cut_that_ends_on_somebody_else_asking_the_next_question_is_refused() -> None:
    problems, _ = checked(INTERVIEW, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ], voices=["S1", "S1", "S2"])
    assert any("那你後來還敢再去嗎" in problem and "question" in problem for problem in problems)

def test_a_question_the_same_voice_ends_on_is_an_ending() -> None:
    # 「你們覺得呢？」 to the camera is how a lot of videos close.
    problems, _ = checked(INTERVIEW, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ], voices=["S1", "S1", "S1"])
    assert problems == []

def test_a_question_nobody_told_the_voices_of_is_not_judged() -> None:
    problems, _ = checked(INTERVIEW, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ])
    assert problems == []

def opening_on(first_line: str) -> List[str]:
    """Check a cut of two lines whose first words are `first_line`.

    Args:
        first_line: What the video opens on.

    Returns:
        The problems.
    """
    problems, _ = checked([(1.0, 3.0, first_line), (8.0, 10.0, "真的很值得。")], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[1].id, beat_id="b1"),
    ])
    return problems

def test_a_video_that_opens_on_a_word_leaning_on_what_came_before_is_refused() -> None:
    for line in ("所以我們決定去爬山。", "然後我們就出發了。", "因為天氣很好。", "還有一件事。"):
        assert any(line in problem and "opens" in problem for problem in opening_on(line)), line

def test_a_video_that_opens_on_a_particle_no_sentence_starts_with_is_refused() -> None:
    assert any("opens" in problem for problem in opening_on("的時候我們才出發。"))

def test_a_video_may_open_on_an_exclamation_or_a_plain_sentence() -> None:
    for line in ("哇今天天氣也太好了吧。", "欸你們看這個。", "今天我們去爬山。", "那天我們去爬山。"):
        assert opening_on(line) == [], line

def test_a_window_refused_for_its_opening_still_says_it_stops_short() -> None:
    problems, notes = checked([(1.0, 7.0, "所以我們決定一路走到山頂再下來。"), (8.0, 10.0, "真的很值得。")],
                              lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=3.0)),
        Selection(clip_id=spoken[1].id, beat_id="b1"),
    ])
    assert any("所以" in problem for problem in problems)
    assert any("stop before the sentence ends" in note for note in notes)

def test_a_video_that_opens_partway_through_a_sentence_is_refused() -> None:
    problems, _ = checked([(1.0, 7.0, "今天我們決定一路走到山頂再下來。"), (8.0, 10.0, "真的很值得。")],
                          lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1", trim=Trim(kind=TrimKind.RANGE, from_seconds=3.0,
                                                                to_seconds=6.0)),
        Selection(clip_id=spoken[1].id, beat_id="b1"),
    ])
    assert any("partway" in problem and "starts" in problem for problem in problems)

STORY = [
    (1.0, 3.0, "今天我們去爬山。"),
    (3.5, 5.0, "山上風很大。"),
    (8.0, 10.0, "因為昨天下過雨。"),
    (12.0, 14.0, "然後我們就到山頂了。"),
    (15.0, 17.0, "真的很值得。"),
]

def test_a_line_leaning_on_a_cause_that_was_cut_out_is_refused_after_a_join() -> None:
    problems, _ = checked(STORY, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
        Selection(clip_id=spoken[4].id, beat_id="b1"),
    ])
    assert any("因為昨天下過雨" in problem and "after a join" in problem for problem in problems)

def test_a_line_that_carries_the_story_on_may_follow_a_join() -> None:
    problems, _ = checked(STORY, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[3].id, beat_id="b1"),
        Selection(clip_id=spoken[4].id, beat_id="b1"),
    ])
    assert problems == []

def test_a_line_heard_straight_after_the_one_before_it_in_the_file_leans_on_what_was_heard() -> None:
    problems, _ = checked(STORY, lambda spoken: [
        Selection(clip_id=spoken[1].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
        Selection(clip_id=spoken[4].id, beat_id="b1"),
    ])
    assert problems == []

def test_a_window_nobody_hears_opens_on_nothing() -> None:
    # A shot kept for its picture under the music says no 「因為」 to anybody.
    problems, _ = checked(STORY, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1", volume=0),
        Selection(clip_id=spoken[4].id, beat_id="b1"),
    ])
    assert problems == []

def test_a_join_that_leans_on_purpose_passes_with_its_reason() -> None:
    problems, _ = checked(STORY, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1", edge_reason="字卡已經交代下過雨，這句接得上"),
        Selection(clip_id=spoken[4].id, beat_id="b1"),
    ])
    assert problems == []

def test_a_join_into_the_middle_of_a_sentence_is_refused() -> None:
    problems, _ = checked([(1.0, 3.0, "今天我們去爬山。"), (5.0, 11.0, "我們一路慢慢走到山頂再下來。"),
                           (12.0, 14.0, "真的很值得。")], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
        Selection(clip_id=spoken[1].id, beat_id="b1", trim=Trim(kind=TrimKind.RANGE, from_seconds=3.0,
                                                                to_seconds=7.0)),
    ])
    assert any("after a join" in problem and "partway" in problem for problem in problems)

# One stretch the transcriber ran two sentences together in, with a pause between them.
RUN_ON = [("今天我們一路走", 1.0, 3.0), ("走得很累。", 3.0, 4.0), ("後來我們到了山頂。", 5.0, 7.0)]

def test_a_join_at_a_pause_inside_a_run_on_stretch_starts_a_sentence() -> None:
    problems, _ = checked([(0.1, 0.4, "出發。"), (0.5, 0.9, "天還沒亮。"), RUN_ON, (12.0, 14.0, "真的很值得。")],
                          lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=2.0)),
        Selection(clip_id=spoken[3].id, beat_id="b1"),
    ])
    assert problems == []

def test_the_last_line_is_the_last_sentence_heard_not_the_start_of_its_stretch() -> None:
    stretch = [("然後呢我們走了很久。", 1.0, 4.0), ("今天就到這裡。", 5.0, 7.0)]
    problems, _ = checked([(0.2, 0.8, "出發。"), stretch], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[1].id, beat_id="b1"),
    ])
    assert problems == []

# The transcriber laid these words end to end and stretched 「我」 over the pause before it; a
# second listen heard nobody between 3.2s and 3.6s.
GLUED = [("今天我們", 1.0, 2.0), ("走得很累", 2.0, 3.0), ("我", 3.0, 4.0), ("後來到了山頂。", 4.0, 6.0)]

def test_a_pause_a_second_listen_heard_begins_a_phrase_the_word_timings_ran_over() -> None:
    problems, _ = checked([(0.1, 0.4, "出發。"), (0.5, 0.9, "天還沒亮。"), GLUED, (12.0, 14.0, "真的很值得。")],
                          lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=3.0)),
        Selection(clip_id=spoken[3].id, beat_id="b1"),
    ], voiced=[(0.1, 0.9), (1.0, 3.2), (3.6, 6.0), (12.0, 14.0)])
    assert problems == []

def test_without_a_second_listen_words_laid_end_to_end_are_one_phrase() -> None:
    problems, _ = checked([(0.1, 0.4, "出發。"), (0.5, 0.9, "天還沒亮。"), GLUED, (12.0, 14.0, "真的很值得。")],
                          lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=3.0)),
        Selection(clip_id=spoken[3].id, beat_id="b1"),
    ])
    assert any("partway" in problem for problem in problems)

def test_a_cut_in_real_quiet_does_not_cut_into_the_word_stretched_over_it() -> None:
    # 「那」 was given 3.5s to 6.5s, but nobody speaks until 5.7s: the cut at 5.0s takes nothing off.
    stretch = [("那", 3.5, 6.5), ("我們就回家了。", 6.5, 8.5)]
    problems, _ = checked([(0.1, 0.4, "出發。"), (1.0, 3.0, "天還沒亮。"), stretch, (12.0, 14.0, "真的很值得。")],
                          lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=3.5)),
        Selection(clip_id=spoken[3].id, beat_id="b1"),
    ], voiced=[(0.1, 0.4), (1.0, 3.0), (5.7, 8.5), (12.0, 14.0)])
    assert problems == []

def test_a_phrase_opened_after_the_particle_the_transcriber_hung_on_its_front_starts_clean() -> None:
    # 「首先呢」 heard with the pause before 「呢」: starting on 「先」 takes only the particle off.
    stretch = [("首先", 1.0, 1.6), ("呢", 2.0, 2.3), ("先把它關機。", 2.3, 4.0)]
    problems, _ = checked([(0.1, 0.4, "出發。"), (0.5, 0.9, "天還沒亮。"), stretch, (12.0, 14.0, "真的很值得。")],
                          lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=1.7)),
        Selection(clip_id=spoken[3].id, beat_id="b1"),
    ])
    assert problems == []

def test_a_refused_ending_says_where_its_phrase_finishes_and_what_that_costs_the_video() -> None:
    problems, _ = checked(TALK, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[3].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.0), speed=2.0),
    ])
    # The phrase finishes at 18.00s of the file; 4s of it at double speed is 2s of the video.
    assert any("18.00s" in problem and "2.0s more of the video" in problem for problem in problems)

def test_a_cut_that_ends_on_the_next_subject_with_a_comma_after_it_is_pointed_out() -> None:
    # Transcribers write 「接下來，」, and the comma alone does not make what follows a new line.
    _, notes = checked([*NEXT_SUBJECT[:2], [("接下來，", 8.0, 8.8), ("我們去吃飯。", 8.8, 10.0)]], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ])
    assert any("接下來，我們去吃飯" in note for note in pointed(notes))

def test_a_question_whose_answer_no_turn_covers_is_not_judged() -> None:
    problems, _ = checked(INTERVIEW, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ], voices=[None, None, "S2"])
    assert problems == []

def test_a_reason_on_one_of_two_selections_of_the_same_line_excuses_that_one_only() -> None:
    problems, _ = checked(STORY, lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1", edge_reason="字卡已經交代下過雨，這句接得上"),
        Selection(clip_id=spoken[4].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
        Selection(clip_id=spoken[4].id, beat_id="b1"),
    ])
    assert len([problem for problem in problems if "因為昨天下過雨" in problem]) == 1

def test_a_video_whose_first_words_are_the_first_said_in_the_file_is_told_to_open_elsewhere() -> None:
    # There is no earlier start to offer: nothing is said before 「所以」.
    assert any("open on another piece" in problem for problem in opening_on("所以我們決定去爬山。"))

def test_a_video_may_open_on_a_word_that_only_starts_with_a_particle() -> None:
    assert opening_on("的確很值得去一次。") == []

def test_an_ending_cut_in_real_quiet_does_not_stop_partway_through_the_word_stretched_over_it() -> None:
    # 「真的很值得。」 was given 12s to 16s, but nobody speaks after 13.5s: ending at 14s finishes it.
    problems, _ = checked([*TALK[:3], (12.0, 16.0, "真的很值得。")], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[3].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.0)),
    ], voiced=[(1.0, 3.0), (3.5, 5.0), (8.0, 10.0), (12.0, 13.5)])
    assert problems == []

def joined_at(stretch: Sequence[Union[Tuple[float, float, str], Words]], **given) -> List[str]:
    """Check a cut that joins into the stretch starting at 4s, after a line from elsewhere in the file.

    The words run end to end, so no gap between them says where a phrase begins.

    Args:
        stretch: The transcriber's stretches from 1s to 6s.
        **given: Passed on to `checked`.

    Returns:
        The problems.
    """
    problems, _ = checked([(0.1, 0.4, "出發。"), *stretch, (12.0, 14.0, "真的很值得。")], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[-2].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=2.0)),
        Selection(clip_id=spoken[-1].id, beat_id="b1"),
    ], **given)
    return problems

def test_words_laid_end_to_end_with_nothing_between_them_are_one_phrase() -> None:
    assert any("partway" in problem for problem in joined_at(
        [[("今天我們一路走", 1.0, 3.0), ("走得很累", 3.0, 4.0), ("後來我們到了山頂。", 4.0, 6.0)]]))

def test_punctuation_ends_a_phrase_the_word_timings_run_straight_through() -> None:
    assert joined_at([[("今天我們一路走", 1.0, 3.0), ("走得很累。", 3.0, 4.0), ("後來我們到了山頂。", 4.0, 6.0)]]) == []

def test_a_space_the_transcriber_put_between_words_ends_a_phrase() -> None:
    assert joined_at([[("今天我們一路走", 1.0, 3.0), ("走得很累", 3.0, 4.0), (" 後來我們到了山頂。", 4.0, 6.0)]]) == []

def test_a_measured_silence_the_word_timings_run_over_ends_a_phrase() -> None:
    assert joined_at([[("今天我們一路走", 1.0, 3.0), ("走得很累", 3.0, 4.0), ("後來我們到了山頂。", 4.0, 6.0)]],
                     quiet=[(3.6, 4.0)]) == []

def test_a_silence_that_began_before_the_earlier_word_and_runs_to_the_next_ends_a_phrase() -> None:
    assert joined_at([[("今天我們一路走", 1.0, 3.0), ("走得很累", 3.0, 4.0), ("後來我們到了山頂。", 4.0, 6.0)]],
                     quiet=[(2.9, 4.0)]) == []

def test_a_gap_between_two_listens_is_not_lost_behind_a_shorter_pause_inside_it() -> None:
    # Asked of the cuts directly: through a plan, the shorter pause also splits the transcriber's line,
    # and the window then opens elsewhere without ever asking about the word at 4.0.
    cuts = CleanCuts(words=((1.0, 3.0), (3.0, 4.0), (4.0, 6.0)), voiced=((0.0, 2.5), (4.0, 10.0)),
                     pauses=((2.9, 3.4),), transcribed=True, listened=True)
    assert cuts.begins_phrase(2)

def test_a_long_pause_is_not_lost_behind_a_shorter_one_nested_in_it() -> None:
    assert joined_at([[("今天我們一路走", 1.0, 3.0), ("走得很累", 3.0, 4.0), ("後來我們到了山頂。", 4.0, 6.0)]],
                     quiet=[(2.0, 4.0), (2.5, 3.5)]) == []

def test_a_silence_ending_just_inside_the_earlier_word_s_start_is_a_pause_before_it() -> None:
    assert any("partway" in problem for problem in joined_at(
        [[("今天我們一路走", 1.0, 3.0), ("走得很累", 3.0, 4.0), ("後來我們到了山頂。", 4.0, 6.0)]], quiet=[(2.5, 3.2)]))

def test_a_sentence_the_transcriber_started_starts_a_phrase() -> None:
    assert joined_at([[("今天我們一路走", 1.0, 3.0), ("走得很累", 3.0, 4.0)], [("後來我們到了山頂", 4.0, 6.0)]]) == []

def test_a_cut_that_ends_on_the_next_subject_after_a_word_and_a_comma_is_pointed_out() -> None:
    _, notes = checked([*NEXT_SUBJECT[:2], [("好，", 8.0, 8.4), ("接下來", 8.4, 9.0), ("我們去吃飯。", 9.0, 10.0)]],
                          lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ])
    assert any("好，接下來我們去吃飯" in note and "end on the piece before it" in note for note in pointed(notes))

def test_a_next_subject_pointed_out_says_where_the_phrase_before_it_finishes() -> None:
    _, notes = checked([*NEXT_SUBJECT[:2], [("好吃。", 8.0, 8.6), ("接下來我們去吃飯。", 8.6, 10.0)]], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ])
    assert any("8.60s of the file" in note for note in pointed(notes))

def test_a_refused_ending_offers_the_phrase_before_it_in_the_window() -> None:
    problems, _ = checked([*TALK[:3], [("今天天氣很好。", 12.0, 13.0), ("我們一路走上來", 13.0, 18.0)]], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[3].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=3.0)),
    ])
    assert any("partway" in problem and "18.00s" in problem and "13.00s of the file" in problem
               for problem in problems)

def test_a_question_split_by_a_comma_is_still_somebody_else_asking() -> None:
    problems, _ = checked([*INTERVIEW[:2], [("那你呢，", 8.0, 8.8), ("後來還敢再去嗎？", 8.8, 10.0)]], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ], voices=["S1", "S1", "S2"])
    assert any("那你呢，後來還敢再去嗎" in problem and "question" in problem for problem in problems)

def test_an_ending_on_a_window_merged_from_two_selections_names_the_last_one() -> None:
    # The two lines nearly touch, so they play as one window; its end is the second selection's.
    holder: List[SemanticClip] = []

    def choose(spoken: List[SemanticClip]) -> List[Selection]:
        holder.extend(spoken)
        return [
            Selection(clip_id=spoken[0].id, beat_id="b1"),
            Selection(clip_id=spoken[1].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.0)),
        ]

    problems, _ = checked([(1.0, 3.0, "今天我們去爬山。"), (3.1, 9.0, "真的很值得這樣一路走上來。")], choose)
    assert any(f"{holder[1].id} ends" in problem and "partway" in problem for problem in problems)

def test_a_join_refused_after_its_particle_comes_off_offers_the_phrase_before_the_particle() -> None:
    # 「呢」 belongs to 「首先呢」: opening on it would open on a tail, so the start offered is 「首先」's.
    stretch = [("首先", 1.0, 1.6), ("呢", 2.0, 2.3), ("所以把它關機。", 2.3, 4.0)]
    problems, _ = checked([(0.1, 0.4, "出發。"), (0.5, 0.9, "天還沒亮。"), stretch, (12.0, 14.0, "真的很值得。")],
                          lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=1.7)),
        Selection(clip_id=spoken[3].id, beat_id="b1"),
    ])
    assert any("所以" in problem and "phrase before it in the file, at 1.00s" in problem for problem in problems)

def test_a_window_like_the_refused_ending_still_says_it_stops_short() -> None:
    # Only the last window's ending is refused; the same cut heard earlier is still owed its note.
    _, notes = checked(TALK, lambda spoken: [
        Selection(clip_id=spoken[3].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.0)),
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[3].id, beat_id="b1", trim=Trim(kind=TrimKind.HEAD, seconds=2.0)),
    ])
    assert any("stop before the sentence ends" in note for note in notes)

def test_a_join_partway_through_a_phrase_is_offered_its_first_word_past_the_particle() -> None:
    # Starting where the phrase begins would open on 「呢」, a tail; 「先把」 at 2.3s is where it can start.
    stretch = [("首先", 1.0, 1.6), ("呢", 2.0, 2.3), ("先把", 2.3, 2.9), ("它關機。", 2.9, 4.0)]
    problems, _ = checked([(0.1, 0.4, "出發。"), (0.5, 0.9, "天還沒亮。"), stretch, (12.0, 14.0, "真的很值得。")],
                          lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=1.1)),
        Selection(clip_id=spoken[3].id, beat_id="b1"),
    ])
    assert any("partway" in problem and "2.30s of the file" in problem for problem in problems)

def test_a_sound_made_before_speaking_does_not_hide_the_word_that_leans() -> None:
    assert any("所以" in problem and "opens" in problem for problem in opening_on("嗯，所以我們決定去爬山。"))

def test_a_sound_made_before_speaking_does_not_hide_the_next_subject() -> None:
    _, notes = checked([*NEXT_SUBJECT[:2], (8.0, 10.0, "嗯接下來我們去吃飯。")], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ])
    assert any("接下來" in note for note in pointed(notes))

def test_a_pause_the_word_before_was_stretched_over_does_not_also_begin_the_word_after() -> None:
    # 「我們」 was given 1.2s to 2.0s but said from 1.8s: the pause is before 「我們」, not after it,
    # so 「去吃飯」 is still partway through 「我們去吃飯好開心」.
    stretch = [("今天", 1.0, 1.2), ("我們", 1.2, 2.0), ("去吃飯", 2.0, 3.0), ("好開心", 3.0, 6.0)]
    problems, _ = checked([(0.1, 0.4, "出發。"), stretch, (12.0, 14.0, "真的很值得。")], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[1].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=4.0)),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ], voiced=[(0.1, 0.4), (1.0, 1.2), (1.8, 6.0), (12.0, 14.0)])
    assert any("partway" in problem and "starts" in problem for problem in problems)

# Real transcripts split far finer than a line a word: most words are one character, and a
# comma stays on the word before it.

def test_a_filler_the_transcriber_gave_its_own_comma_does_not_hide_the_word_that_leans() -> None:
    problems, _ = checked([[("啊,", 1.0, 1.3), ("所以", 1.3, 1.8), ("我們決定去爬山。", 1.8, 3.0)],
                           (8.0, 10.0, "真的很值得。")], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[1].id, beat_id="b1"),
    ])
    assert any("所以" in problem and "opens" in problem for problem in problems)

def test_the_earlier_start_offered_is_never_a_filler_on_its_own() -> None:
    # Opening on 「啊,」 would still open on 「所以」, so the start offered is the line before it.
    stretch = [("出門前", 1.0, 1.5), ("先檢查。", 1.5, 2.0), ("啊,", 2.5, 2.8), ("所以", 2.8, 3.3), ("我們出發。", 3.3, 4.0)]
    problems, _ = checked([(0.1, 0.4, "出發。"), stretch, (12.0, 14.0, "真的很值得。")], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[1].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=1.2)),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ])
    assert any("所以" in problem and "phrase before it in the file, at 1.00s" in problem for problem in problems)

def test_a_join_into_a_word_split_in_two_is_partway_through_it() -> None:
    for first, second in (("的", "確很值得去一次。"), ("了", "解之後我們出發。")):
        stretch = [(first, 1.0, 1.2), (second[0], 1.2, 1.4), (second[1:], 1.4, 3.0)]
        problems, _ = checked([(0.1, 0.4, "出發。"), stretch, (12.0, 14.0, "真的很值得。")], lambda spoken: [
            Selection(clip_id=spoken[0].id, beat_id="b1"),
            Selection(clip_id=spoken[1].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=1.8)),
            Selection(clip_id=spoken[2].id, beat_id="b1"),
        ])
        assert any("partway" in problem for problem in problems), first

def test_then_or_also_partway_through_a_finished_line_ends_it_well() -> None:
    for last_line in ([("先把它關機，", 8.0, 9.0), ("再來把電池拔掉。", 9.0, 10.0)],
                      [("這家很好吃，", 8.0, 9.0), ("另外價格也便宜。", 9.0, 10.0)]):
        problems, notes = checked([*NEXT_SUBJECT[:2], last_line], lambda spoken: [
            Selection(clip_id=spoken[0].id, beat_id="b1"),
            Selection(clip_id=spoken[2].id, beat_id="b1"),
        ])
        assert problems == [] and pointed(notes) == [], last_line[0][0]

def test_a_silence_nearer_the_end_of_the_earlier_word_is_after_it() -> None:
    # The transcriber started 「走得很累」 late and ran it on to 4.0s; the quiet from 3.3s is after it.
    assert joined_at([[("今天我們一路走", 1.0, 3.0), ("走得很累", 3.0, 4.0), ("後來我們到了山頂。", 4.0, 6.0)]],
                     quiet=[(3.3, 3.95)]) == []

def ending_on(last_line: Words, *more: Union[Tuple[float, float, str], Words], **given) -> List[str]:
    """Check a cut of two lines from one file whose last words are `last_line`, after any `more` stretches.

    Args:
        last_line: The stretch the video ends on.
        more: Stretches after the first line and before the last one, selected in order too.
        **given: Passed on to `checked`.

    Returns:
        The problems, and the notes saying the last line may open the next subject.
    """
    problems, notes = checked([(1.0, 3.0, "今天我們去爬山。"), *more, last_line], lambda spoken: [
        Selection(clip_id=clip.id, beat_id="b1") for clip in spoken
    ], **given)
    return problems + pointed(notes)

def test_a_pause_with_no_comma_parts_a_line_the_way_a_comma_does() -> None:
    # Most breaks in real transcripts carry no punctuation: a pause alone.
    assert ending_on([("先把它關機", 8.0, 8.7), ("再來把電池拔掉。", 9.0, 10.0)]) == []
    assert any("next subject" in said for said in ending_on([("接下來", 8.0, 8.6), ("我們去吃飯。", 8.9, 10.0)]))

def test_a_line_split_across_two_windows_running_straight_on_is_still_one_line() -> None:
    assert any("next subject" in said for said in ending_on((9.6, 10.6, "我們去吃飯。"), (8.0, 8.6, "接下來，")))

def test_a_join_partway_through_a_leaning_line_is_not_offered_its_start() -> None:
    # Starting on 「所以」 would be refused for leaning, so the start offered is the line before it.
    stretch = [("出門前先檢查。", 5.0, 7.0), ("所以", 8.0, 8.4), ("我們就出發了。", 8.4, 10.0)]
    problems, _ = checked([(0.1, 0.4, "出發。"), stretch, (12.0, 14.0, "真的很值得。")], lambda spoken: [
        Selection(clip_id=spoken[0].id, beat_id="b1"),
        Selection(clip_id=spoken[1].id, beat_id="b1", trim=Trim(kind=TrimKind.TAIL, seconds=1.6)),
        Selection(clip_id=spoken[2].id, beat_id="b1"),
    ])
    assert any("partway" in problem and "5.00s" in problem and "8.00s" not in problem for problem in problems)

def test_a_word_that_only_starts_with_a_particle_may_open_a_video() -> None:
    assert opening_on("吧台在那邊。") == []

def opened_at(stretch: Words, at: float, **given) -> List[str]:
    """Check a cut that joins into `stretch` at `at` seconds of the file and plays it to its end.

    A line from earlier in the file comes first, and one after it is skipped, so the join is not
    straight on.

    Args:
        stretch: The stretch joined into, its words as `(text, start, end)`.
        at: Where in the file the join opens.
        **given: Passed on to `checked`.

    Returns:
        The problems about where a piece after a join opens.
    """
    def choose(spoken: List[SemanticClip]) -> List[Selection]:
        joined = spoken[2].source_range
        return [
            Selection(clip_id=spoken[0].id, beat_id="b1"),
            Selection(clip_id=spoken[2].id, beat_id="b1", trim=Trim(
                kind=TrimKind.RANGE, from_seconds=round(at - joined.start, 3),
                to_seconds=round(joined.end - joined.start, 3))),
            Selection(clip_id=spoken[3].id, beat_id="b1"),
        ]

    problems, _ = checked([(0.1, 0.4, "出發。"), (1.0, 2.0, "天還沒亮。"), stretch, (12.0, 14.0, "真的很值得。")],
                          choose, **given)
    return [problem for problem in problems if "after a join" in problem]

def follows(stretch: Words, at: float, **given) -> None:
    """Assert a join at `at` is refused, and that starting where the refusal offers is not."""
    refused = opened_at(stretch, at, **given)
    assert len(refused) == 1, refused
    offered = re.search(r"start on [^,]*?, at (\d+\.\d+)s", refused[0])
    assert offered is not None, refused[0]
    assert opened_at(stretch, float(offered.group(1)), **given) == [], refused[0]

def test_the_start_offered_past_a_pause_the_word_was_stretched_over_can_be_cut_at() -> None:
    follows([("首先", 3.5, 4.6), ("呢", 4.6, 4.8), ("先把它關機。", 4.8, 6.0)], 4.8,
            voiced=[(0.1, 0.4), (1.0, 2.0), (3.5, 3.6), (4.4, 6.0), (12.0, 14.0)])

def test_the_phrase_offered_before_a_leaning_one_is_never_a_lone_particle() -> None:
    follows([("首先", 3.0, 3.6), ("呢", 4.0, 4.3), ("所以", 4.8, 5.3), ("把它關機。", 5.3, 6.0)], 4.8)

def test_the_phrase_offered_before_a_leaning_one_does_not_lean_itself() -> None:
    follows([("我們吃完早餐。", 3.0, 4.5), ("所以先檢查。", 5.0, 7.0), ("所以", 8.0, 8.4), ("我們就出發了。", 8.4, 10.0)],
            8.4)
