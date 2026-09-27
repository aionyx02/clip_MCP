"""The second listen: how a file's voiced stretches are handed to the speech model."""

from app.engine.relisten import JOIN_SECONDS, STRETCH_PAD_SECONDS, stretches_to_transcribe

def test_stretches_close_together_are_heard_as_one_phrase() -> None:
    pieces = stretches_to_transcribe([(1.0, 2.0), (2.0 + JOIN_SECONDS / 2, 3.0), (6.0, 7.0)], 10.0)
    assert pieces == [(1.0 - STRETCH_PAD_SECONDS, 3.0 + STRETCH_PAD_SECONDS),
                      (6.0 - STRETCH_PAD_SECONDS, 7.0 + STRETCH_PAD_SECONDS)]

def test_the_padding_never_runs_into_the_next_stretch_or_off_the_file() -> None:
    pieces = stretches_to_transcribe([(0.1, 1.0), (1.9, 2.5), (9.8, 9.95)], 10.0)
    assert pieces[0][0] == 0.0 and pieces[-1][1] == 10.0
    assert all(first[1] <= second[0] for first, second in zip(pieces, pieces[1:]))
