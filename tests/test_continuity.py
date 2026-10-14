"""Tests for finding a shot dropped between two that belong together."""

from datetime import datetime, timedelta, timezone

from app.engine.continuity import FAR, Shot, backward_steps, strays

MORNING = datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc)

def at(minutes: float, **fields) -> Shot:
    """A shot taken this many minutes after the morning began."""
    return Shot(label=f"s{minutes:g}", moment=MORNING + timedelta(minutes=minutes), **fields)

def test_an_afternoon_shot_between_two_morning_ones_is_named() -> None:
    found = strays([at(0), at(1), at(245), at(2), at(3)])
    assert len(found) == 1 and found[0].startswith("s245 sits between s1 and s2")

def test_two_strays_in_a_row_are_named_together() -> None:
    found = strays([at(0), at(300), at(301), at(2)])
    assert len(found) == 1 and found[0].startswith("s300, s301 sits between s0 and s2")

def test_a_day_told_in_order_is_left_alone() -> None:
    assert strays([at(minutes) for minutes in (0, 40, 90, 200, 380, 600)]) == []

def test_a_new_part_may_begin_anywhere_in_time() -> None:
    assert strays([at(0), at(245, opens_part=True), at(1)]) == []
    assert strays([at(0), at(245, exempt=True), at(1)]) == []

def test_a_shot_off_the_topic_its_neighbours_share_is_named() -> None:
    found = strays([Shot("a", topic="測試"), Shot("b", topic="聊天"), Shot("c", topic="測試")])
    assert found and "「測試」" in found[0] and "「聊天」" in found[0]

def test_nothing_is_said_about_a_shot_whose_time_is_unknown() -> None:
    assert strays([at(0), Shot("x"), at(1)]) == []
    assert FAR > timedelta(0)

def test_a_part_that_opens_on_the_night_and_goes_back_to_the_day_is_named() -> None:
    # No stray between two that agree: the night opens the part, and the day follows it.
    shots = [at(0), at(720, opens_part=True), at(10), at(20)]
    assert strays(shots) == []
    by_time, by_light = backward_steps(shots)
    assert len(by_time) == 1 and by_time[0].startswith("s10 was shot at") and by_light == []
    # A reason given, or a new part, is a step back meant.
    assert backward_steps([at(720), at(10, exempt=True)]) == ([], [])
    assert backward_steps([at(720), at(10, opens_part=True)]) == ([], [])
    assert backward_steps([at(720), at(600)]) == ([], [])

def test_with_no_times_a_dark_shot_then_a_bright_one_is_only_a_note() -> None:
    _, by_light = backward_steps([Shot("night", exposure=0.1), Shot("day", exposure=0.6)])
    assert [pair for _, pair in by_light] == [("night", "day")]
    assert backward_steps([Shot("day", exposure=0.6), Shot("night", exposure=0.1)]) == ([], [])
