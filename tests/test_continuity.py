"""Tests for finding a shot dropped between two that belong together."""

from datetime import datetime, timedelta, timezone

from app.engine.continuity import FAR, Shot, strays

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
