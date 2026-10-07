"""Laying a track's clips out in frames: clips that touch share a frame boundary."""

from decimal import Decimal
from fractions import Fraction

import pytest

from app.engine.builder import _layout
from app.models.timeline import Clip, TimeRange


def clip(clip_id: str, start: str, end: str, at: str, speed: float = 1.0) -> Clip:
    """Build a clip of one file at a place on the timeline."""
    return Clip(id=clip_id, asset_id="x", timeline_in=Decimal(at), speed=speed,
                source_range=TimeRange(start=Decimal(start), end=Decimal(end)))


def test_a_clip_at_another_speed_touches_the_next_rather_than_overlapping_it() -> None:
    # Played 1.1x, the first ends at 10.3834545…s; the next starts at 10.383, kept to the millisecond.
    # At 30 fps the one rounds up to frame 312 and the other down to 311.
    first = clip("a", "0", "11.4218", "0", speed=1.1)
    second = clip("b", "20", "25", "10.383")
    laid = _layout([first, second], Fraction(30))
    assert [(segment.start_frame, segment.end_frame) for segment in laid] == [(0, 311), (311, 461)]


def test_a_real_overlap_is_still_refused() -> None:
    with pytest.raises(ValueError, match="overlaps the previous clip"):
        _layout([clip("a", "0", "5", "0"), clip("b", "0", "5", "4.9")], Fraction(30))
