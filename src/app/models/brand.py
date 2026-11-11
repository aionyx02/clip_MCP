"""A brand kit: the logo, colour, font and closing card every video of one brand wears.

Said once and kept for every project after, like a preference, but as settings
rather than words: a plan names the kit, and the compile puts its logo over the
cut and its closing card at the end, where the length, the music and every
check already count them.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.models.timeline.parts import (
    COLOUR, TITLE_CARD_LONGEST, TITLE_CARD_SHORTEST, LogoCorner,
)


def _utc_now() -> datetime:
    """The time now, in UTC."""
    return datetime.now(timezone.utc)


class EndCard(BaseModel):
    """The card a brand's videos close on: what to do next, with the logo above it."""

    title: str = Field(..., min_length=1, max_length=60, description="The call to action, such as 「追蹤看更多」")
    subtitle: Optional[str] = Field(default=None, max_length=80, description="A smaller line under it")
    # Provisional: long enough to read a short call to action twice; not measured.
    seconds: float = Field(default=3.0, ge=float(TITLE_CARD_SHORTEST), le=float(TITLE_CARD_LONGEST),
                           description="How long it stays up; counted in the cut's length")
    logo: bool = Field(default=True, description="Draw the kit's logo above the words")


class BrandKit(BaseModel):
    """How every video of one brand looks, kept across projects."""

    name: str = Field(..., min_length=1, max_length=40, description="What the user calls it; a plan names it by this")
    logo_asset_id: Optional[str] = Field(
        default=None, description="The logo, a picture already in the library; a PNG with a clear background "
                                  "sits on the footage without a box around it",
    )
    logo_corner: LogoCorner = Field(default="top_right")
    # Provisional: faint enough not to pull the eye, solid enough to read; not measured.
    logo_opacity: float = Field(default=0.85, gt=0, le=1)
    # Provisional: about what broadcasters' corner bugs take; not measured.
    logo_width: float = Field(default=0.12, ge=0.03, le=0.4, description="As a share of the frame's width")
    colour: Optional[str] = Field(
        default=None, pattern=COLOUR,
        description="The brand's colour, behind its closing card and behind any title card asked for on a "
                    "`colour` background without a colour of its own",
    )
    font: Optional[str] = Field(
        default=None, max_length=80,
        description="The font family of the captions, title cards and words over the footage",
    )
    end_card: Optional[EndCard] = Field(default=None, description="The card every video closes on")
    saved_at: datetime = Field(default_factory=_utc_now)
