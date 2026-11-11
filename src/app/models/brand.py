"""A brand kit: the logos, colours, font and closing words every video of one brand wears.

Said once and kept for every project after, like a preference, but as settings
rather than words: a plan names the kit, and the compile puts its closing words
over the end of the last shot (or a closing card after it), its colours on the
cards and the captions, and its logo in a corner when the kit asks for one.
"""

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.timeline.parts import (
    COLOUR, TITLE_CARD_LONGEST, TITLE_CARD_SHORTEST, LogoCorner,
)

# Provisional: long enough to read a short call to action twice over the last shot; not measured.
OVERLAY_SECONDS = 3.0
# Provisional: a closing card is extra length nobody watches for; long enough to read three or
# four words once. Not measured.
CARD_SECONDS = 1.5


def _utc_now() -> datetime:
    """The time now, in UTC."""
    return datetime.now(timezone.utc)


def luminance(colour: str) -> float:
    """The relative luminance of a colour, 0 for black and 1 for white (WCAG 2's formula).

    Args:
        colour: Six hex digits after a `#`, or `black` or `white`.

    Returns:
        Its luminance.
    """
    hexed = {"black": "#000000", "white": "#ffffff"}.get(colour, colour)

    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(int(hexed[index:index + 2], 16)) for index in (1, 3, 5))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


class EndCard(BaseModel):
    """What a brand's videos close on: what to do next, over the last shot or on a card of its own."""

    title: str = Field(..., min_length=1, max_length=60, description="The call to action, such as 「追蹤看更多」")
    subtitle: Optional[str] = Field(default=None, max_length=60, description="A smaller line under it")
    mode: Literal["overlay", "card"] = Field(
        default="overlay",
        description="`overlay`: the words over the last seconds of the last shot, adding nothing to the length; "
                    "`card`: a card of its own after the last shot, counted in the length",
    )
    seconds: Optional[float] = Field(
        default=None, ge=float(TITLE_CARD_SHORTEST), le=float(TITLE_CARD_LONGEST),
        description=f"How long it stays up; null is {OVERLAY_SECONDS:g}s over the last shot or "
                    f"{CARD_SECONDS:g}s for a card. Over the last shot, that shot has to play at least this long",
    )
    logo: bool = Field(default=True, description="For a card: draw the kit's logo above the words")

    @property
    def held(self) -> float:
        """How long it stays up, its mode's default when not said."""
        if self.seconds is not None:
            return self.seconds
        return OVERLAY_SECONDS if self.mode == "overlay" else CARD_SECONDS


class BrandKit(BaseModel):
    """How every video of one brand looks, kept across projects."""

    name: str = Field(..., min_length=1, max_length=40, description="What the user calls it; a plan names it by this")
    logo_asset_id: Optional[str] = Field(
        default=None, description="The logo, a picture already in the library; the light version, for dark "
                                  "backgrounds. A PNG with a clear background sits on the footage without a box",
    )
    logo_dark_asset_id: Optional[str] = Field(
        default=None, description="The dark version of the logo, for light backgrounds, if the brand has one",
    )
    corner_logo: bool = Field(
        default=False, description="Hold the logo in a corner over the whole cut; only when the user asks for it",
    )
    logo_corner: LogoCorner = Field(default="top_right")
    # Provisional: faint enough not to pull the eye, solid enough to read; not measured.
    logo_opacity: float = Field(default=0.85, gt=0, le=1)
    # Provisional: about what broadcasters' corner bugs take; not measured.
    logo_width: float = Field(default=0.12, ge=0.03, le=0.4, description="As a share of the frame's width")
    colour: Optional[str] = Field(
        default=None, pattern=COLOUR,
        description="The brand's main colour, behind its closing card and behind any title card asked for on a "
                    "`colour` background without a colour of its own",
    )
    accent: Optional[str] = Field(
        default=None, pattern=COLOUR,
        description="The brand's accent colour, for the title line of its cards and its closing words",
    )
    text_colour: str = Field(
        default="white", pattern=COLOUR,
        description="The colour of the words on its cards, its closing words and the captions",
    )
    font: Optional[str] = Field(
        default=None, max_length=80,
        description="The font family of the captions, title cards and words over the footage",
    )
    end_card: Optional[EndCard] = Field(default=None, description="What every video closes on")
    saved_at: datetime = Field(default_factory=_utc_now)

    @property
    def logos(self) -> set:
        """The files the kit takes as its logo."""
        return {asset_id for asset_id in (self.logo_asset_id, self.logo_dark_asset_id) if asset_id}

    @property
    def corner_logo_asset_id(self) -> Optional[str]:
        """The logo held over the footage: the light one, which reads on most pictures, else the dark one."""
        return self.logo_asset_id or self.logo_dark_asset_id

    def logo_on(self, background: Optional[str]) -> Optional[str]:
        """The logo that stands out from a card's background.

        Args:
            background: The card's plain colour, or None for footage behind
                it, blurred and darkened, which takes the light logo.

        Returns:
            Whichever of the two logos contrasts more with the background,
            measured as WCAG contrast against white and against black; the
            only logo when there is one; None when there is none.
        """
        if background is None or not (self.logo_asset_id and self.logo_dark_asset_id):
            return self.corner_logo_asset_id
        lit = luminance(background)
        against_white, against_black = 1.05 / (lit + 0.05), (lit + 0.05) / 0.05
        return self.logo_asset_id if against_white >= against_black else self.logo_dark_asset_id
