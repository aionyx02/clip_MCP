"""What a project is made of: time ranges, clips, tracks, captions, title cards, text and the project itself."""

from decimal import Decimal
from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class TimeRange(BaseModel):
    """A time span in seconds, where `end` must be greater than `start`."""

    start : Decimal = Field(..., ge=0, description="Start time (seconds)")
    end : Decimal = Field(..., ge=0, description="End time (seconds)")

    @model_validator(mode="after")
    def validate_range(self):
        """Ensure the range has a positive duration.

        Returns:
            The validated `TimeRange` instance.

        Raises:
            ValueError: If `end` is less than or equal to `start`.
        """
        if self.end <= self.start:
            raise ValueError("End time must be greater than start time")
        return self

    @property
    def duration(self) -> Decimal :
        """Length of the range in seconds.

        Returns:
            The difference between `end` and `start`.
        """
        return self.end - self.start

class CueWord(BaseModel):
    """One word of a caption with its own timing, for captions that light up word by word.

    The times are in whatever the caption holding them is measured in: source
    seconds on a stored caption, timeline seconds once it has been placed. The
    caption is the anchor either way, so the words never need rewriting on
    their own.
    """

    start: Decimal = Field(..., ge=0, description="When the word begins (seconds)")
    end: Decimal = Field(..., ge=0, description="When it ends (seconds)")
    text: str = Field(..., description="The word itself")

class SubtitleCue(BaseModel):
    """One caption, anchored to the words it transcribes rather than to a moment in the cut.

    A caption is a record of something said at a particular second of a
    particular file, and that stays true however the cut is rearranged. Storing
    where it lands on the timeline instead meant every caption was wrong the
    moment a clip moved, so captions had to be made last and made again after
    any change. Anchored here, they follow the footage: a clip that moves
    carries its lines with it, a clip that is deleted takes its lines with it,
    and a clip split in two hands the line across the cut to both halves.

    `place_cues` works out where each one falls in the cut as it stands.
    """

    id: str = Field(default="", description="Stable ID for this caption, so a single line can be corrected on its own")
    asset_id: str = Field(..., min_length=1, description="Source file the words were spoken in")
    source_start: Decimal = Field(..., ge=0, description="When they begin in that file (seconds)")
    source_end: Decimal = Field(..., ge=0, description="When they end in that file (seconds)")
    text: str = Field(..., description="The caption text; it is wrapped to fit the frame when rendered")
    secondary: str = Field(
        default="",
        description="A second line under the first, for a bilingual caption. Whoever writes the captions "
                    "writes this too — nothing here translates anything",
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Who said it, joined across files (`V1`, `V2`), or the file's own label (`S1`) for footage "
                    "analyzed before voices were kept. The caption style decides "
                    "whether that reaches the screen as a name, a colour, or not at all",
    )
    words: List[CueWord] = Field(
        default_factory=list,
        description="Word timings inside this caption, in source seconds, for captions that light up word by "
                    "word. Empty is normal: a caption written by hand has none, and one lights up whole instead",
    )
    remembered_in: Optional[str] = Field(
        default=None,
        description="For a caption remembered against its file: the ID of the project it was written in",
    )
    local: bool = Field(
        default=False,
        description="Written for this project only — a title, or words put to fit this cut's order — so it is "
                    "never remembered against the file for other projects, and captioning this project again "
                    "keeps it over the file's own words",
    )

    @model_validator(mode="after")
    def validate_span(self):
        """Ensure the caption covers a positive stretch of its source file.

        Returns:
            The validated `SubtitleCue` instance.

        Raises:
            ValueError: If `source_end` is not after `source_start`.
        """
        if self.source_end <= self.source_start:
            raise ValueError("a subtitle must end after it starts")
        return self

class PlacedCue(BaseModel):
    """One caption worked out onto the cut as it stands now.

    The same stored caption can be placed more than once — a clip split in two
    shows its line on both sides of the cut — so this carries the `cue_id` it
    came from rather than being one itself. That is also what `edit_subtitle`
    takes: correcting a word corrects every place it appears.
    """

    cue_id: str = Field(..., description="The stored caption this came from")
    start: Decimal = Field(..., ge=0, description="When it appears (seconds on the timeline)")
    end: Decimal = Field(..., ge=0, description="When it disappears (seconds on the timeline)")
    text: str = Field(..., description="The caption text")
    secondary: str = Field(default="", description="The second line, for a bilingual caption")
    speaker: Optional[str] = Field(default=None, description="Who said it, as the speaker split labelled them")
    words: List[CueWord] = Field(
        default_factory=list, description="Word timings, in timeline seconds; empty when the caption has none",
    )

class VoiceCleanup(BaseModel):
    """Repairs applied to a recorded voice, each one asked for on purpose.

    Off unless somebody says otherwise, and never guessed at from a
    measurement: every one of these throws part of the recording away, and
    which part is worth losing depends on what the recording is for. A room
    tone somebody wanted is hiss to a denoiser.

    They run in this order, which is the order they undo each other least in:
    the rumble goes first so the denoiser is not busy modelling it, and the
    sibilance last, on what is left.
    """

    rumble: bool = Field(
        default=False,
        description="Take out the low roar under a recording — traffic, air conditioning, a hand on the "
                    "camera. Costs nothing on a voice, which has nothing down there",
    )
    hiss: bool = Field(
        default=False,
        description="Take out the steady background hiss a phone or a clip-on microphone leaves. Overdone it "
                    "makes a voice sound like it is underwater, so this is a light pass rather than a deep one",
    )
    sibilance: bool = Field(
        default=False,
        description="Soften harsh S sounds, which a microphone held close picks up. Leave it off unless they "
                    "are actually harsh: it takes the edge off the whole voice with them",
    )

    @property
    def is_nothing(self) -> bool:
        """Whether this asks for no repair at all.

        Returns:
            True when every repair is off, which is what a clip with nothing
            wrong with it wants.
        """
        return not (self.rumble or self.hiss or self.sibilance)

class SpeakerMark(str, Enum):
    """How a caption says who is talking."""

    OFF = "off"
    NAME = "name"
    COLOUR = "colour"
    BOTH = "both"

class CaptionStyle(BaseModel):
    """How burned-in captions are drawn, as fractions of the frame.

    Fractions rather than pixels, for the same reason a layout box is: the
    same style has to work at 1080 and at 4K, and at whatever shape the
    project is. Every number here is a platform convention rather than a
    measurement of anything — where a phone puts its own buttons over the
    picture, and how big text has to be to read on one. They are collected in
    `CAPTION_PRESETS` so that changing one changes it everywhere.
    """

    font: Optional[str] = Field(
        default=None, description="Font family to ask for; null uses the server's configured default",
    )
    size_fraction: float = Field(
        default=1 / 16, gt=0, le=0.5, description="Text size as a fraction of the frame's shorter side",
    )
    bottom_fraction: Optional[float] = Field(
        default=None,
        ge=0,
        lt=1,
        description="How far above the bottom the text sits, as a fraction of the height. Null keeps clear of "
                    "the controls a phone draws over a vertical video, and sits lower on a landscape one",
    )
    side_fraction: float = Field(
        default=0.08, ge=0, lt=0.5, description="Margin at each side, as a fraction of the width",
    )
    outline_fraction: float = Field(
        default=1 / 16, ge=0, le=0.5, description="Outline thickness as a fraction of the text size",
    )
    karaoke: bool = Field(
        default=False,
        description="Light each word as it is said, rather than showing the whole caption at once. Needs the "
                    "word timings `generate_subtitles` puts on a caption; one without them lights up whole",
    )
    speaker_mark: SpeakerMark = Field(
        default=SpeakerMark.OFF,
        description="How a caption says who is talking: `name` puts it in front of the line, `colour` gives "
                    "each speaker their own, `both` does both, `off` says nothing",
    )
    speaker_names: Dict[str, str] = Field(
        default_factory=dict,
        description="Real names for the speaker labels, such as {'V1': '阿明'}. A label with no name "
                    "keeps the label, because a caption reading 'S2' is still better than one crediting the "
                    "wrong person",
    )
    secondary_scale: float = Field(
        default=0.75, gt=0, le=1, description="How big the second line of a bilingual caption is, against the first",
    )
    single_line: bool = Field(
        default=True,
        description="Never stack a caption into several lines: one too long for a line is shown as several "
                    "lines one after another, split between words and timed to when they are said. Off, a "
                    "long caption wraps upwards into the picture. A bilingual caption keeps its two lines",
    )
    min_seconds: float = Field(
        default=0.7,
        ge=0,
        le=5,
        description="Shortest time a caption stays on screen. A shorter one is held on while its shot has "
                    "room before the next caption; one that cannot be, such as the last words of a video, is "
                    "found by the check before a render. 0.7 reads two characters at 0.35s each; change it "
                    "when the user asks",
    )

# Provisional (roadmap §13): `CaptionStyle.min_seconds` above — how long a caption has to stay
# on screen to be read.

# Where each platform's own furniture sits over the picture, how large text has to be to
# read on a phone held at arm's length, and how heavy an outline it takes to stay legible
# there. A phone is watched over whatever the footage is doing, so the vertical presets
# carry more outline than the one meant for a screen. Conventions, not measurements: they
# move when the apps move, which is why they are in one place.
CAPTION_PRESETS: Dict[str, CaptionStyle] = {
    # What the server did before there were presets: clear of a phone's controls on a
    # vertical video, lower on a landscape one.
    "plain": CaptionStyle(),
    # Landscape, watched on a screen rather than held: smaller text, and only the player's
    # own control bar to stay above.
    "youtube": CaptionStyle(size_fraction=1 / 20, bottom_fraction=0.10, side_fraction=0.10,
                            outline_fraction=1 / 18),
    # The caption, the handle and the audio credit take the bottom of a Reel. Captions sit
    # just above them rather than clear of the whole stack: one line placed low is what
    # captioned short video looks like, and a block halfway up the frame covers the subject.
    "reels": CaptionStyle(size_fraction=1 / 16, bottom_fraction=0.16, side_fraction=0.08,
                          outline_fraction=1 / 10, karaoke=True),
    # Same again, and the buttons up the right-hand side push the safe area in further.
    "tiktok": CaptionStyle(size_fraction=1 / 15, bottom_fraction=0.18, side_fraction=0.10,
                           outline_fraction=1 / 9, karaoke=True),
    # A vertical video minutes long — a podcast episode, a vlog. The Reels safe area, without
    # words lighting up: over a few minutes that reads as a gimmick, not as emphasis.
    "vertical": CaptionStyle(size_fraction=1 / 16, bottom_fraction=0.16, side_fraction=0.08,
                             outline_fraction=1 / 10),
}

class ClipFit(BaseModel):
    """How a shot of another shape sits in the frame: cropped to fill it, or shown whole.

    A landscape shot in a vertical video, or a vertical one in a landscape
    video. Left unset, the shot decides for itself: one with a face in it is
    cropped around the face; one without — scenery, hands at work, a screen —
    is shown whole over a blurred, enlarged copy of itself.
    """

    mode: Literal["fill", "whole"] = Field(
        ...,
        description="`fill` crops the shot to fill the frame; `whole` shows all of it, small, over a blurred "
                    "copy of itself",
    )
    center_x: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="For `fill`: where the crop is centred across the picture, 0 the left edge and 1 the right. "
                    "Set, the crop stays there instead of following a face",
    )
    center_y: Optional[float] = Field(
        default=None, ge=0, le=1, description="For `fill`: the same down the picture, 0 the top",
    )
    zoom: float = Field(default=1.0, ge=1, le=4, description="For `fill`: how much closer than filling the frame")

# How a photo moves while it is on screen. A photo held perfectly still reads as the video
# having stopped; a slow push in reads as somebody looking at it.
Motion = Literal["push", "pull", "pan_left", "pan_right", "none"]

class ClipLayout(BaseModel):
    """Where a clip is drawn in the frame, as fractions of the output size.

    Fractions rather than pixels, so the same layout works whatever the
    project's resolution is. A clip with no layout fills the frame.
    """

    x: float = Field(default=0.0, ge=0, le=1, description="Left edge as a fraction of the output width")
    y: float = Field(default=0.0, ge=0, le=1, description="Top edge as a fraction of the output height")
    # No upper bound here: a box that is too big is caught below, with a message that says why.
    width: float = Field(default=1.0, gt=0, description="Width as a fraction of the output width")
    height: float = Field(default=1.0, gt=0, description="Height as a fraction of the output height")

    @model_validator(mode="after")
    def validate_box(self):
        """Ensure the box stays inside the frame.

        Returns:
            The validated `ClipLayout` instance.

        Raises:
            ValueError: If the box runs past the right or bottom edge.
        """
        if self.x + self.width > 1.0001 or self.y + self.height > 1.0001:
            raise ValueError(
                f"a clip at x {self.x}, y {self.y} sized {self.width} x {self.height} runs outside the frame; "
                "x + width and y + height must each stay within 1.0"
            )
        return self

class ColorAdjust(BaseModel):
    """Picture adjustments applied to a clip, in the language of the controls people expect."""

    brightness: float = Field(default=0.0, ge=-1, le=1, description="Lighter above 0, darker below; 0 leaves it alone")
    contrast: float = Field(default=1.0, ge=0, le=4, description="1.0 leaves it alone; above 1 deepens blacks and brightens highlights")
    saturation: float = Field(default=1.0, ge=0, le=3, description="1.0 leaves it alone; 0 is black and white; above 1 is more colourful")
    temperature: Optional[int] = Field(
        default=None,
        ge=1000,
        le=40000,
        description="White balance in kelvin: below about 6500 looks warmer, above looks cooler; null leaves it alone",
    )

    @property
    def is_neutral(self) -> bool:
        """Whether these settings change nothing.

        Returns:
            True when every adjustment is at its resting value.
        """
        return (
            self.brightness == 0.0
            and self.contrast == 1.0
            and self.saturation == 1.0
            and self.temperature is None
        )

# Said the same way on a clip and on the operations that set it.
KEEP_LEVEL = (
    "Play this clip's sound at the level it was recorded, relative to everything else, instead of bringing its "
    "talking to the level of the rest — for a whisper or a shout that is meant to be one. Its volume still applies"
)

class WipeDirection(str, Enum):
    """Which way a wipe travels across the frame."""

    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

# A colour a dip can pass through: six hex digits, or the two names worth having by name.
# Anything else has to be given as hex on purpose — FFmpeg knows a long list of colour
# names, and a typo in one of them would only ever surface as a render that failed.
COLOUR = r"^(#[0-9a-fA-F]{6}|black|white)$"

class _Transition(BaseModel):
    """What every transition has: how long it runs.

    The length is the whole of it, and it ends on the cut rather than
    straddling it. That is what keeps every clip where it is: the transition
    eats the incoming clip's run-up, which comes from outside its source
    range, so the sequence never changes length and nothing after it has to
    be shifted. A transition centred on the cut would move both sides, which
    is a different design and not this one.
    """

    seconds: Decimal = Field(..., gt=0, description="How long the transition runs, ending on the cut")

    @property
    def run_up(self) -> Decimal:
        """How much picture the incoming clip has to supply from before its in point.

        Returns:
            The seconds it reaches back, which for a single mix is the whole
            transition.
        """
        return self.seconds

    @property
    def least_frames(self) -> int:
        """How many frames this transition needs to exist at all.

        Returns:
            One, for a transition that is a single mix.
        """
        return 1

class Dissolve(_Transition):
    """The incoming picture mixes up through the outgoing one."""

    kind: Literal["dissolve"]

class Wipe(_Transition):
    """The incoming picture travels across the frame over the outgoing one."""

    kind: Literal["wipe"]
    direction: WipeDirection = Field(
        default=WipeDirection.LEFT, description="Which way it travels; `left` moves the edge towards the left",
    )

class Dip(_Transition):
    """The picture goes to a colour and comes back out of it on the other side.

    Two mixes rather than one: the outgoing picture fades to the colour over
    the first half, the incoming rises out of it over the second. So a dip
    only reaches back into the incoming clip for half its length, where a
    dissolve or a wipe reaches back for all of it.
    """

    kind: Literal["dip"]
    through: str = Field(
        default="black",
        pattern=COLOUR,
        description="The colour to pass through: `black`, `white`, or a hex colour such as `#1b2a4a`",
    )

    @property
    def run_up(self) -> Decimal:
        """How much picture the incoming clip has to supply from before its in point.

        Returns:
            Half the transition: the colour covers the first mix, and only the
            second one reaches into this clip.
        """
        return self.seconds / 2

    @property
    def least_frames(self) -> int:
        """How many frames this transition needs to exist at all.

        Returns:
            Two: one to go into the colour and one to come out of it.
        """
        return 2

Transition = Annotated[Union[Dissolve, Wipe, Dip], Field(discriminator="kind")]

# Times are kept to the millisecond, while a clip played at another speed ends wherever its length
# divides out to and a file ends wherever its last frame does. Within this, two clips touch rather
# than overlap, and a clip ends at the end of its file rather than past it. The one tolerance every
# part of the server uses for it.
TOUCHING_SECONDS = Decimal("0.001")
# What the renderer can stretch sound to without stacking filters on top of each other
# past the point where anybody would want to listen to the result.
MIN_SPEED = 0.25
MAX_SPEED = 8.0

# Provisional (roadmap §13): how long a title card stays up when nothing says otherwise.
TITLE_CARD_SECONDS = Decimal("2.5")
# The shortest and longest a title card can be: under half a second nobody reads it, and a
# card that stays ten seconds is a slide, not a title.
TITLE_CARD_SHORTEST = Decimal("0.5")
TITLE_CARD_LONGEST = Decimal(10)

CardBackground = Literal["blur", "picture", "colour"]

def card_background(asked: Optional[str], photo: Optional[str]) -> str:
    """The background a new title card gets: the one asked for, else its photo as it is, else the shot blurred."""
    return asked or ("picture" if photo else "blur")

class TitleCard(BaseModel):
    """A title on a picture of its own, taking up time in the cut.

    A card has no file of its own. Its clip reads one frame of the file behind
    it — the shot it leads into, unless it was given a photo — and the card
    says how that frame is shown and what is written over it.
    """

    title: str = Field(..., min_length=1, max_length=60, description="The main line, large and centred")
    subtitle: Optional[str] = Field(default=None, max_length=80, description="A smaller line under it, if any")
    background: CardBackground = Field(
        default="blur",
        description="`blur`: the frame behind it blurred and darkened; `picture`: that frame as it is, darkened a "
                    "little so the words read; `colour`: a plain colour",
    )
    colour: str = Field(default="#111114", pattern=COLOUR, description="The colour for a `colour` background")
    at: Decimal = Field(default=Decimal(0), ge=0, description="Which second of the file behind it is shown")
    beat_id: Optional[str] = Field(
        default=None, description="The part of the plan it opens, when the plan put it there",
    )

TextStyle = Literal["name", "place", "headline", "free"]

class TextLayer(BaseModel):
    """Words over a stretch of footage, pinned to what is on screen rather than to a time in the cut.

    The stretch is seconds of the clip's own file, so the words go wherever
    that footage goes: moved, trimmed, sped up or cut out, they follow it,
    and they show only for as much of the stretch as the cut still plays.
    """

    id: str
    text: str = Field(..., min_length=1, max_length=60)
    second: Optional[str] = Field(
        default=None, max_length=60,
        description="A smaller second line: who a named person is, or what a place is",
    )
    style: TextStyle = Field(
        default="name",
        description="`name`: a name bar low on the left; `place`: a place, high on the left; `headline`: large in "
                    "the middle; `free`: wherever `x` and `y` put it",
    )
    start: Decimal = Field(..., ge=0, description="First second of the clip's file it is shown over")
    end: Decimal = Field(..., gt=0, description="Second of the clip's file it stops at")
    x: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Where it was moved to across the frame, 0 to 1: the centre, or the left edge of a name or a "
                    "place; null where its style puts it",
    )
    y: Optional[float] = Field(default=None, ge=0, le=1,
                               description="Where it was moved to down the frame, 0 to 1: its centre; null likewise")
    by_plan: bool = Field(
        default=False,
        description="Put there by the plan and rebuilt with it; false once anybody changes it, and then kept",
    )
    hidden: bool = Field(
        default=False,
        description="Words the plan put there that were taken off by hand: kept out of sight, so the next "
                    "compile does not put them back",
    )

class Clip(BaseModel):
    """A segment of a source asset placed on a track.

    A clip that a plan compiled carries where it came from, and whether
    anybody has touched it since. Those two together are what lets a plan be
    compiled a second time without throwing away the changes made by hand in
    between: the compiler rebuilds what it made, and leaves alone what it did
    not.
    """

    id: str
    asset_id: str
    source_range: TimeRange = Field(..., description="Source time range of the clip")
    timeline_in: Decimal = Field(..., ge=0, description="Start time on the timeline")
    from_plan_id: Optional[str] = Field(default=None, description="Plan that compiled this clip, if one did")
    from_clip_ids: List[str] = Field(
        default_factory=list,
        description="Semantic clips this was compiled from; what the compiler matches on when it runs again",
    )
    pinned: bool = Field(
        default=False,
        description="Adjusted by hand, so compiling the plan again keeps it as it is rather than rebuilding it",
    )
    speed: float = Field(
        default=1.0,
        ge=MIN_SPEED,
        le=MAX_SPEED,
        description="How fast the clip plays. 2.0 runs it at double speed and half the length, 0.5 at half speed "
                    "and twice the length. The sound is stretched to match",
    )
    preserve_pitch: bool = Field(
        default=True,
        description="Keep voices at the pitch they were recorded at when the clip is sped up or slowed down. "
                    "False resamples instead, so the sound rises and falls with the speed the way tape does — "
                    "which is a mistake on someone talking and the whole point of a comedy speed-up",
    )
    volume: float = Field(default=1.0, ge=0, description="Audio gain; 1.0 keeps the original level, 0.5 halves it")
    audio_lead: Decimal = Field(
        default=Decimal(0),
        ge=0,
        description="Seconds this clip's sound starts before its picture does, taken from before its source range. "
                    "This is the J cut: the next scene is heard under the end of the one still on screen",
    )
    audio_lag: Decimal = Field(
        default=Decimal(0),
        ge=0,
        description="Seconds this clip's sound runs on after its picture has cut away, taken from after its source "
                    "range. This is the L cut: a line finishes over the shot that follows it",
    )
    audio_fade_in: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-in length at the clip start (seconds)")
    audio_fade_out: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-out length at the clip end (seconds)")
    cleanup: VoiceCleanup = Field(
        default_factory=VoiceCleanup,
        description="Repairs applied to the voice on this clip; nothing by default",
    )
    keep_level: bool = Field(default=False, description=KEEP_LEVEL)
    transition_in: Optional[Transition] = Field(
        default=None,
        description="How this clip's picture arrives over the end of the clip before it: a dissolve, a wipe, or a "
                    "dip through a colour. The transition ends where this clip starts and the extra picture comes "
                    "from before its source range, so the cut stays where it is and the sequence keeps its "
                    "length. Null is a straight cut",
    )
    video_fade_in: Decimal = Field(default=Decimal(0), ge=0, description="Fade in from black at the clip start (seconds)")
    video_fade_out: Decimal = Field(default=Decimal(0), ge=0, description="Fade out to black at the clip end (seconds)")
    color: Optional[ColorAdjust] = Field(default=None, description="Picture adjustments; null leaves the picture as shot")
    layout: Optional[ClipLayout] = Field(
        default=None,
        description="Where the clip is drawn, for clips on a video track above the first; null fills the frame",
    )
    fit: Optional[ClipFit] = Field(
        default=None,
        description="How a shot of another shape than the frame sits in it; null lets the shot decide",
    )
    motion: Optional[Motion] = Field(
        default=None,
        description="For a photo: `push` closer, `pull` back, `pan_left`, `pan_right`, or `none` to hold it "
                    "still; null is a slow push. Ignored for video",
    )
    card: Optional[TitleCard] = Field(
        default=None,
        description="Set on a title card: what it says and how its background looks. A card is silent, and its "
                    "source range is only its length: it shows one frame of its file, at `card.at`",
    )
    texts: List[TextLayer] = Field(
        default_factory=list,
        description="Words over this clip's footage, each pinned to seconds of its file",
    )

    @property
    def shown_texts(self) -> List[TextLayer]:
        """The words that are on screen: all but the plan's ones taken off by hand."""
        return [text for text in self.texts if not text.hidden]

    @property
    def run_up(self) -> Decimal:
        """How much picture this clip needs from before its in point.

        A transition is fed from there, so that it can run under the outgoing
        clip while this one is still to come. A dissolve and a wipe are one
        mix and reach back for their whole length; a dip is two mixes with a
        colour between them, and only the second of those touches this clip,
        so it reaches back for half.

        Returns:
            The seconds of run-up, 0 for a straight cut.
        """
        return Decimal(0) if self.transition_in is None else self.transition_in.run_up

    @property
    def video_source_start(self) -> Decimal:
        """Where in the source this clip's picture begins.

        Plain arithmetic, for the same reason `audio_source_start` is: a
        transition reaching past the start of the file has to be reportable.

        Returns:
            The second in the source, which is negative when the transition
            asks for more picture than the file has before the clip.
        """
        return self.source_range.start - self.run_up * Decimal(str(self.speed))

    @property
    def audio_source_start(self) -> Decimal:
        """Where in the source this clip's sound begins.

        Plain arithmetic rather than a range, because a lead that reaches back
        past the start of the file has to be reportable. Building a `TimeRange`
        here would raise on the way to the sentence explaining what is wrong.

        Returns:
            The second in the source, which is negative when the lead asks for
            more sound than the file has before the picture.
        """
        return self.source_range.start - self.audio_lead * Decimal(str(self.speed))

    @property
    def audio_source_end(self) -> Decimal:
        """Where in the source this clip's sound ends.

        Returns:
            The second in the source, which may be past the end of the file
            when the lag asks for more than it has.
        """
        return self.source_range.end + self.audio_lag * Decimal(str(self.speed))

    @property
    def audio_source_range(self) -> TimeRange:
        """Source range this clip's sound is taken from.

        The picture's range, opened out by whatever the lead and the lag ask
        for. Those are timeline seconds, so playback speed turns them into
        source seconds the same way it does for the clip's own length.

        Returns:
            The range. Equal to `source_range` when neither is set.

        Raises:
            ValueError: If it falls outside the file. `validate_project` says
                so in a sentence first; this is the backstop.
        """
        return TimeRange(start=self.audio_source_start, end=self.audio_source_end)

    @property
    def audio_timeline_in(self) -> Decimal:
        """Time on the timeline at which this clip's sound comes in.

        Returns:
            `timeline_in`, less the lead.
        """
        return self.timeline_in - self.audio_lead

    @property
    def audio_timeline_out(self) -> Decimal:
        """Time on the timeline at which this clip's sound stops.

        Returns:
            `timeline_out`, plus the lag.
        """
        return self.timeline_out + self.audio_lag

    @property
    def audio_timeline_duration(self) -> Decimal:
        """How long this clip's sound runs on the timeline.

        Returns:
            The picture's length plus the lead and the lag. This, rather than
            `timeline_duration`, is what the clip's fades have to fit inside.
        """
        return self.timeline_duration + self.audio_lead + self.audio_lag

    @property
    def timeline_duration(self) -> Decimal:
        """Length of the clip on the timeline after applying playback speed.

        Returns:
            The source range duration divided by `speed`, in seconds.
        """
        return self.source_range.duration / Decimal(str(self.speed))

    @property
    def timeline_out(self) -> Decimal:
        """Time on the timeline at which the clip ends.

        Returns:
            `timeline_in` plus `timeline_duration`, in seconds.
        """
        return self.timeline_in + self.timeline_duration

class Marker(BaseModel):
    """A named point on the timeline: where one part of the video begins.

    A cut has a shape — an opening, a middle, an ending — and until now that
    shape lived only in the plan and was lost the moment the plan was
    compiled. A marker is that shape written onto the timeline, where the
    things that follow it can read it: where the music changes, how dense the
    B-roll should be, which caption style applies.

    Markers name a moment, not a stretch. Where one part ends is where the
    next begins, and the last one runs to the end of the cut.
    """

    id: str
    name: str = Field(..., min_length=1, description="What this part is, such as 開場 or 結尾")
    timeline_in: Decimal = Field(..., ge=0, description="Where it begins on the timeline (seconds)")
    from_beat_id: Optional[str] = Field(
        default=None,
        description="Beat of the plan this came from. Markers with one are the compiler's and are rebuilt "
                    "every time the plan is compiled; markers without one were put there by hand and are kept",
    )

class TrackType(str, Enum):
    """Kind of media a track carries."""

    VIDEO = "video"
    AUDIO = "audio"

class Track(BaseModel):
    """An ordered sequence of clips of a single media type."""

    id: str
    track_type: TrackType
    clips: List[Clip] = Field(default_factory=list)
    duck_under_speech: bool = Field(
        default=False,
        description="Lower this audio track automatically while the video's own sound is loud; audio tracks only",
    )
    voice: Optional[bool] = Field(
        default=None,
        description="A voice recorded apart from the picture, such as a narration: the footage's own sound "
                    "and any music drop while it speaks. Null decides from the recording, which counts as a "
                    "voice when it was transcribed and sentences cover at least half of it, unless the track "
                    "ducks under speech; audio tracks only",
    )

class BranchPoint(BaseModel):
    """Where a branch came from: which project, at which of its versions."""

    project_id: str
    commit: str = Field(..., description="The version of that project it started as, a commit in the history")

class Project(BaseModel):
    """An editing project: output format settings plus its tracks."""

    id: str
    name: Optional[str] = Field(default=None, description="What the editor calls this project, such as 'EP1 台北'")
    branched_from: Optional[BranchPoint] = Field(
        default=None, description="For a branch, the project and version it was started from; null for a first cut",
    )
    version: int = Field(default=1, description="Project version")
    fps_num: int = Field(default=30, gt=0, description="Frame rate numerator (e.g. 30000 for 29.97 fps)")
    fps_den: int = Field(default=1, gt=0, description="Frame rate denominator (e.g. 1001 for 29.97 fps)")
    width: int = Field(default=1920, gt=0, multiple_of=2, description="Output width (pixels); must be even")
    height: int = Field(default=1080, gt=0, multiple_of=2, description="Output height (pixels); must be even")
    tracks: List[Track] = Field(default_factory=list)
    markers: List[Marker] = Field(
        default_factory=list,
        description="Where each part of the video begins, in timeline order",
    )
    subtitles: List[SubtitleCue] = Field(
        default_factory=list,
        description="Captions burned into the picture when render_project is called with burn_subtitles",
    )
    deleted_subtitles: List[SubtitleCue] = Field(
        default_factory=list,
        description="Captions deleted by hand, as they were, so captioning again leaves the same words out",
    )
    caption_style: CaptionStyle = Field(
        default_factory=CaptionStyle,
        description="How those captions are drawn: where they sit, how big they are, whether they light up "
                    "word by word, and whether they say who is talking",
    )

    @field_validator("subtitles", mode="before")
    @classmethod
    def drop_captions_from_before_they_followed_the_footage(cls, value):
        """Leave behind captions stored against the timeline instead of the footage.

        Captions used to record where they landed in the cut. Those carry no
        `asset_id`, so they cannot be placed, and refusing them would be worse
        than dropping them: the project would not open at all, and a project
        that will not open cannot have its captions made again — which is what
        the roadmap asks for instead of a migration.

        Args:
            value: Whatever was stored under `subtitles`.

        Returns:
            The captions that can still be placed. A project stored before the
            change opens with none, and `generate_subtitles` makes them again.
        """
        if not isinstance(value, list):
            return value
        return [cue for cue in value if not isinstance(cue, dict) or "asset_id" in cue]

    @property
    def video_tracks(self) -> List[Track]:
        """The video tracks in the order they are drawn.

        Returns:
            The video tracks, the first being the base and the rest drawn on
            top of it.
        """
        return [track for track in self.tracks if track.track_type == TrackType.VIDEO]

    @property
    def base_video_track(self) -> Optional[Track]:
        """The video track that carries the sequence.

        Returns:
            The first video track, or None when the project has none.
        """
        return next(iter(self.video_tracks), None)

    @computed_field(description="Length of the edited video (seconds): the end of the last clip on the bottom video track")
    @property
    def duration(self) -> Decimal:
        """Length of the edited video.

        The first video track is the base: it decides how long the output is.
        Audio tracks and any video track above it do not extend the output;
        anything past this point is cut.

        Returns:
            The end of the last clip on the base video track in seconds, or
            zero if the project has no video clips.
        """
        base = self.base_video_track
        return max((clip.timeline_out for clip in base.clips), default=Decimal(0)) if base else Decimal(0)
