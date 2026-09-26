from decimal import Decimal
from enum import Enum
from typing import Annotated, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from app.models.media import Asset


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
}

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

# What the renderer can stretch sound to without stacking filters on top of each other
# past the point where anybody would want to listen to the result.
MIN_SPEED = 0.25
MAX_SPEED = 8.0

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

class Project(BaseModel):
    """An editing project: output format settings plus its tracks."""

    id: str
    name: Optional[str] = Field(default=None, description="What the editor calls this project, such as 'EP1 台北'")
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

class AddTrackOp(BaseModel):
    """Edit operation that appends an empty track to the project."""

    action: Literal["add_track"] = "add_track"
    track_id: str = Field(..., description="ID of the new track; must be unique within the project")
    track_type: TrackType
    duck_under_speech: bool = Field(
        default=False,
        description="Lower this track automatically while the video's own sound is loud; audio tracks only",
    )

class _NewClipSpec(BaseModel):
    """Fields shared by operations that create a clip."""

    track_id: str
    clip_id: str = Field(..., description="ID of the new clip; must be unique within the track")
    asset_id: str = Field(..., description="ID returned by import_asset")
    source_range: TimeRange = Field(..., description="Segment of the asset to use")
    speed: float = Field(default=1.0, gt=0)
    preserve_pitch: bool = Field(default=True, description="Keep voices at their recorded pitch when the speed changes")
    volume: float = Field(default=1.0, ge=0, description="Audio gain; 1.0 keeps the original level, 0.5 halves it")
    audio_fade_in: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-in length at the clip start (seconds)")
    audio_fade_out: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-out length at the clip end (seconds)")
    # These three are on the spec because `compile_plan` rebuilds a pinned clip by
    # inserting it again. Leaving them off silently dropped a hand-made J cut or
    # transition on the next compile, which is the one thing pinning promises not to do.
    audio_lead: Decimal = Field(default=Decimal(0), ge=0, description="Seconds the sound starts before the picture (J cut)")
    audio_lag: Decimal = Field(default=Decimal(0), ge=0, description="Seconds the sound runs on after the picture (L cut)")
    cleanup: VoiceCleanup = Field(default_factory=VoiceCleanup, description="Repairs applied to the voice")
    transition_in: Optional[Transition] = Field(
        default=None, description="How the picture arrives over the clip before it; null is a straight cut",
    )
    video_fade_in: Decimal = Field(default=Decimal(0), ge=0, description="Fade in from black at the clip start (seconds)")
    video_fade_out: Decimal = Field(default=Decimal(0), ge=0, description="Fade out to black at the clip end (seconds)")
    color: Optional[ColorAdjust] = Field(default=None, description="Picture adjustments; null leaves the picture as shot")
    layout: Optional[ClipLayout] = Field(
        default=None,
        description="Where the clip is drawn, for clips on a video track above the first; null fills the frame",
    )

class AddClipOp(_NewClipSpec):
    """Edit operation that places a clip at an absolute timeline position without moving other clips."""

    action: Literal["add_clip"] = "add_clip"
    timeline_in: Decimal = Field(..., ge=0, description="Start time on the timeline (seconds)")

class InsertClipOp(_NewClipSpec):
    """Edit operation that inserts a clip into the sequence, shifting later clips to make room."""

    action: Literal["insert_clip"] = "insert_clip"
    before_clip_id: Optional[str] = Field(
        default=None,
        description="Insert at the start of this clip; omit to append after the last clip on the track",
    )

class TrimClipOp(BaseModel):
    """Edit operation that changes the source range of a clip."""

    action: Literal["trim_clip"] = "trim_clip"
    track_id: str
    clip_id: str
    new_source_range: TimeRange = Field(..., description="New segment of the asset to use")
    ripple: bool = Field(
        default=True,
        description="Shift later clips by the change in duration, so no gap or overlap is created",
    )

class DeleteOp(BaseModel):
    """Edit operation that removes a clip from a track."""

    action: Literal["delete_clip"] = "delete_clip"
    track_id: str
    clip_id: str
    ripple: bool = Field(
        default=True,
        description="Shift later clips earlier to close the space the clip occupied; false leaves a gap",
    )

class MoveClipOp(BaseModel):
    """Edit operation that moves a clip to an absolute timeline position without moving other clips."""

    action: Literal["move_clip"] = "move_clip"
    track_id: str
    clip_id: str
    new_timeline_in: Decimal = Field(..., ge=0, description="New start time on the timeline (seconds)")

class SplitClipOp(BaseModel):
    """Edit operation that cuts one clip into two at a point inside it."""

    action: Literal["split_clip"] = "split_clip"
    track_id: str
    clip_id: str = Field(..., description="Clip to cut; it keeps this ID and becomes the first half")
    new_clip_id: str = Field(..., description="ID for the second half; must be unique within the track")
    at: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Cut point as a time on the timeline (seconds); give this or at_source, not both",
    )
    at_source: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Cut point as a time in the source file (seconds), as transcripts report it",
    )

    @model_validator(mode="after")
    def validate_cut_point(self):
        """Ensure exactly one way of naming the cut point was given.

        Returns:
            The validated `SplitClipOp` instance.

        Raises:
            ValueError: If both `at` and `at_source` are set, or neither is.
        """
        if (self.at is None) == (self.at_source is None):
            raise ValueError("give exactly one of at (timeline seconds) or at_source (source seconds)")
        return self

class ReorderClipOp(BaseModel):
    """Edit operation that moves a clip to another place in the sequence, closing the space it leaves and opening space where it lands."""

    action: Literal["reorder_clip"] = "reorder_clip"
    track_id: str
    clip_id: str
    before_clip_id: Optional[str] = Field(
        default=None,
        description="Move the clip to where this clip starts; omit to move it to the end of the track",
    )

class FitTrackOp(BaseModel):
    """Edit operation that makes an audio track cover the video exactly, with its fades in the right place.

    Written for the usual case of one song on the track: when the video runs
    past everything on it, the longest clip is the one repeated, on the
    assumption that it is the untrimmed piece of that song.
    """

    action: Literal["fit_track"] = "fit_track"
    track_id: str
    loop: bool = Field(
        default=True,
        description="Loop back to the start once the song runs out; playing more of a song that still has material left happens either way",
    )
    fade_in: Optional[Decimal] = Field(default=None, ge=0, description="Fade-in to put on the first clip (seconds)")
    fade_out: Optional[Decimal] = Field(default=None, ge=0, description="Fade-out to put on the new last clip (seconds)")

class EditSubtitleOp(BaseModel):
    """Edit operation that changes or removes one caption, leaving the rest untouched."""

    action: Literal["edit_subtitle"] = "edit_subtitle"
    cue_id: str = Field(
        ...,
        description="ID of the caption: `id` as generate_subtitles reports it, `cue_id` as get_subtitles does",
    )
    text: Optional[str] = Field(default=None, description="New text for this caption")
    secondary: Optional[str] = Field(
        default=None, description="New second line for a bilingual caption; an empty string takes it away",
    )
    speaker: Optional[str] = Field(default=None, description="Who said it, overriding what the speaker split decided")
    source_start: Optional[Decimal] = Field(
        default=None, ge=0, description="New start in the source file (seconds), to catch a line that comes up early",
    )
    source_end: Optional[Decimal] = Field(
        default=None, ge=0, description="New end in the source file (seconds), to hold a line on screen longer",
    )
    delete: bool = Field(default=False, description="Remove this caption entirely")

    @model_validator(mode="after")
    def validate_text(self):
        """Ensure a caption is not blanked out instead of removed.

        Returns:
            The validated `EditSubtitleOp` instance.

        Raises:
            ValueError: If `text` is given but holds nothing to show.
        """
        if self.text is not None and not self.text.strip():
            raise ValueError("a caption needs text; set delete to remove it instead")
        return self

class SetSubtitlesOp(BaseModel):
    """Edit operation that replaces the project's captions."""

    action: Literal["set_subtitles"] = "set_subtitles"
    cues: List[SubtitleCue] = Field(default_factory=list, description="The complete new set of captions; an empty list removes them")

class SetCaptionStyleOp(BaseModel):
    """Edit operation that changes how the captions are drawn.

    A preset is a starting point, not a lock: anything given alongside it wins,
    so `{"preset": "tiktok", "karaoke": false}` is that platform's safe area
    without the word-by-word lighting.
    """

    action: Literal["set_caption_style"] = "set_caption_style"
    preset: Optional[str] = Field(
        default=None,
        description="Where the video is going: `youtube`, `reels`, `tiktok`, or `plain` for no platform's "
                    "furniture at all. Each one sets the safe area, the text size and the outline",
    )
    style: Optional[CaptionStyle] = Field(
        default=None, description="Settings to apply on top of the preset, or on their own",
    )

    @model_validator(mode="after")
    def validate_preset(self):
        """Ensure the operation names a preset that exists and asks for something.

        Returns:
            The validated `SetCaptionStyleOp` instance.

        Raises:
            ValueError: If the preset is unknown, or neither field is given.
        """
        if self.preset is not None and self.preset not in CAPTION_PRESETS:
            raise ValueError(
                f"there is no caption preset called {self.preset!r}; the presets are "
                f"{', '.join(sorted(CAPTION_PRESETS))}"
            )
        if self.preset is None and self.style is None:
            raise ValueError("give a preset, a style, or both")
        return self

class SetClipLookOp(BaseModel):
    """Edit operation that changes a clip's picture: how it comes in, its colour, and where it sits.

    Omitted fields stay unchanged.
    """

    action: Literal["set_clip_look"] = "set_clip_look"
    track_id: str
    clip_id: str
    transition_in: Optional[Transition] = Field(
        default=None,
        description="How this clip's picture arrives over the end of the clip before it: a dissolve, a wipe, or a "
                    "dip through a colour. Use clear_transition to go back to a straight cut",
    )
    clear_transition: bool = Field(default=False, description="Make the clip a straight cut again")
    video_fade_in: Optional[Decimal] = Field(default=None, ge=0, description="Fade in from black at the clip start (seconds)")
    video_fade_out: Optional[Decimal] = Field(default=None, ge=0, description="Fade out to black at the clip end (seconds)")
    color: Optional[ColorAdjust] = Field(
        default=None,
        description="Picture adjustments to apply; only the ones named here change, the clip's other adjustments stay",
    )
    clear_color: bool = Field(default=False, description="Remove the clip's picture adjustments and leave it as shot")
    layout: Optional[ClipLayout] = Field(default=None, description="Where the clip is drawn in the frame")
    clear_layout: bool = Field(default=False, description="Make the clip fill the frame again")

    @model_validator(mode="after")
    def validate_intent(self):
        """Ensure the operation does not both set and clear the same thing.

        Returns:
            The validated `SetClipLookOp` instance.

        Raises:
            ValueError: If a value is given alongside the flag that clears it.
        """
        for value, clear, name, cleared in (
            (self.color, self.clear_color, "color", "clear_color"),
            (self.layout, self.clear_layout, "layout", "clear_layout"),
            (self.transition_in, self.clear_transition, "transition_in", "clear_transition"),
        ):
            if value is not None and clear:
                raise ValueError(f"give either {name} or {cleared}, not both")
        return self

class RenameProjectOp(BaseModel):
    """Edit operation that gives the project a name, or takes it away again."""

    action: Literal["rename_project"] = "rename_project"
    name: Optional[str] = Field(
        default=None, max_length=120, description="What to call the project; null goes back to no name",
    )

class SetTrackAudioOp(BaseModel):
    """Edit operation that changes a whole track's audio behaviour; omitted fields stay unchanged."""

    action: Literal["set_track_audio"] = "set_track_audio"
    track_id: str
    duck_under_speech: Optional[bool] = Field(
        default=None,
        description="Lower this track automatically while the video's own sound is loud; audio tracks only",
    )

class SetClipAudioOp(BaseModel):
    """Edit operation that changes a clip's audio level, fades, or how far its sound runs outside its picture.

    Omitted fields stay unchanged.
    """

    action: Literal["set_clip_audio"] = "set_clip_audio"
    track_id: str
    clip_id: str
    volume: Optional[float] = Field(default=None, ge=0, description="Audio gain; 1.0 keeps the original level, 0.5 halves it")
    audio_fade_in: Optional[Decimal] = Field(default=None, ge=0, description="Audio fade-in length at the clip start (seconds)")
    audio_fade_out: Optional[Decimal] = Field(default=None, ge=0, description="Audio fade-out length at the clip end (seconds)")
    audio_lead: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Seconds this clip's sound starts before its picture — a J cut. 0 puts them back together",
    )
    audio_lag: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Seconds this clip's sound runs on after its picture — an L cut. 0 puts them back together",
    )
    cleanup: Optional[VoiceCleanup] = Field(
        default=None,
        description="Repairs to apply to the voice on this clip. Only the ones named here change, so asking "
                    "for the hiss to go does not put the rumble back",
    )

class SetMarkersOp(BaseModel):
    """Edit operation that replaces the timeline's structure markers.

    The whole set at once, like captions: a marker means something only
    against the ones either side of it, so they are written together.
    """

    action: Literal["set_markers"] = "set_markers"
    markers: List[Marker] = Field(default_factory=list, description="The markers; an empty list clears them")

class SetClipSpeedOp(BaseModel):
    """Edit operation that changes how fast a clip plays.

    The clip's length on the timeline changes with it — twice the speed is
    half the length — so by default what follows moves up or back to keep the
    sequence tight, the same way retrimming does.
    """

    action: Literal["set_clip_speed"] = "set_clip_speed"
    track_id: str
    clip_id: str
    speed: float = Field(
        ...,
        ge=MIN_SPEED,
        le=MAX_SPEED,
        description="1.0 is as shot, 2.0 twice as fast and half as long, 0.5 half as fast and twice as long",
    )
    preserve_pitch: Optional[bool] = Field(
        default=None,
        description="True keeps voices at their recorded pitch; false lets the sound rise and fall with the speed. "
                    "Omit to leave the clip's current setting alone",
    )
    ripple: bool = Field(
        default=True,
        description="Shift later clips by the change in duration, so no gap or overlap is created",
    )

class SetClipPinnedOp(BaseModel):
    """Edit operation that protects a clip from being rebuilt, or gives it back to the plan.

    Adjusting a compiled clip by hand pins it on its own, so this is mostly
    how a clip is handed back: unpin it and the next compile rebuilds it from
    the plan like any other.
    """

    action: Literal["set_clip_pinned"] = "set_clip_pinned"
    track_id: str
    clip_id: str
    pinned: bool = Field(..., description="True to keep this clip as it is; false to let the plan rebuild it")

EditOperation = Annotated[
    Union[
        AddTrackOp, AddClipOp, InsertClipOp, TrimClipOp, DeleteOp, MoveClipOp,
        SplitClipOp, ReorderClipOp, RenameProjectOp, SetTrackAudioOp, SetClipAudioOp,
        SetClipLookOp, SetClipPinnedOp, SetClipSpeedOp, SetMarkersOp, SetSubtitlesOp, EditSubtitleOp,
        SetCaptionStyleOp, FitTrackOp,
    ],
    Field(discriminator="action"),
]

# Touching a compiled clip through one of these is the user changing it by hand, which
# is what pinning records. Nobody has to remember to say so, which is the point: the
# alternative is a feedback loop that wipes their work every time it comes round.
_EDITS_BY_HAND = (
    TrimClipOp, MoveClipOp, SplitClipOp, ReorderClipOp, SetClipLookOp, SetClipAudioOp, SetClipSpeedOp,
)

CueOrder = Tuple[str, Decimal, Decimal]

def cue_order(cue: SubtitleCue) -> CueOrder:
    """Order captions by where the words are, not by where they land.

    The cut decides where a caption appears and the cut can change; the file
    and the second the words were said cannot, so that is what keeps the stored
    order — and with it the IDs — stable across an edit.

    Args:
        cue: The caption.

    Returns:
        A sort key over its source file and time.
    """
    return (cue.asset_id, cue.source_start, cue.source_end)

def number_cues(cues: List[SubtitleCue]) -> List[SubtitleCue]:
    """Give every caption an ID, keeping the ones that already have a unique one.

    Args:
        cues: Captions in timeline order.

    Returns:
        The captions, each with a non-empty, unique `id`.
    """
    seen: set[str] = set()
    numbered: List[SubtitleCue] = []
    for index, cue in enumerate(cues, start=1):
        identifier = cue.id if cue.id and cue.id not in seen else f"c{index}"
        while identifier in seen:
            identifier = f"{identifier}x"
        seen.add(identifier)
        numbered.append(cue if cue.id == identifier else cue.model_copy(update={"id": identifier}))
    return numbered

def _find_track(project: Project, track_id: str) -> Track:
    """Look up a track by ID.

    Args:
        project: Project to search.
        track_id: ID of the track to find.

    Returns:
        The matching track.

    Raises:
        ValueError: If the project has no track with `track_id`.
    """
    for track in project.tracks:
        if track.id == track_id:
            return track
    raise ValueError(f"track {track_id} not found")

def _find_clip(track: Track, clip_id: str) -> Clip:
    """Look up a clip by ID.

    Args:
        track: Track to search.
        clip_id: ID of the clip to find.

    Returns:
        The matching clip.

    Raises:
        ValueError: If the track has no clip with `clip_id`.
    """
    for clip in track.clips:
        if clip.id == clip_id:
            return clip
    raise ValueError(f"clip {clip_id} not found on track {track.id}")

def _new_clip(track: Track, spec: _NewClipSpec, timeline_in: Decimal) -> Clip:
    """Create a clip from an operation's clip fields.

    Args:
        track: Track the clip will be added to.
        spec: Operation describing the clip.
        timeline_in: Start time of the clip on the timeline.

    Returns:
        The new clip. It is not added to the track.

    Raises:
        ValueError: If the track already has a clip with the same ID.
    """
    if any(clip.id == spec.clip_id for clip in track.clips):
        raise ValueError(f"clip {spec.clip_id} already exists on track {track.id}")
    return Clip(
        id=spec.clip_id,
        asset_id=spec.asset_id,
        source_range=spec.source_range,
        timeline_in=timeline_in,
        speed=spec.speed,
        preserve_pitch=spec.preserve_pitch,
        volume=spec.volume,
        audio_fade_in=spec.audio_fade_in,
        audio_fade_out=spec.audio_fade_out,
        audio_lead=spec.audio_lead,
        audio_lag=spec.audio_lag,
        cleanup=spec.cleanup,
        transition_in=spec.transition_in,
        video_fade_in=spec.video_fade_in,
        video_fade_out=spec.video_fade_out,
        color=spec.color,
        layout=spec.layout,
    )

def _shift_clips(track: Track, starting_at: Decimal, delta: Decimal) -> None:
    """Move every clip that starts at or after a position by the same amount.

    Args:
        track: Track whose clips are moved.
        starting_at: Timeline position (seconds); clips starting earlier stay put.
        delta: Seconds to add to each affected clip's `timeline_in`.
    """
    for clip in track.clips:
        if clip.timeline_in >= starting_at:
            clip.timeline_in += delta

def _fit_track(project: Project, track: Track, op: "FitTrackOp", assets: Mapping[str, Asset]) -> None:
    """Make an audio track cover the video exactly, with its fades in the right place.

    Four passes over the track: trim whatever runs past the end and drop
    whatever starts past it, play more of the source when there is some left,
    loop the song when there is not, and put the fades on whichever clips are
    first and last once all that is settled.

    Args:
        project: Project the track belongs to; its video decides the length.
        track: Audio track to fit.
        op: The `fit_track` operation being applied.
        assets: Registered assets keyed by ID, for the length of each source.

    Raises:
        ValueError: If the track is a video track, or the project has no video
            to fit it to.
    """
    if track.track_type == TrackType.VIDEO:
        raise ValueError(
            f"track {track.id} is a video track and sets the length itself; fit_track is for audio tracks"
        )
    video_end = project.duration
    if video_end <= 0:
        raise ValueError("the project has no video yet, so there is nothing to fit the track to")

    kept: List[Clip] = []
    for clip in sorted(track.clips, key=lambda item: item.timeline_in):
        if clip.timeline_in >= video_end:
            continue
        if clip.timeline_out > video_end:
            clip.source_range = TimeRange(
                start=clip.source_range.start,
                end=clip.source_range.start + (video_end - clip.timeline_in) * Decimal(str(clip.speed)),
            )
        kept.append(clip)

    if kept and kept[-1].timeline_out < video_end:
        # Play more of the source before looping: a song trimmed by an earlier fit should
        # carry on from where it stopped rather than jump back to the beginning. This is not
        # looping, so `loop: false` does not turn it off: there is nothing to repeat yet.
        last = kept[-1]
        asset = assets.get(last.asset_id)
        if asset is not None and asset.duration is not None:
            room = Decimal(str(asset.duration)) - last.source_range.end
            needed = (video_end - last.timeline_out) * Decimal(str(last.speed))
            if room > 0:
                last.source_range = TimeRange(
                    start=last.source_range.start,
                    end=last.source_range.end + min(room, needed),
                )

    if op.loop and kept:
        # The longest clip is the untrimmed one, so looping it repeats the whole song rather than an offcut.
        unit = max(kept, key=lambda item: item.source_range.duration)
        copies = 1
        while kept[-1].timeline_out < video_end:
            position = kept[-1].timeline_out
            remaining = (video_end - position) * Decimal(str(unit.speed))
            if remaining <= Decimal("0.001"):
                break
            copies += 1
            identifier = f"{unit.id}-{copies}"
            while any(item.id == identifier for item in kept):
                copies += 1
                identifier = f"{unit.id}-{copies}"
            kept.append(Clip(
                id=identifier,
                asset_id=unit.asset_id,
                source_range=TimeRange(
                    start=unit.source_range.start,
                    end=unit.source_range.start + min(unit.source_range.duration, remaining),
                ),
                timeline_in=position,
                speed=unit.speed,
                volume=unit.volume,
            ))

    track.clips = kept
    if not kept:
        return

    # Whatever the track looked like before, the fades now belong to its real first and last clips.
    longest_fade_out = max((clip.audio_fade_out for clip in kept), default=Decimal(0))
    for clip in kept[:-1]:
        clip.audio_fade_out = Decimal(0)
    for clip in kept[1:]:
        clip.audio_fade_in = Decimal(0)
    if op.fade_in is not None:
        kept[0].audio_fade_in = op.fade_in
    kept[-1].audio_fade_out = op.fade_out if op.fade_out is not None else longest_fade_out
    kept[0].audio_fade_in = min(kept[0].audio_fade_in, kept[0].timeline_duration)
    kept[-1].audio_fade_out = min(kept[-1].audio_fade_out, kept[-1].timeline_duration)
    if kept[0] is kept[-1]:
        # One clip carrying both fades: shrink them together rather than letting them overlap.
        only = kept[0]
        total = only.audio_fade_in + only.audio_fade_out
        if total > only.timeline_duration:
            scale = only.timeline_duration / total
            only.audio_fade_in *= scale
            only.audio_fade_out *= scale

def _touched_clips(track: Track, op: EditOperation) -> List[Clip]:
    """Find the clips a hand edit changed.

    Args:
        track: Track the operation ran on.
        op: The operation, already applied.

    Returns:
        The clips it named that are still on the track. A `split_clip` names
        both halves, because the cut between them is the change.
    """
    wanted = {getattr(op, "clip_id", None), getattr(op, "new_clip_id", None)} - {None}
    return [clip for clip in track.clips if clip.id in wanted]

def apply_operation(project: Project, op: EditOperation, assets: Mapping[str, Asset]) -> None:
    """Apply a single edit operation to a project in place.

    Rippling operations (`insert_clip`, `reorder_clip`, and `trim_clip` and
    `delete_clip` unless `ripple` is false) shift the clips that follow the
    edit point on the same track, so the rest of that track keeps its order
    and spacing. `split_clip` changes no position at all: the two halves
    together occupy exactly what the original clip did.
    Clips on other tracks never move. Clips on the
    affected track are kept sorted by `timeline_in`, the order in which they
    are rendered.

    Args:
        project: Project to modify.
        op: Operation to apply.
        assets: Registered assets keyed by ID, for operations that need to
            know how long a source file is. `fit_track` uses them to play more
            of a song before it loops back to the start.

    Raises:
        ValueError: If the operation references a track or clip that does not
            exist, or adds a track or clip whose ID is already in use.
    """
    if isinstance(op, AddTrackOp):
        if any(track.id == op.track_id for track in project.tracks):
            raise ValueError(f"track {op.track_id} already exists")
        project.tracks.append(
            Track(id=op.track_id, track_type=op.track_type, duck_under_speech=op.duck_under_speech)
        )
        return

    if isinstance(op, SetMarkersOp):
        named: dict = {}
        for marker in op.markers:
            if marker.id in named:
                raise ValueError(f"two markers share the id {marker.id!r}; a marker is named once")
            named[marker.id] = marker
        project.markers = sorted(op.markers, key=lambda item: (item.timeline_in, item.id))
        return

    if isinstance(op, RenameProjectOp):
        project.name = op.name.strip() if op.name and op.name.strip() else None
        return

    if isinstance(op, SetSubtitlesOp):
        project.subtitles = number_cues(sorted(op.cues, key=cue_order))
        return

    if isinstance(op, EditSubtitleOp):
        cue = next((item for item in project.subtitles if item.id == op.cue_id), None)
        if cue is None:
            known = ", ".join(item.id for item in project.subtitles[:5]) or "none"
            raise ValueError(f"caption {op.cue_id} not found; the project's captions start with {known}")
        if op.delete:
            project.subtitles = [item for item in project.subtitles if item.id != op.cue_id]
            return
        named = {
            field: value for field, value in
            (("text", op.text), ("secondary", op.secondary), ("speaker", op.speaker),
             ("source_start", op.source_start), ("source_end", op.source_end))
            if value is not None
        }
        # Correcting the words leaves the word timings describing words that are no longer
        # there, and a caption lighting up against the wrong syllables is worse than one
        # lighting up whole. They go, and `generate_subtitles` makes them again.
        if op.text is not None:
            named["words"] = []
        # Re-run the model's own rules on the edited cue: model_copy skips them.
        updated = SubtitleCue.model_validate(cue.model_copy(update=named).model_dump())
        project.subtitles = sorted(
            [updated if item.id == op.cue_id else item for item in project.subtitles],
            key=cue_order,
        )
        return

    if isinstance(op, SetCaptionStyleOp):
        # A preset is where the settings start, so anything given with it wins. Applied
        # over the preset rather than over the project's current style: asking for a
        # platform and getting it half-mixed with the last one is nobody's intent.
        base = CAPTION_PRESETS[op.preset] if op.preset else project.caption_style
        if op.style is None:
            project.caption_style = base.model_copy(deep=True)
        else:
            project.caption_style = base.model_copy(update=op.style.model_dump(exclude_unset=True), deep=True)
        return

    track = _find_track(project, op.track_id)
    if isinstance(op, SetTrackAudioOp):
        if op.duck_under_speech is not None:
            track.duck_under_speech = op.duck_under_speech
        return
    if isinstance(op, AddClipOp):
        track.clips.append(_new_clip(track, op, op.timeline_in))
    elif isinstance(op, InsertClipOp):
        if op.before_clip_id is None:
            position = max((clip.timeline_out for clip in track.clips), default=Decimal(0))
        else:
            position = _find_clip(track, op.before_clip_id).timeline_in
        clip = _new_clip(track, op, position)
        _shift_clips(track, position, clip.timeline_duration)
        track.clips.append(clip)
    elif isinstance(op, TrimClipOp):
        clip = _find_clip(track, op.clip_id)
        old_out = clip.timeline_out
        clip.source_range = op.new_source_range
        if op.ripple:
            _shift_clips(track, old_out, clip.timeline_out - old_out)
    elif isinstance(op, MoveClipOp):
        _find_clip(track, op.clip_id).timeline_in = op.new_timeline_in
    elif isinstance(op, DeleteOp):
        clip = _find_clip(track, op.clip_id)
        track.clips.remove(clip)
        if op.ripple:
            _shift_clips(track, clip.timeline_out, -clip.timeline_duration)
    elif isinstance(op, SplitClipOp):
        clip = _find_clip(track, op.clip_id)
        if any(other.id == op.new_clip_id for other in track.clips):
            raise ValueError(f"clip {op.new_clip_id} already exists on track {track.id}")
        if op.at is not None:
            at_source = clip.source_range.start + (op.at - clip.timeline_in) * Decimal(str(clip.speed))
        else:
            at_source = op.at_source
        if not clip.source_range.start < at_source < clip.source_range.end:
            raise ValueError(
                f"clip {clip.id} cannot be split at source {at_source}s: the cut must fall inside "
                f"{clip.source_range.start}s to {clip.source_range.end}s "
                f"({clip.timeline_in}s to {clip.timeline_out}s on the timeline)"
            )
        tail = Clip(
            id=op.new_clip_id,
            asset_id=clip.asset_id,
            source_range=TimeRange(start=at_source, end=clip.source_range.end),
            timeline_in=clip.timeline_in + (at_source - clip.source_range.start) / Decimal(str(clip.speed)),
            speed=clip.speed,
            volume=clip.volume,
            # A fade-out belongs to the end of the original clip, which is now the end of the tail.
            audio_fade_out=clip.audio_fade_out,
            video_fade_out=clip.video_fade_out,
            # The halves are the same shot, so they keep the same look and the same origin.
            color=clip.color,
            layout=clip.layout,
            from_plan_id=clip.from_plan_id,
            from_clip_ids=list(clip.from_clip_ids),
        )
        clip.source_range = TimeRange(start=clip.source_range.start, end=at_source)
        clip.audio_fade_out = Decimal(0)
        clip.video_fade_out = Decimal(0)
        # Keep the fades the halves inherited inside their new, shorter lengths.
        clip.audio_fade_in = min(clip.audio_fade_in, clip.timeline_duration)
        clip.video_fade_in = min(clip.video_fade_in, clip.timeline_duration)
        tail.audio_fade_out = min(tail.audio_fade_out, tail.timeline_duration)
        tail.video_fade_out = min(tail.video_fade_out, tail.timeline_duration)
        track.clips.append(tail)
    elif isinstance(op, ReorderClipOp):
        if op.before_clip_id == op.clip_id:
            raise ValueError(f"clip {op.clip_id} cannot be moved before itself")
        clip = _find_clip(track, op.clip_id)
        target = _find_clip(track, op.before_clip_id) if op.before_clip_id is not None else None
        track.clips.remove(clip)
        _shift_clips(track, clip.timeline_out, -clip.timeline_duration)
        # Read the target's position after the space has closed, so it is where the clip really lands.
        position = target.timeline_in if target is not None else max(
            (other.timeline_out for other in track.clips), default=Decimal(0)
        )
        _shift_clips(track, position, clip.timeline_duration)
        clip.timeline_in = position
        track.clips.append(clip)
    elif isinstance(op, FitTrackOp):
        _fit_track(project, track, op, assets)
    elif isinstance(op, SetClipLookOp):
        clip = _find_clip(track, op.clip_id)
        for field in ("video_fade_in", "video_fade_out"):
            value = getattr(op, field)
            if value is not None:
                setattr(clip, field, value)
        if op.clear_transition:
            clip.transition_in = None
        elif op.transition_in is not None:
            clip.transition_in = op.transition_in
        if op.clear_color:
            clip.color = None
        elif op.color is not None:
            # Only the adjustments the caller named move, so asking for more colour
            # after asking for more light does not undo the light.
            named = op.color.model_dump(exclude_unset=True)
            clip.color = clip.color.model_copy(update=named) if clip.color is not None else op.color
        if op.clear_layout:
            clip.layout = None
        elif op.layout is not None:
            clip.layout = op.layout
    elif isinstance(op, SetClipAudioOp):
        clip = _find_clip(track, op.clip_id)
        for field in ("volume", "audio_fade_in", "audio_fade_out", "audio_lead", "audio_lag"):
            value = getattr(op, field)
            if value is not None:
                setattr(clip, field, value)
        if op.cleanup is not None:
            # Only the repairs the caller named, so asking for the hiss to go does not
            # put the rumble back — the same rule the colour adjustments follow.
            clip.cleanup = clip.cleanup.model_copy(update=op.cleanup.model_dump(exclude_unset=True))
    elif isinstance(op, SetClipSpeedOp):
        clip = _find_clip(track, op.clip_id)
        was = clip.timeline_out
        clip.speed = op.speed
        if op.preserve_pitch is not None:
            clip.preserve_pitch = op.preserve_pitch
        if op.ripple:
            _shift_clips(track, was, clip.timeline_out - was)
    elif isinstance(op, SetClipPinnedOp):
        _find_clip(track, op.clip_id).pinned = op.pinned

    if isinstance(op, _EDITS_BY_HAND):
        # Whatever this operation touched was compiled by a plan, it has now been
        # changed by hand, so the next compile leaves it alone. `split_clip` pins
        # both halves: the cut between them is the change.
        for touched in _touched_clips(track, op):
            if touched.from_plan_id is not None:
                touched.pinned = True
    track.clips.sort(key=lambda clip: clip.timeline_in)

def validate_project(project: Project, assets: Mapping[str, Asset]) -> None:
    """Check that a project's timeline is semantically consistent.

    These rules hold regardless of what the renderer supports: every clip
    references a registered asset, stays within that asset's duration, does
    not overlap another clip on the same track, and has fades that fit within
    its length. Overlap has two deliberate exceptions. A clip whose sound
    leads or lags overlaps its neighbour's sound, which is what a J or an L
    cut is. A clip with a transition runs its picture over the end of the one
    before it, whether that is a dissolve, a wipe or a dip through a colour.
    Neither moves a clip: the cut stays where it is, and both draw the extra
    media from outside the clip's own source range. Only audio tracks duck under speech, and only clips on a
    video track above the base one are drawn in a layout box.

    Args:
        project: Project to check.
        assets: Registered assets, keyed by asset ID.

    Raises:
        ValueError: If any rule is violated. The message names the offending
            clip and explains how to fix it.
    """
    base = project.base_video_track
    for track in project.tracks:
        if track.duck_under_speech and track.track_type != TrackType.AUDIO:
            raise ValueError(f"track {track.id}: only audio tracks can duck under speech")
        previous: Optional[Clip] = None
        for clip in sorted(track.clips, key=lambda c: c.timeline_in):
            asset = assets.get(clip.asset_id)
            if asset is None:
                raise ValueError(f"clip {clip.id}: asset {clip.asset_id} not found")
            if clip.layout is not None and (track is base or track.track_type != TrackType.VIDEO):
                raise ValueError(
                    f"clip {clip.id}: a layout box only means something on a video track above the base one; "
                    f"track {track.id} is not drawn on top of anything, so its clips fill the frame"
                )
            if asset.duration is not None and clip.source_range.end > asset.duration:
                raise ValueError(
                    f"clip {clip.id}: source range ends at {clip.source_range.end}s, "
                    f"but asset {asset.id} is only {asset.duration}s long"
                )
            if clip.transition_in is not None:
                transition = clip.transition_in
                if previous is None or previous.timeline_out != clip.timeline_in:
                    raise ValueError(
                        f"clip {clip.id}: a {transition.kind} runs this clip in over the one before it, and there "
                        f"is no clip ending where this one starts on track {track.id}. To come up from black, "
                        "use video_fade_in"
                    )
                if transition.seconds > previous.timeline_duration:
                    raise ValueError(
                        f"clip {clip.id}: a {transition.seconds}s {transition.kind} is longer than clip "
                        f"{previous.id}, which runs {previous.timeline_duration}s and is what it would run over"
                    )
                if clip.video_source_start < 0:
                    raise ValueError(
                        f"clip {clip.id}: a {transition.seconds}s {transition.kind} reaches back to "
                        f"{clip.video_source_start}s of asset {asset.id}, before the file starts"
                    )
                # A dip is two mixes with a colour between them, so it needs a frame for
                # each. Shorter than that and there is no dip to see, only a filtergraph
                # asking for a mix of no length. Each kind says how little it can live on.
                frames = transition.seconds * project.fps_num / project.fps_den
                if frames < transition.least_frames:
                    raise ValueError(
                        f"clip {clip.id}: a {transition.seconds}s {transition.kind} is under "
                        f"{transition.least_frames} frame(s) at {project.fps_num}/{project.fps_den}fps, "
                        "which is less than it takes to draw one"
                    )
            if (clip.audio_lead or clip.audio_lag) and not asset.has_audio:
                raise ValueError(
                    f"clip {clip.id}: asks for its sound to run outside its picture, but asset {asset.id} "
                    "has no sound to run"
                )
            if clip.audio_timeline_in < 0:
                raise ValueError(
                    f"clip {clip.id}: a lead of {clip.audio_lead}s would start its sound at "
                    f"{clip.audio_timeline_in}s, before the timeline begins"
                )
            if clip.audio_source_start < 0:
                raise ValueError(
                    f"clip {clip.id}: a lead of {clip.audio_lead}s reaches back to "
                    f"{clip.audio_source_start}s of asset {asset.id}, before the file starts"
                )
            if asset.duration is not None and clip.audio_source_end > asset.duration:
                raise ValueError(
                    f"clip {clip.id}: a lag of {clip.audio_lag}s reaches to "
                    f"{clip.audio_source_end}s of asset {asset.id}, which is only {asset.duration}s long"
                )
            # Against the sound's length, not the picture's: a lead and a lag make the sound
            # longer than the clip, and a fade belongs to the sound.
            if clip.audio_fade_in + clip.audio_fade_out > clip.audio_timeline_duration:
                raise ValueError(
                    f"clip {clip.id}: audio fades ({clip.audio_fade_in}s in + {clip.audio_fade_out}s out) "
                    f"are longer than the clip's sound ({clip.audio_timeline_duration}s)"
                )
            if clip.video_fade_in + clip.video_fade_out > clip.timeline_duration:
                raise ValueError(
                    f"clip {clip.id}: video fades ({clip.video_fade_in}s in + {clip.video_fade_out}s out) "
                    f"are longer than the clip ({clip.timeline_duration}s)"
                )
            if previous is not None and clip.timeline_in < previous.timeline_out:
                raise ValueError(
                    f"clip {clip.id} starts at {clip.timeline_in}s and overlaps clip {previous.id}, "
                    f"which ends at {previous.timeline_out}s on track {track.id}"
                )
            previous = clip
