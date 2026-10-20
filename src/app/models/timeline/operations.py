"""The edits a project takes, a class for each, and EditOperation, the union an edit is read into."""

from decimal import Decimal
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from app.models.timeline.parts import (
    CAPTION_PRESETS, COLOUR, KEEP_LEVEL, MAX_SPEED, MIN_SPEED, TITLE_CARD_LONGEST, TITLE_CARD_SECONDS,
    TITLE_CARD_SHORTEST, CaptionStyle, CardBackground, ClipFit, ClipLayout, ColorAdjust, Marker, Motion, SubtitleCue,
    TextStyle, TimeRange, TitleCard, TrackType, Transition, VoiceCleanup,
)


class AddTrackOp(BaseModel):
    """Edit operation that appends an empty track to the project."""

    action: Literal["add_track"] = "add_track"
    track_id: str = Field(..., description="ID of the new track; must be unique within the project")
    track_type: TrackType
    duck_under_speech: bool = Field(
        default=False,
        description="Lower this track automatically while the video's own sound is loud; audio tracks only",
    )
    voice: Optional[bool] = Field(default=None, description="A voice recorded apart from the picture, such as a narration: the footage's own sound "
                    "and any music drop while it speaks. Null decides from the recording, which counts as a "
                    "voice when it was transcribed and sentences cover at least half of it, unless the track "
                    "ducks under speech; audio tracks only")

class NewClipSpec(BaseModel):
    """Fields shared by operations that create a clip.

    A new clip's look and repairs — transition, colour, layout, J and L cuts,
    voice cleanup — are set by `set_clip_look` and `set_clip_audio` later in
    the same batch, so the schema describes each of them once. They are still
    taken here, unlisted, because `compile_plan` rebuilds a pinned clip by
    inserting it again: leaving them off silently dropped a hand-made J cut or
    transition on the next compile, the one thing pinning promises not to do.
    """

    track_id: str
    clip_id: str = Field(..., description="ID of the new clip; must be unique within the track")
    asset_id: str = Field(..., description="ID returned by import_asset")
    source_range: TimeRange = Field(..., description="Segment of the asset to use")
    speed: float = Field(default=1.0, gt=0)
    preserve_pitch: bool = Field(default=True, description="Keep voices at their recorded pitch when the speed changes")
    volume: float = Field(default=1.0, ge=0, description="Audio gain; 1.0 keeps the original level, 0.5 halves it")
    audio_fade_in: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-in length at the clip start (seconds)")
    audio_fade_out: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-out length at the clip end (seconds)")
    audio_lead: SkipJsonSchema[Decimal] = Field(default=Decimal(0), ge=0)
    audio_lag: SkipJsonSchema[Decimal] = Field(default=Decimal(0), ge=0)
    cleanup: SkipJsonSchema[VoiceCleanup] = Field(default_factory=VoiceCleanup)
    keep_level: SkipJsonSchema[bool] = False
    transition_in: SkipJsonSchema[Optional[Transition]] = None
    video_fade_in: SkipJsonSchema[Decimal] = Field(default=Decimal(0), ge=0)
    video_fade_out: SkipJsonSchema[Decimal] = Field(default=Decimal(0), ge=0)
    color: SkipJsonSchema[Optional[ColorAdjust]] = None
    layout: SkipJsonSchema[Optional[ClipLayout]] = None
    fit: SkipJsonSchema[Optional[ClipFit]] = None
    motion: SkipJsonSchema[Optional[Motion]] = None
    # A title card the compiler puts in, or one made by hand that it puts back.
    card: SkipJsonSchema[Optional[TitleCard]] = None

class AddClipOp(NewClipSpec):
    """Edit operation that places a clip at an absolute timeline position without moving other clips.

    Its transition, colour, layout and sound repairs are set with
    `set_clip_look` and `set_clip_audio` in the same batch.
    """

    action: Literal["add_clip"] = "add_clip"
    timeline_in: Decimal = Field(..., ge=0, description="Start time on the timeline (seconds)")

class InsertClipOp(NewClipSpec):
    """Edit operation that inserts a clip into the sequence, shifting later clips to make room.

    Its transition, colour, layout and sound repairs are set with
    `set_clip_look` and `set_clip_audio` in the same batch.
    """

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
    # Where each song can loop on the beat, by asset ID: filled in by the server from the
    # songs' analyses, which the timeline has no way to read.
    loops: SkipJsonSchema[Dict[str, Tuple[Decimal, Decimal]]] = Field(default_factory=dict)

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
    timeline_start: Optional[Decimal] = Field(
        default=None, ge=0,
        description="New start as a time in the finished cut (seconds), as `get_subtitles` reports it; turned into "
                    "the file's own seconds through the clip playing then. Give this or `source_start`, not both",
    )
    timeline_end: Optional[Decimal] = Field(
        default=None, ge=0,
        description="New end as a time in the finished cut (seconds). Both times have to fall inside one clip of "
                    "the caption's file. Give this or `source_end`, not both",
    )
    for_all_projects: bool = Field(
        default=False,
        description="Also remember this caption against its file, so every project captioning this footage starts "
                    "from it. Only for a correction of what was said — a misheard word or a name. Left off, it stays "
                    "in this project: anything worded for this cut, such as 「第三、四版」, would be wrong in the next",
    )

    @model_validator(mode="after")
    def validate_one_clock(self):
        """Refuse a time given both ways at once.

        Returns:
            The validated `EditSubtitleOp` instance.

        Raises:
            ValueError: If a start or an end is given both in the file's seconds and the cut's.
        """
        if self.source_start is not None and self.timeline_start is not None:
            raise ValueError("give the start as source_start or timeline_start, not both")
        if self.source_end is not None and self.timeline_end is not None:
            raise ValueError("give the end as source_end or timeline_end, not both")
        return self

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

class AddSubtitleOp(BaseModel):
    """Edit operation that adds one caption, leaving the rest untouched.

    For a line the transcript missed, or text written for a picture nobody
    speaks over — 「原本的杯子區」 over a shot of the counter.
    """

    action: Literal["add_subtitle"] = "add_subtitle"
    asset_id: Optional[str] = Field(
        default=None, min_length=1,
        description="The file the caption belongs to; left out with `timeline_start`, it is the file playing then",
    )
    source_start: Optional[Decimal] = Field(default=None, ge=0, description="When it starts in that file (seconds)")
    source_end: Optional[Decimal] = Field(default=None, ge=0, description="When it ends in that file (seconds)")
    text: str = Field(..., min_length=1, description="The caption text")
    secondary: str = Field(default="", description="Second line of a bilingual caption")
    speaker: Optional[str] = Field(default=None, description="Who says it, if anybody")
    timeline_start: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Where it starts as a time in the finished cut (seconds), as `get_subtitles` reports it; turned into "
                    "the file's own seconds through the clip playing then. Give this or `source_start`, not both",
    )
    timeline_end: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Where it ends as a time in the finished cut (seconds). Both times have to fall inside one clip of "
                    "the caption's file. Give this or `source_end`, not both",
    )
    for_all_projects: bool = Field(default=False, description="As on `edit_subtitle`")

    @model_validator(mode="after")
    def validate_where(self):
        """Ensure the caption says where it goes, one way only.

        Returns:
            The validated `AddSubtitleOp` instance.

        Raises:
            ValueError: Unless it gives a file and its seconds, or two times in the cut.
        """
        by_source = self.source_start is not None or self.source_end is not None
        by_cut = self.timeline_start is not None or self.timeline_end is not None
        if by_source == by_cut:
            raise ValueError("place a caption with asset_id, source_start and source_end, or with timeline_start "
                             "and timeline_end — one of the two")
        if by_source and (self.asset_id is None or self.source_start is None or self.source_end is None):
            raise ValueError("a caption placed in its file needs asset_id, source_start and source_end")
        if by_cut and (self.timeline_start is None or self.timeline_end is None):
            raise ValueError("a caption placed in the cut needs both timeline_start and timeline_end")
        return self

class SetSubtitlesOp(BaseModel):
    """Edit operation that replaces the project's captions."""

    action: Literal["set_subtitles"] = "set_subtitles"
    cues: List[SubtitleCue] = Field(default_factory=list, description="The complete new set of captions; an empty list removes them")
    for_all_projects: bool = Field(
        default=False,
        description="Remember these captions against their files for every project, as each one's `local` says. "
                    "Left off, every one is kept to this project",
    )

class SetCaptionStyleOp(BaseModel):
    """Edit operation that changes how the captions are drawn.

    A preset is a starting point, not a lock: anything given alongside it wins,
    so `{"preset": "tiktok", "karaoke": false}` is that platform's safe area
    without the word-by-word lighting. Without a preset, only the settings
    given change and the rest stay as the project has them, a vertical
    video's placement included.
    """

    action: Literal["set_caption_style"] = "set_caption_style"
    preset: Optional[str] = Field(
        default=None,
        # Listed from the presets themselves, so the description cannot name fewer than there are.
        description="Where the video is going: one of "
                    + ", ".join(f"`{name}`" for name in CAPTION_PRESETS)
                    + " (`plain` for no platform's furniture at all). Each one sets the safe area, the text "
                      "size and the outline",
    )
    style: Optional[CaptionStyle] = Field(
        default=None,
        description="Settings to change. With a preset they go on top of it, which replaces the whole style; "
                    "on their own they change only themselves, and every setting not given stays as it is. The "
                    "result gives the style as it now stands",
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

class AddTitleCardOp(BaseModel):
    """Edit operation that puts a title card into the sequence, shifting later clips to make room.

    Only when the user asks for one. Its background is the shot it leads into,
    blurred, unless it is given a photo or a colour.
    """

    action: Literal["add_title_card"] = "add_title_card"
    track_id: str = Field(..., description="The main video track")
    clip_id: str = Field(..., description="ID of the new card; must be unique within the track")
    before_clip_id: Optional[str] = Field(
        default=None,
        description="Put it at the start of this clip; omit to put it after the last clip",
    )
    title: str = Field(..., min_length=1, max_length=60)
    subtitle: Optional[str] = Field(default=None, max_length=80)
    seconds: Decimal = Field(default=TITLE_CARD_SECONDS, ge=TITLE_CARD_SHORTEST, le=TITLE_CARD_LONGEST)
    background: Optional[CardBackground] = Field(
        default=None,
        description="Null: `picture` with a photo, else `blur` of the shot next to it",
    )
    colour: Optional[str] = Field(default=None, pattern=COLOUR, description="For a `colour` background")
    photo_asset_id: Optional[str] = Field(default=None, description="A photo from the library to put behind it")

class SetTitleCardOp(BaseModel):
    """Edit operation that changes a title card: its words, its background or its length."""

    action: Literal["set_title_card"] = "set_title_card"
    track_id: str
    clip_id: str
    title: Optional[str] = Field(default=None, min_length=1, max_length=60)
    subtitle: Optional[str] = Field(default=None, max_length=80, description="An empty string takes it away")
    seconds: Optional[Decimal] = Field(
        default=None, ge=TITLE_CARD_SHORTEST, le=TITLE_CARD_LONGEST,
        description="Its new length; later clips move to keep up",
    )
    background: Optional[CardBackground] = None
    colour: Optional[str] = Field(default=None, pattern=COLOUR)
    photo_asset_id: Optional[str] = Field(default=None, description="A photo from the library to put behind it")

class TextTimes(BaseModel):
    """When a text layer shows: seconds of the clip's file, or seconds of the cut."""

    start: Optional[Decimal] = Field(default=None, ge=0, description="First second of the clip's file it covers")
    end: Optional[Decimal] = Field(default=None, gt=0, description="Second of the clip's file it stops at")
    timeline_start: Optional[Decimal] = Field(
        default=None, ge=0, description="Or: where it starts in the cut, in seconds; turned into the file's own",
    )
    timeline_end: Optional[Decimal] = Field(default=None, gt=0, description="Or: where it stops in the cut")

class AddTextOp(TextTimes):
    """Edit operation that puts words over a clip, pinned to its footage.

    Only when the user asks for words on screen. Left without times it
    covers the whole clip.
    """

    action: Literal["add_text"] = "add_text"
    track_id: str
    clip_id: str = Field(..., description="The clip whose footage it goes over")
    text_id: str = Field(..., description="ID of the new text; unique within the clip")
    text: str = Field(..., min_length=1, max_length=60)
    second: Optional[str] = Field(default=None, max_length=60, description="A smaller second line, if any")
    style: TextStyle = "name"
    x: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Put it here across instead of where its style goes, 0 to 1: the centre, or the left edge of a "
                    "name or a place; it is kept inside the frame's margins",
    )
    y: Optional[float] = Field(default=None, ge=0, le=1,
                               description="Put its middle here down the frame instead, 0 to 1; kept off the captions")
    by_plan: SkipJsonSchema[bool] = False
    hidden: SkipJsonSchema[bool] = False

class SetTextOp(TextTimes):
    """Edit operation that changes a text layer: its words, its style, where or when it shows."""

    action: Literal["set_text"] = "set_text"
    track_id: str
    clip_id: str
    text_id: str
    text: Optional[str] = Field(default=None, min_length=1, max_length=60)
    second: Optional[str] = Field(default=None, max_length=60, description="An empty string takes it away")
    style: Optional[TextStyle] = Field(
        default=None, description="Changing it puts the words where the new style goes, unless x and y are given too",
    )
    x: Optional[float] = Field(default=None, ge=0, le=1, description="Move it: see the text's own `x`")
    y: Optional[float] = Field(default=None, ge=0, le=1, description="Move it: see the text's own `y`")

class RemoveTextOp(BaseModel):
    """Edit operation that takes a text layer off a clip."""

    action: Literal["remove_text"] = "remove_text"
    track_id: str
    clip_id: str
    text_id: str

class SetClipLookOp(BaseModel):
    """Edit operation that changes a clip's picture: how it comes in, its colour, and where it sits.

    A transition runs the clip in over the end of the one before it and ends
    on the cut rather than straddling it, so it never changes how long the
    sequence runs. Omitted fields stay unchanged.
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
    fit: Optional[ClipFit] = Field(
        default=None,
        description="How a shot of another shape than the frame sits in it — a landscape shot in a vertical video: "
                    "`fill` crops it, around a centre of your choosing or the face; `whole` shows all of it over a "
                    "blurred copy",
    )
    clear_fit: bool = Field(default=False, description="Let the shot decide again: crop around a face, else show it whole")
    motion: Optional[Motion] = Field(
        default=None,
        description="For a photo: how it moves while on screen — `push`, `pull`, `pan_left`, `pan_right` or `none`",
    )

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
            (self.fit, self.clear_fit, "fit", "clear_fit"),
            (self.transition_in, self.clear_transition, "transition_in", "clear_transition"),
        ):
            if value is not None and clear:
                raise ValueError(f"give either {name} or {cleared}, not both")
        return self

class RenameProjectOp(BaseModel):
    """Edit operation that gives the project a new name. Every project has one."""

    action: Literal["rename_project"] = "rename_project"
    name: str = Field(..., min_length=1, max_length=120, description="What to call the project")

    @field_validator("name")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Refuse a name that is only spaces, which would leave the project nameless."""
        if not value.strip():
            raise ValueError("a project needs a name")
        return value.strip()

class SetTrackAudioOp(BaseModel):
    """Edit operation that changes a whole track's audio behaviour; omitted fields stay unchanged.

    A `voice` track is a narration recorded apart from the picture, which the
    footage's own sound and the music drop under. Left unsaid, a track counts
    as a voice when its recording was transcribed and is mostly sentences.
    """

    action: Literal["set_track_audio"] = "set_track_audio"
    track_id: str
    duck_under_speech: Optional[bool] = Field(
        default=None,
        description="Lower this track automatically while the video's own sound is loud; audio tracks only",
    )
    voice: Optional[bool] = Field(
        default=None,
        description="True or false to say whether this track is a voice recorded apart from the picture; "
                    "omit to leave it as it is. A track nobody has said this about is judged from its recording",
    )
    voice_auto: bool = Field(
        default=False,
        description="Forget what was said about `voice`, and judge from the recording again",
    )

class SetClipAudioOp(BaseModel):
    """Edit operation that changes a clip's audio level, fades, or how far its sound runs outside its picture.

    A lead brings the next scene's sound in under the end of the shot still on
    screen (a J cut); a lag lets a line finish over the shot that follows (an
    L cut). Both take their extra sound from outside the clip's
    `source_range`. Omitted fields stay unchanged.
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
    keep_level: Optional[bool] = Field(default=None, description=KEEP_LEVEL)

class SetMarkersOp(BaseModel):
    """Edit operation that replaces the timeline's structure markers: where each part of the video begins.

    The whole set at once, like captions: a marker means something only
    against the ones either side of it, so they are written together.
    `compile_plan` writes one per beat; markers added by hand have no beat
    behind them and survive the next compile.
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
        SetClipLookOp, SetClipPinnedOp, SetClipSpeedOp, SetMarkersOp, SetSubtitlesOp, EditSubtitleOp, AddSubtitleOp,
        SetCaptionStyleOp, FitTrackOp, AddTitleCardOp, SetTitleCardOp, AddTextOp, SetTextOp, RemoveTextOp,
    ],
    Field(discriminator="action"),
]
