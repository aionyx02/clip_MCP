from decimal import Decimal
from enum import Enum
from typing import Annotated, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, Field, computed_field, model_validator

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

class SubtitleCue(BaseModel):
    """One caption with its place on the edited timeline."""

    id: str = Field(default="", description="Stable ID for this caption, so a single line can be corrected on its own")
    start: Decimal = Field(..., ge=0, description="When the caption appears (seconds on the timeline)")
    end: Decimal = Field(..., ge=0, description="When the caption disappears (seconds on the timeline)")
    text: str = Field(..., description="The caption text; it is wrapped to fit the frame when rendered")

    @model_validator(mode="after")
    def validate_span(self):
        """Ensure the caption is on screen for a positive length of time.

        Returns:
            The validated `SubtitleCue` instance.

        Raises:
            ValueError: If `end` is not after `start`.
        """
        if self.end <= self.start:
            raise ValueError("a subtitle must end after it starts")
        return self

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
    speed: float = Field(default=1.0, gt=0)
    volume: float = Field(default=1.0, ge=0, description="Audio gain; 1.0 keeps the original level, 0.5 halves it")
    audio_fade_in: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-in length at the clip start (seconds)")
    audio_fade_out: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-out length at the clip end (seconds)")
    video_fade_in: Decimal = Field(default=Decimal(0), ge=0, description="Fade in from black at the clip start (seconds)")
    video_fade_out: Decimal = Field(default=Decimal(0), ge=0, description="Fade out to black at the clip end (seconds)")
    color: Optional[ColorAdjust] = Field(default=None, description="Picture adjustments; null leaves the picture as shot")
    layout: Optional[ClipLayout] = Field(
        default=None,
        description="Where the clip is drawn, for clips on a video track above the first; null fills the frame",
    )

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
    subtitles: List[SubtitleCue] = Field(
        default_factory=list,
        description="Captions burned into the picture when render_project is called with burn_subtitles",
    )

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
    volume: float = Field(default=1.0, ge=0, description="Audio gain; 1.0 keeps the original level, 0.5 halves it")
    audio_fade_in: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-in length at the clip start (seconds)")
    audio_fade_out: Decimal = Field(default=Decimal(0), ge=0, description="Audio fade-out length at the clip end (seconds)")
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
    cue_id: str = Field(..., description="ID of the caption, as generate_subtitles and get_subtitles report it")
    text: Optional[str] = Field(default=None, description="New text for this caption")
    start: Optional[Decimal] = Field(default=None, ge=0, description="New start on the timeline (seconds)")
    end: Optional[Decimal] = Field(default=None, ge=0, description="New end on the timeline (seconds)")
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

class SetClipLookOp(BaseModel):
    """Edit operation that changes a clip's picture: its fades from and to black, and its colour; omitted fields stay unchanged."""

    action: Literal["set_clip_look"] = "set_clip_look"
    track_id: str
    clip_id: str
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
        for value, clear, name in (
            (self.color, self.clear_color, "color"),
            (self.layout, self.clear_layout, "layout"),
        ):
            if value is not None and clear:
                raise ValueError(f"give either {name} or clear_{name}, not both")
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
    """Edit operation that changes a clip's audio level or fades; omitted fields stay unchanged."""

    action: Literal["set_clip_audio"] = "set_clip_audio"
    track_id: str
    clip_id: str
    volume: Optional[float] = Field(default=None, ge=0, description="Audio gain; 1.0 keeps the original level, 0.5 halves it")
    audio_fade_in: Optional[Decimal] = Field(default=None, ge=0, description="Audio fade-in length at the clip start (seconds)")
    audio_fade_out: Optional[Decimal] = Field(default=None, ge=0, description="Audio fade-out length at the clip end (seconds)")

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
        SetClipLookOp, SetClipPinnedOp, SetSubtitlesOp, EditSubtitleOp, FitTrackOp,
    ],
    Field(discriminator="action"),
]

# Touching a compiled clip through one of these is the user changing it by hand, which
# is what pinning records. Nobody has to remember to say so, which is the point: the
# alternative is a feedback loop that wipes their work every time it comes round.
_EDITS_BY_HAND = (TrimClipOp, MoveClipOp, SplitClipOp, ReorderClipOp, SetClipLookOp, SetClipAudioOp)

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
        volume=spec.volume,
        audio_fade_in=spec.audio_fade_in,
        audio_fade_out=spec.audio_fade_out,
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

    if isinstance(op, RenameProjectOp):
        project.name = op.name.strip() if op.name and op.name.strip() else None
        return

    if isinstance(op, SetSubtitlesOp):
        project.subtitles = number_cues(sorted(op.cues, key=lambda cue: cue.start))
        return

    if isinstance(op, EditSubtitleOp):
        cue = next((item for item in project.subtitles if item.id == op.cue_id), None)
        if cue is None:
            known = ", ".join(item.id for item in project.subtitles[:5]) or "none"
            raise ValueError(f"caption {op.cue_id} not found; the project's captions start with {known}")
        if op.delete:
            project.subtitles = [item for item in project.subtitles if item.id != op.cue_id]
            return
        # Re-run the model's own rules on the edited cue: model_copy skips them.
        updated = SubtitleCue.model_validate(cue.model_copy(update={
            field: value for field, value in
            (("text", op.text), ("start", op.start), ("end", op.end)) if value is not None
        }).model_dump())
        project.subtitles = sorted(
            [updated if item.id == op.cue_id else item for item in project.subtitles],
            key=lambda item: item.start,
        )
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
        for field in ("volume", "audio_fade_in", "audio_fade_out"):
            value = getattr(op, field)
            if value is not None:
                setattr(clip, field, value)
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
    its length. Only audio tracks duck under speech, and only clips on a
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
            if clip.audio_fade_in + clip.audio_fade_out > clip.timeline_duration:
                raise ValueError(
                    f"clip {clip.id}: audio fades ({clip.audio_fade_in}s in + {clip.audio_fade_out}s out) "
                    f"are longer than the clip ({clip.timeline_duration}s)"
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
