from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field


class Asset(BaseModel):
    """A registered source media file and the probe facts editing relies on."""

    id: str
    path: str = Field(..., description="Absolute path to the media file")
    duration: Optional[Decimal] = Field(default=None, description="Container duration (seconds), if known")
    has_video: bool
    has_audio: bool

class Span(BaseModel):
    """A time span within a media file, in seconds."""

    start: float
    end: float

class TranscriptWord(BaseModel):
    """A transcribed word or character with its timing."""

    start: float
    end: float
    text: str

class TranscriptSegment(BaseModel):
    """A transcribed sentence or phrase with its timing."""

    start: float
    end: float
    text: str
    words: List[TranscriptWord] = Field(default_factory=list)

class Transcript(BaseModel):
    """Speech transcribed from a media file."""

    language: str = Field(..., description="Detected or requested language code, such as 'zh'")
    model: str = Field(..., description="Speech recognition model that produced the transcript")
    chinese_variant: Optional[str] = Field(default=None, description="Chinese variant the text was converted to, such as 'zh-TW'")
    segments: List[TranscriptSegment] = Field(default_factory=list)

class Measured(BaseModel):
    """A stretch of a file, and numbers measured over the whole of it.

    Subclasses add the numbers. Everything else about them is the same, and
    the code that averages a measurement onto a clip relies on that: every
    field but `start` and `end` is a number it can weight by overlap, and a
    field left null is one that was not measured rather than one measured as
    zero.
    """

    start: float
    end: float

class ShotMeasurement(Measured):
    """How one shot was shot, measured rather than judged.

    One record per detected shot. Every field is a number the analysis pass
    measured off the picture; none of them says whether the shot is good
    enough, because that depends on what the shot is for. A wobbly handheld
    close-up is a mistake in an interview and the point of a vlog.
    """

    exposure: float = Field(..., description="Mean brightness, 0.0 black to 1.0 white, on the full range whatever "
                                             "range the file carries")
    contrast: float = Field(..., description="Spread between the dark and bright ends of the picture, 0.0 to 1.0")
    blur: Optional[float] = Field(
        default=None,
        description="How blurred the picture is; larger is blurrier, and a sharp shot sits near 5. Null where "
                    "the picture has no detail to measure it on, such as a frame of flat black",
    )
    motion: Optional[float] = Field(
        default=None,
        description="Mean change from one sampled frame to the next, 0.0 for a still shot. Null for a shot with "
                    "only one sample in it, since a change needs two",
    )
    shake: Optional[float] = Field(
        default=None,
        description="Mean change in the camera's own movement, in frame widths; near 0 for a tripod or a smooth pan, "
                    "and far higher for handheld. Null when FFmpeg was built without libvidstab",
    )

# Digital silence measures as minus infinity decibels, which no arithmetic survives, so
# this stands for it. Nothing real is ever this quiet, which is the point: it reads as
# silence rather than as a level.
SILENCE_DB = -120.0

class SoundMeasurement(Measured):
    """What the sound was like over one second, in decibels relative to full scale."""

    loudness: float = Field(..., description="RMS level; -120 stands for digital silence")
    peak: float = Field(..., description="Loudest sample in the window; near 0 means it is at the ceiling")
    noise_floor: float = Field(..., description="Level of the quietest part, which is the hiss under everything")
    flatness: float = Field(
        ...,
        description="How flat the waveform is where it peaks. A clipped recording peaks near 0 dB and is flat there, "
                    "so a high peak together with a high flatness is what clipping looks like",
    )

class FaceMeasurement(Measured):
    """Who was on screen during one second.

    The field names carry the word `face` even though the class already says
    so, because these end up in one flat set of scores on a clip next to
    `exposure` and `loudness`, where a bare `x` would mean nothing.

    Only faces are found, not people: somebody turned away from the camera is
    not counted, and this is what makes it worth saying that `faces` is 0
    rather than that nobody is there.
    """

    faces: int = Field(..., description="Most faces seen at once during the second")
    face_share: Optional[float] = Field(
        default=None,
        description="Area of the largest face as a share of the frame, from 0 to 1. Null when there is no face",
    )
    face_x: Optional[float] = Field(
        default=None,
        description="Centre of the largest face across the frame, 0 at the left edge and 1 at the right",
    )
    face_y: Optional[float] = Field(
        default=None,
        description="Centre of the largest face down the frame, 0 at the top and 1 at the bottom",
    )

class SpeakerTurn(BaseModel):
    """A stretch of a file where one person was speaking.

    The labels are this file's own. `S1` in one recording has nothing to do
    with `S1` in another: diarization says how many voices there were and
    which stretches belong together, not whose they are. Putting names to them
    is for somebody who knows.
    """

    start: float
    end: float
    speaker: str = Field(..., description="Label for the voice, such as 'S1'; unique within this file only")

class Voice(BaseModel):
    """What one voice in a file sounds like, as a vector.

    A diarization label only means something inside its own file, so two
    recordings of the same conversation come back with two unrelated sets of
    them. This is what makes them comparable: the embedding model turns a
    voice into a point, and the same person lands near themselves in a
    different file. Nothing here decides that two are the same — that is a
    threshold, and it lives with the layer that joins them up.

    Kept per label rather than per turn. A label's turns are the same person by
    construction, so measuring them together gives a steadier point than any
    one turn does, and one vector per voice rather than per turn keeps an
    hour-long interview from carrying a few hundred of them.
    """

    speaker: str = Field(..., description="The label this belongs to, as the speaker split wrote it")
    embedding: List[float] = Field(..., min_length=1, description="Where this voice sits in the embedding model's space")
    seconds: float = Field(..., gt=0, description="How much talking it was measured over; a longer look is a surer one")

class AnalysisRecipe(BaseModel):
    """What produced an analysis, so that an out-of-date one can be recognised.

    An analysis is a measurement, and a measurement taken with different
    instruments is not comparable with one taken now. Changing a detection
    threshold or the speech model does not corrupt what is stored — it makes
    it answer a slightly different question — so the recipe travels with the
    result and staleness is reported rather than guessed at.
    """

    version: int = Field(default=0, description="Bumped when the analysis pass itself changes")
    detectors: str = Field(default="", description="Covers every detection threshold and sampling rate")
    speech_model: Optional[str] = Field(default=None, description="Speech model used, when speech was transcribed")
    speaker_model: Optional[str] = Field(default=None, description="Speaker model used, when voices were told apart")

    def differs_from(self, current: "AnalysisRecipe") -> bool:
        """Say whether an analysis taken this way is out of date.

        Args:
            current: The recipe the analysis pass would use now.

        Returns:
            True when the pass or its thresholds have changed, or when a
            transcript or a set of speaker turns was made with a model that is
            no longer the one in use. An analysis that never ran a model is not
            made stale by a change to it: there is nothing in it that model
            wrote.
        """
        if (self.version, self.detectors) != (current.version, current.detectors):
            return True
        for used, now in ((self.speech_model, current.speech_model), (self.speaker_model, current.speaker_model)):
            if used is not None and used != now:
                return True
        return False

class MediaAnalysis(BaseModel):
    """Machine-readable description of an asset's content, used to plan edits."""

    asset_id: str
    duration: float
    scenes: List[Span] = Field(default_factory=list, description="Shots between detected scene changes")
    black_frames: List[Span] = Field(default_factory=list, description="Stretches of (nearly) black picture")
    frozen_frames: List[Span] = Field(default_factory=list, description="Stretches where the picture does not change")
    silences: List[Span] = Field(default_factory=list, description="Stretches without audible sound")
    shots: List[ShotMeasurement] = Field(default_factory=list, description="How each shot was shot, one record per shot")
    sound: List[SoundMeasurement] = Field(default_factory=list, description="What the sound was like, one record per second")
    faces: List[FaceMeasurement] = Field(default_factory=list, description="Who was on screen, one record per second")
    speakers: List[SpeakerTurn] = Field(default_factory=list, description="Which voice was speaking when")
    voices: List[Voice] = Field(
        default_factory=list,
        description="What each of those voices sounds like, one per label, so the same person can be "
                    "recognised in another file",
    )
    transcript: Optional[Transcript] = None
    recipe: AnalysisRecipe = Field(default_factory=AnalysisRecipe, description="What produced this analysis")
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
