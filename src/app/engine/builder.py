import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from statistics import median
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from app.engine import loudness, resources
from app.engine.ffmpeg import escape_filter_path
from app.engine.reframe import Framing, crop_filter
from app.models.media import Asset, MediaAnalysis
from app.models.timeline import Clip, Dip, Project, TrackType, Transition, VoiceCleanup, Wipe

AUDIO_SAMPLE_RATE = 48000
# Streaming platforms normalize to about -14 LUFS, so delivering at that level avoids being turned down.
DEFAULT_LOUDNESS_TARGET = -14.0
# Ducking where the transcripts say who is talking: the music drops this many decibels,
# starting a little before the first word and coming back slowly after the last; talking
# with less than DUCK_HOLD_SECONDS between is held as one stretch. Provisional.
DUCK_DEPTH_DB = 10.0
DUCK_ATTACK_SECONDS = 0.15
DUCK_RELEASE_SECONDS = 0.6
DUCK_HOLD_SECONDS = 0.8
# Ducking by level, for footage nobody transcribed: hold the music well under speech, and
# come back slowly so it does not pump between words.
DUCK_THRESHOLD = 0.03
DUCK_RATIO = 10
DUCK_ATTACK_MS = 20
DUCK_RELEASE_MS = 500
# Ducking the footage under a voice recorded apart from it. The footage's own sound is the
# place — a street, a café — so it is held down rather than taken away. A narration on a
# clip-on microphone is often recorded far quieter than a camera hears a street: the first
# edit this was made for had the street ten decibels over the voice. So the narration is
# first brought to VOICE_KEY_DB (LUFS) whatever level it was recorded at, and it is at
# that level that it is heard and that it drives the ducking: ordinary speech then sits
# about thirteen decibels over the threshold and the place drops by about that much.
# Slower to come back than music, so the place does not swell between words. Provisional.
VOICE_KEY_DB = -20.0
VOICE_DUCK_THRESHOLD = 0.02
VOICE_DUCK_RATIO = 20
VOICE_DUCK_ATTACK_MS = 30
VOICE_DUCK_RELEASE_MS = 800
# How much of a recording has to be sentences for it to count as somebody talking. A
# narration is sentences with breaths between (the first one measured 90%); a song with
# words has an intro, instrumental breaks and an outro. Provisional.
VOICE_SPEECH_SHARE = 0.5
# A gain derived from the measured loudness of a mix that is silent all the way through
# comes out infinite and fills the stream with NaN; the AAC encoder then refuses the frame
# and the whole render fails. `loudness.settle_gain` answers zero for silence, and this
# stays as the guard behind it. That happens whenever nothing on the timeline has sound — footage shot
# on a muted microphone, or a cut with no audio track at all. Replacing NaN with zero
# hands the encoder the silence it was actually given, and leaves every real mix
# untouched. Clamping the samples or converting them to integers instead is not an
# alternative: both turn the NaN into full-scale noise. Two expressions, because
# everything reaching the mix has already been formatted to stereo.
SILENCE_GUARD = r"aeval=exprs=if(isnan(val(0))\,0\,val(0))|if(isnan(val(1))\,0\,val(1)):c=same"
# What the voice repairs are set to. All three are deliberately gentle: each one works by
# throwing part of the recording away, and the failure people actually notice is not too
# much hiss left but a voice that has been scrubbed until it sounds underwater. A hundred
# hertz is below anything a human voice puts out, so the high-pass costs the voice nothing;
# twelve decibels of noise reduction is about half of what afftdn will do. Provisional.
RUMBLE_HZ = 100
HISS_REDUCTION_DB = 12
HISS_FLOOR_DB = -40
SIBILANCE_INTENSITY = 0.4
# Raised whenever the picture chain changes in a way its text does not show, so a
# cached picture made by the old chain is never mistaken for one made by the new.
PICTURE_CACHE_VERSION = 1
# The rate the stems of a sound preview are written at: they are measured, not listened to.
STEM_SAMPLE_RATE = 8000

@dataclass(frozen=True)
class Talking:
    """What the transcripts say about the talking in a cut, for placing the music under it.

    Attributes:
        spans: Where somebody is heard talking, as `(start, end)` in timeline
            seconds — the footage's own sound and any narration.
        music_gains: For each music track, by track ID, and each clip on it,
            by clip ID, the gain in dB that brings that clip's song to the
            level of the talking, so that a clip's volume says how far under
            the voices it sits. Per clip, because two songs on one track can
            be mastered ten decibels apart. A clip left out is played at its
            own level.
        clip_gains: For each clip on the sequence, by clip ID, the gain in dB
            that brings its talking to the level of the rest, or keeps the
            place it was shot in under that talking. A clip left out plays at
            the level it was recorded.
    """

    spans: Tuple[Tuple[float, float], ...] = ()
    music_gains: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    clip_gains: Mapping[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class Segment:
    """A contiguous span of the rendered output, in frames.

    Attributes:
        start_frame: First output frame of the span (inclusive).
        end_frame: Output frame at which the span ends (exclusive).
        clip: Clip rendered in the span, or `None` for a gap.
    """

    start_frame: int
    end_frame: int
    clip: Optional[Clip] = None

def _is_voice(clips: List[Clip], analyses: Mapping[str, MediaAnalysis]) -> bool:
    """Tell from the recordings on a track whether it is somebody talking.

    Not from the beat: a narration on its own file is analyzed as music, and
    syllables are strong enough onsets that it comes back with a tempo — the
    first narration this was made for measured 135.7 BPM. What a narration
    has that a song does not is sentences through most of it.

    Args:
        clips: The track's clips.
        analyses: Analyses by asset ID.

    Returns:
        True when every clip that can be heard comes from a file that was
        transcribed, with sentences covering at least `VOICE_SPEECH_SHARE` of
        the stretch the clip plays. A file never analyzed says nothing, so it
        is not a voice.
    """
    heard = [clip for clip in clips if clip.volume != 0]
    if not heard:
        return False
    for clip in heard:
        analysis = analyses.get(clip.asset_id)
        if analysis is None or analysis.transcript is None:
            return False
        low, high = float(clip.source_range.start), float(clip.source_range.end)
        spoken = sum(
            max(0.0, min(segment.end, high) - max(segment.start, low))
            for segment in analysis.transcript.segments
        )
        if high <= low or spoken / (high - low) < VOICE_SPEECH_SHARE:
            return False
    return True

def voice_keys(
    project: Project,
    analyses: Mapping[str, MediaAnalysis],
    loudness: Callable[[Clip], Optional[float]],
) -> Dict[str, float]:
    """Find the tracks that carry a voice recorded apart from the picture.

    Decided here, at render time, from what the recordings are, because the
    edit that needs it most — a narration laid over footage by hand — is the
    one nobody stops to label.

    Args:
        project: Project to render.
        analyses: Analyses by asset ID, to tell a voice from a song.
        loudness: How loud a clip's talking is, in LUFS before its volume,
            or nothing when it cannot be measured. Measured rather than read
            from the analysis, because an analysis made before per-second
            levels were kept has none, and those are the edits already made.

    Returns:
        The gain in dB that brings each voice track's talking to
        `VOICE_KEY_DB`, keyed by track ID; it is heard at that level and
        ducks the footage from it. Zero when its level could not be measured.
    """
    keys: Dict[str, float] = {}
    for track in project.tracks:
        if track.track_type != TrackType.AUDIO or not track.clips:
            continue
        # A track set to duck under speech was put there as music, whatever it has words in.
        automatic = not track.duck_under_speech and _is_voice(track.clips, analyses)
        if track.voice is False or (track.voice is None and not automatic):
            continue
        levels = [
            level + 20 * math.log10(clip.volume)
            for clip in track.clips
            if clip.volume != 0 and (level := loudness(clip)) is not None
        ]
        keys[track.id] = VOICE_KEY_DB - median(levels) if levels else 0.0
    return keys

def _duck_volume(speech: Sequence[Tuple[float, float]]) -> str:
    """Write the volume filter that lowers music wherever somebody is talking.

    Stretches of talking closer together than `DUCK_HOLD_SECONDS` are held as
    one, so the music does not swell back up between two sentences. The drop
    starts `DUCK_ATTACK_SECONDS` before the first word, so the music is
    already down when the voice arrives, and comes back over
    `DUCK_RELEASE_SECONDS`. Held stretches never overlap once ramped, so each
    contributes its own ramp and they add up to at most one.

    Args:
        speech: Where somebody is talking, as `(start, end)` in timeline
            seconds.

    Returns:
        A `volume` filter evaluated once per frame of sound; a plain pass
        through when there is no talking.
    """
    held: List[List[float]] = []
    for start, end in sorted(speech):
        if held and start - held[-1][1] < DUCK_HOLD_SECONDS:
            held[-1][1] = max(held[-1][1], end)
        else:
            held.append([start, end])
    if not held:
        return "anull"
    attack, release = DUCK_ATTACK_SECONDS, DUCK_RELEASE_SECONDS
    ramps = "+".join(
        f"clip((t-{start - attack:.3f})/{attack:g},0,1)*clip(({end + release:.3f}-t)/{release:g},0,1)"
        for start, end in held
    )
    drop = 1 - 10 ** (-DUCK_DEPTH_DB / 20)
    return f"volume='1-{drop:.4f}*({ramps})':eval=frame"

def _round_half_up(value: Fraction) -> int:
    """Round a fraction to the nearest integer, rounding halves up.

    Args:
        value: Value to round.

    Returns:
        The nearest integer.
    """
    return math.floor(value + Fraction(1, 2))

def _frame_to_sample(frame: int, fps: Fraction) -> int:
    """Convert an output frame boundary to the matching audio sample boundary.

    Args:
        frame: Frame index on the output timeline.
        fps: Output frame rate.

    Returns:
        The nearest sample index at `AUDIO_SAMPLE_RATE`.
    """
    return _round_half_up(Fraction(frame) * AUDIO_SAMPLE_RATE / fps)

def _seconds_to_samples(seconds: Decimal) -> int:
    """Convert a length in seconds to a whole number of audio samples.

    Args:
        seconds: The length.

    Returns:
        The sample count at `AUDIO_SAMPLE_RATE`.
    """
    return _round_half_up(Fraction(seconds) * AUDIO_SAMPLE_RATE)

def _format_seconds(value: Fraction) -> str:
    """Format a time value as a plain decimal string FFmpeg accepts.

    Args:
        value: Time in seconds.

    Returns:
        The time with microsecond precision, without exponent notation.
    """
    return f"{float(value):.6f}"

def _layout(clips: List[Clip], fps: Fraction) -> List[Segment]:
    """Lay out a track's clips as consecutive clip and gap segments.

    Clip boundaries are snapped to the nearest output frame. Each boundary is
    rounded from its absolute timeline position, so rounding errors do not
    accumulate across clips. Empty time before and between clips becomes gap
    segments; the layout ends where the last clip ends.

    Args:
        clips: Clips of one track, in any order.
        fps: Output frame rate.

    Returns:
        Segments covering the track from frame 0 without holes.

    Raises:
        ValueError: If a clip overlaps the previous one or is shorter than one
            frame.
    """
    segments: List[Segment] = []
    cursor = 0
    for clip in sorted(clips, key=lambda c: c.timeline_in):
        start = _round_half_up(Fraction(clip.timeline_in) * fps)
        end = _round_half_up(Fraction(clip.timeline_out) * fps)
        if start < cursor:
            raise ValueError(f"clip {clip.id} overlaps the previous clip")
        if end <= start:
            raise ValueError(f"clip {clip.id} is shorter than one frame")
        if start > cursor:
            segments.append(Segment(cursor, start))
        segments.append(Segment(start, end, clip))
        cursor = end
    return segments

def _input_args(clip: Clip, asset: Asset, frames: int, fps: Fraction) -> List[str]:
    """Build the FFmpeg input options that read a clip's source segment.

    One extra frame is read; the clip's filters trim to the exact length.

    Args:
        clip: Clip whose source segment is read.
        asset: Asset the clip references.
        frames: Frames to read, which is the clip's length on the output
            timeline plus the run-up a transition needs.
        fps: Output frame rate.

    Returns:
        The `-ss`, `-t`, and `-i` options for the input. The seek starts at
        `video_source_start`, which is the clip's own in point unless a
        transition runs it in, in which case it is that much earlier. The length is in
        source seconds, so a clip running at double speed reads twice as much
        of the file as it occupies on the timeline.
    """
    return [
        "-ss", _format_seconds(Fraction(clip.video_source_start)),
        "-t", _format_seconds(Fraction(frames + 1) * Fraction(str(clip.speed)) / fps),
        "-i", asset.path,
    ]

def _audio_input_args(clip: Clip, asset: Asset, fps: Fraction) -> List[str]:
    """Build the FFmpeg input options that read a clip's sound on its own.

    A clip whose sound leads its picture needs audio from before the point the
    picture input is seeked to, so it gets an input of its own rather than the
    picture's being moved — moving it would shift every frame. Only clips that
    actually lead or lag pay for the extra input.

    Args:
        clip: Clip whose sound is read.
        asset: Asset the clip references.
        fps: Output frame rate, used for the same one-frame margin the picture
            input takes.

    Returns:
        The `-ss`, `-t`, and `-i` options for the input.
    """
    return [
        "-ss", _format_seconds(Fraction(clip.audio_source_range.start)),
        "-t", _format_seconds(Fraction(clip.audio_source_range.duration) + Fraction(1) / fps),
        "-i", asset.path,
    ]

def overlay_box(clip: Clip, width: int, height: int) -> Tuple[int, int, int, int]:
    """Work out where a clip is drawn, in whole even pixels.

    Args:
        clip: Clip whose layout is read; no layout means the whole frame.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        The box as `(x, y, width, height)`. Sizes are even, because the
        encoder's chroma subsampling needs even dimensions.
    """
    layout = clip.layout
    if layout is None:
        return 0, 0, width, height
    box_width = max(2, int(round(width * layout.width / 2)) * 2)
    box_height = max(2, int(round(height * layout.height / 2)) * 2)
    return (
        min(int(round(width * layout.x)), width - box_width),
        min(int(round(height * layout.y)), height - box_height),
        box_width,
        box_height,
    )

def _speed_video_filter(clip: Clip) -> str:
    """Stretch or compress a clip's picture to its playback speed.

    Applied before the output frame rate is settled, so that `fps` resamples
    whatever the new timing produced rather than the other way round: doing it
    after would drop or repeat frames that the stretch had already decided on.

    Args:
        clip: Clip whose speed is applied.

    Returns:
        The filter with a leading comma, or an empty string at normal speed.
    """
    if clip.speed == 1.0:
        return ""
    return f",setpts={_format_seconds(Fraction(1) / Fraction(str(clip.speed)))}*PTS"

def _xfade_name(transition: Transition) -> str:
    """Name the FFmpeg transition that draws this one.

    Asked here rather than answered by the transition itself: `xfade` names are
    this renderer's vocabulary, and `models/` is meant to hold data and rules
    with no idea what draws them. How much picture a transition needs and how
    few frames it can live on are facts about the transition and do live there.

    Args:
        transition: The clip's transition.

    Returns:
        The `xfade` transition name. A dip is drawn as two plain mixes through
        a colour rather than one named transition, so it answers with the mix
        each of its halves uses.
    """
    if isinstance(transition, Wipe):
        return f"wipe{transition.direction.value}"
    return "fade"

def _colour_source(colour: str) -> str:
    """Write a colour the way the `color` filter reads it.

    Args:
        colour: `black`, `white`, or a `#RRGGBB` hex colour.

    Returns:
        The colour as a filter argument. Hex is rewritten with an `0x` prefix:
        a `#` is not special to a filtergraph today, and relying on that to
        stay true is not worth the saving.
    """
    return f"0x{colour[1:]}" if colour.startswith("#") else colour

def _cleanup_filters(cleanup: VoiceCleanup) -> List[str]:
    """Build the repairs asked for on one clip's voice.

    Every one of these throws part of the recording away, so none of them runs
    unless it was asked for by name. The settings are deliberately gentle: a
    denoiser turned up far enough to remove all the hiss removes the top of the
    voice with it, and the result is worse than the hiss was.

    Args:
        cleanup: The repairs asked for.

    Returns:
        The filters in the order they undo each other least: the rumble first,
        so the denoiser is not busy modelling it; the sibilance last, on what
        is left. Empty when nothing was asked for.
    """
    if cleanup.is_nothing:
        return []
    filters = []
    if cleanup.rumble:
        # Nothing in a human voice lives below this, so it costs the voice nothing.
        filters.append(f"highpass=f={RUMBLE_HZ}")
    if cleanup.hiss:
        filters.append(f"afftdn=nr={HISS_REDUCTION_DB}:nf={HISS_FLOOR_DB}")
    if cleanup.sibilance:
        filters.append(f"deesser=i={SIBILANCE_INTENSITY}")
    return filters

def _speed_audio_filters(clip: Clip) -> List[str]:
    """Stretch a clip's sound to its playback speed, with or without its pitch.

    Two different things, which is why the clip says which it wants.
    `atempo` resamples the sound in the time domain and leaves the pitch where
    it was, so a person sped up still sounds like themselves. Changing the
    sample rate under it and resampling back is what tape did: everything
    rises and falls with the speed. The first is what somebody talking needs;
    the second is the whole point of a comedy speed-up, and there was no way
    to ask for it.

    `atempo` takes a factor between 0.5 and 2 before the result starts to fall
    apart, so anything further is reached by applying it more than once. The
    factors multiply back to the speed asked for.

    Args:
        clip: Clip whose speed and pitch setting are applied.

    Returns:
        The filters, or an empty list at normal speed.
    """
    if clip.speed == 1.0:
        return []
    speed = Fraction(str(clip.speed))
    if not clip.preserve_pitch:
        # Rounded to a whole rate because that is what asetrate takes; the length is
        # settled by the pad and trim straight after, so the rounding costs nothing.
        return [f"asetrate={round(AUDIO_SAMPLE_RATE * speed)}", f"aresample={AUDIO_SAMPLE_RATE}"]
    remaining, steps = speed, []
    while remaining > 2:
        steps.append(Fraction(2))
        remaining /= 2
    while remaining < Fraction(1, 2):
        steps.append(Fraction(1, 2))
        remaining *= 2
    steps.append(remaining)
    # Resampled again afterwards, which looks redundant and is not. `atempo` hands on a
    # stream whose timing the `apad` and `atrim` that settle the length then read wrong,
    # and they cut it to a quarter of its length and pad the rest with silence — so a
    # clip with any speed on it came out mostly silent. `aresample` reclocks the stream
    # and they measure it correctly. The other branch ends in one already.
    return [f"atempo={float(step):.6f}" for step in steps] + [f"aresample={AUDIO_SAMPLE_RATE}"]

def _picture_chain(
    clip: Clip,
    frames: int,
    width: int,
    height: int,
    fps: Fraction,
    framing: Optional[Framing],
) -> str:
    """Build everything that turns a clip's decoded source into its picture.

    The same chain for the sequence and for everything drawn over it, so a
    clip on an upper track plays at its speed and wears its look exactly the
    way one on the sequence does.

    Args:
        clip: The clip.
        frames: Frames of picture to produce, run-up included.
        width: Width of the box it fills.
        height: Height of the box it fills.
        fps: Output frame rate.
        framing: Where its crop sits, or None for the middle.

    Returns:
        The filters, without input or output labels.
    """
    rate = f"{fps.numerator}/{fps.denominator}"
    return (
        f"setpts=PTS-STARTPTS{_speed_video_filter(clip)},fps={rate},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"{crop_filter(width, height, framing)},setsar=1,format=yuv420p,"
        f"tpad=stop_mode=clone:stop_duration=1,trim=end_frame={frames},setpts=PTS-STARTPTS"
        f"{_clip_video_filter(clip, frames, fps)}"
    )

def _clip_video_filter(clip: Clip, frames: int, fps: Fraction) -> str:
    """Build the look part of a clip's video chain: its colour and its fades.

    Args:
        clip: Clip whose look is applied.
        frames: Length of the clip on the output timeline, in frames.
        fps: Output frame rate.

    Returns:
        The filters to append, each with a leading comma, or an empty string
        when the clip is left as shot.
    """
    chain: List[str] = []
    color = clip.color
    if color is not None and not color.is_neutral:
        if (color.brightness, color.contrast, color.saturation) != (0.0, 1.0, 1.0):
            chain.append(
                f"eq=brightness={color.brightness:g}:contrast={color.contrast:g}:saturation={color.saturation:g}"
            )
        if color.temperature is not None:
            chain.append(f"colortemperature=temperature={color.temperature:g}")
    # Fades are counted in frames, like everything else on the timeline, so they land on exact boundaries.
    fade_in = _fade_frames(clip.video_fade_in, frames, fps)
    if fade_in:
        chain.append(f"fade=t=in:start_frame=0:nb_frames={fade_in}")
    fade_out = _fade_frames(clip.video_fade_out, frames, fps)
    if fade_out:
        chain.append(f"fade=t=out:start_frame={frames - fade_out}:nb_frames={fade_out}")
    return "".join(f",{item}" for item in chain)

def _fade_frames(seconds: Decimal, frames: int, fps: Fraction) -> int:
    """Convert a fade length in seconds to whole frames that fit inside a clip.

    Args:
        seconds: Fade length in seconds.
        frames: Length of the clip in frames.
        fps: Output frame rate.

    Returns:
        The fade length in frames: zero when no fade was asked for, and at
        least one frame otherwise.
    """
    if seconds <= 0:
        return 0
    return max(1, min(frames, round(Fraction(seconds) * fps)))

def _silence_filter(samples: int, output_label: str) -> str:
    """Build a filter that produces silence of an exact length.

    Args:
        samples: Number of samples to produce at `AUDIO_SAMPLE_RATE`.
        output_label: Filtergraph label of the result, such as `[a0]`.

    Returns:
        The filter chain.
    """
    return f"anullsrc=r={AUDIO_SAMPLE_RATE}:cl=stereo,atrim=end_sample={samples}{output_label}"

def _clip_audio_filter(input_label: str, clip: Clip, samples: int, output_label: str, gain_db: float = 0.0) -> str:
    """Build the filter chain that turns a clip's source audio into its output audio.

    The audio is resampled to 48 kHz stereo, padded or trimmed to exactly
    `samples`, and then has the clip's volume and fades applied.

    Args:
        input_label: Filtergraph label of the source audio, such as `[2:a]`.
        clip: Clip providing volume and fade settings.
        samples: Exact number of output samples.
        output_label: Filtergraph label of the result.
        gain_db: What the clip is turned by to sit where it belongs in the
            mix, on top of its own volume.

    Returns:
        The filter chain.
    """
    chain = [
        "asetpts=PTS-STARTPTS",
        f"aresample={AUDIO_SAMPLE_RATE}",
        "aformat=sample_fmts=fltp:channel_layouts=stereo",
        # Before the stretch and before the level: a denoiser models what it is given, so
        # giving it something already sped up or turned down is giving it the wrong thing.
        *_cleanup_filters(clip.cleanup),
        # Before the length is settled: the stretch is what decides how long the sound runs,
        # and padding or trimming it first would make that decision for it.
        *_speed_audio_filters(clip),
        f"apad=whole_len={samples}",
        f"atrim=end_sample={samples}",
    ]
    level = clip.volume * 10 ** (gain_db / 20)
    if level != 1.0:
        chain.append(f"volume={level:.6g}")
    length = Fraction(samples, AUDIO_SAMPLE_RATE)
    if clip.audio_fade_in > 0:
        chain.append(f"afade=t=in:st=0:d={_format_seconds(min(Fraction(clip.audio_fade_in), length))}")
    if clip.audio_fade_out > 0:
        fade_out = min(Fraction(clip.audio_fade_out), length)
        chain.append(f"afade=t=out:st={_format_seconds(length - fade_out)}:d={_format_seconds(fade_out)}")
    chain.append("asetpts=PTS-STARTPTS")
    return f"{input_label}{','.join(chain)}{output_label}"

class FFmpegRenderer:
    """Translates a project timeline into an FFmpeg command."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        """Initialize the renderer.

        Args:
            ffmpeg_bin: Path to, or name of, the FFmpeg executable.
        """
        self.ffmpeg_bin = ffmpeg_bin

    def check_supported(self, project: Project, assets: Mapping[str, Asset]) -> None:
        """Reject timeline features this renderer cannot render yet.

        The first video track is the base and decides the output length.
        Further video tracks are drawn on top of it, each clip in the box its
        `layout` gives. Video-track clips need assets with a video stream, audio-track clips need assets with an
        audio stream, every asset needs a known duration, and every clip must
        be at least one frame long.

        Args:
            project: Project to check.
            assets: Registered assets, keyed by asset ID.

        Raises:
            ValueError: If the project uses an unsupported feature or cannot be
                laid out into frames.
        """
        fps = Fraction(project.fps_num, project.fps_den)
        for track in project.tracks:
            for clip in track.clips:
                asset = assets.get(clip.asset_id)
                if asset is None:
                    raise ValueError(f"clip {clip.id}: asset {clip.asset_id} not found")
                if track.track_type == TrackType.VIDEO and not asset.has_video:
                    raise ValueError(f"clip {clip.id}: asset {asset.id} has no video stream; put audio-only assets on an audio track")
                if track.track_type == TrackType.AUDIO and not asset.has_audio:
                    raise ValueError(f"clip {clip.id}: asset {asset.id} has no audio stream, so it cannot be used on audio track {track.id}")
                if asset.duration is None:
                    raise ValueError(f"clip {clip.id}: asset {asset.id} has no known duration; still images are not supported yet")
            _layout(track.clips, fps)

    def _picture(
        self,
        inputs: List[List[str]],
        pieces: List[Dict[str, object]],
        clip: Clip,
        asset: Asset,
        frames: int,
        fps: Fraction,
        box: Tuple[int, int],
        framing: Optional[Framing],
        cache_dir: Optional[str],
    ) -> Tuple[str, int, bool]:
        """Add a clip's picture to a render, from its source or from the cache.

        With a cache, a clip's picture is rendered once, on its own, into a
        file named after exactly what went into it — the input options, every
        filter, and the source file's size and modification time — and every
        later render that asks for the same picture reads that file instead.
        Named by content rather than by which clip it is, so a clip trimmed
        by hand, or recoloured, or reframed, is a new picture without anybody
        having to say so, and two projects using the same shot the same way
        share it.

        Args:
            inputs: The render's inputs so far; one is added.
            pieces: Pictures still to be rendered into the cache; one is added
                when this picture is not there yet.
            clip: The clip.
            asset: Its file.
            frames: Frames of picture, run-up included.
            fps: Output frame rate.
            box: `(width, height)` it fills.
            framing: Where its crop sits.
            cache_dir: Where cached pictures live, or None to read the source.

        Returns:
            `(filters, input_index, cached)` — the chain reading its input,
            without an output label, the input it reads, and whether that
            input is a cached picture rather than the source, which has no
            sound in it.
        """
        args = _input_args(clip, asset, frames, fps)
        chain = _picture_chain(clip, frames, box[0], box[1], fps, framing)
        index = len(inputs)
        if cache_dir is None:
            inputs.append(args)
            return f"[{index}:v]{chain}", index, False
        stat = os.stat(asset.path)
        recipe = json.dumps([PICTURE_CACHE_VERSION, args, chain, stat.st_size, stat.st_mtime_ns])
        path = os.path.join(cache_dir, hashlib.sha256(recipe.encode("utf-8")).hexdigest()[:32] + ".mp4")
        if not os.path.exists(path) and not any(piece["path"] == path for piece in pieces):
            # Written under a name of its own and moved into place only once it is whole,
            # so a render stopped halfway never leaves a truncated picture to be reused.
            part = f"{path[:-4]}.{uuid.uuid4().hex[:8]}.part.mp4"
            rate = f"{fps.numerator}/{fps.denominator}"
            pieces.append({
                "command": [
                    self.ffmpeg_bin, "-y", "-loglevel", "error", "-nostats", *args,
                    "-filter_complex", f"[0:v]{chain}[v]", "-map", "[v]", "-an", "-r", rate,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", part,
                ],
                "part": part,
                "path": path,
                "duration_seconds": float(Fraction(frames) / fps),
            })
        inputs.append(["-i", path])
        rate = f"{fps.numerator}/{fps.denominator}"
        return (
            f"[{index}:v]setpts=PTS-STARTPTS,fps={rate},"
            f"tpad=stop_mode=clone:stop_duration=1,trim=end_frame={frames},setpts=PTS-STARTPTS",
            index,
            True,
        )

    def plan_segments(self, project: Project) -> List[Segment]:
        """Lay out the video track as consecutive clip and gap segments.

        The video track defines the output: it starts at frame 0 and ends
        where the last video clip ends.

        Args:
            project: Project to lay out.

        Returns:
            Segments covering the output without holes; empty if the project
            has no video track or no video clips.

        Raises:
            ValueError: If a clip overlaps the previous one or is shorter than
                one frame.
        """
        track = project.base_video_track
        return _layout(track.clips, Fraction(project.fps_num, project.fps_den)) if track else []

    def build_incremental(
        self,
        project: Project,
        assets: Mapping[str, Asset],
        output_path: str,
        cache_dir: str,
        loudness_target: Optional[float] = DEFAULT_LOUDNESS_TARGET,
        subtitle_path: Optional[str] = None,
        framing: Optional[Mapping[str, Framing]] = None,
        chapters_path: Optional[str] = None,
        voices: Optional[Mapping[str, float]] = None,
        talking: Optional[Talking] = None,
    ) -> Tuple[List[str], List[Dict[str, object]]]:
        """Build a preview render that only renders the pictures it has not rendered before.

        Every clip's picture is taken from the cache when it is there and
        rendered into it when it is not; the render itself then only puts the
        cached pictures together, lays anything over them, and mixes the
        sound — which is always mixed afresh, because it is cheap and because
        ducking and loudness depend on everything at once. Changing one cut
        re-renders the one or two pictures it touched.

        Args:
            project: Project to render, at the size the preview is wanted at.
            assets: Registered assets, keyed by asset ID.
            output_path: Destination path of the rendered file.
            cache_dir: Where cached pictures live.
            loudness_target: As for `build_command`.
            subtitle_path: As for `build_command`.
            framing: As for `build_command`.
            chapters_path: As for `build_command`.
            voices: As for `build_command`.
            talking: As for `build_command`.

        Returns:
            `(command, pieces)` — the render, and the pictures to render into
            the cache first, each a dictionary with its `command`, the `part`
            file it writes, the `path` to move that to once it is finished,
            and its `duration_seconds`. Empty when everything was cached.
        """
        return self._build(
            project, assets, output_path, True, loudness_target, subtitle_path, framing, chapters_path, cache_dir,
            voices=voices, talking=talking,
        )

    def build_sound(
        self,
        project: Project,
        assets: Mapping[str, Asset],
        mix_path: str,
        voice_path: str,
        music_path: str,
        loudness_target: Optional[float] = DEFAULT_LOUDNESS_TARGET,
        voices: Optional[Mapping[str, float]] = None,
        talking: Optional[Talking] = None,
    ) -> List[str]:
        """Build a render of the sound alone, with its two halves beside it.

        The mix is exactly the one a full render would carry — the same
        placement, ducking and loudness — so listening to it is hearing the
        cut. The two stems are what that mix is made of, kept apart so they
        can be measured against each other: the footage's own sound, and the
        audio tracks after ducking. No picture is decoded, so this takes a
        small part of what a render does.

        Args:
            project: Project whose sound is rendered.
            assets: Registered assets, keyed by asset ID.
            mix_path: Where the mix goes, as AAC.
            voice_path: Where the footage's sound goes, as mono WAV.
            music_path: Where the audio tracks go, as mono WAV.
            loudness_target: As for `build_command`.

        Returns:
            The command.
        """
        return self._build(
            project, assets, mix_path, False, loudness_target, None, None, None, None, (voice_path, music_path),
            voices, talking,
        )[0]

    def build_mix(
        self,
        project: Project,
        assets: Mapping[str, Asset],
        mix_path: str,
        voices: Optional[Mapping[str, float]] = None,
        talking: Optional[Talking] = None,
    ) -> List[str]:
        """Build a render of the mix alone, before its loudness is set.

        What `loudness.settle_gain` measures to find the gain a render's
        graph is waiting for: the same placement and ducking, no picture.

        Args:
            project: Project whose sound is rendered.
            assets: Registered assets, keyed by asset ID.
            mix_path: Where the mix goes, as 32-bit float WAV.
            voices: As for `build_command`.
            talking: As for `build_command`.

        Returns:
            The command.
        """
        return self._build(project, assets, mix_path, False, None, None, None, None, None, (), voices, talking)[0]

    def build_command(
        self,
        project: Project,
        assets: Mapping[str, Asset],
        output_path: str,
        is_preview: bool = False,
        loudness_target: Optional[float] = DEFAULT_LOUDNESS_TARGET,
        subtitle_path: Optional[str] = None,
        framing: Optional[Mapping[str, Framing]] = None,
        chapters_path: Optional[str] = None,
        voices: Optional[Mapping[str, float]] = None,
        talking: Optional[Talking] = None,
    ) -> List[str]:
        """Build the FFmpeg arguments that render a project to a file.

        Every video clip is placed at its timeline position, and gaps are
        filled with black frames and silence. Each source is normalized to
        the project format: scaled to cover `width` x `height` and
        cropped — from the middle, or where `framing` says, converted to the project frame rate, and resampled to
        48 kHz stereo; sources without audio contribute silence. Each audio
        track is laid out the same way, cut or padded to the video length,
        and mixed with the video's own audio at the clips' volumes and fades.
        A track marked `duck_under_speech` is compressed against the video's
        own sound, so music drops while someone is talking; a narration on an
        audio track named in `voices` counts as talking too, and the video's
        own sound drops under it the same way. The finished
        mix is normalized to `loudness_target`, so files rendered from
        different footage all come out at the same level. Video frame counts and audio
        sample counts are all derived from the same absolute frame
        boundaries, so the streams stay in sync without drift. Previews are
        then scaled to 480p and encoded for speed.

        Args:
            project: Project whose timeline is rendered.
            assets: Registered assets, keyed by asset ID.
            output_path: Destination path of the rendered file.
            is_preview: Whether to render a fast, low-resolution preview.
            loudness_target: Integrated loudness of the finished mix in LUFS.
                The graph then carries `loudness.GAIN_TOKEN` in place of the
                gain, which `loudness.settle_gain` finds on the `build_mix`
                render and `loudness.with_gain` writes in;
                None leaves the levels exactly as mixed.
            subtitle_path: ASS subtitle file to burn into the picture, sized
                for this project's output format; None burns nothing.
            framing: Where each clip's crop sits, keyed by clip ID, from
                `reframe.frame_project`. A clip without one is cropped from
                the middle.
            chapters_path: An FFmpeg metadata file of chapters to carry in the
                output, from `delivery.chapter_metadata`; None carries none.
            voices: The audio tracks that carry a voice recorded apart from
                the picture, from `voice_keys`, with the gain that brings each
                one's speech to a known level.
                While one speaks, the footage's own sound drops, and so does
                any track marked `duck_under_speech`. None treats no track as
                a voice.
            talking: Where the transcripts say somebody is talking, and the
                gain that brings each music track to the level of it. Given
                it, music marked `duck_under_speech` drops `DUCK_DEPTH_DB`
                wherever somebody is talking, however loud they are; without
                it, music ducks under the level of the footage's own sound.

        Returns:
            The command as an argument list.

        Raises:
            ValueError: If the project uses an unsupported feature, has no
                video clips, or has overlapping or sub-frame clips.
        """
        return self._build(
            project, assets, output_path, is_preview, loudness_target, subtitle_path, framing, chapters_path, None,
            voices=voices, talking=talking,
        )[0]

    def _build(
        self,
        project: Project,
        assets: Mapping[str, Asset],
        output_path: str,
        is_preview: bool,
        loudness_target: Optional[float],
        subtitle_path: Optional[str],
        framing: Optional[Mapping[str, Framing]],
        chapters_path: Optional[str],
        cache_dir: Optional[str],
        stems: Optional[Tuple[str, str]] = None,
        voices: Optional[Mapping[str, float]] = None,
        talking: Optional[Talking] = None,
    ) -> Tuple[List[str], List[Dict[str, object]]]:
        """Build a render, reading pictures from their sources or from a cache.

        Args:
            project: Project whose timeline is rendered.
            assets: Registered assets, keyed by asset ID.
            output_path: Destination path of the rendered file.
            is_preview: Whether to encode for speed.
            loudness_target: Integrated loudness of the finished mix in LUFS.
            subtitle_path: ASS subtitle file to burn in, or None.
            framing: Where each clip's crop sits, keyed by clip ID.
            chapters_path: FFmpeg metadata file of chapters, or None.
            cache_dir: Where cached pictures live, or None to read every
                picture from its source.
            stems: `(voice_path, music_path)` to render the sound alone: the
                mix to `output_path`, and beside it what the footage says and
                what the audio tracks play after ducking, each as its own
                file. No picture is decoded at all.
            voices: The audio tracks that are a voice recorded apart from
                the picture, from `voice_keys`: the footage's sound and the
                music duck under them.
            talking: As for `build_command`.

        Returns:
            `(command, pieces)`; `pieces` is always empty without a cache.
        """
        sound_only = stems is not None
        speech = talking.spans if talking is not None and talking.spans else None
        music_gains = talking.music_gains if talking is not None else {}
        clip_gains = talking.clip_gains if talking is not None else {}
        self.check_supported(project, assets)
        segments = self.plan_segments(project)
        if not segments:
            raise ValueError("the project has no clips on a video track to render")

        fps = Fraction(project.fps_num, project.fps_den)
        rate = f"{project.fps_num}/{project.fps_den}"
        size = f"{project.width}x{project.height}"
        framed = framing or {}
        total_frames = segments[-1].end_frame
        total_samples = _frame_to_sample(total_frames, fps)

        inputs: List[List[str]] = []
        pieces: List[Dict[str, object]] = []
        filters: List[str] = []
        video_labels: List[str] = []
        audio_labels: List[str] = []
        # A run is a stretch of straight cuts, which concatenates. Runs are separated by
        # transitions, which do not: each one runs over the run before it.
        runs: List[List[int]] = [[]]
        transitions: List[Tuple[Transition, int]] = []

        for index, segment in enumerate(segments):
            frames = segment.end_frame - segment.start_frame
            samples = _frame_to_sample(segment.end_frame, fps) - _frame_to_sample(segment.start_frame, fps)

            # A transition needs a clip ending exactly where this one starts to run over.
            # validate_project has already refused anything else; this is what renders it.
            # The run-up is what this clip carries, which for a dip is only its second
            # half — the colour covers the first.
            run_up = 0
            if segment.clip is not None and segment.clip.transition_in is not None and index > 0:
                run_up = _round_half_up(Fraction(segment.clip.run_up) * fps)
            if run_up and runs[-1]:
                runs.append([index])
                transitions.append((segment.clip.transition_in, run_up))
            else:
                run_up = 0
                runs[-1].append(index)

            if segment.clip is None:
                if not sound_only:
                    filters.append(f"color=c=black:s={size}:r={rate},format=yuv420p,trim=end_frame={frames}[v{index}]")
            else:
                asset = assets[segment.clip.asset_id]
                picture_frames = frames + run_up
                if sound_only:
                    # With no picture there is nothing to share an input with.
                    input_index, cached = -1, True
                else:
                    picture, input_index, cached = self._picture(
                        inputs, pieces, segment.clip, asset, picture_frames, fps,
                        (project.width, project.height), framed.get(segment.clip.id), cache_dir,
                    )
                    filters.append(f"{picture}[v{index}]")
                if asset.has_audio:
                    clip = segment.clip
                    lead, lag = _seconds_to_samples(clip.audio_lead), _seconds_to_samples(clip.audio_lag)
                    # The picture's input starts early by any run-up a transition needs, and
                    # the sound must not: read from there it would run late by exactly the
                    # length of the transition. So a clip with a run-up reads its sound on
                    # its own, from its own in point — as does one whose picture is cached,
                    # since a cached picture has no sound in it.
                    if lead or lag or run_up or cached:
                        audio_index = len(inputs)
                        inputs.append(_audio_input_args(clip, asset, fps))
                    else:
                        audio_index = input_index
                    at = _frame_to_sample(segment.start_frame, fps) - lead
                    filters.append(
                        _clip_audio_filter(
                            f"[{audio_index}:a]", clip, samples + lead + lag, f"[araw{index}]",
                            clip_gains.get(clip.id, 0.0),
                        )
                    )
                    placed = f"[a{index}]"
                    if at:
                        filters.append(f"[araw{index}]adelay={at}S:all=1{placed}")
                    else:
                        filters.append(f"[araw{index}]anull{placed}")
                    audio_labels.append(placed)

            video_labels.append(f"[v{index}]")

        # The picture is concatenated; the sound is placed and mixed. Concatenating it too
        # would put every clip's sound strictly after the one before, which is exactly what a
        # J or an L cut is not. Each clip's sound is delayed to its own absolute position
        # instead, so nothing accumulates across a long timeline either.
        joined = "[joinedv]"
        run_labels: List[str] = []
        for run_index, run in enumerate(runs if not sound_only else []):
            labels = [video_labels[index] for index in run]
            if len(labels) == 1:
                run_labels.append(labels[0])
                continue
            run_label = f"[run{run_index}]"
            filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0{run_label}")
            run_labels.append(run_label)

        # Each run after the first arrives carrying its own run-up, so the transition that
        # eats the run-up leaves the timeline exactly as long as it was. The transition ends
        # on the cut rather than straddling it, which is what keeps every clip where it is.
        timebase = f"settb={project.fps_den}/{project.fps_num}"
        if sound_only:
            run_labels = []
        if len(run_labels) > 1:
            # xfade refuses two inputs whose timebases differ, and they do: a run of one
            # segment carries the frame rate's, a concatenated run carries the muxer's.
            # Both are put on the frame rate's before they meet.
            for run_index, run_label in enumerate(run_labels):
                settled = f"[tb{run_index}]"
                filters.append(f"{run_label}{timebase}{settled}")
                run_labels[run_index] = settled

        carried = run_labels[0] if run_labels else ""
        covered = sum(segments[index].end_frame - segments[index].start_frame for index in runs[0])
        for run_index in range(1, len(run_labels)):
            transition, run_up = transitions[run_index - 1]
            label = joined if run_index == len(run_labels) - 1 else f"[mixed{run_index}]"
            if isinstance(transition, Dip):
                # Two mixes with a colour between them. The colour runs the whole length
                # of the dip, the outgoing picture fades into it over the first half, and
                # the incoming rises out of it over the second — which is the half the
                # incoming clip carried its run-up for. The first mix hands on something
                # exactly as long as the timeline was, so the second mix's offset is the
                # same arithmetic a one-mix transition uses.
                whole = _round_half_up(Fraction(transition.seconds) * fps)
                colour, dipped = f"[dip{run_index}]", f"[dipped{run_index}]"
                filters.append(
                    f"color=c={_colour_source(transition.through)}:s={size}:r={rate},format=yuv420p,"
                    f"trim=end_frame={whole},setpts=PTS-STARTPTS,{timebase}{colour}"
                )
                filters.append(
                    f"{carried}{colour}xfade=transition=fade"
                    f":duration={_format_seconds(Fraction(whole - run_up) / fps)}"
                    f":offset={_format_seconds(Fraction(covered - whole) / fps)}{dipped}"
                )
                carried = dipped
            filters.append(
                f"{carried}{run_labels[run_index]}xfade=transition={_xfade_name(transition)}"
                f":duration={_format_seconds(Fraction(run_up) / fps)}"
                f":offset={_format_seconds(Fraction(covered - run_up) / fps)}{label}"
            )
            carried = label
            covered += sum(segments[index].end_frame - segments[index].start_frame for index in runs[run_index])
        if len(run_labels) == 1:
            filters.append(f"{carried}null{joined}")
        if not audio_labels:
            filters.append(_silence_filter(total_samples, "[mainaudio]"))
        else:
            mixed = audio_labels[0] if len(audio_labels) == 1 else "[laid]"
            if len(audio_labels) > 1:
                # normalize=0 sums rather than dividing by the number of inputs, so a clip
                # playing alone keeps its own level and an overlap is the two of them together.
                filters.append(
                    f"{''.join(audio_labels)}amix=inputs={len(audio_labels)}"
                    f":duration=longest:normalize=0{mixed}"
                )
            filters.append(
                f"{mixed}apad=whole_len={total_samples},atrim=end_sample={total_samples}[mainaudio]"
            )
        # Video tracks above the first are drawn on top of it, each clip inside the box its layout gives.
        overlay_audio: List[str] = []
        for track_index, track in enumerate(project.video_tracks[1:]):
            track_labels: List[str] = []
            for segment_index, segment in enumerate(_layout(track.clips, fps)):
                if segment.start_frame >= total_frames:
                    break
                end_frame = min(segment.end_frame, total_frames)
                frames = end_frame - segment.start_frame
                samples = _frame_to_sample(end_frame, fps) - _frame_to_sample(segment.start_frame, fps)
                audio_label = f"[o{track_index}s{segment_index}]"
                if segment.clip is None:
                    filters.append(_silence_filter(samples, audio_label))
                    track_labels.append(audio_label)
                    continue

                clip = segment.clip
                asset = assets[clip.asset_id]
                if sound_only:
                    input_index, cached = -1, True
                else:
                    box_x, box_y, box_width, box_height = overlay_box(clip, project.width, project.height)
                    over = f"[ov{track_index}_{segment_index}]"
                    picture, input_index, cached = self._picture(
                        inputs, pieces, clip, asset, frames, fps, (box_width, box_height), framed.get(clip.id),
                        cache_dir,
                    )
                    # Hold the overlay back to its place on the timeline, then stop drawing when it runs out.
                    filters.append(f"{picture},setpts=PTS+{segment.start_frame}/({rate})/TB{over}")
                    composited = f"[ovr{track_index}_{segment_index}]"
                    filters.append(
                        f"{joined}{over}overlay=x={box_x}:y={box_y}:eof_action=pass:repeatlast=0{composited}"
                    )
                    joined = composited
                if cached and asset.has_audio:
                    input_index = len(inputs)
                    inputs.append(_input_args(clip, asset, frames, fps))

                if asset.has_audio:
                    filters.append(_clip_audio_filter(f"[{input_index}:a]", clip, samples, audio_label))
                else:
                    filters.append(_silence_filter(samples, audio_label))
                track_labels.append(audio_label)

            if track_labels:
                track_label = f"[ot{track_index}]"
                filters.append(
                    f"{''.join(track_labels)}concat=n={len(track_labels)}:v=0:a=1,"
                    f"apad=whole_len={total_samples},atrim=end_sample={total_samples}{track_label}"
                )
                overlay_audio.append(track_label)

        if sound_only:
            pass
        elif subtitle_path is not None:
            # Burn before any preview downscale, so the ASS file's own resolution matches the picture.
            filters.append(f"{joined}subtitles=filename='{escape_filter_path(subtitle_path)}'[subbedv]")
            joined = "[subbedv]"
        if sound_only:
            pass
        elif is_preview and cache_dir is None:
            # A cached preview is rendered at its own small size from the start.
            filters.append(f"{joined}scale=-2:480[outv]")
        else:
            filters.append(f"{joined}null[outv]")

        mix_labels = ["[mainaudio]", *overlay_audio]
        ducking: List[str] = []
        narration: List[Tuple[str, float]] = []
        audio_tracks = [track for track in project.tracks if track.track_type == TrackType.AUDIO]
        for track_index, track in enumerate(audio_tracks):
            track_labels: List[str] = []
            for segment_index, segment in enumerate(_layout(track.clips, fps)):
                if segment.start_frame >= total_frames:
                    break
                label = f"[t{track_index}s{segment_index}]"
                samples = _frame_to_sample(segment.end_frame, fps) - _frame_to_sample(segment.start_frame, fps)
                if segment.clip is None:
                    filters.append(_silence_filter(samples, label))
                else:
                    input_index = len(inputs)
                    frames = segment.end_frame - segment.start_frame
                    inputs.append(_input_args(segment.clip, assets[segment.clip.asset_id], frames, fps))
                    filters.append(_clip_audio_filter(
                        f"[{input_index}:a]", segment.clip, samples, label,
                        # Brought to where the talking is, so a clip's volume says how far under
                        # the voices it sits rather than how far under its own mastering.
                        music_gains.get(track.id, {}).get(segment.clip.id, 0.0),
                    ))
                track_labels.append(label)
            if track_labels:
                track_label = f"[t{track_index}]"
                filters.append(
                    f"{''.join(track_labels)}concat=n={len(track_labels)}:v=0:a=1,"
                    f"apad=whole_len={total_samples},atrim=end_sample={total_samples}{track_label}"
                )
                if voices and track.id in voices:
                    narration.append((track_label, voices[track.id]))
                elif track.duck_under_speech:
                    ducking.append(track_label)
                mix_labels.append(track_label)

        # The footage's own sound, and anything laid over the picture, come first in the mix;
        # they are what a narration is spoken over.
        footage = 1 + len(overlay_audio)
        # What music ducks under when nothing says where the talking is: the footage's own
        # sound, joined by any narration.
        heard = "[mainaudio]"
        spoken = set(range(footage))
        if narration:
            # Every narration is brought to the same level first, which is what it is heard
            # at and what it ducks the place with, so a quiet recording ends up over the
            # place as far as a loud one does.
            keys: List[str] = []
            for index, (label, gain) in enumerate(narration):
                kept, key = f"[narr{index}]", f"[narrkey{index}]"
                filters.append(f"{label}volume={gain:g}dB,asplit=2{kept}{key}")
                mix_labels[mix_labels.index(label)] = kept
                keys.append(key)
            spoken = {mix_labels.index(f"[narr{index}]") for index in range(len(narration))}
            copies = [f"[vkey{index}]" for index in range(footage + (1 if ducking and speech is None else 0))]
            joined = keys[0] if len(keys) == 1 else f"{''.join(keys)}amix=inputs={len(keys)}:duration=first:normalize=0,"
            filters.append(f"{joined}asplit={len(copies)}{''.join(copies)}")
            if ducking and speech is None:
                # The footage's own speech still ducks the music, so a copy is kept before it is lowered.
                filters.append("[mainaudio]asplit=2[mainkeep][mainspeech]")
                mix_labels[0] = "[mainkeep]"
            for index in range(footage):
                lowered = f"[under{index}]"
                filters.append(
                    f"{mix_labels[index]}{copies[index]}sidechaincompress="
                    f"threshold={VOICE_DUCK_THRESHOLD}:ratio={VOICE_DUCK_RATIO}:"
                    f"attack={VOICE_DUCK_ATTACK_MS}:release={VOICE_DUCK_RELEASE_MS}{lowered}"
                )
                mix_labels[index] = lowered
            if ducking and speech is None:
                filters.append(f"[mainspeech]{copies[-1]}amix=inputs=2:duration=first:normalize=0[allspeech]")
                heard = "[allspeech]"

        if ducking and speech is not None:
            # Where the transcripts say somebody is talking, the music drops by a set amount,
            # however loud they were recorded or were turned: nothing about the level of the
            # voice decides whether the music gets out of its way.
            envelope = _duck_volume(speech)
            for label in ducking:
                ducked_label = f"[{label.strip('[]')}duck]"
                filters.append(f"{label}{envelope}{ducked_label}")
                mix_labels[mix_labels.index(label)] = ducked_label
        elif ducking:
            # sidechaincompress consumes its sidechain, so the speech needs one copy per ducked
            # track, and one more to mix when it is the footage's sound and nothing else took it.
            copies = [f"[speech{index}]" for index in range(len(ducking))]
            if heard == "[mainaudio]":
                filters.append(f"[mainaudio]asplit={len(copies) + 1}[speechmix]{''.join(copies)}")
                mix_labels[0] = "[speechmix]"
            else:
                filters.append(f"{heard}asplit={len(copies)}{''.join(copies)}")
            for label, copy in zip(ducking, copies):
                ducked_label = f"[{label.strip('[]')}duck]"
                filters.append(
                    f"{label}{copy}sidechaincompress="
                    f"threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}:"
                    f"attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}{ducked_label}"
                )
                mix_labels[mix_labels.index(label)] = ducked_label

        if sound_only and stems:
            # Every lane is split, one copy to the mix and one to its stem: who is talking on
            # one side — the footage's own sound, or the narration when there is one — and
            # everything else as ducked on the other, which with a narration includes the
            # place it was spoken over.
            lanes = {"voice": [], "music": []}
            for index, label in enumerate(mix_labels):
                kept, stem = f"[mixed{index}]", f"[stem{index}]"
                filters.append(f"{label}asplit=2{kept}{stem}")
                mix_labels[index] = kept
                lanes["voice" if index in spoken else "music"].append(stem)
            for lane, labels in lanes.items():
                if not labels:
                    filters.append(_silence_filter(total_samples, f"[{lane}]"))
                elif len(labels) == 1:
                    filters.append(f"{labels[0]}anull[{lane}]")
                else:
                    filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0[{lane}]")
        if len(mix_labels) == 1:
            filters.append(f"{mix_labels[0]}anull[mixa]")
        else:
            # normalize=0 keeps every input at its own volume instead of dividing by the number of inputs.
            filters.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:normalize=0[mixa]")

        if loudness_target is None:
            filters.append("[mixa]anull[outa]")
        else:
            # The gain is not known yet: the mix is measured on its own first, and the gain
            # that reaches the target is written in over the token (see `loudness`). The
            # limiter oversamples, so come back to the exact length afterwards; the guard
            # sits before the format is settled for the encoder.
            filters.append(
                f"[mixa]{loudness.chain(loudness_target, loudness.GAIN_TOKEN)},{SILENCE_GUARD},"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"apad=whole_len={total_samples},atrim=end_sample={total_samples}[outa]"
            )

        # Log errors only and suppress console stats; render workers read progress through -progress instead.
        cmd = [self.ffmpeg_bin, "-y", "-loglevel", "error", "-nostats"]
        # Every input is opened at once and stays open for the whole render, so on a long
        # timeline the decoders, not the encoder, are what fills memory. Past a handful of
        # them each one is held to a single thread, which keeps one decoded picture in
        # flight per input instead of one per core.
        threads = resources.decoder_threads(len(inputs))
        for input_args in inputs:
            if threads:
                cmd.extend(["-threads", str(threads)])
            cmd.extend(input_args)
        if chapters_path is not None:
            # Read last, after every media input, so it takes no part in the graph.
            cmd.extend(["-f", "ffmetadata", "-i", chapters_path])
        if sound_only:
            # The mix to listen to, and the two stems to measure: mono and at a low rate,
            # since nothing listens to them.
            # A mix to be measured is kept exact, peaks over full scale and all.
            codec = ["-c:a", "aac", "-b:a", "128k"] if stems else ["-c:a", "pcm_f32le"]
            cmd.extend(["-filter_complex", ";".join(filters), "-map", "[outa]", *codec, output_path])
            for lane, path in zip(("voice", "music"), stems):
                cmd.extend(["-map", f"[{lane}]", "-ac", "1", "-ar", str(STEM_SAMPLE_RATE), "-c:a", "pcm_s16le", path])
            return cmd, pieces
        cmd.extend(["-filter_complex", ";".join(filters), "-map", "[outv]", "-map", "[outa]"])
        if chapters_path is not None:
            cmd.extend(["-map_chapters", str(len(inputs))])

        if is_preview:
            cmd.extend([
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-b:a", "96k"
            ])
        else:
            cmd.extend([
                "-r", rate,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                # At 192k the encoder's own ringing put hard transients up to 2 dB over the
                # limiter, depending only on where they fell in its frames; at 256k they held
                # at 2.5 dB under full scale however they fell.
                "-c:a", "aac", "-b:a", "256k"
            ])

        cmd.append(output_path)
        return cmd, pieces

    def output_duration(self, project: Project) -> Fraction:
        """Compute the exact duration of the rendered output.

        Args:
            project: Project to measure.

        Returns:
            The duration in seconds, from the start of the timeline to the end
            of the last clip after frame snapping; zero if there are no clips.

        Raises:
            ValueError: If the timeline cannot be laid out into frames.
        """
        segments = self.plan_segments(project)
        if not segments:
            return Fraction(0)
        return Fraction(segments[-1].end_frame * project.fps_den, project.fps_num)
