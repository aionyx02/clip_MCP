import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import List, Mapping, Optional, Tuple
from app.engine import resources
from app.models.media import Asset
from app.models.timeline import Clip, Project, TrackType

AUDIO_SAMPLE_RATE = 48000
# Streaming platforms normalize to about -14 LUFS, so delivering at that level avoids being turned down.
DEFAULT_LOUDNESS_TARGET = -14.0
LOUDNESS_TRUE_PEAK = -1.5
LOUDNESS_RANGE = 11.0
# Ducking: hold the music well under speech, and come back slowly so it does not pump between words.
DUCK_THRESHOLD = 0.03
DUCK_RATIO = 10
DUCK_ATTACK_MS = 20
DUCK_RELEASE_MS = 500
# loudnorm derives its gain from the measured loudness of what it is given. A mix that
# is silent all the way through measures as -inf LUFS, so that gain comes out infinite
# and fills the stream with NaN; the AAC encoder then refuses the frame and the whole
# render fails. That happens whenever nothing on the timeline has sound — footage shot
# on a muted microphone, or a cut with no audio track at all. Replacing NaN with zero
# hands the encoder the silence it was actually given, and leaves every real mix
# untouched. Clamping the samples or converting them to integers instead is not an
# alternative: both turn the NaN into full-scale noise. Two expressions, because
# everything reaching the mix has already been formatted to stereo.
SILENCE_GUARD = r"aeval=exprs=if(isnan(val(0))\,0\,val(0))|if(isnan(val(1))\,0\,val(1)):c=same"

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
        frames: Length of the clip on the output timeline, in frames.
        fps: Output frame rate.

    Returns:
        The `-ss`, `-t`, and `-i` options for the input.
    """
    return [
        "-ss", _format_seconds(Fraction(clip.source_range.start)),
        "-t", _format_seconds(Fraction(frames + 1) / fps),
        "-i", asset.path,
    ]

def _overlay_box(clip: Clip, width: int, height: int) -> Tuple[int, int, int, int]:
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

def _escape_filter_path(path: str) -> str:
    """Make a file path safe to put inside a quoted filtergraph argument.

    A path breaks a filtergraph three times over: on Windows the backslashes
    are escapes and the drive letter's colon separates filter options, and an
    apostrophe anywhere in the path closes the quoted section, which then
    leaves any comma or bracket in it to the graph parser.

    Args:
        path: Path to a file the filtergraph reads, written between single
            quotes at the call site.

    Returns:
        The path with forward slashes, an escaped colon, and each apostrophe
        quoted out and escaped for both the filter and the graph.
    """
    return path.replace("\\", "/").replace("'", r"'\\\''").replace(":", r"\:")

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

def _clip_audio_filter(input_label: str, clip: Clip, samples: int, output_label: str) -> str:
    """Build the filter chain that turns a clip's source audio into its output audio.

    The audio is resampled to 48 kHz stereo, padded or trimmed to exactly
    `samples`, and then has the clip's volume and fades applied.

    Args:
        input_label: Filtergraph label of the source audio, such as `[2:a]`.
        clip: Clip providing volume and fade settings.
        samples: Exact number of output samples.
        output_label: Filtergraph label of the result.

    Returns:
        The filter chain.
    """
    chain = [
        "asetpts=PTS-STARTPTS",
        f"aresample={AUDIO_SAMPLE_RATE}",
        "aformat=sample_fmts=fltp:channel_layouts=stereo",
        f"apad=whole_len={samples}",
        f"atrim=end_sample={samples}",
    ]
    if clip.volume != 1.0:
        chain.append(f"volume={clip.volume:g}")
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
        `layout` gives. Every clip plays at normal speed. Video-track clips
        need assets with a video stream, audio-track clips need assets with an
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
                if clip.speed != 1.0:
                    raise ValueError(f"clip {clip.id}: speed other than 1.0 is not supported yet")
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

    def build_command(
        self,
        project: Project,
        assets: Mapping[str, Asset],
        output_path: str,
        is_preview: bool = False,
        loudness_target: Optional[float] = DEFAULT_LOUDNESS_TARGET,
        subtitle_path: Optional[str] = None,
    ) -> List[str]:
        """Build the FFmpeg arguments that render a project to a file.

        Every video clip is placed at its timeline position, and gaps are
        filled with black frames and silence. Each source is normalized to
        the project format: scaled to cover `width` x `height` and
        center-cropped, converted to the project frame rate, and resampled to
        48 kHz stereo; sources without audio contribute silence. Each audio
        track is laid out the same way, cut or padded to the video length,
        and mixed with the video's own audio at the clips' volumes and fades.
        A track marked `duck_under_speech` is compressed against the video's
        own sound, so music drops while someone is talking, and the finished
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
            loudness_target: Integrated loudness of the finished mix in LUFS;
                None leaves the levels exactly as mixed.
            subtitle_path: ASS subtitle file to burn into the picture, sized
                for this project's output format; None burns nothing.

        Returns:
            The command as an argument list.

        Raises:
            ValueError: If the project uses an unsupported feature, has no
                video clips, or has overlapping or sub-frame clips.
        """
        self.check_supported(project, assets)
        segments = self.plan_segments(project)
        if not segments:
            raise ValueError("the project has no clips on a video track to render")

        fps = Fraction(project.fps_num, project.fps_den)
        rate = f"{project.fps_num}/{project.fps_den}"
        size = f"{project.width}x{project.height}"
        total_frames = segments[-1].end_frame
        total_samples = _frame_to_sample(total_frames, fps)

        inputs: List[List[str]] = []
        filters: List[str] = []
        video_labels: List[str] = []
        audio_labels: List[str] = []

        for index, segment in enumerate(segments):
            frames = segment.end_frame - segment.start_frame
            samples = _frame_to_sample(segment.end_frame, fps) - _frame_to_sample(segment.start_frame, fps)

            if segment.clip is None:
                filters.append(f"color=c=black:s={size}:r={rate},format=yuv420p,trim=end_frame={frames}[v{index}]")
                filters.append(_silence_filter(samples, f"[a{index}]"))
            else:
                asset = assets[segment.clip.asset_id]
                input_index = len(inputs)
                inputs.append(_input_args(segment.clip, asset, frames, fps))
                filters.append(
                    f"[{input_index}:v]setpts=PTS-STARTPTS,fps={rate},"
                    f"scale={project.width}:{project.height}:force_original_aspect_ratio=increase,"
                    f"crop={project.width}:{project.height},setsar=1,format=yuv420p,"
                    f"tpad=stop_mode=clone:stop_duration=1,trim=end_frame={frames},setpts=PTS-STARTPTS"
                    f"{_clip_video_filter(segment.clip, frames, fps)}[v{index}]"
                )
                if asset.has_audio:
                    filters.append(_clip_audio_filter(f"[{input_index}:a]", segment.clip, samples, f"[a{index}]"))
                else:
                    filters.append(_silence_filter(samples, f"[a{index}]"))

            video_labels.append(f"[v{index}]")
            audio_labels.append(f"[a{index}]")

        # Concatenate video and audio separately: a joint concat pads each segment's audio up to its
        # video duration, which would add sub-frame drift that accumulates across segments.
        joined = "[joinedv]"
        filters.append(f"{''.join(video_labels)}concat=n={len(segments)}:v=1:a=0{joined}")
        filters.append(f"{''.join(audio_labels)}concat=n={len(segments)}:v=0:a=1[mainaudio]")
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
                input_index = len(inputs)
                inputs.append(_input_args(clip, asset, frames, fps))
                box_x, box_y, box_width, box_height = _overlay_box(clip, project.width, project.height)
                over = f"[ov{track_index}_{segment_index}]"
                filters.append(
                    f"[{input_index}:v]setpts=PTS-STARTPTS,fps={rate},"
                    f"scale={box_width}:{box_height}:force_original_aspect_ratio=increase,"
                    f"crop={box_width}:{box_height},setsar=1,format=yuv420p,"
                    f"tpad=stop_mode=clone:stop_duration=1,trim=end_frame={frames},setpts=PTS-STARTPTS"
                    f"{_clip_video_filter(clip, frames, fps)},"
                    # Hold the overlay back to its place on the timeline, then stop drawing when it runs out.
                    f"setpts=PTS+{segment.start_frame}/({rate})/TB{over}"
                )
                composited = f"[ovr{track_index}_{segment_index}]"
                filters.append(
                    f"{joined}{over}overlay=x={box_x}:y={box_y}:eof_action=pass:repeatlast=0{composited}"
                )
                joined = composited

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

        if subtitle_path is not None:
            # Burn before any preview downscale, so the ASS file's own resolution matches the picture.
            filters.append(f"{joined}subtitles=filename='{_escape_filter_path(subtitle_path)}'[subbedv]")
            joined = "[subbedv]"
        if is_preview:
            filters.append(f"{joined}scale=-2:480[outv]")
        else:
            filters.append(f"{joined}null[outv]")

        mix_labels = ["[mainaudio]", *overlay_audio]
        ducking: List[str] = []
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
                    filters.append(_clip_audio_filter(f"[{input_index}:a]", segment.clip, samples, label))
                track_labels.append(label)
            if track_labels:
                track_label = f"[t{track_index}]"
                filters.append(
                    f"{''.join(track_labels)}concat=n={len(track_labels)}:v=0:a=1,"
                    f"apad=whole_len={total_samples},atrim=end_sample={total_samples}{track_label}"
                )
                mix_labels.append(track_label)
                if track.duck_under_speech:
                    ducking.append(track_label)

        if ducking:
            # sidechaincompress consumes its sidechain, so the speech needs one copy per ducked track plus one to mix.
            copies = [f"[speech{index}]" for index in range(len(ducking) + 1)]
            filters.append(f"[mainaudio]asplit={len(copies)}{''.join(copies)}")
            mix_labels[0] = copies[0]
            for index, label in enumerate(ducking):
                ducked_label = f"[{label.strip('[]')}duck]"
                filters.append(
                    f"{label}{copies[index + 1]}sidechaincompress="
                    f"threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}:"
                    f"attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}{ducked_label}"
                )
                mix_labels[mix_labels.index(label)] = ducked_label

        if len(mix_labels) == 1:
            filters.append(f"{mix_labels[0]}anull[mixa]")
        else:
            # normalize=0 keeps every input at its own volume instead of dividing by the number of inputs.
            filters.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:normalize=0[mixa]")

        if loudness_target is None:
            filters.append("[mixa]anull[outa]")
        else:
            # loudnorm resamples to 192 kHz internally, so come back to the project rate and to
            # the exact length. The guard sits after that, where there are a quarter as many
            # samples to look at, and before the format is settled for the encoder.
            filters.append(
                f"[mixa]loudnorm=I={loudness_target:g}:TP={LOUDNESS_TRUE_PEAK:g}:LRA={LOUDNESS_RANGE:g},"
                f"aresample={AUDIO_SAMPLE_RATE},{SILENCE_GUARD},"
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
        cmd.extend(["-filter_complex", ";".join(filters), "-map", "[outv]", "-map", "[outa]"])

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
                "-c:a", "aac", "-b:a", "192k"
            ])

        cmd.append(output_path)
        return cmd

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
