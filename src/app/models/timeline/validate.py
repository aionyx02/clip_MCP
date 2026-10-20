"""The checks a project must pass after every edit."""

from decimal import Decimal
from typing import Mapping, Optional

from app.models.media import Asset
from app.models.timeline.parts import TOUCHING_SECONDS, Clip, Project, TrackType


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
        if track.voice and track.track_type != TrackType.AUDIO:
            raise ValueError(
                f"track {track.id}: only an audio track can be a voice recorded apart from the picture; "
                "the sound of a video track is the footage's own"
            )
        if track.voice and track.duck_under_speech:
            raise ValueError(
                f"track {track.id}: a voice cannot duck under speech, since it is the speech; "
                "turn duck_under_speech off, or say the track is not a voice"
            )
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
            if clip.card is not None:
                if track is not base:
                    raise ValueError(f"clip {clip.id}: a title card goes on the main video track, not {track.id}")
                if clip.volume != 0:
                    raise ValueError(f"clip {clip.id}: a title card is silent; its sound cannot be turned up")
                if asset.duration is not None and clip.card.at > Decimal(str(asset.duration)) + TOUCHING_SECONDS:
                    raise ValueError(
                        f"clip {clip.id}: the card shows {clip.card.at}s of asset {asset.id}, "
                        f"which is only {asset.duration}s long"
                    )
            elif asset.duration is not None and clip.source_range.end > Decimal(str(asset.duration)) + TOUCHING_SECONDS:
                raise ValueError(
                    f"clip {clip.id}: source range ends at {clip.source_range.end}s, "
                    f"but asset {asset.id} is only {asset.duration}s long"
                )
            if clip.transition_in is not None:
                transition = clip.transition_in
                if previous is None or abs(previous.timeline_out - clip.timeline_in) > TOUCHING_SECONDS:
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
                # A title card and a photo hold one picture, which is there before the cut as much as after it.
                if clip.video_source_start < 0 and clip.card is None and not asset.still:
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
            if asset.duration is not None and clip.audio_source_end > Decimal(str(asset.duration)) + TOUCHING_SECONDS:
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
            if previous is not None and clip.timeline_in < previous.timeline_out - TOUCHING_SECONDS:
                raise ValueError(
                    f"clip {clip.id} starts at {clip.timeline_in}s and overlaps clip {previous.id}, "
                    f"which ends at {previous.timeline_out}s on track {track.id}"
                )
            previous = clip
