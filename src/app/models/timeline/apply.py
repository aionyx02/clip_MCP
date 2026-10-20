"""Applying an edit operation to a project, with the helpers that place clips and captions as it does."""

from decimal import Decimal
from typing import List, Mapping, Optional, Tuple

from app.models.media import Asset
from app.models.timeline.operations import (
    AddClipOp, AddSubtitleOp, AddTextOp, AddTitleCardOp, AddTrackOp, DeleteOp, EditOperation, EditSubtitleOp,
    FitTrackOp, InsertClipOp, MoveClipOp, NewClipSpec, RemoveTextOp, RenameProjectOp, ReorderClipOp, SetCaptionStyleOp,
    SetClipAudioOp, SetClipLookOp, SetClipPinnedOp, SetClipSpeedOp, SetMarkersOp, SetSubtitlesOp, SetTextOp,
    SetTitleCardOp, SetTrackAudioOp, SplitClipOp, TextTimes, TrimClipOp,
)
from app.models.timeline.parts import (
    CAPTION_PRESETS, Clip, CueWord, Project, SubtitleCue, TextLayer, TimeRange, TitleCard, Track, TrackType,
    card_background,
)


def _in_the_file(
    project: "Project", start: Decimal, end: Decimal, asset_id: Optional[str],
) -> Tuple[str, Decimal, Decimal]:
    """Turn two times in the cut into the file and seconds a caption is anchored to.

    A caption belongs to the words of a file, not to a moment of the cut, so a
    time in the cut only means something through the clip playing then — and
    both ends have to come through the same one, or the caption would claim
    seconds of the file the cut does not play there.

    Args:
        project: The project, as it stands before this edit.
        start: Where the caption starts in the cut.
        end: Where it ends.
        asset_id: The caption's file, when it already has one.

    Returns:
        `(asset_id, source_start, source_end)`.

    Raises:
        ValueError: If no clip of that file plays at `start`, or `end` is past it.
    """
    tracks = [project.base_video_track] if project.base_video_track else []
    tracks += [track for track in project.tracks if track.track_type == TrackType.AUDIO]
    playing = [
        clip for track in tracks for clip in track.clips
        if clip.timeline_in <= start < clip.timeline_out and (asset_id is None or clip.asset_id == asset_id)
    ]
    if not playing:
        whose = f" of {asset_id}" if asset_id else ""
        raise ValueError(f"no clip{whose} plays at {start}s of the cut, so a caption cannot start there")
    clip = playing[0]
    if end > clip.timeline_out:
        raise ValueError(
            f"a caption's times have to stay inside one clip of its file: clip {clip.id} plays from "
            f"{clip.timeline_in}s to {clip.timeline_out}s of the cut"
        )
    speed = Decimal(str(clip.speed))

    def to_source(moment: Decimal) -> Decimal:
        return clip.source_range.start + (moment - clip.timeline_in) * speed

    return clip.asset_id, to_source(start), to_source(end)

def _cut_span(project: "Project", cue: SubtitleCue) -> Tuple[Decimal, Decimal]:
    """Where a caption now sits in the cut, for the end of it an edit leaves alone.

    Args:
        project: The project.
        cue: The caption.

    Returns:
        `(start, end)` in the cut's seconds, through the first clip of its file
        that plays any of it.

    Raises:
        ValueError: If no clip plays any of it, so it has no time in the cut.
    """
    tracks = [project.base_video_track] if project.base_video_track else []
    tracks += [track for track in project.tracks if track.track_type == TrackType.AUDIO]
    for clip in sorted((clip for track in tracks for clip in track.clips), key=lambda item: item.timeline_in):
        first = max(cue.source_start, clip.source_range.start)
        last = min(cue.source_end, clip.source_range.end)
        if clip.asset_id == cue.asset_id and last > first:
            speed = Decimal(str(clip.speed))
            return (clip.timeline_in + (first - clip.source_range.start) / speed,
                    clip.timeline_in + (last - clip.source_range.start) / speed)
    raise ValueError(f"caption {cue.id} is not in the cut, so it has no time in it to change")

def mostly_over(cue: SubtitleCue, gone: SubtitleCue) -> bool:
    """Say whether a caption is mostly the words of another, in the same file.

    Args:
        cue: The caption.
        gone: The other, such as one deleted by hand.

    Returns:
        True when more than half of `cue`'s stretch of the file is `gone`'s:
        captioning again breaks lines differently, so the same words come
        back cut at other places.
    """
    if cue.asset_id != gone.asset_id:
        return False
    shared = min(cue.source_end, gone.source_end) - max(cue.source_start, gone.source_start)
    return shared > 0 and shared * 2 > cue.source_end - cue.source_start

def _bare(text: str) -> str:
    """Strip a caption to the characters that are said, for matching words to it."""
    return "".join(character for character in text if character.isalnum())

def retimed_words(words: List[CueWord], text: str) -> List[CueWord]:
    """Carry a caption's word timings over to corrected text, where they still fit.

    Args:
        words: The caption's words as timed.
        text: The corrected text.

    Returns:
        The same timings with the corrected characters, when the correction
        changed characters one for one; otherwise none, since timings for
        words that are gone would light up the wrong syllables.
    """
    said = _bare(text)
    if not words or len(said) != sum(len(_bare(word.text)) for word in words):
        return []
    retimed, at = [], 0
    for word in words:
        length = len(_bare(word.text))
        retimed.append(word.model_copy(update={"text": said[at:at + length] if length else word.text}))
        at += length
    return retimed

def words_as_written(words: List[CueWord], text: str) -> List[CueWord]:
    """Put a caption's text, as written, onto its timed words, for lighting up word by word.

    The words carry what the transcript heard; the text is what somebody may
    since have corrected, punctuation and all. Drawn from the words, a
    correction would quietly turn back into the transcript on screen (#85).

    Args:
        words: The caption's timed words.
        text: Its text.

    Returns:
        The same timings carrying the text's own characters — each word its
        said characters and the punctuation after them — when the text says
        the same number of characters; otherwise none, so it lights up whole.
    """
    if not words or sum(len(_bare(word.text)) for word in words) != len(_bare(text)):
        return []
    written, at = [], 0
    for index, word in enumerate(words):
        wanted, taken = len(_bare(word.text)), 0
        if not wanted and index < len(words) - 1:
            # A word with nothing said in it keeps its time and shows nothing; the text goes to the rest.
            written.append(word.model_copy(update={"text": ""}))
            continue
        start = at
        while at < len(text) and taken < wanted:
            taken += text[at].isalnum()
            at += 1
        # What follows up to the next said character goes with this word; the last takes the rest.
        while at < len(text) and (not text[at].isalnum() or index == len(words) - 1):
            at += 1
        written.append(word.model_copy(update={"text": text[start:at]}))
    return written

# Touching a compiled clip through one of these is the user changing it by hand, which
# is what pinning records. Nobody has to remember to say so, which is the point: the
# alternative is a feedback loop that wipes their work every time it comes round.
_EDITS_BY_HAND = (
    TrimClipOp, MoveClipOp, SplitClipOp, ReorderClipOp, SetClipLookOp, SetClipAudioOp, SetClipSpeedOp,
    SetTitleCardOp,
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

def _new_clip(track: Track, spec: NewClipSpec, timeline_in: Decimal) -> Clip:
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
        keep_level=spec.keep_level,
        transition_in=spec.transition_in,
        video_fade_in=spec.video_fade_in,
        video_fade_out=spec.video_fade_out,
        color=spec.color,
        layout=spec.layout,
        fit=spec.fit,
        motion=spec.motion,
        card=spec.card,
    )

def _find_text(clip: Clip, text_id: str) -> TextLayer:
    """Find a text layer on a clip.

    Raises:
        ValueError: If the clip has no text with that ID.
    """
    # Words taken off by hand are gone as far as anybody editing is concerned.
    for text in clip.shown_texts:
        if text.id == text_id:
            return text
    known = ", ".join(text.id for text in clip.shown_texts) or "none"
    raise ValueError(f"clip {clip.id} has no text {text_id}; its texts: {known}")

def _text_times(clip: Clip, op: TextTimes, start: Decimal, end: Decimal) -> Tuple[Decimal, Decimal]:
    """Work out the stretch of a clip's file a text layer covers.

    Times in the cut are turned into the file's own seconds through the clip,
    so the words stay with the footage when the cut changes around it.

    Args:
        clip: The clip the text is on.
        op: The operation, with file times, cut times, or neither.
        start: The start to keep when the operation gives none.
        end: The end to keep when the operation gives none.

    Returns:
        `(start, end)` in seconds of the clip's file.

    Raises:
        ValueError: If the stretch is empty or misses the clip's footage altogether.
    """
    speed = Decimal(str(clip.speed))

    def from_cut(seconds: Decimal) -> Decimal:
        return clip.source_range.start + (seconds - clip.timeline_in) * speed

    if op.timeline_start is not None:
        start = from_cut(op.timeline_start)
    if op.timeline_end is not None:
        end = from_cut(op.timeline_end)
    if op.start is not None:
        start = op.start
    if op.end is not None:
        end = op.end
    start = max(start, Decimal(0))
    if end <= start:
        raise ValueError(f"clip {clip.id}: a text has to end after it starts ({start}s to {end}s of the file)")
    if end <= clip.source_range.start or start >= clip.source_range.end:
        raise ValueError(
            f"clip {clip.id}: {start}s to {end}s of the file is not in the clip, which plays "
            f"{clip.source_range.start}s to {clip.source_range.end}s of it ({clip.timeline_in}s to "
            f"{clip.timeline_out}s of the cut)"
        )
    return start, end

def _card_backdrop(
    track: Track,
    before_clip_id: Optional[str],
    photo_asset_id: Optional[str],
    assets: Mapping[str, Asset],
) -> Tuple[str, Decimal]:
    """Choose the file a title card shows behind its words, and which second of it.

    Args:
        track: The track the card goes on.
        before_clip_id: The clip the card leads into, or None for the end.
        photo_asset_id: A photo asked for, which wins.
        assets: Registered assets.

    Returns:
        `(asset_id, at)`: a photo from its start; else the first frame of the
        shot the card leads into; else, at the end, the last frame of the
        shot before it.

    Raises:
        ValueError: If the photo is not a picture, or there is no shot to
            borrow a frame from and no photo was given.
    """
    if photo_asset_id is not None:
        photo = assets.get(photo_asset_id)
        if photo is None:
            raise ValueError(f"asset {photo_asset_id} not found")
        if not photo.has_video:
            raise ValueError(f"asset {photo_asset_id} has no picture to put behind a title card")
        return photo.id, Decimal(0)
    shots = [clip for clip in sorted(track.clips, key=lambda c: c.timeline_in) if clip.card is None]
    if before_clip_id is not None:
        following = [clip for clip in shots if clip.timeline_in >= _find_clip(track, before_clip_id).timeline_in]
        if following:
            return following[0].asset_id, following[0].source_range.start
    if shots:
        last = shots[-1]
        # A frame just inside its end: its very last instant may be past its last frame.
        return last.asset_id, max(last.source_range.start, last.source_range.end - Decimal("0.1"))
    raise ValueError("a title card borrows its background from a shot next to it, and there is none yet; "
                     "give it a photo with photo_asset_id")

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

    if op.loop and kept and kept[-1].timeline_out < video_end:
        # The longest clip is the untrimmed one, so looping it repeats the whole song rather than an offcut.
        unit = max(kept, key=lambda item: item.source_range.duration)
        # A song with a loop point turns round on it, on the beat, instead of playing its ending
        # and starting again from the top: the clip that ran out stops at the loop's end, and
        # every repeat is the loop.
        span = op.loops.get(unit.asset_id)
        last = kept[-1]
        if (span is not None and last.asset_id == unit.asset_id
                and last.source_range.start < span[1] <= last.source_range.end and span[0] < span[1]):
            last.source_range = TimeRange(start=last.source_range.start, end=span[1])
            repeat = TimeRange(start=span[0], end=span[1])
        else:
            repeat = unit.source_range
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
                    start=repeat.start,
                    end=repeat.start + min(repeat.duration, remaining),
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
            Track(id=op.track_id, track_type=op.track_type, duck_under_speech=op.duck_under_speech, voice=op.voice)
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
        project.name = op.name
        return

    if isinstance(op, SetSubtitlesOp):
        cues = op.cues if op.for_all_projects else [cue.model_copy(update={"local": True}) for cue in op.cues]
        project.subtitles = number_cues(sorted(cues, key=cue_order))
        return

    if isinstance(op, AddSubtitleOp):
        taken = {item.id for item in project.subtitles}
        number = len(project.subtitles) + 1
        while f"c{number}" in taken:
            number += 1
        if op.timeline_start is not None:
            asset_id, source_start, source_end = _in_the_file(project, op.timeline_start, op.timeline_end,
                                                              op.asset_id)
        else:
            asset_id, source_start, source_end = op.asset_id, op.source_start, op.source_end
        added = SubtitleCue(
            id=f"c{number}", asset_id=asset_id, source_start=source_start, source_end=source_end,
            text=op.text, secondary=op.secondary, speaker=op.speaker, local=not op.for_all_projects,
        )
        project.subtitles = sorted([*project.subtitles, added], key=cue_order)
        # A caption put back by hand where one was deleted is wanted there after all.
        project.deleted_subtitles = [gone for gone in project.deleted_subtitles if not mostly_over(added, gone)]
        return
    if isinstance(op, EditSubtitleOp):
        cue = next((item for item in project.subtitles if item.id == op.cue_id), None)
        if cue is None:
            known = ", ".join(item.id for item in project.subtitles[:5]) or "none"
            raise ValueError(f"caption {op.cue_id} not found; the project's captions start with {known}")
        if op.delete:
            project.subtitles = [item for item in project.subtitles if item.id != op.cue_id]
            # Remembered, so captioning again after the cut changes does not bring it back (#79).
            project.deleted_subtitles = [*project.deleted_subtitles, cue.model_copy(update={"id": ""})]
            return
        source_start, source_end = op.source_start, op.source_end
        if op.timeline_start is not None or op.timeline_end is not None:
            placed_start, placed_end = _cut_span(project, cue)
            _, start, end = _in_the_file(project, op.timeline_start if op.timeline_start is not None else placed_start,
                                         op.timeline_end if op.timeline_end is not None else placed_end, cue.asset_id)
            source_start = start if op.timeline_start is not None else source_start
            source_end = end if op.timeline_end is not None else source_end
        named = {
            field: value for field, value in
            (("text", op.text), ("secondary", op.secondary), ("speaker", op.speaker),
             ("source_start", source_start), ("source_end", source_end))
            if value is not None
        }
        # Kept to this project unless it is said to be for all of them: words put to one cut,
        # remembered against the footage, turn up in every cut made from it after.
        named["local"] = not op.for_all_projects
        # Correcting the words leaves the word timings describing words that are no longer
        # there, and a caption lighting up against the wrong syllables is worse than one
        # lighting up whole — unless the correction only swaps characters one for one, the
        # commonest correction there is (掐 for 掰), and then each word keeps its timing.
        if op.text is not None:
            named["words"] = retimed_words(cue.words, op.text)
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
        if op.voice_auto:
            track.voice = None
        elif op.voice is not None:
            track.voice = op.voice
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
    elif isinstance(op, AddTitleCardOp):
        if track is not project.base_video_track:
            raise ValueError(f"a title card goes on the main video track, not {track.id}")
        if op.before_clip_id is None:
            position = max((clip.timeline_out for clip in track.clips), default=Decimal(0))
        else:
            position = _find_clip(track, op.before_clip_id).timeline_in
        asset_id, at = _card_backdrop(track, op.before_clip_id, op.photo_asset_id, assets)
        background = card_background(op.background, op.photo_asset_id)
        clip = _new_clip(track, NewClipSpec(
            track_id=track.id, clip_id=op.clip_id, asset_id=asset_id,
            source_range=TimeRange(start=Decimal(0), end=op.seconds),
        ), position)
        clip.card = TitleCard(title=op.title, subtitle=op.subtitle or None, background=background,
                              at=at, **({"colour": op.colour} if op.colour else {}))
        # A card is silent: whatever the file behind it says belongs to the shot.
        clip.volume = 0.0
        _shift_clips(track, position, clip.timeline_duration)
        track.clips.append(clip)
    elif isinstance(op, SetTitleCardOp):
        clip = _find_clip(track, op.clip_id)
        if clip.card is None:
            raise ValueError(f"clip {clip.id} is not a title card; set_title_card only changes cards")
        card = clip.card.model_copy()
        if op.title is not None:
            card.title = op.title
        if op.subtitle is not None:
            card.subtitle = op.subtitle or None
        if op.photo_asset_id is not None:
            clip.asset_id, card.at = _card_backdrop(track, None, op.photo_asset_id, assets)
            card.background = "picture"
        if op.background is not None:
            card.background = op.background
        if op.colour is not None:
            card.colour = op.colour
        clip.card = card
        if op.seconds is not None:
            old_out = clip.timeline_out
            clip.source_range = TimeRange(start=clip.source_range.start, end=clip.source_range.start + op.seconds)
            clip.speed = 1.0
            _shift_clips(track, old_out, clip.timeline_out - old_out)
    elif isinstance(op, AddTextOp):
        clip = _find_clip(track, op.clip_id)
        if track.track_type != TrackType.VIDEO or clip.card is not None:
            raise ValueError(f"clip {clip.id}: words go over footage on a video track; a title card has its own")
        if any(text.id == op.text_id for text in clip.texts):
            raise ValueError(f"clip {clip.id} already has a text {op.text_id}")
        start, end = _text_times(clip, op, clip.source_range.start, clip.source_range.end)
        clip.texts.append(TextLayer(id=op.text_id, text=op.text, second=op.second or None, style=op.style,
                                    start=start, end=end, x=op.x, y=op.y, by_plan=op.by_plan, hidden=op.hidden))
    elif isinstance(op, SetTextOp):
        clip = _find_clip(track, op.clip_id)
        layer = _find_text(clip, op.text_id)
        start, end = _text_times(clip, op, layer.start, layer.end)
        # Changed by somebody, so the next compile keeps it rather than rebuilding it from the plan.
        changed = layer.model_copy(update={"start": start, "end": end, "by_plan": False})
        if op.text is not None:
            changed.text = op.text
        if op.second is not None:
            changed.second = op.second or None
        if op.style is not None:
            if op.style != layer.style and op.style != "free" and op.x is None and op.y is None:
                # A new style puts the words where that style goes, not where the old ones were moved.
                changed.x = changed.y = None
            changed.style = op.style
        if op.x is not None:
            changed.x = op.x
        if op.y is not None:
            changed.y = op.y
        clip.texts = [changed if text.id == layer.id else text for text in clip.texts]
    elif isinstance(op, RemoveTextOp):
        clip = _find_clip(track, op.clip_id)
        layer = _find_text(clip, op.text_id)
        if layer.by_plan:
            # The plan would put it back on the next compile: kept, out of sight, as taken off by hand.
            clip.texts = [text.model_copy(update={"hidden": True, "by_plan": False}) if text.id == layer.id else text
                          for text in clip.texts]
        else:
            clip.texts = [text for text in clip.texts if text.id != layer.id]
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
            # The halves are the same shot, so they keep the same look and the same origin —
            # and the same words over it, each half showing what falls in its own stretch.
            color=clip.color,
            layout=clip.layout,
            fit=clip.fit,
            motion=clip.motion,
            card=clip.card,
            texts=[text.model_copy() for text in clip.texts],
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
        if op.clear_fit:
            clip.fit = None
        elif op.fit is not None:
            clip.fit = op.fit
        if op.motion is not None:
            clip.motion = op.motion
    elif isinstance(op, SetClipAudioOp):
        clip = _find_clip(track, op.clip_id)
        for field in ("volume", "audio_fade_in", "audio_fade_out", "audio_lead", "audio_lag", "keep_level"):
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
