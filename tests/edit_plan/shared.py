"""What the edit plan tests build their footage, plans and compiled cuts from."""

from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from app.engine.plan.beats import compile_pieces
from app.engine.plan.checks import check_plan
from app.engine.plan.compile import compile_operations
from app.engine.plan.pieces import Piece
from app.engine.sections import build_sections
from app.engine.semantic import build_timeline, clean_cuts
from app.models.media import Asset, MediaAnalysis, Rhythm, TranscriptWord
from app.models.plan import (
    BeatRole, Beat, BeatTransition, EditPlan, Established, Fact, Rejection, Selection, Trim, TrimKind,
)
from app.models.semantic import SectionChoice, SemanticClip, SemanticTimeline
from app.models.timeline.parts import Project
from app.runtime import repo as runtime_repo
from app.tools.catalog import import_asset
from app.tools.planning import save_plan
from test_semantic_timeline import ASSET_ID, analysis, asset


FOOTAGE = dict(
    duration=20.0,
    segments=[
        (1.0, 3.0, "第一句話"),
        (3.5, 5.0, "第二句話"),
        (8.0, 10.0, "第三句話"),
        (12.0, 14.0, "第四句話"),
    ],
    silences=[(0.0, 1.0), (3.0, 3.5), (5.0, 8.0), (10.0, 12.0), (14.0, 20.0)],
)

def sourced(asset_id: str = ASSET_ID, seconds: float = 20.0) -> Asset:
    """Build an asset that knows how long it is.

    Args:
        asset_id: ID for the asset.
        seconds: Its duration.

    Returns:
        The asset.
    """
    return asset(asset_id).model_copy(update={"duration": Decimal(str(seconds))})

def footage() -> tuple:
    """Build a timeline over one file of talk.

    Returns:
        `(clips_by_id, children, assets, ordered_clips)`.
    """
    made = analysis(**FOOTAGE)
    _, clips = build_timeline({ASSET_ID: sourced()}, {ASSET_ID: made})
    return {clip.id: clip for clip in clips}, {}, {ASSET_ID: sourced()}, clips

def sectioned() -> tuple:
    """Build a timeline with the whole file as one section.

    Returns:
        `(clips_by_id, children, assets, section_id)`.
    """
    made = analysis(**FOOTAGE)
    timeline, clips = build_timeline({ASSET_ID: sourced()}, {ASSET_ID: made})
    sections, utterances = build_sections(timeline.id, clips, [
        SectionChoice(first_clip_id=clips[0].id, last_clip_id=clips[-1].id, name="全部", summary="整段"),
    ])
    everything = [*utterances, *sections]
    children = {sections[0].id: [clip for clip in utterances]}
    return {clip.id: clip for clip in everything}, children, {ASSET_ID: sourced()}, sections[0].id

def plan_for(selections: List[Selection], **fields) -> EditPlan:
    """Build a plan over one beat.

    Args:
        selections: The chosen footage.
        **fields: Other plan fields.

    Returns:
        The plan.
    """
    return EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash",
        beats=[Beat(id="b1", name="主體")],
        selections=selections,
        **fields,
    )

def with_a_fact(plan: EditPlan, start: float, end: float) -> EditPlan:
    """Give a plan of several parts the one fact it needs, established by its first piece.

    Args:
        plan: The plan.
        start: Where in the first piece's file the fact is said.
        end: Where that stops.

    Returns:
        The plan, with the fact and its evidence.
    """
    first = plan.selections[0].model_copy(update={"establishes": [Established(fact="topic", evidence=(start, end))]})
    return plan.model_copy(update={
        "facts": [Fact(id="topic", says="這支在講什麼")], "selections": [first, *plan.selections[1:]],
    })

def speech_of(clips: List[SemanticClip]) -> List[SemanticClip]:
    """Pick out the clips where someone is talking.

    Args:
        clips: Clips to filter.

    Returns:
        The speech clips in order.
    """
    return [clip for clip in clips if clip.text]

def spoken_words(*triples) -> MediaAnalysis:
    """Build an analysis of one sentence whose words are timed individually.

    `analysis()` gives each sentence a single word spanning it, which is
    enough for everything else but hides where the words actually are — and
    where the words are is the whole question here.

    Args:
        *triples: `(text, start, end)` per word.

    Returns:
        The analysis, with silence before the first word and after the last.
    """
    first, last = triples[0][1], triples[-1][2]
    made = analysis(
        20.0,
        segments=[(first, last, "".join(word for word, _, _ in triples))],
        silences=[(0.0, first), (last, 20.0)],
    )
    return made.model_copy(update={"transcript": made.transcript.model_copy(update={
        "segments": [made.transcript.segments[0].model_copy(update={
            "words": [TranscriptWord(text=word, start=start, end=end) for word, start, end in triples],
        })],
    })})

SENTENCE = (("今天", 2.0, 3.0), ("四點", 3.0, 4.0), ("就起床", 4.0, 5.5), ("了", 5.5, 6.0))

def compiled(made: MediaAnalysis, trims: Dict[int, Trim] = {}, which: str = "speech") -> tuple:
    """Compile every clip of one kind in a file, in order.

    Args:
        made: The analysis to derive the timeline from.
        trims: A trim for the selection at that position, where one is wanted.
        which: `speech` to select the talking, `quiet` to select the rest.

    Returns:
        `(pieces, notes)`.
    """
    one = sourced(seconds=made.duration)
    _, clips = build_timeline({ASSET_ID: one}, {ASSET_ID: made})
    chosen = speech_of(clips) if which == "speech" else [clip for clip in clips if not clip.text]
    plan = plan_for([
        Selection(clip_id=clip.id, beat_id="b1", trim=trims.get(place, Trim()))
        for place, clip in enumerate(chosen)
    ])
    return compile_pieces(plan, {clip.id: clip for clip in clips}, {}, {ASSET_ID: one},
                          {ASSET_ID: clean_cuts(made)})

# B-roll: picture laid over the cut while the sound underneath keeps running. The rules
# about how long it may run and what it may cover are arithmetic, so they live in the
# compiler and are refused with a reason rather than hoped for in a guide.

BROLL_ASSET = "asset-2"

def covered() -> tuple:
    """A timeline of talk, plus a second file of silent picture to lay over it.

    Returns:
        `(clips_by_id, children, assets, ordered_clips)`.
    """
    talk, quiet = analysis(**FOOTAGE), analysis(
        duration=20.0, silences=[(0.0, 20.0)], scenes=[(0.0, 20.0)], asset_id=BROLL_ASSET,
    )
    assets = {ASSET_ID: sourced(), BROLL_ASSET: sourced(BROLL_ASSET)}
    _, clips = build_timeline(assets, {ASSET_ID: talk, BROLL_ASSET: quiet})
    return {clip.id: clip for clip in clips}, {}, assets, clips

@pytest.fixture
def planned(media: Path) -> dict:
    """Import a real file, build a timeline over it, and save a plan for it.

    Returns:
        A dictionary with the `plan_id`, `timeline_id` and the chosen clips.
    """
    from app.tools.timelines import build_semantic_timeline, query_clips

    asset_id = import_asset(str(media / "wide.mp4"))["id"]
    runtime_repo.save_analysis(analysis(asset_id=asset_id, **FOOTAGE))
    built = build_semantic_timeline([asset_id], rebuild=True)
    speech = query_clips(timeline_id=built["timeline_id"], text="句話")["clips"]
    saved = save_plan(EditPlan(
        timeline_id=built["timeline_id"], timeline_input_hash="",
        goal="做一支短的",
        beats=[Beat(id="b1", name="主體", intent="講完一件事")],
        selections=[
            Selection(clip_id=speech[0]["clip_id"], beat_id="b1", rationale="開場最清楚"),
            Selection(clip_id=speech[2]["clip_id"], beat_id="b1", rationale="收在這句"),
        ],
        rejected=[Rejection(clip_id=speech[1]["clip_id"], reason="重複了")],
    ))
    return {**saved, "clips": speech}

# --- structure markers ------------------------------------------------------------------------

def beats_plan(selections: List[Selection], beats: List[Beat]) -> EditPlan:
    """Build a plan whose beats are given rather than assumed.

    Args:
        selections: The chosen footage.
        beats: The parts of the video.

    Returns:
        The plan.
    """
    return EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash", beats=beats, selections=selections,
    )

# Music that changes with the parts of the video, and cuts put on its beat. Where the
# beat is comes from the analysis; everything here is arithmetic on those times.

SONG = "song"

def song_of(beats, seconds: float = 30.0) -> tuple:
    """A music file with its beat already measured.

    Args:
        beats: Where the beats are, in song seconds; None for a song whose
            beat was never measured.
        seconds: How long it runs.

    Returns:
        `(asset, cuts)` for the song.
    """
    song = sourced(SONG, seconds=seconds).model_copy(update={"has_video": False})
    rhythm = None if beats is None else Rhythm(tempo=120.0 if beats else None, beats=list(beats))
    return song, clean_cuts(MediaAnalysis(asset_id=SONG, duration=seconds, rhythm=rhythm))

def every(period: float, first: float = 0.0, seconds: float = 30.0) -> List[float]:
    """A steady beat.

    Args:
        period: Seconds between beats.
        first: Where the first one is.
        seconds: How far it runs.

    Returns:
        The beats.
    """
    return [round(first + period * index, 3) for index in range(int((seconds - first) / period) + 1)]

def cut_points(pieces: List[Piece]) -> List[float]:
    """Where each picture cut lands on the timeline.

    Args:
        pieces: The compiled windows.

    Returns:
        The seconds between one window and the next.
    """
    points, position = [], 0.0
    for piece in pieces[:-1]:
        position += piece.played
        points.append(round(position, 3))
    return points

def checked(plan: EditPlan, by_id: dict, assets: dict, cuts=None) -> tuple:
    """Check a plan with everything known about its files.

    Args:
        plan: The plan.
        by_id: The clips.
        assets: The assets.
        cuts: What is known about each file.

    Returns:
        `(problems, notes)`.
    """
    timeline = SemanticTimeline(
        id=plan.timeline_id, asset_ids=[ASSET_ID], input_hash=plan.timeline_input_hash, derivation_version=1,
    )
    return check_plan(plan, timeline, by_id, {}, assets, cuts)

def parts() -> tuple:
    """Three shots in three beats, each one a part of the video with its own music.

    Returns:
        `(plan, clips_by_id, assets)`, the plan having no music yet.
    """
    by_id, _, assets, clips = footage()
    talk = speech_of(clips)
    plan = EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash",
        beats=[Beat(id="b1", name="開場", role=BeatRole.HOOK), Beat(id="b2", name="中段", role=BeatRole.TURN),
               Beat(id="b3", name="結尾", role=BeatRole.PAYOFF)],
        selections=[Selection(clip_id=talk[0].id, beat_id="b1"), Selection(clip_id=talk[2].id, beat_id="b2"),
                    Selection(clip_id=talk[3].id, beat_id="b3")],
    )
    opened, closed = talk[0].source_range.start, talk[0].source_range.end
    plan = with_a_fact(plan, opened + (closed - opened) * 0.4, opened + (closed - opened) * 0.6)
    other = sourced("other", seconds=60.0).model_copy(update={"has_video": False})
    return plan, by_id, {**assets, SONG: song_of(every(0.5), 60.0)[0], "other": other}

def music_by_hand(plan: EditPlan, by_id: dict, assets: dict, index: int = 0, **changes) -> tuple:
    """Compile a plan's music, then change one stretch of it by hand, which pins it.

    Args:
        plan: The plan.
        by_id: The clips.
        assets: The assets.
        index: Which stretch to change.
        **changes: The clip fields changed by hand.

    Returns:
        `(project, laid)` — a project holding the music with that stretch
        changed, and the music operations the first compile made.
    """
    from app.models.timeline.parts import Clip, TimeRange, Track, TrackType

    operations, provenance = compile_operations(plan, by_id, {}, assets)
    laid = [op for op in operations if op.get("track_id") == "music" and op["action"] == "add_clip"]
    music = []
    for position, op in enumerate(laid):
        clip = Clip(
            id=op["clip_id"], asset_id=op["asset_id"], timeline_in=Decimal(str(op["timeline_in"])),
            source_range=TimeRange(start=Decimal(str(op["source_range"]["start"])),
                                   end=Decimal(str(op["source_range"]["end"]))),
            volume=op["volume"], from_plan_id=plan.id, from_clip_ids=provenance[op["clip_id"]]["from_clip_ids"],
        )
        if position == index:
            clip = clip.model_copy(update={**changes, "pinned": True})
        music.append(clip)
    return Project(id="p", tracks=[Track(id="music", track_type=TrackType.AUDIO, clips=music)]), laid

def joined_plan(transition: Optional[BeatTransition], lead: Optional[float], opening: float) -> tuple:
    """The plan `joined` compiles, with what compiling or checking it needs.

    Returns:
        `(plan, clips by ID, children, assets)`.
    """
    by_id, children, assets, clips = covered()
    talk = speech_of(clips)[0]
    walk = next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = EditPlan(
        timeline_id="tl_test", timeline_input_hash="hash",
        beats=[Beat(id="b1", name="講", role="turn"),
               Beat(id="b2", name="走", role="payoff", transition_in=transition, sound_lead=lead)],
        selections=[
            Selection(clip_id=talk.id, beat_id="b1"),
            Selection(clip_id=walk.id, beat_id="b2",
                      trim=Trim(kind=TrimKind.RANGE, from_seconds=opening, to_seconds=opening + 3.0)),
        ],
    )
    opened, closed = talk.source_range.start, talk.source_range.end
    plan = with_a_fact(plan, opened + (closed - opened) * 0.4, opened + (closed - opened) * 0.6)
    return plan, by_id, children, assets
