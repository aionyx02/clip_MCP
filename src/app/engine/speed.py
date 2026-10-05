"""How long the slow parts of an analysis took on this computer, and how long the next will take.

Before anything has run, the only guide is a table by the computer's tier.
Every analysis long enough to say something then leaves behind how many
seconds transcription and telling the voices apart took for how many seconds
of footage, and from then on the estimate is this computer's own. Only this
computer's: a workspace can move to another one, and a desktop's speed says
nothing about a laptop's.

Only the two stages that make a user wait are measured. Faces, search and the
FFmpeg pass take seconds to a minute anywhere.
"""

import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.engine.analysis import whisper_model_for
from app.engine.diarize import speaker_model_name
from app.engine.machine import Machine
from app.storage.repo import Repository

# Provisional (roadmap §13): minutes an hour of footage takes to transcribe accurately,
# by tier, before anything has been measured on this computer. `None` is no upper bound.
# Measured 2026-10-05 on an RTX 4060 laptop with an i7-14700HX: about 2 minutes an hour
# on the card, 10 on the processor, 13 held to two threads. The low ends come from that
# one fast computer; the high ends leave room for slower ones.
HOUR_MINUTES = {"high": (3, 10), "mid": (10, 40), "low": (30, None)}
# Provisional (roadmap §13): how many times quicker a fast transcription is than an
# accurate one, for the same table, until a fast one has been measured here. Measured
# 1.9 to 2.3 on the processor of the same computer.
FAST_SPEEDUP = 2
# Provisional (roadmap §13): minutes an hour of footage takes to tell the voices apart,
# by tier, before anything has been measured. It runs on the processor everywhere, so the
# card does not help; measured about 5 minutes an hour on an i7-14700HX.
SPEAKERS_HOUR_MINUTES = {"high": (3, 8), "mid": (3, 8), "low": (8, 20)}
# Provisional (roadmap §13): a file shorter than this is mostly loading the model, so its
# speed says little about the next hour of footage.
MIN_MEASURED_SECONDS = 120.0
# Provisional (roadmap §13): how many recent runs an estimate is the median of, how many
# are kept, and how much room is left above it.
RECENT_SAMPLES = 5
KEPT_SAMPLES = 20
HEADROOM = 1.5

ROUGH = "rough, from the hardware; not yet measured on this computer"
MEASURED = "measured on this computer"
# Diarization runs on the processor whatever the computer has.
SPEAKERS_DEVICE = "cpu"

@dataclass(frozen=True)
class Measured:
    """How long one stage of one analysis took.

    Attributes:
        stage: `transcription` or `speakers`.
        device: `cuda` or `cpu`, as it actually ran.
        model: The model it ran.
        seconds: Time spent, not counting loading the model.
    """

    stage: str
    device: str
    model: str
    seconds: float

def record(repo: Repository, runs: List[Measured], media_seconds: float, computer: str) -> None:
    """Keep how fast each stage ran, when the file was long enough to tell.

    Args:
        repo: Where the runs are kept.
        runs: The stages of one finished analysis.
        media_seconds: How long the file is.
        computer: This computer, as `machine.identity` names it.
    """
    if media_seconds < MIN_MEASURED_SECONDS:
        return
    for run in runs:
        repo.add_speed_sample(computer, run.stage, run.device, run.model, run.seconds, media_seconds, KEPT_SAMPLES)

def _stage(
    repo: Repository, computer: str, stage: str, device: str, model: str, footage_seconds: float,
    table: Tuple[float, Optional[float]],
) -> Tuple[Tuple[float, Optional[float]], bool]:
    """Estimate one stage: from this computer's recent runs, or from the hardware table.

    Args:
        repo: Where past runs are kept.
        computer: This computer.
        stage: `transcription` or `speakers`.
        device: Where the stage runs.
        model: The model it runs.
        footage_seconds: How much footage it runs over.
        table: Minutes an hour takes by the hardware table, high None for no bound.

    Returns:
        `(low, high)` in seconds, high None for no bound, and whether it was measured.
    """
    samples = repo.speed_samples(computer, stage, device, model, limit=RECENT_SAMPLES)
    if samples:
        typical = statistics.median(samples) * footage_seconds
        return (typical, typical * HEADROOM), True
    low, high = table
    hours = footage_seconds / 3600
    return (low * 60 * hours, None if high is None else high * 60 * hours), False

def _minutes(seconds: Tuple[float, Optional[float]]) -> List[Optional[int]]:
    """A range in whole minutes, never shorter than one, rounded outwards at the top."""
    low = max(1, round(seconds[0] / 60))
    return [low, None if seconds[1] is None else max(low, math.ceil(seconds[1] / 60))]

def estimate(repo: Repository, here: Machine, footage: Dict[str, float], speakers: bool) -> dict:
    """Estimate how long transcribing, and telling the voices apart in, some footage takes here.

    Args:
        repo: Where past runs are kept.
        here: This computer.
        footage: Seconds of footage by transcription choice, `accurate` and
            `fast`; a batch can hold both, when some files keep the accurate
            transcript they already have.
        speakers: Whether the voices will be told apart too.

    Returns:
        `transcription_minutes` and `speakers_minutes` as `[low, high]`, high
        null when there is no telling; `speakers_minutes` null when not asked
        for; and `basis`, measured only when every figure was.
    """
    low, high, all_measured = 0.0, 0.0, True
    for transcription, seconds in footage.items():
        if not seconds:
            continue
        table = HOUR_MINUTES[here.tier]
        if transcription == "fast":
            table = (table[0] / FAST_SPEEDUP, None if table[1] is None else table[1] / FAST_SPEEDUP)
        (part_low, part_high), measured = _stage(repo, here.identity, "transcription", here.device,
                                                 whisper_model_for(transcription), seconds, table)
        low += part_low
        high = None if high is None or part_high is None else high + part_high
        all_measured = all_measured and measured
    found = {"transcription_minutes": _minutes((low, high)), "speakers_minutes": None}
    if speakers:
        spoken, measured = _stage(repo, here.identity, "speakers", SPEAKERS_DEVICE, speaker_model_name(),
                                  sum(footage.values()), SPEAKERS_HOUR_MINUTES[here.tier])
        found["speakers_minutes"] = _minutes(spoken)
        all_measured = all_measured and measured
    found["basis"] = MEASURED if all_measured else ROUGH
    return found
