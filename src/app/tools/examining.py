"""Reading the footage: analysis jobs, the transcript heard again where it is in doubt, and what an analysis holds.

Also songs as they were heard, which the library, plans and the music tools all read.
"""

import difflib
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Annotated, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import Field

from app.models.media import Asset, HeardAgain, MediaAnalysis, Transcript, TranscriptWord
from app.models.timeline.parts import Project, TrackType
from app.models.job import Job, JobKind
from app.engine import clap, machine, resources, speed
from app.engine.analysis import listen_again as listen_again_in_file
from app.engine.analysis import (
    BALANCED_JOIN, current_recipe, doubtful_spots, marked_text, sound_note, transcription_of, unsure_words,
    whisper_model_for, whisper_model_name,
)
from app.engine.diarize import speaker_model_name
from app.engine.rhythm import rhythm_model_name
from app.engine.faces import framing_note
from app.engine.semantic import EDGE_TOLERANCE_SECONDS, was_made_up
from app.engine.ffmpeg import hidden_window_flags
from app.engine.library import MUSIC, SPEECH_SHARE, classify, library_of, speech_share
from app.engine import music as music_energy
from app.engine.frames import format_timestamp
from app.engine.listen import cut_sound, levels_every
from app.engine.subtitles import place_cues
from app import runtime
from app.runtime import mcp
from app.tools import rendering, shared

def song_tags() -> Dict[str, Dict[str, List[str]]]:
    """The moods and styles that set each analyzed song apart from the rest of the library.

    Returns:
        `moods` and `styles` by song.
    """
    sensed = {}
    for asset in runtime.repo.list_assets():
        if library_of(asset) != MUSIC:
            continue
        analysis = runtime.repo.get_analysis(asset.id)
        if analysis is not None and analysis.music is not None:
            sensed[asset.id] = analysis.music
    moods = clap.ranked({song: sense.moods for song, sense in sensed.items()})
    styles = clap.ranked({song: sense.styles for song, sense in sensed.items()})
    return {song: {"moods": moods.get(song, []), "styles": styles.get(song, [])} for song in sensed}

def sounds_like(asset_id: str, analysis: MediaAnalysis, heard: Mapping[str, Mapping[str, List[str]]]) -> dict:
    """A song in a line: its moods and styles, how it ends and where it is fullest."""
    summary = music_energy.summary(analysis.music.energy)
    return {**heard.get(asset_id, {}), "ends": summary["ends"], "fullest": summary["fullest"],
            # The speech detector put it with music; the model hears talking. Asked about, not moved.
            **({"sounds_like_speech": True} if analysis.music.sounds_like_speech else {})}

def analyzing_assets() -> set:
    """The files an analysis is queued or running for."""
    return {job.asset_id for job in runtime.repo.jobs_to_show(datetime.now(timezone.utc))[0] if job.kind == JobKind.ANALYZE}

def song_details(asset: Asset, analysis: Optional[MediaAnalysis], heard: Mapping[str, Mapping[str, List[str]]],
                 analyzing: set) -> dict:
    """What is known about a song, for the AI's list and the editor's page alike.

    Args:
        asset: The song.
        analysis: Its analysis, if any.
        heard: From `song_tags`.
        analyzing: From `analyzing_assets`.

    Returns:
        Its `tempo`, its moods, styles, ending and fullest part once heard,
        and `analyzing` until then — queued or running, since every song
        added is analyzed by itself.
    """
    found: dict = {}
    if analysis is not None and analysis.rhythm is not None:
        found["tempo"] = analysis.rhythm.tempo
    if analysis is not None and analysis.music is not None:
        found.update(sounds_like(asset.id, analysis, heard))
    if asset.id in analyzing or analysis is None or analysis.music is None:
        found["analyzing"] = True
    return found

def is_stale(analysis: MediaAnalysis) -> bool:
    """Say whether an analysis was taken in a way the server no longer uses.

    Args:
        analysis: Analysis to check.

    Returns:
        True when the detection thresholds, the measurements taken, or the
        speech model have changed since it was made. Nothing in it is wrong;
        it just answers a slightly different question from a fresh one, which
        matters most when two assets are being compared with each other.
    """
    # A fast or balanced transcript is one the user chose, not one made with a model since replaced.
    speech_model = analysis.recipe.speech_model
    current = speech_model if transcription_of(speech_model) in ("fast", "balanced") else whisper_model_name()
    return analysis.recipe.differs_from(current_recipe(
        current, speaker_model=speaker_model_name(), rhythm_model=rhythm_model_name(), clap_model=clap.MODEL_ID,
    ))

def transcription_kind(analysis: Optional[MediaAnalysis]) -> Optional[str]:
    """Say how an asset's speech was transcribed: `fast`, `balanced`, `accurate`, or None when it was not."""
    return transcription_of(None if analysis is None or analysis.transcript is None else analysis.transcript.model)

def _transcribed(asset: Asset, transcribe: bool) -> bool:
    """Whether an analysis of this file transcribes it: never a song, whatever was asked."""
    return transcribe and asset.has_audio and library_of(asset) != MUSIC

def start_analysis(asset: Asset, transcribe: bool, transcription: str, language: Optional[str] = None,
                   prompt: Optional[str] = None, chinese_variant: Optional[str] = None, diarize: bool = True,
                   speakers: Optional[int] = None, priority: int = 0) -> Job:
    """Queue one file's analysis.

    Args:
        asset: The file.
        transcribe: Whether speech is transcribed; never for a song.
        transcription: `accurate`, `balanced` or `fast`.
        language: Spoken language code, or None to detect it.
        prompt: Text that guides transcription.
        chinese_variant: Script and regional variant for a Chinese transcript.
        diarize: Whether to tell the voices apart.
        speakers: How many people are talking, when known.
        priority: Lower goes first; a song added by itself waits behind footage.

    Returns:
        The job.
    """
    job = Job(kind=JobKind.ANALYZE, asset_id=asset.id, priority=priority)
    job.work_dir = os.path.join(runtime.WORKSPACE_DIR, "jobs", job.job_id)
    with_speech = _transcribed(asset, transcribe)
    job.memory_estimate = resources.analysis_memory_bytes(
        with_speech,
        # The larger model is what has to fit; a balanced transcription loads one after the other.
        whisper_model_for(transcription).split(BALANCED_JOIN)[-1],
        duration=float(asset.duration),
        diarize=with_speech and diarize,
        detect_faces=asset.has_video,
    )
    return runtime.job_manager.start_job(job, {
        # What the analysis will do, not what was asked: a song is never transcribed, and a worker
        # that holds a speech model lets it go before an analysis that does not use it.
        "transcribe": with_speech,
        "transcription": transcription,
        "language": language,
        "prompt": prompt,
        "chinese_variant": chinese_variant,
        "diarize": diarize,
        "speakers": speakers,
    })

# How good each kind of transcription is, so a file is never analyzed into a worse one.
TRANSCRIPTION_RANK = {"fast": 0, "balanced": 1, "accurate": 2}

def plan_analyses(
    assets: Sequence[Asset], transcribe: bool, transcription: str, again: bool,
) -> Tuple[List[Tuple[Asset, str]], List[dict], List[str]]:
    """Decide which files an analysis starts, and with which transcription.

    A transcript is never made worse: asked for less than it has, a file is
    skipped while current, and analyzed again as well as it was when it is
    out of date or `again` is given. A transcript asked to be better —
    fast to balanced or accurate, balanced to accurate — is redone without
    `again`; fast to balanced keeps the fast transcript and hears only its
    unsure lines again.

    Args:
        assets: The files asked for.
        transcribe: Whether speech is transcribed.
        transcription: `accurate`, `balanced` or `fast`, as asked.
        again: Redo files whose analysis is current.

    A file already queued or being analyzed is left to that job rather than
    analyzed twice.

    Returns:
        `(to_start, skipped, upgrading)`: each file to start with the
        transcription it gets, each file left alone as `{asset_id, reason}`
        (#88), and the files whose transcript is being made better.
    """
    to_start: List[Tuple[Asset, str]] = []
    skipped: List[dict] = []
    upgrading: List[str] = []
    active = {job.asset_id: job for job in runtime.repo.jobs_to_show(datetime.now(timezone.utc))[0]
              if job.kind == JobKind.ANALYZE and job.status.is_active}
    for asset in assets:
        if asset.id in active:
            job = active[asset.id]
            skipped.append({"asset_id": asset.id, "reason": f"already {job.status.value} in job {job.job_id}; "
                                                              "follow it with get_job"})
            continue
        existing = runtime.repo.get_analysis(asset.id)
        had = transcription_kind(existing)
        upgrade = (_transcribed(asset, transcribe) and had is not None
                   and TRANSCRIPTION_RANK[transcription] > TRANSCRIPTION_RANK[had])
        # Heard as a song, never transcribed; footage now, so it is listened to again for what is said.
        heard_as_song = (existing is not None and existing.transcript is None and existing.rhythm is not None
                         and _transcribed(asset, transcribe))
        if not again and not upgrade and not heard_as_song and existing is not None and not is_stale(existing):
            skipped.append({"asset_id": asset.id, "reason": "its analysis is current; again=true to redo it"})
            continue
        if upgrade:
            upgrading.append(asset.id)
        better = had if had is not None and TRANSCRIPTION_RANK[had] > TRANSCRIPTION_RANK[transcription] else None
        to_start.append((asset, better or transcription))
    return to_start, skipped, upgrading

def _estimate(to_start: Sequence[Tuple[Asset, str]], transcribe: bool, diarize: bool,
              upgrading: Sequence[str]) -> Optional[dict]:
    """How long the files about to be analyzed take here, or None when nothing in them is transcribed."""
    footage: Dict[str, float] = {"accurate": 0.0, "balanced": 0.0, "fast": 0.0}
    for asset, transcription in to_start:
        if _transcribed(asset, transcribe):
            footage[transcription] += float(asset.duration)
    if not any(footage.values()):
        return None
    here = machine.profile()
    found = {"tier": here.tier, "footage_minutes": round(sum(footage.values()) / 60, 1),
             **speed.estimate(runtime.repo, here, footage, diarize)}
    if upgrading:
        chosen = {transcription for asset, transcription in to_start if asset.id in upgrading}
        found["note"] = (
            "files whose transcript is being made better keep their picture and sound measurements; "
            + ("made accurate, the whole transcript is heard again" if "accurate" in chosen else "")
            + ("; " if chosen == {"accurate", "balanced"} else "")
            + ("a fast one made balanced has only its unsure lines heard again, so it takes less than this says"
               if "balanced" in chosen else "")
        )
    return found

@mcp.tool()
def analyze_asset(
    asset_ids: List[str],
    transcribe: bool = True,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    chinese_variant: Optional[Literal["zh-TW", "zh-HK", "zh-Hant", "zh-Hans"]] = None,
    diarize: bool = True,
    speakers: Annotated[Optional[int], Field(ge=1, le=20)] = None,
    again: bool = False,
    transcription: Literal["accurate", "balanced", "fast"] = "accurate",
    dry_run: bool = False,
) -> dict:
    """Start analyzing what assets contain, in the background.

    Takes every file to analyze in one call — a whole imported folder at
    once — and starts one job per file. A file that already has a current
    analysis is left alone unless `again` is true, so the same call can be
    repeated over a folder after a few files are added.

    One decoding pass detects scene changes, black and frozen picture, and
    silences, and alongside them measures each shot — exposure, contrast,
    blur, motion, camera shake — each second of sound (level, peak, noise
    floor, and how squared off the waveform is where it peaks), and each
    second of picture for faces: how many, how big the largest is, and where
    it sits in frame. If `transcribe` is true and the asset has audio, speech
    is then transcribed locally with word-level timestamps that stay accurate
    on long recordings, which is the slow part by far, and the voices are told
    apart so that each clip knows who was speaking.

    Returns immediately: poll `get_job` with all the job IDs at once until
    they are done, then read the results with `get_analysis`, or search them
    through `query_clips` once a semantic timeline is built.

    Everything runs on this machine. The models are downloaded the first time
    they are needed — the speech model is the large one, the rest are tens of
    megabytes — and they are kept inside the workspace, so deleting the
    workspace deletes them too. `stage` says when a download is what a job is
    waiting on.

    Args:
        asset_ids: IDs of the assets to analyze. The same settings apply to
            all of them; analyze files that need different ones in separate
            calls.
        transcribe: Whether to transcribe speech.
        language: Spoken language code, such as "zh" or "en"; omit to detect it.
        prompt: Text that guides transcription, such as names and terms that
            appear in the recording.
        diarize: Whether to tell the voices apart. Only done alongside a
            transcript, since a speaker label with nothing said under it has
            nowhere to go. Turn it off for footage with one person in it if
            the extra minute matters.
        speakers: How many people are talking, when that is known. Worth
            giving for an interview or a two-hander: told the number, the
            server has to find exactly that many, and cannot split one person
            who leaned closer to the microphone into two. Omit it to let the
            number be worked out.
        chinese_variant: Convert a Chinese transcript to this script and
            regional variant: "zh-TW" (Traditional, Taiwan characters and
            vocabulary, e.g. 視頻 becomes 影片), "zh-HK" (Traditional, Hong
            Kong), "zh-Hant" (Traditional, no vocabulary changes), or "zh-Hans"
            (Simplified). Omit to keep the speech model's output, which may mix
            scripts.
        again: Analyze files that already have a current analysis too,
            replacing it — for example with a `prompt` or `speakers` the first
            run did not have. Only what would come out different is redone:
            the transcript, the voices when `speakers` is given; the scan and
            the faces are kept.
        transcription: "accurate", the default; "fast": about twice as
            quick without an NVIDIA card, with more misheard words and looser
            word timings; or "balanced": fast, then only the lines it was
            unsure of heard again by the accurate model — most of the quality
            for a little more than fast. The user's trade to make: on a
            low-tier computer, or a mid-tier one with more than about an hour
            of footage, call with `dry_run` first and ask them with the
            times. A transcript asked to be better is redone without `again`
            (fast to balanced hears only the unsure lines again); one is
            never made worse.
        dry_run: Start nothing; say what would start and how long it would
            take, as `estimates` for "accurate", "balanced" and "fast", so
            the user can choose before waiting on any.

    Returns:
        A dictionary with `jobs`, one `asset_id`, `job_id`, `status` and
        `stage` per job started, and `skipped`, each asset left alone with
        its `reason`: its analysis is current, or a job already queued or
        running for it, named. Analyses run one or two at a time, so that
        several of them cannot exhaust the machine's memory between them;
        `stage` says what a job that has not started yet is waiting for.
        Waiting costs nothing and needs no action: keep polling `get_job`.
        `upgrading` lists the files whose fast transcript is being replaced
        with an accurate one. With `dry_run`, `would_start` lists the files
        in place of `jobs`.

        Files in the music library are beat-tracked and never transcribed.
        `music` lists the ones started, or with `dry_run` the ones that
        would be, each with how much of what was listened to is speech; one
        that is mostly speech, or named like a narration, may have been filed
        there by mistake — ask the user, and move it with `set_asset_library`
        before analyzing it. A file never sorted into a library is sorted
        first; `unsorted` lists any that could not be, with why — one with no
        picture is then analyzed as music.

        When anything started is transcribed, `estimate` says how long that
        takes here for all of it together: `footage_minutes`,
        `transcription_minutes` and `speakers_minutes` as `[low, high]` (high
        null when there is no telling; `speakers_minutes` null when voices are
        not told apart), the computer's `tier`, and `basis` — this computer's
        own past runs, or a rough figure from its hardware — and a `note`
        when upgraded files are analyzed whole. Tell the user in a sentence
        when it runs past a few minutes.

    Raises:
        ValueError: If an asset does not exist or has no known duration;
            nothing is started then.
    """
    assets = [shared.get_asset(asset_id) for asset_id in dict.fromkeys(asset_ids)]
    for asset in assets:
        if asset.duration is None:
            raise ValueError(f"asset {asset.id} has no known duration and cannot be analyzed")

    if dry_run:
        choices = {}
        for choice in ("accurate", "balanced", "fast"):
            to_start, skipped, upgrading = plan_analyses(assets, transcribe, choice, again)
            choices[choice] = _estimate(to_start, transcribe, diarize, upgrading)
        to_start, skipped, upgrading = plan_analyses(assets, transcribe, transcription, again)
        result = {"would_start": [asset.id for asset, _ in to_start], "skipped": skipped, "upgrading": upgrading,
                  "estimates": choices}
        music = _music_heard(asset for asset, _ in to_start)
        if music:
            result["music"] = music
        return result

    unsorted = []
    for asset in assets:
        if asset.library is None:
            # Sorted when it was added, or when the server started; listened to now if neither managed.
            try:
                asset.library = classify(asset)
            except (OSError, RuntimeError, ValueError) as error:
                unsorted.append({"asset_id": asset.id, "reason": str(error)})
                continue
            runtime.repo.save_asset(asset)

    started: List[dict] = []
    to_start, skipped, upgrading = plan_analyses(assets, transcribe, transcription, again)
    for asset, chosen in to_start:
        job = start_analysis(asset, transcribe, chosen, language, prompt, chinese_variant, diarize, speakers)
        started.append({"asset_id": asset.id, "job_id": job.job_id, "status": job.status.value, "stage": job.stage})
    result: dict = {"jobs": started, "skipped": skipped, "upgrading": upgrading}
    if unsorted:
        result["unsorted"] = unsorted
    music = _music_heard(asset for asset, _ in to_start)
    if music:
        result["music"] = music
    found = _estimate(to_start, transcribe, diarize, upgrading)
    if found is not None:
        result["estimate"] = found
    return result

def _music_heard(assets: Iterable[Asset]) -> List[dict]:
    """Say, for each file in the music library, how much of it is somebody talking.

    Args:
        assets: The files about to be analyzed.

    Returns:
        One entry per music file: its `asset_id`, `name`, `speech_share`
        (null when its sound could not be read), and `sounds_like_speech`
        when that share is what would have filed it as footage.
    """
    heard = []
    for asset in assets:
        if not asset.has_audio or library_of(asset) != MUSIC:
            continue
        share = speech_share(asset.path, float(asset.duration or 0))
        entry = {"asset_id": asset.id, "name": os.path.basename(asset.path),
                 "speech_share": None if share is None else round(share, 2)}
        if share is not None and share >= SPEECH_SHARE:
            entry["sounds_like_speech"] = True
        heard.append(entry)
    return heard

# Provisional (roadmap §13): the longest stretch heard again at once, short enough to come back
# inside the time most AI clients allow a tool call.
LISTEN_AGAIN_SECONDS = 30.0

def bare_words(text: str) -> str:
    """Text reduced to what is said, for comparing two hearings: no spaces or punctuation."""
    return re.sub(r"[\s\W_]+", "", text)

@mcp.tool()
def listen_again(
    start: float,
    end: float,
    asset_id: Optional[str] = None,
    language: Optional[str] = None,
    job_id: Optional[str] = None,
    keep: bool = False,
) -> dict:
    """Hear one stretch of a file or of a finished render again, to check a word, a line or the mix.

    The transcript is one long pass, and a word it misheard — 「沒那麼順」
    written as 「怎麼那麼順」 — is misheard for everything that reads it. This
    transcribes only that stretch of sound, from scratch: no prompt, no text
    before it. Where the two disagree, or the words come back unsure, believe
    neither — give the user `audio_path` to listen to and ask.

    When a cut is refused for landing in a word the transcript placed wrong,
    hear that stretch with `keep`: its word times are then what cuts in this
    file are judged and moved by, in place of the transcript's. Only what was
    heard is kept, never a time you write; the plans stay valid. A hearing
    that does not say the same words as the transcript, or is unsure of any,
    is not kept — `not_kept` says why; play `audio_path` to the user and ask.
    Start and end the stretch in a pause, so it holds whole words.

    Given a render's `job_id` instead of a file, it hears what the viewer
    will, a shot at a time so the words after a cut are not lost to the ones
    before it: whether a word was cut into at an edit, and, from `levels`, whether
    the music dips under speech and two songs really cross-fade where the
    plan says.

    Args:
        start: Where the stretch starts, in seconds of the file or the render.
        end: Where it ends; at most 30 seconds after `start`.
        asset_id: The file to hear. Give this or `job_id`.
        language: Spoken language code; omit to detect it.
        job_id: A finished render to hear, preview or full.
        keep: For a file: keep these word times over the transcript's.

    Returns:
        A dictionary with `heard`, the new hearing with unsure words in ⟦ ⟧;
        `words`, each with `start`, `end` and its `probability`; `levels`,
        the sound's level in dBFS every `every_seconds` from `start` — speech
        sits well above a ducked bed, a gap is near -60; `audio_path`, a WAV of
        the stretch for the user to play; for a file, `stored`, what the
        transcript says there, or null, `agrees`, whether the two say the
        same words — never when the hearing has the marks of a line the
        recogniser makes up (a subtitle credit, a sign-off, words that take
        no time, one word looping), and `made_up` is then true — and, when
        `keep` was given, `kept` or `not_kept` — kept word by word where
        only some of it was heard alike and for sure, with `kept_words` and
        `left_as_transcribed`; for a render, `captions`, the ones on screen in
        the stretch, `changed_since`, true when the project has been edited
        since it was rendered, and `unheard`, where it is loud enough to be
        talking with no word heard — hear that stretch again on its own, or
        play `audio_path`; it may be a song.

    Raises:
        ValueError: If neither or both of a file and a render are given, it
            has no sound, or the stretch is empty, too long or outside it.
    """
    if (asset_id is None) == (job_id is None):
        raise ValueError("give either asset_id, to hear a file, or job_id, to hear a render")
    if keep and job_id is not None:
        raise ValueError("only what a file says is kept; a render is the cut, and is heard to check it")
    if job_id is not None:
        return _listen_to_render(job_id, start, end, language)
    asset = shared.get_asset(asset_id)
    if not asset.has_audio:
        raise ValueError(f"asset {asset_id} has no sound to hear again")
    _hearable(start, end)
    if asset.duration is not None and start >= float(asset.duration):
        raise ValueError(f"the file is only {float(asset.duration):g}s long")
    known = runtime.repo.get_analysis(asset_id)
    transcript = known.transcript if known is not None else None
    if keep and transcript is None:
        raise ValueError(f"asset {asset_id} has no transcript to correct; analyze it first")
    keep_as = os.path.join(runtime.WORKSPACE_DIR, "temp", "listen", f"{asset_id[:8]}-{start:.2f}-{end:.2f}.wav")
    fresh = listen_again_in_file(
        asset.path, start, end, language or (transcript.language if transcript else None), keep_as,
        chinese_variant=transcript.chinese_variant if transcript else None,
    )
    stored = None
    if transcript is not None:
        said = [word.text for segment in transcript.segments for word in segment.words
                if word.end > start and word.start < end]
        stored = "".join(said).strip() or None
    plain_heard = "".join(segment.text for segment in fresh.segments)
    # The same recogniser makes the same lines up over the same café hum, so two hearings
    # of a made-up line agreeing says nothing about whether anyone said it.
    made_up = any(was_made_up(segment) for segment in fresh.segments)
    result = {
        **_heard(fresh, keep_as, start),
        "stored": stored,
        "agrees": not made_up and stored is not None and bare_words(plain_heard) == bare_words(stored),
    }
    if made_up:
        result["made_up"] = True
    if keep:
        heard_words = [word for segment in fresh.segments for word in segment.words]
        result.update(_keep_hearing(known, HeardAgain(start=start, end=end, words=heard_words), result["agrees"],
                                    made_up))
    return result

def _alike_times(
    stored: List[TranscriptWord], heard: List[TranscriptWord],
) -> Tuple[List[TranscriptWord], List[str], List[str], List[str]]:
    """Take the heard times of the words two hearings agree on, and the transcript's for the rest.

    Matched character by character, so two hearings that cut a line into
    words differently still line up. A word takes its heard time only when
    every character of it was heard the same, all of them in one heard word
    holding nothing else, that word was heard for sure, and the time leaves
    the words either side in order.

    Args:
        stored: The transcript's words in the stretch, in order.
        heard: The words heard again there, in order.

    Returns:
        `(words, apart, doubted, retimed)`: the transcript's words, each with
        the heard time where the two agree; the transcript's words they
        disagree on; those they agree on but the hearing was unsure of, which
        keep the transcript's time; and those that took the heard time.
    """
    def characters(words: List[TranscriptWord]) -> Tuple[str, List[int]]:
        text, owner = [], []
        for index, word in enumerate(words):
            for character in bare_words(word.text):
                text.append(character)
                owner.append(index)
        return "".join(text), owner

    said, said_by = characters(stored)
    again, again_by = characters(heard)
    paired: Dict[int, set] = {}
    claimed: Dict[int, set] = {}
    matched = set()
    for block in difflib.SequenceMatcher(a=said, b=again, autojunk=False).get_matching_blocks():
        for step in range(block.size):
            one, other = said_by[block.a + step], again_by[block.b + step]
            matched.add(block.a + step)
            paired.setdefault(one, set()).add(other)
            claimed.setdefault(other, set()).add(one)
    owned = {index: sum(1 for owner in said_by if owner == index) for index in range(len(stored))}
    out: List[TranscriptWord] = list(stored)
    retimed: set = set()
    apart: List[str] = []
    doubted: List[str] = []
    for index, word in enumerate(stored):
        others = paired.get(index, set())
        whole = sum(1 for at, owner in enumerate(said_by) if owner == index and at in matched) == owned[index]
        if not owned[index]:
            continue
        if not (whole and len(others) == 1):
            apart.append(word.text.strip())
            continue
        match = heard[next(iter(others))]
        alone = claimed.get(next(iter(others))) == {index} and len(bare_words(match.text)) == owned[index]
        if alone and unsure_words([match]):
            # Heard alike but not for sure: its time is no better than the transcript's (#98).
            doubted.append(word.text.strip())
        elif alone:
            out[index] = word.model_copy(update={"start": match.start, "end": match.end,
                                                 "probability": match.probability})
            retimed.add(index)
    # Two neighbours whose times now overlap — one heard again, one not, or two heard apart —
    # are put back as the transcript had them, until the words run in order again.
    changed = True
    while changed:
        changed = False
        for index in range(len(out) - 1):
            if out[index].end > out[index + 1].start + EDGE_TOLERANCE_SECONDS and {index, index + 1} & retimed:
                for undo in {index, index + 1} & retimed:
                    out[undo] = stored[undo]
                    retimed.discard(undo)
                changed = True
    return out, apart, doubted, [stored[index].text.strip() for index in sorted(retimed)]

def _keep_hearing(known: MediaAnalysis, stretch: HeardAgain, agrees: bool, made_up: bool = False) -> dict:
    """Keep a stretch heard again over the transcript's word times, when it can be believed.

    Args:
        known: The file's analysis the hearing was taken against.
        stretch: What was heard, and where.
        agrees: Whether the hearing says the same words as the transcript there.
        made_up: Whether the hearing is one the recogniser makes up, in which
            case none of it is believed.

    Returns:
        `kept` when it was kept, or `not_kept` saying why it was not. Where the
        two say different words, the times of the ones they agree on are
        kept and the rest stay the transcript's (#87).
    """
    # A word half in the stretch would be kept twice: its outside part from the transcript,
    # and the whole of it as heard. The stretch is moved to take it in or leave it out.
    across = [word for segment in known.transcript.segments for word in segment.words
              if word.start < stretch.start < word.end or word.start < stretch.end < word.end]
    if across:
        opens = min([stretch.start, *(word.start for word in across if word.start < stretch.start < word.end)])
        closes = max([stretch.end, *(word.end for word in across if word.start < stretch.end < word.end)])
        # Rounded outward, so the stretch said takes the word in however long the file is; snapped
        # first, so a time already in whole milliseconds is not pushed into the next word.
        opens, closes = math.floor(round(opens * 1000, 6)) / 1000, math.ceil(round(closes * 1000, 6)) / 1000
        return {"not_kept": f"the transcript has {'、'.join(word.text.strip() for word in across)} running across the "
                            f"edge of the stretch; hear {opens:.3f}s to {closes:.3f}s again with keep"}
    partly, listed = "", {}
    unsure = unsure_words(stretch.words)
    if not agrees or unsure:
        # Word by word: the ones heard alike and for sure take the heard time, and the rest
        # keep the transcript's — a word in doubt costs only itself, not the stretch (#87, #98).
        stored = [word for segment in known.transcript.segments for word in segment.words
                  if word.start >= stretch.start and word.end <= stretch.end]
        alike, apart, doubted, retimed = ([], [], [], []) if made_up else _alike_times(stored, stretch.words)
        if not retimed:
            if agrees:
                return {"not_kept": f"the hearing was unsure of {'、'.join(unsure)}; play audio_path to the user "
                                    "and ask"}
            return {"not_kept": "the hearing and the transcript say different words here, so neither is believed; "
                                "play audio_path to the user and ask, or hear a stretch that starts and ends in a pause"}
        stretch = stretch.model_copy(update={"words": alike})
        said = [f" for the {len(retimed)} word(s) the two heard alike and the hearing was sure of"]
        if apart:
            said.append(f"「{'、'.join(apart)}」, heard differently, keep the transcript's words and times")
        if doubted:
            said.append(f"「{'、'.join(doubted)}」, heard unsure, keep the transcript's times")
        partly = "; ".join(said) + (" — play audio_path to the user and ask about those" if apart or doubted else "")
        listed = {"kept_words": retimed, "left_as_transcribed": apart + doubted}
    # Read again before writing, so an analysis taken again meanwhile is not overwritten
    # with the one this started from.
    current = runtime.repo.get_analysis(known.asset_id)
    if current is None or current.analyzed_at != known.analyzed_at:
        return {"not_kept": "the file was analyzed again while this was heard; hear the stretch again"}
    current.heard_again = [
        *(item for item in current.heard_again if item.end <= stretch.start or item.start >= stretch.end), stretch,
    ]
    runtime.repo.save_analysis(current)
    return {"kept": f"the word times heard from {stretch.start:g}s to {stretch.end:g}s are now what cuts in this "
                    f"file are judged by{partly}; check the cut again", **listed}

# Provisional (roadmap §13): how long each level `listen_again` reports covers — fine
# enough to see a fade or a dip under a word, coarse enough that thirty seconds is sixty numbers.
LEVEL_EVERY_SECONDS = 0.5

def _heard(fresh: Transcript, keep_as: str, start: float) -> dict:
    """What `listen_again` says about any stretch it heard.

    Args:
        fresh: The new hearing, in seconds of the file or render.
        keep_as: The stretch's sound, kept for the user to play.
        start: Where the stretch starts, which the levels count from.

    Returns:
        `heard`, `words`, `levels` and `audio_path`.
    """
    return {
        "heard": "".join(marked_text(segment) for segment in fresh.segments),
        "words": [
            {"text": word.text.strip(), "start": word.start, "end": word.end, "probability": word.probability}
            for segment in fresh.segments for word in segment.words
        ],
        "levels": {"start": start, "every_seconds": LEVEL_EVERY_SECONDS,
                   "db": levels_every(keep_as, LEVEL_EVERY_SECONDS)},
        "audio_path": keep_as,
    }

def _hearable(start: float, end: float) -> None:
    """Refuse a stretch `listen_again` cannot take: empty, or longer than it hears at once.

    Raises:
        ValueError: Saying which.
    """
    if end <= start:
        raise ValueError(f"the stretch has to end after it starts ({start}s to {end}s)")
    if end - start > LISTEN_AGAIN_SECONDS:
        raise ValueError(f"hear a stretch of at most {LISTEN_AGAIN_SECONDS:g} seconds at a time; this one is "
                         f"{end - start:g}s")

# A stretch of a render this loud for this long with no word heard in it is very likely
# somebody talking that the speech model missed — or a song, which only listening tells
# apart. Speech in a finished render sits near -20 dBFS and a ducked bed well under -30.
# Provisional, both.
UNHEARD_LOUD_DB = -28.0
UNHEARD_LOUD_SECONDS = 1.0
# A piece of a render shorter than this either side of a cut is heard with the shot it
# belongs to rather than on its own: too short to hold a word. Provisional.
MIN_HEARD_SECONDS = 0.3

def _listen_to_render(job_id: str, start: float, end: float, language: Optional[str]) -> dict:
    """`listen_again` over a finished render rather than a file.

    Args:
        job_id: The render.
        start: Where the stretch starts, in seconds of the render.
        end: Where it ends.
        language: Spoken language code, or None for the footage's own.

    Returns:
        What `_heard` says, with the `captions` on screen in the stretch and
        `changed_since`.
    """
    job, path = rendering.finished_render(job_id)
    _hearable(start, end)
    project = rendering.rendered_from(job)
    if project is not None and not rendering.changed_since(job) and start >= float(project.duration):
        raise ValueError(f"the render is only {float(project.duration):g}s long")
    keep_as = os.path.join(runtime.WORKSPACE_DIR, "temp", "listen", f"render-{job_id[:8]}-{start:.2f}-{end:.2f}.wav")
    known = None
    for clip in (project.base_video_track.clips if project and project.base_video_track else []):
        analysis = runtime.repo.get_analysis(clip.asset_id)
        if analysis is not None and analysis.transcript is not None:
            known = analysis.transcript
            break
    # Heard a shot at a time: the speech model, given a stretch across a cut, wrote down the
    # words before it and nothing after, though the voice there was loud and clear.
    cuts = sorted({float(clip.timeline_in) for clip in project.base_video_track.clips} if (
        project is not None and project.base_video_track is not None and not rendering.changed_since(job)) else set())
    edges = [start] + [at for at in cuts if start + MIN_HEARD_SECONDS <= at <= end - MIN_HEARD_SECONDS] + [end]
    language = language or (known.language if known else None)
    variant = known.chinese_variant if known else None
    # The words are heard from the cut's own sound without the music under it, which turned
    # 「買一送四」 into 「買壽司」 sixteen decibels down (#89); the user plays the mix as rendered.
    voice = _voice_of_render(job, project)
    heard_from = voice or path
    stem, extension = os.path.splitext(keep_as)
    pieces = []
    for index, (opened, closed) in enumerate(zip(edges, edges[1:])):
        # Each shot's own sound is only what the model is handed; the user plays the whole stretch.
        piece_as = f"{stem}-{index}{extension}"
        pieces.append(listen_again_in_file(heard_from, opened, closed, language, piece_as, chinese_variant=variant))
        os.remove(piece_as)
    cut_sound(path, start, end, keep_as)
    fresh = pieces[0].model_copy(update={"segments": [segment for piece in pieces for segment in piece.segments]})
    shown = [cue for cue in (place_cues(project, project.subtitles) if project and job.captioned else [])
             if float(cue.end) > start and float(cue.start) < end]
    result = {
        **_heard(fresh, keep_as, start),
        "captions": [f"{format_timestamp(float(cue.start))} {cue.text}" for cue in shown],
        "changed_since": rendering.changed_since(job),
    }
    if voice is not None:
        result["heard_without_music"] = True
        # Loud without words is judged on the talking alone, so a song is not taken for a voice.
        spoken = os.path.join(runtime.WORKSPACE_DIR, "temp", "listen", f"render-{job_id[:8]}-voice-{start:.2f}-{end:.2f}.wav")
        cut_sound(voice, start, end, spoken)
        loud = levels_every(spoken, LEVEL_EVERY_SECONDS)
        os.remove(spoken)
    else:
        loud = result["levels"]["db"]
    unheard = loud_without_words(loud, start, fresh)
    if unheard:
        result["unheard"] = ", ".join(f"{opened:.1f}–{closed:.1f}s" for opened, closed in unheard)
    return result

def _voice_of_render(job: Job, project: Optional[Project]) -> Optional[str]:
    """Make, once, the sound of a render's own footage without its music, to hear words from.

    Args:
        job: The render.
        project: The project it was rendered from.

    Returns:
        The path of the stem, kept beside the render's work; None when the
        project has changed since, so the stem would not be what was
        rendered, when nothing plays under the footage, or when it cannot be
        made — the render itself is heard then.
    """
    if project is None or rendering.changed_since(job) or not job.work_dir:
        return None
    voice = os.path.join(job.work_dir, "voice.wav")
    if os.path.exists(voice):
        return voice
    audio = [track for track in project.tracks if track.track_type == TrackType.AUDIO and track.clips]
    if not audio:
        return None
    # The stem is the footage's own sound: a narration on a track of its own, as the render
    # decides one, is not in it.
    voices = rendering.voices_of(project)
    if voices:
        return None
    os.makedirs(job.work_dir, exist_ok=True)
    music, mix = os.path.join(job.work_dir, "music.wav"), os.path.join(job.work_dir, "sound.m4a")
    try:
        command = runtime.renderer.build_sound(project, shared.referenced_assets(project), mix, voice, music, None,
                                               voices=voices, talking=rendering.talking_of(project, voices))
        done = subprocess.run(command, capture_output=True, creationflags=hidden_window_flags())
    except (OSError, ValueError):
        return None
    for leftover in (music, mix):
        if os.path.exists(leftover):
            os.remove(leftover)
    return voice if done.returncode == 0 and os.path.exists(voice) else None

def loud_without_words(levels: Sequence[float], start: float, fresh: Transcript) -> List[Tuple[float, float]]:
    """Find where a render is loud enough to be somebody talking and the hearing has no word.

    Args:
        levels: The sound's level every `LEVEL_EVERY_SECONDS`, in dBFS.
        start: Where the levels start, in seconds of the render.
        fresh: The hearing.

    Returns:
        Each such stretch as `(start, end)` in seconds of the render, at
        least `UNHEARD_LOUD_SECONDS` long.
    """
    said = [(word.start, word.end) for segment in fresh.segments for word in segment.words]
    found: List[Tuple[float, float]] = []
    opened = None
    for index, level in enumerate([*levels, None]):
        at = start + index * LEVEL_EVERY_SECONDS
        loud = level is not None and level >= UNHEARD_LOUD_DB and not any(
            begins < at + LEVEL_EVERY_SECONDS and ends > at for begins, ends in said)
        if loud and opened is None:
            opened = at
        elif not loud and opened is not None:
            if at - opened >= UNHEARD_LOUD_SECONDS:
                found.append((opened, at))
            opened = None
    return found

@mcp.tool()
def get_analysis(
    asset_id: str,
    start: Annotated[float, Field(ge=0)] = 0,
    end: Optional[float] = None,
    include_words: bool = False,
) -> dict:
    """Return the analysis of an asset for a time range.

    Only items overlapping the range are returned. Transcripts of long
    recordings are large, so read them in windows of about 10 minutes. Times
    are seconds in the source file and can be used directly as a clip's
    `source_range`.

    Args:
        asset_id: ID of the analyzed asset.
        start: Range start in seconds.
        end: Range end in seconds; omit for the end of the asset.
        include_words: Whether to include word-level timings in transcript
            segments. Use them to cut exactly at a word; leave them out to
            save space.

    Returns:
        A dictionary with the asset's `notes` where it has them — how it may
        be used, which every cut from it has to respect — its `duration`, `analyzed_at`,
        `transcription` (`fast` — expect misheard words, check them before
        trusting a cut to a word — `balanced`, `accurate`, or null), `doubtful_spots`
        (how many stretches in the whole file the speech model was unsure
        of; null for a transcript made before that was kept), and the
        overlapping `scenes`, `black_frames`, `frozen_frames`, `silences`,
        `shots`, and `transcript` (`language`, `model`, `chinese_variant`, and
        `segments` with `start`, `end`, and `text`), or null for `transcript`
        if speech was not transcribed.

        In `text`, words the speech model was unsure of are wrapped in ⟦ ⟧.
        The marks are only here: captions and the video never carry them, so
        never copy them into anything the user will see. Read the marked
        stretches with what the footage is about in mind — names and terms
        are what it mishears — and fix what is wrong through
        `generate_subtitles`' `fix_words` or by correcting the caption; check
        by listening before trusting a cut to a marked word.

        `shots` carries one record per shot in the range — `exposure`,
        `contrast`, `blur`, `motion` and `shake` — while `sound` and `faces`
        summarise the range's per-second records rather than listing them,
        since an hour of those is thousands of rows. `faces` says how much of
        the range had anybody on screen, how many there usually were, and
        where the largest face sat across the frame and how far it moved,
        which is what a vertical reframe needs. `speakers` lists the stretches
        each voice held; the labels are this file's own and mean nothing
        outside it. `rhythm` is for a file in the music library, and gives
        the `tempo` in beats per minute and the `beats` in the range; its `tempo` is null when the music has no steady pulse, and
        the whole of it is null for footage, or for music analyzed before
        beats were measured. `music`, for a song, has the `moods` and
        `styles` that set it apart in the library (how it sounds to a model,
        right about half the time on mood), its `sections` of quiet, middle
        and full energy, where it is `fullest`, and how it `ends`.

        All of these are measurements and none of them is a verdict: whether a
        shot is too dark or too wobbly depends on what it is for.

        `recipe` says what measured it, and `stale` is true when the server
        has since changed how it measures. To pick material out of this, use
        `query_clips`: every measurement here is also averaged onto the
        semantic clips, where it can be filtered on.

    Raises:
        ValueError: If the asset has not been analyzed yet.
    """
    analysis = runtime.repo.get_analysis(asset_id)
    if analysis is None:
        raise ValueError(f"asset {asset_id} has not been analyzed; call analyze_asset first")
    end = analysis.duration if end is None else end

    transcript = None
    if analysis.transcript is not None:
        exclude = None if include_words else {"words"}
        transcript = {
            "language": analysis.transcript.language,
            "model": analysis.transcript.model,
            "chinese_variant": analysis.transcript.chinese_variant,
            "segments": [
                {**segment.model_dump(exclude=exclude), "text": marked_text(segment)}
                for segment in analysis.transcript.segments
                if segment.end > start and segment.start < end
            ],
        }
    notes = shared.get_asset(asset_id).notes
    return {
        "asset_id": asset_id,
        **({"notes": notes} if notes else {}),
        "duration": analysis.duration,
        "analyzed_at": analysis.analyzed_at.isoformat(),
        "recipe": analysis.recipe.model_dump(),
        "stale": is_stale(analysis),
        "transcription": transcription_kind(analysis),
        "doubtful_spots": None if analysis.transcript is None else doubtful_spots(analysis.transcript.segments),
        "range": {"start": start, "end": end},
        "scenes": shared.overlapping(analysis.scenes, start, end),
        "black_frames": shared.overlapping(analysis.black_frames, start, end),
        "frozen_frames": shared.overlapping(analysis.frozen_frames, start, end),
        "silences": shared.overlapping(analysis.silences, start, end),
        "shots": shared.overlapping(analysis.shots, start, end),
        "sound": sound_note(shared.covering(analysis.sound, start, end)),
        "faces": framing_note(shared.covering(analysis.faces, start, end)),
        "speakers": shared.overlapping(analysis.speakers, start, end),
        "rhythm": None if analysis.rhythm is None else {
            "tempo": analysis.rhythm.tempo,
            "beats": [beat for beat in analysis.rhythm.beats if start <= beat < end],
        },
        "music": None if analysis.music is None or library_of(shared.get_asset(asset_id)) != MUSIC else {
            **song_tags().get(asset_id, {}),
            **music_energy.summary(analysis.music.energy),
        },
        "transcript": transcript,
    }
