"""The evidence pack a judge reads about one finished film, made from the film alone.

    python benchmark/evidence.py <film.mp4> <output folder>

It does not read the project or the plan, so an AI's film, a hand-cut film and
the baseline are treated alike. See rubric.md for what is judged.

The judge cannot hear, so the film's sound comes as numbers and spectrograms it
can read: this is the sound described, not the sound listened to.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.engine.sound_quality import RATE, measure, read_sound, spectrogram

# Provisional: how different two frames must be to count as a cut; FFmpeg's scene score from 0 to 1.
SCENE_THRESHOLD = 0.15
OVERVIEW_SECONDS = 6
MAX_CUT_PAIRS = 40
# Provisional: a pause shorter than this between two words is the inside of a sentence.
SENTENCE_PAUSE_SECONDS = 0.25
# Provisional: the shortest black stretch and frozen stretch counted, in seconds.
BLACK_SECONDS = 0.3
FROZEN_SECONDS = 1
# Provisional: how far inside a word a cut must land to count, since word times are approximate.
WORD_EDGE_SECONDS = 0.03
# Provisional: the stretch each row of the sound table covers, and each spectrogram.
SOUND_WINDOW_SECONDS = 10
SPECTROGRAM_SECONDS = 60
# Provisional: how much sound either side of a cut is compared, and how big a change in level
# a listener would catch as a jump, in decibels.
CUT_SOUND_SECONDS = 0.5
CUT_JUMP_DB = 6.0
# Provisional: what counts as silence in the film, by level and by length.
SILENCE_DBFS = -40
SILENCE_SECONDS = 1
LEVEL_FLOOR_DBFS = -96.0


def measured(film: str) -> Dict[str, object]:
    """Length, loudness, bad picture and cuts."""
    length = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                   film], capture_output=True, text=True, check=True).stdout.strip())
    log = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", film,
         "-af", f"ebur128=peak=true,silencedetect=n={SILENCE_DBFS}dB:d={SILENCE_SECONDS}",
         "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    cuts = [round(float(found), 3) for found in re.findall(r"Parsed_showinfo.*?pts_time:([\d.]+)", log)]
    summary = log.rsplit("Summary:", 1)[-1]
    loudness = re.findall(r"I:\s+(-?[\d.]+) LUFS", summary)
    loudness_range = re.findall(r"LRA:\s+(-?[\d.]+) LU", summary)
    peak = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    return {
        "seconds": round(length, 2),
        "integrated_lufs": float(loudness[0]) if loudness else None,
        "loudness_range_lu": float(loudness_range[0]) if loudness_range else None,
        "true_peak_dbfs": float(peak[0]) if peak else None,
        "silences": silences(log, length),
        "cuts": cuts,
        "shots": len(cuts) + 1,
        "average_shot_seconds": round(length / (len(cuts) + 1), 2),
    }


def silences(log: str, length: float) -> List[List[float]]:
    """The stretches FFmpeg found silent; one still running at the end runs to the end of the film."""
    starts = [float(found) for found in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(found) for found in re.findall(r"silence_end: (-?[\d.]+)", log)]
    return [[round(max(start, 0), 2), round(ends[index] if index < len(ends) else length, 2)]
            for index, start in enumerate(starts)]


def bad_picture(film: str) -> Dict[str, int]:
    """Stretches of black and of frozen picture, which need every frame rather than only the cuts."""
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", film, "-an",
                          "-vf", f"blackdetect=d={BLACK_SECONDS},freezedetect=d={FROZEN_SECONDS}", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    return {"black_stretches": log.count("black_start"), "frozen_stretches": log.count("freeze_start")}


def transcript(film: str) -> List[dict]:
    """The words said in the film, with their times."""
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3-turbo", device="auto", compute_type="auto")
    segments, _ = model.transcribe(film, language="zh", word_timestamps=True)
    return [{"start": round(segment.start, 2), "end": round(segment.end, 2), "text": segment.text,
             "words": [(word.start, word.end) for word in segment.words or []]} for segment in segments]


def cut_signals(cuts: List[float], lines: List[dict]) -> Dict[str, object]:
    """How many cuts land in a word, or in the middle of a sentence."""
    words = [word for line in lines for word in line["words"]]
    in_word, in_sentence = [], []
    for cut in cuts:
        if any(start + WORD_EDGE_SECONDS < cut < end - WORD_EDGE_SECONDS for start, end in words):
            in_word.append(cut)
        for line in lines:
            pairs = list(zip(line["words"], line["words"][1:]))
            if any(one[1] <= cut <= two[0] and two[0] - one[1] < SENTENCE_PAUSE_SECONDS for one, two in pairs):
                in_sentence.append(cut)
    return {"cuts_inside_a_word": in_word, "cuts_inside_a_sentence": in_sentence}


def level(samples: np.ndarray) -> float:
    """The level of some samples in dBFS."""
    if not len(samples):
        return LEVEL_FLOOR_DBFS
    return round(max(10 * float(np.log10(max(float(np.mean(samples ** 2)), 1e-12))), LEVEL_FLOOR_DBFS), 1)


def sound_table(samples: np.ndarray) -> List[dict]:
    """How each stretch of the film sounds, measured."""
    size = SOUND_WINDOW_SECONDS * RATE
    return [{"start": at // RATE, "level_dbfs": level(samples[at:at + size]), **measure(samples[at:at + size])}
            for at in range(0, len(samples), size)]


def sound_jumps(samples: np.ndarray, cuts: List[float]) -> List[dict]:
    """Cuts where the level changes enough for a listener to catch it."""
    size = round(CUT_SOUND_SECONDS * RATE)
    found = []
    for cut in cuts:
        at = round(cut * RATE)
        before, after = level(samples[max(at - size, 0):at]), level(samples[at:at + size])
        if abs(after - before) >= CUT_JUMP_DB:
            found.append({"at": cut, "before_dbfs": before, "after_dbfs": after})
    return found


def spectrograms(film: str, length: float, folder: Path) -> None:
    """The film's sound drawn a minute at a time."""
    for index, start in enumerate(range(0, int(length) + 1, SPECTROGRAM_SECONDS)):
        if start < length:
            drawn = spectrogram(film, start, min(SPECTROGRAM_SECONDS, length - start))
            (folder / f"sound-{index:02d}.png").write_bytes(drawn)


def overview(film: str, folder: Path) -> None:
    """One frame every few seconds, as sheets of twenty."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", film, "-vf",
                    f"fps=1/{OVERVIEW_SECONDS},scale=320:-2,tile=5x4", str(folder / "overview-%02d.jpg")], check=True)


def cut_pairs(film: str, cuts: List[float], length: float, folder: Path) -> None:
    """The frame before and after each cut, spread over the film when there are many."""
    usable = [cut for cut in cuts if cut + 0.2 < length]
    picked = sorted({round(index * len(usable) / MAX_CUT_PAIRS) for index in range(MAX_CUT_PAIRS)})         if len(usable) > MAX_CUT_PAIRS else range(len(usable))
    chosen = [usable[index] for index in picked]
    (folder / "cuts.json").write_text(json.dumps(chosen), encoding="utf-8")
    for index, cut in enumerate(chosen):
        for name, at in (("a", max(cut - 0.1, 0)), ("b", cut + 0.1)):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", film, "-frames:v", "1",
                            "-vf", "scale=320:-2", str(folder / f"cut-{index:02d}{name}.jpg")], check=True)

def main() -> None:
    film, folder = sys.argv[1], Path(sys.argv[2])
    folder.mkdir(parents=True, exist_ok=True)
    numbers = {**measured(film), **bad_picture(film)}
    lines = transcript(film)
    numbers.update(cut_signals(numbers["cuts"], lines))
    samples = read_sound(film, 0, numbers["seconds"])
    numbers["sound_jumps_at_cuts"] = sound_jumps(samples, numbers["cuts"])
    (folder / "numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / "transcript.txt").write_text(
        "\n".join(f"{line['start']:7.2f}–{line['end']:7.2f} {line['text'].strip()}" for line in lines), encoding="utf-8")
    (folder / "sound.json").write_text(json.dumps(sound_table(samples), ensure_ascii=False, indent=1), encoding="utf-8")
    spectrograms(film, numbers["seconds"], folder)
    overview(film, folder)
    cut_pairs(film, numbers["cuts"], numbers["seconds"], folder)


if __name__ == "__main__":
    main()
