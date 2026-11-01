"""The evidence pack a judge reads about one finished film, made from the film alone.

    python benchmark/evidence.py <film.mp4> <output folder>

It does not read the project or the plan, so an AI's film, a hand-cut film and
the baseline are treated alike. See rubric.md for what is judged.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Provisional: how different two frames must be to count as a cut; FFmpeg's scene score from 0 to 1.
SCENE_THRESHOLD = 0.15
OVERVIEW_SECONDS = 6
MAX_CUT_PAIRS = 40
# Provisional: a pause shorter than this between two words is the inside of a sentence.
SENTENCE_PAUSE_SECONDS = 0.25


def measured(film: str) -> Dict[str, object]:
    """Length, loudness, bad picture and cuts."""
    length = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                   film], capture_output=True, text=True, check=True).stdout.strip())
    log = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", film,
         "-af", "ebur128", "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    cuts = [round(float(found), 3) for found in re.findall(r"Parsed_showinfo.*?pts_time:([\d.]+)", log)]
    loudness = re.findall(r"I:\s+(-?[\d.]+) LUFS", log.rsplit("Summary:", 1)[-1])
    return {
        "seconds": round(length, 2),
        "integrated_lufs": float(loudness[0]) if loudness else None,
        "cuts": cuts,
        "shots": len(cuts) + 1,
        "average_shot_seconds": round(length / (len(cuts) + 1), 2),
    }


def bad_picture(film: str) -> Dict[str, int]:
    """Stretches of black and of frozen picture, which need every frame rather than only the cuts."""
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", film, "-an",
                          "-vf", "blackdetect=d=0.3,freezedetect=d=1", "-f", "null", "-"],
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
        if any(start + 0.03 < cut < end - 0.03 for start, end in words):
            in_word.append(cut)
        for line in lines:
            pairs = list(zip(line["words"], line["words"][1:]))
            if any(one[1] <= cut <= two[0] and two[0] - one[1] < SENTENCE_PAUSE_SECONDS for one, two in pairs):
                in_sentence.append(cut)
    return {"cuts_inside_a_word": in_word, "cuts_inside_a_sentence": in_sentence}


def overview(film: str, folder: Path) -> None:
    """One frame every few seconds, as sheets of twenty."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", film, "-vf",
                    f"fps=1/{OVERVIEW_SECONDS},scale=320:-2,tile=5x4", str(folder / "overview-%02d.jpg")], check=True)


def cut_pairs(film: str, cuts: List[float], length: float, folder: Path) -> None:
    """The frame before and after each cut, spread over the film when there are many."""
    step = max(1, len(cuts) // MAX_CUT_PAIRS)
    chosen = [cut for cut in cuts[::step][:MAX_CUT_PAIRS] if cut + 0.2 < length]
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
    (folder / "numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / "transcript.txt").write_text(
        "\n".join(f"{line['start']:7.2f}–{line['end']:7.2f} {line['text'].strip()}" for line in lines), encoding="utf-8")
    overview(film, folder)
    cut_pairs(film, numbers["cuts"], numbers["seconds"], folder)


if __name__ == "__main__":
    main()
