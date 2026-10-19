"""A balanced transcription: the fast model over everything, the accurate one over the lines it was unsure of."""

from pathlib import Path

import pytest

from app import runtime, server
from app.tools import examining
from app.engine import analysis as analysis_module
from app.engine import speed
from app.engine.analysis import (
    BALANCED_JOIN, FAST_WHISPER_MODEL, REFINE_PAD_SECONDS, _unsure_stretches, refine_unsure_lines, transcription_of,
    whisper_model_for, whisper_model_name,
)
from app.models.media import MediaAnalysis, Transcript, TranscriptSegment, TranscriptWord
from test_speed import computer, repo  # noqa: F401  (the repo fixture)


def line(start: float, end: float, text: str, sure: float = 0.9) -> TranscriptSegment:
    step = (end - start) / len(text)
    return TranscriptSegment(start=start, end=end, text=text, words=[
        TranscriptWord(start=round(start + step * index, 3), end=round(start + step * (index + 1), 3),
                       text=character, probability=sure)
        for index, character in enumerate(text)
    ])


def fast(*segments: TranscriptSegment) -> Transcript:
    return Transcript(language="zh", model=FAST_WHISPER_MODEL, segments=list(segments))


def test_a_balanced_transcript_is_known_by_its_models() -> None:
    model = whisper_model_for("balanced")
    assert model == f"{FAST_WHISPER_MODEL}{BALANCED_JOIN}{whisper_model_name()}"
    assert transcription_of(model) == "balanced" and transcription_of(whisper_model_name()) == "accurate"


def test_only_the_unsure_lines_are_heard_again() -> None:
    segments = [line(0, 2, "今天天氣"), line(3, 5, "我們出發", sure=0.2), line(5.1, 6, "好嗎", sure=0.3),
                line(8, 9, "走吧")]
    # The two unsure lines next to each other are heard together, padded but never into a sure line.
    assert _unsure_stretches(segments, 10.0) == [(3 - REFINE_PAD_SECONDS, 6 + REFINE_PAD_SECONDS)]


def test_the_accurate_model_s_words_take_the_place_of_the_unsure_ones(monkeypatch, tmp_path: Path) -> None:
    import subprocess

    heard_at = []

    def accurate(device, path, duration, language, prompt, on_progress, is_cancelled, on_measured, model_name):
        heard_at.append((round(duration, 3), model_name))
        return Transcript(language="zh", model=model_name, segments=[line(0.3, 2.3, "我們出發")])

    monkeypatch.setattr(analysis_module, "_run_whisper", accurate)
    monkeypatch.setattr(analysis_module.subprocess, "run",
                        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""))
    transcript = fast(line(0, 2, "今天天氣"), line(3, 5, "我門出發", sure=0.2), line(7, 8, "走吧"))
    better = refine_unsure_lines("x.mp4", transcript, 10.0, None, [], "large-v3-turbo", lambda f: None,
                                 lambda: False, str(tmp_path))
    assert heard_at == [(2.6, "large-v3-turbo")]
    assert "".join(segment.text for segment in better.segments) == "今天天氣我們出發走吧"
    assert transcription_of(better.model) == "balanced"
    # Placed at the file's own seconds: the stretch began at 2.7.
    moved = next(word for segment in better.segments for word in segment.words if word.text == "我")
    assert moved.start == pytest.approx(3.0)


def test_a_transcript_is_never_made_worse(media: Path) -> None:
    asset = runtime.repo.get_assets([server.import_asset(str(media / "silent.mp4"))["id"]])
    asset = next(iter(asset.values())).model_copy(update={"has_audio": True})
    for had, asked, wanted in [("balanced", "fast", "balanced"), ("fast", "balanced", "balanced"),
                               ("balanced", "accurate", "accurate"), ("accurate", "balanced", "accurate")]:
        runtime.repo.save_analysis(MediaAnalysis(
            asset_id=asset.id, duration=4.0,
            transcript=Transcript(language="zh", model=whisper_model_for(had), segments=[]),
        ))
        to_start, _, upgrading = examining.plan_analyses([asset], True, asked, again=True)
        assert [chosen for _, chosen in to_start] == [wanted], (had, asked)
        assert bool(upgrading) == (examining.TRANSCRIPTION_RANK[asked] > examining.TRANSCRIPTION_RANK[had])


def test_balanced_costs_more_than_fast_and_less_than_accurate(repo) -> None:  # noqa: F811
    times = {choice: speed.estimate(repo, computer("mid"), {choice: 3600}, False)["transcription_minutes"][0]
             for choice in ("fast", "balanced", "accurate")}
    assert times["fast"] < times["balanced"] < times["accurate"]


def test_a_balanced_transcript_is_heard_again_with_its_accurate_model() -> None:
    from app.engine.analysis import heard_with
    assert heard_with("small+large-v3-turbo") == "large-v3-turbo"
    assert heard_with("small") == "small"


def test_a_line_the_accurate_model_hears_nothing_in_is_kept(monkeypatch, tmp_path: Path) -> None:
    import subprocess

    monkeypatch.setattr(analysis_module, "_run_whisper", lambda device, path, duration, *rest: Transcript(
        language="zh", model=rest[-1], segments=[]))
    monkeypatch.setattr(analysis_module.subprocess, "run",
                        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""))
    transcript = fast(line(0, 2, "今天天氣"), line(3, 5, "我們出發", sure=0.2))
    kept = refine_unsure_lines("x.mp4", transcript, 10.0, None, [], "large-v3-turbo", lambda f: None,
                               lambda: False, str(tmp_path))
    assert "".join(segment.text for segment in kept.segments) == "今天天氣我們出發"
