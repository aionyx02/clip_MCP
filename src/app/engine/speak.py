"""Speaking a narration in the user's own voice, with a speech model.

A narration written for a picture nobody talks over — 「原本的杯子區」 over a
shot of the counter — can go on as a caption or be spoken. Spoken, it is
spoken by F5-TTS, which clones a voice from a few seconds of it: the user's
own, taken from their footage, so the narration sounds like the rest of the
video. Whether a narration is spoken at all is the user's to say, never
assumed, and they are told what the model's licence allows before it is.

F5-TTS runs on PyTorch, which this server deliberately does not depend on
(the speaker models run on onnxruntime for the same reason). So it lives in
an environment of its own inside the workspace, installed the first time a
narration is spoken and run as a separate process: nobody who never asks for
a spoken narration downloads the five gigabytes it takes.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from app.engine import models
from app.engine.ffmpeg import OperationCancelled, hidden_window_flags

# The model and the licence its makers publish for the weights. The code is MIT; the
# pretrained weights are CC BY-NC 4.0 because the Emilia data they were trained on is.
DEFAULT_MODEL = "F5TTS_v1_Base"
DEFAULT_LICENSE = "CC BY-NC 4.0 (F5-TTS pretrained weights, trained on the Emilia dataset): no commercial use"
# Versions that work together on Windows. torchaudio from 2.9 reads audio through
# torchcodec, which needs FFmpeg's shared libraries rather than the ffmpeg.exe the rest
# of this server uses; 2.8 is also the version F5-TTS's own instructions install.
F5_TTS_VERSION = "1.1.22"
TORCH_PACKAGES = ("torch==2.8.0", "torchaudio==2.8.0")
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
ENV_PYTHON = "3.12"
INSTALLED_MARK = "installed.txt"
# How long a voice sample should be. F5-TTS clips its reference at about twelve seconds,
# and under three it has too little of the voice to go on. Provisional.
REFERENCE_SHORTEST = 3.0
REFERENCE_LONGEST = 10.0
# Silence put after the voice sample. With the sample ending right on its last word the
# model swallowed the first word of what it was asked to say — 「原本的杯子區」 came out
# as 「本的杯子區」 — and with this much after it, it did not.
REFERENCE_TAIL_SECONDS = 0.8
# How closely what is heard in a narration has to match what it was asked to say before
# it is taken, ignoring punctuation and script; below it, it is spoken again, once.
# Provisional: a recogniser mishears a few characters of perfectly good speech.
SAID_IT_ALL = 0.85

# A sentence is a voice sample only where the transcript and the second listen agree
# on its words this closely: the model is told exactly what the sample says, and a
# sample whose words are wrong — 「試工差」 for 「試公差」 — comes out mumbled. Provisional.
SAMPLE_AGREEMENT = 0.9

def said_it_all(asked: str, heard: str) -> float:
    """Say how closely two versions of the same words agree.

    Args:
        asked: What was meant to be said.
        heard: What a recogniser heard.

    Returns:
        A ratio from 0 to 1, over the letters and characters alone, with
        traditional and simplified Chinese read as the same.
    """
    from difflib import SequenceMatcher

    return SequenceMatcher(None, _plain(asked), _plain(heard)).ratio()

def _plain(text: str) -> str:
    """Keep only the letters and characters of a text, in one Chinese script and one case."""
    kept = "".join(character for character in text if character.isalnum()).lower()
    if any("一" <= character <= "鿿" for character in kept):
        import opencc

        kept = opencc.OpenCC("t2s.json").convert(kept)
    return kept

@dataclass(frozen=True)
class SpeechModel:
    """The speech model narrations are spoken with, and what it may be used for.

    Attributes:
        name: F5-TTS's name for the model architecture.
        checkpoint: Weights to load instead of the pretrained ones, if any.
        vocab: The vocabulary that goes with them.
        license: What the weights may be used for.
        commercial_use: Whether that includes commercial use.
    """

    name: str
    checkpoint: Optional[str]
    vocab: Optional[str]
    license: str
    commercial_use: bool

def speech_model() -> SpeechModel:
    """Read which speech model narrations are spoken with.

    The pretrained weights unless `CLIP_MCP_TTS_CHECKPOINT` names others — a
    model trained on data that allows commercial use, say — with
    `CLIP_MCP_TTS_VOCAB` for their vocabulary, `CLIP_MCP_TTS_LICENSE` for
    what they may be used for, and `CLIP_MCP_TTS_COMMERCIAL=1` when that
    includes commercial use. Weights nobody has said the licence of are
    treated as not allowing it.

    Returns:
        The model.
    """
    checkpoint = os.environ.get("CLIP_MCP_TTS_CHECKPOINT") or None
    if checkpoint is None:
        return SpeechModel(DEFAULT_MODEL, None, None, DEFAULT_LICENSE, False)
    return SpeechModel(
        os.environ.get("CLIP_MCP_TTS_MODEL", DEFAULT_MODEL),
        checkpoint,
        os.environ.get("CLIP_MCP_TTS_VOCAB") or None,
        os.environ.get("CLIP_MCP_TTS_LICENSE", "not given"),
        os.environ.get("CLIP_MCP_TTS_COMMERCIAL") == "1",
    )

def tts_dir() -> str:
    """Directory holding the speech model's environment and weights, beside the other models."""
    return os.path.join(models.models_dir(), "tts")

def env_python() -> str:
    """The Python of the speech model's own environment."""
    folder = os.path.join(tts_dir(), "env")
    return os.path.join(folder, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(folder, "bin", "python")

def is_installed() -> bool:
    """Say whether the speech model's environment is ready."""
    return os.path.isfile(env_python()) and os.path.isfile(os.path.join(tts_dir(), INSTALLED_MARK))

def install(on_progress: Callable[[float, str], None], is_cancelled: Callable[[], bool]) -> None:
    """Set up the speech model's environment, once.

    With uv, which is how this server is installed and run: a Python of its
    own, PyTorch built for the graphics card when there is one, and F5-TTS.

    Args:
        on_progress: Called with the completed fraction and what is happening.
        is_cancelled: Polled between steps to stop early.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If uv is missing or a step fails.
    """
    uv = os.environ.get("CLIP_MCP_UV") or shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is needed to install the speech model and was not found; install uv, or set CLIP_MCP_UV")
    python = env_python()
    folder = os.path.dirname(os.path.dirname(python))
    torch_index = ["--index-url", TORCH_CUDA_INDEX] if _has_nvidia() else []
    steps: Sequence[Tuple[str, List[str]]] = (
        ("making an environment for the speech model", [uv, "venv", "--allow-existing", "--python", ENV_PYTHON, folder]),
        ("downloading PyTorch, about 3 GB, once", [uv, "pip", "install", "--python", python, *TORCH_PACKAGES, *torch_index]),
        ("downloading F5-TTS, once", [uv, "pip", "install", "--python", python, f"f5-tts=={F5_TTS_VERSION}"]),
        # It comes with newer torchaudio and is not used by 2.8, but loads and fails without FFmpeg's DLLs.
        ("tidying up", [uv, "pip", "uninstall", "--python", python, "torchcodec"]),
    )
    for index, (stage, command) in enumerate(steps):
        if is_cancelled():
            raise OperationCancelled()
        on_progress(index / len(steps), stage)
        done = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, creationflags=hidden_window_flags())
        if done.returncode != 0 and stage != "tidying up":
            raise RuntimeError(f"{stage} failed: {done.stderr.decode('utf-8', 'replace').strip()[-800:]}")
    with open(os.path.join(tts_dir(), INSTALLED_MARK), "w", encoding="utf-8") as mark:
        mark.write(f"f5-tts {F5_TTS_VERSION}, {', '.join(TORCH_PACKAGES)}\n")
    on_progress(1.0, "installed")

def _has_nvidia() -> bool:
    """Say whether there is an NVIDIA graphics card to build PyTorch for."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return shutil.which("nvidia-smi") is not None

def speak(
    reference_wav: str,
    reference_text: str,
    text: str,
    output_wav: str,
    speed: float = 1.0,
    model: Optional[SpeechModel] = None,
) -> None:
    """Speak a narration in the voice of a reference recording.

    Args:
        reference_wav: A few seconds of the voice, as a WAV file.
        reference_text: Exactly what is said in it.
        text: What to say.
        output_wav: Where to write the narration.
        speed: How fast to say it; 1.0 as the model speaks.
        model: The speech model; `speech_model()` when not given.

    Raises:
        RuntimeError: If the model fails or writes nothing.
    """
    model = model or speech_model()
    command = [
        env_python(), "-m", "f5_tts.infer.infer_cli", "-m", model.name,
        "-r", reference_wav, "-s", reference_text, "-t", text,
        "-o", os.path.dirname(output_wav), "-w", os.path.basename(output_wav),
        "--speed", f"{speed:g}", "--remove_silence",
    ]
    if model.checkpoint:
        command += ["-p", model.checkpoint]
    if model.vocab:
        command += ["-v", model.vocab]
    # The weights and everything else it downloads stay in the workspace with the environment.
    env = dict(os.environ, HF_HOME=os.path.join(tts_dir(), "hf"), PYTHONIOENCODING="utf-8")
    done = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, env=env,
                          creationflags=hidden_window_flags())
    if done.returncode != 0 or not os.path.isfile(output_wav):
        raise RuntimeError(f"the speech model failed: {done.stderr.decode('utf-8', 'replace').strip()[-800:]}")

def reference_candidates(sentences: Sequence[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """Choose the sentences a voice can be taken from, best first.

    Args:
        sentences: `(start, end, text)` of sentences one person says, in
            source seconds.

    Returns:
        Those between `REFERENCE_SHORTEST` and `REFERENCE_LONGEST` long, the
        longest first: more of the voice is a better copy of it.
    """
    fitting = [item for item in sentences if REFERENCE_SHORTEST <= item[1] - item[0] <= REFERENCE_LONGEST and item[2].strip()]
    return sorted(fitting, key=lambda item: item[1] - item[0], reverse=True)

# What a narration job holds while it speaks: the model's weights and PyTorch in memory,
# most of it on the graphics card where there is one. Measured loosely on a laptop GPU
# and rounded up; it only has to keep a render from starting beside it on a small machine.
NARRATION_MEMORY_BYTES = 4 * 1024 ** 3
