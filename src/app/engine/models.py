"""Where the models live, and how they get here.

Every model this server runs is kept inside the workspace, under `models/`.
Nothing is written to a cache in the user's home directory, so deleting the
workspace deletes the models with it and the project takes up what it appears
to take up. That is the whole point of this module: a model is project data,
not a machine-wide installation. `CLIP_MCP_MODELS` overrides where they go,
for anyone who wants the opposite trade.

A model is fetched the first time something needs it and never again. Each one
is pinned by the SHA-256 of what was downloaded, so a truncated download or a
changed release is a failure with a reason rather than a model that quietly
behaves differently. Downloads land on a temporary name and are moved into
place only once they have been verified, so an interrupted fetch leaves
nothing that a later run would mistake for a finished model.
"""

import hashlib
import os
import shutil
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
from app.workspace import workspace_dir

MEGABYTE = 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60

# The speech model built most recently, and the key it was built for. A worker that
# analyses several files in a row reuses it instead of loading the weights again. Only a
# worker keeps it: the MCP server also hears stretches again, and a server that kept the
# weights after one of those would hold gigabytes for as long as it runs.
_keep_speech_model = False
_speech_model_lock = threading.Lock()
_speech_model: Optional[Any] = None
_speech_model_key: Optional[Tuple[str, str, str]] = None

@dataclass(frozen=True)
class Model:
    """One model file, and where it comes from.

    Attributes:
        name: What to call it when telling somebody it is being downloaded.
        path: Where it is kept, relative to the model directory.
        url: Where it is fetched from.
        sha256: Digest of the downloaded file, checked before it is kept. For
            an archive this covers the archive, not the file taken out of it.
        size: Roughly how big the download is, in bytes, so that a caller can
            say what it is about to cost before it starts.
        member: For an archive, the path inside it to extract; `None` when the
            download is the model itself.
    """

    name: str
    path: str
    url: str
    sha256: str
    size: int
    member: Optional[str] = None

# Speaker diarization. Segmentation says when somebody is talking and when the voice
# changes; the embedding model says whether two stretches are the same person. The
# embedding model is trained on Mandarin and English together, which is what this is
# mostly pointed at.
SPEAKER_SEGMENTATION = Model(
    name="speaker segmentation",
    path="diarization/segmentation.onnx",
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    sha256="24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    size=7 * MEGABYTE,
    member="sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
)
SPEAKER_EMBEDDING = Model(
    name="speaker embedding",
    path="diarization/embedding.onnx",
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    sha256="aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
    size=27 * MEGABYTE,
)
# Face detection. Pinned to the commit rather than to a branch: the digest below is the
# whole safety net, and on a branch the day the file is regenerated is the day face
# detection stops working entirely instead of quietly changing.
FACE_DETECTION = Model(
    name="face detection",
    path="faces/yunet.onnx",
    url="https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a"
        "/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size=227 * 1024,
)

def models_dir() -> str:
    """Directory holding every model.

    Returns:
        The absolute path. By default `models/` inside the workspace, so that
        it follows wherever the workspace is and deleting it takes the
        models with it. `CLIP_MCP_MODELS` moves them elsewhere for anyone who
        would rather keep several gigabytes off that drive, or share one copy
        between workspaces — at the cost of the thing the default buys, which
        is that the project takes up what it appears to take up.
    """
    override = os.environ.get("CLIP_MCP_MODELS")
    if override:
        return os.path.abspath(override)
    return os.path.join(workspace_dir(), "models")

def whisper_dir() -> str:
    """Directory the speech recognition model is downloaded into.

    faster-whisper fetches its own weights, so this is handed to it as a
    download root rather than being filled in by `ensure`. It sits beside the
    other models for the same reason they are there: so that the workspace
    holds everything.

    Returns:
        The absolute path.
    """
    return os.path.join(models_dir(), "whisper")

def keep_speech_model() -> None:
    """Keep the speech model loaded between calls, for a worker that runs one analysis after another."""
    global _keep_speech_model
    _keep_speech_model = True

def speech_model(
    name: str, device: str, download_root: str, before_build: Optional[Callable[[], None]] = None,
) -> Any:
    """Load the speech model, reusing the one built last for the same settings.

    A worker that analyses one file after another would otherwise load the
    weights again for every file. Nothing is kept unless `keep_speech_model`
    was called, as a worker does. Only the model built most recently is kept:
    asking for a different name, device or download root drops it before the
    new one is built, so two of these large models are never held at once.
    A check of free memory belongs in `before_build`, which runs only when a
    model is about to be built, after the old one was let go: a model already
    held needs no more memory.

    Args:
        name: The model's name, or a path to it.
        device: `"cuda"` or `"cpu"`.
        download_root: Where the weights are kept.
        before_build: Called before a model is built, never when the one kept
            is returned; it raises to stop the build.

    Returns:
        A `faster_whisper.WhisperModel` for those settings.
    """
    global _speech_model, _speech_model_key
    key = (name, device, download_root)
    with _speech_model_lock:
        if _speech_model is not None and _speech_model_key == key:
            return _speech_model
        # Let the old weights go before the new ones are asked for.
        _speech_model = None
        _speech_model_key = None
        if before_build is not None:
            before_build()
        from faster_whisper import WhisperModel

        model = WhisperModel(
            name,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
            download_root=download_root,
        )
        if _keep_speech_model:
            _speech_model, _speech_model_key = model, key
        return model

def model_path(model: Model) -> str:
    """Where a model is kept.

    Args:
        model: The model.

    Returns:
        Its absolute path, whether or not it has been downloaded.
    """
    return os.path.join(models_dir(), model.path)

def is_downloaded(model: Model) -> bool:
    """Say whether a model is already here.

    Args:
        model: The model.

    Returns:
        True when its file is in place.
    """
    return os.path.exists(model_path(model))

def _fetch(model: Model, target: str, on_progress: Optional[Callable[[float], None]]) -> None:
    """Download a model to a path, verifying it before it is kept.

    Args:
        model: The model to fetch.
        target: Final path for the model file.
        on_progress: Called with the completed fraction, when given.

    Raises:
        RuntimeError: If the download fails, or if what arrived is not what
            was expected.
    """
    os.makedirs(os.path.dirname(target), exist_ok=True)
    handle, staged = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".part")
    os.close(handle)
    try:
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(model.url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                expected = int(response.headers.get("Content-Length") or 0)
                received = 0
                with open(staged, "wb") as file:
                    while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                        file.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                        if on_progress is not None and expected:
                            on_progress(min(received / expected, 1.0))
        except (urllib.error.URLError, OSError) as error:
            raise RuntimeError(f"could not download the {model.name} model from {model.url}: {error}") from error

        if digest.hexdigest() != model.sha256:
            raise RuntimeError(
                f"the {model.name} model downloaded from {model.url} is not the expected file "
                f"(sha256 {digest.hexdigest()}, expected {model.sha256}); it was not kept"
            )

        if model.member is None:
            os.replace(staged, target)
            staged = None
            return
        with tarfile.open(staged, "r:*") as archive:
            # A missing member raises, and a member that is a directory comes back as None.
            try:
                found = archive.extractfile(model.member)
            except KeyError:
                found = None
            if found is None:
                raise RuntimeError(f"the {model.name} download does not contain {model.member}")
            with found, open(f"{staged}.out", "wb") as file:
                shutil.copyfileobj(found, file)
        os.replace(f"{staged}.out", target)
    finally:
        for leftover in (staged, f"{staged}.out" if staged else None):
            if leftover and os.path.exists(leftover):
                os.remove(leftover)

def ensure(model: Model, on_progress: Optional[Callable[[float], None]] = None) -> str:
    """Make sure a model is on disk, downloading it the first time.

    Args:
        model: The model needed.
        on_progress: Called with the completed fraction of the download. Not
            called at all when the model is already here, which is the usual
            case.

    Returns:
        The path to the model file.

    Raises:
        RuntimeError: If it has to be downloaded and cannot be, or if what
            arrives does not match the digest it is pinned to.
    """
    target = model_path(model)
    if not os.path.exists(target):
        _fetch(model, target, on_progress)
    return target
