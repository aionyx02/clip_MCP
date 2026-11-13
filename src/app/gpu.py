"""The NVIDIA libraries speech recognition needs on the card, and where they come from.

CTranslate2, which runs the speech model, brings its own CUDA runtime but
loads cuBLAS 12 by name the first time a model is built on the card:
`cublas64_12.dll`, which loads `cublasLt64_12.dll`. Nothing else is needed:
transcribing on an RTX 4060 loaded those two and the cuDNN stub CTranslate2
ships, and no other cuDNN library (measured for #298). Without them the card
is found but the model fails, so the card is chosen only when they are found.

They are looked for in this order, and the first that has both is used:

1. `bundled`: the GPU pack the installer, or the editor, downloaded into the
   installation's own `gpu` folder.
2. `system`: a CUDA Toolkit's, on PATH or under `CUDA_PATH`.
3. `none`: speech recognition runs on the processor, and `status` says why.

The pack is one pinned wheel from PyPI, checked against its SHA-256 before
anything is taken out of it; only the two libraries are kept.
"""

import functools
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

LIBRARIES = ("cublas64_12.dll", "cublasLt64_12.dll")
FOLDER = "gpu"
STAMP = "pack.json"

@dataclass(frozen=True)
class Pack:
    """The pinned download the GPU pack is taken from.

    Attributes:
        name: The wheel's file name.
        url: Where it is downloaded from.
        sha256: Its checksum; anything else is refused.
        size: Its size in bytes, said to the user before they choose it.
        members: Where each library is inside it.
        installed_size: What the libraries take on disk once taken out.
    """

    name: str
    url: str
    sha256: str
    size: int
    members: tuple
    installed_size: int

# cuBLAS 12.9: the newest 12.x with a Windows wheel. 12.8 and later also run on RTX 50 cards;
# a driver older than the toolkit still runs it (CUDA minor version compatibility), checked
# on driver 566.14, which reports CUDA 12.7.
PACK = Pack(
    name="nvidia_cublas_cu12-12.9.1.4-py3-none-win_amd64.whl",
    url="https://files.pythonhosted.org/packages/45/a1/a17fade6567c57452cfc8f967a40d1035bb9301db52f27808167fbb2be2f/"
        "nvidia_cublas_cu12-12.9.1.4-py3-none-win_amd64.whl",
    sha256="1e5fee10662e6e52bd71dec533fbbd4971bb70a5f24f3bc3793e5c2e9dc640bf",
    size=553_153_899,
    members=("nvidia/cublas/bin/cublas64_12.dll", "nvidia/cublas/bin/cublasLt64_12.dll"),
    installed_size=771_188_224,
)
CHUNK = 1024 * 1024
TIMEOUT = 60

@dataclass(frozen=True)
class Libraries:
    """Where cuBLAS was found.

    Attributes:
        source: `bundled`, `system` or `none`.
        folder: The folder holding both libraries, or None.
    """

    source: str
    folder: Optional[Path]

def install_folder() -> Optional[Path]:
    """The installer's folder, when this copy came from the installer."""
    from app.ui import updates

    return updates.installed_by_setup()

def bundled_folder() -> Optional[Path]:
    """Where the GPU pack goes in this installation; None for a copy not from the installer."""
    root = install_folder()
    return root / FOLDER if root else None

def _has_all(folder: Path) -> bool:
    return all((folder / name).is_file() for name in LIBRARIES)

def _system_folders() -> List[Path]:
    """The folders the system offers libraries from: PATH, then a CUDA Toolkit's own."""
    folders = [Path(entry) for entry in os.environ.get("PATH", "").split(os.pathsep) if entry.strip()]
    for name, value in os.environ.items():
        if name.upper() == "CUDA_PATH" or name.upper().startswith("CUDA_PATH_V"):
            folders.append(Path(value) / "bin")
    return folders

def _on_windows() -> bool:
    """Whether this is Windows, where the libraries are looked for: a seam tests can change without touching `sys`."""
    return sys.platform == "win32"


def find() -> Libraries:
    """Look for cuBLAS 12: the bundled pack first, then the system's.

    Returns:
        Where it is, or `none`. Elsewhere than Windows the libraries are the
        system's to provide and are not looked for: `system`, with no folder.
    """
    if not _on_windows():
        return Libraries("system", None)
    bundled = bundled_folder()
    if bundled is not None and _has_all(bundled):
        return Libraries("bundled", bundled)
    for folder in _system_folders():
        try:
            if _has_all(folder):
                return Libraries("system", folder)
        except OSError:
            continue
    return Libraries("none", None)

@functools.lru_cache(maxsize=1)
def use() -> Libraries:
    """Find cuBLAS and load it from where it was found, once per process.

    Loaded by full path, so that when CTranslate2 asks for it by name it gets
    this copy and not whichever comes first on PATH: the bundled pack wins
    over an older toolkit's.

    Returns:
        Where it was loaded from, or `none` when it was not found or would not load.
    """
    found = find()
    if found.folder is None:
        return found
    try:
        _load(found.folder)
    except OSError:
        return Libraries("none", found.folder)
    return found

def _load(folder: Path) -> None:
    """Load both libraries from `folder`, and let anything loaded after them look there too."""
    import ctypes

    os.add_dll_directory(str(folder))
    # cublasLt first: cublas64_12 loads it by name as it loads.
    for name in reversed(LIBRARIES):
        ctypes.WinDLL(str(folder / name))
    os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")

# Set when a transcription on the card failed and was done on the processor instead, so the
# fallback is said rather than silent.
last_failure: Optional[str] = None

def note_failure(error: str) -> None:
    """Keep why the card failed, for `status`, and say it on stderr where clients log it."""
    global last_failure
    last_failure = error
    print(f"clip-mcp: speech recognition fell back to the processor: {error}", file=sys.stderr)

def card_count() -> int:
    """How many NVIDIA cards CTranslate2's CUDA runtime can use: 0 without a recent enough driver."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except (ImportError, OSError, RuntimeError):
        return 0

def status() -> dict:
    """Say where speech recognition runs and why, for `clip-mcp check` and the editor.

    Returns:
        `device` (`cuda` or `cpu`), `libraries` (`bundled`, `system`, `none`),
        `folder`, `card` (whether a usable NVIDIA card is here), `reason` in
        words and `why` as a key for the editor to put in its own words
        (`forced`, `no_card`, `wont_load`, `no_libraries`, `bundled`,
        `system`), `last_failure`, and what the pack would take:
        `pack_size`, `pack_installed_size`, `can_install`, `pack_installed`.
    """
    forced = os.environ.get("CLIP_MCP_WHISPER_DEVICE", "auto").lower()
    card = card_count() > 0
    libraries = use() if card or forced == "cuda" else find()
    bundled = bundled_folder()
    if forced in ("cpu", "cuda"):
        device, why, reason = forced, "forced", f"CLIP_MCP_WHISPER_DEVICE is set to {forced}"
    elif not card:
        device, why, reason = "cpu", "no_card", "no NVIDIA card with a driver for CUDA 12 was found"
    elif libraries.source == "none" and libraries.folder is not None:
        device, why, reason = "cpu", "wont_load", f"cuBLAS in {libraries.folder} would not load"
    elif libraries.source == "none":
        device, why = "cpu", "no_libraries"
        reason = ("an NVIDIA card is here but cuBLAS 12 is not: install the GPU pack"
                  + (" from the editor" if bundled else ", or a CUDA 12 Toolkit"))
    else:
        device, why = "cuda", libraries.source
        reason = (f"the GPU pack in {libraries.folder}" if libraries.source == "bundled"
                  else f"the system's cuBLAS in {libraries.folder}" if libraries.folder
                  else "the system's CUDA libraries")
    return {
        "device": device, "why": why, "libraries": libraries.source,
        "folder": str(libraries.folder) if libraries.folder else None,
        "card": card, "reason": reason, "last_failure": last_failure,
        "pack_size": PACK.size, "pack_installed_size": PACK.installed_size,
        "can_install": bundled is not None and card, "pack_installed": bundled is not None and _has_all(bundled),
    }

def _download(target: Path, on_progress: Callable[[int, int], None]) -> None:
    """Fetch the pinned wheel, picking up a download that stopped part way."""
    have = target.stat().st_size if target.exists() else 0
    if have >= PACK.size:
        return
    request = urllib.request.Request(PACK.url, headers={"User-Agent": "clip-mcp", "Range": f"bytes={have}-"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        # A server that ignores the range sends the whole file again.
        mode = "ab" if response.status == 206 else "wb"
        done = have if mode == "ab" else 0
        with open(target, mode) as handle:
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                done += len(chunk)
                on_progress(done, PACK.size)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()

def install_pack(on_progress: Callable[[int, int], None] = lambda done, total: None) -> Path:
    """Download the GPU pack into this installation and keep only the libraries.

    Args:
        on_progress: Told the bytes downloaded so far and the total.

    Returns:
        The folder the libraries are in.

    Raises:
        RuntimeError: Not an installed copy, or the download does not match
            its checksum (it is deleted, so the next try starts over).
        OSError: The download or the copy failed.
    """
    folder = bundled_folder()
    if folder is None:
        raise RuntimeError("the GPU pack goes into a copy installed by the installer; "
                           "a copy run from source uses a CUDA 12 Toolkit's cuBLAS")
    stamp = folder / STAMP
    if _has_all(folder) and stamp.is_file() and json.loads(stamp.read_text(encoding="utf-8")).get("sha256") == PACK.sha256:
        return folder
    staging = folder.with_name(FOLDER + ".download")
    staging.mkdir(parents=True, exist_ok=True)
    wheel = staging / PACK.name
    _download(wheel, on_progress)
    found = _sha256(wheel)
    if found != PACK.sha256:
        wheel.unlink(missing_ok=True)
        raise RuntimeError(f"the GPU pack download does not match its checksum ({found}); try again")
    unpacked = folder.with_name(FOLDER + ".new")
    shutil.rmtree(unpacked, ignore_errors=True)
    unpacked.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for member in PACK.members:
            with archive.open(member) as source, open(unpacked / Path(member).name, "wb") as out:
                shutil.copyfileobj(source, out, CHUNK)
    (unpacked / STAMP).write_text(json.dumps({"wheel": PACK.name, "sha256": PACK.sha256}), encoding="utf-8")
    shutil.rmtree(folder, ignore_errors=True)
    unpacked.rename(folder)
    shutil.rmtree(staging, ignore_errors=True)
    use.cache_clear()
    return folder

def remove_pack() -> bool:
    """Take the GPU pack out of this installation; whether there was one."""
    folder = bundled_folder()
    if folder is None or not folder.exists():
        return False
    shutil.rmtree(folder)
    use.cache_clear()
    return True
