"""The Python half of build_payload.ps1: launchers, slimming, a smoke test, and sizes.

    python payload_tools.py launcher <uv trampoline> <out exe> <python path relative to out>
    python payload_tools.py slim <payload>
    python payload_tools.py smoke <payload>
    python payload_tools.py sizes <payload> [<json out>]

Run with the build machine's Python for `launcher`, `slim` and `sizes`, and with
the payload's own (`runtime\\python.exe -I -B`) for `smoke`, which is the point of it.
"""

import ctypes
import json
import os
import shutil
import sys
from pathlib import Path

# --- launchers -------------------------------------------------------------------------------
# uv writes each console script as a small launcher with the Python to start held in a resource,
# UV_PYTHON_PATH, which it resolves against the launcher's own folder when it is relative. A
# relocatable venv's launchers say `python.exe`; ours say `..\runtime\python.exe`, so `bin\`
# keeps the place it had in the uv-based installs, where AI clients were pointed at it.

RT_RCDATA = 10
PYTHON_PATH = "UV_PYTHON_PATH"

def _read_resource(exe: Path, name: str) -> bytes:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LoadLibraryExW.restype = ctypes.c_void_p
    kernel.FindResourceW.restype = ctypes.c_void_p
    kernel.FindResourceW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p]
    kernel.LoadResource.restype = ctypes.c_void_p
    kernel.LoadResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.LockResource.restype = ctypes.c_void_p
    kernel.LockResource.argtypes = [ctypes.c_void_p]
    kernel.SizeofResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
    module = kernel.LoadLibraryExW(str(exe), None, 0x2 | 0x20)  # as data, as an image resource
    if not module:
        raise OSError(ctypes.get_last_error(), f"cannot read {exe}")
    try:
        found = kernel.FindResourceW(module, name, RT_RCDATA)
        if not found:
            raise OSError(f"{exe} has no {name}: not a uv launcher this script knows")
        return ctypes.string_at(kernel.LockResource(kernel.LoadResource(module, found)), kernel.SizeofResource(module, found))
    finally:
        kernel.FreeLibrary(module)

def launcher(trampoline: Path, out: Path, python: str) -> None:
    """Copy a uv launcher and point it at `python`, relative to where it will sit."""
    original = _read_resource(trampoline, PYTHON_PATH).decode()
    if Path(original).name.lower() != Path(python).name.lower():
        raise SystemExit(f"{trampoline} starts {original}, not {Path(python).name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(trampoline, out)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.BeginUpdateResourceW.restype = ctypes.c_void_p
    kernel.BeginUpdateResourceW.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
    kernel.UpdateResourceW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ushort,
                                       ctypes.c_char_p, ctypes.c_ulong]
    kernel.EndUpdateResourceW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    data = python.encode()
    handle = kernel.BeginUpdateResourceW(str(out), 0)
    if not handle or not kernel.UpdateResourceW(handle, RT_RCDATA, PYTHON_PATH, 0, data, len(data)) \
            or not kernel.EndUpdateResourceW(handle, 0):
        raise OSError(ctypes.get_last_error(), f"cannot write {out}")
    if _read_resource(out, PYTHON_PATH) != data:
        raise SystemExit(f"{out} did not take the new Python path")
    print(f"launcher {out.name} -> {python}")

# --- slimming --------------------------------------------------------------------------------
# What the payload leaves out, each with why clip-mcp cannot miss it. Paths are relative to the
# runtime folder; `*` matches within a name. Checked after removal by `smoke`, and by the
# analysis and render the release is tested with.

REMOVE = [
    ("Scripts", "the packages' own console scripts, with the build folder written into them, and"
                " sherpa-onnx's second copy of its C libraries; clip-mcp's launchers are in bin"),
    ("include", "C headers, for building extensions"),
    ("libs", "import libraries, for building extensions"),
    ("Lib/ensurepip", "installs pip, which is left out too"),
    ("Lib/idlelib", "the IDLE editor"),
    ("Lib/turtledemo", "turtle graphics demos"),
    ("DLLs/_test*.pyd", "CPython's own test modules"),
    ("DLLs/_ctypes_test.pyd", "CPython's own test module"),
    ("Lib/site-packages/pip", "nothing is installed into the payload after it is built"),
    ("Lib/site-packages/pip-*.dist-info", "pip's record"),
    ("Lib/site-packages/pythonwin", "PyWin32's IDE and MFC bindings: mcp and dulwich use only win32api,"
                                    " win32con, win32job and pywintypes"),
    ("Lib/site-packages/PyWin32.chm", "PyWin32's help file"),
    ("Lib/site-packages/adodbapi", "PyWin32's database adapter, unused"),
    ("Lib/site-packages/isapi", "PyWin32's IIS extension support, unused"),
    ("Lib/site-packages/cv2/opencv_videoio_ffmpeg*.dll", "OpenCV's own FFmpeg, for cv2.VideoCapture; clip-mcp"
                                                         " reads stills with cv2.imread and video with FFmpeg"),
    ("Lib/site-packages/jieba/lac_small", "jieba's PaddlePaddle mode: only with enable_paddle(), never called"),
    ("Lib/site-packages/jieba/analyse", "jieba's keyword extraction, unused: captions use cut and posseg"),
    ("Lib/site-packages/jieba/posseg/*.p", "Jython's pickled tables; CPython imports the .py ones"),
    ("Lib/site-packages/jieba/finalseg/*.p", "Jython's pickled tables; CPython imports the .py ones"),
    ("Lib/site-packages/sherpa_onnx/lib/*.lib", "link libraries, for building against sherpa-onnx"),
    ("Lib/site-packages/sherpa_onnx/include", "C headers"),
]
# Folders removed wherever they are: bytecode is written again on first use, and test suites
# are never imported by the package they test.
REMOVE_DIRS = {"__pycache__": "bytecode, made again on first use", "tests": "test suites"}
# PyWin32 calls its test folders `test`; the standard library's `test` package is not touched.
PYWIN32_TESTS = ("win32/test", "win32com/test", "win32comext/*/test")

def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

def _remove(path: Path) -> int:
    size = _size(path)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return size

def slim(payload: Path) -> dict:
    """Leave out of the payload what clip-mcp never uses, saying what each removal saved."""
    runtime = payload / "runtime"
    saved: dict = {}
    for pattern, why in REMOVE:
        parent = runtime / Path(pattern).parent
        for found in sorted(parent.glob(Path(pattern).name)) if parent.is_dir() else []:
            saved[str(found.relative_to(payload))] = (_remove(found), why)
    site = runtime / "Lib" / "site-packages"
    for pattern in PYWIN32_TESTS:
        for found in site.glob(pattern):
            saved[str(found.relative_to(payload))] = (_remove(found), "PyWin32's test suites")
    for folder, why in REMOVE_DIRS.items():
        for found in sorted(payload.rglob(folder), key=lambda path: len(path.parts), reverse=True):
            if found.is_dir():
                saved.setdefault(f"every {folder}", [0, why])
                saved[f"every {folder}"] = (saved[f"every {folder}"][0] + _remove(found), why)
    for name, (size, why) in sorted(saved.items(), key=lambda item: -item[1][0]):
        print(f"{size / 2 ** 20:8.1f} MB  {name}  ({why})")
    print(f"{sum(size for size, _ in saved.values()) / 2 ** 20:8.1f} MB  removed in all")
    return {name: size for name, (size, _) in saved.items()}

# --- smoke test ------------------------------------------------------------------------------

def smoke(payload: Path) -> None:
    """Import every clip-mcp module and run what the slimming came close to, in the payload's Python.

    Raises:
        SystemExit: Something the program uses is missing.
    """
    import importlib
    import pkgutil
    import tempfile

    if not Path(sys.prefix).resolve() == (payload / "runtime").resolve():
        raise SystemExit(f"run this with {payload / 'runtime' / 'python.exe'} -I -B, not {sys.executable}")
    # Importing the tools opens a workspace: a throwaway one, never the user's.
    os.environ["CLIP_MCP_WORKSPACE"] = tempfile.mkdtemp(prefix="clip-mcp-smoke-")
    import app

    failed = []
    for module in pkgutil.walk_packages(app.__path__, "app."):
        try:
            importlib.import_module(module.name)
        except Exception as error:  # noqa: BLE001 - every failure is reported, then the build stops
            failed.append(f"{module.name}: {type(error).__name__}: {error}")
    for name in ("faster_whisper", "ctranslate2", "onnxruntime", "sherpa_onnx", "av", "cv2", "tkinter", "win32job",
                 "opencc", "pypinyin", "tokenizers", "PIL", "numpy", "dulwich", "fastmcp"):
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001
            failed.append(f"{name}: {type(error).__name__}: {error}")
    if failed:
        raise SystemExit("smoke test failed:\n  " + "\n  ".join(failed))
    import cv2
    import jieba
    import jieba.posseg

    assert hasattr(cv2, "FaceDetectorYN") and cv2.imread(str(payload / "nothing.png")) is None
    assert list(jieba.cut("我們今天去台南")) and [pair.flag for pair in jieba.posseg.cut("我們今天去台南")]
    from importlib.metadata import version

    print(f"smoke test passed: clip-mcp {version('clip-mcp')} from {Path(app.__file__).parent}")

# --- sizes -----------------------------------------------------------------------------------

PARTS = {
    "python": ("runtime", lambda path: not path.is_relative_to(Path("runtime/Lib/site-packages"))),
    "packages": ("runtime/Lib/site-packages", lambda path: True),
    "program": ("program", lambda path: True),
    "launchers": ("bin", lambda path: True),
    "ffmpeg": ("ffmpeg", lambda path: True),
    "gpu pack": ("gpu", lambda path: True),
}

def sizes(payload: Path) -> dict:
    """What each part of the payload takes, uncompressed."""
    found = {}
    for part, (folder, keep) in PARTS.items():
        root = payload / folder
        if root.exists():
            found[part] = sum(item.stat().st_size for item in root.rglob("*")
                              if item.is_file() and keep(item.relative_to(payload)))
    found["total"] = sum(found.values())
    for part, size in found.items():
        print(f"{part:<10} {size / 2 ** 20:9.1f} MB")
    return found

def main() -> None:
    command, *rest = sys.argv[1:]
    if command == "launcher":
        launcher(Path(rest[0]), Path(rest[1]), rest[2])
    elif command == "slim":
        slim(Path(rest[0]))
    elif command == "smoke":
        smoke(Path(rest[0]))
    elif command == "sizes":
        found = sizes(Path(rest[0]))
        if len(rest) > 1:
            Path(rest[1]).write_text(json.dumps(found, indent=1), encoding="utf-8")
    else:
        raise SystemExit(__doc__)

if __name__ == "__main__":
    main()
