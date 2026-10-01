"""The system's own open-file and open-folder windows.

A file dropped on a web page arrives as its contents, never as where it is,
so the page could only copy it — and footage copied into the workspace is
footage on the disk twice. Asking Windows for the file instead gives its
path, and nothing is copied.
"""

import threading
from typing import List

from app.server import MEDIA_EXTENSIONS

# Only one dialog at a time: a second click while one is open would stack another
# behind it, which a beginner reads as the app having frozen.
_open = threading.Lock()

def _media_filter() -> List[tuple]:
    """The file types offered, media first."""
    patterns = " ".join(f"*{extension}" for extension in sorted(MEDIA_EXTENSIONS))
    return [("影片與聲音", patterns), ("所有檔案", "*.*")]

def pick(kind: str) -> List[str]:
    """Show an open window and wait for the user.

    Args:
        kind: `files` for one or more media files, `folder` for a folder.

    Returns:
        What was chosen; empty when the window was cancelled, or another one
        is already open.
    """
    if not _open.acquire(blocking=False):
        return []
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        # Over the editor's window rather than behind it.
        root.attributes("-topmost", True)
        root.update()
        try:
            if kind == "folder":
                chosen = filedialog.askdirectory(parent=root, title="選擇素材資料夾", mustexist=True)
                return [chosen] if chosen else []
            chosen = filedialog.askopenfilenames(parent=root, title="加入素材", filetypes=_media_filter())
            return list(chosen)
        finally:
            root.destroy()
    finally:
        _open.release()
