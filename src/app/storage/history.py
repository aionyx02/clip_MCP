"""Every version of every cut, kept as a git repository.

The database holds what each project, plan and caption is now; this keeps
what each of them has been. It only ever grows: a write to the database is
followed by a commit here, and going back to an old version reads it out of
here and writes it into the database again, which is one more commit. Nothing
is ever rewritten, so a mistaken step back is undone the same way.

Plain git, so an AI with a shell can read it with `git log` and `git diff`,
and the version panel draws its graph from the same commits. Only text goes
in — projects, plans, captions kept against their footage, and the library's
list of files — never footage, renders, analyses or caches.

Every commit says who made it and, in its first line, what changed in words
a person would use: the user's own words when the AI was given them, or a
summary of the change. Writes made during one tool call or one editor action
are one commit, so a version is a step somebody took.
"""

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from dulwich.errors import NotTreeError
from dulwich.object_store import tree_lookup_path
from dulwich.repo import Repo
from filelock import FileLock

# Who a commit is written by when nobody said: a write outside any tool call or editor
# action, such as starting the history or a background job.
SERVER_AUTHOR = "clip-mcp"
# Every commit carries the same address: the name says who; the address only has to be valid.
AUTHOR_EMAIL = "clip-mcp@localhost"
# How many characters of a commit's first line: long enough for a sentence, short enough
# for the graph to show whole.
SUBJECT_CHARACTERS = 72
# Characters a file system will not take in a name, plus the control range.
_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def canonical(data: object) -> bytes:
    """Write data the same way every time, so a diff shows only what changed.

    Keys sorted, one field to a line, seconds to the millisecond — a time kept
    as `104.7166886666666666666666667` would change on every save without
    anything having changed.

    Args:
        data: JSON-like data; `Decimal` and float values are rounded to three places.

    Returns:
        UTF-8 text ending in a newline.
    """
    def rounded(value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, (float, Decimal)):
            number = round(float(value), 3)
            return int(number) if number.is_integer() else number
        if isinstance(value, dict):
            return {str(key): rounded(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [rounded(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    return (json.dumps(rounded(data), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def file_stem(name: str) -> str:
    """A name made safe to be part of a file's name in the history.

    Args:
        name: A project's or a file's name.

    Returns:
        The name with anything a file system refuses replaced, at most 60 characters.
    """
    return " ".join(_FORBIDDEN.sub(" ", name).split())[:60].strip(" .")


@dataclass
class Batch:
    """The writes of one step somebody took, to become one commit.

    Attributes:
        author: Who took it: `AI（Claude Code）`, `編輯器`.
        note: The user's own words for it, when somebody passed them on.
        doing: What was done, in a few words, when no note was given — the
            tool or the editor action.
        files: Each file written, by its path in the history, with what it
            held before and holds now (None when it is not there).
        said: A summary of each change, in the order they were made.
    """

    author: str
    note: str = ""
    doing: str = ""
    files: Dict[str, Tuple[Optional[bytes], Optional[bytes]]] = field(default_factory=dict)
    said: List[str] = field(default_factory=list)


_batch: ContextVar[Optional[Batch]] = ContextVar("history_batch", default=None)


@dataclass(frozen=True)
class Version:
    """One commit in the history, as the AI and the version panel read it.

    Attributes:
        commit: The commit's ID.
        when: When it was made.
        author: Who made it.
        subject: Its first line: what changed, in words.
        body: The rest: which files and the details.
        parents: The commits it follows.
    """

    commit: str
    when: datetime
    author: str
    subject: str
    body: str
    parents: Tuple[str, ...]


class History:
    """The workspace's history repository, written to by every process that edits.

    The MCP server and the editor app are separate processes, so each commit
    is made under a lock file: one writer at a time, and each sees the other's
    last commit before making its own.
    """

    def __init__(self, root: str, snapshot: Optional[Callable[[], Dict[str, bytes]]] = None):
        """Point at the repository, which is made the first time anything is written.

        Args:
            root: The repository's folder.
            snapshot: Everything there is to keep, as `{path: content}` — what
                the first commit holds, so the history starts from how things
                stand rather than from nothing.
        """
        self.root = os.path.abspath(root)
        self._snapshot = snapshot
        self._lock = FileLock(self.root + ".lock")
        # What `counts` last found, by folder, with the commit it was found at.
        self._counted: Dict[str, Tuple[str, Dict[str, int]]] = {}

    @contextmanager
    def step(self, author: str, note: str = "", doing: str = "") -> Iterator[Batch]:
        """Gather every write made inside the block into one commit.

        A step inside a step belongs to the outer one: a tool that calls
        another still makes one version.

        Args:
            author: Who is taking the step.
            note: The user's own words for it, if any were passed on.
            doing: What is being done, for the summary when there is no note.

        Yields:
            The batch being gathered.
        """
        outer = _batch.get()
        if outer is not None:
            yield outer
            return
        batch = Batch(author=author, note=note.strip(), doing=doing)
        token = _batch.set(batch)
        try:
            yield batch
        finally:
            _batch.reset(token)
        if batch.files:
            self._commit(batch)

    def write(self, path: str, content: Optional[bytes], said: str = "") -> None:
        """Record that a file in the history now holds this, or is gone.

        Inside a step it joins that step's commit; outside one it is a commit
        of its own.

        Args:
            path: Where it lives in the history, with forward slashes.
            content: What it holds now, or None when it is gone.
            said: What changed, in words, for the commit's message.
        """
        batch = _batch.get()
        if batch is None:
            with self.step(SERVER_AUTHOR):
                self.write(path, content, said)
            return
        before = batch.files[path][0] if path in batch.files else self._read_current(path)
        batch.files[path] = (before, content)
        if said and said not in batch.said:
            batch.said.append(said)

    def working_path(self, prefix: str, suffix: str) -> Optional[str]:
        """Find a file in the working copy by its folder and how its name ends.

        Args:
            prefix: Its folder, such as `projects/`.
            suffix: How its name ends, such as `-<project ID>.json`.

        Returns:
            Its path in the history, or None when there is none.
        """
        folder = os.path.join(self.root, *prefix.strip("/").split("/"))
        if not os.path.isdir(folder):
            return None
        for name in os.listdir(folder):
            if name.endswith(suffix):
                return f"{prefix.strip('/')}/{name}"
        return None

    def _read_current(self, path: str) -> Optional[bytes]:
        """What a file holds in the working copy now, or None."""
        full = os.path.join(self.root, *path.split("/"))
        if not os.path.exists(full):
            return None
        with open(full, "rb") as file:
            return file.read()

    def _repo(self) -> Repo:
        """Open the repository, making it and its first commit if there is none yet."""
        if os.path.isdir(os.path.join(self.root, ".git")):
            return Repo(self.root)
        os.makedirs(self.root, exist_ok=True)
        repo = Repo.init(self.root)
        if self._snapshot is not None:
            paths = []
            for path, content in self._snapshot().items():
                self._put(path, content)
                paths.append(path)
            if paths:
                repo.get_worktree().stage(paths)
            self._make_commit(repo, "開始記錄版本", "從這裡開始，每次修改都會留下一個版本。", SERVER_AUTHOR)
        return repo

    def _put(self, path: str, content: Optional[bytes]) -> None:
        """Write a file into the working copy, or take it away."""
        full = os.path.join(self.root, *path.split("/"))
        if content is None:
            if os.path.exists(full):
                os.remove(full)
            return
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as file:
            file.write(content)

    def _commit(self, batch: Batch) -> None:
        """Write a step's files and commit them, unless nothing really changed."""
        with self._lock:
            repo = self._repo()
            try:
                changed = [path for path, (_, now) in batch.files.items() if self._read_current(path) != now]
                if not changed:
                    return
                for path in changed:
                    self._put(path, batch.files[path][1])
                repo.get_worktree().stage(changed)
                subject = batch.note or "；".join(batch.said) or batch.doing or "修改"
                details = [f"由{batch.author}" + (f"：{batch.doing}" if batch.doing else ""), *batch.said,
                           "", *(f"- {path}" for path in changed)]
                self._make_commit(repo, subject, "\n".join(details), batch.author)
            finally:
                repo.close()

    @staticmethod
    def _make_commit(repo: Repo, subject: str, body: str, author: str) -> None:
        """Commit what is staged, with a first line a person can read and nobody's signature.

        Never signed: these are the app's own records, and signing would ask
        for whatever key the user's git is set up with on every edit.
        """
        line = " ".join(subject.split())
        if len(line) > SUBJECT_CHARACTERS:
            line = line[:SUBJECT_CHARACTERS - 1] + "…"
        who = f"{author} <{AUTHOR_EMAIL}>".encode("utf-8")
        stamp = time.time()
        repo.get_worktree().commit(
            message=f"{line}\n\n{body}\n".encode("utf-8"), author=who, committer=who,
            author_timestamp=stamp, commit_timestamp=stamp, sign=False,
        )

    def versions(self, paths: List[str], limit: int = 50) -> List[Version]:
        """The commits that changed any of these files, newest first.

        Args:
            paths: Files in the history.
            limit: Most commits to return.

        Returns:
            The versions; empty when there is no history yet.
        """
        if not os.path.isdir(os.path.join(self.root, ".git")):
            return []
        with Repo(self.root) as repo:
            if not _has_head(repo):
                return []
            walker = repo.get_walker(paths=[path.encode("utf-8") for path in paths], max_entries=limit)
            return [_version(entry.commit) for entry in walker]

    def versions_of(self, prefix: str, suffix: str, limit: int = 50) -> List[Version]:
        """The commits that changed a file found by how its name ends, whatever it was called then.

        A project's file is named after the project, so renaming it renames
        the file; its ID, at the end of the name, is what stays.

        Args:
            prefix: Its folder, such as `projects/`.
            suffix: How its name ends, such as `<project ID>.json`.
            limit: Most commits to return.

        Returns:
            The versions, newest first.
        """
        if not os.path.isdir(os.path.join(self.root, ".git")):
            return []
        folder = prefix.encode("utf-8")
        ending = suffix.encode("utf-8")
        found: List[Version] = []
        with Repo(self.root) as repo:
            if not _has_head(repo):
                return []
            for entry in repo.get_walker(paths=[folder.rstrip(b"/")]):
                for change in entry.changes():
                    # A merge reports one list of changes per parent; there are none here, but read both shapes.
                    for one in (change if isinstance(change, list) else [change]):
                        paths = [side.path for side in (one.old, one.new) if side is not None and side.path]
                        if any(path.startswith(folder) and path.endswith(ending) for path in paths):
                            found.append(_version(entry.commit))
                            break
                    else:
                        continue
                    break
                if len(found) >= limit:
                    break
        return found

    def read(self, path: str, commit: str) -> Optional[bytes]:
        """What a file held at one commit.

        Args:
            path: The file in the history.
            commit: The commit's ID, or the start of it.

        Returns:
            Its content, or None when it was not there then.

        Raises:
            KeyError: If there is no such commit.
        """
        with self._lock, Repo(self.root) as repo:
            found = self._resolve(repo, commit)
            try:
                _, sha = tree_lookup_path(repo.get_object, repo[found].tree, path.encode("utf-8"))
            except (KeyError, NotTreeError):
                return None
            return repo[sha].data

    def find(self, prefix: str, suffix: str, commit: str) -> Optional[str]:
        """Find a file by the start and end of its path at one commit, whatever its name was then.

        A project's file carries its name, which changes; its ID, at the end, does not.

        Args:
            prefix: Its folder, such as `projects/`.
            suffix: How its name ends, such as `-<project ID>.json`.
            commit: The commit.

        Returns:
            The path, or None when there was no such file then.
        """
        with self._lock, Repo(self.root) as repo:
            tree = repo[repo[self._resolve(repo, commit)].tree]
            folder = prefix.rstrip("/").encode("utf-8")
            if folder not in tree:
                return None
            for entry in repo[tree[folder][1]].items():
                name = entry.path.decode("utf-8")
                if name.endswith(suffix):
                    return f"{prefix.rstrip('/')}/{name}"
        return None

    @staticmethod
    def _resolve(repo: Repo, commit: str) -> bytes:
        """Turn a whole or shortened commit ID into the commit's ID."""
        wanted = commit.strip().lower()
        if len(wanted) == 40:
            sha = wanted.encode("ascii")
            if sha in repo:
                return sha
        elif len(wanted) >= 6:
            for entry in repo.get_walker():
                if entry.commit.id.decode("ascii").startswith(wanted):
                    return entry.commit.id
        raise KeyError(f"there is no version {commit}")

    def counts(self, prefix: str) -> Dict[str, int]:
        """How many versions each file in a folder has, by the ID its name ends with, in one walk.

        Remembered until the next commit, since the projects page asks for every
        project at once each time it is drawn.

        Args:
            prefix: The folder, such as `projects/`.

        Returns:
            The number of commits that changed each, by ID.
        """
        head = self.head()
        if head is None:
            return {}
        cached = self._counted.get(prefix)
        if cached and cached[0] == head:
            return cached[1]
        folder = prefix.encode("utf-8")
        counted: Dict[str, int] = {}
        with Repo(self.root) as repo:
            for entry in repo.get_walker(paths=[folder.rstrip(b"/")]):
                seen = set()
                for change in entry.changes():
                    for one in (change if isinstance(change, list) else [change]):
                        for side in (one.old, one.new):
                            if side is not None and side.path and side.path.startswith(folder):
                                stem = side.path.decode("utf-8").rsplit("/", 1)[-1].removesuffix(".json")
                                seen.add(stem[-36:])
                for found in seen:
                    counted[found] = counted.get(found, 0) + 1
        self._counted[prefix] = (head, counted)
        return counted

    def head(self) -> Optional[str]:
        """The newest commit's ID, or None before anything was kept."""
        if not os.path.isdir(os.path.join(self.root, ".git")):
            return None
        with Repo(self.root) as repo:
            return repo.head().decode("ascii") if _has_head(repo) else None


def _has_head(repo: Repo) -> bool:
    """Whether anything has been committed yet."""
    try:
        repo.head()
    except KeyError:
        return False
    return True

def _version(commit) -> Version:
    """Read a dulwich commit as a `Version`."""
    message = commit.message.decode("utf-8", errors="replace")
    subject, _, body = message.partition("\n")
    author = commit.author.decode("utf-8", errors="replace").rsplit(" <", 1)[0]
    return Version(
        commit=commit.id.decode("ascii"),
        when=datetime.fromtimestamp(commit.author_time, timezone.utc),
        author=author,
        subject=subject.strip(),
        body=body.strip(),
        parents=tuple(parent.decode("ascii") for parent in commit.parents),
    )


_fingerprints: Dict[Tuple[str, int, int], Optional[str]] = {}
# How much of each end of a file its fingerprint reads: enough to tell two takes apart,
# little enough that a library of hundreds of files is read in a moment.
FINGERPRINT_BYTES = 1 << 16


def fingerprint(path: str) -> Optional[str]:
    """Something that stays the same when a file is moved and changes when it is not the same file.

    The whole file is not read: footage runs to gigabytes. Its size and both
    ends of it tell one take from another, and are remembered for as long as
    the file's size and time do not change.

    Args:
        path: The file.

    Returns:
        A short hex digest, or None when the file is not there.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key not in _fingerprints:
        digest = hashlib.sha256(str(stat.st_size).encode("ascii"))
        with open(path, "rb") as file:
            digest.update(file.read(FINGERPRINT_BYTES))
            if stat.st_size > FINGERPRINT_BYTES:
                file.seek(max(FINGERPRINT_BYTES, stat.st_size - FINGERPRINT_BYTES))
                digest.update(file.read(FINGERPRINT_BYTES))
        _fingerprints[key] = digest.hexdigest()[:16]
    return _fingerprints[key]


def current_batch() -> Optional[Batch]:
    """The step being gathered in this call, if any."""
    return _batch.get()


def describe_project(before: Optional[Mapping], after: Optional[Mapping]) -> str:
    """Say what changed in a project, in words a person would use.

    Args:
        before: The project as it was, or None when it is new.
        after: The project as it is, or None when it is gone.

    Returns:
        A short sentence.
    """
    if after is None:
        return f"刪除了專案「{(before or {}).get('name') or ''}」"
    name = after.get("name") or ""
    if before is None:
        return f"建立了專案「{name}」"
    said = []
    if before.get("name") != name:
        said.append(f"改名為「{name}」")

    def clips(project: Mapping) -> List[Mapping]:
        return [clip for track in project.get("tracks", []) for clip in track.get("clips", [])]

    old, new = clips(before), clips(after)
    if [_clip_key(clip) for clip in old] != [_clip_key(clip) for clip in new]:
        moved = len(old) != len(new)
        said.append(f"剪接改成 {len(new)} 段" if moved else f"調整了 {sum(_clip_key(a) != _clip_key(b) for a, b in zip(old, new))} 段剪接")
    elif old != new:
        said.append("調整了片段的聲音或畫面")
    before_cues = {cue.get("id"): cue.get("text") for cue in before.get("subtitles", [])}
    after_cues = {cue.get("id"): cue.get("text") for cue in after.get("subtitles", [])}
    if before_cues != after_cues:
        if not before_cues:
            said.append(f"上了 {len(after_cues)} 句字幕")
        else:
            edited = sum(1 for key, text in after_cues.items() if before_cues.get(key, text) != text)
            added = len(set(after_cues) - set(before_cues))
            gone = len(set(before_cues) - set(after_cues))
            parts = [f"改了 {edited} 句字幕"] * bool(edited) + [f"加了 {added} 句"] * bool(added) + [f"拿掉 {gone} 句"] * bool(gone)
            said.append("、".join(parts) or "調整了字幕時間")
    if before.get("caption_style") != after.get("caption_style"):
        said.append("改了字幕樣式")
    return "，".join(said) or "調整了專案"


def _clip_key(clip: Mapping) -> tuple:
    """What makes a cut the same cut: which file, which part of it, where it sits."""
    source = clip.get("source_range") or {}
    return (clip.get("asset_id"), str(source.get("start")), str(source.get("end")), str(clip.get("timeline_in")))
