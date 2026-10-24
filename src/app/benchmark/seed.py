"""Writing a draft benchmark case from footage already on this machine.

A case is not something to type out by hand. The footage is on a drive, the
workspace has probably transcribed some of it, and the hashes have to be read
off the files themselves; so this builds the draft and leaves the two things
only a person can supply — the instruction and the review — to them.

It never lists a folder to decide what the footage is. Finished cuts sit
beside the raw files in the folders people keep them in, and a listing would
pick one up as source material and hand the answer to whatever is being
measured. Files are named one at a time, on purpose.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from app.benchmark.case import BenchmarkCase, CaseWord, FileTranscript, FootageFile, file_digest

def _duration(path: Path) -> Optional[float]:
    """Read a file's duration, if FFprobe can be asked and answers.

    Seeding has to work on a machine without FFmpeg — a corpus can be built
    from files copied off a camera anywhere — so a probe that fails is a
    missing duration rather than an error.

    Args:
        path: The file to probe.

    Returns:
        Its duration in seconds, or None when it could not be read.
    """
    try:
        from app.engine.probe import probe_file
        value = probe_file(str(path)).get("format", {}).get("duration")
        return float(value) if value is not None else None
    except Exception:
        return None

def _transcript(repository, path: Path, name: str) -> Optional[FileTranscript]:
    """Copy the workspace's transcript of a file into a draft transcript.

    The transcript is the recogniser's own output, so what comes back is a
    draft: something for a person to correct, never evidence on its own.

    Args:
        repository: The workspace repository, or a stand-in for a test.
        path: Absolute path of the file on disk.
        name: The file's path relative to the corpus root.

    Returns:
        The draft transcript, or None when the workspace has no analysis of
        the file or no transcript in it.
    """
    asset = repository.find_asset_by_path(str(path))
    if asset is None:
        return None
    analysis = repository.get_analysis(asset.id)
    if analysis is None or analysis.transcript is None:
        return None
    words = [
        CaseWord(text=word.text, start=word.start, end=word.end)
        for segment in analysis.transcript.segments
        for word in segment.words
    ]
    return FileTranscript(file=name, words=words, checked_by="draft")

def main(argv: Optional[Sequence[str]] = None, repository=None) -> int:
    """Build a draft case from files under a corpus root.

    Args:
        argv: Arguments after the module name; `sys.argv[1:]` when omitted.
        repository: The workspace repository to copy transcripts from. Opened
            lazily when omitted, so a caller that already has one — a test,
            or the editor — does not open a second database.

    Returns:
        0 when the case was written, 1 when it was refused because the file
        already exists and `--force` was not given.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.benchmark.seed",
        description="Write a draft benchmark case for footage already on this machine.",
    )
    parser.add_argument("--root", required=True, type=Path,
                        help="Folder the corpus paths are relative to, holding the footage")
    parser.add_argument("--case-id", required=True, help="Name for the case, which is also its file name")
    parser.add_argument("--instruction", required=True, help="What was asked for, in the user's own words")
    parser.add_argument("--corpus", type=Path, default=Path("corpus"),
                        help="Folder the case is written to (default: corpus in the current folder)")
    parser.add_argument("--instruction-by", choices=("user", "summarised", "derived"), default="user",
                        help="Whose words the instruction is (default: user)")
    parser.add_argument("--split", choices=("tune", "holdout"), default="tune",
                        help="Whether the case may be used to tune constants (default: tune)")
    parser.add_argument("--force", action="store_true", help="Overwrite a case that already exists")
    parser.add_argument("files", nargs="+", help="Footage files, named one by one and relative to --root")
    args = parser.parse_args(argv)

    if repository is None:
        from app.runtime import repo as workspace_repo
        repository = workspace_repo

    root = args.root
    footage: List[str] = []
    pinned: List[FootageFile] = []
    transcripts: List[FileTranscript] = []
    for name in args.files:
        path = Path(name)
        if path.is_absolute():
            try:
                relative = path.relative_to(root)
            except ValueError:
                print(f"{name} is not under {root}", file=sys.stderr)
                return 1
        else:
            relative = path
            path = root / relative
        if not path.is_file():
            print(f"{name} is not a file under {root}", file=sys.stderr)
            return 1
        name = relative.as_posix()
        digest, size = file_digest(path)
        footage.append(name)
        pinned.append(FootageFile(
            file=name, sha256=digest, bytes=size, duration=_duration(path),
        ))
        transcript = _transcript(repository, path, name)
        if transcript is not None:
            transcripts.append(transcript)

    case = BenchmarkCase(
        id=args.case_id,
        instruction=args.instruction,
        footage=footage,
        footage_files=pinned,
        transcripts=transcripts,
        split=args.split,
        instruction_by=args.instruction_by,
        annotated_by="draft",
    )

    output = args.corpus / f"{case.id}.json"
    if output.exists() and not args.force:
        print(f"{output} already exists; pass --force to overwrite it", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(case.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
