"""What a benchmark case is made of.

Footage on its own is not a question. The same six files are a good answer at
ninety seconds and a bad one at six minutes, so a case is footage *plus* the
instruction someone gave *plus* what the answer is not allowed to get wrong.
Instructions run from exact to vague on purpose: a vague one is the only kind
that measures judgement.

Annotations say which stretches have to survive and which have to go. They are
deliberately not a model answer. There is more than one good cut of anything,
so a case scores whether the mines were stepped on and whether the point was
found, not how closely the result resembles one person's version.

Which is why a case says where its instruction and its annotations came from.
A brief the person wrote before anything was cut is worth more than one
reconstructed afterwards, and shortening theirs sits between the two: the
requirements are still theirs, but whoever shortened them knew the answer.
A case seeded from footage somebody has already cut starts with every chosen
stretch marked must-keep, and that is a model answer wearing an annotation's
clothes. It stays marked `draft` until a person has taken out everything that
was merely chosen rather than necessary, and a draft scores nothing.

Every annotation anchors to a file path and a time inside that file. Asset IDs
are generated per workspace, so a corpus keyed on them would score nothing on a
second machine; paths survive being copied about, which is what a corpus is
for.

A case also says what its footage *is*, by digest. The files are often too
large for git, so the repository keeps the case and the person keeps the
footage; without a hash a case that points at the wrong cut of a file, or at a
re-export, would score something and never say that it had measured the wrong
thing.
"""

import hashlib
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

# A digest is read and written as lower-case hex, which is what `hashlib` gives
# and what a hand-edited case is most likely to copy.
SHA256_PATTERN = r"^[0-9a-f]{64}$"
DIGEST_BLOCK_BYTES = 1024 * 1024

class MarkedSpan(BaseModel):
    """A stretch of one source file that the annotation has something to say about."""

    file: str = Field(..., min_length=1, description="Path to the footage, relative to the corpus root")
    start: float = Field(..., ge=0, description="Where it starts in that file, in seconds")
    end: float = Field(..., gt=0, description="Where it ends in that file, in seconds")
    why: str = Field(default="", description="Why it has to stay, or why it has to go")

    @model_validator(mode="after")
    def validate_range(self):
        """Ensure the span has a positive length.

        Returns:
            The validated span.

        Raises:
            ValueError: If `end` is not after `start`.
        """
        if self.end <= self.start:
            raise ValueError(f"{self.file}: a span ends at {self.end}s, which is not after its start at {self.start}s")
        return self

    @property
    def duration(self) -> float:
        """Length of the span in seconds.

        Returns:
            The seconds it covers.
        """
        return self.end - self.start

class FootageFile(BaseModel):
    """One source file a case is made of, identified by size and digest.

    The file itself usually stays on the machine that shot it, so the case
    carries what is needed to tell whether the footage it is about is the
    footage in front of the scorer.

    Attributes:
        file: Path to the footage, relative to the corpus root.
        sha256: Digest of the file, lower-case hex.
        bytes: Length of the file in bytes.
        duration: Length of the file in seconds, when it is known.
        source: How to get the file, in the person's own words: where it was
            shot, which drive it is on, where it can be fetched from.
    """

    file: str = Field(..., min_length=1, description="Path to the footage, relative to the corpus root")
    sha256: str = Field(..., pattern=SHA256_PATTERN, description="Digest of the file, lower-case hex")
    bytes: int = Field(..., gt=0, description="Length of the file in bytes")
    duration: Optional[float] = Field(default=None, gt=0, description="Length of the file in seconds, if known")
    source: str = Field(default="", description="How to get this file, in the person's own words")

class CaseWord(BaseModel):
    """One word of a checked transcript, with where it was said.

    A word is a token the person would agree with rather than whatever the
    recogniser chose to call one: Chinese arrives a character at a time and
    English a word at a time, and both are words here.
    """

    text: str = Field(..., min_length=1, description="The word as it was said")
    start: float = Field(..., ge=0, description="When it starts in that file, in seconds")
    end: float = Field(..., gt=0, description="When it ends in that file, in seconds")

    @model_validator(mode="after")
    def validate_range(self):
        """Ensure the word has a positive length.

        Returns:
            The validated word.

        Raises:
            ValueError: If `end` is not after `start`.
        """
        if self.end <= self.start:
            raise ValueError(f"a word ends at {self.end}s, which is not after its start at {self.start}s")
        return self

class FileTranscript(BaseModel):
    """A transcript of one source file, and who has checked it.

    A draft transcript is the recogniser's own output copied into the corpus.
    It is a starting point for a person to correct, not evidence, so it scores
    nothing until `checked_by` says somebody went through it.
    """

    file: str = Field(..., min_length=1, description="Path to the footage, relative to the corpus root")
    words: List[CaseWord] = Field(
        default_factory=list,
        description="Every word said in the whole file, in order: it is scored against everything the recogniser "
                    "heard in that file. Empty when nobody talks",
    )
    checked_by: Literal["user", "draft"] = Field(
        default="draft",
        description="'draft' is the recogniser's own output nobody has corrected yet, which scores nothing",
    )

    @model_validator(mode="after")
    def validate_order(self):
        """Ensure the words run forwards through the file.

        A transcript out of order is not a slightly wrong answer, it is a file
        that was assembled the wrong way, so it is refused rather than sorted.

        Returns:
            The validated transcript.

        Raises:
            ValueError: If a word starts before the one in front of it.
        """
        previous = None
        for word in self.words:
            if previous is not None and word.start < previous:
                raise ValueError(
                    f"{self.file}: a word starts at {word.start}s, before the one in front of it at {previous}s"
                )
            previous = word.start
        return self

class CaptionLine(BaseModel):
    """Where a caption line should break, as the person would break it.

    The recogniser knows where words are but not where a line stops being
    readable, so this is written by hand and scored later.
    """

    file: str = Field(..., min_length=1, description="Path to the footage, relative to the corpus root")
    start: float = Field(..., ge=0, description="When the line appears, in seconds")
    end: float = Field(..., gt=0, description="When it leaves, in seconds")
    text: str = Field(..., min_length=1, description="The words the line should hold")

    @model_validator(mode="after")
    def validate_range(self):
        """Ensure the line has a positive length.

        Returns:
            The validated line.

        Raises:
            ValueError: If `end` is not after `start`.
        """
        if self.end <= self.start:
            raise ValueError(
                f"{self.file}: a caption line ends at {self.end}s, which is not after its start at {self.start}s"
            )
        return self

class BenchmarkCase(BaseModel):
    """One question: this footage, this instruction, and these rules about the answer."""

    id: str = Field(..., min_length=1, description="Stable name for this case, used to compare runs")
    instruction: str = Field(..., min_length=1, description="What the user asked for, in their own words")
    footage: List[str] = Field(..., min_length=1, description="The source files, relative to the corpus root")
    footage_files: List[FootageFile] = Field(
        default_factory=list,
        description="The same files with the size and digest the corpus was built from. Empty for a case written "
                    "before footage was pinned, whose footage is then taken on trust",
    )
    kinds: List[str] = Field(
        default_factory=list,
        description="What sort of footage this is, in free text: 'solo vlog', 'conversation', 'noisy outdoors', "
                    "'music under speech', 'mixed cameras', 'code switching'",
    )
    split: Literal["tune", "holdout"] = Field(
        default="tune",
        description="Which half of the corpus this case belongs to. A `holdout` case is never used to tune a "
                    "constant or a rule — only to check the result afterwards, so a number tuned on it is not "
                    "a number that predicts anything",
    )
    target_seconds: Optional[float] = Field(
        default=None, gt=0, description="How long the cut should run, when the instruction implies a length",
    )
    must_keep: List[MarkedSpan] = Field(
        default_factory=list, description="Stretches the cut has to contain to have answered the instruction",
    )
    must_drop: List[MarkedSpan] = Field(
        default_factory=list, description="Stretches that must not reach the cut at all",
    )
    cut_windows: List[MarkedSpan] = Field(
        default_factory=list,
        description="Stretches a cut point may land in. Files with none are files nobody has ruled on, "
                    "not files where every cut is wrong",
    )
    rubric: List[str] = Field(
        default_factory=list,
        description="What a good cut of this footage looks like, one judgeable statement per line. "
                    "Carried for L3, not scored here",
    )
    transcripts: List[FileTranscript] = Field(
        default_factory=list,
        description="Checked transcripts of the footage, which are what transcript accuracy is measured against",
    )
    caption_lines: List[CaptionLine] = Field(
        default_factory=list,
        description="Where the person would break a caption line. Carried now, scored later",
    )
    instruction_by: Literal["user", "summarised", "derived"] = Field(
        default="user",
        description="Whose words the instruction is, and how far from the ask it has travelled. 'user' is what "
                    "they wrote; 'summarised' is what they wrote, shortened by somebody who has since seen the "
                    "cut; 'derived' was reconstructed from the cut itself and never asked by anyone",
    )
    annotated_by: Literal["user", "draft"] = Field(
        default="user",
        description="Who settled the annotations. 'draft' means nobody has been through them yet",
    )

    @property
    def is_draft(self) -> bool:
        """Whether this case is still somebody's homework.

        A case whose annotations were extracted from an existing cut marks
        everything that cut used as must-keep, which scores a new plan on how
        closely it reproduces an old one. That is the one thing the corpus is
        not for, so until a person has been through the list and taken out
        what was merely chosen rather than necessary, this case measures
        nothing.

        Returns:
            True while the annotations are an unreviewed draft.
        """
        return self.annotated_by == "draft"

    @model_validator(mode="after")
    def validate_annotations(self):
        """Ensure every annotation points at footage the case actually has.

        A span naming a file that is not in `footage` scores nothing and says
        nothing, so it is a broken case rather than an empty result.

        Returns:
            The validated case.

        Raises:
            ValueError: If an annotation names a file the case does not list,
                the same stretch is marked both must-keep and must-drop, or a
                file is pinned twice in `footage_files`.
        """
        listed = set(self.footage)
        for span in [*self.must_keep, *self.must_drop, *self.cut_windows]:
            if span.file not in listed:
                raise ValueError(
                    f"case {self.id}: an annotation is about {span.file}, which is not in the case's footage"
                )
        for keep in self.must_keep:
            for drop in self.must_drop:
                if keep.file == drop.file and keep.start < drop.end and drop.start < keep.end:
                    raise ValueError(
                        f"case {self.id}: {keep.file} {max(keep.start, drop.start)}s to "
                        f"{min(keep.end, drop.end)}s is marked both must-keep and must-drop"
                    )
        pinned = [entry.file for entry in self.footage_files]
        repeated = sorted({name for name in pinned if pinned.count(name) > 1})
        if repeated:
            raise ValueError(f"case {self.id}: {', '.join(repeated)} is pinned twice in footage_files")
        for entry in self.footage_files:
            if entry.file not in listed:
                raise ValueError(
                    f"case {self.id}: footage_files pins {entry.file}, which is not in the case's footage"
                )
        for transcript in self.transcripts:
            if transcript.file not in listed:
                raise ValueError(
                    f"case {self.id}: a transcript is about {transcript.file}, which is not in the case's footage"
                )
        for line in self.caption_lines:
            if line.file not in listed:
                raise ValueError(
                    f"case {self.id}: a caption line is about {line.file}, which is not in the case's footage"
                )
        return self

def load_case(path: Path) -> BenchmarkCase:
    """Read one case from a JSON file.

    Args:
        path: The file to read.

    Returns:
        The case.

    Raises:
        ValueError: If the file is not valid JSON, or the case does not hold
            up. The message names the file, because a corpus is edited by hand
            and the line that is wrong is the only thing worth saying.
    """
    try:
        return BenchmarkCase.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error

def load_cases(directory: Path) -> List[BenchmarkCase]:
    """Read every case in a corpus directory, in name order.

    Args:
        directory: Directory holding one `.json` file per case.

    Returns:
        The cases, ordered by file name so a run is reproducible.

    Raises:
        ValueError: If the directory does not exist, or two cases share an ID.
            Duplicate IDs would silently overwrite each other's scores.
    """
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory; a corpus is a folder of .json cases")
    paths = sorted(directory.glob("*.json"))
    cases, seen = [], {}
    for path in paths:
        case = load_case(path)
        if case.id in seen:
            raise ValueError(f"{path.name} and {seen[case.id]} are both case {case.id!r}")
        seen[case.id] = path.name
        cases.append(case)
    return cases

def file_digest(path: Union[str, Path]) -> Tuple[str, int]:
    """Measure a file's digest and size without holding it in memory.

    Read a million bytes at a time: footage runs to gigabytes, and a digest
    that needed the whole file in memory would fail on exactly the files a
    corpus is for.

    Args:
        path: The file to read.

    Returns:
        Its sha256 as lower-case hex, and its length in bytes.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while block := handle.read(DIGEST_BLOCK_BYTES):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size

def verify_footage(case: BenchmarkCase, root: Union[str, Path]) -> List[str]:
    """Check that the footage a case pins is the footage on this machine.

    A case naming no `footage_files` has nothing to check and is taken on
    trust; one that pins them is only worth scoring if every file matches.

    Args:
        case: The case whose footage is being checked.
        root: The corpus root the case's paths are relative to.

    Returns:
        One line per file that is missing, a different size, or a different
        digest; empty when everything matches.
    """
    base = Path(root)
    problems: List[str] = []
    for entry in case.footage_files:
        path = base / entry.file
        if not path.is_file():
            problems.append(f"{entry.file} is missing under {base}")
            continue
        digest, size = file_digest(path)
        if size != entry.bytes:
            problems.append(
                f"{entry.file} has changed size: {size} bytes now, {entry.bytes} recorded"
            )
            continue
        if digest != entry.sha256:
            problems.append(
                f"{entry.file} has changed digest: {digest} now, {entry.sha256} recorded"
            )
    return problems
