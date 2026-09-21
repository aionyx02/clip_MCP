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
"""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

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

class BenchmarkCase(BaseModel):
    """One question: this footage, this instruction, and these rules about the answer."""

    id: str = Field(..., min_length=1, description="Stable name for this case, used to compare runs")
    instruction: str = Field(..., min_length=1, description="What the user asked for, in their own words")
    footage: List[str] = Field(..., min_length=1, description="The source files, relative to the corpus root")
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
        description="Stretches a cut point may land in. Files with none are files nobody has ruled on, not files where every cut is wrong",
    )
    rubric: List[str] = Field(
        default_factory=list,
        description="What a good cut of this footage looks like, one judgeable statement per line. Carried for L3, not scored here",
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
                or the same stretch is marked both must-keep and must-drop.
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
