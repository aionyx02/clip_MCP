"""How far a production has got, and whether it is finished, read off what is already stored.

Nothing here is kept: a production keeps only what was asked, and every step is
worked out again from the footage's analyses, the semantic timelines, the plan,
the project and its renders. An edit made by hand in the editor changes those,
and so changes what is read here, without anybody having to say so.

It is finished in one of three ways. The user called it off. Every shape it is
for has a finished video of one version of its project, made after it started,
by whoever made it: an export from the editor is the user's own decision and
counts as much as one the AI started. Or its project is gone. Shapes are compared
by the size they come out at, never by name: a landscape project rendered as it
is delivers `landscape`.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from app.engine.reframe import framed_size
from app.models.job import Job, JobKind, JobStatus
from app.models.production import Production
from app.models.timeline.parts import Project

# The steps a production goes through, in order.
STEPS = ("analysed", "timeline", "plan", "project", "cut", "previewed", "answered", "delivered")
# Once there is a cut on the project, how it got there no longer matters: a project the user
# cut by hand needs no plan before it.
AFTER_THE_CUT = STEPS.index("previewed")


@dataclass(frozen=True)
class Footing:
    """What is stored that a production's progress is read off.

    Attributes:
        analysed: Which of its files have an analysis.
        timelines: The files each semantic timeline was built over.
        plan_id: The plan it is cut from: its own, or the one its project's
            sequence was compiled from; None when there is neither.
        project: Its project; None when it has none, or the one it had is gone.
        cut: Whether the project has anything on its sequence.
        picture: The fingerprint of the project's picture as it is now.
        renders: The project's finished renders, previews included.
        questions: How many questions the AI asked the user on the project
            are still open and unanswered.
    """

    analysed: FrozenSet[str]
    timelines: Sequence[FrozenSet[str]]
    plan_id: Optional[str]
    project: Optional[Project]
    cut: bool
    picture: Optional[str]
    renders: Sequence[Job]
    questions: int


def _finals(production: Production, renders: Sequence[Job]) -> List[Job]:
    """The finished full renders of its project made since it started, by whoever started them."""
    return [job for job in renders
            if job.kind == JobKind.RENDER and not job.preview and job.status == JobStatus.COMPLETED
            and job.project_id == production.project_id and job.created_at >= production.created_at]


def missing_shapes(production: Production, project: Project, renders: Sequence[Job]) -> List[Optional[str]]:
    """Which shapes it is for have no finished video yet.

    The shapes have to be met by one version of the project together, so a
    portrait video of the first cut and a landscape one of the second are not
    the two shapes of one video. Sizes are worked out from the project as it
    is now.

    Args:
        production: The production.
        project: Its project.
        renders: The project's finished renders.

    Returns:
        Its `outputs` still to make, in its order; empty when one version has
        them all. Measured against the newest version with a finished video,
        or all of them when there is none.
    """
    size = lambda frame: framed_size(project.width, project.height, frame)
    finals = sorted(_finals(production, renders), key=lambda job: job.created_at)
    by_version: Dict[Optional[int], Set[Tuple[int, int]]] = {}
    for job in finals:
        by_version.setdefault(job.project_version, set()).add(size(job.frame))
    if any(all(size(frame) in made for frame in production.outputs) for made in by_version.values()):
        return []
    made = by_version[finals[-1].project_version] if finals else set()
    return [frame for frame in production.outputs if size(frame) not in made]


def ended(production: Production, project: Optional[Project], renders: Sequence[Job]) -> Optional[str]:
    """Say whether a production is finished, and how.

    Args:
        production: The production.
        project: Its project as it is now; None when it has none or it is gone.
        renders: The project's finished renders.

    Returns:
        `closed` when the user called it off, `project_gone` when the project
        it was bound to no longer exists, `delivered` when every shape it is
        for has a finished video of one version; None while it goes on.
    """
    if production.closed is not None:
        return "closed"
    if production.project_id is None:
        return None
    if project is None:
        return "project_gone"
    return None if missing_shapes(production, project, renders) else "delivered"


def progress(production: Production, footing: Footing) -> dict:
    """Work out which steps a production has done, and which comes next.

    Args:
        production: The production.
        footing: What is stored about it.

    Returns:
        `steps`, each step and whether it is done; `stage`, the first one not
        done (None when all are); `missing`, the shapes still to deliver; and
        `ended`, as `ended` says.
    """
    project = footing.project
    wanted = set(production.asset_ids)
    current = [job for job in footing.renders if job.preview and job.status == JobStatus.COMPLETED
               and footing.picture is not None and job.picture == footing.picture]
    missing = missing_shapes(production, project, footing.renders) if project is not None else list(
        production.outputs)
    done = {
        "analysed": wanted <= footing.analysed,
        "timeline": not wanted or any(wanted <= built for built in footing.timelines),
        "plan": footing.plan_id is not None,
        "project": project is not None,
        "cut": project is not None and footing.cut,
        "previewed": bool(current),
        "answered": footing.questions == 0,
        "delivered": project is not None and not missing,
    }
    order = STEPS[AFTER_THE_CUT:] if done["cut"] else STEPS
    return {
        "steps": done,
        "stage": next((step for step in order if not done[step]), None),
        "missing": missing,
        "ended": ended(production, project, footing.renders),
    }


# What to do at each step, for the AI picking a production up.
NEXT: Mapping[str, str] = {
    "analysed": "analyze the files not analysed yet with analyze_asset",
    "timeline": "build the semantic timeline over its files with build_semantic_timeline",
    "plan": "write the plan with save_plan, passing production_id",
    "project": "make the project with create_project, passing production_id, then compile_plan the plan onto it",
    "cut": "compile_plan the plan onto the project",
    "previewed": "render a preview of the cut as it is now with render_project and look at it",
    "answered": "the user has not answered every question asked on the project: get_comments lists them",
    "delivered": "render the final cut in each shape still missing with render_project, is_preview: false",
}
