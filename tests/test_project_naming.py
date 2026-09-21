"""Tests for calling a project something other than its ID."""

from app.models.timeline import Project
from app.server import MAX_OUTPUT_NAME, _output_name, create_project, get_project, list_projects
from helpers import edit

def test_a_project_can_be_named_when_it_is_created() -> None:
    project = create_project(name="EP1 台北", width=1080, height=1920)
    assert project["name"] == "EP1 台北"
    assert get_project(project["id"])["name"] == "EP1 台北"

def test_a_project_without_a_name_says_so_rather_than_inventing_one() -> None:
    project = create_project()
    assert project["name"] is None
    # Left out of the state the same way a clip's untouched fields are.
    assert "name" not in get_project(project["id"])

def test_the_listing_shows_the_names_so_several_projects_can_be_told_apart() -> None:
    first = create_project(name="EP1 台北")["id"]
    second = create_project(name="EP2 台南")["id"]
    listed = {row["id"]: row["name"] for row in list_projects()["projects"]}
    assert listed[first] == "EP1 台北" and listed[second] == "EP2 台南"

def test_a_project_can_be_renamed_afterwards() -> None:
    project = create_project(name="暫定")["id"]
    edit(project, [{"action": "rename_project", "name": "EP3 結案這一天"}])
    assert get_project(project)["name"] == "EP3 結案這一天"

def test_renaming_to_nothing_takes_the_name_away() -> None:
    project = create_project(name="EP1")["id"]
    edit(project, [{"action": "rename_project"}])
    assert "name" not in get_project(project)

def test_a_name_that_is_only_spaces_does_not_count_as_one() -> None:
    assert create_project(name="   ")["name"] is None


def named(name, kind="output"):
    """Name the render of a project called this.

    Args:
        name: What the project is called, or None for an unnamed one.
        kind: `output` or `preview`.

    Returns:
        The file name a render would be written to.
    """
    return _output_name(Project(id="c91b0000-0000-0000-0000-000000000000", name=name, width=1080, height=1920), kind)


def test_a_render_is_named_after_the_project() -> None:
    """Two uuids in a path is why finished cuts get copied somewhere else under a made-up name."""
    assert named("EP1 台北") == "EP1 台北_output.mp4"
    assert named("EP1 台北", "preview") == "EP1 台北_preview.mp4"


def test_a_name_a_filesystem_would_refuse_is_made_safe() -> None:
    """Whoever is editing names the project, so a render cannot assume anything about it."""
    assert named(r'a/b\c:d*e?f"g<h>i|j') == "a b c d e f g h i j_output.mp4"
    # Control characters go the same way, and the runs of spaces they leave collapse.
    assert named("EP1\n\t台北") == "EP1 台北_output.mp4"


def test_a_project_with_no_usable_name_falls_back_to_its_id() -> None:
    """A file called `_output.mp4`, or one ending in a dot, is worse than an unreadable one."""
    for name in (None, "", "   ", "...", "///"):
        assert named(name) == "c91b0000-0000-0000-0000-000000000000_output.mp4", name


def test_a_very_long_name_is_cut_to_something_a_filesystem_will_take() -> None:
    """Paths have limits, and a project name has none."""
    assert named("影" * 200) == "影" * MAX_OUTPUT_NAME + "_output.mp4"