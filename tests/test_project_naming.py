"""Tests for calling a project something other than its ID."""

from app.server import create_project, get_project, list_projects
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
