"""Tests that the skill and the server describe the same thing.

The guide tells the model which tools and operations exist and which extra
files to read. If they drift apart the model is told to call something that
is not there, which these tests are meant to catch.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import get_args

import pytest

from app.models.timeline import EditOperation
from app.server import SKILLS_DIR, mcp

SKILL_DIR = Path(SKILLS_DIR) / "clip-editing"
SKILL = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")

def resources() -> dict:
    """List the skill resources the server publishes.

    Returns:
        Each resource's text, keyed by URI.
    """
    async def read_all() -> dict:
        listed = await mcp._list_resources()
        out = {}
        for resource in listed:
            result = await mcp.read_resource(str(resource.uri))
            out[str(resource.uri)] = result.contents[0].content
        return out

    return asyncio.run(read_all())

def operation_actions() -> set:
    """Collect every edit operation's `action` value.

    Returns:
        The action names the server accepts.
    """
    actions = set()
    for model in get_args(get_args(EditOperation)[0]):
        actions.add(get_args(model.model_fields["action"].annotation)[0])
    return actions

def tool_names() -> set:
    """Collect every registered tool name.

    Returns:
        The tool names the server exposes.
    """
    return {tool.name for tool in asyncio.run(mcp._list_tools())}

def test_every_file_the_guide_points_at_is_published() -> None:
    referenced = set(re.findall(r"skill://clip-editing/[\w.\-]+", SKILL))
    assert referenced, "the guide should point at its supporting files"
    published = resources()
    for uri in referenced:
        assert uri in published, f"{uri} is referenced by SKILL.md but not published"
        assert len(published[uri]) > 200, f"{uri} is published but nearly empty"

def test_the_manifest_lists_every_file_in_the_skill(tmp_path: Path) -> None:
    manifest = json.loads(resources()["skill://clip-editing/_manifest"])
    listed = {entry["path"] for entry in manifest["files"]}
    on_disk = {path.name for path in SKILL_DIR.glob("*.md")}
    assert listed == on_disk

def test_every_tool_is_mentioned_in_the_guide() -> None:
    # A tool the guide never names is a tool the model will not use.
    missing = sorted(name for name in tool_names() if name not in SKILL)
    assert missing == [], f"tools missing from SKILL.md: {missing}"

def test_every_edit_operation_is_mentioned_in_the_guide() -> None:
    missing = sorted(action for action in operation_actions() if action not in SKILL)
    assert missing == [], f"operations missing from SKILL.md: {missing}"

def test_the_guide_never_names_an_operation_that_does_not_exist() -> None:
    used = set(re.findall(r'"action":\s*"([a-z_]+)"', SKILL))
    unknown = sorted(used - operation_actions())
    assert unknown == [], f"SKILL.md uses operations the server does not accept: {unknown}"

@pytest.mark.parametrize("name", ["pacing-and-structure.md", "examples.md"])
def test_the_split_out_files_say_what_they_are_for(name: str) -> None:
    text = (SKILL_DIR / name).read_text(encoding="utf-8")
    assert text.startswith("# ")
    assert "SKILL.md" in text, "a supporting file should say where it is referenced from"
