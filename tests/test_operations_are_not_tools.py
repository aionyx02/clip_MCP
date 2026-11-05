"""Tests for what an AI is told when it calls an edit as if it were a tool of its own."""

import asyncio

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from app.models.timeline.operations import OPERATION_NAMES
from app.server import mcp

def call(name: str) -> None:
    """Call a tool the way a client does, by name, with no arguments.

    Args:
        name: The tool's name.
    """
    async def calling() -> None:
        async with Client(mcp) as client:
            await client.call_tool(name, {})
    asyncio.run(calling())

def test_every_edit_is_named() -> None:
    assert {"set_caption_style", "insert_clip", "remove_text"} <= OPERATION_NAMES

def test_an_edit_called_as_a_tool_says_where_it_is() -> None:
    with pytest.raises(ToolError, match=r"operation of apply_edits.*\"action\": \"set_caption_style\""):
        call("set_caption_style")

def test_a_name_that_is_nothing_is_still_unknown() -> None:
    with pytest.raises(ToolError, match="Unknown tool") as refused:
        call("no_such_thing")
    assert "apply_edits" not in str(refused.value)
