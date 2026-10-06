"""What a client is shown of the tools: short enough to be read whole, and the right ones named first."""

import asyncio

from fastmcp import Client

from app.server import CORE_TOOLS, mcp

# Claude Code cuts an MCP tool's description off at 2048 characters, and the rule that
# matters is as likely to be in the last paragraph as the first. Kept a little under.
MAX_DESCRIPTION = 2000


def listed() -> dict:
    """Every tool as a client receives it."""
    async def go():
        async with Client(mcp) as client:
            return {tool.name: tool for tool in await client.list_tools()}
    return asyncio.run(go())


def test_every_tool_description_is_read_whole() -> None:
    long = {name: len(tool.description or "") for name, tool in listed().items()
            if len(tool.description or "") > MAX_DESCRIPTION}
    assert long == {}, f"move the detail into the arguments or the skill: {long}"


def test_the_tools_named_first_are_all_real() -> None:
    tools = listed()
    assert [name for name in CORE_TOOLS if name not in tools] == []
    assert all(name in mcp.instructions for name in CORE_TOOLS)


def test_a_new_clip_is_described_without_what_set_clip_look_describes() -> None:
    operations = listed()["apply_edits"].inputSchema["properties"]["operations"]["items"]
    variants = {variant["properties"]["action"].get("const") or variant["properties"]["action"].get("default"): variant
                for variant in operations.get("oneOf") or operations.get("anyOf")}
    assert "transition_in" not in variants["insert_clip"]["properties"]
    assert "transition_in" in variants["set_clip_look"]["properties"]
