"""A file can be named by its exact file name wherever a tool wants an asset ID."""

import asyncio
import shutil
from pathlib import Path

from fastmcp import Client

from app.server import import_asset, mcp


def call(name: str, arguments: dict):
    """Call a tool the way a client does; returns the structured result or the error's message."""
    async def go():
        async with Client(mcp) as client:
            result = await client.call_tool(name, arguments, raise_on_error=False)
            return result.content[0].text if result.is_error else result.structured_content
    return asyncio.run(go())


def test_a_file_is_found_by_its_name(media: Path, tmp_path: Path) -> None:
    named = tmp_path / "抽杯架_名字找得到.mp4"
    shutil.copy(media / "tall.mp4", named)
    asset_id = import_asset(str(named))["id"]
    planned = call("analyze_asset", {"asset_ids": [named.name], "dry_run": True})
    assert planned["would_start"] == [asset_id]


def test_two_files_with_one_name_are_not_guessed_between(media: Path, tmp_path: Path) -> None:
    ids = []
    for folder in ("一", "二"):
        (tmp_path / folder).mkdir()
        shutil.copy(media / "tall.mp4", tmp_path / folder / "同名的檔案.mp4")
        ids.append(import_asset(str(tmp_path / folder / "同名的檔案.mp4"))["id"])
    said = call("get_analysis", {"asset_id": "同名的檔案.mp4"})
    assert "2 files are called 同名的檔案.mp4" in said and all(asset_id in said for asset_id in ids)


def test_a_name_nothing_has_reaches_the_tool_as_it_was() -> None:
    assert "asset 沒有這個檔案.mp4 " in call("get_analysis", {"asset_id": "沒有這個檔案.mp4"})
