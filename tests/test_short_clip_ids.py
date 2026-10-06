"""The AI sees a clip as `u0034` everywhere, and can write it that way everywhere."""

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client

from app.models.plan import EditPlan, short_clip_ids, with_whole_clip_ids
from app.server import build_semantic_timeline, import_asset, mcp, query_clips
from app.server import repo as server_repo
from test_edit_plan import FOOTAGE
from test_semantic_timeline import analysis


def call(name: str, arguments: dict):
    """Call a tool the way a client does, through everything the server puts in front of it.

    Returns:
        The result's structured content, or the error's message.
    """
    async def go():
        async with Client(mcp) as client:
            result = await client.call_tool(name, arguments, raise_on_error=False)
            if result.is_error:
                return result.content[0].text
            return result.structured_content
    return asyncio.run(go())


@pytest.fixture
def timeline(media: Path) -> dict:
    """A timeline over one real file, and the IDs of its spoken clips, whole."""
    asset_id = import_asset(str(media / "wide.mp4"))["id"]
    server_repo.save_analysis(analysis(asset_id=asset_id, **FOOTAGE))
    built = build_semantic_timeline([asset_id], rebuild=True)
    speech = query_clips(timeline_id=built["timeline_id"], text="句話")["clips"]
    return {"timeline_id": built["timeline_id"], "clips": [clip["clip_id"] for clip in speech], "asset_id": asset_id}


def test_the_timeline_is_taken_off_every_clip_id_said_back() -> None:
    said = "tl_37f37b83:u0034 runs into tl_37f37b83:s0002, timeline tl_37f37b83 itself stays"
    assert short_clip_ids(said) == "u0034 runs into s0002, timeline tl_37f37b83 itself stays"


def test_a_plan_written_with_short_ids_is_stored_whole() -> None:
    plan = EditPlan.model_validate({
        "timeline_id": "tl_0000abcd",
        "selections": [{"clip_id": "u0001", "beat_id": "b1", "trim": {"kind": "keep", "keep_clip_ids": ["u0002"]}}],
        "rejected": [{"clip_id": "tl_0000abcd:u0003"}],
    })
    assert plan.selections[0].clip_id == "tl_0000abcd:u0001"
    assert plan.selections[0].trim.keep_clip_ids == ["tl_0000abcd:u0002"]
    assert plan.rejected[0].clip_id == "tl_0000abcd:u0003"


def test_only_fields_that_name_a_clip_are_made_whole() -> None:
    made = with_whole_clip_ids({"beat_id": "b1", "rationale": "u0001 說得最好", "clip_id": "u0001"}, "tl_0000abcd")
    assert made == {"beat_id": "b1", "rationale": "u0001 說得最好", "clip_id": "tl_0000abcd:u0001"}


def test_a_client_never_sees_a_whole_clip_id(timeline: dict) -> None:
    short = [clip.split(":")[1] for clip in timeline["clips"]]
    saved = call("save_plan", {"plan": {
        "timeline_id": timeline["timeline_id"], "goal": "做一支短的",
        "beats": [{"id": "b1", "name": "主體", "intent": "講完一件事"}],
        "selections": [{"clip_id": short[0], "beat_id": "b1", "rationale": "開場"}],
    }})
    assert saved["problems"] == []
    # Stored whole, so nothing that reads the plan has to know about the short way.
    assert server_repo.get_plan(saved["plan_id"]).selections[0].clip_id == timeline["clips"][0]

    read = call("get_plan", {"plan_id": saved["plan_id"]})
    assert read["plan"]["selections"][0]["clip_id"] == short[0]
    assert read["plan"]["timeline_id"] == timeline["timeline_id"]
    found = call("query_clips", {"timeline_id": timeline["timeline_id"], "text": "句話"})
    assert ":" not in json.dumps([clip["clip_id"] for clip in found["clips"]])

    amended = call("amend_plan", {"plan_id": saved["plan_id"], "expected_version": 1, "amendments": [
        {"action": "add_selection", "selection": {"clip_id": short[1], "beat_id": "b1"}},
    ]})
    assert amended["problems"] == []
    assert [item.clip_id for item in server_repo.get_plan(saved["plan_id"]).selections] == timeline["clips"][:2]


def test_an_error_names_the_clip_the_short_way(timeline: dict) -> None:
    said = call("amend_plan", {"plan_id": call("save_plan", {"plan": {
        "timeline_id": timeline["timeline_id"],
        "beats": [{"id": "b1", "name": "主體", "intent": "講完一件事"}],
        "selections": [{"clip_id": timeline["clips"][0], "beat_id": "b1"}],
    }})["plan_id"], "expected_version": 1, "amendments": [{"action": "drop_selection", "clip_id": "u9999"}]})
    assert "u9999" in said and f"{timeline['timeline_id']}:" not in said


def test_a_clip_is_read_by_its_short_id(timeline: dict) -> None:
    short = timeline["clips"][0].split(":")[1]
    clip = call("get_semantic_clip", {"clip_id": short, "timeline_id": timeline["timeline_id"]})
    assert clip["clip_id"] == short


def test_reading_through_footage_carries_each_file_s_notes_once(timeline: dict) -> None:
    from app.server import edit_asset

    edit_asset(timeline["asset_id"], "後面片段請剪除")
    try:
        read = query_clips(timeline_id=timeline["timeline_id"], brief=True)
        assert read["notes"] == {"wide.mp4": "後面片段請剪除"}
        assert not any("後面片段請剪除" in line for line in read["lines"])
    finally:
        edit_asset(timeline["asset_id"], "")
