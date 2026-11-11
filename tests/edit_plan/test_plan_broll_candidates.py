"""B-roll candidates: which shots are offered over a stretch, on what facts, and picks pinned to the offer."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest
from pydantic import TypeAdapter

from app.engine import meaning
from app.engine.plan import broll_candidates as offering
from app.engine.plan.beats import plan_pieces
from app.engine.plan.broll import Stretch, stretch_over
from app.models.media import Asset, Span
from app.models.plan import BrollShot, PlanAmendment, Selection
from app.models.semantic import ClipKind, ClipLevel, SemanticClip, SemanticTimeline, Tag, TagSource
from app.runtime import repo as runtime_repo
from app.tools.catalog import import_asset
from app.tools.editing import create_project
from app.tools.planning import amend_plan, compile_plan, propose_broll, save_plan, validate_plan
from test_meaning_search import _stand_in
from test_semantic_timeline import analysis

from edit_plan.shared import BROLL_ASSET, covered, plan_for, speech_of

TALK = "talk"
NOON = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def shot(clip_id: str, asset_id: str, seconds: float = 8.0, kind: ClipKind = ClipKind.AMBIENT, start: float = 0.0,
         **scores) -> SemanticClip:
    """One clip of a made-up timeline."""
    return SemanticClip(id=f"tl_x:{clip_id}", timeline_id="tl_x", asset_id=asset_id, level=ClipLevel.UTTERANCE,
                        source_range=Span(start=start, end=start + seconds), kind=kind, scores=scores)


def filed(asset_id: str, name: str, minutes: float = 0.0, has_video: bool = True) -> Asset:
    """A file with a name, recorded some minutes after noon."""
    return Asset(id=asset_id, path=f"/footage/{name}", has_video=has_video, has_audio=True,
                 recorded_at=NOON + timedelta(minutes=minutes))


def stretch(said: str = "可以看一下它的機構", under=(TALK,), seconds: float = 4.0, exposure: float = 0.2) -> Stretch:
    """A stretch over a talk, as `stretch_over` would work it out."""
    return Stretch(over_clip_id="tl_x:u0000", opened=0.0, seconds=seconds, said=said, under=tuple(under),
                   exposure=exposure)


def offered(clips: List[SemanticClip], assets: dict, said: str = "可以看一下它的機構", **fields) -> tuple:
    """Offer candidates over one stretch, with the stand-in meaning model."""
    pool = offering.eligible(clips, assets)
    found = stretch(said, **fields)
    texts = {clip.id: offering.reading(clip, assets[clip.asset_id]) for clip in pool}
    passages = _stand_in([text for text in texts.values()]) if texts else []
    question = _stand_in([said], query=True)[0]
    meaning_of = {clip_id: float(vector @ question) for (clip_id, text), vector in zip(texts.items(), passages) if text}
    return offering.offer_candidates(found, plan_for([]), [], pool, assets, meaning_of, NOON)


def ids(candidates: List[dict]) -> List[str]:
    return [item["clip_id"].split(":")[1] for item in candidates]


# --- what is never offered ---------------------------------------------------------------------

def test_the_talk_itself_and_what_is_no_shot_are_left_out_with_counts() -> None:
    assets = {
        TALK: filed(TALK, "1-講解.mp4"), "a": filed("a", "2-機構測試.mp4"), "b": filed("b", "3-人.mp4"),
        "c": filed("c", "4-音樂.wav", has_video=False), "d": filed("d", "5-黑畫面.mp4"), "e": filed("e", "6-閃.mp4"),
    }
    clips = [
        shot("u0000", TALK, faces=0.0),                                 # the same camera, between lines
        shot("u0001", "a"),                                             # a cutaway
        shot("u0002", "b", kind=ClipKind.SPEECH, faces=0.9),           # somebody else talking to camera
        shot("u0003", "c"),                                             # no picture
        shot("u0004", "d", kind=ClipKind.UNUSABLE),                    # black
        shot("u0005", "e", seconds=0.5),                                # a flicker
    ]
    candidates, left_out, _ = offered(clips, assets)
    assert ids(candidates) == ["u0001"]
    assert left_out == {
        "the same camera as the picture it would cover": 1, "somebody talking to camera": 1, "no picture": 1,
        "black or frozen": 1, "shorter than 1s": 1,
    }


def test_somebody_talking_off_camera_over_a_demonstration_is_still_offered() -> None:
    # Hands showing the thing while the voice explains it: speech, but no face, and good B-roll.
    assets = {TALK: filed(TALK, "1-講解.mp4"), "a": filed("a", "9-講解測試機構2.mp4")}
    candidates, _, _ = offered([shot("u0001", "a", kind=ClipKind.SPEECH, faces=0.0)], assets)
    assert ids(candidates) == ["u0001"]


# --- the order, and why ------------------------------------------------------------------------

def test_shots_sharing_a_word_with_what_is_said_come_first_and_say_so() -> None:
    assets = {TALK: filed(TALK, "1-講解.mp4"), "a": filed("a", "26-成品展示.mp4"), "b": filed("b", "13-機構測試4.mp4")}
    candidates, _, spread = offered([shot("u0001", "a"), shot("u0002", "b")], assets)
    assert ids(candidates) == ["u0002", "u0001"]
    first = candidates[0]
    assert first["matched"] == ["機構"] and first["rank"] == 1
    assert "shares 「機構」" in first["why"] and "file name" in first["why"]
    assert candidates[1]["matched"] == [] and "shares no word" in candidates[1]["why"]
    assert spread is not None and spread["highest"] >= spread["lowest"]


def test_what_is_said_in_a_shot_is_not_what_it_shows() -> None:
    # Two talks sharing 「等一下」 say nothing about either picture.
    assets = {TALK: filed(TALK, "1-講解.mp4"), "a": filed("a", "DJI_0012.MP4")}
    other = shot("u0001", "a", kind=ClipKind.SPEECH, faces=0.0).model_copy(update={"text": "等一下看它的機構"})
    candidates, _, _ = offered([other], assets)
    assert candidates[0].get("meaning") is None and "nothing says what is in it" in candidates[0]["why"]


def test_a_description_and_tags_are_read_as_well_as_the_file_name() -> None:
    assets = {TALK: filed(TALK, "1-講解.mp4"), "a": filed("a", "IMG_0001.MOV")}
    described = shot("u0001", "a").model_copy(update={
        "description": "手把杯子從架上抽出來", "tags": [Tag(value="機構特寫", source=TagSource.MODEL)]})
    candidates, _, _ = offered([described], assets)
    assert candidates[0]["matched"] == ["機構"]
    assert [source.split(" ")[0] for source in candidates[0]["read_from"]] == ["description", "tags"]


def test_shots_nothing_describes_are_offered_nearest_the_talk_in_recording_time() -> None:
    assets = {TALK: filed(TALK, "DJI_0001.MP4"), **{
        name: filed(name, f"DJI_{index:04d}.MP4", minutes) for index, (name, minutes) in
        enumerate((("far", 300), ("near", -3), ("mid", 40)), start=2)}}
    candidates, _, _ = offered([shot("u0001", "far"), shot("u0002", "near"), shot("u0003", "mid")], assets)
    assert ids(candidates) == ["u0002", "u0003", "u0001"]
    assert "recorded 3m before the talk" in candidates[0]["why"]


def test_a_few_places_are_kept_for_footage_nobody_labelled() -> None:
    assets = {TALK: filed(TALK, "1-講解.mp4"), "x": filed("x", "DJI_0099.MP4")}
    labelled = []
    for index in range(12):
        assets[f"f{index}"] = filed(f"f{index}", f"{index}-機構測試{index}.mp4")
        labelled.append(shot(f"u{index + 1:04d}", f"f{index}"))
    candidates, left_out, _ = offered([*labelled, shot("u0099", "x")], assets)
    assert len(candidates) == offering.CANDIDATES_PER_STRETCH
    assert ids(candidates)[-1] == "u0099"
    assert sum(left_out.values()) == 13 - offering.CANDIDATES_PER_STRETCH


def test_one_file_does_not_fill_the_list() -> None:
    assets = {TALK: filed(TALK, "1-講解.mp4"), "a": filed("a", "2-機構測試.mp4"), "b": filed("b", "3-成品.mp4")}
    same = [shot(f"u{index:04d}", "a", start=index * 10.0).model_copy(update={"description": f"機構 第{index}段"})
            for index in range(1, 5)]
    candidates, _, _ = offered([*same, shot("u0009", "b")], assets)
    assert sum(1 for item in candidates if item["file"] == "2-機構測試.mp4") == offering.PER_FILE
    assert "u0009" in ids(candidates)


def test_clips_of_one_file_that_read_the_same_are_one_candidate_the_longest() -> None:
    assets = {TALK: filed(TALK, "1-講解.mp4"), "a": filed("a", "2-機構測試.mp4")}
    candidates, left_out, _ = offered([shot("u0001", "a", seconds=2.0), shot("u0002", "a", seconds=9.0, start=5)],
                                      assets)
    assert ids(candidates) == ["u0002"]
    assert left_out == {"another clip of the same file reads the same; the longest is offered": 1}


# --- the facts beside each one -----------------------------------------------------------------

def test_each_candidate_says_whether_it_fits_how_steady_and_how_bright_it_is() -> None:
    assets = {TALK: filed(TALK, "1-講解.mp4"), "a": filed("a", "2-機構測試.mp4"), "b": filed("b", "3-機構.mp4")}
    clips = [shot("u0001", "a", seconds=2.0, shake=0.05, exposure=0.6, motion=0.001),
             shot("u0002", "b", seconds=9.0, shake=0.0, exposure=0.22, motion=0.03)]
    candidates, _, _ = offered(clips, assets, seconds=4.0, exposure=0.2)
    rough, clean = sorted(candidates, key=lambda item: item["clip_id"])
    assert rough["fits"] is False and rough["from_seconds_up_to"] == 0
    assert any("handheld" in note for note in rough["notes"])
    assert any("brighter than the picture it covers" in note for note in rough["notes"])
    assert any("barely moves" in note for note in rough["notes"])
    assert clean["fits"] is True and clean["from_seconds_up_to"] == 5.0 and "notes" not in clean
    assert (clean["shake"], clean["exposure"], clean["motion"]) == (0.0, 0.22, 0.03)


def test_a_shot_already_used_or_already_on_screen_says_so() -> None:
    by_id, children, assets, clips = covered()
    speech, quiet = speech_of(clips), next(clip for clip in clips if clip.asset_id == BROLL_ASSET)
    plan = plan_for([Selection(clip_id=speech[0].id, beat_id="b1"), Selection(clip_id=quiet.id, beat_id="b1")],
                    broll=[BrollShot(clip_id=quiet.id, over_clip_id=speech[1].id, seconds=1.0)])
    pieces = plan_pieces(plan, by_id, children, assets)
    found, _ = stretch_over(plan, pieces, by_id, speech[0].id)
    candidates, _, _ = offering.offer_candidates(found, plan, pieces, offering.eligible(clips, assets), assets,
                                                 None, None)
    notes = next(item for item in candidates if item["clip_id"] == quiet.id)["notes"]
    assert any("already shows" in note and "repeated" in note for note in notes)
    assert any(f"already covers {speech[1].id}" in note for note in notes)


def test_the_same_best_shot_offered_everywhere_is_said_on_each() -> None:
    slots = [{"over_clip_id": over, "candidates": [{"clip_id": "u1"}, {"clip_id": "u2"}]} for over in ("a", "b")]
    offering.mark_variety(slots)
    assert slots[0]["candidates"][0]["also_near_the_top_over"] == ["b"]
    assert slots[1]["candidates"][1]["also_near_the_top_over"] == ["a"]


def test_words_shared_are_whole_and_filler_is_not_one() -> None:
    assert offering.matched_words("可以看一下它的機構這樣抽下來", "一下 機構測試") == ["機構"]
    assert offering.matched_words("the hinge works", "Hinge close-up") == ["hinge"]
    assert offering.file_label(filed("a", "13-機構測試4.mp4")) == "機構測試4"
    assert offering.file_label(filed("a", "DJI_20260828125230_0735_D.MP4")) == ""
    assert offering.file_label(filed("a", "video_2026-08-29_23-19-10.mp4")) == ""


# --- picks pinned to the offer ------------------------------------------------------------------

@pytest.fixture
def two_files(media: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A plan of talk over `wide.mp4`, with a silent second file on the timeline to cover it with."""
    from app.models.plan import Beat, EditPlan
    from app.tools.timelines import build_semantic_timeline, query_clips

    monkeypatch.setattr(meaning, "embed", _stand_in)
    from edit_plan.shared import FOOTAGE

    talk = import_asset(str(media / "wide.mp4"))["id"]
    runtime_repo.save_analysis(analysis(asset_id=talk, **FOOTAGE))
    quiet = import_asset(str(media / "silent.mp4"))["id"]
    runtime_repo.save_analysis(analysis(duration=4.0, scenes=[(0.0, 4.0)], asset_id=quiet))
    built = build_semantic_timeline([talk, quiet], rebuild=True)
    speech = query_clips(timeline_id=built["timeline_id"], text="句話")["clips"]
    cover = query_clips(timeline_id=built["timeline_id"], asset_ids=[quiet])["clips"][0]
    saved = save_plan(EditPlan(timeline_id=built["timeline_id"], beats=[Beat(id="b1", name="主體")],
                               selections=[Selection(clip_id=speech[place]["clip_id"], beat_id="b1")
                                           for place in (0, 2)]))
    over = speech[0]["clip_id"]
    return {"plan_id": saved["plan_id"], "over": over, "cover": cover["clip_id"], "talk": speech[2]["clip_id"],
            "offer": propose_broll(saved["plan_id"], over_clip_ids=[over])}


def pick(two_files: dict, **fields) -> dict:
    """Put one shot in the plan, as offered unless a field says otherwise."""
    version = runtime_repo.get_plan(two_files["plan_id"]).version
    return amend_plan(two_files["plan_id"], version, TypeAdapter(List[PlanAmendment]).validate_python([{
        "action": "add_broll", "shot": {"clip_id": two_files["cover"], "over_clip_id": two_files["over"],
                                        "seconds": 1.5, "candidate_hash": two_files["offer"]["candidate_hash"],
                                        **fields}}]))


def test_a_pick_from_the_offer_compiles(two_files: dict) -> None:
    assert pick(two_files)["problems"] == []
    project = create_project(name="插入", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=two_files["plan_id"])


@pytest.mark.parametrize("fields, said", [
    ({"candidate_hash": ""}, "was not picked from propose_broll's candidates"),
    ({"candidate_hash": "000000000000"}, "no candidates were offered on this timeline under hash"),
    ({"over_clip_id": "TALK"}, "offered nothing over"),
    ({"clip_id": "TALK"}, "is not one of the candidates offered over"),
])
def test_a_pick_that_was_not_offered_is_refused_with_the_reason(two_files: dict, fields: dict, said: str) -> None:
    fields = {key: two_files["talk"] if value == "TALK" else value for key, value in fields.items()}
    problems = pick(two_files, **fields)["problems"]
    assert any(said in problem for problem in problems), problems
    with pytest.raises(ValueError, match="cannot be compiled"):
        project = create_project(name="插入", width=640, height=360)["id"]
        compile_plan(project_id=project, expected_version=1, plan_id=two_files["plan_id"])


def test_a_pick_from_footage_described_since_is_stale(two_files: dict) -> None:
    from app.models.semantic import ClipDescription
    from app.tools.timelines import set_clip_tags

    assert pick(two_files)["problems"] == []
    set_clip_tags([ClipDescription(clip_id=two_files["cover"], description="夜裡的店門口", tags=["夜景"])],
                  written_by="test", reviewed=True)
    problems = validate_plan(two_files["plan_id"])["problems"]
    assert any("what the footage's clips are known to show" in problem and "changed since" in problem
               for problem in problems), problems
    # Proposing again gives a new hash, and the same pick under it holds.
    two_files["offer"] = propose_broll(two_files["plan_id"], over_clip_ids=[two_files["over"]])
    assert pick(two_files)["problems"] == []


def test_a_clip_that_cannot_be_covered_is_refused_by_name(two_files: dict) -> None:
    offered_now = propose_broll(two_files["plan_id"], over_clip_ids=[two_files["cover"]])
    assert offered_now["refused"] == {two_files["cover"]: "the cut does not use it"}


def test_an_offer_is_pinned_to_what_it_was_made_from() -> None:
    timeline = SemanticTimeline(id="tl_x", asset_ids=[], input_hash="h", derivation_version=1)
    one = offering.stretch_inputs(stretch("機構"), timeline, "pool")
    other = offering.stretch_inputs(stretch("成品"), timeline, "pool")
    assert one != other and one["said"] != other["said"]
    first = offering.candidate_hash({"tl_x:u0000": {"inputs": one, "candidates": ["a", "b"]}})
    assert first == offering.candidate_hash({"tl_x:u0000": {"inputs": one, "candidates": ["a", "b"]}})
    assert first != offering.candidate_hash({"tl_x:u0000": {"inputs": one, "candidates": ["a"]}})


def test_a_described_shot_under_a_silent_stretch_is_not_called_undescribed() -> None:
    assets = {TALK: filed(TALK, "1-講解.mp4"), "a": filed("a", "IMG_0001.MOV")}
    described = shot("u0001", "a").model_copy(update={"tags": [Tag(value="機構特寫", source=TagSource.MODEL)]})
    pool = offering.eligible([described], assets)
    # Nothing said under it, so nothing is read by meaning, as `propose_broll` leaves it.
    candidates, _, _ = offering.offer_candidates(stretch(""), plan_for([]), [], pool, assets, {}, NOON)
    assert "nothing is said under this stretch" in candidates[0]["why"]
    assert "nothing says what is in it" not in candidates[0]["why"]
