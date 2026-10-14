"""Title cards and words in a plan: compiled into the cut, and what was made by hand kept across a recompile."""

from typing import List

from pydantic import TypeAdapter

from app.engine.plan import compile_operations
from app.models.plan import Beat, PlanText, PlanTitleCard, Selection, SetBeatTitleOp, SetSelectionTextsOp, apply_amendment
from app.models.timeline import EditOperation, Project, apply_operation
from test_edit_plan import beats_plan, footage, speech_of

OPERATIONS = TypeAdapter(List[EditOperation])


def compiled(plan, by_id, children, assets, project: Project = None) -> Project:
    """Compile a plan onto a project the way `compile_plan` does: apply, then stamp where each clip came from."""
    operations, provenance = compile_operations(plan, by_id, children, assets, project)
    project = (project or Project(id="p", width=1920, height=1080)).model_copy(deep=True)
    for op in OPERATIONS.validate_python(operations):
        apply_operation(project, op, assets)
    for track in project.tracks:
        for clip in track.clips:
            origin = provenance.get(clip.id)
            if origin is not None:
                clip.from_plan_id, clip.from_clip_ids, clip.pinned = (
                    origin["from_plan_id"], list(origin["from_clip_ids"]), origin["pinned"])
    return project


def by_hand(project: Project, assets, *operations: dict) -> None:
    for op in OPERATIONS.validate_python(list(operations)):
        apply_operation(project, op, assets)


def two_parts(by_id, clips, **first):
    speech = speech_of(clips)
    return beats_plan(
        [Selection(clip_id=speech[0].id, beat_id="open"), Selection(clip_id=speech[2].id, beat_id="body")],
        [Beat(id="open", name="開場", **first), Beat(id="body", name="主體")],
    )


def test_a_part_opens_on_its_card_and_everything_after_makes_room() -> None:
    by_id, children, assets, clips = footage()
    plain = compiled(two_parts(by_id, clips), by_id, children, assets)
    plan = two_parts(by_id, clips, title_card=PlanTitleCard(title="開場", seconds=2))
    project = compiled(plan, by_id, children, assets)
    sequence = sorted(project.base_video_track.clips, key=lambda clip: clip.timeline_in)
    card = sequence[0]
    assert card.card.title == "開場" and card.card.beat_id == "open" and card.volume == 0
    # Its frame is the part's first shot, and everything after it is two seconds later.
    assert card.asset_id == sequence[1].asset_id and float(card.card.at) == float(sequence[1].source_range.start)
    assert float(project.duration) == float(plain.duration) + 2
    assert [float(marker.timeline_in) for marker in project.markers] == [
        0.0, float(next(m for m in plain.markers if m.id == "body").timeline_in) + 2]


def test_words_in_the_plan_land_on_the_footage_they_belong_to() -> None:
    by_id, children, assets, clips = footage()
    plan = two_parts(by_id, clips)
    target = plan.selections[1].clip_id
    apply_amendment(plan, SetSelectionTextsOp(clip_id=target, texts=[PlanText(text="王小明", second="店長")]))
    project = compiled(plan, by_id, children, assets)
    [carrier] = [clip for clip in project.base_video_track.clips if clip.texts]
    assert target in carrier.from_clip_ids
    [text] = carrier.texts
    assert (text.text, text.second, text.by_plan) == ("王小明", "店長", True)


def test_a_recompile_keeps_what_was_made_by_hand() -> None:
    by_id, children, assets, clips = footage()
    plan = two_parts(by_id, clips, title_card=PlanTitleCard(title="開場"))
    project = compiled(plan, by_id, children, assets)
    shots = sorted((clip for clip in project.base_video_track.clips if clip.card is None), key=lambda c: c.timeline_in)
    plan_card = next(clip for clip in project.base_video_track.clips if clip.card)
    by_hand(project, assets,
            {"action": "add_title_card", "track_id": "main", "clip_id": "mine", "title": "第二段",
             "before_clip_id": shots[1].id},
            {"action": "add_text", "track_id": "main", "clip_id": shots[1].id, "text_id": "who", "text": "老闆"},
            {"action": "set_title_card", "track_id": "main", "clip_id": plan_card.id, "title": "改過的開場"})
    # The plan changes, and is compiled again over the work.
    apply_amendment(plan, SetBeatTitleOp(beat_id="open", title_card=PlanTitleCard(title="計畫的新標題")))
    again = compiled(plan, by_id, children, assets, project)
    cards = sorted((clip for clip in again.base_video_track.clips if clip.card), key=lambda clip: clip.timeline_in)
    # The plan's card was changed by hand, so the hand's version stays; the hand-made one comes back
    # in front of the shot it shows.
    assert [clip.card.title for clip in cards] == ["改過的開場", "第二段"]
    after_mine = min((clip for clip in again.base_video_track.clips if clip.timeline_in >= cards[1].timeline_out),
                     key=lambda clip: clip.timeline_in)
    assert after_mine.asset_id == cards[1].asset_id and after_mine.texts[0].text == "老闆"


def test_a_card_nobody_touched_follows_the_plan() -> None:
    by_id, children, assets, clips = footage()
    plan = two_parts(by_id, clips, title_card=PlanTitleCard(title="舊"))
    project = compiled(plan, by_id, children, assets)
    apply_amendment(plan, SetBeatTitleOp(beat_id="open", title_card=None))
    again = compiled(plan, by_id, children, assets, project)
    assert not any(clip.card for clip in again.base_video_track.clips)


def plan_with_words(by_id, clips):
    plan = two_parts(by_id, clips)
    target = plan.selections[1].clip_id
    apply_amendment(plan, SetSelectionTextsOp(clip_id=target, texts=[PlanText(text="王小明", second="店長")]))
    return plan


def test_words_of_the_plan_taken_off_by_hand_stay_off() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_with_words(by_id, clips)
    project = compiled(plan, by_id, children, assets)
    [carrier] = [clip for clip in project.base_video_track.clips if clip.texts]
    by_hand(project, assets, {"action": "remove_text", "track_id": "main", "clip_id": carrier.id,
                              "text_id": carrier.texts[0].id})
    again = compiled(plan, by_id, children, assets, project)
    assert not any(clip.shown_texts for clip in again.base_video_track.clips)


def test_words_of_the_plan_changed_by_hand_are_not_doubled() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_with_words(by_id, clips)
    project = compiled(plan, by_id, children, assets)
    [carrier] = [clip for clip in project.base_video_track.clips if clip.texts]
    by_hand(project, assets, {"action": "set_text", "track_id": "main", "clip_id": carrier.id,
                              "text_id": carrier.texts[0].id, "text": "王大明"})
    again = compiled(plan, by_id, children, assets, project)
    words = [text.text for clip in again.base_video_track.clips for text in clip.shown_texts]
    assert words == ["王大明"]


def test_a_card_of_another_plan_is_not_carried() -> None:
    by_id, children, assets, clips = footage()
    project = compiled(two_parts(by_id, clips, title_card=PlanTitleCard(title="別的")), by_id, children, assets)
    for clip in project.base_video_track.clips:
        if clip.card:
            clip.from_plan_id = "another"
    again = compiled(two_parts(by_id, clips), by_id, children, assets, project)
    assert not any(clip.card for clip in again.base_video_track.clips)


def test_new_words_in_the_plan_come_in_beside_ones_changed_by_hand() -> None:
    by_id, children, assets, clips = footage()
    plan = plan_with_words(by_id, clips)
    project = compiled(plan, by_id, children, assets)
    [carrier] = [clip for clip in project.base_video_track.clips if clip.texts]
    by_hand(project, assets, {"action": "remove_text", "track_id": "main", "clip_id": carrier.id,
                              "text_id": carrier.texts[0].id})
    target = plan.selections[1].clip_id
    apply_amendment(plan, SetSelectionTextsOp(clip_id=target, texts=[PlanText(text="李大華")]))
    again = compiled(plan, by_id, children, assets, project)
    assert [text.text for clip in again.base_video_track.clips for text in clip.shown_texts] == ["李大華"]


def test_words_of_the_plan_are_named_as_the_ai_is_shown_them() -> None:
    from app.models.plan import short_clip_ids

    by_id, children, assets, clips = footage()
    plan = plan_with_words(by_id, clips)
    project = compiled(plan, by_id, children, assets)
    [carrier] = [clip for clip in project.base_video_track.clips if clip.texts]
    # Everything said to the AI has the timeline taken off clip IDs; a text's own ID must survive that.
    [text] = carrier.texts
    assert short_clip_ids(text.id) == text.id
    by_hand(project, assets, {"action": "set_text", "track_id": "main", "clip_id": carrier.id,
                              "text_id": short_clip_ids(text.id), "text": "王大明"})
    assert [text.text for text in carrier.texts] == ["王大明"]


def test_words_changed_by_hand_under_the_old_name_are_not_doubled() -> None:
    from app.models.plan import short_clip_ids

    by_id, children, assets, clips = footage()
    plan = plan_with_words(by_id, clips)
    project = compiled(plan, by_id, children, assets)
    [carrier] = [clip for clip in project.base_video_track.clips if clip.texts]
    target = plan.selections[1].clip_id
    # Named with the whole clip ID, the way they were before.
    carrier.texts[0].id = carrier.texts[0].id.replace("plan-" + target.split(":")[1], "plan-" + target)
    by_hand(project, assets, {"action": "set_text", "track_id": "main", "clip_id": carrier.id,
                              "text_id": carrier.texts[0].id, "text": "王大明"})
    again = compiled(plan, by_id, children, assets, project)
    [kept] = [text for clip in again.base_video_track.clips for text in clip.shown_texts]
    # Kept, and named from now on as the AI is shown it.
    assert kept.text == "王大明" and short_clip_ids(kept.id) == kept.id
