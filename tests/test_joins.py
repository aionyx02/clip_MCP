"""Finding every join of a cut, and laying the words said across it onto the cut's time."""

from pathlib import Path

from app import server
from app.engine.delivery import check_delivery
from app.engine.joins import joins_in, words_heard
from app.models.media import MediaAnalysis, Transcript, TranscriptSegment, TranscriptWord
from helpers import build_project, edit, insert, video_track


def said(asset_id: str, *words: tuple) -> None:
    """Store a transcript of one sentence per `(start, end, text)`."""
    server.repo.save_analysis(MediaAnalysis(asset_id=asset_id, duration=10.0, transcript=Transcript(
        language="zh", model="test", segments=[
            TranscriptSegment(start=start, end=end, text=text, words=[TranscriptWord(start=start, end=end, text=text)])
            for start, end, text in words
        ],
    )))


def test_every_kind_of_join_is_listed_in_the_order_it_plays(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([
        video_track(), insert("a", asset, 0, 4), insert("b", asset, 5, 9),
        video_track("over"), {**insert("r", asset, 2, 3, track_id="over"), "action": "add_clip", "timeline_in": 1.0},
    ])
    edit(project, [{"action": "set_markers", "markers": [
        {"id": "m1", "name": "開場", "timeline_in": 0}, {"id": "m2", "name": "結果", "timeline_in": 4},
    ]}])
    joins = joins_in(server.repo.get_project(project), {asset: "wide.mp4"})
    assert [(join.at, join.kind) for join in joins] == [(1.0, "b-roll in"), (2.0, "b-roll out"), (4.0, "cut")]
    assert joins[0].before.startswith("clip a") and joins[0].after.startswith("clip r")
    assert joins[2].before == "clip a: wide.mp4 0.00–4.00s" and joins[2].after == "clip b: wide.mp4 5.00–9.00s"
    assert joins[2].part == "結果" and joins[0].part is None


def test_a_join_says_where_the_sound_changes_over_when_it_leads_the_picture(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = build_project([video_track(), insert("a", asset, 0, 4), insert("b", asset, 5, 9, audio_lead=0.5)])
    (join,) = joins_in(server.repo.get_project(project), {})
    assert join.at == 4.0 and join.sound_at == 3.5


def test_the_words_across_a_join_are_laid_on_the_cut_and_a_word_cut_into_is_marked(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    said(asset, (2.0, 3.0, "前面"), (3.5, 4.5, "被切"), (6.0, 6.5, "後面"))
    project = build_project([video_track(), insert("a", asset, 0, 4), insert("b", asset, 5, 9)])
    heard = words_heard(server.repo.get_project(project), {asset: server.repo.get_analysis(asset)}, 2.5, 5.5)
    assert [(word.text, word.clip_id, word.cut) for word in heard] == [
        ("前面", "a", False), ("被切", "a", True), ("後面", "b", False),
    ]
    # 「被切」 is heard only up to where clip a ends; 「後面」 a second into clip b.
    assert heard[1].end == 4.0 and heard[2].start == 5.0


def test_what_check_render_finds_at_a_join_is_put_on_it(media: Path) -> None:
    asset = server.import_asset(str(media / "wide.mp4"))["id"]
    project = server.repo.get_project(build_project([
        video_track(), insert("a", asset, 0, 4), insert("b", asset, 5, 9), insert("c", asset, 9.5, 10),
    ]))
    talking = MediaAnalysis(asset_id=asset, duration=10.0, transcript=Transcript(language="zh", model="test", segments=[
        TranscriptSegment(start=3.5, end=4.5, text="被切", words=[TranscriptWord(start=3.5, end=4.5, text="被切")]),
    ]))
    findings = check_delivery(project, {asset: talking})
    near, far = joins_in(project, {})
    assert [finding.check for finding in server._found_at(near, findings)] == ["mid_speech"]
    assert server._found_at(far, findings) == []
