"""Tests for what the check before a render says about a picture in a corner or a half."""

from decimal import Decimal

from app.engine.builder import Talking
from app.engine.delivery import check_delivery
from app.engine.reframe import frame_project
from app.models.media import Asset, FaceMeasurement, MediaAnalysis
from app.models.timeline.parts import (
    Clip, ClipLayout, LogoLayer, PlacedCue, Project, TimeRange, TitleCard, Track, TrackType,
)

ASSETS = {
    "talk": Asset(id="talk", path="talk.mp4", duration=Decimal(60), has_video=True, has_audio=True,
                  width=1920, height=1080),
    "close": Asset(id="close", path="close.mp4", duration=Decimal(60), has_video=True, has_audio=True,
                   width=1920, height=1080),
    "logo": Asset(id="logo", path="logo.png", duration=Decimal(1), has_video=True, has_audio=False,
                  width=400, height=200),
}

def talking_head(face_x: float, face_y: float) -> MediaAnalysis:
    """A minute of somebody talking with their face at one place in the frame, a fiftieth of it."""
    return MediaAnalysis(asset_id="talk", duration=60.0, faces=[
        FaceMeasurement(start=float(second), end=float(second + 1), faces=1, face_share=0.02,
                        face_x=face_x, face_y=face_y)
        for second in range(60)
    ])

def cut(layout: ClipLayout, width: int = 1920, height: int = 1080, volume: float = 0.0, ducked: bool = False,
        **fields) -> Project:
    """Ten seconds of the talk with the close-up laid over it from 2 s to 6 s."""
    talk = Clip(id="a", asset_id="talk", source_range=TimeRange(start=Decimal(0), end=Decimal(10)),
                timeline_in=Decimal(0))
    over = Clip(id="pip", asset_id="close", source_range=TimeRange(start=Decimal(0), end=Decimal(4)),
                timeline_in=Decimal(2), layout=layout, volume=volume)
    return Project(id="p", name="測試", width=width, height=height, tracks=[
        Track(id="main", track_type=TrackType.VIDEO, clips=[talk]),
        Track(id="broll", track_type=TrackType.VIDEO, clips=[over], duck_under_speech=ducked),
    ], **fields)

def insets_found(project: Project, analyses: dict, **options) -> list:
    """The `inset` findings for a cut, with its sequence framed the way the render frames it."""
    faces = {asset_id: analysis.faces for asset_id, analysis in analyses.items()}
    framing = frame_project(project, ASSETS, faces)
    return [finding for finding in check_delivery(project, analyses, framing=framing, assets=ASSETS, **options)
            if finding.check == "inset"]

CORNER = ClipLayout(place="top_right", size="medium")

def test_a_corner_over_the_speaker_s_face_is_said_and_left_where_it_is() -> None:
    found = insets_found(cut(CORNER), {"talk": talking_head(0.85, 0.2)})
    assert len(found) == 1
    assert "top-right corner (clip pip) covers the speaker's face" in found[0].message
    assert "It is left where it is" in found[0].message
    assert found[0].spans == ((2.0, 6.0),)

def test_a_corner_clear_of_the_face_says_nothing() -> None:
    assert insets_found(cut(ClipLayout(place="bottom_left", size="medium")), {"talk": talking_head(0.85, 0.2)}) == []

def test_a_corner_clear_of_the_face_in_landscape_can_cover_it_once_cropped_for_a_phone() -> None:
    # In the middle of a wide shot the face is well left of the top-right corner. Cropped to a
    # portrait frame on the face, it is in the middle of a narrow one, and the corner reaches it.
    head = {"talk": talking_head(0.5, 0.2)}
    assert insets_found(cut(CORNER), head) == []
    found = insets_found(cut(CORNER, 1080, 1920), head)
    assert len(found) == 1 and "1080x1920 frame" in found[0].message

def test_a_split_screen_never_covers_the_face_it_moved_aside() -> None:
    assert insets_found(cut(ClipLayout(place="right")), {"talk": talking_head(0.6, 0.4)}) == []

def test_a_caption_that_runs_past_two_lines_is_said_where_it_meets_a_bottom_corner() -> None:
    long = PlacedCue(cue_id="c1", start=Decimal(3), end=Decimal(5),
                     text="這個抽杯架的機構其實很簡單只要往下一拉杯子就會自己掉下來真的很方便",
                     secondary="The mechanism is simple: pull it down and a cup drops out")
    short = PlacedCue(cue_id="c2", start=Decimal(7), end=Decimal(8), text="很方便")
    found = insets_found(cut(ClipLayout(place="bottom_right", size="large")), {}, captions=[long, short])
    assert len(found) == 1 and "captions are drawn over" in found[0].message
    assert found[0].spans == ((3.0, 5.0),)
    # Two lines and a top corner meet nothing.
    assert insets_found(cut(ClipLayout(place="top_right", size="large")), {}, captions=[long, short]) == []

def test_a_corner_over_the_logo_is_said() -> None:
    logo = LogoLayer(asset_id="logo", corner="top_right")
    found = insets_found(cut(CORNER, logo=logo), {})
    assert len(found) == 1 and "over the logo in the top-right corner" in found[0].message
    assert insets_found(cut(ClipLayout(place="top_left", size="small"), logo=logo), {}) == []

def test_a_picture_heard_at_full_level_over_the_talking_is_said() -> None:
    talking = Talking(spans=((1.0, 9.0),))
    found = insets_found(cut(CORNER, volume=1.0), {}, talking=talking)
    assert len(found) == 1 and "plays its own sound at full level" in found[0].message
    # Ducked, or turned down, it is not doubling anybody.
    assert insets_found(cut(CORNER, volume=1.0, ducked=True), {}, talking=talking) == []
    assert insets_found(cut(CORNER), {}, talking=talking) == []

def test_a_corner_over_a_title_card_is_a_covered_card() -> None:
    card = Clip(id="t", asset_id="talk", source_range=TimeRange(start=Decimal(0), end=Decimal(3)),
                timeline_in=Decimal(0), volume=0.0, card=TitleCard(title="開場"))
    talk = Clip(id="a", asset_id="talk", source_range=TimeRange(start=Decimal(0), end=Decimal(10)),
                timeline_in=Decimal(3))
    over = Clip(id="pip", asset_id="close", source_range=TimeRange(start=Decimal(0), end=Decimal(2)),
                timeline_in=Decimal(2), layout=CORNER)
    project = Project(id="p", name="測試", tracks=[
        Track(id="main", track_type=TrackType.VIDEO, clips=[card, talk]),
        Track(id="broll", track_type=TrackType.VIDEO, clips=[over]),
    ])
    found = [finding for finding in check_delivery(project, {}) if finding.check == "covered_card"]
    assert len(found) == 1 and "top-right corner" in found[0].message
