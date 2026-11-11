"""Sound effects: a library of their own, moments offered by rule, picks pinned to them, and heard in the render."""

import shutil
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import List

import pytest
from pydantic import TypeAdapter

from app import runtime
from app.engine import fetch, interchange
from app.engine.library import SFX, SFX_LONGEST_SECONDS, classify
from app.engine.plan.effects import SFX_UNDER_TALK_DB, effect_moments, effect_volume, moments_hash
from app.models.media import Asset
from app.models.plan import PlanAmendment
from app.models.timeline.parts import Clip, Project, TextLayer, TimeRange, TitleCard, Track, TrackType
from app.tools import catalog, delivery, rendering, soundtrack, upkeep
from app.tools.catalog import import_asset
from app.tools.editing import create_project, get_project
from app.tools.planning import amend_plan, compile_plan
from helpers import level, render


@pytest.fixture(scope="module")
def effects(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Make two short sounds: a bright pop and a low hit, far from the footage's 440 Hz tone.

    Returns:
        A folder holding `pop.wav` (0.3 s at 3 kHz) and `hit.wav` (0.5 s at 120 Hz).
    """
    folder = tmp_path_factory.mktemp("effects")
    for name, frequency, seconds in (("pop.wav", 3000, 0.3), ("hit.wav", 120, 0.5)):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"sine=frequency={frequency}:duration={seconds}", "-c:a", "pcm_s16le", str(folder / name)],
                       check=True, capture_output=True)
    return folder


def steps(*operations: dict) -> list:
    """Validate plan amendments the way a client's would be."""
    return TypeAdapter(List[PlanAmendment]).validate_python(list(operations))


def compiled(planned: dict) -> str:
    """Compile the planned cut on a new project, and return the project's ID."""
    project = create_project(name="音效", width=640, height=360)["id"]
    compile_plan(project_id=project, expected_version=1, plan_id=planned["plan_id"])
    return project


def pick(planned: dict, project: str, asset_id: str, kind: str = "cut", **fields) -> dict:
    """Write one effect into the plan at the first moment of a kind, and return what was proposed."""
    offered = soundtrack.propose_sound_effects(project)
    moment = next(moment for moment in offered["moments"] if moment["kind"] == kind)
    version = runtime.repo.get_plan(planned["plan_id"]).version
    amend_plan(planned["plan_id"], version, steps({"action": "set_sound_effects", "sound_effects": {
        "candidate_hash": fields.pop("candidate_hash", offered["candidate_hash"]),
        "picks": [{"candidate": fields.pop("candidate", moment["id"]), "asset_id": asset_id, **fields}],
    }}))
    return {**offered, "moment": moment}


# --- the library ------------------------------------------------------------------------------

def test_a_short_sound_with_no_picture_is_a_sound_effect_whose_kind_is_only_what_was_said(effects: Path) -> None:
    short = Asset(id="s", path="pop.wav", duration=SFX_LONGEST_SECONDS - 1, has_video=False, has_audio=True)
    assert classify(short) == SFX
    shot = short.model_copy(update={"has_video": True})
    assert classify(shot) == "footage"

    added = import_asset(str(effects / "pop.wav"), sound_kinds=["Pop", " pop", "click"])
    assert added["library"] == SFX and added["sound_kinds"] == ["pop", "click"]
    listed = catalog.list_assets(kind="sfx", limit=200)["assets"]
    assert any(item["id"] == added["id"] and item["sound_kinds"] == ["pop", "click"] for item in listed)
    assert added["id"] not in {item["id"] for item in catalog.list_assets(kind="music", limit=200)["assets"]}
    # Imported again without kinds, what was said is kept; nothing listens for it.
    assert import_asset(str(effects / "pop.wav"))["sound_kinds"] == ["pop", "click"]
    assert catalog.edit_asset(added["id"], sound_kinds=["whoosh"])["sound_kinds"] == ["whoosh"]


def test_a_file_moves_to_the_effects_with_its_kinds_and_a_picture_never_does(media: Path, effects: Path) -> None:
    song = import_asset(str(media / "song.mp3"))["id"]
    assert catalog.set_asset_library([song], "sfx", sound_kinds=["riser"])["moved"] == ["song.mp3"]
    assert runtime.repo.get_assets([song])[song].sound_kinds == ["riser"]
    catalog.set_asset_library([song], "music")
    shot = import_asset(str(media / "wide.mp4"))["id"]
    with pytest.raises(ValueError, match="has a picture"):
        catalog.set_asset_library([shot], "sfx")
    with pytest.raises(ValueError, match="only with library sfx"):
        catalog.set_asset_library([song], "music", sound_kinds=["pop"])
    with pytest.raises(ValueError, match="sfx library first"):
        catalog.edit_asset(shot, sound_kinds=["pop"])


def test_effects_come_only_from_the_sites_with_an_effects_licence(effects: Path, monkeypatch: pytest.MonkeyPatch,
                                                                   tmp_path: Path) -> None:
    def fake_download(url, folder, title):
        Path(folder).mkdir(parents=True, exist_ok=True)
        return shutil.copy(effects / "pop.wav", Path(folder) / f"{title}.wav")

    for module in (catalog, delivery, rendering, soundtrack, upkeep):
        monkeypatch.setattr(module, "output_dir", lambda: tmp_path)
    monkeypatch.setattr(fetch, "download", fake_download)
    with pytest.raises(ValueError, match="sound effects are only taken from Pixabay Music, Mixkit"):
        soundtrack.add_music_from_url("https://incompetech.com/a.mp3", "https://incompetech.com/a", "A",
                                      sound_effect=True)
    url, page = "https://assets.mixkit.co/sfx/whoosh.wav", "https://mixkit.co/free-sound-effects/whoosh/"
    shown = soundtrack.add_music_from_url(url, page, "Whoosh", sound_effect=True, sound_kinds=["whoosh"])
    assert shown["licence"] == "Mixkit Sound Effects Free License" and shown["credit"] is None
    taken = soundtrack.add_music_from_url(url, page, "Whoosh", sound_effect=True, sound_kinds=["whoosh"],
                                          confirm_plan=shown["plan"])
    asset = runtime.repo.get_assets([taken["id"]])[taken["id"]]
    assert asset.library == SFX and asset.sound_kinds == ["whoosh"]
    assert asset.source.licence == "Mixkit Sound Effects Free License"
    assert Path(taken["path"]).parent == tmp_path / "音效"


# --- the moments ------------------------------------------------------------------------------

def _clip(clip_id: str, start: float, end: float, at: float, made_of: List[str], **fields) -> Clip:
    return Clip(id=clip_id, asset_id="a" * 12, source_range=TimeRange(start=Decimal(str(start)), end=Decimal(str(end))),
                timeline_in=Decimal(str(at)), from_clip_ids=made_of, **fields)


def _cut(shift: float = 0.0, extra: bool = False) -> Project:
    card = _clip("p001", 0, 2, 0, [], card=TitleCard(title="開場", beat_id="open"))
    first = _clip("p002", 10 - shift, 14, 2, ["tl_0000aaaa:u0001"],
                  texts=[TextLayer(id="plan-u0001-ab", text="重點", start=Decimal("11"), end=Decimal("13"))])
    second = _clip("p003", 20, 24, 6 - shift, ["tl_0000aaaa:u0002"], speed=1.5,
                   transition_in={"kind": "dissolve", "seconds": Decimal("0.5")})
    shots = [card, first, second] + ([_clip("p004", 30, 32, 8.67 - shift, ["tl_0000aaaa:u0003"])] if extra else [])
    cover = _clip("b001", 0, 1.5, 3, ["tl_0000aaaa:u0009", "tl_0000aaaa:u0001"], volume=0)
    return Project(id="p", tracks=[Track(id="main", track_type=TrackType.VIDEO, clips=shots),
                                   Track(id="broll", track_type=TrackType.VIDEO, clips=[cover])])


def test_every_cut_card_words_and_cover_is_offered_and_named_by_what_makes_it() -> None:
    moments = {moment.id: moment for moment in effect_moments(_cut())}
    assert set(moments) == {"cut:u0001", "cut:u0002", "title_card:card-open", "words:plan-u0001-ab",
                            "broll:u0009+u0001"}
    assert moments["cut:u0001"].marks == ("card",)
    assert moments["cut:u0002"].marks == ("transition", "speed")
    assert moments["words:plan-u0001-ab"].at == pytest.approx(3.0)
    # Retrimmed earlier in the cut, every moment moves and none changes: the picks still stand.
    assert moments_hash(effect_moments(_cut(shift=0.5))) == moments_hash(effect_moments(_cut()))
    # A shot more is a moment more.
    assert moments_hash(effect_moments(_cut(extra=True))) != moments_hash(effect_moments(_cut()))


# --- picking and compiling --------------------------------------------------------------------

def test_an_effect_is_laid_at_the_cut_it_was_picked_for_and_never_ducks(planned: dict, effects: Path) -> None:
    pop = import_asset(str(effects / "pop.wav"), sound_kinds=["pop"])["id"]
    project = compiled(planned)
    offered = pick(planned, project, pop, volume=2.0)
    assert offered["moments"] and all(moment["kind"] == "cut" for moment in offered["moments"])
    assert any(effect["id"] == pop and effect["sound_kinds"] == ["pop"] for effect in offered["effects"])
    compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=planned["plan_id"])

    state = runtime.repo.get_project(project)
    sfx = next(track for track in state.tracks if track.id == "sfx")
    assert sfx.track_type == TrackType.AUDIO and not sfx.duck_under_speech and sfx.voice is False
    [laid] = sfx.clips
    assert float(laid.timeline_in) == pytest.approx(offered["moment"]["at"], abs=0.001)
    assert laid.volume == effect_volume(2.0) == pytest.approx(2.0 * 10 ** (-SFX_UNDER_TALK_DB / 20), abs=1e-4)
    assert laid.from_clip_ids == [offered["moment"]["id"], pop] and laid.from_plan_id == planned["plan_id"]
    again = soundtrack.propose_sound_effects(project)
    assert again["chosen"]["current"] is True and again["candidate_hash"] == offered["candidate_hash"]

    # A compile with no effects takes them away: the plan owns them.
    amend_plan(planned["plan_id"], runtime.repo.get_plan(planned["plan_id"]).version,
               steps({"action": "set_sound_effects", "sound_effects": None}))
    compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=planned["plan_id"])
    assert not any(track.clips for track in runtime.repo.get_project(project).tracks if track.id == "sfx")


@pytest.mark.parametrize("wrong, reason", [
    ({"candidate_hash": "000000000000"}, "moments have changed"),
    ({"candidate": "cut:u9999"}, "not one of the moments"),
    # The first cut comes about two seconds in, so a three-second run-up starts before the video.
    ({"lead": 3.0}, "before the video begins"),
])
def test_a_pick_that_does_not_hold_up_is_handed_back_and_nothing_is_compiled(
    planned: dict, effects: Path, wrong: dict, reason: str,
) -> None:
    pop = import_asset(str(effects / "pop.wav"))["id"]
    project = compiled(planned)
    pick(planned, project, pop, **wrong)
    before = get_project(project)["version"]
    with pytest.raises(ValueError, match=reason):
        compile_plan(project_id=project, expected_version=before, plan_id=planned["plan_id"])
    assert get_project(project)["version"] == before


def test_a_file_that_is_not_a_sound_effect_is_refused_rather_than_played(planned: dict, media: Path) -> None:
    song = import_asset(str(media / "song.mp3"))["id"]
    catalog.set_asset_library([song], "music")
    project = compiled(planned)
    pick(planned, project, song)
    with pytest.raises(ValueError, match="in the music library, not sound effects"):
        compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=planned["plan_id"])


# --- heard, shown and accounted for -----------------------------------------------------------

def test_the_effect_is_heard_at_its_cut_and_the_talking_is_not_ducked_by_it(
    planned: dict, effects: Path, tmp_path: Path,
) -> None:
    pop = import_asset(str(effects / "pop.wav"), sound_kinds=["pop"])["id"]
    project = compiled(planned)
    plain = runtime.repo.get_project(project)
    render(project, tmp_path / "plain.mp4", loudness_target=None, talking=rendering.talking_of(plain, {}))
    at = pick(planned, project, pop)["moment"]["at"]
    compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=planned["plan_id"])
    with_effect = runtime.repo.get_project(project)
    render(project, tmp_path / "pop.mp4", loudness_target=None, talking=rendering.talking_of(with_effect, {}))

    # The pop's 3 kHz is there for the 0.3 s after the cut and nowhere before it.
    assert level(tmp_path / "pop.mp4", at + 0.05, 0.2, freq=3000) > level(tmp_path / "pop.mp4", at - 1.0, 0.5,
                                                                           freq=3000) + 20
    assert level(tmp_path / "pop.mp4", at + 0.05, 0.2, freq=3000) > level(tmp_path / "plain.mp4", at + 0.05, 0.2,
                                                                           freq=3000) + 20
    # The footage's own 440 Hz is as loud with the effect over it as without: nothing ducks under an effect.
    for start in (at + 0.05, at - 1.0):
        assert level(tmp_path / "pop.mp4", start, 0.2, freq=440) == pytest.approx(
            level(tmp_path / "plain.mp4", start, 0.2, freq=440), abs=1.0)


def test_the_effect_is_shown_played_listed_and_accounted_for(planned: dict, effects: Path) -> None:
    pop = import_asset(str(effects / "pop.wav"), sound_kinds=["pop"])["id"]
    project = compiled(planned)
    pick(planned, project, pop)
    compile_plan(project_id=project, expected_version=get_project(project)["version"], plan_id=planned["plan_id"])
    state = runtime.repo.get_project(project)

    text = rendering.preview_project(project).content[0].text
    assert any(line.startswith("-- sound effect on track sfx: clip s001 pop.wav (pop) at") for line in text.splitlines())

    described = rendering.playback_of(state)
    played = next(track for track in described["tracks"] if track["id"] == "sfx")
    assert played["type"] == "audio" and not played["duck"] and played["clips"][0]["asset_id"] == pop

    checked = rendering.check_render(project)
    # The pop lands as the next sentence starts: said, never a finding that stops the render.
    assert checked["sound_effects_on_words"][0]["effect"] == "pop.wav"
    assert not any(finding["check"] == "music" for finding in checked["findings"])

    _, behind = interchange.write_edl(state, rendering.shared.referenced_assets(state))
    # An EDL is one picture track with its sound: the effects are among what it cannot carry.
    assert any(line.startswith("tracks other than the sequence") and "sfx" in line for line in behind)
    facts = soundtrack.export_notices(project)["facts"]
    assert any(fact.startswith("sound_effect | pop.wav") and "licence: nobody has said" in fact for fact in facts)
    assert soundtrack.music_credits(project)["unknown"] == ["pop.wav"]
