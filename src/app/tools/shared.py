"""Helpers more than one kind of tool needs: finding assets, filling in what old assets lack, shaping results.

Kept apart from any one tool module so that none of them has to import another
for these, and the modules can be read and moved one at a time.
"""

import os
from decimal import Decimal
from typing import Dict, List, Mapping, Optional, Sequence

from app import runtime
from app.engine.probe import picture_size, probe_file, recorded_at
from app.models.media import Asset, Measured
from app.models.timeline import Project
from app.storage.history import current_batch

MAX_STORYBOARD_TILES = 36
# Reading a whole timeline at once: an hour of talk is a few hundred utterances.
MAX_TIMELINE_CLIPS = 5000

def asset_id_for(name: str) -> str:
    """Find the asset a file name means, for a tool given a name where it wants an ID.

    Only an exact name: guessing which of `0828.mp4` and `0828(1).mp4` was
    meant is worse than one more look at the list.

    Args:
        name: An asset ID, or a file's name as `list_assets` shows it.

    Returns:
        The asset's ID, or `name` as it was when it is an ID or no file has
        that name, so the tool says what it says about an asset it cannot find.

    Raises:
        ValueError: If more than one file has that name.
    """
    if not isinstance(name, str) or runtime.repo.get_assets([name]):
        return name
    named = [asset for asset in runtime.repo.list_assets() if os.path.basename(asset.path) == name]
    if len(named) > 1:
        raise ValueError(
            f"{len(named)} files are called {name}: "
            + "; ".join(f"{asset.id} ({asset.path})" for asset in named)
            + ". Give the ID of the one meant"
        )
    return named[0].id if named else name

def referenced_assets(project: Project) -> dict[str, Asset]:
    """Fetch the assets referenced by a project's clips.

    Args:
        project: Project whose clips are inspected.

    Returns:
        The registered assets used by the project, keyed by asset ID. IDs that
        are not registered are omitted.
    """
    return runtime.repo.get_assets(clip.asset_id for track in project.tracks for clip in track.clips)

def get_asset(asset_id: str) -> Asset:
    """Fetch a registered asset.

    Args:
        asset_id: ID of the asset.

    Returns:
        The asset.

    Raises:
        ValueError: If no asset with `asset_id` exists.
    """
    asset = runtime.repo.get_assets([asset_id]).get(asset_id)
    if asset is None:
        raise ValueError(f"asset {asset_id} not found")
    return asset

def covering(records: Sequence[Measured], start: float, end: float) -> list:
    """Select the records that reach into a time range.

    Args:
        records: Records to filter; anything with `start` and `end`.
        start: Range start in seconds.
        end: Range end in seconds.

    Returns:
        The overlapping records themselves, for a caller that has to read them.
    """
    return [record for record in records if record.end > start and record.start < end]

def overlapping(spans: Sequence[Measured], start: float, end: float) -> List[dict]:
    """Select the spans that overlap a time range.

    Args:
        spans: Spans to filter; anything with `start` and `end` attributes.
        start: Range start in seconds.
        end: Range end in seconds.

    Returns:
        The overlapping spans, serialized as dictionaries.
    """
    return [span.model_dump() for span in covering(spans, start, end)]

def plain(value):
    """Turn the Decimals a model keeps its seconds in into plain numbers, all the way down.

    Times are kept as Decimal so the timeline adds up to the millisecond, but a
    client reading them gets strings — `"12.5"` — and compares or adds them
    wrongly. What leaves the server is numbers; what is stored stays exact.

    Args:
        value: What a tool is about to return.

    Returns:
        The same, with every Decimal a float.
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value

def dated(assets: Mapping[str, Asset]) -> Dict[str, Asset]:
    """Make sure every asset knows when it was recorded, where anything says.

    Assets imported before the time was kept do not, so it is read off the
    file the first time it is asked for and saved, the same way `sized` does
    the picture size.

    Args:
        assets: The assets, keyed by ID.

    Returns:
        The same assets, each with `recorded_at` filled in where it can be.
    """
    found = {}
    for asset_id, asset in assets.items():
        if asset.recorded_at is None and os.path.isfile(asset.path):
            try:
                moment = recorded_at(probe_file(asset.path), asset.path)
            except (OSError, RuntimeError):
                moment = None
            if moment is not None:
                asset = asset.model_copy(update={"recorded_at": moment})
                runtime.repo.save_asset(asset)
        found[asset_id] = asset
    return found

def sized(assets: Mapping[str, Asset]) -> Dict[str, Asset]:
    """Make sure every asset with a picture knows how big it is.

    Assets imported before the size was kept do not. Reframing needs it, so it
    is read off the file the first time it is asked for and saved, rather than
    making anybody import everything again.

    Args:
        assets: The assets, keyed by ID.

    Returns:
        The same assets, sized where the file could still be read. One whose
        file has gone is returned as it was; the render reports the missing
        file in its own words.
    """
    found: Dict[str, Asset] = {}
    for asset_id, asset in assets.items():
        if asset.has_video and asset.width is None:
            try:
                size = picture_size(probe_file(asset.path))
            except (FileNotFoundError, RuntimeError):
                size = None
            if size is not None:
                asset = asset.model_copy(update={"width": size[0], "height": size[1]})
                runtime.repo.save_asset(asset)
        found[asset_id] = asset
    return found

def say_in_history(words: str) -> None:
    """Give the version being made these words, unless the user's own were passed on."""
    batch = current_batch()
    if batch is not None and not batch.note and words:
        batch.note = words

def glossed(text: str, glossary: Optional[Mapping[str, str]] = None) -> str:
    """Text the transcriber wrote, with every word in the glossary corrected as the captions are.

    Args:
        text: The text.
        glossary: Heard words mapped to meant ones; the workspace's when not given.

    Returns:
        The text corrected.
    """
    for heard, meant in (runtime.repo.glossary() if glossary is None else glossary).items():
        text = text.replace(heard, meant)
    return text
