"""Tests for the model store: where models go, and what happens when one arrives wrong.

The point of the store is that a model is project data rather than a
machine-wide installation, so what is checked here is mostly where files land
and what is refused. The download machinery is exercised over `file://` URLs,
which `urllib` serves without a network and which therefore let the failure
cases — a wrong digest, a missing member, a URL that is not there — be tested
rather than described.
"""

import hashlib
import os
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from app.engine import models

def as_url(path: Path) -> str:
    """Turn a local file into a URL the downloader will read.

    Args:
        path: File to serve.

    Returns:
        Its `file://` URL.
    """
    return path.as_uri()

def local_model(path: Path, content: bytes = b"a model, more or less", **overrides) -> models.Model:
    """Write a file and describe it as a model to be fetched from disk.

    Args:
        path: File to write.
        content: What to put in it.
        **overrides: Fields to change on the resulting model.

    Returns:
        The model, pinned to the digest of what was written unless overridden.
    """
    path.write_bytes(content)
    model = models.Model(
        name="test",
        path="test/model.bin",
        url=as_url(path),
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )
    return replace(model, **overrides)

@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the model store at an empty directory for one test.

    Returns:
        The directory it was pointed at.
    """
    folder = tmp_path / "models"
    monkeypatch.setenv("CLIP_MCP_MODELS", str(folder))
    return folder

# --- where models go ---------------------------------------------------------------------

def test_models_live_in_the_workspace_so_deleting_it_deletes_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model is project data. Nothing is written to a cache under the user's home."""
    monkeypatch.delenv("CLIP_MCP_MODELS", raising=False)
    monkeypatch.setenv("CLIP_MCP_WORKSPACE", os.path.join("somewhere", "else"))
    assert models.models_dir() == os.path.abspath(os.path.join("somewhere", "else", "models"))
    # The speech model too, which is the big one and the one that would otherwise
    # end up in a shared cache of its own.
    assert models.whisper_dir().startswith(models.models_dir())
    assert models.model_path(models.FACE_DETECTION).startswith(models.models_dir())

def test_the_model_directory_can_be_moved_off_the_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sharing one copy between workspaces is the trade the override exists for."""
    monkeypatch.setenv("CLIP_MCP_WORKSPACE", "ignored")
    monkeypatch.setenv("CLIP_MCP_MODELS", os.path.join("elsewhere", "weights"))
    assert models.models_dir() == os.path.abspath(os.path.join("elsewhere", "weights"))

# --- fetching ------------------------------------------------------------------------------

def test_a_model_is_fetched_once_and_then_left_alone(store: Path, tmp_path: Path) -> None:
    """Downloading tens of megabytes again on every analysis is the thing to avoid."""
    source = tmp_path / "source.bin"
    model = local_model(source)
    assert not models.is_downloaded(model)

    path = models.ensure(model)
    assert Path(path) == store / "test" / "model.bin"
    assert models.is_downloaded(model)

    # With the source gone, a second call can only succeed by not fetching anything.
    source.unlink()
    assert models.ensure(model) == path

def test_a_download_that_is_not_what_it_should_be_is_refused(store: Path, tmp_path: Path) -> None:
    """A changed release or a truncated download must not become a model that quietly differs."""
    model = local_model(tmp_path / "source.bin", sha256="0" * 64)
    with pytest.raises(RuntimeError, match="not the expected file"):
        models.ensure(model)
    assert not models.is_downloaded(model)
    # Nothing is left behind for a later run to mistake for a finished model.
    assert list((store / "test").glob("*")) == []

def test_a_url_that_is_not_there_says_which_model_it_was(store: Path, tmp_path: Path) -> None:
    """The message has to name the model, because a job failing on one is the reason."""
    model = local_model(tmp_path / "source.bin", url=as_url(tmp_path / "absent.bin"))
    with pytest.raises(RuntimeError, match="could not download the test model"):
        models.ensure(model)
    assert not models.is_downloaded(model)

def test_progress_is_reported_while_a_model_downloads(store: Path, tmp_path: Path) -> None:
    """A job that looks stuck for a minute is a job somebody cancels."""
    model = local_model(tmp_path / "source.bin", content=b"x" * (3 * models.DOWNLOAD_CHUNK_BYTES))
    seen: list = []
    models.ensure(model, seen.append)
    assert seen and seen[-1] == pytest.approx(1.0)
    assert seen == sorted(seen)

# --- archives --------------------------------------------------------------------------------

def test_a_model_inside_an_archive_is_taken_out_of_it(store: Path, tmp_path: Path) -> None:
    """One of the releases ships a tarball, and what is wanted is one file from it."""
    inner = tmp_path / "model.onnx"
    inner.write_bytes(b"the weights")
    archive = tmp_path / "release.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        tar.add(inner, arcname="release/model.onnx")
        tar.add(inner, arcname="release/README.md")

    model = local_model(tmp_path / "unused.bin", url=as_url(archive), member="release/model.onnx")
    model = replace(model, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    assert Path(models.ensure(model)).read_bytes() == b"the weights"

def test_an_archive_without_the_file_in_it_is_refused(store: Path, tmp_path: Path) -> None:
    """Better to fail naming the missing member than to leave an empty model behind."""
    archive = tmp_path / "release.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        tar.add(tmp_path, arcname="release")

    model = local_model(tmp_path / "unused.bin", url=as_url(archive), member="release/missing.onnx")
    model = replace(model, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="does not contain release/missing.onnx"):
        models.ensure(model)
    assert not models.is_downloaded(model)

# --- the models this server actually uses ------------------------------------------------------

def test_every_model_is_pinned_to_a_digest_and_a_place() -> None:
    """An unpinned model is one that can change under a project without anybody noticing."""
    declared = [models.SPEAKER_SEGMENTATION, models.SPEAKER_EMBEDDING, models.FACE_DETECTION]
    for model in declared:
        assert len(model.sha256) == 64, model.name
        assert model.url.startswith("https://"), model.name
        assert model.size > 0, model.name
    assert len({model.path for model in declared}) == len(declared)
