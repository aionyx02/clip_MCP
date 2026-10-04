"""Finding clips by what they are about, on this machine.

A search by words only finds the words: "the part where the product is
introduced" matches nothing unless somebody said "introduce". This turns each
clip's words and description into a vector with a small multilingual model,
and a request into one the same way, and ranks clips by how close they are.

It all runs here. The model is downloaded once into the workspace, like the
others, and runs on the CPU through onnxruntime, which the speech recogniser
already brings; nothing about the footage leaves the machine to be searched.
The AI asks a question and gets back a ranked handful of clips — never the
whole transcript, and never the video.

The model is multilingual-e5-small, quantized to 8 bits: on a set of
Traditional Chinese and English questions about footage it put the right clip
first more often than bge-small-zh, a Chinese-only model a fifth of its size,
and it reads English as well, at about a thousand clips a second on a laptop CPU.
"""

import hashlib
import threading
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.engine import models

# Pinned to a commit of the repository the ONNX export lives in: the digests are the
# safety net, and a branch could be regenerated under them.
_SOURCE = "https://huggingface.co/Xenova/multilingual-e5-small/resolve/761b726dd34fb83930e26aab4e9ac3899aa1fa78"
SEARCH_MODEL = models.Model(
    name="meaning search",
    path="search/multilingual-e5-small-q8.onnx",
    url=f"{_SOURCE}/onnx/model_quantized.onnx",
    sha256="f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193",
    size=113 * models.MEGABYTE,
)
SEARCH_TOKENIZER = models.Model(
    name="meaning search tokenizer",
    path="search/multilingual-e5-small-tokenizer.json",
    url=f"{_SOURCE}/tokenizer.json",
    sha256="0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    size=16 * models.MEGABYTE,
)
# Stored with every vector, so vectors from another model are never compared with these.
MODEL_ID = "multilingual-e5-small-q8"
DIMENSIONS = 384
# A clip's words past this many tokens add little to what it is about, and cost time.
MAX_TOKENS = 256
BATCH = 32

_lock = threading.Lock()
_loaded: Optional[tuple] = None

def download_size() -> int:
    """Bytes still to download before a search by meaning can run; 0 once it is here."""
    return sum(model.size for model in (SEARCH_MODEL, SEARCH_TOKENIZER) if not models.is_downloaded(model))

def _load() -> tuple:
    """The model and its tokenizer, downloaded the first time and kept in memory after."""
    global _loaded
    with _lock:
        if _loaded is None:
            import onnxruntime
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(models.ensure(SEARCH_TOKENIZER))
            tokenizer.enable_padding()
            tokenizer.enable_truncation(MAX_TOKENS)
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3
            session = onnxruntime.InferenceSession(models.ensure(SEARCH_MODEL), options,
                                                   providers=["CPUExecutionProvider"])
            _loaded = (session, tokenizer, {item.name for item in session.get_inputs()})
        return _loaded

def embed(texts: Sequence[str], query: bool = False) -> np.ndarray:
    """Turn texts into unit vectors that are close when the texts mean similar things.

    Args:
        texts: What to embed.
        query: These are questions rather than the clips they are asked of;
            the model was trained to tell the two apart by a prefix.

    Returns:
        One row of `DIMENSIONS` float32 values per text, each of length 1.
    """
    session, tokenizer, inputs = _load()
    prefix = "query: " if query else "passage: "
    rows = []
    for first in range(0, len(texts), BATCH):
        encoded = tokenizer.encode_batch([prefix + text for text in texts[first:first + BATCH]])
        ids = np.array([item.ids for item in encoded], dtype=np.int64)
        mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = session.run(None, feed)[0]
        # The mean of the real tokens, as the model was trained to be read.
        pooled = (hidden * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
        rows.append(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12))
    return np.concatenate(rows).astype(np.float32) if rows else np.zeros((0, DIMENSIONS), dtype=np.float32)

def digest(text: str) -> str:
    """What a stored vector was made from, so one made from older words is made again."""
    return hashlib.sha1(f"{MODEL_ID}\n{text}".encode("utf-8")).hexdigest()

def rank(question: str, passages: Dict[str, np.ndarray]) -> List[tuple]:
    """Order passages by how close they are to a question.

    Args:
        question: What the user is looking for, in their words.
        passages: Vectors by ID, as `embed` makes them.

    Returns:
        `(id, similarity)` pairs, closest first. Similarity is the cosine,
        and this model packs it tightly — on real footage, a close match and
        an unrelated sentence differ by a few hundredths around 0.85 — so it
        orders the clips of one search and means nothing compared across two.
    """
    if not passages:
        return []
    ids = list(passages)
    scores = np.stack([passages[key] for key in ids]) @ embed([question], query=True)[0]
    return sorted(zip(ids, (float(score) for score in scores)), key=lambda pair: -pair[1])
