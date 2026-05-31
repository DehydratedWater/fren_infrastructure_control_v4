"""OpenAI embedding service — text-embedding-3-small (1536 dims).

Short texts embed directly. Long texts are split into chunks, each embedded
separately, all stored in the embedding_chunks table pointing back to the source.
"""

from __future__ import annotations

from functools import lru_cache

from app.settings import get_settings

MODEL = "text-embedding-3-small"
DIMS = 1536
# text-embedding-3-small: 8192 token limit
# Worst-case ~2.5 chars/token for mixed content, so 16000 chars ≈ 6400 tokens
_MAX_CHUNK_CHARS = 16000
_OVERLAP_CHARS = 400


@lru_cache(maxsize=1)
def _get_client():
    from openai import OpenAI

    return OpenAI(api_key=get_settings().openai_api_key)


def chunk_text(text: str) -> list[str]:
    """Split text into chunks that fit within the embedding model's token limit.

    Returns a list of 1+ chunks. Short texts return a single-element list.
    """
    t = text.strip()
    if not t:
        return []
    if len(t) <= _MAX_CHUNK_CHARS:
        return [t]
    chunks = []
    start = 0
    while start < len(t):
        end = start + _MAX_CHUNK_CHARS
        chunk = t[start:end]
        chunks.append(chunk)
        start = end - _OVERLAP_CHARS  # overlap for context continuity
    return chunks


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text. Truncates if too long.

    Returns zero vector on empty input or missing key.
    For long texts, use chunk_text() + get_embeddings_batch() instead.
    """
    if not text or not text.strip():
        return [0.0] * DIMS
    if not get_settings().openai_api_key:
        return [0.0] * DIMS
    t = text.strip()
    if len(t) > _MAX_CHUNK_CHARS:
        t = t[:_MAX_CHUNK_CHARS]
    resp = _get_client().embeddings.create(input=[t], model=MODEL)
    return resp.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of texts. Each text must be within token limit.

    Returns zero vectors for empty inputs.
    """
    if not texts:
        return []
    if not get_settings().openai_api_key:
        return [[0.0] * DIMS] * len(texts)
    # Truncate each text to stay within limits
    processed = []
    for t in texts:
        t = t.strip()
        if len(t) > _MAX_CHUNK_CHARS:
            t = t[:_MAX_CHUNK_CHARS]
        processed.append(t)
    # Filter empties but track indices
    non_empty: list[tuple[int, str]] = [(i, t) for i, t in enumerate(processed) if t]
    if not non_empty:
        return [[0.0] * DIMS] * len(texts)
    indices, cleaned = zip(*non_empty, strict=False)
    resp = _get_client().embeddings.create(input=list(cleaned), model=MODEL)
    results: list[list[float]] = [[0.0] * DIMS] * len(texts)
    for idx, emb_data in zip(indices, resp.data, strict=False):
        results[idx] = emb_data.embedding
    return results
