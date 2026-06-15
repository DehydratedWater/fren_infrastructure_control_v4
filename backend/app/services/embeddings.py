"""Embedding service — local bge-m3 (A4000, OpenAI-compatible vLLM) by default.

Which model is used is settings-driven and verifiable at runtime:
  * EMBEDDING_BASE_URL (default http://192.168.0.42:8083/v1) → local bge-m3
  * EMBEDDING_MODEL    (default BAAI/bge-m3), 1024 dims, 8192-token context
Memories/messages embed PRIVATELY on our own hardware — nothing leaves the box.
If EMBEDDING_BASE_URL is cleared, it falls back to OpenAI text-embedding-3-small
(1536) using OPENAI_API_KEY. `active_model()` reports exactly what is in use.

Short texts embed directly. Long texts are chunked (each chunk embedded) and the
chunks stored in embedding_chunks pointing back to the source.
"""

from __future__ import annotations

from functools import lru_cache

from app.settings import get_settings

# text-embedding model token limits are ~8192; ~2.5 chars/token worst case, so
# 16000 chars (~6400 tokens) is a safe per-chunk cap for both bge-m3 and OpenAI.
_MAX_CHUNK_CHARS = 16000
_OVERLAP_CHARS = 400


def _resolve() -> tuple[str, str, int, str]:
    """(base_url, model, dims, api_key) for the ACTIVE embedding backend.

    base_url set → local bge-m3 (default). base_url empty → OpenAI fallback.
    """
    s = get_settings()
    if s.embedding_base_url:
        return s.embedding_base_url, s.embedding_model, int(s.embedding_dims), (s.embedding_api_key or "EMPTY")
    # OpenAI fallback (legacy): text-embedding-3-small, 1536 dims.
    return "", "text-embedding-3-small", 1536, s.openai_api_key


def active_model() -> dict:
    """What embedding backend is actually in use — for /health + smoke checks."""
    base_url, model, dims, key = _resolve()
    return {
        "model": model,
        "dims": dims,
        "endpoint": base_url or "https://api.openai.com (OpenAI)",
        "local": bool(base_url),
        "configured": bool(base_url or key),
    }


def dims() -> int:
    return _resolve()[2]


@lru_cache(maxsize=1)
def _get_client():
    from openai import OpenAI

    base_url, _model, _dims, key = _resolve()
    kwargs: dict = {"api_key": key or "EMPTY"}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def chunk_text(text: str) -> list[str]:
    """Split text into chunks that fit the model's token limit. Short text → [t]."""
    t = text.strip()
    if not t:
        return []
    if len(t) <= _MAX_CHUNK_CHARS:
        return [t]
    chunks = []
    start = 0
    while start < len(t):
        end = start + _MAX_CHUNK_CHARS
        chunks.append(t[start:end])
        start = end - _OVERLAP_CHARS  # overlap for context continuity
    return chunks


def get_embedding(text: str) -> list[float]:
    """Embed one text (truncated if long). Zero vector on empty input or when no
    backend is configured (no base_url AND no key)."""
    base_url, model, n_dims, key = _resolve()
    if not text or not text.strip():
        return [0.0] * n_dims
    if not base_url and not key:
        return [0.0] * n_dims
    t = text.strip()[:_MAX_CHUNK_CHARS]
    resp = _get_client().embeddings.create(input=[t], model=model)
    return resp.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch (each within token limit). Zero vectors for empty inputs."""
    base_url, model, n_dims, key = _resolve()
    if not texts:
        return []
    if not base_url and not key:
        return [[0.0] * n_dims] * len(texts)
    processed = [t.strip()[:_MAX_CHUNK_CHARS] for t in texts]
    non_empty: list[tuple[int, str]] = [(i, t) for i, t in enumerate(processed) if t]
    if not non_empty:
        return [[0.0] * n_dims] * len(texts)
    indices, cleaned = zip(*non_empty, strict=False)
    resp = _get_client().embeddings.create(input=list(cleaned), model=model)
    results: list[list[float]] = [[0.0] * n_dims] * len(texts)
    for idx, emb_data in zip(indices, resp.data, strict=False):
        results[idx] = emb_data.embedding
    return results
