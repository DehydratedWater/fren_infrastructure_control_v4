"""Memories repository — CRUD + tag/embedding search."""

from __future__ import annotations

from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


def _format_vector(embedding: list[float]) -> str:
    """Format Python list as pgvector string literal '[0.1,0.2,...]'."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


class MemoriesRepo:
    async def create(
        self,
        memory_id: str,
        title: str,
        content: str,
        *,
        tags: list[str] | None = None,
        category: str = "",
        source: str = "user",
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "mid": memory_id,
            "title": title,
            "content": content,
            "tags": tags or [],
            "category": category,
            "source": source,
        }
        if embedding:
            params["embedding"] = _format_vector(embedding)
            sql = """
                INSERT INTO memories (memory_id, title, content, tags, category, source, embedding)
                VALUES (:mid, :title, :content, CAST(:tags AS text[]), :category, :source,
                        CAST(:embedding AS vector))
                RETURNING *
            """
        else:
            sql = """
                INSERT INTO memories (memory_id, title, content, tags, category, source)
                VALUES (:mid, :title, :content, CAST(:tags AS text[]), :category, :source)
                RETURNING *
            """
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM memories WHERE memory_id = :mid",
                {"mid": memory_id},
            )

    async def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def delete(self, memory_id: str) -> bool:
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                "DELETE FROM memories WHERE memory_id = :mid RETURNING memory_id",
                {"mid": memory_id},
            )
            return row is not None

    async def update(
        self,
        memory_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        category: str | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"mid": memory_id}
        if title is not None:
            sets.append("title = :title")
            params["title"] = title
        if content is not None:
            sets.append("content = :content")
            params["content"] = content
        if tags is not None:
            sets.append("tags = CAST(:tags AS text[])")
            params["tags"] = tags
        if category is not None:
            sets.append("category = :category")
            params["category"] = category
        if embedding is not None:
            sets.append("embedding = CAST(:embedding AS vector)")
            params["embedding"] = _format_vector(embedding)
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE memories SET {', '.join(sets)} WHERE memory_id = :mid RETURNING *",
                params,
            )

    async def search_by_text(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """ILIKE fallback when embeddings are unavailable."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM memories
                WHERE title ILIKE :q OR content ILIKE :q
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"q": f"%{query}%", "limit": limit},
            )

    async def search_by_tags(self, tags: list[str], *, limit: int = 20) -> list[dict[str, Any]]:
        """Find memories with overlapping tags (GIN-indexed)."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT *
                FROM memories
                WHERE tags && CAST(:tags AS text[])
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"tags": tags, "limit": limit},
            )

    async def search_by_embedding(
        self,
        embedding: list[float],
        *,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Cosine similarity search on memories."""
        vec = _format_vector(embedding)
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT *, 1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM memories
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> CAST(:vec AS vector)) > :threshold
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :limit
                """,
                {"vec": vec, "limit": limit, "threshold": threshold},
            )

    async def search_hybrid(
        self,
        embedding: list[float],
        tags: list[str],
        *,
        limit: int = 10,
        threshold: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Combined tag + embedding search with tag boost."""
        vec = _format_vector(embedding)
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT *,
                    1 - (embedding <=> CAST(:vec AS vector)) AS similarity,
                    COALESCE((SELECT COUNT(*) FROM unnest(tags) t WHERE t = ANY(CAST(:tags AS text[]))), 0) AS tag_overlap,
                    (1 - (embedding <=> CAST(:vec AS vector)))
                        + COALESCE((SELECT COUNT(*) FROM unnest(tags) t WHERE t = ANY(CAST(:tags AS text[]))), 0) * 0.1 AS combined_score
                FROM memories
                WHERE embedding IS NOT NULL
                  AND (
                    1 - (embedding <=> CAST(:vec AS vector)) > :threshold
                    OR tags && CAST(:tags AS text[])
                  )
                ORDER BY combined_score DESC
                LIMIT :limit
                """,
                {"vec": vec, "tags": tags, "limit": limit, "threshold": threshold},
            )
