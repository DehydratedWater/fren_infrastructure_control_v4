"""Embedding chunks repository — stores chunked embeddings for long texts."""

from __future__ import annotations

from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


def _format_vector(embedding: list[float]) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


class EmbeddingChunksRepo:
    async def store_chunks(
        self,
        source_table: str,
        source_id: str,
        chunks: list[tuple[int, str, list[float]]],
    ) -> int:
        """Store multiple (chunk_index, chunk_text, embedding) for a source row.

        Deletes existing chunks for this source first, then inserts new ones.
        Returns number of chunks stored.
        """
        async with get_async_session() as s:
            await execute_sql(
                s,
                "DELETE FROM embedding_chunks WHERE source_table = :st AND source_id = :sid",
                {"st": source_table, "sid": source_id},
            )
            for idx, text, emb in chunks:
                vec = _format_vector(emb)
                await execute_sql(
                    s,
                    """
                    INSERT INTO embedding_chunks (source_table, source_id, chunk_index, chunk_text, embedding)
                    VALUES (:st, :sid, :idx, :text, CAST(:vec AS vector))
                    """,
                    {"st": source_table, "sid": source_id, "idx": idx, "text": text, "vec": vec},
                )
        return len(chunks)

    async def search(
        self,
        embedding: list[float],
        *,
        source_table: str | None = None,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Cosine similarity search on chunks, optionally filtered by source_table."""
        vec = _format_vector(embedding)
        table_filter = "AND source_table = :st" if source_table else ""
        params: dict[str, Any] = {"vec": vec, "limit": limit, "threshold": threshold}
        if source_table:
            params["st"] = source_table
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT source_table, source_id, chunk_index,
                       LEFT(chunk_text, 500) AS text_preview,
                       1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM embedding_chunks
                WHERE 1 - (embedding <=> CAST(:vec AS vector)) > :threshold
                  {table_filter}
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :limit
                """,
                params,
            )

    async def count_for_source(self, source_table: str, source_id: str) -> int:
        """Count existing chunks for a given source row."""
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                "SELECT COUNT(*) AS cnt FROM embedding_chunks WHERE source_table = :st AND source_id = :sid",
                {"st": source_table, "sid": source_id},
            )
            return row["cnt"] if row else 0

    async def delete_for_source(self, source_table: str, source_id: str) -> int:
        """Delete all chunks for a given source row."""
        async with get_async_session() as s:
            result = await execute_sql(
                s,
                "DELETE FROM embedding_chunks WHERE source_table = :st AND source_id = :sid",
                {"st": source_table, "sid": source_id},
            )
            return result.rowcount
