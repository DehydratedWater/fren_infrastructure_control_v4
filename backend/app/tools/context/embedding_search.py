"""Embedding search tool — cross-table semantic search via pgvector.

Searches both direct embeddings on source tables AND chunked embeddings
in the embedding_chunks table (for long texts split into multiple pieces).
"""

from __future__ import annotations

import asyncio

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        default="search-all",
        description="search-all|search-messages|search-memories|search-facts|search-discoveries|search-documents|search-transcripts|chunk-embed|check-chunks (default: search-all)",
    )
    query: str = Field(default="", description="Search query text")
    source_table: str = Field(default="", description="Source table (for chunk-embed/check-chunks)")
    source_id: str = Field(default="", description="Source row ID (for chunk-embed/check-chunks)")
    force: bool = Field(default=False, description="Force re-embedding (for chunk-embed)")
    limit: int = Field(default=10, description="Max results")
    threshold: float = Field(default=0.3, description="Minimum cosine similarity (0-1)")


class Output(BaseModel):
    success: bool = True
    results: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


_TABLE_CONFIG = {
    "chat_messages": {"text_col": "message", "id_col": "id", "extra_cols": "sender, timestamp, date"},
    "memories": {"text_col": "content", "id_col": "memory_id", "extra_cols": "title, tags, category"},
    "user_facts": {"text_col": "fact_text", "id_col": "fact_id", "extra_cols": "category"},
    "profile_discoveries": {"text_col": "discovery", "id_col": "id", "extra_cols": "confidence, status"},
    "documents": {"text_col": "extracted_text", "id_col": "doc_id", "extra_cols": "original_filename, mime_type"},
}

# Tables whose long text can be chunked into embedding_chunks via chunk-embed.
_CHUNKABLE = {
    "documents": {"text_col": "extracted_text", "id_col": "doc_id"},
    "youtube_videos": {"text_col": "transcript", "id_col": "video_id"},
}


def _format_vector(embedding: list[float]) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


class EmbeddingSearchTool(ScriptTool[Input, Output]):
    name = "embedding_search"
    description = "Cross-table semantic search using embeddings"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _get_embedding(self, text: str) -> list[float] | None:
        try:
            from app.services.embeddings import get_embedding

            emb = await asyncio.to_thread(get_embedding, text)
            if all(v == 0.0 for v in emb[:10]):
                return None
            return emb
        except Exception:
            return None

    async def _search_table(
        self,
        table: str,
        embedding: list[float],
        *,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[dict]:
        from app.db.session import fetch_all, get_async_session

        cfg = _TABLE_CONFIG[table]
        vec = _format_vector(embedding)
        sql = f"""
            SELECT {cfg["id_col"]}, {cfg["extra_cols"]},
                   LEFT({cfg["text_col"]}, 500) AS text_preview,
                   1 - (embedding <=> CAST(:vec AS vector)) AS similarity,
                   '{table}' AS source_table
            FROM {table}
            WHERE embedding IS NOT NULL
              AND 1 - (embedding <=> CAST(:vec AS vector)) > :threshold
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :limit
        """
        async with get_async_session() as s:
            return await fetch_all(s, sql, {"vec": vec, "limit": limit, "threshold": threshold})

    async def _search_chunks(
        self,
        embedding: list[float],
        tables: list[str],
        *,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[dict]:
        from app.db.repos.embedding_chunks import EmbeddingChunksRepo

        repo = EmbeddingChunksRepo()
        all_chunk_results: list[dict] = []
        for table in tables:
            rows = await repo.search(embedding, source_table=table, limit=limit, threshold=threshold)
            all_chunk_results.extend(rows)
        return all_chunk_results

    async def _chunk_embed(self, source_table: str, source_id: str, force: bool) -> Output:
        if not source_table or not source_id:
            return Output(success=False, error="--source_table and --source_id required")

        cfg = _CHUNKABLE.get(source_table)
        if not cfg:
            return Output(
                success=False,
                error=f"Table '{source_table}' not chunkable. Supported: {list(_CHUNKABLE)}",
            )

        from app.db.repos.embedding_chunks import EmbeddingChunksRepo

        chunks_repo = EmbeddingChunksRepo()

        if not force:
            existing = await chunks_repo.count_for_source(source_table, source_id)
            if existing > 0:
                return Output(
                    success=True,
                    results=[
                        {
                            "source_table": source_table,
                            "source_id": source_id,
                            "chunk_count": existing,
                            "already_existed": True,
                        }
                    ],
                    count=existing,
                )

        # Read text from source table
        from app.db.session import fetch_one, get_async_session

        async with get_async_session() as s:
            row = await fetch_one(
                s,
                f"SELECT {cfg['text_col']} AS text_content FROM {source_table} WHERE {cfg['id_col']} = :sid",
                {"sid": source_id},
            )
        if not row:
            return Output(success=False, error=f"Not found: {source_table}/{source_id}")

        text = row.get("text_content", "")
        if not text or not text.strip():
            return Output(success=False, error=f"No text content in {source_table}/{source_id}")

        from app.services.embeddings import chunk_text, get_embeddings_batch

        text_chunks = chunk_text(text)
        if not text_chunks:
            return Output(success=False, error=f"No chunks produced for {source_table}/{source_id}")

        embeddings = await asyncio.to_thread(get_embeddings_batch, text_chunks)
        chunk_data = [(i, t, e) for i, (t, e) in enumerate(zip(text_chunks, embeddings, strict=True))]
        count = await chunks_repo.store_chunks(source_table, source_id, chunk_data)

        return Output(
            success=True,
            results=[
                {
                    "source_table": source_table,
                    "source_id": source_id,
                    "chunk_count": count,
                    "already_existed": False,
                }
            ],
            count=count,
        )

    async def _check_chunks(self, source_table: str, source_id: str) -> Output:
        if not source_table or not source_id:
            return Output(success=False, error="--source_table and --source_id required")

        from app.db.repos.embedding_chunks import EmbeddingChunksRepo

        count = await EmbeddingChunksRepo().count_for_source(source_table, source_id)
        return Output(
            success=True,
            results=[
                {
                    "source_table": source_table,
                    "source_id": source_id,
                    "has_chunks": count > 0,
                    "chunk_count": count,
                }
            ],
            count=count,
        )

    async def _run(self, inp: Input) -> Output:
        # Non-search commands
        if inp.command == "chunk-embed":
            return await self._chunk_embed(inp.source_table, inp.source_id, inp.force)
        if inp.command == "check-chunks":
            return await self._check_chunks(inp.source_table, inp.source_id)

        if not inp.query:
            return Output(success=False, error="query is required")

        embedding = await self._get_embedding(inp.query)
        if not embedding:
            return Output(success=False, error="Embedding unavailable (no API key?)")

        cmd_to_tables = {
            "search-messages": ["chat_messages"],
            "search-memories": ["memories"],
            "search-facts": ["user_facts"],
            "search-discoveries": ["profile_discoveries"],
            "search-documents": ["documents"],
            "search-transcripts": ["youtube_videos"],
            "search-all": [*_TABLE_CONFIG, "youtube_videos"],
        }

        tables = cmd_to_tables.get(inp.command)
        if tables is None:
            return Output(success=False, error=f"Unknown command: {inp.command}")

        all_results: list[dict] = []

        # Search direct embeddings (only tables with embedding column)
        for table in tables:
            if table in _TABLE_CONFIG:
                rows = await self._search_table(table, embedding, limit=inp.limit, threshold=inp.threshold)
                all_results.extend(rows)

        # Search chunked embeddings (works for any table)
        chunk_rows = await self._search_chunks(embedding, tables, limit=inp.limit, threshold=inp.threshold)
        all_results.extend(chunk_rows)

        # Deduplicate: prefer highest similarity per (source_table, source_id)
        seen: dict[tuple[str, str], dict] = {}
        for r in all_results:
            table_name = r.get("source_table", "")
            # Normalize ID key across direct and chunk results
            row_id = str(
                r.get("source_id") or r.get("id") or r.get("memory_id") or r.get("fact_id") or r.get("doc_id") or ""
            )
            key = (table_name, row_id)
            existing = seen.get(key)
            if not existing or r.get("similarity", 0) > existing.get("similarity", 0):
                seen[key] = r

        deduped = sorted(seen.values(), key=lambda r: r.get("similarity", 0), reverse=True)
        deduped = deduped[: inp.limit]

        return Output(success=True, results=deduped, count=len(deduped))


if __name__ == "__main__":
    EmbeddingSearchTool.run()
