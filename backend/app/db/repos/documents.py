"""Documents repository -- uploaded document storage and analysis tracking."""

from __future__ import annotations

from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class DocumentsRepo:
    async def create(
        self,
        doc_id: str,
        original_filename: str,
        *,
        file_path: str = "",
        mime_type: str = "",
        file_size_bytes: int = 0,
        extracted_text: str = "",
        chunk_count: int = 0,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "doc_id": doc_id,
            "original_filename": original_filename,
            "file_path": file_path,
            "mime_type": mime_type,
            "file_size_bytes": file_size_bytes,
            "extracted_text": extracted_text,
            "chunk_count": chunk_count,
        }
        if embedding:
            vec = "[" + ",".join(str(v) for v in embedding) + "]"
            params["embedding"] = vec
            sql = """
                INSERT INTO documents
                    (doc_id, original_filename, file_path, mime_type,
                     file_size_bytes, extracted_text, chunk_count, embedding)
                VALUES
                    (:doc_id, :original_filename, :file_path, :mime_type,
                     :file_size_bytes, :extracted_text, :chunk_count, CAST(:embedding AS vector))
                RETURNING *
            """
        else:
            sql = """
                INSERT INTO documents
                    (doc_id, original_filename, file_path, mime_type,
                     file_size_bytes, extracted_text, chunk_count)
                VALUES
                    (:doc_id, :original_filename, :file_path, :mime_type,
                     :file_size_bytes, :extracted_text, :chunk_count)
                RETURNING *
            """
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM documents WHERE doc_id = :doc_id",
                {"doc_id": doc_id},
            )

    async def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM documents
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"limit": limit},
            )

    async def update_analysis(
        self,
        doc_id: str,
        *,
        analysis: str,
        analysis_status: str,
    ) -> dict[str, Any] | None:
        sql = """
            UPDATE documents
            SET analysis = :analysis,
                analysis_status = :analysis_status,
                updated_at = NOW()
            WHERE doc_id = :doc_id
            RETURNING *
        """
        async with get_async_session() as s:
            return await fetch_one(
                s,
                sql,
                {
                    "doc_id": doc_id,
                    "analysis": analysis,
                    "analysis_status": analysis_status,
                },
            )

    async def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM documents
                WHERE extracted_text ILIKE :q OR analysis ILIKE :q
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"q": f"%{query}%", "limit": limit},
            )
