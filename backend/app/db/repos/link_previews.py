"""Link previews repository — URL metadata cache for chat_messages enrichment."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


def _format_vector(embedding: list[float]) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


class LinkPreviewsRepo:
    async def get(self, url: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM link_previews WHERE url = :url",
                {"url": url},
            )

    async def get_many(self, urls: list[str]) -> dict[str, dict[str, Any]]:
        if not urls:
            return {}
        async with get_async_session() as s:
            rows = await fetch_all(
                s,
                "SELECT * FROM link_previews WHERE url = ANY(CAST(:urls AS text[]))",
                {"urls": urls},
            )
        return {r["url"]: r for r in rows}

    async def upsert(
        self,
        url: str,
        *,
        title: str | None = None,
        description: str | None = None,
        site_name: str | None = None,
        og_title: str | None = None,
        og_description: str | None = None,
        status: str = "ok",
        http_status: int | None = None,
        error: str | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "url": url,
            "title": title,
            "description": description,
            "site_name": site_name,
            "og_title": og_title,
            "og_description": og_description,
            "status": status,
            "http_status": http_status,
            "error": error,
            "fetched_at": datetime.now().astimezone(),
        }
        if embedding:
            params["embedding"] = _format_vector(embedding)
            sql = """
                INSERT INTO link_previews
                    (url, title, description, site_name, og_title, og_description,
                     fetched_at, status, http_status, error, embedding)
                VALUES
                    (:url, :title, :description, :site_name, :og_title, :og_description,
                     :fetched_at, :status, :http_status, :error, CAST(:embedding AS vector))
                ON CONFLICT (url) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    site_name = EXCLUDED.site_name,
                    og_title = EXCLUDED.og_title,
                    og_description = EXCLUDED.og_description,
                    fetched_at = EXCLUDED.fetched_at,
                    status = EXCLUDED.status,
                    http_status = EXCLUDED.http_status,
                    error = EXCLUDED.error,
                    embedding = EXCLUDED.embedding
                RETURNING *
            """
        else:
            sql = """
                INSERT INTO link_previews
                    (url, title, description, site_name, og_title, og_description,
                     fetched_at, status, http_status, error)
                VALUES
                    (:url, :title, :description, :site_name, :og_title, :og_description,
                     :fetched_at, :status, :http_status, :error)
                ON CONFLICT (url) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    site_name = EXCLUDED.site_name,
                    og_title = EXCLUDED.og_title,
                    og_description = EXCLUDED.og_description,
                    fetched_at = EXCLUDED.fetched_at,
                    status = EXCLUDED.status,
                    http_status = EXCLUDED.http_status,
                    error = EXCLUDED.error
                RETURNING *
            """
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def search_by_embedding(
        self, embedding: list[float], *, limit: int = 10, threshold: float = 0.3
    ) -> list[dict[str, Any]]:
        vec = _format_vector(embedding)
        sql = """
            SELECT url, title, description, site_name, og_title, og_description,
                   fetched_at, status,
                   1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM link_previews
            WHERE embedding IS NOT NULL AND status = 'ok'
              AND 1 - (embedding <=> CAST(:vec AS vector)) > :threshold
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :limit
        """
        async with get_async_session() as s:
            return await fetch_all(s, sql, {"vec": vec, "limit": limit, "threshold": threshold})

    async def list_pending(self, *, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT url FROM link_previews WHERE status = 'pending' ORDER BY fetched_at LIMIT :limit",
                {"limit": limit},
            )

    async def mark_status(self, url: str, status: str) -> None:
        async with get_async_session() as s:
            await execute_sql(
                s,
                "UPDATE link_previews SET status = :status WHERE url = :url",
                {"url": url, "status": status},
            )
