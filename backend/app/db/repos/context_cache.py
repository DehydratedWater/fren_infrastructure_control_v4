"""Context cache repository -- central artifact registry."""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class ContextCacheRepo:
    async def create(
        self,
        cache_id: str,
        artifact_type: str,
        *,
        entity_type: str = "",
        entity_id: str = "",
        file_path: str = "",
        summary: str = "",
        tags: list[str] | None = None,
        content_class: str = "public",
        source_agent: str = "",
        expires_hours: int = 0,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "cache_id": cache_id,
            "artifact_type": artifact_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "file_path": file_path,
            "summary": summary,
            "tags": json.dumps(tags or []),
            "content_class": content_class,
            "source_agent": source_agent,
        }
        if expires_hours > 0:
            sql = """
                INSERT INTO context_cache
                    (cache_id, artifact_type, entity_type, entity_id, file_path,
                     summary, tags, content_class, source_agent, expires_at)
                VALUES
                    (:cache_id, :artifact_type, :entity_type, :entity_id, :file_path,
                     :summary, CAST(:tags AS jsonb), :content_class, :source_agent,
                     NOW() + make_interval(hours => :hours))
                RETURNING *
            """
            params["hours"] = expires_hours
        else:
            sql = """
                INSERT INTO context_cache
                    (cache_id, artifact_type, entity_type, entity_id, file_path,
                     summary, tags, content_class, source_agent)
                VALUES
                    (:cache_id, :artifact_type, :entity_type, :entity_id, :file_path,
                     :summary, CAST(:tags AS jsonb), :content_class, :source_agent)
                RETURNING *
            """
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def get(self, cache_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM context_cache WHERE cache_id = :cid",
                {"cid": cache_id},
            )

    async def list_recent(
        self,
        *,
        hours: int = 24,
        limit: int = 20,
        content_class: str = "public",
    ) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            if content_class == "full":
                # No content_class filter -- return all
                return await fetch_all(
                    s,
                    """
                    SELECT * FROM context_cache
                    WHERE created_at > NOW() - make_interval(hours => :hours)
                      AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """,
                    {"hours": hours, "limit": limit},
                )
            return await fetch_all(
                s,
                """
                SELECT * FROM context_cache
                WHERE created_at > NOW() - make_interval(hours => :hours)
                  AND (expires_at IS NULL OR expires_at > NOW())
                  AND content_class = :cls
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"hours": hours, "limit": limit, "cls": content_class},
            )

    async def list_by_tags(
        self,
        tags: list[str],
        *,
        hours: int = 72,
        limit: int = 20,
        content_class: str = "public",
    ) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM context_cache
                WHERE tags ?| CAST(:tags AS text[])
                  AND created_at > NOW() - make_interval(hours => :hours)
                  AND (expires_at IS NULL OR expires_at > NOW())
                  AND (content_class = :cls OR :cls = 'full')
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"tags": tags, "hours": hours, "limit": limit, "cls": content_class},
            )

    async def list_by_type(
        self,
        artifact_type: str,
        *,
        hours: int = 72,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM context_cache
                WHERE artifact_type = :atype
                  AND created_at > NOW() - make_interval(hours => :hours)
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"atype": artifact_type, "hours": hours, "limit": limit},
            )

    async def search(
        self,
        query: str,
        *,
        hours: int = 72,
        limit: int = 20,
        content_class: str = "public",
    ) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM context_cache
                WHERE summary ILIKE :q
                  AND created_at > NOW() - make_interval(hours => :hours)
                  AND (expires_at IS NULL OR expires_at > NOW())
                  AND (content_class = :cls OR :cls = 'full')
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"q": f"%{query}%", "hours": hours, "limit": limit, "cls": content_class},
            )

    async def list_newest(
        self,
        *,
        artifact_type: str | None = None,
        q: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Newest-first rows for the dashboard artifact gallery (no time cutoff).

        Optional ``artifact_type`` equality filter and ``q`` ILIKE substring
        filter on summary (``%``/``_``/``\\`` in the query are escaped so they
        match literally). Read-only.
        """
        conds: list[str] = ["1=1"]
        params: dict[str, Any] = {"limit": limit}
        if artifact_type:
            conds.append("artifact_type = :atype")
            params["atype"] = artifact_type
        if q:
            escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conds.append("summary ILIKE :q ESCAPE '\\'")
            params["q"] = f"%{escaped}%"
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM context_cache
                WHERE {" AND ".join(conds)}
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                params,
            )

    async def distinct_types(self) -> list[dict[str, Any]]:
        """Distinct artifact_type values with row counts, biggest first.

        Read-only aggregate for the dashboard type-filter buttons.
        """
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT artifact_type, COUNT(*) AS n
                FROM context_cache
                GROUP BY artifact_type
                ORDER BY n DESC, artifact_type ASC
                """,
                {},
            )

    async def delete(self, cache_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                "DELETE FROM context_cache WHERE cache_id = :cid",
                {"cid": cache_id},
            )
            return r.rowcount > 0

    async def delete_expired(self) -> int:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM context_cache WHERE expires_at IS NOT NULL AND expires_at < NOW()")
            return r.rowcount


async def add_to_cache(
    artifact_type: str,
    summary: str,
    *,
    entity_type: str = "",
    entity_id: str = "",
    file_path: str = "",
    tags: list[str] | None = None,
    content_class: str = "public",
    source_agent: str = "",
    expires_hours: int = 0,
) -> dict[str, Any]:
    """Convenience function for tool integration -- generates cache_id automatically."""
    repo = ContextCacheRepo()
    cache_id = f"ctx_{uuid.uuid4().hex[:12]}"
    return await repo.create(
        cache_id,
        artifact_type,
        entity_type=entity_type,
        entity_id=entity_id,
        file_path=file_path,
        summary=summary,
        tags=tags,
        content_class=content_class,
        source_agent=source_agent,
        expires_hours=expires_hours,
    )
