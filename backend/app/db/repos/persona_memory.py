"""Persona memory repos — persona_interests, topic_nodes, pending_thoughts, rss_feeds.

Gives Twily her own interest backlog, a topic tree over user's recurring themes,
a motivation-scored pending-thoughts queue, and an editable RSS feed registry.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


def _format_vector(embedding: list[float]) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


# ───────────────────────── persona_interests ─────────────────────────


class PersonaInterestsRepo:
    async def create(
        self,
        *,
        topic: str,
        stance: str | None = None,
        source: str = "self_reflection",
        source_url: str | None = None,
        embedding: list[float] | None = None,
        novelty_score: float = 0.5,
        expires_at: datetime | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "topic": topic,
            "stance": stance,
            "source": source,
            "source_url": source_url,
            "novelty": novelty_score,
            "expires_at": expires_at,
        }
        if embedding:
            params["embedding"] = _format_vector(embedding)
            sql = """
                INSERT INTO persona_interests
                    (topic, stance, source, source_url, embedding, novelty_score, expires_at)
                VALUES (:topic, :stance, :source, :source_url,
                        CAST(:embedding AS vector), :novelty, :expires_at)
                RETURNING *
            """
        else:
            sql = """
                INSERT INTO persona_interests
                    (topic, stance, source, source_url, novelty_score, expires_at)
                VALUES (:topic, :stance, :source, :source_url, :novelty, :expires_at)
                RETURNING *
            """
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def list_active(self, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM persona_interests
                WHERE expires_at IS NULL OR expires_at > NOW()
                ORDER BY novelty_score DESC, created_at DESC
                LIMIT :limit
                """,
                {"limit": limit},
            )

    async def list_top_by_novelty(self, *, limit: int = 5) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM persona_interests
                WHERE (expires_at IS NULL OR expires_at > NOW())
                ORDER BY novelty_score DESC
                LIMIT :limit
                """,
                {"limit": limit},
            )

    async def search_by_embedding(
        self, embedding: list[float], *, limit: int = 10, threshold: float = 0.3
    ) -> list[dict[str, Any]]:
        vec = _format_vector(embedding)
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT *, 1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM persona_interests
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> CAST(:vec AS vector)) > :threshold
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :limit
                """,
                {"vec": vec, "limit": limit, "threshold": threshold},
            )

    async def mark_surfaced(self, interest_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE persona_interests
                SET surface_count = surface_count + 1,
                    last_surfaced_at = NOW(),
                    novelty_score = GREATEST(novelty_score * 0.85, 0.05)
                WHERE id = :id
                RETURNING *
                """,
                {"id": interest_id},
            )

    async def prune_expired(self) -> int:
        async with get_async_session() as s:
            result = await execute_sql(
                s,
                """
                DELETE FROM persona_interests
                WHERE (expires_at IS NOT NULL AND expires_at < NOW())
                   OR (surface_count > 3 AND created_at < NOW() - INTERVAL '21 days')
                """,
            )
            return result.rowcount or 0

    async def delete(self, interest_id: int) -> bool:
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                "DELETE FROM persona_interests WHERE id = :id RETURNING id",
                {"id": interest_id},
            )
            return row is not None


# ───────────────────────── topic_nodes ─────────────────────────


class TopicNodesRepo:
    async def create_node(
        self,
        *,
        label: str,
        summary: str | None,
        embedding: list[float],
        parent_id: int | None = None,
        depth: int = 0,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO topic_nodes (parent_id, label, summary, embedding, depth, hit_count, last_hit_at)
                VALUES (:parent_id, :label, :summary, CAST(:embedding AS vector), :depth, 1, NOW())
                RETURNING *
                """,
                {
                    "parent_id": parent_id,
                    "label": label,
                    "summary": summary,
                    "embedding": _format_vector(embedding),
                    "depth": depth,
                },
            )

    async def find_nearest_child(
        self,
        parent_id: int | None,
        embedding: list[float],
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Return children of parent_id ordered by cosine similarity."""
        vec = _format_vector(embedding)
        parent_filter = "parent_id IS NULL" if parent_id is None else "parent_id = :parent_id"
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT *, 1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM topic_nodes
                WHERE {parent_filter} AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :limit
                """,
                {"vec": vec, "parent_id": parent_id, "limit": limit},
            )

    async def bump_hit(self, node_id: int) -> None:
        async with get_async_session() as s:
            await execute_sql(
                s,
                """
                UPDATE topic_nodes
                SET hit_count = hit_count + 1, last_hit_at = NOW()
                WHERE id = :id
                """,
                {"id": node_id},
            )

    async def update_summary(self, node_id: int, summary: str) -> None:
        async with get_async_session() as s:
            await execute_sql(
                s,
                "UPDATE topic_nodes SET summary = :s WHERE id = :id",
                {"s": summary, "id": node_id},
            )

    async def set_salience(self, node_id: int, salience: float) -> None:
        async with get_async_session() as s:
            await execute_sql(
                s,
                "UPDATE topic_nodes SET salience = :s WHERE id = :id",
                {"s": salience, "id": node_id},
            )

    async def list_all(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(s, "SELECT * FROM topic_nodes", {})

    async def list_by_salience(self, *, limit: int = 10) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM topic_nodes ORDER BY salience DESC LIMIT :limit",
                {"limit": limit},
            )

    async def list_stale_for_drift(
        self, *, min_salience: float = 0.2, min_hours_since_surfaced: int = 72, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Stale-but-salient nodes worth bringing up again."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM topic_nodes
                WHERE salience >= :min_sal
                  AND (last_surfaced_at IS NULL OR last_surfaced_at < NOW() - INTERVAL '1 hour' * :hours)
                ORDER BY salience DESC, COALESCE(last_surfaced_at, 'epoch'::timestamptz) ASC
                LIMIT :limit
                """,
                {"min_sal": min_salience, "hours": min_hours_since_surfaced, "limit": limit},
            )

    async def mark_surfaced(self, node_id: int) -> None:
        async with get_async_session() as s:
            await execute_sql(
                s,
                "UPDATE topic_nodes SET last_surfaced_at = NOW() WHERE id = :id",
                {"id": node_id},
            )

    async def prune(self, *, max_salience: float = 0.05, max_hit_count: int = 2, min_age_days: int = 14) -> int:
        async with get_async_session() as s:
            result = await execute_sql(
                s,
                """
                DELETE FROM topic_nodes
                WHERE salience < :max_sal
                  AND hit_count < :max_hits
                  AND created_at < NOW() - INTERVAL '1 day' * :days
                """,
                {"max_sal": max_salience, "max_hits": max_hit_count, "days": min_age_days},
            )
            return result.rowcount or 0


# ───────────────────────── pending_thoughts ─────────────────────────


class PendingThoughtsRepo:
    async def create(
        self,
        *,
        content: str,
        kind: str,
        motivation_score: float,
        motivation_breakdown: dict[str, Any] | None = None,
        topic_node_id: int | None = None,
        persona_interest_id: int | None = None,
    ) -> dict[str, Any]:
        import json

        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO pending_thoughts
                    (content, kind, motivation_score, motivation_breakdown,
                     topic_node_id, persona_interest_id)
                VALUES (:content, :kind, :score, CAST(:breakdown AS jsonb), :node_id, :interest_id)
                RETURNING *
                """,
                {
                    "content": content,
                    "kind": kind,
                    "score": motivation_score,
                    "breakdown": json.dumps(motivation_breakdown) if motivation_breakdown else None,
                    "node_id": topic_node_id,
                    "interest_id": persona_interest_id,
                },
            )

    async def peek_top(
        self, *, kinds: list[str] | None = None, limit: int = 3, min_motivation: float = 0.0
    ) -> list[dict[str, Any]]:
        """Return top unconsumed thoughts by motivation_score. Does NOT consume."""
        if kinds:
            sql = """
                SELECT * FROM pending_thoughts
                WHERE consumed_at IS NULL
                  AND kind = ANY(CAST(:kinds AS text[]))
                  AND motivation_score >= :min_mot
                ORDER BY motivation_score DESC, created_at ASC
                LIMIT :limit
            """
            params: dict[str, Any] = {"kinds": kinds, "limit": limit, "min_mot": min_motivation}
        else:
            sql = """
                SELECT * FROM pending_thoughts
                WHERE consumed_at IS NULL AND motivation_score >= :min_mot
                ORDER BY motivation_score DESC, created_at ASC
                LIMIT :limit
            """
            params = {"limit": limit, "min_mot": min_motivation}
        async with get_async_session() as s:
            return await fetch_all(s, sql, params)

    async def consume(self, thought_id: int, by: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE pending_thoughts
                SET consumed_at = NOW(), consumed_by = :by
                WHERE id = :id AND consumed_at IS NULL
                RETURNING *
                """,
                {"id": thought_id, "by": by},
            )

    async def expire_old(self, *, hours: int = 48) -> int:
        async with get_async_session() as s:
            result = await execute_sql(
                s,
                """
                DELETE FROM pending_thoughts
                WHERE consumed_at IS NULL
                  AND created_at < NOW() - INTERVAL '1 hour' * :hours
                """,
                {"hours": hours},
            )
            return result.rowcount or 0

    async def count_unconsumed(self) -> int:
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                "SELECT COUNT(*) AS c FROM pending_thoughts WHERE consumed_at IS NULL",
                {},
            )
            return int(row["c"]) if row else 0

    async def trim_queue(self, *, max_size: int = 30) -> int:
        """Drop lowest-scored unconsumed thoughts if queue exceeds max_size."""
        async with get_async_session() as s:
            result = await execute_sql(
                s,
                """
                DELETE FROM pending_thoughts
                WHERE id IN (
                    SELECT id FROM pending_thoughts
                    WHERE consumed_at IS NULL
                    ORDER BY motivation_score ASC, created_at ASC
                    OFFSET :max_size
                )
                """,
                {"max_size": max_size},
            )
            return result.rowcount or 0

    async def list_recent(self, *, limit: int = 20, consumed: bool | None = None) -> list[dict[str, Any]]:
        where = ""
        if consumed is True:
            where = "WHERE consumed_at IS NOT NULL"
        elif consumed is False:
            where = "WHERE consumed_at IS NULL"
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"SELECT * FROM pending_thoughts {where} ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )


# ───────────────────────── rss_feeds ─────────────────────────


class RssFeedsRepo:
    async def list_all(self, *, theme: str | None = None, enabled: bool | None = None) -> list[dict[str, Any]]:
        where_parts: list[str] = []
        params: dict[str, Any] = {}
        if theme:
            where_parts.append("theme = :theme")
            params["theme"] = theme
        if enabled is not None:
            where_parts.append("enabled = :enabled")
            params["enabled"] = enabled
        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"SELECT * FROM rss_feeds {where_sql} ORDER BY theme, name",
                params,
            )

    async def get(self, feed_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM rss_feeds WHERE id = :id", {"id": feed_id})

    async def create(
        self, *, theme: str, url: str, name: str, max_items_per_run: int = 3, enabled: bool = True
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO rss_feeds (theme, url, name, max_items_per_run, enabled)
                VALUES (:theme, :url, :name, :max_items, :enabled)
                RETURNING *
                """,
                {
                    "theme": theme,
                    "url": url,
                    "name": name,
                    "max_items": max_items_per_run,
                    "enabled": enabled,
                },
            )

    async def update(
        self,
        feed_id: int,
        *,
        theme: str | None = None,
        url: str | None = None,
        name: str | None = None,
        max_items_per_run: int | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any] | None:
        sets: list[str] = []
        params: dict[str, Any] = {"id": feed_id}
        if theme is not None:
            sets.append("theme = :theme")
            params["theme"] = theme
        if url is not None:
            sets.append("url = :url")
            params["url"] = url
        if name is not None:
            sets.append("name = :name")
            params["name"] = name
        if max_items_per_run is not None:
            sets.append("max_items_per_run = :max_items")
            params["max_items"] = max_items_per_run
        if enabled is not None:
            sets.append("enabled = :enabled")
            params["enabled"] = enabled
        if not sets:
            return await self.get(feed_id)
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE rss_feeds SET {', '.join(sets)} WHERE id = :id RETURNING *",
                params,
            )

    async def toggle(self, feed_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "UPDATE rss_feeds SET enabled = NOT enabled WHERE id = :id RETURNING *",
                {"id": feed_id},
            )

    async def delete(self, feed_id: int) -> bool:
        async with get_async_session() as s:
            row = await fetch_one(s, "DELETE FROM rss_feeds WHERE id = :id RETURNING id", {"id": feed_id})
            return row is not None

    async def mark_fetched(
        self,
        feed_id: int,
        *,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE rss_feeds
                SET last_fetched_at = NOW(), last_status = :status, last_error = :error
                WHERE id = :id
                RETURNING *
                """,
                {"id": feed_id, "status": status, "error": error},
            )

    async def due_for_fetch(self, *, min_age_hours: int = 6, limit: int = 100) -> list[dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(hours=min_age_hours)
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM rss_feeds
                WHERE enabled = TRUE
                  AND (last_fetched_at IS NULL OR last_fetched_at < :cutoff)
                ORDER BY COALESCE(last_fetched_at, 'epoch'::timestamptz) ASC
                LIMIT :limit
                """,
                {"cutoff": cutoff, "limit": limit},
            )
