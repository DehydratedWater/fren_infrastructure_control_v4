"""Context pins repository — discussion topics, pinned context, document references."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class ContextPinsRepo:
    # ── Topics ──

    async def create_topic(
        self,
        topic_name: str,
        topic_summary: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            # Deactivate any currently active topic
            await execute_sql(
                s,
                "UPDATE discussion_topics SET is_active = FALSE, updated_at = NOW() WHERE is_active = TRUE",
            )
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO discussion_topics (topic_name, topic_summary, session_id)
                VALUES (:name, :summary, :sid)
                RETURNING *
                """,
                {"name": topic_name, "summary": topic_summary, "sid": session_id},
            )

    async def end_topic(self, topic_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE discussion_topics
                SET is_active = FALSE, updated_at = NOW()
                WHERE id = :tid
                RETURNING *
                """,
                {"tid": topic_id},
            )

    async def get_active_topic(self) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM discussion_topics WHERE is_active = TRUE ORDER BY updated_at DESC LIMIT 1",
            )

    async def list_topics(self, limit: int = 10) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM discussion_topics ORDER BY updated_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def update_topic(self, topic_id: int, topic_summary: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE discussion_topics
                SET topic_summary = :summary, updated_at = NOW()
                WHERE id = :tid
                RETURNING *
                """,
                {"tid": topic_id, "summary": topic_summary},
            )

    async def touch_topic(self, topic_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "UPDATE discussion_topics SET updated_at = NOW() WHERE id = :tid RETURNING *",
                {"tid": topic_id},
            )

    # ── Pins ──

    async def create_pin(
        self,
        content: str,
        content_type: str,
        topic_id: int,
        *,
        source_table: str = "",
        source_id: str = "",
        relevance_score: float = 1.0,
        expires_hours: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "content": content,
            "ctype": content_type,
            "tid": topic_id,
            "src_table": source_table,
            "src_id": source_id,
            "score": relevance_score,
            "meta": json.dumps(metadata or {}),
        }
        if expires_hours > 0:
            sql = """
                INSERT INTO context_pins
                    (content, content_type, topic_id, source_table, source_id,
                     relevance_score, expires_at, metadata)
                VALUES
                    (:content, :ctype, :tid, :src_table, :src_id,
                     :score, NOW() + make_interval(hours => :hours), CAST(:meta AS jsonb))
                RETURNING *
            """
            params["hours"] = expires_hours
        else:
            sql = """
                INSERT INTO context_pins
                    (content, content_type, topic_id, source_table, source_id,
                     relevance_score, metadata)
                VALUES
                    (:content, :ctype, :tid, :src_table, :src_id,
                     :score, CAST(:meta AS jsonb))
                RETURNING *
            """
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def delete_pin(self, pin_id: int) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM context_pins WHERE id = :pid", {"pid": pin_id})
            return r.rowcount > 0

    async def get_pins(
        self,
        topic_id: int,
        *,
        content_type: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            if content_type:
                return await fetch_all(
                    s,
                    """
                    SELECT * FROM context_pins
                    WHERE topic_id = :tid AND content_type = :ctype
                      AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY relevance_score DESC, pin_timestamp DESC
                    LIMIT :limit
                    """,
                    {"tid": topic_id, "ctype": content_type, "limit": limit},
                )
            return await fetch_all(
                s,
                """
                SELECT * FROM context_pins
                WHERE topic_id = :tid
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY relevance_score DESC, pin_timestamp DESC
                LIMIT :limit
                """,
                {"tid": topic_id, "limit": limit},
            )

    # ── Document References ──

    async def add_document_ref(
        self,
        document_id: str,
        topic_id: int,
        *,
        reference_reason: str = "",
        accessed_by: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO document_references
                    (document_id, topic_id, reference_reason, accessed_by, metadata)
                VALUES
                    (:did, :tid, :reason, :by, CAST(:meta AS jsonb))
                RETURNING *
                """,
                {
                    "did": document_id,
                    "tid": topic_id,
                    "reason": reference_reason,
                    "by": accessed_by,
                    "meta": json.dumps(metadata or {}),
                },
            )

    async def get_document_refs(self, topic_id: int, limit: int = 10) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM document_references
                WHERE topic_id = :tid
                ORDER BY reference_timestamp DESC
                LIMIT :limit
                """,
                {"tid": topic_id, "limit": limit},
            )

    # ── Composite ──

    async def get_current_context(
        self,
        *,
        include_pins: bool = True,
        include_docs: bool = True,
        message_limit: int = 10,
    ) -> dict[str, Any] | None:
        topic = await self.get_active_topic()
        if not topic:
            return None

        result: dict[str, Any] = {"topic": topic, "pins": [], "document_refs": [], "recent_messages": []}

        if include_pins:
            result["pins"] = await self.get_pins(topic["id"])

        if include_docs:
            result["document_refs"] = await self.get_document_refs(topic["id"])

        if message_limit > 0:
            from app.db.repos.chat import ChatMessagesRepo

            msgs = await ChatMessagesRepo().get_recent(limit=message_limit)
            msgs.reverse()  # get_recent returns DESC, want chronological
            result["recent_messages"] = msgs

        return result

    # ── Cleanup ──

    async def cleanup_expired(self) -> int:
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                "DELETE FROM context_pins WHERE expires_at IS NOT NULL AND expires_at < NOW()",
            )
            return r.rowcount
