"""Workflow master repository — session and message management."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class WorkflowMasterSessionsRepo:
    async def get_active(self) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                SELECT * FROM workflow_master_sessions
                WHERE status NOT IN ('completed', 'cancelled')
                ORDER BY created_at DESC LIMIT 1
            """,
            )

    async def get_or_create(self) -> dict[str, Any]:
        active = await self.get_active()
        if active:
            return active
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO workflow_master_sessions (status) VALUES ('active_creating')
                RETURNING *
            """,
            )  # type: ignore[return-value]

    async def update_status(self, session_id: int, status: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE workflow_master_sessions SET status = :status, updated_at = NOW()
                WHERE id = :sid RETURNING *
            """,
                {"sid": session_id, "status": status},
            )

    async def clear_active(self) -> int:
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                """
                UPDATE workflow_master_sessions SET status = 'cancelled', updated_at = NOW()
                WHERE status NOT IN ('completed', 'cancelled')
            """,
            )
            return r.rowcount


class WorkflowMasterMessagesRepo:
    async def save(
        self,
        session_id: int,
        role: str,
        content: str,
        *,
        message_type: str = "message",
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO workflow_master_messages
                    (session_id, role, content, message_type, metadata)
                VALUES (:sid, :role, :content, :mtype, CAST(:meta AS jsonb))
                RETURNING *
            """,
                {
                    "sid": session_id,
                    "role": role,
                    "content": content,
                    "mtype": message_type,
                    "meta": json.dumps(metadata or {}),
                },
            )  # type: ignore[return-value]

    async def get_history(self, session_id: int, *, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM workflow_master_messages
                WHERE session_id = :sid ORDER BY created_at ASC LIMIT :limit
            """,
                {"sid": session_id, "limit": limit},
            )
