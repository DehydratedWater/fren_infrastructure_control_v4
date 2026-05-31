"""Agent notes repository — key-value store with TTL."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class AgentNotesRepo:
    async def set(
        self,
        note_key: str,
        note_value: Any,
        *,
        expires_hours: int = 24,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO agent_notes (note_key, note_value, expires_at)
                VALUES (:key, CAST(:val AS jsonb), NOW() + make_interval(hours => :hours))
                ON CONFLICT (note_key) DO UPDATE
                SET note_value = CAST(:val AS jsonb),
                    expires_at = NOW() + make_interval(hours => :hours),
                    updated_at = NOW()
                RETURNING *
            """,
                {
                    "key": note_key,
                    "val": json.dumps(note_value),
                    "hours": expires_hours,
                },
            )  # type: ignore[return-value]

    async def get(self, note_key: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                SELECT * FROM agent_notes
                WHERE note_key = :key AND (expires_at IS NULL OR expires_at > NOW())
            """,
                {"key": note_key},
            )

    async def get_by_prefix(self, prefix: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM agent_notes
                WHERE note_key LIKE :prefix
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY updated_at DESC
            """,
                {"prefix": f"{prefix}%"},
            )

    async def delete(self, note_key: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM agent_notes WHERE note_key = :key RETURNING id", {"key": note_key})
            return r.fetchone() is not None

    async def cleanup_expired(self) -> int:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM agent_notes WHERE expires_at < NOW()")
            return r.rowcount
