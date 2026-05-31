"""Cross-summary repo for bidirectional awareness between main bot and RP bot."""

from __future__ import annotations

from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class CrossSummaryRepo:
    async def write(
        self,
        direction: str,
        chat_id: int,
        summary: str,
        *,
        context_window: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO rp_cross_summaries (direction, chat_id, summary, context_window)
                VALUES (:dir, :cid, :summary, :ctx)
                RETURNING *
                """,
                {"dir": direction, "cid": chat_id, "summary": summary, "ctx": context_window},
            )

    async def read_latest(self, direction: str, chat_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                SELECT * FROM rp_cross_summaries
                WHERE direction = :dir AND chat_id = :cid
                ORDER BY created_at DESC LIMIT 1
                """,
                {"dir": direction, "cid": chat_id},
            )

    async def read_recent(self, direction: str, chat_id: int, *, limit: int = 5) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM rp_cross_summaries
                WHERE direction = :dir AND chat_id = :cid
                ORDER BY created_at DESC LIMIT :limit
                """,
                {"dir": direction, "cid": chat_id, "limit": limit},
            )
