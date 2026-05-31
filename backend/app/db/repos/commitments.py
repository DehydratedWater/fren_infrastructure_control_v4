"""Commitments repository."""

from __future__ import annotations

from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class CommitmentsRepo:
    async def save(
        self,
        commitment_id: str,
        pattern_type: str,
        commitment_text: str,
        confidence: float,
        *,
        full_match: str | None = None,
        source_message: str | None = None,
        source: str = "telegram",
        status: str = "pending",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO commitments (commitment_id, pattern_type, commitment_text,
                    confidence, date, full_match, source_message, source, status)
                VALUES (:cid, :pt, :ct, :conf, CURRENT_DATE, :fm, :sm, :src, :st)
                ON CONFLICT (commitment_id) DO NOTHING
                RETURNING *
            """,
                {
                    "cid": commitment_id,
                    "pt": pattern_type,
                    "ct": commitment_text,
                    "conf": confidence,
                    "fm": full_match,
                    "sm": source_message,
                    "src": source,
                    "st": status,
                },
            )  # type: ignore[return-value]

    async def get_pending(self, *, days: int = 7) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM commitments
                WHERE status = 'pending'
                  AND date >= CURRENT_DATE - :days * INTERVAL '1 day'
                ORDER BY date DESC
            """,
                {"days": days},
            )

    async def get_today(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM commitments WHERE date = CURRENT_DATE ORDER BY detected_at DESC",
            )

    async def update_status(
        self, commitment_id: str, status: str, *, goal_id: str | None = None
    ) -> dict[str, Any] | None:
        params: dict[str, Any] = {"cid": commitment_id, "st": status}
        extra = ""
        if goal_id:
            extra = ", linked_goal_id = :gid"
            params["gid"] = goal_id
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE commitments SET status = :st{extra} WHERE commitment_id = :cid RETURNING *",
                params,
            )
