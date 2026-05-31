"""RP ban rules repository — anti-cliche pattern storage."""

from __future__ import annotations

from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class BanRuleRepo:
    async def create(
        self,
        adventure_id: int,
        rule: str,
        *,
        source: str = "auto",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO rp_ban_rules (adventure_id, rule, source)
                VALUES (:aid, :rule, :source)
                ON CONFLICT (adventure_id, rule) DO UPDATE SET is_active = TRUE
                RETURNING *
                """,
                {"aid": adventure_id, "rule": rule, "source": source},
            )

    async def list_active(self, adventure_id: int, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM rp_ban_rules
                WHERE adventure_id = :aid AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"aid": adventure_id, "limit": limit},
            )

    async def deactivate(self, rule_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "UPDATE rp_ban_rules SET is_active = FALSE WHERE id = :id RETURNING *",
                {"id": rule_id},
            )
