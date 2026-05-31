"""Checker state repository (single-row table)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

from app.db.session import fetch_one, get_async_session


class CheckerStateRepo:
    async def get(self) -> dict[str, Any]:
        async with get_async_session() as s:
            row = await fetch_one(s, "SELECT * FROM checker_state WHERE id = 1")
            return row or {}

    async def update(
        self,
        *,
        last_reminder_at: datetime | str | None = None,
        last_trigger_reason: str | None = None,
    ) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {}
        if last_reminder_at is not None:
            if isinstance(last_reminder_at, str):
                from datetime import datetime as _dt

                last_reminder_at = _dt.fromisoformat(last_reminder_at)
            sets.append("last_reminder_at = CAST(:lra AS timestamptz)")
            params["lra"] = last_reminder_at
        if last_trigger_reason is not None:
            sets.append("last_trigger_reason = :ltr")
            params["ltr"] = last_trigger_reason
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE checker_state SET {", ".join(sets)} WHERE id = 1 RETURNING *
            """,
                params,
            )

    async def get_category_cooldowns(self) -> dict[str, str]:
        """Get per-category last-trigger timestamps as {category: iso_timestamp}."""
        row = await self.get()
        raw = row.get("last_triggers")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        return raw if isinstance(raw, dict) else {}

    async def set_category_cooldown(self, category: str, timestamp: datetime) -> None:
        """Set the last-trigger timestamp for a specific category."""
        cooldowns = await self.get_category_cooldowns()
        cooldowns[category] = timestamp.isoformat()
        async with get_async_session() as s:
            await fetch_one(
                s,
                "UPDATE checker_state SET last_triggers = CAST(:lt AS jsonb), updated_at = NOW() WHERE id = 1 RETURNING id",
                {"lt": json.dumps(cooldowns)},
            )

    async def record_tier_send(self, tier: str, timestamp: datetime) -> None:
        """Record a proactive send for a tier. Stored as `tier:<name>` in last_triggers
        to share the column with category cooldowns without key collision."""
        if ":" in tier:
            raise ValueError("tier name must not contain ':'")
        cooldowns = await self.get_category_cooldowns()
        cooldowns[f"tier:{tier}"] = timestamp.isoformat()
        async with get_async_session() as s:
            await fetch_one(
                s,
                "UPDATE checker_state SET last_triggers = CAST(:lt AS jsonb), updated_at = NOW() WHERE id = 1 RETURNING id",
                {"lt": json.dumps(cooldowns)},
            )

    async def get_tier_last_send(self, tier: str) -> str | None:
        """Return ISO timestamp of last proactive send for a tier, or None if never."""
        cooldowns = await self.get_category_cooldowns()
        val = cooldowns.get(f"tier:{tier}")
        return val if isinstance(val, str) else None

    async def get_all_tier_sends(self) -> dict[str, str]:
        """Return all tier-prefixed sends as {tier_name: iso_ts}, prefix stripped."""
        cooldowns = await self.get_category_cooldowns()
        return {k[len("tier:") :]: v for k, v in cooldowns.items() if k.startswith("tier:") and isinstance(v, str)}
