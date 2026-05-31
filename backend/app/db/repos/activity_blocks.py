"""Activity blocks repository -- structured activity timeline with freeze semantics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import date, datetime

from app.db.session import execute_sql, fetch_all, get_async_session


class ActivityBlocksRepo:
    async def freeze_old_blocks(self, block_date: date) -> int:
        """Freeze blocks that started more than 6 hours ago."""
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                """
                UPDATE activity_blocks
                SET frozen_at = NOW(), updated_at = NOW()
                WHERE block_date = :block_date
                  AND frozen_at IS NULL
                  AND started_at < NOW() - INTERVAL '6 hours'
                """,
                {"block_date": block_date},
            )
            return r.rowcount

    async def delete_unfrozen(self, block_date: date) -> int:
        """Delete all unfrozen (rewritable) blocks for a date."""
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                """
                DELETE FROM activity_blocks
                WHERE block_date = :block_date
                  AND frozen_at IS NULL
                """,
                {"block_date": block_date},
            )
            return r.rowcount

    async def insert_blocks(self, block_date: date, blocks: list[dict[str, Any]]) -> int:
        """Batch insert blocks for a date. Returns count inserted."""
        if not blocks:
            return 0
        count = 0
        async with get_async_session() as s:
            for b in blocks:
                await execute_sql(
                    s,
                    """
                    INSERT INTO activity_blocks
                        (block_date, started_at, ended_at, activity_type, title,
                         description, application, project, environment,
                         health_snapshot, tags, confidence)
                    VALUES
                        (:block_date, :started_at, :ended_at, :activity_type, :title,
                         :description, :application, :project,
                         CAST(:environment AS jsonb), CAST(:health_snapshot AS jsonb),
                         CAST(:tags AS jsonb), :confidence)
                    """,
                    {
                        "block_date": block_date,
                        "started_at": b["started_at"],
                        "ended_at": b.get("ended_at"),
                        "activity_type": b.get("activity_type", "unknown"),
                        "title": b.get("title", ""),
                        "description": b.get("description", ""),
                        "application": b.get("application", ""),
                        "project": b.get("project", ""),
                        "environment": json.dumps(b.get("environment", {})),
                        "health_snapshot": json.dumps(b.get("health_snapshot", {})),
                        "tags": json.dumps(b.get("tags", [])),
                        "confidence": b.get("confidence", 1.0),
                    },
                )
                count += 1
        return count

    async def replace_recent(self, block_date: date, blocks: list[dict[str, Any]]) -> dict[str, int]:
        """Main entry point: freeze old → delete unfrozen → insert new blocks."""
        frozen = await self.freeze_old_blocks(block_date)
        deleted = await self.delete_unfrozen(block_date)
        inserted = await self.insert_blocks(block_date, blocks)
        return {"frozen": frozen, "deleted": deleted, "inserted": inserted}

    async def get_frozen_blocks(self, block_date: date) -> list[dict[str, Any]]:
        """Get immutable (frozen) blocks for a date — used as LLM context."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM activity_blocks
                WHERE block_date = :block_date
                  AND frozen_at IS NOT NULL
                ORDER BY started_at
                """,
                {"block_date": block_date},
            )

    async def get_all_blocks(self, block_date: date) -> list[dict[str, Any]]:
        """Get all blocks (frozen + unfrozen) for a date."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM activity_blocks
                WHERE block_date = :block_date
                ORDER BY started_at
                """,
                {"block_date": block_date},
            )

    async def get_recent_blocks(self, hours: int = 6) -> list[dict[str, Any]]:
        """Get blocks from the last N hours — for the thinking agent."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM activity_blocks
                WHERE started_at > NOW() - make_interval(hours => :hours)
                ORDER BY started_at
                """,
                {"hours": hours},
            )

    async def get_range(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Get blocks within a datetime range."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM activity_blocks
                WHERE started_at >= :start AND started_at < :end
                ORDER BY started_at
                """,
                {"start": start, "end": end},
            )
