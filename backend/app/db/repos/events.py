"""Events repository — raw SQL, async."""

from __future__ import annotations

import json
from datetime import date as date_type
from datetime import datetime, timedelta
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class EventsRepo:
    async def create(
        self,
        event_id: str,
        category: str,
        title: str,
        occurred_at: Any,
        date: Any,
        *,
        subcategory: str | None = None,
        value: str | None = None,
        unit: str | None = None,
        source: str = "manual",
        source_message_id: int | None = None,
        metadata: dict | None = None,
        quantity: float | None = None,
        cost: float | None = None,
        currency: str | None = None,
        duration_minutes: int | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO events (event_id, category, subcategory, title, value, unit,
                    source, source_message_id, occurred_at, date, metadata,
                    quantity, cost, currency, duration_minutes)
                VALUES (:event_id, :category, :subcategory, :title, :value, :unit,
                    :source, :source_message_id, :occurred_at, :date,
                    CAST(:metadata AS jsonb),
                    :quantity, :cost, :currency, :duration_minutes)
                RETURNING *
            """,
                {
                    "event_id": event_id,
                    "category": category,
                    "subcategory": subcategory,
                    "title": title,
                    "value": value,
                    "unit": unit,
                    "source": source,
                    "source_message_id": source_message_id,
                    "occurred_at": occurred_at,
                    "date": date,
                    "metadata": json.dumps(metadata or {}),
                    "quantity": quantity,
                    "cost": cost,
                    "currency": currency or None,
                    "duration_minutes": duration_minutes,
                },
            )  # type: ignore[return-value]

    async def exists_for_message(self, source_message_id: int, category: str) -> bool:
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                """
                SELECT 1 FROM events
                WHERE source_message_id = :mid AND category = :cat
                LIMIT 1
            """,
                {"mid": source_message_id, "cat": category},
            )
            return row is not None

    async def exists_similar(
        self,
        category: str,
        occurred_at: Any,
        *,
        subcategory: str | None = None,
        value: str | None = None,
        window_minutes: int = 30,
    ) -> bool:
        """Check if a similar event exists within a time window (any source)."""
        conds = ["category = :cat", "occurred_at BETWEEN :t_start AND :t_end"]
        params: dict[str, Any] = {
            "cat": category,
            "t_start": occurred_at - timedelta(minutes=window_minutes),
            "t_end": occurred_at + timedelta(minutes=window_minutes),
        }
        if subcategory:
            conds.append("LOWER(subcategory) = LOWER(:sub)")
            params["sub"] = subcategory
        if value:
            conds.append("value = :val")
            params["val"] = value
        where = " AND ".join(conds)
        async with get_async_session() as s:
            row = await fetch_one(s, f"SELECT 1 FROM events WHERE {where} LIMIT 1", params)
            return row is not None

    async def get(self, event_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM events WHERE event_id = :eid", {"eid": event_id})

    async def list(
        self,
        *,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        conds: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if category:
            conds.append("category = :category")
            params["category"] = category
        if date_from:
            conds.append("date >= :date_from")
            params["date_from"] = date_type.fromisoformat(date_from)
        if date_to:
            conds.append("date <= :date_to")
            params["date_to"] = date_type.fromisoformat(date_to)
        if source:
            conds.append("source = :source")
            params["source"] = source
        where = " AND ".join(conds) if conds else "1=1"
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM events WHERE {where}
                ORDER BY occurred_at DESC LIMIT :limit
            """,
                params,
            )

    async def list_since_id(self, since_id: int, *, limit: int = 200) -> list[dict[str, Any]]:
        """Events with DB row id greater than *since_id*, oldest-first.

        Cursor-style read used by periodic consumers (event→habit bridge) so
        each event is processed exactly once across cron ticks.
        """
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM events
                WHERE id > :since_id
                ORDER BY id ASC LIMIT :limit
            """,
                {"since_id": since_id, "limit": limit},
            )

    async def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM events
                ORDER BY occurred_at DESC LIMIT :limit
            """,
                {"limit": limit},
            )

    async def list_by_category(self, category: str, days: int = 30) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM events
                WHERE category = :category
                  AND date >= CURRENT_DATE - CAST(:days AS integer)
                ORDER BY occurred_at ASC
            """,
                {"category": category, "days": days},
            )

    async def update(
        self,
        event_id: str,
        *,
        category: str | None = None,
        subcategory: str | None = None,
        title: str | None = None,
        value: str | None = None,
        unit: str | None = None,
        occurred_at: Any | None = None,
        quantity: float | None = None,
        cost: float | None = None,
        currency: str | None = None,
        duration_minutes: int | None = None,
    ) -> dict[str, Any] | None:
        """Update an existing event. Only non-None fields are changed."""
        sets: list[str] = []
        params: dict[str, Any] = {"eid": event_id}
        if category is not None:
            sets.append("category = :category")
            params["category"] = category
        if subcategory is not None:
            sets.append("subcategory = :subcategory")
            params["subcategory"] = subcategory
        if title is not None:
            sets.append("title = :title")
            params["title"] = title
        if value is not None:
            sets.append("value = :value")
            params["value"] = value
        if unit is not None:
            sets.append("unit = :unit")
            params["unit"] = unit
        if occurred_at is not None:
            sets.append("occurred_at = :occurred_at")
            params["occurred_at"] = occurred_at
            if isinstance(occurred_at, datetime):
                sets.append("date = :date")
                params["date"] = occurred_at.date()
        if quantity is not None:
            sets.append("quantity = :quantity")
            params["quantity"] = quantity
        if cost is not None:
            sets.append("cost = :cost")
            params["cost"] = cost
        if currency is not None:
            sets.append("currency = :currency")
            params["currency"] = currency
        if duration_minutes is not None:
            sets.append("duration_minutes = :duration_minutes")
            params["duration_minutes"] = duration_minutes
        if not sets:
            return await self.get(event_id)
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE events SET {', '.join(sets)} WHERE event_id = :eid RETURNING *",
                params,
            )

    async def delete(self, event_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                "DELETE FROM events WHERE event_id = :eid RETURNING id",
                {"eid": event_id},
            )
            return r.fetchone() is not None

    async def get_extraction_state(self) -> dict[str, Any]:
        async with get_async_session() as s:
            row = await fetch_one(s, "SELECT * FROM event_extraction_state WHERE id = 1")
            return row or {"last_processed_message_id": 0, "last_run_at": None}

    async def update_extraction_state(self, last_processed_message_id: int) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE event_extraction_state
                SET last_processed_message_id = :mid, last_run_at = NOW(), updated_at = NOW()
                WHERE id = 1 RETURNING *
            """,
                {"mid": last_processed_message_id},
            )  # type: ignore[return-value]

    async def get_daily_summary(self, date: str) -> list[dict[str, Any]]:
        date_obj = date_type.fromisoformat(date) if isinstance(date, str) else date
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT category, COUNT(*) as count
                FROM events
                WHERE date = :date
                GROUP BY category
                ORDER BY count DESC
            """,
                {"date": date_obj},
            )
