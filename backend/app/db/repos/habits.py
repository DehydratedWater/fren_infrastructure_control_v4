"""Habits repository — raw SQL, async."""

from __future__ import annotations

import json
from datetime import date as _date
from datetime import time as _time
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class HabitsRepo:
    async def create(
        self,
        habit_id: str,
        title: str,
        frequency_type: str,
        *,
        description: str | None = None,
        importance_level: int = 3,
        frequency_detail: dict | None = None,
        preferred_time_start: str | None = None,
        preferred_time_end: str | None = None,
        generates_type: str = "none",
        generation_template: dict | None = None,
        validation_rules: dict | None = None,
        linked_priority_id: str | None = None,
        linked_goal_id: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO habits (habit_id, title, description, importance_level,
                    frequency_type, frequency_detail, preferred_time_start, preferred_time_end,
                    generates_type, generation_template, validation_rules,
                    linked_priority_id, linked_goal_id, category, tags, metadata)
                VALUES (:habit_id, :title, :description, :importance_level,
                    :frequency_type, CAST(:frequency_detail AS jsonb),
                    CAST(:pts AS time), CAST(:pte AS time),
                    :generates_type, CAST(:generation_template AS jsonb),
                    CAST(:validation_rules AS jsonb),
                    :linked_priority_id, :linked_goal_id, :category, :tags,
                    CAST(:metadata AS jsonb))
                RETURNING *
            """,
                {
                    "habit_id": habit_id,
                    "title": title,
                    "description": description,
                    "importance_level": importance_level,
                    "frequency_type": frequency_type,
                    "frequency_detail": json.dumps(frequency_detail or {}),
                    "pts": preferred_time_start,
                    "pte": preferred_time_end,
                    "generates_type": generates_type,
                    "generation_template": json.dumps(generation_template or {}),
                    "validation_rules": json.dumps(validation_rules or {"type": "manual"}),
                    "linked_priority_id": linked_priority_id,
                    "linked_goal_id": linked_goal_id,
                    "category": category,
                    "tags": tags or [],
                    "metadata": json.dumps(metadata or {}),
                },
            )  # type: ignore[return-value]

    async def get(self, habit_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM habits WHERE habit_id = :hid", {"hid": habit_id})

    async def list(
        self, *, status: str | None = None, category: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        conds: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if status:
            conds.append("status = :status")
            params["status"] = status
        if category:
            conds.append("category = :category")
            params["category"] = category
        where = " AND ".join(conds) if conds else "1=1"
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM habits WHERE {where}
                ORDER BY importance_level DESC, created_at DESC LIMIT :limit
            """,
                params,
            )

    async def update(self, habit_id: str, **fields: Any) -> dict[str, Any] | None:
        sets: list[str] = []
        params: dict[str, Any] = {"hid": habit_id}
        for k, v in fields.items():
            if v is not None:
                if k in ("frequency_detail", "generation_template", "validation_rules", "metadata"):
                    v = json.dumps(v)
                    sets.append(f"{k} = CAST(:{k} AS jsonb)")
                else:
                    sets.append(f"{k} = :{k}")
                params[k] = v
        if not sets:
            return await self.get(habit_id)
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE habits SET {", ".join(sets)}
                WHERE habit_id = :hid RETURNING *
            """,
                params,
            )

    async def delete(self, habit_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM habits WHERE habit_id = :hid RETURNING id", {"hid": habit_id})
            return r.fetchone() is not None

    # ── Occurrences ──

    async def create_occurrence(
        self,
        occurrence_id: str,
        habit_id: str,
        scheduled_date: str,
        *,
        scheduled_time_start: str | None = None,
        scheduled_time_end: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO habit_occurrences (occurrence_id, habit_id, scheduled_date,
                    scheduled_time_start, scheduled_time_end)
                VALUES (:oid, :hid, CAST(:sd AS date),
                    CAST(:sts AS time), CAST(:ste AS time))
                ON CONFLICT (habit_id, scheduled_date) DO NOTHING
                RETURNING *
            """,
                {
                    "oid": occurrence_id,
                    "hid": habit_id,
                    "sd": _date.fromisoformat(scheduled_date) if isinstance(scheduled_date, str) else scheduled_date,
                    "sts": _time.fromisoformat(scheduled_time_start)
                    if isinstance(scheduled_time_start, str)
                    else scheduled_time_start,
                    "ste": _time.fromisoformat(scheduled_time_end)
                    if isinstance(scheduled_time_end, str)
                    else scheduled_time_end,
                },
            )  # type: ignore[return-value]

    async def complete_occurrence(self, occurrence_id: str, *, notes: str | None = None) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE habit_occurrences
                SET status = 'completed', completed_at = NOW(), notes = COALESCE(:notes, notes)
                WHERE occurrence_id = :oid RETURNING *
            """,
                {"oid": occurrence_id, "notes": notes},
            )

    async def skip_occurrence(self, occurrence_id: str, *, reason: str | None = None) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE habit_occurrences
                SET status = 'skipped', skipped_at = NOW(), skip_reason = :reason
                WHERE occurrence_id = :oid RETURNING *
            """,
                {"oid": occurrence_id, "reason": reason},
            )

    async def get_due_today(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT ho.*, h.title as habit_title, h.importance_level
                FROM habit_occurrences ho
                JOIN habits h ON h.habit_id = ho.habit_id
                WHERE ho.scheduled_date = CURRENT_DATE AND ho.status = 'pending'
                ORDER BY h.importance_level DESC
            """,
            )

    async def get_occurrences(self, habit_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM habit_occurrences
                WHERE habit_id = :hid
                ORDER BY scheduled_date DESC LIMIT :limit
            """,
                {"hid": habit_id, "limit": limit},
            )
