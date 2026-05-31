"""Daily routines repository — raw SQL, async.

Predefined recurring tasks per weekday with optional time-gated visibility.
Separate from habits: no streaks, just "did you do it today?"
"""

from __future__ import annotations

from datetime import date as _date
from datetime import time as _time
from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


def _to_time(val: str | _time | None) -> _time | None:
    """Convert HH:MM or HH:MM:SS string to time object (asyncpg needs native types)."""
    if val is None:
        return None
    if isinstance(val, _time):
        return val
    parts = str(val).split(":")
    return _time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)


def _local_now() -> tuple[_date, _time, int]:
    """Return (local_date, local_time, weekday_0based) in user timezone."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from app.settings import get_settings

    tz = ZoneInfo(get_settings().user_timezone)
    now = datetime.now(tz)
    return now.date(), now.time(), now.weekday()


class DailyRoutinesRepo:
    async def create(
        self,
        routine_id: str,
        title: str,
        *,
        description: str | None = None,
        weekdays: list[int] | None = None,
        visible_from: str | None = None,
        visible_until: str | None = None,
        sort_order: int = 0,
        status: str = "active",
        category: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO daily_routines
                    (routine_id, title, description, weekdays,
                     visible_from, visible_until, sort_order, status, category)
                VALUES (:rid, :title, :desc, CAST(:weekdays AS SMALLINT[]),
                        :vf, :vu, :sort, :status, :cat)
                RETURNING *
                """,
                {
                    "rid": routine_id,
                    "title": title,
                    "desc": description,
                    "weekdays": weekdays or [],
                    "vf": _to_time(visible_from),
                    "vu": _to_time(visible_until),
                    "sort": sort_order,
                    "status": status,
                    "cat": category,
                },
            )

    async def get(self, routine_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM daily_routines WHERE routine_id = :rid",
                {"rid": routine_id},
            )

    async def list(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where = "WHERE 1=1"
        params: dict[str, Any] = {"limit": limit}
        if status:
            where += " AND status = :status"
            params["status"] = status
        if category:
            where += " AND category = :cat"
            params["cat"] = category
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"SELECT * FROM daily_routines {where} ORDER BY sort_order, created_at LIMIT :limit",
                params,
            )

    async def update(self, routine_id: str, **fields: Any) -> dict[str, Any] | None:
        if not fields:
            return await self.get(routine_id)
        sets = []
        params: dict[str, Any] = {"rid": routine_id}
        for key, val in fields.items():
            if key == "weekdays":
                sets.append(f"weekdays = CAST(:p_{key} AS SMALLINT[])")
                params[f"p_{key}"] = val if val is not None else []
            elif key in ("visible_from", "visible_until"):
                sets.append(f"{key} = :p_{key}")
                params[f"p_{key}"] = _to_time(val)
            else:
                sets.append(f"{key} = :p_{key}")
                params[f"p_{key}"] = val
        sets.append("updated_at = NOW()")
        set_clause = ", ".join(sets)
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE daily_routines SET {set_clause} WHERE routine_id = :rid RETURNING *",
                params,
            )

    async def delete(self, routine_id: str) -> bool:
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                "DELETE FROM daily_routines WHERE routine_id = :rid RETURNING id",
                {"rid": routine_id},
            )
            return row is not None

    async def get_due_today(self) -> list[dict[str, Any]]:
        """Return active routines visible right now and not yet completed today."""
        local_date, local_time, weekday = _local_now()
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT r.*
                FROM daily_routines r
                LEFT JOIN daily_routine_completions c
                    ON c.routine_id = r.routine_id AND c.completed_date = :today
                WHERE r.status = 'active'
                  AND (r.weekdays = '{}' OR :dow = ANY(r.weekdays))
                  AND (r.visible_from IS NULL OR :now_time >= r.visible_from)
                  AND (r.visible_until IS NULL OR :now_time <= r.visible_until)
                  AND c.id IS NULL
                ORDER BY r.sort_order, r.created_at
                """,
                {"today": local_date, "dow": weekday, "now_time": local_time},
            )

    async def get_checklist(self) -> list[dict[str, Any]]:
        """Return all routines for today with completion status."""
        local_date, local_time, weekday = _local_now()
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT r.*,
                       c.completed_at IS NOT NULL AS completed,
                       c.completed_at,
                       c.notes AS completion_notes,
                       CASE
                           WHEN r.visible_from IS NOT NULL
                                AND :now_time < r.visible_from
                           THEN false
                           ELSE true
                       END AS currently_visible
                FROM daily_routines r
                LEFT JOIN daily_routine_completions c
                    ON c.routine_id = r.routine_id AND c.completed_date = :today
                WHERE r.status = 'active'
                  AND (r.weekdays = '{}' OR :dow = ANY(r.weekdays))
                ORDER BY r.sort_order, r.created_at
                """,
                {"today": local_date, "dow": weekday, "now_time": local_time},
            )

    async def complete(
        self,
        routine_id: str,
        *,
        date: _date | None = None,
        notes: str | None = None,
    ) -> dict[str, Any] | None:
        if date is None:
            date, _, _ = _local_now()
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO daily_routine_completions (routine_id, completed_date, notes)
                VALUES (:rid, :dt, :notes)
                ON CONFLICT (routine_id, completed_date)
                DO UPDATE SET completed_at = NOW(), notes = COALESCE(:notes, daily_routine_completions.notes)
                RETURNING *
                """,
                {"rid": routine_id, "dt": date, "notes": notes},
            )

    async def uncomplete(
        self,
        routine_id: str,
        *,
        date: _date | None = None,
    ) -> bool:
        if date is None:
            date, _, _ = _local_now()
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                """
                DELETE FROM daily_routine_completions
                WHERE routine_id = :rid AND completed_date = :dt
                RETURNING id
                """,
                {"rid": routine_id, "dt": date},
            )
            return row is not None

    async def get_completions(self, routine_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM daily_routine_completions
                WHERE routine_id = :rid
                ORDER BY completed_date DESC
                LIMIT :limit
                """,
                {"rid": routine_id, "limit": limit},
            )

    async def get_completion_stats(self, *, days: int = 7) -> list[dict[str, Any]]:
        """Per-day stats for the last N days: total scheduled vs completed."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                WITH dates AS (
                    SELECT generate_series(
                        CURRENT_DATE - CAST(:days AS integer) + 1,
                        CURRENT_DATE,
                        '1 day'::interval
                    )::date AS d
                ),
                scheduled AS (
                    SELECT dates.d,
                           r.routine_id
                    FROM dates
                    CROSS JOIN daily_routines r
                    WHERE r.status = 'active'
                      AND (r.weekdays = '{}' OR CAST(EXTRACT(ISODOW FROM dates.d) - 1 AS SMALLINT) = ANY(r.weekdays))
                ),
                completed AS (
                    SELECT c.completed_date, c.routine_id
                    FROM daily_routine_completions c
                    WHERE c.completed_date >= CURRENT_DATE - CAST(:days AS integer) + 1
                )
                SELECT s.d AS date,
                       COUNT(DISTINCT s.routine_id) AS total,
                       COUNT(DISTINCT c.routine_id) AS completed,
                       CASE WHEN COUNT(DISTINCT s.routine_id) > 0
                            THEN ROUND(COUNT(DISTINCT c.routine_id)::numeric / COUNT(DISTINCT s.routine_id) * 100)
                            ELSE 0
                       END AS percentage
                FROM scheduled s
                LEFT JOIN completed c ON c.completed_date = s.d AND c.routine_id = s.routine_id
                GROUP BY s.d
                ORDER BY s.d
                """,
                {"days": days},
            )
