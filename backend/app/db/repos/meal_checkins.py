"""Meal check-in repository — daily meal status tracking."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class MealCheckinsRepo:
    async def create(
        self,
        checkin_id: str,
        meal_date: date,
        meal_type: str,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO meal_checkins (checkin_id, date, meal_type)
                VALUES (:cid, :d, :mt)
                ON CONFLICT (date, meal_type) DO NOTHING
                RETURNING *
                """,
                {"cid": checkin_id, "d": meal_date, "mt": meal_type},
            ) or await fetch_one(
                s,
                "SELECT * FROM meal_checkins WHERE date = :d AND meal_type = :mt",
                {"d": meal_date, "mt": meal_type},
            )  # type: ignore[return-value]

    async def get_today(self) -> list[dict[str, Any]]:
        today = date.today()
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM meal_checkins WHERE date = :d ORDER BY id",
                {"d": today},
            )

    async def get_by_meal(self, meal_date: date, meal_type: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM meal_checkins WHERE date = :d AND meal_type = :mt",
                {"d": meal_date, "mt": meal_type},
            )

    async def update_status(self, meal_date: date, meal_type: str, status: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["status = :status", "updated_at = NOW()"]
        params: dict[str, Any] = {"d": meal_date, "mt": meal_type, "status": status}
        idx = 1
        for k, v in fields.items():
            if v is not None:
                pk = f"p{idx}"
                params[pk] = v
                sets.append(f"{k} = :{pk}")
                idx += 1
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE meal_checkins SET {', '.join(sets)} WHERE date = :d AND meal_type = :mt RETURNING *",
                params,
            )

    async def record_response(
        self,
        meal_date: date,
        meal_type: str,
        *,
        user_response: str = "",
        meal_source: str = "",
        location: str = "",
    ) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE meal_checkins SET
                    status = 'responded',
                    user_response = :resp,
                    meal_source = :src,
                    location = :loc,
                    responded_at = :now,
                    updated_at = NOW()
                WHERE date = :d AND meal_type = :mt RETURNING *
                """,
                {
                    "d": meal_date,
                    "mt": meal_type,
                    "resp": user_response,
                    "src": meal_source,
                    "loc": location,
                    "now": datetime.now(),
                },
            )

    async def get_history(self, *, days: int = 7) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM meal_checkins
                WHERE date >= CURRENT_DATE - CAST(:days AS integer)
                ORDER BY date DESC, id
                """,
                {"days": days},
            )

    async def ensure_today_entries(self) -> list[dict[str, Any]]:
        """Create pending entries for today's meals if they don't exist."""
        today = date.today()
        results = []
        for mt in ("breakfast", "lunch", "dinner"):
            cid = f"meal_{today.strftime('%Y%m%d')}_{mt}"
            row = await self.create(cid, today, mt)
            results.append(row)
        return results

    async def get_unanswered(self) -> list[dict[str, Any]]:
        today = date.today()
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM meal_checkins
                WHERE date = :d AND status IN ('pending', 'asked')
                ORDER BY id
                """,
                {"d": today},
            )
