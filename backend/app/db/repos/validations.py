"""Validations and conclusions repositories."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class ValidationsRepo:
    async def save(
        self,
        validation_id: str,
        attempt_id: str,
        approach_type: str,
        validated: bool,
        effectiveness: float,
        *,
        validation_category: str = "general",
        assumptions_tested: list | None = None,
        conditions_for_success: list | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO validations (validation_id, attempt_id, approach_type,
                    validated, effectiveness, date, validation_category,
                    assumptions_tested, conditions_for_success, notes)
                VALUES (:vid, :aid, :atype, :val, :eff, CURRENT_DATE, :vcat,
                    CAST(:asmp AS jsonb), CAST(:cond AS jsonb), :notes)
                ON CONFLICT (validation_id) DO UPDATE SET
                    effectiveness = :eff, notes = :notes
                RETURNING *
            """,
                {
                    "vid": validation_id,
                    "aid": attempt_id,
                    "atype": approach_type,
                    "val": validated,
                    "eff": effectiveness,
                    "vcat": validation_category,
                    "asmp": json.dumps(assumptions_tested or []),
                    "cond": json.dumps(conditions_for_success or []),
                    "notes": notes,
                },
            )  # type: ignore[return-value]

    async def get_by_date(self, date: str) -> list[dict[str, Any]]:
        from datetime import date as _date

        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM validations WHERE date = CAST(:d AS date) ORDER BY created_at DESC",
                {"d": _date.fromisoformat(date) if isinstance(date, str) else date},
            )

    async def get_last_30_days(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM validations
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY date DESC
            """,
            )


class ConclusionsRepo:
    async def save(
        self,
        conclusion_id: str,
        month: str,
        *,
        total_validations: int = 0,
        validated_count: int = 0,
        invalidated_count: int = 0,
        approach_stats: dict | None = None,
        most_effective_approaches: list | None = None,
        least_effective_approaches: list | None = None,
        successful_conditions: dict | None = None,
        recommendations: list | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO monthly_conclusions (conclusion_id, month, total_validations,
                    validated_count, invalidated_count, approach_stats,
                    most_effective_approaches, least_effective_approaches,
                    successful_conditions, recommendations)
                VALUES (:cid, :month, :tv, :vc, :ic, CAST(:as_ AS jsonb),
                    CAST(:me AS jsonb), CAST(:le AS jsonb),
                    CAST(:sc AS jsonb), CAST(:rec AS jsonb))
                ON CONFLICT (conclusion_id) DO UPDATE SET
                    total_validations = :tv, approach_stats = CAST(:as_ AS jsonb),
                    recommendations = CAST(:rec AS jsonb)
                RETURNING *
            """,
                {
                    "cid": conclusion_id,
                    "month": month,
                    "tv": total_validations,
                    "vc": validated_count,
                    "ic": invalidated_count,
                    "as_": json.dumps(approach_stats or {}),
                    "me": json.dumps(most_effective_approaches or []),
                    "le": json.dumps(least_effective_approaches or []),
                    "sc": json.dumps(successful_conditions or {}),
                    "rec": json.dumps(recommendations or []),
                },
            )  # type: ignore[return-value]

    async def get_all(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(s, "SELECT * FROM monthly_conclusions ORDER BY month DESC")

    async def get_latest(self) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM monthly_conclusions ORDER BY month DESC LIMIT 1")
