"""Strategies and influence attempts repositories."""

from __future__ import annotations

import json
from datetime import date as _date
from datetime import datetime as _datetime
from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


def _to_date(v: str | _date) -> _date:
    """Convert string or date to datetime.date (asyncpg requires native types)."""
    if isinstance(v, _date):
        return v
    return _date.fromisoformat(v)


def _to_datetime(v: str | _datetime) -> _datetime:
    """Convert string or datetime to datetime.datetime (asyncpg requires native types)."""
    if isinstance(v, _datetime):
        return v
    return _datetime.fromisoformat(v)


class StrategiesRepo:
    async def create(
        self,
        strategy_id: str,
        date: str,
        *,
        focus_goals: list | None = None,
        time_blocks: list | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO daily_strategies (strategy_id, date, focus_goals, time_blocks, notes)
                VALUES (:strategy_id, CAST(:date AS date), CAST(:focus_goals AS jsonb), CAST(:time_blocks AS jsonb), :notes)
                RETURNING *
            """,
                {
                    "strategy_id": strategy_id,
                    "date": _to_date(date),
                    "focus_goals": json.dumps(focus_goals or []),
                    "time_blocks": json.dumps(time_blocks or []),
                    "notes": notes,
                },
            )  # type: ignore[return-value]

    async def get_today(self) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM daily_strategies WHERE date = CURRENT_DATE")

    async def get(self, strategy_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM daily_strategies WHERE strategy_id = :sid", {"sid": strategy_id})

    async def update(self, strategy_id: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"sid": strategy_id}
        for k, v in fields.items():
            if v is not None:
                if k in ("focus_goals", "time_blocks", "completion_summary"):
                    v = json.dumps(v)
                    sets.append(f"{k} = CAST(:{k} AS jsonb)")
                else:
                    sets.append(f"{k} = :{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE daily_strategies SET {", ".join(sets)}
                WHERE strategy_id = :sid RETURNING *
            """,
                params,
            )

    async def list_recent(self, *, limit: int = 14) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM daily_strategies ORDER BY date DESC LIMIT :limit
            """,
                {"limit": limit},
            )


class InfluenceRepo:
    async def create(
        self,
        attempt_id: str,
        influence_type: str,
        message_sent: str,
        date: str,
        sent_at: str,
        *,
        strategy_id: str | None = None,
        goal_id: str | None = None,
        assumptions: list | None = None,
        expected_outcome: str | None = None,
        campaign_id: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO influence_attempts (attempt_id, strategy_id, goal_id,
                    influence_type, message_sent, assumptions, expected_outcome,
                    date, sent_at, campaign_id)
                VALUES (:aid, :sid, :gid, :itype, :msg, CAST(:asmp AS jsonb),
                    :expected, CAST(:date AS date), CAST(:sent_at AS timestamptz), :campaign_id)
                RETURNING *
            """,
                {
                    "aid": attempt_id,
                    "sid": strategy_id,
                    "gid": goal_id,
                    "itype": influence_type,
                    "msg": message_sent,
                    "asmp": json.dumps(assumptions or []),
                    "expected": expected_outcome,
                    "date": _to_date(date),
                    "sent_at": _to_datetime(sent_at),
                    "campaign_id": campaign_id,
                },
            )  # type: ignore[return-value]

    async def evaluate(self, attempt_id: str, actual_outcome: str, effectiveness_score: float) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE influence_attempts
                SET actual_outcome = :out, effectiveness_score = :score, evaluated_at = NOW()
                WHERE attempt_id = :aid RETURNING *
            """,
                {"aid": attempt_id, "out": actual_outcome, "score": effectiveness_score},
            )

    async def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM influence_attempts ORDER BY sent_at DESC LIMIT :limit
            """,
                {"limit": limit},
            )
