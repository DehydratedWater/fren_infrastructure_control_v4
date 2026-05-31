"""Nudge campaigns repository — strategic persuasion campaign tracking."""

from __future__ import annotations

import json
from datetime import datetime as _datetime
from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


def _to_datetime(v: str | _datetime) -> _datetime:
    if isinstance(v, _datetime):
        return v
    return _datetime.fromisoformat(v)


# Exponential moving average weight for responsiveness updates
_EMA_ALPHA = 0.3


class NudgeCampaignsRepo:
    async def create(
        self,
        campaign_id: str,
        target_type: str,
        target_id: str,
        target_title: str,
        current_tactic: str,
        *,
        tactic_rationale: str | None = None,
        escalation_level: int = 1,
        max_escalation: int = 4,
        min_interval_minutes: int = 60,
        max_nudges_per_day: int = 3,
        notes: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO nudge_campaigns (
                    campaign_id, target_type, target_id, target_title,
                    current_tactic, tactic_rationale, escalation_level,
                    max_escalation, min_interval_minutes, max_nudges_per_day, notes
                ) VALUES (
                    :cid, :ttype, :tid, :ttitle,
                    :tactic, :rationale, :esc,
                    :max_esc, :min_int, :max_npd, :notes
                )
                RETURNING *
                """,
                {
                    "cid": campaign_id,
                    "ttype": target_type,
                    "tid": target_id,
                    "ttitle": target_title,
                    "tactic": current_tactic,
                    "rationale": tactic_rationale,
                    "esc": escalation_level,
                    "max_esc": max_escalation,
                    "min_int": min_interval_minutes,
                    "max_npd": max_nudges_per_day,
                    "notes": notes,
                },
            )  # type: ignore[return-value]

    async def get(self, campaign_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM nudge_campaigns WHERE campaign_id = :cid",
                {"cid": campaign_id},
            )

    async def get_active(self, *, limit: int = 10) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM nudge_campaigns
                WHERE status = 'active'
                  AND (paused_until IS NULL OR paused_until < NOW())
                ORDER BY escalation_level DESC, last_nudge_at ASC NULLS FIRST
                LIMIT :limit
                """,
                {"limit": limit},
            )

    async def get_by_target(self, target_type: str, target_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                SELECT * FROM nudge_campaigns
                WHERE target_type = :ttype AND target_id = :tid AND status = 'active'
                """,
                {"ttype": target_type, "tid": target_id},
            )

    async def update(self, campaign_id: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"cid": campaign_id}
        for k, v in fields.items():
            if v is not None:
                if k == "tactic_history":
                    v = json.dumps(v)
                    sets.append(f"{k} = CAST(:{k} AS jsonb)")
                elif k in (
                    "started_at",
                    "paused_until",
                    "completed_at",
                    "last_nudge_at",
                    "last_reaction_at",
                    "last_action_at",
                ):
                    sets.append(f"{k} = CAST(:{k} AS timestamptz)")
                    v = _to_datetime(v)
                else:
                    sets.append(f"{k} = :{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE nudge_campaigns SET {", ".join(sets)}
                WHERE campaign_id = :cid RETURNING *
                """,
                params,
            )

    async def record_nudge(self, campaign_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE nudge_campaigns SET
                    total_nudges = total_nudges + 1,
                    nudges_today = nudges_today + 1,
                    last_nudge_at = NOW(),
                    updated_at = NOW()
                WHERE campaign_id = :cid RETURNING *
                """,
                {"cid": campaign_id},
            )

    async def record_reaction(self, campaign_id: str, reaction_type: str) -> dict[str, Any] | None:
        """Record user reaction: ignored | acknowledged | effective."""
        campaign = await self.get(campaign_id)
        if not campaign:
            return None

        # Update the correct counter
        counter_col = {
            "ignored": "nudges_ignored",
            "acknowledged": "nudges_acknowledged",
            "effective": "nudges_effective",
        }.get(reaction_type)
        if not counter_col:
            return None

        # Compute new responsiveness score via EMA
        old_score = campaign.get("responsiveness_score") or 0.5
        reaction_value = {"ignored": 0.0, "acknowledged": 0.3, "effective": 1.0}[reaction_type]
        new_score = round(_EMA_ALPHA * reaction_value + (1 - _EMA_ALPHA) * old_score, 4)

        # Best tactic tracking
        best_update = ""
        params: dict[str, Any] = {"cid": campaign_id, "new_score": new_score}
        if reaction_type == "effective":
            best_update = ", best_tactic = :best_tactic, last_action_at = NOW()"
            params["best_tactic"] = campaign.get("current_tactic")

        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE nudge_campaigns SET
                    {counter_col} = {counter_col} + 1,
                    last_reaction_at = NOW(),
                    responsiveness_score = :new_score,
                    updated_at = NOW()
                    {best_update}
                WHERE campaign_id = :cid RETURNING *
                """,
                params,
            )

    async def rotate_tactic(
        self,
        campaign_id: str,
        new_tactic: str,
        rationale: str,
        escalation_level: int,
    ) -> dict[str, Any] | None:
        campaign = await self.get(campaign_id)
        if not campaign:
            return None

        # Archive current tactic to history
        history = campaign.get("tactic_history") or []
        if isinstance(history, str):
            history = json.loads(history)

        history.append(
            {
                "tactic": campaign.get("current_tactic"),
                "escalation": campaign.get("escalation_level"),
                "started_at": str(campaign.get("started_at") if not history else campaign.get("updated_at")),
                "ended_at": _datetime.utcnow().isoformat(),
                "nudge_count": campaign.get("total_nudges", 0),
                "ignored": campaign.get("nudges_ignored", 0),
                "acknowledged": campaign.get("nudges_acknowledged", 0),
                "effective": campaign.get("nudges_effective", 0),
            }
        )

        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE nudge_campaigns SET
                    current_tactic = :tactic,
                    tactic_rationale = :rationale,
                    escalation_level = :esc,
                    tactic_history = CAST(:history AS jsonb),
                    updated_at = NOW()
                WHERE campaign_id = :cid RETURNING *
                """,
                {
                    "cid": campaign_id,
                    "tactic": new_tactic,
                    "rationale": rationale,
                    "esc": escalation_level,
                    "history": json.dumps(history),
                },
            )

    async def reset_daily_counts(self) -> None:
        async with get_async_session() as s:
            await fetch_one(
                s,
                "UPDATE nudge_campaigns SET nudges_today = 0 WHERE status = 'active' RETURNING id",
            )

    async def get_effectiveness_report(self, *, days: int = 30) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT
                    current_tactic as tactic,
                    COUNT(*) as campaign_count,
                    SUM(total_nudges) as total_nudges,
                    SUM(nudges_effective) as total_effective,
                    SUM(nudges_ignored) as total_ignored,
                    SUM(nudges_acknowledged) as total_acknowledged,
                    AVG(responsiveness_score) as avg_responsiveness,
                    AVG(CASE WHEN total_nudges > 0
                        THEN nudges_effective::float / total_nudges
                        ELSE 0 END) as avg_effectiveness_rate
                FROM nudge_campaigns
                WHERE created_at >= NOW() - CAST(:days || ' days' AS interval)
                GROUP BY current_tactic
                ORDER BY avg_effectiveness_rate DESC
                """,
                {"days": str(days)},
            )
