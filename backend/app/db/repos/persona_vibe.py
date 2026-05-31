"""Persona vibe state + style event log repos.

VibeStateRepo maintains a continuous 5-weight palette blend per chat_id that
drifts (EMA-smoothed) after each turn. StyleEventsRepo logs rule-scorer
rewrites for dashboard tuning.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session

WEIGHT_KEYS = (
    "w_warm_snarky",
    "w_dry_ironic",
    "w_caring_edge",
    "w_playful_flirt",
    "w_debate_socratic",
)

DEFAULT_WEIGHTS = {
    "w_warm_snarky": 0.40,
    "w_dry_ironic": 0.15,
    "w_caring_edge": 0.15,
    "w_playful_flirt": 0.10,
    "w_debate_socratic": 0.20,
}


def _renormalize(weights: dict[str, float]) -> dict[str, float]:
    """Clamp each weight to [0.02, 0.85] then renormalize to sum=1.0."""
    clamped = {k: max(0.02, min(0.85, float(weights[k]))) for k in WEIGHT_KEYS}
    total = sum(clamped.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in clamped.items()}


# ───────────────────────── persona_vibe_state ─────────────────────────


class VibeStateRepo:
    async def get(self, chat_id: int) -> dict[str, Any]:
        """Fetch the vibe state for chat_id. Creates a default row if missing."""
        async with get_async_session() as s:
            row = await fetch_one(s, "SELECT * FROM persona_vibe_state WHERE chat_id = :cid", {"cid": chat_id})
            if row:
                return row
            # Auto-seed with defaults on first access.
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO persona_vibe_state (chat_id) VALUES (:cid)
                ON CONFLICT (chat_id) DO UPDATE SET chat_id = :cid
                RETURNING *
                """,
                {"cid": chat_id},
            )

    async def set_weights(self, chat_id: int, weights: dict[str, float]) -> dict[str, Any]:
        norm = _renormalize(weights)
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                UPDATE persona_vibe_state
                SET w_warm_snarky = :w1,
                    w_dry_ironic = :w2,
                    w_caring_edge = :w3,
                    w_playful_flirt = :w4,
                    w_debate_socratic = :w5,
                    updated_at = NOW()
                WHERE chat_id = :cid
                RETURNING *
                """,
                {
                    "cid": chat_id,
                    "w1": norm["w_warm_snarky"],
                    "w2": norm["w_dry_ironic"],
                    "w3": norm["w_caring_edge"],
                    "w4": norm["w_playful_flirt"],
                    "w5": norm["w_debate_socratic"],
                },
            )

    async def drift(
        self,
        chat_id: int,
        delta: dict[str, float],
        *,
        trigger: str,
        user_tone: str,
        ema: float = 0.75,
        axis_delta: float = 0.0,
        arousal_delta: float = 0.0,
        decay_half_life_min: float = 45.0,
    ) -> dict[str, Any]:
        """Apply EMA drift: new = ema*old + (1-ema)*(old+delta), renormalize.

        Before applying the new delta, weights decay toward DEFAULT_WEIGHTS
        based on time elapsed since `updated_at` (exponential decay with
        configurable half-life). Axes decay toward 0.

        `delta` maps weight keys → signed floats (e.g. {"w_debate_socratic": +0.08}).
        Unspecified keys drift only via renormalization.
        `axis_delta` shifts ironic_genuine_axis directly (bounded -1..+1).
        """
        current = await self.get(chat_id)

        # --- Time-based decay toward equilibrium ---
        updated_at = current.get("updated_at")
        if updated_at and decay_half_life_min > 0:
            if hasattr(updated_at, "timestamp"):
                elapsed_min = (datetime.now(UTC) - updated_at.astimezone(UTC)).total_seconds() / 60.0
            else:
                elapsed_min = 0.0
            if elapsed_min > 1.0:
                # decay_factor: 1.0 → no decay (just updated), 0.0 → fully returned to default
                decay_factor = math.exp(-0.693 * elapsed_min / decay_half_life_min)
                for key in WEIGHT_KEYS:
                    old = float(current[key])
                    default = DEFAULT_WEIGHTS[key]
                    current[key] = default + decay_factor * (old - default)
                # Axes decay toward 0
                current["ironic_genuine_axis"] = decay_factor * float(current["ironic_genuine_axis"])
                current["arousal_axis"] = decay_factor * float(current.get("arousal_axis") or 0.0)

        mix = 1.0 - ema
        target: dict[str, float] = {}
        for key in WEIGHT_KEYS:
            old = float(current[key])
            d = float(delta.get(key, 0.0))
            target[key] = ema * old + mix * (old + d)
        norm = _renormalize(target)

        new_axis = max(-1.0, min(1.0, float(current["ironic_genuine_axis"]) + axis_delta))
        current_arousal = float(current.get("arousal_axis") or 0.0)
        new_arousal = max(-1.0, min(1.0, current_arousal + arousal_delta))

        async with get_async_session() as s:
            result = await fetch_one(  # type: ignore[assignment]
                s,
                """
                UPDATE persona_vibe_state
                SET w_warm_snarky = :w1,
                    w_dry_ironic = :w2,
                    w_caring_edge = :w3,
                    w_playful_flirt = :w4,
                    w_debate_socratic = :w5,
                    ironic_genuine_axis = :axis,
                    arousal_axis = :arousal,
                    last_trigger = :trigger,
                    last_user_tone = :tone,
                    drift_count = drift_count + 1,
                    updated_at = NOW()
                WHERE chat_id = :cid
                RETURNING *
                """,
                {
                    "cid": chat_id,
                    "w1": norm["w_warm_snarky"],
                    "w2": norm["w_dry_ironic"],
                    "w3": norm["w_caring_edge"],
                    "w4": norm["w_playful_flirt"],
                    "w5": norm["w_debate_socratic"],
                    "axis": new_axis,
                    "arousal": new_arousal,
                    "trigger": trigger,
                    "tone": user_tone[:20] if user_tone else None,
                },
            )
            if result:
                # Append a time-series snapshot so the dashboard can render drift history.
                await execute_sql(
                    s,
                    """
                    INSERT INTO persona_vibe_history
                        (chat_id, w_warm_snarky, w_dry_ironic, w_caring_edge,
                         w_playful_flirt, w_debate_socratic, ironic_genuine_axis,
                         arousal_axis, trigger, user_tone, drift_count)
                    VALUES (:cid, :w1, :w2, :w3, :w4, :w5, :axis, :arousal,
                            :trigger, :tone, :count)
                    """,
                    {
                        "cid": chat_id,
                        "w1": norm["w_warm_snarky"],
                        "w2": norm["w_dry_ironic"],
                        "w3": norm["w_caring_edge"],
                        "w4": norm["w_playful_flirt"],
                        "w5": norm["w_debate_socratic"],
                        "axis": new_axis,
                        "arousal": new_arousal,
                        "trigger": trigger,
                        "tone": user_tone[:20] if user_tone else None,
                        "count": int(result["drift_count"]),
                    },
                )
            return result  # type: ignore[return-value]

    async def history(self, chat_id: int, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return the last N drift snapshots, oldest first for timeline rendering."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM (
                    SELECT * FROM persona_vibe_history
                    WHERE chat_id = :cid
                    ORDER BY recorded_at DESC
                    LIMIT :limit
                ) sub
                ORDER BY recorded_at ASC
                """,
                {"cid": chat_id, "limit": limit},
            )

    async def reset(self, chat_id: int) -> dict[str, Any]:
        """Reset weights + axis to defaults (keeps drift_count for debugging)."""
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                UPDATE persona_vibe_state
                SET w_warm_snarky = 0.40,
                    w_dry_ironic = 0.15,
                    w_caring_edge = 0.15,
                    w_playful_flirt = 0.10,
                    w_debate_socratic = 0.20,
                    ironic_genuine_axis = 0.0,
                    arousal_axis = 0.0,
                    last_trigger = 'reset',
                    updated_at = NOW()
                WHERE chat_id = :cid
                RETURNING *
                """,
                {"cid": chat_id},
            )


# ───────────────────────── persona_style_events ─────────────────────────


class StyleEventsRepo:
    async def log(
        self,
        chat_id: int,
        *,
        violation_type: str,
        details: str | None = None,
        before: str | None = None,
        after: str | None = None,
        enforced: bool = True,
    ) -> None:
        async with get_async_session() as s:
            await execute_sql(
                s,
                """
                INSERT INTO persona_style_events
                    (chat_id, violation_type, details, before_text, after_text, enforced)
                VALUES (:cid, :vtype, :details, :before, :after, :enforced)
                """,
                {
                    "cid": chat_id,
                    "vtype": violation_type[:40],
                    "details": details,
                    "before": before,
                    "after": after,
                    "enforced": enforced,
                },
            )

    async def log_many(self, chat_id: int, events: list[dict[str, Any]]) -> None:
        """Bulk insert. Each event dict needs: violation_type, details, before, after, enforced."""
        if not events:
            return
        async with get_async_session() as s:
            for e in events:
                await execute_sql(
                    s,
                    """
                    INSERT INTO persona_style_events
                        (chat_id, violation_type, details, before_text, after_text, enforced)
                    VALUES (:cid, :vtype, :details, :before, :after, :enforced)
                    """,
                    {
                        "cid": chat_id,
                        "vtype": str(e.get("violation_type", ""))[:40],
                        "details": e.get("details"),
                        "before": e.get("before"),
                        "after": e.get("after"),
                        "enforced": bool(e.get("enforced", True)),
                    },
                )

    async def list_recent(
        self,
        chat_id: int,
        *,
        limit: int = 50,
        violation_type: str | None = None,
    ) -> list[dict[str, Any]]:
        where = "WHERE chat_id = :cid"
        params: dict[str, Any] = {"cid": chat_id, "limit": limit}
        if violation_type:
            where += " AND violation_type = :vtype"
            params["vtype"] = violation_type
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"SELECT * FROM persona_style_events {where} ORDER BY created_at DESC LIMIT :limit",
                params,
            )

    async def count_by_type(self, chat_id: int, *, since_hours: int = 24) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT violation_type, COUNT(*) AS n
                FROM persona_style_events
                WHERE chat_id = :cid
                  AND created_at > NOW() - INTERVAL '1 hour' * :hours
                GROUP BY violation_type
                ORDER BY n DESC
                """,
                {"cid": chat_id, "hours": since_hours},
            )
