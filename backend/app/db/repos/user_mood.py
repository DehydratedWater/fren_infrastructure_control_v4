"""User mood state + history repos.

UserMoodRepo maintains Twily's running estimate of the user's emotional state
across 5 independent dimensions (0-1 each, no sum-to-1 constraint).  Drifted
via EMA from the same tone classifier that drives Twily's palette blend.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session

MOOD_KEYS = ("energy", "valence", "stress", "engagement", "openness")

DEFAULT_MOODS: dict[str, float] = {
    "energy": 0.5,
    "valence": 0.5,
    "stress": 0.3,
    "engagement": 0.5,
    "openness": 0.5,
}

# Labels for the dimension furthest from its neutral center (0.5, except stress=0.3).
_MOOD_LABELS: dict[str, tuple[str, str]] = {
    "energy": ("low_energy", "high_energy"),
    "valence": ("negative", "positive"),
    "stress": ("calm", "stressed"),
    "engagement": ("disengaged", "engaged"),
    "openness": ("closed", "open"),
}


def _dominant_mood(state: dict[str, Any]) -> str:
    """Pick the dimension furthest from its default center."""
    best_key = "valence"
    best_dist = 0.0
    for key in MOOD_KEYS:
        val = float(state.get(key, DEFAULT_MOODS[key]))
        center = DEFAULT_MOODS[key]
        dist = abs(val - center)
        if dist > best_dist:
            best_dist = dist
            best_key = key
    val = float(state.get(best_key, DEFAULT_MOODS[best_key]))
    low_label, high_label = _MOOD_LABELS[best_key]
    return high_label if val >= DEFAULT_MOODS[best_key] else low_label


def _clamp_moods(moods: dict[str, float]) -> dict[str, float]:
    """Clamp each dimension to [0.0, 1.0]."""
    return {k: max(0.0, min(1.0, float(moods[k]))) for k in MOOD_KEYS}


class UserMoodRepo:
    async def latest(self) -> dict[str, Any] | None:
        """READ-ONLY: the most recently updated mood state row, or ``None``.

        Unlike ``get`` this never auto-seeds a row — safe for the dashboard,
        which must never write.
        """
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM user_mood_state ORDER BY updated_at DESC LIMIT 1",
                {},
            )

    async def get(self, chat_id: int) -> dict[str, Any]:
        """Fetch mood state for chat_id. Creates a default row if missing."""
        async with get_async_session() as s:
            row = await fetch_one(s, "SELECT * FROM user_mood_state WHERE chat_id = :cid", {"cid": chat_id})
            if row:
                return row
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO user_mood_state (chat_id) VALUES (:cid)
                ON CONFLICT (chat_id) DO UPDATE SET chat_id = :cid
                RETURNING *
                """,
                {"cid": chat_id},
            )

    async def drift(
        self,
        chat_id: int,
        delta: dict[str, float],
        *,
        trigger: str,
        ema: float = 0.40,
        decay_half_life_min: float = 60.0,
    ) -> dict[str, Any]:
        """Apply EMA drift: new = ema*old + (1-ema)*(old+delta), clamp [0,1].

        Time-based decay toward DEFAULT_MOODS before applying the new delta.
        """
        current = await self.get(chat_id)

        # --- Time-based decay toward defaults ---
        updated_at = current.get("updated_at")
        if updated_at and decay_half_life_min > 0:
            if hasattr(updated_at, "timestamp"):
                elapsed_min = (datetime.now(UTC) - updated_at.astimezone(UTC)).total_seconds() / 60.0
            else:
                elapsed_min = 0.0
            if elapsed_min > 1.0:
                decay_factor = math.exp(-0.693 * elapsed_min / decay_half_life_min)
                for key in MOOD_KEYS:
                    old = float(current[key])
                    default = DEFAULT_MOODS[key]
                    current[key] = default + decay_factor * (old - default)

        mix = 1.0 - ema
        target: dict[str, float] = {}
        for key in MOOD_KEYS:
            old = float(current[key])
            d = float(delta.get(key, 0.0))
            target[key] = ema * old + mix * (old + d)
        clamped = _clamp_moods(target)

        dominant = _dominant_mood(clamped)

        async with get_async_session() as s:
            result = await fetch_one(  # type: ignore[assignment]
                s,
                """
                UPDATE user_mood_state
                SET energy = :energy,
                    valence = :valence,
                    stress = :stress,
                    engagement = :engagement,
                    openness = :openness,
                    dominant_mood = :dominant,
                    last_trigger = :trigger,
                    drift_count = drift_count + 1,
                    updated_at = NOW()
                WHERE chat_id = :cid
                RETURNING *
                """,
                {
                    "cid": chat_id,
                    "energy": clamped["energy"],
                    "valence": clamped["valence"],
                    "stress": clamped["stress"],
                    "engagement": clamped["engagement"],
                    "openness": clamped["openness"],
                    "dominant": dominant,
                    "trigger": trigger,
                },
            )
            if result:
                await execute_sql(
                    s,
                    """
                    INSERT INTO user_mood_history
                        (chat_id, energy, valence, stress, engagement, openness,
                         dominant_mood, trigger, drift_count)
                    VALUES (:cid, :energy, :valence, :stress, :engagement, :openness,
                            :dominant, :trigger, :count)
                    """,
                    {
                        "cid": chat_id,
                        "energy": clamped["energy"],
                        "valence": clamped["valence"],
                        "stress": clamped["stress"],
                        "engagement": clamped["engagement"],
                        "openness": clamped["openness"],
                        "dominant": dominant,
                        "trigger": trigger,
                        "count": int(result["drift_count"]),
                    },
                )
            return result  # type: ignore[return-value]

    async def history(self, chat_id: int, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return the last N mood snapshots, oldest first for timeline rendering."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM (
                    SELECT * FROM user_mood_history
                    WHERE chat_id = :cid
                    ORDER BY recorded_at DESC
                    LIMIT :limit
                ) sub
                ORDER BY recorded_at ASC
                """,
                {"cid": chat_id, "limit": limit},
            )
