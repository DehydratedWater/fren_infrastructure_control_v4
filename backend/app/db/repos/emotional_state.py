"""Emotional state repository — personality core mood tracking."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class EmotionalStateRepo:
    async def save(
        self,
        *,
        emotions: list[dict],
        description: str = "",
        chain_of_thought: str = "",
        mood_shift: str = "",
        response_guidance: str = "",
        private_thoughts: str = "",
        stimuli_summary: str = "",
        raw_xml: str = "",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO emotional_state
                    (emotions, description, chain_of_thought, mood_shift,
                     response_guidance, private_thoughts, stimuli_summary, raw_xml)
                VALUES
                    (CAST(:emotions AS jsonb), :description, :chain_of_thought, :mood_shift,
                     :response_guidance, :private_thoughts, :stimuli_summary, :raw_xml)
                RETURNING *
                """,
                {
                    "emotions": json.dumps(emotions),
                    "description": description,
                    "chain_of_thought": chain_of_thought,
                    "mood_shift": mood_shift,
                    "response_guidance": response_guidance,
                    "private_thoughts": private_thoughts,
                    "stimuli_summary": stimuli_summary[:500],
                    "raw_xml": raw_xml,
                },
            )  # type: ignore[return-value]

    async def get_current(self) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM emotional_state ORDER BY created_at DESC LIMIT 1",
                {},
            )

    async def get_history(self, *, hours: int = 24, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT id, emotions, description, mood_shift, stimuli_summary, created_at
                FROM emotional_state
                WHERE created_at > NOW() - make_interval(hours => :hours)
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"hours": hours, "limit": limit},
            )

    async def get_aggregates(self, *, period: str = "daily", days: int = 7) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM emotional_state_aggregates
                WHERE period = :period
                  AND period_start > NOW() - make_interval(days => :days)
                ORDER BY period_start DESC
                """,
                {"period": period, "days": days},
            )

    async def update_aggregates(self) -> None:
        """Compute hourly aggregate for the current hour."""
        async with get_async_session() as s:
            rows = await fetch_all(
                s,
                """
                SELECT emotions FROM emotional_state
                WHERE created_at > date_trunc('hour', NOW())
                """,
                {},
            )
            if not rows:
                return

            # Count emotions across all evaluations this hour
            counts: dict[str, int] = {}
            intensities: dict[str, list[float]] = {}
            for row in rows:
                for emo in row.get("emotions") or []:
                    name = emo.get("name", "unknown")
                    intensity = float(emo.get("intensity", 0.5))
                    counts[name] = counts.get(name, 0) + 1
                    intensities.setdefault(name, []).append(intensity)

            if not counts:
                return

            dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
            dominant_avg = sum(intensities[dominant]) / len(intensities[dominant])

            await execute_sql(
                s,
                """
                INSERT INTO emotional_state_aggregates
                    (period, period_start, dominant_emotion, dominant_intensity,
                     emotion_counts, evaluation_count)
                VALUES
                    ('hourly', date_trunc('hour', NOW()), :dominant, :intensity,
                     CAST(:counts AS jsonb), :eval_count)
                ON CONFLICT (period, period_start) DO UPDATE
                SET dominant_emotion = :dominant,
                    dominant_intensity = :intensity,
                    emotion_counts = CAST(:counts AS jsonb),
                    evaluation_count = :eval_count
                """,
                {
                    "dominant": dominant,
                    "intensity": dominant_avg,
                    "counts": json.dumps(counts),
                    "eval_count": len(rows),
                },
            )
