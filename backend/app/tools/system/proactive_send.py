"""Proactive send tier-aware cooldown — shared send-budget floor.

Each proactive agent (meal_planner, nudge_strategist, winddown, evening_focus)
calls `check` before emitting guidance and `stamp` after. Lower-priority tiers
defer when a higher-priority send happened within their cooldown window, so
task reminders aren't buried under vibe/meal chatter.

Tier hierarchy (high → low): task > event > meal > vibe ≈ health.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from src import ScriptTool
from pydantic import BaseModel, Field

# Lower rank = higher priority. task is always allowed (cooldown=0).
_TIER_RANK: dict[str, int] = {
    "task": 0,
    "event": 1,
    "meal": 2,
    "vibe": 3,
    "health": 3,
}

# How long THIS tier defers when a higher-rank tier was sent recently (minutes).
# task=0 means it never defers; always sends.
_TIER_COOLDOWN_MIN: dict[str, int] = {
    "task": 0,
    "event": 15,
    "meal": 45,
    "vibe": 60,
    "health": 60,
}

VALID_TIERS = tuple(_TIER_RANK.keys())


class Input(BaseModel):
    command: str = Field(description="check|stamp|list")
    tier: str = Field(default="", description=f"Tier name: one of {VALID_TIERS}")


class Output(BaseModel):
    success: bool = True
    can_send: bool = True
    reason: str = ""
    blocked_by: str = ""
    minutes_since_block: float = -1
    cooldown_min: int = 0
    tier: str = ""
    sends: dict[str, str] = Field(default_factory=dict)
    error: str = ""


class ProactiveSendTool(ScriptTool[Input, Output]):
    name = "proactive_send"
    description = "Tier-aware proactive-send cooldown floor shared across proactive agents"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        if inp.command == "list":
            from app.db.repos.checker import CheckerStateRepo

            sends = await CheckerStateRepo().get_all_tier_sends()
            return Output(success=True, sends=sends)

        if inp.tier not in _TIER_RANK:
            return Output(
                success=False,
                error=f"Unknown tier '{inp.tier}'. Valid: {VALID_TIERS}",
            )

        if inp.command == "check":
            return await self._check(inp.tier)

        if inp.command == "stamp":
            return await self._stamp(inp.tier)

        return Output(success=False, error=f"Unknown command: {inp.command}")

    async def _check(self, tier: str) -> Output:
        from app.db.repos.checker import CheckerStateRepo

        my_rank = _TIER_RANK[tier]
        my_cd = _TIER_COOLDOWN_MIN[tier]
        if my_cd == 0:
            return Output(success=True, can_send=True, tier=tier, cooldown_min=0, reason="tier_never_defers")

        now = datetime.now(UTC)
        sends = await CheckerStateRepo().get_all_tier_sends()
        for other_tier, iso_ts in sends.items():
            other_rank = _TIER_RANK.get(other_tier)
            if other_rank is None or other_rank >= my_rank:
                continue
            try:
                ts = datetime.fromisoformat(iso_ts)
            except (TypeError, ValueError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            elapsed_min = (now - ts).total_seconds() / 60
            if elapsed_min < my_cd:
                return Output(
                    success=True,
                    can_send=False,
                    tier=tier,
                    cooldown_min=my_cd,
                    blocked_by=other_tier,
                    minutes_since_block=round(elapsed_min, 1),
                    reason=(
                        f"higher-priority tier '{other_tier}' sent "
                        f"{round(elapsed_min, 1)} min ago (cooldown={my_cd} min)"
                    ),
                )
        return Output(
            success=True, can_send=True, tier=tier, cooldown_min=my_cd, reason="no higher-priority block active"
        )

    async def _stamp(self, tier: str) -> Output:
        from app.db.repos.checker import CheckerStateRepo

        now = datetime.now(UTC)
        await CheckerStateRepo().record_tier_send(tier, now)
        return Output(success=True, tier=tier, reason=f"stamped at {now.isoformat()}")


if __name__ == "__main__":
    ProactiveSendTool.run()
