"""Strategy Tracker — daily strategies and influence attempts."""

from __future__ import annotations

import asyncio
from datetime import datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="create|get-today|get|log-attempt|get-attempts|update-outcome|list-recent")
    strategy_id: str = Field(default="", description="Strategy ID")
    attempt_id: str = Field(default="", description="Attempt ID")
    date: str = Field(default="", description="Date (YYYY-MM-DD)")
    focus_goals: str = Field(default="", description="Comma-separated goal IDs")
    time_blocks: str = Field(default="", description="JSON time blocks")
    notes: str = Field(default="", description="Notes")
    goal_id: str = Field(default="", description="Goal ID for attempt")
    influence_type: str = Field(default="", description="Type of influence attempt")
    message: str = Field(default="", description="Message sent")
    outcome: str = Field(default="", description="Actual outcome")
    score: float = Field(default=-1.0, description="Effectiveness score (0.0-1.0)")


class Output(BaseModel):
    success: bool = True
    strategy: dict = Field(default_factory=dict)
    strategies: list[dict] = Field(default_factory=list)
    attempt: dict = Field(default_factory=dict)
    attempts: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class StrategyTrackerTool(ScriptTool[Input, Output]):
    name = "strategy_tracker"
    description = "Manage daily strategies and influence attempt tracking"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.strategies import InfluenceRepo, StrategiesRepo

        strat_repo = StrategiesRepo()
        infl_repo = InfluenceRepo()

        if inp.command == "create":
            import json

            sid = f"strat_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            focus = [g.strip() for g in inp.focus_goals.split(",") if g.strip()] if inp.focus_goals else []
            blocks = json.loads(inp.time_blocks) if inp.time_blocks else []
            s = await strat_repo.create(
                strategy_id=sid,
                date=inp.date or datetime.now().strftime("%Y-%m-%d"),
                focus_goals=focus,
                time_blocks=blocks,
                notes=inp.notes or None,
            )
            return Output(success=True, strategy=s)

        if inp.command == "get-today":
            s = await strat_repo.get_today()
            if not s:
                return Output(success=False, error="No strategy for today")
            return Output(success=True, strategy=s)

        if inp.command == "get":
            s = await strat_repo.get(inp.strategy_id)
            if not s:
                return Output(success=False, error=f"Strategy not found: {inp.strategy_id}")
            return Output(success=True, strategy=s)

        if inp.command == "list-recent":
            ss = await strat_repo.list_recent()
            return Output(success=True, strategies=ss, count=len(ss))

        if inp.command == "log-attempt":
            aid = f"attempt_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            a = await infl_repo.create(
                attempt_id=aid,
                influence_type=inp.influence_type,
                message_sent=inp.message,
                date=inp.date or datetime.now().strftime("%Y-%m-%d"),
                sent_at=datetime.now().isoformat(),
                strategy_id=inp.strategy_id or None,
                goal_id=inp.goal_id or None,
            )
            return Output(success=True, attempt=a)

        if inp.command == "get-attempts":
            attempts = await infl_repo.list_recent()
            return Output(success=True, attempts=attempts, count=len(attempts))

        if inp.command == "update-outcome":
            if inp.score < 0:
                return Output(success=False, error="score required")
            a = await infl_repo.evaluate(inp.attempt_id, inp.outcome, inp.score)
            if not a:
                return Output(success=False, error=f"Attempt not found: {inp.attempt_id}")
            return Output(success=True, attempt=a)

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    StrategyTrackerTool.run()
