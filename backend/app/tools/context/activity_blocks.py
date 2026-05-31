"""Activity blocks tool -- structured activity timeline for agents."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="get-recent|get-day|get-range")
    hours: int = Field(default=6, description="Hours to look back (for get-recent)")
    date: str = Field(default="", description="Date YYYY-MM-DD (for get-day)")
    start: str = Field(default="", description="Start datetime ISO (for get-range)")
    end: str = Field(default="", description="End datetime ISO (for get-range)")


class Output(BaseModel):
    success: bool = True
    blocks: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


def _format_block(b: dict) -> dict:
    """Simplify block for agent consumption."""
    return {
        "started_at": str(b["started_at"]),
        "ended_at": str(b["ended_at"]) if b.get("ended_at") else None,
        "activity_type": b["activity_type"],
        "title": b["title"],
        "description": b["description"],
        "application": b["application"],
        "project": b["project"],
        "environment": b.get("environment", {}),
        "health_snapshot": b.get("health_snapshot", {}),
        "tags": b.get("tags", []),
        "confidence": b["confidence"],
        "frozen": b.get("frozen_at") is not None,
    }


class ActivityBlocksTool(ScriptTool[Input, Output]):
    name = "activity_blocks"
    description = "Structured activity timeline with time-ranged blocks (coding, browsing, gaming, sleeping, etc.)"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.activity_blocks import ActivityBlocksRepo

        repo = ActivityBlocksRepo()

        if inp.command == "get-recent":
            rows = await repo.get_recent_blocks(hours=inp.hours)
            blocks = [_format_block(r) for r in rows]
            return Output(success=True, blocks=blocks, count=len(blocks))

        if inp.command == "get-day":
            if not inp.date:
                return Output(success=False, error="--date required (YYYY-MM-DD)")
            from datetime import date as date_cls

            d = date_cls.fromisoformat(inp.date)
            rows = await repo.get_all_blocks(d)
            blocks = [_format_block(r) for r in rows]
            return Output(success=True, blocks=blocks, count=len(blocks))

        if inp.command == "get-range":
            if not inp.start or not inp.end:
                return Output(success=False, error="--start and --end required (ISO datetime)")
            start = datetime.fromisoformat(inp.start).replace(tzinfo=UTC)
            end = datetime.fromisoformat(inp.end).replace(tzinfo=UTC)
            rows = await repo.get_range(start, end)
            blocks = [_format_block(r) for r in rows]
            return Output(success=True, blocks=blocks, count=len(blocks))

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    ActivityBlocksTool.run()
