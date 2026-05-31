"""Tool History — query the tool execution audit log."""

from __future__ import annotations

import asyncio

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="recent|errors|stats")
    tool_name: str = Field(default="", description="Filter by tool name (for recent)")
    hours: int = Field(default=24, description="Lookback window in hours")
    limit: int = Field(default=30, description="Max results")


class Output(BaseModel):
    success: bool = True
    data: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class ToolHistoryTool(ScriptTool[Input, Output]):
    name = "tool_history"
    description = "Query the tool execution audit log"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.tool_logs import ToolLogsRepo

        repo = ToolLogsRepo()
        cmd = inp.command

        if cmd == "recent":
            rows = await repo.list_recent(
                tool_name=inp.tool_name or None,
                hours=inp.hours,
                limit=inp.limit,
            )
            return Output(success=True, data=rows, count=len(rows))

        if cmd == "errors":
            rows = await repo.list_errors(hours=inp.hours, limit=inp.limit)
            return Output(success=True, data=rows, count=len(rows))

        if cmd == "stats":
            rows = await repo.get_tool_stats(hours=inp.hours)
            return Output(success=True, data=rows, count=len(rows))

        return Output(success=False, error=f"Unknown command: {cmd}")


if __name__ == "__main__":
    ToolHistoryTool.run()
