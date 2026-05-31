"""Cron history — query cron execution history and stats."""

import asyncio
from typing import Any, ClassVar

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="list|stats|log")
    limit: int = Field(default=50, description="Max records to return")
    date: str = Field(default="", description="Filter by date (YYYY-MM-DD)")
    mode: str = Field(default="", description="Filter by mode")
    execution_id: str = Field(default="", description="Execution ID for log command")
    start: str = Field(default="", description="Start timestamp for log")
    end: str = Field(default="", description="End timestamp for log")
    exit_code: int = Field(default=0, description="Exit code for log")
    log_file: str = Field(default="", description="Log file path")
    triggered_by: str = Field(default="cron", description="Trigger source")


class Output(BaseModel):
    success: bool = True
    executions: list[dict] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
    execution: dict = Field(default_factory=dict)
    error: str = ""


class CronHistoryTool(ScriptTool[Input, Output]):
    name: ClassVar[str] = "cron_history"
    description: ClassVar[str] = "Query cron execution history and statistics"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.cron import CronExecutionsRepo

        repo = CronExecutionsRepo()

        if inp.command == "list":
            return await self._list(repo, inp)
        if inp.command == "stats":
            return await self._stats(inp)
        if inp.command == "log":
            return await self._log(repo, inp)

        return Output(success=False, error=f"Unknown command: {inp.command}")

    async def _list(self, repo: Any, inp: Input) -> Output:
        from app.db.session import fetch_all, get_async_session

        query = "SELECT * FROM cron_executions WHERE 1=1"
        params: dict[str, Any] = {}
        if inp.date:
            query += " AND DATE(started_at) = :date::date"
            params["date"] = inp.date
        if inp.mode:
            query += " AND mode = :mode"
            params["mode"] = inp.mode
        query += " ORDER BY started_at DESC LIMIT :limit"
        params["limit"] = inp.limit

        async with get_async_session() as s:
            rows = await fetch_all(s, query, params)
        return Output(executions=rows)

    async def _stats(self, inp: Input) -> Output:
        from app.db.session import fetch_all, get_async_session

        query = """
            SELECT mode, COUNT(*) as total,
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                   AVG(duration_seconds) as avg_duration
            FROM cron_executions
        """
        params: dict[str, Any] = {}
        if inp.date:
            query += " WHERE DATE(started_at) = :date::date"
            params["date"] = inp.date
        query += " GROUP BY mode ORDER BY total DESC"

        async with get_async_session() as s:
            rows = await fetch_all(s, query, params)
        return Output(stats={"modes": rows})

    async def _log(self, repo: Any, inp: Input) -> Output:
        import uuid

        eid = inp.execution_id or str(uuid.uuid4())[:8]
        row = await repo.create(eid, inp.mode, inp.start, triggered_by=inp.triggered_by)
        if inp.end:
            status = "completed" if inp.exit_code == 0 else "failed"
            row = await repo.complete(eid, exit_code=inp.exit_code, status=status)
        return Output(execution=row or {})
