"""Cron Manager — manage cron executions and workflow scheduling."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="log-start|log-complete|list-recent|workflow-start|workflow-complete|workflow-recent"
    )
    execution_id: str = Field(default="", description="Execution ID")
    mode: str = Field(default="", description="Cron mode/agent name")
    workflow_id: str = Field(default="", description="Workflow ID")
    workflow_name: str = Field(default="", description="Workflow name")
    input_text: str = Field(default="", description="Input text for workflow")
    triggered_by: str = Field(default="cron", description="Trigger source: cron|manual|bot")
    exit_code: int = Field(default=0, description="Exit code (0=success)")
    status: str = Field(default="completed", description="Final status")
    output: str = Field(default="", description="Output text")
    error_text: str = Field(default="", description="Error text")
    limit: int = Field(default=20, description="Result limit")


class Output(BaseModel):
    success: bool = True
    execution: dict = Field(default_factory=dict)
    executions: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class CronManagerTool(ScriptTool[Input, Output]):
    name = "cron_manager"
    description = "Log and query cron and workflow executions"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.cron import CronExecutionsRepo, WorkflowExecutionsRepo

        cron_repo = CronExecutionsRepo()
        wf_repo = WorkflowExecutionsRepo()
        cmd = inp.command

        # ── Cron Executions ──
        if cmd == "log-start":
            eid = inp.execution_id or f"cron_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{id(inp) % 0xFFFF:04x}"
            e = await cron_repo.create(
                execution_id=eid,
                mode=inp.mode,
                started_at=datetime.now(UTC),
                triggered_by=inp.triggered_by,
            )
            return Output(success=True, execution=e)

        if cmd == "log-complete":
            e = await cron_repo.complete(
                inp.execution_id,
                exit_code=inp.exit_code,
                status=inp.status,
            )
            if not e:
                return Output(success=False, error=f"Execution not found: {inp.execution_id}")
            return Output(success=True, execution=e)

        if cmd == "list-recent":
            es = await cron_repo.list_recent(limit=inp.limit)
            return Output(success=True, executions=es, count=len(es))

        # ── Workflow Executions ──
        if cmd == "workflow-start":
            eid = inp.execution_id or f"wf_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{id(inp) % 0xFFFF:04x}"
            e = await wf_repo.create(
                execution_id=eid,
                workflow_id=inp.workflow_id,
                workflow_name=inp.workflow_name,
                started_at=datetime.now(UTC),
                input_text=inp.input_text or None,
                triggered_by=inp.triggered_by,
            )
            return Output(success=True, execution=e)

        if cmd == "workflow-complete":
            e = await wf_repo.complete(
                inp.execution_id,
                exit_code=inp.exit_code,
                output=inp.output or None,
                error=inp.error_text or None,
            )
            if not e:
                return Output(success=False, error=f"Workflow execution not found: {inp.execution_id}")
            return Output(success=True, execution=e)

        if cmd == "workflow-recent":
            es = await wf_repo.list_recent(limit=inp.limit)
            return Output(success=True, executions=es, count=len(es))

        return Output(success=False, error=f"Unknown command: {cmd}")


if __name__ == "__main__":
    CronManagerTool.run()
