"""Cron Runner — manage cron executions and workflow scheduling.

Faithful v4 port of v3 ``fren/tools/system/cron_manager.py`` (the cron-execution
runner/logger). In v3 this was an ``open_agent_compiler.runtime.ScriptTool`` that
the scheduler and bot used to log the start/completion of cron and workflow
executions and to query recent runs. v4 has no open_agent_compiler dependency,
so it is ported as a plain async class preserving every command branch
(log-start, log-complete, list-recent, workflow-start, workflow-complete,
workflow-recent) and the execution-id generation EXACTLY.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

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


class CronManagerTool:
    """Log and query cron and workflow executions."""

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

    @classmethod
    def run(cls) -> None:
        """One-shot CLI entrypoint (parity with v3 ScriptTool.run())."""
        import argparse
        import json as _json

        parser = argparse.ArgumentParser(prog=cls.name, description=cls.description)
        parser.add_argument("--command", required=True)
        parser.add_argument("--execution-id", default="")
        parser.add_argument("--mode", default="")
        parser.add_argument("--workflow-id", default="")
        parser.add_argument("--workflow-name", default="")
        parser.add_argument("--input-text", default="")
        parser.add_argument("--triggered-by", default="cron")
        parser.add_argument("--exit-code", type=int, default=0)
        parser.add_argument("--status", default="completed")
        parser.add_argument("--output", default="")
        parser.add_argument("--error-text", default="")
        parser.add_argument("--limit", type=int, default=20)
        args = parser.parse_args()
        inp = Input(
            command=args.command,
            execution_id=args.execution_id,
            mode=args.mode,
            workflow_id=args.workflow_id,
            workflow_name=args.workflow_name,
            input_text=args.input_text,
            triggered_by=args.triggered_by,
            exit_code=args.exit_code,
            status=args.status,
            output=args.output,
            error_text=args.error_text,
            limit=args.limit,
        )
        out = cls().execute(inp)
        print(_json.dumps(out.model_dump(), default=str))


def main() -> None:
    CronManagerTool.run()


if __name__ == "__main__":
    main()
