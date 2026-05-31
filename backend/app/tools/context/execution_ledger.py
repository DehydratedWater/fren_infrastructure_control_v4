"""Execution ledger tool — DB-backed artifact store for inter-agent coordination.

Replaces thought_transfer's destructive file-based reads with non-destructive,
versioned, run-scoped artifacts. Multiple consumers can read the same artifact.

Commands:
    start-run       Create a new execution run, returns run_id
    write-artifact  Store a typed artifact (non-destructive, auto-versioned)
    read-artifact   Read latest version of artifact (marks consumed, does NOT delete)
    complete-run    Mark run completed/failed with optional contract check
    get-run-status  Check run state and list its artifacts
    supersede-run   Mark a run superseded by a newer run
    has-artifact    Check if an artifact type exists for a run
"""

from __future__ import annotations

import asyncio
import json

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="REQUIRED. One of: start-run, write-artifact, read-artifact, complete-run, get-run-status, supersede-run, has-artifact"
    )
    run_id: str = Field(default="", description="Execution run ID")
    interaction_mode: str = Field(default="", description="quick_chat|workflow|analysis|full_flow")
    domain: str = Field(default="", description="Domain: todo, goal, habit, food, research, etc.")
    owner: str = Field(default="", description="Agent that owns this run or produced the artifact")
    root_message_id: str = Field(default="", description="Telegram message_id that started this run")
    root_session_id: str = Field(default="", description="OpenCode session ID")
    artifact_type: str = Field(
        default="",
        description="resolved_context|routing_decision|thinking_output|analysis_brief|response_plan|delivery_result|background_job_result",
    )
    content: str = Field(default="", description="Artifact payload (string content)")
    payload: str = Field(default="", description="Artifact payload as JSON object")
    consumer: str = Field(default="", description="Agent consuming the artifact")
    status: str = Field(default="completed", description="Run status: completed|failed")
    contract_passed: str = Field(default="", description="true|false — did the run meet its contract?")
    superseded_by: str = Field(default="", description="run_id of the newer run")


class Output(BaseModel):
    success: bool = True
    run_id: str = ""
    artifact_id: str = ""
    artifact_type: str = ""
    content: str = ""
    payload: dict | None = None
    run: dict | None = None
    artifacts: list[dict] = Field(default_factory=list)
    exists: bool = False
    error: str = ""


class ExecutionLedgerTool(ScriptTool[Input, Output]):
    name = "execution_ledger"
    description = (
        "DB-backed artifact store for inter-agent coordination. "
        "Use start-run to begin tracking, write-artifact to store data, "
        "read-artifact to retrieve (non-destructive). "
        "Example: --command write-artifact --run_id {id} --artifact_type routing_decision --content '...' --owner my_agent"
    )
    stream_field = "content"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        repo = ExecutionLedgerRepo()
        cmd = inp.command

        if cmd == "start-run":
            if not inp.interaction_mode:
                return Output(success=False, error="--interaction_mode required")
            msg_id = int(inp.root_message_id) if inp.root_message_id else None
            run = await repo.start_run(
                interaction_mode=inp.interaction_mode,
                domain=inp.domain,
                owner=inp.owner,
                root_message_id=msg_id,
                root_session_id=inp.root_session_id,
            )
            return Output(success=True, run_id=run.get("run_id", ""), run=run)

        if cmd == "write-artifact":
            if not inp.run_id:
                return Output(success=False, error="--run_id required")
            if not inp.artifact_type:
                return Output(success=False, error="--artifact_type required")
            # Accept either --content (string) or --payload (JSON)
            if inp.payload:
                try:
                    payload_data = json.loads(inp.payload)
                except json.JSONDecodeError:
                    payload_data = {"content": inp.payload}
            elif inp.content:
                payload_data = {"content": inp.content}
            else:
                return Output(success=False, error="--content or --payload required")

            art = await repo.write_artifact(
                inp.run_id,
                inp.artifact_type,
                payload_data,
                producer=inp.owner,
            )
            return Output(
                success=True,
                artifact_id=art.get("artifact_id", ""),
                artifact_type=inp.artifact_type,
                run_id=inp.run_id,
            )

        if cmd == "read-artifact":
            if not inp.run_id:
                return Output(success=False, error="--run_id required")
            if not inp.artifact_type:
                return Output(success=False, error="--artifact_type required")

            art = await repo.read_artifact(
                inp.run_id,
                inp.artifact_type,
                consumer=inp.consumer,
            )
            if not art:
                return Output(
                    success=False,
                    error=f"Artifact not found: {inp.artifact_type} for run {inp.run_id}",
                )
            payload = art.get("payload", {})
            content = payload.get("content", "") if isinstance(payload, dict) else str(payload)
            return Output(
                success=True,
                artifact_id=art.get("artifact_id", ""),
                artifact_type=inp.artifact_type,
                run_id=inp.run_id,
                content=content,
                payload=payload if isinstance(payload, dict) else {"content": payload},
            )

        if cmd == "complete-run":
            if not inp.run_id:
                return Output(success=False, error="--run_id required")
            cp = None
            if inp.contract_passed:
                cp = inp.contract_passed.lower() == "true"
            run = await repo.complete_run(
                inp.run_id,
                status=inp.status,
                contract_passed=cp,
            )
            return Output(success=True, run_id=inp.run_id, run=run)

        if cmd == "get-run-status":
            if not inp.run_id:
                return Output(success=False, error="--run_id required")
            status = await repo.get_run_status(inp.run_id)
            if "error" in status:
                return Output(success=False, error=status["error"])
            return Output(
                success=True,
                run_id=inp.run_id,
                run=status,
                artifacts=status.get("artifacts", []),
            )

        if cmd == "supersede-run":
            if not inp.run_id:
                return Output(success=False, error="--run_id required")
            if not inp.superseded_by:
                return Output(success=False, error="--superseded_by required")
            await repo.supersede_run(inp.run_id, inp.superseded_by)
            return Output(success=True, run_id=inp.run_id)

        if cmd == "has-artifact":
            if not inp.run_id:
                return Output(success=False, error="--run_id required")
            if not inp.artifact_type:
                return Output(success=False, error="--artifact_type required")
            exists = await repo.has_artifact(inp.run_id, inp.artifact_type)
            return Output(success=True, run_id=inp.run_id, artifact_type=inp.artifact_type, exists=exists)

        return Output(success=False, error=f"Unknown command: {cmd}")


if __name__ == "__main__":
    ExecutionLedgerTool.run()
