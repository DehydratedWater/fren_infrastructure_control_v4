"""Ralf manager tool — CRUD for ralf_processes, ralf_stages, attempts, logs.

Exposes all Ralf workflow state operations to agents via a single script tool.
Used by the 5 Ralf agents (dispatcher, planner, plan_evaluator, executor, step_evaluator)
and by scripts/ralf_ping.py.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description=(
            "create-ralf|get-ralf|list-active|list-all|update-status|heartbeat"
            "|set-task-name|set-total-stages|set-current-stage|set-completed|set-failed|set-budget|total-attempts"
            "|create-stage|list-stages|get-stage|update-stage-status|update-stage-criteria|delete-stages"
            "|create-attempt|get-latest-attempt|get-attempt|count-attempts"
            "|update-attempt-outcome|set-attempt-session"
            "|log-entry|list-logs|list-stage-logs"
            "|kv-set|kv-get|kv-list|kv-delete"
            "|lock-acquire|lock-release|lock-get|lock-list-held"
            "|add-amendment|list-amendments|count-unread-amendments|mark-amendments-read"
        )
    )
    ralf_id: str = Field(default="", description="Ralf process identifier")
    user_request: str = Field(default="", description="Original user task description")
    content_class: str = Field(default="public", description="public|nsfw|secret")
    model: str = Field(default="", description="Model variant label for subagent invocations")
    task_name: str = Field(default="", description="Short human task name")
    status: str = Field(default="", description="Status for update-status")
    last_error: str = Field(default="", description="Error message")
    stuck_reason: str = Field(default="", description="Reason ralf is stuck/failed")
    current_stage: int = Field(default=0, description="Current stage number (1-indexed)")
    total_stages: int = Field(default=0, description="Total number of stages")
    max_total_attempts: int = Field(
        default=0, description="Budget: max executor attempts across all stages (0=unchanged)"
    )
    deadline_at: str = Field(default="", description="Budget: ISO 8601 deadline timestamp")

    stage_number: int = Field(default=0, description="Stage number (1-indexed)")
    stage_name: str = Field(default="", description="Short stage name")
    goal: str = Field(default="", description="Stage goal")
    finalization_criteria: str = Field(default="", description="Testable completion predicate")
    notes: str = Field(default="", description="Stage notes")
    stage_status: str = Field(default="", description="Stage status update")

    attempt_number: int = Field(default=0, description="Attempt number (1-indexed)")
    outcome: str = Field(default="", description="in_progress|awaiting_eval|approved|retry|impossible")
    evaluator_verdict: str = Field(default="", description="Evaluator verdict (short)")
    evaluator_notes: str = Field(default="", description="Evaluator notes (longer)")
    session_id: str = Field(default="", description="Opencode session id of executor run")

    log_type: str = Field(default="", description="idea|problem|approach|tool_result|progress")
    log_entry: str = Field(default="", description="Log text content")

    resource_key: str = Field(default="", description="Lock resource key (e.g. comfyui_instance_0)")
    ttl_seconds: int = Field(default=600, description="Lock TTL in seconds")
    key: str = Field(default="", description="KV store key")
    value: str = Field(default="", description="KV store value")
    value_type: str = Field(default="text", description="KV value type: text|json|int|float|bool")
    explanation: str = Field(default="", description="KV store explanation — what this key represents")
    created_by: str = Field(default="", description="Agent name that set the KV entry")

    limit: int = Field(default=50, description="List limit")

    note: str = Field(default="", description="Amendment note text (soft refinement from user)")
    unread_only: bool = Field(default=False, description="For list-amendments: only return unread")
    up_to_id: int = Field(default=0, description="For mark-amendments-read: cap (0=all unread)")


class Output(BaseModel):
    success: bool = True
    ralf_id: str = ""
    process: dict | None = None
    processes: list[dict] = Field(default_factory=list)
    stage: dict | None = None
    stages: list[dict] = Field(default_factory=list)
    attempt: dict | None = None
    logs: list[dict] = Field(default_factory=list)
    kv: dict | None = None
    kvs: list[dict] = Field(default_factory=list)
    lock: dict | None = None
    locks: list[dict] = Field(default_factory=list)
    amendment: dict | None = None
    amendments: list[dict] = Field(default_factory=list)
    acquired: bool = False
    count: int = 0
    error: str = ""


class RalfManagerTool(ScriptTool[Input, Output]):
    name: ClassVar[str] = "ralf_manager"
    description: ClassVar[str] = "Manage Ralf multi-stage workflow state in the database"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.session import set_null_pool

        set_null_pool(enabled=True)

        from app.db.repos.ralf import (
            RalfAmendmentsRepo,
            RalfKVRepo,
            RalfLocksRepo,
            RalfProcessesRepo,
            RalfStagesRepo,
            RalfStepAttemptsRepo,
            RalfStepLogsRepo,
        )

        processes = RalfProcessesRepo()
        stages = RalfStagesRepo()
        attempts = RalfStepAttemptsRepo()
        logs = RalfStepLogsRepo()
        kv = RalfKVRepo()
        locks = RalfLocksRepo()
        amendments = RalfAmendmentsRepo()

        try:
            # ── Processes ──
            if inp.command == "create-ralf":
                if not inp.user_request:
                    return Output(success=False, error="user_request required")
                from datetime import datetime

                deadline = None
                if inp.deadline_at:
                    try:
                        deadline = datetime.fromisoformat(inp.deadline_at)
                    except ValueError:
                        return Output(success=False, error=f"invalid deadline_at: {inp.deadline_at}")
                # Auto-default model to active chat model (user-facing) if not provided
                model_name = inp.model or None
                if not model_name:
                    try:
                        from app.telegram.state import get_model  # TODO(v4-port): app.telegram not yet ported to v4; get_model() resolves the active chat model. Guarded by the surrounding try/except so it degrades to model_name=None until ported.

                        model_name = get_model()
                    except Exception:
                        model_name = None
                row = await processes.create(
                    user_request=inp.user_request,
                    content_class=inp.content_class or "public",
                    model=model_name,
                    max_total_attempts=inp.max_total_attempts or 40,
                    deadline_at=deadline,
                )
                return Output(success=True, ralf_id=row.get("ralf_id", ""), process=row)

            if inp.command == "get-ralf":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                proc = await processes.get(inp.ralf_id)
                if not proc:
                    return Output(success=False, error=f"ralf {inp.ralf_id} not found")
                stages_list = await stages.list_for_ralf(inp.ralf_id)
                kv_list = await kv.list_for_ralf(inp.ralf_id)
                return Output(success=True, process=proc, stages=stages_list, kvs=kv_list)

            if inp.command == "list-active":
                procs = await processes.list_active()
                return Output(success=True, processes=procs, count=len(procs))

            if inp.command == "list-all":
                procs = await processes.list_all(limit=inp.limit)
                return Output(success=True, processes=procs, count=len(procs))

            if inp.command == "update-status":
                if not inp.ralf_id or not inp.status:
                    return Output(success=False, error="ralf_id and status required")
                row = await processes.update_status(
                    inp.ralf_id,
                    inp.status,
                    last_error=inp.last_error or None,
                    stuck_reason=inp.stuck_reason or None,
                )
                return Output(success=True, process=row)

            if inp.command == "heartbeat":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                await processes.heartbeat(inp.ralf_id)
                return Output(success=True)

            if inp.command == "set-task-name":
                if not inp.ralf_id or not inp.task_name:
                    return Output(success=False, error="ralf_id and task_name required")
                await processes.set_task_name(inp.ralf_id, inp.task_name)
                return Output(success=True)

            if inp.command == "set-total-stages":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                await processes.set_total_stages(inp.ralf_id, inp.total_stages)
                return Output(success=True)

            if inp.command == "set-current-stage":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                await processes.set_current_stage(inp.ralf_id, inp.current_stage)
                return Output(success=True)

            if inp.command == "set-completed":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                await processes.set_completed(inp.ralf_id)
                return Output(success=True)

            if inp.command == "set-failed":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                await processes.set_failed(inp.ralf_id, inp.stuck_reason or "unspecified")
                return Output(success=True)

            if inp.command == "set-budget":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                from datetime import datetime

                deadline = None
                if inp.deadline_at:
                    try:
                        deadline = datetime.fromisoformat(inp.deadline_at)
                    except ValueError:
                        return Output(success=False, error=f"invalid deadline_at: {inp.deadline_at}")
                row = await processes.set_budget(
                    inp.ralf_id,
                    max_total_attempts=inp.max_total_attempts or None,
                    deadline_at=deadline,
                )
                return Output(success=True, process=row)

            if inp.command == "total-attempts":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                n = await processes.total_attempts(inp.ralf_id)
                return Output(success=True, count=n)

            # ── Stages ──
            if inp.command == "create-stage":
                if not inp.ralf_id or not inp.stage_name or not inp.goal or not inp.finalization_criteria:
                    return Output(
                        success=False,
                        error="ralf_id, stage_name, goal, finalization_criteria required",
                    )
                row = await stages.create(
                    ralf_id=inp.ralf_id,
                    stage_number=inp.stage_number,
                    stage_name=inp.stage_name,
                    goal=inp.goal,
                    finalization_criteria=inp.finalization_criteria,
                    notes=inp.notes,
                )
                return Output(success=True, stage=row)

            if inp.command == "list-stages":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                rows = await stages.list_for_ralf(inp.ralf_id)
                return Output(success=True, stages=rows, count=len(rows))

            if inp.command == "get-stage":
                if not inp.ralf_id or not inp.stage_number:
                    return Output(success=False, error="ralf_id and stage_number required")
                row = await stages.get_stage(inp.ralf_id, inp.stage_number)
                if not row:
                    return Output(success=False, error="stage not found")
                return Output(success=True, stage=row)

            if inp.command == "update-stage-status":
                if not inp.ralf_id or not inp.stage_number or not inp.stage_status:
                    return Output(success=False, error="ralf_id, stage_number, stage_status required")
                row = await stages.update_status(
                    inp.ralf_id,
                    inp.stage_number,
                    inp.stage_status,
                    notes=inp.notes or None,
                )
                return Output(success=True, stage=row)

            if inp.command == "update-stage-criteria":
                if not inp.ralf_id or not inp.stage_number:
                    return Output(success=False, error="ralf_id and stage_number required")
                row = await stages.update_criteria(
                    inp.ralf_id,
                    inp.stage_number,
                    goal=inp.goal or None,
                    finalization_criteria=inp.finalization_criteria or None,
                    notes=inp.notes or None,
                )
                return Output(success=True, stage=row)

            if inp.command == "delete-stages":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                await stages.delete_all_for_ralf(inp.ralf_id)
                return Output(success=True)

            # ── Attempts ──
            if inp.command == "create-attempt":
                if not inp.ralf_id or not inp.stage_number:
                    return Output(success=False, error="ralf_id and stage_number required")
                row = await attempts.create(ralf_id=inp.ralf_id, stage_number=inp.stage_number)
                return Output(success=True, attempt=row)

            if inp.command == "get-latest-attempt":
                if not inp.ralf_id or not inp.stage_number:
                    return Output(success=False, error="ralf_id and stage_number required")
                row = await attempts.get_latest(inp.ralf_id, inp.stage_number)
                if not row:
                    return Output(success=True, attempt=None)
                return Output(success=True, attempt=row)

            if inp.command == "get-attempt":
                if not inp.ralf_id or not inp.stage_number or not inp.attempt_number:
                    return Output(success=False, error="ralf_id, stage_number, attempt_number required")
                row = await attempts.get(inp.ralf_id, inp.stage_number, inp.attempt_number)
                if not row:
                    return Output(success=False, error="attempt not found")
                return Output(success=True, attempt=row)

            if inp.command == "count-attempts":
                if not inp.ralf_id or not inp.stage_number:
                    return Output(success=False, error="ralf_id and stage_number required")
                n = await attempts.count_attempts(inp.ralf_id, inp.stage_number)
                return Output(success=True, count=n)

            if inp.command == "update-attempt-outcome":
                if not inp.ralf_id or not inp.stage_number or not inp.attempt_number or not inp.outcome:
                    return Output(
                        success=False,
                        error="ralf_id, stage_number, attempt_number, outcome required",
                    )
                row = await attempts.update_outcome(
                    inp.ralf_id,
                    inp.stage_number,
                    inp.attempt_number,
                    inp.outcome,
                    evaluator_verdict=inp.evaluator_verdict or None,
                    evaluator_notes=inp.evaluator_notes or None,
                )
                return Output(success=True, attempt=row)

            if inp.command == "set-attempt-session":
                if not inp.ralf_id or not inp.stage_number or not inp.attempt_number or not inp.session_id:
                    return Output(
                        success=False,
                        error="ralf_id, stage_number, attempt_number, session_id required",
                    )
                await attempts.set_session_id(inp.ralf_id, inp.stage_number, inp.attempt_number, inp.session_id)
                return Output(success=True)

            # ── Logs ──
            if inp.command == "log-entry":
                if (
                    not inp.ralf_id
                    or not inp.stage_number
                    or not inp.attempt_number
                    or not inp.log_type
                    or not inp.log_entry
                ):
                    return Output(
                        success=False,
                        error="ralf_id, stage_number, attempt_number, log_type, log_entry required",
                    )
                row = await logs.append(
                    ralf_id=inp.ralf_id,
                    stage_number=inp.stage_number,
                    attempt_number=inp.attempt_number,
                    log_type=inp.log_type,
                    log_entry=inp.log_entry,
                )
                return Output(success=True, logs=[row])

            if inp.command == "list-logs":
                if not inp.ralf_id or not inp.stage_number or not inp.attempt_number:
                    return Output(success=False, error="ralf_id, stage_number, attempt_number required")
                rows = await logs.list_for_attempt(inp.ralf_id, inp.stage_number, inp.attempt_number)
                return Output(success=True, logs=rows, count=len(rows))

            if inp.command == "list-stage-logs":
                if not inp.ralf_id or not inp.stage_number:
                    return Output(success=False, error="ralf_id and stage_number required")
                rows = await logs.list_for_stage(inp.ralf_id, inp.stage_number)
                return Output(success=True, logs=rows, count=len(rows))

            # ── KV Store ──
            if inp.command == "kv-set":
                if not inp.ralf_id or not inp.key or not inp.value or not inp.explanation:
                    return Output(success=False, error="ralf_id, key, value, explanation required")
                row = await kv.set(
                    ralf_id=inp.ralf_id,
                    key=inp.key,
                    value=inp.value,
                    explanation=inp.explanation,
                    value_type=inp.value_type or "text",
                    created_by=inp.created_by,
                )
                return Output(success=True, kv=row)

            if inp.command == "kv-get":
                if not inp.ralf_id or not inp.key:
                    return Output(success=False, error="ralf_id and key required")
                row = await kv.get(inp.ralf_id, inp.key)
                if not row:
                    return Output(success=True, kv=None)
                return Output(success=True, kv=row)

            if inp.command == "kv-list":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                rows = await kv.list_for_ralf(inp.ralf_id)
                return Output(success=True, kvs=rows, count=len(rows))

            if inp.command == "kv-delete":
                if not inp.ralf_id or not inp.key:
                    return Output(success=False, error="ralf_id and key required")
                await kv.delete(inp.ralf_id, inp.key)
                return Output(success=True)

            # ── Locks ──
            if inp.command == "lock-acquire":
                if not inp.ralf_id or not inp.resource_key:
                    return Output(success=False, error="ralf_id and resource_key required")
                row = await locks.acquire(
                    resource_key=inp.resource_key,
                    ralf_id=inp.ralf_id,
                    ttl_seconds=inp.ttl_seconds or 600,
                    stage_number=inp.stage_number or None,
                    notes=inp.notes,
                )
                if row is None:
                    existing = await locks.get(inp.resource_key)
                    return Output(success=True, acquired=False, lock=existing)
                return Output(success=True, acquired=True, lock=row)

            if inp.command == "lock-release":
                if not inp.ralf_id or not inp.resource_key:
                    return Output(success=False, error="ralf_id and resource_key required")
                await locks.release(inp.resource_key, inp.ralf_id)
                return Output(success=True)

            if inp.command == "lock-get":
                if not inp.resource_key:
                    return Output(success=False, error="resource_key required")
                row = await locks.get(inp.resource_key)
                return Output(success=True, lock=row)

            if inp.command == "lock-list-held":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                rows = await locks.list_held_by(inp.ralf_id)
                return Output(success=True, locks=rows, count=len(rows))

            # ── Amendments (soft user refinements for a running ralf) ──
            if inp.command == "add-amendment":
                if not inp.ralf_id or not inp.note:
                    return Output(success=False, error="ralf_id and note required")
                stage_when = inp.current_stage or inp.stage_number or None
                row = await amendments.add(
                    ralf_id=inp.ralf_id,
                    note=inp.note,
                    stage_number_when_added=stage_when,
                )
                return Output(success=True, amendment=row)

            if inp.command == "list-amendments":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                rows = await amendments.list_for_ralf(inp.ralf_id, unread_only=inp.unread_only)
                return Output(success=True, amendments=rows, count=len(rows))

            if inp.command == "count-unread-amendments":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                n = await amendments.count_unread(inp.ralf_id)
                return Output(success=True, count=n)

            if inp.command == "mark-amendments-read":
                if not inp.ralf_id:
                    return Output(success=False, error="ralf_id required")
                stage_when = inp.current_stage or inp.stage_number or None
                n = await amendments.mark_read(
                    inp.ralf_id,
                    up_to_id=inp.up_to_id or None,
                    stage_number_when_read=stage_when,
                )
                return Output(success=True, count=n)

            return Output(success=False, error=f"Unknown command: {inp.command}")
        except Exception as e:
            return Output(success=False, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    RalfManagerTool.run()
