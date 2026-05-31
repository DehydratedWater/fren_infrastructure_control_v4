"""Workflow master session manager tool — session state and history."""

from __future__ import annotations

import asyncio
import json

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="get-status|get-history|clear|save-message|get-or-create")
    role: str = Field(default="", description="Message role (user/assistant/system)")
    content: str = Field(default="", description="Message content")
    message_type: str = Field(default="message", description="Message type")
    metadata: str = Field(default="{}", description="JSON metadata")
    limit: int = Field(default=100, description="Max messages for history")


class Output(BaseModel):
    success: bool = True
    active: bool = False
    session_id: int | None = None
    status: str = ""
    messages: list[dict] = Field(default_factory=list)
    message_id: int | None = None
    cleared_sessions: int = 0
    created_at: str = ""
    updated_at: str = ""
    error: str = ""


class WmSessionManagerTool(ScriptTool[Input, Output]):
    name = "wm_session_manager"
    description = "Manage workflow master sessions and history"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.workflow_master import WorkflowMasterMessagesRepo, WorkflowMasterSessionsRepo

        sessions = WorkflowMasterSessionsRepo()
        messages = WorkflowMasterMessagesRepo()

        if inp.command == "get-status":
            session = await sessions.get_active()
            if not session:
                return Output(success=True, active=False)
            return Output(
                success=True,
                active=True,
                session_id=session["id"],
                status=session.get("status", ""),
                created_at=str(session.get("created_at", "")),
                updated_at=str(session.get("updated_at", "")),
            )

        if inp.command == "get-history":
            session = await sessions.get_active()
            if not session:
                return Output(success=True, active=False, messages=[])
            msgs = await messages.get_history(session["id"], limit=inp.limit)
            return Output(
                success=True,
                active=True,
                session_id=session["id"],
                messages=[
                    {
                        "role": m.get("role", ""),
                        "content": m.get("content", ""),
                        "message_type": m.get("message_type", ""),
                        "created_at": str(m.get("created_at", "")),
                    }
                    for m in msgs
                ],
            )

        if inp.command == "clear":
            count = await sessions.clear_active()
            return Output(success=True, cleared_sessions=count)

        if inp.command == "save-message":
            session = await sessions.get_or_create()
            try:
                meta = json.loads(inp.metadata) if inp.metadata else {}
            except json.JSONDecodeError:
                meta = {}
            msg = await messages.save(
                session_id=session["id"],
                role=inp.role,
                content=inp.content,
                message_type=inp.message_type,
                metadata=meta,
            )
            return Output(success=True, session_id=session["id"], message_id=msg.get("id"))

        if inp.command == "get-or-create":
            session = await sessions.get_or_create()
            return Output(
                success=True,
                active=True,
                session_id=session["id"],
                status=session.get("status", ""),
                created_at=str(session.get("created_at", "")),
            )

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    WmSessionManagerTool.run()
