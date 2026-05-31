"""Agent notes tool — persistent key-value memory for agents."""

from __future__ import annotations

import asyncio
import json

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description=(
            "get-note|set-note|delete-note|list-notes|check-reminded|mark-reminded"
            "|mark-acknowledged|check-time-block|mark-time-block-notified"
            "|check-question|mark-question-asked|get-acknowledgments"
            "|check-user-said|cleanup-expired"
            "|read-scratchpad|write-scratchpad|append-scratchpad"
        )
    )
    key: str = Field(default="", description="Note key")
    value: str = Field(default="", description="JSON value string")
    prefix: str = Field(default="", description="Key prefix for listing")
    todo_id: str = Field(default="", description="Todo ID for reminder checks")
    block_id: str = Field(default="", description="Time block ID")
    topic: str = Field(default="", description="Topic for acknowledgments")
    message: str = Field(default="", description="Message text")
    hash: str = Field(default="", description="Question hash")
    text: str = Field(default="", description="Text content (scratchpad/question)")
    hours: float = Field(default=0, description="Hours for expiry/check")
    minutes: int = Field(default=30, description="Minutes for time block check")
    expires_hours: float = Field(default=24.0, description="Expiry in hours for set-note")


class Output(BaseModel):
    success: bool = True
    note: dict | None = None
    notes: list[dict] = Field(default_factory=list)
    count: int = 0
    was_reminded: bool = False
    was_notified: bool = False
    was_asked: bool = False
    found: bool = False
    text: str = ""
    written: bool = False
    appended: bool = False
    length: int = 0
    deleted: int = 0
    error: str = ""


SCRATCHPAD_KEY = "checker_scratchpad"
SCRATCHPAD_MAX = 2000


class AgentNotesTool(ScriptTool[Input, Output]):
    name = "agent_notes"
    description = "Persistent key-value memory for agents with TTL and scratchpad"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.agent_notes import AgentNotesRepo

        repo = AgentNotesRepo()

        if inp.command == "get-note":
            note = await repo.get(inp.key)
            if note:
                return Output(success=True, note=note, found=True)
            return Output(success=True, found=False)

        if inp.command == "set-note":
            try:
                val = json.loads(inp.value) if inp.value else {}
            except (json.JSONDecodeError, TypeError):
                val = inp.value
            note = await repo.set(inp.key, val, expires_hours=int(inp.expires_hours))
            return Output(success=True, note=note)

        if inp.command == "delete-note":
            deleted = await repo.delete(inp.key)
            return Output(success=deleted, found=deleted)

        if inp.command == "list-notes":
            notes = await repo.get_by_prefix(inp.prefix)
            return Output(success=True, notes=notes, count=len(notes))

        if inp.command == "check-reminded":
            hours = inp.hours if inp.hours > 0 else 2.0
            note = await repo.get(f"last_reminder:{inp.todo_id}")
            if not note:
                return Output(success=True, was_reminded=False)
            from datetime import UTC, datetime, timedelta

            updated = note.get("updated_at")
            if updated and isinstance(updated, datetime):
                was_reminded = (datetime.now(UTC) - updated.replace(tzinfo=UTC)) < timedelta(hours=hours)
            else:
                was_reminded = False
            return Output(success=True, was_reminded=was_reminded)

        if inp.command == "mark-reminded":
            note = await repo.set(
                f"last_reminder:{inp.todo_id}",
                {"message": inp.message},
                expires_hours=24,
            )
            return Output(success=True, note=note)

        if inp.command == "mark-acknowledged":
            note = await repo.set(f"user_said:{inp.topic}", {"message": inp.message}, expires_hours=4)
            return Output(success=True, note=note)

        if inp.command == "check-time-block":
            note = await repo.get(f"time_block:{inp.block_id}")
            if not note:
                return Output(success=True, was_notified=False)
            from datetime import UTC, datetime, timedelta

            updated = note.get("updated_at")
            if updated and isinstance(updated, datetime):
                was_notified = (datetime.now(UTC) - updated.replace(tzinfo=UTC)) < timedelta(minutes=inp.minutes)
            else:
                was_notified = False
            return Output(success=True, was_notified=was_notified)

        if inp.command == "mark-time-block-notified":
            note = await repo.set(f"time_block:{inp.block_id}", {}, expires_hours=2)
            return Output(success=True, note=note)

        if inp.command == "check-question":
            hours = inp.hours if inp.hours > 0 else 4.0
            note = await repo.get(f"question:{inp.hash}")
            if not note:
                return Output(success=True, was_asked=False)
            from datetime import UTC, datetime, timedelta

            updated = note.get("updated_at")
            if updated and isinstance(updated, datetime):
                was_asked = (datetime.now(UTC) - updated.replace(tzinfo=UTC)) < timedelta(hours=hours)
            else:
                was_asked = False
            return Output(success=True, was_asked=was_asked)

        if inp.command == "mark-question-asked":
            note = await repo.set(f"question:{inp.hash}", {"text": inp.text}, expires_hours=24)
            return Output(success=True, note=note)

        if inp.command == "get-acknowledgments":
            notes = await repo.get_by_prefix("user_said:")
            return Output(success=True, notes=notes, count=len(notes))

        if inp.command == "check-user-said":
            note = await repo.get(f"user_said:{inp.topic}")
            return Output(success=True, note=note, found=note is not None)

        if inp.command == "cleanup-expired":
            count = await repo.cleanup_expired()
            return Output(success=True, deleted=count)

        if inp.command == "read-scratchpad":
            note = await repo.get(SCRATCHPAD_KEY)
            txt = ""
            if note:
                val = note.get("note_value", {})
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        txt = val
                        val = None
                if isinstance(val, dict):
                    txt = val.get("text", "")
            return Output(success=True, text=txt, length=len(txt))

        if inp.command == "write-scratchpad":
            txt = inp.text
            if len(txt) > SCRATCHPAD_MAX:
                txt = txt[-SCRATCHPAD_MAX:]
            await repo.set(SCRATCHPAD_KEY, {"text": txt}, expires_hours=8760)
            return Output(success=True, written=True, length=len(txt))

        if inp.command == "append-scratchpad":
            note = await repo.get(SCRATCHPAD_KEY)
            existing = ""
            if note:
                val = note.get("note_value", {})
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        existing = val
                        val = None
                if isinstance(val, dict):
                    existing = val.get("text", "")
            combined = f"{existing}\n{inp.text}" if existing else inp.text
            if len(combined) > SCRATCHPAD_MAX:
                combined = combined[-SCRATCHPAD_MAX:]
            await repo.set(SCRATCHPAD_KEY, {"text": combined}, expires_hours=8760)
            return Output(success=True, appended=True, length=len(combined))

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    AgentNotesTool.run()
