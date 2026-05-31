"""User mood manager — read-only access to Twily's estimate of user emotional state.

Used by the dashboard and /vibe command to display user mood data.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="get-mood|history")
    chat_id: int = Field(default=0, description="Telegram chat_id")
    limit: int = Field(default=100, description="history: row limit")


class Output(BaseModel):
    success: bool = True
    state: dict | None = None
    history: list[dict] = Field(default_factory=list)
    error: str = ""


def _serialize(row: dict[str, Any]) -> dict[str, Any]:
    """Make a DB row JSON-safe (convert datetimes)."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


class UserMoodManagerTool(ScriptTool[Input, Output]):
    name = "user_mood_manager"
    description = "Read Twily's estimate of the user's emotional state"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.user_mood import UserMoodRepo

        chat_id = inp.chat_id
        if not chat_id:
            try:
                from app.settings import get_settings

                raw = get_settings().chat_id
                chat_id = int(raw) if raw else 0
            except Exception:
                chat_id = 0
        if not chat_id:
            return Output(success=False, error="chat_id is required (no default configured)")

        cmd = inp.command

        if cmd == "get-mood":
            state = await UserMoodRepo().get(chat_id)
            return Output(success=True, state=_serialize(state))

        if cmd == "history":
            rows = await UserMoodRepo().history(chat_id, limit=inp.limit)
            return Output(success=True, history=[_serialize(r) for r in rows])

        return Output(success=False, error=f"Unknown command: {cmd}")


if __name__ == "__main__":
    UserMoodManagerTool.run()
