"""Peek tool — read-only view into Twily's pending_thoughts queue.

Returns the top unconsumed thoughts by motivation_score without consuming.
Safe to expose to chat agents for drift cues.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import ClassVar

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    kinds: str = Field(
        default="opener,share,callback,contrarian",
        description="CSV of thought kinds to include",
    )
    limit: int = Field(default=3, description="Max thoughts to return")
    min_motivation: float = Field(default=0.4, description="Minimum motivation_score filter")


class Output(BaseModel):
    success: bool = True
    thoughts: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class PeekThoughtTool(ScriptTool[Input, Output]):
    name: ClassVar[str] = "peek_thought"
    description: ClassVar[str] = "Read-only peek at Twily's top pending_thoughts (no consume)"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.persona_memory import PendingThoughtsRepo
        from app.db.session import set_null_pool

        set_null_pool(enabled=True)
        repo = PendingThoughtsRepo()
        try:
            kinds = [k.strip() for k in inp.kinds.split(",") if k.strip()] or None
            rows = await repo.peek_top(kinds=kinds, limit=inp.limit, min_motivation=inp.min_motivation)
            display = [
                {
                    "content": r.get("content"),
                    "kind": r.get("kind"),
                    "motivation": round(float(r.get("motivation_score", 0)), 2),
                    "age_minutes": _age_minutes(r.get("created_at")),
                }
                for r in rows
            ]
            return Output(thoughts=display, count=len(display))
        except Exception as e:
            return Output(success=False, error=f"{type(e).__name__}: {e}")


def _age_minutes(ts) -> int | None:
    if ts is None:
        return None
    now = datetime.now(UTC)
    if getattr(ts, "tzinfo", None) is None:
        return None
    delta = now - ts.astimezone(UTC)
    return int(delta.total_seconds() // 60)
