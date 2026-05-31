"""RP story log manager — append entries, read history.

Agents use this to maintain the narrative log: dialogue, narration,
actions, and system events within an adventure.
"""

from __future__ import annotations

import asyncio
import json

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="append|get-recent|get-all|get-turn-count")
    adventure_id: int = Field(description="RP adventure ID")
    turn_number: int = Field(default=0, description="Turn number (required for append)")
    speaker: str = Field(default="", description="Speaker name (append)")
    content: str = Field(default="", description="Entry content (required for append)")
    entry_type: str = Field(default="dialogue", description="Entry type: dialogue|narration|action|system")
    metadata: str = Field(default="{}", description="JSON metadata string")
    limit: int = Field(default=20, description="Row limit for get-recent")


class Output(BaseModel):
    success: bool = True
    entry: dict | None = None
    entries: list[dict] = Field(default_factory=list)
    turn_count: int = 0
    error: str = ""


class RPStoryManagerTool(ScriptTool[Input, Output]):
    name = "rp_story_manager"
    description = "Manage RP story log — append entries, read history"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.rp_adventure import StoryLogRepo

        cmd = inp.command

        if cmd == "append":
            if not inp.content:
                return Output(success=False, error="content is required for append")
            row = await StoryLogRepo().append(
                inp.adventure_id,
                inp.turn_number,
                inp.content,
                speaker=inp.speaker or None,
                entry_type=inp.entry_type,
                metadata=inp.metadata,
            )
            return Output(success=True, entry=_serialize(row))

        if cmd == "get-recent":
            rows = await StoryLogRepo().get_recent(inp.adventure_id, limit=inp.limit)
            return Output(success=True, entries=[_serialize(r) for r in rows])

        if cmd == "get-all":
            rows = await StoryLogRepo().get_all(inp.adventure_id)
            return Output(success=True, entries=[_serialize(r) for r in rows])

        if cmd == "get-turn-count":
            count = await StoryLogRepo().get_turn_count(inp.adventure_id)
            return Output(success=True, turn_count=count)

        return Output(success=False, error=f"Unknown command: {cmd}")


def _serialize(row: dict | None) -> dict:
    """Coerce datetime/decimal fields for JSON output."""
    if not row:
        return {}
    out = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif isinstance(v, dict | list):
            out[k] = v
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


if __name__ == "__main__":
    RPStoryManagerTool.run()
