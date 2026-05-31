"""RP world state manager — set/get world aspects for an adventure.

Agents use this to track environmental state: weather, time of day,
location descriptions, political tensions, etc.
"""

from __future__ import annotations

import asyncio
import json

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="set|get|get-all")
    adventure_id: int = Field(description="RP adventure ID")
    aspect: str = Field(default="", description="World aspect key (required for set/get)")
    value: str = Field(default="", description="Aspect value (required for set)")


class Output(BaseModel):
    success: bool = True
    aspect_data: dict | None = None
    aspects: list[dict] = Field(default_factory=list)
    error: str = ""


class RPWorldManagerTool(ScriptTool[Input, Output]):
    name = "rp_world_manager"
    description = "Manage RP world state aspects"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.rp_adventure import WorldStateRepo

        cmd = inp.command

        if cmd == "set":
            if not inp.aspect:
                return Output(success=False, error="aspect is required for set")
            if not inp.value:
                return Output(success=False, error="value is required for set")
            row = await WorldStateRepo().set_aspect(inp.adventure_id, inp.aspect, inp.value)
            return Output(success=True, aspect_data=_serialize(row))

        if cmd == "get":
            if not inp.aspect:
                return Output(success=False, error="aspect is required for get")
            row = await WorldStateRepo().get_aspect(inp.adventure_id, inp.aspect)
            if not row:
                return Output(success=False, error=f"Aspect '{inp.aspect}' not found")
            return Output(success=True, aspect_data=_serialize(row))

        if cmd == "get-all":
            rows = await WorldStateRepo().get_all(inp.adventure_id)
            return Output(success=True, aspects=[_serialize(r) for r in rows])

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
    RPWorldManagerTool.run()
