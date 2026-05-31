"""RP cross-summary manager — bidirectional summaries between main bot and RP bot.

Enables context sharing: the main bot writes summaries for the RP bot to read,
and vice versa, so each side has awareness of what happened on the other.
"""

from __future__ import annotations

import asyncio
import json

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="write|read-latest|read-recent")
    direction: str = Field(default="", description="rp_to_main|main_to_rp")
    chat_id: int = Field(default=0, description="Telegram chat_id (falls back to settings)")
    summary: str = Field(default="", description="Summary text (required for write)")
    context_window: str = Field(default="", description="Context window label")
    limit: int = Field(default=5, description="Row limit for read-recent")


class Output(BaseModel):
    success: bool = True
    summary_data: dict | None = None
    summaries: list[dict] = Field(default_factory=list)
    error: str = ""


class RPCrossSummaryTool(ScriptTool[Input, Output]):
    name = "rp_cross_summary"
    description = "Bidirectional summaries between main bot and RP bot"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.rp_cross_summary import CrossSummaryRepo

        cmd = inp.command
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

        if not inp.direction:
            return Output(success=False, error="direction is required (rp_to_main|main_to_rp)")
        if inp.direction not in ("rp_to_main", "main_to_rp"):
            return Output(success=False, error=f"Invalid direction: {inp.direction}")

        if cmd == "write":
            if not inp.summary:
                return Output(success=False, error="summary is required for write")
            row = await CrossSummaryRepo().write(
                inp.direction,
                chat_id,
                inp.summary,
                context_window=inp.context_window or None,
            )
            return Output(success=True, summary_data=_serialize(row))

        if cmd == "read-latest":
            row = await CrossSummaryRepo().read_latest(inp.direction, chat_id)
            if not row:
                return Output(success=True, summary_data=None)
            return Output(success=True, summary_data=_serialize(row))

        if cmd == "read-recent":
            rows = await CrossSummaryRepo().read_recent(inp.direction, chat_id, limit=inp.limit)
            return Output(success=True, summaries=[_serialize(r) for r in rows])

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
    RPCrossSummaryTool.run()
