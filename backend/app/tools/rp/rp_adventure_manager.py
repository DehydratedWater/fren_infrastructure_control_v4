"""RP adventure manager — create, get, list, update status/scene, increment turn.

Exposes AdventureRepo to agents via a single script tool.
"""

from __future__ import annotations

import asyncio
import json

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="create|get|get-active|list|update-status|update-scene|update-config|increment-turn"
    )
    chat_id: int = Field(default=0, description="Telegram chat_id (falls back to settings.chat_id)")
    adventure_id: int = Field(default=0, description="Adventure ID (required for get/update commands)")
    title: str = Field(default="", description="Adventure title (create)")
    setting: str = Field(default="", description="Adventure setting description (create)")
    genre: str = Field(default="fantasy", description="Genre (create)")
    tone: str = Field(default="narrative", description="Tone (create)")
    status: str = Field(default="", description="New status (update-status)")
    current_scene: str = Field(default="", description="Scene description (create/update-scene)")
    current_time: str = Field(default="", description="In-world time (update-config, stored as inworld_time)")
    current_date: str = Field(default="", description="In-world date (update-config, stored as inworld_date)")
    context_summary: str = Field(default="", description="Compressed summary of old story (update-config)")
    cot_mode: str = Field(
        default="", description="CoT framework: narrative_audit|minimal|character_focus|off (update-config)"
    )
    narrative_mode: str = Field(
        default="",
        description="Narrative mode: balanced|slice_of_reality|cinematic|dark_simulation|comedic (update-config)",
    )
    writing_style: str = Field(default="", description="Writing style preset key (update-config)")
    limit: int = Field(default=20, description="Row limit (list)")


class Output(BaseModel):
    success: bool = True
    adventure: dict | None = None
    adventures: list[dict] = Field(default_factory=list)
    error: str = ""


class RPAdventureManagerTool(ScriptTool[Input, Output]):
    name = "rp_adventure_manager"
    description = "Manage RP adventures — create, get, list, update status"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.rp_adventure import AdventureRepo

        cmd = inp.command
        chat_id = inp.chat_id
        if not chat_id:
            try:
                from app.settings import get_settings

                raw = get_settings().chat_id
                chat_id = int(raw) if raw else 0
            except Exception:
                chat_id = 0
        inp.chat_id = chat_id

        repo = AdventureRepo()

        if cmd == "create":
            if not inp.title:
                return Output(success=False, error="title is required for create")
            if not inp.setting:
                return Output(success=False, error="setting is required for create")
            if not chat_id:
                return Output(success=False, error="chat_id is required")
            row = await repo.create(
                chat_id,
                inp.title,
                inp.setting,
                genre=inp.genre,
                tone=inp.tone,
                current_scene=inp.current_scene or None,
            )
            return Output(success=True, adventure=_serialize(row))

        if cmd == "get":
            if not inp.adventure_id:
                return Output(success=False, error="adventure_id is required for get")
            row = await repo.get(inp.adventure_id)
            if not row:
                return Output(success=False, error=f"Adventure {inp.adventure_id} not found")
            return Output(success=True, adventure=_serialize(row))

        if cmd == "get-active":
            if not chat_id:
                return Output(success=False, error="chat_id is required")
            row = await repo.get_active(chat_id)
            if not row:
                return Output(success=False, error="No active adventure found")
            return Output(success=True, adventure=_serialize(row))

        if cmd == "list":
            if not chat_id:
                return Output(success=False, error="chat_id is required")
            rows = await repo.list_all(chat_id, limit=inp.limit)
            return Output(success=True, adventures=[_serialize(r) for r in rows])

        if cmd == "update-status":
            if not inp.adventure_id:
                return Output(success=False, error="adventure_id is required for update-status")
            if not inp.status:
                return Output(success=False, error="status is required for update-status")
            row = await repo.update_status(inp.adventure_id, inp.status)
            if not row:
                return Output(success=False, error=f"Adventure {inp.adventure_id} not found")
            return Output(success=True, adventure=_serialize(row))

        if cmd == "update-scene":
            if not inp.adventure_id:
                return Output(success=False, error="adventure_id is required for update-scene")
            if not inp.current_scene:
                return Output(success=False, error="current_scene is required for update-scene")
            row = await repo.update_scene(inp.adventure_id, inp.current_scene)
            if not row:
                return Output(success=False, error=f"Adventure {inp.adventure_id} not found")
            return Output(success=True, adventure=_serialize(row))

        if cmd == "increment-turn":
            if not inp.adventure_id:
                return Output(success=False, error="adventure_id is required for increment-turn")
            row = await repo.increment_turn(inp.adventure_id)
            if not row:
                return Output(success=False, error=f"Adventure {inp.adventure_id} not found")
            return Output(success=True, adventure=_serialize(row))

        if cmd == "update-config":
            if not inp.adventure_id:
                return Output(success=False, error="adventure_id is required for update-config")
            fields = {}
            if inp.current_time:
                fields["inworld_time"] = inp.current_time
            if inp.current_date:
                fields["inworld_date"] = inp.current_date
            if inp.current_scene:
                fields["current_scene"] = inp.current_scene
            if inp.context_summary:
                fields["context_summary"] = inp.context_summary
            if inp.cot_mode:
                fields["cot_mode"] = inp.cot_mode
            if inp.narrative_mode:
                fields["narrative_mode"] = inp.narrative_mode
            if inp.writing_style:
                fields["writing_style"] = inp.writing_style
            if not fields:
                return Output(success=False, error="No config fields provided")
            row = await repo.update_config(inp.adventure_id, **fields)
            if not row:
                return Output(success=False, error=f"Adventure {inp.adventure_id} not found")
            return Output(success=True, adventure=_serialize(row))

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
    RPAdventureManagerTool.run()
