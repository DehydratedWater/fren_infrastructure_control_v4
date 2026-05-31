"""RP character manager — CRUD for rp_characters + persona loading.

Agents use this to create, inspect, update, and list characters within an
adventure, as well as to load a formatted persona prompt for injection into
agent context.
"""

from __future__ import annotations

import asyncio
import json

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="create|get|list|list-brief|list-detailed|list-all|update|load-persona")
    adventure_id: int = Field(default=0, description="Adventure ID (required for create/list/list-all)")
    character_id: int = Field(default=0, description="Character ID (required for get/update/load-persona)")
    name: str = Field(default="", description="Character name")
    role: str = Field(default="npc", description="Role: pc|npc|companion|villain|etc.")
    personality: str = Field(default="", description="Personality description")
    background: str = Field(default="", description="Background / backstory")
    knowledge: str = Field(default="", description="What the character knows")
    appearance: str = Field(default="", description="Physical appearance")
    location: str = Field(default="", description="Current location")
    mood: str = Field(default="", description="Current mood")
    inventory: str = Field(default="", description="Inventory (free-form or JSON)")
    stats: str = Field(default="", description="Stats (free-form or JSON)")
    status: str = Field(default="", description="Status: active|dead|absent|etc.")
    hidden_layer: str = Field(default="", description="What the character truly feels beneath the surface")
    current_goal: str = Field(default="", description="Character's active short-term goal")
    pressure: str = Field(default="", description="Environmental/social pressure on the character")
    trust_map: str = Field(default="", description="Trust level per other character (JSON: {name: 0.0-1.0})")
    dialogue_color: str = Field(default="", description="Hex color for character's dialogue")
    current_outfit: str = Field(default="", description="What the character is currently wearing")


class Output(BaseModel):
    success: bool = True
    character: dict | None = None
    characters: list[dict] = Field(default_factory=list)
    persona: str = ""
    error: str = ""


class RPCharacterManagerTool(ScriptTool[Input, Output]):
    name = "rp_character_manager"
    description = "Manage RP characters — create, get, list, update, load-persona"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.rp_adventure import CharacterRepo

        repo = CharacterRepo()
        cmd = inp.command

        if cmd == "create":
            if not inp.adventure_id:
                return Output(success=False, error="adventure_id is required")
            if not inp.name:
                return Output(success=False, error="name is required")
            if not inp.personality:
                return Output(success=False, error="personality is required")
            row = await repo.create(
                inp.adventure_id,
                inp.name,
                inp.personality,
                role=inp.role,
                background=inp.background or None,
                knowledge=inp.knowledge or None,
                appearance=inp.appearance or None,
                location=inp.location or None,
            )
            return Output(success=True, character=_serialize(row))

        if cmd == "get":
            if not inp.character_id:
                return Output(success=False, error="character_id is required")
            row = await repo.get(inp.character_id)
            if not row:
                return Output(success=False, error=f"Character {inp.character_id} not found")
            return Output(success=True, character=_serialize(row))

        if cmd == "list":
            if not inp.adventure_id:
                return Output(success=False, error="adventure_id is required")
            rows = await repo.list_active(inp.adventure_id)
            return Output(success=True, characters=[_serialize(r) for r in rows])

        if cmd == "list-brief":
            if not inp.adventure_id:
                return Output(success=False, error="adventure_id is required")
            rows = await repo.list_active(inp.adventure_id)
            brief = [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "role": r.get("role", "npc"),
                    "location": r.get("location", ""),
                    "mood": r.get("mood", "neutral"),
                    "status": r.get("status", "active"),
                }
                for r in rows
            ]
            return Output(success=True, characters=brief)

        if cmd == "list-all":
            if not inp.adventure_id:
                return Output(success=False, error="adventure_id is required")
            rows = await repo.list_all(inp.adventure_id)
            return Output(success=True, characters=[_serialize(r) for r in rows])

        if cmd == "list-detailed":
            if not inp.adventure_id:
                return Output(success=False, error="adventure_id is required")
            rows = await repo.list_active(inp.adventure_id)
            detailed = []
            for r in rows:
                d = _serialize(r)
                # Format the NPC priority stack for easy reading
                stack = []
                if r.get("hidden_layer"):
                    stack.append(f"  Hidden: {r['hidden_layer']}")
                trust = r.get("trust_map", {})
                if trust and isinstance(trust, dict):
                    trust_strs = [f"{k}: {v}" for k, v in trust.items()]
                    stack.append(f"  Trust: {{{', '.join(trust_strs)}}}")
                if r.get("pressure"):
                    stack.append(f"  Pressure: {r['pressure']}")
                if r.get("current_goal"):
                    stack.append(f"  Goal: {r['current_goal']}")
                if stack:
                    d["priority_stack"] = "\n".join(stack)
                detailed.append(d)
            return Output(success=True, characters=detailed)

        if cmd == "update":
            if not inp.character_id:
                return Output(success=False, error="character_id is required")
            fields: dict[str, str] = {}
            for key in (
                "mood",
                "location",
                "inventory",
                "stats",
                "status",
                "knowledge",
                "personality",
                "background",
                "hidden_layer",
                "current_goal",
                "pressure",
                "trust_map",
                "dialogue_color",
                "current_outfit",
            ):
                val = getattr(inp, key)
                if val:
                    fields[key] = val
            row = await repo.update(inp.character_id, **fields)
            if not row:
                return Output(success=False, error=f"Character {inp.character_id} not found")
            return Output(success=True, character=_serialize(row))

        if cmd == "load-persona":
            if not inp.character_id:
                return Output(success=False, error="character_id is required")
            row = await repo.get(inp.character_id)
            if not row:
                return Output(success=False, error=f"Character {inp.character_id} not found")
            persona = _format_persona(row)
            return Output(success=True, character=_serialize(row), persona=persona)

        return Output(success=False, error=f"Unknown command: {cmd}")


def _format_persona(row: dict) -> str:
    """Build a formatted persona prompt from a character row."""
    lines = [f"## Character: {row['name']}"]
    lines.append(f"Role: {row.get('role', 'npc')}")
    if row.get("personality"):
        lines.append(f"Personality: {row['personality']}")
    if row.get("background"):
        lines.append(f"Background: {row['background']}")
    if row.get("knowledge"):
        lines.append(f"Knowledge: {row['knowledge']}")
    if row.get("appearance"):
        lines.append(f"Appearance: {row['appearance']}")
    if row.get("current_outfit"):
        lines.append(f"Current Outfit: {row['current_outfit']}")
    if row.get("location"):
        lines.append(f"Current Location: {row['location']}")
    if row.get("mood"):
        lines.append(f"Current Mood: {row['mood']}")
    if row.get("inventory"):
        lines.append(f"Inventory: {row['inventory']}")
    # NPC Priority Stack (Phase 4)
    if any(row.get(k) for k in ("hidden_layer", "trust_map", "current_goal", "pressure")):
        lines.append("")
        lines.append("NPC Priority Stack:")
        if row.get("hidden_layer"):
            lines.append(f"  Hidden Layer: {row['hidden_layer']}")
        trust = row.get("trust_map", {})
        if trust and isinstance(trust, dict):
            trust_strs = [f"{k}: {v}" for k, v in trust.items()]
            lines.append(f"  Trust Map: {{{', '.join(trust_strs)}}}")
        if row.get("pressure"):
            lines.append(f"  Pressure: {row['pressure']}")
        if row.get("current_goal"):
            lines.append(f"  Current Goal: {row['current_goal']}")
    if row.get("dialogue_color"):
        lines.append(f"Dialogue Color: {row['dialogue_color']}")
    return "\n".join(lines)


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
    RPCharacterManagerTool.run()
