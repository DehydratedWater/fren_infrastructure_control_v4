"""Briefing Preferences — manage daily briefing section toggles and instructions."""

from __future__ import annotations

import asyncio

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="list|get|enable|disable|set-instructions|add|remove|reset")
    section: str = Field(default="", description="Section name (e.g. goals, weather, news)")
    instructions: str = Field(default="", description="Custom instructions for a section")
    priority: int = Field(default=-1, description="Display priority (lower = shown first)")
    enabled: bool = Field(default=True, description="Whether section is enabled (for add)")


class Output(BaseModel):
    success: bool = True
    preference: dict = Field(default_factory=dict)
    preferences: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class BriefingPreferencesTool(ScriptTool[Input, Output]):
    name = "briefing_preferences"
    description = "Manage daily briefing section preferences — toggle, reorder, and customize sections"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.briefing_preferences import BriefingPreferencesRepo

        repo = BriefingPreferencesRepo()

        if inp.command == "list":
            prefs = await repo.list_all()
            return Output(success=True, preferences=prefs, count=len(prefs))

        if inp.command == "get":
            if not inp.section:
                return Output(success=False, error="section is required")
            p = await repo.get(inp.section)
            if not p:
                return Output(success=False, error=f"Section not found: {inp.section}")
            return Output(success=True, preference=p)

        if inp.command == "enable":
            if not inp.section:
                return Output(success=False, error="section is required")
            p = await repo.toggle(inp.section, True)
            if not p:
                return Output(success=False, error=f"Section not found: {inp.section}")
            return Output(success=True, preference=p)

        if inp.command == "disable":
            if not inp.section:
                return Output(success=False, error="section is required")
            p = await repo.toggle(inp.section, False)
            if not p:
                return Output(success=False, error=f"Section not found: {inp.section}")
            return Output(success=True, preference=p)

        if inp.command == "set-instructions":
            if not inp.section:
                return Output(success=False, error="section is required")
            p = await repo.update_instructions(inp.section, inp.instructions)
            if not p:
                return Output(success=False, error=f"Section not found: {inp.section}")
            return Output(success=True, preference=p)

        if inp.command == "add":
            if not inp.section:
                return Output(success=False, error="section is required")
            kwargs: dict = {"enabled": inp.enabled}
            if inp.instructions:
                kwargs["instructions"] = inp.instructions
            if inp.priority >= 0:
                kwargs["priority"] = inp.priority
            p = await repo.upsert(inp.section, **kwargs)
            return Output(success=True, preference=p or {})

        if inp.command == "remove":
            if not inp.section:
                return Output(success=False, error="section is required")
            ok = await repo.delete(inp.section)
            return Output(success=ok, error="" if ok else f"Section not found: {inp.section}")

        if inp.command == "reset":
            prefs = await repo.reset_defaults()
            return Output(success=True, preferences=prefs, count=len(prefs))

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    BriefingPreferencesTool.run()
