"""User facts manager — manage user-defined personal facts."""

import asyncio
import uuid
from typing import ClassVar

from src import ScriptTool
from pydantic import BaseModel, Field

VALID_CATEGORIES = {"personality", "health", "schedule", "preferences", "work", "relationships"}


class Input(BaseModel):
    command: str = Field(description="add|list|delete|get-for-agents")
    category: str = Field(default="", description="Fact category")
    text: str = Field(default="", description="Fact text")
    fact_id: str = Field(default="", description="Fact ID for delete")


class Output(BaseModel):
    success: bool = True
    facts: list[dict] = Field(default_factory=list)
    fact: dict = Field(default_factory=dict)
    grouped: dict = Field(default_factory=dict)
    error: str = ""


class UserFactsManagerTool(ScriptTool[Input, Output]):
    name: ClassVar[str] = "user_facts_manager"
    description: ClassVar[str] = "Manage user-defined personal facts for agent personalization"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.user_facts import UserFactsRepo

        repo = UserFactsRepo()

        if inp.command == "add":
            if inp.category and inp.category not in VALID_CATEGORIES:
                return Output(success=False, error=f"Invalid category. Valid: {', '.join(sorted(VALID_CATEGORIES))}")
            fid = str(uuid.uuid4())[:8]
            row = await repo.add(fid, inp.category, inp.text)
            return Output(fact=row)

        if inp.command == "list":
            rows = await repo.get_all(category=inp.category or None)
            return Output(facts=rows)

        if inp.command == "delete":
            ok = await repo.delete(inp.fact_id)
            return Output(success=ok, error="" if ok else "Fact not found")

        if inp.command == "get-for-agents":
            grouped = await repo.get_formatted()
            return Output(grouped=grouped)

        return Output(success=False, error=f"Unknown command: {inp.command}")
