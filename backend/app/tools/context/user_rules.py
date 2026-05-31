"""User rules tool — manage persistent user directives for all agents."""

from __future__ import annotations

import asyncio
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="add|list|remove|get — add a new rule, list active rules, "
        "remove (deactivate) a rule by ID, or get a single rule"
    )
    rule: str = Field(default="", description="The rule text (for 'add' command)")
    category: str = Field(
        default="general",
        description="Rule category: general, nudging, persona, communication, health, work",
    )
    rule_id: int = Field(default=0, description="Rule ID (for 'remove' and 'get' commands)")


class Output(BaseModel):
    success: bool = True
    item: dict[str, Any] = Field(default_factory=dict)
    items: list[dict[str, Any]] = Field(default_factory=list)
    formatted: str = ""
    error: str = ""


class UserRulesTool(ScriptTool[Input, Output]):
    name = "user_rules"
    description = "Manage persistent user rules/directives that all agents must follow"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.user_rules import UserRulesRepo

        repo = UserRulesRepo()

        if inp.command == "add":
            if not inp.rule:
                return Output(success=False, error="rule text is required")
            item = await repo.add(rule=inp.rule, category=inp.category)
            return Output(item=item)

        if inp.command == "list":
            cat = inp.category if inp.category != "general" else None
            items = await repo.list_active(category=cat)
            formatted = await repo.format_rules_prompt()
            return Output(items=items, formatted=formatted)

        if inp.command == "remove":
            if not inp.rule_id:
                return Output(success=False, error="rule_id is required")
            item = await repo.deactivate(inp.rule_id)
            if not item:
                return Output(success=False, error=f"Rule {inp.rule_id} not found")
            return Output(item=item)

        if inp.command == "get":
            if not inp.rule_id:
                return Output(success=False, error="rule_id is required")
            item = await repo.get(inp.rule_id)
            if not item:
                return Output(success=False, error=f"Rule {inp.rule_id} not found")
            return Output(item=item)

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    UserRulesTool.run()
