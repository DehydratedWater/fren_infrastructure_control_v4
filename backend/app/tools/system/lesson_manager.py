"""Lesson manager tool — manage agent lessons learned from past mistakes."""

from __future__ import annotations

import asyncio
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="add|list|remove|get|promote — add a lesson, list active, "
        "remove (deactivate) by ID, get by ID, promote situational to systemic"
    )
    lesson: str = Field(default="", description="The lesson text (for 'add')")
    lesson_type: str = Field(default="systemic", description="systemic (permanent) or situational (auto-expires)")
    category: str = Field(
        default="general",
        description="correction|tool_error|duplicate_action|task_management|communication|general",
    )
    source_pattern: str = Field(default="", description="What triggered this lesson")
    confidence: float = Field(default=0.8, description="Confidence 0.0-1.0")
    expires_hours: int = Field(default=0, description="Hours until expiry (0 = never, for situational)")
    lesson_id: int = Field(default=0, description="Lesson ID (for remove/get/promote)")


class Output(BaseModel):
    success: bool = True
    item: dict[str, Any] = Field(default_factory=dict)
    items: list[dict[str, Any]] = Field(default_factory=list)
    formatted: str = ""
    error: str = ""


class LessonManagerTool(ScriptTool[Input, Output]):
    name = "lesson_manager"
    description = "Manage agent lessons learned from past mistakes"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.agent_lessons import AgentLessonsRepo

        repo = AgentLessonsRepo()

        if inp.command == "add":
            if not inp.lesson:
                return Output(success=False, error="lesson text is required")
            item = await repo.add(
                lesson=inp.lesson,
                lesson_type=inp.lesson_type,
                category=inp.category,
                source_pattern=inp.source_pattern or None,
                confidence=inp.confidence,
                expires_hours=inp.expires_hours or None,
                created_by="user",
            )
            return Output(item=item)

        if inp.command == "list":
            items = await repo.list_active()
            formatted = await repo.format_lessons_prompt()
            return Output(items=items, formatted=formatted)

        if inp.command == "remove":
            if not inp.lesson_id:
                return Output(success=False, error="lesson_id is required")
            item = await repo.deactivate(inp.lesson_id)
            if not item:
                return Output(success=False, error=f"Lesson {inp.lesson_id} not found")
            return Output(item=item)

        if inp.command == "get":
            if not inp.lesson_id:
                return Output(success=False, error="lesson_id is required")
            item = await repo.get(inp.lesson_id)
            if not item:
                return Output(success=False, error=f"Lesson {inp.lesson_id} not found")
            return Output(item=item)

        if inp.command == "promote":
            if not inp.lesson_id:
                return Output(success=False, error="lesson_id is required")
            item = await repo.promote_to_systemic(inp.lesson_id)
            if not item:
                return Output(success=False, error=f"Lesson {inp.lesson_id} not found")
            return Output(item=item)

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    LessonManagerTool.run()
