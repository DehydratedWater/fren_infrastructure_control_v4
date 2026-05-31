"""Todo Manager — CRUD for tasks with deadlines and goal alignment."""

from __future__ import annotations

import asyncio
from datetime import datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="add|get|list|update|complete|delete|get-today|get-overdue|get-week|get-upcoming|get-no-deadline"
    )
    todo_id: str = Field(default="", description="Todo ID")
    title: str = Field(default="", description="Todo title")
    description: str = Field(default="", description="Todo description")
    status: str = Field(default="", description="Filter by status")
    priority: str = Field(default="medium", description="Priority level")
    category: str = Field(default="personal", description="Category")
    deadline: str = Field(default="", description="Deadline (ISO or relative)")
    linked_goal_id: str = Field(default="", description="Linked goal ID")
    estimated_minutes: int = Field(default=0, description="Estimated time in minutes")
    tags: str = Field(default="", description="Comma-separated tags")
    url: str = Field(default="", description="URL associated with this todo")
    dependencies: str = Field(default="", description="Comma-separated todo_ids this task depends on")
    date: str = Field(default="", description="Date filter (YYYY-MM-DD)")


class Output(BaseModel):
    success: bool = True
    todo: dict = Field(default_factory=dict)
    todos: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class TodoManagerTool(ScriptTool[Input, Output]):
    name = "todo_manager"
    description = "Manage todos/tasks with deadlines, categories, and goal alignment"
    output_note = "If the user requested this, confirm the result via send_message.py"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.todos import TodosRepo

        repo = TodosRepo()

        if inp.command == "add":
            tid = f"todo_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            tags = [t.strip() for t in inp.tags.split(",") if t.strip()] if inp.tags else None
            deps = [int(d.strip()) for d in inp.dependencies.split(",") if d.strip()] if inp.dependencies else None
            todo = await repo.create(
                todo_id=tid,
                title=inp.title,
                date_str=inp.date or datetime.now().strftime("%Y-%m-%d"),
                description=inp.description or None,
                priority=inp.priority,
                category=inp.category,
                deadline=inp.deadline or None,
                linked_goal_id=inp.linked_goal_id or None,
                estimated_minutes=inp.estimated_minutes or None,
                tags=tags,
                url=inp.url or None,
                dependencies=deps,
            )
            return Output(success=True, todo=todo)

        if inp.command == "get":
            todo = await repo.get(inp.todo_id)
            if not todo:
                return Output(success=False, error=f"Todo not found: {inp.todo_id}")
            return Output(success=True, todo=todo)

        if inp.command == "list":
            todos = await repo.list(
                status=inp.status or None,
                category=inp.category if inp.category != "personal" else None,
                date=inp.date or None,
                linked_goal_id=inp.linked_goal_id or None,
            )
            return Output(success=True, todos=todos, count=len(todos))

        if inp.command == "update":
            fields = {}
            if inp.title:
                fields["title"] = inp.title
            if inp.description:
                fields["description"] = inp.description
            if inp.priority and inp.priority != "medium":
                fields["priority"] = inp.priority
            if inp.category and inp.category != "personal":
                fields["category"] = inp.category
            if inp.deadline:
                fields["deadline"] = inp.deadline
            if inp.status:
                fields["status"] = inp.status
            if inp.estimated_minutes:
                fields["estimated_minutes"] = inp.estimated_minutes
            if inp.date:
                fields["date"] = inp.date
            if inp.url:
                fields["url"] = inp.url
            if inp.dependencies:
                fields["dependencies"] = [int(d.strip()) for d in inp.dependencies.split(",") if d.strip()]
            todo = await repo.update(inp.todo_id, **fields)
            if not todo:
                return Output(success=False, error=f"Todo not found: {inp.todo_id}")
            return Output(success=True, todo=todo)

        if inp.command == "complete":
            todo = await repo.complete(inp.todo_id)
            if not todo:
                return Output(success=False, error=f"Todo not found: {inp.todo_id}")
            return Output(success=True, todo=todo)

        if inp.command == "delete":
            ok = await repo.delete(inp.todo_id)
            return Output(success=ok, error="" if ok else f"Todo not found: {inp.todo_id}")

        if inp.command == "get-today":
            todos = await repo.get_today()
            return Output(success=True, todos=todos, count=len(todos))

        if inp.command == "get-overdue":
            todos = await repo.get_overdue()
            return Output(success=True, todos=todos, count=len(todos))

        if inp.command == "get-week":
            todos = await repo.list(limit=100)
            return Output(success=True, todos=todos, count=len(todos))

        if inp.command == "get-upcoming":
            todos = await repo.get_upcoming()
            return Output(success=True, todos=todos, count=len(todos))

        if inp.command == "get-no-deadline":
            todos = await repo.get_no_deadline()
            return Output(success=True, todos=todos, count=len(todos))

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    TodoManagerTool.run()
