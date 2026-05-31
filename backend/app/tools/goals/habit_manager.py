"""Habit Manager — CRUD + occurrence tracking + streaks."""

from __future__ import annotations

import asyncio
from datetime import datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="add|get|list|update|delete|complete|skip|due-today|occurrences|stats|generate-today"
    )
    habit_id: str = Field(default="", description="Habit ID")
    title: str = Field(default="", description="Habit title")
    description: str = Field(default="", description="Habit description")
    frequency: str = Field(default="daily", description="Frequency: daily|weekly|monthly|custom")
    importance: int = Field(default=3, description="Importance level (1-5)")
    status: str = Field(default="", description="Status filter or new status")
    category: str = Field(default="", description="Category filter or value")
    date: str = Field(default="", description="Date (YYYY-MM-DD)")
    time_start: str = Field(default="", description="Preferred start time (HH:MM)")
    time_end: str = Field(default="", description="Preferred end time (HH:MM)")
    notes: str = Field(default="", description="Notes for completion")
    reason: str = Field(default="", description="Reason for skipping")
    linked_priority_id: str = Field(default="", description="Linked priority ID")
    linked_goal_id: str = Field(default="", description="Linked goal ID")


class Output(BaseModel):
    success: bool = True
    habit: dict = Field(default_factory=dict)
    habits: list[dict] = Field(default_factory=list)
    occurrences: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class HabitManagerTool(ScriptTool[Input, Output]):
    name = "habit_manager"
    description = "Manage recurring habits with occurrence tracking and streaks"
    output_note = "If the user requested this, confirm the result via send_message.py"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.habits import HabitsRepo

        repo = HabitsRepo()

        if inp.command == "add":
            hid = f"habit_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            habit = await repo.create(
                habit_id=hid,
                title=inp.title,
                frequency_type=inp.frequency,
                description=inp.description or None,
                importance_level=inp.importance,
                preferred_time_start=inp.time_start or None,
                preferred_time_end=inp.time_end or None,
                linked_priority_id=inp.linked_priority_id or None,
                linked_goal_id=inp.linked_goal_id or None,
                category=inp.category or None,
            )
            return Output(success=True, habit=habit)

        if inp.command == "get":
            habit = await repo.get(inp.habit_id)
            if not habit:
                return Output(success=False, error=f"Habit not found: {inp.habit_id}")
            return Output(success=True, habit=habit)

        if inp.command == "list":
            habits = await repo.list(
                status=inp.status or None,
                category=inp.category or None,
            )
            return Output(success=True, habits=habits, count=len(habits))

        if inp.command == "update":
            fields = {}
            for k in ("title", "description", "status", "category"):
                v = getattr(inp, k)
                if v:
                    fields[k] = v
            if inp.importance != 3:
                fields["importance_level"] = inp.importance
            if inp.time_start:
                fields["preferred_time_start"] = inp.time_start
            if inp.time_end:
                fields["preferred_time_end"] = inp.time_end
            habit = await repo.update(inp.habit_id, **fields)
            if not habit:
                return Output(success=False, error=f"Habit not found: {inp.habit_id}")
            return Output(success=True, habit=habit)

        if inp.command == "delete":
            ok = await repo.delete(inp.habit_id)
            return Output(success=ok, error="" if ok else f"Habit not found: {inp.habit_id}")

        if inp.command == "complete":
            date = inp.date or datetime.now().strftime("%Y-%m-%d")
            oid = f"occ_{inp.habit_id}_{date}"
            # Ensure occurrence exists
            await repo.create_occurrence(oid, inp.habit_id, date)
            occ = await repo.complete_occurrence(oid, notes=inp.notes or None)
            return Output(success=bool(occ), habit=occ or {})

        if inp.command == "skip":
            date = inp.date or datetime.now().strftime("%Y-%m-%d")
            oid = f"occ_{inp.habit_id}_{date}"
            await repo.create_occurrence(oid, inp.habit_id, date)
            occ = await repo.skip_occurrence(oid, reason=inp.reason or None)
            return Output(success=bool(occ), habit=occ or {})

        if inp.command == "due-today":
            due = await repo.get_due_today()
            return Output(success=True, occurrences=due, count=len(due))

        if inp.command == "occurrences":
            occs = await repo.get_occurrences(inp.habit_id)
            return Output(success=True, occurrences=occs, count=len(occs))

        if inp.command == "stats":
            habit = await repo.get(inp.habit_id)
            if not habit:
                return Output(success=False, error=f"Habit not found: {inp.habit_id}")
            return Output(success=True, habit=habit)

        if inp.command == "generate-today":
            date = inp.date or datetime.now().strftime("%Y-%m-%d")
            habits = await repo.list(status="active")
            generated = 0
            for h in habits:
                oid = f"occ_{h['habit_id']}_{date}"
                result = await repo.create_occurrence(oid, h["habit_id"], date)
                if result:
                    generated += 1
            return Output(success=True, count=generated)

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    HabitManagerTool.run()
