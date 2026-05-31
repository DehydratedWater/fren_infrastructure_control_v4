"""Routine Manager — daily checklist CRUD + completion tracking."""

from __future__ import annotations

import asyncio
from datetime import date, datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="add|get|list|update|delete|complete|uncomplete|due-now|checklist|stats")
    routine_id: str = Field(default="", description="Routine ID")
    title: str = Field(default="", description="Routine title")
    description: str = Field(default="", description="Routine description")
    weekdays: str = Field(default="", description="Comma-separated weekday numbers (0=Mon..6=Sun), empty=every day")
    visible_from: str = Field(default="", description="Time window start (HH:MM)")
    visible_until: str = Field(default="", description="Time window end (HH:MM)")
    sort_order: int = Field(default=0, description="Sort order within the day")
    status: str = Field(default="", description="Status filter or value: active|paused|archived")
    category: str = Field(default="", description="Category: health|medication|planning|fitness|other")
    date: str = Field(default="", description="Date (YYYY-MM-DD) for complete/uncomplete")
    notes: str = Field(default="", description="Notes for completion")
    days: int = Field(default=7, description="Number of days for stats")


class Output(BaseModel):
    success: bool = True
    routine: dict = Field(default_factory=dict)
    routines: list[dict] = Field(default_factory=list)
    completions: list[dict] = Field(default_factory=list)
    stats: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


def _parse_weekdays(s: str) -> list[int] | None:
    """Parse '0,1,2,3,4' → [0,1,2,3,4]. Empty string → None (every day)."""
    if not s.strip():
        return None
    return [int(d.strip()) for d in s.split(",") if d.strip().isdigit()]


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    return date.fromisoformat(s)


def _coerce(row: dict) -> dict:
    """Coerce non-JSON-serializable fields."""
    out = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif isinstance(v, list | tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


class RoutineManagerTool(ScriptTool[Input, Output]):
    name = "routine_manager"
    description = "Manage daily routines checklist — predefined recurring tasks by weekday with time windows"
    output_note = "If the user requested this, confirm the result via send_message.py"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.daily_routines import DailyRoutinesRepo

        repo = DailyRoutinesRepo()

        if inp.command == "add":
            rid = f"routine_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            weekdays = _parse_weekdays(inp.weekdays)
            routine = await repo.create(
                routine_id=rid,
                title=inp.title,
                description=inp.description or None,
                weekdays=weekdays if weekdays is not None else [],
                visible_from=inp.visible_from or None,
                visible_until=inp.visible_until or None,
                sort_order=inp.sort_order,
                category=inp.category or None,
            )
            return Output(success=True, routine=_coerce(routine))

        if inp.command == "get":
            routine = await repo.get(inp.routine_id)
            if not routine:
                return Output(success=False, error=f"Routine not found: {inp.routine_id}")
            return Output(success=True, routine=_coerce(routine))

        if inp.command == "list":
            routines = await repo.list(
                status=inp.status or None,
                category=inp.category or None,
            )
            return Output(success=True, routines=[_coerce(r) for r in routines], count=len(routines))

        if inp.command == "update":
            fields: dict = {}
            if inp.title:
                fields["title"] = inp.title
            if inp.description:
                fields["description"] = inp.description
            if inp.weekdays != "":
                weekdays = _parse_weekdays(inp.weekdays)
                fields["weekdays"] = weekdays if weekdays is not None else []
            if inp.visible_from:
                fields["visible_from"] = inp.visible_from
            if inp.visible_until:
                fields["visible_until"] = inp.visible_until
            if inp.sort_order:
                fields["sort_order"] = inp.sort_order
            if inp.status:
                fields["status"] = inp.status
            if inp.category:
                fields["category"] = inp.category
            routine = await repo.update(inp.routine_id, **fields)
            if not routine:
                return Output(success=False, error=f"Routine not found: {inp.routine_id}")
            return Output(success=True, routine=_coerce(routine))

        if inp.command == "delete":
            ok = await repo.delete(inp.routine_id)
            if not ok:
                return Output(success=False, error=f"Routine not found: {inp.routine_id}")
            return Output(success=True)

        if inp.command == "complete":
            dt = _parse_date(inp.date)
            result = await repo.complete(inp.routine_id, date=dt, notes=inp.notes or None)
            if not result:
                return Output(success=False, error=f"Failed to complete: {inp.routine_id}")
            return Output(success=True, routine=_coerce(result))

        if inp.command == "uncomplete":
            dt = _parse_date(inp.date)
            ok = await repo.uncomplete(inp.routine_id, date=dt)
            if not ok:
                return Output(success=False, error=f"No completion found for: {inp.routine_id}")
            return Output(success=True)

        if inp.command == "due-now":
            routines = await repo.get_due_today()
            return Output(success=True, routines=[_coerce(r) for r in routines], count=len(routines))

        if inp.command == "checklist":
            routines = await repo.get_checklist()
            return Output(success=True, routines=[_coerce(r) for r in routines], count=len(routines))

        if inp.command == "stats":
            stats = await repo.get_completion_stats(days=inp.days)
            return Output(success=True, stats=[_coerce(s) for s in stats])

        return Output(success=False, error=f"Unknown command: {inp.command}")
