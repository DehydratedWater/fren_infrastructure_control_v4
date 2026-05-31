"""Validation tracker — track approach effectiveness and generate conclusions."""

import asyncio
import json
import uuid
from typing import Any, ClassVar

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="save-validation|get-date|get-last-30-days|get-conclusions|get-latest-conclusion|generate-monthly-conclusion"
    )
    attempt_id: str = Field(default="", description="Influence attempt ID")
    validated: str = Field(default="true", description="Whether approach was validated")
    effectiveness: float = Field(default=0.0, description="Effectiveness score 0.0-1.0")
    approach_type: str = Field(default="", description="Approach type")
    assumptions_tested: str = Field(default="", description="JSON array of tested assumptions")
    conditions_for_success: str = Field(default="", description="JSON array of conditions")
    notes: str = Field(default="", description="Notes")
    date: str = Field(default="", description="Date (YYYY-MM-DD)")
    month: str = Field(default="", description="Month (YYYY-MM)")


class Output(BaseModel):
    success: bool = True
    validation: dict = Field(default_factory=dict)
    validations: list[dict] = Field(default_factory=list)
    conclusion: dict = Field(default_factory=dict)
    conclusions: list[dict] = Field(default_factory=list)
    error: str = ""


class ValidationTrackerTool(ScriptTool[Input, Output]):
    name: ClassVar[str] = "validation_tracker"
    description: ClassVar[str] = "Track approach effectiveness and generate monthly conclusions"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.validations import ConclusionsRepo, ValidationsRepo

        v_repo = ValidationsRepo()
        c_repo = ConclusionsRepo()

        if inp.command == "save-validation":
            assumptions = json.loads(inp.assumptions_tested) if inp.assumptions_tested else []
            conditions = json.loads(inp.conditions_for_success) if inp.conditions_for_success else []
            row = await v_repo.save(
                validation_id=str(uuid.uuid4())[:8],
                attempt_id=inp.attempt_id,
                approach_type=inp.approach_type,
                validated=inp.validated.lower() in ("true", "1", "yes"),
                effectiveness=inp.effectiveness,
                assumptions_tested=assumptions,
                conditions_for_success=conditions,
                notes=inp.notes or None,
            )
            return Output(validation=row)

        if inp.command == "get-date":
            rows = await v_repo.get_by_date(inp.date)
            return Output(validations=rows, success=bool(rows))

        if inp.command == "get-last-30-days":
            rows = await v_repo.get_last_30_days()
            return Output(validations=rows)

        if inp.command == "get-conclusions":
            rows = await c_repo.get_all()
            return Output(conclusions=rows)

        if inp.command == "get-latest-conclusion":
            row = await c_repo.get_latest()
            return Output(conclusion=row or {}, success=row is not None)

        if inp.command == "generate-monthly-conclusion":
            conclusion = await self._generate_conclusion(inp.month, v_repo, c_repo)
            return Output(conclusion=conclusion)

        return Output(success=False, error=f"Unknown command: {inp.command}")

    async def _generate_conclusion(self, month: str, v_repo: Any, c_repo: Any) -> dict:
        from app.db.session import fetch_all, get_async_session

        async with get_async_session() as s:
            rows = await fetch_all(
                s,
                """
                SELECT * FROM validations
                WHERE TO_CHAR(date, 'YYYY-MM') = :month
                ORDER BY date
            """,
                {"month": month},
            )

        if not rows:
            return {"error": f"No validations found for {month}"}

        total = len(rows)
        validated = sum(1 for r in rows if r.get("validated"))
        approach_stats: dict[str, dict] = {}
        for r in rows:
            at = r.get("approach_type", "unknown")
            st = approach_stats.setdefault(at, {"count": 0, "total_eff": 0.0, "scores": []})
            st["count"] += 1
            eff = float(r.get("effectiveness", 0))
            st["total_eff"] += eff
            st["scores"].append(eff)

        for _at, st in approach_stats.items():
            st["avg"] = round(st["total_eff"] / st["count"], 2) if st["count"] else 0
            st["min"] = min(st["scores"]) if st["scores"] else 0
            st["max"] = max(st["scores"]) if st["scores"] else 0
            del st["scores"]
            del st["total_eff"]

        best = max(approach_stats.items(), key=lambda x: x[1]["avg"])[0] if approach_stats else None
        worst = min(approach_stats.items(), key=lambda x: x[1]["avg"])[0] if approach_stats else None

        conditions: list[str] = []
        for r in rows:
            if float(r.get("effectiveness", 0)) >= 0.7:
                conds = r.get("conditions_for_success", [])
                if isinstance(conds, str):
                    conds = json.loads(conds) if conds else []
                conditions.extend(conds)

        recs = []
        for at, st in approach_stats.items():
            if st["avg"] >= 0.7:
                recs.append(f"Continue using {at} (avg effectiveness: {st['avg']})")
            elif st["avg"] < 0.3:
                recs.append(f"Reduce use of {at} (avg effectiveness: {st['avg']})")

        row = await c_repo.save(
            conclusion_id=str(uuid.uuid4())[:8],
            month=month,
            total_validations=total,
            validated_count=validated,
            invalidated_count=total - validated,
            approach_stats=approach_stats,
            most_effective_approaches=[best] if best else [],
            least_effective_approaches=[worst] if worst else [],
            successful_conditions=dict.fromkeys(set(conditions), True) if conditions else {},
            recommendations=recs,
        )
        return row
