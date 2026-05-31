"""Priority Manager — importance/immediacy scoring with audits."""

from __future__ import annotations

import asyncio
from datetime import datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="add|get|list|update|delete|link|unlink|get-linked|audit|audit-history|matrix")
    priority_id: str = Field(default="", description="Priority ID")
    title: str = Field(default="", description="Priority title")
    description: str = Field(default="", description="Priority description")
    immediacy: float = Field(default=-1.0, description="Immediacy score (0.0-1.0)")
    importance: float = Field(default=-1.0, description="Importance score (0.0-1.0)")
    real_importance: float = Field(default=-1.0, description="Audited real importance")
    status: str = Field(default="", description="Status filter or value")
    category: str = Field(default="", description="Category")
    entity_type: str = Field(default="", description="goal|todo for link commands")
    entity_id: str = Field(default="", description="Entity ID for link commands")
    weight: float = Field(default=0.5, description="Contribution weight for links")
    notes: str = Field(default="", description="Alignment notes or audit notes")


class Output(BaseModel):
    success: bool = True
    priority: dict = Field(default_factory=dict)
    priorities: list[dict] = Field(default_factory=list)
    mappings: list[dict] = Field(default_factory=list)
    audits: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class PriorityManagerTool(ScriptTool[Input, Output]):
    name = "priority_manager"
    description = "Manage priorities with importance/immediacy scoring and audit trails"
    output_note = "If the user requested this, confirm the result via send_message.py"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.priorities import PrioritiesRepo

        repo = PrioritiesRepo()

        if inp.command == "add":
            pid = f"prio_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            p = await repo.create(
                priority_id=pid,
                title=inp.title,
                immediacy=inp.immediacy,
                importance=inp.importance,
                description=inp.description or None,
                category=inp.category or None,
            )
            return Output(success=True, priority=p)

        if inp.command == "get":
            p = await repo.get(inp.priority_id)
            if not p:
                return Output(success=False, error=f"Priority not found: {inp.priority_id}")
            return Output(success=True, priority=p)

        if inp.command == "list":
            ps = await repo.list(
                status=inp.status or None,
                category=inp.category or None,
            )
            return Output(success=True, priorities=ps, count=len(ps))

        if inp.command == "update":
            fields = {}
            for k in ("title", "description", "status", "category"):
                v = getattr(inp, k)
                if v:
                    fields[k] = v
            if inp.immediacy >= 0:
                fields["immediacy"] = inp.immediacy
            if inp.importance >= 0:
                fields["importance"] = inp.importance
            p = await repo.update(inp.priority_id, **fields)
            if not p:
                return Output(success=False, error=f"Priority not found: {inp.priority_id}")
            return Output(success=True, priority=p)

        if inp.command == "delete":
            ok = await repo.delete(inp.priority_id)
            return Output(success=ok, error="" if ok else "Priority not found or has links")

        if inp.command == "link":
            m = await repo.add_mapping(
                inp.priority_id,
                inp.entity_type,
                inp.entity_id,
                contribution_weight=inp.weight,
                alignment_notes=inp.notes or None,
            )
            return Output(success=bool(m), priority=m or {})

        if inp.command == "unlink":
            ok = await repo.remove_mapping(inp.priority_id, inp.entity_type, inp.entity_id)
            return Output(success=ok)

        if inp.command == "get-linked":
            ms = await repo.get_mappings(inp.priority_id)
            return Output(success=True, mappings=ms, count=len(ms))

        if inp.command == "audit":
            if inp.real_importance < 0:
                return Output(success=False, error="real_importance required for audit")
            aid = f"audit_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            p = await repo.get(inp.priority_id)
            prev = float(p.get("real_importance") or 0) if p else None
            a = await repo.add_audit(
                audit_id=aid,
                priority_id=inp.priority_id,
                new_real_importance=inp.real_importance,
                metrics_snapshot={},
                previous_real_importance=prev,
                audit_notes=inp.notes or None,
            )
            return Output(success=True, priority=a)

        if inp.command == "audit-history":
            audits = await repo.get_audits(inp.priority_id)
            return Output(success=True, audits=audits, count=len(audits))

        if inp.command == "matrix":
            ps = await repo.list(status="active")
            return Output(success=True, priorities=ps, count=len(ps))

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    PriorityManagerTool.run()
