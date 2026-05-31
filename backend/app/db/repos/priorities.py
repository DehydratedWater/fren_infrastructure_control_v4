"""Priorities repository — raw SQL, async."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class PrioritiesRepo:
    async def create(
        self,
        priority_id: str,
        title: str,
        immediacy: float,
        importance: float,
        *,
        description: str | None = None,
        category: str | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO priorities (priority_id, title, immediacy, importance,
                    description, category, metadata)
                VALUES (:pid, :title, :imm, :imp, :desc, :cat, :meta)
                RETURNING *
            """,
                {
                    "pid": priority_id,
                    "title": title,
                    "imm": immediacy,
                    "imp": importance,
                    "desc": description,
                    "cat": category,
                    "meta": json.dumps(metadata) if metadata else None,
                },
            )  # type: ignore[return-value]

    async def get(self, priority_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM priorities WHERE priority_id = :pid", {"pid": priority_id})

    async def list(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        conds: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if status:
            conds.append("status = :status")
            params["status"] = status
        if category:
            conds.append("category = :category")
            params["category"] = category
        where = " AND ".join(conds) if conds else "1=1"
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM priorities WHERE {where}
                ORDER BY importance DESC, immediacy DESC LIMIT :limit
            """,
                params,
            )

    async def update(self, priority_id: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"pid": priority_id}
        for k, v in fields.items():
            if v is not None:
                if k == "metadata":
                    v = json.dumps(v)
                    sets.append(f"{k} = CAST(:{k} AS jsonb)")
                else:
                    sets.append(f"{k} = :{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE priorities SET {", ".join(sets)}
                WHERE priority_id = :pid RETURNING *
            """,
                params,
            )

    async def delete(self, priority_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s, "DELETE FROM priorities WHERE priority_id = :pid RETURNING id", {"pid": priority_id}
            )
            return r.fetchone() is not None

    # ── Mappings ──

    async def add_mapping(
        self,
        priority_id: str,
        entity_type: str,
        entity_id: str,
        *,
        contribution_weight: float = 0.5,
        alignment_notes: str | None = None,
    ) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO priority_mappings (priority_id, entity_type, entity_id,
                    contribution_weight, alignment_notes)
                VALUES (:pid, :et, :eid, :cw, :notes)
                ON CONFLICT (priority_id, entity_type, entity_id) DO UPDATE
                SET contribution_weight = :cw, alignment_notes = :notes
                RETURNING *
            """,
                {
                    "pid": priority_id,
                    "et": entity_type,
                    "eid": entity_id,
                    "cw": contribution_weight,
                    "notes": alignment_notes,
                },
            )

    async def get_mappings(self, priority_id: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(s, "SELECT * FROM priority_mappings WHERE priority_id = :pid", {"pid": priority_id})

    async def remove_mapping(self, priority_id: str, entity_type: str, entity_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                """
                DELETE FROM priority_mappings
                WHERE priority_id = :pid AND entity_type = :et AND entity_id = :eid
                RETURNING id
            """,
                {"pid": priority_id, "et": entity_type, "eid": entity_id},
            )
            return r.fetchone() is not None

    # ── Audits ──

    async def add_audit(
        self,
        audit_id: str,
        priority_id: str,
        new_real_importance: float,
        metrics_snapshot: dict,
        *,
        previous_real_importance: float | None = None,
        audit_notes: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                """
                INSERT INTO priority_audits (audit_id, priority_id,
                    previous_real_importance, new_real_importance,
                    metrics_snapshot, audit_notes)
                VALUES (:aid, :pid, :prev, :new, :snap_jsonb, :notes)
                RETURNING *
            """,
                {
                    "aid": audit_id,
                    "pid": priority_id,
                    "prev": previous_real_importance,
                    "new": new_real_importance,
                    "snap_jsonb": json.dumps(metrics_snapshot),
                    "notes": audit_notes,
                },
            )
            # Also update the priority itself
            await execute_sql(
                s,
                """
                UPDATE priorities
                SET real_importance = :ri, audit_count = audit_count + 1,
                    last_audit_at = NOW(), updated_at = NOW()
                WHERE priority_id = :pid
            """,
                {"ri": new_real_importance, "pid": priority_id},
            )
            return row  # type: ignore[return-value]

    async def get_audits(self, priority_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM priority_audits
                WHERE priority_id = :pid
                ORDER BY audited_at DESC LIMIT :limit
            """,
                {"pid": priority_id, "limit": limit},
            )
