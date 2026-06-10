"""Goals repository — raw SQL, async."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class GoalsRepo:
    async def create(
        self,
        goal_id: str,
        level: int,
        title: str,
        *,
        description: str | None = None,
        parent_goal_id: str | None = None,
        status: str = "active",
        priority: str = "medium",
        deadline: str | None = None,
        date_str: str | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO goals (goal_id, level, title, description, parent_goal_id,
                                   status, priority, deadline, date, metadata)
                VALUES (:goal_id, :level, :title, :description, :parent_goal_id,
                        :status, :priority, :deadline, :date, :metadata)
                RETURNING *
            """,
                {
                    "goal_id": goal_id,
                    "level": level,
                    "title": title,
                    "description": description,
                    "parent_goal_id": parent_goal_id,
                    "status": status,
                    "priority": priority,
                    "deadline": deadline,
                    "date": date.fromisoformat(date_str) if date_str else None,
                    "metadata": json.dumps(metadata or {}),
                },
            )  # type: ignore[return-value]

    async def get(self, goal_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM goals WHERE goal_id = :gid", {"gid": goal_id})

    async def list_active(self, *, level: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        conds = ["status = 'active'"]
        params: dict[str, Any] = {"limit": limit}
        if level is not None:
            conds.append("level = :level")
            params["level"] = level
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM goals WHERE {" AND ".join(conds)}
                ORDER BY created_at DESC LIMIT :limit
            """,
                params,
            )

    async def update(self, goal_id: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"gid": goal_id}
        for k, v in fields.items():
            if v is not None:
                if k == "metadata" or k == "child_goal_ids":
                    v = json.dumps(v)
                    sets.append(f"{k} = CAST(:{k} AS jsonb)")
                else:
                    sets.append(f"{k} = :{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE goals SET {", ".join(sets)}
                WHERE goal_id = :gid RETURNING *
            """,
                params,
            )

    async def delete(self, goal_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM goals WHERE goal_id = :gid RETURNING id", {"gid": goal_id})
            return r.fetchone() is not None

    async def search_by_keyword(self, keywords: list[str], *, limit: int = 10) -> list[dict[str, Any]]:
        kw_clauses = []
        params: dict[str, Any] = {"limit": limit}
        for i, kw in enumerate(keywords):
            key = f"kw_{i}"
            kw_clauses.append(f"(title ILIKE CAST(:{key} AS text) OR description ILIKE CAST(:{key} AS text))")
            params[key] = f"%{kw}%"
        where = f"status = 'active' AND ({' OR '.join(kw_clauses)})" if kw_clauses else "status = 'active'"
        sql = f"SELECT * FROM goals WHERE {where} ORDER BY created_at DESC LIMIT :limit"
        async with get_async_session() as s:
            return await fetch_all(s, sql, params)

    async def get_children(self, parent_goal_id: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s, "SELECT * FROM goals WHERE parent_goal_id = :pid ORDER BY level, created_at", {"pid": parent_goal_id}
            )

    async def list_with_children(self, *, limit: int = 300) -> list[dict[str, Any]]:
        """All non-archived goals (parents AND children), shallow-first.

        Read-only listing used by the dashboard to rebuild the goal hierarchy
        in Python — ordered by level then created_at so parents come before
        their children.
        """
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM goals
                WHERE status != 'archived'
                ORDER BY level ASC, created_at ASC LIMIT :limit
            """,
                {"limit": limit},
            )
