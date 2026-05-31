"""Todos repository — raw SQL, async."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class TodosRepo:
    async def create(
        self,
        todo_id: str,
        title: str,
        date_str: str,
        *,
        description: str | None = None,
        status: str = "pending",
        priority: str = "medium",
        source: str = "user",
        source_metadata: dict | None = None,
        deadline: str | None = None,
        estimated_minutes: int | None = None,
        category: str = "personal",
        tags: list[str] | None = None,
        linked_goal_id: str | None = None,
        subtasks: list | None = None,
        url: str | None = None,
        dependencies: list[int] | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            # Convert date string to date object for database
            date_obj = date.fromisoformat(date_str) if isinstance(date_str, str) else date_str
            # Build params dict
            params = {
                "todo_id": todo_id,
                "title": title,
                "description": description,
                "status": status,
                "priority": priority,
                "source": source,
                "estimated_minutes": estimated_minutes,
                "category": category,
                "linked_goal_id": linked_goal_id,
                "date": date_obj,
            }
            # Only include JSON columns if they have non-empty values
            if source_metadata:
                params["source_metadata"] = json.dumps(source_metadata)
            if deadline:
                params["deadline"] = datetime.fromisoformat(deadline) if isinstance(deadline, str) else deadline
            if tags:
                params["tags"] = json.dumps(tags)
            if subtasks:
                params["subtasks"] = json.dumps(subtasks)

            # Build SQL dynamically based on available params
            columns = ["todo_id", "title", "status", "priority", "source", "date"]
            values = [":todo_id", ":title", ":status", ":priority", ":source", ":date"]

            if description is not None:
                columns.append("description")
                values.append(":description")

            if "source_metadata" in params:
                columns.append("source_metadata")
                values.append(":source_metadata")
            else:
                columns.append("source_metadata")
                values.append("DEFAULT")

            if "deadline" in params:
                columns.append("deadline")
                values.append(":deadline")
            else:
                columns.append("deadline")
                values.append("DEFAULT")

            if "estimated_minutes" in params:
                columns.append("estimated_minutes")
                values.append(":estimated_minutes")
            else:
                columns.append("estimated_minutes")
                values.append("DEFAULT")

            if category != "personal":
                columns.append("category")
                values.append(":category")
            else:
                columns.append("category")
                values.append("DEFAULT")

            if "tags" in params:
                columns.append("tags")
                values.append(":tags")
            else:
                columns.append("tags")
                values.append("DEFAULT")

            if "linked_goal_id" in params:
                columns.append("linked_goal_id")
                values.append(":linked_goal_id")
            else:
                columns.append("linked_goal_id")
                values.append("DEFAULT")

            if "subtasks" in params:
                columns.append("subtasks")
                values.append(":subtasks")
            else:
                columns.append("subtasks")
                values.append("DEFAULT")

            if dependencies:
                params["dependencies"] = json.dumps(dependencies)
                columns.append("dependencies")
                values.append(":dependencies")
            else:
                columns.append("dependencies")
                values.append("DEFAULT")

            if url:
                params["url"] = url
                columns.append("url")
                values.append(":url")

            sql = f"""
                INSERT INTO todos ({", ".join(columns)})
                VALUES ({", ".join(values)})
                RETURNING *
            """

            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def get(self, todo_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM todos WHERE todo_id = :tid", {"tid": todo_id})

    async def list(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        date: str | None = None,
        linked_goal_id: str | None = None,
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
        if date:
            from datetime import date as _date

            conds.append("date = CAST(:date AS date)")
            params["date"] = _date.fromisoformat(date) if isinstance(date, str) else date
        if linked_goal_id:
            conds.append("linked_goal_id = :lgid")
            params["lgid"] = linked_goal_id
        where = " AND ".join(conds) if conds else "1=1"
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM todos WHERE {where}
                ORDER BY created_at DESC LIMIT :limit
            """,
                params,
            )

    async def update(self, todo_id: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"tid": todo_id}
        for k, v in fields.items():
            if v is not None:
                if k in ("tags", "subtasks", "source_metadata", "goal_alignment", "dependencies"):
                    v = json.dumps(v)
                    sets.append(f"{k} = CAST(:{k} AS jsonb)")
                elif k == "deadline":
                    v = datetime.fromisoformat(v) if isinstance(v, str) else v
                    sets.append(f"{k} = :{k}")
                elif k == "date":
                    v = date.fromisoformat(v) if isinstance(v, str) else v
                    sets.append(f"{k} = :{k}")
                else:
                    sets.append(f"{k} = :{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE todos SET {", ".join(sets)}
                WHERE todo_id = :tid RETURNING *
            """,
                params,
            )

    async def complete(self, todo_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE todos SET status = 'completed', completed_at = NOW()
                WHERE todo_id = :tid RETURNING *
            """,
                {"tid": todo_id},
            )

    async def delete(self, todo_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM todos WHERE todo_id = :tid RETURNING id", {"tid": todo_id})
            return r.fetchone() is not None

    async def search_by_keyword(
        self, keywords: list[str], *, status: str = "", limit: int = 10
    ) -> list[dict[str, Any]]:
        conditions = ["1=1"]
        params: dict[str, Any] = {"limit": limit}
        if status:
            conditions.append("status = CAST(:status AS text)")
            params["status"] = status
        kw_clauses = []
        for i, kw in enumerate(keywords):
            key = f"kw_{i}"
            kw_clauses.append(f"(title ILIKE CAST(:{key} AS text) OR description ILIKE CAST(:{key} AS text))")
            params[key] = f"%{kw}%"
        if kw_clauses:
            conditions.append(f"({' OR '.join(kw_clauses)})")
        sql = f"SELECT * FROM todos WHERE {' AND '.join(conditions)} ORDER BY created_at DESC LIMIT :limit"
        async with get_async_session() as s:
            return await fetch_all(s, sql, params)

    async def get_overdue(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM todos
                WHERE status = 'pending' AND deadline < NOW()
                ORDER BY deadline ASC
            """,
            )

    async def get_today(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM todos
                WHERE status = 'pending'
                  AND (
                    date = CURRENT_DATE
                    OR (deadline IS NULL AND (date IS NULL OR date <= CURRENT_DATE))
                  )
                ORDER BY priority DESC, created_at
            """,
            )

    async def get_upcoming(self, days: int = 7) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM todos
                WHERE status = 'pending'
                  AND (
                    (date > CURRENT_DATE AND date <= CURRENT_DATE + CAST(:days AS integer))
                    OR (deadline > NOW() AND deadline <= NOW() + make_interval(days => CAST(:days AS integer)))
                  )
                ORDER BY COALESCE(deadline, date + TIME '23:59:59') ASC, priority DESC
            """,
                {"days": days},
            )

    async def get_no_deadline(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM todos
                WHERE status = 'pending'
                  AND deadline IS NULL
                  AND (date IS NULL OR date <= CURRENT_DATE)
                ORDER BY priority DESC, created_at
            """,
            )
