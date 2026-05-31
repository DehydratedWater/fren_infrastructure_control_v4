"""User facts repository."""

from __future__ import annotations

from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class UserFactsRepo:
    async def add(
        self, fact_id: str, category: str, fact_text: str, *, embedding: list[float] | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"fid": fact_id, "cat": category, "txt": fact_text}
        if embedding:
            vec = "[" + ",".join(str(v) for v in embedding) + "]"
            params["embedding"] = vec
            sql = """
                INSERT INTO user_facts (fact_id, category, fact_text, embedding)
                VALUES (:fid, :cat, :txt, CAST(:embedding AS vector))
                RETURNING *
            """
        else:
            sql = """
                INSERT INTO user_facts (fact_id, category, fact_text)
                VALUES (:fid, :cat, :txt)
                RETURNING *
            """
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def get_all(self, *, category: str | None = None) -> list[dict[str, Any]]:
        if category:
            async with get_async_session() as s:
                return await fetch_all(
                    s,
                    "SELECT * FROM user_facts WHERE category = :cat ORDER BY created_at",
                    {"cat": category},
                )
        async with get_async_session() as s:
            return await fetch_all(s, "SELECT * FROM user_facts ORDER BY category, created_at")

    async def delete(self, fact_id: str) -> bool:
        async with get_async_session() as s:
            result = await fetch_one(
                s,
                "DELETE FROM user_facts WHERE fact_id = :fid RETURNING fact_id",
                {"fid": fact_id},
            )
            return result is not None

    async def get_formatted(self) -> dict[str, list[str]]:
        rows = await self.get_all()
        grouped: dict[str, list[str]] = {}
        for r in rows:
            cat = r.get("category", "other")
            grouped.setdefault(cat, []).append(r.get("fact_text", ""))
        return grouped
