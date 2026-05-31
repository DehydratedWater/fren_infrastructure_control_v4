"""Agent lessons repository — learned behavioral patterns from past mistakes."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class AgentLessonsRepo:
    async def add(
        self,
        lesson: str,
        *,
        lesson_type: str = "systemic",
        category: str = "general",
        source_pattern: str | None = None,
        source_message_ids: list[int] | None = None,
        confidence: float = 0.8,
        expires_hours: int | None = None,
        created_by: str = "lesson_extractor",
    ) -> dict[str, Any]:
        ids_json = json.dumps(source_message_ids or [])
        if expires_hours and expires_hours > 0:
            sql = """
                INSERT INTO agent_lessons
                    (lesson, lesson_type, category, source_pattern, source_message_ids,
                     confidence, created_by, expires_at)
                VALUES
                    (:lesson, :type, :category, :pattern, CAST(:ids AS jsonb),
                     :confidence, :created_by, NOW() + make_interval(hours => :hours))
                RETURNING *
            """
            params: dict[str, Any] = {
                "lesson": lesson,
                "type": lesson_type,
                "category": category,
                "pattern": source_pattern,
                "ids": ids_json,
                "confidence": confidence,
                "created_by": created_by,
                "hours": expires_hours,
            }
        else:
            sql = """
                INSERT INTO agent_lessons
                    (lesson, lesson_type, category, source_pattern, source_message_ids,
                     confidence, created_by)
                VALUES
                    (:lesson, :type, :category, :pattern, CAST(:ids AS jsonb),
                     :confidence, :created_by)
                RETURNING *
            """
            params = {
                "lesson": lesson,
                "type": lesson_type,
                "category": category,
                "pattern": source_pattern,
                "ids": ids_json,
                "confidence": confidence,
                "created_by": created_by,
            }
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def list_active(self, *, category: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            if category:
                return await fetch_all(
                    s,
                    """
                    SELECT * FROM agent_lessons
                    WHERE active = TRUE AND category = :category
                      AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY hit_count DESC, confidence DESC
                    LIMIT :limit
                """,
                    {"category": category, "limit": limit},
                )
            return await fetch_all(
                s,
                """
                SELECT * FROM agent_lessons
                WHERE active = TRUE
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY hit_count DESC, confidence DESC
                LIMIT :limit
            """,
                {"limit": limit},
            )

    async def get(self, lesson_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM agent_lessons WHERE id = :id", {"id": lesson_id})

    async def deactivate(self, lesson_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "UPDATE agent_lessons SET active = FALSE WHERE id = :id RETURNING *",
                {"id": lesson_id},
            )

    async def reinforce(self, lesson_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE agent_lessons
                SET hit_count = hit_count + 1, last_reinforced_at = NOW()
                WHERE id = :id RETURNING *
            """,
                {"id": lesson_id},
            )

    async def promote_to_systemic(self, lesson_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE agent_lessons
                SET lesson_type = 'systemic', expires_at = NULL
                WHERE id = :id RETURNING *
            """,
                {"id": lesson_id},
            )

    async def find_similar(self, lesson_text: str) -> list[dict[str, Any]]:
        """Find active lessons with similar text for dedup."""
        # Use first 50 chars as search key
        search_key = lesson_text[:50].lower()
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM agent_lessons
                WHERE active = TRUE AND LOWER(lesson) LIKE :pattern
                  AND (expires_at IS NULL OR expires_at > NOW())
            """,
                {"pattern": f"%{search_key}%"},
            )

    async def cleanup_expired(self) -> int:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM agent_lessons WHERE expires_at IS NOT NULL AND expires_at < NOW()")
            return r.rowcount

    async def archive_stale(self, days: int = 30) -> int:
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                """
                UPDATE agent_lessons SET active = FALSE
                WHERE active = TRUE AND lesson_type = 'systemic'
                  AND last_reinforced_at < NOW() - make_interval(days => :days)
            """,
                {"days": days},
            )
            return r.rowcount

    async def format_lessons_prompt(self) -> str:
        """Format active lessons as a prompt section for agent injection."""
        lessons = await self.list_active(limit=20)
        if not lessons:
            return ""

        lines = ["## Agent Lessons (learned from past mistakes — follow these)"]
        for l in lessons:  # noqa: E741
            cat = l.get("category", "general")
            text = l.get("lesson", "")
            ltype = l.get("lesson_type", "systemic")
            hits = l.get("hit_count", 1)
            suffix = f" (x{hits})" if hits > 1 else ""
            if ltype == "situational":
                lines.append(f"- [{cat}, temp] {text}{suffix}")
            else:
                lines.append(f"- [{cat}] {text}{suffix}")

        return "\n".join(lines)
