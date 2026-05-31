"""User rules repository — persistent directives from user to all agents."""

from __future__ import annotations

from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class UserRulesRepo:
    async def add(
        self,
        rule: str,
        *,
        category: str = "general",
        created_by: str = "user",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO user_rules (rule, category, created_by)
                VALUES (:rule, :category, :created_by)
                RETURNING *
            """,
                {"rule": rule, "category": category, "created_by": created_by},
            )  # type: ignore[return-value]

    async def list_active(self, *, category: str | None = None) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            if category:
                return await fetch_all(
                    s,
                    """
                    SELECT * FROM user_rules
                    WHERE active = TRUE AND category = :category
                    ORDER BY created_at ASC
                """,
                    {"category": category},
                )
            return await fetch_all(
                s,
                """
                SELECT * FROM user_rules
                WHERE active = TRUE
                ORDER BY created_at ASC
            """,
                {},
            )

    async def deactivate(self, rule_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE user_rules SET active = FALSE
                WHERE id = :id RETURNING *
            """,
                {"id": rule_id},
            )

    async def get(self, rule_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM user_rules WHERE id = :id",
                {"id": rule_id},
            )

    async def format_rules_prompt(self) -> str:
        """Format all active rules as a prompt section for agent injection."""
        rules = await self.list_active()
        if not rules:
            return ""

        lines = ["## User Rules (MUST follow — only the user can change these)"]
        for r in rules:
            category = r.get("category", "general")
            rule_text = r.get("rule", "")
            lines.append(f"- [{category}] {rule_text}")

        return "\n".join(lines)
