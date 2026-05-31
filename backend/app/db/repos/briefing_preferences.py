"""Briefing preferences repository — raw SQL, async."""

from __future__ import annotations

from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session

# Default sections with (section, enabled, priority)
_DEFAULTS = [
    ("goals", True, 10),
    ("todos", True, 20),
    ("habits", True, 30),
    ("priorities", True, 40),
    ("calendar", True, 50),
    ("strategies", True, 60),
    ("email", False, 70),
    ("weather", False, 80),
    ("news", False, 90),
    ("research", False, 100),
    ("profile_insights", False, 110),
]


class BriefingPreferencesRepo:
    async def list_all(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM briefing_preferences ORDER BY priority ASC",
            )

    async def list_enabled(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM briefing_preferences WHERE enabled = true ORDER BY priority ASC",
            )

    async def get(self, section: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM briefing_preferences WHERE section = :section",
                {"section": section},
            )

    async def upsert(self, section: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"section": section}
        for k, v in fields.items():
            if v is not None and k in ("enabled", "instructions", "priority"):
                sets.append(f"{k} = :{k}")
                params[k] = v

        # Try update first
        existing = await self.get(section)
        if existing:
            async with get_async_session() as s:
                return await fetch_one(
                    s,
                    f"""
                    UPDATE briefing_preferences SET {", ".join(sets)}
                    WHERE section = :section RETURNING *
                    """,
                    params,
                )
        else:
            # Insert new section
            enabled = fields.get("enabled", True)
            instructions = fields.get("instructions")
            priority = fields.get("priority", 50)
            async with get_async_session() as s:
                return await fetch_one(
                    s,
                    """
                    INSERT INTO briefing_preferences (section, enabled, instructions, priority)
                    VALUES (:section, :enabled, :instructions, :priority)
                    RETURNING *
                    """,
                    {
                        "section": section,
                        "enabled": enabled,
                        "instructions": instructions,
                        "priority": priority,
                    },
                )

    async def toggle(self, section: str, enabled: bool) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE briefing_preferences SET enabled = :enabled, updated_at = NOW()
                WHERE section = :section RETURNING *
                """,
                {"section": section, "enabled": enabled},
            )

    async def update_instructions(self, section: str, instructions: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE briefing_preferences SET instructions = :instructions, updated_at = NOW()
                WHERE section = :section RETURNING *
                """,
                {"section": section, "instructions": instructions},
            )

    async def delete(self, section: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                "DELETE FROM briefing_preferences WHERE section = :section RETURNING id",
                {"section": section},
            )
            return r.fetchone() is not None

    async def reset_defaults(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            await execute_sql(s, "DELETE FROM briefing_preferences")
            for section, enabled, priority in _DEFAULTS:
                await execute_sql(
                    s,
                    """
                    INSERT INTO briefing_preferences (section, enabled, priority)
                    VALUES (:section, :enabled, :priority)
                    """,
                    {"section": section, "enabled": enabled, "priority": priority},
                )
            return await fetch_all(
                s,
                "SELECT * FROM briefing_preferences ORDER BY priority ASC",
            )
