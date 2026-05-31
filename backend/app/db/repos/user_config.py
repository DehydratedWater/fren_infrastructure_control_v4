"""User config repository — key-value store for user preferences."""

from __future__ import annotations

from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class UserConfigRepo:
    async def get(self, config_key: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM user_config WHERE config_key = :key",
                {"key": config_key},
            )

    async def set(self, config_key: str, config_value: str) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO user_config (config_key, config_value, updated_at)
                VALUES (:key, :val, NOW())
                ON CONFLICT (config_key) DO UPDATE
                SET config_value = :val, updated_at = NOW()
                RETURNING *
                """,
                {"key": config_key, "val": config_value},
            )  # type: ignore[return-value]

    async def list(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM user_config ORDER BY config_key",
            )

    async def delete(self, config_key: str) -> None:
        from sqlalchemy import text

        async with get_async_session() as s, s.begin():
            await s.execute(
                text("DELETE FROM user_config WHERE config_key = :key"),
                {"key": config_key},
            )

    # ── persona_prose model override helpers ──
    # These layer on top of settings.persona_prose_provider / persona_prose_model.
    # Set via /model_chat; cleared via /model_chat default.
    PERSONA_PROSE_PROVIDER_KEY = "persona_prose_provider"
    PERSONA_PROSE_MODEL_KEY = "persona_prose_model"

    async def get_persona_prose_override(self) -> tuple[str, str] | None:
        """Return (provider, model) if an override is set, else None."""
        prov = await self.get(self.PERSONA_PROSE_PROVIDER_KEY)
        mod = await self.get(self.PERSONA_PROSE_MODEL_KEY)
        if prov and mod:
            return str(prov["config_value"]), str(mod["config_value"])
        return None

    async def set_persona_prose_override(self, provider: str, model: str) -> None:
        await self.set(self.PERSONA_PROSE_PROVIDER_KEY, provider)
        await self.set(self.PERSONA_PROSE_MODEL_KEY, model)

    async def clear_persona_prose_override(self) -> None:
        await self.delete(self.PERSONA_PROSE_PROVIDER_KEY)
        await self.delete(self.PERSONA_PROSE_MODEL_KEY)
