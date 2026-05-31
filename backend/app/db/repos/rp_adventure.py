"""RP adventure repos - adventures, characters, world state, story log."""

from __future__ import annotations

from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class AdventureRepo:
    async def create(
        self,
        chat_id: int,
        title: str,
        setting: str,
        *,
        genre: str = "fantasy",
        tone: str = "narrative",
        current_scene: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO rp_adventures (chat_id, title, setting, genre, tone, current_scene)
                VALUES (:chat_id, :title, :setting, :genre, :tone, :scene)
                RETURNING *
                """,
                {
                    "chat_id": chat_id,
                    "title": title,
                    "setting": setting,
                    "genre": genre,
                    "tone": tone,
                    "scene": current_scene,
                },
            )

    async def get(self, adventure_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM rp_adventures WHERE id = :id", {"id": adventure_id})

    async def get_active(self, chat_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM rp_adventures WHERE chat_id = :cid AND status = 'active' ORDER BY updated_at DESC LIMIT 1",
                {"cid": chat_id},
            )

    async def list_all(self, chat_id: int, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM rp_adventures WHERE chat_id = :cid ORDER BY created_at DESC LIMIT :limit",
                {"cid": chat_id, "limit": limit},
            )

    async def update_status(self, adventure_id: int, status: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "UPDATE rp_adventures SET status = :status, updated_at = NOW() WHERE id = :id RETURNING *",
                {"id": adventure_id, "status": status},
            )

    async def update_scene(self, adventure_id: int, scene: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "UPDATE rp_adventures SET current_scene = :scene, updated_at = NOW() WHERE id = :id RETURNING *",
                {"id": adventure_id, "scene": scene},
            )

    async def increment_turn(self, adventure_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "UPDATE rp_adventures SET turn_count = turn_count + 1, updated_at = NOW() WHERE id = :id RETURNING *",
                {"id": adventure_id},
            )

    async def update_config(self, adventure_id: int, **fields: Any) -> dict[str, Any] | None:
        """Update adventure config fields: cot_mode, narrative_mode, writing_style, current_time, current_date, context_summary."""
        allowed = {
            "cot_mode",
            "narrative_mode",
            "writing_style",
            "inworld_time",
            "inworld_date",
            "current_scene",
            "context_summary",
            "prose_provider",
            "prose_model",
        }
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return await self.get(adventure_id)
        set_clause = ", ".join(f"{k} = :{k}" for k in updates)
        updates["id"] = adventure_id
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE rp_adventures SET {set_clause}, updated_at = NOW() WHERE id = :id RETURNING *",
                updates,
            )


class CharacterRepo:
    async def create(
        self,
        adventure_id: int,
        name: str,
        personality: str,
        *,
        role: str = "npc",
        background: str | None = None,
        knowledge: str | None = None,
        appearance: str | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO rp_characters
                    (adventure_id, name, role, personality, background, knowledge, appearance, location)
                VALUES (:aid, :name, :role, :personality, :bg, :knowledge, :appearance, :location)
                RETURNING *
                """,
                {
                    "aid": adventure_id,
                    "name": name,
                    "role": role,
                    "personality": personality,
                    "bg": background,
                    "knowledge": knowledge,
                    "appearance": appearance,
                    "location": location,
                },
            )

    async def get(self, character_id: int) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM rp_characters WHERE id = :id", {"id": character_id})

    async def list_active(self, adventure_id: int) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM rp_characters WHERE adventure_id = :aid AND status = 'active' ORDER BY created_at",
                {"aid": adventure_id},
            )

    async def list_all(self, adventure_id: int) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM rp_characters WHERE adventure_id = :aid ORDER BY created_at",
                {"aid": adventure_id},
            )

    async def get_by_location(self, adventure_id: int, location: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM rp_characters WHERE adventure_id = :aid AND status = 'active' AND location = :loc ORDER BY created_at",
                {"aid": adventure_id, "loc": location},
            )

    async def update(self, character_id: int, **fields: Any) -> dict[str, Any] | None:
        allowed = {
            "mood",
            "location",
            "inventory",
            "stats",
            "status",
            "knowledge",
            "personality",
            "background",
            "hidden_layer",
            "trust_map",
            "current_goal",
            "pressure",
            "dialogue_color",
            "current_outfit",
        }
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return await self.get(character_id)
        set_clause = ", ".join(f"{k} = :{k}" for k in updates)
        updates["id"] = character_id
        async with get_async_session() as s:
            return await fetch_one(s, f"UPDATE rp_characters SET {set_clause} WHERE id = :id RETURNING *", updates)


class WorldStateRepo:
    async def set_aspect(self, adventure_id: int, aspect: str, value: str) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO rp_world_state (adventure_id, aspect, value)
                VALUES (:aid, :aspect, :value)
                ON CONFLICT (adventure_id, aspect)
                DO UPDATE SET value = :value, updated_at = NOW()
                RETURNING *
                """,
                {"aid": adventure_id, "aspect": aspect, "value": value},
            )

    async def get_all(self, adventure_id: int) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM rp_world_state WHERE adventure_id = :aid ORDER BY aspect",
                {"aid": adventure_id},
            )

    async def get_aspect(self, adventure_id: int, aspect: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM rp_world_state WHERE adventure_id = :aid AND aspect = :aspect",
                {"aid": adventure_id, "aspect": aspect},
            )


class StoryLogRepo:
    async def append(
        self,
        adventure_id: int,
        turn_number: int,
        content: str,
        *,
        speaker: str | None = None,
        entry_type: str = "dialogue",
        metadata: str = "{}",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO rp_story_log
                    (adventure_id, turn_number, speaker, content, entry_type, metadata)
                VALUES (:aid, :turn, :speaker, :content, :etype, CAST(:meta AS jsonb))
                RETURNING *
                """,
                {
                    "aid": adventure_id,
                    "turn": turn_number,
                    "speaker": speaker,
                    "content": content,
                    "etype": entry_type,
                    "meta": metadata,
                },
            )

    async def get_recent(self, adventure_id: int, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM (
                    SELECT * FROM rp_story_log
                    WHERE adventure_id = :aid
                    ORDER BY turn_number DESC, id DESC
                    LIMIT :limit
                ) sub ORDER BY turn_number ASC, id ASC
                """,
                {"aid": adventure_id, "limit": limit},
            )

    async def get_all(self, adventure_id: int) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM rp_story_log WHERE adventure_id = :aid ORDER BY turn_number, id",
                {"aid": adventure_id},
            )

    async def get_by_entry_type(self, adventure_id: int, entry_type: str, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM (
                    SELECT * FROM rp_story_log
                    WHERE adventure_id = :aid AND entry_type = :etype
                    ORDER BY turn_number DESC, id DESC
                    LIMIT :limit
                ) sub ORDER BY turn_number ASC, id ASC
                """,
                {"aid": adventure_id, "etype": entry_type, "limit": limit},
            )

    async def get_turn_count(self, adventure_id: int) -> int:
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                "SELECT COALESCE(MAX(turn_number), 0) AS max_turn FROM rp_story_log WHERE adventure_id = :aid",
                {"aid": adventure_id},
            )
            return int(row["max_turn"]) if row else 0

    async def get_range(self, adventure_id: int, *, from_turn: int, to_turn: int) -> list[dict[str, Any]]:
        """Get story entries within an inclusive turn range, in natural order."""
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM rp_story_log
                WHERE adventure_id = :aid
                  AND turn_number BETWEEN :lo AND :hi
                ORDER BY turn_number ASC, id ASC
                """,
                {"aid": adventure_id, "lo": from_turn, "hi": to_turn},
            )


class SummaryRepo:
    """Progressive summaries maintained by the orchestrator for unbounded memory."""

    async def upsert(
        self,
        adventure_id: int,
        window: str,
        text: str,
        *,
        covers_from_turn: int | None = None,
        covers_to_turn: int | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                """
                INSERT INTO rp_summaries
                    (adventure_id, window_name, text, covers_from_turn, covers_to_turn)
                VALUES (:aid, :window, :text, :lo, :hi)
                ON CONFLICT (adventure_id, window_name) DO UPDATE
                SET text = :text,
                    covers_from_turn = :lo,
                    covers_to_turn = :hi,
                    updated_at = NOW()
                RETURNING id, adventure_id, window_name AS window, text,
                          covers_from_turn, covers_to_turn, updated_at
                """,
                {
                    "aid": adventure_id,
                    "window": window,
                    "text": text,
                    "lo": covers_from_turn,
                    "hi": covers_to_turn,
                },
            )
            return row or {}

    async def get_all(self, adventure_id: int) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT id, adventure_id, window_name AS window, text,
                       covers_from_turn, covers_to_turn, updated_at
                FROM rp_summaries
                WHERE adventure_id = :aid
                ORDER BY window_name
                """,
                {"aid": adventure_id},
            )


class RecallPinRepo:
    """Single-use recall pins: orchestrator injects forgotten context for one upcoming turn."""

    async def add(self, adventure_id: int, turn: int, text: str) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                """
                INSERT INTO rp_recall_pins (adventure_id, turn, text)
                VALUES (:aid, :turn, :text)
                RETURNING *
                """,
                {"aid": adventure_id, "turn": turn, "text": text},
            )

    async def get_active(self, adventure_id: int, *, limit: int = 5) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM rp_recall_pins
                WHERE adventure_id = :aid
                ORDER BY created_at DESC
                LIMIT :limit
                """,
                {"aid": adventure_id, "limit": limit},
            )

    async def clear(self, adventure_id: int) -> int:
        async with get_async_session() as s:
            result = await execute_sql(
                s,
                "DELETE FROM rp_recall_pins WHERE adventure_id = :aid",
                {"aid": adventure_id},
            )
            return result.rowcount or 0
