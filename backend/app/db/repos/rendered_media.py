"""Rendered media repository — track ComfyUI render outputs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


def _make_media_id(media_type: str) -> str:
    prefix = "img" if media_type == "image" else "vid"
    now = datetime.now(UTC)
    return f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class RenderedMediaRepo:
    """CRUD for the rendered_media table."""

    async def create(
        self,
        *,
        media_type: str,
        file_path: str,
        positive_prompt: str,
        workflow_id: str = "",
        negative_prompt: str = "",
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        elapsed_seconds: float | None = None,
        source_agent: str = "",
        source_ralf_id: str = "",
        source_stage_number: int | None = None,
        source_attempt_number: int | None = None,
        reference_media_id: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        media_id = _make_media_id(media_type)
        sql = """
            INSERT INTO rendered_media
                (media_id, media_type, file_path, workflow_id, positive_prompt, negative_prompt,
                 seed, width, height, elapsed_seconds, source_agent, source_ralf_id,
                 source_stage_number, source_attempt_number, reference_media_id, notes)
            VALUES
                (:media_id, :media_type, :file_path, :workflow_id, :positive_prompt, :negative_prompt,
                 :seed, :width, :height, :elapsed_seconds, :source_agent, :source_ralf_id,
                 :source_stage_number, :source_attempt_number, :reference_media_id, :notes)
            RETURNING *
        """
        params = {
            "media_id": media_id,
            "media_type": media_type,
            "file_path": file_path,
            "workflow_id": workflow_id or None,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt or None,
            "seed": seed,
            "width": width,
            "height": height,
            "elapsed_seconds": elapsed_seconds,
            "source_agent": source_agent or None,
            "source_ralf_id": source_ralf_id or None,
            "source_stage_number": source_stage_number,
            "source_attempt_number": source_attempt_number,
            "reference_media_id": reference_media_id or None,
            "notes": notes or None,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {"media_id": media_id}

    async def get(self, media_id: str) -> dict[str, Any] | None:
        sql = "SELECT * FROM rendered_media WHERE media_id = :media_id"
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"media_id": media_id})
            return dict(row) if row else None

    async def list_for_ralf(self, source_ralf_id: str, *, stage_number: int | None = None) -> list[dict[str, Any]]:
        if stage_number is not None:
            sql = """
                SELECT * FROM rendered_media
                WHERE source_ralf_id = :ralf_id AND source_stage_number = :stage
                ORDER BY created_at
            """
            params = {"ralf_id": source_ralf_id, "stage": stage_number}
        else:
            sql = """
                SELECT * FROM rendered_media
                WHERE source_ralf_id = :ralf_id
                ORDER BY created_at
            """
            params = {"ralf_id": source_ralf_id}
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, params)
            return [dict(r) for r in rows]

    async def list_recent(self, *, media_type: str = "", limit: int = 20) -> list[dict[str, Any]]:
        if media_type:
            sql = """
                SELECT * FROM rendered_media
                WHERE media_type = :media_type
                ORDER BY created_at DESC LIMIT :limit
            """
            params = {"media_type": media_type, "limit": limit}
        else:
            sql = "SELECT * FROM rendered_media ORDER BY created_at DESC LIMIT :limit"
            params = {"limit": limit}
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, params)
            return [dict(r) for r in rows]

    async def set_notes(self, media_id: str, notes: str) -> None:
        sql = "UPDATE rendered_media SET notes = :notes WHERE media_id = :media_id"
        async with get_async_session() as s:
            await execute_sql(s, sql, {"media_id": media_id, "notes": notes})

    async def list_older_than(self, cutoff: datetime) -> list[dict[str, Any]]:
        """Rows created before *cutoff* — retention-cleanup candidates."""
        sql = "SELECT * FROM rendered_media WHERE created_at < :cutoff ORDER BY created_at"
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {"cutoff": cutoff})
            return [dict(r) for r in rows]

    async def delete(self, media_id: str) -> bool:
        sql = "DELETE FROM rendered_media WHERE media_id = :media_id"
        async with get_async_session() as s:
            result = await execute_sql(s, sql, {"media_id": media_id})
            return bool(getattr(result, "rowcount", 0))
