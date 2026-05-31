"""YouTube user preferences and video feedback repositories."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class YouTubePreferencesRepo:
    async def upsert(
        self,
        preference_id: str,
        user_id: str,
        key: str,
        value: str,
        *,
        confidence: float = 0.5,
        reason: str = "explicit",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO youtube_user_preferences
                    (preference_id, user_id, preference_key, preference_value, confidence, updated_reason)
                VALUES (:pid, :uid, :key, :val, :conf, :reason)
                ON CONFLICT (user_id, preference_key) DO UPDATE SET
                    preference_value = EXCLUDED.preference_value,
                    confidence = EXCLUDED.confidence,
                    updated_reason = EXCLUDED.updated_reason,
                    updated_at = NOW()
                RETURNING *
                """,
                {
                    "pid": preference_id,
                    "uid": user_id,
                    "key": key,
                    "val": value,
                    "conf": confidence,
                    "reason": reason,
                },
            )  # type: ignore[return-value]

    async def get(self, user_id: str, key: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM youtube_user_preferences WHERE user_id = :uid AND preference_key = :key",
                {"uid": user_id, "key": key},
            )

    async def list_for_user(self, user_id: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM youtube_user_preferences WHERE user_id = :uid ORDER BY preference_key",
                {"uid": user_id},
            )

    async def delete(self, user_id: str, key: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s,
                "DELETE FROM youtube_user_preferences WHERE user_id = :uid AND preference_key = :key RETURNING id",
                {"uid": user_id, "key": key},
            )
            return r.fetchone() is not None


class YouTubeVideoFeedbackRepo:
    async def create(
        self,
        feedback_id: str,
        user_id: str,
        yt_video_id: str,
        feedback_type: str,
        feedback_text: str = "",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO youtube_video_feedback
                    (feedback_id, user_id, yt_video_id, feedback_type, feedback_text)
                VALUES (:fid, :uid, :vid, :ftype, :ftxt)
                RETURNING *
                """,
                {
                    "fid": feedback_id,
                    "uid": user_id,
                    "vid": yt_video_id,
                    "ftype": feedback_type,
                    "ftxt": feedback_text,
                },
            )  # type: ignore[return-value]

    async def list_for_video(self, yt_video_id: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM youtube_video_feedback WHERE yt_video_id = :vid ORDER BY created_at DESC",
                {"vid": yt_video_id},
            )

    async def list_recent(self, user_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM youtube_video_feedback WHERE user_id = :uid ORDER BY created_at DESC LIMIT :limit",
                {"uid": user_id, "limit": limit},
            )

    async def get_disliked_video_ids(self, user_id: str) -> list[str]:
        async with get_async_session() as s:
            rows = await fetch_all(
                s,
                """
                SELECT DISTINCT yt_video_id FROM youtube_video_feedback
                WHERE user_id = :uid AND feedback_type IN ('disliked', 'too_obscure')
                """,
                {"uid": user_id},
            )
            return [r["yt_video_id"] for r in rows]

    async def get_search_hints(self, user_id: str) -> dict[str, Any]:
        """Build a summary of preferences + feedback for search context."""
        prefs = await YouTubePreferencesRepo().list_for_user(user_id)
        disliked = await self.get_disliked_video_ids(user_id)
        recent = await self.list_recent(user_id, limit=10)

        hints: dict[str, Any] = {}
        for p in prefs:
            val = p["preference_value"]
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                val = json.loads(val)
            hints[p["preference_key"]] = {"value": val, "confidence": p["confidence"]}

        return {
            "preferences": hints,
            "disliked_video_ids": disliked,
            "recent_feedback_count": len(recent),
        }
