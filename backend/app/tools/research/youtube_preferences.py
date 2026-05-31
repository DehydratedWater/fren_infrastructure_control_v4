"""YouTube Preferences — manage user preferences and video feedback."""

from __future__ import annotations

import asyncio
from datetime import datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="Preference: set-preference|get-preference|list-preferences|delete-preference; "
        "Feedback: log-feedback|list-feedback; Hints: get-search-hints"
    )
    user_id: str = Field(default="default", description="User ID")
    preference_key: str = Field(default="", description="Preference key (e.g. preferred_genres, min_view_threshold)")
    preference_value: str = Field(default="", description="Preference value (JSON or plain text)")
    confidence: float = Field(default=0.5, description="Confidence 0-1")
    reason: str = Field(default="explicit", description="Update reason: explicit or implicit")
    yt_video_id: str = Field(default="", description="YouTube video ID")
    feedback_type: str = Field(default="", description="Feedback type: liked|disliked|too_obscure|good_rec")
    feedback_text: str = Field(default="", description="Free-text feedback")
    limit: int = Field(default=20, description="Max results to return")


class Output(BaseModel):
    success: bool = True
    item: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    search_hints: dict = Field(default_factory=dict)
    error: str = ""


class YouTubePreferencesTool(ScriptTool[Input, Output]):
    name = "youtube_preferences"
    description = "Manage YouTube user preferences and video feedback"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.youtube_preferences import YouTubePreferencesRepo, YouTubeVideoFeedbackRepo

        cmd = inp.command

        # ── Preferences ──
        if cmd == "set-preference":
            if not inp.preference_key or not inp.preference_value:
                return Output(success=False, error="preference_key and preference_value required")
            pid = f"pref_{inp.user_id}_{inp.preference_key}"
            p = await YouTubePreferencesRepo().upsert(
                pid,
                inp.user_id,
                inp.preference_key,
                inp.preference_value,
                confidence=inp.confidence,
                reason=inp.reason,
            )
            return Output(success=True, item=p)

        if cmd == "get-preference":
            if not inp.preference_key:
                return Output(success=False, error="preference_key required")
            p = await YouTubePreferencesRepo().get(inp.user_id, inp.preference_key)
            return Output(success=bool(p), item=p or {}, error="" if p else "Preference not found")

        if cmd == "list-preferences":
            ps = await YouTubePreferencesRepo().list_for_user(inp.user_id)
            return Output(success=True, items=ps, count=len(ps))

        if cmd == "delete-preference":
            if not inp.preference_key:
                return Output(success=False, error="preference_key required")
            ok = await YouTubePreferencesRepo().delete(inp.user_id, inp.preference_key)
            return Output(success=ok)

        # ── Feedback ──
        if cmd == "log-feedback":
            if not inp.yt_video_id or not inp.feedback_type:
                return Output(success=False, error="yt_video_id and feedback_type required")
            fid = f"fb_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            fb = await YouTubeVideoFeedbackRepo().create(
                fid,
                inp.user_id,
                inp.yt_video_id,
                inp.feedback_type,
                inp.feedback_text,
            )
            return Output(success=True, item=fb)

        if cmd == "list-feedback":
            if inp.yt_video_id:
                fbs = await YouTubeVideoFeedbackRepo().list_for_video(inp.yt_video_id)
            else:
                fbs = await YouTubeVideoFeedbackRepo().list_recent(inp.user_id, limit=inp.limit)
            return Output(success=True, items=fbs, count=len(fbs))

        # ── Search hints ──
        if cmd == "get-search-hints":
            hints = await YouTubeVideoFeedbackRepo().get_search_hints(inp.user_id)
            return Output(success=True, search_hints=hints)

        return Output(success=False, error=f"Unknown command: {cmd}")


if __name__ == "__main__":
    YouTubePreferencesTool.run()
