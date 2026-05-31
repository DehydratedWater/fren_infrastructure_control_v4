"""Context cache tool -- central registry for background artifacts."""

from __future__ import annotations

import asyncio
import json
import uuid

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="add|get|list-recent|list-by-tags|list-by-type|search|cleanup")
    cache_id: str = Field(default="", description="Cache entry ID (for get)")
    artifact_type: str = Field(default="", description="Type: youtube_video, research_analysis, screenshot, etc.")
    entity_type: str = Field(default="", description="Referenced table name (youtube_videos, invoices, etc.)")
    entity_id: str = Field(default="", description="Referenced row ID (vid_xxx, inv_xxx, etc.)")
    file_path: str = Field(default="", description="Relative file path from project root")
    summary: str = Field(default="", description="1-3 sentence summary of the artifact")
    tags: str = Field(default="", description='JSON array of tags: \'["youtube","ai"]\'')
    content_class: str = Field(default="public", description="public|nsfw|secret")
    source_agent: str = Field(default="", description="Agent/tool that produced this artifact")
    query: str = Field(default="", description="Search query (for search command)")
    hours: int = Field(default=24, description="Time window in hours")
    limit: int = Field(default=20, description="Max results to return")
    expires_hours: int = Field(default=0, description="TTL in hours (0 = no expiry)")


class Output(BaseModel):
    success: bool = True
    item: dict | None = None
    items: list[dict] = Field(default_factory=list)
    error: str = ""


class ContextCacheTool(ScriptTool[Input, Output]):
    name = "context_cache"
    description = "Central registry of background artifacts (YouTube videos, research, images, invoices, etc.)"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.context_cache import ContextCacheRepo

        repo = ContextCacheRepo()

        if inp.command == "add":
            if not inp.artifact_type:
                return Output(success=False, error="--artifact_type required")
            cache_id = inp.cache_id or f"ctx_{uuid.uuid4().hex[:12]}"
            tags = json.loads(inp.tags) if inp.tags else []
            item = await repo.create(
                cache_id,
                inp.artifact_type,
                entity_type=inp.entity_type,
                entity_id=inp.entity_id,
                file_path=inp.file_path,
                summary=inp.summary,
                tags=tags,
                content_class=inp.content_class,
                source_agent=inp.source_agent,
                expires_hours=inp.expires_hours,
            )
            return Output(success=True, item=item)

        if inp.command == "get":
            if not inp.cache_id:
                return Output(success=False, error="--cache_id required")
            item = await repo.get(inp.cache_id)
            if item:
                return Output(success=True, item=item)
            return Output(success=False, error=f"Not found: {inp.cache_id}")

        if inp.command == "list-recent":
            items = await repo.list_recent(
                hours=inp.hours,
                limit=inp.limit,
                content_class=inp.content_class,
            )
            return Output(success=True, items=items)

        if inp.command == "list-by-tags":
            if not inp.tags:
                return Output(success=False, error="--tags required (JSON array)")
            tags = json.loads(inp.tags)
            items = await repo.list_by_tags(
                tags,
                hours=inp.hours,
                limit=inp.limit,
                content_class=inp.content_class,
            )
            return Output(success=True, items=items)

        if inp.command == "list-by-type":
            if not inp.artifact_type:
                return Output(success=False, error="--artifact_type required")
            items = await repo.list_by_type(
                inp.artifact_type,
                hours=inp.hours,
                limit=inp.limit,
            )
            return Output(success=True, items=items)

        if inp.command == "search":
            if not inp.query:
                return Output(success=False, error="--query required")
            items = await repo.search(
                inp.query,
                hours=inp.hours,
                limit=inp.limit,
                content_class=inp.content_class,
            )
            return Output(success=True, items=items)

        if inp.command == "cleanup":
            count = await repo.delete_expired()
            return Output(success=True, item={"deleted": count})

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    ContextCacheTool.run()
