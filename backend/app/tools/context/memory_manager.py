"""Memory manager tool — create, search, and manage explicit memories."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="create|search|search-tags|search-semantic|list|get|delete|update")
    title: str = Field(default="", description="Memory title")
    content: str = Field(default="", description="Memory content")
    tags: str = Field(default="", description="Comma-separated tags")
    category: str = Field(default="", description="Memory category")
    memory_id: str = Field(default="", description="Memory ID for get/delete/update")
    query: str = Field(default="", description="Search query text")
    limit: int = Field(default=20, description="Max results to return")


class Output(BaseModel):
    success: bool = True
    memory: dict | None = None
    memories: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class MemoryManagerTool(ScriptTool[Input, Output]):
    name = "memory_manager"
    description = "Create, search, and manage persistent memories"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    def _gen_id(self, title: str) -> str:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        h = hashlib.md5(f"{title}{ts}".encode()).hexdigest()[:8]
        return f"mem_{ts}_{h}"

    def _parse_tags(self, tags_str: str) -> list[str]:
        if not tags_str:
            return []
        return [t.strip() for t in tags_str.split(",") if t.strip()]

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding, returning None on failure."""
        try:
            from app.services.embeddings import get_embedding

            emb = await asyncio.to_thread(get_embedding, text)
            # Check if it's a zero vector (no API key or empty)
            if all(v == 0.0 for v in emb[:10]):
                return None
            return emb
        except Exception:
            return None

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.memories import MemoriesRepo

        repo = MemoriesRepo()

        if inp.command == "create":
            if not inp.title:
                return Output(success=False, error="title is required")
            memory_id = self._gen_id(inp.title)
            tags = self._parse_tags(inp.tags)
            embed_text = f"{inp.title}\n{inp.content}" if inp.content else inp.title
            embedding = await self._get_embedding(embed_text)
            mem = await repo.create(
                memory_id,
                inp.title,
                inp.content,
                tags=tags,
                category=inp.category,
                source="user",
                embedding=embedding,
            )
            return Output(success=True, memory=mem)

        if inp.command == "get":
            if not inp.memory_id:
                return Output(success=False, error="memory_id is required")
            mem = await repo.get(inp.memory_id)
            if not mem:
                return Output(success=False, error=f"Memory {inp.memory_id} not found")
            return Output(success=True, memory=mem)

        if inp.command == "list":
            mems = await repo.list_recent(limit=inp.limit)
            return Output(success=True, memories=mems, count=len(mems))

        if inp.command == "delete":
            if not inp.memory_id:
                return Output(success=False, error="memory_id is required")
            deleted = await repo.delete(inp.memory_id)
            if not deleted:
                return Output(success=False, error=f"Memory {inp.memory_id} not found")
            return Output(success=True)

        if inp.command == "update":
            if not inp.memory_id:
                return Output(success=False, error="memory_id is required")
            tags = self._parse_tags(inp.tags) if inp.tags else None
            # Re-embed if content or title changed
            embedding = None
            if inp.title or inp.content:
                embed_text = f"{inp.title}\n{inp.content}" if inp.content else inp.title
                embedding = await self._get_embedding(embed_text)
            mem = await repo.update(
                inp.memory_id,
                title=inp.title or None,
                content=inp.content or None,
                tags=tags,
                category=inp.category or None,
                embedding=embedding,
            )
            if not mem:
                return Output(success=False, error=f"Memory {inp.memory_id} not found")
            return Output(success=True, memory=mem)

        if inp.command == "search-tags":
            tags = self._parse_tags(inp.tags or inp.query)
            if not tags:
                return Output(success=False, error="tags or query required")
            mems = await repo.search_by_tags(tags, limit=inp.limit)
            return Output(success=True, memories=mems, count=len(mems))

        if inp.command == "search-semantic":
            if not inp.query:
                return Output(success=False, error="query is required")
            embedding = await self._get_embedding(inp.query)
            if not embedding:
                mems = await repo.search_by_text(inp.query, limit=inp.limit)
                return Output(success=True, memories=mems, count=len(mems))
            mems = await repo.search_by_embedding(embedding, limit=inp.limit)
            return Output(success=True, memories=mems, count=len(mems))

        if inp.command == "search":
            if not inp.query:
                return Output(success=False, error="query is required")
            tags = self._parse_tags(inp.tags)
            embedding = await self._get_embedding(inp.query)
            if embedding and tags:
                mems = await repo.search_hybrid(embedding, tags, limit=inp.limit)
            elif embedding:
                mems = await repo.search_by_embedding(embedding, limit=inp.limit)
            elif tags:
                mems = await repo.search_by_tags(tags, limit=inp.limit)
            else:
                mems = await repo.search_by_text(inp.query, limit=inp.limit)
            return Output(success=True, memories=mems, count=len(mems))

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    MemoryManagerTool.run()
