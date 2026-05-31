"""Website Monitor — check websites for changes and run search queries."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="check-website|check-all|check-topic-websites|run-search-queries")
    website_id: str = Field(default="", description="Website ID")
    topic_id: str = Field(default="", description="Topic ID")


class Output(BaseModel):
    success: bool = True
    item: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class WebsiteMonitorTool(ScriptTool[Input, Output]):
    name = "website_monitor"
    description = "Check websites for content changes and run periodic search queries"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        cmd = inp.command

        if cmd == "check-website":
            return await self._check_website(inp.website_id)

        if cmd == "check-all":
            return await self._check_all()

        if cmd == "check-topic-websites":
            return await self._check_topic_websites(inp.topic_id)

        if cmd == "run-search-queries":
            return await self._run_search_queries(inp.topic_id)

        return Output(success=False, error=f"Unknown command: {cmd}")

    async def _fetch_content(self, url: str) -> str:
        """Fetch website text content via httpx."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "FrenResearchBot/1.0"})
            resp.raise_for_status()
            return resp.text

    async def _check_website(self, website_id: str) -> Output:
        from app.db.repos.research import TopicWebsiteRepo, WebsiteSnapshotRepo

        website = await TopicWebsiteRepo().get(website_id)
        if not website:
            return Output(success=False, error=f"Website not found: {website_id}")

        try:
            content = await self._fetch_content(website["url"])
        except Exception as e:
            return Output(success=False, error=f"Failed to fetch {website['url']}: {e}")

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:64]
        has_changes = content_hash != (website.get("last_content_hash") or "")

        diff_summary = ""
        if has_changes and website.get("last_content_hash"):
            diff_summary = f"Content changed (hash: {content_hash[:12]}...)"

        sid = f"snap_{datetime.now().strftime('%Y%m%d%H%M')}_{id(website_id) % 0xFFFFFFFF:08x}"
        snapshot = await WebsiteSnapshotRepo().create(
            sid,
            website_id,
            content_text=content[:50000],
            content_hash=content_hash,
            has_changes=has_changes,
            diff_summary=diff_summary,
        )
        await TopicWebsiteRepo().mark_checked(website_id, content_hash)

        return Output(
            success=True,
            item={
                "website_id": website_id,
                "url": website["url"],
                "has_changes": has_changes,
                "snapshot_id": snapshot["snapshot_id"],
                "diff_summary": diff_summary,
            },
        )

    async def _check_all(self) -> Output:
        from app.db.repos.research import TopicWebsiteRepo

        websites = await TopicWebsiteRepo().list_active()
        results: list[dict[str, Any]] = []
        for w in websites:
            r = await self._check_website(w["website_id"])
            results.append(r.item if r.success else {"website_id": w["website_id"], "error": r.error})
        return Output(success=True, items=results, count=len(results))

    async def _check_topic_websites(self, topic_id: str) -> Output:
        from app.db.repos.research import TopicWebsiteRepo

        websites = await TopicWebsiteRepo().list_for_topic(topic_id)
        results: list[dict[str, Any]] = []
        for w in websites:
            r = await self._check_website(w["website_id"])
            results.append(r.item if r.success else {"website_id": w["website_id"], "error": r.error})
        return Output(success=True, items=results, count=len(results))

    async def _run_search_queries(self, topic_id: str) -> Output:
        from app.db.repos.research import TopicSearchQueryRepo
        from app.tools.research.web_search import WebSearchTool

        repo = TopicSearchQueryRepo()
        if topic_id:
            queries = await repo.list_for_topic(topic_id)
        else:
            queries = await repo.list_active()

        results: list[dict[str, Any]] = []
        search = WebSearchTool()
        for q in queries:
            from app.tools.research.web_search import Input as SearchInput

            search_inp = SearchInput(query=q["query"], max_results=5)
            search_out = search.execute(search_inp)
            await repo.mark_run(q["query_id"])
            results.append(
                {
                    "query_id": q["query_id"],
                    "query": q["query"],
                    "topic_name": q.get("topic_name", ""),
                    "results": search_out.item.get("organic_results", []) if search_out.success else [],
                }
            )

        return Output(success=True, items=results, count=len(results))


if __name__ == "__main__":
    WebsiteMonitorTool.run()
