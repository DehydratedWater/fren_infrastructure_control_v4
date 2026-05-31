"""Link search tool — find URLs shared in chat, optionally anchored to a name or topic.

Complements chat_history.py's keyword-blind search of bare URLs. Pairs each URL with
the enriched link_preview (title + description) when available, so an agent can tell
what a bare link was *about* without opening it.

Commands:
  - list: enumerate URLs shared in a date range, with preview titles
  - around-name: find chat messages that name someone and return URLs ± N messages
  - search-previews: semantic search over link_previews.embedding (topic-match links)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="list|around-name|search-previews")
    days: int = Field(default=30, description="Time window in days (list, around-name)")
    from_date: str = Field(default="", description="Start date YYYY-MM-DD (overrides days)")
    to_date: str = Field(default="", description="End date YYYY-MM-DD inclusive")
    sender: str = Field(default="", description="Filter messages by sender (user|twily)")
    name: str = Field(default="", description="Name to anchor search around (around-name)")
    window: int = Field(default=5, description="Messages before/after the name mention (around-name)")
    query: str = Field(default="", description="Semantic query (search-previews)")
    limit: int = Field(default=50, description="Max results")
    threshold: float = Field(default=0.25, description="Min similarity (search-previews)")


class LinkHit(BaseModel):
    url: str
    message_id: int | None = None
    timestamp: str = ""
    sender: str = ""
    message_excerpt: str = ""
    title: str = ""
    description: str = ""
    site_name: str = ""
    status: str = ""
    similarity: float | None = None


class Output(BaseModel):
    success: bool = True
    count: int = 0
    hits: list[LinkHit] = Field(default_factory=list)
    error: str = ""


def _parse_date(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


class LinkSearchTool(ScriptTool[Input, Output]):
    name = "link_search"
    description = "Find URLs shared in chat; enrich with cached link_preview title/description."

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        if inp.command == "list":
            return await self._list(inp)
        if inp.command == "around-name":
            return await self._around_name(inp)
        if inp.command == "search-previews":
            return await self._search_previews(inp)
        return Output(success=False, error=f"Unknown command: {inp.command}")

    def _window_bounds(self, inp: Input) -> tuple[datetime, datetime | None]:
        from_dt = _parse_date(inp.from_date)
        to_dt = _parse_date(inp.to_date)
        if from_dt is None:
            from_dt = datetime.now() - timedelta(days=max(1, inp.days))
        if to_dt is not None:
            # include the full day
            to_dt = to_dt + timedelta(days=1) - timedelta(seconds=1)
        return from_dt, to_dt

    async def _list(self, inp: Input) -> Output:
        from app.db.repos.link_previews import LinkPreviewsRepo
        from app.db.session import fetch_all, get_async_session
        from app.services.link_enricher import extract_urls

        from_dt, to_dt = self._window_bounds(inp)

        where = ["(message LIKE '%http://%' OR message LIKE '%https://%')", "timestamp >= :from_dt"]
        params: dict = {"from_dt": from_dt, "limit": inp.limit}
        if to_dt is not None:
            where.append("timestamp <= :to_dt")
            params["to_dt"] = to_dt
        if inp.sender:
            where.append("sender = :sender")
            params["sender"] = inp.sender
        sql = f"""
            SELECT id, timestamp, sender, message
            FROM chat_messages
            WHERE {" AND ".join(where)}
            ORDER BY timestamp DESC
            LIMIT :limit
        """
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, params)

        hits: list[LinkHit] = []
        seen_urls: set[tuple[int, str]] = set()
        for r in rows:
            urls = extract_urls(r.get("message", "") or "")
            for u in urls:
                key = (int(r["id"]), u)
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                hits.append(
                    LinkHit(
                        url=u,
                        message_id=int(r["id"]),
                        timestamp=str(r.get("timestamp") or "")[:19],
                        sender=str(r.get("sender") or ""),
                        message_excerpt=str(r.get("message") or "")[:160],
                    )
                )

        previews = await LinkPreviewsRepo().get_many([h.url for h in hits])
        for h in hits:
            p = previews.get(h.url)
            if p:
                h.title = p.get("title") or p.get("og_title") or ""
                h.description = p.get("description") or p.get("og_description") or ""
                h.site_name = p.get("site_name") or ""
                h.status = p.get("status") or ""
        return Output(success=True, count=len(hits), hits=hits)

    async def _around_name(self, inp: Input) -> Output:
        if not inp.name:
            return Output(success=False, error="--name required for around-name")
        from app.db.repos.link_previews import LinkPreviewsRepo
        from app.db.session import fetch_all, get_async_session
        from app.services.link_enricher import extract_urls

        from_dt, to_dt = self._window_bounds(inp)

        # Find messages mentioning the name
        mention_where = ["message ILIKE :pat", "timestamp >= :from_dt"]
        params: dict = {"pat": f"%{inp.name}%", "from_dt": from_dt, "limit": inp.limit}
        if to_dt is not None:
            mention_where.append("timestamp <= :to_dt")
            params["to_dt"] = to_dt
        sql_mentions = f"""
            SELECT id, timestamp FROM chat_messages
            WHERE {" AND ".join(mention_where)}
            ORDER BY timestamp DESC
            LIMIT :limit
        """
        async with get_async_session() as s:
            mentions = await fetch_all(s, sql_mentions, params)

        if not mentions:
            return Output(success=True, count=0, hits=[])

        # For each mention, fetch ± window messages and collect URLs
        all_hits: list[LinkHit] = []
        seen: set[tuple[int, str]] = set()
        for m in mentions:
            mid = int(m["id"])
            sql_nearby = """
                SELECT id, timestamp, sender, message FROM (
                    SELECT id, timestamp, sender, message
                    FROM chat_messages
                    WHERE id <= :mid
                    ORDER BY id DESC LIMIT :win
                ) ab
                UNION ALL
                SELECT id, timestamp, sender, message FROM (
                    SELECT id, timestamp, sender, message
                    FROM chat_messages
                    WHERE id > :mid
                    ORDER BY id ASC LIMIT :win
                ) af
                ORDER BY id ASC
            """
            async with get_async_session() as s:
                rows = await fetch_all(s, sql_nearby, {"mid": mid, "win": inp.window})
            for r in rows:
                if inp.sender and r.get("sender") != inp.sender:
                    continue
                urls = extract_urls(r.get("message", "") or "")
                for u in urls:
                    key = (int(r["id"]), u)
                    if key in seen:
                        continue
                    seen.add(key)
                    all_hits.append(
                        LinkHit(
                            url=u,
                            message_id=int(r["id"]),
                            timestamp=str(r.get("timestamp") or "")[:19],
                            sender=str(r.get("sender") or ""),
                            message_excerpt=str(r.get("message") or "")[:160],
                        )
                    )

        previews = await LinkPreviewsRepo().get_many([h.url for h in all_hits])
        for h in all_hits:
            p = previews.get(h.url)
            if p:
                h.title = p.get("title") or p.get("og_title") or ""
                h.description = p.get("description") or p.get("og_description") or ""
                h.site_name = p.get("site_name") or ""
                h.status = p.get("status") or ""

        all_hits.sort(key=lambda h: h.timestamp, reverse=True)
        return Output(success=True, count=len(all_hits), hits=all_hits)

    async def _search_previews(self, inp: Input) -> Output:
        if not inp.query:
            return Output(success=False, error="--query required for search-previews")
        from app.db.repos.link_previews import LinkPreviewsRepo
        from app.services.embeddings import get_embedding

        emb = await asyncio.to_thread(get_embedding, inp.query)
        if not emb or all(v == 0.0 for v in emb[:10]):
            return Output(success=False, error="Embedding unavailable (no API key?)")

        rows = await LinkPreviewsRepo().search_by_embedding(emb, limit=inp.limit, threshold=inp.threshold)
        hits = [
            LinkHit(
                url=r["url"],
                title=r.get("title") or r.get("og_title") or "",
                description=r.get("description") or r.get("og_description") or "",
                site_name=r.get("site_name") or "",
                status=r.get("status") or "",
                similarity=r.get("similarity"),
            )
            for r in rows
        ]
        return Output(success=True, count=len(hits), hits=hits)


if __name__ == "__main__":
    LinkSearchTool.run()
