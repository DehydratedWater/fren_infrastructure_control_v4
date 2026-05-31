"""Link enrichment tool — fetch URL metadata, cache in link_previews, embed preview text."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="fetch-url|enrich-message|backfill|reembed-messages|list-pending|show")
    url: str = Field(default="", description="URL to fetch (for fetch-url, show)")
    message_id: int = Field(default=0, description="chat_messages.id (for enrich-message)")
    days: int = Field(default=90, description="Backfill window (days) — backfill / reembed-messages")
    limit: int = Field(default=200, description="Max rows to process per call")
    force: bool = Field(default=False, description="Re-fetch URLs / re-embed messages even when already done")
    concurrency: int = Field(default=4, description="Parallel HTTP fetches (backfill)")


class Output(BaseModel):
    success: bool = True
    fetched: int = 0
    skipped: int = 0
    errors: int = 0
    reembedded: int = 0
    urls: list[dict] = Field(default_factory=list)
    preview: dict = Field(default_factory=dict)
    error: str = ""


class LinkEnrichTool(ScriptTool[Input, Output]):
    name = "link_enrich"
    description = "Fetch URL metadata (title, description, og tags) and cache it for search."

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _embed(self, text: str) -> list[float] | None:
        if not text.strip():
            return None
        try:
            from app.settings import get_settings

            if not get_settings().openai_api_key:
                return None
            from app.services.embeddings import get_embedding

            emb = await asyncio.to_thread(get_embedding, text)
            if all(v == 0.0 for v in emb[:10]):
                return None
            return emb
        except Exception:
            return None

    async def _persist_preview(self, result: dict) -> dict:
        """Store a fetch_preview result in link_previews, embedding preview text when present."""
        from app.db.repos.link_previews import LinkPreviewsRepo
        from app.services.link_enricher import build_preview_text

        repo = LinkPreviewsRepo()
        text = build_preview_text(result)  # type: ignore[arg-type]
        embedding = await self._embed(text) if text else None
        row = await repo.upsert(
            url=str(result["url"]),
            title=str(result.get("title") or "") or None,
            description=str(result.get("description") or "") or None,
            site_name=str(result.get("site_name") or "") or None,
            og_title=str(result.get("og_title") or "") or None,
            og_description=str(result.get("og_description") or "") or None,
            status=str(result.get("status") or "error"),
            http_status=result.get("http_status") if isinstance(result.get("http_status"), int) else None,
            error=str(result.get("error") or "") or None,
            embedding=embedding,
        )
        out = dict(row)
        out.pop("embedding", None)  # vector blob — not useful in tool output
        out["has_embedding"] = embedding is not None
        return out

    async def _fetch_one(self, url: str, *, force: bool) -> dict:
        from app.db.repos.link_previews import LinkPreviewsRepo
        from app.services.link_enricher import fetch_preview

        repo = LinkPreviewsRepo()
        if not force:
            existing = await repo.get(url)
            if existing and existing.get("status") == "ok":
                return {**existing, "reused": True}
        result = await fetch_preview(url)
        row = await self._persist_preview(result)
        row["reused"] = False
        return row

    async def _run(self, inp: Input) -> Output:
        if inp.command == "fetch-url":
            if not inp.url:
                return Output(success=False, error="--url required")
            row = await self._fetch_one(inp.url, force=inp.force)
            return Output(
                success=True,
                fetched=0 if row.get("reused") else 1,
                preview=row,
            )

        if inp.command == "show":
            if not inp.url:
                return Output(success=False, error="--url required")
            from app.db.repos.link_previews import LinkPreviewsRepo

            row = await LinkPreviewsRepo().get(inp.url)
            if not row:
                return Output(success=False, error=f"No preview stored for {inp.url}")
            clean = dict(row)
            has_emb = clean.get("embedding") is not None
            clean.pop("embedding", None)
            clean["has_embedding"] = has_emb
            return Output(success=True, preview=clean)

        if inp.command == "enrich-message":
            if not inp.message_id:
                return Output(success=False, error="--message_id required")
            return await self._enrich_message(inp.message_id, force=inp.force)

        if inp.command == "backfill":
            return await self._backfill(inp.days, limit=inp.limit, force=inp.force, concurrency=inp.concurrency)

        if inp.command == "reembed-messages":
            return await self._reembed_messages(inp.days, limit=inp.limit, force=inp.force)

        if inp.command == "list-pending":
            from app.db.repos.link_previews import LinkPreviewsRepo

            rows = await LinkPreviewsRepo().list_pending(limit=inp.limit)
            return Output(success=True, urls=[dict(r) for r in rows])

        return Output(success=False, error=f"Unknown command: {inp.command}")

    async def _enrich_message(self, message_id: int, *, force: bool) -> Output:
        from app.db.session import fetch_one, get_async_session
        from app.services.link_enricher import extract_urls

        async with get_async_session() as s:
            row = await fetch_one(
                s,
                "SELECT id, message FROM chat_messages WHERE id = :id",
                {"id": message_id},
            )
        if not row:
            return Output(success=False, error=f"chat_messages id {message_id} not found")
        urls = extract_urls(row.get("message", "") or "")
        if not urls:
            return Output(success=True, fetched=0, skipped=1, urls=[])
        results: list[dict] = []
        for u in urls:
            r = await self._fetch_one(u, force=force)
            results.append({"url": u, "status": r.get("status"), "title": r.get("title"), "reused": r.get("reused")})
        fetched = sum(1 for r in results if r.get("status") == "ok" and not r.get("reused"))
        errors = sum(1 for r in results if r.get("status") != "ok")
        return Output(
            success=True, fetched=fetched, skipped=len(results) - fetched - errors, errors=errors, urls=results
        )

    async def _backfill(self, days: int, *, limit: int, force: bool, concurrency: int) -> Output:
        """Walk chat_messages with URLs in the last N days, enrich each unique URL."""
        from app.db.repos.link_previews import LinkPreviewsRepo
        from app.db.session import fetch_all, get_async_session
        from app.services.link_enricher import extract_urls, fetch_previews

        since = datetime.now().astimezone() - timedelta(days=days)
        async with get_async_session() as s:
            rows = await fetch_all(
                s,
                """
                SELECT id, message FROM chat_messages
                WHERE timestamp >= :since
                  AND (message LIKE '%http://%' OR message LIKE '%https://%')
                ORDER BY timestamp DESC
                LIMIT :limit
                """,
                {"since": since, "limit": limit},
            )
        url_set: dict[str, None] = {}
        for r in rows:
            for u in extract_urls(r.get("message", "") or ""):
                url_set.setdefault(u, None)
        urls = list(url_set)
        if not urls:
            return Output(success=True, fetched=0, skipped=0, errors=0)

        repo = LinkPreviewsRepo()
        if not force:
            existing = await repo.get_many(urls)
            todo = [u for u in urls if existing.get(u, {}).get("status") != "ok"]
        else:
            todo = urls

        results = await fetch_previews(todo, concurrency=concurrency)
        fetched = 0
        errors = 0
        summary: list[dict] = []
        for r in results:
            row = await self._persist_preview(r)
            status = row.get("status")
            if status == "ok":
                fetched += 1
            else:
                errors += 1
            summary.append(
                {
                    "url": row.get("url"),
                    "status": status,
                    "title": row.get("title"),
                    "error": row.get("error"),
                }
            )
        skipped = len(urls) - len(todo)
        return Output(success=True, fetched=fetched, skipped=skipped, errors=errors, urls=summary)

    async def _reembed_messages(self, days: int, *, limit: int, force: bool) -> Output:
        """Re-embed chat_messages containing URLs using message + link-preview text.

        Before: bare URL like "https://untools.co/" embeds to meaningless vector.
        After: embedding source = "https://untools.co/\n---LINK: Tools for better thinking |
        Untools\nCollection of thinking tools..." — matches topical queries.
        """
        from app.db.repos.link_previews import LinkPreviewsRepo
        from app.db.session import execute_sql, fetch_all, get_async_session
        from app.services.embeddings import get_embeddings_batch
        from app.services.link_enricher import build_preview_text, extract_urls

        since = datetime.now().astimezone() - timedelta(days=days)
        where = "(message LIKE '%http://%' OR message LIKE '%https://%') AND timestamp >= :since"
        if not force:
            where += " AND embedding IS NULL"
        async with get_async_session() as s:
            rows = await fetch_all(
                s,
                f"SELECT id, message FROM chat_messages WHERE {where} ORDER BY id DESC LIMIT :limit",
                {"since": since, "limit": limit},
            )

        if not rows:
            return Output(success=True, reembedded=0, skipped=0, errors=0)

        # Collect all unique URLs and pull their previews in one DB hit
        url_set: dict[str, None] = {}
        per_msg_urls: dict[int, list[str]] = {}
        for r in rows:
            urls = extract_urls(r.get("message", "") or "")
            per_msg_urls[int(r["id"])] = urls
            for u in urls:
                url_set.setdefault(u, None)
        previews = await LinkPreviewsRepo().get_many(list(url_set))

        # Build composite text per message
        to_embed: list[tuple[int, str]] = []
        skipped = 0
        for r in rows:
            mid = int(r["id"])
            msg = (r.get("message") or "").strip()
            urls = per_msg_urls.get(mid, [])
            parts: list[str] = [msg] if msg else []
            for u in urls:
                p = previews.get(u)
                if not p or p.get("status") != "ok":
                    continue
                preview_text = build_preview_text(dict(p))
                if preview_text:
                    parts.append(f"---LINK {u}---\n{preview_text}")
            if len(parts) <= 1:
                # No enrichment available — skip so embedding_search still falls back to text
                skipped += 1
                continue
            to_embed.append((mid, "\n".join(parts)))

        if not to_embed:
            return Output(success=True, reembedded=0, skipped=skipped, errors=0)

        # Batch-embed (OpenAI supports up to ~2048 inputs per call; we stay well under)
        batch_size = 64
        reembedded = 0
        errors = 0
        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i : i + batch_size]
            texts = [t for _, t in batch]
            embeddings = await asyncio.to_thread(get_embeddings_batch, texts)
            async with get_async_session() as s:
                for (mid, _), emb in zip(batch, embeddings, strict=False):
                    if not emb or all(v == 0.0 for v in emb[:10]):
                        errors += 1
                        continue
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    await execute_sql(
                        s,
                        "UPDATE chat_messages SET embedding = CAST(:vec AS vector) WHERE id = :id",
                        {"vec": vec, "id": mid},
                    )
                    reembedded += 1
        return Output(success=True, reembedded=reembedded, skipped=skipped, errors=errors)


if __name__ == "__main__":
    LinkEnrichTool.run()
