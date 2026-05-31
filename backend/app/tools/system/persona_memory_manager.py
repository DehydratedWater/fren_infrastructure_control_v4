"""Persona memory manager — CRUD for persona_interests / pending_thoughts / rss_feeds.

Exposes persona-side memory operations to agents via a single script tool.
Used by twily_curator, inner_monologue, relationship_initiator, and the chat drift path.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description=(
            "create-interest|list-interests|top-interests|search-interests|mark-interest-surfaced"
            "|prune-interests|delete-interest"
            "|peek-thought|consume-thought|expire-thoughts|trim-thoughts|list-thoughts|count-thoughts"
            "|list-feeds|get-feed|create-feed|update-feed|toggle-feed|delete-feed|mark-feed-fetched|feeds-due"
        )
    )
    # persona_interests
    interest_id: int = Field(default=0, description="persona_interests.id")
    topic: str = Field(default="", description="Interest topic")
    stance: str = Field(default="", description="Twily's angle/opinion on the topic")
    source: str = Field(default="self_reflection", description="rss|article|user_echo|self_reflection|dream")
    source_url: str = Field(default="", description="Source URL for provenance")
    novelty_score: float = Field(default=0.5, description="Initial novelty 0-1")
    embedding_text: str = Field(default="", description="If non-empty, compute embedding from this text and store")
    query_text: str = Field(default="", description="Semantic search query text (embedded on the fly)")
    threshold: float = Field(default=0.3, description="Cosine similarity threshold for search")

    # pending_thoughts
    thought_id: int = Field(default=0, description="pending_thoughts.id")
    content: str = Field(default="", description="Thought content (1-3 sentences)")
    kind: str = Field(default="share", description="opener|question|share|callback|contrarian")
    motivation_score: float = Field(default=0.5, description="Motivation 0-1")
    motivation_breakdown: str = Field(
        default="", description="JSON object: {curiosity, persona_fit, silence_fit, drift_need}"
    )
    topic_node_id: int = Field(default=0, description="FK topic_nodes.id (0=none)")
    persona_interest_id: int = Field(default=0, description="FK persona_interests.id (0=none)")
    kinds: str = Field(default="", description="CSV kinds for peek-thought filter")
    min_motivation: float = Field(default=0.0, description="Min motivation_score for peek")
    consumed_by: str = Field(default="", description="inner_monologue|relationship_initiator|chat_drift")
    hours: int = Field(default=48, description="Expiry window in hours")
    max_queue_size: int = Field(default=30, description="Cap for unconsumed queue")
    include_consumed: bool = Field(default=False, description="list-thoughts: include consumed rows")

    # rss_feeds
    feed_id: int = Field(default=0, description="rss_feeds.id")
    theme: str = Field(default="", description="Feed theme label")
    url: str = Field(default="", description="Feed URL")
    name: str = Field(default="", description="Short display name")
    max_items_per_run: int = Field(default=3, description="Max items curator pulls per run")
    enabled: bool = Field(default=True, description="Feed enabled flag")
    filter_enabled: str = Field(default="", description="list-feeds filter: 'true'|'false'|'' (any)")
    fetch_status: str = Field(default="", description="mark-feed-fetched: ok|error|timeout|empty")
    fetch_error: str = Field(default="", description="mark-feed-fetched: error message")
    min_age_hours: int = Field(default=6, description="feeds-due: min age in hours since last fetch")

    limit: int = Field(default=50, description="List limit")


class Output(BaseModel):
    success: bool = True
    interest: dict | None = None
    interests: list[dict] = Field(default_factory=list)
    thought: dict | None = None
    thoughts: list[dict] = Field(default_factory=list)
    feed: dict | None = None
    feeds: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class PersonaMemoryManagerTool(ScriptTool[Input, Output]):
    name: ClassVar[str] = "persona_memory_manager"
    description: ClassVar[str] = "Manage persona_interests / pending_thoughts / rss_feeds"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.session import set_null_pool

        set_null_pool(enabled=True)

        from app.db.repos.persona_memory import (
            PendingThoughtsRepo,
            PersonaInterestsRepo,
            RssFeedsRepo,
        )

        interests = PersonaInterestsRepo()
        thoughts = PendingThoughtsRepo()
        feeds = RssFeedsRepo()

        try:
            # ── persona_interests ──
            if inp.command == "create-interest":
                emb = None
                if inp.embedding_text.strip():
                    from app.services.embeddings import get_embedding

                    emb = get_embedding(inp.embedding_text)
                row = await interests.create(
                    topic=inp.topic,
                    stance=inp.stance or None,
                    source=inp.source,
                    source_url=inp.source_url or None,
                    embedding=emb,
                    novelty_score=inp.novelty_score,
                )
                return Output(interest=row)

            if inp.command == "list-interests":
                rows = await interests.list_active(limit=inp.limit)
                return Output(interests=rows, count=len(rows))

            if inp.command == "top-interests":
                rows = await interests.list_top_by_novelty(limit=inp.limit)
                return Output(interests=rows, count=len(rows))

            if inp.command == "search-interests":
                if not inp.query_text.strip():
                    return Output(success=False, error="query_text required")
                from app.services.embeddings import get_embedding

                emb = get_embedding(inp.query_text)
                rows = await interests.search_by_embedding(emb, limit=inp.limit, threshold=inp.threshold)
                return Output(interests=rows, count=len(rows))

            if inp.command == "mark-interest-surfaced":
                row = await interests.mark_surfaced(inp.interest_id)
                return Output(interest=row)

            if inp.command == "prune-interests":
                n = await interests.prune_expired()
                return Output(count=n)

            if inp.command == "delete-interest":
                ok = await interests.delete(inp.interest_id)
                return Output(success=ok)

            # ── pending_thoughts ──
            if inp.command == "peek-thought":
                kinds = [k.strip() for k in inp.kinds.split(",") if k.strip()] or None
                rows = await thoughts.peek_top(kinds=kinds, limit=inp.limit, min_motivation=inp.min_motivation)
                return Output(thoughts=rows, count=len(rows))

            if inp.command == "consume-thought":
                if not inp.consumed_by.strip():
                    return Output(success=False, error="consumed_by required")
                row = await thoughts.consume(inp.thought_id, inp.consumed_by)
                return Output(thought=row, success=row is not None)

            if inp.command == "expire-thoughts":
                n = await thoughts.expire_old(hours=inp.hours)
                return Output(count=n)

            if inp.command == "trim-thoughts":
                n = await thoughts.trim_queue(max_size=inp.max_queue_size)
                return Output(count=n)

            if inp.command == "list-thoughts":
                consumed = None if inp.include_consumed else False
                rows = await thoughts.list_recent(limit=inp.limit, consumed=consumed)
                return Output(thoughts=rows, count=len(rows))

            if inp.command == "count-thoughts":
                return Output(count=await thoughts.count_unconsumed())

            # Note: thought creation is done directly by thought_forger.py (uses embeddings
            # + motivation math), not through this CLI.

            # ── rss_feeds ──
            if inp.command == "list-feeds":
                enabled_filter: bool | None
                if inp.filter_enabled.lower() == "true":
                    enabled_filter = True
                elif inp.filter_enabled.lower() == "false":
                    enabled_filter = False
                else:
                    enabled_filter = None
                rows = await feeds.list_all(theme=inp.theme or None, enabled=enabled_filter)
                return Output(feeds=rows, count=len(rows))

            if inp.command == "get-feed":
                row = await feeds.get(inp.feed_id)
                return Output(feed=row, success=row is not None)

            if inp.command == "create-feed":
                if not (inp.theme and inp.url and inp.name):
                    return Output(success=False, error="theme, url, name required")
                row = await feeds.create(
                    theme=inp.theme,
                    url=inp.url,
                    name=inp.name,
                    max_items_per_run=inp.max_items_per_run,
                    enabled=inp.enabled,
                )
                return Output(feed=row)

            if inp.command == "update-feed":
                row = await feeds.update(
                    inp.feed_id,
                    theme=inp.theme or None,
                    url=inp.url or None,
                    name=inp.name or None,
                    max_items_per_run=inp.max_items_per_run if inp.max_items_per_run else None,
                    enabled=inp.enabled,
                )
                return Output(feed=row, success=row is not None)

            if inp.command == "toggle-feed":
                row = await feeds.toggle(inp.feed_id)
                return Output(feed=row, success=row is not None)

            if inp.command == "delete-feed":
                ok = await feeds.delete(inp.feed_id)
                return Output(success=ok)

            if inp.command == "mark-feed-fetched":
                row = await feeds.mark_fetched(
                    inp.feed_id,
                    status=inp.fetch_status,
                    error=inp.fetch_error or None,
                )
                return Output(feed=row, success=row is not None)

            if inp.command == "feeds-due":
                rows = await feeds.due_for_fetch(min_age_hours=inp.min_age_hours, limit=inp.limit)
                return Output(feeds=rows, count=len(rows))

            return Output(success=False, error=f"unknown command: {inp.command}")

        except Exception as e:
            return Output(success=False, error=f"{type(e).__name__}: {e}")
