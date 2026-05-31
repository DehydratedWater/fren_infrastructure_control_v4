"""RP Memory toolbox — progressive summaries, RAG recall, and pinning.

Commands (called by the orchestrator agent):

- `index   --adventure_id N [--limit 50]`  Embed any un-embedded recent story entries
                                           into `embedding_chunks` with source_table
                                           "rp_story" and source_id "{adv}:{log_id}".
- `search  --adventure_id N --query "..." [--limit 5]`  Cosine-similarity RAG search
                                           over story entries for this adventure.
- `summarize --adventure_id N --window recent|mid|distant`  Summarize the matching
                                           turn window via the orchestrator model and
                                           upsert into `rp_summaries`.
- `pin     --adventure_id N --turn T --text "..."`  Drop a one-shot recall pin that
                                           the next prose call will include in its
                                           system prompt.

Window semantics:
- recent:  last 30 raw entries → summary covers turns [max-60 .. max-30]
- mid:     turns [max-200 .. max-60]
- distant: turns [0       .. max-200]
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field

_RECENT_WINDOW = 30
_MID_WINDOW = 200
_INDEX_BATCH = 50


class Input(BaseModel):
    command: str = Field(description="index|search|summarize|pin")
    adventure_id: int = Field(default=0, description="Adventure ID")
    query: str = Field(default="", description="RAG query text (search)")
    window: str = Field(default="recent", description="Summary window: recent|mid|distant")
    turn: int = Field(default=0, description="Turn number for pin")
    text: str = Field(default="", description="Recall pin text")
    limit: int = Field(default=5, description="Result limit")


class Output(BaseModel):
    success: bool = True
    results: list[dict] = Field(default_factory=list)
    summary: str = ""
    indexed: int = 0
    pin: dict | None = None
    error: str = ""


class RPMemoryTool(ScriptTool[Input, Output]):
    name = "rp_memory"
    description = (
        "RP memory operations: index story entries for RAG, search past turns, "
        "maintain progressive summaries, drop single-use recall pins."
    )

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        cmd = inp.command
        if inp.adventure_id <= 0:
            return Output(success=False, error="adventure_id is required")

        if cmd == "index":
            return await self._index(inp.adventure_id, inp.limit or _INDEX_BATCH)
        if cmd == "search":
            if not inp.query:
                return Output(success=False, error="query is required for search")
            return await self._search(inp.adventure_id, inp.query, inp.limit or 5)
        if cmd == "summarize":
            return await self._summarize(inp.adventure_id, inp.window or "recent")
        if cmd == "pin":
            if not inp.text:
                return Output(success=False, error="text is required for pin")
            return await self._pin(inp.adventure_id, inp.turn, inp.text)
        return Output(success=False, error=f"unknown command: {cmd}")

    # ── index ────────────────────────────────────────────────────────────
    async def _index(self, adventure_id: int, limit: int) -> Output:
        from app.db.repos.embedding_chunks import EmbeddingChunksRepo
        from app.db.repos.rp_adventure import StoryLogRepo
        from app.services.embeddings import get_embedding

        story_repo = StoryLogRepo()
        chunks_repo = EmbeddingChunksRepo()

        rows = await story_repo.get_recent(adventure_id, limit=limit)
        if not rows:
            return Output(success=True, indexed=0)

        indexed = 0
        for row in rows:
            log_id = row.get("id")
            content = (row.get("content") or "").strip()
            if not content or log_id is None:
                continue
            source_id = f"{adventure_id}:{log_id}"
            existing = await chunks_repo.count_for_source("rp_story", source_id)
            if existing > 0:
                continue
            speaker = row.get("speaker") or ""
            etype = row.get("entry_type") or ""
            turn = row.get("turn_number") or 0
            text_for_embed = f"[turn {turn}] ({etype}) {speaker}: {content}".strip()
            try:
                embedding = get_embedding(text_for_embed)
            except Exception:  # OpenAI key missing, offline, etc.
                continue
            if not any(embedding):
                continue
            await chunks_repo.store_chunks(
                "rp_story",
                source_id,
                [(0, text_for_embed, embedding)],
            )
            indexed += 1
        return Output(success=True, indexed=indexed)

    # ── search ───────────────────────────────────────────────────────────
    async def _search(self, adventure_id: int, query: str, limit: int) -> Output:
        from app.db.repos.embedding_chunks import EmbeddingChunksRepo
        from app.services.embeddings import get_embedding

        try:
            q_emb = get_embedding(query)
        except Exception as e:
            return Output(success=False, error=f"embedding failed: {e}")
        if not any(q_emb):
            return Output(success=False, error="empty query embedding (missing OPENAI_API_KEY?)")

        prefix = f"{adventure_id}:"
        # Fetch a generous pool then filter by adventure_id prefix so we respect adventure scoping.
        raw = await EmbeddingChunksRepo().search(
            q_emb, source_table="rp_story", limit=max(limit * 4, 20), threshold=0.25
        )
        filtered = [r for r in raw if (r.get("source_id") or "").startswith(prefix)]
        results = [
            {
                "source_id": r.get("source_id"),
                "text": r.get("text_preview"),
                "similarity": float(r.get("similarity") or 0.0),
            }
            for r in filtered[:limit]
        ]
        return Output(success=True, results=results)

    # ── summarize ────────────────────────────────────────────────────────
    async def _summarize(self, adventure_id: int, window: str) -> Output:
        from app.db.repos.rp_adventure import StoryLogRepo, SummaryRepo

        # TODO(v4-port): app.telegram.rp_prose not yet ported
        from app.telegram import rp_prose

        if window not in {"recent", "mid", "distant"}:
            return Output(success=False, error=f"unknown window: {window}")

        story_repo = StoryLogRepo()
        max_turn = await story_repo.get_turn_count(adventure_id)
        if max_turn <= 0:
            return Output(success=False, error="no story entries yet")

        if window == "recent":
            lo = max(1, max_turn - (_RECENT_WINDOW * 2))
            hi = max(1, max_turn - _RECENT_WINDOW)
        elif window == "mid":
            lo = max(1, max_turn - _MID_WINDOW)
            hi = max(1, max_turn - _RECENT_WINDOW)
        else:  # distant
            lo = 1
            hi = max(1, max_turn - _MID_WINDOW)

        if lo >= hi:
            return Output(success=True, summary="", error="range too small to summarize")

        entries = await story_repo.get_range(adventure_id, from_turn=lo, to_turn=hi)
        if not entries:
            return Output(success=True, summary="", error="no entries in window")

        rendered = _render_entries(entries)
        system = (
            "You are a story archivist. Summarize the following roleplay log fragment "
            "into a dense, factual recap suitable for later reuse as context. Preserve "
            "specific names, places, objects, promises, wounds, revelations, and "
            "relationship shifts. Omit purple prose. Use 6-12 tight sentences."
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Turns {lo}-{hi} of the adventure (window: {window}).\n\n{rendered}\n\nWrite the summary now."
                ),
            }
        ]
        try:
            summary_text = await rp_prose.generate_prose(system, messages, role="orchestrator")
        except Exception as e:
            return Output(success=False, error=f"summarization failed: {e}")

        summary_text = (summary_text or "").strip()
        if not summary_text:
            return Output(success=False, error="empty summary")

        await SummaryRepo().upsert(
            adventure_id,
            window,
            summary_text,
            covers_from_turn=lo,
            covers_to_turn=hi,
        )
        return Output(success=True, summary=summary_text)

    # ── pin ──────────────────────────────────────────────────────────────
    async def _pin(self, adventure_id: int, turn: int, text: str) -> Output:
        from app.db.repos.rp_adventure import RecallPinRepo

        row = await RecallPinRepo().add(adventure_id, turn, text.strip())
        return Output(success=True, pin=_serialize(row))


def _render_entries(entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for e in entries:
        turn = e.get("turn_number") or 0
        speaker = e.get("speaker") or ""
        etype = e.get("entry_type") or ""
        content = (e.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"[t{turn}] ({etype}) {speaker}: {content}")
    return "\n".join(lines)


def _serialize(row: dict | None) -> dict:
    if not row:
        return {}
    out: dict[str, Any] = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


if __name__ == "__main__":
    RPMemoryTool.run()
