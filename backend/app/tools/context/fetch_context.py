"""Unified cross-source retrieval tool — fast heuristic search across all memory systems."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta

import httpx
from src import ScriptTool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Stop words for keyword extraction
# ---------------------------------------------------------------------------
STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "was",
        "did",
        "we",
        "i",
        "me",
        "my",
        "what",
        "how",
        "when",
        "about",
        "that",
        "this",
        "it",
        "for",
        "of",
        "to",
        "in",
        "on",
        "at",
        "do",
        "you",
        "remember",
        "tell",
        "are",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "does",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "and",
        "or",
        "but",
        "not",
        "no",
        "so",
        "if",
        "then",
        "than",
        "with",
        "from",
        "by",
        "up",
        "out",
        "any",
        "all",
        "some",
    }
)

# Source weights applied BEFORE dedup so cross-source matches keep the boost
SOURCE_WEIGHTS: dict[str, float] = {
    "context_pins": 1.5,
    "memories": 1.2,
    "context_cache": 1.3,
    "telegram_log": 1.1,
    "embedding_chunks": 1.0,
    "chat_messages": 1.0,
}

STEP_TIMEOUT = 3.0  # seconds per retrieval step
TELEGRAM_LOG_API = "http://192.168.0.80:5050"
TELEGRAM_LOG_DAYS = 7  # how many days back to search


# ---------------------------------------------------------------------------
# Input / Output models
# ---------------------------------------------------------------------------
class Input(BaseModel):
    command: str = Field(description="fetch")
    query: str = Field(default="", description="What to search for")
    max_results: int = Field(default=10, description="Max results to return")
    threshold: float = Field(default=0.3, description="Min relevance threshold")


class Output(BaseModel):
    success: bool = True
    status: str = ""  # "success" | "no_results" | "error"
    confidence: str = ""  # "high" | "medium" | "low"
    results: list[dict] = Field(default_factory=list)
    context_summary: str = ""
    active_items: list[dict] = Field(default_factory=list)
    sources_queried: int = 0
    total_results: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Pure helper functions (no I/O — easily unit-testable)
# ---------------------------------------------------------------------------
def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a natural-language query."""
    # Grab quoted phrases first
    quoted = re.findall(r'"([^"]+)"', query)
    # Strip quotes from the remaining text before splitting
    remaining = re.sub(r'"[^"]*"', "", query)
    words = remaining.lower().split()
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return list(dict.fromkeys(keywords + quoted))[:7]  # dedupe, preserve order


def _apply_weights(results: list[dict]) -> list[dict]:
    """Apply source-specific weight multipliers to relevance scores."""
    out = []
    for r in results:
        w = SOURCE_WEIGHTS.get(r.get("source", ""), 1.0)
        out.append({**r, "relevance": round(r.get("relevance", 0.0) * w, 4)})
    return out


def _deduplicate(results: list[dict]) -> list[dict]:
    """Keep the highest-scoring entry per (source, source_id) pair."""
    best: dict[tuple[str, str], dict] = {}
    for r in results:
        key = (r.get("source", ""), str(r.get("source_id", "")))
        existing = best.get(key)
        if existing is None or r.get("relevance", 0) > existing.get("relevance", 0):
            best[key] = r
    return sorted(best.values(), key=lambda x: x.get("relevance", 0), reverse=True)


def _compute_confidence(results: list[dict]) -> str:
    """Determine confidence level based on result score distribution.

    Calibrated for text-embedding-3-small where relevant results cluster 0.2-0.5.
    """
    high_scores = [r for r in results if r.get("relevance", 0) > 0.45]
    med_scores = [r for r in results if r.get("relevance", 0) > 0.35]
    if len(high_scores) >= 3:
        return "high"
    if len(med_scores) >= 1:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Normalizers — convert repo-specific dicts to common schema
# ---------------------------------------------------------------------------
def _norm_embedding(row: dict) -> dict:
    return {
        "source": "embedding_chunks",
        "source_id": f"{row.get('source_table', '')}:{row.get('source_id', '')}:{row.get('chunk_index', '')}",
        "text": row.get("text_preview", row.get("chunk_text", "")),
        "relevance": row.get("similarity", row.get("score", 0.0)),
        "metadata": {
            "source_table": row.get("source_table", ""),
            "source_id": row.get("source_id", ""),
        },
    }


def _norm_memory(row: dict) -> dict:
    return {
        "source": "memories",
        "source_id": str(row.get("memory_id", row.get("id", ""))),
        "text": f"{row.get('title', '')}\n{row.get('content', '')}".strip(),
        "relevance": row.get("similarity", row.get("score", 0.0)),
        "metadata": {
            "tags": row.get("tags", []),
            "category": row.get("category", ""),
        },
    }


def _norm_pin(row: dict, relevance: float) -> dict:
    return {
        "source": "context_pins",
        "source_id": str(row.get("id", "")),
        "text": row.get("content", row.get("summary", "")),
        "relevance": relevance,
        "metadata": {"topic_id": row.get("topic_id", "")},
    }


def _norm_chat(row: dict, relevance: float) -> dict:
    return {
        "source": "chat_messages",
        "source_id": str(row.get("id", "")),
        "text": row.get("message", row.get("content", "")),
        "relevance": relevance,
        "metadata": {"sender": row.get("sender", ""), "created_at": str(row.get("timestamp", ""))},
    }


def _norm_cache(row: dict, relevance: float) -> dict:
    return {
        "source": "context_cache",
        "source_id": str(row.get("cache_id", "")),
        "text": row.get("summary", ""),
        "relevance": relevance,
        "metadata": {
            "artifact_type": row.get("artifact_type", ""),
            "entity_type": row.get("entity_type", ""),
            "entity_id": row.get("entity_id", ""),
            "file_path": row.get("file_path", ""),
            "tags": row.get("tags", []),
        },
    }


def _norm_telegram(msg: dict, relevance: float) -> dict:
    text = msg.get("message_text", "")
    entities = msg.get("entities") or []
    urls = [e["extracted_text"] for e in entities if e.get("entity_type") in ("url", "text_link")]
    if urls:
        text = text + "\nURLs: " + ", ".join(urls)
    author = msg.get("message_author", {})
    return {
        "source": "telegram_log",
        "source_id": str(msg.get("message_id", msg.get("id", ""))),
        "text": text,
        "relevance": relevance,
        "metadata": {
            "time": msg.get("message_time", ""),
            "author": author.get("nickname") or author.get("full_name", ""),
            "hashtags": [e["extracted_text"] for e in entities if e.get("entity_type") == "hashtag"],
            "urls": urls,
        },
    }


def _build_context_summary(results: list[dict], active_items: list[dict]) -> str:
    """Build a one-sentence context summary from top results — no LLM needed."""
    if not results and not active_items:
        return ""
    parts: list[str] = []
    # Summarize sources found
    sources = {}
    for r in results:
        src = r.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    source_parts = []
    _labels = {
        "memories": ("memory", "memories"),
        "chat_messages": ("chat message", "chat messages"),
        "embedding_chunks": ("document chunk", "document chunks"),
        "context_pins": ("pinned item", "pinned items"),
        "context_cache": ("cached artifact", "cached artifacts"),
        "telegram_log": ("telegram log entry", "telegram log entries"),
    }
    for src, count in sources.items():
        singular, plural = _labels.get(src, (src, src + "s"))
        source_parts.append(f"{count} {singular if count == 1 else plural}")
    if source_parts:
        parts.append(f"Found {', '.join(source_parts)}")
    # Top result preview
    if results:
        top = results[0]
        text = top.get("text", "")
        if len(text) > 120:
            text = text[:120].rsplit(" ", 1)[0] + "..."
        parts.append(f"top hit ({top.get('source', '')}, {top.get('relevance', 0):.2f}): {text}")
    # Active items count
    todos = [i for i in active_items if i.get("type") == "todo"]
    goals = [i for i in active_items if i.get("type") == "goal"]
    item_parts = []
    if todos:
        item_parts.append(f"{len(todos)} related todo{'s' if len(todos) > 1 else ''}")
    if goals:
        item_parts.append(f"{len(goals)} related goal{'s' if len(goals) > 1 else ''}")
    if item_parts:
        parts.append(f"plus {', '.join(item_parts)}")
    return "; ".join(parts) + "." if parts else ""


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------
class FetchContextTool(ScriptTool[Input, Output]):
    name = "fetch_context"
    description = "Unified retrieval from all memory systems — fast cross-source search"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        if inp.command == "fetch":
            try:
                return await asyncio.wait_for(self._fetch_pipeline(inp), timeout=15.0)
            except TimeoutError:
                return Output(success=True, status="error", error="Retrieval timed out after 15s")
            except Exception as exc:
                return Output(success=False, status="error", error=str(exc))

        if inp.command == "fetch-deep":
            return Output(success=False, status="error", error="fetch-deep not yet implemented")

        return Output(success=False, status="error", error=f"Unknown command: {inp.command}")

    async def _fetch_pipeline(self, inp: Input) -> Output:
        keywords = _extract_keywords(inp.query)
        if not keywords:
            return Output(
                success=True,
                status="no_results",
                confidence="low",
                error="Could not extract meaningful keywords from query",
            )

        all_results: list[dict] = []
        sources_queried = 0
        query_vec: list[float] | None = None

        # Step 1: Context pin check
        pin_results, pin_ok = await self._step_pins(keywords)
        if pin_ok:
            sources_queried += 1
        all_results.extend(pin_results)

        # Step 2: Get embedding (reused in Steps 3 & 4)
        query_vec = await self._get_embedding(inp.query)

        # Step 3: Embedding search (cross-table)
        if query_vec:
            emb_results, emb_ok = await self._step_embeddings(
                query_vec,
                limit=inp.max_results,
                threshold=inp.threshold,
            )
            if emb_ok:
                sources_queried += 1
            all_results.extend(emb_results)

        # Step 4: Memory hybrid search
        if query_vec:
            mem_results, mem_ok = await self._step_memory_hybrid(
                query_vec,
                keywords,
                limit=inp.max_results,
            )
            if mem_ok:
                sources_queried += 1
            all_results.extend(mem_results)

        # Step 3.5: Context cache (recent artifacts — YouTube, documents, screenshots, etc.)
        cache_results, cache_ok = await self._step_context_cache(keywords)
        if cache_ok:
            sources_queried += 1
        all_results.extend(cache_results)

        # Step 4.5: Telegram log (last 7 days — links, hashtags, notes)
        tg_results, tg_ok = await self._step_telegram_log(keywords)
        if tg_ok:
            sources_queried += 1
        all_results.extend(tg_results)

        # Step 5: Recent chat history scan (72h)
        chat_results, chat_ok = await self._step_chat_history(keywords)
        if chat_ok:
            sources_queried += 1
        all_results.extend(chat_results)

        # Active items (todos/goals)
        active_items = await self._step_active_items(keywords)
        sources_queried += 1  # counts as a source even if empty

        # Merge, weight, dedup, rank
        weighted = _apply_weights(all_results)
        deduped = _deduplicate(weighted)
        top = deduped[: inp.max_results]
        confidence = _compute_confidence(top)
        context_summary = _build_context_summary(top, active_items)

        status = "success" if top else "no_results"
        return Output(
            success=True,
            status=status,
            confidence=confidence,
            results=top,
            context_summary=context_summary,
            active_items=active_items,
            sources_queried=sources_queried,
            total_results=len(top),
        )

    # ------------------------------------------------------------------
    # Retrieval steps — each wrapped in try/except + timeout
    # ------------------------------------------------------------------

    async def _get_embedding(self, text: str) -> list[float] | None:
        try:
            from app.services.embeddings import get_embedding

            emb = await asyncio.wait_for(
                asyncio.to_thread(get_embedding, text),
                timeout=STEP_TIMEOUT,
            )
            if all(v == 0.0 for v in emb[:10]):
                return None
            return emb
        except Exception:
            return None

    async def _step_pins(self, keywords: list[str]) -> tuple[list[dict], bool]:
        try:
            from app.db.repos.context_pins import ContextPinsRepo

            repo = ContextPinsRepo()
            active_topic = await asyncio.wait_for(repo.get_active_topic(), timeout=STEP_TIMEOUT)
            if not active_topic:
                return [], True
            pins = await asyncio.wait_for(
                repo.get_pins(active_topic["id"]),
                timeout=STEP_TIMEOUT,
            )
            # Score pins by keyword overlap with topic name + pin content
            topic_text = (active_topic.get("name", "") + " " + active_topic.get("summary", "")).lower()
            results = []
            for pin in pins:
                pin_text = (pin.get("content", "") + " " + pin.get("summary", "")).lower()
                combined = topic_text + " " + pin_text
                matches = sum(1 for kw in keywords if kw.lower() in combined)
                if matches > 0:
                    relevance = min(0.5 + (matches / len(keywords)) * 0.5, 1.0)
                    results.append(_norm_pin(pin, relevance))
            return results, True
        except Exception:
            return [], False

    async def _step_embeddings(
        self,
        query_vec: list[float],
        *,
        limit: int,
        threshold: float,
    ) -> tuple[list[dict], bool]:
        try:
            from app.db.repos.embedding_chunks import EmbeddingChunksRepo

            repo = EmbeddingChunksRepo()
            rows = await asyncio.wait_for(
                repo.search(query_vec, limit=limit, threshold=threshold),
                timeout=STEP_TIMEOUT,
            )
            return [_norm_embedding(r) for r in rows], True
        except Exception:
            return [], False

    async def _step_memory_hybrid(
        self,
        query_vec: list[float],
        keywords: list[str],
        *,
        limit: int,
    ) -> tuple[list[dict], bool]:
        try:
            from app.db.repos.memories import MemoriesRepo

            repo = MemoriesRepo()
            rows = await asyncio.wait_for(
                repo.search_hybrid(query_vec, keywords, limit=limit, threshold=0.2),
                timeout=STEP_TIMEOUT,
            )
            return [_norm_memory(r) for r in rows], True
        except Exception:
            return [], False

    async def _step_telegram_log(self, keywords: list[str]) -> tuple[list[dict], bool]:
        """Search the user's personal Telegram log (last N days) for keyword matches."""
        try:
            pattern = re.compile(
                r"(?:" + "|".join(re.escape(kw) for kw in keywords) + r")",
                re.I,
            )
            results: list[dict] = []
            # Use a generous timeout for the full telegram log scan
            tg_timeout = 5.0
            async with httpx.AsyncClient(base_url=TELEGRAM_LOG_API, timeout=tg_timeout) as client:
                end = datetime.now(UTC)
                start = end - timedelta(days=TELEGRAM_LOG_DAYS)
                resp = await client.post(
                    "/stored_data/fetch_blocks_by_date",
                    json={
                        "start_date": start.strftime("%Y-%m-%d"),
                        "end_date": end.strftime("%Y-%m-%d"),
                    },
                )
                resp.raise_for_status()
                blocks = resp.json().get("blocks", [])
                if not blocks:
                    return [], True

                # Fetch ALL blocks in batches of 30
                all_block_ids = [b["message_block_id"] for b in blocks]
                for i in range(0, len(all_block_ids), 30):
                    batch = all_block_ids[i : i + 30]
                    resp = await client.post(
                        "/stored_data/fetch_messages_by_blocks",
                        json={"message_block_ids": batch},
                    )
                    resp.raise_for_status()
                    for block in resp.json().get("blocks", []):
                        for msg in block.get("messages", []):
                            text = msg.get("message_text", "") or ""
                            entities = msg.get("entities") or []
                            urls = " ".join(
                                e.get("extracted_text", "") or ""
                                for e in entities
                                if e.get("entity_type") in ("url", "text_link")
                            )
                            searchable = text + " " + urls
                            found = pattern.findall(searchable)
                            if found:
                                relevance = min(0.3 + (len(found) / len(keywords)) * 0.4, 0.9)
                                results.append(_norm_telegram(msg, relevance))

            return results, True
        except Exception:
            return [], False

    async def _step_context_cache(self, keywords: list[str]) -> tuple[list[dict], bool]:
        try:
            from app.db.repos.context_cache import ContextCacheRepo

            repo = ContextCacheRepo()
            # Two search paths: tag overlap + keyword search on summaries
            results: list[dict] = []
            seen_ids: set[str] = set()

            # Tag search (72h window)
            tag_rows = await asyncio.wait_for(
                repo.list_by_tags(keywords, hours=72, limit=10),
                timeout=STEP_TIMEOUT,
            )
            for row in tag_rows:
                cid = row.get("cache_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    tags = row.get("tags", [])
                    overlap = sum(1 for kw in keywords if kw.lower() in [t.lower() for t in tags])
                    relevance = min(0.4 + (overlap / max(len(keywords), 1)) * 0.4, 0.9)
                    results.append(_norm_cache(row, relevance))

            # Keyword search on summaries (72h window)
            for kw in keywords[:3]:  # limit to top 3 keywords to stay fast
                kw_rows = await asyncio.wait_for(
                    repo.search(kw, hours=72, limit=5),
                    timeout=STEP_TIMEOUT,
                )
                for row in kw_rows:
                    cid = row.get("cache_id", "")
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        results.append(_norm_cache(row, 0.4))

            return results, True
        except Exception:
            return [], False

    async def _step_chat_history(self, keywords: list[str]) -> tuple[list[dict], bool]:
        try:
            from app.db.repos.chat import ChatMessagesRepo

            repo = ChatMessagesRepo()
            recent = await asyncio.wait_for(
                repo.get_since_hours(hours=72, limit=200),
                timeout=STEP_TIMEOUT,
            )
            pattern = re.compile(
                r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b",
                re.I,
            )
            results = []
            for msg in recent:
                content = msg.get("message", msg.get("content", ""))
                found = pattern.findall(content)
                if found:
                    relevance = min(0.3 + (len(found) / len(keywords)) * 0.4, 0.9)
                    results.append(_norm_chat(msg, relevance))
            return results, True
        except Exception:
            return [], False

    async def _step_active_items(self, keywords: list[str]) -> list[dict]:
        items: list[dict] = []
        try:
            from app.db.repos.todos import TodosRepo

            todos = await asyncio.wait_for(
                TodosRepo().search_by_keyword(keywords, status="pending", limit=5),
                timeout=STEP_TIMEOUT,
            )
            for t in todos:
                items.append({"type": "todo", "id": t.get("todo_id", ""), "title": t.get("title", "")})
        except Exception:
            pass
        try:
            from app.db.repos.goals import GoalsRepo

            goals = await asyncio.wait_for(
                GoalsRepo().search_by_keyword(keywords, limit=5),
                timeout=STEP_TIMEOUT,
            )
            for g in goals:
                items.append({"type": "goal", "id": g.get("goal_id", ""), "title": g.get("title", "")})
        except Exception:
            pass
        return items


if __name__ == "__main__":
    FetchContextTool.run()
