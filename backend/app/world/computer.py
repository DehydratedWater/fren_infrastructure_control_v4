"""The in-world computer — Twily's research mechanic.

When the narrator decides Twily sits at her desk to look something up, the world
doesn't hallucinate the answer: it runs a REAL web search through the existing
SearchAPI tool and feeds the actual results back so the "computer responds" with
genuine information she can react to and learn from. This is the bridge that
lets her curiosity touch the real world.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MAX_RESULTS = 5


async def research(query: str) -> dict:
    """Run a real web search for `query`. Returns
    {ok, query, results:[{title,link,snippet}], summary}. Never raises — a failed
    lookup just yields ok=False with an empty result set (the narrator then plays
    it as 'the connection's flaky today')."""
    query = (query or "").strip()
    if not query:
        return {"ok": False, "query": "", "results": [], "summary": ""}

    try:
        from app.tools.research.web_search import Input, WebSearchTool

        tool = WebSearchTool()
        # ScriptTool.execute is sync (wraps asyncio.run); run it off the loop.
        import asyncio

        out = await asyncio.to_thread(
            tool.execute, Input(command="search", query=query, max_results=MAX_RESULTS)
        )
    except Exception:  # noqa: BLE001
        logger.exception("world.computer: search failed for %r", query)
        return {"ok": False, "query": query, "results": [], "summary": ""}

    if not getattr(out, "success", False):
        logger.warning("world.computer: search unsuccessful for %r: %s", query, getattr(out, "error", ""))
        return {"ok": False, "query": query, "results": [], "summary": getattr(out, "error", "")}

    results = [
        {
            "title": str(r.get("title", ""))[:200],
            "link": str(r.get("link", "")),
            "snippet": str(r.get("snippet", ""))[:500],
        }
        for r in (out.items or [])[:MAX_RESULTS]
    ]
    summary = _summarize(results)
    return {"ok": True, "query": query, "results": results, "summary": summary}


def _summarize(results: list[dict]) -> str:
    """A compact plain-text digest the narrator reads back as 'what the screen
    shows'. We keep it factual; the narrator turns it into prose."""
    if not results:
        return "No useful results came back."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "(untitled)"
        snippet = r.get("snippet") or ""
        lines.append(f"{i}. {title} — {snippet}".strip())
    return "\n".join(lines)
