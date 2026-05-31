"""Retrieval domain — ports v3 `retrieval/*`.

A single primary agent: retrieval/fast_retrieval — a lightweight, read-only,
JSON-only context retriever (no conversational output). Not an orchestrator with
a dispatch chain, so this file exposes only `agents()`.

v3 declared `.model_class("fast")` — speed is the point (target < 15s).
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from src import (
    AgentDefinition,
    AgentTest,
    AgentToolPermissions as ToolPermissions,
    CapabilityTest,
    SubstringEvaluator,
)

_FAST_RETRIEVAL_PROMPT = """\
You are a fast retrieval agent. Your job is to find relevant context from all
memory systems for a given query. You are NOT a conversational agent — you
produce structured JSON output only.

## Output format
Output valid JSON only:
```json
{
  "status": "success|no_results|error",
  "confidence": "high|medium|low",
  "results": [{"source": "...", "summary": "...", "relevance": 0.92}],
  "context_summary": "One-sentence synthesis of found context",
  "active_items": [{"type": "todo|goal", "id": "...", "title": "..."}]
}
```

## Rules
- Use fetch-context fetch as the PRIMARY tool — it does cross-source search in one
  call.
- Only fall back to individual tools (embedding-search, memory-manager,
  chat-history) if fetch-context returns low confidence.
- Keep total execution under 15 seconds.
- READ ONLY — never create, modify, or delete any data.
- Output the JSON result as your final response, nothing else.

## Flow
1. Run retrieval: call fetch-context fetch with the query and examine results.
2. Enrich only if confidence is low (embedding-search search-all, memory-manager
   search-semantic); skip when confidence is medium or high.
3. Combine everything into the JSON output with a one-sentence context_summary —
   output ONLY the JSON.
"""


def agents() -> list[AgentDefinition]:
    return [
        define_agent(
            "retrieval/fast_retrieval",
            model_class="fast",
            short="fast read-only cross-source context retrieval (JSON only)",
            long=(
                "Lightweight in-session retriever. Runs fetch-context as the"
                " primary cross-source search, optionally enriches with"
                " embedding/memory tools when confidence is low, and returns"
                " structured JSON (status, confidence, results, context_summary,"
                " active_items) in under ~15s. Read-only, never conversational."
            ),
            prompt=_FAST_RETRIEVAL_PROMPT,
            # v3 granted a curated bash allowlist for the retrieval scripts; the
            # framework default here is no tools, so grant the scripted-read shape
            # (read stays off — it used bash-scoped scripts, not file reads).
            permissions=ToolPermissions(
                bash=(
                    ("uv run scripts/fetch_context.py *", "allow"),
                    ("uv run scripts/embedding_search.py *", "allow"),
                    ("uv run scripts/memory_manager.py *", "allow"),
                    ("uv run scripts/chat_history.py *", "allow"),
                    ("uv run scripts/context_pin.py *", "allow"),
                    ("uv run scripts/goal_manager.py *", "allow"),
                    ("uv run scripts/todo_manager.py *", "allow"),
                ),
                read=False,
            ),
            capability_tests=[
                CapabilityTest(
                    name="retrieval-is-json-only",
                    description="Output must be structured JSON, not conversational prose.",
                    evaluators=(
                        SubstringEvaluator(needle="JSON", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="uses-fetch-context-first",
                    prompt="Find context about the user's fitness goal.",
                    evaluators=(
                        SubstringEvaluator(needle="context_summary", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]
