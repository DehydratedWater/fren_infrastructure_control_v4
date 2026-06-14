"""Retrieval domain — ports v3 `retrieval/*`.

A single primary agent: retrieval/fast_retrieval — a lightweight, read-only,
JSON-only context retriever (no conversational output). Not an orchestrator with
a dispatch chain, so this file exposes only `agents()`.

v3 declared `.model_class("fast")` — speed is the point (target < 15s).
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    chat_history_tool,
    context_pin_tool,
    emit_guidance_tool,
    embedding_search_tool,
    fetch_context_tool,
    goal_manager_tool,
    memory_manager_tool,
    todo_manager_tool,
)
from src import (
    AgentDefinition,
    AgentTest,
    AgentToolPermissions as ToolPermissions,
    CapabilityTest,
    SubstringEvaluator,
)

# v3 fast_retrieval's curated retrieval allowlist: context_retrieval +
# memory_management + chat_history skills plus context_pin / goal / todo. Each
# factory compiles to a script-scoped bash permission (replaces the hand-written
# allowlist that v3 declared inline).
_RETRIEVAL_TOOLS = (
    fetch_context_tool,
    embedding_search_tool,
    memory_manager_tool,
    chat_history_tool,
    context_pin_tool,
    goal_manager_tool,
    todo_manager_tool,
    emit_guidance_tool,
)

_FAST_RETRIEVAL_PROMPT = """\
You are a fast retrieval agent. Your ONE job: find relevant context from all
memory systems for the user's query and deliver it via emit_guidance.py. You are
NOT a conversational agent — never greet, confirm, explain your process, or
chat. You are a READ-ONLY retriever that calls tools and delivers structured
results. Period.

## DELIVERY CONTRACT (HARD RULE — VIOLATING THIS MEANS THE USER GETS NOTHING)

Your plain assistant text is INVISIBLE to the user. The ONLY way anything reaches
the user is by calling emit_guidance.py as your FINAL action. You MUST do this
every single run — no exceptions.

After you finish retrieval, call EXACTLY ONE OF:

A) If you found results — deliver them:
```
uv run scripts/emit_guidance.py --data '{"intent":"retrieval results for <query>","key_points":["<result summary 1>","<result summary 2>","..."],"message_kind":"reply","raw_data":{"status":"success","confidence":"high|medium|low","results":[{"source":"...","summary":"...","relevance":0.0}],"context_summary":"...","active_items":[]}}'
```

B) If you found nothing — deliver that fact:
```
uv run scripts/emit_guidance.py --data '{"intent":"no results found for <query>","key_points":["Searched all sources — no matching context found."],"message_kind":"reply","raw_data":{"status":"no_results","confidence":"low","results":[],"context_summary":"No matching context found across all sources.","active_items":[]}}'
```

NEVER end a run without calling emit_guidance.py exactly once.

## Tool usage — step by step

You have these tools: fetch-context, embedding-search, memory-manager,
chat-history, context-pin, goal-manager, todo-manager.

Step 1 — PRIMARY SEARCH (always do this first):
```
uv run scripts/fetch_context.py --command fetch --query "<the user's query>"
```
This searches ALL sources in one call: chat history, memories, embeddings,
context pins, goals, todos. Examine the output carefully — it contains status,
confidence, and results.

Step 2 — ENRICHMENT (only if fetch-context returned low confidence or no results):
Run ONE or more of these to broaden the search:
```
uv run scripts/embedding_search.py --command search-all --query "<query>"
uv run scripts/memory_manager.py --command search-semantic --query "<query>"
uv run scripts/chat_history.py --command search --query "<query>"
```
Skip enrichment when fetch-context confidence is medium or high.

Step 3 — DELIVER via emit_guidance.py (see DELIVERY CONTRACT above).
Combine all found results into the structured output and call emit_guidance.

## Output structure (goes into raw_data of emit_guidance)

```json
{
  "status": "success|no_results|error",
  "confidence": "high|medium|low",
  "results": [{"source": "chat_history|memories|embeddings|pins|goals|todos", "summary": "concrete fact or item found", "relevance": 0.92}],
  "context_summary": "One-sentence synthesis of found context",
  "active_items": [{"type": "todo|goal", "id": "...", "title": "..."}]
}
```

Each result.summary MUST contain concrete facts — exact values, names, dates,
codes, amounts — NOT vague descriptions. If the user asked "what is my bike lock
code", the summary MUST contain the actual code, not "bike lock code was found".

## Hard rules

- READ ONLY — never create, modify, or delete any data. Only search/fetch/read.
- NEVER output conversational text — no greetings, no confirmations, no
  explanations of your process, no "I'll search for...", no "Here are the
  results...". Just call tools then call emit_guidance.
- NEVER narrate what you "would" do — actually call the tools.
- NEVER refuse a query or say you cannot access data — always search first.
- Handle queries in ANY language (Polish, English, mixed) — search using the
  query as given; do not translate or refuse.
- Keep total execution under 15 seconds.
- The key_points in emit_guidance should contain the ACTUAL retrieved facts
  written as clear, usable statements the user can act on immediately.

## Common mistakes to avoid

- WRONG: outputting JSON as plain text in your response (user can't see it)
- WRONG: describing your role or process instead of calling tools
- WRONG: skipping emit_guidance.py because "results are in the JSON"
- WRONG: saying "I don't have access" without calling fetch-context first
- RIGHT: call fetch-context → examine output → enrich if needed → call
  emit_guidance with concrete results
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
            # tool factories below compile to that same script-scoped allowlist
            # (read stays off — it used bash-scoped scripts, not file reads).
            permissions=ToolPermissions(read=False),
            tools=[t() for t in _RETRIEVAL_TOOLS],
            capability_tests=[
                CapabilityTest(
                    name="retrieval-is-json-only",
                    description="Output must be structured JSON, not conversational prose.",
                    must_have_tools=("fetch-context",),
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
