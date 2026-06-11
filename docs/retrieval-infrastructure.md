# Retrieval infrastructure — sources, agents, and the autoloop QA suite

How Twily finds things, how that ability is tested with exact ground truth,
and how to run/extend the retrieval autoloops.

## The retrieval stack (production)

One unified entry point and six sources behind it:

| source | store | reached via |
|--------|-------|-------------|
| context pins | `context_pins` | `fetch-context` (weight 1.5) |
| memories | `memories` (pgvector) | `fetch-context` (1.2) / `memory-manager` |
| artifact cache | `context_cache` | `fetch-context` (1.3) |
| **digital journal** | telegram-log service on **orp** (the orange pi, `192.168.0.80:5050`) | `fetch-context` (1.1) |
| embedding chunks | `embedding_chunks` (transcripts, documents) | `fetch-context` (1.0) / `embedding-search search-transcripts` |
| chat history | `chat_messages` (pgvector) | `fetch-context` (1.0) / `chat-history` / `embedding-search` |

`fetch-context` (backend/app/tools/context/fetch_context.py) fans out to all
sources in parallel (3 s/source, 15 s total), weights, dedupes, ranks, and
returns `{status, confidence, results[], context_summary}`.

**How Twily retrieves:** any agent can call the tools directly, but the
designed path is spawning **`retrieval/fast_retrieval`** (a read-only,
JSON-only subagent with all six tools, <15 s contract) via
`opencode_manager.py run --agent retrieval/fast_retrieval '<question>'`.
`persona/responding` and the RALF executors use exactly this.

**Self-examination:** `session-inspector` (backend/app/tools/system/
session_inspector.py) queries the opencode SQLite store (find-by-text,
find-by-time, session trees); `execution_runs`/`execution_artifacts` hold
agent run traces; the `profile/*` agents run v3-style analysis cycles over
chat history. YouTube transcripts come from SearchAPI.io
(`youtube_fetcher.py` — NOT an MCP server) and persist into
`youtube_videos.transcript` + searchable `embedding_chunks`.

## The QA suite — exact ground truth on a 20k-message haystack

Methodology: the framework's canary method
(OpenCodeCompilerV2/docs/retrieval-testing.md). Everything below runs against
the ISOLATED autoloop DB only.

### 1. Seed the corpus

```bash
# DATABASE_URL must point at <db>_autoloop (run_autoloop.sh does this);
# the seeder REFUSES any other target.
python -m app seed-retrieval            # full: copy + canaries + embeddings
python -m app seed-retrieval --no-embed # fast, skips OpenAI embedding pass
```

What it does (backend/app/agents/retrieval_corpus.py):
- copies the REAL v3 corpus — 19,799 chat messages (Feb–Jun 2026) — from the
  v3 DB (port 5452, opened read-only) into autoloop `chat_messages`;
- plants **10 canary messages** (globally unique facts at known timestamps:
  wifi password `X9-KITE-42`, bike lock `7351`, fan `NF-A12x25`, flight
  `LO1923`, …) — `CANARIES` in the module is the single source of truth,
  each entry carrying its question + expected facts;
- copies 200 full YT transcripts + plants **1 canary transcript** (the
  "silent server rack" video: facts `19 dB`, `612 euro`) with embedded
  chunks, so transcript-path probes can't be answered from chat;
- embeds everything with `text-embedding-3-small` — the SAME encoder prod
  uses, so similarity behaviour matches production.
- idempotent (`metadata.seed_source` markers); re-run any time.

Verified baseline: the bike-lock canary is the top semantic hit (0.69) over
the full 20k haystack; the canary transcript is the top transcript-chunk hit.

### 2. The probe categories (backend/app/agents/retrieval_probes.py)

| category | count | ground truth | evaluators |
|----------|-------|--------------|------------|
| exact | 10 | canary facts | `FactRecallEvaluator` (+ refusal guard) |
| open-ended | 4 | corpus themes (EN+PL) | groundedness judge + refusal guard |
| seeded transcripts | 2 | canary video only | `FactRecallEvaluator` |
| journal (live orp) | 1 | live service | judge: must CHECK, may be empty |
| self-exam | 2 | canary + planted date; own sessions | `FactRecallEvaluator` / judge |

Suite targets: `retrieval/fast_retrieval` (all 19) and `persona/responding`
(6 end-to-end — the answer must survive the emit_guidance delivery path;
the evaluator grades the emitted payload, not assistant text).

### 3. Run the autoloop

```bash
./run_autoloop.sh --retrieval-probes              # tune both suite agents
./run_autoloop.sh --retrieval-probes --agent retrieval/fast_retrieval
```

The suite merges into the standard judged loop (`_judge_test_suite`), so the
teacher rewrites prompts against concrete failures ("missing: 7351") and
per-test metrics land in the snapshots as `score_floor:by_name:retrieval:…`.

### 4. RALF loop end-to-end

```bash
./run_autoloop.sh --refresh   # optional: fresh DB copy first, then reseed
python -m app ralf-smoke      # under run_autoloop.sh env
```

`ralf-smoke` (backend/app/agents/ralf_smoke.py) runs ONE production-faithful
RALF cycle: planner → plan evaluation → executors → step evaluators, chain
self-driving via detached spawns, against the seeded DB. The smoke question
needs **two facts from two different stores** (chat canary `7351` +
transcript canary `NF-A12x25`), so a pass proves cross-source retrieval and
synthesis through the whole loop. Exit 0 = completed + all facts recalled;
1 = finished but missed/failed; 2 = no terminal state before the cap.

The five RALF agents are also individually improvable in the normal fleet
loop (judge tests + corpus packs).

## Extending the suite

- New exact probe: add a `CANARIES` entry (message, ts, facts, question) in
  retrieval_corpus.py, re-run `seed-retrieval` — retrieval_probes picks it up
  automatically (canary_tests() iterates CANARIES).
- New store: plant a canary in that store (the only honest way to force its
  search path) and add a category in retrieval_probes.py.
- Open-ended: extend `_OPEN_ENDED` with (question, groundedness rubric).

## Known issue — session-start blocked tool calls

Most opencode sessions open with 1–3 denied tool calls before the agent
settles into its allow-list. Tracked separately; the tool-discipline signal
(blocked attempts) is already surfaced to the judge/teacher via the runner's
3-tuple, so the autoloop pressures it down over time.
