# LangChain first-contact tier (snappy) routing into the opencode suite

Status: PLANNED (design approved 2026-06-15). Implements a fast in-process
first-contact agent in front of the existing opencode agent suite.

## Why

Every conversational turn currently runs a heavy opencode agent (orchestrator /
twily_chat) — a subprocess + a thinking-model generation across multiple tool
turns (~80–215s). Most turns are simple (banter, a quick question, "add a
task"). v3 had a fast tier that answered most turns directly and only escalated.
We restore that as a **LangChain + local-qwen, in-process** first-contact agent
built on the framework's existing `src/interactive` primitive.

## Framework primitive (already exists — reuse, don't rebuild)

`OpenCodeCompilerV2/src/interactive/`:
- `build_interactive_spec(agent_definition, live_profile)` → `InteractiveAgentSpec`
  (model + system_prompt + ToolSpecs) from the SAME `AgentDefinition`.
- `bindings/langchain_binding.py`: `build_chat_model(spec)` → streaming
  `ChatOpenAI` on the live provider; tool conversion.
- `runner.py: run_interactive(...)`: framework-owned in-process tool-calling loop
  that records `ToolCallRecord`s → the SAME autoloop probes/evaluators apply, so
  the first-contact agent is autoloop-optimisable like every other agent.
- `OpenAICompatClient`: local OpenAI-compatible (vLLM qwen) client; handles the
  Qwen3.x empty-content quirk.

So the work is a NEW lightweight agent definition + handoff tools + a v4 runtime
entry + handler wiring. No new runtime engine.

## Decisions (approved)

1. Delivery: first-contact emits PersonaGuidance → `persona_prose` renders it
   (ONE voice renderer everywhere; consistent with the suite). Speed comes from
   first-contact being lightweight + avoiding the heavy orchestrator, not from
   skipping persona_prose.
2. Agent: a NEW lightweight `persona/twily_first_contact` (small prompt distilled
   from v3 twily_chat's conversational core). The rich twily_chat stays as a
   Tier-1 escalation target.
3. Routing is a SPECTRUM, not a single handoff:
   - DIRECT TOOLS — cheap CRUD the FC agent does itself, inline: check tasks,
     add task, edit task, mark done; read recent context. (Reuse the existing
     ScriptTools as ToolSpecs in the interactive runtime.)
   - CALL-AND-WAIT micro-specialists — small specialists FC invokes and BLOCKS
     on for the result (it acts as a micro-orchestrator). e.g. a quick lookup.
   - CONSTRUCT-PROMPT + HAND OFF — for big replanning / research / RALF /
     multi-step, FC writes a PRECISE instruction prompt and dispatches to a
     heavy agent (persona/orchestrator or a task_master-style agent),
     fire-and-forget + a quick ack; the suite delivers the full result later.
4. Local qwen only, no model switching (a "live" SplitProfile pins
   model_class → local-vllm qwen).

## Components to build

A. Live SplitProfile (v4): `model_class → local-vllm-remote/qwen35-27b` for the
   interactive binding. (Worker profile unchanged.)

B. `persona/twily_first_contact` AgentDefinition (v4 domain):
   - Lightweight prompt: persona voice + the routing contract (answer directly
     vs direct-tool vs call-and-wait vs hand-off-heavy). Distilled, NOT 35K.
   - Tools:
     - direct: todo_manager / habit CRUD subset (the cheap ops), recent context.
     - `call_specialist(agent, task, wait=true)` — spawn opencode agent, await,
       return its result inline (micro-orchestrator path).
     - `handoff(agent, instruction)` — construct a precise prompt, spawn
       opencode agent fire-and-forget, return "dispatched" → FC acks.
     - emit_guidance (delivery, per decision 1).
   - Compiled to BOTH targets (interactive runtime + opencode) → autoloop-testable.

C. Handoff tools (v4): thin wrappers over the existing `spawn_agent` /
   opencode_manager path so cooldown/gate/persona_prose still apply downstream.
   `call_specialist` awaits `spawn_agent(...)`; `handoff` fire-and-forgets.

D. v4 runtime entry: `app/agents/first_contact.py` — build the spec from the
   compiled FC definition + live profile, run `run_interactive` on the user
   message with the tool executor wired to B/C. Returns either delivered text
   (via persona_prose) or a dispatch ack.

E. Handler wiring (`_debounce_dispatch`): deterministic media fast-path stays
   first; conversational turns → first-contact (in-process) instead of straight
   to twily_chat/orchestrator. FC answers directly or routes.

F. Tests + autoloop probes: FC route-vs-answer decisions + reply quality, via
   the interactive runner (records ToolCallRecords). Per the
   "autoloop-optimizable or incomplete" rule.

G. Deploy + verify (rebuild bot, smoke a banter turn = fast, a "deep research"
   turn = hands off + acks, an "add a task" turn = direct tool).

## Concrete build notes (APIs verified 2026-06-15)

- Runner: `from src.interactive import run_interactive` (runner.py:288).
  `run_interactive(spec, user_input, *, tool_runner, client=None, sink=None,
  max_tool_rounds=8, history=None, **params) -> RunResult`. `client` defaults to
  `OpenAICompatClient.from_spec(spec)`. `tool_runner` is `(tool_name, args) ->
  result_text` (same shape as the LangChain binding). It records ToolCallRecords.
- Spec: `from src.interactive.spec import build_interactive_spec`.
  `build_interactive_spec(agent=<AgentDefinition>, live_profile=<VariantSpec>,
  output_schema=None)`. Tools come from the agent's ScriptTools (auto → ToolSpec).
- LIVE preset CAVEAT: the worker preset `QWEN35_27B` uses `model_id="qwen35-27b"`
  (the opencode.json KEY). The interactive client calls vLLM DIRECTLY, so the
  LIVE preset's `model_id` MUST be the SERVED id
  `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8`, with `provider_options`
  `{base_url: "http://192.168.0.42:8082/v1", api_key_env: "VLLM_API_KEY"}`
  (= `_VLLM_REMOTE` in app/agents/config.py). Define `QWEN35_27B_LIVE` + a
  `LIVE_PROFILE` SplitProfile returning it for every model_class.
- Qwen quirk handled by OpenAICompatClient (empty content → reads reasoning
  field); reasoning/thinking stays ON.
- Framework `src/interactive` is solid: 43 tests pass.

## Build phases (resume here)
A. config.py: `QWEN35_27B_LIVE` preset + `LIVE_PROFILE`. (low-risk, no hot path)
B. domains: `persona/twily_first_contact` AgentDefinition (lightweight prompt +
   direct-tool subset + call_specialist + handoff + emit_guidance).
C. tools: `call_specialist` (await spawn_agent) + `handoff` (fire-and-forget).
D. `app/agents/first_contact.py`: build spec + run_interactive + tool_runner.
E. handler wiring in `_debounce_dispatch` (PRODUCTION hot path — do carefully).
F. tests + autoloop probes (route-vs-answer).
G. rebuild + deploy + smoke.

## Compatibility

Opencode suite, delivery gate, proactive cooldown, send_message, vLLM priority
lanes — all unchanged; FC sits in FRONT. The escalation targets are the existing
agents. Same AgentDefinition/ToolSpec primitives → same autoloop coverage.
