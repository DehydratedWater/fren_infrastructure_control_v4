# fren infrastructure control — v4

A clean-room rebuild of `fren_infrastructure_control_v3` on the new
**open-agent-compiler** framework (currently `../OpenCodeCompilerV2`). v4 is the
"final test" of the framework: it must reach **full feature parity** with v3,
be **fully replaceable** (drop-in), be **fully testable**, and — new in v4 —
give **every agent its own auto-improvement loop + tests** and **every
distinguished orchestration branch its own path-test + optimisation**.

## Architecture

Two ways to use a fleet agent, from shared definitions:

- **Workers** — long-running, side-effecting agents compiled to opencode
  artifacts (`.opencode/agents/*.md`) and run fire-and-forget. Provider: **z.ai**
  coding plan.
- **Interactive** — streaming chat agents bound to LangChain on a **local
  qwen** (z.ai can't drive LangChain). Same definitions, resolved through a
  different model profile.

```
backend/app/agents/
  config.py      # ModelPresets, 7 worker variants, the split profile, live profile
  _authoring.py  # define_agent(...) — terse AgentDefinition construction
  domains/       # agents grouped by domain; each exposes agents()
  registry.py    # collect all domains → AgentRegistry (+ promoted-prompt merge)
  compile.py     # compile_fleet() → all variants via CompileScript
  branches.py    # BranchTests: distinguished orchestrator paths
  improve.py     # per-agent + per-branch loops via the framework fleet harness
```

### Per-agent & per-branch self-improvement
Each agent carries embedded `capability_tests` + `agent_tests`. Each orchestrator
contributes `BranchTest`s (expected dispatch chain). `app/agents/improve.py`
turns both into `ImprovementUnit`s and runs them in parallel via the framework's
`run_fleet`, promoting any winner that clears its threshold + hard criteria into
`.oac/promoted/` — picked up on the next compile. Both tiers: a deterministic
mock gates every round; a live `opencode` run precedes promotion.

### Split-profile + vision passthrough
The `splitqwen35` variant routes each agent by `model_class` (fast / analytical)
to co-located vLLM servers, while **vision** agents pass through unchanged
(keep their own model) — the framework's `SplitProfile.passthrough_classes`,
matching v3's `resolve()->None`.

## Status — rebuild in progress

| Phase | Scope | State |
|------|-------|-------|
| P0 | framework prep (branch test+optimise, split parity, fleet harness) | ✅ in OpenCodeCompilerV2 |
| P1 | foundation: config, authoring, registry, compile, branches, improve | ✅ persona slice |
| P2 | tooling parity (ScriptTools / scripts CLI) | ⏳ |
| P3 | agent fleet (~105 agents, all domains) | ⏳ persona done |
| P4 | branches for every orchestrator | ⏳ persona done |
| P5 | surface: Telegram bot, dashboard, persona_prose, fast-path | ⏳ |
| P6 | integrations: vLLM, ComfyUI, STT/TTS, Google, Tuya, Garmin, search | ⏳ |
| P7 | feature-parity audit vs v3 + acceptance | ⏳ |

## Dev

```bash
cd backend
PYTHONPATH=../../OpenCodeCompilerV2:. python -m pytest tests/ -q
```

Docker: `docker compose up` (mounts `../OpenCodeCompilerV2` at `/srv/oac`).
Secrets live in a gitignored `.env` (see `.env.example`).
