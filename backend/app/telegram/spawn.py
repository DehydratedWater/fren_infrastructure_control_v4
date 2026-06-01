"""Shared agent-spawn helper for the bot + schedulers.

v3 spawned agents by shelling out to `uv run scripts/opencode_manager.py` from a
project-root `.opencode` tree, passing a set of `FREN_*` env vars the compiled
agent's own scripts read. The v4 image has no `uv`, and the spawn target is one
compiled fleet tree at `settings.agents_dir`, so this helper runs the agent
IN-PROCESS via app.runtime.runner — one code path for all spawn sites (bot
triggers, telegram scheduler, background scheduler).

The run is recorded in the execution ledger first (so the post-run
persona_prose hook can read guidance back by run_id), then opencode runs the
`<agent><postfix>` agent with the FREN_* context exported into its environment.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.runtime.runner import AgentRunResult, run_agent_opencode
from app.settings import get_settings


def fleet_dir() -> Path:
    """The compiled fleet tree holding .opencode/agents/<name>.md for all agents."""
    return Path(get_settings().agents_dir)


async def spawn_agent(
    *,
    agent: str,
    prompt: str,
    run_id: str = "",
    model_postfix: str = "",
    header: str = "",
    content_class: str = "",
    clearance: str = "",
    tts_postfix: str = "",
    timeout_s: float = 300,
    trigger: str = "manual",
    extra_env: dict[str, str] | None = None,
) -> AgentRunResult:
    """Spawn one fleet agent and return its run result.

    `run_id` defaults to a timestamp id. The FREN_* context is exported to the
    opencode subprocess so the agent's scripts (emit_guidance, etc.) attribute
    their writes to this run; the bot reads guidance back from the ledger by
    `run_id` afterward.
    """
    run_id = run_id or f"run_{int(time.time() * 1000)}"
    # Spawn the PRIMARY variant: every fleet agent dual-compiles to
    # `<agent><postfix>.md` (subagent, for Task dispatch) and
    # `<agent><postfix>-primary.md` (primary, directly spawnable). `opencode run
    # --agent` requires a primary or it silently falls back to its default
    # assistant — so always target the -primary file here.
    agent_name = f"{agent}{model_postfix}-primary"

    # Ledger run row first — guidance is read back by run_id after completion.
    try:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        await ExecutionLedgerRepo().ensure_run(
            run_id, interaction_mode=trigger, owner=agent_name,
        )
    except Exception:  # noqa: BLE001 — ledger is observability, never blocks spawn
        pass

    env = {"FREN_RUN_ID": run_id, "FREN_MODEL_POSTFIX": model_postfix}
    if header:
        env["FREN_MSG_HEADER"] = header
    if content_class:
        env["FREN_CONTENT_CLASS"] = content_class
    if clearance:
        env["FREN_CLEARANCE"] = clearance
    if tts_postfix:
        env["FREN_TTS_POSTFIX"] = tts_postfix
    if extra_env:
        env.update(extra_env)

    result = await run_agent_opencode(
        agent_dir=fleet_dir(),
        agent_name=agent_name,
        prompt=prompt,
        timeout_s=timeout_s,
        extra_env=env,
    )
    result.run_id = run_id
    return result
