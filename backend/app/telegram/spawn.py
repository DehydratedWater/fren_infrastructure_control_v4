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
    # Proactive coordination (v3 background-cooldown parity): a cron-triggered
    # run is unsolicited, so tag its inline delivery (emit_guidance ->
    # send_message.py runs INSIDE this agent subprocess and inherits this env)
    # as "proactive" — the delivery gate then suppresses it if the user is
    # actively chatting or the bot just spoke. User-initiated triggers
    # (chat/chatbot/workflow/...) stay "reply" and are never cooldown-gated.
    # An explicit extra_env FREN_MSG_KIND wins over this default.
    if trigger == "cron" and "FREN_MSG_KIND" not in env:
        env["FREN_MSG_KIND"] = "proactive"

    result = await run_agent_opencode(
        agent_dir=fleet_dir(),
        agent_name=agent_name,
        prompt=prompt,
        timeout_s=timeout_s,
        extra_env=env,
        # Cron/proactive runs go to the low-priority vLLM lane so user replies
        # preempt them on the shared :8082 endpoint (see runner.run_agent_opencode).
        background=(trigger == "cron"),
    )
    result.run_id = run_id

    # Close out the ledger run row so it doesn't stay status='running' forever
    # (the dashboard + supersede logic depend on a terminal status). Best-effort:
    # a failure here must NEVER block or fail the agent run.
    try:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        await ExecutionLedgerRepo().complete_run(
            run_id,
            status="completed" if result.ok else "failed",
            contract_passed=result.ok,
        )
    except Exception:  # noqa: BLE001 — ledger is observability, never blocks spawn
        pass

    # Persist the parsed trajectory (assistant output + the ordered tool calls)
    # as ONE `run_trace` artifact so the dashboard can replay what the agent
    # reasoned and which tools/commands it ran. This trajectory is otherwise
    # used only for evaluation and discarded. Best-effort: a failure here must
    # NEVER block or fail the agent run.
    try:
        await _write_run_trace(run_id, result)
    except Exception:  # noqa: BLE001 — trace is observability, never blocks spawn
        pass

    return result


# Cap each free-text field so a chatty run can't bloat the DB. Output text and
# per-tool command/args are truncated to this many chars; the count of tool
# calls is also capped (long flailing loops are summarised, not stored whole).
_TRACE_TEXT_CAP = 4000
_TRACE_MAX_CALLS = 200
# Ordered trajectory caps: each text/command/output field is truncated to the
# text cap, and the timeline itself is capped at this many items (a true count
# is kept). Long flailing loops are summarised, not stored whole.
_TRACE_MAX_TRAJECTORY = 400
# Retention: keep only the newest N run_trace artifacts. The dashboard only ever
# shows recent runs, and traces are written once per agent run (the
# periodic_checker alone fires every 5 min ≈ 288/day), so an unbounded table
# would bloat. ~2000 traces covers roughly a week of churn.
_TRACE_KEEP = 2000


def _truncate(s: str, cap: int = _TRACE_TEXT_CAP) -> str:
    if len(s) <= cap:
        return s
    return s[:cap] + f"… [truncated {len(s) - cap} chars]"


def _tool_call_payload(tc: object) -> dict[str, str | None]:
    """Flatten one ToolCallRecord to a small dict for the trace payload.

    `command` is the bash command an agent ran (the load-bearing arg for the
    dashboard) when present, else a compact dump of the call's args. Both are
    truncated.
    """
    args = getattr(tc, "args", None) or {}
    command = ""
    if isinstance(args, dict):
        command = str(args.get("command") or "")
        if not command and args:
            # Non-bash tool: show its args compactly.
            import json as _json

            try:
                command = _json.dumps(args, default=str)
            except Exception:  # noqa: BLE001
                command = str(args)
    return {
        "name": str(getattr(tc, "name", "") or ""),
        "command": _truncate(command),
        "error": (_truncate(str(tc.error), 200) if getattr(tc, "error", None) else None),
    }


def _trajectory_item_payload(item: dict) -> dict[str, object]:
    """Flatten + truncate one ordered-trajectory item for the trace payload.

    Preserves ``kind`` and the load-bearing fields per kind (text → text; tool →
    name/command; result → output/error/status). All free text is capped so a
    chatty run can't bloat the DB.
    """
    kind = str(item.get("kind") or "")
    if kind == "text":
        return {"kind": "text", "text": _truncate(str(item.get("text") or ""))}
    if kind == "tool":
        return {
            "kind": "tool",
            "name": str(item.get("name") or ""),
            "command": _truncate(str(item.get("command") or "")),
            "error": (_truncate(str(item.get("error")), 200) if item.get("error") else None),
        }
    if kind == "result":
        return {
            "kind": "result",
            "name": str(item.get("name") or ""),
            "output": _truncate(str(item.get("output") or "")),
            "error": (_truncate(str(item.get("error")), 200) if item.get("error") else None),
            "status": str(item.get("status") or ""),
        }
    # unknown kind: keep it minimal but don't drop it
    return {"kind": kind or "unknown"}


async def _write_run_trace(run_id: str, result: AgentRunResult) -> None:
    from app.db.repos.execution_ledger import ExecutionLedgerRepo

    calls = list(getattr(result, "tool_calls", []) or [])
    traj = list(getattr(result, "trajectory", []) or [])
    payload = {
        "text": _truncate(result.text or ""),
        "tool_calls": [_tool_call_payload(tc) for tc in calls[:_TRACE_MAX_CALLS]],
        "tool_call_count": len(calls),
        # Ordered, interleaved timeline (narration → tool → result → …) in stream
        # order. Capped in length with a true count preserved.
        "trajectory": [
            _trajectory_item_payload(it) for it in traj[:_TRACE_MAX_TRAJECTORY]
        ],
        "trajectory_count": len(traj),
        "ok": result.ok,
        "error": _truncate(str(result.error), 200) if result.error else None,
    }
    repo = ExecutionLedgerRepo()
    await repo.write_artifact(run_id, "run_trace", payload, producer="runner")
    # Trim old traces so the table stays bounded (best-effort; not fatal).
    try:
        await repo.prune_artifacts_by_type("run_trace", _TRACE_KEEP)
    except Exception:  # noqa: BLE001
        pass
