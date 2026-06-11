"""OpenCode session manager — spawn fleet agents, capture artifacts (v4).

The agent-spawn entrypoint the Telegram bot + scheduler shell out to. Faithful
to v3's contract (run | list | stop | logs; run_id; model_postfix; the
XDG_DATA_HOME + PWD + stdin=DEVNULL + --format json invariants) but spawns from
v4's compiled fleet tree at `settings.agents_dir` (one tree holding all 137
agents) and records the run in the execution ledger so the bot's post-run
persona_prose hook can read guidance back.

The low-level subprocess + event-stream parse is delegated to
app.runtime.runner (single source of truth for the opencode invariants).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from pydantic import BaseModel, Field
from src import ScriptTool

from app.runtime.runner import run_agent_opencode
from app.settings import get_settings


class Input(BaseModel):
    command: str = Field(default="run", description="run | list | stop | logs")
    agent: str = Field(default="", description="Agent name to spawn")
    prompt: str = Field(default="", description="Prompt for the agent")
    run_id: str = Field(default="", description="Run ID for tracking")
    session_id: str = Field(default="", description="Session ID for logs")
    timeout: int = Field(default=300, description="Timeout in seconds")
    model_postfix: str = Field(default="", description="Model variant postfix")


class Output(BaseModel):
    ok: bool
    command: str
    result: dict = Field(default_factory=dict)


def _fleet_dir() -> Path:
    """The compiled fleet tree (holds .opencode/agents/<name>.md for all agents).

    settings.agents_dir is the CONTAINER volume path (/data/agents); on host
    runs (RALF chain hand-offs, autoloop workspace) it doesn't exist — fall
    back to the caller's cwd when that is itself a compiled fleet tree. This
    is what lets ralf_spawn.py work identically in prod and in the loop.
    """
    configured = Path(get_settings().agents_dir)
    if configured.is_dir():
        return configured
    cwd = Path.cwd()
    if (cwd / ".opencode" / "agents").is_dir():
        return cwd
    return configured


async def _spawn(inp: Input) -> dict:
    # Target the -primary variant (see app/telegram/spawn.py): opencode `run
    # --agent` needs a primary-mode agent or it falls back to its default.
    agent_name = f"{inp.agent}{inp.model_postfix}-primary"
    run_id = inp.run_id or f"run_{int(time.time() * 1000)}"
    agent_dir = _fleet_dir()

    # Record run start in the ledger (best-effort; the bot reads guidance back).
    try:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        await ExecutionLedgerRepo().ensure_run(
            run_id, interaction_mode="worker", owner=agent_name,
        )
    except Exception:  # noqa: BLE001 — ledger is observability, never blocks spawn
        pass

    result = await run_agent_opencode(
        agent_dir=agent_dir,
        agent_name=agent_name,
        prompt=inp.prompt,
        timeout_s=float(inp.timeout),
    )
    status = "completed" if result.ok else (
        "timeout" if result.error and "timeout" in result.error else "error"
    )
    return {
        "run_id": run_id,
        "status": status,
        "agent": agent_name,
        "text": result.text,
        "tool_calls": [c.name for c in result.tool_calls],
        "error": result.error,
    }


async def _list_sessions() -> dict:
    sessions_dir = (
        _fleet_dir() / ".opencode" / "data" / "opencode" / "storage" / "session"
    )
    sessions: list[dict] = []
    if sessions_dir.exists():
        for f in sorted(sessions_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                sessions.append({"id": data.get("id"), "title": data.get("title")})
            except (OSError, json.JSONDecodeError):
                continue
    return {"sessions": sessions}


async def _logs(inp: Input) -> dict:
    try:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        artifacts = await ExecutionLedgerRepo().list_artifacts(inp.run_id)
        return {"run_id": inp.run_id, "artifacts": artifacts}
    except Exception as exc:  # noqa: BLE001
        return {"run_id": inp.run_id, "artifacts": [], "error": str(exc)[:200]}


class OpenCodeManagerTool(ScriptTool[Input, Output]):
    name = "opencode-manager"
    description = "Spawn and manage OpenCode agent sessions"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        if inp.command == "run":
            res = await _spawn(inp)
            return Output(ok=res.get("status") == "completed", command="run", result=res)
        if inp.command == "list":
            return Output(ok=True, command="list", result=await _list_sessions())
        if inp.command == "logs":
            return Output(ok=True, command="logs", result=await _logs(inp))
        if inp.command == "stop":
            # No persistent process registry across invocations; opencode runs
            # are bounded by their own timeout. Report no-op for parity.
            return Output(ok=True, command="stop", result={"stopped": []})
        return Output(ok=False, command=inp.command, result={"error": "unknown command"})
