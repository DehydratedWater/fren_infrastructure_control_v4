"""ralf_spawn — detached hand-off between RALF chain stages.

Accepted forms (all equivalent where it matters):
    python scripts/ralf_spawn.py workflows/twily_ralf_plan_evaluation ralf_id=<id>
    python scripts/ralf_spawn.py --agent workflows/twily_ralf_execution --ralf_id <id> --stage_number 2
    python scripts/ralf_spawn.py --ralf_id <id>          # infer next stage from DB

The inference form exists because models reach for the flag style used by
every other script (ralf_manager --command ... --ralf_id ...) — the
2026-06-11 smoke showed the plan evaluator calling `--ralf_id <id>` with no
agent and stalling the chain on a usage error. When the agent is omitted,
the script reads the process state and spawns whatever the chain needs next:

    status planning     -> workflows/twily_ralf_planning
    status plan_review  -> workflows/twily_ralf_plan_evaluation
    status executing    -> workflows/twily_ralf_execution (first non-approved
                           stage, attempt = prior attempts + 1)

Spawns the next agent as a DETACHED subprocess (start_new_session, output
discarded) via opencode_manager, so the calling session ends immediately.
cwd stays the compiled workspace; env inherits DATABASE_URL etc. and gains
~/.opencode/bin on PATH (the opencode binary).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

CHAIN = {
    "planning": "workflows/twily_ralf_planning",
    "plan_review": "workflows/twily_ralf_plan_evaluation",
}


def _parse(argv: list[str]) -> tuple[str, dict[str, str]]:
    agent = ""
    params: dict[str, str] = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            key = a[2:]
            val = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else ""
            i += 2 if val else 1
            if key == "agent":
                agent = val
            else:
                params[key] = val
        elif "=" in a:
            k, _, v = a.partition("=")
            params[k] = v
            i += 1
        elif "/" in a and not agent:
            agent = a
            i += 1
        elif a.startswith("ralf_") and "ralf_id" not in params:
            params["ralf_id"] = a  # bare ralf id (models do this too)
            i += 1
        else:
            i += 1
    return agent, params


async def _infer(ralf_id: str) -> tuple[str, dict[str, str]]:
    """Read the process and decide which chain agent runs next."""
    import asyncpg

    dsn = os.environ.get("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn)
    try:
        proc = await conn.fetchrow(
            "SELECT status, current_stage FROM ralf_processes WHERE ralf_id=$1", ralf_id)
        if proc is None:
            raise SystemExit(json.dumps({"ok": False, "error": f"no ralf {ralf_id!r}"}))
        status = proc["status"]
        if status in CHAIN:
            return CHAIN[status], {}
        if status in ("executing", "running"):
            stage = await conn.fetchrow(
                "SELECT stage_number FROM ralf_stages WHERE ralf_id=$1"
                " AND status NOT IN ('approved', 'completed')"
                " ORDER BY stage_number LIMIT 1", ralf_id)
            if stage is None:
                raise SystemExit(json.dumps(
                    {"ok": False, "error": "all stages done — nothing to spawn"}))
            n = stage["stage_number"]
            last = await conn.fetchrow(
                "SELECT attempt_number, outcome,"
                " EXTRACT(EPOCH FROM (now() - started_at)) AS age_s"
                " FROM ralf_step_attempts WHERE ralf_id=$1 AND stage_number=$2"
                " ORDER BY attempt_number DESC LIMIT 1", ralf_id, n)
            if last is None:
                return "workflows/twily_ralf_execution", {
                    "stage_number": str(n), "attempt_number": "1"}
            outcome, age = last["outcome"], float(last["age_s"] or 0)
            if outcome == "approved":
                # The attempt is approved but the stage row hasn't flipped yet
                # (auto-chain fires on update-attempt-outcome, BEFORE the
                # evaluator's update-stage-status) — the stage is done; move
                # on. Without this, the race spawned a redundant re-run.
                nxt = await conn.fetchrow(
                    "SELECT stage_number FROM ralf_stages WHERE ralf_id=$1"
                    " AND stage_number > $2 ORDER BY stage_number LIMIT 1",
                    ralf_id, n)
                if nxt is None:
                    raise SystemExit(json.dumps(
                        {"ok": False,
                         "error": "final stage approved — nothing to spawn"}))
                return "workflows/twily_ralf_execution", {
                    "stage_number": str(nxt["stage_number"]),
                    "attempt_number": "1"}
            if outcome == "awaiting_eval":
                return "workflows/twily_ralf_step_evaluator", {
                    "stage_number": str(n),
                    "attempt_number": str(last["attempt_number"])}
            if outcome == "in_progress":
                if age < 1500:  # a live executor session — don't double-spawn
                    raise SystemExit(json.dumps(
                        {"ok": True, "skipped": "attempt in progress"
                         f" ({int(age)}s old) — not spawning a duplicate"}))
                # Executor died mid-attempt (session cap, crash): the step
                # evaluator is the recovery path — it verifies the attempt's
                # ACTUAL artifacts (kv/logs/data) and verdicts approve/retry.
                return "workflows/twily_ralf_step_evaluator", {
                    "stage_number": str(n),
                    "attempt_number": str(last["attempt_number"])}
            # retry (or any settled non-approved outcome) -> next attempt
            return "workflows/twily_ralf_execution", {
                "stage_number": str(n),
                "attempt_number": str(int(last["attempt_number"]) + 1)}
        raise SystemExit(json.dumps(
            {"ok": False, "error": f"status {status!r} is terminal — nothing to spawn"}))
    finally:
        await conn.close()


async def _executor_already_running(ralf_id: str, stage: str) -> bool:
    """Freshness guard for EXPLICIT executor spawns: the tool-layer auto-chain
    and the agent's own backup spawn both fire on a hand-off — without this,
    every stage ran twice (observed live as attempts 4.1 + 4.2)."""
    import asyncpg

    dsn = os.environ.get("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    try:
        conn = await asyncpg.connect(dsn)
    except Exception:  # noqa: BLE001 — no DB, no guard; spawning is the priority
        return False
    try:
        age = await conn.fetchval(
            "SELECT EXTRACT(EPOCH FROM (now() - started_at))"
            " FROM ralf_step_attempts WHERE ralf_id=$1 AND stage_number=$2"
            " AND outcome='in_progress' ORDER BY attempt_number DESC LIMIT 1",
            ralf_id, int(stage))
        return age is not None and float(age) < 1500
    finally:
        await conn.close()


def main() -> int:
    agent, params = _parse(sys.argv[1:])
    ralf_id = params.get("ralf_id", "")
    if not agent:
        if not ralf_id:
            print(json.dumps({"ok": False,
                              "error": "usage: ralf_spawn.py [<agent_id>] --ralf_id <id> [k=v ...]"}))
            return 2
        agent, extra = asyncio.run(_infer(ralf_id))
        params.update(extra)
    elif agent.endswith("twily_ralf_execution") and ralf_id and params.get("stage_number"):
        if asyncio.run(_executor_already_running(ralf_id, params["stage_number"])):
            print(json.dumps({"ok": True,
                              "skipped": "an executor for this stage is already"
                              " running — not spawning a duplicate"}))
            return 0

    prompt = " ".join(f"{k}={v}" for k, v in params.items())

    # The detached child must find the opencode binary: agent sessions get
    # ~/.opencode/bin injected via _branch_env, but a fresh Popen env doesn't.
    env = dict(os.environ)
    home = os.path.expanduser("~")
    env["PATH"] = ":".join([f"{home}/.opencode/bin", f"{home}/.local/bin",
                            env.get("PATH", "")])

    child = subprocess.Popen(  # noqa: S603 — fixed argv, params passed as one prompt arg
        [
            sys.executable, "scripts/opencode_manager.py",
            "--command", "run",
            "--agent", agent,
            "--prompt", prompt,
            "--timeout", "1800",
        ],
        cwd=os.getcwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(json.dumps({"ok": True, "spawned": agent, "prompt": prompt, "pid": child.pid}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
