"""RALF loop end-to-end smoke probe — answer a question through the full loop.

WHAT THIS TESTS: the production RALF chain (planning → plan_evaluation →
execution → step_evaluator → … completion), self-chaining exactly as in prod
(each agent spawns the next via scripts/ralf_spawn.py), against the SEEDED
autoloop DB. The research question's ground truth spans TWO sources — a chat
canary (bike lock code 7351) and a transcript canary (Noctua NF-A12x25) — so
a passing run proves the loop can retrieve AND synthesize across sources.

HOW IT WORKS:
1. Creates a ralf_processes row directly (what the dispatcher's START does).
2. Runs workflows/twily_ralf_planning synchronously in the autoloop opencode
   workspace; from there the chain drives ITSELF (detached spawns).
3. Polls the DB until the process reaches a terminal status or the wall-clock
   cap; collects every executor output, step log and emitted guidance.
4. Grades the collected output with the framework's FactRecallEvaluator —
   the same evaluator the retrieval suite uses — and prints a verdict.

Usage (env DATABASE_URL must point at the seeded autoloop DB):
    python -m app ralf-smoke [--timeout-min 25] [--question "..."]

Exit code 0 = facts recalled (loop answered correctly); 1 = loop finished
but missed facts; 2 = loop never reached a terminal state in time.
"""

from __future__ import annotations

import asyncio
import time

SMOKE_QUESTION = (
    "Research task: find out (a) what fan model is installed in my server"
    " rack and (b) my bike lock code. Both are somewhere in my stored data —"
    " chat history, saved video transcripts, notes. Report both facts"
    " explicitly in your final summary."
)
SMOKE_FACTS = ["NF-A12x25", "7351"]


async def _run(question: str, timeout_min: float, ws) -> int:
    from src import FactRecallEvaluator, FactSpec
    from src.testing.evaluation import RunContext, evaluate

    from app.db.repos.ralf import (
        RalfProcessesRepo,
        RalfStagesRepo,
        RalfStepAttemptsRepo,
        RalfStepLogsRepo,
    )
    from app.runtime.runner import run_agent_opencode

    procs = RalfProcessesRepo()
    proc_row = await procs.create(user_request=question, content_class="public")
    ralf_id = proc_row["ralf_id"]
    print(f"[ralf-smoke] created ralf {ralf_id!r}")

    print("[ralf-smoke] running planner (chain self-drives from here) ...")
    res = await run_agent_opencode(
        agent_dir=ws, agent_name="workflows/twily_ralf_planning-primary",
        prompt=f"ralf_id={ralf_id}", timeout_s=600,
    )
    if res.error:
        print(f"[ralf-smoke] planner session error: {res.error}")

    deadline = time.monotonic() + timeout_min * 60
    status = ""
    while time.monotonic() < deadline:
        proc = await procs.get(ralf_id)
        status = (proc or {}).get("status", "missing")
        if status in ("completed", "failed", "stuck"):
            break
        await asyncio.sleep(15)
    print(f"[ralf-smoke] final status={status!r}")

    stages = await RalfStagesRepo().list_for_ralf(ralf_id)
    outputs: list[str] = []
    attempts_repo = RalfStepAttemptsRepo()
    logs_repo = RalfStepLogsRepo()
    for st in stages:
        n_attempts = await attempts_repo.count_attempts(ralf_id, st["stage_number"])
        for n in range(1, n_attempts + 1):
            att = await attempts_repo.get(ralf_id, st["stage_number"], n)
            for f in ("executor_output", "result_summary"):
                if att and att.get(f):
                    outputs.append(str(att[f]))
        for lg in await logs_repo.list_for_stage(ralf_id, st["stage_number"]):
            outputs.append(str(lg.get("log_text") or "") + " "
                           + str(lg.get("reasoning") or ""))
    blob = "\n".join(outputs)
    print(f"[ralf-smoke] stages={len(stages)} collected_output_chars={len(blob)}")

    ev = FactRecallEvaluator(
        name="ralf-smoke-recall",
        facts=tuple(FactSpec(any_of=(f,)) for f in SMOKE_FACTS),
    )
    r = evaluate(ev, RunContext(output=blob))
    print(f"[ralf-smoke] {r.evidence} (score={r.score:.2f})")

    if status not in ("completed", "failed", "stuck"):
        print("[ralf-smoke] VERDICT: TIMEOUT — loop never reached a terminal state")
        return 2
    if r.passed and status == "completed":
        print("[ralf-smoke] VERDICT: PASS — loop completed and recalled all facts")
        return 0
    print("[ralf-smoke] VERDICT: FAIL — "
          + ("missed facts" if status == "completed" else f"loop {status}"))
    return 1


def main(argv: list[str]) -> None:
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="app ralf-smoke")
    p.add_argument("--timeout-min", type=float, default=25.0)
    p.add_argument("--question", default=SMOKE_QUESTION)
    args = p.parse_args(argv)
    # The fleet compile drives opencode warm-up sessions through asyncio.run
    # internally, so it must happen OUTSIDE our own event loop.
    from app.agents.improve_live import _ensure_fleet_compiled

    print("[ralf-smoke] compiling fleet workspace ...")
    ws = _ensure_fleet_compiled()
    sys.exit(asyncio.run(_run(args.question, args.timeout_min, ws)))
