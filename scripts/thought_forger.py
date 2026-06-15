#!/usr/bin/env python3
"""Thought forger cron entrypoint (every 30 min during waking hours).

The forging is LLM work, so it lives in the fleet agent
``persona/thought_forger`` (defined in app/agents/domains/persona.py, with
LLM-judge probes gating unlisted-interest motivations and re-forged pending
thoughts). This wrapper just spawns that agent the same way the scheduler runs
agent jobs — ``spawn_agent`` against the compiled fleet, honoring
FREN_MODEL_POSTFIX — and exits non-zero on failure.

Usage:
    python scripts/thought_forger.py
"""

from __future__ import annotations

import asyncio
import os
import sys

AGENT = "persona/thought_forger"
# Heavy thinking-on persona agent on the low-priority `-bg` vLLM lane: under GPU
# contention its first token is slow, and 270s starved it (zero output, 100%
# timeouts at the :00/:30 cron herd). Widened to 480s (under the job's 510s
# budget) — the same bump that recovered activity_summarizer (12:15/12:30 ✓).
TIMEOUT_S = 480.0


def build_prompt() -> str:
    return (
        "Run the periodic thought-forging pass.\n"
        "Housekeep the pending_thoughts queue first (expire-thoughts 48h, "
        "trim-thoughts 30, count-thoughts — skip forging if the queue is "
        "full). Then read the top persona interests, the user's current "
        "context, and the unconsumed pending thoughts; forge up to 3 "
        "motivation-scored thoughts bridging a listed interest with a live "
        "user thread, and persist each via persona-memory-manager "
        "create-thought. Never cite unlisted interests; never re-forge a "
        "bridge already pending."
    )


async def _run() -> int:
    from app.telegram.spawn import spawn_agent

    result = await spawn_agent(
        agent=AGENT,
        prompt=build_prompt(),
        model_postfix=os.environ.get("FREN_MODEL_POSTFIX", ""),
        trigger="cron",
        timeout_s=TIMEOUT_S,
    )
    if not result.ok:
        print(f"[thought_forger] agent run failed: {result.error}", file=sys.stderr)
        return 1
    print(f"[thought_forger] {AGENT} completed")
    return 0


def main() -> None:
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
