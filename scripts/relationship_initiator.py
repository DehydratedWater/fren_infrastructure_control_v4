#!/usr/bin/env python3
"""Relationship initiator cron entrypoint.

The decide-and-compose logic is LLM work, so it lives in the fleet agent
``persona/relationship_initiator`` (defined in app/agents/domains/persona.py,
with LLM-judge probes gating generic/off-context openers and a skip probe).
This wrapper just spawns that agent the same way the scheduler runs agent jobs
— ``spawn_agent`` against the compiled fleet, honoring FREN_MODEL_POSTFIX —
and exits non-zero on failure.

Usage:
    python scripts/relationship_initiator.py
"""

from __future__ import annotations

import asyncio
import os
import sys

AGENT = "persona/relationship_initiator"
TIMEOUT_S = 110.0  # under the schedule job's 120s budget


def build_prompt() -> str:
    return (
        "Run the proactive relationship-initiation check.\n"
        "Apply the gates first (user_busy note, a Twily message in the last "
        "60 minutes, the 3-per-day initiation cap and consecutive-ignored "
        "backoff via the agent_notes initiation: prefix). If a gate fails, "
        "SKIP. Otherwise prefer a curated pending thought (peek-thought, then "
        "consume-thought), else compose a brief opener grounded in the "
        "conversation digest, relationship memories and style lessons; deliver "
        "via emit_guidance and record the initiation in agent_notes."
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
        print(f"[relationship_initiator] agent run failed: {result.error}", file=sys.stderr)
        return 1
    print(f"[relationship_initiator] {AGENT} completed")
    return 0


def main() -> None:
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
