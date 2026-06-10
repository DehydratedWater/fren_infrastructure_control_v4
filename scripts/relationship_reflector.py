#!/usr/bin/env python3
"""Relationship reflector cron entrypoint (weekly, Sunday evening).

The reflection is LLM work, so it lives in the fleet agent
``persona/relationship_reflector`` (defined in app/agents/domains/persona.py,
with an LLM-judge probe gating boilerplate strategy). This wrapper just spawns
that agent the same way the scheduler runs agent jobs — ``spawn_agent``
against the compiled fleet, honoring FREN_MODEL_POSTFIX — and exits non-zero
on failure.

Usage:
    python scripts/relationship_reflector.py
"""

from __future__ import annotations

import asyncio
import os
import sys

AGENT = "persona/relationship_reflector"
TIMEOUT_S = 1700.0  # under the schedule job's 1800s budget


def build_prompt() -> str:
    return (
        "Run the weekly relationship reflection.\n"
        "Gather the week's data (agent_notes initiation:/connection: prefixes, "
        "chat volume + samples, relationship memories, current style lessons). "
        "If there are fewer than ~3 interactions, stop without reflecting. "
        "Otherwise produce the trend-with-evidence reflection and persist it: "
        "write the relationship_strategy agent-note (expires 168h), create new "
        "relationship memories via memory-manager, and add deduplicated "
        "communication_style lessons via lesson-manager. Optionally send a "
        "one-line summary via emit_guidance."
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
        print(f"[relationship_reflector] agent run failed: {result.error}", file=sys.stderr)
        return 1
    print(f"[relationship_reflector] {AGENT} completed")
    return 0


def main() -> None:
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
