#!/usr/bin/env python3
"""Lesson extractor cron entrypoint.

The extraction is LLM work, so it lives in the fleet agent
``support/lesson_extractor`` (defined in app/agents/domains/support.py, with
its own LLM-judge probes gating invented lessons). This wrapper spawns that
agent the same way the scheduler runs agent jobs — ``spawn_agent`` against the
compiled fleet, honoring FREN_MODEL_POSTFIX — and exits non-zero on failure.

Usage:
    python scripts/lesson_extractor.py            # analyze chat since the cursor
    python scripts/lesson_extractor.py --hours 6  # first-run lookback override
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

AGENT = "support/lesson_extractor"
TIMEOUT_S = 570.0  # under the schedule job's 600s budget
DEFAULT_LOOKBACK_HOURS = 3


def build_prompt(hours: int) -> str:
    return (
        "Run the periodic lesson extraction.\n"
        "Read your cursor (agent_notes key lesson_extractor_cursor), fetch the "
        "chat messages since it (or the last "
        f"{hours}h on a first run), extract behavioral lessons from clear "
        "mistakes/corrections only, store them via lesson-manager (deduplicating "
        "against the active lessons), and advance the cursor to the highest "
        "message id you processed."
    )


async def _run(hours: int) -> int:
    from app.telegram.spawn import spawn_agent

    result = await spawn_agent(
        agent=AGENT,
        prompt=build_prompt(hours),
        model_postfix=os.environ.get("FREN_MODEL_POSTFIX", ""),
        trigger="cron",
        timeout_s=TIMEOUT_S,
    )
    if not result.ok:
        print(f"[lesson_extractor] agent run failed: {result.error}", file=sys.stderr)
        return 1
    print(f"[lesson_extractor] {AGENT} completed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Lesson extractor — learn from agent mistakes")
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_LOOKBACK_HOURS,
        help=f"Hours lookback for first run (default: {DEFAULT_LOOKBACK_HOURS})",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.hours)))


if __name__ == "__main__":
    main()
