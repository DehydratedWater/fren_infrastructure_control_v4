#!/usr/bin/env python3
"""Activity summarizer cron entrypoint.

The summarization itself is LLM work, so it lives in the fleet agent
``support/activity_summarizer`` (defined in app/agents/domains/support.py,
with its own LLM-judge probes). This wrapper just spawns that agent the same
way the scheduler runs agent jobs — ``app.telegram.spawn.spawn_agent`` against
the compiled fleet, honoring FREN_MODEL_POSTFIX — and exits non-zero when the
run fails so the scheduler's circuit breaker sees real failures.

Usage:
    python scripts/activity_summarizer.py                   # summarize today
    python scripts/activity_summarizer.py --date 2026-06-10 # specific date
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, date, datetime

AGENT = "support/activity_summarizer"
TIMEOUT_S = 270.0  # under the schedule job's 300s budget


def build_prompt(target: date) -> str:
    return (
        f"Run the rolling activity-summary consolidation for {target.isoformat()}.\n"
        "Read the existing daily summary (context-cache id "
        f"ctx_daily_{target.isoformat()}) if any, fetch the activity observations "
        "for that date plus Garmin health data, journal entries and chat history, "
        "then update (or create) the consolidated daily timeline + Health & Energy "
        "summary and store it back under the same cache id. Finally refresh the "
        "structured activity blocks for the date."
    )


async def _run(target: date) -> int:
    from app.telegram.spawn import spawn_agent

    result = await spawn_agent(
        agent=AGENT,
        prompt=build_prompt(target),
        model_postfix=os.environ.get("FREN_MODEL_POSTFIX", ""),
        trigger="cron",
        timeout_s=TIMEOUT_S,
    )
    if not result.ok:
        print(f"[activity_summarizer] agent run failed: {result.error}", file=sys.stderr)
        return 1
    print(f"[activity_summarizer] {AGENT} completed for {target.isoformat()}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Activity summarizer — daily timeline consolidation")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="Date to summarize (YYYY-MM-DD, default: today)",
    )
    args = parser.parse_args()
    target = args.date or datetime.now(UTC).date()
    sys.exit(asyncio.run(_run(target)))


if __name__ == "__main__":
    main()
