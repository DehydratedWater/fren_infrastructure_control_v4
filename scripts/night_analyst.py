#!/usr/bin/env python3
"""Night analyst cron entrypoint.

The analysis is LLM work, so it lives in the fleet agent
``support/night_analyst`` (defined in app/agents/domains/support.py, with
LLM-judge probes gating invented correlations). This wrapper just spawns that
agent the same way the scheduler runs agent jobs — ``spawn_agent`` against the
compiled fleet, honoring FREN_MODEL_POSTFIX — and exits non-zero on failure so
the scheduler's circuit breaker sees real failures.

Usage:
    python scripts/night_analyst.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime

AGENT = "support/night_analyst"
TIMEOUT_S = 3300.0  # under the schedule job's 3600s budget


def build_prompt(date_iso: str) -> str:
    return (
        f"Run the nightly deep cross-domain analysis for {date_iso}.\n"
        "Gather recent events, activity blocks, goals, habits, chat themes and "
        "Garmin health data; read the previous run's findings (night-analysis) "
        "so you never repeat them; correlate across domains; persist the report "
        "via context-cache (artifact type night_analysis_report) plus a "
        "night_analysis memory, and deliver the <<night_analysis>> summary via "
        "emit_guidance. If the data shows no strong patterns, say so and skip "
        "the delivery — never invent findings."
    )


async def _run(date_iso: str) -> int:
    from app.telegram.spawn import spawn_agent

    result = await spawn_agent(
        agent=AGENT,
        prompt=build_prompt(date_iso),
        model_postfix=os.environ.get("FREN_MODEL_POSTFIX", ""),
        trigger="cron",
        timeout_s=TIMEOUT_S,
    )
    if not result.ok:
        print(f"[night_analyst] agent run failed: {result.error}", file=sys.stderr)
        return 1
    print(f"[night_analyst] {AGENT} completed for {date_iso}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Night analyst — deep overnight cross-domain analysis")
    parser.add_argument("--date", default=None, help="Analysis date label (YYYY-MM-DD, default: today)")
    args = parser.parse_args()
    date_iso = args.date or datetime.now(UTC).date().isoformat()
    sys.exit(asyncio.run(_run(date_iso)))


if __name__ == "__main__":
    main()
