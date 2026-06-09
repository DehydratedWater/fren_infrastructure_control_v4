#!/usr/bin/env python3
"""Event → Habit bridge cron entrypoint.

Thin wrapper around ``app.bridge.event_habit.run_bridge`` — matches newly
detected life events against active habits using the autoloop-tunable policy
(``DEFAULT_POLICY`` or the promoted ``policy:event_habit_bridge`` snapshot)
and auto-completes today's habit occurrences through the habits repo.

Usage:
    python scripts/event_habit_bridge.py
"""

from __future__ import annotations

import asyncio
import logging


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from app.bridge.event_habit import run_bridge

    summary = asyncio.run(run_bridge())
    print(
        f"[event_habit_bridge] events={summary['events']} "
        f"completions={summary['completions']} skipped={summary['skipped']}"
    )


if __name__ == "__main__":
    main()
