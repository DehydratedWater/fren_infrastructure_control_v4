#!/usr/bin/env python3
"""Hourly goal-progress cron entrypoint.

Thin wrapper around ``app.tools.goals.goal_progress_cron`` — backfills LLM
matching questions, then runs ``GoalProgressAutoUpdaterTool`` with a 2-hour
lookback (v3 parity).

Usage:
    python scripts/goal_progress_auto_updater_cron.py [--lookback-hours 2]
"""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Goal progress auto-update cron")
    parser.add_argument("--lookback-hours", type=int, default=2, help="Evidence lookback window")
    args = parser.parse_args()

    from app.tools.goals.goal_progress_cron import main as cron_main

    cron_main(lookback_hours=args.lookback_hours)


if __name__ == "__main__":
    main()
