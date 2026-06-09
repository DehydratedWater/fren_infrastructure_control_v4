#!/usr/bin/env python3
"""Ralf housekeeping cron entrypoint.

Thin wrapper around ``app.tools.system.ralf_cleanup.run`` — deletes rendered
media older than the retention window (unless marked as a ralf winner) and
releases expired locks. File deletion is containment-checked against the
configured media roots; nothing outside them is ever touched.

Usage:
    python scripts/ralf_cleanup.py [--keep-days 7] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import logging


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ralf housekeeping — rendered media + expired locks")
    parser.add_argument("--keep-days", type=int, default=7, help="Keep rendered media newer than N days")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without deleting")
    args = parser.parse_args()

    from app.tools.system.ralf_cleanup import run

    asyncio.run(run(args.keep_days, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
