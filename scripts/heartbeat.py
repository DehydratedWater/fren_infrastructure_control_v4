#!/usr/bin/env python3
"""Proactive autonomy heartbeat — cron entrypoint.

Runs ONE heartbeat tick for the given --mode (day|evening|winddown|night). The
reasoning + routing lives in app.agents.heartbeat (in-process thinking-on triage,
no opencode spin-up). This wrapper just runs it the way the scheduler runs other
script jobs, and exits non-zero on failure so the circuit breaker sees it.

Usage:
    python scripts/heartbeat.py --mode day
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def _run(mode: str) -> int:
    from app.agents.heartbeat import run_heartbeat

    result = await run_heartbeat(mode)
    if not result.get("ok"):
        print(f"[heartbeat] tick failed: {result.get('error')}", file=sys.stderr)
        return 1
    print(f"[heartbeat:{mode}] decision={result.get('decision')} "
          f"category={result.get('category')} acted={result.get('acted')}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Proactive autonomy heartbeat tick")
    p.add_argument("--mode", default="day", choices=["day", "evening", "winddown", "night"])
    args = p.parse_args()
    sys.exit(asyncio.run(_run(args.mode)))


if __name__ == "__main__":
    main()
