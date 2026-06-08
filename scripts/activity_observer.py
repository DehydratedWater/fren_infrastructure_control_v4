#!/usr/bin/env python3
"""Activity observer cron entrypoint.

Thin wrapper around ``app.tools.system.activity_observer`` — captures a camera
frame, describes it with the vision vLLM, and writes an activity_blocks row so
the proactive context loader has live, changing room-state material.

Usage:
    python scripts/activity_observer.py [webcam|desk|both]
"""

from __future__ import annotations

import asyncio
import sys


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("webcam", "desk", "both") else "webcam"
    from app.tools.system.activity_observer import run

    asyncio.run(run(command=command))


if __name__ == "__main__":
    main()
