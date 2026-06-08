#!/usr/bin/env python3
"""Inner monologue cron entrypoint.

Thin wrapper around ``app.tools.system.inner_monologue`` — generates one private
thought and stores it in the memories table tagged ``inner_monologue``, which is
what the proactive context loader + conversation digest read for voice cues.

Usage:
    python scripts/inner_monologue.py
"""

from __future__ import annotations

import asyncio


def main() -> None:
    from app.tools.system.inner_monologue import run

    asyncio.run(run())


if __name__ == "__main__":
    main()
