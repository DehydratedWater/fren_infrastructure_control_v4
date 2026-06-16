#!/usr/bin/env python3
"""Background world turn — cron entrypoint.

Advances Twily's life by one beat (she chooses her own next action), the way the
scheduler runs other script jobs. Periodically also promotes her distilled world
memories into her persona (so her sim life shapes who she is on Telegram).

This is a BACKGROUND job: it must NOT message the user. The scheduler is
configured with synth_fallback disabled for it; we also keep stdout terse.

Usage:
    python scripts/world_turn.py                 # one autonomous beat
    python scripts/world_turn.py --promote        # beat + promote memories
    python scripts/world_turn.py --promote-only    # just promote (no beat)
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def _run(*, promote: bool, promote_only: bool) -> int:
    # short-lived process: avoid holding a shared pool
    from app.db.session import set_null_pool

    set_null_pool(True)

    from app.world.loader import DEFAULT_PACKAGE

    rc = 0
    if not promote_only:
        from app.world.turn import run_world_turn

        result = await run_world_turn(world_id=DEFAULT_PACKAGE, trigger="auto")
        if not result.get("ok"):
            print(f"[world_turn] beat failed: {result.get('error')}", file=sys.stderr)
            rc = 1
        else:
            print(
                f"[world_turn] turn={result.get('turn')} "
                f"clock={result.get('clock_label')} ({result.get('day_phase')}) "
                f"moved={result.get('moved')} researched={result.get('researched')}"
            )

    if promote or promote_only:
        from app.world.integrate import promote_world_memories

        counts = await promote_world_memories(DEFAULT_PACKAGE)
        print(f"[world_turn] promoted memories={counts['memories']} interests={counts['interests']}")

    return rc


def main() -> None:
    p = argparse.ArgumentParser(description="Advance Twily's world by one beat")
    p.add_argument("--promote", action="store_true",
                   help="also promote distilled world memories into persona memory")
    p.add_argument("--promote-only", action="store_true",
                   help="skip the beat; only promote memories")
    args = p.parse_args()
    sys.exit(asyncio.run(_run(promote=args.promote, promote_only=args.promote_only)))


if __name__ == "__main__":
    main()
