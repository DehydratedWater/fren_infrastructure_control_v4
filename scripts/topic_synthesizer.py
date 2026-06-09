#!/usr/bin/env python3
"""Topic synthesizer cron entrypoint — ``--expire-only`` mode only (for now).

Thin wrapper around ``app.tools.persona.topic_synthesizer.expire_only`` —
prunes stale persona_interests and expires/trims the pending_thoughts queue
(job ``pending_thoughts_expire``, daily).

The FULL nightly MemTree-style topic-tree rebuild (v3's default mode) is NOT
ported yet; the ``topic_synthesizer`` schedule job stays disabled until it is,
and invoking this script without ``--expire-only`` fails loudly rather than
pretending to synthesize.

Usage:
    python scripts/topic_synthesizer.py --expire-only
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Topic synthesizer (expire-only mode ported)")
    parser.add_argument("--expire-only", action="store_true", help="Prune stale thoughts + interests")
    args = parser.parse_args()

    if not args.expire_only:
        print(
            "[topic_synthesizer] FATAL: the full topic-tree rebuild is not ported to v4 yet — "
            "only --expire-only is supported (job pending_thoughts_expire)."
        )
        sys.exit(2)

    from app.tools.persona.topic_synthesizer import expire_only

    try:
        asyncio.run(expire_only())
    except Exception as e:  # noqa: BLE001 — cron entrypoint: fail with a clear one-liner
        print(f"[topic_synthesizer] FATAL: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
