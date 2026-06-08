#!/usr/bin/env python3
"""Conversation digest cron entrypoint.

Thin wrapper around ``app.tools.context.conversation_digest`` — generates the
rolling situational digest and stores it in ``agent_notes['conversation_digest']``
for the scheduler's ``_enrich_prompt`` to prepend to every proactive agent.

Usage:
    python scripts/conversation_digest.py [--hours 12] [--print]
"""

from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversation digest — rolling situational summary")
    parser.add_argument("--hours", type=int, default=12, help="Hours of chat history to look back")
    parser.add_argument("--print", action="store_true", dest="print_only", help="Print current digest, no update")
    args = parser.parse_args()

    from app.tools.context.conversation_digest import get_digest, run

    if args.print_only:
        print(asyncio.run(get_digest()) or "No conversation digest found")
    else:
        asyncio.run(run(hours=args.hours))


if __name__ == "__main__":
    main()
