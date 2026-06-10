#!/usr/bin/env python3
"""Topic synthesizer cron entrypoint — full rebuild + ``--expire-only`` modes.

Two schedule jobs share this script (same names as v3):

- ``pending_thoughts_expire`` runs ``--expire-only``: pure data plumbing
  through ``app.tools.persona.topic_synthesizer.expire_only`` (prune stale
  persona_interests, expire/trim pending_thoughts). No LLM.
- ``topic_synthesizer`` (nightly, full mode, no flag): the clustering is LLM
  work, so it lives in the fleet agent ``persona/topic_synthesizer`` (defined
  in app/agents/domains/persona.py, with LLM-judge probes gating unsupported
  topics and duplicates). Full mode spawns that agent the same way the
  scheduler runs agent jobs — ``spawn_agent`` against the compiled fleet,
  honoring FREN_MODEL_POSTFIX — and exits non-zero on failure.

Usage:
    python scripts/topic_synthesizer.py                # full nightly rebuild
    python scripts/topic_synthesizer.py --expire-only  # cheap pruning pass
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

AGENT = "persona/topic_synthesizer"
TIMEOUT_S = 870.0  # under the schedule job's 900s budget


def build_prompt() -> str:
    return (
        "Run the nightly topic synthesis.\n"
        "Read the recent user-side chat themes, events and memories, plus the "
        "existing persona interests as the dedup baseline. Cluster the new "
        "material into 0-6 deduplicated topics with novelty scores; persist "
        "each genuinely new topic via persona-memory-manager create-interest "
        "(source user_echo), mark-interest-surfaced for themes matching an "
        "existing interest, then run the prune-interests / expire-thoughts / "
        "trim-thoughts housekeeping. Every topic must be supported by the "
        "material — an uneventful day may yield zero new topics."
    )


async def _run_full() -> int:
    from app.telegram.spawn import spawn_agent

    result = await spawn_agent(
        agent=AGENT,
        prompt=build_prompt(),
        model_postfix=os.environ.get("FREN_MODEL_POSTFIX", ""),
        trigger="cron",
        timeout_s=TIMEOUT_S,
    )
    if not result.ok:
        print(f"[topic_synthesizer] agent run failed: {result.error}", file=sys.stderr)
        return 1
    print(f"[topic_synthesizer] {AGENT} completed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Topic synthesizer (full rebuild or --expire-only)")
    parser.add_argument("--expire-only", action="store_true", help="Prune stale thoughts + interests")
    args = parser.parse_args()

    if args.expire_only:
        from app.tools.persona.topic_synthesizer import expire_only

        try:
            asyncio.run(expire_only())
        except Exception as e:  # noqa: BLE001 — cron entrypoint: fail with a clear one-liner
            print(f"[topic_synthesizer] FATAL: {type(e).__name__}: {e}")
            sys.exit(1)
        return

    sys.exit(asyncio.run(_run_full()))


if __name__ == "__main__":
    main()
