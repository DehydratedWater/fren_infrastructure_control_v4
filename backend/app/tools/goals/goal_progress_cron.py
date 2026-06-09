"""Hourly goal-progress cron — v3 parity wrapper around the auto-updater tool.

v3's ``scripts/goal_progress_auto_updater_cron.py`` ported to v4 (the cron
entrypoint at ``scripts/goal_progress_auto_updater_cron.py`` is a thin wrapper
around :func:`main`): runs the auto-updater with a 2-hour lookback to catch
any evidence (activities, events, habit completions, todos) the event
extractor's inline trigger missed, after backfilling LLM matching questions
for goals that lack them. Plain plumbing — all decisions live inside
``GoalProgressAutoUpdaterTool`` (itself LLM-matched and config-tunable).
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_HOURS = 2


def backfill_missing_questions() -> None:
    """Backfill matching questions for goals that don't have them yet."""
    try:
        from app.tools.goals import goal_manager as gm

        result = gm.GoalManagerTool().execute(gm.Input(command="backfill-keywords"))
        if result.count > 0:
            logger.info("Backfilled matching questions for %d goals", result.count)
    except Exception:
        logger.warning("Matching question backfill failed (non-critical)", exc_info=True)


def run(lookback_hours: int = DEFAULT_LOOKBACK_HOURS):
    """One cron pass: backfill questions, then run the update cycle."""
    from app.tools.goals import goal_progress_auto_updater as gpau

    backfill_missing_questions()
    return gpau.GoalProgressAutoUpdaterTool().execute(
        gpau.Input(command="run", lookback_hours=lookback_hours)
    )


def main(lookback_hours: int = DEFAULT_LOOKBACK_HOURS) -> None:
    result = run(lookback_hours)
    if result.success:
        logger.info(
            "Goal progress update: %d updates, %d skipped. %s",
            result.updates_made,
            result.updates_skipped,
            result.message,
        )
    else:
        logger.error("Goal progress update failed: %s", result.error)
        sys.exit(1)
