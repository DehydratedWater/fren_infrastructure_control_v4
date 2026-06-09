"""Topic synthesizer — v4 port of the v3 script's ``--expire-only`` mode.

Cheap nightly pruning of persona memory (job ``pending_thoughts_expire``):

  - ``persona_interests``: drop expired / over-surfaced stale interests
    (``PersonaInterestsRepo.prune_expired``);
  - ``pending_thoughts``: expire unconsumed thoughts older than 48h and trim
    the queue to its 30-item cap (``PendingThoughtsRepo.expire_old`` /
    ``trim_queue``).

Pure data plumbing through the existing repos — no decisions, no LLM. The
FULL nightly MemTree-style topic-tree rebuild (v3 ``_synthesize``) is NOT
ported yet; the ``topic_synthesizer`` schedule job stays disabled until it is.
"""

from __future__ import annotations

DEFAULT_THOUGHT_MAX_AGE_HOURS = 48
DEFAULT_QUEUE_MAX_SIZE = 30


async def expire_only(
    *,
    thought_max_age_hours: int = DEFAULT_THOUGHT_MAX_AGE_HOURS,
    queue_max_size: int = DEFAULT_QUEUE_MAX_SIZE,
) -> dict[str, int]:
    """Prune stale persona_interests + pending_thoughts. Returns counts."""
    from app.db.repos.persona_memory import PendingThoughtsRepo, PersonaInterestsRepo

    interests = PersonaInterestsRepo()
    thoughts = PendingThoughtsRepo()

    pruned_interests = await interests.prune_expired()
    expired_thoughts = await thoughts.expire_old(hours=thought_max_age_hours)
    trimmed = await thoughts.trim_queue(max_size=queue_max_size)

    print(
        f"[topic_synthesizer] expire-only — pruned_interests={pruned_interests} "
        f"expired_thoughts={expired_thoughts} trimmed={trimmed}"
    )
    return {
        "pruned_interests": pruned_interests,
        "expired_thoughts": expired_thoughts,
        "trimmed": trimmed,
    }
