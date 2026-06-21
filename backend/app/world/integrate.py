"""Feed Twily's world back into her persona.

Her life in the sim is only meaningful if it *changes* her — so distilled world
memories get promoted into her real persona memory, and the things she chose to
research become persona interests. This runs periodically (a cron job) and after
notable turns. Embeddings are best-effort: the embedding backend now raises on
failure, so we catch and store without a vector rather than losing the memory.
"""

from __future__ import annotations

import logging
from typing import Any

from app.world.loader import DEFAULT_PACKAGE, get_package
from app.world.state import WorldStateRepo

logger = logging.getLogger(__name__)


async def _embed(text: str) -> list[float] | None:
    try:
        from app.services.embeddings import get_embedding

        return await get_embedding(text)
    except Exception:  # noqa: BLE001 — embedding backend may be down; store without
        logger.debug("world.integrate: embedding skipped for memory")
        return None


async def promote_world_memories(world_id: str = DEFAULT_PACKAGE, *, limit: int = 12) -> dict[str, int]:
    """Convert her recent research into persona interests (gentle curiosity
    shaping). Episodic world memories are intentionally NOT promoted — see below.
    Returns counts."""
    repo = WorldStateRepo(world_id)
    pending = await repo.unconsumed_memories(limit=limit)

    # NOTE: we deliberately do NOT promote episodic world memories (Sol said X,
    # the dropout, etc.) into the chat-retrieved `memories` table — those leaked
    # world specifics into assistant replies (a query topically matching them made
    # a clueless reply path confabulate/disavow). She's shaped GENTLY instead, via
    # interests only (her curiosities — harmless topics). Episodic beats stay in
    # the world's own tables and surface only in the world UI. We still consume the
    # pending rows so they don't re-scan.
    promoted_ids = [int(m["id"]) for m in pending]
    await repo.mark_memories_consumed(promoted_ids)

    interests = await _promote_research_interests(world_id)
    logger.info("world.integrate[%s]: %d interests (episodic promotion disabled)", world_id, interests)
    return {"memories": 0, "interests": interests, "scanned": len(pending)}


async def _promote_research_interests(world_id: str) -> int:
    """Turn the things she chose to look up into persona interests (her curiosity
    in the sim becomes her curiosity in real life)."""
    from app.db.repos.persona_memory import PersonaInterestsRepo

    repo = WorldStateRepo(world_id)
    research = await repo.recent_research(limit=6)
    if not research:
        return 0
    interests = PersonaInterestsRepo()
    seen: set[str] = set()
    made = 0
    for r in research:
        q = str(r.get("query", "")).strip()
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        summary = str(r.get("summary", ""))[:300]
        emb = await _embed(q)
        try:
            await interests.create(
                topic=q,
                stance=f"Looked this up while tinkering at home — {summary}" if summary else None,
                source="world_research",
                embedding=emb,
                novelty_score=0.6,
            )
            made += 1
        except Exception:  # noqa: BLE001
            logger.exception("world.integrate: failed to create interest for %r", q)
    return made


async def recent_life_summary(world_id: str = DEFAULT_PACKAGE, *, turns: int = 12) -> str:
    """A compact prose digest of her recent life — for surfacing in other systems
    (e.g. the heartbeat could mention what she's been up to)."""
    repo = WorldStateRepo(world_id)
    events = await repo.recent_events(limit=turns)
    if not events:
        return ""
    lines: list[str] = []
    for e in events:
        if e.get("kind") in ("narration", "action", "research", "move"):
            lines.append(str(e.get("content", "")).strip())
    return " ".join(x for x in lines if x)[:1500]
