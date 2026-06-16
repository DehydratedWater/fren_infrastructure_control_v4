"""Feed Twily's world back into her persona.

Her life in the sim is only meaningful if it *changes* her — so distilled world
memories get promoted into her real persona memory, and the things she chose to
research become persona interests. This runs periodically (a cron job) and after
notable turns. Embeddings are best-effort: the embedding backend now raises on
failure, so we catch and store without a vector rather than losing the memory.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.world.loader import DEFAULT_PACKAGE, get_package
from app.world.state import WorldStateRepo

logger = logging.getLogger(__name__)

# only memories at/above this importance cross into her persona
_PROMOTE_THRESHOLD = 0.6


async def _embed(text: str) -> list[float] | None:
    try:
        from app.services.embeddings import get_embedding

        return await get_embedding(text)
    except Exception:  # noqa: BLE001 — embedding backend may be down; store without
        logger.debug("world.integrate: embedding skipped for memory")
        return None


async def promote_world_memories(world_id: str = DEFAULT_PACKAGE, *, limit: int = 12) -> dict[str, int]:
    """Promote unconsumed, important world memories into persona memory and
    convert recent research into persona interests. Returns counts."""
    from app.db.repos.memories import MemoriesRepo

    repo = WorldStateRepo(world_id)
    pkg = get_package(world_id)
    pending = await repo.unconsumed_memories(limit=limit)

    promoted_ids: list[int] = []
    mem_repo = MemoriesRepo()
    promoted = 0
    for m in pending:
        if float(m.get("importance", 0)) < _PROMOTE_THRESHOLD:
            promoted_ids.append(int(m["id"]))  # consume low-value ones too (don't re-scan)
            continue
        content = str(m.get("content", "")).strip()
        if not content:
            promoted_ids.append(int(m["id"]))
            continue
        emb = await _embed(content)
        try:
            await mem_repo.create(
                memory_id=f"world_{world_id}_{uuid.uuid4().hex[:12]}",
                title=f"From her life in {pkg.name}",
                content=content,
                tags=["inner_life", "world", world_id],
                category="inner_life",
                source="world",
                embedding=emb,
            )
            promoted += 1
        except Exception:  # noqa: BLE001
            logger.exception("world.integrate: failed to promote memory %s", m.get("id"))
            continue
        promoted_ids.append(int(m["id"]))

    await repo.mark_memories_consumed(promoted_ids)

    interests = await _promote_research_interests(world_id)
    logger.info("world.integrate[%s]: promoted %d memories, %d interests", world_id, promoted, interests)
    return {"memories": promoted, "interests": interests, "scanned": len(pending)}


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
