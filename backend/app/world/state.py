"""Runtime world state — Postgres persistence for the life-sim.

Thin async raw-SQL repos over the `world_*` tables (migration 003), matching the
fren repo pattern (app.db.session helpers, parameterized SQL, no ORM). The turn
engine reads the session + recent events, runs the LLM beat, then writes the
resulting events / state deltas back here.
"""

from __future__ import annotations

import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session
from app.world import clock
from app.world.models import WorldPackage


class WorldStateRepo:
    """One world's live state + its logs. `world_id` keys everything."""

    def __init__(self, world_id: str) -> None:
        self.world_id = world_id

    # ── session lifecycle ────────────────────────────────────────────────
    async def get_session(self) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s, "SELECT * FROM world_sessions WHERE world_id = :w", {"w": self.world_id}
            )

    async def ensure_session(self, pkg: WorldPackage) -> dict[str, Any]:
        """Get-or-create the world session from the package's opening scenario."""
        existing = await self.get_session()
        if existing:
            return existing
        start_minutes = (pkg.scenario.start_hour % 24) * 60
        persona_state = {
            "mood": "settled and curious",
            "energy": 80,
            "focus": "the day ahead",
        }
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                """
                INSERT INTO world_sessions
                    (world_id, package_id, current_location_id, clock_minutes,
                     day_count, turn_count, persona_state)
                VALUES (:w, :pkg, :loc, :clk, 1, 0, CAST(:ps AS jsonb))
                ON CONFLICT (world_id) DO NOTHING
                RETURNING *
                """,
                {
                    "w": self.world_id,
                    "pkg": pkg.id,
                    "loc": pkg.scenario.starting_location_id,
                    "clk": start_minutes,
                    "ps": json.dumps(persona_state),
                },
            )
        if row is None:  # lost a race; read it back
            row = await self.get_session()
        assert row is not None
        # seed the opening narration once
        if int(row.get("turn_count", 0)) == 0:
            await self.add_event(
                turn=0, kind="system", actor="narrator",
                content=pkg.scenario.opening_narration,
                location_id=pkg.scenario.starting_location_id,
            )
        # seed npc affinities
        await self._seed_npcs(pkg)
        return row

    async def _seed_npcs(self, pkg: WorldPackage) -> None:
        async with get_async_session() as s:
            for npc in pkg.npcs:
                await execute_sql(
                    s,
                    """
                    INSERT INTO world_npc_state (world_id, npc_id, affinity)
                    VALUES (:w, :n, :a)
                    ON CONFLICT (world_id, npc_id) DO NOTHING
                    """,
                    {"w": self.world_id, "n": npc.id, "a": int(npc.default_affinity)},
                )

    async def advance_turn(
        self,
        *,
        new_location_id: str | None,
        minutes: int,
        persona_state: dict[str, Any],
        visitor_present: bool | None = None,
    ) -> dict[str, Any]:
        """Bump the clock/day/turn counter and persist location + persona_state."""
        sess = await self.get_session()
        assert sess is not None
        clk, day = clock.advance(
            int(sess["clock_minutes"]), int(sess["day_count"]), minutes
        )
        loc = new_location_id or sess["current_location_id"]
        vis = sess["visitor_present"] if visitor_present is None else visitor_present
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                """
                UPDATE world_sessions
                   SET current_location_id = :loc,
                       clock_minutes = :clk,
                       day_count = :day,
                       turn_count = turn_count + 1,
                       persona_state = CAST(:ps AS jsonb),
                       visitor_present = :vis,
                       updated_at = NOW()
                 WHERE world_id = :w
                RETURNING *
                """,
                {
                    "loc": loc, "clk": clk, "day": day,
                    "ps": json.dumps(persona_state), "vis": bool(vis),
                    "w": self.world_id,
                },
            )
        assert row is not None
        return row

    async def set_visitor_present(self, present: bool) -> None:
        async with get_async_session() as s:
            await execute_sql(
                s,
                "UPDATE world_sessions SET visitor_present = :p, updated_at = NOW() "
                "WHERE world_id = :w",
                {"p": present, "w": self.world_id},
            )

    # ── events (the life log) ────────────────────────────────────────────
    async def add_event(
        self, *, turn: int, kind: str, actor: str, content: str,
        location_id: str | None = None, meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not content or not content.strip():
            return None
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO world_events (world_id, turn, kind, actor, content, location_id, meta)
                VALUES (:w, :t, :k, :a, :c, :loc, CAST(:m AS jsonb))
                RETURNING *
                """,
                {
                    "w": self.world_id, "t": turn, "k": kind, "a": actor,
                    "c": content.strip(), "loc": location_id,
                    "m": json.dumps(meta or {}),
                },
            )

    async def recent_events(self, limit: int = 60, before_id: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"w": self.world_id, "lim": limit}
        clause = ""
        if before_id is not None:
            clause = "AND id < :bid"
            params["bid"] = before_id
        async with get_async_session() as s:
            rows = await fetch_all(
                s,
                f"""
                SELECT * FROM world_events
                 WHERE world_id = :w {clause}
                 ORDER BY id DESC
                 LIMIT :lim
                """,
                params,
            )
        return list(reversed(rows))  # chronological

    async def events_for_prompt(self, limit: int = 24) -> list[dict[str, Any]]:
        return await self.recent_events(limit=limit)

    # ── npc relationships ────────────────────────────────────────────────
    async def npc_states(self) -> dict[str, dict[str, Any]]:
        async with get_async_session() as s:
            rows = await fetch_all(
                s, "SELECT * FROM world_npc_state WHERE world_id = :w", {"w": self.world_id}
            )
        return {r["npc_id"]: r for r in rows}

    async def bump_affinity(self, npc_id: str, delta: int, turn: int) -> None:
        delta = max(-10, min(10, int(delta)))
        async with get_async_session() as s:
            await execute_sql(
                s,
                """
                INSERT INTO world_npc_state (world_id, npc_id, affinity, last_seen_turn)
                VALUES (:w, :n, :d, :t)
                ON CONFLICT (world_id, npc_id) DO UPDATE
                   SET affinity = GREATEST(-100, LEAST(100, world_npc_state.affinity + :d)),
                       last_seen_turn = :t,
                       updated_at = NOW()
                """,
                {"w": self.world_id, "n": npc_id, "d": delta, "t": turn},
            )

    async def mark_seen(self, npc_ids: list[str], turn: int) -> None:
        if not npc_ids:
            return
        async with get_async_session() as s:
            for nid in npc_ids:
                await execute_sql(
                    s,
                    """
                    INSERT INTO world_npc_state (world_id, npc_id, last_seen_turn)
                    VALUES (:w, :n, :t)
                    ON CONFLICT (world_id, npc_id) DO UPDATE
                       SET last_seen_turn = :t, updated_at = NOW()
                    """,
                    {"w": self.world_id, "n": nid, "t": turn},
                )

    # ── research log (the in-world computer) ─────────────────────────────
    async def add_research(
        self, *, turn: int, query: str, summary: str, results: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO world_research (world_id, turn, query, summary, results)
                VALUES (:w, :t, :q, :s, CAST(:r AS jsonb))
                RETURNING *
                """,
                {"w": self.world_id, "t": turn, "q": query, "s": summary,
                 "r": json.dumps(results)},
            )

    async def recent_research(self, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM world_research WHERE world_id = :w ORDER BY id DESC LIMIT :lim",
                {"w": self.world_id, "lim": limit},
            )

    # ── memories (feed persona later) ────────────────────────────────────
    async def add_memory(
        self, *, turn: int, content: str, importance: float = 0.5,
        kind: str = "episodic", location_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not content or not content.strip():
            return None
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO world_memories (world_id, turn, content, importance, kind, location_id)
                VALUES (:w, :t, :c, :imp, :k, :loc)
                RETURNING *
                """,
                {"w": self.world_id, "t": turn, "c": content.strip(),
                 "imp": importance, "k": kind, "loc": location_id},
            )

    async def unconsumed_memories(self, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM world_memories
                 WHERE world_id = :w AND consumed = FALSE
                 ORDER BY importance DESC, id ASC
                 LIMIT :lim
                """,
                {"w": self.world_id, "lim": limit},
            )

    async def mark_memories_consumed(self, ids: list[int]) -> None:
        if not ids:
            return
        async with get_async_session() as s:
            await execute_sql(
                s,
                "UPDATE world_memories SET consumed = TRUE "
                "WHERE world_id = :w AND id = ANY(:ids)",
                {"w": self.world_id, "ids": ids},
            )
