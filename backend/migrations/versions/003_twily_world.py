"""Twily's world — runtime state for the background roleplay life-sim.

Five tables, all scoped by `world_id` (one row-set per world package, so several
worlds could coexist later):

* `world_sessions`   — the singleton-ish live state per world (where she is, the
                       in-world clock, her persona_state blob, turn counter).
* `world_events`     — the append-only life log / stream (narration, her actions
                       and speech, NPC lines, moves, research, visitor input).
* `world_npc_state`  — per-NPC relationship warmth + light memory notes.
* `world_research`   — the in-world computer's real web lookups (via SearchAPI).
* `world_memories`   — distilled, importance-scored memories from her life that
                       later feed back into her persona (see world/integrate.py).

Raw `op.execute(...)` SQL, matching 001/002 style. No vector columns (the
optional embedding for memories is added later if needed).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "003_twily_world"
down_revision: str | None = "002_bge_m3_embeddings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS world_sessions (
            world_id            VARCHAR(100) PRIMARY KEY,
            package_id          VARCHAR(100) NOT NULL,
            current_location_id VARCHAR(120) NOT NULL,
            clock_minutes       BIGINT       NOT NULL DEFAULT 0,
            day_count           INTEGER      NOT NULL DEFAULT 1,
            turn_count          BIGINT       NOT NULL DEFAULT 0,
            persona_state       JSONB        NOT NULL DEFAULT '{}'::jsonb,
            visitor_present     BOOLEAN      NOT NULL DEFAULT FALSE,
            started_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS world_events (
            id            BIGSERIAL PRIMARY KEY,
            world_id      VARCHAR(100) NOT NULL,
            turn          BIGINT       NOT NULL DEFAULT 0,
            kind          VARCHAR(32)  NOT NULL,   -- narration|action|speech|npc|research|move|mood|system|visitor
            actor         VARCHAR(120) NOT NULL DEFAULT 'narrator',
            content       TEXT         NOT NULL,
            location_id   VARCHAR(120),
            meta          JSONB        NOT NULL DEFAULT '{}'::jsonb,
            created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_world_events_world_id ON world_events(world_id, id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_world_events_turn ON world_events(world_id, turn)")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS world_npc_state (
            world_id       VARCHAR(100) NOT NULL,
            npc_id         VARCHAR(120) NOT NULL,
            affinity       INTEGER      NOT NULL DEFAULT 0,   -- -100..100
            notes          JSONB        NOT NULL DEFAULT '[]'::jsonb,
            last_seen_turn BIGINT       NOT NULL DEFAULT 0,
            updated_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            PRIMARY KEY (world_id, npc_id)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS world_research (
            id          BIGSERIAL PRIMARY KEY,
            world_id    VARCHAR(100) NOT NULL,
            turn        BIGINT       NOT NULL DEFAULT 0,
            query       TEXT         NOT NULL,
            summary     TEXT         NOT NULL DEFAULT '',
            results     JSONB        NOT NULL DEFAULT '[]'::jsonb,
            created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_world_research_world_id ON world_research(world_id, id)")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS world_memories (
            id           BIGSERIAL PRIMARY KEY,
            world_id     VARCHAR(100) NOT NULL,
            turn         BIGINT       NOT NULL DEFAULT 0,
            content      TEXT         NOT NULL,
            kind         VARCHAR(32)  NOT NULL DEFAULT 'episodic',
            importance   REAL         NOT NULL DEFAULT 0.5,
            location_id  VARCHAR(120),
            consumed     BOOLEAN      NOT NULL DEFAULT FALSE,  -- promoted into persona memory yet?
            created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_world_memories_world_id ON world_memories(world_id, id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_world_memories_unconsumed "
        "ON world_memories(world_id, consumed) WHERE consumed = FALSE"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS world_memories CASCADE")
    op.execute("DROP TABLE IF EXISTS world_research CASCADE")
    op.execute("DROP TABLE IF EXISTS world_npc_state CASCADE")
    op.execute("DROP TABLE IF EXISTS world_events CASCADE")
    op.execute("DROP TABLE IF EXISTS world_sessions CASCADE")
