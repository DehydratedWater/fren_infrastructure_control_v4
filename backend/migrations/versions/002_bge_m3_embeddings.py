"""Switch embedding columns 1536 → 1024 (OpenAI text-embedding-3-small → bge-m3).

The embedding service now uses the local bge-m3 vLLM (A4000, 1024-dim) instead of
OpenAI's 1536-dim model. Existing 1536-d vectors are incompatible, so every
vector column is cleared, re-dimensioned to vector(1024), and its hnsw index
rebuilt. Re-population is done OUT of band by scripts/reembed_all.py (migrations
must not call the embedding endpoint).

Style follows 001: raw `op.execute(...)` SQL.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "002_bge_m3_embeddings"
down_revision: str | None = "001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, hnsw index name or None) — every vector column in the schema.
_VECTOR_COLS: list[tuple[str, str | None]] = [
    ("chat_messages", "idx_chat_messages_embedding"),
    ("documents", "idx_documents_embedding"),
    ("embedding_chunks", "idx_ec_embedding"),
    ("link_previews", "idx_link_previews_embedding"),
    ("memories", "idx_memories_embedding"),
    ("night_analysis_findings", "idx_findings_embedding"),
    ("persona_interests", None),
    ("profile_discoveries", "idx_profile_discoveries_embedding"),
    ("topic_nodes", None),
    ("user_facts", "idx_user_facts_embedding"),
]


def _redim(dims: int) -> None:
    for table, index in _VECTOR_COLS:
        if index:
            op.execute(f"DROP INDEX IF EXISTS {index}")
        # Clear incompatible vectors first, then change the column dimension.
        op.execute(f"UPDATE {table} SET embedding = NULL WHERE embedding IS NOT NULL")
        op.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector({dims})")
        if index:
            op.execute(
                f"CREATE INDEX IF NOT EXISTS {index} ON {table} "
                "USING hnsw (embedding vector_cosine_ops)"
            )


def upgrade() -> None:
    _redim(1024)


def downgrade() -> None:
    _redim(1536)
