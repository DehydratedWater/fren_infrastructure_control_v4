#!/usr/bin/env python3
"""Re-embed every populated source table with the ACTIVE embedding model (bge-m3).

Run AFTER migration 002 (which cleared the old 1536-d OpenAI vectors and
re-dimensioned the columns to vector(1024)). Embeds each row's source text via
app.services.embeddings (now the local bge-m3 vLLM on the A4000) and writes the
vector back. Idempotent: re-run safely; pass --force to re-embed rows that
already have an embedding.

Usage:
    python scripts/reembed_all.py            # embed rows missing an embedding
    python scripts/reembed_all.py --force    # re-embed everything
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# (table, pk col, source-text SQL expression)
_SPECS: list[tuple[str, str, str]] = [
    ("memories", "id", "content"),
    ("persona_interests", "id", "concat_ws(' — ', topic, stance)"),
    ("profile_discoveries", "id", "discovery"),
    ("user_facts", "fact_id", "fact_text"),
    ("chat_messages", "id", "message"),
    ("night_analysis_findings", "id", "concat_ws(' ', title, content)"),
    ("topic_nodes", "id", "concat_ws(' — ', label, summary)"),
    ("link_previews", "url", "concat_ws(' ', title, description)"),
    ("documents", "doc_id", "extracted_text"),
]
_BATCH = 64


async def _reembed_table(table: str, pk: str, text_expr: str, *, force: bool) -> tuple[int, int]:
    from app.db.session import execute_sql, fetch_all, get_async_session
    from app.services.embeddings import get_embeddings_batch

    where = f"({text_expr}) IS NOT NULL AND length(trim({text_expr})) > 0"
    if not force:
        where += " AND embedding IS NULL"
    sel = f"SELECT {pk} AS pk, {text_expr} AS txt FROM {table} WHERE {where}"
    async with get_async_session() as s:
        rows = await fetch_all(s, sel, {})
    if not rows:
        return 0, 0

    done = 0
    for i in range(0, len(rows), _BATCH):
        batch = rows[i:i + _BATCH]
        vecs = await asyncio.to_thread(get_embeddings_batch, [str(r["txt"]) for r in batch])
        async with get_async_session() as s:
            for r, vec in zip(batch, vecs, strict=False):
                if not vec or not any(vec):  # skip zero vectors (embed failed)
                    continue
                literal = "[" + ",".join(repr(float(x)) for x in vec) + "]"
                await execute_sql(
                    s,
                    f"UPDATE {table} SET embedding = CAST(:v AS vector) WHERE {pk} = :pk",
                    {"v": literal, "pk": r["pk"]},
                )
                done += 1
        print(f"  [{table}] {min(i + _BATCH, len(rows))}/{len(rows)}", flush=True)
    return done, len(rows)


async def _run(force: bool) -> int:
    from app.services.embeddings import active_model

    info = active_model()
    print(f"[reembed] model={info['model']} dims={info['dims']} endpoint={info['endpoint']} "
          f"local={info['local']} force={force}")
    if not info["configured"]:
        print("[reembed] no embedding backend configured — aborting", file=sys.stderr)
        return 1
    total = 0
    for table, pk, expr in _SPECS:
        try:
            done, seen = await _reembed_table(table, pk, expr, force=force)
            if seen:
                print(f"[reembed] {table}: embedded {done}/{seen}")
            total += done
        except Exception as exc:  # noqa: BLE001 — keep going across tables
            print(f"[reembed] {table}: ERROR {exc}", file=sys.stderr)
    print(f"[reembed] done — {total} rows embedded with {info['model']}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Re-embed all source tables with the active model")
    p.add_argument("--force", action="store_true", help="re-embed rows that already have an embedding")
    args = p.parse_args()
    sys.exit(asyncio.run(_run(args.force)))


if __name__ == "__main__":
    main()
