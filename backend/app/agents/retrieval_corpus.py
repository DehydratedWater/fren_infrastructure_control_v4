"""Seed a LARGE retrieval corpus into the (isolated) autoloop DB.

The retrieval autoloop suite (see retrieval_probes.py) needs a realistic,
large haystack — a few hundred rows can't expose ranking, recall, or latency
problems. This module copies the REAL v3 conversation corpus (~19.8k
chat_messages, Feb–Jun 2026) and the v3 YouTube transcript library (~1.6k
videos with full transcripts) into the CURRENT DATABASE_URL target, then
plants CANARY messages: synthetic, uniquely-identifiable facts at known
timestamps that give FactRecallEvaluator exact ground truth (needle-in-a-
19k-message-haystack).

SAFETY:
- v3 (port 5452) is opened READ-ONLY (no writes are ever issued to it).
- The target is whatever DATABASE_URL points at — run this ONLY against the
  `<db>_autoloop` copy (run_autoloop.sh exports that URL). The seeder refuses
  to run if the target DB name does not end in `_autoloop`.
- Idempotent: seeded rows carry metadata.seed_source; reruns skip existing.

Usage (env DATABASE_URL must point at the autoloop DB):
    python -m app seed-retrieval [--no-embed] [--limit N] [--yt-limit N]
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

V3_DSN = "postgresql://fren:fren@localhost:5452/fren"

SEED_SOURCE_CORPUS = "v3_corpus"
SEED_SOURCE_CANARY = "retrieval_canary"

# Synthetic needles with EXACT ground truth, spread across the corpus
# timeline. Each is phrased like a real user message; the `facts` aliases are
# what a correct retrieval answer must surface (FactRecallEvaluator), and the
# fact values are globally unique in the corpus so recall is unambiguous.
CANARIES: list[dict[str, Any]] = [
    {
        "ts": "2026-02-20 21:14:00+01",
        "message": "btw zapisz: hasło do wifi w domku w górach to X9-KITE-42, "
                   "bo zawsze zapominam",
        "facts": ["X9-KITE-42"],
        "question": "What is the wifi password of the mountain cabin?",
    },
    {
        "ts": "2026-03-02 09:41:00+01",
        "message": "I booked the dentist — Dr Marlena Wójcik, March 14th at "
                   "11:30, remember that for me",
        "facts": ["Marlena Wójcik", "11:30"],
        "question": "Who is my dentist and what time was the March appointment?",
    },
    {
        "ts": "2026-03-09 17:03:00+01",
        "message": "new bike lock code is 7351, old one is dead",
        "facts": ["7351"],
        "question": "What is my bike lock code?",
    },
    {
        "ts": "2026-03-21 13:22:00+01",
        "message": "ordered the ergonomic chair, model Markus II, 899 zł, "
                   "delivery in two weeks",
        "facts": ["Markus II", "899"],
        "question": "Which ergonomic chair model did I order and for how much?",
    },
    {
        "ts": "2026-04-03 08:15:00+02",
        "message": "blood test results came: ferritin 87 ng/ml, doctor says "
                   "it's fine now",
        "facts": ["87"],
        "question": "What was my ferritin result from the April blood test?",
    },
    {
        "ts": "2026-04-12 23:55:00+02",
        "message": "the loud fan in the server rack is a Noctua NF-A12x25, "
                   "ordering a second one",
        "facts": ["NF-A12x25"],
        "question": "What fan model is in my server rack?",
    },
    {
        "ts": "2026-04-25 19:30:00+02",
        "message": "babcia powiedziała że do pierogów daje 3 łyżki kwaśnej "
                   "śmietany, zapisz to do przepisu",
        "facts": ["3 łyżki", "śmietan"],
        "question": "Ile kwaśnej śmietany daje babcia do pierogów?",
    },
    {
        "ts": "2026-05-06 07:58:00+02",
        "message": "got assigned parking spot B-14 at the office, level -2",
        "facts": ["B-14"],
        "question": "Which parking spot do I have at the office?",
    },
    {
        "ts": "2026-05-15 20:44:00+02",
        "message": "flight to Lisbon booked: LO1923 on May 21st, window seat",
        "facts": ["LO1923", "May 21"],
        "question": "What is my flight number to Lisbon and the date?",
    },
    {
        "ts": "2026-05-28 16:09:00+02",
        "message": "fyi the orange pi that runs the journal logger is "
                   "hostname orp, 192.168.0.80",
        "facts": ["orp", "192.168.0.80"],
        "question": "What is the hostname and IP of the orange pi running the journal logger?",
    },
]


# A canary TRANSCRIPT: its facts exist NOWHERE in chat history, so answering
# the retrieval:yt:* probes proves the transcript/chunk search path works.
CANARY_VIDEO = {
    "video_id": "seed_canary_rack_video",
    "title": "Quiet homelab: taming a screaming server rack",
    "transcript": (
        "Welcome back. Today we finally silence the rack. The stock fans were"
        " unbearable — measured 43 dB right next to the desk. I swapped them"
        " for a single Noctua NF-A12x25 on the exhaust and added rubber"
        " grommets on the rails. After the swap the meter reads 19 dB at one"
        " meter, which is basically inaudible over the room tone. Total cost"
        " of the whole silent-rack build came to 612 euro including the"
        " sound-dampening panels. The biggest win was actually the grommets —"
        " they killed the chassis resonance completely. Next episode we look"
        " at undervolting the switch."
    ),
}


def _target_dsn() -> str:
    url = os.environ.get("DATABASE_URL", "")
    dsn = url.replace("postgresql+asyncpg://", "postgresql://")
    db = dsn.rsplit("/", 1)[-1].split("?")[0]
    if not db.endswith("_autoloop"):
        raise SystemExit(
            f"[seed-retrieval] REFUSING: target DB {db!r} is not an _autoloop "
            "copy — never seed synthetic data into prod."
        )
    return dsn


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


async def _copy_messages(src, dst, *, limit: int | None) -> int:
    rows = await src.fetch(
        "SELECT timestamp, timestamp_unix, sender, message, chat_id,"
        " message_id, username, metadata, date FROM chat_messages"
        " ORDER BY timestamp" + (f" LIMIT {int(limit)}" if limit else "")
    )
    existing = await dst.fetchval(
        "SELECT count(*) FROM chat_messages WHERE metadata->>'seed_source' = $1",
        SEED_SOURCE_CORPUS,
    )
    if existing:
        print(f"[seed-retrieval] corpus already seeded ({existing} rows) — skipping copy")
        return 0
    inserted = 0
    batch: list[tuple] = []
    for r in rows:
        meta = dict(json.loads(r["metadata"]) if isinstance(r["metadata"], str)
                    else (r["metadata"] or {}))
        meta["seed_source"] = SEED_SOURCE_CORPUS
        batch.append((
            r["timestamp"], r["timestamp_unix"], r["sender"], r["message"],
            r["chat_id"], r["message_id"], r["username"], json.dumps(meta),
            r["date"],
        ))
        if len(batch) >= 1000:
            inserted += await _flush_messages(dst, batch)
            batch = []
    if batch:
        inserted += await _flush_messages(dst, batch)
    return inserted


async def _flush_messages(dst, batch: list[tuple]) -> int:
    await dst.executemany(
        "INSERT INTO chat_messages (timestamp, timestamp_unix, sender, message,"
        " chat_id, message_id, username, metadata, date)"
        " VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9)",
        batch,
    )
    return len(batch)


async def _plant_canaries(dst) -> int:
    existing = await dst.fetchval(
        "SELECT count(*) FROM chat_messages WHERE metadata->>'seed_source' = $1",
        SEED_SOURCE_CANARY,
    )
    if existing:
        print(f"[seed-retrieval] canaries already planted ({existing}) — skipping")
        return 0
    from datetime import datetime

    planted = 0
    for c in CANARIES:
        ts = datetime.fromisoformat(c["ts"])
        await dst.execute(
            "INSERT INTO chat_messages (timestamp, timestamp_unix, sender,"
            " message, chat_id, message_id, username, metadata, date)"
            " VALUES ($1, $2, 'user', $3, 'seed', NULL, 'dw', $4::jsonb, $5)",
            ts, ts.timestamp(), c["message"],
            json.dumps({"seed_source": SEED_SOURCE_CANARY}), ts.date(),
        )
        planted += 1
    return planted


async def _copy_youtube(src, dst, *, limit: int | None) -> int:
    existing = await dst.fetchval(
        "SELECT count(*) FROM youtube_videos WHERE transcript_status = 'seeded'"
    )
    if existing:
        print(f"[seed-retrieval] yt transcripts already seeded ({existing}) — skipping")
        return 0
    rows = await src.fetch(
        "SELECT video_id, yt_video_id, channel_id, title, transcript"
        " FROM youtube_videos WHERE length(transcript) > 500"
        " ORDER BY created_at DESC" + (f" LIMIT {int(limit)}" if limit else "")
    )
    inserted = 0
    for r in rows:
        await dst.execute(
            "INSERT INTO youtube_videos (video_id, yt_video_id, channel_id,"
            " title, transcript, transcript_status)"
            " VALUES ($1,$2,$3,$4,$5,'seeded')"
            " ON CONFLICT (video_id) DO NOTHING",
            f"seed_{r['video_id']}", r["yt_video_id"], None, r["title"],
            r["transcript"],
        )
        inserted += 1
    return inserted


async def _plant_canary_video(dst, *, embed: bool) -> int:
    """Insert the canary transcript and (when embedding) its searchable chunks
    into embedding_chunks — the path search-transcripts actually queries."""
    inserted = await dst.execute(
        "INSERT INTO youtube_videos (video_id, yt_video_id, title, transcript,"
        " transcript_status) VALUES ($1, $1, $2, $3, 'seeded')"
        " ON CONFLICT (video_id) DO NOTHING",
        CANARY_VIDEO["video_id"], CANARY_VIDEO["title"],
        CANARY_VIDEO["transcript"],
    )
    if not embed:
        return 0
    have = await dst.fetchval(
        "SELECT count(*) FROM embedding_chunks WHERE source_table ="
        " 'youtube_videos' AND source_id = $1", CANARY_VIDEO["video_id"],
    )
    if have:
        return 0
    from app.services.embeddings import chunk_text, get_embeddings_batch

    chunks = chunk_text(f"{CANARY_VIDEO['title']}\n{CANARY_VIDEO['transcript']}")
    vecs = get_embeddings_batch(chunks)
    for i, (c, v) in enumerate(zip(chunks, vecs)):
        await dst.execute(
            "INSERT INTO embedding_chunks (source_table, source_id,"
            " chunk_index, chunk_text, embedding)"
            " VALUES ('youtube_videos', $1, $2, $3, $4::vector)",
            CANARY_VIDEO["video_id"], i, c, _vec_literal(v),
        )
    return len(chunks)


async def _embed_messages(dst, *, batch_size: int = 256) -> int:
    """Embed seeded messages that don't have an embedding yet (OpenAI
    text-embedding-3-small — the SAME encoder prod search uses, so similarity
    behaviour in the loop matches production)."""
    from app.services.embeddings import get_embeddings_batch

    total = 0
    while True:
        rows = await dst.fetch(
            "SELECT id, message FROM chat_messages"
            " WHERE embedding IS NULL AND metadata->>'seed_source' IS NOT NULL"
            " AND length(message) >= 8 ORDER BY id LIMIT $1",
            batch_size,
        )
        if not rows:
            break
        vecs = get_embeddings_batch([r["message"][:8000] for r in rows])
        for r, v in zip(rows, vecs):
            await dst.execute(
                "UPDATE chat_messages SET embedding = $2::vector WHERE id = $1",
                r["id"], _vec_literal(v),
            )
        total += len(rows)
        print(f"[seed-retrieval] embedded {total} messages ...", flush=True)
    return total


async def seed(*, embed: bool = True, limit: int | None = None,
               yt_limit: int | None = None) -> dict[str, int]:
    import asyncpg

    dst_dsn = _target_dsn()
    src = await asyncpg.connect(V3_DSN)
    # READ-ONLY guard on the v3 source: any accidental write raises.
    await src.execute("SET default_transaction_read_only = on")
    dst = await asyncpg.connect(dst_dsn)
    try:
        copied = await _copy_messages(src, dst, limit=limit)
        canaries = await _plant_canaries(dst)
        yt = await _copy_youtube(src, dst, limit=yt_limit)
        canary_chunks = await _plant_canary_video(dst, embed=embed)
        embedded = 0
        if embed:
            embedded = await _embed_messages(dst)
        total = await dst.fetchval("SELECT count(*) FROM chat_messages")
        print(
            f"[seed-retrieval] done: corpus+{copied} canaries+{canaries} "
            f"yt+{yt} canary_video_chunks+{canary_chunks} embedded={embedded};"
            f" chat_messages total={total}"
        )
        return {"copied": copied, "canaries": canaries, "yt": yt,
                "canary_video_chunks": canary_chunks,
                "embedded": embedded, "total": int(total)}
    finally:
        await src.close()
        await dst.close()


def main(argv: list[str]) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="app seed-retrieval")
    p.add_argument("--no-embed", action="store_true",
                   help="skip embedding computation (faster; semantic probes will skip)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap copied corpus messages (default: all ~19.8k)")
    p.add_argument("--yt-limit", type=int, default=200,
                   help="cap copied YT transcripts (default 200; 0 = all)")
    args = p.parse_args(argv)
    asyncio.run(seed(
        embed=not args.no_embed, limit=args.limit,
        yt_limit=(args.yt_limit or None),
    ))
