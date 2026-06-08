"""Read-only data access for the monitoring dashboard.

Every function here is READ-ONLY: it either delegates to an existing repo or runs
a small read-only ``SELECT`` through the shared ``app.db.session`` helpers. The
dashboard NEVER writes to the DB and NEVER triggers an agent.

The functions return plain dicts/lists shaped for the templates and are written
to degrade gracefully on empty tables (returning ``[]`` / ``None`` / zeroed
counts) so the dashboard renders even on a fresh DB.
"""

from __future__ import annotations

import contextlib
from datetime import date
from typing import Any

from app.db.repos.activity_blocks import ActivityBlocksRepo
from app.db.repos.agent_notes import AgentNotesRepo
from app.db.repos.chat import ChatMessagesRepo
from app.db.repos.emotional_state import EmotionalStateRepo
from app.db.repos.memories import MemoriesRepo
from app.db.session import fetch_all, fetch_one, get_async_session

# ── persona_response classification ───────────────────────────────────────────


def classify_persona_response(payload: dict[str, Any]) -> tuple[bool, str]:
    """Classify a persona_response artifact payload as delivered vs skipped.

    A response counts as DELIVERED when it carries non-empty ``delivered_text``
    and is not explicitly flagged as a skip. Everything else (explicit
    ``skipped``/``kind == 'skip'``, or empty/missing ``delivered_text``) is a
    SKIP.

    Returns ``(is_skip, delivered_text)``.
    """
    payload = payload or {}
    text = (payload.get("delivered_text") or "").strip()
    explicit_skip = bool(payload.get("skipped")) or payload.get("kind") == "skip"
    delivered_flag = payload.get("delivered")
    if explicit_skip or not text or delivered_flag is False:
        return True, text
    return False, text


# ── proactive messages (persona_response artifacts) ───────────────────────────


async def recent_persona_responses(limit: int = 40) -> dict[str, Any]:
    """Latest-version persona_response per run, newest first, with counts.

    A single run can carry several persona_response versions (e.g. the prose
    artifact then a later ``skip`` artifact). We collapse to the latest version
    per run via ``DISTINCT ON`` so the dashboard shows the final outcome.
    """
    sql = """
        SELECT DISTINCT ON (a.run_id)
            a.run_id, a.version, a.producer, a.payload, a.created_at, r.owner
        FROM execution_artifacts a
        LEFT JOIN execution_runs r ON r.run_id = a.run_id
        WHERE a.artifact_type = 'persona_response'
        ORDER BY a.run_id, a.version DESC
    """
    async with get_async_session() as s:
        rows = await fetch_all(s, sql, {})

    items: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") or {}
        is_skip, text = classify_persona_response(payload)
        items.append(
            {
                "run_id": row["run_id"],
                "owner": row.get("owner") or payload.get("owner") or "",
                "producer": row.get("producer") or "",
                "kind": payload.get("kind") or "",
                "created_at": row.get("created_at"),
                "is_skip": is_skip,
                "delivered_text": text,
                # normalised key for repetition grouping
                "norm": _normalise_for_repeat(text),
            }
        )
    items.sort(key=lambda i: (i["created_at"] is not None, i["created_at"]), reverse=True)

    delivered = [i for i in items if not i["is_skip"]]
    # Flag repeats: any delivered message whose normalised text appears more than
    # once gets the same repeat-group number so the template can badge them.
    groups: dict[str, int] = {}
    counts: dict[str, int] = {}
    for i in delivered:
        counts[i["norm"]] = counts.get(i["norm"], 0) + 1
    next_group = 1
    for i in items:
        if not i["is_skip"] and counts.get(i["norm"], 0) > 1:
            if i["norm"] not in groups:
                groups[i["norm"]] = next_group
                next_group += 1
            i["repeat_group"] = groups[i["norm"]]
        else:
            i["repeat_group"] = None

    skipped_total = sum(1 for i in items if i["is_skip"])
    items = items[:limit]
    return {
        "messages": items,
        "delivered_count": len(delivered),
        "skipped_count": skipped_total,
        "total": len(rows),
        "repeat_groups": next_group - 1,
    }


def _normalise_for_repeat(text: str) -> str:
    """Cheap normalisation so near-identical proactive messages collide.

    Lowercases, strips punctuation/whitespace and keeps the first ~120 chars —
    enough to make obvious copy-paste repeats group together without a full
    similarity model.
    """
    if not text:
        return ""
    cleaned = "".join(c for c in text.lower() if c.isalnum() or c.isspace())
    cleaned = " ".join(cleaned.split())
    return cleaned[:120]


# ── agent runs ────────────────────────────────────────────────────────────────


async def recent_runs(limit: int = 30) -> list[dict[str, Any]]:
    """Recent execution_runs with their artifact counts."""
    sql = """
        SELECT r.run_id, r.owner, r.status, r.interaction_mode, r.domain,
               r.started_at, r.completed_at,
               COALESCE(c.n, 0) AS artifact_count
        FROM execution_runs r
        LEFT JOIN (
            SELECT run_id, COUNT(*) AS n
            FROM execution_artifacts GROUP BY run_id
        ) c ON c.run_id = r.run_id
        ORDER BY r.started_at DESC
        LIMIT :limit
    """
    async with get_async_session() as s:
        return await fetch_all(s, sql, {"limit": limit})


# ── single run detail (view-the-session) ──────────────────────────────────────


async def run_detail(run_id: str) -> dict[str, Any] | None:
    """One run's header + trajectory + persona artifacts, for ``/run/{id}``.

    Read-only. Returns ``None`` when the run row doesn't exist. The trajectory
    comes from the ``run_trace`` artifact the runner persists (assistant output +
    ordered tool calls); it may be missing for older runs, in which case
    ``trace`` is ``None`` and the page shows "no trajectory captured".
    """
    run_sql = """
        SELECT run_id, owner, status, interaction_mode, domain,
               started_at, completed_at, contract_passed
        FROM execution_runs WHERE run_id = :run_id
    """
    art_sql = """
        SELECT artifact_type, version, producer, payload, created_at
        FROM execution_artifacts WHERE run_id = :run_id
        ORDER BY artifact_type, version DESC
    """
    async with get_async_session() as s:
        run = await fetch_one(s, run_sql, {"run_id": run_id})
        if not run:
            return None
        arts = await fetch_all(s, art_sql, {"run_id": run_id})

    run = dict(run)
    trace: dict[str, Any] | None = None
    persona: list[dict[str, Any]] = []
    seen_persona: set[str] = set()
    for a in arts:
        atype = a.get("artifact_type")
        payload = a.get("payload") or {}
        if isinstance(payload, str):
            import json

            try:
                payload = json.loads(payload)
            except Exception:  # noqa: BLE001
                payload = {"content": payload}
        if atype == "run_trace" and trace is None:
            # rows are ordered version DESC within type → first is the latest
            trace = {
                "text": payload.get("text") or "",
                "tool_calls": payload.get("tool_calls") or [],
                "tool_call_count": payload.get("tool_call_count"),
                "ok": payload.get("ok"),
                "error": payload.get("error"),
                "producer": a.get("producer") or "",
            }
        elif atype in ("persona_guidance", "persona_response"):
            # collapse to the latest version per type (rows are version DESC)
            if atype in seen_persona:
                continue
            seen_persona.add(atype)
            persona.append(
                {
                    "artifact_type": atype,
                    "producer": a.get("producer") or "",
                    "created_at": a.get("created_at"),
                    "payload": payload,
                }
            )

    return {"run": run, "trace": trace, "persona": persona}


# ── context the agents are fed ────────────────────────────────────────────────


async def conversation_digest() -> str | None:
    """Render the current conversation_digest note value's ``digest`` field."""
    note = await AgentNotesRepo().get("conversation_digest")
    if not note:
        return None
    val = note.get("note_value")
    if isinstance(val, dict):
        return val.get("digest") or None
    if isinstance(val, str):
        return val or None
    return None


async def inner_monologue(limit: int = 8) -> list[dict[str, Any]]:
    """Recent inner_monologue memories (tagged inner_monologue)."""
    return await MemoriesRepo().search_by_tags(["inner_monologue"], limit=limit)


async def emotional_state() -> dict[str, Any] | None:
    return await EmotionalStateRepo().get_current()


async def recent_activity_blocks(hours: int = 12) -> list[dict[str, Any]]:
    repo = ActivityBlocksRepo()
    blocks = await repo.get_recent_blocks(hours=hours)
    if blocks:
        return blocks
    # fall back to today's blocks if nothing in the recent window
    with contextlib.suppress(Exception):
        return await repo.get_all_blocks(date.today())
    return []


# ── chat ──────────────────────────────────────────────────────────────────────


async def recent_chat(limit: int = 30) -> list[dict[str, Any]]:
    """Recent chat messages, oldest→newest (timeline order)."""
    rows = await ChatMessagesRepo().get_recent(limit=limit)
    return list(reversed(rows))


# ── health strip ──────────────────────────────────────────────────────────────


async def db_ok() -> bool:
    try:
        async with get_async_session() as s:
            await fetch_one(s, "SELECT 1 AS ok", {})
        return True
    except Exception:
        return False


async def health() -> dict[str, Any]:
    """Top-strip health summary: db reachability, counts, last scheduler fire."""
    info: dict[str, Any] = {
        "db_ok": False,
        "chat_count": 0,
        "run_count": 0,
        "persona_count": 0,
        "last_run_at": None,
        "last_chat_at": None,
        "qwen_url": "",
    }
    from app.settings import get_settings

    with contextlib.suppress(Exception):
        info["qwen_url"] = get_settings().local_llm_base_url

    try:
        async with get_async_session() as s:
            await fetch_one(s, "SELECT 1 AS ok", {})
            info["db_ok"] = True
            counts = await fetch_one(
                s,
                """
                SELECT
                    (SELECT COUNT(*) FROM chat_messages) AS chat_count,
                    (SELECT COUNT(*) FROM execution_runs) AS run_count,
                    (SELECT COUNT(*) FROM execution_artifacts
                       WHERE artifact_type = 'persona_response') AS persona_count,
                    (SELECT MAX(started_at) FROM execution_runs) AS last_run_at,
                    (SELECT MAX(timestamp) FROM chat_messages) AS last_chat_at
                """,
                {},
            )
            if counts:
                info.update(counts)
    except Exception:
        # DB unreachable — leave defaults, db_ok stays False.
        pass
    return info
