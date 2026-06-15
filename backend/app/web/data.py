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
import json
import zlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.db.repos.activity_blocks import ActivityBlocksRepo
from app.db.repos.agent_notes import AgentNotesRepo
from app.db.repos.chat import ChatMessagesRepo
from app.db.repos.context_cache import ContextCacheRepo
from app.db.repos.emotional_state import EmotionalStateRepo
from app.db.repos.events import EventsRepo
from app.db.repos.goals import GoalsRepo
from app.db.repos.habits import HabitsRepo
from app.db.repos.memories import MemoriesRepo
from app.db.repos.persona_memory import PendingThoughtsRepo, PersonaInterestsRepo
from app.db.repos.persona_vibe import StyleEventsRepo, VibeStateRepo
from app.db.repos.priorities import PrioritiesRepo
from app.db.repos.todos import TodosRepo
from app.db.repos.user_mood import UserMoodRepo
from app.db.session import fetch_all, fetch_one, get_async_session


# ── shared helpers ────────────────────────────────────────────────────────────


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _age_hours(ts: Any, now: datetime | None = None) -> float | None:
    """Hours elapsed since ``ts`` (datetime), or ``None`` if absent/invalid.

    Naive datetimes are assumed UTC (the DB stores TIMESTAMPTZ; naive values
    only appear in tests/fixtures).
    """
    if not isinstance(ts, datetime):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    now = now or _now_utc()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return max(0.0, (now - ts).total_seconds() / 3600.0)


def freshness_class(ts: Any, now: datetime | None = None) -> str:
    """Health-chip freshness: green (<6h) → ``ok``, amber (<24h) → ``warn``,
    older → ``bad``, missing → ``""`` (neutral pill)."""
    age = _age_hours(ts, now)
    if age is None:
        return ""
    if age < 6:
        return "ok"
    if age < 24:
        return "warn"
    return "bad"


def _human_size(size: Any) -> str:
    """1234 → '1.2 KB'. Tolerates None/garbage → ''."""
    try:
        n = float(size)
    except (TypeError, ValueError):
        return ""
    if n < 0:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def _as_dict(value: Any) -> dict[str, Any]:
    """JSONB column → dict (handles dict, JSON string, None)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        with contextlib.suppress(Exception):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
    return {}


def _as_list(value: Any) -> list[str]:
    """JSONB array column → list of strings (handles list, JSON string, None)."""
    if isinstance(value, str) and value.strip():
        with contextlib.suppress(Exception):
            value = json.loads(value)
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v not in (None, "")]
    return []


def _f(value: Any, default: float | None = None) -> float | None:
    """Numeric column (incl. Decimal/str) → float, or ``default`` on junk."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_utc(ts: datetime) -> datetime:
    """Naive datetimes are assumed UTC (same policy as ``_age_hours``)."""
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts

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


async def recent_heartbeats(limit: int = 40) -> list[dict[str, Any]]:
    """Recent proactive autonomy heartbeat ticks with their decision, for the
    Heartbeat panel — so each wake-up is monitorable at a glance (mode, what it
    decided, category, the message/reasoning) without opening every run."""
    sql = """
        SELECT r.run_id, r.owner, r.status, r.started_at,
               a.payload AS decision
        FROM execution_runs r
        LEFT JOIN LATERAL (
            SELECT payload FROM execution_artifacts
            WHERE run_id = r.run_id AND artifact_type = 'persona_guidance'
            ORDER BY version DESC LIMIT 1
        ) a ON TRUE
        WHERE r.interaction_mode = 'heartbeat'
        ORDER BY r.started_at DESC
        LIMIT :limit
    """
    async with get_async_session() as s:
        rows = await fetch_all(s, sql, {"limit": limit})
    out: list[dict[str, Any]] = []
    for row in rows:
        d = row.get("decision")
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except (ValueError, TypeError):
                d = {}
        d = d if isinstance(d, dict) else {}
        out.append({
            "run_id": row.get("run_id"),
            "mode": (row.get("owner") or "").replace("persona/heartbeat-", "") or "—",
            "status": row.get("status"),
            "started_at": row.get("started_at"),
            "decision": d.get("decision") or "—",
            "category": d.get("category") or "—",
            "message": str(d.get("draft") or d.get("reasoning") or "")[:240],
        })
    return out


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
                # Ordered, interleaved timeline (narration/tool/result in stream
                # order). Absent for older traces → template falls back to the
                # flat text + tool_calls view.
                "trajectory": payload.get("trajectory") or [],
                "trajectory_count": payload.get("trajectory_count"),
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


# ── Mind tab (agent internal state: mood, vibe, interests, thoughts) ──────────

# (key, human label) pairs — column names confirmed against user_mood_state /
# persona_vibe_state in migrations/versions/001_initial_schema.py.
MOOD_DIMS: tuple[tuple[str, str], ...] = (
    ("energy", "energy"),
    ("valence", "valence"),
    ("stress", "stress"),
    ("engagement", "engagement"),
    ("openness", "openness"),
)

VIBE_DIMS: tuple[tuple[str, str], ...] = (
    ("w_warm_snarky", "warm-snarky"),
    ("w_dry_ironic", "dry-ironic"),
    ("w_caring_edge", "caring-edge"),
    ("w_playful_flirt", "playful-flirt"),
    ("w_debate_socratic", "debate-socratic"),
)

VIBE_AXES: tuple[tuple[str, str], ...] = (
    ("ironic_genuine_axis", "ironic ↔ genuine"),
    ("arousal_axis", "arousal"),
)


def _meter(value: Any) -> dict[str, Any]:
    """0..1 float → {value, pct} for a horizontal meter bar. Bad input → 0."""
    try:
        v = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        v = 0.0
    return {"value": round(v, 3), "pct": int(round(v * 100))}


def shape_mood(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """user_mood_state row → template dict (meters + dominant + freshness)."""
    if not row:
        return None
    return {
        "meters": [
            {"key": key, "label": label, **_meter(row.get(key))} for key, label in MOOD_DIMS
        ],
        "dominant_mood": row.get("dominant_mood") or "",
        "last_trigger": row.get("last_trigger") or "",
        "drift_count": row.get("drift_count"),
        "updated_at": row.get("updated_at"),
        "freshness": freshness_class(row.get("updated_at")),
    }


def shape_vibe(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """persona_vibe_state row → template dict (palette bars + axes).

    Axes are -1..+1 → rendered as a centred pct (0..100, 50 = neutral).
    Note: the vibe table stores no directives string — directives are composed
    at prompt-build time, so the panel shows weights/axes/trigger only.
    """
    if not row:
        return None
    bars = [
        {"key": key, "label": label, **_meter(row.get(key))} for key, label in VIBE_DIMS
    ]
    axes = []
    for key, label in VIBE_AXES:
        try:
            v = max(-1.0, min(1.0, float(row.get(key) or 0.0)))
        except (TypeError, ValueError):
            v = 0.0
        axes.append({"key": key, "label": label, "value": round(v, 3), "pct": int(round((v + 1.0) * 50))})
    return {
        "chat_id": row.get("chat_id"),
        "bars": bars,
        "axes": axes,
        "last_trigger": row.get("last_trigger") or "",
        "last_user_tone": row.get("last_user_tone") or "",
        "drift_count": row.get("drift_count"),
        "updated_at": row.get("updated_at"),
        "freshness": freshness_class(row.get("updated_at")),
    }


def shape_vibe_history(rows: list[dict[str, Any]], keep: int = 10) -> list[dict[str, Any]]:
    """Last ``keep`` vibe drift snapshots (newest last) → compact table rows."""
    shaped = []
    for r in rows[-keep:]:
        shaped.append(
            {
                "recorded_at": r.get("recorded_at"),
                "trigger": r.get("trigger") or "",
                "user_tone": r.get("user_tone") or "",
                "weights": [
                    {"key": key, "label": label, **_meter(r.get(key))} for key, label in VIBE_DIMS
                ],
            }
        )
    return shaped


def shape_interest(row: dict[str, Any]) -> dict[str, Any]:
    """persona_interests row → template dict."""
    return {
        "topic": row.get("topic") or "",
        "stance": row.get("stance") or "",
        "source": row.get("source") or "",
        "novelty": _meter(row.get("novelty_score")),
        "surface_count": row.get("surface_count") or 0,
        "last_surfaced_at": row.get("last_surfaced_at"),
        "created_at": row.get("created_at"),
    }


def shape_thought(row: dict[str, Any]) -> dict[str, Any]:
    """pending_thoughts row → template dict with parsed motivation breakdown."""
    breakdown = _as_dict(row.get("motivation_breakdown"))
    parts = []
    for k, v in breakdown.items():
        with contextlib.suppress(TypeError, ValueError):
            parts.append({"key": str(k), "value": round(float(v), 2)})
    parts.sort(key=lambda p: p["value"], reverse=True)
    return {
        "content": row.get("content") or "",
        "kind": row.get("kind") or "",
        "motivation": _meter(row.get("motivation_score")),
        "breakdown": parts,
        "created_at": row.get("created_at"),
        "consumed_at": row.get("consumed_at"),
        "consumed_by": row.get("consumed_by") or "",
    }


async def mind() -> dict[str, Any]:
    """Everything the Mind tab shows. Read-only; every sub-panel degrades to
    ``None``/``[]`` on an empty table so a fresh DB still renders."""
    mood_row = await UserMoodRepo().latest()
    vibe_row = await VibeStateRepo().latest()

    vibe_history: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    if vibe_row and vibe_row.get("chat_id") is not None:
        cid = int(vibe_row["chat_id"])
        with contextlib.suppress(Exception):
            vibe_history = shape_vibe_history(await VibeStateRepo().history(cid, limit=10))
        with contextlib.suppress(Exception):
            violations = await StyleEventsRepo().count_by_type(cid, since_hours=24)

    interests_rows = await PersonaInterestsRepo().list_active(limit=12)
    thoughts_rows = await PendingThoughtsRepo().list_recent(limit=10)

    return {
        "mood": shape_mood(mood_row),
        "vibe": shape_vibe(vibe_row),
        "vibe_history": vibe_history,
        "violations": violations,
        "interests": [shape_interest(r) for r in interests_rows],
        "thoughts": [shape_thought(r) for r in thoughts_rows],
    }


# ── persona prose traces (LLM audit log) ─────────────────────────────────────


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    """Nearest-rank percentile over an ascending list. Empty → None."""
    if not sorted_vals:
        return None
    idx = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


def shape_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    """One traces-list SQL row (JSON fields extracted as text) → template dict."""

    def _int(v: Any) -> int | None:
        with contextlib.suppress(TypeError, ValueError):
            return int(float(v))
        return None

    fallback_raw = row.get("fallback")
    fallback = (
        fallback_raw if isinstance(fallback_raw, bool)
        else str(fallback_raw or "").strip().lower() == "true"
    )
    return {
        "run_id": row.get("run_id") or "",
        "created_at": row.get("created_at"),
        "kind": row.get("kind") or "",
        "model": row.get("model") or "",
        "duration_ms": _int(row.get("duration_ms")),
        "input_tokens": _int(row.get("input_tokens")),
        "output_tokens": _int(row.get("output_tokens")),
        "fallback": fallback,
    }


def trace_stats(traces: list[dict[str, Any]], now: datetime | None = None) -> dict[str, Any]:
    """Stats strip over shaped trace rows: count_24h, p50/p95 duration, fallback rate."""
    now = now or _now_utc()
    count_24h = 0
    for t in traces:
        age = _age_hours(t.get("created_at"), now)
        if age is not None and age < 24:
            count_24h += 1
    durations = sorted(float(t["duration_ms"]) for t in traces if t.get("duration_ms") is not None)
    fallbacks = sum(1 for t in traces if t.get("fallback"))
    return {
        "total": len(traces),
        "count_24h": count_24h,
        "p50_ms": _percentile(durations, 0.50),
        "p95_ms": _percentile(durations, 0.95),
        "fallback_rate": round(fallbacks / len(traces), 3) if traces else 0.0,
    }


TRACE_STATS_WINDOW = 200


async def prose_traces(limit: int = 50) -> dict[str, Any]:
    """Newest-first persona_prose_trace list + a stats strip.

    Traces live in ``execution_artifacts`` (artifact_type='persona_prose_trace',
    written by app.telegram.persona_prose). The list query extracts only the
    cheap JSON scalars — full payloads (system prompts, raw output) are fetched
    one-at-a-time on the detail page. Stats are computed in Python over the last
    ``TRACE_STATS_WINDOW`` rows.
    """
    sql = """
        SELECT * FROM (
            SELECT DISTINCT ON (a.run_id)
                a.run_id, a.created_at,
                a.payload->>'kind' AS kind,
                a.payload->>'model' AS model,
                a.payload->>'duration_ms' AS duration_ms,
                a.payload->>'input_tokens' AS input_tokens,
                a.payload->>'output_tokens' AS output_tokens,
                a.payload->>'fallback_triggered' AS fallback
            FROM execution_artifacts a
            WHERE a.artifact_type = 'persona_prose_trace'
            ORDER BY a.run_id, a.version DESC
        ) sub
        ORDER BY created_at DESC
        LIMIT :window
    """
    async with get_async_session() as s:
        rows = await fetch_all(s, sql, {"window": TRACE_STATS_WINDOW})
    traces = [shape_trace_row(r) for r in rows]
    return {
        "traces": traces[:limit],
        "stats": trace_stats(traces),
        "window": TRACE_STATS_WINDOW,
        "cap": limit,
    }


def shape_trace_detail(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Full persona_prose_trace artifact row → detail-page dict.

    Everything renders inside escaped ``<pre>`` blocks in the template, so this
    only normalises shapes (payload may arrive as a JSON string).
    """
    if not row:
        return None
    payload = _as_dict(row.get("payload"))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = []
    context_summary = _as_dict(payload.get("context_summary"))
    guidance = _as_dict(payload.get("guidance"))
    return {
        "run_id": row.get("run_id") or "",
        "created_at": row.get("created_at"),
        "producer": row.get("producer") or "",
        "kind": payload.get("kind") or "",
        "model": payload.get("model") or "",
        "provider": payload.get("provider") or "",
        "duration_ms": payload.get("duration_ms"),
        "input_tokens": payload.get("input_tokens"),
        "output_tokens": payload.get("output_tokens"),
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
        "fallback": bool(payload.get("fallback_triggered")),
        "system_prompt": payload.get("system_prompt") or "",
        "messages": messages,
        "raw_output": payload.get("raw_output") or "",
        "thinking": payload.get("thinking") or "",
        "stripped_output": payload.get("stripped_output") or "",
        "delivered_text": payload.get("delivered_text") or "",
        "context_summary": context_summary,
        "guidance": guidance,
    }


async def prose_trace_detail(run_id: str) -> dict[str, Any] | None:
    """Latest persona_prose_trace artifact for one run, or ``None``."""
    sql = """
        SELECT run_id, producer, payload, created_at
        FROM execution_artifacts
        WHERE artifact_type = 'persona_prose_trace' AND run_id = :run_id
        ORDER BY version DESC
        LIMIT 1
    """
    async with get_async_session() as s:
        row = await fetch_one(s, sql, {"run_id": run_id})
    return shape_trace_detail(row)


# ── image gallery (rendered selfies/renders + camera captures) ────────────────

# The media kinds the gallery serves, mapped to their sub-dir under data_dir.
# Both live on the persistent fren_v4_data volume (mounted at /data) so they
# survive container recreates. Anything not in this map is rejected by the route.
MEDIA_KINDS: dict[str, str] = {
    "rendered": "rendered",   # ComfyUI selfies/renders copied here by render workers
    "captures": "captures",   # camera captures
}

# Only these extensions are ever listed or served — no arbitrary files leak out.
IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def media_root(kind: str) -> Path | None:
    """Resolved absolute dir for a media ``kind`` under ``settings.data_dir``.

    Returns ``None`` for an unknown kind (the route turns that into a 404). The
    returned path is resolved so the route can verify served files stay within.
    """
    sub = MEDIA_KINDS.get(kind)
    if sub is None:
        return None
    from app.settings import get_settings

    return (Path(get_settings().data_dir) / sub).resolve()


def safe_media_path(kind: str, name: str) -> Path | None:
    """Resolve ``<data_dir>/<kind>/<name>`` SAFELY, or ``None`` if rejected.

    Rejects anything that isn't a plain image filename living directly inside the
    allowed media dir: path-traversal (``..``, absolute paths, nested dirs/
    separators) and non-image extensions all return ``None``. The final resolved
    path is re-checked to be a direct child of the resolved media root, so even a
    symlink/``..`` that slips past the string checks can't escape the dir.
    """
    root = media_root(kind)
    if root is None:
        return None
    # No separators, no traversal, no absolute paths — a bare filename only.
    if not name or "/" in name or "\\" in name or "\x00" in name:
        return None
    if name != Path(name).name or name in (".", ".."):
        return None
    if Path(name).suffix.lower() not in IMAGE_EXTS:
        return None
    candidate = (root / name).resolve()
    # Belt-and-suspenders: the resolved file must sit DIRECTLY under the root.
    if candidate.parent != root:
        return None
    return candidate


async def _context_meta_by_filename(
    limit: int = 200,
) -> dict[str, dict[str, Any]]:
    """Map basename → {summary, created_at} from selfie/generated context_cache.

    Read-only best-effort enrichment: pulls recent context_cache rows tagged
    ``selfie``/``generated`` (or of those artifact_types) and indexes them by the
    basename of their ``file_path`` so the gallery can show a prompt preview next
    to a matching rendered/captured file. Returns ``{}`` on any DB error so the
    gallery still renders from the filesystem alone.
    """
    sql = """
        SELECT file_path, summary, created_at
        FROM context_cache
        WHERE (tags ?| ARRAY['selfie','generated','render','selfies']
               OR artifact_type IN ('selfie','generated','render','image'))
          AND file_path <> ''
          AND (expires_at IS NULL OR expires_at > NOW())
        ORDER BY created_at DESC
        LIMIT :limit
    """
    out: dict[str, dict[str, Any]] = {}
    with contextlib.suppress(Exception):
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {"limit": limit})
        for row in rows:
            fp = (row.get("file_path") or "").strip()
            if not fp:
                continue
            base = Path(fp).name
            # newest first → keep the first (most recent) seen per filename
            if base not in out:
                out[base] = {
                    "summary": (row.get("summary") or "").strip(),
                    "created_at": row.get("created_at"),
                }
    return out


def _list_dir_images(root: Path, kind: str) -> list[dict[str, Any]]:
    """List image files directly under ``root`` as gallery entries (no recurse).

    Each entry: ``{kind, name, mtime, size}``. Missing dir → ``[]``. Hidden files
    and non-image extensions are skipped. Pure filesystem, no DB.
    """
    items: list[dict[str, Any]] = []
    with contextlib.suppress(FileNotFoundError, NotADirectoryError, PermissionError):
        for entry in root.iterdir():
            if not entry.is_file() or entry.name.startswith("."):
                continue
            if entry.suffix.lower() not in IMAGE_EXTS:
                continue
            with contextlib.suppress(OSError):
                st = entry.stat()
                items.append(
                    {
                        "kind": kind,
                        "name": entry.name,
                        "mtime": datetime.fromtimestamp(st.st_mtime),
                        "size": st.st_size,
                    }
                )
    return items


# Valid gallery filter values: every media kind, plus "all" (no filter).
IMAGE_FILTERS: tuple[str, ...] = ("all", *MEDIA_KINDS)


def normalize_image_filter(value: Any) -> str:
    """Server-side validation of the gallery ``kind`` query param.

    Anything that isn't exactly a known media kind falls back to ``"all"`` —
    the filter never reaches the filesystem as raw user input.
    """
    if isinstance(value, str) and value in MEDIA_KINDS:
        return value
    return "all"


async def recent_images(limit: int = 60, kind: str = "all") -> dict[str, Any]:
    """Newest-first gallery listing, optionally filtered to one media kind.

    Lists images on the filesystem (rendered + captures), sorts by mtime newest
    first, caps at ``limit``, then enriches each with a matching context_cache
    prompt/summary by filename when available. ``kind`` is normalised via
    ``normalize_image_filter`` (unknown → "all"). Degrades to a pure-filesystem
    listing (no prompts) if the DB is unreachable, and to an empty list if no
    media dirs exist yet. Read-only throughout.
    """
    kind = normalize_image_filter(kind)
    kinds = list(MEDIA_KINDS) if kind == "all" else [kind]

    all_items: list[dict[str, Any]] = []
    for k in kinds:
        root = media_root(k)
        if root is not None:
            all_items.extend(_list_dir_images(root, k))

    all_items.sort(key=lambda i: i["mtime"], reverse=True)
    truncated = len(all_items) > limit
    all_items = all_items[:limit]

    meta = await _context_meta_by_filename()
    for item in all_items:
        m = meta.get(item["name"])
        item["prompt"] = m["summary"] if m else ""
        item["size_h"] = _human_size(item.get("size"))

    return {"images": all_items, "cap": limit, "truncated": truncated, "kind": kind}


# ── Life tab (goals / todos / habits / priorities) ────────────────────────────

# goals.level is constrained to 1..6 in the schema — indent depth is capped to
# the same six levels so a malformed tree can't push rows off-screen.
GOAL_MAX_DEPTH = 5


def shape_goal(row: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """One goals row → template dict (progress meter + overdue flag)."""
    now = now or _now_utc()
    status = (row.get("status") or "").lower()
    deadline = row.get("deadline")
    overdue = False
    if isinstance(deadline, datetime) and status == "active":
        overdue = _as_utc(deadline) < now
    pct = _f(row.get("progress_percent"), 0.0) or 0.0
    level = int(_f(row.get("level"), 1.0) or 1.0)
    return {
        "goal_id": row.get("goal_id") or "",
        "parent_goal_id": row.get("parent_goal_id") or "",
        "level": max(1, min(level, GOAL_MAX_DEPTH + 1)),
        "title": row.get("title") or "",
        "status": status or "active",
        "priority": row.get("priority") or "",
        "progress": {"pct": int(max(0.0, min(100.0, pct)))},
        "deadline": deadline,
        "overdue": overdue,
        "depth": 0,  # assigned by shape_goal_tree
    }


def shape_goal_tree(
    rows: list[dict[str, Any]], now: datetime | None = None
) -> list[dict[str, Any]]:
    """goals rows → flattened DFS order with ``depth`` for indentation.

    Robust against dirty data: orphans (parent_goal_id pointing at a missing/
    archived goal) become roots indented by their own level, self-parents are
    treated as roots, and cycles can't recurse (visited set) — nothing crashes,
    every goal is rendered exactly once.
    """
    now = now or _now_utc()
    shaped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        s = shape_goal(row, now)
        gid = s["goal_id"]
        if not gid or gid in shaped:
            continue
        shaped[gid] = s
        order.append(gid)

    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for gid in order:
        pid = shaped[gid]["parent_goal_id"]
        if pid and pid != gid and pid in shaped:
            children.setdefault(pid, []).append(gid)
        else:
            roots.append(gid)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def walk(gid: str, depth: int) -> None:
        if gid in seen:
            return
        seen.add(gid)
        node = shaped[gid]
        node["depth"] = max(0, min(depth, GOAL_MAX_DEPTH))
        out.append(node)
        for child in children.get(gid, []):
            walk(child, depth + 1)

    for gid in roots:
        walk(gid, shaped[gid]["level"] - 1)
    # pure cycles (a→b→a, no root) — emit flat so nothing silently disappears
    for gid in order:
        walk(gid, shaped[gid]["level"] - 1)
    return out


TODO_UPCOMING_DAYS = 7


def shape_todo(row: dict[str, Any]) -> dict[str, Any]:
    """One todos row → template dict (due label computed by bucket_todos)."""
    return {
        "todo_id": row.get("todo_id") or "",
        "title": row.get("title") or "",
        "status": (row.get("status") or "").lower(),
        "priority": (row.get("priority") or "").lower(),
        "category": row.get("category") or "",
        "linked_goal_id": row.get("linked_goal_id") or "",
        "goal_title": "",  # filled by life() from the goals listing
        "due_label": "",
    }


def bucket_todos(
    rows: list[dict[str, Any]], now: datetime | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Open todos → overdue / today / upcoming (next 7 days) buckets.

    The due moment is ``deadline`` (TIMESTAMPTZ) when present, else ``date``
    (DATE, never NULL in the schema). Deadline-carrying todos go overdue the
    second the deadline passes; date-only todos go overdue the day after.
    Completed/cancelled rows and anything due beyond the window are dropped.
    Buckets come back sorted soonest-first.
    """
    now = now or _now_utc()
    today = now.date()
    horizon = today + timedelta(days=TODO_UPCOMING_DAYS)
    buckets: dict[str, list[tuple[Any, dict[str, Any]]]] = {
        "overdue": [], "today": [], "upcoming": [],
    }
    for row in rows:
        status = (row.get("status") or "").lower()
        if status not in ("pending", "in_progress"):
            continue
        deadline = row.get("deadline")
        ddate = row.get("date")
        item = shape_todo(row)
        if isinstance(deadline, datetime):
            due = _as_utc(deadline)
            item["due_label"] = due.strftime("%m-%d %H:%M")
            if due < now:
                key = "overdue"
            elif due.date() == today:
                key = "today"
            elif due.date() <= horizon:
                key = "upcoming"
            else:
                continue
            sort_key = (due.date().isoformat(), due.strftime("%H:%M"))
        elif isinstance(ddate, date) and not isinstance(ddate, datetime):
            item["due_label"] = ddate.strftime("%m-%d")
            if ddate < today:
                key = "overdue"
            elif ddate == today:
                key = "today"
            elif ddate <= horizon:
                key = "upcoming"
            else:
                continue
            sort_key = (ddate.isoformat(), "")
        else:
            continue  # no due information → not shown in the dated buckets
        buckets[key].append((sort_key, item))
    return {
        key: [item for _, item in sorted(pairs, key=lambda p: p[0])]
        for key, pairs in buckets.items()
    }


HABIT_RATE_DAYS = 30


def shape_habits(
    habit_rows: list[dict[str, Any]],
    due_today_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """habits + due-today occurrences + 30d occurrence stats → template dict."""
    stats_by_id: dict[str, dict[str, Any]] = {}
    for r in stats_rows:
        hid = r.get("habit_id")
        if hid:
            stats_by_id[hid] = r

    due_today = [
        {
            "habit_id": r.get("habit_id") or "",
            "title": r.get("habit_title") or r.get("title") or "",
            "importance_level": int(_f(r.get("importance_level"), 0.0) or 0.0),
        }
        for r in due_today_rows
    ]
    due_ids = {d["habit_id"] for d in due_today}

    habits = []
    for r in habit_rows:
        hid = r.get("habit_id") or ""
        st = stats_by_id.get(hid)
        rate = None
        if st:
            scheduled = int(_f(st.get("scheduled"), 0.0) or 0.0)
            completed = int(_f(st.get("completed"), 0.0) or 0.0)
            if scheduled > 0:
                rate = {
                    "pct": int(round(100.0 * completed / scheduled)),
                    "completed": completed,
                    "scheduled": scheduled,
                }
        habits.append(
            {
                "habit_id": hid,
                "title": r.get("title") or "",
                "frequency_type": r.get("frequency_type") or "",
                "streak": int(_f(r.get("current_streak"), 0.0) or 0.0),
                "best_streak": int(_f(r.get("best_streak"), 0.0) or 0.0),
                "rate": rate,
                "due_today": hid in due_ids,
            }
        )
    return {"due_today": due_today, "habits": habits}


# Eisenhower quadrants from the priorities table's importance × immediacy
# (both NUMERIC 0..1 — verified in migrations 001_initial_schema.py).
PRIORITY_QUADRANTS: tuple[tuple[str, str], ...] = (
    ("do", "important · immediate"),
    ("plan", "important · can wait"),
    ("delegate", "immediate · low importance"),
    ("drop", "low · later"),
)

QUADRANT_THRESHOLD = 0.5
MISALIGN_THRESHOLD = 0.15


def assign_quadrant(importance: Any, immediacy: Any) -> str:
    """importance × immediacy (0..1) → Eisenhower quadrant key."""
    hi_imp = (_f(importance, 0.0) or 0.0) >= QUADRANT_THRESHOLD
    hi_imm = (_f(immediacy, 0.0) or 0.0) >= QUADRANT_THRESHOLD
    if hi_imp and hi_imm:
        return "do"
    if hi_imp:
        return "plan"
    if hi_imm:
        return "delegate"
    return "drop"


def shape_priorities(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """priorities rows → quadrant lists with misalignment flags.

    ``misaligned`` = audited ``real_importance`` deviating from the stated
    importance by ≥ MISALIGN_THRESHOLD (uses the generated ``importance_delta``
    column when present, recomputes otherwise; never-audited rows are never
    flagged).
    """
    quadrants: dict[str, list[dict[str, Any]]] = {k: [] for k, _ in PRIORITY_QUADRANTS}
    misaligned_count = 0
    for r in rows:
        imp = _f(r.get("importance"))
        imm = _f(r.get("immediacy"))
        real = _f(r.get("real_importance"))
        delta = _f(r.get("importance_delta"))
        if delta is None and real is not None and imp is not None:
            delta = real - imp
        misaligned = delta is not None and abs(delta) >= MISALIGN_THRESHOLD
        if misaligned:
            misaligned_count += 1
        quadrants[assign_quadrant(imp, imm)].append(
            {
                "priority_id": r.get("priority_id") or "",
                "title": r.get("title") or "",
                "category": r.get("category") or "",
                "importance": imp,
                "immediacy": imm,
                "real_importance": real,
                "delta": delta,
                "misaligned": misaligned,
            }
        )
    return {"quadrants": quadrants, "misaligned": misaligned_count, "total": len(rows)}


async def life() -> dict[str, Any]:
    """Everything the Life tab shows. Read-only; each sub-panel independently
    degrades to empty on a missing table / unreachable DB so the tab always
    renders."""
    goals_rows: list[dict[str, Any]] = []
    todos_rows: list[dict[str, Any]] = []
    habit_rows: list[dict[str, Any]] = []
    due_today_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    prio_rows: list[dict[str, Any]] = []
    with contextlib.suppress(Exception):
        goals_rows = await GoalsRepo().list_with_children(limit=300)
    with contextlib.suppress(Exception):
        todos_rows = await TodosRepo().list(status="pending", limit=300)
    with contextlib.suppress(Exception):
        habit_rows = await HabitsRepo().list(status="active", limit=100)
    with contextlib.suppress(Exception):
        due_today_rows = await HabitsRepo().get_due_today()
    with contextlib.suppress(Exception):
        stats_rows = await HabitsRepo().completion_stats(days=HABIT_RATE_DAYS)
    with contextlib.suppress(Exception):
        prio_rows = await PrioritiesRepo().list(status="active", limit=100)

    goals = shape_goal_tree(goals_rows)
    goal_titles = {g["goal_id"]: g["title"] for g in goals}
    todos = bucket_todos(todos_rows)
    for bucket in todos.values():
        for t in bucket:
            t["goal_title"] = goal_titles.get(t["linked_goal_id"], "")

    return {
        "goals": goals,
        "todos": todos,
        "habits": shape_habits(habit_rows, due_today_rows, stats_rows),
        "priorities": shape_priorities(prio_rows),
    }


# ── /events page (event timeline + category bar chart + daily strip) ──────────

EVENT_TIMELINE_DAYS = 7
EVENT_CHART_DAYS = 30
EVENT_TIMELINE_CAP = 200


def normalize_event_category(value: Any, categories: list[str]) -> str:
    """Server-side validation of the ``category`` query param.

    Anything that isn't exactly one of the categories present in the DB falls
    back to ``"all"`` — raw user input never reaches the SQL filter.
    """
    if isinstance(value, str) and value in categories:
        return value
    return "all"


def category_bars(counts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """count_by_category rows → bar-chart entries scaled to the max count."""
    shaped = []
    for row in counts:
        cat = row.get("category")
        n = int(_f(row.get("count"), 0.0) or 0.0)
        if not cat or n < 0:
            continue
        shaped.append({"category": str(cat), "count": n, "pct": 0})
    max_n = max((b["count"] for b in shaped), default=0)
    if max_n > 0:
        for b in shaped:
            b["pct"] = int(round(100.0 * b["count"] / max_n))
    return shaped


def daily_strip(
    rows: list[dict[str, Any]], *, days: int = EVENT_CHART_DAYS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Sparse daily_counts rows → dense last-``days`` series scaled to the max.

    Missing days are filled with zero so the strip is a fixed-width calendar
    ending today.
    """
    now = now or _now_utc()
    today = now.date()
    counts: dict[date, int] = {}
    for row in rows:
        d = row.get("date")
        if isinstance(d, datetime):
            d = d.date()
        if isinstance(d, date):
            counts[d] = int(_f(row.get("count"), 0.0) or 0.0)
    series = []
    for offset in range(days, -1, -1):
        d = today - timedelta(days=offset)
        series.append({"date": d.isoformat(), "count": counts.get(d, 0), "pct": 0})
    max_n = max((p["count"] for p in series), default=0)
    if max_n > 0:
        for p in series:
            p["pct"] = int(round(100.0 * p["count"] / max_n))
    return {"days": series, "max": max_n, "total": sum(p["count"] for p in series)}


def shape_event_row(row: dict[str, Any]) -> dict[str, Any]:
    """One events row → timeline-table dict (value/cost/duration condensed)."""
    parts: list[str] = []
    value = row.get("value")
    if value not in (None, ""):
        unit = row.get("unit") or ""
        parts.append(f"{value} {unit}".strip())
    quantity = _f(row.get("quantity"))
    if quantity is not None and not value:
        parts.append(f"×{quantity:g}")
    cost = _f(row.get("cost"))
    if cost is not None:
        parts.append(f"{cost:g} {row.get('currency') or ''}".strip())
    duration = _f(row.get("duration_minutes"))
    if duration is not None:
        parts.append(f"{int(duration)} min")
    return {
        "event_id": row.get("event_id") or "",
        "occurred_at": row.get("occurred_at"),
        "category": row.get("category") or "",
        "subcategory": row.get("subcategory") or "",
        "title": row.get("title") or "",
        "detail": " · ".join(parts),
        "source": row.get("source") or "",
    }


async def events_page(category: Any = "all") -> dict[str, Any]:
    """Everything the /events page shows. Read-only, degrades to empty."""
    counts: list[dict[str, Any]] = []
    with contextlib.suppress(Exception):
        counts = await EventsRepo().count_by_category(days=EVENT_CHART_DAYS)
    bars = category_bars(counts)
    categories = [b["category"] for b in bars]
    category = normalize_event_category(category, categories)

    timeline_rows: list[dict[str, Any]] = []
    date_from = (_now_utc().date() - timedelta(days=EVENT_TIMELINE_DAYS)).isoformat()
    with contextlib.suppress(Exception):
        timeline_rows = await EventsRepo().list(
            category=None if category == "all" else category,
            date_from=date_from,
            limit=EVENT_TIMELINE_CAP,
        )

    strip: dict[str, Any] | None = None
    if category != "all":
        daily_rows: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            daily_rows = await EventsRepo().daily_counts(category, days=EVENT_CHART_DAYS)
        strip = daily_strip(daily_rows, days=EVENT_CHART_DAYS)

    return {
        "category": category,
        "categories": categories,
        "bars": bars,
        "events": [shape_event_row(r) for r in timeline_rows],
        "strip": strip,
        "days": EVENT_TIMELINE_DAYS,
        "chart_days": EVENT_CHART_DAYS,
        "cap": EVENT_TIMELINE_CAP,
    }


# ── /artifacts page (context_cache gallery) ───────────────────────────────────

ARTIFACTS_CAP = 100
ARTIFACT_SUMMARY_PREVIEW = 220
ARTIFACT_PALETTE_SIZE = 6
SEARCH_Q_MAX_LEN = 80


def artifact_type_class(artifact_type: Any) -> str:
    """Stable per-type CSS color class (art-c0..art-c5). crc32, not hash() —
    hash() is salted per process, which would reshuffle colors every restart."""
    name = artifact_type if isinstance(artifact_type, str) else ""
    return f"art-c{zlib.crc32(name.encode('utf-8')) % ARTIFACT_PALETTE_SIZE}"


def normalize_artifact_type(value: Any, types: list[str]) -> str:
    """Server-side validation of the ``type`` query param — anything that isn't
    exactly a type present in the DB falls back to ``"all"``."""
    if isinstance(value, str) and value in types:
        return value
    return "all"


def normalize_search_q(value: Any) -> str:
    """Server-side normalisation of the ``q`` search param: non-strings → "",
    whitespace collapsed, capped at SEARCH_Q_MAX_LEN chars. (ILIKE wildcard
    escaping happens in the repo; HTML escaping in the autoescaped template.)"""
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(value.split())
    return cleaned[:SEARCH_Q_MAX_LEN]


def shape_artifact(row: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """One context_cache row → gallery dict (badge class, preview, tags)."""
    now = now or _now_utc()
    summary = (row.get("summary") or "").strip()
    atype = row.get("artifact_type") or ""
    expires_at = row.get("expires_at")
    expired = isinstance(expires_at, datetime) and _as_utc(expires_at) < now
    entity_type = row.get("entity_type") or ""
    entity_id = row.get("entity_id") or ""
    return {
        "cache_id": row.get("cache_id") or "",
        "artifact_type": atype,
        "type_class": artifact_type_class(atype),
        "summary": summary,
        "preview": summary[:ARTIFACT_SUMMARY_PREVIEW],
        "has_more": len(summary) > ARTIFACT_SUMMARY_PREVIEW,
        "tags": _as_list(row.get("tags")),
        "source_agent": row.get("source_agent") or "",
        "entity": f"{entity_type}:{entity_id}" if entity_type and entity_id else "",
        "created_at": row.get("created_at"),
        "expires_at": expires_at,
        "expired": expired,
    }


async def artifacts_page(atype: Any = "all", q: Any = "") -> dict[str, Any]:
    """Everything the /artifacts page shows. Read-only, degrades to empty."""
    type_rows: list[dict[str, Any]] = []
    with contextlib.suppress(Exception):
        type_rows = await ContextCacheRepo().distinct_types()
    types = []
    for row in type_rows:
        name = row.get("artifact_type")
        if not name:
            continue
        types.append(
            {
                "name": str(name),
                "n": int(_f(row.get("n"), 0.0) or 0.0),
                "type_class": artifact_type_class(str(name)),
            }
        )
    type_names = [t["name"] for t in types]
    atype = normalize_artifact_type(atype, type_names)
    q = normalize_search_q(q)

    rows: list[dict[str, Any]] = []
    with contextlib.suppress(Exception):
        rows = await ContextCacheRepo().list_newest(
            artifact_type=None if atype == "all" else atype,
            q=q or None,
            limit=ARTIFACTS_CAP,
        )
    return {
        "type": atype,
        "types": types,
        "q": q,
        "artifacts": [shape_artifact(r) for r in rows],
        "cap": ARTIFACTS_CAP,
    }


# ── health strip ──────────────────────────────────────────────────────────────


async def db_ok() -> bool:
    try:
        async with get_async_session() as s:
            await fetch_one(s, "SELECT 1 AS ok", {})
        return True
    except Exception:
        return False


async def health() -> dict[str, Any]:
    """Top-strip health summary: db reachability, counts, last scheduler fire,
    plus freshness chips (green <6h / amber <24h / red older) for the agent's
    internal-state tables (mood / vibe / interests)."""
    info: dict[str, Any] = {
        "db_ok": False,
        "chat_count": 0,
        "run_count": 0,
        "persona_count": 0,
        "last_run_at": None,
        "last_chat_at": None,
        "qwen_url": "",
        "mood_updated_at": None,
        "mood_fresh": "",
        "vibe_updated_at": None,
        "vibe_fresh": "",
        "interests_updated_at": None,
        "interests_fresh": "",
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
                    (SELECT MAX(timestamp) FROM chat_messages) AS last_chat_at,
                    (SELECT MAX(updated_at) FROM user_mood_state) AS mood_updated_at,
                    (SELECT MAX(updated_at) FROM persona_vibe_state) AS vibe_updated_at,
                    (SELECT MAX(GREATEST(created_at, COALESCE(last_surfaced_at, created_at)))
                       FROM persona_interests) AS interests_updated_at
                """,
                {},
            )
            if counts:
                info.update(counts)
    except Exception:
        # DB unreachable — leave defaults, db_ok stays False.
        pass
    info["mood_fresh"] = freshness_class(info.get("mood_updated_at"))
    info["vibe_fresh"] = freshness_class(info.get("vibe_updated_at"))
    info["interests_fresh"] = freshness_class(info.get("interests_updated_at"))
    return info
