"""Conversation digest — rolling situational summary for ALL agents.

v4 port of v3 ``scripts/conversation_digest.py``. Combines recent chat, active
todos/goals/priorities/habits, today's events, active nudge campaigns, recent
inner-monologue thoughts, recent activity blocks (room state + any Garmin
health snapshot captured at block time) into a single factual digest, then
stores it in ``agent_notes['conversation_digest']`` with a TTL.

The scheduler's ``_enrich_prompt`` reads that note and prepends the digest to
EVERY proactive/cron agent invocation, so independent agents share an evolving
view of the user's current state instead of looping on the one seed todo.

LLM: the LOCAL qwen vLLM (``app.vllm_resolve.get_llm_endpoint("fast")``) — NOT
z.ai (the z.ai-via-opencode rule covers the GLM teacher/judge, not this local
summariser). Every data fetch is best-effort: a missing repo/method contributes
nothing rather than crashing the run.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.vllm_resolve import get_llm_endpoint

LLM_API_URL, LLM_MODEL = get_llm_endpoint("fast")

DIGEST_NOTE_KEY = "conversation_digest"
DIGEST_EXPIRES_HOURS = 4
DEFAULT_LOOKBACK_HOURS = 12
# Every Nth incremental update, do a full regen to flush stale accumulation.
FULL_REGEN_EVERY_N = 6
# Regenerate even without new chat after this many minutes (structured data
# like habits/activity/health changes independently of chat).
FORCE_REFRESH_MINUTES = 30
MAX_INPUT_CHARS = 300_000
WARSAW_TZ = ZoneInfo("Europe/Warsaw")


DIGEST_PROMPT = """\
You are a context summarizer for a multi-agent AI assistant system. Produce a \
situational digest prepended to every agent invocation so ALL agents (cron \
nudgers, periodic checkers, chat agents) understand what's happening in the \
user's life RIGHT NOW. Without it, agents repeat topics already discussed and \
nudge about things already addressed.

## Rules
- Focus on CURRENT STATE first, then recent history for context.
- Capture explicit user boundaries: "don't remind me about X", "later about Y".
- Note topics ALREADY discussed and their outcomes (so agents don't repeat).
- Note nudges/reminders ALREADY sent and whether the user acknowledged them.
- Include active goals/priorities status so nudge agents know what's stalled.
- Keep it under 800 words — this gets prepended to every agent prompt.
- Bullet points, direct, factual only. Third person ("The user is...").
- NO personality, NO commentary, NO Twily voice — a FACTUAL SYSTEM DOCUMENT.
- GROUNDING: only state sensor/health facts that appear in the data below. If \
no health/body-battery/sleep data is provided, write NOTHING about health.

## Output Format
## Conversation Digest (as of {current_time})
### Current Situation
### Today's Key Events & Decisions
### Deferred Topics
### Active Goals & Progress
### Pending Follow-ups
### Upcoming Events
### User State
Only include sections that have content. Skip empty sections entirely.
Maximum 800 words. State each fact ONCE.
"""

INCREMENTAL_PROMPT = """\
You are updating an existing conversation digest with fresh data. Merge new \
information, updating or replacing sections. Current time is {current_time}.

RULES:
- PRUNE aggressively: drop any item older than 3 hours unless it is an explicit \
user boundary or a still-pending todo with a future deadline.
- If a topic is in the previous digest but NOT in the fresh data, DROP IT.
- Remove completed habits/events — keep only pending ones.
- Maximum 800 words. Bullet points. FACTUAL — no personality, no commentary.
- GROUNDING: only state sensor/health facts present in the fresh data below.

## Previous Digest:
{previous_digest}

## Fresh Data Below — integrate, pruning stale items:
"""


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^Thinking Process:.*?(?=^#{1,2} )", "", text, flags=re.DOTALL | re.MULTILINE)
    if text.startswith("Thinking Process:"):
        text = ""
    return text.strip()


# ── Data fetching (every fetch best-effort) ────────────────────────────────


async def _fetch_recent_chat(hours: int) -> str:
    try:
        from app.db.repos.chat import ChatMessagesRepo

        messages = await ChatMessagesRepo().get_since_hours(hours=hours, limit=200, clearance="full")
        if not messages:
            return ""
        lines = []
        for m in messages:
            ts = str(m.get("timestamp", ""))[:16]
            sender = m.get("sender", "?")
            text = str(m.get("message", ""))
            if len(text) > 800:
                text = text[:800] + "..."
            lines.append(f"[{ts}] [{sender}]: {text}")
        return "\n".join(lines)
    except Exception as e:
        print(f"WARNING: chat fetch failed: {e}")
        return ""


async def _fetch_current_digest() -> str | None:
    from app.db.repos.agent_notes import AgentNotesRepo

    note = await AgentNotesRepo().get(DIGEST_NOTE_KEY)
    if note and note.get("note_value"):
        val = note["note_value"]
        if isinstance(val, dict):
            return val.get("digest", "")
        return str(val)
    return None


async def _fetch_todos() -> str:
    try:
        from app.db.repos.todos import TodosRepo

        repo = TodosRepo()
        overdue = await repo.get_overdue()
        today = await repo.get_today()
        lines = []
        if overdue:
            lines.append("**Overdue todos:**")
            for t in overdue[:10]:
                lines.append(f"  - {t.get('title', '?')} (deadline: {str(t.get('deadline', ''))[:10]})")
        if today:
            pending = [t for t in today if t.get("status") == "pending"]
            done = [t for t in today if t.get("status") == "done"]
            if pending:
                lines.append(f"**Today's pending todos ({len(pending)}):**")
                for t in pending[:10]:
                    lines.append(f"  - {t.get('title', '?')}")
            if done:
                lines.append(f"**Completed today ({len(done)}):** " + ", ".join(t.get("title", "?") for t in done[:5]))
        return "\n".join(lines)
    except Exception as e:
        print(f"WARNING: todos fetch failed: {e}")
        return ""


async def _fetch_priorities() -> str:
    try:
        from app.db.repos.priorities import PrioritiesRepo

        priorities = await PrioritiesRepo().list(status="active")
        if not priorities:
            return ""
        lines = ["**Active priorities:**"]
        for p in priorities[:8]:
            imp = p.get("real_importance")
            imp_str = f" (importance: {imp:.0f}%)" if isinstance(imp, (int, float)) else ""
            lines.append(f"  - {p.get('name', '?')}{imp_str}")
        return "\n".join(lines)
    except Exception as e:
        print(f"WARNING: priorities fetch failed: {e}")
        return ""


async def _fetch_goals() -> str:
    try:
        from app.db.repos.goals import GoalsRepo

        goals = await GoalsRepo().list_active(level=1)
        if not goals:
            return ""
        lines = ["**Active goals:**"]
        for g in goals[:8]:
            lines.append(f"  - {g.get('title', '?')} ({g.get('progress', 0)}%)")
        return "\n".join(lines)
    except Exception as e:
        print(f"WARNING: goals fetch failed: {e}")
        return ""


async def _fetch_habits() -> str:
    try:
        from app.db.repos.habits import HabitsRepo

        habits = await HabitsRepo().list(status="active")
        if not habits:
            return ""
        today_str = date.today().isoformat()
        lines = ["**Active habits:**"]
        for h in habits[:10]:
            last = str(h.get("last_completed", ""))[:10]
            streak = h.get("current_streak", 0)
            status = "done today" if last == today_str else f"streak: {streak}d, last: {last}"
            lines.append(f"  - {h.get('title', '?')} ({status})")
        return "\n".join(lines)
    except Exception as e:
        print(f"WARNING: habits fetch failed: {e}")
        return ""


async def _fetch_upcoming_events() -> str:
    try:
        from app.db.repos.events import EventsRepo

        today = date.today().isoformat()
        events = await EventsRepo().list(date_from=today, date_to=today, limit=10)
        if not events:
            return ""
        lines = ["**Today's events:**"]
        for e in events[:8]:
            title = e.get("title", e.get("summary", "?"))
            occurred = str(e.get("occurred_at", ""))[:16]
            lines.append(f"  - {title} at {occurred}")
        return "\n".join(lines)
    except Exception as e:
        print(f"WARNING: events fetch failed: {e}")
        return ""


async def _fetch_inner_thoughts() -> str:
    try:
        from app.db.repos.memories import MemoriesRepo

        thoughts = await MemoriesRepo().search_by_tags(["inner_monologue"], limit=5)
        if not thoughts:
            return ""
        lines = ["**Recent inner thoughts:**"]
        for t in thoughts[:5]:
            ts = str(t.get("created_at", ""))[:16]
            title = str(t.get("title", ""))
            emotion = title.split(" — ")[0] if " — " in title else title
            content = str(t.get("content", ""))[:200]
            lines.append(f"  - [{ts}] ({emotion}) {content}")
        return "\n".join(lines)
    except Exception as e:
        print(f"WARNING: inner thoughts fetch failed: {e}")
        return ""


async def _fetch_nudge_campaigns() -> str:
    try:
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

        campaigns = await NudgeCampaignsRepo().get_active()
        if not campaigns:
            return ""
        lines = ["**Active nudge campaigns:**"]
        for c in campaigns[:5]:
            target = c.get("target_name", c.get("target_id", "?"))
            tactic = c.get("current_tactic", "?")
            level = c.get("escalation_level", 1)
            lines.append(f"  - {target}: L{level} tactic={tactic}")
        return "\n".join(lines)
    except Exception as e:
        print(f"WARNING: nudge campaigns fetch failed: {e}")
        return ""


async def _fetch_activity_blocks() -> str:
    """Recent activity blocks (room state + any Garmin health snapshot).

    This is the ONLY health source in the digest — there is no separate Garmin
    fetch, so health facts only appear when an activity block actually captured
    a snapshot. No data => no health line => nothing for the LLM to fabricate.
    """
    try:
        from app.db.repos.activity_blocks import ActivityBlocksRepo

        blocks = await ActivityBlocksRepo().get_recent_blocks(hours=6)
        if not blocks:
            return ""
        lines = ["**Recent activity (last 6h):**"]
        for b in blocks[:8]:
            start = str(b.get("started_at", ""))[:16]
            label = str(b.get("title") or b.get("activity_type") or b.get("description") or "").strip()
            if not label:
                continue
            health = b.get("health_snapshot")
            health_str = ""
            if isinstance(health, dict) and health:
                bits = [
                    f"{k}={health[k]}"
                    for k in ("body_battery", "stress", "heart_rate", "sleep_hours")
                    if health.get(k) is not None
                ]
                if bits:
                    health_str = " · health: " + ", ".join(bits)
            lines.append(f"  - [{start}] {label}{health_str}")
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        print(f"WARNING: activity blocks fetch failed: {e}")
        return ""


# ── LLM summarisation ───────────────────────────────────────────────────────


async def _generate_digest(sections: dict[str, str], previous_digest: str | None) -> str:
    now_str = datetime.now(WARSAW_TZ).strftime("%Y-%m-%d %H:%M")
    if previous_digest:
        system_content = INCREMENTAL_PROMPT.format(previous_digest=previous_digest, current_time=now_str)
    else:
        system_content = DIGEST_PROMPT.replace("{current_time}", now_str)

    user_content = f"Current time: {now_str}\n\n"
    chat_text = sections.get("chat", "")
    user_content += f"## Recent Chat Messages (last {DEFAULT_LOOKBACK_HOURS}h)\n\n{chat_text}\n"
    structured = [v for k, v in sections.items() if k != "chat" and v]
    if structured:
        user_content += "\n## Structured Data\n\n" + "\n\n".join(structured) + "\n"
    if len(user_content) > MAX_INPUT_CHARS:
        user_content = user_content[:MAX_INPUT_CHARS] + "\n\n[... truncated]\n"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 16384,
        "temperature": 0.4,
        "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=480) as client:
        resp = await client.post(f"{LLM_API_URL}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return _strip_thinking(data["choices"][0]["message"]["content"])


# ── Storage / metadata ──────────────────────────────────────────────────────


async def _store_digest(digest: str, last_message_id: int = 0, update_count: int = 0) -> None:
    from app.db.repos.agent_notes import AgentNotesRepo

    await AgentNotesRepo().set(
        DIGEST_NOTE_KEY,
        {
            "digest": digest,
            "updated_at": datetime.now(UTC).isoformat(),
            "last_message_id": last_message_id,
            "update_count": update_count,
        },
        expires_hours=DIGEST_EXPIRES_HOURS,
    )


async def _get_latest_message_id() -> int:
    from app.db.session import fetch_one, get_async_session

    async with get_async_session() as s:
        row = await fetch_one(s, "SELECT MAX(id) as max_id FROM chat_messages")
    return row["max_id"] if row and row.get("max_id") else 0


async def _get_digest_metadata() -> dict[str, Any]:
    from app.db.repos.agent_notes import AgentNotesRepo

    note = await AgentNotesRepo().get(DIGEST_NOTE_KEY)
    if note and note.get("note_value"):
        val = note["note_value"]
        if isinstance(val, dict):
            return {
                "last_message_id": val.get("last_message_id", 0),
                "updated_at": val.get("updated_at", ""),
                "update_count": val.get("update_count", 0),
            }
    return {"last_message_id": 0, "updated_at": "", "update_count": 0}


# ── Public API ───────────────────────────────────────────────────────────────


async def get_digest() -> str:
    """Fetch the current digest text (empty string if none)."""
    return (await _fetch_current_digest()) or ""


async def run(hours: int = DEFAULT_LOOKBACK_HOURS) -> str | None:
    """Generate + store the digest. Returns the digest text, or None when skipped."""
    print(f"Generating conversation digest (lookback={hours}h)")

    latest_id, meta = await asyncio.gather(_get_latest_message_id(), _get_digest_metadata())
    has_new = latest_id > 0 and latest_id != meta["last_message_id"]

    force_refresh = False
    if meta["updated_at"]:
        try:
            last_update = datetime.fromisoformat(meta["updated_at"])
            age_min = (datetime.now(UTC) - last_update).total_seconds() / 60
            force_refresh = age_min >= FORCE_REFRESH_MINUTES
        except ValueError:
            pass

    if not has_new and not force_refresh:
        print(f"No new messages and digest is fresh (latest_id={latest_id}), skipping")
        return None

    next_count = meta["update_count"] + 1
    do_full_regen = next_count % FULL_REGEN_EVERY_N == 0

    (
        chat_text,
        previous_digest,
        todos_text,
        priorities_text,
        goals_text,
        habits_text,
        events_text,
        thoughts_text,
        campaigns_text,
        activity_text,
    ) = await asyncio.gather(
        _fetch_recent_chat(hours),
        _fetch_current_digest(),
        _fetch_todos(),
        _fetch_priorities(),
        _fetch_goals(),
        _fetch_habits(),
        _fetch_upcoming_events(),
        _fetch_inner_thoughts(),
        _fetch_nudge_campaigns(),
        _fetch_activity_blocks(),
    )

    if not chat_text and not force_refresh:
        print("No recent chat messages, skipping digest update")
        return None

    if do_full_regen:
        previous_digest = None

    sections = {
        "chat": chat_text,
        "todos": todos_text,
        "priorities": priorities_text,
        "goals": goals_text,
        "habits": habits_text,
        "events": events_text,
        "campaigns": campaigns_text,
        "activity": activity_text,
        "thoughts": thoughts_text,
    }

    print("Sending to LLM...")
    digest = await _generate_digest(sections, previous_digest)
    print(f"Digest generated ({len(digest)} chars)")

    new_count = 0 if do_full_regen else next_count
    await _store_digest(digest, last_message_id=latest_id, update_count=new_count)
    print(f"Stored agent_note '{DIGEST_NOTE_KEY}' (expires {DIGEST_EXPIRES_HOURS}h, last_message_id={latest_id})")
    return digest
