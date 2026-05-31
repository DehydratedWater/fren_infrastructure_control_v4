"""Periodic Checker — lightweight 5-minute check for intervention triggers.

Faithful v4 port of v3 ``fren/tools/system/periodic_checker.py``. The v3 module
was an ``open_agent_compiler.runtime.ScriptTool`` invoked once per 5-minute tick
by the scheduler (``script:scripts/periodic_checker.py``). v4 does not depend on
open_agent_compiler, so the checker is ported as a plain async class
(``PeriodicCheckerTool``) that preserves the trigger-detection logic, the global
blockers, the per-category cooldowns, and the accumulation/priority ordering
EXACTLY. The Pydantic Input/Output contract is kept so callers (the scheduler
tick, a /periodic command, or a thin CLI) see the same shape as v3.

``run()`` / ``main()`` provide the one-shot entrypoint that the v3 ScriptTool
``.run()`` classmethod exposed.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

WAKING_HOURS_START = 8
WAKING_HOURS_END = 22
GLOBAL_MIN_COOLDOWN_MINUTES = 5  # Minimum gap between ANY messages
USER_IDLE_THRESHOLD_MINUTES = 60
TASK_NUDGE_IDLE_MINUTES = 120  # Only nudge about pending tasks after 2h idle
CALENDAR_LOOKAHEAD_MINUTES = 30  # Alert for events starting within this window
CHRONIC_OVERDUE_HOURS = 24  # After this, suggest rescheduling instead of nagging

# Per-category cooldowns (minutes)
_CATEGORY_COOLDOWNS: dict[str, int] = {
    "upcoming_calendar_event": 15,
    "overdue_todos": 60,
    "overdue_reschedule_suggestion": 120,
    "idle_during_block": 60,
    "pending_tasks": 120,
    "untracked_conversation_tasks": 120,
    "unfulfilled_actions": 180,
    "incomplete_daily_routines": 60,
}

# Patterns that indicate the user intends to do a task
_TASK_INTENT_PATTERNS = [
    re.compile(r"\b(?:I need to|I should|I have to|I must|I gotta)\b", re.IGNORECASE),
    re.compile(r"\b(?:remind me to|don't let me forget)\b", re.IGNORECASE),
    re.compile(r"\b(?:I want to start|I'm planning to|tomorrow I'll|I'll)\b", re.IGNORECASE),
    re.compile(r"\b(?:add to my (?:list|todo|tasks)|put on my todo)\b", re.IGNORECASE),
    re.compile(r"\b(?:I'm going to|gonna|need to get)\b", re.IGNORECASE),
]


class Input(BaseModel):
    command: str = Field(default="check", description="check|get-state|dry-run")
    force: bool = Field(default=False, description="Skip waking-hours and work-hours checks")


class Output(BaseModel):
    success: bool = True
    trigger: bool = False
    reason: str = ""
    details: dict = Field(default_factory=dict)
    triggers: list[dict] = Field(default_factory=list)
    state: dict = Field(default_factory=dict)
    error: str = ""


class PeriodicCheckerTool:
    """Lightweight periodic check for Twily intervention triggers."""

    name = "periodic_checker"
    description = "Lightweight periodic check for Twily intervention triggers"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        if inp.command == "get-state":
            from app.db.repos.checker import CheckerStateRepo

            state = await CheckerStateRepo().get()
            return Output(success=True, state=state)

        if inp.command in ("check", "dry-run"):
            return await self._run_check(dry_run=(inp.command == "dry-run"), force=inp.force)

        return Output(success=False, error=f"Unknown command: {inp.command}")

    async def _category_on_cooldown(self, category: str, now: datetime) -> bool:
        """Check if a specific trigger category is still on cooldown."""
        from app.db.repos.checker import CheckerStateRepo

        cooldowns = await CheckerStateRepo().get_category_cooldowns()
        last_str = cooldowns.get(category)
        if not last_str:
            return False
        try:
            last_dt = datetime.fromisoformat(last_str)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=UTC)
            elapsed = (now - last_dt).total_seconds() / 60
            return elapsed < _CATEGORY_COOLDOWNS.get(category, 30)
        except (ValueError, TypeError):
            return False

    async def _record_trigger(self, category: str, now: datetime, dry_run: bool) -> None:
        """Record that a trigger fired for cooldown tracking."""
        if dry_run:
            return
        from app.db.repos.checker import CheckerStateRepo

        repo = CheckerStateRepo()
        await repo.set_category_cooldown(category, now)
        # Also update legacy fields for backwards compat
        await repo.update(last_reminder_at=now, last_trigger_reason=category)

    async def _run_check(self, dry_run: bool = False, force: bool = False) -> Output:
        from zoneinfo import ZoneInfo

        from app.settings import get_settings
        from app.db.repos.agent_notes import AgentNotesRepo
        from app.db.repos.chat import ChatMessagesRepo
        from app.db.repos.checker import CheckerStateRepo
        from app.db.repos.strategies import StrategiesRepo
        from app.db.repos.todos import TodosRepo

        now = datetime.now(UTC)
        local_now = now.astimezone(ZoneInfo(get_settings().user_timezone))
        hour = local_now.hour

        # 1. Waking hours check — global blocker (skipped with --force)
        if not force and (hour < WAKING_HOURS_START or hour >= WAKING_HOURS_END):
            return Output(
                success=True,
                trigger=False,
                reason="outside_waking_hours",
                details={"hour": hour, "range": f"{WAKING_HOURS_START}-{WAKING_HOURS_END}"},
            )

        # 1.5. Work hours check — global blocker (skipped with --force)
        if not force:
            try:
                from app.db.repos.user_config import UserConfigRepo

                wh_row = await UserConfigRepo().get("work_hours")
                if wh_row:
                    wh_val = wh_row.get("config_value", "")
                    if isinstance(wh_val, str) and "-" in wh_val:
                        start_h, end_h = int(wh_val.split("-")[0]), int(wh_val.split("-")[1])
                        if start_h <= hour < end_h:
                            return Output(
                                success=True,
                                trigger=False,
                                reason="work_hours",
                                details={"hour": hour, "work_hours": wh_val},
                            )
            except Exception:
                pass

        # 2. User busy/DND check — global blocker
        busy_note = await AgentNotesRepo().get("user_busy")
        if busy_note:
            busy_until = busy_note.get("note_value", {})
            if isinstance(busy_until, dict):
                until_str = busy_until.get("until", "")
                reason = busy_until.get("reason", "busy")
                if until_str:
                    try:
                        until_dt = datetime.fromisoformat(until_str)
                        if hasattr(until_dt, "tzinfo") and until_dt.tzinfo is None:
                            until_dt = until_dt.replace(tzinfo=UTC)
                        if now < until_dt:
                            return Output(
                                success=True,
                                trigger=False,
                                reason="user_busy",
                                details={"until": until_str, "reason": reason},
                            )
                        await AgentNotesRepo().delete("user_busy")
                    except ValueError:
                        pass

        # 3. Recent conversation check — global blocker
        recent_msgs = await ChatMessagesRepo().get_recent(limit=15)
        user_msgs = [m for m in recent_msgs if m.get("sender") == "user"]
        _BUSY_KEYWORDS = ("at work", "busy", "in a meeting", "don't disturb", "later", "not now", "leave me")
        for msg in user_msgs[:5]:
            text = str(msg.get("message", "")).lower()
            if any(kw in text for kw in _BUSY_KEYWORDS):
                msg_ts = msg.get("timestamp_unix", 0)
                if msg_ts:
                    mins_ago = (now.timestamp() - float(msg_ts)) / 60
                    if mins_ago < 180:
                        return Output(
                            success=True,
                            trigger=False,
                            reason="user_recently_busy",
                            details={"message": text[:100], "minutes_ago": round(mins_ago)},
                        )

        # 4. Global minimum cooldown — prevent message spam
        state = await CheckerStateRepo().get()
        last_reminder = state.get("last_reminder_at")
        if last_reminder:
            if isinstance(last_reminder, str):
                try:
                    last_dt = datetime.fromisoformat(last_reminder)
                except ValueError:
                    last_dt = None
            else:
                last_dt = last_reminder
            if last_dt:
                if hasattr(last_dt, "tzinfo") and last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=UTC)
                elapsed = (now - last_dt).total_seconds() / 60
                if elapsed < GLOBAL_MIN_COOLDOWN_MINUTES:
                    return Output(
                        success=True,
                        trigger=False,
                        reason="global_cooldown",
                        details={"minutes_since_last": round(elapsed), "cooldown": GLOBAL_MIN_COOLDOWN_MINUTES},
                    )

        # ── Accumulate all triggers (per-category cooldowns) ──
        triggers: list[dict] = []

        # 5. Calendar upcoming event check
        if not await self._category_on_cooldown("upcoming_calendar_event", now):
            try:
                from app.services import calendar_client

                time_min = now.isoformat()
                time_max = (now + timedelta(minutes=CALENDAR_LOOKAHEAD_MINUTES)).isoformat()
                events = await asyncio.to_thread(calendar_client.list_events, time_min=time_min, time_max=time_max)
                upcoming = []
                for ev in events:
                    start_info = ev.get("start", {})
                    dt_str = start_info.get("dateTime", "")
                    if not dt_str:
                        continue
                    try:
                        from dateutil.parser import isoparse

                        ev_start = isoparse(dt_str)
                        minutes_until = (ev_start - now).total_seconds() / 60
                        if 0 <= minutes_until <= CALENDAR_LOOKAHEAD_MINUTES:
                            upcoming.append(
                                {"summary": ev.get("summary", "Untitled"), "minutes_until": round(minutes_until)}
                            )
                    except (ValueError, TypeError):
                        continue
                if upcoming:
                    triggers.append(
                        {
                            "reason": "upcoming_calendar_event",
                            "details": {"events": upcoming, "count": len(upcoming)},
                        }
                    )
            except Exception:
                pass

        # 6. Overdue todos check
        overdue = await TodosRepo().get_overdue()
        if overdue:
            chronic = []
            fresh = []
            for t in overdue:
                deadline_str = t.get("deadline", "")
                overdue_hours = None
                if deadline_str:
                    try:
                        from dateutil.parser import isoparse

                        deadline_dt = isoparse(str(deadline_str))
                        if deadline_dt.tzinfo is None:
                            deadline_dt = deadline_dt.replace(tzinfo=UTC)
                        overdue_hours = (now - deadline_dt).total_seconds() / 3600
                    except (ValueError, TypeError):
                        pass
                if overdue_hours is not None and overdue_hours >= CHRONIC_OVERDUE_HOURS:
                    chronic.append({**t, "overdue_hours": round(overdue_hours)})
                else:
                    fresh.append(t)

            if chronic and not await self._category_on_cooldown("overdue_reschedule_suggestion", now):
                titles = [f"{t.get('title', '')[:50]} ({t['overdue_hours']}h)" for t in chronic[:3]]
                triggers.append(
                    {
                        "reason": "overdue_reschedule_suggestion",
                        "details": {"count": len(chronic), "titles": titles},
                    }
                )

            if fresh and not await self._category_on_cooldown("overdue_todos", now):
                titles = [t.get("title", "")[:50] for t in fresh[:3]]
                triggers.append(
                    {
                        "reason": "overdue_todos",
                        "details": {"count": len(fresh), "titles": titles},
                    }
                )

        # 7. Strategy time block check
        if not await self._category_on_cooldown("idle_during_block", now):
            today_strategy = await StrategiesRepo().get_today()
            if today_strategy:
                time_blocks = today_strategy.get("time_blocks") or []
                current_time = now.strftime("%H:%M")
                active_block = None
                for block in time_blocks:
                    if not isinstance(block, dict):
                        continue
                    start = block.get("start", "")
                    end = block.get("end", "")
                    if start <= current_time <= end:
                        active_block = block
                        break

                if active_block:
                    if user_msgs:
                        last_ts = user_msgs[0].get("timestamp_unix", 0)
                        idle_minutes = (now.timestamp() - float(last_ts)) / 60 if last_ts else 999
                    else:
                        idle_minutes = 999

                    if idle_minutes >= USER_IDLE_THRESHOLD_MINUTES:
                        triggers.append(
                            {
                                "reason": "idle_during_block",
                                "details": {
                                    "block": active_block,
                                    "idle_minutes": round(idle_minutes),
                                    "threshold": USER_IDLE_THRESHOLD_MINUTES,
                                },
                            }
                        )

        # 8. Pending tasks nudge
        if not await self._category_on_cooldown("pending_tasks", now):
            today_tasks = await TodosRepo().get_today()
            pending = [t for t in today_tasks if t.get("status") == "pending"]
            if pending:
                if user_msgs:
                    last_ts = user_msgs[0].get("timestamp_unix", 0)
                    idle_minutes = (now.timestamp() - float(last_ts)) / 60 if last_ts else 999
                else:
                    idle_minutes = 999

                if idle_minutes >= TASK_NUDGE_IDLE_MINUTES:
                    titles = [t.get("title", "")[:60] for t in pending[:5]]
                    priorities = [t.get("priority", "medium") for t in pending[:5]]
                    triggers.append(
                        {
                            "reason": "pending_tasks",
                            "details": {
                                "count": len(pending),
                                "titles": titles,
                                "priorities": priorities,
                                "idle_minutes": round(idle_minutes),
                            },
                        }
                    )

        # 9. Unfulfilled actions — scan recent Twily messages for promises/actions she claimed to do
        if not await self._category_on_cooldown("unfulfilled_actions", now):
            unfulfilled = await self._check_unfulfilled_actions(recent_msgs, now)
            if unfulfilled:
                triggers.append(
                    {
                        "reason": "unfulfilled_actions",
                        "details": {"actions": unfulfilled, "count": len(unfulfilled)},
                    }
                )

        # 10. Untracked conversation tasks — scan chat for task-like language not yet in todos
        if not await self._category_on_cooldown("untracked_conversation_tasks", now):
            untracked = await self._check_untracked_tasks(user_msgs, now)
            if untracked:
                triggers.append(
                    {
                        "reason": "untracked_conversation_tasks",
                        "details": {"tasks": untracked, "count": len(untracked)},
                    }
                )

        # 11. Incomplete daily routines
        if not await self._category_on_cooldown("incomplete_daily_routines", now):
            try:
                from app.db.repos.daily_routines import DailyRoutinesRepo

                pending_routines = await DailyRoutinesRepo().get_due_today()
                if pending_routines:
                    titles = [r.get("title", "")[:60] for r in pending_routines[:5]]
                    triggers.append(
                        {
                            "reason": "incomplete_daily_routines",
                            "details": {"count": len(pending_routines), "titles": titles},
                        }
                    )
            except Exception:
                pass

        # ── Return results ──
        if not triggers:
            return Output(
                success=True,
                trigger=False,
                reason="no_triggers",
                triggers=[],
                details={
                    "checks_passed": [
                        "waking_hours",
                        "calendar",
                        "overdue",
                        "time_blocks",
                        "pending_tasks",
                        "untracked_tasks",
                    ]
                },
            )

        # Record cooldowns for all fired triggers
        for t in triggers:
            await self._record_trigger(t["reason"], now, dry_run)

        # Primary trigger = first in list (priority order from checks above)
        primary = triggers[0]
        return Output(
            success=True,
            trigger=True,
            reason=primary["reason"],
            details=primary["details"],
            triggers=triggers,
        )

    async def _check_untracked_tasks(self, user_msgs: list[dict], now: datetime) -> list[dict]:
        """Scan recent user messages for task-like language not matched by existing todos."""
        from app.db.repos.todos import TodosRepo

        # Only check messages from last 2 hours
        cutoff = now.timestamp() - 7200
        recent_task_msgs = []
        for msg in user_msgs:
            msg_ts = msg.get("timestamp_unix", 0)
            if msg_ts and float(msg_ts) >= cutoff:
                text = str(msg.get("message", ""))
                for pattern in _TASK_INTENT_PATTERNS:
                    match = pattern.search(text)
                    if match:
                        # Extract the task phrase (rest of sentence after the pattern)
                        start = match.end()
                        remainder = text[start:].strip()
                        # Take first sentence/clause
                        task_phrase = re.split(r"[.!?\n]", remainder)[0].strip()
                        if len(task_phrase) > 3:
                            recent_task_msgs.append(task_phrase[:100])
                        break

        if not recent_task_msgs:
            return []

        # Get recent todos to cross-reference
        today_tasks = await TodosRepo().get_today()
        all_pending = await TodosRepo().list(status="pending")
        existing_titles = {t.get("title", "").lower() for t in (today_tasks + all_pending)}

        # Simple fuzzy match: check if any extracted task phrase is already a todo
        untracked = []
        for phrase in recent_task_msgs:
            phrase_lower = phrase.lower()
            matched = any(phrase_lower in title or title in phrase_lower for title in existing_titles if title)
            if not matched:
                untracked.append({"phrase": phrase})

        return untracked[:3]  # Cap at 3 to avoid noise

    async def _check_unfulfilled_actions(self, all_msgs: list[dict], now: datetime) -> list[dict]:
        """Scan recent Twily messages for promises/actions she claimed to do, then check if they happened."""
        # Patterns indicating Twily promised to do something
        _ACTION_PATTERNS = [
            re.compile(r"\b(?:let me|I'll|I will|on it|one sec|hold on|working on)\b", re.IGNORECASE),
            re.compile(
                r"\b(?:sending|searching|looking|checking|adding|creating)\b.*(?:now|for you|right away)", re.IGNORECASE
            ),
            re.compile(r"\b(?:I'm going to|gonna|about to)\b", re.IGNORECASE),
        ]

        # Only check messages from last 2 hours
        cutoff = now.timestamp() - 7200
        twily_promises: list[dict] = []

        for msg in all_msgs:
            if msg.get("sender") != "assistant":
                continue
            msg_ts = msg.get("timestamp_unix", 0)
            if not msg_ts or float(msg_ts) < cutoff:
                continue
            text = str(msg.get("message", ""))
            for pattern in _ACTION_PATTERNS:
                if pattern.search(text):
                    twily_promises.append(
                        {
                            "message": text[:200],
                            "timestamp": float(msg_ts),
                        }
                    )
                    break

        if not twily_promises:
            return []

        # Check if user got a follow-up from Twily after each promise
        # A promise is "fulfilled" if Twily sent another message within 10 minutes after it
        unfulfilled = []
        for promise in twily_promises:
            promise_ts = promise["timestamp"]
            # Look for Twily messages sent 1-10 minutes after the promise
            followup_found = False
            for msg in all_msgs:
                if msg.get("sender") != "assistant":
                    continue
                msg_ts = float(msg.get("timestamp_unix", 0))
                delta = msg_ts - promise_ts
                if 60 < delta < 600:  # 1-10 minutes after
                    followup_found = True
                    break
            if not followup_found and (now.timestamp() - promise_ts) > 600:
                unfulfilled.append(promise)

        return unfulfilled[:3]

    @classmethod
    def run(cls) -> None:
        """One-shot CLI entrypoint (parity with v3 ScriptTool.run()).

        v3's ScriptTool.run() parsed argv into the Input model. Here we keep a
        minimal arg surface (--command / --force) sufficient for the scheduler's
        ``script:`` invocation and manual runs; output is printed as JSON.
        """
        import argparse
        import json as _json

        parser = argparse.ArgumentParser(prog=cls.name, description=cls.description)
        parser.add_argument("--command", default="check", help="check|get-state|dry-run")
        parser.add_argument("--force", action="store_true")
        args = parser.parse_args()
        out = cls().execute(Input(command=args.command, force=args.force))
        print(_json.dumps(out.model_dump(), default=str))


def main() -> None:
    PeriodicCheckerTool.run()


if __name__ == "__main__":
    main()
