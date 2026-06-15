"""Proactive autonomy heartbeat — the agent's periodic wake-up loop.

Replaces the every-5-min `periodic_checker` reminder agent (and, by mode,
winddown/evening/night) with ONE in-process thinking-on triage call over rich,
deterministically-assembled evidence — then deterministic routing. See
docs/plans/proactive-autonomy-heartbeat.md.

Why this shape (not a heavy opencode agent, not LangGraph):
  * The old design spun up a full opencode+qwen agent every tick just to call a
    deterministic tool that said "nothing to do" ~100% of the time, AND it never
    read the agent's own internal work (pending_thoughts forged every 30 min died
    unseen). This engine assembles the rich picture itself and reasons once.
  * The flow is linear — assemble → reason (thinking ON) → decide → route — and
    the 256k context means we pre-load everything, so a request/response call
    beats a tool-loop (more testable, no tool-flail). Routing side-effects are
    plain Python on the structured decision.

Autonomy (the more the better) with ONE hard guardrail: it may PROPOSE tasks but
NEVER create todos/goals/calendar entries unless the user explicitly asked — the
heartbeat writes only its OWN state (consumes pending_thoughts, records its run).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Modes: time-windowed personas/policies of the SAME engine (unifies
# periodic/winddown/evening/night). `escalate_agent` is the heavy specialist this
# mode falls through to for nuanced/action-heavy cases (preserves their behaviour
# while the cheap triage gates the wake-up).
MODES: dict[str, dict[str, str]] = {
    "day": {
        "escalate_agent": "goals/periodic_checker",
        "policy": (
            "Daytime (08-21). Reasons to wake: a genuine reminder/nudge; the user "
            "is procrastinating on something they said mattered; an agreement is "
            "slipping; a dropped thread deserves follow-up; an old/forgotten task "
            "is worth resurfacing; you connected two events into an insight; you "
            "(the agent) forged a genuinely interesting thought worth sharing; or "
            "after a long quiet gap, a warm hello / something about yourself to "
            "build the relationship. Default to SKIP on a busy/quiet tick."
        ),
    },
    "evening": {
        "escalate_agent": "goals/evening_focus",
        "policy": (
            "Evening (21-24). Help the user close the day / set up tomorrow; "
            "reflect on what got done. Gentle. Skip if nothing is worth raising."
        ),
    },
    "winddown": {
        "escalate_agent": "goals/winddown",
        "policy": (
            "Late night (00-05). Help the user wind down and sleep. The later it "
            "is the more urgent (set `urgency` 0-5 accordingly). If real action is "
            "needed (camera/desk check, turning lights off, a sleepy selfie, firm "
            "escalation) choose decision=escalate so the winddown specialist acts. "
            "If only a gentle nudge is needed choose decision=message."
        ),
    },
    "night": {
        "escalate_agent": "goals/periodic_checker",
        "policy": (
            "Deep night. Mostly reflect/prepare silently; deliver to the user only "
            "if something is genuinely urgent. Default SKIP."
        ),
    },
}

_BASE_PROMPT = """\
You are Twily's proactive autonomy heartbeat — the agent's periodic wake-up.

You run on a timer. Below is EVERYTHING you currently know: the recent
conversation, the user's open commitments/agreements, deterministic
reminder-triggers (calendar/todos/routines), your OWN internal work (pending
thoughts you forged, your strategies and inner monologue), procrastination /
dropped-thread signals, and how long it's been since you last spoke.

Decide, ONCE, whether there is a GENUINE reason to reach the user right now — and
if so, what and how. You have broad autonomy: remind, nudge, surface a forged
insight, follow up on a slipping agreement, resurface a worthwhile old task,
share a connection you noticed, kick off your own research, or — after a long
quiet gap — just say hello / share something about yourself to build the
relationship. You may also choose to do nothing.

## Decision
Return the structured decision object:
- decision="skip"     → say/do nothing this tick. This is the RIGHT answer most
  ticks. Pick it when nothing is genuinely worth the user's attention, the user
  is busy, or you'd repeat something already said recently.
- decision="message"  → send the user a message NOW. Put the FINAL message text,
  in Twily's warm voice, in `draft`. Set `category`.
- decision="escalate" → the situation is nuanced/action-heavy; hand off to the
  full specialist agent for deep reasoning + delivery. Put context in `reasoning`.
- decision="act"      → kick off autonomous work: set `route_agent` to the
  specialist to run (e.g. a research agent) and `reasoning` to its instruction.

## HARD RULES
- You may PROPOSE tasks (category="propose_task", as a suggestion in `draft`), but
  you must NEVER create/edit/delete todos, goals, or calendar entries, and never
  instruct another agent to do so unless the user EXPLICITLY asked. Suggest only.
- Anti-spam: a hello / fun remark / share-about-yourself is welcome ONLY because a
  lot of time has passed — not every tick. If you spoke recently and nothing is
  new, SKIP.
- Grounding: only reference facts present in the evidence below. Never invent
  health figures, events, or todos that aren't shown.
- Anti-repetition: do not raise something you already raised in the recent
  conversation, or that the user deferred.
- If you surface one of your pending thoughts, list its id(s) in `uses_thought_ids`.
"""

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["skip", "message", "escalate", "act"]},
        "category": {
            "type": "string",
            "enum": [
                "reminder", "nudge", "share_insight", "agreement_followup",
                "procrastination", "dropped_thread", "stale_task_review",
                "event_connection", "propose_task", "self_research", "self_plan",
                "plan_execution", "social_checkin", "share_about_self",
                "fun_remark", "winddown", "none", "other",
            ],
        },
        "urgency": {"type": "integer", "description": "0 (calm) - 5 (urgent), for winddown escalation"},
        "reasoning": {"type": "string", "description": "why — brief; shown in the dashboard + audit"},
        "draft": {"type": "string", "description": "the message to send, Twily voice, when decision=message"},
        "route_agent": {"type": "string", "description": "specialist agent id when decision=act"},
        "route_action": {"type": "string", "description": "named physical action when decision=act (e.g. lights_off)"},
        "uses_thought_ids": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"},
    },
    "required": ["decision", "category", "reasoning"],
}

# Agents the heartbeat may autonomously spawn (decision=act). A safelist: research
# / self-directed work, never anything that mutates the user's data without ask.
_ACT_AGENT_SAFELIST = {
    "research/techtree_orchestrator",
    "research/deep_researcher",
    "persona/topic_synthesizer",
    "persona/thought_forger",
    "goals/strategy_tracker",
}

_MAX_TOKENS = 6000
_CLIENT_TIMEOUT_S = 200.0


# ── evidence assembly (deterministic — this is where the missing signals enter) ──

async def _pending_thoughts(limit: int = 6) -> list[dict]:
    """Top unconsumed forged thoughts by motivation — the agent's own internal
    work that today never reaches the user proactively (the biggest gap)."""
    from app.db.session import fetch_all, get_async_session

    sql = """
        SELECT id, content, kind, motivation_score
        FROM pending_thoughts
        WHERE consumed_at IS NULL
        ORDER BY motivation_score DESC NULLS LAST, created_at DESC
        LIMIT :limit
    """
    try:
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {"limit": limit})
            return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001 — evidence is best-effort
        logger.debug("heartbeat: pending_thoughts fetch failed", exc_info=True)
        return []


async def _mark_thoughts_consumed(ids: list[int]) -> None:
    if not ids:
        return
    from app.db.session import execute_sql, get_async_session

    sql = """
        UPDATE pending_thoughts
        SET consumed_at = :now, consumed_by = 'heartbeat'
        WHERE id = ANY(:ids) AND consumed_at IS NULL
    """
    try:
        async with get_async_session() as s:
            await execute_sql(s, sql, {"now": datetime.now(UTC), "ids": list(ids)})
    except Exception:  # noqa: BLE001
        logger.debug("heartbeat: mark-consumed failed", exc_info=True)


async def _deterministic_triggers() -> dict:
    """The existing periodic_checker tool output, used as EVIDENCE (not the gate)."""
    try:
        from app.tools.system.periodic_checker import PeriodicCheckerTool

        out = await PeriodicCheckerTool()._run_check(dry_run=True, force=True)
        return {
            "trigger": out.trigger,
            "reason": out.reason,
            "triggers": out.triggers,
        }
    except Exception:  # noqa: BLE001
        logger.debug("heartbeat: deterministic triggers failed", exc_info=True)
        return {}


async def _open_commitments(limit: int = 8) -> list[dict]:
    """Open agreements/commitments (extracted by event_extractor; deduped there).
    Stored as agent_notes of type 'commitment' until a dedicated table lands."""
    try:
        from app.db.repos.agent_notes import AgentNotesRepo

        note = await AgentNotesRepo().get("open_commitments")
        if note and isinstance(note.get("note_value"), (list, dict)):
            val = note["note_value"]
            items = val if isinstance(val, list) else val.get("items", [])
            return list(items)[:limit]
    except Exception:  # noqa: BLE001
        pass
    return []


async def _strategies_and_monologue() -> dict:
    out: dict[str, Any] = {}
    try:
        from app.db.repos.strategies import StrategiesRepo

        today = await StrategiesRepo().get_today()
        if today:
            out["today_strategy"] = {
                "focus": today.get("focus") or today.get("summary"),
                "time_blocks": today.get("time_blocks"),
            }
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.db.repos.agent_notes import AgentNotesRepo

        for key in ("inner_monologue", "conversation_digest"):
            n = await AgentNotesRepo().get(key)
            if n and n.get("note_value"):
                v = n["note_value"]
                out[key] = (v.get("digest") or v.get("text") if isinstance(v, dict) else str(v))
    except Exception:  # noqa: BLE001
        pass
    return out


async def _recent_chat(limit: int = 16) -> tuple[list[dict], dict]:
    """Recent chat as chronological messages + (last_user_age_min, last_bot_age_min)."""
    msgs: list[dict] = []
    ages = {"last_user_age_min": None, "last_bot_age_min": None}
    try:
        from app.db.repos.chat import ChatMessagesRepo

        rows = await ChatMessagesRepo().get_recent(limit=limit)
        now = datetime.now(UTC).timestamp()
        for m in rows:  # most-recent-first
            ts = float(m.get("timestamp_unix") or 0)
            age = round((now - ts) / 60) if ts else None
            if m.get("sender") == "twily":
                if ages["last_bot_age_min"] is None:
                    ages["last_bot_age_min"] = age
            elif ages["last_user_age_min"] is None:
                ages["last_user_age_min"] = age
        for m in reversed(rows):
            text = str(m.get("message") or "")
            if text:
                role = "assistant" if m.get("sender") == "twily" else "user"
                msgs.append({"role": role, "content": text})
    except Exception:  # noqa: BLE001
        pass
    return msgs, ages


def _evidence_block(mode: str, *, triggers: dict, thoughts: list[dict],
                    commitments: list[dict], strat: dict, ages: dict) -> str:
    """The single user-turn evidence dump the triage reasons over."""
    from zoneinfo import ZoneInfo

    from app.settings import get_settings

    local = datetime.now(ZoneInfo(get_settings().user_timezone))
    parts = [
        f"## NOW\nmode={mode} · local_time={local:%Y-%m-%d %H:%M (%a)} · "
        f"last_user_msg={ages.get('last_user_age_min')} min ago · "
        f"last_twily_msg={ages.get('last_bot_age_min')} min ago",
    ]
    if thoughts:
        parts.append(
            "## YOUR PENDING THOUGHTS (forged by you; share the genuinely good ones)\n"
            + "\n".join(
                f"- [id {t['id']} · motivation {t.get('motivation_score')}] {str(t.get('content'))[:400]}"
                for t in thoughts
            )
        )
    if commitments:
        parts.append(
            "## OPEN COMMITMENTS / AGREEMENTS\n"
            + "\n".join(f"- {str(c.get('text') or c)[:200]}" for c in commitments)
        )
    if triggers:
        parts.append("## DETERMINISTIC TRIGGERS (calendar/todos/routines)\n"
                     + json.dumps(triggers, default=str)[:2000])
    if strat:
        parts.append("## YOUR STRATEGIES / INNER STATE\n" + json.dumps(strat, default=str)[:2000])
    parts.append(
        "## DECIDE\nReturn the decision object. Most ticks should be skip. "
        "If you message, write the full text in `draft` in Twily's voice."
    )
    return "\n\n".join(parts)


def _build_spec(mode: str):
    from src.interactive.runner import OpenAICompatClient
    from src.interactive.spec import InteractiveAgentSpec

    from app.agents.config import QWEN35_27B_LIVE

    policy = MODES.get(mode, MODES["day"])["policy"]
    spec = InteractiveAgentSpec(
        agent_id="persona/heartbeat",
        model=QWEN35_27B_LIVE,
        system_prompt=f"{_BASE_PROMPT}\n\n## MODE\n{policy}",
        tools=(),
        output_schema=DECISION_SCHEMA,
    )
    client = OpenAICompatClient.from_spec(spec)
    # Generous room: thinking-on reasoning + the structured answer must not truncate.
    client.default_params["max_tokens"] = _MAX_TOKENS
    return spec, client


# ── routing (deterministic side-effects on the structured decision) ──

def _deliver_message(draft: str, run_id: str) -> bool:
    """Send the drafted proactive message through the existing delivery pipeline
    (gate/cooldown/dedup/style/persist) via send_message.py."""
    from app.settings import get_settings

    if not draft.strip():
        return False
    try:
        subprocess.run(
            ["python", "scripts/send_message.py", "--message", draft],
            cwd=str(get_settings().agents_dir),
            env={**os.environ, "FREN_RUN_ID": run_id, "FREN_MSG_KIND": "proactive"},
            capture_output=True, text=True, timeout=60, check=False,
        )
        return True
    except Exception:  # noqa: BLE001
        logger.exception("heartbeat: send_message failed")
        return False


def _spawn_detached(agent: str, instruction: str) -> bool:
    """Fire-and-forget an opencode specialist (it delivers its own result)."""
    from app.settings import get_settings

    try:
        subprocess.Popen(
            ["python", "scripts/opencode_manager.py", "--command", "run",
             "--agent", agent, "--prompt", instruction],
            cwd=str(get_settings().agents_dir), env={**os.environ},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:  # noqa: BLE001
        logger.exception("heartbeat: spawn %s failed", agent)
        return False


async def run_heartbeat(mode: str = "day") -> dict:
    """One heartbeat tick. Assemble evidence → ONE thinking-on triage call →
    route. Returns a small result dict (also recorded in the execution ledger)."""
    import asyncio

    from src.interactive import run_interactive

    mode = mode if mode in MODES else "day"
    run_id = f"run_{uuid.uuid4().hex[:16]}"
    try:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        await ExecutionLedgerRepo().ensure_run(
            run_id, interaction_mode="heartbeat", owner=f"persona/heartbeat-{mode}",
        )
    except Exception:  # noqa: BLE001
        pass

    triggers, thoughts, commitments, strat = await asyncio.gather(
        _deterministic_triggers(), _pending_thoughts(), _open_commitments(),
        _strategies_and_monologue(),
    )
    history, ages = await _recent_chat()
    evidence = _evidence_block(
        mode, triggers=triggers, thoughts=thoughts, commitments=commitments,
        strat=strat, ages=ages,
    )

    spec, client = _build_spec(mode)
    client.default_params.setdefault("timeout", _CLIENT_TIMEOUT_S)

    def _go():
        return run_interactive(
            spec, evidence, client=client, history=history, max_tool_rounds=1,
        )

    try:
        result = await asyncio.to_thread(_go)
    except Exception as exc:  # noqa: BLE001
        logger.exception("heartbeat tick failed: %s", exc)
        await _complete(run_id, "failed", False, None, None)
        return {"ok": False, "mode": mode, "run_id": run_id, "error": str(exc)}

    decision = result.structured if isinstance(result.structured, dict) else {}
    kind = str(decision.get("decision") or "skip")
    category = str(decision.get("category") or "none")
    acted = False

    if kind == "message":
        acted = _deliver_message(str(decision.get("draft") or ""), run_id)
        if acted:
            await _mark_thoughts_consumed(
                [int(i) for i in (decision.get("uses_thought_ids") or []) if str(i).isdigit()]
            )
    elif kind == "escalate":
        target = MODES[mode]["escalate_agent"]
        acted = _spawn_detached(target, str(decision.get("reasoning") or "Proactive check — decide and act."))
    elif kind == "act":
        agent = str(decision.get("route_agent") or "")
        if agent in _ACT_AGENT_SAFELIST:
            acted = _spawn_detached(agent, str(decision.get("reasoning") or ""))
        else:
            logger.info("heartbeat: act blocked (agent %r not in safelist)", agent)

    await _complete(run_id, "completed", True, decision, result)
    logger.info("heartbeat[%s]: decision=%s category=%s acted=%s", mode, kind, category, acted)
    return {"ok": True, "mode": mode, "run_id": run_id, "decision": kind,
            "category": category, "acted": acted}


async def _complete(run_id: str, status: str, passed: bool,
                    decision: dict | None, result: Any) -> None:
    try:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        repo = ExecutionLedgerRepo()
        if decision is not None:
            await repo.write_artifact(run_id, "persona_guidance", decision, producer="heartbeat")
        traj = []
        if result is not None:
            traj = [{"kind": "text", "text": (result.output_text or "")[:4000]}]
        await repo.write_artifact(
            run_id, "run_trace",
            {"text": (decision or {}).get("reasoning", ""), "tool_calls": [],
             "tool_call_count": 0, "trajectory": traj, "trajectory_count": len(traj),
             "ok": passed, "error": getattr(result, "error", None)},
            producer="heartbeat",
        )
        await repo.complete_run(run_id, status=status, contract_passed=passed)
    except Exception:  # noqa: BLE001
        logger.debug("heartbeat: ledger complete failed", exc_info=True)
