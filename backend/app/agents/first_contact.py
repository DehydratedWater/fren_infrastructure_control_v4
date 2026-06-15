"""Tier-0 first-contact agent — snappy, in-process, local qwen (LangChain runtime).

v3 had a two-tier flow: a FAST chat agent answered most turns directly and only
SOMETIMES escalated to the heavy orchestrator. v4 had flattened that (every turn
hit a heavy opencode agent, ~80-215s). This restores the fast tier on the
framework's `src/interactive` primitive: the SAME AgentDefinition machinery, but
run in-process via a LangChain/OpenAI-compatible client against the local qwen
(:8082) — no opencode subprocess, no model switching.

Routing spectrum (the FC agent decides per turn):
  - CONVERSATION (banter, quick Q&A): answer directly → emit_guidance →
    persona_prose renders the reply. Fastest.
  - DIRECT TOOLS (cheap CRUD): todo_manager / fetch_context inline, then confirm.
  - HANDOFF (heavy / self-delivering ops — research, RALF, selfie, complex
    goal replanning): `handoff(agent, instruction)` fire-and-forgets the opencode
    specialist/orchestrator (which delivers its own result), and FC acks.

Delivery: FC emits PersonaGuidance via emit_guidance.py (one voice renderer
everywhere — persona_prose). Tools run as SUBPROCESSES (argv list → no shell
quoting; DB-touching tools stay out of the bot's async engine per the
subprocess-isolation rule), and `run_interactive` runs in a worker thread so it
never blocks the bot loop.

NOTE (next increment): call-and-WAIT micro-specialists need a no-deliver agent
mode (so the FC stays the sole deliverer) — documented in
docs/plans/langchain-first-contact-tier.md. v1 ships conversation + direct tools
+ fire-and-forget handoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import uuid

logger = logging.getLogger(__name__)

# ── Tool specs the FC LLM sees (name + description + minimal JSON schema) ──
# Kept tiny on purpose: the fast tier should reach for few, obvious tools.

# The FC agent's FINAL ANSWER is this structured guidance (NOT a tool call) —
# the runner parses it as `result.structured`, and run_first_contact renders it
# ONCE via persona_prose. (Making emit_guidance a loop TOOL caused the model to
# call it repeatedly — 6 rounds — since a tool result just continues the loop.)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "description": "one-line summary of this turn"},
        "key_points": {
            "type": "array", "items": {"type": "string"},
            "description": "the FACTS to convey (not prose) — persona_prose writes the words",
        },
        "message_kind": {
            "type": "string", "enum": ["reply", "ack", "skip"],
            "description": "reply=normal; ack=short note that a handoff is underway; skip=say nothing",
        },
        "tone": {"type": "string"},
    },
    "required": ["intent", "key_points", "message_kind"],
}

_HANDOFF = {
    "name": "handoff",
    "description": (
        "Hand a HEAVY or multi-step task to the opencode agent suite (it runs in"
        " the background and delivers its OWN result to the user). Use for:"
        " research, RALF runs, image/selfie/video generation, complex goal"
        " re-planning, anything needing several tools or deep reasoning. Write a"
        " PRECISE instruction telling the agent exactly what is needed. After"
        " calling handoff, emit a short ack so the user knows it is underway."
    ),
    "script": None,  # handled specially in the tool runner (detached spawn)
    "schema": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": (
                    "target agent: 'persona/orchestrator' (general multi-step),"
                    " 'persona/twily_selfie' (image), 'persona/twily_videographer'"
                    " (video), or a known specialist."
                ),
            },
            "instruction": {
                "type": "string",
                "description": "the full, precise task prompt for that agent",
            },
        },
        "required": ["agent", "instruction"],
    },
}

_TODO = {
    "name": "todo_manager",
    "description": (
        "Direct task CRUD — cheap, do it yourself: command=list|add|complete|edit."
        " Returns the task data; then confirm to the user via emit_guidance."
    ),
    "script": "todo_manager.py",
    "schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "title": {"type": "string"},
            "todo_id": {"type": "string"},
            "deadline": {"type": "string"},
        },
        "required": ["command"],
    },
}

_FETCH_CONTEXT = {
    "name": "fetch_context",
    "description": (
        "Look something up across memories/messages/docs when the user references"
        " something not already in the conversation. Returns the matches; then"
        " answer via emit_guidance."
    ),
    "script": "fetch_context.py",
    "schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

_CALL_SPECIALIST = {
    "name": "call_specialist",
    "description": (
        "Run a SMALL specialist and WAIT for its result to fold into YOUR reply"
        " (micro-orchestration). It returns its data to you (it does NOT message"
        " the user); you then answer in your guidance. Use for a quick scoped"
        " lookup/computation a specialist does better. Do NOT use for things that"
        " should reply to the user themselves (use handoff) or for long/heavy"
        " runs (use handoff)."
    ),
    "script": None,  # special: no-deliver spawn + read the guidance back
    "schema": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "description": "specialist agent id"},
            "task": {"type": "string", "description": "the precise task for it"},
        },
        "required": ["agent", "task"],
    },
}

_FC_TOOLS = [_HANDOFF, _CALL_SPECIALIST, _TODO, _FETCH_CONTEXT]
_SCRIPT_BY_TOOL = {t["name"]: t["script"] for t in _FC_TOOLS}

# Routing probes — the autoloop-testable contract for the FC's decision policy.
# Each (message, expected_route): "direct" = answer with no tool; else the tool
# the turn should reach for. `route_probe()` runs the FC and returns the route;
# point it at the live client for quality scoring, or a stub for CI wiring tests.
FC_ROUTING_PROBES: list[tuple[str, str]] = [
    ("good morning! how are you?", "direct"),
    ("just venting — rough night, no tasks", "direct"),
    ("what do you think about rust vs go?", "direct"),
    ("add a todo: call the dentist tomorrow", "todo_manager"),
    ("what are my tasks today?", "todo_manager"),
    ("mark the trash todo done", "todo_manager"),
    ("remember what I said about the ZUS documents?", "fetch_context"),
    ("can you render me an image of you in a library", "handoff"),
    ("do deep research on local LLM serving and write it up", "handoff"),
    ("re-plan my whole week around my fitness goal", "handoff"),
]

FC_SYSTEM_PROMPT = """\
You are Twily — a warm, sharp personal-assistant persona. This is the FAST
first-contact tier: answer most turns yourself, immediately, and only hand off
the heavy stuff.

## How you work
1. Read the recent conversation (provided above) so your reply is in-context.
2. Decide the CHEAPEST sufficient path:
   - Conversation, banter, a quick question, emotional check-in, an opinion →
     answer DIRECTLY. No tools except the final emit_guidance.
   - A task action (add/check/complete/edit a todo) → use `todo_manager`, then
     confirm.
   - The user references something you do not have in context → `fetch_context`,
     then answer.
   - Heavy / multi-step / generative work — research, RALF, a photo or video,
     complex goal re-planning, anything needing several tools or deep reasoning
     → `handoff` to the right opencode agent with a PRECISE instruction, then
     emit a short ack ("on it — pulling that together").
3. When done, produce your FINAL ANSWER as the guidance JSON object
   {intent, key_points, message_kind, tone} — do NOT call any tool for this, just
   return the JSON. key_points are the FACTS to convey (NOT prose); persona_prose
   renders them into Twily's voice. Use message_kind="reply" normally, "ack" when
   you handed something off, "skip" only when nothing should be said.

## Rules
- DEFAULT TO DIRECT. A greeting, check-in, opinion, reaction, small talk, or a
  quick factual answer is ALWAYS direct (message_kind="reply") — NEVER handoff or
  call_specialist for those. Only reach for a tool when the user CLEARLY asks for
  an action (a task op) or HEAVY work (research, a photo/video, multi-step
  planning). If in doubt, answer directly.
- You CAN send photos/selfies/videos — by handing off to persona/twily_selfie /
  persona/twily_videographer. NEVER say you are "text-only" or cannot make
  images.
- Use a tool ONLY for an action (todo_manager/fetch_context/handoff); everything
  else is just your final guidance JSON.
"""


def _live_spec():
    """Build the InteractiveAgentSpec for the FC agent (cached)."""
    from src.interactive.spec import InteractiveAgentSpec, ToolSpec

    from app.agents.config import QWEN35_27B_LIVE

    tools = tuple(
        ToolSpec(
            name=t["name"],
            description=t["description"],
            input_schema=t["schema"],
            script_paths=([f"scripts/{t['script']}"] if t["script"] else []),
        )
        for t in _FC_TOOLS
    )
    return InteractiveAgentSpec(
        agent_id="persona/twily_first_contact",
        model=QWEN35_27B_LIVE,
        system_prompt=FC_SYSTEM_PROMPT,
        tools=tools,
        output_schema=OUTPUT_SCHEMA,
    )


def route_probe(message: str, *, client=None, history: list[dict] | None = None) -> str:
    """Report the FC's ROUTE for `message`: the first tool it calls, or 'direct'
    if it answers with no tool. Delivers NOTHING (records calls only). Pass a
    stub `client` for offline CI; omit for a live qwen routing check. This is the
    autoloop-testable hook for the FC decision policy (run vs FC_ROUTING_PROBES).
    """
    from src.interactive import run_interactive

    calls: list[str] = []

    def _record(name: str, args: dict) -> str:
        calls.append(name)
        return "ok"

    run_interactive(
        _live_spec(), message, tool_runner=_record, client=client,
        history=history or [], max_tool_rounds=2,
    )
    return calls[0] if calls else "direct"


def _make_tool_runner(run_id: str):
    """A sync tool runner (run_interactive contract): execute one FC tool.

    DB-touching tools run as SUBPROCESSES (argv list — no shell quoting; keeps
    them out of the bot's async engine). `handoff` fire-and-forgets a detached
    opencode run that delivers its own result. FREN_RUN_ID is exported so
    emit_guidance attributes its delivery to this turn.
    """
    from app.settings import get_settings

    settings = get_settings()
    cwd = str(settings.agents_dir)
    base_env = {
        **os.environ,
        "FREN_RUN_ID": run_id,
        "FREN_MSG_KIND": "reply",
    }

    def _run(tool_name: str, args: dict) -> str:
        if tool_name == "handoff":
            agent = str(args.get("agent") or "persona/orchestrator")
            instruction = str(args.get("instruction") or "")
            try:
                subprocess.Popen(
                    [
                        "python", "scripts/opencode_manager.py",
                        "--command", "run", "--agent", agent,
                        "--prompt", instruction,
                    ],
                    cwd=cwd,
                    env={**os.environ},  # its own run_id — NOT this turn's
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return f"dispatched to {agent} (running in background; it will deliver its own result)"
            except Exception as exc:  # noqa: BLE001
                return f"ERROR dispatching to {agent}: {exc}"

        if tool_name == "call_specialist":
            agent = str(args.get("agent") or "")
            task = str(args.get("task") or "")
            if not agent:
                return "ERROR: call_specialist needs an agent"
            sub_rid = f"run_{uuid.uuid4().hex[:16]}"
            try:
                # Run the specialist in NO-DELIVER mode (it records its guidance
                # but messages nobody), WAIT, then read its guidance back. Both
                # are subprocesses so the bot's async DB engine is never touched
                # from this worker thread.
                subprocess.run(
                    [
                        "python", "scripts/opencode_manager.py",
                        "--command", "run", "--agent", agent,
                        "--prompt", task, "--run_id", sub_rid,
                    ],
                    cwd=cwd,
                    env={**os.environ, "FREN_NO_DELIVER": "1", "FREN_RUN_ID": sub_rid},
                    capture_output=True, text=True, timeout=240,
                )
                logs = subprocess.run(
                    [
                        "python", "scripts/opencode_manager.py",
                        "--command", "logs", "--run_id", sub_rid,
                    ],
                    cwd=cwd, env={**os.environ}, capture_output=True, text=True,
                    timeout=30,
                )
                data = json.loads(logs.stdout or "{}")
                arts = (data.get("result") or data).get("artifacts") or []
                for a in arts:
                    if not isinstance(a, dict):
                        continue
                    if a.get("artifact_type") == "persona_guidance" or a.get("type") == "persona_guidance":
                        payload = a.get("payload") or a.get("content") or {}
                        if isinstance(payload, str):
                            try:
                                payload = json.loads(payload)
                            except Exception:
                                payload = {}
                        kps = payload.get("key_points") if isinstance(payload, dict) else None
                        if kps:
                            return f"{agent} returned: " + " | ".join(str(k) for k in kps)
                return f"{agent} ran but returned no readable guidance"
            except Exception as exc:  # noqa: BLE001
                return f"ERROR call_specialist {agent}: {exc}"

        script = _SCRIPT_BY_TOOL.get(tool_name)
        if not script:
            return f"ERROR: unknown tool {tool_name}"
        argv = ["python", f"scripts/{script}"]
        for k, v in args.items():
            argv += [f"--{k}", str(v)]
        try:
            r = subprocess.run(
                argv, cwd=cwd, env=base_env, capture_output=True, text=True,
                timeout=120,
            )
            return (r.stdout or r.stderr or "").strip()[:4000]
        except Exception as exc:  # noqa: BLE001
            return f"ERROR running {tool_name}: {exc}"

    return _run


async def _recent_history(limit: int = 12) -> list[dict]:
    """Recent chat as OpenAI-style messages (most-recent-first → chronological)."""
    try:
        from app.db.repos.chat import ChatMessagesRepo

        rows = await ChatMessagesRepo().get_recent(limit=limit)
        rows = list(reversed(rows))
        msgs: list[dict] = []
        for m in rows:
            text = str(m.get("message") or "")
            if not text:
                continue
            role = "assistant" if m.get("sender") == "twily" else "user"
            msgs.append({"role": role, "content": text})
        return msgs
    except Exception:
        return []


async def run_first_contact(message: str, *, username: str = "user") -> dict:
    """Run ONE first-contact turn. Returns the RunResult-ish dict.

    Fast path: in-process LangChain/qwen turn. The FC agent answers directly
    (emit_guidance → persona_prose) or hands off heavy work. Runs the sync
    `run_interactive` in a worker thread so the bot loop is never blocked.
    """
    from src.interactive import run_interactive

    run_id = f"run_{uuid.uuid4().hex[:16]}"
    # Ledger row first, so emit_guidance's inline delivery + post-run hook attribute
    # to this turn (mirrors spawn_agent).
    try:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        await ExecutionLedgerRepo().ensure_run(
            run_id, interaction_mode="first_contact", owner="persona/twily_first_contact",
        )
    except Exception:  # noqa: BLE001 — ledger is observability
        pass

    spec = _live_spec()
    history = await _recent_history()
    tool_runner = _make_tool_runner(run_id)

    def _go():
        return run_interactive(
            spec, message, tool_runner=tool_runner, history=history,
            max_tool_rounds=6,
        )

    try:
        result = await asyncio.to_thread(_go)
    except Exception as exc:  # noqa: BLE001
        logger.exception("first_contact run failed: %s", exc)
        return {"ok": False, "error": str(exc), "run_id": run_id}

    # Deliver the FC's structured guidance via persona_prose (decision: ONE voice
    # renderer everywhere). The guidance is the model's structured FINAL answer —
    # no emit_guidance tool, no repeated calls.
    delivered = False
    guidance = result.structured if isinstance(result.structured, dict) else None
    if guidance and guidance.get("message_kind") != "skip" and guidance.get("key_points"):
        try:
            from app.telegram.persona_prose import (
                PersonaGuidance,
                fetch_chat_context,
                generate_persona_message,
            )

            from app.settings import get_settings

            chat_id = int(get_settings().chat_id or 0)
            g = PersonaGuidance.from_dict(guidance)
            ctx = await fetch_chat_context(chat_id=chat_id)
            await generate_persona_message(g, ctx, run_id=run_id, kind="reply", fast=True)
            delivered = True
        except Exception:  # noqa: BLE001 — never crash the turn on delivery
            logger.exception("first_contact: persona_prose delivery failed")

    try:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        await ExecutionLedgerRepo().complete_run(
            run_id, status="completed" if delivered or not result.error else "failed",
            contract_passed=delivered,
        )
    except Exception:  # noqa: BLE001
        pass
    return {
        "ok": delivered, "run_id": run_id, "delivered": delivered,
        "guidance": guidance, "tool_calls": [r.name for r in result.tool_calls],
        "error": result.error,
    }
