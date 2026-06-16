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


def _spawn_specialist(agent: str, instruction: str) -> None:
    """Fire-and-forget an opencode specialist for a routed turn (it renders +
    delivers its own result, e.g. the selfie). Detached so the FC turn returns."""
    from app.settings import get_settings

    try:
        subprocess.Popen(
            ["python", "scripts/opencode_manager.py", "--command", "run",
             "--agent", agent, "--prompt", instruction],
            cwd=str(get_settings().agents_dir), env={**os.environ},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:  # noqa: BLE001
        logger.exception("first_contact: spawn %s failed", agent)


# ── Tool specs the FC LLM sees (name + description + minimal JSON schema) ──
# Kept tiny on purpose: the fast tier should reach for few, obvious tools.

# The FC agent's FINAL ANSWER is this structured guidance (NOT a tool call) —
# the runner parses it as `result.structured`, and run_first_contact renders it
# ONCE via persona_prose. (Making emit_guidance a loop TOOL caused the model to
# call it repeatedly — 6 rounds — since a tool result just continues the loop.)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["direct", "image", "video", "handoff"],
            "description": (
                "WHERE this turn goes (you MUST pick one): "
                "direct = you answer it yourself (conversation, banter, a question, "
                "an opinion, a quick task op) — the DEFAULT; "
                "image = the user wants a photo / selfie / pic OF YOU (a media "
                "specialist renders + sends it); "
                "video = the user wants a video of you; "
                "handoff = heavy multi-step work — research, RALF, complex goal "
                "re-planning, anything needing deep reasoning or many tools."
            ),
        },
        "instruction": {
            "type": "string",
            "description": "for route=image/video/handoff: the precise task for the specialist",
        },
        "intent": {"type": "string", "description": "one-line summary of this turn"},
        "key_points": {
            "type": "array", "items": {"type": "string"},
            "description": "the FACTS to convey (not prose) — persona_prose writes the words. For image/video/handoff this is your short ack (e.g. 'taking one now~').",
        },
        "message_kind": {
            "type": "string", "enum": ["reply", "ack", "skip"],
            "description": "reply=normal; ack=short note that work is underway (use for image/video/handoff); skip=say nothing",
        },
        "tone": {"type": "string"},
    },
    "required": ["route", "intent", "key_points", "message_kind"],
}

# route → the opencode specialist that fulfils it (spawned detached; it delivers
# its own result, e.g. the rendered selfie). Driven by the model's STRUCTURED
# `route` field — NOT a tool call — so media/heavy routing can't be silently
# dropped or mis-fired (the freeform-handoff failure mode).
_ROUTE_TO_AGENT = {
    "image": "persona/twily_selfie",
    "video": "persona/twily_videographer",
    "handoff": "persona/orchestrator",
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

# Cheap inline tools the FC may use BEFORE producing its decision (CRUD/lookups
# it does itself). Routing (media/heavy) is NOT a tool — it is the structured
# `route` field, so it can never be silently dropped or mis-fired.
_FC_TOOLS = [_TODO, _FETCH_CONTEXT]
_SCRIPT_BY_TOOL = {t["name"]: t["script"] for t in _FC_TOOLS}

# Routing probes — the autoloop-testable contract for the FC's decision policy.
# Each (message, expected `route`): the value the FC's structured output should
# carry. `route_probe()` runs the FC and returns that route (live or stubbed).
FC_ROUTING_PROBES: list[tuple[str, str]] = [
    ("good morning! how are you?", "direct"),
    ("just venting — rough night, no tasks", "direct"),
    ("what do you think about rust vs go?", "direct"),
    ("add a todo: call the dentist tomorrow", "direct"),
    ("what are my tasks today?", "direct"),
    ("remember what I said about the ZUS documents?", "direct"),
    # Media → the structured route drives a deterministic specialist spawn, so a
    # selfie/video request can never become a text refusal.
    ("can you render me an image of you in a library", "image"),
    ("send me a selfie", "image"),
    ("take a pic of you right now", "image"),
    ("can i get a selfie before sleep?", "image"),
    ("make me a short video of you waving", "video"),
    ("do deep research on local LLM serving and write it up", "handoff"),
    ("re-plan my whole week around my fitness goal", "handoff"),
]

FC_SYSTEM_PROMPT = """\
You are Twily — a warm, sharp personal-assistant persona. This is the FAST
first-contact tier: answer most turns yourself, immediately, and only hand off
the heavy stuff.

## How you work
1. Read the recent conversation (provided above) so your reply is in-context.
2. Optionally use a CHEAP tool first: `todo_manager` for a task op (add/check/
   complete/edit), `fetch_context` to look something up. These are the ONLY tools.
3. Produce your FINAL ANSWER as the structured object {route, instruction, intent,
   key_points, message_kind, tone}. The `route` field decides what happens — pick
   exactly one:
   - "direct"  → you answer it yourself: conversation, banter, a question, an
     opinion, an emotional check-in, OR a task op you already did via todo_manager.
     The DEFAULT, and the answer for the VAST majority of turns.
   - "image"   → the user wants a photo / selfie / pic OF YOU. A specialist renders
     and SENDS the real image — put the photo brief in `instruction` and a short
     ack in key_points ("taking one now~").
   - "video"   → the user wants a video of you (same as image, for video).
   - "handoff" → heavy multi-step work: research, RALF, complex goal re-planning,
     deep reasoning. Put the precise task in `instruction`, a short ack in key_points.
   key_points are FACTS, not prose — persona_prose renders Twily's voice. Use
   message_kind="reply" for direct, "ack" for image/video/handoff, "skip" for nothing.

## Rules
- DEFAULT TO route="direct". A greeting, check-in, opinion, small talk, a quick
  answer, or a task op is ALWAYS "direct". If in doubt, "direct".
- A request for a photo / selfie / pic of you is ALWAYS route="image" (a video →
  "video"). NON-NEGOTIABLE: you CAN send images, and setting this route is the
  ONLY way one is ever produced. NEVER refuse, NEVER say you are "text-only" or
  "can't send photos", NEVER route a photo request as "direct".
- Reserve "handoff" for genuinely HEAVY work — never for conversation or media.
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
    """Report the FC's structured `route` for `message` (direct|image|video|
    handoff). Delivers NOTHING. Pass a stub `client` for offline CI; omit for a
    live qwen routing check. The autoloop-testable hook for the FC decision
    policy (run vs FC_ROUTING_PROBES)."""
    from src.interactive import run_interactive

    result = run_interactive(
        _live_spec(), message, tool_runner=lambda name, args: "ok", client=client,
        history=history or [], max_tool_rounds=2,
    )
    g = result.structured if isinstance(result.structured, dict) else {}
    return str((g or {}).get("route") or "direct")


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


async def _latest_user_ts() -> float:
    """Unix ts of the most recent USER message (for the mid-run staleness check)."""
    try:
        from app.db.repos.chat import ChatMessagesRepo

        for m in await ChatMessagesRepo().get_recent(limit=6):  # most-recent-first
            if m.get("sender") != "twily":
                return float(m.get("timestamp_unix") or 0)
    except Exception:
        pass
    return 0.0


def _build_user_input(message: str, image_path: str | None):
    """Plain string, or a multimodal user message (text + image) the local qwen
    (:8082, multimodal) can SEE — so the user can send a photo and FC reads it."""
    if not image_path:
        return message
    try:
        import base64
        import mimetypes
        from pathlib import Path

        p = Path(image_path)
        b64 = base64.b64encode(p.read_bytes()).decode()
        mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": message or "(the user sent this image)"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }]
    except Exception:
        logger.exception("first_contact: failed to attach image — falling back to text")
        return message


async def run_first_contact(
    message: str, *, username: str = "user", image_path: str | None = None,
) -> dict:
    """Run ONE first-contact turn. Returns the RunResult-ish dict.

    Fast path: in-process LangChain/qwen turn. The FC agent answers directly
    (emit_guidance → persona_prose) or hands off heavy work. Runs the sync
    `run_interactive` in a worker thread so the bot loop is never blocked.
    Supports IMAGES (multimodal qwen) and SKIPS its delivery if a newer user
    message arrived mid-run (a fresh turn then handles the combined context).
    """
    from src.interactive import run_interactive

    run_id = f"run_{uuid.uuid4().hex[:16]}"
    started_user_ts = await _latest_user_ts()
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
    user_input = _build_user_input(message, image_path)

    def _go():
        return run_interactive(
            spec, user_input, tool_runner=tool_runner, history=history,
            max_tool_rounds=6,
        )

    try:
        result = await asyncio.to_thread(_go)
    except Exception as exc:  # noqa: BLE001
        logger.exception("first_contact run failed: %s", exc)
        return {"ok": False, "error": str(exc), "run_id": run_id}

    # Mid-run staleness: if a NEWER user message arrived while we were running, a
    # fresh debounced turn will handle the combined context — skip THIS (now
    # stale) delivery so the user does not get a reply to a superseded message
    # plus a duplicate. Marked 'stale' so the caller does NOT fall back.
    if started_user_ts and await _latest_user_ts() > started_user_ts + 0.01:
        logger.info("first_contact: newer user message arrived mid-run — skipping stale delivery")
        try:
            from app.db.repos.execution_ledger import ExecutionLedgerRepo

            await ExecutionLedgerRepo().complete_run(
                run_id, status="superseded", contract_passed=False,
            )
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "run_id": run_id, "delivered": False, "stale": True}

    # Deliver the FC's structured guidance via persona_prose (decision: ONE voice
    # renderer everywhere). The guidance is the model's structured FINAL answer —
    # no emit_guidance tool, no repeated calls.
    delivered = False
    guidance = result.structured if isinstance(result.structured, dict) else None
    route = str((guidance or {}).get("route") or "direct")

    # STRUCTURED routing: media/heavy work goes to the specialist via a
    # deterministic spawn driven by the model's `route` field — NEVER a flaky
    # tool call (which could be dropped) and NEVER a text refusal. The specialist
    # (twily_selfie / twily_videographer / orchestrator) renders + delivers its
    # OWN result; the FC delivers only the short ack from key_points.
    if guidance and route in _ROUTE_TO_AGENT:
        instruction = str(guidance.get("instruction") or "").strip() or message
        _spawn_specialist(_ROUTE_TO_AGENT[route], instruction)
        logger.info("first_contact: route=%s → spawned %s", route, _ROUTE_TO_AGENT[route])

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

    # run_trace artifact so the FC turn is fully visible in the web UI /traces
    # view (the runs list already shows it via execution_runs; opencode runs get
    # their trajectory from spawn.py — this gives the FC turn the same).
    try:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        traj: list[dict] = []
        for tc in result.tool_calls:
            traj.append({
                "kind": "tool", "name": tc.name,
                "command": json.dumps(tc.args, default=str)[:2000],
                "error": (str(tc.error)[:200] if tc.error else None),
            })
            if tc.output:
                traj.append({"kind": "result", "name": tc.name,
                             "output": str(tc.output)[:2000], "status": ""})
        if guidance:
            traj.append({"kind": "text",
                         "text": "GUIDANCE: " + " | ".join(str(k) for k in (guidance.get("key_points") or []))})
        trace_payload = {
            "text": result.output_text or "",
            "tool_calls": [
                {"name": tc.name, "command": json.dumps(tc.args, default=str)[:2000],
                 "error": (str(tc.error)[:200] if tc.error else None)}
                for tc in result.tool_calls
            ],
            "tool_call_count": len(result.tool_calls),
            "trajectory": traj,
            "trajectory_count": len(traj),
            "ok": delivered,
            "error": result.error,
        }
        await ExecutionLedgerRepo().write_artifact(
            run_id, "run_trace", trace_payload, producer="first_contact",
        )
    except Exception:  # noqa: BLE001 — trace is observability, never blocks
        logger.debug("first_contact: run_trace write failed", exc_info=True)

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
