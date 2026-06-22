"""Post-run delivery verifier — keeps every opencode flow HONEST.

When an opencode flow finishes, this runs ONE cheap local-qwen audit that asks a
single question: *did the user actually get what they asked for?* It compares the
original request (the prompt) against what was really delivered back (the
persona_response text + a summary of the tools the agent ran). If the request was
NOT delivered, it states WHAT went wrong and hands the task to the orchestrator
with a precise instruction to retry/reroute a different way.

Live failure modes this catches (all observed in the YouTube-link incident):
  - the agent printed a fake command as TEXT and ran zero tools (video_analyst
    "uv run scripts/research_manager.py …" — nothing executed);
  - the agent fetched the data but a follow-up turn misclassified itself as an
    "inner thought, no new message" and SKIPPED delivery (the
    "I'm pulling the transcript now…" that never arrived);
  - a refusal of something she can actually do ("I can't access YouTube links").

Design guards:
  - bounded to ONE re-dispatch (``depth`` guard) — it can never loop;
  - conversation/banter is treated as delivered (no reroute);
  - a deterministic pre-filter skips the LLM call for plain chat that promised
    nothing, so banter stays cheap (audit only when the turn was a task trigger
    OR the reply contains a "I'll do X" promise marker);
  - fire-and-forget: callers never block the user-facing path on it.
"""

from __future__ import annotations

import logging
import re
import uuid

logger = logging.getLogger(__name__)

# Triggers that represent a concrete DELIVERABLE the user is owed (always audit).
_TASK_TRIGGERS = {
    "video_analysis",
    "document_analysis",
    "workflow",
    "handoff",
    "research",
    "ralf",
    "delivery_retry",  # audited too, but depth guard blocks a second reroute
}

# "I'll do X / it's coming" language with no result behind it is the classic
# promise-without-delivery tell — audit those even on the plain chat path.
_PROMISE_RE = re.compile(
    r"\b("
    r"pulling (it|the|them)|i'?ll (list|get|send|find|have|put together|pull|grab|compile|draw|make|build|gather)"
    r"|in a (moment|sec|second|minute|bit)|give me a (sec|moment|minute)"
    r"|let me (get|pull|grab|check|look|fetch|find)"
    r"|generating|fetching|compiling|working on it|on it|coming (right )?up"
    r"|i'?m (pulling|fetching|getting|generating|compiling|working|looking)"
    r"|hold on|one (sec|moment|minute)|stand by|right away"
    r")\b",
    re.IGNORECASE,
)

_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "delivered": {
            "type": "boolean",
            "description": "did the user actually RECEIVE what they asked for (the real content/answer/file), not just a promise of it?",
        },
        "is_conversation": {
            "type": "boolean",
            "description": "was the user's message just banter / a greeting / an emotional check-in / had no concrete deliverable? (then delivered is irrelevant)",
        },
        "what_went_wrong": {
            "type": "string",
            "description": "if not delivered: ONE sentence on what failed (e.g. 'promised the transcript analysis but never sent it', 'ran no tools', 'falsely claimed it can't access YouTube')",
        },
        "reroute_instruction": {
            "type": "string",
            "description": "if not delivered: a precise instruction for the orchestrator — exactly WHAT to produce and deliver, and a hint at a DIFFERENT approach to try",
        },
        "confidence": {"type": "number", "description": "0..1 confidence that it was genuinely NOT delivered"},
    },
    "required": ["delivered", "is_conversation", "confidence"],
}

_AUDIT_SYSTEM = """\
You are a strict delivery auditor for a personal-assistant persona named Twily.
You are given (1) what the USER asked for and (2) what Twily ACTUALLY delivered
back to them, plus a summary of the tools she ran. Decide whether the user's
request was genuinely FULFILLED and the result DELIVERED.

Rules — be precise, not generous:
- If the user's message was casual conversation, banter, a greeting, an emotional
  check-in, an opinion ask, or had no concrete deliverable → is_conversation=true
  (delivered does not matter).
- A PROMISE is NOT a delivery. "I'm pulling the transcript now", "I'll list them
  in a moment", "generating…", "let me look", "on it" — if the actual content the
  user asked for is NOT present in what was delivered, delivered=false.
- A REFUSAL of something she can do is a failure. If she said she "can't access
  YouTube / can't read links / is text-only" but the user asked her to analyze a
  link or media → delivered=false.
- If she ran NO meaningful tools and produced only vague chatter for a concrete
  task → delivered=false.
- If the concrete answer / content / file the user asked for IS present in the
  delivered text → delivered=true.
- When delivered=false, write what_went_wrong (one sentence) and a precise
  reroute_instruction telling an orchestrator EXACTLY what to produce and deliver,
  plus a hint at a different approach to try.
Return ONLY the structured object.
"""


_YT_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([\w-]{11})"
)


async def _gather_grounding(request: str, *, max_body_chars: int = 6000) -> str:
    """Carry ALREADY-produced work into the recovery run so the orchestrator gets
    DATA, not a treasure hunt.

    The recovery target (persona/orchestrator) has a read-side toolset but NOT
    every fetch tool — e.g. it can discover that a YouTube transcript was cached
    but has no tool to read the transcript BODY (that lives in youtube_videos,
    reachable only via youtube_fetcher, which it lacks). So we pull the actual
    bodies and inline a bounded excerpt. This is the difference between "go find
    out how to get a transcript" (the orchestrator flails — it has no such tool)
    and "here is the transcript, extract the tracks" (trivially solvable).

    Grounding is keyed on ENTITIES IN THE REQUEST (e.g. a YouTube id in the URL),
    not a time window — so it is robust whether the verifier fires seconds or
    hours after the flow. Recent context_cache rows are added as a supplement."""
    lines: list[str] = []
    try:
        from app.db.session import fetch_all, get_async_session

        async with get_async_session() as s:
            # 1) Entity-keyed: a YouTube id in the request → its transcript body.
            m = _YT_ID_RE.search(request or "")
            if m:
                yt_id = m.group(1)
                rows = await fetch_all(
                    s,
                    "SELECT video_id, title, transcript FROM youtube_videos "
                    "WHERE yt_video_id = :y AND transcript IS NOT NULL AND transcript <> '' "
                    "ORDER BY length(transcript) DESC LIMIT 1",
                    {"y": yt_id},
                )
                if rows:
                    title = str(rows[0].get("title") or "").strip()
                    body = str(rows[0].get("transcript") or "")
                    excerpt = body[:max_body_chars]
                    trunc = " …(truncated)" if len(body) > max_body_chars else ""
                    lines.append(
                        f"- YouTube transcript ALREADY fetched for {yt_id}"
                        + (f" — \"{title}\"" if title else "")
                        + f" ({len(body)} chars)."
                    )
                    lines.append(
                        f"\n  TRANSCRIPT TEXT ({len(excerpt)} of {len(body)} chars){trunc}:\n"
                        f"  \"\"\"\n{excerpt}\n  \"\"\"\n"
                    )

            # 2) Supplement: anything the just-failed flow cached very recently.
            recent = await fetch_all(
                s,
                """
                SELECT artifact_type, entity_id, summary
                FROM context_cache
                WHERE created_at > now() - interval '45 minutes'
                ORDER BY created_at DESC LIMIT 5
                """,
                {},
            )
            for r in recent:
                summary = str(r.get("summary") or "").strip()
                if summary and summary not in "\n".join(lines):
                    eid = str(r.get("entity_id") or "")
                    atype = str(r.get("artifact_type") or "")
                    lines.append(f"- [{atype}] {summary}" + (f" (entity_id={eid})" if eid else ""))
    except Exception:  # noqa: BLE001
        logger.debug("delivery_verify: grounding gather failed", exc_info=True)
    return "\n".join(lines).strip()


async def _collect_delivery(run_id: str) -> dict:
    """Gather what a run actually delivered: the user-facing text + a terse tool
    summary + whether the run skipped delivery entirely."""
    delivered_text = ""
    tool_summary = ""
    skipped = False
    try:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        arts = await ExecutionLedgerRepo().list_artifacts(run_id)
        for a in arts:
            payload = a.get("payload") or {}
            if isinstance(payload, str):
                import json

                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
            atype = a.get("artifact_type")
            if atype == "persona_response":
                if payload.get("skipped") or payload.get("kind") == "skip":
                    skipped = True
                txt = str(payload.get("delivered_text") or "")
                if txt:
                    delivered_text = txt
            elif atype == "run_trace":
                tcs = payload.get("tool_calls") or []
                names = [str(t.get("name") or "") for t in tcs if isinstance(t, dict)]
                tool_summary = ", ".join(n for n in names if n) or "(no tools called)"
                # an agent that only emitted text and called nothing is a red flag
                if not names and not tool_summary:
                    tool_summary = "(no tools called)"
    except Exception:  # noqa: BLE001
        logger.debug("delivery_verify: collect failed for %s", run_id, exc_info=True)
    return {"delivered_text": delivered_text, "tool_summary": tool_summary, "skipped": skipped}


async def _audit(request: str, delivery: dict) -> dict:
    """One cheap structured local-qwen call: was it delivered?"""
    import asyncio

    from src.interactive import run_interactive
    from src.interactive.runner import OpenAICompatClient
    from src.interactive.spec import InteractiveAgentSpec

    from app.agents.config import QWEN35_27B_LIVE

    spec = InteractiveAgentSpec(
        agent_id="persona/delivery_audit", model=QWEN35_27B_LIVE,
        system_prompt=_AUDIT_SYSTEM, tools=(), output_schema=_AUDIT_SCHEMA,
    )
    client = OpenAICompatClient.from_spec(spec)
    client.default_params["max_tokens"] = 800

    delivered_text = delivery.get("delivered_text") or ("(NOTHING was delivered — the turn was skipped)" if delivery.get("skipped") else "(nothing delivered)")
    user_msg = (
        f"## USER ASKED FOR\n{request.strip()}\n\n"
        f"## TWILY ACTUALLY DELIVERED\n{delivered_text.strip()}\n\n"
        f"## TOOLS SHE RAN\n{delivery.get('tool_summary') or '(unknown)'}\n\n"
        "Audit it. Return ONLY the structured object."
    )
    try:
        res = await asyncio.to_thread(
            lambda: run_interactive(spec, user_msg, client=client, history=[], max_tool_rounds=1)
        )
        return res.structured if isinstance(res.structured, dict) else {}
    except Exception:  # noqa: BLE001
        logger.debug("delivery_verify: audit call failed", exc_info=True)
        return {}


async def verify_and_maybe_reroute(
    *, agent: str, run_id: str, request: str, trigger: str = "", depth: int = 0,
) -> None:
    """Audit one finished flow's delivery; if the request was NOT delivered, hand
    it to the orchestrator with a precise retry/reroute instruction.

    Bounded to a single re-dispatch (``depth`` guard). Safe to fire-and-forget.
    """
    if depth >= 1:
        return  # the retry flow itself is not re-rerouted — single bounded attempt
    if not request or not request.strip():
        return
    try:
        from app.telegram.persona_prose import is_excluded_agent

        if is_excluded_agent(agent):
            return
    except Exception:  # noqa: BLE001
        pass
    # Don't fire a follow-up into a dying bot.
    try:
        from app.telegram import bot as _bot

        if getattr(_bot, "_shutting_down", False):
            return
    except Exception:  # noqa: BLE001
        pass

    delivery = await _collect_delivery(run_id)

    # Deterministic pre-filter: audit task triggers and skipped turns always;
    # for plain chat, only audit when the reply made a promise (else it's banter).
    task_like = trigger in _TASK_TRIGGERS
    promised = bool(_PROMISE_RE.search(delivery.get("delivered_text") or ""))
    if not task_like and not delivery.get("skipped") and not promised:
        return

    verdict = await _audit(request, delivery)
    if not verdict:
        return
    if verdict.get("delivered") or verdict.get("is_conversation"):
        return
    try:
        confidence = float(verdict.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.6:
        return

    what = str(verdict.get("what_went_wrong") or "the result was never delivered").strip()
    how = str(verdict.get("reroute_instruction") or "Complete the task and deliver the result.").strip()
    logger.info(
        "delivery_verify: run=%s agent=%s NOT delivered (%.2f) — rerouting to orchestrator: %s",
        run_id, agent, confidence, what,
    )

    # Carry the data the failed flow ALREADY produced into the recovery run. This
    # is what makes recovery *feasible* rather than a treasure hunt: the
    # orchestrator's toolset can't re-fetch everything (e.g. it has no tool to
    # read a YouTube transcript body), so we hand it the data inline.
    grounding = await _gather_grounding(request)
    grounding_block = (
        f"\nDATA ALREADY FETCHED — use this directly, do NOT try to re-fetch it "
        f"(you may not have a tool that can):\n{grounding}\n"
        if grounding else ""
    )

    instruction = (
        "A previous attempt to handle the user's request DID NOT deliver it. You are the "
        "recovery orchestrator — fix it and DELIVER.\n\n"
        f"USER'S ORIGINAL REQUEST:\n{request.strip()}\n\n"
        f"WHAT WENT WRONG (first attempt by {agent}):\n{what}\n\n"
        f"WHAT IS STILL NEEDED:\n{how}\n"
        f"{grounding_block}\n"
        "Work with the tools you ACTUALLY have and the data above — do not assume a tool exists. "
        "If the data you need is already inlined above, just use it directly. When you have the "
        "real result, DELIVER it to the user via emit_guidance — do not merely promise it."
    )

    try:
        from app.telegram.bot import _post_run_persona_delivery, _tts_postfix
        from app.telegram.spawn import spawn_agent
        from app.telegram.state import get_model, get_postfix

        postfix = get_postfix(get_model())
        new_run = f"run_{uuid.uuid4().hex[:16]}"
        result = await spawn_agent(
            agent="persona/orchestrator",
            prompt=instruction,
            run_id=new_run,
            model_postfix=postfix,
            tts_postfix=_tts_postfix(),
            timeout_s=600,
            trigger="delivery_retry",
            extra_env={"FREN_MSG_KIND": "reply"},
        )
        await _post_run_persona_delivery("persona/orchestrator", new_run)
        # Single bounded retry: audit the recovery run too, but depth=1 means it
        # can no longer reroute — it only logs if the retry also failed.
        await verify_and_maybe_reroute(
            agent="persona/orchestrator", run_id=new_run, request=request,
            trigger="delivery_retry", depth=1,
        )
        if not result.ok:
            logger.warning("delivery_verify: recovery orchestrator run %s not ok: %s", new_run, result.error)
    except Exception:  # noqa: BLE001
        logger.exception("delivery_verify: reroute to orchestrator failed for run=%s", run_id)
