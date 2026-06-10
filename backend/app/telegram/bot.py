"""Telegram bot — application setup and polling loop."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.settings import get_settings
from app.telegram.scheduler import Scheduler
from app.telegram.spawn import spawn_agent

logger = logging.getLogger(__name__)

_scheduler: Scheduler | None = None

# Strong references to background tasks so they are not GC'd mid-run.
# See https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
_background_personality_tasks: set[asyncio.Task[None]] = set()

# Set by the SIGTERM/SIGINT cleanup handler so error notifications and the
# post-run persona delivery hook stay quiet during Docker restarts — agent
# spawns get torn down with the process and we don't want to spam Telegram.
_shutting_down: bool = False


async def _post_run_persona_delivery(agent: str, run_id: str) -> None:
    """Post-run hook that hands control to persona_prose for delivery.

    Skipped when the bot is shutting down. In that case the agent likely
    didn't get to emit guidance (Docker restart / timeout kill / cleanup),
    and we don't want persona_prose to synthesize a flustered fallback for a
    dying bot process — it would race with shutdown and leak 'something went
    wrong' messages to the user.
    """
    if _shutting_down:
        logger.info("Skipping post-run delivery for %s (bot shutting down)", agent)
        return
    try:
        from app.telegram.persona_prose import deliver_guidance_from_ledger, is_excluded_agent

        if not is_excluded_agent(agent):
            await deliver_guidance_from_ledger(run_id=run_id)
    except Exception:
        logger.exception("post-run persona_prose delivery failed for %s", agent)


async def _send_error_notification(context: str) -> None:
    """Send a Twily-flavored error notification to the user via Telegram.

    No-op if the bot is currently shutting down — Docker-driven restarts
    cascade SIGTERM into child subprocesses, and we don't want to tell the
    user 'something went wrong' every time we ship a new fix.
    """
    if _shutting_down:
        logger.info("Suppressing error notification '%s' — bot is shutting down", context)
        return
    try:
        from telegram import Bot

        settings = get_settings()
        bot = Bot(token=settings.bot_token)
        await bot.initialize()
        await bot.send_message(
            chat_id=settings.chat_id,
            text=f"*taps horn nervously* Something went wrong behind the scenes ({context}). I'll be back in a moment!",
        )
    except Exception as e:
        logger.error("Failed to send error notification: %s", e)


def _get_trigger_commands() -> dict[str, str]:
    """Discover all trigger commands from agent definitions.

    v4 agents don't define a `trigger_command` (the field doesn't exist in the v4
    AgentDefinition and no agent sets one), so there are no slash-trigger commands
    to register. Return empty cleanly instead of importing the unported v3
    `app.compile.collect_all_agents` (which only logged a warning every startup).
    """
    return {}


def _infer_intents(message: str) -> str:
    """Run intent inference on raw message, return formatted section or empty string."""
    try:
        from app.tools.context.intent_inference import Input as IntentInput
        from app.tools.context.intent_inference import IntentInferenceTool

        result = IntentInferenceTool().execute(IntentInput(command="infer", message=message))
        if result.success and result.intents:
            lines = ["## Detected Intents (auto-inferred from message based on keywords)"]
            for intent in result.intents:
                lines.append(f"- **{intent['type']}**: {intent['description']}")
            lines.append(
                "\nUse these to guide your routing. If you handle an intent, note it for the sanity check step."
            )
            return "\n".join(lines)
    except Exception:
        pass
    return ""


def _agent_with_postfix(base_agent: str, model: str) -> str:
    """Resolve agent name with model-specific postfix."""
    from app.telegram.state import get_postfix

    postfix = get_postfix(model)
    return f"{base_agent}{postfix}"


def _tts_postfix() -> str:
    """Get TTS-specific postfix from tts_model state."""
    from app.telegram.state import get_postfix, get_tts_model

    return get_postfix(get_tts_model())


async def trigger_chatbot(
    message: str,
    username: str = "user",
    image_path: str | None = None,
    model: str | None = None,
    content_class: str = "public",
) -> None:
    """Run the persona orchestrator agent for a user message."""
    from app.telegram.state import format_header, get_model, is_local_model

    settings = get_settings()
    project_root = settings.project_root

    if model is None:
        model = get_model()

    # Space before and after @path for opencode file attachment parsing.
    image_ref = f" @{image_path} \n\n" if image_path else ""

    # Assemble in static → semi-static → volatile tiers so vLLM's prefix
    # cache can reuse the top of the prompt across turns. See
    # plans/refactored-forging-shore.md for the full rationale.
    static_sections: list[str] = []
    semi_static_sections: list[str] = []
    volatile_sections: list[str] = []

    # ── TIER 1: STATIC ──
    static_sections.append(
        "⚠️ FINAL ACTION: Emit PersonaGuidance via `uv run scripts/emit_guidance.py --data '{...}'`. "
        "Your text output is INVISIBLE to the user — only emit_guidance.py lets persona_prose deliver "
        "the actual reply. Do NOT call send_message.py."
    )

    # ── TIER 2: SEMI-STATIC ──
    try:
        from app.db.repos.user_rules import UserRulesRepo

        rules_text = await UserRulesRepo().format_rules_prompt()
        if rules_text:
            semi_static_sections.append(rules_text)
    except Exception:
        pass

    try:
        from app.db.repos.agent_lessons import AgentLessonsRepo

        lessons_text = await AgentLessonsRepo().format_lessons_prompt()
        if lessons_text:
            semi_static_sections.append(lessons_text)
    except Exception:
        pass

    # ── TIER 3: VOLATILE ──
    try:
        from app.db.repos.agent_notes import AgentNotesRepo

        repo = AgentNotesRepo()
        note = await repo.get("conversation_digest")
        if note and note.get("note_value"):
            val = note["note_value"]
            digest = val.get("digest", "") if isinstance(val, dict) else str(val)
            if digest:
                volatile_sections.append(digest)
    except Exception:
        pass

    try:
        from app.db.repos.daily_routines import DailyRoutinesRepo

        checklist = await DailyRoutinesRepo().get_checklist()
        if checklist:
            done = [r for r in checklist if r.get("completed")]
            pending = [r for r in checklist if not r.get("completed") and r.get("currently_visible")]
            lines = [f"## Daily Routines: {len(done)}/{len(checklist)} done"]
            if pending:
                lines.append("Pending: " + ", ".join(r["title"][:40] for r in pending[:5]))
            if done:
                lines.append("Done: " + ", ".join(r["title"][:40] for r in done[:5]))
            volatile_sections.append("\n".join(lines))
    except Exception:
        pass

    intents_section = _infer_intents(message)
    if intents_section:
        volatile_sections.append(intents_section)

    volatile_sections.append(f"---\n\n{message}" if message else "---")

    prompt = image_ref + "\n\n".join(static_sections + semi_static_sections + volatile_sections)

    # Brief terminal nudge — static, keeps the steer fresh right before generation.
    prompt += (
        "\n\n---REMINDER---\n"
        "FINAL action: ONE call to `uv run scripts/emit_guidance.py --data '{...}'`. "
        "Text output is invisible. Do NOT call send_message.py."
    )

    from app.telegram.state import get_postfix

    agent = _agent_with_postfix("persona/orchestrator", model)
    header = format_header("work", model)
    clearance = "full" if is_local_model(model) else "public"
    postfix = get_postfix(model)

    # Allocate a run_id so the agent can emit PersonaGuidance into the ledger
    # for the post-run hook to deliver via persona_prose.
    import uuid

    run_id = f"run_{uuid.uuid4().hex[:16]}"

    # Spawn the compiled agent in-process via opencode (settings.agents_dir).
    # spawn_agent writes the ledger run row first so the post-run persona_prose
    # hook can read this run's guidance back by run_id. The FREN_* context is
    # exported to the agent's environment exactly as v3's subprocess did.
    result = await spawn_agent(
        agent=agent,
        prompt=prompt,
        run_id=run_id,
        model_postfix=postfix,
        header=header,
        content_class=content_class,
        clearance=clearance,
        tts_postfix=_tts_postfix(),
        timeout_s=1800,
        trigger="chatbot",
    )
    if not result.ok and not _shutting_down:
        logger.error("Agent failed: %s", result.error)
        await _send_error_notification("orchestrator error")

    # Post-run persona_prose delivery (no-op if agent didn't emit guidance).
    await _post_run_persona_delivery(agent, run_id)


async def _update_personality_core_background(message: str, history_context: str, now_str: str) -> None:
    """Fire-and-forget wrapper: run personality_core evaluation and write
    the result to the emotional_state table. Discards the prompt string
    return value because the planner agent no longer consumes it —
    persona_prose reads from the DB directly via fetch_chat_context.

    Called from trigger_chat_agent via asyncio.create_task so the agent
    subprocess starts immediately instead of waiting 5-15s for the
    personality_core LLM round-trip. By the time persona_prose actually
    queries EmotionalStateRepo.get_current() (15-120s later), this
    background call has long since written its row.
    """
    try:
        await _evaluate_personality_core(message, history_context, now_str)
    except Exception:
        logger.exception("Background personality_core evaluation failed")


async def _evaluate_personality_core(message: str, history_context: str, now_str: str) -> str:
    """Call personality core model to evaluate emotional state. Returns prompt section to append.

    Note: trigger_chat_agent no longer awaits this directly — it fires
    _update_personality_core_background() instead so the agent subprocess
    isn't blocked on the 5-15s LLM round-trip. This function stays in
    place for manual / debug / admin callers that want the prompt string.
    """
    import json

    import httpx

    from app.settings import get_settings
    from app.db.repos.emotional_state import EmotionalStateRepo
    from app.tools.personality.personality_core import (
        DEFAULT_STATE,
        SYSTEM_PROMPT,
        _extract_tag,
        _parse_emotions,
    )

    repo = EmotionalStateRepo()
    settings = get_settings()

    # Check cooldown
    current = await repo.get_current()
    if current:
        from datetime import UTC, datetime

        created = current["created_at"]
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        if (datetime.now(UTC) - created).total_seconds() < 10:
            # Return cached state
            emotions = current.get("emotions", [])
            emo_summary = ", ".join(f"{e['name']}({e['intensity']})" for e in emotions if isinstance(e, dict))
            return (
                "\n\n## Your current emotional state (personality core):\n"
                "NOTE: This is a general vibe, not literal fact. Evaluate with current context "
                "before referencing — don't treat emotion names or intensities as strict values. "
                "May contain hallucinated facts or memories — use only the emotional tone, not specific claims.\n"
                f"Emotions: {emo_summary}\n"
                f"Feeling: {current.get('description', '')}\n"
                f"Guidance: {current.get('response_guidance', '')}\n"
                "(cached — evaluated recently)"
            )

    # Build internal state
    if current:
        internal_state = {
            "emotions": current.get("emotions", DEFAULT_STATE["emotions"]),
            "description": current.get("description", DEFAULT_STATE["description"]),
        }
    else:
        internal_state = DEFAULT_STATE

    # Build stimuli
    stimuli_lines = [f"[CHAT | telegram | {now_str}]"]
    if history_context:
        for line in history_context.split("\n")[-5:]:
            stimuli_lines.append(line)
    stimuli_lines.append(f"[user {now_str[-5:]}] {message}")
    stimuli = "\n".join(stimuli_lines)

    # Call vLLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"INTERNAL STATE:\n{json.dumps(internal_state)}\n\nEXTERNAL STIMULI:\n{stimuli}"},
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"http://{settings.personality_core_host}/v1/chat/completions",
            json={"model": "personality-core", "messages": messages, "max_tokens": 2048, "temperature": 0.7},
        )
        resp.raise_for_status()
        raw_xml = resp.json()["choices"][0]["message"]["content"]

    # Parse
    emotions = _parse_emotions(raw_xml)
    description = _extract_tag(raw_xml, "description")
    response_guidance = _extract_tag(raw_xml, "response_guidance")
    if not emotions:
        emotions = DEFAULT_STATE["emotions"]

    # Save to DB
    await repo.save(
        emotions=emotions,
        description=description,
        chain_of_thought=_extract_tag(raw_xml, "chain_of_thought"),
        mood_shift=_extract_tag(raw_xml, "mood_shift"),
        response_guidance=response_guidance,
        private_thoughts=_extract_tag(raw_xml, "private_thoughts"),
        stimuli_summary=stimuli[:500],
        raw_xml=raw_xml,
    )

    # Build prompt section
    emo_summary = ", ".join(f"{e['name']}({e['intensity']})" for e in emotions if isinstance(e, dict))
    lines = [
        "## Your current emotional state (personality core):",
        "NOTE: This is a general vibe, not literal fact. Evaluate with current context "
        "before referencing — don't treat emotion names or intensities as strict values. "
        "May contain hallucinated facts or memories — use only the emotional tone, not specific claims.",
        f"Emotions: {emo_summary}",
    ]
    if description:
        lines.append(f"Feeling: {description}")
    if response_guidance:
        lines.append(f"Guidance: {response_guidance}")
    return "\n\n" + "\n".join(lines)


async def trigger_chat_agent(
    message: str,
    username: str = "user",
    image_path: str | None = None,
    model: str | None = None,
    content_class: str = "public",
) -> None:
    """Run the lightweight chat agent for /chat mode."""
    from app.db.repos.chat import ChatMessagesRepo
    from app.telegram.state import format_header, get_model, is_local_model

    settings = get_settings()
    project_root = settings.project_root

    if model is None:
        model = get_model()

    clearance = "full" if is_local_model(model) else "public"

    # Phase-1 direct-path branch removed in plan stage S16. All in-scope
    # agents now emit PersonaGuidance; the compiled-agent path runs, then
    # the post-run hook (below) reads the ledger and hands off to
    # persona_prose.generate_persona_message for delivery.

    # Fetch last 24h chat history (filtered by clearance)
    try:
        # Fetch an extended window (base + chunk) and anchor the oldest included
        # message to a chunk-aligned id boundary. This keeps the FIRST row of the
        # history block byte-stable for ~CHUNK_SIZE consecutive new messages, so
        # vLLM's prefix cache can reuse most of the history prefix across turns
        # instead of rolling by one message every turn.
        BASE_HISTORY = 30
        HISTORY_CHUNK = 30
        history = await ChatMessagesRepo().get_history(days=1, limit=BASE_HISTORY + HISTORY_CHUNK, clearance=clearance)
        # Guarantee the last 5 messages are always included (in case the extended
        # window somehow dropped them — defensive).
        recent_5 = await ChatMessagesRepo().get_recent(limit=5, clearance=clearance)
        recent_5.reverse()  # get_recent returns DESC, need ASC
        history_ids = {m.get("id") for m in history}
        for m in recent_5:
            if m.get("id") not in history_ids:
                history.append(m)
        if history:
            newest_id = max((m.get("id") or 0) for m in history)
            if newest_id:
                window_floor = (newest_id // HISTORY_CHUNK) * HISTORY_CHUNK - BASE_HISTORY
                history = [m for m in history if (m.get("id") or 0) >= window_floor]
        history_lines = []
        for m in history:
            ts = str(m.get("timestamp", ""))[:16]  # "2026-02-19 17:03"
            sender = m.get("sender", "?")
            text = str(m.get("message", ""))[:300]
            history_lines.append(f"[{ts}] [{sender}]: {text}")
        history_context = "\n".join(history_lines)
    except Exception:
        history_context = ""

    # Fetch pinned context for current discussion topic
    try:
        from app.db.repos.context_pins import ContextPinsRepo

        current_ctx = await ContextPinsRepo().get_current_context(message_limit=0)
        if current_ctx and current_ctx.get("topic"):
            topic = current_ctx["topic"]
            pin_lines = [f"## Active discussion: {topic.get('topic_name', 'Unknown')}"]
            if topic.get("topic_summary"):
                pin_lines.append(f"Summary: {topic['topic_summary']}")
            pins = current_ctx.get("pins", [])
            if pins:
                pin_lines.append("### Pinned context:")
                for p in pins[:10]:
                    ptype = p.get("content_type", "note")
                    pcontent = str(p.get("content", ""))[:500]
                    pin_lines.append(f"- [{ptype}] {pcontent}")
            doc_refs = current_ctx.get("document_refs", [])
            if doc_refs:
                pin_lines.append("### Referenced documents:")
                for d in doc_refs[:5]:
                    pin_lines.append(f"- doc:{d.get('document_id', '')} — {d.get('reference_reason', '')[:200]}")
            pinned_context = "\n".join(pin_lines)
        else:
            pinned_context = ""
    except Exception:
        pinned_context = ""

    # Fetch recent context cache (last 24h, filtered by clearance)
    try:
        from app.db.repos.context_cache import ContextCacheRepo

        cache_cls = "full" if clearance == "full" else "public"
        repo = ContextCacheRepo()
        recent_cache = await repo.list_recent(hours=24, limit=30, content_class=cache_cls)

        # Always include the 2 most recent daily summaries (today + yesterday),
        # even if they'd otherwise be pushed out by other items.
        summaries_48h = await repo.list_by_type("activity_daily_summary", hours=48, limit=2)

        # Filter: keep last 6 activity_observations (~30 min), cap total at 20 + guaranteed summaries
        obs_count = 0
        max_observations = 48  # 5-min intervals x 48 = 4h of activity history
        filtered_cache = []
        seen_ids: set[str] = set()
        # Prepend guaranteed summaries
        for s in summaries_48h:
            sid = s.get("cache_id", "")
            if sid not in seen_ids:
                filtered_cache.append(s)
                seen_ids.add(sid)
        # Fill rest from recent, keeping observations up to max
        for c in recent_cache:
            cid = c.get("cache_id", "")
            if cid in seen_ids:
                continue
            if c.get("artifact_type") == "activity_observation":
                obs_count += 1
                if obs_count > max_observations:
                    continue
            filtered_cache.append(c)
            seen_ids.add(cid)
            if len(filtered_cache) >= 60:
                break
        cache_lines = []
        for c in filtered_cache:
            ts = str(c.get("created_at", ""))[:16]
            atype = c.get("artifact_type", "?")
            summary = str(c.get("summary", ""))[:300]
            cid = c.get("cache_id", "")
            eid = c.get("entity_id", "")
            eid_part = f" entity_id={eid}" if eid else ""
            cache_lines.append(f"- [{ts}] {atype}:{eid_part} {summary} (ref: {cid})")
        cache_context = "\n".join(cache_lines)
    except Exception:
        cache_context = ""

    from datetime import datetime
    from zoneinfo import ZoneInfo

    now_str = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M")

    # Space before and after @path required for opencode to parse it as a file attachment.
    image_ref = f" @{image_path} \n\n" if image_path else ""
    quoted_msg = f'"{message}"' if message else ""

    # Prompt is assembled in THREE tiers to maximize vLLM prefix-cache hit
    # rate. Static tokens first, semi-static in the middle, per-turn volatile
    # content last. The current timestamp and user message land at the very
    # end so only the tail gets invalidated across requests — everything
    # above stays cacheable on GPU. See plans/refactored-forging-shore.md.
    static_sections: list[str] = []
    semi_static_sections: list[str] = []
    volatile_sections: list[str] = []

    # ── TIER 1: STATIC (cacheable across every request) ──
    static_sections.append(
        "⚠️ FINAL ACTION: Emit PersonaGuidance via `uv run scripts/emit_guidance.py --data '{...}'`. "
        "Your text output is INVISIBLE to the user — only emit_guidance.py lets persona_prose deliver "
        "the actual reply. Do NOT call send_message.py."
    )

    # Knowledge sheet — changes once a day at most, treat as static.
    try:
        from app.db.repos.user_config import UserConfigRepo

        ks = await UserConfigRepo().get("knowledge_sheet")
        if ks and ks.get("config_value"):
            static_sections.append(f"## User context (knowledge sheet):\n{ks['config_value']}")
    except Exception:
        pass

    # ── TIER 2: SEMI-STATIC (cacheable across turns, changes rarely) ──
    try:
        from app.db.repos.user_rules import UserRulesRepo

        rules_text = await UserRulesRepo().format_rules_prompt()
        if rules_text:
            semi_static_sections.append(rules_text)
    except Exception:
        pass

    try:
        from app.db.repos.agent_lessons import AgentLessonsRepo

        lessons_text = await AgentLessonsRepo().format_lessons_prompt()
        if lessons_text:
            semi_static_sections.append(lessons_text)
    except Exception:
        pass

    if pinned_context:
        semi_static_sections.append(pinned_context)

    # ── TIER 3: VOLATILE (changes per-turn or per-minute — tail of prompt) ──
    try:
        from app.db.repos.agent_notes import AgentNotesRepo

        repo = AgentNotesRepo()
        digest_note = await repo.get("conversation_digest")
        if digest_note and digest_note.get("note_value"):
            dval = digest_note["note_value"]
            digest_text = dval.get("digest", "") if isinstance(dval, dict) else str(dval)
            if digest_text:
                volatile_sections.append(digest_text)
    except Exception:
        pass

    try:
        from app.db.repos.memories import MemoriesRepo

        thoughts = await MemoriesRepo().search_by_tags(["inner_monologue"], limit=3)
        if thoughts:
            thought_lines = []
            for t in thoughts:
                ts = str(t.get("created_at", ""))[:16]
                title = str(t.get("title", ""))
                emotion = title.split(" — ")[0] if " — " in title else title
                content = str(t.get("content", ""))
                thought_lines.append(f"- [{ts}] ({emotion}) {content}")
            volatile_sections.append(
                "## Your recent inner thoughts (private — do NOT quote directly):\n" + "\n".join(thought_lines)
            )
    except Exception:
        pass

    try:
        from app.db.repos.ralf import RalfAmendmentsRepo, RalfProcessesRepo

        active = await RalfProcessesRepo().list_active()
        if active:
            amendments_repo = RalfAmendmentsRepo()
            ralf_lines = [
                "## Active Ralfs (multi-stage background tasks currently running)",
                "When the user's message relates to one of these, prefer folding it in via "
                "`ralf_manager add-amendment` instead of spawning a new ralf or claiming "
                "to start the work yourself. See 'Active Ralf handling' below for routing.",
            ]
            for r in active:
                rid = r.get("ralf_id") or ""
                task = (r.get("task_name") or "").strip()
                if not task:
                    ur = (r.get("user_request") or "").strip()
                    task = (ur[:80] + "…") if len(ur) > 80 else ur
                stage_n = int(r.get("current_stage") or 0)
                stage_total = int(r.get("total_stages") or 0)
                stage_str = f"step {stage_n}/{stage_total}" if stage_total else f"step {stage_n}"
                created = str(r.get("created_at") or "")[:16]
                unread = await amendments_repo.count_unread(rid) if rid else 0
                unread_str = f" · {unread} unread amendment{'s' if unread != 1 else ''}" if unread else ""
                ralf_lines.append(
                    f'- {rid} [{r.get("status", "")}] {stage_str} · "{task}" · started {created}{unread_str}'
                )
            volatile_sections.append("\n".join(ralf_lines))
    except Exception:
        logger.exception("active_ralfs injection failed")

    if cache_context:
        volatile_sections.append(
            "## Recent background activity (ref by cache_id via context_cache tool):\n"
            f"{cache_context}\n"
            "To get full content for a cached artifact, use the entity_id with the appropriate tool "
            "(e.g., `uv run scripts/research_manager.py --command get-video --video_id <entity_id>` for youtube)."
        )

    if history_context:
        volatile_sections.append(f"## Recent conversation (last 24h):\n{history_context}")

    # Timestamp lives at the tail so it only invalidates ~100 trailing tokens
    # per minute instead of the whole prompt.
    volatile_sections.append(
        "## AUTHORITATIVE CURRENT TIME\n"
        f"- Current local time in Europe/Warsaw RIGHT NOW: {now_str}\n"
        "- Use this as the source of truth for words like now, today, in X minutes, and time-until-event.\n"
        "- Do NOT infer the current time from chat-history timestamps; those are message times, not the live clock."
    )

    # New-message header and the user turn itself — the most volatile bit,
    # landing absolutely last.
    volatile_sections.append(f"## ⚡ NEW MESSAGE:\n{quoted_msg}")

    prompt = image_ref + "\n\n".join(static_sections + semi_static_sections + volatile_sections)

    # Fire personality_core as a background task — its output is consumed by
    # persona_prose later when it queries EmotionalStateRepo.get_current().
    # By the time the agent finishes and persona_prose runs (~15-120s later),
    # this background call (~5-15s) has written its row to the DB. The
    # planner agent itself doesn't use the emotional state for routing, so
    # we don't append the result to its prompt.
    try:
        _personality_core_bg_task = asyncio.create_task(
            _update_personality_core_background(message, history_context, now_str)
        )
        _background_personality_tasks.add(_personality_core_bg_task)
        _personality_core_bg_task.add_done_callback(_background_personality_tasks.discard)
    except Exception:
        logger.exception("Personality core background fire failed")

    # Detected intents — per-turn volatile; append just before the user
    # message so it lands in the tail of the prompt and doesn't invalidate
    # upstream cache.
    intents_section = _infer_intents(message)
    if intents_section:
        prompt += f"\n\n{intents_section}"

    # Brief terminal nudge — static, cheap, keeps the emit_guidance steer
    # fresh in the model's short-term window right before generation. Not
    # prefix-cacheable (it's after the volatile tail) but single-digit tokens.
    prompt += (
        "\n\n---REMINDER---\n"
        "FINAL action: ONE call to `uv run scripts/emit_guidance.py --data '{...}'`. "
        "Text output is invisible. Do NOT call send_message.py."
    )

    from app.telegram.state import get_postfix

    agent = _agent_with_postfix("persona/twily_chat", model)
    header = format_header("chat", model)
    postfix = get_postfix(model)

    # Allocate a run_id so the agent can emit PersonaGuidance into the ledger
    # and we can deliver it via persona_prose after the agent finishes. Passed
    # as FREN_RUN_ID; agents that haven't been converted yet simply ignore it
    # and keep calling send_message.py directly (no guidance artifact exists →
    # deliver_guidance_from_ledger returns False and is a no-op).
    import uuid

    run_id = f"run_{uuid.uuid4().hex[:16]}"

    # Spawn the compiled chat agent in-process via opencode. The run row is
    # written to the ledger first so the post-run hook can read guidance back
    # by run_id; FREN_* context is exported to the agent's environment.
    result = await spawn_agent(
        agent=agent,
        prompt=prompt,
        run_id=run_id,
        model_postfix=postfix,
        header=header,
        content_class=content_class,
        clearance=clearance,
        tts_postfix=_tts_postfix(),
        timeout_s=1800,
        trigger="chat",
    )
    if not result.ok and not _shutting_down:
        logger.error("Chat agent failed: %s", result.error)
        await _send_error_notification("chat agent error")

    # Post-run persona_prose delivery hook. No-op if the agent didn't emit
    # guidance (the agent still called send_message directly on the old path).
    await _post_run_persona_delivery(agent, run_id)


async def trigger_workflow(
    workflow_agent: str,
    message: str,
    username: str = "user",
    message_id: int = 0,
    model: str | None = None,
) -> None:
    """Run a workflow agent directly."""
    from app.telegram.state import format_header, get_mode, get_model

    settings = get_settings()
    project_root = settings.project_root

    if model is None:
        model = get_model()

    from app.telegram.state import get_postfix

    agent = _agent_with_postfix(workflow_agent, model)
    mode = get_mode()
    header = format_header(mode, model)
    postfix = get_postfix(model)

    # Prepend Telegram message ID so agents can use it for deduplication
    if message_id > 0:
        prompt = f"[TELEGRAM_MESSAGE_ID:{message_id}]\n\n{message}"
    else:
        prompt = message

    import uuid

    run_id = f"run_{uuid.uuid4().hex[:16]}"

    # Spawn the workflow agent in-process via opencode (ledger row first).
    result = await spawn_agent(
        agent=agent,
        prompt=prompt,
        run_id=run_id,
        model_postfix=postfix,
        header=header,
        tts_postfix=_tts_postfix(),
        timeout_s=1800,
        trigger="workflow",
    )
    if not result.ok and not _shutting_down:
        logger.error("Workflow failed: %s", result.error)
        await _send_error_notification("workflow error")

    # Post-run persona_prose delivery (no-op for unconverted agents).
    await _post_run_persona_delivery(agent, run_id)


async def trigger_video_analysis(url: str, original_message: str, model: str | None = None) -> None:
    """Ingest a YouTube video and run the video_analyst agent in the background."""
    from app.telegram.state import get_model, get_postfix

    if model is None:
        model = get_model()
    postfix = get_postfix(model)

    # Phase 1: Ingest — create DB record + fetch transcript. v3 shelled out to
    # `uv run scripts/youtube_fetcher.py --command ingest-url`; v4 runs the
    # same tool in-process via its async dispatch (no subprocess — the v4 image
    # has no `uv`). The tool's .execute() wraps asyncio.run(), which we can't
    # call from inside this running loop, so we await _dispatch directly.
    try:
        from app.tools.research.youtube_fetcher import (
            Input as YouTubeInput,
        )
        from app.tools.research.youtube_fetcher import (
            YouTubeFetcherTool,
        )

        out = await asyncio.wait_for(
            YouTubeFetcherTool()._dispatch(YouTubeInput(command="ingest-url", url=url)),
            timeout=120,
        )
        if not out.success:
            logger.error("YouTube ingest failed for %s: %s", url, out.error)
            return
        video_id = out.item.get("video_id", "")
        if not video_id:
            logger.error("YouTube ingest returned no video_id for %s: %r", url, out.item)
            return
    except TimeoutError:
        logger.error("YouTube ingest timed out after 120s for %s", url)
        return
    except Exception as e:
        logger.error("YouTube ingest failed for %s: %s", url, e)
        return

    # Phase 2: Analyze — run video_analyst agent
    agent = f"support/video_analyst{postfix}"
    prompt = (
        f"Analyze this YouTube video that the user shared.\n"
        f"Video ID: {video_id}\n"
        f"URL: {url}\n"
        f"User's message: {original_message}"
    )

    import uuid

    run_id = f"run_{uuid.uuid4().hex[:16]}"

    result = await spawn_agent(
        agent=agent,
        prompt=prompt,
        run_id=run_id,
        model_postfix=postfix,
        tts_postfix=_tts_postfix(),
        timeout_s=600,
        trigger="video_analysis",
    )
    if not result.ok and not _shutting_down:
        logger.error("Video analyst failed for %s: %s", url, result.error)

    # Post-run persona_prose delivery hook.
    await _post_run_persona_delivery(agent, run_id)


async def trigger_document_analysis(rel_path: str, filename: str, caption: str, model: str | None = None) -> None:
    """Parse a document and run the document_analyst agent in the background."""
    from app.telegram.state import get_model, get_postfix

    if model is None:
        model = get_model()
    postfix = get_postfix(model)

    # Phase 1: Parse — extract text + create DB record. v3 shelled out to
    # `uv run scripts/document_manager.py --command parse`; v4 runs the same
    # tool in-process via its async dispatch (no subprocess — the v4 image has
    # no `uv`). .execute() wraps asyncio.run(), unusable inside this running
    # loop, so we await _dispatch directly.
    try:
        from app.tools.research.document_manager import (
            DocumentManagerTool,
        )
        from app.tools.research.document_manager import (
            Input as DocumentInput,
        )

        out = await asyncio.wait_for(
            DocumentManagerTool()._dispatch(DocumentInput(command="parse", file_path=rel_path)),
            timeout=120,
        )
        if not out.success:
            logger.error("Document parse failed for %s: %s", filename, out.error)
            return
        doc_id = out.item.get("doc_id", "")
        if not doc_id:
            logger.error("Document parse returned no doc_id for %s: %r", filename, out.item)
            return
    except TimeoutError:
        logger.error("Document parse timed out after 120s for %s", filename)
        return
    except Exception as e:
        logger.error("Document parse failed for %s: %s", filename, e)
        return

    # Phase 2: Analyze — run document_analyst agent
    agent = f"support/document_analyst{postfix}"
    prompt = (
        f"Analyze this document that the user uploaded.\n"
        f"Document ID: {doc_id}\n"
        f"Filename: {filename}\n"
        f"User's caption: {caption}"
        if caption
        else f"Analyze this document that the user uploaded.\nDocument ID: {doc_id}\nFilename: {filename}"
    )

    import uuid

    run_id = f"run_{uuid.uuid4().hex[:16]}"

    result = await spawn_agent(
        agent=agent,
        prompt=prompt,
        run_id=run_id,
        model_postfix=postfix,
        tts_postfix=_tts_postfix(),
        timeout_s=600,
        trigger="document_analysis",
    )
    if not result.ok and not _shutting_down:
        logger.error("Document analyst failed for %s: %s", filename, result.error)

    await _post_run_persona_delivery(agent, run_id)


async def trigger_bug_report(prompt: str, model: str | None = None) -> None:
    """Run the bug_reporter agent to file a bug/feature report."""
    from app.telegram.state import get_model, get_postfix

    if model is None:
        model = get_model()
    postfix = get_postfix(model)

    agent = f"support/bug_reporter{postfix}"
    import uuid

    run_id = f"run_{uuid.uuid4().hex[:16]}"

    result = await spawn_agent(
        agent=agent,
        prompt=prompt,
        run_id=run_id,
        model_postfix=postfix,
        tts_postfix=_tts_postfix(),
        timeout_s=300,
        trigger="bug_report",
    )
    if not result.ok and not _shutting_down:
        logger.error("Bug reporter failed: %s", result.error)

    await _post_run_persona_delivery(agent, run_id)


# --- event-loop watchdog ----------------------------------------------------
# The bot froze (silent, zero getUpdates) twice on 2026-06-09/10 — an event-loop
# stall of unknown cause (init:true reaper did NOT prevent the recurrence). This
# self-heals it: an asyncio task touches a heartbeat file every 30s; a DAEMON
# THREAD (runs independently of the frozen loop) os._exit(1)s if the heartbeat
# goes stale, so Docker's `restart: unless-stopped` recovers the bot in seconds
# instead of a multi-hour silent outage. os._exit also dumps a py-spy stack (if
# available) first, so the next stall reveals WHERE it hangs.
_WATCHDOG_HB = Path("/tmp/bot_loop_hb")
_WATCHDOG_STALL_SECS = 180


async def _loop_heartbeat() -> None:
    import time as _time

    while True:
        try:
            _WATCHDOG_HB.write_text(str(_time.time()))
        except Exception:
            logger.exception("loop heartbeat write failed")
        await asyncio.sleep(30)


def _start_loop_watchdog() -> None:
    import threading
    import time as _time

    def _watch() -> None:
        _time.sleep(120)  # startup grace (compile/migrations)
        while True:
            _time.sleep(30)
            try:
                age = _time.time() - float(_WATCHDOG_HB.read_text())
            except Exception:
                age = 0.0  # file not written yet — never kill on a missing HB
            if age > _WATCHDOG_STALL_SECS:
                logger.error("event loop stalled %.0fs — dumping + exiting for restart", age)
                try:
                    import subprocess

                    subprocess.run(
                        ["py-spy", "dump", "--pid", str(os.getpid())],
                        timeout=20, check=False,
                    )
                except Exception:
                    pass
                os._exit(1)

    threading.Thread(target=_watch, daemon=True, name="loop-watchdog").start()


async def _post_init(_app: Application) -> None:
    global _scheduler
    # Self-healing watchdog: heartbeat the loop so a frozen-loop stall restarts.
    _app.create_task(_loop_heartbeat())
    # The dedicated `scheduler` service owns cron in the v4 split deployment.
    # Running the in-bot scheduler too would double-fire every job (per-container
    # state, same schedule.yml). Off by default; set RUN_INTERNAL_SCHEDULER=1 for
    # a single-process (v3-style) deployment with no scheduler service.
    if os.getenv("RUN_INTERNAL_SCHEDULER", "0") == "1":
        _scheduler = Scheduler()
        await _scheduler.start()
        logger.info("In-bot scheduler started (RUN_INTERNAL_SCHEDULER=1)")
    else:
        logger.info("In-bot scheduler disabled; dedicated scheduler service owns cron")


async def _post_shutdown(_app: Application) -> None:
    if _scheduler is not None:
        await _scheduler.stop()


def build_application() -> Application:
    """Build and configure the Telegram bot application."""
    settings = get_settings()

    builder = (
        Application.builder()
        .token(settings.bot_token)
        .concurrent_updates(True)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
    )

    # Use local Bot API server for large file support (up to 2GB)
    if settings.telegram_api_id and settings.telegram_api_hash:
        builder = (
            builder.base_url("http://localhost:8081/bot")
            .base_file_url("http://localhost:8081/file/bot")
            .local_mode(True)
        )
        logger.info("Using local Telegram Bot API server at localhost:8081")

    app = builder.build()

    # Import handlers
    from app.telegram.handlers import (
        handle_agents,
        handle_bug,
        handle_callback_query,
        handle_chat_mode,
        handle_claude_start,
        handle_claude_stop,
        handle_dense_switch,
        handle_document,
        handle_emotions,
        handle_feature,
        handle_glm_mode,
        handle_history,
        handle_local_mode,
        handle_message,
        handle_mode_status,
        handle_model_chat,
        handle_models,
        handle_moe_switch,
        handle_nsfw_mode,
        handle_photo,
        handle_processes,
        handle_sched_model,
        handle_sfw_mode,
        handle_small_switch,
        handle_split_switch,
        handle_start,
        handle_status,
        handle_sticker,
        handle_tts_model,
        handle_unknown_command,
        handle_vibe,
        handle_video,
        handle_voice,
        handle_work_mode,
        handle_workflows,
    )

    # Discover all trigger commands from agent definitions
    trigger_commands = _get_trigger_commands()

    # Built-in commands
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("status", handle_status))
    app.add_handler(CommandHandler("history", handle_history))
    app.add_handler(CommandHandler("workflows", handle_workflows))
    app.add_handler(CommandHandler("agents", handle_agents))
    app.add_handler(CommandHandler("processes", handle_processes))

    # Mode & model commands
    app.add_handler(CommandHandler("chat", handle_chat_mode))
    app.add_handler(CommandHandler("work", handle_work_mode))
    app.add_handler(CommandHandler("glm", handle_glm_mode))
    app.add_handler(CommandHandler("local", handle_local_mode))
    app.add_handler(CommandHandler("mode", handle_mode_status))
    app.add_handler(CommandHandler("models", handle_models))
    app.add_handler(CommandHandler("model_chat", handle_model_chat))
    app.add_handler(CommandHandler("sched_model", handle_sched_model))
    app.add_handler(CommandHandler("tts_model", handle_tts_model))
    app.add_handler(CommandHandler("moe", handle_moe_switch))
    app.add_handler(CommandHandler("dense", handle_dense_switch))
    app.add_handler(CommandHandler("small", handle_small_switch))
    app.add_handler(CommandHandler("split", handle_split_switch))

    # Content mode commands
    app.add_handler(CommandHandler("nsfw", handle_nsfw_mode))
    app.add_handler(CommandHandler("sfw", handle_sfw_mode))
    app.add_handler(CommandHandler("emotions", handle_emotions))
    app.add_handler(CommandHandler("vibe", handle_vibe))

    # Bug/feature reporting
    app.add_handler(CommandHandler("bug", handle_bug))
    app.add_handler(CommandHandler("feature", handle_feature))

    # Claude session management
    app.add_handler(CommandHandler("claude_start", handle_claude_start))
    app.add_handler(CommandHandler("claude_stop", handle_claude_stop))

    # Dynamic trigger command handlers (workflows, workflow_master, server, etc.)
    from app.telegram.handlers import make_workflow_handler

    for cmd_name, agent_path in trigger_commands.items():
        app.add_handler(CommandHandler(cmd_name, make_workflow_handler(agent_path)))

    # Callback query handler (inline keyboard buttons)
    app.add_handler(CallbackQueryHandler(handle_callback_query))

    # Unknown command handler (catch-all for unregistered /commands)
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))

    # Document handler (PDF, DOCX, TXT, CSV, etc.)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Photo handler
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Video handler (video + video_note/round videos)
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video))

    # Sticker handler
    app.add_handler(MessageHandler(filters.Sticker.ALL, handle_sticker))

    # Voice message handler (STT transcription)
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # General message handler (must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Error handler
    async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Unhandled exception: %s", context.error, exc_info=context.error)

    app.add_error_handler(_error_handler)

    return app


def run() -> None:
    """Start the Telegram bot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = get_settings()

    # Resolve PROJECT_ROOT to absolute path and set XDG_DATA_HOME
    # so all child processes (opencode_manager, scheduler jobs) inherit it.
    project_root = Path(settings.project_root).resolve()
    os.environ.setdefault("PROJECT_ROOT", str(project_root))
    os.environ["XDG_DATA_HOME"] = str(project_root / ".opencode" / "data")

    logger.info(
        "Starting Fren Telegram bot (chat_id=%s, root=%s)",
        settings.chat_id,
        project_root,
    )

    app = build_application()

    # Cleanup on shutdown. Sets _shutting_down so that in-flight agent spawns
    # being torn down with the process are NOT reported to the user as
    # "something went wrong" — they're expected during a normal restart / stop
    # cycle. The spawn_agent subprocesses are children of this process and exit
    # with it; nothing extra to terminate here.
    def cleanup(*_):
        global _shutting_down
        _shutting_down = True
        logger.info("Shutting down Fren bot...")

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # Durable self-heal: daemon thread (independent of the asyncio loop) restarts
    # the process if the event loop stalls (see _loop_heartbeat / _start_loop_watchdog).
    _start_loop_watchdog()

    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    run()
