"""Telegram bot handlers — message, command, callback, photo."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update

if TYPE_CHECKING:
    from collections.abc import Callable

    from telegram.ext import ContextTypes

from app.settings import get_settings

logger = logging.getLogger(__name__)

# Track multiselect state: {question_id: set(selected_indices)}
_multiselect_state: dict[str, set[int]] = {}

# Hold strong references to background tasks to prevent GC
_background_tasks: set[asyncio.Task] = set()

# YouTube URL pattern
_YT_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([\w-]{11})"
)

# ── Message debouncing ──
# Collects rapid-fire messages into a single agent dispatch.
# Latest message is primary; earlier ones become context.
_DEBOUNCE_SECONDS = 0.3
_pending_messages: list[dict] = []  # accumulated message data
_debounce_timer: asyncio.Task | None = None


def _fire_and_forget(coro) -> None:
    """Schedule a coroutine as a background task with proper reference tracking."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _active_agent_context() -> str:
    """Build a context note about currently running agents."""
    try:
        from scripts.stop_agents import get_running_agents  # TODO(v4-port): scripts.stop_agents not yet ported
        agents = get_running_agents()
    except Exception:
        # Running-agent awareness not ported to v4 yet — degrade to no context
        # rather than sinking the whole dispatch (the message must still reach
        # the agent).
        return ""
    if not agents:
        return ""
    names = [a.agent_name or "unknown" for a in agents]
    return (
        f"\n\n[SYSTEM: {len(names)} agent(s) currently running: {', '.join(names)}. "
        "Check what they are working on before responding — avoid duplicating work. "
        "If a previous agent is already handling this topic, skip it or add only what's new.]"
    )


# ── Deterministic Fast Path ──
# Maps high-confidence single intents directly to workflow agents,
# bypassing the orchestrator entirely. Preserves personality with pre-built acks.
_INTENT_TO_WORKFLOW: dict[str, str] = {
    "task_completion": "workflows/todo",
    "task_creation": "workflows/todo",
    "task_query": "workflows/todo_goals",
    "habit_completion": "workflows/habit",
    "habit_query": "workflows/habit",
    "goal_management": "workflows/goal",
    "scheduling": "workflows/cron_master",
    "calendar": "workflows/calendar",
    "email": "workflows/email",
    "home_automation": "workflows/server",
}

_FAST_ACK_MESSAGES: dict[str, list[str]] = {
    "workflows/todo": ["On it! *flips through task list*", "Let me handle that~", "Checking your tasks!"],
    "workflows/todo_goals": ["Looking at your tasks~", "Let me pull that up!"],
    "workflows/habit": ["Checking your habits! *sparkle*", "Updating that~"],
    "workflows/goal": ["Looking at your goals! *determined horn glow*", "Checking on that!"],
    "workflows/cron_master": ["Scheduling that! *sparkle*", "Setting that up~"],
    "workflows/calendar": ["Checking your calendar~", "Let me look at that!"],
    "workflows/email": ["Checking your email~", "On it!"],
    "workflows/server": ["Handling that! *taps horn*", "On it~!"],
    "_default": ["On it~!", "Give me a moment!", "Looking into that! *taps horn*"],
}

# Messages that are continuations and should NOT take the fast path
_CONTINUATION_RE = re.compile(
    r"^\s*(?:yes|no|yeah|nah|yep|nope|ok|sure|that one|this one|the first|the second)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def _try_fast_path(message_text: str, *, has_image: bool = False) -> str | None:
    """Check if message qualifies for deterministic fast-path routing.

    Returns the workflow agent path if fast-path applies, None otherwise.
    Guard conditions: single clear intent, no image, no continuation, no emotional signals.
    """
    if has_image:
        return None
    if _CONTINUATION_RE.match(message_text):
        return None
    # Skip if message is very short (likely greeting) or very long (likely complex)
    if len(message_text.strip()) < 5 or len(message_text.strip()) > 300:
        return None

    try:
        from app.tools.context.intent_inference import INTENT_PATTERNS, _normalize

        normalized = _normalize(message_text)
        matches: list[str] = []
        for pattern, intent_type, _desc in INTENT_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                matches.append(intent_type)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        if len(unique) != 1:
            return None  # Ambiguous or no match — let orchestrator handle
        intent = unique[0]
        return _INTENT_TO_WORKFLOW.get(intent)
    except Exception:
        logger.debug("Fast-path check failed", exc_info=True)
        return None


async def _send_fast_ack(workflow: str) -> None:
    """Send a deterministic personality-preserving ack via Telegram bot API."""
    import random

    from telegram import Bot

    settings = get_settings()
    ack_list = _FAST_ACK_MESSAGES.get(workflow, _FAST_ACK_MESSAGES["_default"])
    ack_text = random.choice(ack_list)
    try:
        bot = Bot(token=settings.bot_token)
        await bot.initialize()
        await bot.send_message(chat_id=settings.chat_id, text=ack_text)
    except Exception:
        logger.debug("Failed to send fast ack", exc_info=True)


async def _debounce_dispatch() -> None:
    """Wait for debounce window, then dispatch accumulated messages as one agent call."""
    global _pending_messages
    await asyncio.sleep(_DEBOUNCE_SECONDS)

    batch = _pending_messages.copy()
    _pending_messages.clear()
    if not batch:
        return

    try:
        # Use latest message's settings (model, mode, content_class may have changed)
        latest = batch[-1]
        message_text = latest["text"]
        mode = latest["mode"]
        model = latest["model"]
        content_class = latest["content_class"]
        username = latest["username"]
        image_path = latest.get("image_path")

        # Combine messages: previous become context, latest is primary
        if len(batch) > 1:
            context_lines = []
            for m in batch[:-1]:
                context_lines.append(f"- {m['text']}")
                # Use image_path from any message in the batch
                if m.get("image_path") and not image_path:
                    image_path = m["image_path"]
            message_text = (
                f"[{len(batch)} messages sent in quick succession. "
                f"Previous messages (context, already saved to chat history):\n"
                + "\n".join(context_lines)
                + f"\n\nRespond to the LATEST message:]\n\n{latest['text']}"
            )

        # Add running agent awareness
        message_text += _active_agent_context()

        # Handle YouTube links from all messages in batch
        for m in batch:
            if m.get("yt_url"):
                _fire_and_forget(_handle_youtube_link(m["yt_url"], m["text"], model))

        # Drift the vibe state on EVERY user turn, regardless of routing path.
        # Classifies tone of the latest user message, nudges palette blend via EMA.
        _fire_and_forget(_drift_vibe_state())
        # Drift the USER emotion state too, off the SAME deterministic palette
        # classification (no extra LLM call) — otherwise user_mood never
        # updates from its default row.
        _fire_and_forget(_drift_user_mood(latest["text"]))
        # Run the emotional-core model (:5506) on the latest message so the
        # `emotional_state` snapshot persona_prose injects ("## Current emotional
        # state") is FRESH. v3 parity: bot._update_personality_core_background.
        # Fire-and-forget — the LLM round-trip must not block dispatch; the tool
        # has its own ~10s cooldown.
        _fire_and_forget(_evaluate_personality_core(latest["text"]))

        # ── Fast-path: bypass orchestrator for high-confidence single intents ──
        if mode != "chat" and len(batch) == 1 and not latest.get("yt_url"):
            fast_workflow = _try_fast_path(latest["text"], has_image=bool(image_path))
            if fast_workflow:
                from app.telegram.bot import trigger_workflow

                logger.info("Fast-path routing: %s → %s", latest["text"][:60], fast_workflow)
                _fire_and_forget(_send_fast_ack(fast_workflow))
                _fire_and_forget(
                    trigger_workflow(
                        fast_workflow,
                        message_text,
                        message_id=latest.get("message_id", 0),
                        model=model,
                    )
                )
                return

        # Dispatch single agent
        if mode == "chat":
            from app.telegram.bot import trigger_chat_agent

            _fire_and_forget(
                trigger_chat_agent(
                    message_text, username=username, model=model, image_path=image_path, content_class=content_class
                )
            )
        else:
            from app.telegram.bot import trigger_chatbot

            _fire_and_forget(
                trigger_chatbot(
                    message_text, username=username, model=model, image_path=image_path, content_class=content_class
                )
            )
    except Exception:
        logger.exception("Debounce dispatch failed")


async def _reply_process_count(update: Update) -> None:
    """Send a quick reply showing how many agents are currently running."""
    try:
        from app.telegram.bot import _active_processes
    except Exception:
        # Process-count tracking not ported to v4 yet — skip the count reply
        # rather than raising in the message handler.
        return
    count = len(_active_processes)
    if count > 0:
        await update.effective_message.reply_text(f"({count} running)")  # type: ignore[union-attr]


def _is_allowed(update: Update) -> bool:
    """Check if the message is from the allowed chat."""
    settings = get_settings()
    if not settings.chat_id:
        return True
    return str(update.effective_chat.id) == str(settings.chat_id)  # type: ignore[union-attr]


async def _drift_vibe_state() -> None:
    """Fire-and-forget: classify user tone + nudge palette blend via EMA.

    Runs after every user-message batch (any routing path), so vibe drifts
    on direct chat-mode AND orchestrator-routed messages alike.
    """
    try:
        import subprocess
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[3]
        # Detached subprocess — don't block, don't raise if it fails.
        subprocess.Popen(
            ["python", "scripts/vibe_drift.py"],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        logger.exception("vibe_drift kickoff failed")


async def _drift_user_mood(user_msg: str) -> None:
    """Fire-and-forget: drift the user emotion state from the latest message.

    Rides the SAME deterministic regex trigger-palette classification the vibe
    blend uses (no extra LLM/GPU call), maps it to a 5-axis mood delta, and
    applies EMA drift via UserMoodRepo. Without this hook emotional_state stays
    pinned to its default row forever (the v3 vibe_drift.py user-mood port).
    """
    try:
        from app.db.repos.user_mood import UserMoodRepo
        from app.services.persona_palettes import (
            palette_to_mood_delta,
            select_trigger_palette,
        )

        if not (user_msg or "").strip():
            return
        palette = select_trigger_palette(user_msg)
        delta = palette_to_mood_delta(palette)
        if not any(abs(v) > 0.01 for v in delta.values()):
            return
        # Single-user default: the configured chat_id, else 0 (matches the
        # user_mood_manager tool + dashboard convention).
        try:
            raw = get_settings().chat_id
            chat_id = int(raw) if raw else 0
        except (TypeError, ValueError):
            chat_id = 0
        await UserMoodRepo().drift(chat_id, delta, trigger=palette)
    except Exception:
        logger.exception("user_mood drift failed")


async def _evaluate_personality_core(user_msg: str) -> None:
    """Fire-and-forget: run the emotional-core model (:5506) on the latest user
    message so the `emotional_state` snapshot is fresh for the next persona_prose
    render. The PersonalityCoreTool 'evaluate' command calls the personality-core
    LLM with the message as stimuli, parses the emotions, and writes the
    emotional_state table (+ aggregates) — with a built-in ~10s cooldown. Run in
    a worker thread because the tool's execute() is sync (manages its own loop).
    """
    try:
        if not (user_msg or "").strip():
            return
        import asyncio

        from app.tools.personality.personality_core import Input, PersonalityCoreTool

        await asyncio.to_thread(
            PersonalityCoreTool().execute,
            Input(command="evaluate", stimuli=user_msg),
        )
    except Exception:
        logger.exception("personality_core evaluate failed")


async def _save_user_message(message: str, update: Update, *, content_class: str = "public") -> None:
    """Save incoming user message to chat history."""
    from app.db.repos.chat import ChatMessagesRepo

    sfw_summary = {
        "nsfw": "[private conversation]",
        "secret": "[confidential message]",
    }.get(content_class)

    now = datetime.now(UTC)
    user = update.effective_user
    await ChatMessagesRepo().save(
        sender="user",
        message=message,
        date=now.date(),
        timestamp=now,
        timestamp_unix=now.timestamp(),
        chat_id=str(update.effective_chat.id) if update.effective_chat else None,  # type: ignore[union-attr]
        message_id=update.effective_message.message_id if update.effective_message else None,
        username=user.username if user else None,
        content_class=content_class,
        sfw_summary=sfw_summary,
    )


async def _download_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """Download voice message from Telegram and save locally. Returns file path."""
    if not update.effective_message or not update.effective_message.voice:
        return None

    settings = get_settings()
    voice = update.effective_message.voice
    file = await context.bot.get_file(voice.file_id)

    # Save to data/telegram_voice/YYYY-MM-DD/
    date_str = datetime.now().strftime("%Y-%m-%d")
    voice_dir = Path(settings.project_root) / "data" / "telegram_voice" / date_str
    voice_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{int(time.time())}_{voice.file_unique_id}.ogg"
    filepath = voice_dir / filename

    await file.download_to_drive(str(filepath))
    logger.info("Saved voice: %s", filepath)
    return str(filepath)


async def _transcribe_voice(audio_path: str) -> dict:
    """Send audio to STT service for transcription. Returns {text, language}."""
    import httpx

    settings = get_settings()
    url = f"http://{settings.stt_host}/v1/audio/transcriptions"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(audio_path, "rb") as f:
                resp = await client.post(url, files={"file": ("audio.ogg", f, "audio/ogg")})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error("STT transcription failed: %s", e)
        return {"text": "", "language": "unknown"}


async def _translate_voice(audio_path: str) -> dict:
    """Send audio to STT service for translation to English. Returns {text, source_language}."""
    import httpx

    settings = get_settings()
    url = f"http://{settings.stt_host}/v1/audio/translations"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(audio_path, "rb") as f:
                resp = await client.post(url, files={"file": ("audio.ogg", f, "audio/ogg")})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error("STT translation failed: %s", e)
        return {"text": "", "source_language": "unknown"}


async def _postprocess_transcription(text: str, language: str, model: str | None = None) -> str:
    """Clean up transcription via stt_processor agent. Translate to English if not English."""
    import os

    if not text or not text.strip():
        return text

    # Skip LLM agent entirely if already English — no translation needed
    if language == "en":
        return text

    from app.telegram.state import get_content_mode, get_model, get_postfix

    # SFW: use glm-4.7 flash (coding plan) for translation
    # NSFW: use the current local model
    if get_content_mode() == "nsfw":
        postfix = get_postfix(model or get_model())
    else:
        postfix = get_postfix("glm47")

    settings = get_settings()
    project_root = Path(settings.project_root).resolve()
    xdg_data = str(project_root / ".opencode" / "data")

    agent = f"support/stt_processor{postfix}"

    # Build opencode run command — XDG_DATA_HOME ensures session is visible in web UI
    # TODO(v4-port): verify agent-spawn wiring — direct `opencode run`, points at
    # project_root/.opencode/data; align with v4 agents_dir/runner before going live.
    cmd = [
        "opencode",
        "run",
        "--agent",
        agent,
        text,
    ]

    env = {**os.environ, "XDG_DATA_HOME": xdg_data}

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

        if proc.returncode != 0:
            logger.error(
                "stt_processor agent failed (exit %d): %s",
                proc.returncode,
                stderr.decode()[:500],
            )
            return text

        content = stdout.decode().strip()
        # Strip markdown formatting the agent might wrap output in
        if content.startswith("```") and content.endswith("```"):
            content = content[3:].strip()
            if content.startswith("\n"):
                content = content[1:]
            if content.endswith("```"):
                content = content[:-3].strip()

        if content and len(content) > 3:
            logger.info("stt_processor: '%s' -> '%s'", text[:100], content[:100])
            return content

        logger.error("stt_processor returned empty/short output: '%s'", content)
    except TimeoutError:
        logger.error("stt_processor agent timed out after 60s")
    except Exception as e:
        logger.error("stt_processor agent failed: %s", e)
    return text


async def _download_and_save_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """Download photo from Telegram and save locally. Returns file path."""
    if not update.effective_message or not update.effective_message.photo:
        return None

    settings = get_settings()
    photo = update.effective_message.photo[-1]  # Highest resolution
    file = await context.bot.get_file(photo.file_id)

    # Save to data/telegram_images/YYYY-MM-DD/
    date_str = datetime.now().strftime("%Y-%m-%d")
    img_dir = Path(settings.project_root) / "data" / "telegram_images" / date_str
    img_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{int(time.time())}_{photo.file_unique_id}.jpg"
    filepath = img_dir / filename

    await file.download_to_drive(str(filepath))
    logger.info("Saved image: %s", filepath)

    rel_path = str(filepath.relative_to(settings.project_root))

    # Cache the received image artifact
    try:
        from app.db.repos.context_cache import add_to_cache

        await add_to_cache(
            "telegram_image",
            f"User sent image: {filename} (path: {rel_path})",
            file_path=rel_path,
            tags=["telegram", "image", "received"],
            source_agent="telegram_handler",
        )
    except Exception:
        pass

    # Return relative path from project root (opencode requires @relative/path for image import)
    return rel_path


async def _download_and_save_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """Download sticker from Telegram and save locally. Returns relative file path."""
    msg = update.effective_message
    if not msg or not msg.sticker:
        return None

    sticker = msg.sticker

    settings = get_settings()
    file = await context.bot.get_file(sticker.file_id)

    date_str = datetime.now().strftime("%Y-%m-%d")
    sticker_dir = Path(settings.project_root) / "data" / "telegram_images" / date_str
    sticker_dir.mkdir(parents=True, exist_ok=True)

    # Download original sticker
    if sticker.is_video:
        ext = "webm"
    elif sticker.is_animated:
        ext = "tgs"
    else:
        ext = "webp"

    raw_filename = f"{int(time.time())}_{sticker.file_unique_id}.{ext}"
    raw_filepath = sticker_dir / raw_filename
    await file.download_to_drive(str(raw_filepath))

    # Convert stickers for agent compatibility
    if not sticker.is_animated and not sticker.is_video:
        # Static webp → jpg
        try:
            from PIL import Image

            jpg_filename = f"{int(time.time())}_{sticker.file_unique_id}.jpg"
            jpg_filepath = sticker_dir / jpg_filename
            with Image.open(raw_filepath) as img:
                img.convert("RGB").save(jpg_filepath, "JPEG", quality=90)
            raw_filepath.unlink()
            filepath = jpg_filepath
        except Exception:
            logger.warning("Could not convert sticker to jpg, using webp")
            filepath = raw_filepath
    elif sticker.is_video:
        # Video webm → mp4
        try:
            import subprocess

            mp4_filename = f"{int(time.time())}_{sticker.file_unique_id}.mp4"
            mp4_filepath = sticker_dir / mp4_filename
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw_filepath), "-c:v", "libx264", "-an", str(mp4_filepath)],
                capture_output=True,
                timeout=30,
            )
            if mp4_filepath.exists():
                raw_filepath.unlink()
                filepath = mp4_filepath
            else:
                filepath = raw_filepath
        except Exception:
            logger.warning("Could not convert video sticker to mp4, using webm")
            filepath = raw_filepath
    else:
        # Animated tgs — keep as-is (no easy conversion)
        filepath = raw_filepath

    logger.info("Saved sticker: %s", filepath)

    rel_path = str(filepath.relative_to(settings.project_root))

    # Cache the sticker artifact
    try:
        from app.db.repos.context_cache import add_to_cache

        emoji = sticker.emoji or ""
        set_name = sticker.set_name or ""
        await add_to_cache(
            "telegram_sticker",
            f"User sent sticker: {emoji} from pack '{set_name}' (path: {rel_path})",
            file_path=rel_path,
            tags=["telegram", "sticker", "received"],
            source_agent="telegram_handler",
        )
    except Exception:
        pass

    return rel_path


async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle sticker messages."""
    if not _is_allowed(update):
        return

    sticker_path = await _download_and_save_sticker(update, context)
    if not sticker_path:
        return

    sticker = update.effective_message.sticker  # type: ignore[union-attr]
    emoji = sticker.emoji or "" if sticker else ""
    set_name = sticker.set_name or "" if sticker else ""

    # Build descriptive text for the sticker
    sticker_text = f"[sticker: {emoji}]"
    if set_name:
        sticker_text = f"[sticker: {emoji} from pack '{set_name}']"

    from app.db.repos.chat import ChatMessagesRepo
    from app.telegram.state import get_mode, get_model

    now = datetime.now(UTC)
    user = update.effective_user
    await ChatMessagesRepo().save(
        sender="user",
        message=sticker_text,
        date=now.date(),
        timestamp=now,
        timestamp_unix=now.timestamp(),
        chat_id=str(update.effective_chat.id) if update.effective_chat else None,  # type: ignore[union-attr]
        message_id=update.effective_message.message_id if update.effective_message else None,
        username=user.username if user else None,
        metadata=json.dumps({"sticker_path": sticker_path, "emoji": emoji, "set_name": set_name}),
    )

    # Debounced dispatch — stickers participate in message batching
    global _debounce_timer
    mode = get_mode()
    model = get_model()
    username = update.effective_user.username if update.effective_user else "user"

    _pending_messages.append(
        {
            "text": sticker_text,
            "mode": mode,
            "model": model,
            "content_class": "public",
            "username": username or "user",
            "image_path": sticker_path,
        }
    )

    if _debounce_timer and not _debounce_timer.done():
        _debounce_timer.cancel()
    _debounce_timer = asyncio.create_task(_debounce_dispatch())


# ── Mode & Model Command Handlers ──


async def handle_chat_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /chat command — switch to chat mode."""
    if not _is_allowed(update):
        return
    from app.telegram.state import format_header, set_mode

    set_mode("chat")
    header = format_header("chat")
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"Switched to chat mode! {header} ~Twily"
    )


async def handle_work_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /work command — switch to work mode."""
    if not _is_allowed(update):
        return
    from app.telegram.state import format_header, set_mode

    set_mode("work")
    header = format_header("work")
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"Switched to work mode! {header} ~Twily"
    )


async def handle_glm_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /glm command — switch to default GLM model."""
    if not _is_allowed(update):
        return
    from app.telegram.state import format_header, set_model

    set_model("glm")
    header = format_header()
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"Model: glm-4.5-air {header} ~Twily"
    )


async def handle_local_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /local command — switch to local vLLM model."""
    if not _is_allowed(update):
        return
    from app.telegram.state import format_header, get_default_local_model, get_model_display, set_model

    local_key = get_default_local_model()
    set_model(local_key)
    display = get_model_display(local_key)
    header = format_header()
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"Model: {display} (local) {header} ~Twily"
    )


async def handle_nsfw_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /nsfw command — enable NSFW content mode, force local model."""
    if not _is_allowed(update):
        return
    from app.telegram.state import format_header, get_default_local_model, is_local_model, set_content_mode, set_model

    set_content_mode("nsfw")
    # Force local model if not already local
    if not is_local_model():
        set_model(get_default_local_model())
    header = format_header()
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"NSFW mode enabled (local model only) {header} ~Twily"
    )


async def handle_sfw_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sfw command — return to SFW content mode."""
    if not _is_allowed(update):
        return
    from app.telegram.state import format_header, set_content_mode

    set_content_mode("sfw")
    header = format_header()
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"SFW mode restored {header} ~Twily"
    )


async def handle_emotions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /emotions command — toggle inner monologue on/off."""
    if not _is_allowed(update):
        return
    from app.telegram.state import get_emotions_enabled, set_emotions_enabled

    current = get_emotions_enabled()
    set_emotions_enabled(not current)
    status = "enabled" if not current else "disabled"
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"Inner monologue {status} ~Twily"
    )


async def handle_mode_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /mode command — show current mode + model + content mode."""
    if not _is_allowed(update):
        return
    from app.telegram.state import (
        format_header,
        get_content_mode,
        get_mode,
        get_model,
        get_model_display,
        get_scheduler_model,
        get_tts_model,
    )

    mode = get_mode()
    model = get_model()
    display = get_model_display(model)
    sched_model = get_scheduler_model()
    sched_display = get_model_display(sched_model)
    tts_model = get_tts_model()
    tts_display = get_model_display(tts_model)
    content = get_content_mode()
    header = format_header(mode, model)

    from app.telegram.state import get_vllm_display, get_vllm_variant

    vllm_variant = get_vllm_variant()
    vllm_display = get_vllm_display()
    vllm_line = f"vLLM (8082): {vllm_display}" if vllm_variant != "unknown" else "vLLM (8082): unknown"

    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"Mode: {mode}\nModel: {display}\nScheduler: {sched_display}\nTTS: {tts_display}\n{vllm_line}\nContent: {content}\n{header} ~Twily"
    )


async def handle_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /models command — list all available model hashtags."""
    if not _is_allowed(update):
        return
    from app.agents._config import _MODEL_PRESET_MAP, MODEL_DISPLAY, VARIANT_PRESETS

    lines = ["Available models (use #tag in messages):\n"]
    for key in sorted(VARIANT_PRESETS):
        display = MODEL_DISPLAY.get(key, key)
        tag = f"#{key}"
        preset = _MODEL_PRESET_MAP.get(key)
        port = preset.port if preset else "cloud"
        lines.append(f"  {tag}  →  {display}  [{port}]")

    from app.telegram.state import get_vllm_display, get_vllm_variant

    vllm_variant = get_vllm_variant()
    vllm_display = get_vllm_display()
    lines.append(f"\nvLLM Thinking (8082): {vllm_display} ← {'active' if vllm_variant != 'unknown' else '?'}")
    lines.append("  /dense  →  switch to 27B BF16")
    lines.append("  /moe    →  switch to 122B AWQ")

    await update.effective_message.reply_text("\n".join(lines))  # type: ignore[union-attr]


_VLLM_MODEL_MAP: dict[str, str] = {
    "dense": "localqwen3527b",
    "moe": "localqwen3527b",
    "small": "localqwen3527b",
    "split": "splitqwen35b",
}


async def _vllm_switch(update: Update, variant: str) -> None:
    """Switch vLLM model variant (shared logic for /moe, /dense, /split)."""
    from app.telegram.state import get_vllm_variant, set_model

    current = get_vllm_variant()
    if current == variant:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"Already running {variant}! Use /mode to check."
        )
        return

    # Switch active model tag to match the variant
    model_key = _VLLM_MODEL_MAP.get(variant, "localqwen3527b")
    set_model(model_key)

    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"Switching to {variant}... this will take a few minutes. I'll send updates."
    )

    # Run switcher as background subprocess
    _fire_and_forget(_run_vllm_switch(variant))


async def _run_vllm_switch(variant: str) -> None:
    """Background task: run the vLLM switch script."""
    settings = get_settings()
    project_root = settings.project_root
    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            "scripts/vllm_model_switch.py",
            "--command",
            "switch",
            "--variant",
            variant,
            cwd=str(project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=660)
    except TimeoutError:
        logger.error("vLLM switch to %s timed out after 660s", variant)
    except Exception:
        logger.exception("vLLM switch to %s failed", variant)


async def handle_moe_switch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /moe command — switch to 122B MoE model."""
    if not _is_allowed(update):
        return
    await _vllm_switch(update, "moe")


async def handle_dense_switch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /dense command — switch to 27B Dense model."""
    if not _is_allowed(update):
        return
    await _vllm_switch(update, "dense")


async def handle_small_switch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /small command — switch to MoE 35B-A3B heretic model."""
    if not _is_allowed(update):
        return
    await _vllm_switch(update, "small")


async def handle_split_switch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /split command — switch to split mode (MoE fast + Dense analytical)."""
    if not _is_allowed(update):
        return
    await _vllm_switch(update, "split")


async def handle_sched_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sched_model command — set model for scheduled/cron tasks.

    Usage: /sched_model #localqwen35 (or any model hashtag)
    Without args: show current scheduler model.
    """
    if not _is_allowed(update):
        return
    from app.telegram.state import (
        get_model_display,
        get_scheduler_model,
        parse_model_tag,
        set_scheduler_model,
    )

    args = update.effective_message.text.strip() if update.effective_message and update.effective_message.text else ""  # type: ignore[union-attr]
    # Strip the /sched_model prefix
    args = args.split(None, 1)[1] if len(args.split(None, 1)) > 1 else ""

    if not args:
        model = get_scheduler_model()
        display = get_model_display(model)
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"Scheduler model: {display}\nUse /sched_model #tag to change"
        )
        return

    tag_model = parse_model_tag(args)
    if tag_model:
        set_scheduler_model(tag_model)
        display = get_model_display(tag_model)
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"Scheduler model set to: {display} ~Twily"
        )
    else:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "Unknown model tag. Use /models to see available tags."
        )


async def handle_tts_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /tts_model command — set model for TTS formatting.

    Usage: /tts_model #localqwen35 (or any model hashtag)
    Without args: show current TTS model.
    """
    if not _is_allowed(update):
        return
    from app.telegram.state import (
        get_model_display,
        get_tts_model,
        parse_model_tag,
        set_tts_model,
    )

    args = update.effective_message.text.strip() if update.effective_message and update.effective_message.text else ""  # type: ignore[union-attr]
    args = args.split(None, 1)[1] if len(args.split(None, 1)) > 1 else ""

    if not args:
        model = get_tts_model()
        display = get_model_display(model)
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"TTS model: {display}\nUse /tts_model #tag to change"
        )
        return

    tag_model = parse_model_tag(args)
    if tag_model:
        set_tts_model(tag_model)
        display = get_model_display(tag_model)
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"TTS model set to: {display} ~Twily"
        )
    else:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "Unknown model tag. Use /models to see available tags."
        )


# ── Command Handlers ──


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    if not _is_allowed(update):
        return
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        "*taps horn excitedly* Hey there! I'm Twily! ~Twily\n\nUse /help to see what I can do!"
    )


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command."""
    if not _is_allowed(update):
        return
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        "*adjusts glasses* Systems operational! ~Twily"
    )


async def handle_vibe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /vibe command — render and send vibe state chart (deterministic, no agent)."""
    if not _is_allowed(update):
        return
    try:
        from app.telegram.vibe_chart import render_vibe_chart  # TODO(v4-port): app.telegram.vibe_chart not yet ported

        chat_id = update.effective_chat.id  # type: ignore[union-attr]
        image_path = await render_vibe_chart(chat_id)
        with open(image_path, "rb") as photo:
            await update.effective_message.reply_photo(photo=photo, caption="Twily's Vibe")  # type: ignore[union-attr]
    except Exception:
        logger.exception("Failed to render vibe chart")
        await update.effective_message.reply_text("Failed to render vibe chart.")  # type: ignore[union-attr]


async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unrecognized /commands — show error and list available commands."""
    if not _is_allowed(update):
        return

    message_text = update.effective_message.text or ""  # type: ignore[union-attr]
    cmd = message_text.split()[0] if message_text.strip() else "?"

    from app.telegram.bot import _get_trigger_commands

    trigger_commands = _get_trigger_commands()
    available = "\n".join(f"  /{c}" for c in sorted(trigger_commands))

    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"Unknown command: {cmd}\n\nAvailable commands:\n{available}"
    )


def make_workflow_handler(agent_path: str) -> Callable:
    """Create a handler function for a specific workflow agent."""

    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _is_allowed(update):
            return

        # Extract arguments after the command
        message_text = update.effective_message.text or ""  # type: ignore[union-attr]
        parts = message_text.split(maxsplit=1)
        cmd_name = parts[0].lstrip("/") if parts else agent_path.rsplit("/", 1)[-1]
        args = parts[1] if len(parts) > 1 else cmd_name

        # Save to chat history
        await _save_user_message(message_text, update)
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"<<{cmd_name}>> Starting..."
        )

        # Run workflow in background with current model
        from app.telegram.bot import trigger_workflow
        from app.telegram.state import get_model

        model = get_model()
        message_id = update.effective_message.message_id if update.effective_message else 0
        _fire_and_forget(trigger_workflow(agent_path, args, message_id=message_id, model=model))
        await _reply_process_count(update)

    return handler


# ── Message Handlers ──


async def handle_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history command — show recent chat history."""
    if not _is_allowed(update):
        return

    from app.db.repos.chat import ChatMessagesRepo

    repo = ChatMessagesRepo()
    msgs = await repo.get_history(days=1, limit=20)

    if not msgs:
        await update.effective_message.reply_text("No recent messages.")  # type: ignore[union-attr]
        return

    lines = []
    for m in msgs:
        ts = str(m.get("timestamp", ""))[:16]
        sender = m.get("sender", "?")
        text = str(m.get("message", ""))[:150]
        lines.append(f"[{ts}] {sender}: {text}")

    await update.effective_message.reply_text(  # type: ignore[union-attr]
        "\n".join(lines[-20:])
    )


async def handle_workflows(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /workflows command — list available workflow commands."""
    if not _is_allowed(update):
        return

    from app.telegram.bot import _get_trigger_commands

    commands = _get_trigger_commands()
    if not commands:
        await update.effective_message.reply_text("No workflows available.")  # type: ignore[union-attr]
        return

    lines = ["Available workflows:\n"]
    for cmd, agent_path in sorted(commands.items()):
        lines.append(f"/{cmd} - {agent_path}")

    await update.effective_message.reply_text("\n".join(lines))  # type: ignore[union-attr]


async def _process_user_response(message_text: str) -> None:
    """Process user message for task completions (background)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            "scripts/response_processor.py",
            "--command",
            "process-message",
            "--message",
            message_text,
            "--auto_complete",
            "true",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)
    except Exception as e:
        logger.debug("Response processing failed: %s", e)


def _extract_youtube_url(text: str) -> str | None:
    """Extract the first YouTube URL from text, or None."""
    m = _YT_URL_RE.search(text)
    return m.group(0) if m else None


async def _handle_youtube_link(url: str, message_text: str, model: str) -> None:
    """Background: ingest YouTube video and run analysis agent."""
    try:
        from app.telegram.bot import trigger_video_analysis

        await trigger_video_analysis(url, message_text, model=model)
    except Exception as e:
        logger.error("YouTube analysis failed for %s: %s", url, e)


def _resolve_content_class(content_tags: set[str]) -> str:
    """Determine content_class from parsed content tags and content_mode."""
    from app.telegram.state import get_content_mode

    if "secret" in content_tags:
        return "secret"
    if "nsfw" in content_tags or get_content_mode() == "nsfw":
        return "nsfw"
    return "public"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages."""
    if not _is_allowed(update):
        return

    message_text = update.effective_message.text or ""  # type: ignore[union-attr]
    if not message_text.strip():
        return

    # Parse model tags (#glm, #glm45, #glm47, #glm5, #local, #localglm45air, #localgptoss120b)
    from app.telegram.state import (
        get_default_local_model,
        get_mode,
        get_model,
        is_local_model,
        parse_content_tags,
        parse_model_tag,
        set_model,
        strip_content_tags,
        strip_model_tag,
    )

    tag_model = parse_model_tag(message_text)
    if tag_model:
        set_model(tag_model)
        message_text = strip_model_tag(message_text)
        if not message_text.strip():
            # Tag only — just confirm model switch
            from app.telegram.state import format_header, get_model_display

            display = get_model_display(tag_model)
            header = format_header()
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"Model: {display} {header} ~Twily"
            )
            return

    # Parse content tags (#nsfw, #secret)
    content_tags = parse_content_tags(message_text)
    message_text = strip_content_tags(message_text)
    content_class = _resolve_content_class(content_tags)

    # Force local model for non-public content
    if content_class != "public" and not is_local_model():
        set_model(get_default_local_model())

    # Save to chat history
    await _save_user_message(message_text, update, content_class=content_class)

    # Process user response in background (detect completions/acknowledgments)
    _fire_and_forget(_process_user_response(message_text))

    # Debounced dispatch — collect rapid-fire messages into a single agent call
    global _debounce_timer
    mode = get_mode()
    model = get_model()
    username = update.effective_user.username if update.effective_user else "user"
    yt_url = _extract_youtube_url(message_text)

    _pending_messages.append(
        {
            "text": message_text,
            "mode": mode,
            "model": model,
            "content_class": content_class,
            "username": username or "user",
            "yt_url": yt_url,
            "message_id": update.effective_message.message_id if update.effective_message else 0,
        }
    )

    # Cancel previous timer and start fresh
    if _debounce_timer and not _debounce_timer.done():
        _debounce_timer.cancel()
    _debounce_timer = asyncio.create_task(_debounce_dispatch())

    await _reply_process_count(update)


_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc", ".csv", ".md", ".log"}
_VIDEO_DOCUMENT_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document file uploads — parse and trigger background analysis.

    Also handles video files sent as documents (large videos that exceed Telegram's
    inline video size limit get sent as document attachments).
    """
    if not _is_allowed(update):
        return

    doc = update.effective_message.document  # type: ignore[union-attr]
    if not doc:
        return

    filename = doc.file_name or ""
    ext = Path(filename).suffix.lower() if filename else ""
    mime = doc.mime_type or ""

    # Check if this is a video sent as document
    is_video_doc = ext in _VIDEO_DOCUMENT_EXTENSIONS or mime.startswith("video/")
    if is_video_doc:
        await _handle_video_document(update, context, doc, filename, ext)
        return

    if ext not in _DOCUMENT_EXTENSIONS:
        # Not a supported document type — ignore silently
        return

    settings = get_settings()

    # Download file
    file = await context.bot.get_file(doc.file_id)
    date_str = datetime.now().strftime("%Y-%m-%d")
    doc_dir = Path(settings.project_root) / "data" / "telegram_documents" / date_str
    doc_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = f"{int(time.time())}_{doc.file_unique_id}_{filename}"
    filepath = doc_dir / safe_filename
    await file.download_to_drive(str(filepath))
    logger.info("Saved document: %s", filepath)

    rel_path = str(filepath.relative_to(settings.project_root))
    caption = update.effective_message.caption or ""  # type: ignore[union-attr]

    # Save to chat history
    await _save_user_message(caption or f"[document: {filename}]", update)

    # Phase 1 (sync): parse and extract text
    from app.telegram.state import get_model

    model = get_model()
    _fire_and_forget(_handle_document_upload(rel_path, filename, caption, model))


async def _handle_video_document(
    update: Update, context: ContextTypes.DEFAULT_TYPE, doc: object, filename: str, ext: str
) -> None:
    """Handle a video file sent as a document attachment (large videos)."""
    settings = get_settings()

    try:
        file = await context.bot.get_file(doc.file_id)  # type: ignore[attr-defined]
    except Exception as e:
        if "too big" in str(e).lower():
            size_mb = (doc.file_size / 1024 / 1024) if hasattr(doc, "file_size") and doc.file_size else 0  # type: ignore[attr-defined]
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"Video is too large ({size_mb:.0f}MB) — Bot API cloud limit is 20MB. "
                "Enable local Bot API server for files up to 2GB."
            )
        else:
            logger.error("Failed to download video document: %s", e)
        return

    # Save to data/telegram_videos/ (same location as regular videos)
    date_str = datetime.now().strftime("%Y-%m-%d")
    vid_dir = Path(settings.project_root) / "data" / "telegram_videos" / date_str
    vid_dir.mkdir(parents=True, exist_ok=True)

    safe_ext = ext.lstrip(".") or "mp4"
    safe_filename = f"{int(time.time())}_{doc.file_unique_id}.{safe_ext}"  # type: ignore[attr-defined]
    filepath = vid_dir / safe_filename
    await file.download_to_drive(str(filepath))
    logger.info("Saved video document: %s (%s)", filepath, filename)

    rel_path = str(filepath.relative_to(settings.project_root))

    # Cache artifact
    try:
        from app.db.repos.context_cache import add_to_cache

        await add_to_cache(
            "telegram_video",
            f"User sent video (as document): {filename} (path: {rel_path})",
            file_path=rel_path,
            tags=["telegram", "video", "received", "document"],
            source_agent="telegram_handler",
        )
    except Exception:
        pass

    caption = update.effective_message.caption or ""  # type: ignore[union-attr]

    # Reuse the same routing logic as handle_video
    from app.telegram.state import (
        get_default_local_model,
        get_mode,
        get_model,
        is_local_model,
        parse_content_tags,
        parse_model_tag,
        set_model,
        strip_content_tags,
        strip_model_tag,
    )

    tag_model = parse_model_tag(caption)
    if tag_model:
        set_model(tag_model)
        caption = strip_model_tag(caption)

    content_tags = parse_content_tags(caption)
    caption = strip_content_tags(caption)
    content_class = _resolve_content_class(content_tags)

    if content_class != "public" and not is_local_model():
        set_model(get_default_local_model())

    # Save to chat history
    from app.db.repos.chat import ChatMessagesRepo

    now = datetime.now(UTC)
    user = update.effective_user
    await ChatMessagesRepo().save(
        sender="user",
        message=caption or "[video]",
        date=now.date(),
        timestamp=now,
        timestamp_unix=now.timestamp(),
        chat_id=str(update.effective_chat.id) if update.effective_chat else None,  # type: ignore[union-attr]
        message_id=update.effective_message.message_id if update.effective_message else None,
        username=user.username if user else None,
        metadata=json.dumps({"video_path": rel_path}),
        content_class=content_class,
    )

    mode = get_mode()
    model = get_model()
    username = user.username if user else "user"

    if mode == "chat":
        from app.telegram.bot import trigger_chat_agent

        _fire_and_forget(
            trigger_chat_agent(
                caption,
                username=username or "user",
                image_path=rel_path,
                model=model,
                content_class=content_class,
            )
        )
    else:
        from app.telegram.bot import trigger_chatbot

        _fire_and_forget(
            trigger_chatbot(
                caption,
                username=username or "user",
                image_path=rel_path,
                model=model,
                content_class=content_class,
            )
        )

    await _reply_process_count(update)


async def _handle_document_upload(rel_path: str, filename: str, caption: str, model: str) -> None:
    """Background: parse document and trigger analysis agent."""
    try:
        from app.telegram.bot import trigger_document_analysis

        await trigger_document_analysis(rel_path, filename, caption, model=model)
    except Exception as e:
        logger.error("Document analysis failed for %s: %s", filename, e)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo messages."""
    if not _is_allowed(update):
        return

    # Download image
    image_path = await _download_and_save_image(update, context)
    if not image_path:
        return

    caption = update.effective_message.caption or ""  # type: ignore[union-attr]

    # Parse model tags from caption
    from app.telegram.state import (
        get_default_local_model,
        get_mode,
        get_model,
        is_local_model,
        parse_content_tags,
        parse_model_tag,
        set_model,
        strip_content_tags,
        strip_model_tag,
    )

    tag_model = parse_model_tag(caption)
    if tag_model:
        set_model(tag_model)
        caption = strip_model_tag(caption)

    # Parse content tags
    content_tags = parse_content_tags(caption)
    caption = strip_content_tags(caption)
    content_class = _resolve_content_class(content_tags)

    if content_class != "public" and not is_local_model():
        set_model(get_default_local_model())

    # Save to chat history with image metadata
    from app.db.repos.chat import ChatMessagesRepo

    sfw_summary = {
        "nsfw": "[private conversation]",
        "secret": "[confidential message]",
    }.get(content_class)

    now = datetime.now(UTC)
    user = update.effective_user
    await ChatMessagesRepo().save(
        sender="user",
        message=caption or "[photo]",
        date=now.date(),
        timestamp=now,
        timestamp_unix=now.timestamp(),
        chat_id=str(update.effective_chat.id) if update.effective_chat else None,  # type: ignore[union-attr]
        message_id=update.effective_message.message_id if update.effective_message else None,
        username=user.username if user else None,
        metadata=json.dumps({"image_path": image_path}),
        content_class=content_class,
        sfw_summary=sfw_summary,
    )

    # Check if caption is a workflow command
    if caption.startswith("/"):
        parts = caption.split(maxsplit=1)
        cmd = parts[0].lstrip("/")
        args = parts[1] if len(parts) > 1 else ""

        from app.telegram.bot import _get_trigger_commands

        trigger_commands = _get_trigger_commands()
        if cmd in trigger_commands:
            from app.telegram.bot import trigger_workflow

            model = get_model()
            message_id = update.effective_message.message_id if update.effective_message else 0
            _fire_and_forget(
                trigger_workflow(
                    trigger_commands[cmd],
                    f"@{image_path}\n\n{args}" if args else f"@{image_path}",
                    message_id=message_id,
                    model=model,
                )
            )
            return

    # Debounced dispatch — photos participate in message batching
    global _debounce_timer
    mode = get_mode()
    model = get_model()
    username = update.effective_user.username if update.effective_user else "user"

    _pending_messages.append(
        {
            "text": caption or "[photo]",
            "mode": mode,
            "model": model,
            "content_class": content_class,
            "username": username or "user",
            "image_path": image_path,
        }
    )

    if _debounce_timer and not _debounce_timer.done():
        _debounce_timer.cancel()
    _debounce_timer = asyncio.create_task(_debounce_dispatch())


async def _download_and_save_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """Download video from Telegram and save locally. Returns relative file path."""
    msg = update.effective_message
    if not msg:
        return None

    # Support both regular videos and video notes (round videos)
    video = msg.video or msg.video_note
    if not video:
        return None

    settings = get_settings()
    try:
        file = await context.bot.get_file(video.file_id)
    except Exception as e:
        if "too big" in str(e).lower():
            logger.warning("Video too large for Bot API download: %s", e)
            return None
        raise

    # Save to data/telegram_videos/YYYY-MM-DD/
    date_str = datetime.now().strftime("%Y-%m-%d")
    vid_dir = Path(settings.project_root) / "data" / "telegram_videos" / date_str
    vid_dir.mkdir(parents=True, exist_ok=True)

    ext = "mp4"
    if hasattr(video, "mime_type") and video.mime_type:
        ext = video.mime_type.split("/")[-1] if "/" in video.mime_type else "mp4"
    filename = f"{int(time.time())}_{video.file_unique_id}.{ext}"
    filepath = vid_dir / filename

    await file.download_to_drive(str(filepath))
    logger.info("Saved video: %s", filepath)

    rel_path = str(filepath.relative_to(settings.project_root))

    # Cache the received video artifact
    try:
        from app.db.repos.context_cache import add_to_cache

        duration = getattr(video, "duration", None)
        dur_str = f", duration: {duration}s" if duration else ""
        await add_to_cache(
            "telegram_video",
            f"User sent video: {filename} (path: {rel_path}{dur_str})",
            file_path=rel_path,
            tags=["telegram", "video", "received"],
            source_agent="telegram_handler",
        )
    except Exception:
        pass

    return rel_path


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle video and video_note messages."""
    if not _is_allowed(update):
        return

    video_path = await _download_and_save_video(update, context)
    if not video_path:
        # Notify user if video couldn't be downloaded (likely too large)
        msg = update.effective_message
        video = msg.video or msg.video_note if msg else None
        size_mb = (video.file_size / 1024 / 1024) if video and video.file_size else 0
        if size_mb > 20:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"Video is too large ({size_mb:.0f}MB) — Telegram Bot API limit is 20MB. Try a shorter or compressed video."
            )
        else:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "Couldn't download the video. Try sending it again or as a shorter clip."
            )
        return

    caption = update.effective_message.caption or ""  # type: ignore[union-attr]

    # Parse model tags from caption
    from app.telegram.state import (
        get_default_local_model,
        get_mode,
        get_model,
        is_local_model,
        parse_content_tags,
        parse_model_tag,
        set_model,
        strip_content_tags,
        strip_model_tag,
    )

    tag_model = parse_model_tag(caption)
    if tag_model:
        set_model(tag_model)
        caption = strip_model_tag(caption)

    content_tags = parse_content_tags(caption)
    caption = strip_content_tags(caption)
    content_class = _resolve_content_class(content_tags)

    if content_class != "public" and not is_local_model():
        set_model(get_default_local_model())

    # Save to chat history
    from app.db.repos.chat import ChatMessagesRepo

    sfw_summary = {
        "nsfw": "[private conversation]",
        "secret": "[confidential message]",
    }.get(content_class)

    now = datetime.now(UTC)
    user = update.effective_user
    await ChatMessagesRepo().save(
        sender="user",
        message=caption or "[video]",
        date=now.date(),
        timestamp=now,
        timestamp_unix=now.timestamp(),
        chat_id=str(update.effective_chat.id) if update.effective_chat else None,  # type: ignore[union-attr]
        message_id=update.effective_message.message_id if update.effective_message else None,
        username=user.username if user else None,
        metadata=json.dumps({"video_path": video_path}),
        content_class=content_class,
        sfw_summary=sfw_summary,
    )

    # Route based on mode — treat same as photo (pass video_path as image_path)
    mode = get_mode()
    model = get_model()
    username = update.effective_user.username if update.effective_user else "user"

    if mode == "chat":
        from app.telegram.bot import trigger_chat_agent

        _fire_and_forget(
            trigger_chat_agent(
                caption,
                username=username or "user",
                image_path=video_path,
                model=model,
                content_class=content_class,
            )
        )
    else:
        from app.telegram.bot import trigger_chatbot

        _fire_and_forget(
            trigger_chatbot(
                caption,
                username=username or "user",
                image_path=video_path,
                model=model,
                content_class=content_class,
            )
        )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages — transcribe via STT, translate if Polish, route as text."""
    if not _is_allowed(update):
        return

    # Download voice file
    audio_path = await _download_voice(update, context)
    if not audio_path:
        return

    # Transcribe
    result = await _transcribe_voice(audio_path)
    raw_text = result.get("text", "").strip()
    language = result.get("language", "unknown")

    if not raw_text:
        logger.warning("STT returned empty transcription for %s", audio_path)
        return

    logger.info("STT [%s]: %s", language, raw_text[:200])

    # Post-process: clean up + translate to English if not English
    message_text = await _postprocess_transcription(raw_text, language)

    # Reply with transcript so the user sees what was heard
    was_translated = language != "en" and language != "unknown"
    if was_translated:
        reply = f"[{language}] {raw_text}\n\n[en] {message_text}"
    else:
        reply = f"[{language}] {message_text}"
    # Telegram has a 4096-char limit; chunk if needed
    for i in range(0, len(reply), 4096):
        await update.effective_message.reply_text(reply[i : i + 4096])  # type: ignore[union-attr]

    # Parse model tags from transcription
    from app.telegram.state import (
        get_default_local_model,
        get_mode,
        get_model,
        is_local_model,
        parse_content_tags,
        parse_model_tag,
        set_model,
        strip_content_tags,
        strip_model_tag,
    )

    tag_model = parse_model_tag(message_text)
    if tag_model:
        set_model(tag_model)
        message_text = strip_model_tag(message_text)

    # Parse content tags
    content_tags = parse_content_tags(message_text)
    message_text = strip_content_tags(message_text)
    content_class = _resolve_content_class(content_tags)

    if content_class != "public" and not is_local_model():
        set_model(get_default_local_model())

    if not message_text.strip():
        return

    # Save to chat history with voice metadata
    from app.db.repos.chat import ChatMessagesRepo

    sfw_summary = {
        "nsfw": "[private conversation]",
        "secret": "[confidential message]",
    }.get(content_class)

    now = datetime.now(UTC)
    user = update.effective_user
    await ChatMessagesRepo().save(
        sender="user",
        message=message_text,
        date=now.date(),
        timestamp=now,
        timestamp_unix=now.timestamp(),
        chat_id=str(update.effective_chat.id) if update.effective_chat else None,  # type: ignore[union-attr]
        message_id=update.effective_message.message_id if update.effective_message else None,
        username=user.username if user else None,
        metadata=json.dumps({"voice_path": audio_path, "stt_language": language, "stt_raw": raw_text}),
        content_class=content_class,
        sfw_summary=sfw_summary,
    )

    # Process user response in background
    _fire_and_forget(_process_user_response(message_text))

    # Route based on mode (same as text messages)
    mode = get_mode()
    model = get_model()
    username = update.effective_user.username if update.effective_user else "user"

    if mode == "chat":
        from app.telegram.bot import trigger_chat_agent

        _fire_and_forget(
            trigger_chat_agent(message_text, username=username or "user", model=model, content_class=content_class)
        )
    else:
        from app.telegram.bot import trigger_chatbot

        _fire_and_forget(
            trigger_chatbot(message_text, username=username or "user", model=model, content_class=content_class)
        )


# ── Agent Management Handler ──


async def _handle_bug_or_feature(update: Update, context: ContextTypes.DEFAULT_TYPE, report_type: str) -> None:
    """Shared logic for /bug and /feature commands."""
    if not _is_allowed(update):
        return

    message_text = update.effective_message.text or ""  # type: ignore[union-attr]
    parts = message_text.split(maxsplit=1)
    description = parts[1] if len(parts) > 1 else ""

    if not description:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"Usage: /{report_type} <description>\n"
            f"Tip: Reply to a Twily message with /{report_type} to trace its session."
        )
        return

    # Build prompt
    prompt_parts = [f"Report type: {report_type}", f"Description: {description}"]

    # If replying to a message, extract timestamp and text for session correlation
    reply = update.effective_message.reply_to_message  # type: ignore[union-attr]
    if reply:
        ts = reply.date.timestamp() if reply.date else 0
        reply_text = reply.text or reply.caption or ""
        if not reply_text:
            # Media message without text — include type info for correlation
            media_type = "photo" if reply.photo else "video" if reply.video else "audio" if reply.audio else "media"
            reply_text = f"[{media_type} message, no caption, message_id={reply.message_id}]"
        prompt_parts.append(f"\n## Reply-to Context\nTimestamp: {ts}\nMessage: {reply_text[:1000]}")

    prompt = "\n".join(prompt_parts)

    # Save to chat history
    await _save_user_message(message_text, update)

    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"Filing {report_type} report..."
    )

    from app.telegram.bot import trigger_bug_report
    from app.telegram.state import get_model

    model = get_model()
    _fire_and_forget(trigger_bug_report(prompt, model=model))


async def _run_claude_session_on_host(action: str) -> str:
    """Run claude_session.sh on the host via Docker nsenter container.

    The bot runs in a container but screen/claude live on the host filesystem.
    We use the bind-mounted Docker socket to spawn a privileged one-shot
    container that nsenter's into PID 1's namespaces (the host init), then
    runs the script as the host user.
    """
    settings = get_settings()
    script = str(Path(settings.project_root) / "scripts" / "claude_session.sh")
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "run",
        "--rm",
        "--privileged",
        "--pid=host",
        "--network=host",
        "alpine:3",
        "nsenter",
        "--target",
        "1",
        "--mount",
        "--uts",
        "--ipc",
        "--net",
        "--pid",
        "--",
        "su",
        "-",
        "dw",
        "-c",
        f"{script} {action}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _stderr = await proc.communicate()
    return stdout.decode().strip()


async def handle_claude_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /claude_start — spin up a Claude session with Telegram plugin."""
    if not _is_allowed(update):
        return
    result = await _run_claude_session_on_host("start")
    if result == "ALREADY_RUNNING":
        await update.effective_message.reply_text("Claude session is already running.")  # type: ignore[union-attr]
    else:
        await update.effective_message.reply_text("Claude session started.")  # type: ignore[union-attr]


async def handle_claude_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /claude_stop — stop the Claude Telegram session."""
    if not _is_allowed(update):
        return
    result = await _run_claude_session_on_host("stop")
    if result == "NOT_RUNNING":
        await update.effective_message.reply_text("No Claude session is running.")  # type: ignore[union-attr]
    else:
        await update.effective_message.reply_text("Claude session stopped.")  # type: ignore[union-attr]


async def handle_bug(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /bug command — file a bug report."""
    await _handle_bug_or_feature(update, context, "bug")


async def handle_feature(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /feature command — file a feature request."""
    await _handle_bug_or_feature(update, context, "feature")


def _build_agents_message() -> tuple[str, InlineKeyboardMarkup | None]:
    """Build the agents list message and keyboard from system process scan."""
    from scripts.stop_agents import get_running_agents  # TODO(v4-port): scripts.stop_agents not yet ported

    agents = get_running_agents()

    if not agents:
        return "No running agents.", None

    lines = [f"Running {len(agents)} agent(s):\n"]
    keyboard = []
    for a in agents:
        name = a.agent_name or "unknown"
        lines.append(f"  {name} (PID {a.pid}, {a.process_type})")
        short_name = name.rsplit("/", 1)[-1][:20]
        keyboard.append([InlineKeyboardButton(f"Kill {short_name} (PID {a.pid})", callback_data=f"agent_kill_{a.pid}")])

    keyboard.append(
        [
            InlineKeyboardButton("Stop All", callback_data="agent_stop_all"),
            InlineKeyboardButton("Refresh", callback_data="agent_refresh"),
        ]
    )
    return "\n".join(lines), InlineKeyboardMarkup(keyboard)


async def handle_agents(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /agents command — list running agents with kill buttons."""
    if not _is_allowed(update):
        return

    text, markup = _build_agents_message()
    await update.effective_message.reply_text(text, reply_markup=markup)  # type: ignore[union-attr]


# /processes is an alias for /agents
handle_processes = handle_agents


# ── Callback Query Handler ──


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses (yes/no questions, multiselect)."""
    query = update.callback_query
    if not query or not query.data:
        return

    await query.answer()
    data = query.data

    # ── Agent management callbacks ──
    if data.startswith("agent_kill_"):
        pid = int(data.split("_", 2)[2])
        try:
            import os as _os
            import signal as _signal

            _os.kill(pid, _signal.SIGTERM)
            await query.edit_message_text(f"Sent SIGTERM to PID {pid}")
        except ProcessLookupError:
            await query.edit_message_text(f"PID {pid} already exited")
        except Exception as e:
            await query.edit_message_text(f"Failed to kill PID {pid}: {e}")
        return

    if data == "agent_stop_all":
        import os as _os
        import signal as _signal

        from app.telegram.bot import _active_processes

        found = len(_active_processes)
        stopped = 0
        for proc in list(_active_processes):
            try:
                _os.kill(proc.pid, _signal.SIGTERM)
                stopped += 1
            except Exception:
                pass
        await query.edit_message_text(f"Stopped {stopped}/{found} agent(s)")
        return

    if data == "agent_refresh":
        text, markup = _build_agents_message()
        with contextlib.suppress(Exception):
            await query.edit_message_text(text, reply_markup=markup)
        return

    # Parse callback data: q_YYYYMMDD_XXXXXXXX_action
    parts = data.split("_")
    if len(parts) < 4 or parts[0] != "q":
        return

    # Reconstruct question_id (q_YYYYMMDD_XXXXXXXX)
    question_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
    action = "_".join(parts[3:])

    # Yes/No
    if action in ("yes", "no"):
        # Record the answer
        from app.tools.telegram.question_sender import Input as QInput
        from app.tools.telegram.question_sender import QuestionSenderTool

        tool = QuestionSenderTool()
        await tool._dispatch(QInput(command="record-answer", question_id=question_id, answer=action.capitalize()))

        # Update the message to show the answer
        await query.edit_message_text(
            text=f"{query.message.text}\n\n✅ Answer: {action.capitalize()}"  # type: ignore[union-attr]
        )
        return

    # Multiselect toggle
    if action.startswith("toggle_"):
        idx = int(action.split("_")[1])
        state = _multiselect_state.setdefault(question_id, set())

        if idx in state:
            state.discard(idx)
        else:
            state.add(idx)

        # Rebuild keyboard with checkmarks
        if query.message and query.message.reply_markup:
            old_keyboard = query.message.reply_markup.inline_keyboard
            new_rows = []
            for i, row in enumerate(old_keyboard[:-1]):  # Skip "Done" button
                btn = row[0]
                text = btn.text.lstrip("✅ ")
                if i in state:
                    text = f"✅ {text}"
                new_rows.append([InlineKeyboardButton(text, callback_data=btn.callback_data)])  # type: ignore[arg-type]
            new_rows.append(old_keyboard[-1])  # Keep "Done" button
            await query.edit_message_reply_markup(InlineKeyboardMarkup(new_rows))
        return

    # Multiselect done
    if action == "done":
        state = _multiselect_state.pop(question_id, set())

        # Get option names from the keyboard
        selected_names = []
        if query.message and query.message.reply_markup:
            for i, row in enumerate(query.message.reply_markup.inline_keyboard[:-1]):
                if i in state:
                    selected_names.append(row[0].text.lstrip("✅ "))

        # Record answer
        from app.tools.telegram.question_sender import Input as QInput
        from app.tools.telegram.question_sender import QuestionSenderTool

        tool = QuestionSenderTool()
        await tool._dispatch(
            QInput(
                command="record-answer",
                question_id=question_id,
                answer=", ".join(selected_names) if selected_names else "None selected",
                selected=json.dumps(selected_names),
            )
        )

        await query.edit_message_text(
            text=f"{query.message.text}\n\n✅ Selected: {', '.join(selected_names) or 'None'}"  # type: ignore[union-attr]
        )
        return


async def handle_model_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """View or set the persona_prose model for the main chat bot.

    Distinct from the RP adventure `/model` command — this one controls the
    direct LLM call that turns PersonaGuidance from planner agents into Twily's
    final reply. Storage: user_config k/v keys persona_prose_provider +
    persona_prose_model.

    Usage:
      /model_chat              — list available models + active marker
      /model_chat <name>       — set override (bare name or provider/model)
      /model_chat default      — clear override, fall back to settings
    """
    if not _is_allowed(update):
        return
    try:
        from app.settings import get_settings
        from app.db.repos.user_config import UserConfigRepo
        from app.telegram import rp_prose  # import-only reuse — see rp-isolation.md

        text = update.effective_message.text or ""  # type: ignore[union-attr]
        parts = text.split(maxsplit=1)

        repo = UserConfigRepo()
        settings = get_settings()

        if len(parts) < 2:
            # Show available models + active marker
            override = await repo.get_persona_prose_override()
            if override is not None:
                override_prov, override_mod = override
            else:
                override_prov = ""
                override_mod = ""
            effective_prov = override_prov or settings.persona_prose_provider
            effective_mod = override_mod or settings.persona_prose_model
            active_target = f"{effective_prov}/{effective_mod}"

            available = rp_prose.list_available_models()
            lines: list[str] = [
                "*Chat Persona Models*",
                "",
                f"_Active: {active_target}_" + (" (override)" if override_prov or override_mod else " (from .env)"),
                "",
            ]
            current_provider = ""
            for entry in available:
                if entry["provider"] != current_provider:
                    current_provider = entry["provider"]
                    lines.append(f"*{current_provider}*")
                marker = "  • "
                full = f"{entry['provider']}/{entry['model']}"
                if full == active_target:
                    marker = "  ✓ "
                lines.append(f"{marker}{entry['model']}  ({entry['id']})")
            lines.append("")
            lines.append(
                "Use `/model_chat <name>` or `/model_chat provider/model` to switch. "
                "`/model_chat default` clears the override."
            )
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                "\n".join(lines), parse_mode="Markdown"
            )
            return

        arg = parts[1].strip()

        if arg.lower() in {"default", "reset", "clear", "none"}:
            await repo.clear_persona_prose_override()
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"Chat persona model override cleared. Falling back to "
                f"`{settings.persona_prose_provider}/{settings.persona_prose_model}` "
                f"(from settings).",
                parse_mode="Markdown",
            )
            return

        resolved = rp_prose.resolve_model_arg(arg)
        if not resolved:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"Unknown model: `{arg}`. Use /model_chat to see available options.",
                parse_mode="Markdown",
            )
            return
        provider, model = resolved
        await repo.set_persona_prose_override(provider, model)
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"Chat persona model set to: *{provider}/{model}*",
            parse_mode="Markdown",
        )
    except Exception:
        logger.exception("Failed to handle /model_chat")
        await update.effective_message.reply_text("Failed to handle /model_chat command.")  # type: ignore[union-attr]
