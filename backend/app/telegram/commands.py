"""Explicit slash-command entry points (v3 parity) — THIN routers only.

v3 registered these commands per-agent via `trigger_command` → CommandHandler.
The v4 AgentDefinition has no trigger_command field, so bot.py's
`_get_trigger_commands()` returns {} and the commands silently vanished, even
though every target agent/tool was ported. bot.py registers a catch-all
`MessageHandler(filters.COMMAND, handle_unknown_command)`; handlers.py routes
any command found in this registry through `dispatch_slash_command()` BEFORE
falling back to the unknown-command reply — so the commands are restored
without touching bot.py's registration block.

Rules (mirror the existing handlers):
* `_is_allowed` gate first, exactly like every handler in handlers.py.
* Strip the command, ack immediately, fire-and-forget the agent spawn via the
  SAME `trigger_workflow` path `make_workflow_handler` uses — no new spawn path.
* No business logic here. Exceptions are logged, never raised.
* /memory stores via the memory_manager ScriptTool as an ISOLATED subprocess
  from AGENTS_DIR (the personality_core pattern in handlers.py). Its sync
  execute() wraps asyncio.run() + engine teardown — running it via to_thread
  inside the bot loop corrupts the shared async DB engine. NEVER do that.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from telegram import Update
    from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# ── Command → agent targets (all exist in app/agents/domains/) ──
BRIEF_AGENT = "support/daily_briefer"  # domains/support.py
ANALYSE_AGENT = "support/master_investigator"  # domains/support.py (MASTER_INVESTIGATOR)
INVOICE_AGENT = "workflows/invoice_parser"  # domains/workflows.py (delegates OCR to support/invoice_image_parser)
TECHTREE_AGENT = "research/techtree_orchestrator"  # domains/research.py (TECHTREE_ORCHESTRATOR)
GOAL_AGENT = "goals/twily_goal_interface"  # domains/goals.py
COUNCIL_AGENT = "workflows/council"  # domains/workflows.py
ADVENTURE_AGENT = "rp/adventure_generator"  # domains/rp.py (ADVENTURE_GENERATOR — RP entry)
RALF_AGENT = "workflows/twily_ralf_dispatcher"  # domains/workflows.py (RALF_DISPATCHER — chain entry)


# ── Spawn seam ──
# Single indirection point so tests can monkeypatch `commands._spawn_workflow`
# without importing app.telegram.bot (which needs the real telegram package).


async def _spawn_workflow(agent_path: str, prompt: str, message_id: int) -> None:
    """Spawn one workflow/fleet agent — the exact path make_workflow_handler uses."""
    from app.telegram.bot import trigger_workflow
    from app.telegram.state import get_model

    await trigger_workflow(agent_path, prompt, message_id=message_id, model=get_model())


async def _route_workflow(update: Update, cmd_name: str, agent_path: str, prompt: str) -> None:
    """Save → ack → fire-and-forget spawn → process count (make_workflow_handler UX)."""
    from app.telegram import handlers as _h

    msg = update.effective_message
    message_text = (msg.text or msg.caption or "") if msg else ""
    await _h._save_user_message(message_text, update)
    await msg.reply_text(f"<<{cmd_name}>> Starting...")  # type: ignore[union-attr]

    message_id = msg.message_id if msg else 0
    _h._fire_and_forget(_spawn_workflow(agent_path, prompt, message_id))
    await _h._reply_process_count(update)


def _reply_text(update: Update) -> str:
    """Text (or caption) of the replied-to message, if any."""
    msg = update.effective_message
    reply = getattr(msg, "reply_to_message", None) if msg else None
    if not reply:
        return ""
    return (getattr(reply, "text", None) or getattr(reply, "caption", None) or "").strip()


# ── Command handlers (THIN routers) ──


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    """List the explicit slash commands — auto-generated from the registry."""
    lines = ["Commands:\n"]
    for name in sorted(SLASH_COMMANDS):
        lines.append(f"/{name} — {SLASH_COMMANDS[name].description}")
    lines.append("\nAlso: /workflows (workflow agents), /mode (model state), /agents (running agents).")
    await update.effective_message.reply_text("\n".join(lines))  # type: ignore[union-attr]


async def cmd_brief(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    """Spawn the daily briefer on demand."""
    prompt = args or "Generate the daily briefing now (on-demand /brief command from the user)."
    await _route_workflow(update, "brief", BRIEF_AGENT, prompt)


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    """Store an explicit memory via the memory_manager tool (isolated subprocess)."""
    text = args.strip()
    if not text:
        reply = _reply_text(update)
        if reply:
            text = reply
    if not text:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "Usage: /memory <text to remember>\n(or reply to a message with /memory)"
        )
        return

    from app.telegram import handlers as _h

    msg = update.effective_message
    await _h._save_user_message((msg.text or "") if msg else "", update)
    _store_memory_subprocess(text)
    title = text.splitlines()[0][:80]
    await msg.reply_text(f"<<memory>> Storing: {title}")  # type: ignore[union-attr]


def _store_memory_subprocess(text: str) -> None:
    """Create the memory via scripts/memory_manager.py as a detached subprocess.

    memory_manager is a sync DB-touching ScriptTool — execute() calls
    asyncio.run() and tears down the engine, so it must NEVER run in the bot
    loop (not even via to_thread; that corrupted the shared async engine).
    Same isolation pattern as handlers._evaluate_personality_core: run from
    AGENTS_DIR, where scripts/ is symlinked into the compiled fleet.
    """
    import subprocess

    from app.settings import get_settings

    title = text.splitlines()[0][:80]
    subprocess.Popen(  # noqa: S603 — fixed argv, user text passed as single args
        [
            "python", "scripts/memory_manager.py",
            "--command", "create",
            "--title", title,
            "--content", text[:4000],
            "--tags", "telegram,command",
            "--category", "telegram_command",
        ],
        cwd=str(get_settings().agents_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


async def cmd_analyse(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    """Spawn the investigator on the given text (or the replied-to message)."""
    reply = _reply_text(update)
    if not args and not reply:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "Usage: /analyse <topic or question>\n(or reply to a message with /analyse)"
        )
        return
    prompt = args or reply
    if args and reply:
        prompt = f"{args}\n\n## Quoted message (analyse this)\n{reply[:2000]}"
    await _route_workflow(update, "analyse", ANALYSE_AGENT, prompt)


async def cmd_invoice(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    """Route an invoice photo into the invoice parsing path.

    Text-command flow: works when replying to a photo. A photo WITH the
    caption /invoice never reaches here — handle_photo routes captions via
    dispatch_photo_command below.
    """
    image_path = await _download_reply_photo(update, context)
    if image_path:
        prompt = f"@{image_path}\n\n{args}" if args else f"@{image_path}"
        await _route_workflow(update, "invoice", INVOICE_AGENT, prompt)
        return
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        "Send the invoice photo with the caption /invoice (or reply to the photo with /invoice)."
    )


async def _download_reply_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """Download the photo of the replied-to message. Returns relative path or None.

    Mirrors handlers._download_and_save_image but reads reply_to_message.
    """
    msg = update.effective_message
    reply = getattr(msg, "reply_to_message", None) if msg else None
    photo = getattr(reply, "photo", None) if reply else None
    if not photo:
        return None
    try:
        from app.settings import get_settings

        settings = get_settings()
        file = await context.bot.get_file(photo[-1].file_id)

        date_str = datetime.now().strftime("%Y-%m-%d")
        img_dir = Path(settings.project_root) / "data" / "telegram_images" / date_str
        img_dir.mkdir(parents=True, exist_ok=True)
        filepath = img_dir / f"{int(time.time())}_{photo[-1].file_unique_id}.jpg"
        await file.download_to_drive(str(filepath))
        return str(filepath.relative_to(settings.project_root))
    except Exception:
        logger.exception("Failed to download replied-to photo for /invoice")
        return None


async def cmd_techtree(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    """Spawn the techtree orchestrator for an on-demand analysis."""
    prompt = args or "Run a full on-demand techtree analysis now (/techtree command from the user)."
    await _route_workflow(update, "techtree", TECHTREE_AGENT, prompt)


async def cmd_goal(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    """Route goal text to the goal interface agent; bare /goal shows goals."""
    prompt = args or "Show my current goals and their status."
    await _route_workflow(update, "goal", GOAL_AGENT, prompt)


async def cmd_council(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    """Convene the council of personas on a topic."""
    prompt = args or "Convene the council on my current goals and priorities."
    await _route_workflow(update, "council", COUNCIL_AGENT, prompt)


async def cmd_adventure(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    """Start a new RP adventure via the adventure generator (RP entry agent)."""
    prompt = args or "Create a new adventure and send me the intro."
    await _route_workflow(update, "adventure", ADVENTURE_AGENT, prompt)


async def cmd_ralf(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    """Hand a task to the RALF pipeline dispatcher (plan → review → execute)."""
    # Empty/short input is STATUS MODE by the dispatcher's own contract.
    prompt = args or "ralf status"
    await _route_workflow(update, "ralf", RALF_AGENT, prompt)


# ── Registry ──


@dataclass(frozen=True)
class SlashCommand:
    description: str
    handler: Callable[..., Awaitable[None]]


SLASH_COMMANDS: dict[str, SlashCommand] = {
    "help": SlashCommand("list available commands", cmd_help),
    "brief": SlashCommand("run the daily briefing now", cmd_brief),
    "memory": SlashCommand("store a memory: /memory <text> (or reply to a message)", cmd_memory),
    "analyse": SlashCommand("deep analysis of a topic: /analyse <text> (or reply)", cmd_analyse),
    "invoice": SlashCommand("parse an invoice photo (caption or reply with /invoice)", cmd_invoice),
    "techtree": SlashCommand("run an on-demand techtree analysis", cmd_techtree),
    "goal": SlashCommand("manage goals: /goal <text> (bare = show goals)", cmd_goal),
    "council": SlashCommand("convene the persona council: /council <topic>", cmd_council),
    "adventure": SlashCommand("start a new RP adventure", cmd_adventure),
    "ralf": SlashCommand("run a task through the RALF pipeline: /ralf <task>", cmd_ralf),
}


def command_names() -> set[str]:
    """Names of the explicit slash commands (for the unknown-command listing)."""
    return set(SLASH_COMMANDS)


def _parse_command(text: str) -> tuple[str, str]:
    """Split '/cmd@bot args…' → ('cmd', 'args…'). Empty cmd if not a command."""
    text = (text or "").strip()
    if not text.startswith("/"):
        return "", ""
    parts = text.split(maxsplit=1)
    cmd = parts[0][1:].split("@", 1)[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    return cmd, args


async def dispatch_slash_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Route a /command in this registry. Returns True if it was handled.

    Called from handlers.handle_unknown_command (the catch-all COMMAND
    handler), so anything not in the registry falls through to the existing
    unknown-command reply by returning False.
    """
    from app.telegram import handlers as _h

    msg = update.effective_message
    cmd, args = _parse_command((msg.text or msg.caption or "") if msg else "")
    spec = SLASH_COMMANDS.get(cmd)
    if spec is None:
        return False
    if not _h._is_allowed(update):
        return True  # known command from a disallowed chat — swallow silently
    try:
        await spec.handler(update, context, args)
    except Exception:
        # Exceptions are logged, never raised into the PTB dispatcher.
        logger.exception("/%s command failed", cmd)
        try:
            await msg.reply_text(f"/{cmd} failed — the error was logged.")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 — reply is best-effort
            pass
    return True


async def dispatch_photo_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    cmd: str,
    args: str,
    image_path: str,
) -> bool:
    """Route a photo whose caption is a registry command (handle_photo hook).

    Only /invoice consumes photos today; the photo is already downloaded and
    saved to chat history by handle_photo, so this just acks + spawns.
    """
    from app.telegram import handlers as _h

    if cmd.split("@", 1)[0].lower() != "invoice":
        return False
    if not _h._is_allowed(update):
        return True
    try:
        msg = update.effective_message
        await msg.reply_text("<<invoice>> Starting...")  # type: ignore[union-attr]
        prompt = f"@{image_path}\n\n{args}" if args else f"@{image_path}"
        message_id = msg.message_id if msg else 0
        _h._fire_and_forget(_spawn_workflow(INVOICE_AGENT, prompt, message_id))
        await _h._reply_process_count(update)
    except Exception:
        logger.exception("/invoice photo routing failed")
    return True
