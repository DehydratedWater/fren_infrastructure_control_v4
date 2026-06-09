"""Explicit slash-command entry points (v3 parity) — registry + routing tests.

No live Telegram, DB, or agent spawns: python-telegram-bot is stubbed in
sys.modules (handlers.py only needs a few names at import time), Update/Message
are SimpleNamespace-style fakes, and the spawn seam (commands._spawn_workflow)
plus the handlers side-effect helpers are monkeypatched.
"""

from __future__ import annotations

import asyncio
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

# ── stub `telegram` before importing the handlers module ──────────────────────
import sys

try:  # pragma: no cover — real PTB present in the prod image
    import telegram  # noqa: F401
except ImportError:  # test venv has no python-telegram-bot
    _tg = types.ModuleType("telegram")

    class _Stub:  # placeholder for names handlers.py imports at module level
        def __init__(self, *args, **kwargs):
            pass

    _tg.Update = _Stub
    _tg.InlineKeyboardButton = _Stub
    _tg.InlineKeyboardMarkup = _Stub
    _tg.Bot = _Stub
    _ext = types.ModuleType("telegram.ext")
    _ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
    _tg.ext = _ext
    sys.modules["telegram"] = _tg
    sys.modules["telegram.ext"] = _ext

from app.telegram import commands, handlers  # noqa: E402

EXPECTED_COMMANDS = {
    "help", "brief", "memory", "analyse", "invoice",
    "techtree", "goal", "council", "adventure", "ralf",
}

COMMAND_TO_AGENT = {
    "brief": "support/daily_briefer",
    "analyse": "support/master_investigator",
    "techtree": "research/techtree_orchestrator",
    "goal": "goals/twily_goal_interface",
    "council": "workflows/council",
    "adventure": "rp/adventure_generator",
    "ralf": "workflows/twily_ralf_dispatcher",
}


# ── fakes ──────────────────────────────────────────────────────────────────────


class FakeMessage:
    def __init__(self, text="", caption=None, reply_to_message=None, photo=None, message_id=111):
        self.text = text
        self.caption = caption
        self.reply_to_message = reply_to_message
        self.photo = photo
        self.message_id = message_id
        self.replies: list[str] = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


def make_update(text="", **kwargs):
    msg = FakeMessage(text=text, **kwargs)
    return SimpleNamespace(
        effective_message=msg,
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(username="dw"),
        callback_query=None,
    )


@pytest.fixture(autouse=True)
def harness(monkeypatch):
    """Allow everything, silence side-effect helpers, record spawns/saves."""
    record = SimpleNamespace(spawns=[], saved=[], memories=[])

    monkeypatch.setattr(handlers, "_is_allowed", lambda update: True)

    async def _fake_save(message, update, **kwargs):
        record.saved.append(message)

    async def _fake_count(update):
        return None

    monkeypatch.setattr(handlers, "_save_user_message", _fake_save)
    monkeypatch.setattr(handlers, "_reply_process_count", _fake_count)

    async def _fake_spawn(agent_path, prompt, message_id):
        record.spawns.append((agent_path, prompt, message_id))

    monkeypatch.setattr(commands, "_spawn_workflow", _fake_spawn)
    monkeypatch.setattr(commands, "_store_memory_subprocess", record.memories.append)
    return record


def dispatch(update, context=None):
    """Run dispatch_slash_command and drain the fire-and-forget spawn tasks."""

    async def _run():
        handled = await commands.dispatch_slash_command(update, context or SimpleNamespace())
        for task in list(handlers._background_tasks):
            await task
        return handled

    return asyncio.run(_run())


def dispatch_photo(update, cmd, args, image_path, context=None):
    async def _run():
        handled = await commands.dispatch_photo_command(
            update, context or SimpleNamespace(), cmd, args, image_path
        )
        for task in list(handlers._background_tasks):
            await task
        return handled

    return asyncio.run(_run())


# ── registry ───────────────────────────────────────────────────────────────────


def test_every_lost_command_is_registered():
    assert EXPECTED_COMMANDS <= set(commands.SLASH_COMMANDS)
    for name, spec in commands.SLASH_COMMANDS.items():
        assert spec.description, f"/{name} is missing a description"
        assert callable(spec.handler)


def test_command_names_feed_the_unknown_command_listing():
    assert EXPECTED_COMMANDS <= commands.command_names()


def test_unregistered_command_falls_through(harness):
    update = make_update("/definitely_not_a_command")
    assert dispatch(update) is False
    assert update.effective_message.replies == []
    assert harness.spawns == []


# ── /help ──────────────────────────────────────────────────────────────────────


def test_help_lists_every_registered_command(harness):
    update = make_update("/help")
    assert dispatch(update) is True
    reply = "\n".join(update.effective_message.replies)
    for name in commands.SLASH_COMMANDS:
        assert f"/{name}" in reply, f"/help is missing /{name}"
    assert harness.spawns == []  # /help is deterministic, no agent


# ── /memory ────────────────────────────────────────────────────────────────────


def test_memory_without_args_replies_usage(harness):
    update = make_update("/memory")
    assert dispatch(update) is True
    assert any("Usage: /memory" in r for r in update.effective_message.replies)
    assert harness.memories == []
    assert harness.spawns == []


def test_memory_with_text_stores_via_subprocess_seam(harness):
    update = make_update("/memory the wifi password is in the drawer")
    assert dispatch(update) is True
    assert harness.memories == ["the wifi password is in the drawer"]
    assert any("memory" in r for r in update.effective_message.replies)
    assert harness.spawns == []  # deterministic tool path, no agent spawn


def test_memory_uses_replied_to_text_when_bare(harness):
    reply_msg = SimpleNamespace(text="remember this fact", caption=None, photo=None)
    update = make_update("/memory", reply_to_message=reply_msg)
    assert dispatch(update) is True
    assert harness.memories == ["remember this fact"]


# ── /analyse ───────────────────────────────────────────────────────────────────


def test_analyse_uses_replied_to_text(harness):
    reply_msg = SimpleNamespace(text="some suspicious log line", caption=None, photo=None)
    update = make_update("/analyse", reply_to_message=reply_msg)
    assert dispatch(update) is True
    assert len(harness.spawns) == 1
    agent, prompt, message_id = harness.spawns[0]
    assert agent == COMMAND_TO_AGENT["analyse"]
    assert "some suspicious log line" in prompt
    assert message_id == 111


def test_analyse_combines_args_and_reply(harness):
    reply_msg = SimpleNamespace(text="quoted content", caption=None, photo=None)
    update = make_update("/analyse why does this happen", reply_to_message=reply_msg)
    assert dispatch(update) is True
    _, prompt, _ = harness.spawns[0]
    assert "why does this happen" in prompt
    assert "quoted content" in prompt


def test_analyse_without_text_or_reply_replies_usage(harness):
    update = make_update("/analyse")
    assert dispatch(update) is True
    assert any("Usage: /analyse" in r for r in update.effective_message.replies)
    assert harness.spawns == []


# ── spawn mapping (thin routers fire the right agent via the ONE spawn seam) ──


@pytest.mark.parametrize("cmd", sorted(COMMAND_TO_AGENT))
def test_command_spawns_the_mapped_agent(harness, cmd):
    update = make_update(f"/{cmd} do the thing")
    assert dispatch(update) is True
    assert len(harness.spawns) == 1
    agent, prompt, _ = harness.spawns[0]
    assert agent == COMMAND_TO_AGENT[cmd]
    assert "do the thing" in prompt
    # ack went out immediately, make_workflow_handler style
    assert any("Starting" in r for r in update.effective_message.replies)
    # the raw command message was saved to chat history
    assert harness.saved and harness.saved[0].startswith(f"/{cmd}")


def test_goal_bare_shows_goals(harness):
    update = make_update("/goal")
    assert dispatch(update) is True
    agent, prompt, _ = harness.spawns[0]
    assert agent == COMMAND_TO_AGENT["goal"]
    assert "show my current goals" in prompt.lower()


def test_ralf_bare_is_status_mode(harness):
    update = make_update("/ralf")
    assert dispatch(update) is True
    agent, prompt, _ = harness.spawns[0]
    assert agent == COMMAND_TO_AGENT["ralf"]
    assert "status" in prompt.lower()


def test_command_with_bot_suffix_still_routes(harness):
    update = make_update("/brief@twily_bot")
    assert dispatch(update) is True
    assert harness.spawns[0][0] == COMMAND_TO_AGENT["brief"]


# ── /invoice ───────────────────────────────────────────────────────────────────


def test_invoice_without_photo_instructs_user(harness):
    update = make_update("/invoice")
    assert dispatch(update) is True
    assert any("caption /invoice" in r for r in update.effective_message.replies)
    assert harness.spawns == []


def test_invoice_replying_to_photo_routes_to_parser(harness, monkeypatch):
    async def _fake_download(update, context):
        return "data/telegram_images/2026-06-10/inv.jpg"

    monkeypatch.setattr(commands, "_download_reply_photo", _fake_download)
    reply_msg = SimpleNamespace(text=None, caption=None, photo=[SimpleNamespace(file_id="f")])
    update = make_update("/invoice groceries", reply_to_message=reply_msg)
    assert dispatch(update) is True
    agent, prompt, _ = harness.spawns[0]
    assert agent == "workflows/invoice_parser"
    assert prompt.startswith("@data/telegram_images/2026-06-10/inv.jpg")
    assert "groceries" in prompt


def test_invoice_photo_caption_routes_to_parser(harness):
    update = make_update("", caption="/invoice", photo=[SimpleNamespace(file_id="f")])
    handled = dispatch_photo(update, "invoice", "", "data/telegram_images/2026-06-10/x.jpg")
    assert handled is True
    agent, prompt, _ = harness.spawns[0]
    assert agent == "workflows/invoice_parser"
    assert prompt == "@data/telegram_images/2026-06-10/x.jpg"
    assert any("invoice" in r for r in update.effective_message.replies)


def test_photo_dispatch_ignores_other_commands(harness):
    update = make_update("", caption="/todo buy milk")
    assert dispatch_photo(update, "todo", "buy milk", "data/x.jpg") is False
    assert harness.spawns == []


# ── _is_allowed gate ───────────────────────────────────────────────────────────


def test_is_allowed_denial_short_circuits(harness, monkeypatch):
    monkeypatch.setattr(handlers, "_is_allowed", lambda update: False)
    update = make_update("/brief")
    # handled (swallowed) — but nothing replied, saved, stored, or spawned
    assert dispatch(update) is True
    assert update.effective_message.replies == []
    assert harness.spawns == []
    assert harness.saved == []
    assert harness.memories == []


def test_is_allowed_denial_short_circuits_photo_dispatch(harness, monkeypatch):
    monkeypatch.setattr(handlers, "_is_allowed", lambda update: False)
    update = make_update("", caption="/invoice")
    assert dispatch_photo(update, "invoice", "", "data/x.jpg") is True
    assert update.effective_message.replies == []
    assert harness.spawns == []


# ── integration: the catch-all handler routes through the registry ────────────


def test_handle_unknown_command_routes_registry_commands(harness):
    update = make_update("/brief")

    async def _run():
        await handlers.handle_unknown_command(update, SimpleNamespace())
        for task in list(handlers._background_tasks):
            await task

    asyncio.run(_run())
    assert harness.spawns and harness.spawns[0][0] == COMMAND_TO_AGENT["brief"]
    assert not any("Unknown command" in r for r in update.effective_message.replies)


def test_handler_exception_is_logged_not_raised(harness, monkeypatch):
    async def _boom(update, context, args):
        raise RuntimeError("kaput")

    monkeypatch.setitem(
        commands.SLASH_COMMANDS, "brief",
        commands.SlashCommand("boom", _boom),
    )
    update = make_update("/brief")
    assert dispatch(update) is True  # swallowed, not raised
    assert any("failed" in r for r in update.effective_message.replies)
