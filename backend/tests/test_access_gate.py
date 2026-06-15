"""Telegram access gate — `_is_allowed` must FAIL-CLOSED.

Standalone (no slash-command harness, which stubs `_is_allowed`) so these
exercise the REAL function: an unset CHAT_ID ignores everyone; a set CHAT_ID
admits only that chat.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

# ── stub `telegram` before importing handlers (no PTB in the test venv) ──
try:  # pragma: no cover — real PTB present in the prod image
    import telegram  # noqa: F401
except ImportError:
    _tg = types.ModuleType("telegram")

    class _Stub:
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

from app.telegram import handlers  # noqa: E402


def _upd(chat_id: int) -> SimpleNamespace:
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id))


def test_fail_closed_when_chat_id_unset(monkeypatch):
    # A missing/empty CHAT_ID must ignore EVERYONE — never open the bot to all.
    monkeypatch.setattr(handlers, "get_settings", lambda: SimpleNamespace(chat_id=""))
    monkeypatch.setattr(handlers, "_warned_no_chat_id", False, raising=False)
    assert handlers._is_allowed(_upd(12345)) is False


def test_restricts_to_configured_chat(monkeypatch):
    monkeypatch.setattr(handlers, "get_settings", lambda: SimpleNamespace(chat_id="999"))
    assert handlers._is_allowed(_upd(999)) is True
    assert handlers._is_allowed(_upd(111)) is False
