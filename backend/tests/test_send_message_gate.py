"""Regression lock for the send_message → delivery-gate seam.

A real outage (2026-06-15): the cooldown commit updated the CALL
`_gate_message(..., kind=, last_user_age_s=, last_bot_age_s=)` but NOT the
function signature, so EVERY reply via send_message.py raised TypeError at
runtime ("no response" in Telegram). The gate's own unit tests passed because
they exercise app.delivery.gate.evaluate_message directly, not this wrapper.
These tests call _gate_message exactly as send_message.py does.
"""

from __future__ import annotations

from app.tools.telegram.send_message import _gate_message


def test_gate_accepts_cooldown_kwargs_and_delivers():
    # The exact kwargs the real call site passes — must NOT raise.
    out = _gate_message(
        "hey, here's the thing about your tasks today",
        [],
        None,
        kind="reply",
        last_user_age_s=5.0,
        last_bot_age_s=5.0,
    )
    assert out is None  # None → deliver


def test_gate_legacy_positional_still_delivers():
    assert _gate_message("a normal message", [], None) is None


def test_gate_suppresses_proactive_during_active_chat():
    out = _gate_message(
        "just a little nudge to drink water",
        [],
        None,
        kind="nudge",
        last_user_age_s=10.0,
        last_bot_age_s=999.0,
    )
    assert out is not None and out.suppressed and out.reason == "proactive_user_active"


def test_gate_suppresses_duplicate():
    prev = "the garage door has been open for 20 minutes"
    out = _gate_message(prev, [prev], None, kind="reply")
    assert out is not None and out.suppressed and out.reason == "duplicate"
