"""Her world-life flows into her chat voice (offline).

Pins the wiring that makes "what have you been up to?" draw on her Mooring Wells
day instead of generic assistant chores: the recent-life summary must appear in
the volatile context block persona_prose prepends to the reply turn.
"""

from __future__ import annotations

from app.telegram.persona_prose import ChatContext, _format_volatile_context_block


def test_world_life_surfaces_in_volatile_block():
    ctx = ChatContext(chat_id=1, world_life="I spent the evening fighting the balcony battery, then made risotto.")
    block = _format_volatile_context_block(ctx)
    assert "Mooring Wells" in block
    assert "risotto" in block


def test_no_world_life_no_block():
    ctx = ChatContext(chat_id=1, world_life="")
    block = _format_volatile_context_block(ctx)
    assert "Mooring Wells" not in block
