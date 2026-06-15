"""Deterministic image/video routing — the selfie fast-path.

Regression lock for #213: the qwen planner (orchestrator/twily_chat) refused
image requests ("I'm stuck in text form") instead of delegating, so
persona/twily_selfie never ran. The broadened `selfie` intent + `_try_media_agent`
route generation intents straight to the specialist, in EVERY mode, so a
"send me a selfie" reliably renders+sends. These are the REAL user phrasings
that previously fell through.
"""

from __future__ import annotations

import re

import pytest

from app.tools.context.intent_inference import INTENT_PATTERNS, _normalize

# The exact messages from the live chat where Twily refused (frenv4 chat_messages,
# 2026-06-14/15) plus the canonical generation phrasings.
SELFIE_REQUESTS = [
    "send me a selfie",
    "take a selfie",
    "Can i at least get some cute photo?",          # live refusal #1
    "Well I'm still waiting for the cute photo :)",  # live refusal #2
    "show me yourself",
    "send a pic",
    "can I get a picture of you",
    "make me a goodnight image",
    "draw yourself",
    "gimme a selfie",
]

NON_MEDIA = [
    "what's the weather today",
    "add a task to call mom",
    "remember that I like tea",
    "I need to buy groceries",
    "the photo album is full",   # mentions photo but is not a request
]


def _intents(message: str) -> set[str]:
    norm = _normalize(message)
    return {
        intent_type
        for pattern, intent_type, _desc in INTENT_PATTERNS
        if re.search(pattern, norm, re.IGNORECASE)
    }


# ── pure intent classification (no telegram dep — always runs) ──


@pytest.mark.parametrize("msg", SELFIE_REQUESTS)
def test_selfie_requests_classify_as_selfie(msg):
    assert "selfie" in _intents(msg), msg


@pytest.mark.parametrize("msg", NON_MEDIA)
def test_non_media_not_classified_as_selfie(msg):
    assert "selfie" not in _intents(msg), msg


# ── handler-level routing (needs python-telegram-bot; skipped if absent) ──


@pytest.mark.parametrize("msg", SELFIE_REQUESTS)
def test_selfie_requests_route_to_specialist(msg):
    pytest.importorskip("telegram")
    from app.telegram.handlers import _try_media_agent

    assert _try_media_agent(msg) == "persona/twily_selfie", msg


def test_attached_image_is_not_generation():
    pytest.importorskip("telegram")
    from app.telegram.handlers import _try_media_agent

    # A message WITH an image is analysis, not generation — must not hijack to selfie.
    assert _try_media_agent("what is this picture?", has_image=True) is None


def test_media_map_targets_persona_agents():
    pytest.importorskip("telegram")
    from app.telegram.handlers import _MEDIA_INTENT_TO_AGENT

    assert _MEDIA_INTENT_TO_AGENT["selfie"] == "persona/twily_selfie"
    assert all(a.startswith("persona/") for a in _MEDIA_INTENT_TO_AGENT.values())
