"""Proactive-agent context loader — v3 parity for scheduled agents.

v3's proactive agents (periodic_checker, nudge_strategist, winddown, …) were
context-starved in v4: the scheduler's prompt enrichment carried the digest +
24h chat history, but emotional_state / vibe / inner_thoughts only reached the
agent if the small fleet model chose to call the personality_core / chat_history
tools — which it often skipped, so the proactive voice looped on a few items.

``build_proactive_context_block()`` now loads the SAME volatile sources
persona_prose's render path uses, inline, so the agent always sees them. These
tests assert the full v3 source set is present when data exists and that the
block degrades cleanly (empty string, no crash) when any/all sources are empty.
All mocked — no live DB.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from app.telegram import persona_prose as pp
from app.telegram.persona_prose import (
    _format_activity_blocks,
    _format_inner_thoughts,
    build_proactive_context_block,
)


# ── Pure formatters ────────────────────────────────────────────────────────


def test_format_inner_thoughts_renders_entries():
    out = _format_inner_thoughts(
        [
            {"created_at": "2026-06-08T02:14:00", "title": "worried — late night", "content": "He's still up coding."},
        ]
    )
    assert "## Recent inner thoughts" in out
    assert "(worried)" in out
    assert "still up coding" in out


def test_format_inner_thoughts_empty_is_blank():
    assert _format_inner_thoughts([]) == ""
    # entries with no content contribute nothing
    assert _format_inner_thoughts([{"created_at": "x", "title": "t", "content": ""}]) == ""


def test_format_activity_blocks_includes_health_snapshot():
    out = _format_activity_blocks(
        [
            {
                "started_at": "2026-06-08T01:00:00",
                "ended_at": "2026-06-08T02:00:00",
                "title": "desk / coding",
                "health_snapshot": {"body_battery": 9, "stress": 71},
            }
        ]
    )
    assert "## Recent activity blocks" in out
    assert "desk / coding" in out
    assert "body_battery=9" in out
    assert "stress=71" in out


def test_format_activity_blocks_empty_is_blank():
    assert _format_activity_blocks([]) == ""


# ── build_proactive_context_block — full source set present ────────────────


class _FakeEmotionalRepo:
    async def get_current(self):
        return {"response_guidance": "Close the loop on the work; keep it short."}


class _FakeVibeRepo:
    async def get(self, *, chat_id: int):
        return {
            "w_warm_snarky": 0.4,
            "w_dry_ironic": 0.15,
            "ironic_genuine_axis": 0.0,
            "arousal_axis": 0.0,
        }


class _FakeMemoriesRepo:
    async def search_by_tags(self, tags, *, limit=20):
        assert "inner_monologue" in tags
        return [
            {"created_at": "2026-06-08T02:14:00", "title": "worried — late", "content": "He's still coding at 2am."}
        ]


class _FakeActivityRepo:
    async def get_recent_blocks(self, hours=6):
        return [
            {
                "started_at": "2026-06-08T01:00:00",
                "ended_at": "2026-06-08T02:00:00",
                "title": "desk / coding",
                "health_snapshot": {"body_battery": 9},
            }
        ]


def _patch_repos(monkeypatch, *, emotional, vibe, memories, activity):
    import app.db.repos.emotional_state as es
    import app.db.repos.persona_vibe as pv
    import app.db.repos.memories as mem
    import app.db.repos.activity_blocks as ab

    monkeypatch.setattr(es, "EmotionalStateRepo", emotional)
    monkeypatch.setattr(pv, "VibeStateRepo", vibe)
    monkeypatch.setattr(mem, "MemoriesRepo", memories)
    monkeypatch.setattr(ab, "ActivityBlocksRepo", activity)


async def test_proactive_block_includes_full_v3_source_set(monkeypatch):
    _patch_repos(
        monkeypatch,
        emotional=_FakeEmotionalRepo,
        vibe=_FakeVibeRepo,
        memories=_FakeMemoriesRepo,
        activity=_FakeActivityRepo,
    )

    block = await build_proactive_context_block()

    assert "## Current emotional state" in block
    assert "## Current vibe blend" in block
    assert "## Recent inner thoughts" in block
    assert "## Recent activity blocks" in block
    assert "body_battery=9" in block
    # Anti-fabrication guard frames the block and reports health PRESENT
    # (the activity block carried a health_snapshot).
    assert "GROUNDING CONTRACT" in block
    assert block.index("GROUNDING CONTRACT") < block.index("## Current emotional state")
    assert "Garmin health" in block  # listed among present signals


# ── degrades cleanly when sources are empty ────────────────────────────────


class _EmptyEmotionalRepo:
    async def get_current(self):
        return None


class _EmptyVibeRepo:
    async def get(self, *, chat_id: int):
        return {}


class _EmptyMemoriesRepo:
    async def search_by_tags(self, tags, *, limit=20):
        return []


class _EmptyActivityRepo:
    async def get_recent_blocks(self, hours=6):
        return []


async def test_proactive_block_emits_guard_even_when_all_sources_empty(monkeypatch):
    # When every data source is empty the block is NOT empty: it still carries
    # the grounding contract, because a context-starved tick is exactly when the
    # agent is most tempted to hallucinate sensor/health facts.
    _patch_repos(
        monkeypatch,
        emotional=_EmptyEmotionalRepo,
        vibe=_EmptyVibeRepo,
        memories=_EmptyMemoriesRepo,
        activity=_EmptyActivityRepo,
    )

    block = await build_proactive_context_block()
    assert "GROUNDING CONTRACT" in block
    assert "Signals present THIS tick: NONE" in block
    # health is absent → it must be named in the "do not reference" list
    assert "Garmin health" in block
    assert "FABRICATION" in block
    # no real data sections present
    assert "## Current emotional state" not in block
    assert "## Recent activity blocks" not in block


def test_anti_fabrication_guard_lists_present_and_absent():
    from app.telegram.persona_prose import _anti_fabrication_guard

    # Only emotional_state present → health must be flagged absent.
    g = _anti_fabrication_guard({"emotional_state"})
    assert "GROUNDING CONTRACT" in g
    assert "emotional state" in g
    assert "ABSENT THIS tick" in g
    assert "Garmin health" in g  # absent → named in the do-not-reference list

    # Health present → it appears in the "may use" list, not the absent list.
    g2 = _anti_fabrication_guard({"health", "activity_blocks"})
    assert "may use these" in g2
    assert "Garmin health" in g2


async def test_proactive_block_partial_sources_no_crash(monkeypatch):
    # emotional present, everything else empty → only the emotional section.
    _patch_repos(
        monkeypatch,
        emotional=_FakeEmotionalRepo,
        vibe=_EmptyVibeRepo,
        memories=_EmptyMemoriesRepo,
        activity=_EmptyActivityRepo,
    )

    block = await build_proactive_context_block()
    assert "## Current emotional state" in block
    assert "## Current vibe blend" not in block
    assert "## Recent activity blocks" not in block


async def test_proactive_block_survives_repo_exception(monkeypatch):
    # A repo that raises must not crash the loader — other sources still load.
    class _BoomVibe:
        async def get(self, *, chat_id: int):
            raise RuntimeError("db down")

    _patch_repos(
        monkeypatch,
        emotional=_FakeEmotionalRepo,
        vibe=_BoomVibe,
        memories=_EmptyMemoriesRepo,
        activity=_EmptyActivityRepo,
    )

    block = await build_proactive_context_block()
    # emotional still present; vibe failure swallowed.
    assert "## Current emotional state" in block
    assert "## Current vibe blend" not in block
