"""Scheduler prompt enrichment — proactive agents get the full v3 context set.

``Scheduler._enrich_prompt`` prepends user_rules + agent_lessons +
conversation_digest + the new volatile block (emotional_state / vibe /
inner_thoughts / activity) + the 24h chat history (incl. Twily's own sends,
with an explicit anti-repetition instruction). These tests assert the wiring
end-to-end with every external source mocked — no live DB, LLM, or Telegram.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("croniter")

from app.telegram.scheduler import Scheduler


class _Rules:
    async def format_rules_prompt(self):
        return "## User Rules\n- never call after midnight"


class _Lessons:
    async def format_lessons_prompt(self):
        return "## Agent Lessons\n- [tone] do not lecture about sleep"


class _Notes:
    async def get(self, key):
        assert key == "conversation_digest"
        return {"note_value": {"digest": "User is finishing a deploy, exhausted, 2am."}}


class _Chat:
    async def get_history(self, *, days, limit, clearance):
        # one user msg + one twily send, both within the last hour.
        import time

        now = time.time()
        return [
            {"timestamp": "2026-06-08T01:00", "sender": "user", "message": "still deploying", "timestamp_unix": now - 600},
            {
                "timestamp": "2026-06-08T01:05",
                "sender": "twily",
                "message": "go to bed after this push",
                "timestamp_unix": now - 300,
            },
        ]


async def test_enrich_prompt_carries_full_context_set(monkeypatch):
    import app.db.repos.user_rules as ur
    import app.db.repos.agent_lessons as al
    import app.db.repos.agent_notes as an
    import app.db.repos.chat as ch
    from app.telegram import persona_prose as pp

    monkeypatch.setattr(ur, "UserRulesRepo", _Rules)
    monkeypatch.setattr(al, "AgentLessonsRepo", _Lessons)
    monkeypatch.setattr(an, "AgentNotesRepo", _Notes)
    monkeypatch.setattr(ch, "ChatMessagesRepo", _Chat)

    async def fake_volatile():
        return "## Current emotional state\n- **guidance**: keep it short\n\n## Current vibe blend\n- w_warm_snarky: 0.40"

    monkeypatch.setattr(pp, "build_proactive_context_block", fake_volatile)

    out = await Scheduler._enrich_prompt("## TASK: run periodic check")

    # All v3 context sources present.
    assert "## User Rules" in out
    assert "## Agent Lessons" in out
    assert "User is finishing a deploy" in out  # digest
    assert "## Current emotional state" in out  # NEW volatile block
    assert "## Current vibe blend" in out
    assert "## Chat History (last 24h)" in out
    # Twily's own send is visible (anti-repetition source).
    assert "twily: go to bed after this push" in out
    # Explicit anti-repetition instruction is attached to the history.
    assert "do NOT repeat" in out
    # Task prompt preserved after the context prefix.
    assert "## TASK: run periodic check" in out
    assert out.index("## TASK") > out.index("## Chat History")


async def test_enrich_prompt_degrades_when_sources_empty(monkeypatch):
    import app.db.repos.user_rules as ur
    import app.db.repos.agent_lessons as al
    import app.db.repos.agent_notes as an
    import app.db.repos.chat as ch
    from app.telegram import persona_prose as pp

    class _NoRules:
        async def format_rules_prompt(self):
            return ""

    class _NoLessons:
        async def format_lessons_prompt(self):
            return ""

    class _NoNotes:
        async def get(self, key):
            return None

    class _NoChat:
        async def get_history(self, *, days, limit, clearance):
            return []

    async def empty_volatile():
        return ""

    monkeypatch.setattr(ur, "UserRulesRepo", _NoRules)
    monkeypatch.setattr(al, "AgentLessonsRepo", _NoLessons)
    monkeypatch.setattr(an, "AgentNotesRepo", _NoNotes)
    monkeypatch.setattr(ch, "ChatMessagesRepo", _NoChat)
    monkeypatch.setattr(pp, "build_proactive_context_block", empty_volatile)

    out = await Scheduler._enrich_prompt("## TASK: run check")
    # No prefix, no separator — the bare task prompt is returned unchanged.
    assert out == "## TASK: run check"
