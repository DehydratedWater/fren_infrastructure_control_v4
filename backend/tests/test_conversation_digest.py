"""Conversation digest port — v3 parity, fully mocked (no live DB / no live LLM).

Asserts the digester (a) assembles the structured sections, (b) skips when
there is no new chat and the digest is fresh, (c) stores into agent_notes under
the key the scheduler's _enrich_prompt reads, and (d) never fabricates health —
the activity-block fetch is the only health source.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from app.tools.context import conversation_digest as cd


# ── pure helpers ────────────────────────────────────────────────────────────


def test_strip_thinking_removes_think_blocks():
    assert cd._strip_thinking("<think>plan</think>\n## Digest\n- x") == "## Digest\n- x"


async def test_generate_digest_assembles_sections(monkeypatch):
    captured: dict = {}

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "## Conversation Digest\n- the user is coding"}}]}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["payload"] = json
            return _FakeResp()

    monkeypatch.setattr(cd.httpx, "AsyncClient", _FakeClient)

    out = await cd._generate_digest(
        {"chat": "[10:00] user: hi", "todos": "**Overdue todos:**\n  - file taxes"},
        previous_digest=None,
    )
    assert "coding" in out
    # the structured todo section made it into the user content
    user_msg = captured["payload"]["messages"][1]["content"]
    assert "file taxes" in user_msg
    # local vLLM chat-completions endpoint, thinking disabled
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["chat_template_kwargs"]["enable_thinking"] is False


async def test_activity_blocks_health_only_when_present(monkeypatch):
    class _Repo:
        async def get_recent_blocks(self, hours=6):
            return [
                {"started_at": "2026-06-09T01:00", "title": "desk", "health_snapshot": {"body_battery": 12}},
                {"started_at": "2026-06-09T02:00", "title": "away", "health_snapshot": {}},
            ]

    import app.db.repos.activity_blocks as ab

    monkeypatch.setattr(ab, "ActivityBlocksRepo", _Repo)
    out = await cd._fetch_activity_blocks()
    assert "desk · health: body_battery=12" in out
    # the block with an empty snapshot carries no health text
    assert "away" in out and "away · health" not in out


async def test_activity_blocks_empty_is_blank(monkeypatch):
    class _Repo:
        async def get_recent_blocks(self, hours=6):
            return []

    import app.db.repos.activity_blocks as ab

    monkeypatch.setattr(ab, "ActivityBlocksRepo", _Repo)
    assert await cd._fetch_activity_blocks() == ""


# ── run() orchestration ─────────────────────────────────────────────────────


async def test_run_skips_when_no_new_messages_and_fresh(monkeypatch):
    from datetime import UTC, datetime

    fresh = datetime.now(UTC).isoformat()
    monkeypatch.setattr(cd, "_get_latest_message_id", lambda: _async(5))
    monkeypatch.setattr(
        cd,
        "_get_digest_metadata",
        lambda: _async({"last_message_id": 5, "updated_at": fresh, "update_count": 1}),
    )
    called = {"generated": False}

    async def _boom(*a, **k):
        called["generated"] = True
        return "x"

    monkeypatch.setattr(cd, "_generate_digest", _boom)
    result = await cd.run(hours=12)
    assert result is None
    assert called["generated"] is False


async def test_run_generates_and_stores(monkeypatch):
    monkeypatch.setattr(cd, "_get_latest_message_id", lambda: _async(9))
    monkeypatch.setattr(
        cd,
        "_get_digest_metadata",
        lambda: _async({"last_message_id": 3, "updated_at": "", "update_count": 0}),
    )
    monkeypatch.setattr(cd, "_fetch_recent_chat", lambda hours: _async("[10:00] user: hi"))
    monkeypatch.setattr(cd, "_fetch_current_digest", lambda: _async(None))
    for fn in (
        "_fetch_todos",
        "_fetch_priorities",
        "_fetch_goals",
        "_fetch_habits",
        "_fetch_upcoming_events",
        "_fetch_inner_thoughts",
        "_fetch_nudge_campaigns",
        "_fetch_activity_blocks",
    ):
        monkeypatch.setattr(cd, fn, lambda *a, **k: _async(""))

    async def _gen(sections, previous_digest):
        return "## Conversation Digest\n- the user said hi"

    monkeypatch.setattr(cd, "_generate_digest", _gen)

    stored: dict = {}

    async def _store(digest, last_message_id=0, update_count=0):
        stored["digest"] = digest
        stored["last_message_id"] = last_message_id

    monkeypatch.setattr(cd, "_store_digest", _store)

    result = await cd.run(hours=12)
    assert result is not None and "the user said hi" in result
    assert stored["digest"] == result
    assert stored["last_message_id"] == 9


# small helper to wrap a value in an awaitable for monkeypatched async funcs
async def _async(value):
    return value
