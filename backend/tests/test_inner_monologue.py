"""Inner monologue port — fully mocked (no live DB / no live LLM).

Asserts the writer stores a memory tagged ``inner_monologue`` (the tag the
proactive loader + digest read), skips on empty thought/context, and parses
JSON-with-thinking output.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from app.tools.system import inner_monologue as im


def test_strip_thinking_then_json():
    raw = "<think>let me feel</think>\n{\"emotion\": \"curious\", \"thought\": \"hmm\"}"
    assert im._strip_thinking(raw).startswith("{")


async def _async(value):
    return value


async def test_generate_thought_parses_fenced_json(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "```json\n{\"emotion\": \"content\", \"thought\": \"quiet night\"}\n```"}}
                ]
            }

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json):
            return _Resp()

    monkeypatch.setattr(im.httpx, "AsyncClient", _Client)
    out = await im._generate_thought("Chat:\n[10:00] user: hi\n")
    assert out["emotion"] == "content"
    assert out["thought"] == "quiet night"


async def test_store_thought_tags_inner_monologue(monkeypatch):
    captured: dict = {}

    class _Repo:
        async def create(self, memory_id, title, content, *, tags=None, category="", source="", embedding=None):
            captured.update(memory_id=memory_id, tags=tags, content=content, category=category, source=source)
            return {"memory_id": memory_id}

    import app.db.repos.memories as mem

    monkeypatch.setattr(mem, "MemoriesRepo", _Repo)
    monkeypatch.setattr(im, "_is_night", lambda: False)
    # embedding service raises -> store still proceeds without it
    import app.services.embeddings as emb

    monkeypatch.setattr(emb, "get_embedding", lambda text: (_ for _ in ()).throw(RuntimeError("no model")))

    mid = await im._store_thought({"emotion": "curious", "thought": "I wonder about partitions"})
    assert mid is not None
    assert "inner_monologue" in captured["tags"]
    assert "thought" in captured["tags"]
    assert captured["category"] == "inner_monologue"
    assert captured["source"] == "inner_monologue"


async def test_store_thought_skips_empty():
    assert await im._store_thought({"emotion": "static", "thought": ""}) is None


async def test_run_skips_when_no_context(monkeypatch):
    monkeypatch.setattr(im, "_gather_context", lambda: _async(""))
    # emotions toggle on
    import app.telegram.state as st

    monkeypatch.setattr(st, "get_emotions_enabled", lambda: True)
    assert await im.run() is None


async def test_run_stores_when_thought_produced(monkeypatch):
    import app.telegram.state as st

    monkeypatch.setattr(st, "get_emotions_enabled", lambda: True)
    monkeypatch.setattr(im, "_gather_context", lambda: _async("Chat:\n[10:00] user: hi\n"))
    monkeypatch.setattr(im, "_generate_thought", lambda ctx: _async({"emotion": "curious", "thought": "neat"}))
    stored: dict = {}

    async def _store(thought):
        stored["thought"] = thought
        return "thought_xyz"

    monkeypatch.setattr(im, "_store_thought", _store)
    mid = await im.run()
    assert mid == "thought_xyz"
    assert stored["thought"]["thought"] == "neat"
