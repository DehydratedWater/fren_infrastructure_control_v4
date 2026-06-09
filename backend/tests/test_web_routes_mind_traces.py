"""Route-level tests for the Mind tab, /traces pages and the images filter.

Same conventions as test_web_dashboard.py: every ``app.web.data`` accessor the
routes call is monkeypatched, so the FastAPI TestClient exercises the real
templates/HTMX wiring without a live Postgres. Skipped (whole module) when
fastapi isn't installed in the venv — the data-shaping + template-rendering
coverage for the same features lives in test_web_mind_traces.py and runs
without fastapi.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient

from app.web import data
from app.web.app import create_app

NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)


def _mind_fixture() -> dict[str, Any]:
    return {
        "mood": {
            "meters": [{"key": k, "label": k, "value": 0.5, "pct": 50}
                       for k, _ in data.MOOD_DIMS],
            "dominant_mood": "engaged", "last_trigger": "coding",
            "drift_count": 3, "updated_at": NOW, "freshness": "ok",
        },
        "vibe": {
            "chat_id": 42,
            "bars": [{"key": k, "label": label, "value": 0.2, "pct": 20}
                     for k, label in data.VIBE_DIMS],
            "axes": [{"key": k, "label": label, "value": 0.0, "pct": 50}
                     for k, label in data.VIBE_AXES],
            "last_trigger": "user joked", "last_user_tone": "warm",
            "drift_count": 9, "updated_at": NOW, "freshness": "ok",
        },
        "vibe_history": [],
        "violations": [{"violation_type": "em_dash", "n": 2}],
        "interests": [{
            "topic": "synths", "stance": "", "source": "rss",
            "novelty": {"value": 0.7, "pct": 70}, "surface_count": 1,
            "last_surfaced_at": NOW, "created_at": NOW,
        }],
        "thoughts": [{
            "content": "ask about the build", "kind": "question",
            "motivation": {"value": 0.9, "pct": 90},
            "breakdown": [{"key": "novelty", "value": 0.5}],
            "created_at": NOW, "consumed_at": None, "consumed_by": "",
        }],
    }


_EMPTY_MIND: dict[str, Any] = {"mood": None, "vibe": None, "vibe_history": [],
                               "violations": [], "interests": [], "thoughts": []}


def _traces_fixture() -> dict[str, Any]:
    traces = [{
        "run_id": "tr1", "created_at": NOW, "kind": "nudge", "model": "qwen3.5",
        "duration_ms": 1200, "input_tokens": 100, "output_tokens": 20,
        "fallback": True,
    }]
    return {"traces": traces,
            "stats": {"total": 1, "count_24h": 1, "p50_ms": 1200.0,
                      "p95_ms": 1200.0, "fallback_rate": 1.0},
            "window": 200, "cap": 50}


_EMPTY_TRACES: dict[str, Any] = {
    "traces": [], "stats": {"total": 0, "count_24h": 0, "p50_ms": None,
                            "p95_ms": None, "fallback_rate": 0.0},
    "window": 200, "cap": 50,
}


def _trace_detail_fixture() -> dict[str, Any]:
    return {
        "run_id": "tr1", "created_at": NOW, "producer": "persona_prose",
        "kind": "nudge", "model": "qwen3.5", "provider": "local",
        "duration_ms": 1200, "input_tokens": 100, "output_tokens": 20,
        "temperature": 0.7, "max_tokens": 800, "fallback": False,
        "system_prompt": "be kind <script>alert('x')</script>",
        "messages": [{"role": "user", "content": "hello"}],
        "raw_output": "<think>…</think>hi there",
        "thinking": "…", "stripped_output": "hi there",
        "delivered_text": "hi there",
        "context_summary": {"history_msgs": 5},
        "guidance": {"message_kind": "nudge"},
    }


def _patch(monkeypatch, *, empty: bool = False) -> None:
    async def _ret(v):
        return v

    health = {
        "db_ok": True, "chat_count": 1, "run_count": 1, "persona_count": 1,
        "last_run_at": NOW, "last_chat_at": NOW, "qwen_url": "http://x",
        "mood_updated_at": None if empty else NOW, "mood_fresh": "" if empty else "ok",
        "vibe_updated_at": None if empty else NOW, "vibe_fresh": "" if empty else "ok",
        "interests_updated_at": None, "interests_fresh": "",
    }
    persona = {"messages": [], "delivered_count": 0, "skipped_count": 0,
               "total": 0, "repeat_groups": 0}
    images = {"images": [], "cap": 60, "truncated": False, "kind": "all"}

    monkeypatch.setattr(data, "health", lambda *a, **k: _ret(health))
    monkeypatch.setattr(data, "recent_persona_responses", lambda *a, **k: _ret(persona))
    monkeypatch.setattr(data, "recent_runs", lambda *a, **k: _ret([]))
    monkeypatch.setattr(data, "conversation_digest", lambda *a, **k: _ret(None))
    monkeypatch.setattr(data, "inner_monologue", lambda *a, **k: _ret([]))
    monkeypatch.setattr(data, "emotional_state", lambda *a, **k: _ret(None))
    monkeypatch.setattr(data, "recent_activity_blocks", lambda *a, **k: _ret([]))
    monkeypatch.setattr(data, "recent_chat", lambda *a, **k: _ret([]))
    monkeypatch.setattr(data, "recent_images", lambda *a, **k: _ret(images))
    monkeypatch.setattr(data, "mind",
                        lambda *a, **k: _ret(_EMPTY_MIND if empty else _mind_fixture()))
    monkeypatch.setattr(data, "prose_traces",
                        lambda *a, **k: _ret(_EMPTY_TRACES if empty else _traces_fixture()))

    async def _detail(run_id):
        if empty or run_id != "tr1":
            return None
        return _trace_detail_fixture()

    monkeypatch.setattr(data, "prose_trace_detail", _detail)


@pytest.fixture
def client(monkeypatch) -> TestClient:
    _patch(monkeypatch)
    return TestClient(create_app())


@pytest.fixture
def empty_client(monkeypatch) -> TestClient:
    _patch(monkeypatch, empty=True)
    return TestClient(create_app())


# ── routes return 200, full and empty ─────────────────────────────────────────

ROUTES = ["/", "/partials/mind", "/traces", "/partials/traces",
          "/traces/tr1", "/partials/images", "/partials/images?kind=rendered"]


@pytest.mark.parametrize("route", ROUTES)
def test_routes_return_200(client, route):
    assert client.get(route).status_code == 200


@pytest.mark.parametrize("route", ROUTES)
def test_routes_empty_data_render(empty_client, route):
    assert empty_client.get(route).status_code == 200


# ── index wiring ──────────────────────────────────────────────────────────────


def test_index_has_mind_tab_and_traces_link(client):
    html = client.get("/").text
    assert 'data-tab="mind"' in html
    assert 'id="panel-mind"' in html
    assert 'hx-get="/partials/mind"' in html
    assert 'href="/traces"' in html


def test_index_mind_panel_first_paint(client):
    html = client.get("/").text
    assert "dominant: engaged" in html
    assert "warm-snarky" in html


# ── mind partial ──────────────────────────────────────────────────────────────


def test_mind_partial_renders(client):
    html = client.get("/partials/mind").text
    assert "dominant: engaged" in html
    assert "em_dash × 2" in html
    assert "synths" in html
    assert "ask about the build" in html


def test_mind_partial_empty(empty_client):
    html = empty_client.get("/partials/mind").text
    assert "No user_mood_state yet." in html
    assert "No persona_vibe_state yet." in html


# ── traces ────────────────────────────────────────────────────────────────────


def test_traces_page_lists_and_links(client):
    html = client.get("/traces").text
    assert "/traces/tr1" in html
    assert "qwen3.5" in html
    assert "FALLBACK" in html
    assert 'hx-get="/partials/traces"' in html  # polled list


def test_trace_detail_renders_audit_sections_escaped(client):
    html = client.get("/traces/tr1").text
    assert "system prompt" in html
    assert "hi there" in html
    # the system prompt's <script> is escaped, never raw
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    # detail page never polls
    assert "hx-trigger" not in html


def test_trace_detail_missing(client):
    r = client.get("/traces/nope")
    assert r.status_code == 200
    assert "Trace not found" in r.text


# ── images filter param ───────────────────────────────────────────────────────


def test_images_partial_passes_normalized_kind(monkeypatch, client):
    seen: list[str] = []

    async def fake_images(limit: int = 60, kind: str = "all"):
        seen.append(kind)
        return {"images": [], "cap": 60, "truncated": False, "kind": kind}

    monkeypatch.setattr(data, "recent_images", fake_images)
    assert client.get("/partials/images?kind=rendered").status_code == 200
    assert client.get("/partials/images?kind=evil").status_code == 200
    assert seen == ["rendered", "all"]  # unknown filter normalised server-side


def test_images_partial_filter_buttons(client):
    html = client.get("/partials/images?kind=captures").text
    assert 'hx-get="/partials/images?kind=rendered"' in html
    assert 'hx-get="/partials/images?kind=captures"' in html
