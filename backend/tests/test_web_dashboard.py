"""Web monitoring dashboard — route rendering + skip/delivered classification.

All DB access is mocked: every test patches the ``app.web.data`` accessors the
routes call, so the FastAPI TestClient exercises the real templates/HTMX wiring
without a live Postgres. Three things are guarded:

  1. every route returns 200 and renders its expected section markers,
  2. empty-data renders without raising (graceful empty tables),
  3. the persona_response skip-vs-delivered classification is correct.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")

from fastapi.testclient import TestClient

from app.web import data
from app.web.app import create_app

NOW = datetime(2026, 6, 9, 1, 8, 0)


# ── classification unit tests (no app) ────────────────────────────────────────


def test_classify_delivered():
    is_skip, text = data.classify_persona_response(
        {"kind": "nudge", "delivered_text": "Lights off. Go to sleep 💜"}
    )
    assert is_skip is False
    assert text == "Lights off. Go to sleep 💜"


def test_classify_explicit_skip():
    is_skip, text = data.classify_persona_response(
        {"kind": "skip", "skipped": True, "delivered": False, "delivered_text": ""}
    )
    assert is_skip is True
    assert text == ""


def test_classify_empty_text_is_skip():
    is_skip, _ = data.classify_persona_response({"delivered_text": "   "})
    assert is_skip is True


def test_classify_delivered_false_is_skip():
    is_skip, _ = data.classify_persona_response(
        {"delivered": False, "delivered_text": "had text but not delivered"}
    )
    assert is_skip is True


def test_classify_missing_payload():
    is_skip, text = data.classify_persona_response({})
    assert is_skip is True
    assert text == ""


# ── fixtures: a client with the data layer fully stubbed ──────────────────────


def _persona_fixture() -> dict[str, Any]:
    repeated = "The work is done, your shift is over — go to sleep now."
    items = [
        {
            "run_id": "run_a", "owner": "goals/winddown", "producer": "persona_prose",
            "kind": "nudge", "created_at": NOW, "is_skip": False,
            "delivered_text": repeated, "norm": repeated.lower(), "repeat_group": 1,
        },
        {
            "run_id": "run_b", "owner": "goals/winddown", "producer": "persona_prose",
            "kind": "nudge", "created_at": NOW, "is_skip": False,
            "delivered_text": repeated, "norm": repeated.lower(), "repeat_group": 1,
        },
        {
            "run_id": "run_c", "owner": "goals/nudge_strategist", "producer": "emit_guidance",
            "kind": "skip", "created_at": NOW, "is_skip": True,
            "delivered_text": "", "norm": "", "repeat_group": None,
        },
    ]
    return {
        "messages": items, "delivered_count": 2, "skipped_count": 1,
        "total": 3, "repeat_groups": 1,
    }


def _patch_all(monkeypatch, *, empty: bool = False) -> None:
    persona = (
        {"messages": [], "delivered_count": 0, "skipped_count": 0, "total": 0, "repeat_groups": 0}
        if empty else _persona_fixture()
    )
    runs = [] if empty else [
        {
            "run_id": "run_a", "owner": "support/event_extractor-primary",
            "status": "running", "interaction_mode": "cron", "domain": "support",
            "started_at": NOW, "completed_at": None, "artifact_count": 3,
        }
    ]
    monologue = [] if empty else [
        {"memory_id": "m1", "title": "worried — 01:15", "content": "he's still awake",
         "tags": ["inner_monologue"], "created_at": NOW}
    ]
    emotional = None if empty else {
        "description": "A concentrated hum of care.", "mood_shift": "neutral→urgent",
        "created_at": NOW,
    }
    blocks = [] if empty else [
        {"started_at": NOW, "activity_type": "coding", "title": "OpenCodeCompilerV2"}
    ]
    chat = [] if empty else [
        {"sender": "twily", "message": "Lights off. Go to sleep.", "timestamp": NOW},
        {"sender": "user", "message": "five more minutes", "timestamp": NOW},
    ]
    digest = None if empty else "## Conversation Digest\nUser is awake at 01:08."
    health = {
        "db_ok": not empty, "chat_count": 0 if empty else 42, "run_count": 0 if empty else 7,
        "persona_count": 0 if empty else 3, "last_run_at": None if empty else NOW,
        "last_chat_at": None if empty else NOW, "qwen_url": "http://192.168.0.42:8082/v1",
    }

    async def _ret(v):
        return v

    monkeypatch.setattr(data, "recent_persona_responses", lambda *a, **k: _ret(persona))
    monkeypatch.setattr(data, "recent_runs", lambda *a, **k: _ret(runs))
    monkeypatch.setattr(data, "conversation_digest", lambda *a, **k: _ret(digest))
    monkeypatch.setattr(data, "inner_monologue", lambda *a, **k: _ret(monologue))
    monkeypatch.setattr(data, "emotional_state", lambda *a, **k: _ret(emotional))
    monkeypatch.setattr(data, "recent_activity_blocks", lambda *a, **k: _ret(blocks))
    monkeypatch.setattr(data, "recent_chat", lambda *a, **k: _ret(chat))
    monkeypatch.setattr(data, "health", lambda *a, **k: _ret(health))
    monkeypatch.setattr(data, "db_ok", lambda *a, **k: _ret(not empty))


@pytest.fixture
def client(monkeypatch) -> TestClient:
    _patch_all(monkeypatch, empty=False)
    return TestClient(create_app())


@pytest.fixture
def empty_client(monkeypatch) -> TestClient:
    _patch_all(monkeypatch, empty=True)
    return TestClient(create_app())


# ── route rendering ───────────────────────────────────────────────────────────

ROUTES = ["/", "/partials/health", "/partials/proactive",
          "/partials/runs", "/partials/context", "/partials/chat"]


@pytest.mark.parametrize("route", ROUTES)
def test_routes_return_200(client, route):
    assert client.get(route).status_code == 200


@pytest.mark.parametrize("route", ROUTES)
def test_routes_empty_data_renders(empty_client, route):
    r = empty_client.get(route)
    assert r.status_code == 200
    # no template-rendering exception leaked a 500


def test_index_has_all_sections(client):
    html = client.get("/").text
    assert "Proactive messages" in html
    assert "Agent runs" in html
    assert "Context fed to agents" in html
    assert "Chat" in html
    # HTMX polling wired up
    assert "hx-get=\"/partials/proactive\"" in html
    assert "every 10s" in html


def test_proactive_shows_delivered_and_skip(client):
    html = client.get("/partials/proactive").text
    assert "2 delivered" in html
    assert "1 skipped" in html
    assert "SENT" in html
    assert "SKIP" in html
    # repeated message flagged
    assert "repeat" in html.lower()


def test_proactive_empty(empty_client):
    html = empty_client.get("/partials/proactive").text
    assert "No persona_response artifacts yet." in html
    assert "0 delivered" in html


def test_context_renders_real_fields(client):
    html = client.get("/partials/context").text
    assert "Conversation Digest" in html
    assert "concentrated hum of care" in html
    assert "inner_monologue" in html
    assert "coding" in html


def test_chat_renders_messages(client):
    html = client.get("/partials/chat").text
    assert "Lights off" in html
    assert "five more minutes" in html


def test_runs_render(client):
    html = client.get("/partials/runs").text
    assert "event_extractor-primary" in html
    assert "running" in html


def test_runs_rows_link_to_detail(client):
    html = client.get("/partials/runs").text
    # each runs-panel row links to its /run/{run_id} detail page
    assert "/run/run_a" in html


# ── run detail (view the session) ─────────────────────────────────────────────


def _trace_fixture() -> dict[str, Any]:
    return {
        "run": {
            "run_id": "run_a", "owner": "goals/periodic_checker-primary",
            "status": "completed", "interaction_mode": "cron", "domain": "",
            "started_at": NOW, "completed_at": NOW, "contract_passed": True,
        },
        "trace": {
            "text": "I checked the schedule and nothing is due right now.",
            "tool_calls": [
                {"name": "bash", "command": "python scripts/list_due.py", "error": None},
                {"name": "bash", "command": "ls /forbidden",
                 "error": "a rule prevents you from using ls"},
            ],
            "tool_call_count": 2, "ok": True, "error": None, "producer": "runner",
        },
        "persona": [
            {"artifact_type": "persona_response", "producer": "persona_prose",
             "created_at": NOW, "payload": {"delivered_text": "All quiet — sleep well 💜"}},
        ],
    }


def test_run_detail_renders_header_trace_and_persona(client, monkeypatch):
    async def _ret(_run_id):
        return _trace_fixture()

    monkeypatch.setattr(data, "run_detail", _ret)
    html = client.get("/run/run_a").text
    # header
    assert "goals/periodic_checker-primary" in html
    assert "completed" in html
    assert "cron" in html
    # assistant output
    assert "nothing is due right now" in html
    # ordered tool calls: names + the bash commands
    assert "python scripts/list_due.py" in html
    assert "ls /forbidden" in html
    # the denied call surfaces its reason
    assert "prevents you from using ls" in html
    # persona delivery shown
    assert "sleep well" in html


def test_run_detail_no_trace_degrades(client, monkeypatch):
    async def _ret(_run_id):
        return {
            "run": {
                "run_id": "run_x", "owner": "goals/winddown-primary",
                "status": "completed", "interaction_mode": "cron", "domain": "",
                "started_at": NOW, "completed_at": NOW, "contract_passed": None,
            },
            "trace": None,
            "persona": [],
        }

    monkeypatch.setattr(data, "run_detail", _ret)
    html = client.get("/run/run_x").text
    assert "no trajectory captured" in html.lower()


def test_run_detail_missing_run(client, monkeypatch):
    async def _ret(_run_id):
        return None

    monkeypatch.setattr(data, "run_detail", _ret)
    r = client.get("/run/nope")
    assert r.status_code == 200
    assert "not found" in r.text.lower()


def test_run_detail_data_layer_parses_trace_artifact(monkeypatch):
    """data.run_detail assembles header + latest run_trace + persona from rows."""
    run_row = {
        "run_id": "r1", "owner": "goals/periodic_checker-primary",
        "status": "completed", "interaction_mode": "cron", "domain": "",
        "started_at": NOW, "completed_at": NOW, "contract_passed": True,
    }
    # rows arrive ordered by (artifact_type, version DESC) per the query, so the
    # newer run_trace version comes first.
    art_rows = [
        {"artifact_type": "persona_response", "version": 1, "producer": "persona_prose",
         "created_at": NOW, "payload": {"delivered_text": "hi there"}},
        {"artifact_type": "run_trace", "version": 2, "producer": "runner",
         "created_at": NOW,
         "payload": {"text": "newer", "tool_call_count": 1,
                     "tool_calls": [{"name": "bash", "command": "echo hi"}], "ok": True}},
        {"artifact_type": "run_trace", "version": 1, "producer": "runner",
         "created_at": NOW,
         "payload": {"text": "older", "tool_calls": [], "ok": True}},
    ]

    async def fake_fetch_one(_s, _q, _p):
        return run_row

    async def fake_fetch_all(_s, _q, _p):
        return art_rows

    class _Ctx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(data, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(data, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(data, "get_async_session", lambda: _Ctx())

    import asyncio

    detail = asyncio.run(data.run_detail("r1"))
    assert detail["run"]["owner"] == "goals/periodic_checker-primary"
    # latest version (version DESC ordering) wins
    assert detail["trace"]["text"] == "newer"
    assert detail["trace"]["tool_calls"][0]["command"] == "echo hi"
    assert detail["persona"][0]["payload"]["delivered_text"] == "hi there"


def test_health_strip(client):
    html = client.get("/partials/health").text
    assert "db reachable" in html
    assert "192.168.0.42:8082" in html


def test_healthz_json(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ── recent_persona_responses aggregation (with fetch_all stubbed) ─────────────


def test_recent_persona_responses_classifies_and_groups(monkeypatch):
    repeated = "Go to sleep, your shift is over."
    rows = [
        {"run_id": "r1", "version": 1, "producer": "persona_prose",
         "payload": {"kind": "nudge", "delivered_text": repeated}, "created_at": NOW,
         "owner": "goals/winddown"},
        {"run_id": "r2", "version": 1, "producer": "persona_prose",
         "payload": {"kind": "nudge", "delivered_text": repeated}, "created_at": NOW,
         "owner": "goals/winddown"},
        {"run_id": "r3", "version": 2, "producer": "emit_guidance",
         "payload": {"kind": "skip", "skipped": True, "delivered_text": ""},
         "created_at": NOW, "owner": "goals/nudge_strategist"},
    ]

    async def fake_fetch_all(_s, _q, _p):
        return rows

    class _Ctx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(data, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(data, "get_async_session", lambda: _Ctx())

    import asyncio

    result = asyncio.run(data.recent_persona_responses())
    assert result["delivered_count"] == 2
    assert result["skipped_count"] == 1
    assert result["repeat_groups"] == 1
    # both repeated delivered items share a repeat_group
    rg = [i["repeat_group"] for i in result["messages"] if not i["is_skip"]]
    assert rg == [1, 1]
