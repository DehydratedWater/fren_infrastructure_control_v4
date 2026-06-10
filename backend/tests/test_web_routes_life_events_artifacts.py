"""Route-level tests for the Life tab, /events and /artifacts pages.

Same conventions as test_web_routes_mind_traces.py: every ``app.web.data``
accessor the routes call is monkeypatched, so the FastAPI TestClient exercises
the real templates/HTMX wiring without a live Postgres. Skipped (whole module)
when fastapi isn't installed in the venv — the data-shaping + template-rendering
coverage for the same features lives in test_web_life_events_artifacts.py and
runs without fastapi.
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


def _life_fixture() -> dict[str, Any]:
    return {
        "goals": [{
            "goal_id": "g1", "parent_goal_id": "", "level": 1, "title": "ship v4",
            "status": "active", "priority": "high", "progress": {"pct": 70},
            "deadline": NOW, "overdue": True, "depth": 0,
        }],
        "todos": {
            "overdue": [{"todo_id": "t1", "title": "fix the bot", "status": "pending",
                         "priority": "high", "category": "work",
                         "linked_goal_id": "g1", "goal_title": "ship v4",
                         "due_label": "06-09"}],
            "today": [], "upcoming": [],
        },
        "habits": {
            "due_today": [{"habit_id": "h1", "title": "gym", "importance_level": 5}],
            "habits": [{"habit_id": "h1", "title": "gym", "frequency_type": "daily",
                        "streak": 4, "best_streak": 9,
                        "rate": {"pct": 80, "completed": 24, "scheduled": 30},
                        "due_today": True}],
        },
        "priorities": {
            "quadrants": {
                "do": [{"priority_id": "p1", "title": "ship v1", "category": "work",
                        "importance": 0.9, "immediacy": 0.9,
                        "real_importance": 0.5, "delta": -0.4, "misaligned": True}],
                "plan": [], "delegate": [], "drop": [],
            },
            "misaligned": 1, "total": 1,
        },
    }


_EMPTY_LIFE: dict[str, Any] = {
    "goals": [],
    "todos": {"overdue": [], "today": [], "upcoming": []},
    "habits": {"due_today": [], "habits": []},
    "priorities": {"quadrants": {"do": [], "plan": [], "delegate": [], "drop": []},
                   "misaligned": 0, "total": 0},
}


def _events_fixture(category: str = "all") -> dict[str, Any]:
    return {
        "category": category,
        "categories": ["sleep", "food"],
        "bars": [{"category": "sleep", "count": 10, "pct": 100},
                 {"category": "food", "count": 5, "pct": 50}],
        "events": [{"event_id": "e1", "occurred_at": NOW, "category": "food",
                    "subcategory": "lunch", "title": "ramen <script>x</script>",
                    "detail": "650 kcal", "source": "extracted"}],
        "strip": ({"days": [{"date": "2026-06-10", "count": 2, "pct": 100}],
                   "max": 2, "total": 2} if category != "all" else None),
        "days": 7, "chart_days": 30, "cap": 200,
    }


_EMPTY_EVENTS: dict[str, Any] = {
    "category": "all", "categories": [], "bars": [], "events": [],
    "strip": None, "days": 7, "chart_days": 30, "cap": 200,
}


def _artifacts_fixture() -> dict[str, Any]:
    return {
        "type": "all",
        "types": [{"name": "selfie", "n": 9, "type_class": "art-c1"}],
        "q": "",
        "artifacts": [{
            "cache_id": "c1", "artifact_type": "selfie", "type_class": "art-c1",
            "summary": "a selfie <script>alert('x')</script>",
            "preview": "a selfie <script>alert('x')</script>", "has_more": False,
            "tags": ["render"], "source_agent": "selfie_agent", "entity": "",
            "created_at": NOW, "expires_at": None, "expired": False,
        }],
        "cap": 100,
    }


_EMPTY_ARTIFACTS: dict[str, Any] = {"type": "all", "types": [], "q": "",
                                    "artifacts": [], "cap": 100}


def _patch(monkeypatch, *, empty: bool = False) -> None:
    async def _ret(v):
        return v

    health = {
        "db_ok": True, "chat_count": 1, "run_count": 1, "persona_count": 1,
        "last_run_at": NOW, "last_chat_at": NOW, "qwen_url": "http://x",
        "mood_updated_at": NOW, "mood_fresh": "ok",
        "vibe_updated_at": NOW, "vibe_fresh": "ok",
        "interests_updated_at": None, "interests_fresh": "",
    }
    persona = {"messages": [], "delivered_count": 0, "skipped_count": 0,
               "total": 0, "repeat_groups": 0}
    images = {"images": [], "cap": 60, "truncated": False, "kind": "all"}
    mind = {"mood": None, "vibe": None, "vibe_history": [], "violations": [],
            "interests": [], "thoughts": []}
    traces = {"traces": [], "stats": {"total": 0, "count_24h": 0, "p50_ms": None,
                                      "p95_ms": None, "fallback_rate": 0.0},
              "window": 200, "cap": 50}

    monkeypatch.setattr(data, "health", lambda *a, **k: _ret(health))
    monkeypatch.setattr(data, "recent_persona_responses", lambda *a, **k: _ret(persona))
    monkeypatch.setattr(data, "recent_runs", lambda *a, **k: _ret([]))
    monkeypatch.setattr(data, "conversation_digest", lambda *a, **k: _ret(None))
    monkeypatch.setattr(data, "inner_monologue", lambda *a, **k: _ret([]))
    monkeypatch.setattr(data, "emotional_state", lambda *a, **k: _ret(None))
    monkeypatch.setattr(data, "recent_activity_blocks", lambda *a, **k: _ret([]))
    monkeypatch.setattr(data, "recent_chat", lambda *a, **k: _ret([]))
    monkeypatch.setattr(data, "recent_images", lambda *a, **k: _ret(images))
    monkeypatch.setattr(data, "mind", lambda *a, **k: _ret(mind))
    monkeypatch.setattr(data, "prose_traces", lambda *a, **k: _ret(traces))
    monkeypatch.setattr(
        data, "life", lambda *a, **k: _ret(_EMPTY_LIFE if empty else _life_fixture()))

    async def _events_page(category: Any = "all"):
        if empty:
            return _EMPTY_EVENTS
        return _events_fixture("food" if category == "food" else "all")

    async def _artifacts_page(atype: Any = "all", q: Any = ""):
        return _EMPTY_ARTIFACTS if empty else _artifacts_fixture()

    monkeypatch.setattr(data, "events_page", _events_page)
    monkeypatch.setattr(data, "artifacts_page", _artifacts_page)


@pytest.fixture
def client(monkeypatch) -> TestClient:
    _patch(monkeypatch)
    return TestClient(create_app())


@pytest.fixture
def empty_client(monkeypatch) -> TestClient:
    _patch(monkeypatch, empty=True)
    return TestClient(create_app())


# ── routes return 200, full and empty ─────────────────────────────────────────

ROUTES = ["/", "/partials/life", "/events", "/events?category=food",
          "/partials/events", "/partials/events?category=food",
          "/artifacts", "/artifacts?type=selfie&q=ramen",
          "/partials/artifacts", "/partials/artifacts?type=selfie&q=ramen"]


@pytest.mark.parametrize("route", ROUTES)
def test_routes_return_200(client, route):
    assert client.get(route).status_code == 200


@pytest.mark.parametrize("route", ROUTES)
def test_routes_empty_data_render(empty_client, route):
    assert empty_client.get(route).status_code == 200


# ── index wiring ──────────────────────────────────────────────────────────────


def test_index_has_life_tab_and_nav_links(client):
    html = client.get("/").text
    assert 'data-tab="life"' in html
    assert 'id="panel-life"' in html
    assert 'hx-get="/partials/life"' in html
    assert 'href="/events"' in html
    assert 'href="/artifacts"' in html


def test_index_life_panel_first_paint(client):
    html = client.get("/").text
    assert "ship v4" in html
    assert "fix the bot" in html


# ── life partial ──────────────────────────────────────────────────────────────


def test_life_partial_renders(client):
    html = client.get("/partials/life").text
    assert "ship v4" in html          # goals
    assert "goal: ship v4" in html    # todo linked-goal name
    assert "streak 4" in html         # habits
    assert "quad quad-do" in html     # priorities quadrant
    assert "Δ -0.40" in html          # misalignment flag


def test_life_partial_empty(empty_client):
    html = empty_client.get("/partials/life").text
    assert "No goals yet." in html
    assert "No active priorities yet." in html


# ── events ────────────────────────────────────────────────────────────────────


def test_events_page_renders_chart_and_timeline(client):
    html = client.get("/events").text
    assert "bar-fill" in html
    assert "width: 100%" in html
    assert "ramen" in html
    # event titles are escaped, never raw
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_events_category_param_forwarded_raw(monkeypatch, client):
    seen: list[Any] = []

    async def fake_events_page(category="all"):
        seen.append(category)
        return _EMPTY_EVENTS

    monkeypatch.setattr(data, "events_page", fake_events_page)
    assert client.get("/partials/events?category=food").status_code == 200
    assert client.get("/partials/events?category=../etc").status_code == 200
    # the route forwards raw input; validation lives in data.events_page
    assert seen == ["food", "../etc"]


def test_events_partial_daily_strip_for_selected_category(client):
    html = client.get("/partials/events?category=food").text
    assert "strip-col" in html
    assert "daily counts" in html


# ── artifacts ─────────────────────────────────────────────────────────────────


def test_artifacts_page_renders_gallery_escaped(client):
    html = client.get("/artifacts").text
    assert "selfie_agent" in html
    assert "art-badge art-c1" in html
    # the summary's <script> is escaped, never raw
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    # search form present, read-only (no write controls)
    assert 'name="q"' in html
    assert "cleanup" not in html.lower()


def test_artifacts_type_and_q_params_forwarded(monkeypatch, client):
    seen: list[tuple[Any, Any]] = []

    async def fake_artifacts_page(atype="all", q=""):
        seen.append((atype, q))
        return _EMPTY_ARTIFACTS

    monkeypatch.setattr(data, "artifacts_page", fake_artifacts_page)
    assert client.get("/partials/artifacts?type=selfie&q=ramen").status_code == 200
    assert client.get("/partials/artifacts").status_code == 200
    assert seen == [("selfie", "ramen"), ("all", "")]


def test_artifacts_q_reflected_escaped_not_executable(monkeypatch, client):
    async def fake_artifacts_page(atype="all", q=""):
        out = dict(_EMPTY_ARTIFACTS)
        out["q"] = '"><script>alert(1)</script>'
        return out

    monkeypatch.setattr(data, "artifacts_page", fake_artifacts_page)
    html = client.get('/partials/artifacts?q=x').text
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
