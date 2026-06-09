"""Web monitoring dashboard — route rendering + skip/delivered classification.

All DB access is mocked: every test patches the ``app.web.data`` accessors the
routes call, so the FastAPI TestClient exercises the real templates/HTMX wiring
without a live Postgres. Three things are guarded:

  1. every route returns 200 and renders its expected section markers,
  2. empty-data renders without raising (graceful empty tables),
  3. the persona_response skip-vs-delivered classification is correct.
"""

from __future__ import annotations

import os
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
    images = (
        {"images": [], "cap": 60, "truncated": False}
        if empty else {
            "images": [
                {"kind": "rendered", "name": "comfyui_dl_twily_selfie_00211_.png",
                 "mtime": NOW, "size": 1234, "prompt": "twily winks at the camera"},
                {"kind": "captures", "name": "cam_00042.jpg",
                 "mtime": NOW, "size": 5678, "prompt": ""},
            ],
            "cap": 60, "truncated": False,
        }
    )
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
    monkeypatch.setattr(data, "recent_images", lambda *a, **k: _ret(images))
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
          "/partials/runs", "/partials/context", "/partials/chat",
          "/partials/images"]


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


def test_index_renders_real_tab_bar(client):
    html = client.get("/").text
    # a real tab bar (role=tablist) with the four tabs
    assert 'class="tabbar"' in html
    assert 'role="tablist"' in html
    assert html.count('class="tab"') >= 5
    for name in ("proactive", "runs", "chat", "context", "images"):
        assert f'data-tab="{name}"' in html
    # the tab-mode CSS hook + the default tab the JS selects
    assert "has-tabs" in html
    assert 'DEFAULT = "proactive"' in html


def test_index_no_js_renders_all_panels_server_side(client):
    # Without JS, .has-tabs is never set, so every panel is present in the
    # server-rendered HTML (graceful no-JS fallback shows all panels).
    html = client.get("/").text
    for name in ("proactive", "runs", "chat", "context"):
        assert f'id="panel-{name}"' in html
    # all four panel bodies + their content are server-rendered on first paint
    assert "event_extractor-primary" in html  # runs panel body
    assert "Conversation Digest" in html       # context panel body
    # health strip stays outside the tab panels, always visible
    assert 'id="health"' in html


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


def _timeline_fixture() -> dict[str, Any]:
    return {
        "run": {
            "run_id": "run_t", "owner": "goals/periodic_checker-qwen3527b-primary",
            "status": "completed", "interaction_mode": "cron", "domain": "",
            "started_at": NOW, "completed_at": NOW, "contract_passed": True,
        },
        "trace": {
            "text": "checking…",
            "tool_calls": [],
            "tool_call_count": 1,
            "trajectory": [
                {"kind": "text", "text": "First I narrate my plan."},
                {"kind": "tool", "name": "bash",
                 "command": "python scripts/check.py", "error": None},
                {"kind": "result", "name": "bash", "status": "completed",
                 "output": "all clear", "error": None},
                {"kind": "text", "text": "Then I wrap up."},
            ],
            "trajectory_count": 4,
            "ok": True, "error": None, "producer": "runner",
        },
        "persona": [],
    }


def test_run_detail_renders_ordered_interleaved_timeline(client, monkeypatch):
    async def _ret(_run_id):
        return _timeline_fixture()

    monkeypatch.setattr(data, "run_detail", _ret)
    html = client.get("/run/run_t").text
    # all timeline pieces render
    assert "First I narrate my plan." in html
    assert "python scripts/check.py" in html
    assert "all clear" in html
    assert "Then I wrap up." in html
    # ORDER: the first narration must appear BEFORE the tool that followed it,
    # which must appear before the second narration (chronological timeline).
    i_narr1 = html.index("First I narrate my plan.")
    i_tool = html.index("python scripts/check.py")
    i_result = html.index("all clear")
    i_narr2 = html.index("Then I wrap up.")
    assert i_narr1 < i_tool < i_result < i_narr2
    # rendered as a timeline (not the flat "Tool calls" list)
    assert 'class="timeline"' in html


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


def test_run_detail_flat_trace_falls_back_to_old_view(client, monkeypatch):
    # An old run_trace with no `trajectory` key must still render via the flat
    # assistant-output + tool-calls view (graceful fallback).
    async def _ret(_run_id):
        return _trace_fixture()  # no "trajectory" key

    monkeypatch.setattr(data, "run_detail", _ret)
    html = client.get("/run/run_a").text
    # falls back: the flat "Tool calls" header + flat list, NOT the timeline
    assert "Tool calls" in html
    assert 'class="toollist"' in html
    assert 'class="timeline"' not in html
    assert "python scripts/list_due.py" in html


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


# ── image gallery: tab/partial rendering ──────────────────────────────────────


def test_index_has_images_tab_and_panel(client):
    html = client.get("/").text
    assert 'data-tab="images"' in html
    assert 'id="panel-images"' in html
    # the gallery partial auto-refreshes on its own (slower) cadence
    assert 'hx-get="/partials/images"' in html
    assert "every 30s" in html


def test_images_partial_lists_files_newest_first(client):
    html = client.get("/partials/images").text
    assert "comfyui_dl_twily_selfie_00211_.png" in html
    assert "cam_00042.jpg" in html
    # thumbnails point at the safe /media route
    assert "/media/rendered/comfyui_dl_twily_selfie_00211_.png" in html
    assert "/media/captures/cam_00042.jpg" in html
    # matched prompt preview surfaces
    assert "twily winks at the camera" in html
    # the cap is noted in the UI
    assert "60" in html


def test_images_partial_empty(empty_client):
    html = empty_client.get("/partials/images").text
    assert "no images yet" in html.lower()


# ── image gallery: data layer (filesystem listing + newest-first + cap) ───────


def _stub_data_dir(monkeypatch, tmp_path) -> None:
    """Point data_dir at a tmp dir and stub the context_cache enrichment off."""
    class _S:
        data_dir = tmp_path

    monkeypatch.setattr("app.settings.get_settings", lambda: _S())

    async def _no_meta(*_a, **_k):
        return {}

    monkeypatch.setattr(data, "_context_meta_by_filename", _no_meta)


def test_recent_images_lists_newest_first(monkeypatch, tmp_path):
    import asyncio
    import time

    rendered = tmp_path / "rendered"
    captures = tmp_path / "captures"
    rendered.mkdir()
    captures.mkdir()
    # three files with staggered mtimes (older → newer)
    old = rendered / "old.png"
    mid = captures / "mid.jpg"
    new = rendered / "new.webp"
    for i, f in enumerate((old, mid, new)):
        f.write_bytes(b"x")
        os.utime(f, (1_000_000 + i * 100, 1_000_000 + i * 100))
    # a non-image file must be ignored
    (rendered / "notes.txt").write_text("nope")

    _stub_data_dir(monkeypatch, tmp_path)
    result = asyncio.run(data.recent_images(limit=60))
    names = [im["name"] for im in result["images"]]
    assert names == ["new.webp", "mid.jpg", "old.png"]  # newest first
    assert "notes.txt" not in names
    assert result["truncated"] is False
    del time  # silence unused import if linter complains


def test_recent_images_caps_and_flags_truncated(monkeypatch, tmp_path):
    import asyncio

    rendered = tmp_path / "rendered"
    rendered.mkdir()
    (tmp_path / "captures").mkdir()
    for i in range(5):
        (rendered / f"img_{i:03d}.png").write_bytes(b"x")

    _stub_data_dir(monkeypatch, tmp_path)
    result = asyncio.run(data.recent_images(limit=3))
    assert len(result["images"]) == 3
    assert result["cap"] == 3
    assert result["truncated"] is True


def test_recent_images_missing_dirs_empty(monkeypatch, tmp_path):
    import asyncio

    # neither rendered/ nor captures/ exists under data_dir
    _stub_data_dir(monkeypatch, tmp_path)
    result = asyncio.run(data.recent_images())
    assert result["images"] == []
    assert result["truncated"] is False


def test_recent_images_enriches_with_context_prompt(monkeypatch, tmp_path):
    import asyncio

    rendered = tmp_path / "rendered"
    rendered.mkdir()
    (tmp_path / "captures").mkdir()
    (rendered / "selfie_1.png").write_bytes(b"x")

    class _S:
        data_dir = tmp_path

    monkeypatch.setattr("app.settings.get_settings", lambda: _S())

    async def _meta(*_a, **_k):
        return {"selfie_1.png": {"summary": "a cozy selfie", "created_at": NOW}}

    monkeypatch.setattr(data, "_context_meta_by_filename", _meta)
    result = asyncio.run(data.recent_images())
    assert result["images"][0]["prompt"] == "a cozy selfie"


# ── image gallery: /media route safety (traversal + extension) ────────────────


def test_media_serves_file_from_allowed_dir(monkeypatch, tmp_path):
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    (rendered / "ok.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    class _S:
        data_dir = tmp_path

    monkeypatch.setattr("app.settings.get_settings", lambda: _S())
    c = TestClient(create_app())
    r = c.get("/media/rendered/ok.png")
    assert r.status_code == 200
    assert r.content.startswith(b"\x89PNG")


@pytest.mark.parametrize(
    "url",
    [
        "/media/rendered/../../etc/passwd",
        "/media/rendered/..%2f..%2fetc%2fpasswd",
        "/media/captures/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "/media/rendered/sub%2fnested.png",
    ],
)
def test_media_rejects_path_traversal(monkeypatch, tmp_path, url):
    (tmp_path / "rendered").mkdir()
    (tmp_path / "captures").mkdir()

    class _S:
        data_dir = tmp_path

    monkeypatch.setattr("app.settings.get_settings", lambda: _S())
    c = TestClient(create_app())
    r = c.get(url)
    # never 200 — traversal is blocked (404 from our guard, or 404 from the
    # router refusing to match a path that decodes to extra segments)
    assert r.status_code in (403, 404)
    assert b"passwd" not in r.content.lower() or r.status_code in (403, 404)


def test_media_rejects_non_image_extension(monkeypatch, tmp_path):
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    (rendered / "secret.txt").write_text("top secret")

    class _S:
        data_dir = tmp_path

    monkeypatch.setattr("app.settings.get_settings", lambda: _S())
    c = TestClient(create_app())
    r = c.get("/media/rendered/secret.txt")
    assert r.status_code in (403, 404)


def test_media_rejects_unknown_kind(monkeypatch, tmp_path):
    (tmp_path / "rendered").mkdir()

    class _S:
        data_dir = tmp_path

    monkeypatch.setattr("app.settings.get_settings", lambda: _S())
    c = TestClient(create_app())
    r = c.get("/media/secrets/anything.png")
    assert r.status_code in (403, 404)


def test_safe_media_path_unit(monkeypatch, tmp_path):
    (tmp_path / "rendered").mkdir()

    class _S:
        data_dir = tmp_path

    monkeypatch.setattr("app.settings.get_settings", lambda: _S())
    # valid
    p = data.safe_media_path("rendered", "a.png")
    assert p is not None and p.name == "a.png"
    # traversal / separators / bad ext / unknown kind all rejected
    assert data.safe_media_path("rendered", "../x.png") is None
    assert data.safe_media_path("rendered", "sub/x.png") is None
    assert data.safe_media_path("rendered", "x.txt") is None
    assert data.safe_media_path("rendered", "") is None
    assert data.safe_media_path("nope", "x.png") is None
