"""Mind tab + LLM traces + image-filter port (v3 → v4 dashboard) — data layer
and template rendering.

These tests deliberately do NOT require fastapi (this venv may not have it):

  * data-shaping functions are unit-tested directly with fake repo rows,
  * the new templates are rendered through a plain Jinja2 environment
    (autoescape=True — the same policy starlette's Jinja2Templates uses), so
    escaping and empty-data behaviour are locked even without the app server.

Route-level tests (TestClient) live in test_web_routes_mind_traces.py behind
the same fastapi importorskip guard test_web_dashboard.py uses.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jinja2")

from jinja2 import Environment, FileSystemLoader  # noqa: E402

from app.web import data  # noqa: E402

NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
TEMPLATES_DIR = Path(data.__file__).resolve().parent / "templates"


def _env() -> Environment:
    # autoescape=True mirrors starlette's Jinja2Templates default, so what we
    # assert about escaping here holds in the real app.
    return Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)


def render(template: str, **ctx: Any) -> str:
    return _env().get_template(template).render(**ctx)


# ── freshness chips ───────────────────────────────────────────────────────────


def test_freshness_class_green_amber_red():
    assert data.freshness_class(NOW - timedelta(hours=1), now=NOW) == "ok"
    assert data.freshness_class(NOW - timedelta(hours=12), now=NOW) == "warn"
    assert data.freshness_class(NOW - timedelta(hours=30), now=NOW) == "bad"


def test_freshness_class_boundaries_and_missing():
    assert data.freshness_class(NOW - timedelta(hours=6), now=NOW) == "warn"
    assert data.freshness_class(NOW - timedelta(hours=24), now=NOW) == "bad"
    assert data.freshness_class(None) == ""
    assert data.freshness_class("not a datetime") == ""


def test_freshness_class_naive_datetime_treated_as_utc():
    naive = (NOW - timedelta(hours=2)).replace(tzinfo=None)
    assert data.freshness_class(naive, now=NOW) == "ok"


# ── human-readable sizes ──────────────────────────────────────────────────────


def test_human_size():
    assert data._human_size(0) == "0 B"
    assert data._human_size(999) == "999 B"
    assert data._human_size(2048) == "2.0 KB"
    assert data._human_size(5 * 1024 * 1024) == "5.0 MB"
    assert data._human_size(None) == ""
    assert data._human_size("junk") == ""


# ── image filter validation ───────────────────────────────────────────────────


@pytest.mark.parametrize("value,expected", [
    ("rendered", "rendered"),
    ("captures", "captures"),
    ("all", "all"),
    ("", "all"),
    (None, "all"),
    ("../../etc", "all"),
    ("RENDERED", "all"),
    (123, "all"),
])
def test_normalize_image_filter(value, expected):
    assert data.normalize_image_filter(value) == expected


def _stub_data_dir(monkeypatch, tmp_path) -> None:
    class _S:
        data_dir = tmp_path

    monkeypatch.setattr("app.settings.get_settings", lambda: _S())

    async def _no_meta(*_a, **_k):
        return {}

    monkeypatch.setattr(data, "_context_meta_by_filename", _no_meta)


def test_recent_images_kind_filter(monkeypatch, tmp_path):
    rendered = tmp_path / "rendered"
    captures = tmp_path / "captures"
    rendered.mkdir()
    captures.mkdir()
    (rendered / "r.png").write_bytes(b"x" * 2048)
    (captures / "c.jpg").write_bytes(b"x")

    _stub_data_dir(monkeypatch, tmp_path)

    out = asyncio.run(data.recent_images(kind="rendered"))
    assert [i["name"] for i in out["images"]] == ["r.png"]
    assert out["kind"] == "rendered"
    # size is humanised for the badge
    assert out["images"][0]["size_h"] == "2.0 KB"

    out = asyncio.run(data.recent_images(kind="captures"))
    assert [i["name"] for i in out["images"]] == ["c.jpg"]

    # unknown filter falls back to all
    out = asyncio.run(data.recent_images(kind="nope"))
    assert sorted(i["name"] for i in out["images"]) == ["c.jpg", "r.png"]
    assert out["kind"] == "all"


# ── mood / vibe shaping ───────────────────────────────────────────────────────


def _mood_row() -> dict[str, Any]:
    return {
        "chat_id": 42, "energy": 0.8, "valence": 0.6, "stress": 0.2,
        "engagement": 0.9, "openness": 0.5, "dominant_mood": "engaged",
        "last_trigger": "late-night coding spree", "drift_count": 17,
        "updated_at": NOW - timedelta(hours=1),
    }


def test_shape_mood():
    shaped = data.shape_mood(_mood_row())
    assert shaped is not None
    meters = {m["key"]: m for m in shaped["meters"]}
    assert set(meters) == {"energy", "valence", "stress", "engagement", "openness"}
    assert meters["energy"]["pct"] == 80
    assert meters["stress"]["value"] == 0.2
    assert shaped["dominant_mood"] == "engaged"
    assert shaped["last_trigger"] == "late-night coding spree"
    # freshness is computed against real now — only assert it's a valid class
    assert shaped["freshness"] in ("", "ok", "warn", "bad")


def test_shape_mood_none_and_garbage_values():
    assert data.shape_mood(None) is None
    shaped = data.shape_mood({"energy": "junk", "valence": None})
    assert shaped is not None
    meters = {m["key"]: m for m in shaped["meters"]}
    assert meters["energy"]["pct"] == 0
    assert meters["valence"]["pct"] == 0


def _vibe_row() -> dict[str, Any]:
    return {
        "chat_id": 42,
        "w_warm_snarky": 0.40, "w_dry_ironic": 0.15, "w_caring_edge": 0.15,
        "w_playful_flirt": 0.10, "w_debate_socratic": 0.20,
        "ironic_genuine_axis": -0.5, "arousal_axis": 0.0,
        "last_trigger": "user got sarcastic", "last_user_tone": "snark",
        "drift_count": 99, "updated_at": NOW,
    }


def test_shape_vibe():
    shaped = data.shape_vibe(_vibe_row())
    assert shaped is not None
    bars = {b["key"]: b for b in shaped["bars"]}
    assert set(bars) == {
        "w_warm_snarky", "w_dry_ironic", "w_caring_edge",
        "w_playful_flirt", "w_debate_socratic",
    }
    assert bars["w_warm_snarky"]["pct"] == 40
    labels = [b["label"] for b in shaped["bars"]]
    assert labels == ["warm-snarky", "dry-ironic", "caring-edge", "playful-flirt", "debate-socratic"]
    axes = {a["key"]: a for a in shaped["axes"]}
    # -1..+1 → 0..100 with 50 = neutral
    assert axes["ironic_genuine_axis"]["pct"] == 25
    assert axes["arousal_axis"]["pct"] == 50
    assert shaped["chat_id"] == 42


def test_shape_vibe_none():
    assert data.shape_vibe(None) is None


def test_shape_vibe_history_keeps_last_10():
    rows = [
        {"recorded_at": NOW, "trigger": f"t{i}", "user_tone": "calm",
         "w_warm_snarky": 0.4, "w_dry_ironic": 0.15, "w_caring_edge": 0.15,
         "w_playful_flirt": 0.1, "w_debate_socratic": 0.2}
        for i in range(15)
    ]
    shaped = data.shape_vibe_history(rows, keep=10)
    assert len(shaped) == 10
    assert shaped[-1]["trigger"] == "t14"
    assert len(shaped[0]["weights"]) == 5


# ── interests / pending thoughts shaping ──────────────────────────────────────


def test_shape_interest():
    shaped = data.shape_interest({
        "topic": "analog synthesizers", "stance": "intrigued", "source": "rss",
        "novelty_score": 0.73, "surface_count": 2,
        "last_surfaced_at": NOW, "created_at": NOW,
    })
    assert shaped["topic"] == "analog synthesizers"
    assert shaped["novelty"]["pct"] == 73
    assert shaped["surface_count"] == 2


def test_shape_thought_breakdown_dict_and_json_string():
    base = {
        "content": "ask about the synth build", "kind": "question",
        "motivation_score": 0.9, "created_at": NOW,
        "consumed_at": None, "consumed_by": None,
    }
    d = data.shape_thought({**base, "motivation_breakdown": {"novelty": 0.5, "recency": 0.4}})
    assert [p["key"] for p in d["breakdown"]] == ["novelty", "recency"]  # sorted desc
    s = data.shape_thought({**base, "motivation_breakdown": '{"novelty": 0.5}'})
    assert s["breakdown"] == [{"key": "novelty", "value": 0.5}]
    none = data.shape_thought({**base, "motivation_breakdown": None})
    assert none["breakdown"] == []
    assert none["motivation"]["pct"] == 90


# ── mind() aggregation with fake repos ────────────────────────────────────────


class _FakeMoodRepo:
    async def latest(self):
        return _mood_row()


class _FakeVibeRepo:
    async def latest(self):
        return _vibe_row()

    async def history(self, chat_id, *, limit=100):
        assert chat_id == 42
        return [{"recorded_at": NOW, "trigger": "x", "user_tone": "",
                 "w_warm_snarky": 0.4, "w_dry_ironic": 0.15, "w_caring_edge": 0.15,
                 "w_playful_flirt": 0.1, "w_debate_socratic": 0.2}]


class _FakeStyleRepo:
    async def count_by_type(self, chat_id, *, since_hours=24):
        assert chat_id == 42
        return [{"violation_type": "em_dash", "n": 3}]


class _FakeInterestsRepo:
    async def list_active(self, *, limit=50):
        return [{"topic": "synths", "stance": "", "source": "rss",
                 "novelty_score": 0.7, "surface_count": 1,
                 "last_surfaced_at": None, "created_at": NOW}]


class _FakeThoughtsRepo:
    async def list_recent(self, *, limit=20, consumed=None):
        return [{"content": "hmm", "kind": "muse", "motivation_score": 0.4,
                 "motivation_breakdown": {"novelty": 0.4}, "created_at": NOW,
                 "consumed_at": None, "consumed_by": None}]


class _EmptyRepo:
    async def latest(self):
        return None

    async def list_active(self, *, limit=50):
        return []

    async def list_recent(self, *, limit=20, consumed=None):
        return []


def _patch_mind_repos(monkeypatch, *, empty: bool = False) -> None:
    if empty:
        monkeypatch.setattr(data, "UserMoodRepo", _EmptyRepo)
        monkeypatch.setattr(data, "VibeStateRepo", _EmptyRepo)
        monkeypatch.setattr(data, "PersonaInterestsRepo", _EmptyRepo)
        monkeypatch.setattr(data, "PendingThoughtsRepo", _EmptyRepo)
    else:
        monkeypatch.setattr(data, "UserMoodRepo", _FakeMoodRepo)
        monkeypatch.setattr(data, "VibeStateRepo", _FakeVibeRepo)
        monkeypatch.setattr(data, "StyleEventsRepo", _FakeStyleRepo)
        monkeypatch.setattr(data, "PersonaInterestsRepo", _FakeInterestsRepo)
        monkeypatch.setattr(data, "PendingThoughtsRepo", _FakeThoughtsRepo)


def test_mind_aggregates_all_panels(monkeypatch):
    _patch_mind_repos(monkeypatch)
    mind = asyncio.run(data.mind())
    assert mind["mood"]["dominant_mood"] == "engaged"
    assert mind["vibe"]["chat_id"] == 42
    assert len(mind["vibe_history"]) == 1
    assert mind["violations"] == [{"violation_type": "em_dash", "n": 3}]
    assert mind["interests"][0]["topic"] == "synths"
    assert mind["thoughts"][0]["kind"] == "muse"


def test_mind_empty_tables(monkeypatch):
    _patch_mind_repos(monkeypatch, empty=True)
    mind = asyncio.run(data.mind())
    assert mind["mood"] is None
    assert mind["vibe"] is None
    assert mind["vibe_history"] == []
    assert mind["violations"] == []
    assert mind["interests"] == []
    assert mind["thoughts"] == []


# ── prose trace list / stats shaping ──────────────────────────────────────────


def test_shape_trace_row_parses_json_text_fields():
    shaped = data.shape_trace_row({
        "run_id": "r1", "created_at": NOW, "kind": "nudge",
        "model": "qwen3.5-27b", "duration_ms": "4231.7",
        "input_tokens": "5120", "output_tokens": "180", "fallback": "true",
    })
    assert shaped["duration_ms"] == 4231
    assert shaped["input_tokens"] == 5120
    assert shaped["fallback"] is True
    none = data.shape_trace_row({"run_id": "r2", "fallback": None})
    assert none["duration_ms"] is None
    assert none["fallback"] is False


def test_trace_stats_p50_p95_count_fallback():
    traces = [
        {"created_at": NOW - timedelta(hours=i), "duration_ms": (i + 1) * 100,
         "fallback": i == 0}
        for i in range(10)  # ages 0..9h → all within 24h
    ]
    traces.append({"created_at": NOW - timedelta(hours=48), "duration_ms": None, "fallback": False})
    stats = data.trace_stats(traces, now=NOW)
    assert stats["total"] == 11
    assert stats["count_24h"] == 10
    # durations 100..1000 → p50 ≈ 500/600 region, p95 ≈ 1000
    assert 400 <= stats["p50_ms"] <= 600
    assert stats["p95_ms"] == 1000
    assert stats["fallback_rate"] == round(1 / 11, 3)


def test_trace_stats_empty():
    stats = data.trace_stats([])
    assert stats == {"total": 0, "count_24h": 0, "p50_ms": None, "p95_ms": None,
                     "fallback_rate": 0.0}


def test_prose_traces_with_stubbed_fetch(monkeypatch):
    rows = [
        {"run_id": "r1", "created_at": NOW, "kind": "nudge", "model": "m",
         "duration_ms": "100", "input_tokens": "1", "output_tokens": "2",
         "fallback": "false"},
        {"run_id": "r2", "created_at": NOW - timedelta(hours=1), "kind": "reply",
         "model": "m", "duration_ms": "300", "input_tokens": "3",
         "output_tokens": "4", "fallback": "true"},
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

    out = asyncio.run(data.prose_traces(limit=50))
    assert [t["run_id"] for t in out["traces"]] == ["r1", "r2"]
    assert out["stats"]["total"] == 2
    assert out["stats"]["fallback_rate"] == 0.5


def test_shape_trace_detail_parses_string_payload():
    import json

    payload = {
        "kind": "nudge", "model": "qwen", "provider": "local",
        "system_prompt": "be kind", "messages": [{"role": "user", "content": "hi"}],
        "raw_output": "<think>secret</think>hello", "thinking": "secret",
        "stripped_output": "hello", "delivered_text": "hello",
        "duration_ms": 1200, "input_tokens": 10, "output_tokens": 5,
        "context_summary": {"history_msgs": 4}, "guidance": {"message_kind": "nudge"},
        "fallback_triggered": False,
    }
    shaped = data.shape_trace_detail(
        {"run_id": "r1", "producer": "persona_prose", "created_at": NOW,
         "payload": json.dumps(payload)}
    )
    assert shaped["model"] == "qwen"
    assert shaped["messages"] == [{"role": "user", "content": "hi"}]
    assert shaped["context_summary"] == {"history_msgs": 4}
    assert shaped["guidance"] == {"message_kind": "nudge"}
    assert data.shape_trace_detail(None) is None


# ── template rendering (direct Jinja2, no fastapi needed) ─────────────────────


def _full_mind() -> dict[str, Any]:
    return {
        "mood": data.shape_mood(_mood_row()),
        "vibe": data.shape_vibe(_vibe_row()),
        "vibe_history": data.shape_vibe_history([
            {"recorded_at": NOW, "trigger": "user joked", "user_tone": "warm",
             "w_warm_snarky": 0.5, "w_dry_ironic": 0.1, "w_caring_edge": 0.15,
             "w_playful_flirt": 0.05, "w_debate_socratic": 0.2},
        ]),
        "violations": [{"violation_type": "em_dash", "n": 3}],
        "interests": [data.shape_interest({
            "topic": "analog synthesizers", "stance": "curious", "source": "rss",
            "novelty_score": 0.73, "surface_count": 2,
            "last_surfaced_at": NOW, "created_at": NOW,
        })],
        "thoughts": [data.shape_thought({
            "content": "ask about the synth build", "kind": "question",
            "motivation_score": 0.9, "motivation_breakdown": {"novelty": 0.5},
            "created_at": NOW, "consumed_at": None, "consumed_by": None,
        })],
    }


_EMPTY_MIND = {"mood": None, "vibe": None, "vibe_history": [],
               "violations": [], "interests": [], "thoughts": []}


def test_mind_partial_renders_populated():
    html = render("partials/mind.html", mind=_full_mind())
    assert "dominant: engaged" in html
    assert "warm-snarky" in html
    assert "em_dash × 3" in html
    assert "analog synthesizers" in html
    assert "ask about the synth build" in html
    # meters render as width styles
    assert "width: 80%" in html  # energy 0.8
    # emotional core stays in Context tab — only linked
    assert "Context tab" in html


def test_mind_partial_renders_empty():
    html = render("partials/mind.html", mind=_EMPTY_MIND)
    assert "No user_mood_state yet." in html
    assert "No persona_vibe_state yet." in html
    assert "No persona_interests yet." in html
    assert "No pending_thoughts yet." in html


def test_mind_partial_escapes_thought_content():
    mind = dict(_EMPTY_MIND)
    mind["thoughts"] = [data.shape_thought({
        "content": "<script>alert(1)</script>", "kind": "muse",
        "motivation_score": 0.5, "motivation_breakdown": None,
        "created_at": NOW, "consumed_at": None, "consumed_by": None,
    })]
    html = render("partials/mind.html", mind=mind)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def _traces_ctx(traces=None) -> dict[str, Any]:
    traces = traces if traces is not None else []
    return {"traces": {"traces": traces, "stats": data.trace_stats(traces, now=NOW),
                       "window": 200, "cap": 50}}


def test_traces_partial_renders_list_and_stats():
    rows = [data.shape_trace_row({
        "run_id": "r1", "created_at": NOW, "kind": "nudge", "model": "qwen3.5",
        "duration_ms": "100", "input_tokens": "10", "output_tokens": "5",
        "fallback": "true",
    })]
    html = render("partials/traces.html", **_traces_ctx(rows))
    assert "/traces/r1" in html
    assert "qwen3.5" in html
    assert "FALLBACK" in html
    assert "p50" in html and "p95" in html and "fallback" in html


def test_traces_partial_renders_empty():
    html = render("partials/traces.html", **_traces_ctx([]))
    assert "No persona_prose_trace artifacts yet." in html


def test_traces_page_renders_with_polling():
    html = render("traces.html", **_traces_ctx([]))
    assert 'hx-get="/partials/traces"' in html
    assert "every 45s" in html


def test_trace_detail_renders_and_escapes_script():
    trace = data.shape_trace_detail({
        "run_id": "r1", "producer": "persona_prose", "created_at": NOW,
        "payload": {
            "kind": "nudge", "model": "qwen", "provider": "local",
            "system_prompt": "<script>alert('sys')</script>",
            "messages": [{"role": "user", "content": "<script>alert('msg')</script>"}],
            "raw_output": "<script>alert('raw')</script>",
            "thinking": "pondering <b>hard</b>",
            "stripped_output": "clean text", "delivered_text": "clean text",
            "duration_ms": 1200, "input_tokens": 10, "output_tokens": 5,
            "context_summary": {"history_msgs": 4},
            "guidance": {"message_kind": "nudge", "raw_data": "<script>x</script>"},
            "fallback_triggered": True,
        },
    })
    html = render("trace_detail.html", run_id="r1", trace=trace, missing=False)
    # nothing executable leaks through
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    # all the audit sections are present
    assert "system prompt" in html
    assert "raw output" in html
    assert "delivered text" in html
    assert "guidance payload" in html
    assert "context summary" in html
    assert "history_msgs" in html
    assert "FALLBACK TRIGGERED" in html
    # no polling on the detail page
    assert "hx-trigger" not in html


def test_trace_detail_renders_missing():
    html = render("trace_detail.html", run_id="nope", trace=None, missing=True)
    assert "Trace not found" in html


def test_trace_detail_renders_minimal_payload():
    # a sparse/old trace (most fields absent) must not explode
    trace = data.shape_trace_detail({"run_id": "r2", "producer": "", "created_at": None,
                                     "payload": {}})
    html = render("trace_detail.html", run_id="r2", trace=trace, missing=False)
    assert "Nothing delivered" in html
    assert "No system prompt captured." in html
    assert "No messages captured." in html


def _images_ctx(*, kind="all", items=None, truncated=False):
    return {"images": {"images": items or [], "cap": 60, "truncated": truncated,
                       "kind": kind}}


def test_images_partial_renders_filters_size_and_lightbox():
    items = [{"kind": "rendered", "name": "a.png", "mtime": NOW, "size": 2048,
              "size_h": "2.0 KB", "prompt": "a selfie"}]
    html = render("partials/images.html", **_images_ctx(kind="rendered", items=items))
    # filter buttons with the active one marked
    assert 'hx-get="/partials/images?kind=rendered"' in html
    assert 'hx-get="/partials/images?kind=captures"' in html
    # self-polling root carries the active filter
    assert 'hx-trigger="every 30s"' in html
    assert "imgfilter active" in html
    # size badge + lightbox markup
    assert "2.0 KB" in html
    assert 'class="lightbox"' in html
    assert 'id="lb-1"' in html
    assert 'href="#lb-1"' in html


def test_images_partial_renders_empty_with_filter_note():
    html = render("partials/images.html", **_images_ctx(kind="captures"))
    assert "No images yet" in html
    assert "(filter: captures)" in html


def test_images_partial_renders_without_kind_key():
    # the index route's first paint passes the same dict recent_images returns,
    # but older fixtures/tests omit `kind` — must default to all w/o blowing up
    html = render("partials/images.html",
                  images={"images": [], "cap": 60, "truncated": False})
    assert "No images yet" in html


def test_health_partial_renders_freshness_chips():
    health = {
        "db_ok": True, "chat_count": 1, "run_count": 1, "persona_count": 1,
        "last_run_at": NOW, "last_chat_at": NOW, "qwen_url": "http://x",
        "mood_updated_at": NOW, "mood_fresh": "ok",
        "vibe_updated_at": NOW, "vibe_fresh": "warn",
        "interests_updated_at": None, "interests_fresh": "",
    }
    html = render("partials/health.html", health=health)
    assert "mood" in html
    assert "vibe" in html
    assert "interests" in html
    assert 'class="pill ok">mood' in html.replace("\n", "")
    assert 'class="pill warn">vibe' in html.replace("\n", "")


def test_health_partial_renders_without_new_keys():
    # graceful with the old health shape (e.g. cached fixtures)
    health = {"db_ok": True, "chat_count": 0, "run_count": 0, "persona_count": 0,
              "last_run_at": None, "last_chat_at": None, "qwen_url": ""}
    html = render("partials/health.html", health=health)
    assert "db reachable" in html
