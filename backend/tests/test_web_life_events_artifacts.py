"""Life tab + /events + /artifacts (v3 collected-data pages → v4 dashboard) —
data layer and template rendering.

Same dual pattern as test_web_mind_traces.py — these tests deliberately do NOT
require fastapi (this venv may not have it):

  * data-shaping functions are unit-tested directly with fake repo rows
    (frozen ``NOW`` so overdue/today bucketing is deterministic),
  * the new templates are rendered through a plain Jinja2 environment
    (autoescape=True — the same policy starlette's Jinja2Templates uses), so
    escaping and empty-data behaviour are locked even without the app server.

Route-level tests (TestClient) live in test_web_routes_life_events_artifacts.py
behind the same fastapi importorskip guard test_web_dashboard.py uses.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jinja2")

from jinja2 import Environment, FileSystemLoader  # noqa: E402

from app.web import data  # noqa: E402

NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
TODAY = NOW.date()
TEMPLATES_DIR = Path(data.__file__).resolve().parent / "templates"


def _env() -> Environment:
    # autoescape=True mirrors starlette's Jinja2Templates default, so what we
    # assert about escaping here holds in the real app.
    return Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)


def render(template: str, **ctx: Any) -> str:
    return _env().get_template(template).render(**ctx)


# ── goal hierarchy shaping ────────────────────────────────────────────────────


def _goal(gid: str, level: int = 1, parent: str | None = None, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "goal_id": gid, "level": level, "title": f"goal {gid}",
        "parent_goal_id": parent, "status": "active", "priority": "medium",
        "progress_percent": 40, "deadline": None,
    }
    row.update(over)
    return row


def test_shape_goal_tree_parent_child_order_and_depth():
    rows = [
        _goal("g1", 1),
        _goal("g2", 1),
        _goal("g1a", 2, parent="g1"),
        _goal("g1a1", 3, parent="g1a"),
    ]
    tree = data.shape_goal_tree(rows, now=NOW)
    assert [g["goal_id"] for g in tree] == ["g1", "g1a", "g1a1", "g2"]
    assert [g["depth"] for g in tree] == [0, 1, 2, 0]


def test_shape_goal_tree_orphans_do_not_crash():
    # parent_goal_id points at a goal that's missing (archived/deleted)
    rows = [
        _goal("g1", 1),
        _goal("orphan", 3, parent="ghost-never-existed"),
    ]
    tree = data.shape_goal_tree(rows, now=NOW)
    ids = [g["goal_id"] for g in tree]
    assert ids == ["g1", "orphan"]  # orphan becomes a root, nothing lost
    orphan = tree[1]
    assert orphan["depth"] == 2  # indented by its own level (3 → depth 2)


def test_shape_goal_tree_self_parent_and_cycle_terminate():
    rows = [
        _goal("selfie", 2, parent="selfie"),       # self-parent → root
        _goal("a", 2, parent="b"),                 # a ↔ b pure cycle
        _goal("b", 2, parent="a"),
    ]
    tree = data.shape_goal_tree(rows, now=NOW)
    assert sorted(g["goal_id"] for g in tree) == ["a", "b", "selfie"]  # each once


def test_shape_goal_tree_depth_capped_at_six_levels():
    rows = [_goal("g0", 1)]
    for i in range(1, 9):  # build a 9-deep chain (deeper than the 6 levels)
        rows.append(_goal(f"g{i}", min(i + 1, 6), parent=f"g{i - 1}"))
    tree = data.shape_goal_tree(rows, now=NOW)
    assert max(g["depth"] for g in tree) == data.GOAL_MAX_DEPTH  # capped at 5


def test_shape_goal_overdue_and_progress_clamp():
    overdue = data.shape_goal(
        _goal("g", deadline=NOW - timedelta(days=1), progress_percent=250), now=NOW)
    assert overdue["overdue"] is True
    assert overdue["progress"]["pct"] == 100  # clamped
    future = data.shape_goal(_goal("g", deadline=NOW + timedelta(days=1)), now=NOW)
    assert future["overdue"] is False
    # completed goals are never "overdue", even past deadline
    done = data.shape_goal(
        _goal("g", status="completed", deadline=NOW - timedelta(days=1)), now=NOW)
    assert done["overdue"] is False
    junk = data.shape_goal(_goal("g", progress_percent="junk"), now=NOW)
    assert junk["progress"]["pct"] == 0


# ── todo bucketing (frozen now) ───────────────────────────────────────────────


def _todo(tid: str, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "todo_id": tid, "title": f"todo {tid}", "status": "pending",
        "priority": "medium", "category": "personal",
        "linked_goal_id": None, "deadline": None, "date": TODAY,
    }
    row.update(over)
    return row


def test_bucket_todos_overdue_today_upcoming():
    rows = [
        _todo("past-date", date=TODAY - timedelta(days=2)),
        _todo("past-deadline", deadline=NOW - timedelta(hours=3)),
        _todo("today-date", date=TODAY),
        _todo("today-deadline", deadline=NOW + timedelta(hours=5)),
        _todo("soon", date=TODAY + timedelta(days=3)),
        _todo("edge-7d", date=TODAY + timedelta(days=7)),
        _todo("too-far", date=TODAY + timedelta(days=8)),
        _todo("done", status="completed", date=TODAY),
    ]
    buckets = data.bucket_todos(rows, now=NOW)
    assert [t["todo_id"] for t in buckets["overdue"]] == ["past-date", "past-deadline"]
    assert [t["todo_id"] for t in buckets["today"]] == ["today-date", "today-deadline"]
    assert [t["todo_id"] for t in buckets["upcoming"]] == ["soon", "edge-7d"]
    # too-far (beyond 7 days) and completed rows are dropped entirely
    all_ids = [t["todo_id"] for b in buckets.values() for t in b]
    assert "too-far" not in all_ids
    assert "done" not in all_ids


def test_bucket_todos_deadline_earlier_today_is_overdue():
    # a deadline that already passed TODAY counts as overdue, not "today"
    rows = [_todo("missed", deadline=NOW - timedelta(minutes=10))]
    buckets = data.bucket_todos(rows, now=NOW)
    assert [t["todo_id"] for t in buckets["overdue"]] == ["missed"]
    assert buckets["today"] == []


def test_bucket_todos_naive_deadline_treated_as_utc_and_sorted():
    rows = [
        _todo("b", deadline=(NOW - timedelta(hours=1)).replace(tzinfo=None)),
        _todo("a", date=TODAY - timedelta(days=5)),
    ]
    buckets = data.bucket_todos(rows, now=NOW)
    # both overdue; date-based 5-days-ago sorts before today's missed deadline
    assert [t["todo_id"] for t in buckets["overdue"]] == ["a", "b"]
    assert buckets["overdue"][1]["due_label"]  # deadline label rendered


def test_bucket_todos_empty_and_garbage_rows():
    assert data.bucket_todos([], now=NOW) == {"overdue": [], "today": [], "upcoming": []}
    # a row with neither deadline nor date must not crash (dropped)
    buckets = data.bucket_todos([_todo("x", date=None)], now=NOW)
    assert all(not b for b in buckets.values())


# ── habits shaping ────────────────────────────────────────────────────────────


def test_shape_habits_rates_and_due_today():
    habits = [
        {"habit_id": "h1", "title": "gym", "frequency_type": "daily",
         "current_streak": 4, "best_streak": 9},
        {"habit_id": "h2", "title": "read", "frequency_type": "daily",
         "current_streak": 0, "best_streak": 2},
    ]
    occ = [{"habit_id": "h1", "habit_title": "gym", "importance_level": 5}]
    stats = [{"habit_id": "h1", "scheduled": 30, "completed": 24},
             {"habit_id": "h2", "scheduled": 0, "completed": 0}]
    shaped = data.shape_habits(habits, occ, stats)
    assert shaped["due_today"] == [{"habit_id": "h1", "title": "gym", "importance_level": 5}]
    h1, h2 = shaped["habits"]
    assert h1["rate"] == {"pct": 80, "completed": 24, "scheduled": 30}
    assert h1["due_today"] is True
    assert h2["rate"] is None  # zero scheduled → no rate, not a div-by-zero
    assert h2["due_today"] is False


def test_shape_habits_empty():
    assert data.shape_habits([], [], []) == {"due_today": [], "habits": []}


# ── priorities quadrant assignment ────────────────────────────────────────────


@pytest.mark.parametrize("importance,immediacy,expected", [
    (0.9, 0.9, "do"),
    (0.9, 0.1, "plan"),
    (0.1, 0.9, "delegate"),
    (0.1, 0.1, "drop"),
    (0.5, 0.5, "do"),          # threshold boundary is inclusive
    (0.49, 0.5, "delegate"),
    (Decimal("0.80"), Decimal("0.20"), "plan"),  # NUMERIC arrives as Decimal
    (None, None, "drop"),       # garbage → low/low, never crashes
])
def test_assign_quadrant(importance, immediacy, expected):
    assert data.assign_quadrant(importance, immediacy) == expected


def test_shape_priorities_quadrants_and_misalignment():
    rows = [
        {"priority_id": "p1", "title": "ship v1", "category": "work",
         "importance": Decimal("0.9"), "immediacy": Decimal("0.8"),
         "real_importance": Decimal("0.5"), "importance_delta": Decimal("-0.4")},
        {"priority_id": "p2", "title": "inbox zero", "category": "",
         "importance": Decimal("0.2"), "immediacy": Decimal("0.9"),
         "real_importance": None, "importance_delta": None},
        # delta column absent (older snapshot) → recomputed from real-importance
        {"priority_id": "p3", "title": "health", "category": "",
         "importance": 0.6, "immediacy": 0.1, "real_importance": 0.8},
        # audited but within tolerance → not flagged
        {"priority_id": "p4", "title": "stable", "category": "",
         "importance": 0.6, "immediacy": 0.6, "real_importance": 0.65,
         "importance_delta": 0.05},
    ]
    shaped = data.shape_priorities(rows)
    assert [p["priority_id"] for p in shaped["quadrants"]["do"]] == ["p1", "p4"]
    assert [p["priority_id"] for p in shaped["quadrants"]["delegate"]] == ["p2"]
    assert [p["priority_id"] for p in shaped["quadrants"]["plan"]] == ["p3"]
    assert shaped["quadrants"]["drop"] == []
    by_id = {p["priority_id"]: p for q in shaped["quadrants"].values() for p in q}
    assert by_id["p1"]["misaligned"] is True       # |−0.4| ≥ 0.15
    assert by_id["p2"]["misaligned"] is False      # never audited
    assert by_id["p3"]["misaligned"] is True       # recomputed 0.8−0.6 = +0.2
    assert by_id["p4"]["misaligned"] is False      # |0.05| < 0.15
    assert shaped["misaligned"] == 2
    assert shaped["total"] == 4


def test_shape_priorities_empty():
    shaped = data.shape_priorities([])
    assert shaped["total"] == 0
    assert all(v == [] for v in shaped["quadrants"].values())


# ── life() aggregation with fake repos ────────────────────────────────────────


class _FakeGoalsRepo:
    async def list_with_children(self, *, limit=300):
        return [_goal("g1", 1), _goal("g1a", 2, parent="g1")]


class _FakeTodosRepo:
    async def list(self, *, status=None, limit=300, **_k):
        assert status == "pending"
        return [_todo("t1", linked_goal_id="g1a", date=date.today())]


class _FakeHabitsRepo:
    async def list(self, *, status=None, limit=100, **_k):
        return [{"habit_id": "h1", "title": "gym", "frequency_type": "daily",
                 "current_streak": 3, "best_streak": 5}]

    async def get_due_today(self):
        return [{"habit_id": "h1", "habit_title": "gym", "importance_level": 4}]

    async def completion_stats(self, *, days=30):
        return [{"habit_id": "h1", "scheduled": 10, "completed": 7}]


class _FakePrioritiesRepo:
    async def list(self, *, status=None, limit=100, **_k):
        return [{"priority_id": "p1", "title": "ship", "category": "",
                 "importance": 0.9, "immediacy": 0.9,
                 "real_importance": 0.5, "importance_delta": -0.4}]


class _BoomRepo:
    """Every method raises — life() must degrade per-panel, not explode."""

    def __getattr__(self, name):
        async def _boom(*_a, **_k):
            raise RuntimeError("db down")

        return _boom


def test_life_aggregates_all_panels(monkeypatch):
    monkeypatch.setattr(data, "GoalsRepo", _FakeGoalsRepo)
    monkeypatch.setattr(data, "TodosRepo", _FakeTodosRepo)
    monkeypatch.setattr(data, "HabitsRepo", _FakeHabitsRepo)
    monkeypatch.setattr(data, "PrioritiesRepo", _FakePrioritiesRepo)
    life = asyncio.run(data.life())
    assert [g["goal_id"] for g in life["goals"]] == ["g1", "g1a"]
    # linked-goal title resolved from the goals listing
    today_ids = [t["todo_id"] for t in life["todos"]["today"]]
    assert today_ids == ["t1"]
    assert life["todos"]["today"][0]["goal_title"] == "goal g1a"
    assert life["habits"]["habits"][0]["rate"]["pct"] == 70
    assert life["priorities"]["quadrants"]["do"][0]["misaligned"] is True


def test_life_degrades_per_panel_on_db_errors(monkeypatch):
    monkeypatch.setattr(data, "GoalsRepo", _BoomRepo)
    monkeypatch.setattr(data, "TodosRepo", _BoomRepo)
    monkeypatch.setattr(data, "HabitsRepo", _BoomRepo)
    monkeypatch.setattr(data, "PrioritiesRepo", _BoomRepo)
    life = asyncio.run(data.life())
    assert life["goals"] == []
    assert life["todos"] == {"overdue": [], "today": [], "upcoming": []}
    assert life["habits"] == {"due_today": [], "habits": []}
    assert life["priorities"]["total"] == 0


# ── events: category validation + bar chart math + daily strip ────────────────


def test_normalize_event_category():
    cats = ["sleep", "food", "mood"]
    assert data.normalize_event_category("food", cats) == "food"
    assert data.normalize_event_category("nope", cats) == "all"
    assert data.normalize_event_category("FOOD", cats) == "all"   # exact match only
    assert data.normalize_event_category(None, cats) == "all"
    assert data.normalize_event_category(123, cats) == "all"
    assert data.normalize_event_category("sleep", []) == "all"    # empty DB


def test_category_bars_max_scaling():
    bars = data.category_bars([
        {"category": "sleep", "count": 10},
        {"category": "food", "count": 5},
        {"category": "mood", "count": 0},
    ])
    assert [(b["category"], b["pct"]) for b in bars] == [
        ("sleep", 100), ("food", 50), ("mood", 0),
    ]


def test_category_bars_empty_and_garbage():
    assert data.category_bars([]) == []
    bars = data.category_bars([{"category": None, "count": 3},
                               {"category": "ok", "count": "7"}])
    assert bars == [{"category": "ok", "count": 7, "pct": 100}]


def test_daily_strip_fills_gaps_and_scales():
    rows = [
        {"date": TODAY, "count": 4},
        {"date": TODAY - timedelta(days=2), "count": 2},
    ]
    strip = data.daily_strip(rows, days=6, now=NOW)
    assert len(strip["days"]) == 7  # today + previous 6
    assert strip["days"][-1] == {"date": TODAY.isoformat(), "count": 4, "pct": 100}
    assert strip["days"][-3]["pct"] == 50
    assert strip["days"][0]["count"] == 0  # gap filled
    assert strip["max"] == 4
    assert strip["total"] == 6


def test_daily_strip_empty():
    strip = data.daily_strip([], days=6, now=NOW)
    assert strip["max"] == 0
    assert all(d["pct"] == 0 for d in strip["days"])


def test_shape_event_row_detail_condensed():
    shaped = data.shape_event_row({
        "event_id": "e1", "occurred_at": NOW, "category": "food",
        "subcategory": "lunch", "title": "ramen", "value": "650", "unit": "kcal",
        "cost": Decimal("12.5"), "currency": "EUR", "duration_minutes": 25,
        "source": "extracted",
    })
    assert shaped["detail"] == "650 kcal · 12.5 EUR · 25 min"
    bare = data.shape_event_row({"event_id": "e2", "title": "nap"})
    assert bare["detail"] == ""


class _FakeEventsRepo:
    calls: list[tuple[str, Any]] = []

    async def count_by_category(self, *, days=30):
        type(self).calls.append(("count", days))
        return [{"category": "sleep", "count": 8}, {"category": "food", "count": 4}]

    async def list(self, *, category=None, date_from=None, limit=200, **_k):
        type(self).calls.append(("list", category))
        return [{"event_id": "e1", "occurred_at": NOW, "category": category or "sleep",
                 "title": "slept", "source": "extracted"}]

    async def daily_counts(self, category, *, days=30):
        type(self).calls.append(("daily", category))
        return [{"date": TODAY, "count": 2}]


def test_events_page_known_category(monkeypatch):
    _FakeEventsRepo.calls = []
    monkeypatch.setattr(data, "EventsRepo", _FakeEventsRepo)
    out = asyncio.run(data.events_page("food"))
    assert out["category"] == "food"
    assert out["categories"] == ["sleep", "food"]
    assert out["strip"] is not None
    assert ("list", "food") in _FakeEventsRepo.calls
    assert ("daily", "food") in _FakeEventsRepo.calls


def test_events_page_unknown_category_falls_back_to_all(monkeypatch):
    _FakeEventsRepo.calls = []
    monkeypatch.setattr(data, "EventsRepo", _FakeEventsRepo)
    out = asyncio.run(data.events_page("../etc/passwd"))
    assert out["category"] == "all"
    assert out["strip"] is None                     # no strip for "all"
    assert ("list", None) in _FakeEventsRepo.calls  # unfiltered SQL
    assert all(c[0] != "daily" for c in _FakeEventsRepo.calls)


# ── artifacts: param validation + shaping ─────────────────────────────────────


def test_normalize_artifact_type():
    types = ["selfie", "research_report"]
    assert data.normalize_artifact_type("selfie", types) == "selfie"
    assert data.normalize_artifact_type("evil'; DROP", types) == "all"
    assert data.normalize_artifact_type(None, types) == "all"
    assert data.normalize_artifact_type("selfie", []) == "all"


def test_normalize_search_q():
    assert data.normalize_search_q("  ramen   shop  ") == "ramen shop"
    assert data.normalize_search_q(None) == ""
    assert data.normalize_search_q(123) == ""
    assert data.normalize_search_q("x" * 500) == "x" * data.SEARCH_Q_MAX_LEN
    # wildcard chars survive normalisation (escaped later, in the repo SQL)
    assert data.normalize_search_q("100%_done") == "100%_done"


def test_artifact_type_class_stable_and_bounded():
    c1 = data.artifact_type_class("selfie")
    assert c1 == data.artifact_type_class("selfie")  # stable across calls
    assert c1.startswith("art-c")
    assert 0 <= int(c1.removeprefix("art-c")) < data.ARTIFACT_PALETTE_SIZE
    assert data.artifact_type_class(None).startswith("art-c")


def test_shape_artifact_preview_tags_expiry():
    long_summary = "s" * 300
    shaped = data.shape_artifact({
        "cache_id": "ctx_1", "artifact_type": "research_report",
        "summary": long_summary, "tags": '["news", "ai"]',
        "entity_type": "topic", "entity_id": "t1", "source_agent": "researcher",
        "created_at": NOW - timedelta(hours=2), "expires_at": NOW - timedelta(hours=1),
    }, now=NOW)
    assert shaped["has_more"] is True
    assert shaped["preview"] == "s" * data.ARTIFACT_SUMMARY_PREVIEW
    assert shaped["tags"] == ["news", "ai"]       # JSON-string tags parsed
    assert shaped["expired"] is True
    assert shaped["entity"] == "topic:t1"
    short = data.shape_artifact({"summary": "hi", "tags": ["a"]}, now=NOW)
    assert short["has_more"] is False
    assert short["expired"] is False


class _FakeCacheRepo:
    calls: list[dict[str, Any]] = []

    async def distinct_types(self):
        return [{"artifact_type": "selfie", "n": 9},
                {"artifact_type": "research_report", "n": 3}]

    async def list_newest(self, *, artifact_type=None, q=None, limit=100):
        type(self).calls.append({"artifact_type": artifact_type, "q": q})
        return [{"cache_id": "c1", "artifact_type": artifact_type or "selfie",
                 "summary": f"match {q or ''}".strip(), "tags": [],
                 "created_at": NOW}]


def test_artifacts_page_filters_and_search(monkeypatch):
    _FakeCacheRepo.calls = []
    monkeypatch.setattr(data, "ContextCacheRepo", _FakeCacheRepo)
    out = asyncio.run(data.artifacts_page("selfie", "  ramen  "))
    assert out["type"] == "selfie"
    assert out["q"] == "ramen"
    assert _FakeCacheRepo.calls == [{"artifact_type": "selfie", "q": "ramen"}]
    assert [t["name"] for t in out["types"]] == ["selfie", "research_report"]


def test_artifacts_page_unknown_type_falls_back(monkeypatch):
    _FakeCacheRepo.calls = []
    monkeypatch.setattr(data, "ContextCacheRepo", _FakeCacheRepo)
    out = asyncio.run(data.artifacts_page("nope'; --", None))
    assert out["type"] == "all"
    assert out["q"] == ""
    assert _FakeCacheRepo.calls == [{"artifact_type": None, "q": None}]


# ── template rendering (direct Jinja2, no fastapi needed) ─────────────────────


_EMPTY_LIFE: dict[str, Any] = {
    "goals": [],
    "todos": {"overdue": [], "today": [], "upcoming": []},
    "habits": {"due_today": [], "habits": []},
    "priorities": {"quadrants": {"do": [], "plan": [], "delegate": [], "drop": []},
                   "misaligned": 0, "total": 0},
}


def _full_life() -> dict[str, Any]:
    goals = data.shape_goal_tree([
        _goal("g1", 1, progress_percent=60),
        _goal("g1a", 2, parent="g1", deadline=NOW - timedelta(days=2)),
    ], now=NOW)
    todos = data.bucket_todos([
        _todo("t-over", date=TODAY - timedelta(days=1), priority="high"),
        _todo("t-today", date=TODAY, linked_goal_id="g1"),
        _todo("t-up", date=TODAY + timedelta(days=2)),
    ], now=NOW)
    todos["today"][0]["goal_title"] = "goal g1"
    habits = data.shape_habits(
        [{"habit_id": "h1", "title": "gym", "frequency_type": "daily",
          "current_streak": 4, "best_streak": 9}],
        [{"habit_id": "h1", "habit_title": "gym", "importance_level": 5}],
        [{"habit_id": "h1", "scheduled": 30, "completed": 24}],
    )
    priorities = data.shape_priorities([
        {"priority_id": "p1", "title": "ship v1", "category": "work",
         "importance": 0.9, "immediacy": 0.9,
         "real_importance": 0.5, "importance_delta": -0.4},
        {"priority_id": "p2", "title": "someday", "category": "",
         "importance": 0.1, "immediacy": 0.1,
         "real_importance": None, "importance_delta": None},
    ])
    return {"goals": goals, "todos": todos, "habits": habits, "priorities": priorities}


def test_life_partial_renders_populated():
    html = render("partials/life.html", life=_full_life())
    # goal hierarchy: child indented, overdue deadline flagged red
    assert "goal g1" in html
    assert "margin-left: 18px" in html
    assert "when overdue" in html
    assert "width: 60%" in html            # progress bar
    # todo buckets with priority badge + linked-goal name
    assert "todo t-over" in html
    assert "tag prio high" in html
    assert "goal: goal g1" in html
    # habits: streak + 30d rate bar
    assert "streak 4" in html
    assert "80%" in html
    # priorities: 2x2 quadrants + misalignment flag
    assert "quad quad-do" in html
    assert "quad quad-drop" in html
    assert "ship v1" in html
    assert "Δ -0.40" in html
    assert "1 misaligned" in html


def test_life_partial_renders_empty():
    html = render("partials/life.html", life=_EMPTY_LIFE)
    assert "No goals yet." in html
    assert "Nothing overdue." in html
    assert "Nothing due today." in html
    assert "No active habits yet." in html
    assert "No active priorities yet." in html


def test_life_partial_escapes_titles():
    life = dict(_EMPTY_LIFE)
    life["goals"] = data.shape_goal_tree(
        [_goal("g1", title="<script>alert(1)</script>")], now=NOW)
    html = render("partials/life.html", life=life)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def _events_ctx(**over: Any) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "category": "all", "categories": [], "bars": [], "events": [],
        "strip": None, "days": data.EVENT_TIMELINE_DAYS,
        "chart_days": data.EVENT_CHART_DAYS, "cap": data.EVENT_TIMELINE_CAP,
    }
    ctx.update(over)
    return {"events": ctx}


def test_events_partial_renders_populated():
    bars = data.category_bars([{"category": "sleep", "count": 10},
                               {"category": "food", "count": 5}])
    strip = data.daily_strip([{"date": TODAY, "count": 2}], days=6, now=NOW)
    rows = [data.shape_event_row({"event_id": "e1", "occurred_at": NOW,
                                  "category": "food", "title": "ramen",
                                  "value": "650", "unit": "kcal",
                                  "source": "extracted"})]
    html = render("partials/events.html", **_events_ctx(
        category="food", categories=["sleep", "food"], bars=bars,
        strip=strip, events=rows))
    # filter buttons + active marking; poll URL carries the filter
    assert 'hx-get="/partials/events?category=food"' in html
    assert 'hx-trigger="every 60s"' in html
    assert "imgfilter active" in html
    # bar chart scaled to max
    assert "width: 100%" in html and "width: 50%" in html
    # daily strip columns
    assert "strip-col" in html
    assert "height: 100%" in html
    # timeline row
    assert "ramen" in html
    assert "650 kcal" in html


def test_events_partial_renders_empty():
    html = render("partials/events.html", **_events_ctx())
    assert "No events in the last 30 days." in html
    assert "No events in the last 7 days." in html
    assert "pick a category" in html
    # "all" poll URL carries no filter param
    assert 'hx-get="/partials/events"' in html


def test_events_page_renders_with_nav():
    html = render("events.html", **_events_ctx())
    assert "← dashboard" in html
    assert 'href="/artifacts"' in html
    assert "Events" in html


def _artifacts_ctx(**over: Any) -> dict[str, Any]:
    ctx: dict[str, Any] = {"type": "all", "types": [], "q": "", "artifacts": [],
                           "cap": data.ARTIFACTS_CAP}
    ctx.update(over)
    return {"artifacts": ctx}


def test_artifacts_partial_renders_populated_and_escapes_summary():
    arts = [data.shape_artifact({
        "cache_id": "c1", "artifact_type": "selfie",
        "summary": "<script>alert('pwn')</script>" + "x" * 300,
        "tags": ["render", "<b>tag</b>"], "source_agent": "selfie_agent",
        "created_at": NOW, "expires_at": NOW + timedelta(hours=4),
    }, now=NOW)]
    types = [{"name": "selfie", "n": 9, "type_class": data.artifact_type_class("selfie")}]
    html = render("partials/artifacts.html", **_artifacts_ctx(
        type="selfie", types=types, q="alert", artifacts=arts))
    # nothing executable leaks through autoescaping
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "<b>tag</b>" not in html
    # long summary expandable via <details>
    assert "<details" in html
    assert "selfie_agent" in html
    # type filter buttons + search box carry state; poll URL carries both
    assert 'hx-get="/partials/artifacts?type=selfie&amp;q=alert"' in html
    assert 'value="alert"' in html
    assert "imgfilter active" in html
    # no write controls on the read-only gallery
    assert "cleanup" not in html.lower()
    assert "delete" not in html.lower()


def test_artifacts_partial_renders_empty():
    html = render("partials/artifacts.html", **_artifacts_ctx())
    assert "No context_cache artifacts." in html
    assert 'hx-trigger="every 60s"' in html


def test_artifacts_partial_short_summary_no_details():
    arts = [data.shape_artifact({"cache_id": "c1", "artifact_type": "note",
                                 "summary": "short one", "tags": [],
                                 "created_at": NOW}, now=NOW)]
    html = render("partials/artifacts.html", **_artifacts_ctx(artifacts=arts))
    assert "short one" in html
    assert "<details" not in html


def test_artifacts_page_renders_with_nav():
    html = render("artifacts.html", **_artifacts_ctx())
    assert "← dashboard" in html
    assert 'href="/events"' in html
    assert "read-only" in html


def test_index_page_renders_life_tab_and_nav_links():
    html = render(
        "index.html",
        health={"db_ok": True, "chat_count": 0, "run_count": 0, "persona_count": 0,
                "last_run_at": None, "last_chat_at": None, "qwen_url": ""},
        persona={"messages": [], "delivered_count": 0, "skipped_count": 0,
                 "total": 0, "repeat_groups": 0},
        runs=[], digest=None, monologue=[], emotional=None, blocks=[], chat=[],
        images={"images": [], "cap": 60, "truncated": False, "kind": "all"},
        mind={"mood": None, "vibe": None, "vibe_history": [], "violations": [],
              "interests": [], "thoughts": []},
        life=_EMPTY_LIFE,
    )
    assert 'data-tab="life"' in html
    assert 'id="panel-life"' in html
    assert 'hx-get="/partials/life"' in html
    assert "every 60s" in html
    assert 'href="/events"' in html
    assert 'href="/artifacts"' in html
