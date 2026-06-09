"""Event → habit bridge — policy, probes, autoloop and runner (all offline).

The matching POLICY is pure (`decide_completions`) and autoloop-optimisable
via src.improvement.autoresearch; these tests lock:

  (a) the matching strategies + confidence threshold semantics,
  (b) the time-window and same-day dedup gates,
  (c) DEFAULT_POLICY passing EVERY frozen fixture probe (the shipping gate),
  (d) the offline autoresearch loop running end-to-end (no teacher/DB),
  (e) the DB runner writing occurrences ONLY through the habits repo, with
      the cursor advanced (fully mocked repos).
"""

from __future__ import annotations

from datetime import date

import pytest

pytest.importorskip("sqlalchemy")

from app.bridge import event_habit as eh
from app.bridge.event_habit_probes import (
    BRIDGE_CRITERION,
    HABITS,
    REAL_CASES,
    TODAY,
    _bridge_executable_factory,
    bridge_probe_list,
    improve_bridge,
)
from src.improvement.autoresearch import metrics_from_results, run_probes
from src.improvement.scoring import aggregate_score


# ── (a) pure matching strategies ─────────────────────────────────────────────


def _habit(habit_id="hab_walk", title="Daily walk", category="health", tags=("walking",)):
    return {"habit_id": habit_id, "title": title, "category": category, "tags": list(tags)}


def test_category_in_habit_text_matches_with_high_confidence():
    d = eh.match_event_to_habit({"category": "walk", "title": "Went out"}, _habit())
    assert d.matched and d.confidence == eh.DEFAULT_POLICY["confidence_category"]


def test_subcategory_matches_into_habit_title():
    habit = _habit(habit_id="hab_meds", title="Take concerta", category="medication", tags=())
    d = eh.match_event_to_habit(
        {"category": "medication", "subcategory": "concerta", "title": "took meds"}, habit
    )
    assert d.matched


def test_title_word_overlap_requires_min_word_len():
    habit = _habit(title="Morning run club", category="", tags=())
    # "run" (3 chars) is below min_title_word_len=4 → no overlap match.
    d = eh.match_event_to_habit({"category": "misc", "title": "quick run"}, habit)
    assert not d.matched
    # "morning" overlaps.
    d = eh.match_event_to_habit({"category": "misc", "title": "morning jog"}, habit)
    assert d.matched and d.confidence == eh.DEFAULT_POLICY["confidence_title_overlap"]


def test_synonym_fallback_matches():
    habit = _habit(habit_id="hab_gym", title="Hit the gym", category="fitness", tags=())
    d = eh.match_event_to_habit({"category": "workout", "title": "trained hard"}, habit)
    assert d.matched and d.confidence == eh.DEFAULT_POLICY["confidence_synonym"]


def test_confidence_threshold_gates_low_strategies():
    habit = _habit(habit_id="hab_gym", title="Hit the gym", category="fitness", tags=())
    policy = {**eh.DEFAULT_POLICY, "confidence_threshold": 0.8}
    d = eh.match_event_to_habit({"category": "workout", "title": "trained hard"}, habit, policy)
    assert not d.matched and "below threshold" in d.reason


def test_unrelated_event_never_matches():
    d = eh.match_event_to_habit({"category": "purchase", "title": "Bought a mat"}, _habit())
    assert not d.matched


# ── (b) decide_completions: window + dedup ───────────────────────────────────


def test_decide_completions_time_window_and_dedup():
    today = date(2026, 6, 10)
    events = [
        {"event_id": "e1", "category": "walk", "title": "Morning walk", "date": "2026-06-10"},
        {"event_id": "e2", "category": "walk", "title": "Evening walk", "date": "2026-06-10"},
        {"event_id": "e3", "category": "walk", "title": "Old walk", "date": "2026-05-20"},
    ]
    out = eh.decide_completions(events, [_habit()], today=today)
    assert len(out) == 1  # same-day dedup + stale event dropped
    assert out[0].occurrence_id == "occ_hab_walk_2026-06-10"
    assert out[0].notes.startswith("Auto-completed from walk event")


def test_decide_completions_yesterday_within_window():
    out = eh.decide_completions(
        [{"event_id": "e1", "category": "walk", "title": "walk", "date": "2026-06-09"}],
        [_habit()],
        today=date(2026, 6, 10),
    )
    assert [d.event_date for d in out] == ["2026-06-09"]


# ── (c) DEFAULT_POLICY must pass every frozen probe ──────────────────────────


def test_default_policy_passes_every_fixture_probe():
    results = run_probes(bridge_probe_list(), _bridge_executable_factory(dict(eh.DEFAULT_POLICY)))
    failing = {r.probe_id: r.error or r.score for r in results if not r.passed}
    assert not failing, f"DEFAULT_POLICY fails fixture probes: {failing}"
    metrics = metrics_from_results(results)
    assert metrics["pass_rate"] == 1.0
    assert aggregate_score(BRIDGE_CRITERION, metrics) > 0.0


def test_probe_fixtures_cover_recall_and_precision():
    """The frozen set must keep both sides: completions AND must-not-complete."""
    expects = [bool(c["expect"]) for c in REAL_CASES]
    assert any(expects) and not all(expects)
    # The three core families from the brief are present.
    ids = {c["id"] for c in REAL_CASES}
    assert {"walk_event_completes_daily_walk",
            "medication_event_completes_med_habit",
            "workout_event_completes_exercise"} <= ids


# ── (d) the offline autoresearch loop runs end-to-end ────────────────────────


def test_improve_bridge_runs_offline_and_does_not_promote_into_repo(tmp_path, capsys):
    result = improve_bridge(max_rounds=1, project_root=tmp_path, promote_winner=False)
    out = capsys.readouterr().out
    assert "EVENT-HABIT BRIDGE AUTOLOOP" in out
    assert "baseline: score=" in out
    # The loop produced rounds; nothing was written into the real repo.
    assert result.rounds, "autoresearch loop produced no rounds"
    assert not (tmp_path / ".oac" / "promoted").exists() or not any(
        (tmp_path / ".oac" / "promoted").iterdir()
    )


def test_active_policy_falls_back_to_default(tmp_path):
    eh._clear_policy_cache()
    try:
        assert eh.active_policy(tmp_path) == eh.DEFAULT_POLICY
    finally:
        eh._clear_policy_cache()


# ── (e) DB runner: repos mocked, occurrences written only via the repo ───────


class _FakeNotesRepo:
    store: dict = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, *, expires_hours=24):
        type(self).store[key] = {"note_value": value}
        return {"note_key": key}


class _FakeEventsRepo:
    events: list = []

    async def list_since_id(self, since_id, *, limit=200):
        return [e for e in self.events if e["id"] > since_id][:limit]


class _FakeHabitsRepo:
    habits: list = []
    occurrences: dict = {}
    created: list = []
    completed: list = []

    async def list(self, *, status=None, category=None, limit=50):
        return list(self.habits)

    async def create_occurrence(self, occurrence_id, habit_id, scheduled_date, **kw):
        type(self).created.append(occurrence_id)
        type(self).occurrences.setdefault(habit_id, []).append(
            {"occurrence_id": occurrence_id, "status": "pending"}
        )
        return {"occurrence_id": occurrence_id}

    async def get_occurrences(self, habit_id, *, limit=30):
        return list(self.occurrences.get(habit_id, []))

    async def complete_occurrence(self, occurrence_id, *, notes=None):
        type(self).completed.append((occurrence_id, notes))
        for occs in self.occurrences.values():
            for o in occs:
                if o["occurrence_id"] == occurrence_id:
                    o["status"] = "completed"
        return {"occurrence_id": occurrence_id, "status": "completed"}


@pytest.fixture
def _fresh_fakes(monkeypatch):
    _FakeNotesRepo.store = {}
    _FakeEventsRepo.events = []
    _FakeHabitsRepo.habits = []
    _FakeHabitsRepo.occurrences = {}
    _FakeHabitsRepo.created = []
    _FakeHabitsRepo.completed = []
    import app.db.repos.agent_notes as notes_mod
    import app.db.repos.events as events_mod
    import app.db.repos.habits as habits_mod

    monkeypatch.setattr(notes_mod, "AgentNotesRepo", _FakeNotesRepo)
    monkeypatch.setattr(events_mod, "EventsRepo", _FakeEventsRepo)
    monkeypatch.setattr(habits_mod, "HabitsRepo", _FakeHabitsRepo)


async def test_run_bridge_completes_matching_habit_and_advances_cursor(_fresh_fakes):
    today = date.today().isoformat()
    _FakeEventsRepo.events = [
        {"id": 11, "event_id": "ev_a", "category": "walk", "title": "Went for a walk", "date": today},
        {"id": 12, "event_id": "ev_b", "category": "purchase", "title": "Bought socks", "date": today},
    ]
    _FakeHabitsRepo.habits = [
        {"habit_id": "hab_walk", "title": "Daily walk", "category": "health", "tags": ["walking"]},
    ]

    summary = await eh.run_bridge(dict(eh.DEFAULT_POLICY))

    assert summary == {"events": 2, "completions": 1, "skipped": 0}
    assert _FakeHabitsRepo.created == [f"occ_hab_walk_{today}"]
    completed_id, notes = _FakeHabitsRepo.completed[0]
    assert completed_id == f"occ_hab_walk_{today}"
    assert "Auto-completed from walk event" in notes
    # Cursor advanced to the max event id.
    assert _FakeNotesRepo.store[eh.STATE_KEY]["note_value"] == {"last_event_id": 12}


async def test_run_bridge_skips_already_completed_occurrence(_fresh_fakes):
    today = date.today().isoformat()
    occ_id = f"occ_hab_walk_{today}"
    _FakeEventsRepo.events = [
        {"id": 5, "event_id": "ev_a", "category": "walk", "title": "walk", "date": today},
    ]
    _FakeHabitsRepo.habits = [
        {"habit_id": "hab_walk", "title": "Daily walk", "category": "health", "tags": []},
    ]
    _FakeHabitsRepo.occurrences = {
        "hab_walk": [{"occurrence_id": occ_id, "status": "completed"}]
    }

    summary = await eh.run_bridge(dict(eh.DEFAULT_POLICY))
    assert summary["completions"] == 0 and summary["skipped"] == 1
    assert _FakeHabitsRepo.completed == []


async def test_run_bridge_no_events_is_a_noop(_fresh_fakes):
    summary = await eh.run_bridge(dict(eh.DEFAULT_POLICY))
    assert summary == {"events": 0, "completions": 0, "skipped": 0}
    assert eh.STATE_KEY not in _FakeNotesRepo.store
