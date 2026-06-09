"""Fixture-grounded probes + autoresearch loop for the event→habit bridge.

REAL_CASES is a frozen probe set built from realistic event/habit shapes (the
v3 event extractor's categories × the user's actual habit families: daily
walk, medication, exercise, hydration) plus the precision side — events that
must NOT complete anything (purchases, questions, stale backfills, unrelated
habits).

``improve_bridge()`` wraps ``app.bridge.event_habit.decide_completions`` in
the framework's autoresearch loop (``src.improvement.autoresearch``), mutating
the numeric policy knobs, and promotes a winner that passes EVERY case into
``.oac/promoted/policy:event_habit_bridge.json`` — exactly where
``active_policy()`` looks. Fully deterministic: no teacher, no DB, no network
(mirrors ``app/delivery/gate_probes.py``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from app.bridge.event_habit import (
    COMPONENT_ID,
    DEFAULT_POLICY,
    decide_completions,
)
from src.improvement.autoresearch import (
    Probe,
    metrics_from_results,
    optimize_callable,
    run_probes,
)
from src.improvement.criteria import Criterion, OptimisationCriterion
from src.improvement.loop import LoopResult
from src.improvement.mutators.fields import NumericFieldMutator
from src.improvement.scoring import aggregate_score, hard_pass
from src.model.core.test_model import EqualsEvaluator

# Frozen "today" so every probe is deterministic.
TODAY = date(2026, 6, 10)

# Realistic active-habit fixtures (shape = HabitsRepo.list rows).
HABITS: list[dict[str, Any]] = [
    {"habit_id": "hab_daily_walk", "title": "Daily walk", "category": "health",
     "tags": ["walking", "outdoor"]},
    {"habit_id": "hab_meds", "title": "Take concerta", "category": "medication",
     "tags": ["meds"]},
    {"habit_id": "hab_exercise", "title": "Workout session", "category": "fitness",
     "tags": ["gym", "training"]},
    {"habit_id": "hab_water", "title": "Drink 2L of water", "category": "health",
     "tags": ["hydration", "water"]},
    {"habit_id": "hab_reading", "title": "Read 20 pages", "category": "learning",
     "tags": ["books"]},
]


def _ev(eid: str, category: str, title: str, *, sub: str = "", d: str = "2026-06-10") -> dict[str, Any]:
    return {"event_id": eid, "category": category, "subcategory": sub, "title": title, "date": d}


# One frozen case = one probe: events in, the exact set of completions out
# (as sorted "habit_id:date" strings; "none" when nothing may complete).
REAL_CASES: list[dict[str, Any]] = [
    # ── recall: the three core families must auto-complete ──
    {
        "id": "walk_event_completes_daily_walk",
        "events": [_ev("ev1", "walk", "Went for a walk", )],
        "expect": ["hab_daily_walk:2026-06-10"],
    },
    {
        "id": "medication_event_completes_med_habit",
        "events": [_ev("ev2", "medication", "Took concerta 36mg", sub="concerta")],
        "expect": ["hab_meds:2026-06-10"],
    },
    {
        "id": "workout_event_completes_exercise",
        "events": [_ev("ev3", "workout", "Gym session, 45 min")],
        "expect": ["hab_exercise:2026-06-10"],
    },
    {
        "id": "exercise_synonym_morning_training",
        # category "exercise" + title word "training" → exercise habit via
        # synonyms/tags, NOT via the walk habit.
        "events": [_ev("ev4", "exercise", "Morning training run")],
        "expect": ["hab_exercise:2026-06-10"],
    },
    {
        "id": "polish_spacer_completes_daily_walk",
        # PL title; category "walk" matches habit tag "walking"/title "walk".
        "events": [_ev("ev5", "walk", "Spacer wieczorny")],
        "expect": ["hab_daily_walk:2026-06-10"],
    },
    {
        "id": "drinking_event_completes_water_habit",
        "events": [_ev("ev6", "drinking", "Drank 500ml", )],
        "expect": ["hab_water:2026-06-10"],
    },
    # ── dedup: two same-day walks = ONE completion ──
    {
        "id": "two_walks_one_completion",
        "events": [
            _ev("ev7", "walk", "Morning walk"),
            _ev("ev8", "walk", "Evening walk"),
        ],
        "expect": ["hab_daily_walk:2026-06-10"],
    },
    # ── yesterday is still inside the window ──
    {
        "id": "yesterday_walk_completes_yesterday_occurrence",
        "events": [_ev("ev9", "walk", "Went for a walk", d="2026-06-09")],
        "expect": ["hab_daily_walk:2026-06-09"],
    },
    # ── precision: events that must complete NOTHING ──
    {
        "id": "purchase_completes_nothing",
        "events": [_ev("ev10", "purchase", "Bought a standing desk mat")],
        "expect": [],
    },
    {
        "id": "late_activity_completes_nothing",
        "events": [_ev("ev11", "late_activity", "Still awake at 2am")],
        "expect": [],
    },
    {
        "id": "stale_backfilled_walk_outside_window",
        # A walk from 10 days ago must not flip a historic streak.
        "events": [_ev("ev12", "walk", "Went for a walk", d="2026-05-31")],
        "expect": [],
    },
    {
        "id": "eating_event_no_meal_habit_configured",
        # No meal habit exists in the fixture set → nothing to complete.
        "events": [_ev("ev13", "eating", "Ate lunch")],
        "expect": [],
    },
    {
        "id": "weight_log_does_not_touch_reading",
        "events": [_ev("ev14", "weight", "Logged 78.2 kg")],
        "expect": [],
    },
    # ── mixed batch: exactly the right subset ──
    {
        "id": "mixed_batch_walk_meds_purchase",
        "events": [
            _ev("ev15", "walk", "Walk with dog"),
            _ev("ev16", "medication", "Took concerta 36mg", sub="concerta"),
            _ev("ev17", "purchase", "Bought groceries"),
        ],
        "expect": ["hab_daily_walk:2026-06-10", "hab_meds:2026-06-10"],
    },
]


def _expected_str(expect: list[str]) -> str:
    return ",".join(sorted(expect)) if expect else "none"


def bridge_probe_list() -> list[Probe]:
    """One Probe per frozen case. The executable returns the sorted
    "habit_id:date" completion set as a string; EqualsEvaluator pins it.
    """
    return [
        Probe(
            probe_id=case["id"],
            payload={"events": [dict(e) for e in case["events"]]},
            evaluators=(EqualsEvaluator(expected=_expected_str(case["expect"])),),
            notes=f"expect={_expected_str(case['expect'])}",
        )
        for case in REAL_CASES
    ]


def _bridge_executable_factory(definition: dict):
    """Turn a candidate policy dict into the probeable executable."""

    def execute(payload: dict) -> str:
        decisions = decide_completions(payload["events"], HABITS, definition, today=TODAY)
        if not decisions:
            return "none"
        return ",".join(sorted(f"{d.habit_id}:{d.event_date}" for d in decisions))

    return execute


# Hard gate: EVERY case must pass (a candidate that misses a real habit
# completion — or auto-completes from an unrelated/stale event — cannot win).
BRIDGE_CRITERION = OptimisationCriterion(
    name="event-habit-bridge-quality",
    aggregation="weighted",
    criteria=(
        Criterion(name="all-real-cases", kind="pass_rate", target=1.0,
                  weight=1.0, hard=True),
        Criterion(name="score-lift", kind="score_floor", target=1.0,
                  weight=1.0),
    ),
)


def _bridge_mutators() -> list[NumericFieldMutator]:
    return [
        NumericFieldMutator("confidence_threshold", scale=1.1, minimum=0.3, maximum=0.95),
        NumericFieldMutator("confidence_threshold", scale=0.9, minimum=0.3, maximum=0.95),
        NumericFieldMutator("max_event_age_days", delta=1, minimum=1, maximum=7),
        NumericFieldMutator("max_event_age_days", delta=-1, minimum=1, maximum=7),
        NumericFieldMutator("min_title_word_len", delta=1, minimum=3, maximum=6),
        NumericFieldMutator("min_title_word_len", delta=-1, minimum=3, maximum=6),
    ]


def improve_bridge(
    max_rounds: int = 4,
    *,
    project_root: Path | None = None,
    promote_winner: bool = True,
) -> LoopResult:
    """Autoresearch the bridge policy against the frozen fixture probes.

    Fully deterministic and offline (no teacher, no judge, no DB). When the
    best winner scores >= baseline AND passes every probe, it is promoted via
    the framework snapshot API to ``.oac/promoted/policy:event_habit_bridge.json``.
    """
    from app.bridge import event_habit as bridge_mod
    from src.improvement.snapshot import promote as snapshot_promote
    from src.improvement.snapshot import write_snapshot

    root = Path(project_root) if project_root is not None else bridge_mod.PROJECT_ROOT
    probes = bridge_probe_list()

    baseline_results = run_probes(probes, _bridge_executable_factory(dict(DEFAULT_POLICY)))
    baseline_metrics = metrics_from_results(baseline_results)
    baseline_score = aggregate_score(BRIDGE_CRITERION, baseline_metrics)

    result = optimize_callable(
        component_id=COMPONENT_ID,
        baseline_definition=dict(DEFAULT_POLICY),
        executable_factory=_bridge_executable_factory,
        probes=probes,
        mutators=_bridge_mutators(),
        criterion=BRIDGE_CRITERION,
        max_rounds=max_rounds,
    )

    best = max(
        result.winners,
        key=lambda v: aggregate_score(BRIDGE_CRITERION, v.metrics),
        default=None,
    )

    print("========== EVENT-HABIT BRIDGE AUTOLOOP ==========")
    print(f"probes={len(probes)} rounds={max_rounds} (deterministic, offline)")
    print(
        f"baseline: score={baseline_score:.3f} "
        f"pass_rate={baseline_metrics.get('pass_rate', 0.0):.3f} "
        f"score_mean={baseline_metrics.get('score_mean', 0.0):.3f} "
        f"(threshold={DEFAULT_POLICY['confidence_threshold']}, "
        f"max_age={DEFAULT_POLICY['max_event_age_days']}d)"
    )
    if best is None:
        print("no winners produced — keeping DEFAULT_POLICY")
        return result

    best_score = aggregate_score(BRIDGE_CRITERION, best.metrics)
    best_def = best.definition_copy()
    print(
        f"winner:   score={best_score:.3f} "
        f"pass_rate={best.metrics.get('pass_rate', 0.0):.3f} "
        f"score_mean={best.metrics.get('score_mean', 0.0):.3f} "
        f"(threshold={best_def.get('confidence_threshold')}, "
        f"max_age={best_def.get('max_event_age_days')}d)"
    )

    failing = [
        r.probe_id
        for r in run_probes(probes, _bridge_executable_factory(best_def))
        if not r.passed
    ]
    if failing:
        print(f"winner FAILS {len(failing)} probe(s): {', '.join(failing)}")

    if (
        promote_winner
        and not failing
        and best_score >= baseline_score
        and hard_pass(BRIDGE_CRITERION, best.metrics)
    ):
        snapshots_dir = root / ".oac" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snap_path = write_snapshot(best, snapshots_dir, notes="improve-bridge")
        dest = snapshot_promote(snap_path, root, force=True)
        bridge_mod._clear_policy_cache()
        print(f"PROMOTED → {dest}")
    else:
        print("not promoted (winner must equal/beat baseline AND pass every probe)")

    return result
