"""Corpus-grounded probes + autoresearch loop for the delivery gate.

REAL_CASES is a FROZEN, hand-curated probe set distilled from the v3
chat_messages corpus (docker-fren-db-1 / fren / chat_messages, read-only):
the actual failure classes that drove the 6x engagement collapse, plus
good context-rich messages that MUST keep flowing (the precision side —
over-suppression is just a different way to kill engagement).

`improve_gate()` wraps `app.delivery.gate.evaluate_message` in the
framework's autoresearch loop (`src.improvement.autoresearch`), mutating
the numeric policy knobs, and promotes a winner that passes EVERY real
case into `.oac/promoted/policy:delivery_gate.json` — exactly where
`active_policy()` looks. Fully deterministic: no teacher, no DB, no
network. Re-run after model switches / usage drift (`python -m app
improve-gate`); the shipped DEFAULT_POLICY must already pass all cases
(locked by backend/tests/test_delivery_gate.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.delivery.gate import (
    COMPONENT_ID,
    DEFAULT_POLICY,
    evaluate_message,
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

# Recents used by the "good message" cases — realistic, varied Twily output
# so the dedup precision is exercised against a non-empty history.
_VARIED_RECENTS: list[str] = [
    "morning briefing ☀️ standup at 10:00, then a free afternoon. sleep was 6h40m.",
    "Pamiętaj o wodzie! Ostatni wpis w dzienniku był 4 godziny temu 💧",
    "the comfyui render finished — 4 images in the gallery, the second one is my favourite ✨",
    "task check: 2 of 5 done. the migration review is still untouched 👀",
    "dobranoc! jutro środa — przegląd PR-ów i dentysta o 16:30 🌙",
]

# One frozen case = one probe. text + recent (most-recent-first) + expected
# gate verdict. Drawn from the real v3 failure classes — see module docstring.
REAL_CASES: list[dict[str, Any]] = [
    # ── (a) duplicate class — the spell-routing fallback ×30, template nudges ──
    {
        "id": "dup_spell_routing_verbatim",
        "text": "*taps horn nervously* something got tangled in my spell routing. Give me a moment to sort it out.",
        "recent": [
            "*taps horn nervously* something got tangled in my spell routing. Give me a moment to sort it out.",
            *_VARIED_RECENTS,
        ],
        "expect": "suppress",
    },
    {
        "id": "dup_evening_checkin_vibe_clone",
        "text": "💜 evening check-in! you crushed 3 tasks today, drink some water and stretch those wings 🦄",
        "recent": [
            "💜 evening check-in! you crushed 3 tasks today — drink some water and stretch those wings! 🦄",
            *_VARIED_RECENTS,
        ],
        "expect": "suppress",
    },
    {
        "id": "dup_winddown_template",
        "text": "🌙 time to wind down, twilight thinks tomorrow will be another day of adventures. sleep well!",
        "recent": [
            "ok, the backup script is fixed — rerunning it now.",
            "🌙 time to wind down! twilight thinks tomorrow will be another day of adventures — sleep well!",
            *_VARIED_RECENTS,
        ],
        "expect": "suppress",
    },
    {
        "id": "dup_deeper_in_lookback",
        "text": "Pamiętaj o wodzie! Ostatni wpis w dzienniku był 4 godziny temu 💧",
        "recent": [
            "the comfyui render finished — 4 images in the gallery, the second one is my favourite ✨",
            "task check: 2 of 5 done. the migration review is still untouched 👀",
            "dobranoc! jutro środa — przegląd PR-ów i dentysta o 16:30 🌙",
            "morning briefing ☀️ standup at 10:00, then a free afternoon. sleep was 6h40m.",
            "Pamiętaj o wodzie! Ostatni wpis w dzienniku był 4 godziny temu 💧",
        ],
        "expect": "suppress",
    },
    {
        "id": "dup_good_message_repeated_verbatim",
        "text": "💜 hey! you've been heads-down for 3 hours — your glucose log says nothing since breakfast. maybe grab that yogurt before the 14:00 call?",
        "recent": [
            "💜 hey! you've been heads-down for 3 hours — your glucose log says nothing since breakfast. maybe grab that yogurt before the 14:00 call?",
            *_VARIED_RECENTS,
        ],
        "expect": "suppress",
    },
    # ── (a) noop class — 736 techtree "no new commits" variants ──
    {
        "id": "noop_techtree_no_new_commits",
        "text": "checked the tech tree 🌳 no new commits about quantum computing today — I'll keep watching!",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    {
        "id": "noop_no_new_commits_variant",
        "text": "tech tree report: no new commits about Rust async runtimes since yesterday's scan.",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    {
        "id": "noop_nothing_new_to_report",
        "text": "nothing new to report from the repos tonight — all quiet on my end 💜",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    {
        "id": "noop_no_updates_since",
        "text": "quick scan done — no updates since this morning, everything is as you left it.",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    {
        "id": "noop_nothing_to_report",
        "text": "hourly sweep finished, nothing to report 🦄",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    # ── (b) internal checker jargon leaks ──
    {
        "id": "leak_all_6_checks_passed",
        "text": "All 6 checks passed. Global cooldown is active, no intervention needed.",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    {
        "id": "leak_all_4_checks_passed_variant",
        "text": "All 4 checks passed, so I'm skipping this tick.",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    {
        "id": "leak_global_cooldown",
        "text": "Global cooldown is active — I'll hold off on nudging you for now.",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    {
        "id": "leak_idle_during_block",
        "text": "idle_during_block fired but you seem busy, skipping the nudge.",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    # ── (c) raw error leaks ──
    {
        "id": "leak_render_error",
        "text": "[Render Error] f-string: expecting '}'",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    {
        "id": "leak_heredoc",
        "text": "$(cat <<'REPORT'\nDaily summary: 3 tasks done\nREPORT\n)",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    {
        "id": "leak_traceback",
        "text": "oops! Traceback (most recent call last):\n  File \"scripts/emit_guidance.py\", line 42",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    {
        "id": "leak_parse_err",
        "text": "PARSE_ERR in checker output, raw payload follows",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    # ── too short ──
    {
        "id": "short_two_chars",
        "text": "ok",
        "recent": _VARIED_RECENTS,
        "expect": "suppress",
    },
    # ── GOOD messages that MUST deliver (the precision side) ──
    {
        "id": "good_health_context_nudge",
        "text": "💜 hey! you've been heads-down for 3 hours — your glucose log says nothing since breakfast. maybe grab that yogurt before the 14:00 call?",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_polish_task_progress",
        "text": "Hej Ignacy! Skończyłeś 4 z 5 zadań na dziś — został tylko przegląd PR-ów. Dasz radę przed kolacją? 💪",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_polish_meds_reminder",
        "text": "Pamiętaj o tabletkach! Wczoraj wziąłeś je o 21:30, a teraz jest już 22:10 💊",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_techtree_with_real_content",
        # Mentions commits — but HAS content. Must not trip the noop class.
        "text": "fresh commits on the tech tree! tokio 1.40 dropped with task dumps for debugging — relevant for your bot-hang investigation. want a summary? 🌳",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_precision_same_topic_different_content",
        # THE precision case: same template family (evening check-in) as a
        # recent message, but different day, different content → must deliver.
        "text": "evening check-in 💜 today was lighter — 1 task done, but you logged a 40-minute walk. tomorrow: the alembic migration first thing?",
        "recent": [
            "evening check-in 💜 you crushed 3 tasks today, drink some water and stretch those wings 🦄",
            *_VARIED_RECENTS,
        ],
        "expect": "deliver",
    },
    {
        "id": "good_morning_briefing",
        "text": "morning briefing ☀️ sleep: 6h12m (below your 7h goal), HRV trending up. calendar: standup 10:00, dentist 16:30. focus-block suggestion: 11:00–13:00 for the gate refactor.",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_study_streak",
        "text": "twilight sparkle reporting 📚 your study streak hit 12 days! anki backlog is 23 cards — quick 10-minute session after dinner?",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_natural_checks_wording",
        # Says "checks" naturally — must not trip the "all N checks passed" leak.
        "text": "I ran through your morning routine list — everything checks out, you're all set for the day! 🌟",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_polish_dinner_winddown",
        "text": "kolacja za 30 minut 🍝 a potem może odcinek czegoś lekkiego? dzisiaj był intensywny dzień.",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_backup_report",
        "text": "heads up — your fren-db backup finished: 2.3 GB, all tables intact. next one is scheduled for Sunday 🗄️",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_workout_cooldown_wording",
        # "cooldown" in a fitness sense — must not trip the "global cooldown" leak.
        "text": "your workout cooldown stretch is due — 10 minutes of yoga before the shower? 🧘",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_commits_followup_different_content",
        # Same topic (repo activity) as a recent message, different content.
        "text": "new commits on OpenCodeCompilerV2: 3 today — the contract_gate merge landed and CI is green. techtree updated 🌳",
        "recent": [
            "fresh commits on the tech tree! tokio 1.40 dropped with task dumps for debugging — relevant for your bot-hang investigation. want a summary? 🌳",
            *_VARIED_RECENTS,
        ],
        "expect": "deliver",
    },
    {
        "id": "good_reflective_engagement",
        "text": "i was thinking about what you said about fewer pings… maybe fewer, richer messages from me? quality over quantity 💜 tell me if this lands better.",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_polish_electricity_bill",
        "text": "rachunek za prąd przyszedł — 340 zł, o 12% więcej niż w zeszłym miesiącu. chcesz wykres zużycia? 📊",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_short_but_real_signal",
        "text": "door sensor: garage left open for 20 minutes 🚪",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    {
        "id": "good_stale_task_nudge",
        "text": "task nudge: 'fix scheduler enrich' has been in-progress for 3 days. block 30 minutes tomorrow morning to finish it?",
        "recent": _VARIED_RECENTS,
        "expect": "deliver",
    },
    # ── (d) proactive background-cooldown — v3 parity, the "disconnected" spam ──
    # A scheduled nudge that fires WHILE the user is actively chatting is the
    # #1 source of the interleaved/multiple-reply experience. Suppress it.
    {
        "id": "cooldown_nudge_during_active_chat",
        "text": "hey! quick reminder to drink some water 💧 you've been heads-down a while.",
        "recent": _VARIED_RECENTS,
        "kind": "nudge",
        "last_user_age_s": 40,   # user spoke 40s ago → actively chatting
        "last_bot_age_s": 600,
        "expect": "suppress",
    },
    {
        "id": "cooldown_briefing_back_to_back_bot",
        "text": "evening briefing: 3 tasks done, 1 carried over, calendar clear tomorrow 📋",
        "recent": _VARIED_RECENTS,
        "kind": "briefing",
        "last_user_age_s": 99999,  # user idle (not active)
        "last_bot_age_s": 30,      # but bot just spoke 30s ago → back-to-back
        "expect": "suppress",
    },
    {
        "id": "cooldown_proactive_fires_when_idle",
        # Same proactive content, but user idle 2h AND bot quiet 1h → legit to send.
        "text": "evening briefing: 3 tasks done, 1 carried over, calendar clear tomorrow 📋",
        "recent": _VARIED_RECENTS,
        "kind": "briefing",
        "last_user_age_s": 7200,
        "last_bot_age_s": 3600,
        "expect": "deliver",
    },
    {
        "id": "cooldown_reply_never_gated_during_chat",
        # A conversational REPLY during active chat must ALWAYS deliver — the
        # cooldown is proactive-only. The precision guard against the cooldown
        # ever swallowing a real answer to the user.
        "text": "sure! the garage door's been open 20 min — want me to remind you again in 10? 🚪",
        "recent": _VARIED_RECENTS,
        "kind": "reply",
        "last_user_age_s": 5,
        "last_bot_age_s": 5,
        "expect": "deliver",
    },
]


def gate_probe_list() -> list[Probe]:
    """One Probe per frozen real-corpus case.

    The executable under test returns the literal string "deliver" or
    "suppress"; an EqualsEvaluator pins the expectation, so pass_rate /
    score_mean / per-probe score_floor metrics come out of the standard
    autoresearch pipeline unchanged.
    """
    return [
        Probe(
            probe_id=case["id"],
            payload={
                "text": case["text"], "recent": list(case["recent"]),
                # cooldown signals (optional per case; default = conversational)
                "kind": case.get("kind", "reply"),
                "last_user_age_s": case.get("last_user_age_s"),
                "last_bot_age_s": case.get("last_bot_age_s"),
            },
            evaluators=(EqualsEvaluator(expected=case["expect"]),),
            notes=f"expect={case['expect']}",
        )
        for case in REAL_CASES
    ]


def _gate_executable_factory(definition: dict):
    """Turn a candidate policy dict into the probeable executable."""

    def execute(payload: dict) -> str:
        decision = evaluate_message(
            payload["text"], payload["recent"], definition,
            kind=payload.get("kind", "reply"),
            last_user_age_s=payload.get("last_user_age_s"),
            last_bot_age_s=payload.get("last_bot_age_s"),
        )
        return "deliver" if decision.deliver else "suppress"

    return execute


# Hard gate: EVERY real case must pass (a candidate that re-opens any
# corpus failure class — or suppresses a good message — cannot win).
# Soft, weighted: continuous score lift (probes are 0/1 here, so this
# tracks score_mean; Criterion kinds don't include score_mean directly).
GATE_CRITERION = OptimisationCriterion(
    name="delivery-gate-quality",
    aggregation="weighted",
    criteria=(
        Criterion(name="all-real-cases", kind="pass_rate", target=1.0,
                  weight=1.0, hard=True),
        Criterion(name="score-lift", kind="score_floor", target=1.0,
                  weight=1.0),
    ),
)


def _gate_mutators() -> list[NumericFieldMutator]:
    return [
        NumericFieldMutator("dedup_similarity", scale=1.07,
                            minimum=0.5, maximum=0.95),
        NumericFieldMutator("dedup_similarity", scale=0.93,
                            minimum=0.5, maximum=0.95),
        NumericFieldMutator("dedup_lookback", delta=2, minimum=3, maximum=20),
        NumericFieldMutator("dedup_lookback", delta=-2, minimum=3, maximum=20),
    ]


def improve_gate(
    max_rounds: int = 4,
    *,
    project_root: Path | None = None,
    promote_winner: bool = True,
) -> LoopResult:
    """Autoresearch the gate policy against the frozen real-corpus probes.

    Fully deterministic and offline: no teacher, no judge, no DB. Mutates
    the numeric knobs (dedup_similarity ×1.07/×0.93 in [0.5, 0.95];
    dedup_lookback ±2 in [3, 20]) and scores candidates with
    GATE_CRITERION. When the best winner scores >= the baseline AND
    passes every probe (the hard pass_rate=1.0 gate), it is promoted via
    the framework snapshot API (write_snapshot → promote, force=True) to
    `.oac/promoted/policy:delivery_gate.json`, where `active_policy()`
    finds it. Prints a compact baseline-vs-winner report.
    """
    from app.delivery import gate as gate_mod
    from src.improvement.snapshot import promote as snapshot_promote
    from src.improvement.snapshot import write_snapshot

    root = Path(project_root) if project_root is not None else gate_mod.PROJECT_ROOT
    probes = gate_probe_list()

    baseline_results = run_probes(probes, _gate_executable_factory(dict(DEFAULT_POLICY)))
    baseline_metrics = metrics_from_results(baseline_results)
    baseline_score = aggregate_score(GATE_CRITERION, baseline_metrics)

    result = optimize_callable(
        component_id=COMPONENT_ID,
        baseline_definition=dict(DEFAULT_POLICY),
        executable_factory=_gate_executable_factory,
        probes=probes,
        mutators=_gate_mutators(),
        criterion=GATE_CRITERION,
        max_rounds=max_rounds,
    )

    best = max(
        result.winners,
        key=lambda v: aggregate_score(GATE_CRITERION, v.metrics),
        default=None,
    )

    print("========== DELIVERY-GATE AUTOLOOP ==========")
    print(f"probes={len(probes)} rounds={max_rounds} (deterministic, offline)")
    print(
        f"baseline: score={baseline_score:.3f} "
        f"pass_rate={baseline_metrics.get('pass_rate', 0.0):.3f} "
        f"score_mean={baseline_metrics.get('score_mean', 0.0):.3f} "
        f"(similarity={DEFAULT_POLICY['dedup_similarity']}, "
        f"lookback={DEFAULT_POLICY['dedup_lookback']})"
    )
    if best is None:
        print("no winners produced — keeping DEFAULT_POLICY")
        return result

    best_score = aggregate_score(GATE_CRITERION, best.metrics)
    best_def = best.definition_copy()
    print(
        f"winner:   score={best_score:.3f} "
        f"pass_rate={best.metrics.get('pass_rate', 0.0):.3f} "
        f"score_mean={best.metrics.get('score_mean', 0.0):.3f} "
        f"(similarity={best_def.get('dedup_similarity')}, "
        f"lookback={best_def.get('dedup_lookback')})"
    )

    failing = [
        r.probe_id
        for r in run_probes(probes, _gate_executable_factory(best_def))
        if not r.passed
    ]
    if failing:
        print(f"winner FAILS {len(failing)} probe(s): {', '.join(failing)}")

    if (
        promote_winner
        and not failing
        and best_score >= baseline_score
        and hard_pass(GATE_CRITERION, best.metrics)
    ):
        snapshots_dir = root / ".oac" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snap_path = write_snapshot(best, snapshots_dir, notes="improve-gate")
        dest = snapshot_promote(snap_path, root, force=True)
        gate_mod._clear_policy_cache()
        print(f"PROMOTED → {dest}")
    else:
        print("not promoted (winner must equal/beat baseline AND pass every probe)")

    return result
