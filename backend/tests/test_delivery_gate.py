"""Delivery-quality gate — pure-unit regression lock (no DB / docker / network).

The gate exists because real v3 data showed engagement collapsing 6x while
bot volume grew: near-duplicate proactive spam (736 techtree no-op variants,
an error fallback sent verbatim ×30), internal checker jargon ("All 6 checks
passed" ×110) and raw error leaks ("[Render Error] …") reaching chat.

These tests pin:
  - every frozen REAL_CASES expectation under DEFAULT_POLICY (the shipped
    baseline MUST pass all corpus cases; the autoloop exists for future
    model/usage drift, not to fix the default);
  - reason correctness (leak / noop / duplicate / too_short) and check order;
  - the precision case — similar topic, different content → DELIVER;
  - active_policy fallback + promoted-snapshot round-trip via a tmp .oac dir;
  - the send_message gate seam (_gate_message) output contract;
  - the deterministic improve_gate loop end-to-end against a tmp root.
"""

from __future__ import annotations

import pytest

from app.delivery import gate as gate_mod
from app.delivery.gate import (
    COMPONENT_ID,
    DEFAULT_POLICY,
    GateDecision,
    active_policy,
    evaluate_message,
)
from app.delivery.gate_probes import (
    GATE_CRITERION,
    REAL_CASES,
    gate_probe_list,
    improve_gate,
)

RECENTS = [
    "morning briefing ☀️ standup at 10:00, then a free afternoon.",
    "Pamiętaj o wodzie! Ostatni wpis był 4 godziny temu 💧",
]


@pytest.fixture(autouse=True)
def _fresh_policy_cache():
    gate_mod._clear_policy_cache()
    yield
    gate_mod._clear_policy_cache()


# ── the regression lock: every frozen real-corpus case under DEFAULT_POLICY ──


@pytest.mark.parametrize(
    "case", REAL_CASES, ids=[c["id"] for c in REAL_CASES],
)
def test_real_case_holds_under_default_policy(case):
    decision = evaluate_message(
        case["text"], case["recent"], DEFAULT_POLICY,
        kind=case.get("kind", "reply"),
        last_user_age_s=case.get("last_user_age_s"),
        last_bot_age_s=case.get("last_bot_age_s"),
    )
    got = "deliver" if decision.deliver else "suppress"
    assert got == case["expect"], (
        f"{case['id']}: expected {case['expect']}, got {got} "
        f"(reason={decision.reason}, matched={decision.matched!r})"
    )


def test_real_cases_cover_both_verdicts_and_all_failure_classes():
    """The frozen set must stay balanced: corpus failure classes AND ≥10 good
    messages that must deliver (the precision side)."""
    suppress = [c for c in REAL_CASES if c["expect"] == "suppress"]
    deliver = [c for c in REAL_CASES if c["expect"] == "deliver"]
    assert len(REAL_CASES) >= 25
    assert len(deliver) >= 10
    reasons = {
        evaluate_message(
            c["text"], c["recent"],
            kind=c.get("kind", "reply"),
            last_user_age_s=c.get("last_user_age_s"),
            last_bot_age_s=c.get("last_bot_age_s"),
        ).reason
        for c in suppress
    }
    assert reasons == {
        "duplicate", "noop", "leak", "too_short",
        "proactive_user_active", "proactive_cooldown",
    }


# ── reason correctness + check order ──


def test_leak_reason_and_matched_pattern():
    d = evaluate_message("[Render Error] f-string: expecting '}'", [])
    assert d == GateDecision(deliver=False, reason="leak", matched=r"\[Render Error")


def test_leak_checked_before_noop():
    # "All 6 checks passed" is both internal jargon and a no-op — leak wins
    # because internal machinery must never reach chat regardless.
    d = evaluate_message(
        "All 6 checks passed. Global cooldown is active, no intervention needed.",
        [],
    )
    assert d.deliver is False
    assert d.reason == "leak"


def test_noop_reason():
    d = evaluate_message(
        "checked the tech tree — no new commits about quantum computing today", RECENTS,
    )
    assert (d.deliver, d.reason) == (False, "noop")
    assert d.matched == r"\bno new commits\b"


def test_too_short_reason():
    d = evaluate_message("ok", RECENTS)
    assert (d.deliver, d.reason) == (False, "too_short")


def test_duplicate_reason_reports_matched_message():
    text = "💜 evening check-in! you crushed 3 tasks today 🦄"
    prev = "💜 evening check-in! you crushed 3 tasks today! 🦄"
    d = evaluate_message(text, [prev, *RECENTS])
    assert (d.deliver, d.reason, d.matched) == (False, "duplicate", prev)


def test_duplicate_normalizes_case_and_whitespace():
    d = evaluate_message(
        "Door sensor:  garage left   open 🚪",
        ["door sensor: garage left open 🚪"],
    )
    assert (d.deliver, d.reason) == (False, "duplicate")


# ── proactive background-cooldown (v3 parity) ──


def test_proactive_suppressed_when_user_active():
    # A nudge that fires while the user is mid-conversation is suppressed.
    d = evaluate_message(
        "drink some water 💧", [], kind="nudge", last_user_age_s=30,
    )
    assert (d.deliver, d.reason) == (False, "proactive_user_active")


def test_proactive_suppressed_back_to_back_bot():
    # User idle, but the bot just spoke → suppress the back-to-back proactive.
    d = evaluate_message(
        "evening briefing 📋", [], kind="briefing",
        last_user_age_s=99999, last_bot_age_s=20,
    )
    assert (d.deliver, d.reason) == (False, "proactive_cooldown")


def test_proactive_delivers_when_idle():
    # Same proactive content, but everyone's been quiet → legitimate to send.
    d = evaluate_message(
        "evening briefing 📋", [], kind="briefing",
        last_user_age_s=7200, last_bot_age_s=3600,
    )
    assert d.deliver is True


def test_reply_never_gated_by_cooldown():
    # The cooldown is proactive-only — a conversational reply during active
    # chat must ALWAYS deliver (the precision guard).
    d = evaluate_message(
        "sure, want a reminder in 10? 🚪", [], kind="reply",
        last_user_age_s=2, last_bot_age_s=2,
    )
    assert d.deliver is True


def test_cooldown_unknown_ages_skip_cooldown():
    # last_*_age_s=None (legacy callers) → cooldown is skipped entirely, even
    # for a proactive kind. Backward-compatible by construction.
    d = evaluate_message("evening briefing 📋", [], kind="nudge")
    assert d.deliver is True


def test_leak_still_wins_over_cooldown_skip():
    # Even on the proactive path, a leak in an otherwise-deliverable proactive
    # message is still caught (cooldown returning "ok" doesn't bypass leak).
    d = evaluate_message(
        "[Render Error] briefing failed", [], kind="briefing",
        last_user_age_s=7200, last_bot_age_s=3600,
    )
    assert (d.deliver, d.reason) == (False, "leak")


def test_dedup_respects_lookback_window():
    text = "door sensor: garage left open for 20 minutes 🚪"
    filler = [f"unrelated status update number {i} about something else" for i in range(8)]
    # Verbatim repeat sits *outside* the 8-message lookback → delivers.
    d = evaluate_message(text, [*filler, text])
    assert d.deliver is True
    # Inside the window → suppressed.
    d = evaluate_message(text, [*filler[:7], text])
    assert (d.deliver, d.reason) == (False, "duplicate")


def test_policy_none_falls_back_to_defaults_and_partial_policy_merges():
    assert evaluate_message("ok", [], None).reason == "too_short"
    # A partial promoted dict still carries every DEFAULT_POLICY key.
    d = evaluate_message("[Render Error] boom", [], {"dedup_similarity": 0.9})
    assert d.reason == "leak"


def test_precision_similar_topic_different_content_delivers():
    """THE over-suppression guard: same template family as a recent message,
    different day + different content → must deliver."""
    d = evaluate_message(
        "evening check-in 💜 today was lighter — 1 task done, but you logged a "
        "40-minute walk. tomorrow: the alembic migration first thing?",
        [
            "evening check-in 💜 you crushed 3 tasks today, drink some water "
            "and stretch those wings 🦄",
            *RECENTS,
        ],
    )
    assert (d.deliver, d.reason) == (True, "ok")


def test_good_message_with_empty_recent_history_delivers():
    d = evaluate_message("Hej! Skończyłeś 4 z 5 zadań na dziś 💪", [])
    assert d.deliver is True


# ── active_policy: fallback + promoted round-trip ──


def test_active_policy_falls_back_to_default_without_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_mod, "PROJECT_ROOT", tmp_path)
    assert active_policy() == DEFAULT_POLICY


def test_active_policy_loads_promoted_snapshot_round_trip(tmp_path, monkeypatch):
    """promote() a tuned policy into a tmp .oac dir the same way improve_gate
    does (write_snapshot → promote), and active_policy() must pick it up."""
    from src.improvement.snapshot import promote, write_snapshot
    from src.improvement.version import ComponentVersion

    tuned = {**DEFAULT_POLICY, "dedup_similarity": 0.85, "dedup_lookback": 12}
    version = ComponentVersion.of(COMPONENT_ID, "prompt", tuned)
    snap_path = write_snapshot(version, tmp_path / ".oac" / "snapshots")
    promote(snap_path, tmp_path, force=True)

    monkeypatch.setattr(gate_mod, "PROJECT_ROOT", tmp_path)
    policy = active_policy()
    assert policy["dedup_similarity"] == 0.85
    assert policy["dedup_lookback"] == 12
    # And every default key survives the merge.
    assert set(DEFAULT_POLICY) <= set(policy)


def test_active_policy_is_cached_per_process(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_mod, "PROJECT_ROOT", tmp_path)
    first = active_policy()
    # A promotion landing later is NOT seen until the cache is cleared.
    from src.improvement.snapshot import promote, write_snapshot
    from src.improvement.version import ComponentVersion

    tuned = {**DEFAULT_POLICY, "dedup_lookback": 15}
    snap = write_snapshot(
        ComponentVersion.of(COMPONENT_ID, "prompt", tuned),
        tmp_path / ".oac" / "snapshots",
    )
    promote(snap, tmp_path, force=True)
    assert active_policy() is first  # cached
    gate_mod._clear_policy_cache()
    assert active_policy()["dedup_lookback"] == 15


# ── send_message integration seam (_gate_message) ──


def test_gate_seam_delivers_good_message():
    from app.tools.telegram.send_message import _gate_message

    assert _gate_message("Hej! 4 z 5 zadań zrobione 💪", RECENTS, None) is None


def test_gate_seam_duplicate_keeps_legacy_output_contract():
    from app.tools.telegram.send_message import _gate_message

    text = "*taps horn nervously* something got tangled in my spell routing."
    out = _gate_message(text, [text, *RECENTS], None)
    assert out is not None
    # The historical contract: success=True (the agent's task is complete —
    # never an error, so agents don't retry-spam), zero parts sent, and the
    # DUPLICATE_DETECTED marker the compiled prompts already know.
    assert out.success is True
    assert out.parts_sent == 0
    assert out.error == "DUPLICATE_DETECTED"
    assert out.suppressed is True
    assert out.reason == "duplicate"


@pytest.mark.parametrize(
    ("text", "reason", "error"),
    [
        ("All 6 checks passed, nothing to do.", "leak", "SUPPRESSED_LEAK"),
        ("no new commits about Rust today", "noop", "SUPPRESSED_NOOP"),
        ("ok", "too_short", "SUPPRESSED_TOO_SHORT"),
    ],
)
def test_gate_seam_suppression_reports_ok_with_reason(text, reason, error):
    from app.tools.telegram.send_message import _gate_message

    out = _gate_message(text, RECENTS, None)
    assert out is not None
    assert (out.success, out.parts_sent, out.suppressed) == (True, 0, True)
    assert out.reason == reason
    assert out.error == error


def test_gate_seam_uses_supplied_policy():
    from app.tools.telegram.send_message import _gate_message

    # An (artificially) tiny lookback delivers what the default suppresses.
    text = "door sensor: garage left open for 20 minutes 🚪"
    recents = ["something else entirely happened just now", text]
    assert _gate_message(text, recents, {"dedup_lookback": 1}) is None
    assert _gate_message(text, recents, None) is not None


def test_gate_seam_never_blocks_on_gate_error(monkeypatch):
    """A broken gate must fail-open (deliver), never error the send."""
    import app.delivery.gate as g
    from app.tools.telegram import send_message as sm

    def boom(*a, **k):
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(g, "evaluate_message", boom)
    assert sm._gate_message("a perfectly fine message", RECENTS, None) is None


# ── the autoloop itself: deterministic, offline, promotes into tmp root ──


def test_gate_probe_list_mirrors_real_cases():
    probes = gate_probe_list()
    assert [p.probe_id for p in probes] == [c["id"] for c in REAL_CASES]
    assert all(len(p.evaluators) == 1 for p in probes)


def test_improve_gate_offline_promotes_into_tmp_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate_mod, "PROJECT_ROOT", tmp_path)
    result = improve_gate(max_rounds=2, project_root=tmp_path)

    from src.improvement.scoring import aggregate_score, hard_pass

    best = max(
        result.winners,
        key=lambda v: aggregate_score(GATE_CRITERION, v.metrics),
    )
    # The shipped baseline already passes everything, so the winner must too.
    assert best.metrics["pass_rate"] == 1.0
    assert hard_pass(GATE_CRITERION, best.metrics)

    # The promotion landed where active_policy() looks.
    promoted = tmp_path / ".oac" / "promoted"
    assert any(promoted.glob("*.json")), "winner was not promoted"
    gate_mod._clear_policy_cache()
    policy = active_policy(tmp_path)
    assert set(DEFAULT_POLICY) <= set(policy)
    # The promoted policy still passes every frozen real case.
    for case in REAL_CASES:
        d = evaluate_message(
            case["text"], case["recent"], policy,
            kind=case.get("kind", "reply"),
            last_user_age_s=case.get("last_user_age_s"),
            last_bot_age_s=case.get("last_bot_age_s"),
        )
        got = "deliver" if d.deliver else "suppress"
        assert got == case["expect"], f"promoted policy broke {case['id']}"

    report = capsys.readouterr().out
    assert "baseline:" in report and "winner:" in report
