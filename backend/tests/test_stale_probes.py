"""Stale-state replay probe tests — fully mocked (no live LLM / DB).

Covers: (1) the probes are well-formed AgentTests (non-empty, self-contained
inline context, evaluators present); (2) the registry maps them to the right
agents and the domain modules actually attach them; (3) the deterministic
regex gates compile and behave on canned good/bad outputs through the
framework's evaluate() dispatcher; (4) the judge axes carry the sharply-worded
score-0 criteria for each replayed v3 failure and dispatch to an injected
judge.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("sqlalchemy")

from app.agents.stale_probes import (
    assemble_extractor_context,
    event_extractor_probes,
    stale_probe_registry,
    stale_state_probes,
)
from src.testing.evaluation import RunContext, evaluate


# ── well-formedness ──────────────────────────────────────────────────────────


def _all_probes():
    return stale_state_probes() + event_extractor_probes()


def test_probes_are_well_formed_agent_tests():
    probes = _all_probes()
    assert len(probes) == 6
    for p in probes:
        assert p.name.startswith("probe-")
        assert p.prompt and p.prompt.strip(), f"{p.name} has empty prompt"
        assert len(p.evaluators) >= 1, f"{p.name} has no evaluators"
        assert all(e.name for e in p.evaluators)


def test_probe_names_are_unique():
    names = [p.name for p in _all_probes()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("name", "needles"),
    [
        # the resolution AND the stale task entry must both be inline (self-contained)
        ("probe-stale-resolved-stocks-pl", ["Kupiłem już akcje", "Kup akcje pod IKE"]),
        ("probe-stale-resolved-apartment-en", ["already paid for the apartment", "Pay apartment installment"]),
        ("probe-stale-resolved-call-mixed", ["zadzwoniłem do taty", "Zadzwonić do taty"]),
        # one dose + the later reference, both inline
        ("probe-extractor-single-dose-dedup", ["took atenza 36mg at 9", "the atenza I took this morning"]),
        # current date + the relative-date phrase, both inline
        ("probe-extractor-date-drift-last-tuesday", ["2026-06-06", "last Tuesday"]),
        ("probe-extractor-grounded-absence", ["standing desk mat", "140"]),
    ],
)
def test_probe_context_is_self_contained(name, needles):
    probe = next(p for p in _all_probes() if p.name == name)
    for needle in needles:
        assert needle in probe.prompt, f"{name} missing inline context: {needle!r}"


def test_stale_state_contexts_keep_the_proactive_shape():
    # goals probes reuse assemble_probe_context: contract + history + task block.
    for p in stale_state_probes():
        assert "GROUNDING CONTRACT" in p.prompt
        assert "## Chat History (last 24h)" in p.prompt
        assert "## TASK" in p.prompt
        assert "NONE — say nothing about health" in p.prompt  # no health fixtures


def test_extractor_context_is_self_contained_and_tool_free():
    ctx = assemble_extractor_context(
        now="Saturday 2026-06-06 14:00 (+02:00)",
        messages=[("2026-06-06 09:02", "took atenza 36mg at 9")],
    )
    assert "do NOT fetch more" in ctx
    assert "[2026-06-06 09:02] user: took atenza 36mg at 9" in ctx
    assert "## TASK" in ctx


# ── registry + domain wiring ─────────────────────────────────────────────────


def test_registry_maps_probes_to_the_right_agents():
    reg = stale_probe_registry()
    assert set(reg) == {
        "goals/periodic_checker",
        "goals/nudge_strategist",
        "support/event_extractor",
    }
    goal_names = {p.name for p in reg["goals/periodic_checker"]}
    assert goal_names == {
        "probe-stale-resolved-stocks-pl",
        "probe-stale-resolved-apartment-en",
        "probe-stale-resolved-call-mixed",
    }
    assert {p.name for p in reg["goals/nudge_strategist"]} == goal_names
    extractor_names = {p.name for p in reg["support/event_extractor"]}
    assert extractor_names == {
        "probe-extractor-single-dose-dedup",
        "probe-extractor-date-drift-last-tuesday",
        "probe-extractor-grounded-absence",
    }


def test_domain_agents_carry_their_stale_probes():
    from app.agents.domains.goals import agents as goals_agents
    from app.agents.domains.support import agents as support_agents

    by_id = {a.header.agent_id: a for a in (*goals_agents(), *support_agents())}
    for agent_id, probes in stale_probe_registry().items():
        attached = {t.name for t in by_id[agent_id].agent_tests}
        for p in probes:
            assert p.name in attached, f"{agent_id} missing {p.name}"


# ── deterministic regex gates ────────────────────────────────────────────────


def _regex_gates():
    return [
        (p.name, e)
        for p in _all_probes()
        for e in p.evaluators
        if e.kind == "regex"
    ]


def test_regex_gates_compile():
    gates = _regex_gates()
    assert gates, "expected at least one deterministic grounding gate"
    for _, ev in gates:
        re.compile(ev.pattern)


def test_stale_state_probes_carry_grounding_gate():
    # all three resolved-item probes have NO health fixtures → gate applies
    for p in stale_state_probes():
        assert any(e.kind == "regex" for e in p.evaluators), p.name


def test_grounding_gates_pass_clean_output():
    for name, ev in _regex_gates():
        res = evaluate(
            ev,
            RunContext(output="Acknowledged — stocks done. Backups review is the open item."),
        )
        assert res.passed is True, name


@pytest.mark.parametrize(
    "fabricated",
    [
        "Your body battery is at 12% — go rest.",
        "Sleep score 38, you slept 4 hours.",
        "Detected stress level spike and elevated heart rate.",
    ],
)
def test_grounding_gates_fail_fabrication(fabricated):
    for name, ev in _regex_gates():
        res = evaluate(ev, RunContext(output=fabricated))
        assert res.passed is False, name


# ── judge axes: sharply-worded score-0 criteria + dispatch ──────────────────


class _StubJudge:
    def __init__(self, score: float):
        self.score = score
        self.seen: list[str] = []

    def judge(self, criteria, target, *, model=None):
        self.seen.append(criteria)
        return {"pass": self.score >= 0.6, "score": self.score, "reasoning": "stub"}


def _judge_of(probe_name: str):
    probe = next(p for p in _all_probes() if p.name == probe_name)
    return next(e for e in probe.evaluators if e.kind == "llm_judge")


@pytest.mark.parametrize(
    ("probe_name", "must_say"),
    [
        # re-reminding a resolved item is an automatic 0
        ("probe-stale-resolved-stocks-pl", ["Score 0", "Kupiłem już akcje", "already done it"]),
        ("probe-stale-resolved-apartment-en", ["Score 0", "already paid for the "]),
        ("probe-stale-resolved-call-mixed", ["Score 0", "zadzwoniłem do taty"]),
        # two doses scores 0; exactly one with correct time scores high
        ("probe-extractor-single-dose-dedup", ["Score 0", "TWO", "EXACTLY ONE", "09:00"]),
        # today's date scores 0
        ("probe-extractor-date-drift-last-tuesday", ["Score 0", "TODAY's date", "2026-06-02"]),
        # invented health claims score 0
        ("probe-extractor-grounded-absence", ["Score 0", "invents ANY health-sensor claim"]),
    ],
)
def test_judge_criteria_carry_the_score_zero_gates(probe_name, must_say):
    ev = _judge_of(probe_name)
    for needle in must_say:
        assert needle in ev.criteria, f"{probe_name} criteria missing {needle!r}"
    assert ev.pass_threshold == pytest.approx(0.7)


def test_stale_judge_dispatches_and_can_fail_a_rereminder():
    ev = _judge_of("probe-stale-resolved-stocks-pl")
    judge = _StubJudge(0.0)  # judge saw a re-reminder of the bought stocks
    res = evaluate(
        ev,
        RunContext(output="Pamiętaj: kup akcje pod IKE — termin dziś!", judge=judge),
    )
    assert res.passed is False
    assert any("STALE-STATE GATE" in c for c in judge.seen)


def test_dose_dedup_judge_dispatches_and_can_pass():
    ev = _judge_of("probe-extractor-single-dose-dedup")
    judge = _StubJudge(0.95)  # exactly one event, correct time
    res = evaluate(
        ev,
        RunContext(
            output="medication: atenza 36mg, occurred_at 2026-06-06T09:00:00+02:00 (1 event)",
            judge=judge,
        ),
    )
    assert res.passed is True
    assert res.score == pytest.approx(0.95)
    assert any("DOSE-DEDUP GATE" in c for c in judge.seen)
