"""Autoloop probe suite + evaluator tests — fully mocked (no live LLM / DB).

Covers: (1) context assembly mirrors the live shape and makes health
presence/absence explicit; (2) the deterministic grounding gate passes clean
output and HARD-FAILS fabricated-health output through the framework's
evaluate() dispatcher; (3) the judge axes (variety/anti-repetition/grounded/
skip) dispatch to an injected judge; (4) proactive_probes() produces the four
required axes and is attached to the proactive agents.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from app.agents.proactive_probes import (
    assemble_probe_context,
    grounding_absent_health_regex,
    proactive_probes,
)
from src.testing.evaluation import RunContext, evaluate


# ── context assembly ─────────────────────────────────────────────────────────


def test_assemble_marks_health_absent():
    ctx = assemble_probe_context(digest="d", history=[("t", "user", "hi")])
    assert "GROUNDING CONTRACT" in ctx
    assert "NONE — say nothing about health" in ctx
    assert "## Conversation digest" in ctx
    assert "## Chat History (last 24h)" in ctx
    assert "## TASK" in ctx


def test_assemble_marks_health_present_and_renders_figures():
    ctx = assemble_probe_context(
        history=[("t", "user", "hi")],
        activity=[("t1", "desk / coding", {"body_battery": 11, "stress": 68})],
    )
    assert "Health data present this tick: YES" in ctx
    assert "body_battery=11" in ctx
    assert "stress=68" in ctx


def test_assemble_omits_empty_sections():
    ctx = assemble_probe_context()
    assert "## Conversation digest" not in ctx
    assert "## Chat History" not in ctx
    # the grounding contract + task block are always present
    assert "GROUNDING CONTRACT" in ctx
    assert "## TASK" in ctx


# ── deterministic grounding gate via the framework dispatcher ────────────────


def _gate_evaluator():
    # the gate is a RegexEvaluator; find it on the no-health probe
    probe = next(p for p in proactive_probes() if p.name == "probe-grounded-no-health")
    return next(e for e in probe.evaluators if e.kind == "regex")


def test_grounding_gate_passes_clean_output():
    ev = _gate_evaluator()
    res = evaluate(ev, RunContext(output="The gym habit is at 0/3 this week. Want a quick session?"))
    assert res.passed is True


@pytest.mark.parametrize(
    "fabricated",
    [
        "Your body battery is at 9%, sleep debt is critical.",
        "You're sixteen hours past your bedtime — go to sleep.",
        "Your heart rate is elevated and stress level is high.",
        "Sleep score was 41 and you slept 4 hours.",
    ],
)
def test_grounding_gate_fails_fabrication(fabricated):
    ev = _gate_evaluator()
    res = evaluate(ev, RunContext(output=fabricated))
    assert res.passed is False


def test_grounding_regex_is_case_insensitive():
    import re

    pat = grounding_absent_health_regex()
    assert re.search(pat, "BODY BATTERY at 5%") is None  # fabrication caught regardless of case
    assert re.search(pat, "Let's log the gym session.") is not None


# ── judge axes dispatch to the injected judge ────────────────────────────────


class _StubJudge:
    """Records the criteria it was asked and returns a fixed score."""

    def __init__(self, score: float):
        self.score = score
        self.seen: list[str] = []

    def judge(self, criteria, target, *, model=None):
        self.seen.append(criteria)
        return {"pass": self.score >= 0.6, "score": self.score, "reasoning": "stub"}


def test_skip_judge_axis_scores_via_injected_judge():
    probe = next(p for p in proactive_probes() if p.name == "probe-skip-when-busy")
    judge_ev = next(e for e in probe.evaluators if e.kind == "llm_judge")
    judge = _StubJudge(0.9)
    res = evaluate(judge_ev, RunContext(output="(skip — nothing new)", judge=judge))
    assert res.passed is True
    assert res.score == pytest.approx(0.9)
    assert any("SKIP" in c for c in judge.seen)


def test_variety_judge_axis_can_fail():
    probe = next(p for p in proactive_probes() if p.name == "probe-anti-repetition-evolving")
    judge_ev = next(e for e in probe.evaluators if e.kind == "llm_judge")
    judge = _StubJudge(0.2)  # repeated a deferred topic
    res = evaluate(judge_ev, RunContext(output="Q4 budget report is overdue again!", judge=judge))
    assert res.passed is False
    assert any("REPETITION" in c or "repetition" in c for c in judge.seen)


# ── suite shape + agent wiring ───────────────────────────────────────────────


def test_proactive_probes_cover_all_four_axes():
    names = {p.name for p in proactive_probes()}
    assert "probe-variety-first-tick" in names  # variety
    assert "probe-anti-repetition-evolving" in names  # anti-repetition
    assert "probe-grounded-no-health" in names  # grounded
    assert "probe-grounded-with-health" in names  # grounded (present)
    assert "probe-skip-when-busy" in names  # skip vs deliver


def test_probes_attached_to_proactive_agents():
    from app.agents.domains.goals import agents

    by_id = {a.header.agent_id: a for a in agents()}
    for aid in ("goals/nudge_strategist", "goals/periodic_checker", "goals/winddown"):
        names = {t.name for t in by_id[aid].agent_tests}
        assert any(n.startswith("probe-") for n in names), f"{aid} missing probes"
