"""Study domain — registration, grounding-contract prompts, probes, and the
question→grade branch.

v3's study mode hallucinated questions the material couldn't answer. This
suite locks the v4 rebuild's countermeasures:

1. the three study agents register through the standard domain aggregation
   (`all_agent_defs`) so the compiled fleet can spawn them;
2. every prompt carries the HARD GROUNDING CONTRACT phrases (verbatim
   `source:` line, never-invent, insufficiency declaration);
3. the autoloop probes are well-formed — grounding / insufficiency / PL-EN mix
   on the question_master, incomplete + wrong answers on the grader, one
   grounded-plan judge on the planner — each scored by an LLMJudgeEvaluator
   whose criteria encode the score-0 hallucination clause;
4. the `study::question-then-grade` branch is discovered via
   `app.agents.branches.branches()` and passes the deterministic gate tier
   with its step contracts (same invoker test_branch_step_contracts.py uses).
"""

from __future__ import annotations

import pytest

from app.agents.branches import branches
from app.agents.domains import ALL_DOMAINS, all_agent_defs
from app.agents.domains import study
from app.agents.improve import mock_branch_invoker_factory_for
from src import LLMJudgeEvaluator
from src.testing.branch import run_branch_test

STUDY_IDS = {
    "study/question_master",
    "study/answer_grader",
    "study/session_planner",
}

BRANCH_NAME = "study::question-then-grade"


def _study_agents() -> dict:
    return {
        a.header.agent_id: a
        for a in all_agent_defs()
        if a.header.agent_id in STUDY_IDS
    }


def _study_branch():
    matches = [b for b in branches() if b.name == BRANCH_NAME]
    assert len(matches) == 1, f"expected exactly one {BRANCH_NAME!r} branch"
    return matches[0]


# ── 1. registration through the standard aggregation ────────────────────────


def test_study_domain_is_registered():
    assert study in ALL_DOMAINS


def test_all_three_agents_register_via_all_agent_defs():
    # all_agent_defs() also raises on any duplicate agent_id, so a clean call
    # doubles as the "aggregation intact / no collision" check.
    agents = _study_agents()
    assert set(agents) == STUDY_IDS


def test_study_agents_deliver_via_emit_guidance():
    """Interactive fleet agents deliver in persona voice via emit-guidance
    (mirrors goals/twily_goal_interface / food subagents)."""
    for agent_id, agent in _study_agents().items():
        tool_names = {t.header.name for t in agent.extra_tools}
        assert "emit-guidance" in tool_names, agent_id


# ── 2. grounding-contract prompts ────────────────────────────────────────────


@pytest.mark.parametrize("agent_id", sorted(STUDY_IDS))
def test_prompt_carries_hard_grounding_contract(agent_id):
    prompt = _study_agents()[agent_id].system_prompt
    lower = prompt.lower()
    assert "source:" in lower, f"{agent_id}: no source: line contract"
    assert "verbatim" in lower, f"{agent_id}: source span not required verbatim"
    assert "never invent" in lower or "never introduce" in lower, (
        f"{agent_id}: prompt lacks the never-invent clause"
    )
    assert "grounding contract" in lower, agent_id


def test_question_master_prompt_handles_insufficiency_and_flow():
    prompt = _study_agents()["study/question_master"].system_prompt
    lower = prompt.lower()
    # insufficiency: declare and ask for more material instead of inventing
    assert "insufficient" in lower
    assert "more material" in lower
    # orchestrator flow: dispatch the grader after the user answers
    assert "study/answer_grader" in prompt
    # PL/EN: the user studies in both languages
    assert "polish" in lower and "english" in lower


def test_answer_grader_prompt_grades_zero_to_ten_from_material_only():
    prompt = _study_agents()["study/answer_grader"].system_prompt
    lower = prompt.lower()
    assert "/10" in prompt or "0-10" in prompt
    assert "what was missed" in lower
    assert "model answer" in lower
    # grades against the material, not outside knowledge
    assert "not against your" in lower or "absent from the material" in lower


def test_session_planner_prompt_is_spaced_repetition_material_only():
    prompt = _study_agents()["study/session_planner"].system_prompt
    lower = prompt.lower()
    assert "spaced" in lower or "spacing" in lower
    assert "never invent topics" in lower


# ── 3. probes — well-formed, hallucination scored 0 ──────────────────────────


def _judge_evaluators(test):
    return [e for e in test.evaluators if isinstance(e, LLMJudgeEvaluator)]


def test_question_master_carries_the_three_probes():
    tests = {t.name: t for t in _study_agents()["study/question_master"].agent_tests}
    assert {
        "probe-grounded-question-from-material",
        "probe-insufficient-material-declines",
        "probe-pl-en-mixed-session",
    } <= set(tests)
    for t in tests.values():
        judges = _judge_evaluators(t)
        assert judges, f"{t.name}: no LLMJudgeEvaluator"
        assert t.prompt.strip(), f"{t.name}: empty probe prompt"


def test_grounding_probe_zeroes_hallucinated_questions():
    tests = {t.name: t for t in _study_agents()["study/question_master"].agent_tests}
    grounding = _judge_evaluators(tests["probe-grounded-question-from-material"])[0]
    crit = grounding.criteria.lower()
    # the v3-failure clause: ungrounded / missing-verbatim-source → score 0
    assert "score 0" in crit
    assert "source:" in crit
    assert "verbatim" in crit
    # the probe ships the material inline (2-paragraph fragment)
    probe = tests["probe-grounded-question-from-material"]
    assert "quorum" in probe.prompt and "150 and 300 milliseconds" in probe.prompt


def test_insufficiency_probe_requires_decline_not_invention():
    tests = {t.name: t for t in _study_agents()["study/question_master"].agent_tests}
    probe = tests["probe-insufficient-material-declines"]
    crit = _judge_evaluators(probe)[0].criteria.lower()
    assert "score 0 unless" in crit
    assert "insufficient" in crit
    assert "more material" in crit
    # thin material: the probe's inline material is two sentences
    assert "easier to understand than paxos" in probe.prompt.lower()


def test_pl_en_probe_mixes_polish_material_with_english_request():
    tests = {t.name: t for t in _study_agents()["study/question_master"].agent_tests}
    probe = tests["probe-pl-en-mixed-session"]
    assert "Walidacja krzyżowa" in probe.prompt  # Polish material inline
    assert "in English" in probe.prompt  # English question requested
    crit = _judge_evaluators(probe)[0].criteria.lower()
    assert "verbatim" in crit and "score 0" in crit


def test_answer_grader_probes_middle_band_and_wrong_answer():
    tests = {t.name: t for t in _study_agents()["study/answer_grader"].agent_tests}
    assert {
        "probe-incomplete-answer-middle-band",
        "probe-wrong-answer-low-grade-grounded-correction",
    } <= set(tests)

    incomplete = tests["probe-incomplete-answer-middle-band"]
    crit = _judge_evaluators(incomplete)[0].criteria.lower()
    assert "4-7" in crit  # middle band
    assert "score 0" in crit and "absent from the" in crit  # no outside facts

    wrong = tests["probe-wrong-answer-low-grade-grounded-correction"]
    crit = _judge_evaluators(wrong)[0].criteria.lower()
    assert "0-3" in crit  # low band
    assert "source:" in crit
    # both probes carry material + question + answer inline
    for probe in (incomplete, wrong):
        assert "Material:" in probe.prompt
        assert "Question:" in probe.prompt
        assert "My answer:" in probe.prompt


def test_session_planner_probe_rejects_invented_topics():
    tests = {t.name: t for t in _study_agents()["study/session_planner"].agent_tests}
    assert "probe-plan-no-invented-topics" in tests
    probe = tests["probe-plan-no-invented-topics"]
    crit = _judge_evaluators(probe)[0].criteria.lower()
    assert "score 0" in crit and "invents topics" in crit
    assert "Material:" in probe.prompt and "7 days" in probe.prompt


# ── 4. the question→grade branch on the deterministic gate tier ─────────────


def test_branch_is_discovered_via_fleet_branches():
    branch = _study_branch()
    assert branch.entry_agent == "study/question_master"
    assert branch.path == ("study/answer_grader",)
    assert "study/answer_grader" in branch.subagent_mocks
    assert branch.step_contracts, "branch must carry step contracts"


def test_branch_step_contracts_assert_input_and_output():
    branch = _study_branch()
    (contract,) = branch.step_contracts
    assert contract.step == "study/answer_grader"
    assert contract.input_evaluators, "context-forwarding contract missing"
    assert contract.output_evaluators, "output-discipline contract missing"
    # deterministic satisfiability: the input needle must be a token of the
    # branch prompt (the gate invoker forwards the prompt in the spawn command)
    for ev in contract.input_evaluators:
        assert ev.needle.lower() in branch.prompt.lower()
    # ... and the output needles must be satisfied by the declared mock
    mock = str(branch.subagent_mocks[contract.step]).lower()
    for ev in contract.output_evaluators:
        assert ev.needle.lower() in mock


def test_branch_passes_deterministic_tier_with_contracts():
    """Same gate test_branch_step_contracts.py applies fleet-wide, run
    explicitly for the study branch: full pass (path + joint evaluators +
    every step contract) via the autoloop's deterministic invoker."""
    branch = _study_branch()
    invoker = mock_branch_invoker_factory_for(branch.entry_agent)({})
    result = run_branch_test(branch, invoker)
    failing = [
        f"{r.evaluator_name}: {r.evidence}" for r in result.results if not r.passed
    ]
    assert result.passed, f"{branch.name} failed: {failing}"
    step_results = [
        r for r in result.results if r.evaluator_name.startswith("step:")
    ]
    assert step_results and all(r.passed for r in step_results)
