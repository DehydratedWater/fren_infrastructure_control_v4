"""Per-agent and per-branch improvement, wired to the framework fleet harness.

This is where the fleet's "every agent self-improves, and every branch is
optimised" requirement is assembled. The framework supplies the loops
(`IterativeLoop`, `build_branch_loop`) and the parallel runner (`run_fleet`);
this module supplies the fren-specific EVALUATORS — how an agent / a branch is
actually run to produce a score.

Tiering (the both-tier decision) lives in the injected invoker factories:
- gate tier: a deterministic mock that reflects the candidate prompt so a round
  is cheap and reproducible;
- promote tier: a live `opencode` run (wired in app/runtime/runner.py).

The harness only promotes a winner that clears the score threshold AND the hard
criteria, into `.oac/promoted/`, where the registry picks it up next compile.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Callable

from app.agents.branches import branches
from app.agents.registry import PROJECT_ROOT, all_agents
from src import (
    AgentDefinition,
    AgentTest,
    IdentityMutator,
    ImprovementUnit,
    IterativeLoop,
    LLMPromptRewriter,
    PromptPrefixMutator,
    agent_unit,
    branch_unit,
    build_branch_loop,
    run_fleet,
)
# The improvement Criterion is shadowed at top-level by workflow's Criterion —
# import the optimisation criteria from the improvement package explicitly.
from src.improvement import Criterion, OptimisationCriterion
from src.improvement.mutators import MutationContext
from src.improvement.version import ComponentVersion
from src.testing.branch import BranchInvoker
from src.testing.evaluation import RunContext, ToolCallRecord, evaluate

# How a candidate agent definition is run for ONE agent test → (output, calls).
AgentRunner = Callable[[dict[str, Any], AgentTest], tuple[Any, list[ToolCallRecord]]]
# Build an AgentRunner for a candidate (lets the live tier compile per candidate).
AgentRunnerFactory = Callable[[dict[str, Any]], AgentRunner]
# Build a BranchInvoker for a candidate orchestrator definition.
from src.improvement.branch import BranchInvokerFactory  # noqa: E402

PASS = OptimisationCriterion(
    name="pass-tests",
    criteria=(Criterion(kind="pass_rate", target=1.0, hard=True),),
)

# Graded criterion for judge-scored tests: reward higher score_floor (continuous
# 0..1) rather than all-or-nothing. The best-scoring candidate wins; a candidate
# that lifts the score (e.g. 0.4 → 0.8) is promotable even if not perfect.
GRADED = OptimisationCriterion(
    name="lift-judge-score",
    aggregation="weighted",
    criteria=(Criterion(kind="score_floor", target=1.0, weight=1.0),),
)


def _default_mutators(marker_hint: str = ""):
    """Identity (control) + a prompt-rewriter; a fixed-prefix nudge optional."""
    muts = [IdentityMutator(), LLMPromptRewriter(guidance="Fix the failing tests.")]
    if marker_hint:
        muts.insert(1, PromptPrefixMutator(marker_hint))
    return muts


_PROBE_CACHE: dict[str, str] = {}

# Appended to every probe so the model returns a REAL answer, not a meta-demo.
# (Root cause of many 0-scores: the old prompt said "demonstrate your core
# function", so agents narrated/simulated themselves instead of doing the job.)
_ANTI_META = (
    "\n\nReply with your actual response for THIS request only. Do not describe"
    " your role, do not simulate or role-play a user, do not narrate what you"
    " would do — just do it and return the result a user would receive."
)


def synthesize_probe(agent: AgentDefinition) -> str:
    """One concrete, realistic user message that exercises the agent's role.

    Generated once by the teacher (GLM-5.1) and cached. A concrete task is what
    makes the judge signal real — the old self-referential "demonstrate your
    core function" fallback induced meta-commentary that any honest judge scores
    0. Falls back to a direct task string if the teacher is unavailable.
    """
    aid = agent.header.agent_id
    if aid in _PROBE_CACHE:
        return _PROBE_CACHE[aid]
    role = (agent.usage_explanation_long or agent.usage_explanation_short
            or agent.header.description or aid)
    msg = ""
    try:
        from app.agents.improve_live import _zai_chat
        from app.settings import get_settings

        sys = (
            "You write ONE realistic user message that triggers the described"
            " agent. CRITICAL: the message must be FULLY SELF-CONTAINED — the"
            " agent has NO other context. If the task needs input (a text to"
            " rewrite, data to analyse, a list, a task to categorise), INCLUDE"
            " realistic example content INLINE in the message. Never say 'this"
            " plan' / 'the above' / 'the task' without actually providing it."
            " Output ONLY the user message — no quotes, no preamble."
        )
        usr = (
            f"AGENT ROLE:\n{role}\n\nWrite the single, self-contained user message"
            " now (include any needed input data inline)."
        )
        msg = _zai_chat(
            get_settings().autoloop_teacher_model,
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            max_tokens=900, temperature=0.4,  # GLM-5.1 reasons; don't starve it
        ).strip().strip('"').strip()
    except Exception:  # noqa: BLE001
        msg = ""
    if not msg:
        msg = f"Here is a real task for you: {role[:160]}. Do it now."
    _PROBE_CACHE[aid] = msg
    return msg


def make_judge_test(agent: AgentDefinition) -> AgentTest:
    """Build a strong, graded role-fulfilment LLMJudge test for `agent`.

    This is the satisfiable, continuous 0..1 signal the autoloop climbs (unlike
    brittle substring checks). The prompt is a CONCRETE task (an authored test
    prompt if present, else a teacher-synthesised user message) plus an
    anti-meta guard; the criterion grades whether the reply actually DID the job.
    """
    from src import AgentTest, LLMJudgeEvaluator

    role = (agent.usage_explanation_long or agent.usage_explanation_short
            or agent.header.description or agent.header.agent_id)
    # ALWAYS synthesise a SELF-CONTAINED probe. Authored agent_test prompts are
    # written for their own (often multi-turn / mocked) harness and routinely
    # reference context that a single-shot judge run never supplies ("rewrite the
    # plan in Sarah's voice") — which left Qwen with an underspecified request and
    # produced empty/flailing output scored 0. A teacher-synthesised task is
    # complete on its own.
    prompt = synthesize_probe(agent).rstrip() + _ANTI_META
    criteria = (
        f"The agent's role is: {role}\n"
        f"The user's request is given to the agent. Score how well the response"
        f" ACTUALLY DOES the agent's job for that request, in the agent's voice."
        f" Score 0.0 if the response merely describes its own role, simulates or"
        f" role-plays a user/conversation, narrates what it 'would' do, refuses,"
        f" echoes the prompt, or leaks tool/JSON mechanics instead of answering."
        f" Reward a concrete, correct, on-task result a user could use as-is.\n"
        f"If instead the response is a bracketed note that the agent acted by"
        f" invoking tools/subagents, judge whether THOSE actions are the right"
        f" ones for this request: correct delegation or tool use for the role"
        f" scores high; flailing on wrong, hallucinated, or blocked tools scores"
        f" low."
    )
    return AgentTest(
        name=f"{agent.header.agent_id}::role-fulfilment",
        prompt=prompt,
        evaluators=(LLMJudgeEvaluator(
            name="role-fulfilment", criteria=criteria, pass_threshold=0.7,
        ),),
    )


def _test_expectations(agent: AgentDefinition) -> list[dict[str, Any]]:
    """A human-readable description of what each agent_test expects, fed to the
    LLM rewriter as `context.failures` so it knows the target to satisfy."""
    out: list[dict[str, Any]] = []
    for t in agent.agent_tests:
        for e in t.evaluators:
            exp = {
                "test": t.name,
                "prompt": (t.prompt or (t.turns[0].prompt if t.turns else ""))[:160],
                "evaluator": e.kind,
            }
            for attr in ("needle", "expected", "pattern", "path", "criteria"):
                v = getattr(e, attr, None)
                if v is not None:
                    exp[attr] = v
            out.append(exp)
    return out


def build_agent_evaluator(
    agent: AgentDefinition, runner_factory: AgentRunnerFactory,
    *, failures_sink: list[dict[str, Any]] | None = None, judge: Any = None,
):
    """Score a candidate by running the agent's embedded agent_tests.

    When `failures_sink` is provided, the specific (test, evaluator, output)
    evidence for every FAILED check is appended to it — so the loop's
    MutationContext can feed it to the LLM rewriter (it then knows exactly what
    to fix). The list is cleared each evaluation so it reflects the latest run.
    """

    def evaluator(version: ComponentVersion) -> dict[str, float]:
        tests = agent.agent_tests
        if failures_sink is not None:
            failures_sink.clear()
        if not tests:
            return {"pass_rate": 1.0}  # nothing to fail
        runner = runner_factory(version.definition)
        passes = 0
        scores: list[float] = []
        for t in tests:
            output, calls = runner(version.definition, t)
            # Trajectory-aware: a handoff/tool agent's real output is its tool
            # calls, not assistant text. Without this, every such agent scores a
            # hard 0 (empty text) even when it delegates correctly. Show the judge
            # the actions so it can grade their appropriateness (the rubric in
            # make_judge_test knows to grade delegation, and penalise flailing on
            # wrong/blocked tools).
            judge_output = output
            if not str(output).strip() and calls:
                traj = " -> ".join(c.name for c in calls)
                judge_output = (
                    "[The agent produced no prose reply; it acted by invoking"
                    f" tools/subagents in this order: {traj}.]"
                )
            ctx = RunContext(output=judge_output, tool_calls=list(calls), judge=judge)
            evs = list(t.evaluators)
            results = [evaluate(e, ctx) for e in evs] if evs else []
            ok = all(r.passed for r in results) if results else True
            passes += 1 if ok else 0
            scores.append(
                statistics.fmean([r.score for r in results]) if results else 1.0
            )
            if failures_sink is not None:
                # Capture evidence for any check that didn't fully pass (score < 1)
                # so the rewriter learns WHAT to fix and WHY (judge reasoning).
                for e, r in zip(evs, results):
                    if not r.passed or r.score < 1.0:
                        failures_sink.append({
                            "test": t.name,
                            "prompt": (t.prompt or "")[:200],
                            "evaluator": e.kind,
                            "criterion": getattr(e, "criteria", None)
                            or getattr(e, "needle", None)
                            or getattr(e, "expected", None),
                            "score": round(r.score, 2),
                            "got_output": str(output)[:400],
                            "judge_reasoning": r.evidence[:250],
                        })
        return {
            "pass_rate": passes / len(tests),
            "score_floor": min(scores) if scores else 1.0,
        }

    return evaluator


def build_agent_units(
    runner_factory: AgentRunnerFactory,
    *,
    criterion: OptimisationCriterion = PASS,
    mutators=None,
    max_rounds: int = 2,
    llm: Any = None,
    judge: Any = None,
    only: set[str] | None = None,
    use_judge_test: bool = False,
) -> list[ImprovementUnit]:
    """One improvement unit per agent.

    `llm` (an LLMMutatorClient) is threaded into each loop's MutationContext so
    the LLMPromptRewriter mutator can actually rewrite prompts. `judge` (a
    JudgeClient) is threaded into the RunContext so LLMJudge tests score live.
    When `use_judge_test` is set, EVERY agent gets a generated graded
    role-fulfilment judge test (so all agents — even those without authored
    tests — are improvable on a continuous signal). `only` restricts to a
    subset of agent ids.
    """
    units: list[ImprovementUnit] = []
    for agent in all_agents():
        if only is not None and agent.header.agent_id not in only:
            continue
        # When using the generated judge test, every agent is improvable;
        # otherwise only those that authored agent_tests.
        if use_judge_test:
            agent = agent.model_copy(update={"agent_tests": [make_judge_test(agent)]})
        elif not agent.agent_tests:
            continue
        baseline = ComponentVersion.of(
            agent.header.agent_id, "agent", agent.model_dump(),
        )
        # Shared failures list: the evaluator writes failed-check evidence into
        # it; the MutationContext reads it so the LLM rewriter knows what to fix.
        # Seed it with the test expectations so the FIRST rewrite is on-target
        # even before the baseline run populates concrete failures.
        failures: list[Any] = _test_expectations(agent)
        ctx = None
        if llm is not None:
            ctx = MutationContext(llm=llm, criterion=criterion, failures=failures)
        loop = IterativeLoop(
            baseline=baseline,
            mutators=mutators or _default_mutators(),
            criterion=criterion,
            evaluator=build_agent_evaluator(
                agent, runner_factory,
                failures_sink=failures if llm is not None else None,
                judge=judge,
            ),
            max_rounds=max_rounds,
            mutation_context=ctx,
        )
        units.append(agent_unit(agent.header.agent_id, loop))
    return units


def build_branch_units(
    invoker_factory_for: Callable[[str], BranchInvokerFactory],
    *,
    criterion: OptimisationCriterion = PASS,
    mutators=None,
    max_rounds: int = 2,
    llm: Any = None,
    only: set[str] | None = None,
) -> list[ImprovementUnit]:
    """One improvement unit per branch.

    `invoker_factory_for(entry_agent)` returns the BranchInvokerFactory for that
    orchestrator (so the consumer can wire mock vs live per entry agent). `llm`
    threads into the MutationContext; `only` restricts to a subset of entry ids.
    """
    by_entry: dict[str, list] = {}
    for b in branches():
        if only is not None and b.entry_agent not in only:
            continue
        by_entry.setdefault(b.entry_agent, []).append(b)

    agent_by_id = {a.header.agent_id: a for a in all_agents()}
    units: list[ImprovementUnit] = []
    for entry_agent, tests in by_entry.items():
        entry_def = agent_by_id[entry_agent].model_dump()
        loop = build_branch_loop(
            entry_agent=entry_agent,
            entry_definition=entry_def,
            tests=tests,
            invoker_factory=invoker_factory_for(entry_agent),
            mutators=mutators or _default_mutators(),
            criterion=criterion,
            max_rounds=max_rounds,
            mutation_context=MutationContext(llm=llm) if llm is not None else None,
        )
        units.append(branch_unit(entry_agent, loop))
    return units


def run_improvement(
    *,
    agent_runner_factory: AgentRunnerFactory,
    branch_invoker_factory_for: Callable[[str], BranchInvokerFactory],
    snapshots_dir: Path | None = None,
    promote_threshold: float | None = 1.0,
    project_root: Path | None = None,
    max_workers: int = 4,
    llm: Any = None,
    judge: Any = None,
    only: set[str] | None = None,
    max_rounds: int = 2,
    include_branches: bool = True,
    criterion: OptimisationCriterion = PASS,
    use_judge_test: bool = False,
):
    """Run the full fleet improvement (agents + branches) and (optionally) promote.

    `llm` is the LLMMutatorClient that powers prompt rewriting (the research).
    `judge` is the JudgeClient that scores LLMJudge tests live. `criterion`
    selects the scoring goal (use GRADED for judge-scored continuous lift).
    `only` restricts to a subset of agent/branch ids.
    """
    units = build_agent_units(
        agent_runner_factory, llm=llm, judge=judge, only=only,
        max_rounds=max_rounds, criterion=criterion, use_judge_test=use_judge_test,
    )
    if include_branches:
        units += build_branch_units(
            branch_invoker_factory_for, llm=llm, only=only,
            max_rounds=max_rounds, criterion=criterion,
        )
    return run_fleet(
        units,
        snapshots_dir=snapshots_dir,
        project_root=project_root or PROJECT_ROOT,
        promote_threshold=promote_threshold,
        max_workers=max_workers,
        run_label="fleet-improve",
    )
