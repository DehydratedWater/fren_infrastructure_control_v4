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


def _default_mutators(marker_hint: str = ""):
    """Identity (control) + a prompt-rewriter; a fixed-prefix nudge optional."""
    muts = [IdentityMutator(), LLMPromptRewriter(guidance="Fix the failing tests.")]
    if marker_hint:
        muts.insert(1, PromptPrefixMutator(marker_hint))
    return muts


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
    *, failures_sink: list[dict[str, Any]] | None = None,
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
            ctx = RunContext(output=output, tool_calls=list(calls))
            evs = list(t.evaluators)
            results = [evaluate(e, ctx) for e in evs] if evs else []
            ok = all(r.passed for r in results) if results else True
            passes += 1 if ok else 0
            scores.append(
                statistics.fmean([r.score for r in results]) if results else 1.0
            )
            if failures_sink is not None and not ok:
                for e, r in zip(evs, results):
                    if not r.passed:
                        failures_sink.append({
                            "test": t.name,
                            "prompt": (t.prompt or "")[:160],
                            "evaluator": e.kind,
                            "expected": getattr(e, "needle", None) or getattr(e, "expected", None),
                            "got_output": str(output)[:300],
                            "evidence": r.evidence[:160],
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
    only: set[str] | None = None,
) -> list[ImprovementUnit]:
    """One improvement unit per agent that carries agent_tests.

    `llm` (an LLMMutatorClient) is threaded into each loop's MutationContext so
    the LLMPromptRewriter mutator can actually rewrite prompts. `only` restricts
    to a subset of agent ids.
    """
    units: list[ImprovementUnit] = []
    for agent in all_agents():
        if only is not None and agent.header.agent_id not in only:
            continue
        if not agent.agent_tests:
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
    only: set[str] | None = None,
    max_rounds: int = 2,
    include_branches: bool = True,
):
    """Run the full fleet improvement (agents + branches) and (optionally) promote.

    `llm` is the LLMMutatorClient that powers prompt rewriting (the research).
    `only` restricts to a subset of agent/branch ids. `include_branches` toggles
    the per-branch optimisation pass.
    """
    units = build_agent_units(
        agent_runner_factory, llm=llm, only=only, max_rounds=max_rounds,
    )
    if include_branches:
        units += build_branch_units(
            branch_invoker_factory_for, llm=llm, only=only, max_rounds=max_rounds,
        )
    return run_fleet(
        units,
        snapshots_dir=snapshots_dir,
        project_root=project_root or PROJECT_ROOT,
        promote_threshold=promote_threshold,
        max_workers=max_workers,
        run_label="fleet-improve",
    )
