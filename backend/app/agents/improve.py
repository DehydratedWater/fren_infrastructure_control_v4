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


def build_agent_evaluator(
    agent: AgentDefinition, runner_factory: AgentRunnerFactory,
):
    """Score a candidate by running the agent's embedded agent_tests."""

    def evaluator(version: ComponentVersion) -> dict[str, float]:
        tests = agent.agent_tests
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
) -> list[ImprovementUnit]:
    """One improvement unit per agent that carries agent_tests."""
    units: list[ImprovementUnit] = []
    for agent in all_agents():
        if not agent.agent_tests:
            continue
        baseline = ComponentVersion.of(
            agent.header.agent_id, "agent", agent.model_dump(),
        )
        loop = IterativeLoop(
            baseline=baseline,
            mutators=mutators or _default_mutators(),
            criterion=criterion,
            evaluator=build_agent_evaluator(agent, runner_factory),
            max_rounds=max_rounds,
        )
        units.append(agent_unit(agent.header.agent_id, loop))
    return units


def build_branch_units(
    invoker_factory_for: Callable[[str], BranchInvokerFactory],
    *,
    criterion: OptimisationCriterion = PASS,
    mutators=None,
    max_rounds: int = 2,
) -> list[ImprovementUnit]:
    """One improvement unit per branch.

    `invoker_factory_for(entry_agent)` returns the BranchInvokerFactory for that
    orchestrator (so the consumer can wire mock vs live per entry agent).
    """
    by_entry: dict[str, list] = {}
    for b in branches():
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
):
    """Run the full fleet improvement (agents + branches) and (optionally) promote."""
    units = build_agent_units(agent_runner_factory) + build_branch_units(
        branch_invoker_factory_for
    )
    return run_fleet(
        units,
        snapshots_dir=snapshots_dir,
        project_root=project_root or PROJECT_ROOT,
        promote_threshold=promote_threshold,
        max_workers=max_workers,
        run_label="fleet-improve",
    )
