"""Step-contract gate — every branch's StepContracts hold in the DETERMINISTIC tier.

Every fleet BranchTest now carries `step_contracts` (per-step subagent
assertions: context-forwarding on the dispatch payload + output discipline on
the sub-agent reply). A contract that the deterministic tier cannot satisfy
would zero that branch's score in every autoloop round and block all
promotions — so this suite is the gate that proves each contract is satisfiable
from the branch's own `prompt` + `subagent_mocks` BEFORE it is allowed to gate
anything.

Why NOT the framework's `mock_chain_invoker`: it builds the trajectory with
EMPTY call args, so every input (context-forwarding) evaluator would run
against `{}` and fail. The autoloop's deterministic tier instead uses
`mock_branch_invoker_factory_for` (app.agents.improve — also what
tests/test_fleet_foundation.py drives `build_branch_units` with), which mirrors
the LIVE tier's recording shape from `subagent_dispatch_chain`
(app/runtime/runner.py):

  args   = {"via": "spawn", "command": "… run --agent <step> '<prompt>'"}
  output = <the sub-agent's reply / the step's subagent_mock>

so a contract that passes here asserts exactly the fields a live run records.
"""

from __future__ import annotations

import pytest

from app.agents.branches import branches
from app.agents.improve import mock_branch_invoker_factory_for
from src.testing.branch import run_branch_test

ALL_BRANCHES = branches()
_IDS = [b.name for b in ALL_BRANCHES]


def _run(branch, *, strip_contracts: bool = False):
    if strip_contracts:
        branch = branch.model_copy(update={"step_contracts": ()})
    invoker = mock_branch_invoker_factory_for(branch.entry_agent)({})
    return run_branch_test(branch, invoker)


def test_every_branch_carries_step_contracts():
    """The per-branch subagent-contract coverage decision: all 22 branches."""
    missing = [b.name for b in ALL_BRANCHES if not b.step_contracts]
    assert not missing, f"branches without step_contracts: {missing}"


@pytest.mark.parametrize("branch", ALL_BRANCHES, ids=_IDS)
def test_branch_passes_deterministic_tier_with_contracts(branch):
    """Full pass (path + branch evaluators + every step contract) on the gate
    tier — proves the contracts are deterministic-tier satisfiable."""
    result = _run(branch)
    failing = [
        f"{r.evaluator_name}: {r.evidence}" for r in result.results if not r.passed
    ]
    assert result.passed, f"{branch.name} failed: {failing}"


@pytest.mark.parametrize("branch", ALL_BRANCHES, ids=_IDS)
def test_step_contracts_actually_ran_and_passed(branch):
    """Contracts must have produced results (not been skipped wholesale) and
    every one of them must pass against the recorded dispatches."""
    result = _run(branch)
    step_results = [
        r for r in result.results if r.evaluator_name.startswith("step:")
    ]
    assert step_results, f"{branch.name}: no step-contract results recorded"
    # every contract contributed (>= one result per declared contract)
    assert len(step_results) >= len(branch.step_contracts)
    failing = [
        f"{r.evaluator_name}: {r.evidence}" for r in step_results if not r.passed
    ]
    assert not failing, f"{branch.name} step-contract failures: {failing}"


@pytest.mark.parametrize("branch", ALL_BRANCHES, ids=_IDS)
def test_contracts_do_not_drop_gate_score(branch):
    """Adding contracts must never LOWER a branch's deterministic-tier score —
    a dropped score here would silently weaken every autoloop gate round."""
    with_contracts = _run(branch)
    without_contracts = _run(branch, strip_contracts=True)
    assert with_contracts.score >= without_contracts.score, (
        f"{branch.name}: score dropped from {without_contracts.score:.3f} to"
        f" {with_contracts.score:.3f} once step_contracts were added"
    )


def test_input_contracts_see_forwarded_prompt_not_empty_args():
    """Regression guard for the invoker contract itself: the deterministic
    invoker must record the spawn command (carrying the forwarded prompt) in
    call args — the live tier's `subagent_dispatch_chain` shape — otherwise
    every context-forwarding input evaluator silently runs against `{}`."""
    branch = ALL_BRANCHES[0]
    invoker = mock_branch_invoker_factory_for(branch.entry_agent)({})
    trajectory = invoker(branch)
    spawns = [c for c in trajectory.tool_calls if c.args.get("via") == "spawn"]
    assert spawns, "deterministic invoker recorded no spawn-shaped dispatches"
    for call in spawns:
        assert "--agent" in call.args["command"]
        assert branch.prompt in call.args["command"]
