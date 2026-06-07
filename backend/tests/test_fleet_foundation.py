"""Foundation tests — compile the fleet + run per-agent & per-branch improvement.

Deterministic, no network: the agent runner / branch invoker echo what the
embedded tests expect, so the whole pipeline (define → compile → test →
improve → promote) is exercised without a model.
"""

from __future__ import annotations

from pathlib import Path

from app.agents.compile import compile_fleet
from app.agents.config import DEFAULT_WORKER, WORKER_VARIANTS
from app.agents.improve import build_agent_units, build_branch_units
from app.agents.registry import all_agents, build_registry
from src import run_fleet
from src.testing.branch import BranchTrajectory
from src.testing.evaluation import ToolCallRecord


def test_agents_are_unique_and_persona_present():
    agents = all_agents()
    ids = [a.header.agent_id for a in agents]
    assert len(ids) == len(set(ids))  # no dupes
    assert "persona/orchestrator" in ids


def test_registry_builds_with_a_slot_per_agent():
    reg = build_registry(project_root=Path("/nonexistent"))
    resolved = reg.resolve_config("prod")
    assert "persona/orchestrator" in resolved


def test_compile_default_variant_writes_md(tmp_path):
    compile_fleet(
        target=tmp_path / "build",
        project_root=tmp_path,
        variants=[DEFAULT_WORKER],
    )
    md = {p.relative_to(tmp_path / "build").as_posix()
          for p in (tmp_path / "build").rglob("*.md")}
    assert ".opencode/agents/persona/orchestrator.md" in md
    assert ".opencode/agents/persona/quick_ack.md" in md


def test_split_variant_compiles_with_postfix(tmp_path):
    split = next(v for v in WORKER_VARIANTS if v.name == "splitqwen35")
    compile_fleet(target=tmp_path / "b", project_root=tmp_path, variants=[split])
    names = {p.name for p in (tmp_path / "b").rglob("*.md")}
    assert any(n.endswith("-splitqwen35.md") for n in names)


# --- deterministic runners for the improvement pipeline --------------------

def _agent_runner_factory(_definition):
    from app.agents.improve import is_delivery_agent

    def runner(_defn, test):
        # echo an output that satisfies the persona orchestrator's substring
        # evaluator ("context"). A DELIVERY agent (emit_guidance.py in its
        # allow-list) must DELIVER via emit_guidance or it scores 0 (its assistant
        # text is invisible in production) — so emit the payload through the tool.
        text = "I will analyse the context first, then plan."
        if is_delivery_agent(_defn):
            return (text, [ToolCallRecord(
                name="bash",
                args={"command": "python scripts/emit_guidance.py --data "
                      '\'{"intent":"reply","key_points":["%s"]}\'' % text},
            )])
        return (text, [])
    return runner


def _branch_invoker_factory_for(_entry_agent):
    from app.agents.improve import is_delivery_agent

    def factory(_defn):
        def invoke(test):
            calls = [ToolCallRecord(name=s) for s in test.path]
            # A delivery orchestrator must deliver via emit_guidance or it scores 0.
            if is_delivery_agent(_defn):
                calls.append(ToolCallRecord(
                    name="bash",
                    args={"command": "python scripts/emit_guidance.py --data "
                          '\'{"key_points":["Here is a plan for your week, with '
                          'context."]}\''},
                ))
            return BranchTrajectory(
                output="Here is a plan for your week, with context.",
                tool_calls=calls,
            )
        return invoke
    return factory


def test_agent_units_improve_and_promote(tmp_path):
    units = build_agent_units(_agent_runner_factory)
    assert units, "expected at least one agent with embedded agent_tests"
    result = run_fleet(
        units,
        snapshots_dir=tmp_path / "snaps",
        project_root=tmp_path,
        promote_threshold=1.0,
    )
    assert result.summary()["failed"] == 0
    orch = result.by_id("persona/orchestrator")
    assert orch is not None and orch.winner_score == 1.0


def test_branch_units_run_and_pass(tmp_path):
    units = build_branch_units(_branch_invoker_factory_for)
    assert units
    result = run_fleet(
        units,
        snapshots_dir=tmp_path / "snaps",
        project_root=tmp_path,
        promote_threshold=1.0,
    )
    assert result.summary()["failed"] == 0
    branch = result.by_id("branch:persona/orchestrator")
    assert branch is not None and branch.winner_score == 1.0
