"""Regression tests for the autoloop scoring machinery.

Every bug in the long debugging saga had ZERO coverage, which is why they
recurred. These lock the fixes — all pure / mocked, no live opencode, qwen, or
z.ai:

1. opencode error events are SURFACED, never swallowed as empty text (the
   "Agent not found" mass-zero bug).
2. the teacher (GLM) runs THROUGH opencode, never the raw z.ai API (ToS).
3. probes are disk-cached and fallbacks are never persisted.
4. the judge test is concrete + anti-meta with an LLMJudge evaluator.
5. the evaluator judges the TOOL TRAJECTORY when an agent emits no prose.
6. the workspace resolves to the opencode project root; teacher agents are flat.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("src")

from src import AgentDefinition, AgentHeader, LLMJudgeEvaluator
from src.improvement.version import ComponentVersion
from src.testing.evaluation import ToolCallRecord


def _agent(aid="persona/responding"):
    return AgentDefinition(
        header=AgentHeader(agent_id=aid, name="X", description="d"),
        usage_explanation_long="Rewrites text warmly for users.",
        usage_explanation_short="rewriter",
    )


# --- 1. opencode errors must be surfaced, never swallowed -------------------

def test_opencode_error_event_is_surfaced_not_swallowed():
    from app.runtime.runner import opencode_errors, parse_opencode_events

    stdout = json.dumps({
        "type": "error",
        "error": {"name": "UnknownError",
                  "data": {"message": 'Agent not found: "x". Available agents: build'}},
    })
    # the parser yields NO assistant text for an error-only stream ...
    text, calls = parse_opencode_events(stdout)
    assert text == "" and calls == []
    # ... but the error MUST be detectable, so the runner can surface it as
    # result.error instead of scoring an empty string 0 (THE mass-zero bug).
    errs = opencode_errors(stdout)
    assert errs and "Agent not found" in errs[0]


def test_opencode_errors_empty_for_normal_stream():
    from app.runtime.runner import opencode_errors

    stdout = json.dumps({"part": {"type": "text", "text": "hi"}})
    assert opencode_errors(stdout) == []


def test_make_branch_judge_test_grades_outcome_not_path():
    """Non-single-shot test for orchestrators: grade the OUTCOME (with the dispatch
    path as a soft hint), not a brittle path-match."""
    import app.agents.improve as im
    from src import BranchTest

    b = BranchTest(name="food/orchestrator::suggest", entry_agent="food/orchestrator",
                   prompt="Suggest a quick dinner.", path=("food/food_suggester",))
    t = im.make_branch_judge_test(b)
    assert "Suggest a quick dinner." in t.prompt
    assert isinstance(t.evaluators[0], LLMJudgeEvaluator)
    crit = t.evaluators[0].criteria
    assert "FINAL RESPONSE" in crit and "food/food_suggester" in crit  # outcome + soft hint


def test_build_branch_evaluator_judges_full_session_outcome():
    """The orchestrator is run as a full multi-step session; when it acts via
    tools with no prose, the evaluator shows the JUDGE the dispatch chain."""
    import app.agents.improve as im
    from src import BranchTest
    from src.improvement.version import ComponentVersion
    from src.testing.branch import BranchTrajectory

    b = BranchTest(name="x::y", entry_agent="x", prompt="do the task", path=("a", "b"))
    seen = {}

    class FakeJudge:
        def judge(self, criteria, target, *, model=None):
            seen["target"] = str(target)
            return {"pass": True, "score": 0.9, "reasoning": "outcome ok"}

    def invoker_factory(_defn):
        def invoke(_test):
            # acted via tools, produced no prose
            return BranchTrajectory(output="", tool_calls=[
                ToolCallRecord(name="a"), ToolCallRecord(name="b")])
        return invoke

    ev = im.build_branch_evaluator([b], invoker_factory, judge=FakeJudge())
    metrics = ev(ComponentVersion.of("x", "agent", {"system_prompt": "p", "name": "x"}))
    # the judge saw the dispatch chain (not an empty string)
    assert "acted via" in seen["target"] and "a -> b" in seen["target"]
    assert metrics["score_floor"] == 0.9


def test_blocked_tool_attempts_counts_denied_calls():
    """Behavioural smell: an agent debug-flailing on forbidden commands. Unit
    tests can't run the agent, but the DETECTOR a live smoke uses is tested here."""
    from app.runtime.runner import blocked_tool_attempts

    denied = {"type": "tool", "tool": "bash", "state": {
        "status": "error", "input": {"command": "pip install pydantic"},
        "output": "The user has specified a rule which prevents you from using "
                  "this specific tool call."}}
    ok = {"type": "tool", "tool": "bash", "state": {
        "status": "completed", "input": {"command": "python scripts/x.py"},
        "output": "done"}}
    stdout = "\n".join(json.dumps({"part": p}) for p in (denied, ok, denied))
    assert blocked_tool_attempts(stdout) == 2
    assert blocked_tool_attempts(json.dumps({"part": ok})) == 0


def test_subagent_dispatch_chain_extracts_spawned_agents():
    """An orchestrator spawns sub-agents via `…opencode_manager.py run --agent X`;
    the real dispatch chain must be pulled from those bash commands, not the raw
    `bash` tool names."""
    from app.runtime.runner import subagent_dispatch_chain

    stdout = "\n".join([
        json.dumps({"part": {"type": "tool", "tool": "bash", "state": {"input": {
            "command": "uv run scripts/opencode_manager.py run --agent context_analyzer 'hi'"}}}}),
        json.dumps({"part": {"type": "tool", "tool": "read"}}),
        json.dumps({"part": {"type": "tool", "tool": "bash", "state": {"input": {
            "command": "uv run scripts/opencode_manager.py run --agent persona/thinking 'x'"}}}}),
    ])
    chain = subagent_dispatch_chain(stdout)
    assert [c.name for c in chain] == ["context_analyzer", "persona/thinking"]


# --- 2. teacher runs THROUGH opencode, not the raw z.ai API -----------------

def test_zai_chat_routes_through_opencode_not_raw_api(monkeypatch):
    import app.agents.improve_live as il
    from app.runtime.runner import AgentRunResult

    captured = {}

    async def fake_run(*, agent_dir, agent_name, prompt, timeout_s=120, **kw):
        captured["agent_name"] = agent_name
        captured["prompt"] = prompt
        return AgentRunResult(text="REWRITTEN")

    monkeypatch.setattr(il, "run_agent_opencode", fake_run)
    monkeypatch.setattr(il, "_ensure_teacher_agent",
                        lambda ref: (Path("/tmp"), "teacher_zai_coding_plan_glm_5_1"))
    # a direct z.ai HTTP call would be a ToS violation — make any attempt explode
    import httpx
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: pytest.fail("direct z.ai API call!"))

    out = il._zai_chat("glm-5.1", [
        {"role": "system", "content": "SYS-INSTRUCTION"},
        {"role": "user", "content": "USER-CONTENT"},
    ])
    assert out == "REWRITTEN"
    assert captured["agent_name"].startswith("teacher_")  # ran as a teacher agent
    assert "SYS-INSTRUCTION" in captured["prompt"]
    assert "USER-CONTENT" in captured["prompt"]


# --- 3. probe cache: round-trips, never persists fallbacks ------------------

def test_probe_cache_persists_real_skips_fallback(monkeypatch, tmp_path):
    import app.agents.improve as im

    monkeypatch.setattr(im, "PROJECT_ROOT", tmp_path)
    im._PROBE_CACHE.clear()
    im._PROBE_LOADED = True  # don't read any real disk cache
    im._PROBE_CACHE["a/good"] = "A concrete, self-contained probe with data inline."
    im._PROBE_CACHE["a/bad"] = im._FALLBACK_PREFIX + " do the role. Do it now."

    im._write_probe_cache()
    data = json.loads((tmp_path / ".oac" / "probe_cache.json").read_text())
    assert "a/good" in data
    assert "a/bad" not in data  # transient-fallback must NOT poison the cache


def test_probe_cache_loads_from_disk(monkeypatch, tmp_path):
    import app.agents.improve as im

    (tmp_path / ".oac").mkdir(parents=True)
    (tmp_path / ".oac" / "probe_cache.json").write_text(json.dumps({"x/y": "cached probe"}))
    monkeypatch.setattr(im, "PROJECT_ROOT", tmp_path)
    im._PROBE_CACHE.clear()
    im._PROBE_LOADED = False
    im._load_probe_cache()
    assert im._PROBE_CACHE.get("x/y") == "cached probe"


# --- 4. judge test is concrete + anti-meta with an LLMJudge evaluator -------

def test_make_judge_test_is_anti_meta_and_uses_llm_judge(monkeypatch):
    import app.agents.improve as im

    monkeypatch.setattr(im, "synthesize_probe", lambda agent: "Rewrite: 'Downtime tonight.'")
    t = im.make_judge_test(_agent())
    assert "Rewrite: 'Downtime tonight.'" in t.prompt
    # anti-meta guard present (no "demonstrate your core function" meta-prompt)
    low = t.prompt.lower()
    assert "do not" in low and ("simulate" in low or "narrate" in low)
    assert isinstance(t.evaluators[0], LLMJudgeEvaluator)


# --- 5. evaluator judges the tool trajectory when there's no prose ----------

def test_evaluator_judges_trajectory_when_output_blank(monkeypatch):
    import app.agents.improve as im

    monkeypatch.setattr(im, "synthesize_probe", lambda agent: "do the task")
    seen = {}

    class FakeJudge:
        def judge(self, criteria, target, *, model=None):
            seen["target"] = str(target)
            return {"pass": True, "score": 0.9, "reasoning": "acted via tools"}

    agent = _agent().model_copy(update={"agent_tests": [im.make_judge_test(_agent())]})

    def runner_factory(_defn):
        def runner(_d, _test):
            return "", [ToolCallRecord(name="context_analyzer"),
                        ToolCallRecord(name="priority_planner")]
        return runner

    ev = im.build_agent_evaluator(agent, runner_factory, judge=FakeJudge())
    metrics = ev(ComponentVersion.of("persona/responding", "agent", agent.model_dump()))
    # the judge saw a trajectory rendering, NOT an empty string
    assert "tools/subagents" in seen["target"] or "invoking" in seen["target"]
    assert "context_analyzer" in seen["target"]
    assert metrics["score_floor"] == 0.9


# --- 6. workspace = opencode project root; teacher agents are flat ----------

def test_workspace_resolves_to_opencode_json_dir(monkeypatch, tmp_path):
    import app.agents.improve_live as il

    (tmp_path / "opencode.json").write_text("{}")
    (tmp_path / "backend").mkdir()
    monkeypatch.setattr(il, "_project_root", lambda: tmp_path / "backend")
    assert il._workspace() == tmp_path  # walked up to the dir holding opencode.json


def test_ensure_teacher_agent_is_flat_named(monkeypatch, tmp_path):
    import app.agents.improve_live as il

    (tmp_path / ".opencode" / "agents").mkdir(parents=True)
    monkeypatch.setattr(il, "_ensure_workspace", lambda: tmp_path)
    il._TEACHER_AGENTS.clear()
    ws, name = il._ensure_teacher_agent("zai-coding-plan/glm-5.1")
    assert ws == tmp_path
    assert "/" not in name and name.startswith("teacher_")  # FLAT => discoverable
    md = (tmp_path / ".opencode" / "agents" / f"{name}.md").read_text()
    assert "model: zai-coding-plan/glm-5.1" in md


def test_compile_candidate_installs_flat_name(monkeypatch, tmp_path):
    """A candidate must land FLAT-named (cand_*) — nested/slashed names are the
    'Agent not found' bug."""
    import app.agents.improve_live as il

    (tmp_path / ".opencode" / "agents").mkdir(parents=True)
    monkeypatch.setattr(il, "_ensure_workspace", lambda: tmp_path)

    def fake_compile_one(definition, target):
        # emulate the real compiler writing a NESTED/slashed primary
        d = target / ".opencode" / "agents" / "persona"
        d.mkdir(parents=True, exist_ok=True)
        (d / "responding-primary.md").write_text("---\nmodel: local/x\n---\nbody\n")
        return "persona/responding-primary"

    monkeypatch.setattr(il, "_compile_one", fake_compile_one)
    ws, flat = il._compile_candidate(_agent().model_dump())
    assert ws == tmp_path
    assert "/" not in flat and flat.startswith("cand_")  # FLAT
    assert (tmp_path / ".opencode" / "agents" / f"{flat}.md").exists()
