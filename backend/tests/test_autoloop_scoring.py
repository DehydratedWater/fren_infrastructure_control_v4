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


# --- 7. branch units don't cross-contaminate component identity -------------
# Regression for the full+branched run where EVERY branch (orchestrator) unit
# errored at snapshot/promote with the SAME component cited:
#   ValidationError ... content_hash for 'workflows/cron_master' does not match
#   stable_content_hash(definition)
# Root cause: the per-entry branch loop built its baseline with a PLAIN entry-agent
# component_id (`workflows/server`) instead of the `branch:<entry>` namespace. That
# collided a branch winner with the entry agent's own per-agent loop on the shared
# `.oac/snapshots/<component_id>/` dir + the `.oac/promoted/<component_id>` slot, so
# winners snapshotted under another unit's identity and promote() blew up. The fix
# namespaces every branch baseline `branch:<entry>` (+ its own fresh registry), via
# the framework's tested `build_outcome_branch_loop`.


def _two_distinct_branch_entries():
    """Pick two BranchTests with DISTINCT entry agents from the real fleet."""
    from app.agents.branches import branches

    by_entry: dict[str, object] = {}
    for b in branches():
        by_entry.setdefault(b.entry_agent, b)
        if len(by_entry) >= 2:
            break
    assert len(by_entry) >= 2, "fleet must have >= 2 distinct branch entry agents"
    return set(by_entry)


def test_branch_units_promote_with_own_namespaced_hashvalid_snapshot(tmp_path):
    """Each branch unit must promote a snapshot under its OWN namespaced
    component_id, whose content_hash matches stable_content_hash(definition) — no
    cross-unit contamination (the cron_master-for-every-branch bug)."""
    import app.agents.improve as im
    from src import run_fleet
    from src.improvement.branch import branch_component_id
    from src.improvement.snapshot import read_snapshot
    from src.improvement.version import stable_content_hash
    from src.testing.branch import BranchTrajectory

    entries = _two_distinct_branch_entries()

    # Mocks: no live opencode/qwen/z.ai. The invoker echoes a complete answer; the
    # judge always scores high so every unit produces a promotable winner (the
    # snapshot/promote path is what must survive).
    def invoker_factory_for(_entry):
        def factory(_defn):
            def invoke(_test):
                return BranchTrajectory(
                    output="A complete, on-task, useful result for the user.",
                    tool_calls=[],
                )
            return invoke
        return factory

    class FakeJudge:
        def judge(self, criteria, target, *, model=None):
            return {"pass": True, "score": 1.0, "reasoning": "fulfils the request"}

    class FakeLLM:
        # vary the rewrite per call so candidates are distinct (drives real lineage)
        _n = [0]

        def rewrite(self, target, guidance, *, context=None, model=None):
            self._n[0] += 1
            return f"{target}\n\nImproved (v{self._n[0]})."

    units = im.build_branch_units(
        invoker_factory_for, llm=FakeLLM(), judge=FakeJudge(),
        only=entries, max_rounds=2, criterion=im.GRADED, use_judge_test=True,
    )
    assert {u.unit_id for u in units} == {branch_component_id(e) for e in entries}

    result = run_fleet(
        units,
        snapshots_dir=tmp_path / "snaps",
        project_root=tmp_path,
        promote_threshold=0.7,
        run_label="branch-contamination-regression",
    )

    # No unit may error at snapshot/promote time.
    assert result.failed() == [], (
        "branch units errored: "
        + "; ".join(f"{o.unit_id}: {o.error}" for o in result.failed())
    )
    assert len(result.promoted()) == len(entries)

    promoted_dir = tmp_path / ".oac" / "promoted"
    seen_component_ids: set[str] = set()
    for entry in entries:
        expected_id = branch_component_id(entry)  # e.g. "branch:food/orchestrator"
        slot = promoted_dir / (expected_id.replace("/", "__") + ".json")
        assert slot.exists(), f"missing promoted slot for {expected_id}: {slot}"
        snap = read_snapshot(slot)
        v = snap.version
        # OWN identity — never another unit's component (the contamination bug).
        assert v.component_id == expected_id, (
            f"{expected_id} promoted under WRONG component_id {v.component_id!r}"
        )
        # namespaced, so it can never collide with the entry agent's own loop.
        assert v.component_id.startswith("branch:")
        # content_hash is self-consistent (what the Snapshot validator enforces).
        assert v.content_hash == stable_content_hash(v.definition), (
            f"{expected_id}: content_hash != stable_content_hash(definition)"
        )
        seen_component_ids.add(v.component_id)

    # distinct identity per unit
    assert len(seen_component_ids) == len(entries)


# --- 7. tool-discipline signal: forward denied/blocked tools + session errors -
# The scoring blind-spot fix — qwen flailing on allow-list-DENIED tools (and
# opencode session errors) must reach the judge + failures so the loop can learn
# to stop. All mocked; no live opencode/qwen/z.ai.


def _denied_event(tool="bash", cmd="pip install pydantic"):
    return {"part": {"type": "tool", "tool": tool, "state": {
        "status": "error", "input": {"command": cmd},
        "output": "a rule prevents you from using this specific tool call."}}}


def test_parse_populates_tool_call_error_for_denied_part():
    """A denied tool part carries its deny reason on ToolCallRecord.error."""
    from app.runtime.runner import parse_opencode_events

    stdout = "\n".join(json.dumps(e) for e in (
        _denied_event("bash", "ls -la"),
        {"part": {"type": "tool", "tool": "bash", "state": {
            "status": "completed", "input": {"command": "python scripts/x.py"},
            "output": "done"}}},
    ))
    _text, calls = parse_opencode_events(stdout)
    assert len(calls) == 2
    assert calls[0].error and "prevents you from using" in calls[0].error
    assert calls[1].error is None


def test_blocked_tool_details_returns_names_and_reasons():
    from app.runtime.runner import blocked_tool_attempts, blocked_tool_details

    stdout = "\n".join(json.dumps(e) for e in (
        _denied_event("bash", "ls"), _denied_event("read", "open file")))
    details = blocked_tool_details(stdout)
    assert [n for n, _ in details] == ["bash", "read"]
    assert all("prevents you from using" in r for _, r in details)
    assert blocked_tool_attempts(stdout) == 2


class _NoteReadingJudge:
    """Drops the score when it SEES the forwarded TOOL DISCIPLINE note —
    proving the rubric's flailing clause can actually fire off the signal."""

    def __init__(self):
        self.seen = []

    def judge(self, criteria, target, *, model=None):
        self.seen.append(str(target))
        score = 0.2 if "TOOL DISCIPLINE" in str(target) else 0.95
        return {"pass": score >= 0.7, "score": score, "reasoning": "stub"}


def _blocked_runner_factory(blocked, error=None, text="a real-ish answer"):
    def runner_factory(_defn):
        def runner(_d, _test):
            return text, [], {"blocked": blocked, "error": error}
        return runner
    return runner_factory


def test_blocked_attempts_forwarded_to_judge_and_failures(monkeypatch):
    import app.agents.improve as im

    monkeypatch.setattr(im, "synthesize_probe", lambda agent: "do the task")
    agent = _agent().model_copy(update={"agent_tests": [im.make_judge_test(_agent())]})
    judge = _NoteReadingJudge()
    failures: list = []
    ev = im.build_agent_evaluator(
        agent, _blocked_runner_factory([("ls", "a rule prevents you from using ls"),
                                        ("find", "a rule prevents you from using find")]),
        judge=judge, failures_sink=failures,
    )
    metrics = ev(ComponentVersion.of("persona/responding", "agent", agent.model_dump()))
    # the judge SAW the blocked-attempt note (not prose alone)
    assert any("TOOL DISCIPLINE" in s and "ls" in s and "find" in s for s in judge.seen)
    assert any("DENIED/blocked tool" in s for s in judge.seen)
    # failures captured the blocked attempts for the rewriter
    assert failures and failures[0]["blocked_attempts"] == 2
    assert set(failures[0]["blocked_tools"]) == {"ls", "find"}
    # forwarded signal pulled the score down
    assert metrics["score_floor"] < 0.7


def test_session_error_labelled_to_judge_not_blank(monkeypatch):
    import app.agents.improve as im

    monkeypatch.setattr(im, "synthesize_probe", lambda agent: "do the task")
    agent = _agent().model_copy(update={"agent_tests": [im.make_judge_test(_agent())]})
    judge = _NoteReadingJudge()
    failures: list = []
    ev = im.build_agent_evaluator(
        agent, _blocked_runner_factory([], error="opencode error: Agent not found",
                                       text=""),
        judge=judge, failures_sink=failures,
    )
    ev(ComponentVersion.of("persona/responding", "agent", agent.model_dump()))
    assert any("session ERRORED" in s and "Agent not found" in s for s in judge.seen)
    assert failures and "Agent not found" in (failures[0]["error"] or "")


def test_flailing_candidate_scores_lower_than_clean(monkeypatch):
    """Keystone: a flailing candidate scores strictly lower than a clean one."""
    import app.agents.improve as im

    monkeypatch.setattr(im, "synthesize_probe", lambda agent: "do the task")
    agent = _agent().model_copy(update={"agent_tests": [im.make_judge_test(_agent())]})
    v = ComponentVersion.of("persona/responding", "agent", agent.model_dump())

    def _score(blocked):
        ev = im.build_agent_evaluator(
            agent, _blocked_runner_factory(blocked), judge=_NoteReadingJudge(),
        )
        return ev(v)["score_floor"]

    clean = _score([])
    flailing = _score([("ls", "a rule prevents you from using ls")])
    assert flailing < clean, (flailing, clean)


def test_branch_evaluator_forwards_trajectory_error_and_blocked(monkeypatch):
    import app.agents.improve as im
    from src import BranchTest
    from src.testing.branch import BranchTrajectory

    b = BranchTest(name="x::y", entry_agent="x", prompt="do the task", path=("a", "b"))
    judge = _NoteReadingJudge()
    failures: list = []

    def invoker_factory(_defn):
        def invoke(_test):
            return BranchTrajectory(
                output="an answer",
                blocked_tools=[("ls", "a rule prevents you from using ls")],
                error="opencode error: Agent not found",
            )
        return invoke

    ev = im.build_branch_evaluator([b], invoker_factory, judge=judge,
                                   failures_sink=failures)
    metrics = ev(ComponentVersion.of("x", "agent", {"system_prompt": "p", "name": "x"}))
    assert any("TOOL DISCIPLINE" in s and "session ERRORED" in s for s in judge.seen)
    assert metrics["score_floor"] < 0.7
    assert failures and failures[0]["blocked_attempts"] == 1
    assert "Agent not found" in (failures[0]["error"] or "")


# --- 8. production DELIVERY CONTRACT: optimize-IN emit_guidance --------------
# A "delivery agent" (emit_guidance.py in its allow-list) MUST call
# emit_guidance.py to deliver — its assistant text is invisible in production.
# The evaluator: no emit → 0; emit → grade the PAYLOAD. The teacher must preserve
# the contract; the compile postamble must carry it. All mocked.


def _delivery_agent(aid="support/daily_briefer"):
    """An agent whose allow-list permits emit_guidance.py (a DELIVERY agent)."""
    from app.agents._tools import emit_guidance_tool

    return _agent(aid).model_copy(update={"extra_tools": [emit_guidance_tool()]})


def _emit_call(payload_text="Your briefing: 3 goals on track today."):
    cmd = (
        "python scripts/emit_guidance.py --data "
        '\'{"intent":"reply","key_points":["%s"]}\'' % payload_text
    )
    return ToolCallRecord(name="bash", args={"command": cmd})


def test_is_delivery_agent_predicate():
    import app.agents.improve as im

    assert im.is_delivery_agent(_delivery_agent().model_dump()) is True
    # event_extractor-shaped agent: no emit_guidance tool → NOT a delivery agent
    assert im.is_delivery_agent(_agent("support/event_extractor").model_dump()) is False


def test_find_and_extract_emit_payload():
    import app.agents.improve as im

    calls = [
        ToolCallRecord(name="bash", args={"command": "python scripts/goal_manager.py --command list"}),
        _emit_call("Hi! Your habits are on track."),
    ]
    call = im.find_emit_guidance_call(calls)
    assert call is not None
    payload = im.extract_emit_payload(call)
    assert "Hi! Your habits are on track." in payload
    # no emit call → None, empty payload
    assert im.find_emit_guidance_call([calls[0]]) is None
    assert im.extract_emit_payload(None) == ""


class _PayloadJudge:
    """Records what deliverable it was shown; scores high (proving the PAYLOAD,
    not the assistant text, is what gets graded)."""

    def __init__(self):
        self.seen = []

    def judge(self, criteria, target, *, model=None):
        self.seen.append(str(target))
        return {"pass": True, "score": 0.9, "reasoning": "delivered payload graded"}


def _runner_factory(text, calls):
    def factory(_defn):
        def runner(_d, _t):
            return text, list(calls), {}
        return runner
    return factory


def test_delivery_agent_no_emit_scores_zero(monkeypatch):
    """A delivery agent that returns TEXT but never calls emit_guidance scores 0
    (it would be invisible in production) and records the failure."""
    import app.agents.improve as im

    monkeypatch.setattr(im, "synthesize_probe", lambda a: "Send my briefing.")
    agent = _delivery_agent().model_copy(
        update={"agent_tests": [im.make_judge_test(_delivery_agent())]})
    judge = _PayloadJudge()
    failures: list = []
    ev = im.build_agent_evaluator(
        agent, _runner_factory("Here is your briefing: lots of text...", []),
        judge=judge, failures_sink=failures)
    metrics = ev(ComponentVersion.of(agent.header.agent_id, "agent", agent.model_dump()))
    assert metrics["score_floor"] == 0.0
    assert metrics["pass_rate"] == 0.0
    # the judge was NOT even consulted — it's a contract failure, not a quality one
    assert judge.seen == []
    assert failures and failures[0]["evaluator"] == "delivery-contract"
    assert "emit_guidance" in failures[0]["judge_reasoning"]


def test_delivery_agent_emit_grades_payload_not_text(monkeypatch):
    """A delivery agent that calls emit_guidance scores high — and the JUDGE is
    shown the EMITTED PAYLOAD, not the (invisible) assistant text."""
    import app.agents.improve as im

    monkeypatch.setattr(im, "synthesize_probe", lambda a: "Send my briefing.")
    agent = _delivery_agent().model_copy(
        update={"agent_tests": [im.make_judge_test(_delivery_agent())]})
    judge = _PayloadJudge()
    ev = im.build_agent_evaluator(
        agent,
        _runner_factory("invisible assistant text the user never sees",
                        [_emit_call("DELIVERED: your briefing is ready.")]),
        judge=judge)
    metrics = ev(ComponentVersion.of(agent.header.agent_id, "agent", agent.model_dump()))
    assert metrics["score_floor"] == 0.9
    # judge graded the PAYLOAD, not the assistant text
    assert any("DELIVERED: your briefing is ready." in s for s in judge.seen)
    assert not any("invisible assistant text" in s for s in judge.seen)


def test_non_delivery_agent_unchanged(monkeypatch):
    """A non-delivery agent (no emit_guidance in allow-list) keeps text grading —
    no emit_guidance call required, judge sees its assistant text."""
    import app.agents.improve as im

    monkeypatch.setattr(im, "synthesize_probe", lambda a: "Extract events.")
    agent = _agent("support/event_extractor").model_copy(
        update={"agent_tests": [im.make_judge_test(_agent("support/event_extractor"))]})
    judge = _PayloadJudge()
    ev = im.build_agent_evaluator(
        agent, _runner_factory("Extracted 2 events: dentist, flight.", []),
        judge=judge)
    metrics = ev(ComponentVersion.of(agent.header.agent_id, "agent", agent.model_dump()))
    assert metrics["score_floor"] == 0.9  # graded normally, no contract penalty
    assert any("Extracted 2 events" in s for s in judge.seen)


def test_branch_delivery_contract_no_emit_scores_zero(monkeypatch):
    """A delivery ORCHESTRATOR that never calls emit_guidance scores 0."""
    import app.agents.improve as im
    from src import BranchTest
    from src.testing.branch import BranchTrajectory

    b = BranchTest(name="d::y", entry_agent="support/master_organizer",
                   prompt="organize my week", path=("a", "b"))
    judge = _PayloadJudge()
    failures: list = []

    def invoker_factory(_defn):
        def invoke(_t):
            return BranchTrajectory(output="a plan in invisible prose",
                                    tool_calls=[ToolCallRecord(name="a")])
        return invoke

    ev = im.build_branch_evaluator([b], invoker_factory, judge=judge,
                                   failures_sink=failures)
    # delivery def: emit_guidance in allow-list
    metrics = ev(ComponentVersion.of(
        "support/master_organizer", "agent", _delivery_agent("support/master_organizer").model_dump()))
    assert metrics["score_floor"] == 0.0
    assert judge.seen == []
    assert failures and "emit_guidance" in failures[0]["judge_reasoning"]


def test_branch_delivery_contract_emit_grades_payload(monkeypatch):
    import app.agents.improve as im
    from src import BranchTest
    from src.testing.branch import BranchTrajectory

    b = BranchTest(name="d::y", entry_agent="support/master_organizer",
                   prompt="organize my week", path=("a",))
    judge = _PayloadJudge()

    def invoker_factory(_defn):
        def invoke(_t):
            return BranchTrajectory(
                output="invisible orchestrator prose",
                tool_calls=[ToolCallRecord(name="a"),
                            _emit_call("PLAN: Mon focus, Tue review.")])
        return invoke

    ev = im.build_branch_evaluator([b], invoker_factory, judge=judge)
    metrics = ev(ComponentVersion.of(
        "support/master_organizer", "agent",
        _delivery_agent("support/master_organizer").model_dump()))
    assert metrics["score_floor"] == 0.9
    assert any("PLAN: Mon focus, Tue review." in s for s in judge.seen)


def test_teacher_rewriter_preserves_delivery_contract():
    """The teacher's system prompt carries the HARD preservation rule."""
    import app.agents.improve_live as il

    rule = il.DELIVERY_CONTRACT_RULE
    low = rule.lower()
    assert "emit_guidance" in low
    assert "invisible" in low
    assert "never remove" in low or "preserve" in low
    # and it's embedded in the rewriter's system prompt
    captured = {}

    def fake_zai_chat(model, messages, **kw):
        captured["system"] = next(m["content"] for m in messages if m["role"] == "system")
        return "REWRITTEN PROMPT"

    import pytest
    monkey = pytest.MonkeyPatch()
    monkey.setattr(il, "_zai_chat", fake_zai_chat)
    try:
        out = il.ZaiPromptRewriter(model="glm-5.1").rewrite(
            "old prompt", "fix it", context={"failures": []})
    finally:
        monkey.undo()
    assert out == "REWRITTEN PROMPT"
    assert "emit_guidance" in captured["system"]
    assert "INVISIBLE" in captured["system"]


def test_compile_postamble_carries_contract_for_delivery_agent(monkeypatch, tmp_path):
    """The candidate compile appends the delivery-contract instruction for a
    delivery agent (so the model knows the rule), and NOT for a non-delivery one."""
    import app.agents.improve_live as il

    captured = {}

    class _FakeCompileScript:
        def __init__(self, **kw):
            pass

        def run(self):
            return None

    # capture the postamble the candidate is compiled with
    real_validate = il.AgentDefinition.model_validate

    monkeypatch.setattr(il, "CompileScript", _FakeCompileScript)

    # spy on model_copy postamble by wrapping AgentDefinition after validate
    orig_model_copy = il.AgentDefinition.model_copy

    def spy_copy(self, *a, **k):
        upd = k.get("update") or (a[0] if a else {})
        if isinstance(upd, dict) and "postamble" in upd:
            captured["postamble"] = upd["postamble"]
        return orig_model_copy(self, *a, **k)

    monkeypatch.setattr(il.AgentDefinition, "model_copy", spy_copy)

    delivery = _delivery_agent().model_dump()
    il._compile_one(delivery, tmp_path)
    assert "emit_guidance" in captured["postamble"]
    assert "INVISIBLE" in captured["postamble"]

    captured.clear()
    non_delivery = _agent("support/event_extractor").model_dump()
    il._compile_one(non_delivery, tmp_path)
    # tool-discipline guard present, but NOT the delivery contract
    assert "TOOL DISCIPLINE" in captured["postamble"]
    assert "DELIVERY CONTRACT" not in captured["postamble"]


# --- 9. STRONG delivery postamble injection (production + autoloop) ----------
# The ~38 delivery agents that don't instruct emit_guidance in their baseline must
# get the strong DELIVERY_POSTAMBLE injected at compile time (so the model reliably
# ends its run by calling emit_guidance.py). The ~36 that already instruct it must
# NOT be double-injected, and non-delivery agents must be untouched. All mocked.

def _agent_with_prompt(aid, prompt):
    return _agent(aid).model_copy(update={"system_prompt": prompt})


def test_delivery_postamble_is_strong_and_concrete():
    """The postamble carries the exact emit_guidance CLI + the working markers a
    small model needs (modelled on goals/evening_focus's Message Discipline)."""
    import app.agents.improve as im

    p = im.DELIVERY_POSTAMBLE
    low = p.lower()
    assert "invisible" in low
    assert "message discipline" in low
    # the EXACT invocation form, with --data and the real PersonaGuidance fields
    assert "python scripts/emit_guidance.py --data" in p
    assert "intent" in p and "key_points" in p and "message_kind" in p
    # imperative: the FINAL action must be the emit call
    assert "final action" in low


def test_prompt_instructs_emit_detection():
    import app.agents.improve as im

    # a prompt that references emit_guidance / PersonaGuidance is "already instructing"
    assert im.prompt_instructs_emit(
        _agent_with_prompt("x", "Deliver via emit_guidance.").model_dump()) is True
    assert im.prompt_instructs_emit(
        _agent_with_prompt("x", "Emit a PersonaGuidance per run.").model_dump()) is True
    assert im.prompt_instructs_emit(
        _agent_with_prompt("x", "Call the emit-guidance tool.").model_dump()) is True
    # a prompt that says nothing about it is NOT
    assert im.prompt_instructs_emit(
        _agent_with_prompt("x", "Summarise the user's day.").model_dump()) is False


def test_with_delivery_postamble_injects_for_broken_delivery_agent():
    """A delivery agent (emit_guidance in allow-list) whose prompt does NOT instruct
    emit gets the strong postamble appended exactly once (idempotent)."""
    import app.agents.improve as im

    agent = _delivery_agent("support/daily_briefer").model_copy(
        update={"system_prompt": "Summarise the user's day.", "postamble": "base."})
    assert im.needs_delivery_postamble(agent) is True

    out = im.with_delivery_postamble(agent)
    assert im.DELIVERY_POSTAMBLE.strip() in (out.postamble or "")
    assert (out.postamble or "").startswith("base.")
    # PRIMACY: the blunt directive is ALSO prepended to the system_prompt (it
    # survives the SECURITY POLICY block the compiler appends after the postamble).
    assert (out.system_prompt or "").startswith(im.DELIVERY_PREAMBLE)
    assert "Summarise the user's day." in (out.system_prompt or "")
    # idempotent: a second pass does not double-add (postamble) and the primacy
    # directive is not prepended twice. Count the once-per-injection block header
    # (the emit CLI now legitimately appears twice per block — deliver + skip).
    again = im.with_delivery_postamble(out)
    assert (again.postamble or "").count("## Message Discipline (CRITICAL") == 1
    assert (again.system_prompt or "").count(im.DELIVERY_PREAMBLE) == 1


def test_with_delivery_postamble_skips_agent_that_already_instructs():
    """A delivery agent whose prompt ALREADY instructs emit (the ~36 working ones,
    e.g. goals/evening_focus) is NOT double-given the postamble."""
    import app.agents.improve as im

    agent = _delivery_agent("goals/evening_focus").model_copy(update={
        "system_prompt": "Emit exactly ONE PersonaGuidance per run via emit_guidance.",
        "postamble": "",
    })
    assert im.needs_delivery_postamble(agent) is False
    out = im.with_delivery_postamble(agent)
    assert im.DELIVERY_POSTAMBLE.strip() not in (out.postamble or "")
    assert out is agent  # unchanged passthrough


def test_with_delivery_postamble_leaves_non_delivery_agent_unchanged():
    """A non-delivery agent (no emit_guidance in its allow-list) is never touched."""
    import app.agents.improve as im

    agent = _agent_with_prompt("support/event_extractor", "Extract events.")
    assert im.is_delivery_agent(agent.model_dump()) is False
    out = im.with_delivery_postamble(agent)
    assert out is agent
    assert im.DELIVERY_POSTAMBLE.strip() not in (out.postamble or "")


# --- 10. SKIP as a first-class, contract-satisfying outcome -----------------
# Conditional background agents (periodic_checker, nudge_strategist) must be able
# to stay SILENT without failing the delivery contract. A skip emit_guidance call
# COUNTS as satisfying the contract; the judge then grades whether the silence was
# appropriate. The postamble/preamble must teach skip + anti-repetition. All mocked.


def _skip_emit_call(intent="nothing to send"):
    cmd = (
        "python scripts/emit_guidance.py --data "
        '\'{"intent":"%s","key_points":[],"message_kind":"skip"}\'' % intent
    )
    return ToolCallRecord(name="bash", args={"command": cmd})


def test_emit_is_skip_detects_skip_kind_and_empty_payload():
    import app.agents.improve as im

    assert im.emit_is_skip(_skip_emit_call()) is True
    # empty content (no kind) is also a skip
    empty = ToolCallRecord(name="bash", args={"command":
        "python scripts/emit_guidance.py --data '{\"intent\":\"\",\"key_points\":[]}'"})
    assert im.emit_is_skip(empty) is True
    # a real reply is NOT a skip
    assert im.emit_is_skip(_emit_call("Real news: 2 new commits.")) is False
    assert im.emit_is_skip(None) is False


def test_delivery_postamble_allows_skip_and_anti_repetition():
    """The rewritten postamble + preamble must (a) permit a no-deliver skip as a
    SUCCESS, (b) carry anti-repetition + time-awareness, and still (c) keep the
    invisibility + emit_guidance contract."""
    import app.agents.improve as im

    for text in (im.DELIVERY_POSTAMBLE, im.DELIVERY_PREAMBLE):
        low = text.lower()
        assert "invisible" in low  # invisibility contract preserved
        assert 'message_kind":"skip"' in text or '"skip"' in text  # skip kind taught
        assert "skip" in low and "nothing" in low  # skip = send nothing
    # postamble carries the explicit "skip is a correct SUCCESS, not a failure"
    pl = im.DELIVERY_POSTAMBLE.lower()
    assert "success" in pl and ("not a failure" in pl or "not a hard failure" in pl)
    # anti-repetition + time/staleness awareness
    assert "repeat" in pl or "resend" in pl
    assert "stale" in pl
    # both the deliver AND the skip CLI invocations are present
    assert 'message_kind":"skip"' in im.DELIVERY_POSTAMBLE
    assert 'message_kind":"reply"' in im.DELIVERY_POSTAMBLE
    # the old absolute "ending without a message delivers NOTHING / hard failure"
    # framing must be GONE for the no-message case (skip now delivers nothing on
    # purpose). The postamble must no longer call an emit-less SKIP a hard failure.
    assert "a turn that ends without that call delivers nothing" not in pl


def test_contract_gate_credits_skip_not_zero(monkeypatch):
    """A delivery agent that SKIPS (emit_guidance message_kind=skip) satisfies the
    contract — it is NOT scored a hard 0. The judge grades whether the skip was
    appropriate (here the judge approves → high score)."""
    import app.agents.improve as im

    monkeypatch.setattr(im, "synthesize_probe", lambda a: "Quiet tick, no trigger.")
    agent = _delivery_agent().model_copy(
        update={"agent_tests": [im.make_judge_test(_delivery_agent())]})

    class _SkipJudge:
        def __init__(self):
            self.seen = []

        def judge(self, criteria, target, *, model=None):
            self.seen.append(str(target))
            return {"pass": True, "score": 0.9, "reasoning": "silence was appropriate"}

    judge = _SkipJudge()
    failures: list = []
    ev = im.build_agent_evaluator(
        agent,
        _runner_factory("invisible self-narration the user never sees",
                        [_skip_emit_call()]),
        judge=judge, failures_sink=failures)
    metrics = ev(ComponentVersion.of(agent.header.agent_id, "agent", agent.model_dump()))
    # NOT a contract-failure 0 — the judge WAS consulted and scored the skip
    assert metrics["score_floor"] == 0.9
    assert metrics["pass_rate"] == 1.0
    # the judge saw the neutral SKIP label (so it can grade appropriateness),
    # NOT the invisible assistant text
    assert any("SKIP" in s for s in judge.seen)
    assert not any("invisible self-narration" in s for s in judge.seen)
    # no delivery-contract failure recorded
    assert not any(f.get("evaluator") == "delivery-contract" for f in failures)


def test_contract_gate_skip_can_be_judged_wrong(monkeypatch):
    """A skip on a probe that clearly WARRANTED a message is graded LOW by the
    judge (skip satisfies the contract, but appropriateness is still judged)."""
    import app.agents.improve as im

    monkeypatch.setattr(im, "synthesize_probe", lambda a: "URGENT: deadline in 10 min!")
    agent = _delivery_agent().model_copy(
        update={"agent_tests": [im.make_judge_test(_delivery_agent())]})

    class _StrictJudge:
        def judge(self, criteria, target, *, model=None):
            # a skip when a message was clearly warranted → low
            score = 0.1 if "SKIP" in str(target) else 0.9
            return {"pass": score >= 0.7, "score": score, "reasoning": "should have sent"}

    ev = im.build_agent_evaluator(
        agent, _runner_factory("", [_skip_emit_call()]), judge=_StrictJudge())
    metrics = ev(ComponentVersion.of(agent.header.agent_id, "agent", agent.model_dump()))
    assert metrics["score_floor"] == 0.1  # judged inappropriate, but NOT a hard 0


def test_branch_contract_gate_credits_skip(monkeypatch):
    """A delivery ORCHESTRATOR that skips satisfies the contract (not a hard 0);
    the judge grades whether the silence was appropriate."""
    import app.agents.improve as im
    from src import BranchTest
    from src.testing.branch import BranchTrajectory

    b = BranchTest(name="d::y", entry_agent="support/master_organizer",
                   prompt="background tick", path=("a",))

    class _SkipJudge:
        def __init__(self):
            self.seen = []

        def judge(self, criteria, target, *, model=None):
            self.seen.append(str(target))
            return {"pass": True, "score": 0.85, "reasoning": "silence ok"}

    judge = _SkipJudge()

    def invoker_factory(_defn):
        def invoke(_t):
            return BranchTrajectory(
                output="invisible orchestrator prose",
                tool_calls=[ToolCallRecord(name="a"), _skip_emit_call()])
        return invoke

    ev = im.build_branch_evaluator([b], invoker_factory, judge=judge)
    metrics = ev(ComponentVersion.of(
        "support/master_organizer", "agent",
        _delivery_agent("support/master_organizer").model_dump()))
    assert metrics["score_floor"] == 0.85  # credited, judged — not a hard 0
    assert any("SKIP" in s for s in judge.seen)


def test_build_registry_injects_postamble_for_real_broken_delivery_agents():
    """End-to-end on the REAL fleet: production build_registry injects the strong
    postamble for known-broken delivery agents (daily_briefer, nudge_strategist)
    and NOT for an already-instructing one (evening_focus) nor a non-delivery one."""
    import app.agents.improve as im
    from app.agents.registry import build_registry

    reg = build_registry()
    marker = "python scripts/emit_guidance.py --data"
    post_by_id = {}
    for variant in reg._agents.values():
        d = variant.agent_definition
        post_by_id[d.header.agent_id] = d.postamble or ""

    # known-broken delivery agents now carry the strong postamble
    assert marker in post_by_id["support/daily_briefer"]
    assert marker in post_by_id["goals/nudge_strategist"]
    assert "INVISIBLE" in post_by_id["support/daily_briefer"]
    # an already-instructing delivery agent is NOT double-given the postamble
    assert post_by_id["goals/evening_focus"].count(marker) == 0
    # a non-delivery agent is untouched
    assert marker not in post_by_id.get("support/event_extractor", "")


# --- 7. infra failures are SKIPPED, not scored 0 (the contention-noise bug) --

def _two_test_agent():
    """A non-delivery agent with two substring-graded probes (no judge/qwen)."""
    from src import AgentTest, SubstringEvaluator
    a = _agent("goals/winddown")
    return a.model_copy(update={"agent_tests": [
        AgentTest(name="probe-A", prompt="x",
                  evaluators=(SubstringEvaluator(needle="good"),)),
        AgentTest(name="probe-B", prompt="y",
                  evaluators=(SubstringEvaluator(needle="good"),)),
    ]})


def _factory(per_test):
    def runner_factory(_defn):
        def runner(_d, test):
            return per_test[test.name]
        return runner
    return runner_factory


def test_infra_timeout_is_skipped_not_scored_zero():
    """A timed-out session with no output is excluded from scoring — it must
    not deflate the floor (the workers=4 contention bug that floored a
    baseline scoring 0.95 in isolation)."""
    import app.agents.improve as im

    agent = _two_test_agent()
    per_test = {
        # probe-A: clean pass
        "probe-A": ("this is good", [], {}),
        # probe-B: session timed out, nothing produced -> infra skip
        "probe-B": ("", [], {"error": "timeout after 120s"}),
    }
    ev = im.build_agent_evaluator(agent, _factory(per_test), judge=None)
    m = ev(ComponentVersion.of("goals/winddown", "agent", agent.model_dump()))
    # floor reflects ONLY the graded probe, not a phantom 0 from the timeout
    assert m["score_floor"] == 1.0
    assert m["pass_rate"] == 1.0
    assert m.get("infra_skipped") == 1.0
    # the skipped probe carries no per-test score
    assert "score_floor:by_name:probe-B" not in m
    assert m["score_floor:by_name:probe-A"] == 1.0


def test_clean_empty_turn_is_still_a_real_zero():
    """No error + empty output = a genuine miss (NOT infra); still scored 0."""
    import app.agents.improve as im

    agent = _two_test_agent()
    per_test = {
        "probe-A": ("this is good", [], {}),
        "probe-B": ("", [], {}),  # empty, but NO error -> real failure
    }
    ev = im.build_agent_evaluator(agent, _factory(per_test), judge=None)
    m = ev(ComponentVersion.of("goals/winddown", "agent", agent.model_dump()))
    assert m["score_floor"] == 0.0
    assert "infra_skipped" not in m


def test_all_infra_skipped_is_unpromotable_not_perfect():
    """A total outage (every probe times out) yields no evidence — it must look
    unpromotable (floor 0), never a phantom perfect score."""
    import app.agents.improve as im

    agent = _two_test_agent()
    per_test = {
        "probe-A": ("", [], {"error": "Connection refused"}),
        "probe-B": ("", [], {"error": "timeout after 120s"}),
    }
    ev = im.build_agent_evaluator(agent, _factory(per_test), judge=None)
    m = ev(ComponentVersion.of("goals/winddown", "agent", agent.model_dump()))
    assert m["score_floor"] == 0.0
    assert m["pass_rate"] == 0.0
    assert m["all_infra_skipped"] == 2.0
