"""2026-06 FINAL cron-port batch — the last five v3 cron features as agents +
probes: night_analysis, relationship_initiator, relationship_reflector, the
FULL topic_synthesizer rebuild, and thought_forger.

Per project law: the decision logic lives in fleet agents with LLM-judge
probes (inline realistic context, score-0 gates); the cron entrypoints are
thin spawn wrappers (same script names as v3 so the schedule entries match).
All offline + deterministic: spawn_agent and repos are mocked.

Covers:
  1. the five agents are registered with well-formed judge probes whose
     criteria carry their score-0 gates;
  2. the spawn wrappers are wired like the scheduler's agent jobs (agent id,
     FREN_MODEL_POSTFIX, trigger=cron, timeout under the job budget, exit 1
     on failure);
  3. schedule.yml enables the five jobs with their v3 cron expressions;
  4. the parity pin shrank to exactly {ralf_ping} (permanently excluded —
     superseded by the framework workflow DAG);
  5. the new persona-memory-manager create-thought write path (plumbing for
     persona/thought_forger);
  6. relationship_initiator qualifies as a scheduled delivery agent — the
     registry injects the QUIET-TICK skip clause.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy")

from tests._parity_helpers import REPO_ROOT, schedule_jobs


# ═══════════════════════════════════════════════════════════════════════════
# 1. the five fleet agents exist with their probe suites
# ═══════════════════════════════════════════════════════════════════════════

# agent id -> (domain module name, phrases its judge criteria must carry —
# the score-0 gates that make each probe a real gate, not decoration).
_BATCH2_AGENTS = {
    "support/night_analyst": ("support", ("correlation", "no strong patterns")),
    "persona/relationship_initiator": ("persona", ("pending tasks", "skip")),
    "persona/relationship_reflector": ("persona", ("engagement", "boilerplate")),
    "persona/topic_synthesizer": ("persona", ("unsupported by the input", "one topic")),
    "persona/thought_forger": ("persona", ("not in that list", "already pending")),
}


def _domain_agents(module_name: str) -> dict:
    from app.agents.domains import persona, support

    mod = {"persona": persona, "support": support}[module_name]
    return {a.header.agent_id: a for a in mod.agents()}


def test_batch2_agents_are_defined_with_judge_probes():
    for agent_id, (module_name, needles) in _BATCH2_AGENTS.items():
        agent = _domain_agents(module_name).get(agent_id)
        assert agent is not None, f"{agent_id} missing from the {module_name} domain"
        assert len(agent.agent_tests) >= 1, f"{agent_id} ships no probes"
        # Every probe carries at least one llm_judge with a score-0 gate.
        for t in agent.agent_tests:
            kinds = {e.kind for e in t.evaluators}
            assert "llm_judge" in kinds, f"{agent_id} probe {t.name} has no judge"
            joined_t = " ".join(
                e.criteria for e in t.evaluators if e.kind == "llm_judge"
            ).lower()
            assert "score 0" in joined_t, (
                f"{agent_id} probe {t.name} judge has no explicit score-0 gate"
            )
            # Probes are inline-context replays — they must forbid tool calls.
            assert "do not call any tools" in t.prompt.lower(), (
                f"{agent_id} probe {t.name} is not a self-contained inline probe"
            )
        joined = " ".join(
            e.criteria for t in agent.agent_tests for e in t.evaluators if e.kind == "llm_judge"
        ).lower()
        for needle in needles:
            assert needle in joined, f"{agent_id} judges lost the '{needle}' criterion"


def test_batch2_agents_ship_paired_probes():
    """The spec'd probe pairs exist: grounded+absence (night), grounded+skip
    (initiator), fidelity+dedupe (synthesizer), grounding+freshness (forger)."""
    expectations = {
        "support/night_analyst": 2,
        "persona/relationship_initiator": 2,
        "persona/relationship_reflector": 1,
        "persona/topic_synthesizer": 2,
        "persona/thought_forger": 2,
    }
    for agent_id, n in expectations.items():
        module_name, _ = _BATCH2_AGENTS[agent_id]
        agent = _domain_agents(module_name)[agent_id]
        assert len(agent.agent_tests) >= n, (
            f"{agent_id} must ship at least {n} probes, has {len(agent.agent_tests)}"
        )


def test_batch2_agents_hold_no_file_write_tools():
    """Decision agents persist via their script tools only — never write/edit."""
    for agent_id, (module_name, _) in _BATCH2_AGENTS.items():
        agent = _domain_agents(module_name)[agent_id]
        perms = agent.tool_permissions
        assert not perms.write and not perms.edit, f"{agent_id} holds write/edit"


# ═══════════════════════════════════════════════════════════════════════════
# 2. spawn wrappers — wired like the scheduler's agent jobs
# ═══════════════════════════════════════════════════════════════════════════


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_cron2_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def _spawn_capture(monkeypatch):
    captured: dict = {}

    async def fake_spawn_agent(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True, error=None)

    import app.telegram.spawn as spawn_mod

    monkeypatch.setattr(spawn_mod, "spawn_agent", fake_spawn_agent)
    return captured


async def test_night_analyst_script_spawns_the_agent(_spawn_capture, monkeypatch):
    monkeypatch.setenv("FREN_MODEL_POSTFIX", "-glm51")
    mod = _load_script("night_analyst")
    rc = await mod._run("2026-06-10")
    assert rc == 0
    assert _spawn_capture["agent"] == "support/night_analyst"
    assert _spawn_capture["model_postfix"] == "-glm51"
    assert _spawn_capture["trigger"] == "cron"
    assert "2026-06-10" in _spawn_capture["prompt"]
    assert "night_analysis_report" in _spawn_capture["prompt"]
    # Stays under the schedule job's 3600s budget.
    assert _spawn_capture["timeout_s"] < 3600


async def test_relationship_initiator_script_spawns_the_agent(_spawn_capture, monkeypatch):
    monkeypatch.delenv("FREN_MODEL_POSTFIX", raising=False)
    mod = _load_script("relationship_initiator")
    rc = await mod._run()
    assert rc == 0
    assert _spawn_capture["agent"] == "persona/relationship_initiator"
    assert _spawn_capture["model_postfix"] == ""
    assert _spawn_capture["trigger"] == "cron"
    assert "user_busy" in _spawn_capture["prompt"]
    # Stays under the schedule job's tight 120s budget.
    assert _spawn_capture["timeout_s"] < 120


async def test_relationship_reflector_script_spawns_the_agent(_spawn_capture, monkeypatch):
    monkeypatch.delenv("FREN_MODEL_POSTFIX", raising=False)
    mod = _load_script("relationship_reflector")
    rc = await mod._run()
    assert rc == 0
    assert _spawn_capture["agent"] == "persona/relationship_reflector"
    assert _spawn_capture["trigger"] == "cron"
    assert "relationship_strategy" in _spawn_capture["prompt"]
    assert _spawn_capture["timeout_s"] < 1800


async def test_thought_forger_script_spawns_the_agent(_spawn_capture, monkeypatch):
    monkeypatch.delenv("FREN_MODEL_POSTFIX", raising=False)
    mod = _load_script("thought_forger")
    rc = await mod._run()
    assert rc == 0
    assert _spawn_capture["agent"] == "persona/thought_forger"
    assert _spawn_capture["trigger"] == "cron"
    assert "create-thought" in _spawn_capture["prompt"]
    assert _spawn_capture["timeout_s"] < 300


async def test_topic_synthesizer_full_mode_spawns_the_agent(_spawn_capture, monkeypatch):
    monkeypatch.setenv("FREN_MODEL_POSTFIX", "-glm47")
    mod = _load_script("topic_synthesizer")
    rc = await mod._run_full()
    assert rc == 0
    assert _spawn_capture["agent"] == "persona/topic_synthesizer"
    assert _spawn_capture["model_postfix"] == "-glm47"
    assert _spawn_capture["trigger"] == "cron"
    assert "create-interest" in _spawn_capture["prompt"]
    assert _spawn_capture["timeout_s"] < 900


def test_topic_synthesizer_keeps_the_expire_only_path():
    """EXTEND, don't duplicate: full mode spawns the agent, --expire-only still
    routes to the ported plumbing (app.tools.persona.topic_synthesizer)."""
    src = (REPO_ROOT / "scripts" / "topic_synthesizer.py").read_text()
    assert "--expire-only" in src
    assert "from app.tools.persona.topic_synthesizer import expire_only" in src
    assert "spawn_agent" in src


async def test_batch2_wrappers_exit_nonzero_on_agent_failure(monkeypatch):
    async def failing_spawn_agent(**kwargs):
        return SimpleNamespace(ok=False, error="model not found")

    import app.telegram.spawn as spawn_mod

    monkeypatch.setattr(spawn_mod, "spawn_agent", failing_spawn_agent)

    assert await _load_script("night_analyst")._run("2026-06-10") == 1
    assert await _load_script("relationship_initiator")._run() == 1
    assert await _load_script("relationship_reflector")._run() == 1
    assert await _load_script("thought_forger")._run() == 1
    assert await _load_script("topic_synthesizer")._run_full() == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. schedule entries — enabled with their v3 cron expressions
# ═══════════════════════════════════════════════════════════════════════════


def test_batch2_jobs_are_enabled_with_v3_schedules():
    expected = {
        "night_analysis": ("0 2 * * *", "scripts/night_analyst.py"),
        "relationship_initiator": ("0 9,13,18,22 * * *", "scripts/relationship_initiator.py"),
        "relationship_reflector": ("0 20 * * 0", "scripts/relationship_reflector.py"),
        "topic_synthesizer": ("30 3 * * *", "scripts/topic_synthesizer.py"),
        "thought_forger": ("*/30 7-23 * * *", "scripts/thought_forger.py"),
    }
    jobs = schedule_jobs()
    for name, (cron, script) in expected.items():
        job = jobs.get(name)
        assert job is not None, f"job {name} vanished from schedule.yml"
        assert job.get("enabled") is True, f"job {name} is not enabled"
        assert job.get("cron") == cron, f"job {name} cron drifted: {job.get('cron')!r} != {cron!r}"
        assert job.get("agent") == f"script:{script}", f"job {name} agent drifted"
        assert (REPO_ROOT / script).is_file(), f"{script} missing on disk"


# ═══════════════════════════════════════════════════════════════════════════
# 4. parity pin — only ralf_ping remains disabled (permanently superseded)
# ═══════════════════════════════════════════════════════════════════════════


def test_parity_pin_shrank_to_only_ralf_ping():
    from tests.test_system_parity import REMAINING_DISABLED_SCRIPT_JOBS

    assert REMAINING_DISABLED_SCRIPT_JOBS == {"ralf_ping"}, (
        "the final cron batch is ported — only ralf_ping (superseded by the "
        f"framework workflow DAG) may stay pinned: {REMAINING_DISABLED_SCRIPT_JOBS}"
    )
    job = schedule_jobs().get("ralf_ping")
    assert job is not None and not job.get("enabled"), (
        "ralf_ping must stay disabled — it is superseded, not pending"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. create-thought plumbing — the forger's write path (repos mocked)
# ═══════════════════════════════════════════════════════════════════════════


class _FakeThoughtsRepo:
    created: dict = {}

    async def create(self, **kwargs):
        type(self).created = dict(kwargs)
        return {"id": 41, **kwargs}


@pytest.fixture
def _thoughts_repo(monkeypatch):
    _FakeThoughtsRepo.created = {}
    import app.db.repos.persona_memory as pm

    monkeypatch.setattr(pm, "PendingThoughtsRepo", _FakeThoughtsRepo)
    return _FakeThoughtsRepo


def test_create_thought_persists_via_repo(_thoughts_repo):
    from app.tools.system.persona_memory_manager import Input, PersonaMemoryManagerTool

    out = PersonaMemoryManagerTool().execute(
        Input(
            command="create-thought",
            content="What if your NPCs left pheromone trails?",
            kind="question",
            motivation_score=0.81,
            motivation_breakdown='{"curiosity": 0.9, "persona_fit": 0.8, "silence_fit": 0.7, "drift_need": 0.8}',
            persona_interest_id=12,
        )
    )
    assert out.success and out.thought is not None and out.thought["id"] == 41
    created = _thoughts_repo.created
    assert created["content"] == "What if your NPCs left pheromone trails?"
    assert created["kind"] == "question"
    assert created["motivation_score"] == 0.81
    assert created["motivation_breakdown"] == {
        "curiosity": 0.9, "persona_fit": 0.8, "silence_fit": 0.7, "drift_need": 0.8,
    }
    assert created["persona_interest_id"] == 12
    assert created["topic_node_id"] is None  # 0 → None, no phantom FK


def test_create_thought_requires_content(_thoughts_repo):
    from app.tools.system.persona_memory_manager import Input, PersonaMemoryManagerTool

    out = PersonaMemoryManagerTool().execute(Input(command="create-thought", content="   "))
    assert not out.success and "content required" in out.error
    assert _thoughts_repo.created == {}


def test_create_thought_rejects_bad_breakdown_json(_thoughts_repo):
    from app.tools.system.persona_memory_manager import Input, PersonaMemoryManagerTool

    out = PersonaMemoryManagerTool().execute(
        Input(command="create-thought", content="x", motivation_breakdown="{not json")
    )
    assert not out.success and "not valid JSON" in out.error
    assert _thoughts_repo.created == {}


# ═══════════════════════════════════════════════════════════════════════════
# 6. relationship_initiator is a skip-capable scheduled delivery agent
# ═══════════════════════════════════════════════════════════════════════════


def test_initiator_gets_the_quiet_tick_skip_clause():
    """The initiator carries emit_guidance (delivery agent) and its prompt
    instructs emit — so the registry's with_skip_clause must inject the
    QUIET-TICK allowance, letting a scheduled tick validly send nothing."""
    from app.agents.improve import is_delivery_agent, prompt_instructs_emit, with_skip_clause

    agent = _domain_agents("persona")["persona/relationship_initiator"]
    assert is_delivery_agent(agent), "initiator lost emit_guidance — not a delivery agent"
    assert prompt_instructs_emit(agent), "initiator prompt no longer instructs emit_guidance"
    injected = with_skip_clause(agent)
    assert "QUIET-TICK RULE" in (injected.system_prompt or ""), (
        "with_skip_clause did not inject the skip allowance for the initiator"
    )
