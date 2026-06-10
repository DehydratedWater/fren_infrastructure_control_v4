"""2026-06 cron-port batch — ralf_cleanup, pending-thoughts expiry, the
goal-progress cron wrapper, and the two agent-spawning cron entrypoints.

All offline + deterministic: repos and the agent spawn are mocked; file
deletion runs against tmp dirs and is containment-checked (NEVER outside the
configured media roots — the safe_media_path discipline).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sqlalchemy")

from tests._parity_helpers import REPO_ROOT

from app.tools.system import ralf_cleanup as rc


# ═══════════════════════════════════════════════════════════════════════════
# ralf_cleanup — retention + winner-keep + path containment
# ═══════════════════════════════════════════════════════════════════════════


class _FakeKV:
    winners: set = set()

    async def winner_media_ids(self):
        return set(self.winners)


class _FakeMedia:
    rows: list = []
    deleted: list = []

    async def list_older_than(self, cutoff):
        return [dict(r) for r in self.rows]

    async def delete(self, media_id):
        type(self).deleted.append(media_id)
        return True


class _FakeLocks:
    expired: list = []
    released: int = 0

    async def list_expired(self):
        return list(self.expired)

    async def release_expired(self):
        type(self).released += len(self.expired)
        n = len(self.expired)
        type(self).expired = []
        return n


@pytest.fixture
def _cleanup_fakes(monkeypatch):
    _FakeKV.winners = set()
    _FakeMedia.rows = []
    _FakeMedia.deleted = []
    _FakeLocks.expired = []
    _FakeLocks.released = 0
    import app.db.repos.ralf as ralf_mod
    import app.db.repos.rendered_media as media_mod

    monkeypatch.setattr(ralf_mod, "RalfKVRepo", _FakeKV)
    monkeypatch.setattr(ralf_mod, "RalfLocksRepo", _FakeLocks)
    monkeypatch.setattr(media_mod, "RenderedMediaRepo", _FakeMedia)


def test_is_safely_contained_blocks_escapes(tmp_path):
    root = tmp_path / "rendered"
    root.mkdir()
    inside = root / "img_x.png"
    nested = root / "sub" / "img_y.png"
    outside = tmp_path / "secret.png"
    traversal = root / ".." / "secret.png"
    assert rc.is_safely_contained(inside, [root])
    assert rc.is_safely_contained(nested, [root])
    assert not rc.is_safely_contained(outside, [root])
    assert not rc.is_safely_contained(traversal, [root])  # resolves outside
    assert not rc.is_safely_contained(root, [root])  # the root itself is not a file target


async def test_cleanup_deletes_old_file_inside_root(_cleanup_fakes, tmp_path):
    root = tmp_path / "rendered"
    root.mkdir()
    f = root / "img_old.png"
    f.write_bytes(b"x")
    _FakeMedia.rows = [{"media_id": "img_old", "file_path": str(f)}]

    out = await rc.cleanup_rendered_media(7, roots=[root])
    assert not f.exists()
    assert out["files_removed"] == 1 and out["rows_removed"] == 1
    assert _FakeMedia.deleted == ["img_old"]


async def test_cleanup_keeps_winner_media(_cleanup_fakes, tmp_path):
    root = tmp_path / "rendered"
    root.mkdir()
    f = root / "img_winner.png"
    f.write_bytes(b"x")
    _FakeKV.winners = {"img_winner"}
    _FakeMedia.rows = [{"media_id": "img_winner", "file_path": str(f)}]

    out = await rc.cleanup_rendered_media(7, roots=[root])
    assert f.exists()
    assert out["files_removed"] == 0 and out["rows_removed"] == 0
    assert out["skipped_winners"] == 1
    assert _FakeMedia.deleted == []


async def test_cleanup_never_deletes_outside_the_media_roots(_cleanup_fakes, tmp_path):
    root = tmp_path / "rendered"
    root.mkdir()
    victim = tmp_path / "precious.png"  # OUTSIDE the configured root
    victim.write_bytes(b"x")
    _FakeMedia.rows = [
        {"media_id": "img_evil_abs", "file_path": str(victim)},
        {"media_id": "img_evil_rel", "file_path": str(root / ".." / "precious.png")},
    ]

    out = await rc.cleanup_rendered_media(7, roots=[root])
    assert victim.exists(), "cleanup escaped the configured media roots!"
    assert out["files_removed"] == 0
    assert out["containment_violations"] == 2
    # The anomalous rows are kept visible, not silently dropped.
    assert _FakeMedia.deleted == []


async def test_cleanup_dry_run_touches_nothing(_cleanup_fakes, tmp_path, capsys):
    root = tmp_path / "rendered"
    root.mkdir()
    f = root / "img_old.png"
    f.write_bytes(b"x")
    _FakeMedia.rows = [{"media_id": "img_old", "file_path": str(f)}]

    out = await rc.cleanup_rendered_media(7, dry_run=True, roots=[root])
    assert f.exists() and _FakeMedia.deleted == []
    assert out["files_removed"] == 0 and out["rows_removed"] == 0
    assert "would remove" in capsys.readouterr().out


async def test_cleanup_removes_row_for_already_missing_file(_cleanup_fakes, tmp_path):
    root = tmp_path / "rendered"
    root.mkdir()
    _FakeMedia.rows = [{"media_id": "img_gone", "file_path": str(root / "img_gone.png")}]

    out = await rc.cleanup_rendered_media(7, roots=[root])
    assert out["files_removed"] == 0 and out["rows_removed"] == 1


async def test_cleanup_expired_locks_releases(_cleanup_fakes):
    _FakeLocks.expired = [
        {"resource_key": "comfyui", "holder_ralf_id": "ralf_a"},
        {"resource_key": "tts", "holder_ralf_id": "ralf_b"},
    ]
    n = await rc.cleanup_expired_locks()
    assert n == 2 and _FakeLocks.released == 2


async def test_cleanup_expired_locks_dry_run_only_reports(_cleanup_fakes, capsys):
    _FakeLocks.expired = [{"resource_key": "comfyui", "holder_ralf_id": "ralf_a"}]
    n = await rc.cleanup_expired_locks(dry_run=True)
    assert n == 1 and _FakeLocks.released == 0
    assert "would release" in capsys.readouterr().out


# ═══════════════════════════════════════════════════════════════════════════
# pending_thoughts_expire — topic_synthesizer --expire-only (repos mocked)
# ═══════════════════════════════════════════════════════════════════════════


async def test_expire_only_prunes_via_repos(monkeypatch):
    calls: dict = {}

    class _FakeInterests:
        async def prune_expired(self):
            calls["prune"] = True
            return 4

    class _FakeThoughts:
        async def expire_old(self, *, hours):
            calls["expire_hours"] = hours
            return 7

        async def trim_queue(self, *, max_size):
            calls["trim_max"] = max_size
            return 2

    import app.db.repos.persona_memory as pm

    monkeypatch.setattr(pm, "PersonaInterestsRepo", _FakeInterests)
    monkeypatch.setattr(pm, "PendingThoughtsRepo", _FakeThoughts)

    from app.tools.persona.topic_synthesizer import expire_only

    out = await expire_only()
    assert out == {"pruned_interests": 4, "expired_thoughts": 7, "trimmed": 2}
    # v3 parity knobs: 48h thought expiry, 30-item queue cap.
    assert calls == {"prune": True, "expire_hours": 48, "trim_max": 30}


# ═══════════════════════════════════════════════════════════════════════════
# goal_progress cron — wires the existing tool correctly (tools mocked)
# ═══════════════════════════════════════════════════════════════════════════


def test_goal_progress_cron_wires_the_tool(monkeypatch):
    seen: dict = {}

    class _FakeGoalTool:
        def execute(self, inp):
            seen["goal_cmd"] = inp.command
            return SimpleNamespace(count=3)

    class _FakeUpdaterTool:
        def execute(self, inp):
            seen["updater_cmd"] = inp.command
            seen["lookback"] = inp.lookback_hours
            return SimpleNamespace(
                success=True, updates_made=2, updates_skipped=1, message="ok", error=""
            )

    import app.tools.goals.goal_manager as gm
    import app.tools.goals.goal_progress_auto_updater as gpau

    monkeypatch.setattr(gm, "GoalManagerTool", _FakeGoalTool)
    monkeypatch.setattr(gpau, "GoalProgressAutoUpdaterTool", _FakeUpdaterTool)

    from app.tools.goals import goal_progress_cron

    result = goal_progress_cron.run()
    assert result.success
    # backfill-keywords first (v3 parity), then run with the 2h lookback.
    assert seen == {"goal_cmd": "backfill-keywords", "updater_cmd": "run", "lookback": 2}


def test_goal_progress_cron_main_exits_nonzero_on_failure(monkeypatch):
    class _FakeGoalTool:
        def execute(self, inp):
            return SimpleNamespace(count=0)

    class _FakeUpdaterTool:
        def execute(self, inp):
            return SimpleNamespace(success=False, error="boom")

    import app.tools.goals.goal_manager as gm
    import app.tools.goals.goal_progress_auto_updater as gpau

    monkeypatch.setattr(gm, "GoalManagerTool", _FakeGoalTool)
    monkeypatch.setattr(gpau, "GoalProgressAutoUpdaterTool", _FakeUpdaterTool)

    from app.tools.goals import goal_progress_cron

    with pytest.raises(SystemExit) as exc:
        goal_progress_cron.main()
    assert exc.value.code == 1


def test_goal_progress_cron_backfill_failure_is_non_fatal(monkeypatch):
    class _BrokenGoalTool:
        def execute(self, inp):
            raise RuntimeError("backfill exploded")

    class _FakeUpdaterTool:
        def execute(self, inp):
            return SimpleNamespace(
                success=True, updates_made=0, updates_skipped=0, message="ok", error=""
            )

    import app.tools.goals.goal_manager as gm
    import app.tools.goals.goal_progress_auto_updater as gpau

    monkeypatch.setattr(gm, "GoalManagerTool", _BrokenGoalTool)
    monkeypatch.setattr(gpau, "GoalProgressAutoUpdaterTool", _FakeUpdaterTool)

    from app.tools.goals import goal_progress_cron

    assert goal_progress_cron.run().success  # the cycle still ran


# ═══════════════════════════════════════════════════════════════════════════
# agent-spawning cron entrypoints — spawn wired like the scheduler's agent jobs
# ═══════════════════════════════════════════════════════════════════════════


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_cron_{name}", path)
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


async def test_activity_summarizer_script_spawns_the_agent(_spawn_capture, monkeypatch):
    from datetime import date

    monkeypatch.setenv("FREN_MODEL_POSTFIX", "-glm51")
    mod = _load_script("activity_summarizer")
    rc_ = await mod._run(date(2026, 6, 10))
    assert rc_ == 0
    assert _spawn_capture["agent"] == "support/activity_summarizer"
    assert _spawn_capture["model_postfix"] == "-glm51"
    assert _spawn_capture["trigger"] == "cron"
    assert "ctx_daily_2026-06-10" in _spawn_capture["prompt"]
    # Stays under the schedule job's 300s budget.
    assert _spawn_capture["timeout_s"] < 300


async def test_lesson_extractor_script_spawns_the_agent(_spawn_capture, monkeypatch):
    monkeypatch.delenv("FREN_MODEL_POSTFIX", raising=False)
    mod = _load_script("lesson_extractor")
    rc_ = await mod._run(6)
    assert rc_ == 0
    assert _spawn_capture["agent"] == "support/lesson_extractor"
    assert _spawn_capture["model_postfix"] == ""
    assert "lesson_extractor_cursor" in _spawn_capture["prompt"]
    assert _spawn_capture["timeout_s"] < 600


async def test_spawn_wrappers_exit_nonzero_on_agent_failure(monkeypatch):
    async def failing_spawn_agent(**kwargs):
        return SimpleNamespace(ok=False, error="model not found")

    import app.telegram.spawn as spawn_mod

    monkeypatch.setattr(spawn_mod, "spawn_agent", failing_spawn_agent)

    from datetime import date

    assert await _load_script("activity_summarizer")._run(date(2026, 6, 10)) == 1
    assert await _load_script("lesson_extractor")._run(3) == 1


# ═══════════════════════════════════════════════════════════════════════════
# the new fleet agents exist with their probe suites
# ═══════════════════════════════════════════════════════════════════════════


def test_cron_agents_are_defined_with_probes():
    from app.agents.domains import support

    by_id = {a.header.agent_id: a for a in support.agents()}
    for agent_id, needles in {
        "support/activity_summarizer": ("invent", "health"),
        "support/lesson_extractor": ("re-remind", "transcript"),
    }.items():
        agent = by_id.get(agent_id)
        assert agent is not None, f"{agent_id} missing from the support domain"
        assert len(agent.agent_tests) >= 3, f"{agent_id} must ship at least 3 probes"
        # Every probe carries at least one llm_judge evaluator.
        for t in agent.agent_tests:
            kinds = {e.kind for e in t.evaluators}
            assert "llm_judge" in kinds, f"{agent_id} probe {t.name} has no judge"
        joined = " ".join(
            e.criteria for t in agent.agent_tests for e in t.evaluators if e.kind == "llm_judge"
        ).lower()
        for needle in needles:
            assert needle in joined, f"{agent_id} judges lost the '{needle}' criterion"


# NOTE: the former test_topic_synthesizer_script_rejects_full_mode is gone on
# purpose — the full nightly rebuild IS now ported (agent persona/topic_synthesizer
# + spawn wrapper). Full-mode spawn wiring is covered in test_cron_port_batch2.py;
# --expire-only keeps its coverage above (test_expire_only_prunes_via_repos).
