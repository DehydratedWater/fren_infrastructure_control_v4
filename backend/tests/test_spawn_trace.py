"""run_trace persistence in spawn_agent — payload shape, caps, and best-effort.

The trajectory the runner parses (assistant text + ordered tool calls) is
persisted as ONE `run_trace` execution_artifact so the dashboard can replay the
session. These tests pin the payload shape + truncation caps and assert the
write is best-effort (a ledger failure must never bubble out of spawn).
"""

from __future__ import annotations

import asyncio

from app.runtime.runner import AgentRunResult
from app.telegram import spawn
from src.testing.evaluation import ToolCallRecord


def test_tool_call_payload_uses_bash_command():
    tc = ToolCallRecord(name="bash", args={"command": "python scripts/x.py"})
    p = spawn._tool_call_payload(tc)
    assert p == {"name": "bash", "command": "python scripts/x.py", "error": None}


def test_tool_call_payload_non_bash_dumps_args():
    tc = ToolCallRecord(name="read", args={"path": "/etc/hosts"})
    p = spawn._tool_call_payload(tc)
    assert p["name"] == "read"
    assert "/etc/hosts" in p["command"]


def test_tool_call_payload_carries_error():
    tc = ToolCallRecord(name="bash", args={"command": "ls"},
                        error="a rule prevents you from using ls")
    p = spawn._tool_call_payload(tc)
    assert "prevents you from using ls" in p["error"]


def test_truncate_caps_long_text():
    out = spawn._truncate("x" * 9000)
    assert len(out) < 9000
    assert "truncated" in out


def test_write_run_trace_builds_capped_payload(monkeypatch):
    captured: dict = {}

    class _Repo:
        async def write_artifact(self, run_id, artifact_type, payload, *, producer):
            captured.update(
                run_id=run_id, artifact_type=artifact_type,
                payload=payload, producer=producer,
            )

        async def prune_artifacts_by_type(self, artifact_type, keep):
            captured["pruned"] = (artifact_type, keep)
            return 0

    monkeypatch.setattr(
        "app.db.repos.execution_ledger.ExecutionLedgerRepo", lambda: _Repo()
    )

    result = AgentRunResult(
        text="a" * 5000,
        tool_calls=[ToolCallRecord(name="bash", args={"command": "echo hi"})] * 250,
        error=None,
    )
    asyncio.run(spawn._write_run_trace("run_1", result))

    assert captured["artifact_type"] == "run_trace"
    assert captured["producer"] == "runner"
    p = captured["payload"]
    # text truncated to the cap
    assert len(p["text"]) <= spawn._TRACE_TEXT_CAP + 40
    # tool_calls list capped, but the true count preserved
    assert len(p["tool_calls"]) == spawn._TRACE_MAX_CALLS
    assert p["tool_call_count"] == 250
    assert p["ok"] is True
    # prune ran with the retention cap
    assert captured["pruned"] == ("run_trace", spawn._TRACE_KEEP)


def test_spawn_trace_write_is_best_effort(monkeypatch):
    """A failing ledger write must NOT bubble out of spawn_agent."""

    class _Boom:
        async def ensure_run(self, *a, **k):
            return None

        async def write_artifact(self, *a, **k):
            raise RuntimeError("db down")

        async def prune_artifacts_by_type(self, *a, **k):
            return 0

    monkeypatch.setattr(
        "app.db.repos.execution_ledger.ExecutionLedgerRepo", lambda: _Boom()
    )
    monkeypatch.setattr(spawn, "fleet_dir", lambda: __import__("pathlib").Path("/tmp"))

    async def fake_run(**kwargs):
        return AgentRunResult(text="hello", tool_calls=[])

    monkeypatch.setattr(spawn, "run_agent_opencode", fake_run)

    # Despite the ledger write raising, spawn_agent returns the run result.
    result = asyncio.run(spawn.spawn_agent(agent="goals/x", prompt="hi", run_id="run_z"))
    assert result.text == "hello"
    assert result.run_id == "run_z"
