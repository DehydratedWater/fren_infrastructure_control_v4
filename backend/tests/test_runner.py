"""Runtime runner — event-stream parsing + direct backend (mocked provider).

No live opencode / network: parse_opencode_events is pure; the direct backend
is exercised against a compiled .md with httpx monkeypatched.
"""

from __future__ import annotations

import json

import pytest

from app.runtime.runner import (
    AgentRunResult,
    parse_opencode_events,
    parse_opencode_trajectory,
    run_agent_direct,
)


def _events(*objs) -> str:
    return "\n".join(json.dumps(o) for o in objs)


def test_parse_text_parts_joined():
    stdout = _events(
        {"part": {"type": "text", "text": "Hello"}},
        {"noise": 1},
        {"part": {"type": "text", "text": "world"}},
    )
    text, calls = parse_opencode_events(stdout)
    assert text == "Hello\nworld"
    assert calls == []


def test_parse_extracts_tool_chain_in_order():
    stdout = _events(
        {"part": {"type": "tool", "tool": "context_analyzer", "args": {"q": "x"}}},
        {"part": {"type": "text", "text": "thinking"}},
        {"part": {"type": "tool-invocation", "name": "priority_planner"}},
        {"part": {"type": "subagent", "agent": "todo"}},
    )
    text, calls = parse_opencode_events(stdout)
    assert [c.name for c in calls] == ["context_analyzer", "priority_planner", "todo"]
    assert calls[0].args == {"q": "x"}
    assert "thinking" in text


def test_parse_tool_name_nested_dict_shape():
    stdout = _events({"part": {"type": "tool", "tool": {"name": "web_search"}}})
    _text, calls = parse_opencode_events(stdout)
    assert [c.name for c in calls] == ["web_search"]


def test_parse_falls_back_to_raw_when_no_text_parts():
    stdout = "not json at all\nstill not"
    text, calls = parse_opencode_events(stdout)
    assert text == stdout and calls == []


def test_parse_json_stream_with_no_text_returns_empty_not_raw_json():
    # A run that only emitted tool events (no assistant text) must NOT return the
    # raw JSON event stream — that would poison judges/parsers downstream.
    stdout = _events(
        {"part": {"type": "step_start"}},
        {"part": {"type": "tool", "tool": "bash"}},
        {"part": {"type": "step_finish"}},
    )
    text, calls = parse_opencode_events(stdout)
    assert text == ""  # no assistant text → empty, not raw JSON
    assert [c.name for c in calls] == ["bash"]


# ── ordered, interleaved trajectory ──────────────────────────────────────────


def test_trajectory_preserves_stream_order():
    # narration → tool(+result) → narration → tool(+result), as real opencode
    # emits it. The trajectory must be IN ORDER with each kind tagged.
    stdout = _events(
        {"part": {"type": "text", "text": "Let me check the todos."}},
        {"part": {"type": "tool", "tool": "bash", "callID": "c1",
                  "state": {"status": "completed",
                            "input": {"command": "python scripts/todo.py"},
                            "output": "no todos"}}},
        {"part": {"type": "text", "text": "Nothing pending — done."}},
        {"part": {"type": "tool", "tool": "bash", "callID": "c2",
                  "state": {"status": "error",
                            "input": {"command": "ls /forbidden"},
                            "error": "a rule prevents you from using ls"}}},
    )
    text, calls, traj = parse_opencode_trajectory(stdout)
    kinds = [it["kind"] for it in traj]
    # text, tool, result, text, tool, result — strictly ordered
    assert kinds == ["text", "tool", "result", "text", "tool", "result"]
    # the FIRST narration appears before the FIRST tool, which appears before
    # the SECOND narration (chronological).
    assert traj[0]["text"].startswith("Let me check")
    assert traj[1]["command"] == "python scripts/todo.py"
    assert traj[2]["output"] == "no todos"
    assert traj[3]["text"].startswith("Nothing pending")
    assert traj[4]["command"] == "ls /forbidden"
    assert traj[5]["error"] and "prevents you from using ls" in traj[5]["error"]
    # back-compat flat outputs still correct
    assert [c.name for c in calls] == ["bash", "bash"]
    assert "Let me check the todos." in text and "Nothing pending" in text


def test_trajectory_dedupes_streamed_callid_keeping_position():
    # A tool part streams twice under one callID (pending → completed). The tool
    # item must appear ONCE, at its first-seen position, with the final command.
    stdout = _events(
        {"part": {"type": "text", "text": "thinking"}},
        {"part": {"type": "tool", "tool": "bash", "callID": "c1",
                  "state": {"status": "pending", "input": {"command": "echo hi"}}}},
        {"part": {"type": "tool", "tool": "bash", "callID": "c1",
                  "state": {"status": "completed",
                            "input": {"command": "echo hi"}, "output": "hi"}}},
    )
    _text, calls, traj = parse_opencode_trajectory(stdout)
    kinds = [it["kind"] for it in traj]
    # exactly one tool + one result for the single call
    assert kinds.count("tool") == 1
    assert kinds.count("result") == 1
    assert kinds == ["text", "tool", "result"]
    # flat list also de-duped to one call
    assert len(calls) == 1


def test_parse_events_backcompat_unchanged():
    # the old (text, calls) wrapper still returns the same shape
    stdout = _events(
        {"part": {"type": "text", "text": "hi"}},
        {"part": {"type": "tool", "tool": "bash", "callID": "c1",
                  "state": {"input": {"command": "ls"}}}},
    )
    text, calls = parse_opencode_events(stdout)
    assert text == "hi"
    assert [c.name for c in calls] == ["bash"]


async def test_direct_backend_calls_provider(tmp_path, monkeypatch):
    # write a minimal compiled agent .md
    md = tmp_path / ".opencode" / "agents" / "x.md"
    md.parent.mkdir(parents=True)
    md.write_text("---\nmodel: zai-coding-plan/glm-4.7\n---\nYou are helpful.\n")

    class _Resp:
        def raise_for_status(self):  # noqa: D401
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hi there"}}]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    result = await run_agent_direct(
        agent_dir=tmp_path, agent_name="x", prompt="hello", api_key="k",
    )
    assert isinstance(result, AgentRunResult)
    assert result.ok and result.text == "hi there"


async def test_direct_backend_missing_md_errors(tmp_path):
    result = await run_agent_direct(agent_dir=tmp_path, agent_name="nope", prompt="x")
    assert not result.ok and "not found" in result.error
