"""Proactive autonomy heartbeat — decision → routing, with a stub qwen client.

No live model / DB: a StubClient returns a canned decision, the evidence-assembly
and ledger calls are monkeypatched, and we assert the deterministic routing
(skip delivers nothing; message → send; act → safelisted spawn only; pending
thoughts consumed on share). The decision QUALITY is tuned live/in the autoloop;
these pin the wiring + guardrails.
"""

from __future__ import annotations

import json

import pytest

from app.agents import heartbeat
from src.interactive.runner import ChatResponse


class _StubClient:
    def __init__(self, decision: dict):
        self.default_params: dict = {}
        self._decision = decision
        self.calls = 0

    def complete(self, *, messages, tools, model, **params):
        self.calls += 1
        return ChatResponse(content=json.dumps(self._decision), tool_calls=[])


class _StubLedger:
    async def ensure_run(self, *a, **k):
        return None

    async def write_artifact(self, *a, **k):
        return {}

    async def complete_run(self, *a, **k):
        return {}


@pytest.fixture
def wired(monkeypatch):
    """Patch evidence assembly + ledger; spy on the routing side-effects."""
    spy: dict = {"delivered": None, "spawned": None, "consumed": None}

    async def _t():
        return {"trigger": False, "reason": "no_triggers", "triggers": []}

    async def _th():
        return [{"id": 42, "content": "a forged insight", "motivation_score": 0.9, "kind": "idea"}]

    async def _empty():
        return []

    async def _strat():
        return {}

    async def _chat():
        return ([], {"last_user_age_min": 240, "last_bot_age_min": 600})

    monkeypatch.setattr(heartbeat, "_deterministic_triggers", _t)
    monkeypatch.setattr(heartbeat, "_pending_thoughts", _th)
    monkeypatch.setattr(heartbeat, "_open_commitments", _empty)
    monkeypatch.setattr(heartbeat, "_strategies_and_monologue", _strat)
    monkeypatch.setattr(heartbeat, "_recent_chat", _chat)
    monkeypatch.setattr(
        "app.db.repos.execution_ledger.ExecutionLedgerRepo", lambda: _StubLedger()
    )

    def _deliver(draft, run_id):
        spy["delivered"] = draft
        return True

    def _spawn(agent, instruction):
        spy["spawned"] = agent
        return True

    async def _consume(ids):
        spy["consumed"] = list(ids)

    monkeypatch.setattr(heartbeat, "_deliver_message", _deliver)
    monkeypatch.setattr(heartbeat, "_spawn_detached", _spawn)
    monkeypatch.setattr(heartbeat, "_mark_thoughts_consumed", _consume)
    return spy


def _with_decision(monkeypatch, decision: dict):
    spec, _ = heartbeat._build_spec("day")
    stub = _StubClient(decision)
    monkeypatch.setattr(heartbeat, "_build_spec", lambda mode: (spec, stub))
    return stub


async def test_skip_delivers_nothing(wired, monkeypatch):
    _with_decision(monkeypatch, {"decision": "skip", "category": "none", "reasoning": "quiet tick"})
    res = await heartbeat.run_heartbeat("day")
    assert res["ok"] and res["decision"] == "skip" and res["acted"] is False
    assert wired["delivered"] is None and wired["spawned"] is None


async def test_message_delivers_and_consumes_thought(wired, monkeypatch):
    _with_decision(monkeypatch, {
        "decision": "message", "category": "share_insight",
        "reasoning": "worth sharing", "draft": "hey, I noticed something cool!",
        "uses_thought_ids": [42],
    })
    res = await heartbeat.run_heartbeat("day")
    assert res["acted"] is True and res["category"] == "share_insight"
    assert wired["delivered"] == "hey, I noticed something cool!"
    assert wired["consumed"] == [42]  # the surfaced thought is marked consumed


async def test_escalate_spawns_mode_specialist(wired, monkeypatch):
    _with_decision(monkeypatch, {"decision": "escalate", "category": "agreement_followup",
                                 "reasoning": "nuanced — hand to specialist"})
    res = await heartbeat.run_heartbeat("day")
    assert res["acted"] is True
    assert wired["spawned"] == heartbeat.MODES["day"]["escalate_agent"]


async def test_winddown_escalates_to_winddown_specialist(wired, monkeypatch):
    _with_decision(monkeypatch, {"decision": "escalate", "category": "winddown",
                                 "urgency": 4, "reasoning": "past cutoff, act"})
    res = await heartbeat.run_heartbeat("winddown")
    assert wired["spawned"] == "goals/winddown"


async def test_act_only_spawns_safelisted_agent(wired, monkeypatch):
    _with_decision(monkeypatch, {"decision": "act", "category": "self_research",
                                 "route_agent": "research/techtree_orchestrator",
                                 "reasoning": "go research X"})
    res = await heartbeat.run_heartbeat("day")
    assert res["acted"] is True
    assert wired["spawned"] == "research/techtree_orchestrator"


async def test_act_blocks_non_safelisted_agent(wired, monkeypatch):
    _with_decision(monkeypatch, {"decision": "act", "category": "other",
                                 "route_agent": "goals/todo_manager",
                                 "reasoning": "create a bunch of todos"})
    res = await heartbeat.run_heartbeat("day")
    assert res["acted"] is False           # guardrail: not on the safelist
    assert wired["spawned"] is None


def test_evidence_block_surfaces_pending_thoughts():
    block = heartbeat._evidence_block(
        "day",
        triggers={"trigger": False, "reason": "no_triggers"},
        thoughts=[{"id": 7, "content": "the agent's own idea", "motivation_score": 0.8}],
        commitments=[{"text": "start running this week"}],
        strat={},
        ages={"last_user_age_min": 120, "last_bot_age_min": 300},
    )
    assert "PENDING THOUGHTS" in block and "id 7" in block
    assert "start running this week" in block
    assert "mode=day" in block
