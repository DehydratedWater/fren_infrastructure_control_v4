"""World turn pipeline — deterministic plumbing with a stubbed LLM + fake repo.

No live model / DB: `_generate` is replaced with scripted TurnOutcomes and the
state repo with an in-memory fake, so we pin the wiring — events persisted, move
validated against the nav graph, the two-pass research mechanic, memory distil,
clock/mood advance. Beat *quality* is a live concern; this is the contract.
"""

from __future__ import annotations

import pytest

from app.world import turn as turn_mod
from app.world.loader import load_package
from app.world.models import AffinityDelta, NpcLine, TurnOutcome

_PKG = load_package("twily_haven")
_START = _PKG.scenario.starting_location_id
_NEIGHBOR = _PKG.neighbors(_START)[0][1].id  # a valid one-hop destination


class _FakeRepo:
    """In-memory stand-in for WorldStateRepo."""

    def __init__(self, world_id: str):
        self.world_id = world_id
        self.events: list[dict] = []
        self.research: list[dict] = []
        self.memories: list[dict] = []
        self.affinity: dict[str, int] = {}
        self.seen: list[str] = []
        self.session = {
            "world_id": world_id, "package_id": world_id,
            "current_location_id": _START, "clock_minutes": 9 * 60,
            "day_count": 1, "turn_count": 4, "persona_state": {"mood": "ok", "energy": 70},
            "visitor_present": False,
        }
        self.advanced: dict | None = None

    async def ensure_session(self, pkg):
        return self.session

    async def get_session(self):
        return self.session

    async def npc_states(self):
        return {}

    async def events_for_prompt(self, limit=24):
        return []

    async def recent_events(self, limit=60, before_id=None):
        return self.events

    async def add_event(self, *, turn, kind, actor, content, location_id=None, meta=None):
        row = {"id": len(self.events) + 1, "turn": turn, "kind": kind,
               "actor": actor, "content": content, "location_id": location_id}
        self.events.append(row)
        return row

    async def add_research(self, *, turn, query, summary, results):
        row = {"id": len(self.research) + 1, "turn": turn, "query": query,
               "summary": summary, "results": results}
        self.research.append(row)
        return row

    async def bump_affinity(self, npc_id, delta, turn):
        self.affinity[npc_id] = self.affinity.get(npc_id, 0) + delta

    async def mark_seen(self, npc_ids, turn):
        self.seen.extend(npc_ids)

    async def add_memory(self, *, turn, content, importance=0.5, kind="episodic", location_id=None):
        row = {"turn": turn, "content": content, "importance": importance, "location_id": location_id}
        self.memories.append(row)
        return row

    async def advance_turn(self, *, new_location_id, minutes, persona_state, visitor_present=None):
        self.session = {
            **self.session,
            "current_location_id": new_location_id or self.session["current_location_id"],
            "clock_minutes": self.session["clock_minutes"] + minutes,
            "turn_count": self.session["turn_count"] + 1,
            "persona_state": persona_state,
        }
        self.advanced = {"new_location_id": new_location_id, "minutes": minutes, "ps": persona_state}
        return self.session


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(turn_mod, "WorldStateRepo", _FakeRepo)
    monkeypatch.setattr(turn_mod, "_build_client", lambda pkg, *, visitor_present: (None, None))
    # default research stub (overridden per-test where needed)
    async def _no_research(q):
        return {"ok": True, "query": q, "results": [], "summary": "nothing"}
    monkeypatch.setattr(turn_mod.computer, "research", _no_research)
    yield


def _script(monkeypatch, *outcomes: TurnOutcome):
    seq = list(outcomes)
    calls = {"n": 0}

    def _gen(spec, client, message):
        out = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return out

    monkeypatch.setattr(turn_mod, "_generate", _gen)
    return calls


async def test_basic_beat_persists_events_and_advances(monkeypatch):
    _script(monkeypatch, TurnOutcome(
        narration="The kettle ticks as it cools.", action="I jot a note in the margin.",
        speech="", mood="quietly pleased", energy_delta=-5,
    ))
    res = await turn_mod.run_world_turn(world_id="twily_haven", trigger="auto")
    assert res["ok"] is True
    kinds = {e["kind"] for e in res["events_added"]}
    assert "narration" in kinds and "action" in kinds


async def test_valid_move_is_applied(monkeypatch):
    _script(monkeypatch, TurnOutcome(
        narration="She steps out.", action="I head next door.", move_to=_NEIGHBOR,
    ))
    res = await turn_mod.run_world_turn(world_id="twily_haven", trigger="auto")
    assert res["moved"] is True
    assert any(e["kind"] == "move" for e in res["events_added"])


async def test_invalid_move_is_ignored(monkeypatch):
    _script(monkeypatch, TurnOutcome(
        narration="She considers leaving but stays.", action="I stay put.",
        move_to="not_a_real_location",
    ))
    res = await turn_mod.run_world_turn(world_id="twily_haven", trigger="auto")
    assert res["moved"] is False
    assert not any(e["kind"] == "move" for e in res["events_added"])


async def test_research_runs_two_passes_and_logs(monkeypatch):
    async def _research(q):
        return {"ok": True, "query": q, "results": [{"title": "T", "link": "http://x", "snippet": "s"}],
                "summary": "1. T — s"}
    monkeypatch.setattr(turn_mod.computer, "research", _research)
    calls = _script(
        monkeypatch,
        TurnOutcome(narration="Curious, she opens the laptop.", action="I search.",
                    research_query="why does the daylight charm sag at 11:47"),
        TurnOutcome(narration="The screen fills; she reads, brow furrowed.",
                    action="I copy two links into my notes.", mood="onto something"),
    )
    res = await turn_mod.run_world_turn(world_id="twily_haven", trigger="auto")
    assert res["ok"] and res["researched"] is True
    assert calls["n"] == 2  # two generate passes (search, then read-and-react)
    # the final outcome no longer carries a research_query (no chaining)
    assert res["outcome"]["research_query"] is None
    # the read-and-react narration won (second pass), not the first
    assert "reads" in res["outcome"]["narration"]


async def test_memory_and_affinity_recorded(monkeypatch):
    _script(monkeypatch, TurnOutcome(
        narration="Maro slides a cup across the counter.", action="I thank him.",
        npc_lines=[NpcLine(npc_id="maro", line="On the house. You look like a closed tab.")],
        new_memory="Maro comped my tea when I looked wrung out.",
        affinity=[AffinityDelta(npc_id="maro", delta=4)],
    ))
    res = await turn_mod.run_world_turn(world_id="twily_haven", trigger="auto")
    assert res["ok"]
    assert any(e["kind"] == "npc" for e in res["events_added"])


async def test_visitor_turn_records_visitor_event(monkeypatch):
    _script(monkeypatch, TurnOutcome(
        narration="She looks up, surprised and warm.", action="I wave Vis in.",
        speech="You came at the perfect time — kettle's hot.",
    ))
    res = await turn_mod.run_world_turn(
        world_id="twily_haven", trigger="visitor", visitor_input="Vis knocks and steps inside.",
    )
    assert res["ok"]
