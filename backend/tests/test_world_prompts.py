"""World prompt construction + lore selection (offline)."""

from __future__ import annotations

from app.world import prompts
from app.world.loader import load_package


def _pkg():
    return load_package("twily_haven")


def test_system_prompt_carries_protagonist_and_rules():
    pkg = _pkg()
    sysp = prompts.build_system_prompt(pkg)
    assert pkg.protagonist.name in sysp
    assert "research_query" in sysp  # the computer mechanic is described
    assert "npc_lines" in sysp


def test_system_prompt_adds_visitor_note_when_present():
    pkg = _pkg()
    base = prompts.build_system_prompt(pkg, visitor_present=False)
    vis = prompts.build_system_prompt(pkg, visitor_present=True)
    assert "VISITOR PRESENT" not in base
    assert "VISITOR PRESENT" in vis
    assert pkg.visitor.appearance[:20] in vis


def test_turn_message_has_situation_sections():
    pkg = _pkg()
    start = pkg.scenario.starting_location_id
    session = {
        "current_location_id": start, "clock_minutes": 9 * 60, "day_count": 1,
        "persona_state": {"mood": "curious", "energy": 80},
    }
    msg = prompts.build_turn_message(
        pkg, session=session, events=[], present_npcs=pkg.npcs_at(start), npc_affinity={},
    )
    assert "## NOW" in msg
    assert "## WHERE SHE IS" in msg
    assert "## WHERE SHE COULD GO" in msg
    assert "## YOUR BEAT" in msg


def test_turn_message_visitor_branch():
    pkg = _pkg()
    start = pkg.scenario.starting_location_id
    session = {"current_location_id": start, "clock_minutes": 9 * 60, "day_count": 1, "persona_state": {}}
    msg = prompts.build_turn_message(
        pkg, session=session, events=[], present_npcs=[], npc_affinity={},
        visitor_input="Vis waves and asks what she's working on.",
    )
    assert "## VIS JUST" in msg
    assert "Vis waves" in msg


def test_lore_selection_matches_keywords():
    pkg = _pkg()
    # at least one lore entry should trigger on its own first keyword
    entry = pkg.lorebook[0]
    kw = entry.keywords[0]
    hits = prompts.select_lore(pkg, f"she thinks about {kw} for a while")
    assert any(entry.content == h for h in hits)


def test_research_feedback_renders_results():
    pkg = _pkg()
    start = pkg.scenario.starting_location_id
    session = {"current_location_id": start, "clock_minutes": 9 * 60, "day_count": 1, "persona_state": {}}
    rf = {"ok": True, "query": "why batteries sag at night", "summary": "1. Voltage droop — ...",
          "results": [{"title": "X", "link": "http://x", "snippet": "y"}]}
    msg = prompts.build_turn_message(
        pkg, session=session, events=[], present_npcs=[], npc_affinity={}, research_feedback=rf,
    )
    assert "THE COMPUTER SHOWS" in msg
    assert "why batteries sag at night" in msg
