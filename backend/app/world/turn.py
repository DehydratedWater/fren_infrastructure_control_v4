"""The turn engine — run one beat of Twily's life.

Pipeline (fren-native, reusing the local qwen via src.interactive exactly as the
heartbeat does):

    assemble situation  →  ONE structured narrator call (Twily + world + NPCs)
                        →  [if she researches] real web search + a second narrate
                        →  persist events + apply move/clock/mood/affinity
                        →  distil a memory  →  return the beat

Autonomous ticks (trigger="auto"/"manual") have no input — she chooses her own
next beat. Visitor ticks (trigger="visitor") react to what Vis said/did.
"""

from __future__ import annotations

import logging
from typing import Any

from app.world import clock as world_clock
from app.world import computer, prompts
from app.world.loader import DEFAULT_PACKAGE, get_package
from app.world.models import TURN_SCHEMA, TurnOutcome, WorldPackage
from app.world.state import WorldStateRepo

logger = logging.getLogger(__name__)

_MAX_TOKENS = 5000
_TIMEOUT_S = 200.0


def _build_client(pkg: WorldPackage, *, visitor_present: bool):
    from src.interactive.runner import OpenAICompatClient
    from src.interactive.spec import InteractiveAgentSpec

    from app.agents.config import QWEN35_27B_LIVE

    spec = InteractiveAgentSpec(
        agent_id="world/twily_haven",
        model=QWEN35_27B_LIVE,
        system_prompt=prompts.build_system_prompt(pkg, visitor_present=visitor_present),
        tools=(),
        output_schema=TURN_SCHEMA,
    )
    client = OpenAICompatClient.from_spec(spec)
    client.default_params["max_tokens"] = _MAX_TOKENS
    client.default_params.setdefault("timeout", _TIMEOUT_S)
    return spec, client


def _generate(spec, client, message: str) -> TurnOutcome:
    """Sync LLM call (run under asyncio.to_thread by the caller)."""
    from src.interactive import run_interactive

    result = run_interactive(spec, message, client=client, history=[], max_tool_rounds=1)
    data = result.structured if isinstance(result.structured, dict) else {}
    if not data:
        # fall back to free text as narration so a beat is never wholly lost
        data = {"narration": (result.output_text or "").strip()[:1200], "action": ""}
    return TurnOutcome.model_validate(data)


def _estimate_minutes(outcome: TurnOutcome, *, moved: bool, researched: bool) -> int:
    base = 20
    if moved:
        base += 15
    if researched:
        base += 20
    # a restful beat (positive energy delta, e.g. sleeping) passes more time
    if outcome.energy_delta >= 15:
        base += 60
    return base


async def run_world_turn(
    *,
    world_id: str = DEFAULT_PACKAGE,
    package_id: str = DEFAULT_PACKAGE,
    trigger: str = "auto",
    visitor_input: str | None = None,
) -> dict[str, Any]:
    """Run one beat. Returns {ok, outcome, state, events_added}."""
    import asyncio

    pkg = get_package(package_id)
    repo = WorldStateRepo(world_id)
    session = await repo.ensure_session(pkg)
    turn_no = int(session["turn_count"]) + 1
    loc_id = session["current_location_id"]

    visitor_present = bool(visitor_input) or bool(session.get("visitor_present"))
    if visitor_input:
        await repo.add_event(
            turn=turn_no, kind="visitor", actor="vis",
            content=visitor_input.strip(), location_id=loc_id,
        )

    present_npcs = pkg.npcs_at(loc_id)
    npc_aff = {nid: int(st.get("affinity", 0)) for nid, st in (await repo.npc_states()).items()}
    events = await repo.events_for_prompt(limit=24)
    beats_here = int((session.get("persona_state") or {}).get("beats_here", 0))

    spec, client = _build_client(pkg, visitor_present=visitor_present)
    message = prompts.build_turn_message(
        pkg, session=session, events=events, present_npcs=present_npcs,
        npc_affinity=npc_aff, visitor_input=visitor_input, beats_here=beats_here,
    )

    try:
        outcome = await asyncio.to_thread(_generate, spec, client, message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("world turn generation failed: %s", exc)
        return {"ok": False, "error": str(exc), "world_id": world_id}

    researched = False
    # ── research mechanic: real search, then a second narrate of her reaction ──
    if outcome.research_query:
        researched = True
        rq = outcome.research_query.strip()
        await repo.add_event(turn=turn_no, kind="research", actor="twily",
                             content=f"sits at the computer to look up: {rq}", location_id=loc_id)
        res = await computer.research(rq)
        await repo.add_research(turn=turn_no, query=rq, summary=res.get("summary", ""),
                                results=res.get("results", []))
        message2 = prompts.build_turn_message(
            pkg, session=session, events=events, present_npcs=present_npcs,
            npc_affinity=npc_aff, visitor_input=visitor_input, research_feedback=res,
            beats_here=beats_here,
        )
        try:
            outcome2 = await asyncio.to_thread(_generate, spec, client, message2)
            # keep her decisions from the first beat (mood/move) but take the
            # narration/action of reading the results; never chain another search
            outcome2.research_query = None
            outcome2.move_to = outcome2.move_to or outcome.move_to
            outcome = outcome2
        except Exception:  # noqa: BLE001
            logger.exception("world turn research-narration failed")

    # ── validate move ──
    moved = False
    new_loc = None
    if outcome.move_to:
        neighbor_ids = {dest.id for _c, dest in pkg.neighbors(loc_id)}
        if outcome.move_to in neighbor_ids:
            new_loc = outcome.move_to
            moved = True

    # ── persist the beat's events ──
    added: list[dict] = []
    if outcome.narration:
        added.append(await repo.add_event(turn=turn_no, kind="narration", actor="narrator",
                                           content=outcome.narration, location_id=loc_id) or {})
    if outcome.action:
        added.append(await repo.add_event(turn=turn_no, kind="action", actor="twily",
                                           content=outcome.action, location_id=loc_id) or {})
    if outcome.speech:
        added.append(await repo.add_event(turn=turn_no, kind="speech", actor="twily",
                                           content=outcome.speech, location_id=loc_id) or {})
    for line in outcome.npc_lines:
        if line.line.strip():
            added.append(await repo.add_event(turn=turn_no, kind="npc", actor=line.npc_id,
                                              content=line.line, location_id=loc_id) or {})
    if moved and new_loc:
        dest = pkg.location(new_loc)
        added.append(await repo.add_event(
            turn=turn_no, kind="move", actor="twily",
            content=f"to {dest.name if dest else new_loc}", location_id=new_loc) or {})

    # ── apply state deltas ──
    ps = dict(session.get("persona_state") or {})
    if outcome.mood:
        ps["mood"] = outcome.mood
    energy = int(ps.get("energy", 80)) + int(outcome.energy_delta or 0)
    ps["energy"] = max(0, min(100, energy))
    # restlessness counter: reset on a move, otherwise climb
    ps["beats_here"] = 0 if moved else beats_here + 1

    for aff in outcome.affinity:
        await repo.bump_affinity(aff.npc_id, aff.delta, turn_no)
    await repo.mark_seen([n.id for n in present_npcs], turn_no)

    if outcome.new_memory:
        importance = 0.75 if (visitor_input or outcome.affinity) else 0.55
        await repo.add_memory(turn=turn_no, content=outcome.new_memory,
                              importance=importance, location_id=new_loc or loc_id)

    minutes = _estimate_minutes(outcome, moved=moved, researched=researched)
    # A visit lasts only as long as Vis is actively engaging: an autonomous
    # background beat ends it, so a chat doesn't pin her to one room all night.
    next_visitor_present = bool(visitor_input)
    new_session = await repo.advance_turn(
        new_location_id=new_loc, minutes=minutes, persona_state=ps,
        visitor_present=next_visitor_present,
    )

    return {
        "ok": True,
        "world_id": world_id,
        "turn": turn_no,
        "outcome": outcome.model_dump(),
        "moved": moved,
        "researched": researched,
        "tells_user": outcome.tells_user,
        "clock_label": world_clock.clock_label(int(new_session["clock_minutes"])),
        "day_phase": world_clock.day_phase(int(new_session["clock_minutes"])),
        "events_added": [e for e in added if e],
    }
