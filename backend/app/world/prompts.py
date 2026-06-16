"""Prompt construction for a world turn.

One LLM call per beat does three jobs at once: it *is* Twily (choosing her next
believable action, first person), it is the narrator (the world responding, third
person), and it voices any NPCs present. The structured TURN_SCHEMA keeps the
output machine-usable. These builders assemble the system prompt (stable world
rules + who Twily is) and the per-turn evidence message (the live situation).
"""

from __future__ import annotations

from typing import Any

from app.world import clock
from app.world.models import Npc, WorldPackage

_SYSTEM_TEMPLATE = """You run a cozy, literate life-simulation for a character named {protag_name}. \
This is her private inner life — a {setting_line}. You play THREE roles at once in every beat:

1. {protag_name} herself — choose her next *believable* action given the time of day, her energy, \
her mood, where she is, who's around, and what she's been chasing. Write it first-person in `action` \
(and `speech` if she says something aloud). One small beat at a time: roughly 10–60 in-world minutes, \
not a whole day.
2. The narrator — the world responding around her, third-person, sensory and warm, in `narration`.
3. Any NPCs present — give them real, distinct voices in `npc_lines` (only people actually in the room).

WHO SHE IS:
{protag_block}

HOW THE WORLD WORKS:
- She can stay and do an activity here, or walk to a neighbouring place by setting `move_to` to its \
location id (only ids listed under "Where she could go"). Moving takes her there for the next beat.
- If she gets genuinely curious about something real and she's somewhere with a computer, she can sit \
down to look it up: set `research_query` to what she searches. The computer returns REAL results that \
you'll narrate next beat — so research only when it fits.
- Track her body: `energy_delta` (-20..20; cooking/walking/socialising cost a little, rest/food/tea \
restore), and set `mood` to a short phrase for how she feels after the beat.
- Relationships shift through `affinity` (npc_id + delta -10..10) when something warms or cools between \
her and someone present.
- When a beat genuinely matters — a small revelation, a connection, a decision, something tender or \
funny — put one line in `new_memory`. Most beats don't need one.
- Very rarely, if she'd want to share a thought with Vis (the visitor / the person she talks to outside \
this world), put it in `tells_user`. Keep this rare and earned.

STYLE: cozy, specific, emotionally honest, a little whimsical. Modern life where everyday magic is \
mundane. Avoid melodrama, avoid repeating recent beats, let small ordinary moments breathe. Never break \
character or mention being an AI/model. Return ONLY the structured object.
"""

_VISITOR_NOTE = """
VISITOR PRESENT: Vis is here with her — the person she knows from outside this world, dropped into it \
as a visitor. Vis has just said or done something (see "Vis just…"). React naturally as {protag_name} \
and any NPCs present; this beat is her response to Vis. Vis is described only by appearance: {vis_appearance}
"""


def build_system_prompt(pkg: WorldPackage, *, visitor_present: bool = False) -> str:
    p = pkg.protagonist
    protag_block = "\n".join(
        x for x in [
            f"- Appearance: {p.appearance}" if p.appearance else "",
            f"- Personality: {p.personality}" if p.personality else "",
            f"- Voice: {p.voice}" if p.voice else "",
            f"- Drives: {'; '.join(p.drives)}" if p.drives else "",
            f"- Current goals/arcs: {'; '.join(p.goals)}" if p.goals else "",
        ] if x
    )
    setting_line = pkg.setting.split(".")[0].strip().lower() if pkg.setting else "warm modern town where everyday magic is ordinary"
    sys = _SYSTEM_TEMPLATE.format(
        protag_name=p.name,
        setting_line=setting_line,
        protag_block=protag_block or f"- {p.name}, a curious soul.",
    )
    if pkg.scenario.setting_notes:
        sys += f"\nTONE NOTES: {pkg.scenario.setting_notes}\n"
    if visitor_present:
        sys += _VISITOR_NOTE.format(protag_name=p.name, vis_appearance=pkg.visitor.appearance)
    return sys


def _npc_card(npc: Npc, affinity: int) -> str:
    bits = [f"{npc.name} (id: {npc.id})"]
    if npc.role:
        bits.append(npc.role)
    line = " — ".join(bits)
    extra = []
    if npc.personality:
        extra.append(npc.personality)
    if npc.voice:
        extra.append(f"voice: {npc.voice}")
    extra.append(f"warmth toward her: {affinity:+d}")
    return f"- {line}. " + " ".join(extra)


def select_lore(pkg: WorldPackage, text: str, limit: int = 4) -> list[str]:
    """Keyword-trigger lorebook entries against recent text."""
    low = (text or "").lower()
    hits = [e for e in pkg.lorebook if any(k.lower() in low for k in e.keywords)]
    hits.sort(key=lambda e: e.priority, reverse=True)
    return [e.content for e in hits[:limit]]


def _event_line(ev: dict[str, Any]) -> str:
    kind = ev.get("kind")
    actor = ev.get("actor", "")
    content = str(ev.get("content", "")).strip()
    if not content:
        return ""
    if kind == "npc":
        return f"{actor}: “{content}”"
    if kind == "speech":
        return f"Twily (aloud): “{content}”"
    if kind == "action":
        return f"Twily: {content}"
    if kind == "research":
        return f"[computer] {content}"
    if kind == "move":
        return f"[moves] {content}"
    if kind == "visitor":
        return f"Vis: {content}"
    return content  # narration / system / mood


def build_turn_message(
    pkg: WorldPackage,
    *,
    session: dict[str, Any],
    events: list[dict[str, Any]],
    present_npcs: list[Npc],
    npc_affinity: dict[str, int],
    visitor_input: str | None = None,
    research_feedback: dict | None = None,
) -> str:
    loc = pkg.location(session["current_location_id"])
    clk = int(session["clock_minutes"])
    ps = session.get("persona_state") or {}

    parts: list[str] = []
    parts.append(
        f"## NOW\nDay {session['day_count']} · {clock.clock_label(clk)} "
        f"({clock.day_phase(clk)}) · mood: {ps.get('mood', '—')} · "
        f"energy: {ps.get('energy', '—')}/100"
    )

    where = [f"## WHERE SHE IS\n{loc.name}" + (f" ({loc.kind})" if loc else "")]
    if loc:
        where.append(loc.description)
        if loc.activities:
            acts = ", ".join(f"{a.label} [{a.tag}]" for a in loc.activities)
            where.append(f"She can: {acts}")
    parts.append("\n".join(where))

    if present_npcs:
        cards = "\n".join(_npc_card(n, npc_affinity.get(n.id, 0)) for n in present_npcs)
        parts.append(f"## WHO'S HERE\n{cards}")
    else:
        parts.append("## WHO'S HERE\nNobody — she's on her own right now.")

    if loc:
        nbrs = pkg.neighbors(loc.id)
        if nbrs:
            lines = []
            for conn, dest in nbrs:
                label = conn.label or dest.name
                note = f" ({conn.travel_note})" if conn.travel_note else ""
                lines.append(f"- {dest.id}: {dest.name} — via {label}{note}")
            parts.append("## WHERE SHE COULD GO\n" + "\n".join(lines))

    if events:
        transcript = "\n".join(filter(None, (_event_line(e) for e in events)))
        if transcript:
            parts.append("## RECENTLY (most recent last)\n" + transcript)

    # lore matched against the recent transcript + any visitor input
    lore_text = " ".join(str(e.get("content", "")) for e in events[-6:]) + " " + (visitor_input or "")
    lore = select_lore(pkg, lore_text)
    if lore:
        parts.append("## WORLD NOTES (lore that's relevant now)\n" + "\n".join(f"- {x}" for x in lore))

    if research_feedback and research_feedback.get("results") is not None:
        rf = research_feedback
        if rf.get("ok") and rf.get("summary"):
            parts.append(
                f"## THE COMPUTER SHOWS (real results for “{rf['query']}”)\n{rf['summary']}\n"
                "Narrate her reading and reacting to this — what catches her, what she makes of it."
            )
        else:
            parts.append(
                f"## THE COMPUTER\nHer search for “{rf.get('query','')}” turned up nothing useful "
                "(flaky connection / dead ends). Play it lightly."
            )

    if visitor_input:
        parts.append(f"## VIS JUST…\n{visitor_input}")
        parts.append(
            "## YOUR BEAT\nRespond to Vis as Twily (and any NPCs present). Fill the structured object."
        )
    else:
        parts.append(
            "## YOUR BEAT\nWhat does she do next? Choose one believable beat and fill the structured "
            "object. If she stays put, leave move_to null."
        )
    return "\n\n".join(parts)
