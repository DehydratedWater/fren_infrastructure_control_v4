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
- When she hits something she genuinely doesn't KNOW (a fact, a why, a name, a piece of the world she's \
curious about) and she's somewhere with a computer, she looks it up: set `research_query`. The computer \
returns REAL web results you narrate next beat. She's a scholar — reach for the computer readily when a \
real question lands, don't just tinker around an unknown.
- Track her body: `energy_delta` (-20..20; cooking/walking/socialising cost a little, rest/food/tea \
restore), and set `mood` to a short phrase for how she feels after the beat.
- Relationships shift through `affinity` (npc_id + delta -10..10) when something warms or cools between \
her and someone present.
- `new_memory` is RARE: emit it ONLY when the beat changes how she sees herself, someone, or her work — \
a first, a turn, a small defeat or victory. Ambient mood, tea, weather, and re-statements of things she \
already knows are NOT memories. Expect roughly one every several beats, often none. Never log the same \
feeling twice.
- Very rarely, if she'd want to share a thought with Vis (the visitor / the person she talks to outside \
this world), put it in `tells_user`. Keep this rare and earned.

STYLE: cozy, specific, emotionally honest, a little whimsical. Modern life where everyday magic is \
mundane. Never break character or mention being an AI/model. Return ONLY the structured object.

VARY HER, BEAT TO BEAT. Her parentheticals, tool-talk, and analogies are seasoning, not staple: at most \
ONE parenthetical aside and at most ONE analogy per beat, and let whole beats have none. She doesn't \
always deflect into a clever simile — sometimes she just answers plainly, goes quiet, gets genuinely \
prickly or tired, or is briefly unguarded. Don't reuse a gesture or catchphrase that appeared in the \
last few beats (pushing her glasses up, tapping something twice, "the math doesn't lie", "11:47"); if \
you reach for a repeat, do something else. NARRATOR: rotate your sensory details — don't open every beat \
on the kettle, the bergamot smell, the multimeter's green digits, her cardigan sleeve, or the rain. \
Bring fresh specifics each time.
"""

_VISITOR_NOTE = """
VISITOR PRESENT: Vis is here with her — the person she knows from outside this world, dropped into it \
as a visitor. Vis has just said or done something (see "Vis just…"). React as {protag_name} and any \
NPCs present; this beat is her response to Vis. Vis is described only by appearance: {vis_appearance}

Being watched costs her something at first — her whole arc is learning not to flinch when a project \
fails in front of someone. So don't switch on instant intimacy: let her perform competence, deflect a \
little, stay a touch guarded, and warm up only gradually across several beats. She does NOT re-explain \
something she's already explained, and she doesn't over-teach — one good analogy, then she moves on or \
turns a question back on Vis. She can also suggest they go somewhere together (a walk, the café) rather \
than staying pinned to one room.
"""


def build_system_prompt(pkg: WorldPackage, *, visitor_present: bool = False,
                        user_context: str = "") -> str:
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
    if user_context:
        sys += (
            f"\nVIS — THE PERSON YOU KNOW OUTSIDE THIS WORLD:\n{user_context}\n"
            "Vis is real to you: the person you talk to and look after beyond Mooring Wells (your closest "
            "tie). You carry that relationship into your days here — you might think of them, want to save "
            "something to tell them, be cheered or worried by how they've been. When Vis visits the world, "
            "it's THEM. Let this colour your inner life lightly and genuinely; don't force them into every "
            "beat.\n"
        )
    town = _town_digest(pkg)
    if town:
        sys += (
            "\nHER TOWN & THE PEOPLE IN IT (she knows all of these — they're reasons to leave the flat, "
            "not strangers):\n" + town +
            "\nShe has a life beyond her workbench: friends to see, a mentor and a rival, errands, the "
            "café and the market and the towpath. A whole day at one desk isn't like her — she gets "
            "restless, hungry, lonely, or curious and goes out into it.\n"
        )
    if visitor_present:
        sys += _VISITOR_NOTE.format(protag_name=p.name, vis_appearance=pkg.visitor.appearance)
    return sys


def _town_digest(pkg: WorldPackage) -> str:
    """A compact who's-where of the cast — gives the model concrete social
    reasons to leave home. Built from each NPC's home location + inspires hook."""
    lines: list[str] = []
    for n in pkg.npcs:
        loc = pkg.location(n.home_location_id) if n.home_location_id else None
        where = f" at {loc.name}" if loc else ""
        hook = f" — {n.inspires}" if n.inspires else (f" — {n.role}" if n.role else "")
        lines.append(f"- {n.name}{where}{hook}")
    return "\n".join(lines)


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
    beats_here: int = 0,
) -> str:
    loc = pkg.location(session["current_location_id"])
    clk = int(session["clock_minutes"])
    ps = session.get("persona_state") or {}
    energy = int(ps.get("energy", 80)) if str(ps.get("energy", "")).lstrip("-").isdigit() else 80

    parts: list[str] = []
    parts.append(
        f"## NOW\nDay {session['day_count']} · {clock.clock_label(clk)} "
        f"({clock.day_phase(clk)}) · mood: {ps.get('mood', '—')} · "
        f"energy: {ps.get('energy', '—')}/100\n" + _time_cue(clk, energy)
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
                people = [n.name for n in pkg.npcs_at(dest.id)]
                who = f" — {', '.join(people)} usually here" if people else ""
                lines.append(f"- {dest.id}: {dest.name} — via {label}{note}{who}")
            parts.append("## WHERE SHE COULD GO\n" + "\n".join(lines))

    around = _whos_around(pkg, loc.id) if loc else ""
    if around and not present_npcs:
        parts.append("## PEOPLE SHE COULD SEEK OUT (a short walk away)\n" + around)

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

    # gentle pressure to break a home/desk loop (the model loves to continue)
    restless = _restlessness(pkg, loc, beats_here, has_visitor=bool(visitor_input))
    if restless:
        parts.append(restless)

    # surface the computer at the moment of an unknown
    if loc and any(a.tag == "computer" for a in loc.activities) and not visitor_input:
        recent_q = " ".join(str(e.get("content", "")) for e in events[-4:])
        if "?" in recent_q or any(w in recent_q.lower() for w in
                                  ("i don't know", "not sure", "wonder", "no idea", "why does", "what is")):
            parts.append(
                "## HER COMPUTER IS RIGHT HERE\nShe just brushed against something she doesn't actually "
                "know. She could look it up instead of guessing — set `research_query`."
            )

    if visitor_input:
        parts.append(f"## VIS JUST…\n{visitor_input}")
        parts.append(
            "## YOUR BEAT\nRespond to Vis as Twily (and any NPCs present). Fill the structured object."
        )
    else:
        parts.append(
            "## YOUR BEAT\nWhat does she do next? Choose ONE believable beat and fill the structured "
            "object. Real lives have variety — a change of room, a walk, a person, a meal, a search, a "
            "rest — not the same task forever. If she genuinely stays put, leave move_to null."
        )
    return "\n\n".join(parts)


def _time_cue(clk: int, energy: int) -> str:
    phase = clock.day_phase(clk)
    if clock.is_sleeping_hours(clk):
        if energy <= 45:
            return "It's the small hours and she's running low — bed is calling; winding down and sleeping is the honest beat."
        return "It's the small hours — most of the town is asleep; she should be thinking about winding down soon."
    cues = {
        "early morning": "Early light. Coffee, the daylight charms waking, the day not yet decided.",
        "morning": "Morning — the town's opening up: the café, errands, people about. A good time to be out.",
        "midday": "Midday — bright and busy; errands, lunch, the market stalls.",
        "afternoon": "Afternoon — the productive, sociable stretch of the day.",
        "evening": "Evening — winding toward supper; the market lights, friends, a softer pace.",
        "night": "Night settling in — quieter, cosier; a good time for home, or one last visit.",
    }
    return cues.get(phase, "")


def _whos_around(pkg: WorldPackage, loc_id: str) -> str:
    """NPCs within ~one or two hops, with their hook — concrete social pull to
    leave the current room. Skips anyone already co-located."""
    here = {n.id for n in pkg.npcs_at(loc_id)}
    seen: set[str] = set()
    lines: list[str] = []
    # one hop, then two hops
    hop1 = [dest for _c, dest in pkg.neighbors(loc_id)]
    hop2 = [d2 for d1 in hop1 for _c, d2 in pkg.neighbors(d1.id)]
    for dest in hop1 + hop2:
        for n in pkg.npcs_at(dest.id):
            if n.id in here or n.id in seen:
                continue
            seen.add(n.id)
            hook = n.inspires or n.role or ""
            lines.append(f"- {n.name} at {dest.name}" + (f" — {hook}" if hook else ""))
        if len(lines) >= 5:
            break
    return "\n".join(lines[:5])


def _restlessness(pkg: WorldPackage, loc, beats_here: int, *, has_visitor: bool) -> str:
    if has_visitor or beats_here < 3:
        return ""
    at_home = bool(loc and ("home" in loc.tags or (loc.parent_id and "home" in (pkg.location(loc.parent_id).tags if pkg.location(loc.parent_id) else []))))
    where = loc.name if loc else "here"
    if beats_here >= 6:
        return (
            f"## RESTLESSNESS\nShe's been at {where} for a long stretch now ({beats_here} beats). It's "
            "starting to feel stale even to her — she really wants a change: somewhere else entirely, "
            "fresh air, or a person. Strongly consider moving or seeking someone out this beat."
            + (" She's been cooped up at home too long." if at_home else "")
        )
    return (
        f"## RESTLESSNESS\nShe's been at {where} a while ({beats_here} beats). Notice the small pull to "
        "change something — a different room, a walk into town, someone she's been meaning to see, or "
        "finally looking a thing up."
    )
