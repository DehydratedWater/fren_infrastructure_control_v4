"""Pydantic models for Twily's world — the authored package + runtime turn IO.

Two layers:

* **Authored content** (`WorldPackage` and friends): static, version-controlled,
  *modifiable* world data loaded from YAML under `packages/<id>/`. This is the
  world designer's surface — locations, the people in them, lore, and the two
  characters (Twily, who lives here; Vis, who can visit).
* **Runtime IO** (`TurnContext`, `TurnOutcome`): the per-turn structures the
  engine assembles and the LLM fills. Persistent runtime *state* lives in
  Postgres (see `state.py`), not here.

Everything is plain Pydantic so a package round-trips through YAML cleanly and
the turn LLM call can be schema-validated.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── authored world content ──────────────────────────────────────────────────

LocationKind = Literal["room", "outdoor", "venue", "transit", "abstract"]


class Activity(BaseModel):
    """Something Twily can *do* at a location — the affordances that make a room
    more than a backdrop (cook, sleep, use the computer, tend plants…). `tag`
    is a stable verb the engine/UI can key on; `computer`/`rest`/`cook` carry
    special handling in the turn engine."""

    model_config = ConfigDict(extra="forbid")

    tag: str
    label: str
    description: str = ""


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    kind: LocationKind = "room"
    parent_id: str | None = None  # a room inside a building, a building in a district
    description: str
    image_tags: str = ""  # visual-only prompt for image gen (no lore)
    default_npcs: list[str] = Field(default_factory=list)  # NPC ids usually here
    activities: list[Activity] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    position: tuple[int, int] | None = None  # optional map grid hint


class Connection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_id: str
    to_id: str
    label: str = ""  # "out the front door", "the stairs up", …
    bidirectional: bool = True
    travel_note: str = ""  # flavour: "a ten-minute walk along the canal"


class Npc(BaseModel):
    """A person in Twily's world. Cards are deliberately light — these are
    supporting characters the narrator voices, not full agents."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    role: str = ""  # "barista", "old mentor", "rival tinkerer"
    home_location_id: str | None = None
    appearance: str = ""
    personality: str = ""
    voice: str = ""  # how they talk; speech tics
    description: str = ""  # who they are / their deal
    inspires: str = ""  # what about them tends to spark Twily (drives her growth)
    tags: list[str] = Field(default_factory=list)
    default_affinity: int = 0  # -100..100 starting relationship warmth


class LoreEntry(BaseModel):
    """Keyword-triggered world knowledge injected into the narrator's context."""

    model_config = ConfigDict(extra="forbid")

    id: str
    keywords: list[str]
    content: str
    priority: int = 100  # higher wins when several match and budget is tight


class Faction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str = ""


class Character(BaseModel):
    """A first-class character. Twily is the protagonist who lives the sim; Vis
    is the user's drop-in visitor (visual description only — no backstory)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    appearance: str
    personality: str = ""
    voice: str = ""
    drives: list[str] = Field(default_factory=list)  # what she wants / chases
    goals: list[str] = Field(default_factory=list)  # current arcs to make progress on
    image_tags: str = ""


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    starting_location_id: str
    opening_narration: str
    setting_notes: str = ""  # tone/era guidance for the narrator
    start_hour: int = 8  # in-world clock the sim begins at (0..23)


class WorldPackage(BaseModel):
    """The whole authored world. Loaded from YAML; modifiable by hand. One
    package == one playable world."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    authors: list[str] = Field(default_factory=list)
    setting: str = ""  # one-paragraph elevator pitch of the world

    protagonist: Character  # Twily
    visitor: Character  # Vis (the user's avatar)

    locations: list[Location]
    connections: list[Connection] = Field(default_factory=list)
    npcs: list[Npc] = Field(default_factory=list)
    lorebook: list[LoreEntry] = Field(default_factory=list)
    factions: list[Faction] = Field(default_factory=list)
    scenario: Scenario

    # ── convenience indexes ──
    def location(self, loc_id: str) -> Location | None:
        return next((loc for loc in self.locations if loc.id == loc_id), None)

    def npc(self, npc_id: str) -> Npc | None:
        return next((n for n in self.npcs if n.id == npc_id), None)

    def neighbors(self, loc_id: str) -> list[tuple[Connection, Location]]:
        """Locations reachable from `loc_id` in one hop (honours direction)."""
        out: list[tuple[Connection, Location]] = []
        for c in self.connections:
            if c.from_id == loc_id:
                dest = self.location(c.to_id)
                if dest:
                    out.append((c, dest))
            elif c.bidirectional and c.to_id == loc_id:
                dest = self.location(c.from_id)
                if dest:
                    out.append((c, dest))
        return out

    def npcs_at(self, loc_id: str) -> list[Npc]:
        loc = self.location(loc_id)
        ids = list(loc.default_npcs) if loc else []
        # plus anyone whose home is here
        ids += [n.id for n in self.npcs if n.home_location_id == loc_id and n.id not in ids]
        return [n for n in (self.npc(i) for i in ids) if n is not None]


# ── runtime turn IO ─────────────────────────────────────────────────────────

TurnTrigger = Literal["auto", "manual", "visitor"]


class TurnContext(BaseModel):
    """Everything assembled for one turn before the LLM call (for logging/debug)."""

    model_config = ConfigDict(extra="allow")

    world_id: str
    turn_number: int
    trigger: TurnTrigger
    location_id: str
    clock_minutes: int
    day_count: int
    persona_state: dict[str, Any] = Field(default_factory=dict)
    present_npc_ids: list[str] = Field(default_factory=list)
    visitor_present: bool = False
    visitor_input: str | None = None


class NpcLine(BaseModel):
    npc_id: str
    line: str


class AffinityDelta(BaseModel):
    npc_id: str
    delta: int = 0  # -10..10


class TurnOutcome(BaseModel):
    """The structured result the narrator LLM returns for one turn. The engine
    turns this into persisted events + state deltas."""

    model_config = ConfigDict(extra="ignore")

    narration: str = ""  # 3rd-person prose: what the world does / what happens
    action: str = ""  # first-person: what Twily does this beat
    speech: str = ""  # optional: what Twily says aloud
    feeling: str = ""  # brief internal weather
    npc_lines: list[NpcLine] = Field(default_factory=list)
    move_to: str | None = None  # location id she walks to, if any
    research_query: str | None = None  # if she goes to the computer to look something up
    new_memory: str | None = None  # something from this beat worth remembering
    mood: str | None = None  # her mood after this beat
    energy_delta: int = 0  # -20..20 (tiredness/refreshed)
    affinity: list[AffinityDelta] = Field(default_factory=list)
    tells_user: str | None = None  # an aside she'd want to share with Vis/the user (rare)


# JSON schema the interactive runner enforces on the turn call. Kept in lockstep
# with TurnOutcome but hand-written so we control required/enum exactly.
TURN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "narration": {"type": "string", "description": "3rd-person prose of what happens this beat"},
        "action": {"type": "string", "description": "first-person: what Twily does"},
        "speech": {"type": "string", "description": "optional words Twily says aloud"},
        "feeling": {"type": "string", "description": "brief internal weather (a phrase)"},
        "npc_lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"npc_id": {"type": "string"}, "line": {"type": "string"}},
                "required": ["npc_id", "line"],
            },
        },
        "move_to": {"type": ["string", "null"], "description": "location id she walks to, or null"},
        "research_query": {
            "type": ["string", "null"],
            "description": "if she sits at the computer to look something up, the query; else null",
        },
        "new_memory": {"type": ["string", "null"]},
        "mood": {"type": ["string", "null"]},
        "energy_delta": {"type": "integer"},
        "affinity": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"npc_id": {"type": "string"}, "delta": {"type": "integer"}},
                "required": ["npc_id", "delta"],
            },
        },
        "tells_user": {"type": ["string", "null"]},
    },
    "required": ["narration", "action"],
}
