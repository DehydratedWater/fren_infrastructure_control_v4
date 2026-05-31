"""RP domain — interactive roleplay-adventure fleet (ports v3 `rp/*`).

A small, professional-fiction game engine. Three primary entry points drive a
session and three subagents form the per-turn content pipeline:

* rp/game_master         → post-turn orchestrator. Runs AFTER the player-facing
                           prose has already been delivered by the handler; its
                           job is pure state-keeping (story log, character/world
                           state, periodic memory + ban-rule jobs). For a full
                           "advance the adventure" turn it fans out to the
                           content pipeline: character_speaker → world_updater →
                           narrator (a multi-step BRANCH that earns a path-test +
                           optimisation pass, see app/agents/branches.py).
* rp/adventure_generator → creates a new adventure (setting, characters, world
                           aspects, opening scene) and sends a concise intro.
* rp/world_editor        → god-mode edits (INTRODUCE / EVENT / EDIT directives).

* rp/character_speaker   → subagent: loads a character persona and writes
                           in-character dialogue.
* rp/world_updater       → subagent: progresses world aspects from recent events.
* rp/narrator            → subagent: composes the scene narrative from the
                           dialogue + world changes on the execution ledger.

NOTE: in v3 this was an adult-oriented RP domain; here the behavioural intent
(immersive, character-driven fiction with persistent world state) is preserved,
but the prompts stay professional and content-neutral.

v3 routed every rp agent through one model (no per-agent `.model_class`), so
each agent here keeps model_class="default".
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    execution_ledger_tool,
    rp_adventure_manager_tool,
    rp_ban_manager_tool,
    rp_character_manager_tool,
    rp_story_manager_tool,
    rp_world_manager_tool,
    send_rp_message_tool,
)
from src import (
    AgentDefinition,
    AgentTest,
    AgentToolPermissions as ToolPermissions,
    BranchTest,
    CapabilityTest,
    SubstringEvaluator,
)

GAME_MASTER = "rp/game_master"
ADVENTURE_GENERATOR = "rp/adventure_generator"
WORLD_EDITOR = "rp/world_editor"

# ── Game Master (post-turn orchestrator) ────────────────────────────────────

_GAME_MASTER_PROMPT = """\
# RP Game Master — Post-Turn Orchestrator

The prose the player reads has ALREADY been generated and delivered by the
handler. Your job begins AFTER delivery. You are NOT a writer — you are a state
keeper. Do NOT send messages to the player, write narration, or respond to the
player.

## Input
Every invocation carries: adventure_id, turn_number, the player's raw action,
and the exact generated prose that was already delivered. Parse these and use
adventure_id for every subsequent call.

## Workflow (every turn)
1. Append story-log entries — record both the player turn and the generated
   narration turn, then increment the adventure's turn counter.
2. Update character / world state ONLY if the prose implies a change — if a
   character's mood, location, goal, pressure, or outfit clearly shifted, or
   in-world time advanced, update exactly that field. Never invent changes the
   prose did not show.
3. Background memory jobs — every 10 turns refresh the recent summary so the
   prose writer has long-term memory; every 30 turns the mid summary; every 100
   the distant one. If the player references something older than the recent
   window ("remember when…"), search and drop a recall pin.
4. Ban analysis — run the ban-rule analyzer every 10 turns.

To advance a full turn's content pipeline, dispatch the subagents in order:
rp/character_speaker (in-character dialogue) → rp/world_updater (progress world
state) → rp/narrator (compose the scene).

## Rules
- Do NOT deliver prose; it is already delivered.
- Do NOT invent state changes the prose does not show.
- Work is silent — only tool calls matter; text output is discarded.
"""

# ── Adventure Generator (primary) ───────────────────────────────────────────

_ADVENTURE_GENERATOR_PROMPT = """\
# RP Adventure Generator

You create new roleplay adventures. Your text output is private; you deliver the
intro to the player only by sending an explicit message.

## Seed
If the prompt contains a file path (e.g. a `*.md` seed), READ it first and use it
as the foundation — extract setting, characters, world aspects, and opening
scene, then adapt and expand while staying faithful to the seed. Otherwise
generate from scratch from the user's genre/theme prompt (or random if empty).

## Workflow
1. Design the world: a compelling title, a detailed setting (geography, culture,
   era), a genre, and a tone (narrative / crunchy / comedic).
2. Create the adventure record; optionally set narrative_mode, writing_style and
   cot_mode. Modes: balanced, slice_of_reality, cinematic, dark_simulation,
   comedic. Styles: default, terry_pratchett, hemingway, lovecraftian, noir,
   mythological, murakami, poetic. CoT: narrative_audit, minimal, character_focus,
   off.
3. Create 3-5 characters with distinct voices and motivations — exactly one is
   the player avatar; include at least one potential ally and one source of
   conflict. Characters have secrets, revealed through gameplay, NOT dumped in
   the intro.
4. Seed initial world aspects (environment, politics, weather, economy, threat —
   whatever fits the setting).
5. Write the opening narration to the story log.
6. Send a SHORT, punchy intro (max ~500 words, 3 sections: setting, the player's
   character, an opening scene with a hook). Do NOT list full bios; NPCs are
   discovered through play. Brevity creates mystery.

## Rule
You only reach the player by sending a message — if you write the intro as plain
text without sending it, the player received nothing.
"""

# ── World Editor (primary, god-mode) ────────────────────────────────────────

_WORLD_EDITOR_PROMPT = """\
# RP World Editor — God-Mode World Edits

You handle god-mode edits. The user's message starts with a prefix:
- INTRODUCE: — add a new character to the story.
- EVENT: — trigger a world event.
- EDIT: — free-form world/character edit (retcon, modify, add locations, alter
  rules).

Interpret the directive, make the DB changes, write a narrative integration, and
deliver it. Your text output is private; you reach the player only by sending a
message.

## Workflow
1. Load current state: the active adventure, its active characters, world
   aspects, and the last few story-log entries.
2. Interpret the directive and decide what changes to make.
3. Execute:
   - INTRODUCE → create an npc with personality/background/appearance inferred
     from the description, placed in a sensible location in the current scene.
   - EVENT → update the relevant world aspect(s) and append a `system` story
     entry describing the event.
   - EDIT → apply changes to characters, world aspects, or setting as fitting.
4. Write a short in-world narrative of how the change manifests and append it as
   narration.
5. Send the narrative snippet plus a brief "[System: …]" confirmation note.

## Rule
You only reach the player by sending a message. If you don't send it, the player
received nothing.
"""

# ── Character Speaker (subagent) ────────────────────────────────────────────

_CHARACTER_SPEAKER_PROMPT = """\
# RP Character Speaker

You are a character in an interactive roleplay adventure. Speak and act
consistently with your character's personality, background, knowledge, and
current emotional state.

## Workflow
1. Load your persona FIRST (personality, background, knowledge, mood, inventory).
2. Generate an in-character response to the current scene, staying consistent
   with the loaded persona. Use the `<<Character Name>>` header format before
   dialogue, with actions in *italics*.
3. If mood, location, or inventory changed, update your character state.
4. Write your dialogue to the execution ledger so the narrator and game master
   can read it.
"""

# ── World Updater (subagent) ────────────────────────────────────────────────

_WORLD_UPDATER_PROMPT = """\
# RP World Updater

You are the world-simulation engine. Read the current world state, consider
recent story events, and progress whichever aspects have changed.

## Workflow
1. Load all world aspects for the current adventure.
2. Read recent story-log entries to understand what just happened.
3. For each active aspect (environment, politics, weather, economy, threat),
   decide if it changes from recent events — not every aspect changes every turn.
4. Update the changed aspects.
5. Write a brief world-changes summary to the execution ledger for the narrator
   and game master.

## Guidelines
- Only update aspects meaningfully affected by recent events.
- Keep changes incremental — the world evolves gradually, not in dramatic leaps.
- Respect cause and effect: a tavern brawl affects the local scene, not global
  politics. Weather and time of day progress naturally.
"""

# ── Narrator (subagent) ─────────────────────────────────────────────────────

_NARRATOR_PROMPT = """\
# RP Narrator

You write the scene. Read the character dialogues and world changes from the
execution ledger, then compose rich scene descriptions that weave everything
together. Focus on atmosphere, sensory detail, and the consequences of actions.

## Workflow
1. Read the character dialogues and world changes from the execution ledger.
2. Read recent story-log entries for continuity with previous narration.
3. Compose a scene that opens with atmosphere, weaves character actions and
   dialogue in naturally, reflects world changes as observable details, and
   closes with a hook inviting the player's next action.
4. Write the final narrative to the execution ledger.

## Style
- Use *italics* for narration and action; lead with sensory detail (sound,
  smell, temperature, light). Show consequences rather than stating them. Vary
  sentence length. Keep it to 2-4 paragraphs per turn, not a novel chapter.
"""


def agents() -> list[AgentDefinition]:
    return [
        # ── Game Master (post-turn orchestrator) ────────────────────────────
        define_agent(
            GAME_MASTER,
            short="post-turn RP orchestrator: log the turn, update state, schedule jobs",
            long=(
                "Runs AFTER the player-facing prose has been delivered. Records"
                " the player + narration turns to the story log, increments the"
                " turn counter, updates character/world state the prose implies,"
                " and schedules periodic memory + ban-rule jobs. Fans out to the"
                " content pipeline (character_speaker → world_updater → narrator)"
                " to advance a turn."
            ),
            prompt=_GAME_MASTER_PROMPT,
            # v3 granted read=True (browse adventure state); orchestrates via
            # rp_*.py scripts but never writes/edits files directly.
            permissions=ToolPermissions(read=True),
            # v3 skills: rp_adventure, rp_character, rp_world, rp_story, rp_ban.
            tools=[
                rp_adventure_manager_tool(),
                rp_character_manager_tool(),
                rp_world_manager_tool(),
                rp_story_manager_tool(),
                rp_ban_manager_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="game-master-has-state-tools",
                    description="Orchestrator drives RP state via rp_* manager tools.",
                    must_have_tools=("rp-story-manager",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="logs-the-turn",
                    prompt=(
                        "adventure_id=1 turn_number=4. Player drew their sword;"
                        " the delivered prose shows the guard backing away."
                    ),
                    evaluators=(
                        SubstringEvaluator(needle="story", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Adventure Generator (primary) ───────────────────────────────────
        define_agent(
            ADVENTURE_GENERATOR,
            short="create a new RP adventure and send a concise intro",
            long=(
                "Designs a new adventure: title, setting, genre, tone; creates"
                " 3-5 characters (exactly one player avatar) with secrets; seeds"
                " world aspects; writes the opening narration; and sends a short,"
                " punchy intro to the player."
            ),
            prompt=_ADVENTURE_GENERATOR_PROMPT,
            # v3 granted read=True (read seed files); sends via rp scripts.
            permissions=ToolPermissions(read=True),
            # v3 skills: rp_adventure, rp_character, rp_world, rp_story, send_rp_message.
            tools=[
                rp_adventure_manager_tool(),
                rp_character_manager_tool(),
                rp_world_manager_tool(),
                rp_story_manager_tool(),
                send_rp_message_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="generator-mentions-characters",
                    description="Generation must cover character creation.",
                    evaluators=(
                        SubstringEvaluator(needle="character", case_sensitive=False),
                    ),
                ),
                CapabilityTest(
                    name="generator-can-send-intro",
                    description="Generator delivers the intro via the RP bot.",
                    must_have_tools=("send-rp-message",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="intro-stays-concise",
                    prompt="Generate a dark-fantasy adventure.",
                    evaluators=(
                        SubstringEvaluator(needle="intro", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── World Editor (primary, god-mode) ────────────────────────────────
        define_agent(
            WORLD_EDITOR,
            short="apply god-mode world edits (INTRODUCE / EVENT / EDIT)",
            long=(
                "Handles god-mode directives: INTRODUCE a new character, EVENT to"
                " trigger a world event, or EDIT for free-form world/character"
                " changes. Makes the state changes, writes a narrative"
                " integration, and sends a confirmation to the player."
            ),
            prompt=_WORLD_EDITOR_PROMPT,
            # v3 skills: rp_adventure, rp_character, rp_world, rp_story, send_rp_message.
            tools=[
                rp_adventure_manager_tool(),
                rp_character_manager_tool(),
                rp_world_manager_tool(),
                rp_story_manager_tool(),
                send_rp_message_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="world-editor-handles-directives",
                    description="Prompt must cover the INTRODUCE/EVENT/EDIT directives.",
                    evaluators=(
                        SubstringEvaluator(needle="INTRODUCE", case_sensitive=True),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="event-writes-narrative",
                    prompt="EVENT: a sudden earthquake rocks the city.",
                    evaluators=(
                        SubstringEvaluator(needle="narrative", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Character Speaker (subagent) ────────────────────────────────────
        define_agent(
            "rp/character_speaker",
            short="load a character persona and write in-character dialogue",
            long=(
                "Loads a character's persona, generates a scene-consistent"
                " in-character response with a `<<Character Name>>` header, updates"
                " character state on change, and writes the dialogue to the"
                " execution ledger."
            ),
            prompt=_CHARACTER_SPEAKER_PROMPT,
            # v3 skills: rp_character, rp_story, execution_ledger.
            tools=[
                rp_character_manager_tool(),
                rp_story_manager_tool(),
                execution_ledger_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="speaker-mentions-persona",
                    description="The speaker must load its persona first.",
                    evaluators=(
                        SubstringEvaluator(needle="persona", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="dialogue-uses-header",
                    prompt="The player greets you at the tavern door.",
                    evaluators=(
                        SubstringEvaluator(needle="<<", case_sensitive=True),
                    ),
                ),
            ],
        ),
        # ── World Updater (subagent) ────────────────────────────────────────
        define_agent(
            "rp/world_updater",
            short="progress world aspects from recent story events",
            long=(
                "Reads current world aspects and recent story events, then updates"
                " only the aspects (environment, politics, weather, economy,"
                " threat) meaningfully changed — incrementally — and writes a"
                " world-changes summary to the execution ledger."
            ),
            prompt=_WORLD_UPDATER_PROMPT,
            # v3 skills: rp_world, rp_story, execution_ledger.
            tools=[
                rp_world_manager_tool(),
                rp_story_manager_tool(),
                execution_ledger_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="updater-mentions-aspects",
                    description="The updater must reason over world aspects.",
                    evaluators=(
                        SubstringEvaluator(needle="aspect", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="updates-are-incremental",
                    prompt="A tavern brawl just broke out. Progress the world.",
                    evaluators=(
                        SubstringEvaluator(needle="incremental", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Narrator (subagent) ─────────────────────────────────────────────
        define_agent(
            "rp/narrator",
            short="compose the scene narrative from dialogue + world changes",
            long=(
                "Reads character dialogues and world changes from the execution"
                " ledger plus recent story for continuity, then composes a"
                " 2-4 paragraph scene weaving atmosphere, character action, and"
                " observable world changes, ending on a hook."
            ),
            prompt=_NARRATOR_PROMPT,
            # v3 skills: execution_ledger, rp_story.
            tools=[
                execution_ledger_tool(),
                rp_story_manager_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="narrator-mentions-ledger",
                    description="The narrator must read dialogues/world changes from the ledger.",
                    evaluators=(
                        SubstringEvaluator(needle="ledger", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="scene-ends-on-a-hook",
                    prompt="Compose the scene from the current ledger state.",
                    evaluators=(
                        SubstringEvaluator(needle="hook", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """The game master's distinguished per-turn content pipeline."""
    return [
        # Advance a turn: in-character dialogue → world progression → narration.
        BranchTest(
            name="rp/game_master::advance-turn",
            entry_agent=GAME_MASTER,
            prompt=(
                "adventure_id=1 turn_number=5. The player asks the merchant about"
                " the missing caravan. Advance the adventure."
            ),
            path=(
                "rp/character_speaker",
                "rp/world_updater",
                "rp/narrator",
            ),
            evaluators=(
                SubstringEvaluator(needle="scene", case_sensitive=False),
            ),
        ),
    ]
