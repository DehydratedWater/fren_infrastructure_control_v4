"""Vis Simulation domain — synthetic character training-data generator (v3 `vis_simulation/*`).

The orchestrator runs a classic generation pipeline: read a journal excerpt,
generate a scenario, create the record, simulate a 10-message conversation,
analyze character depth, and score quality. The four generation/analysis steps
are subagents, so the orchestrator's dispatch CHAIN (scenario_generator →
conversation_simulator → character_analyzer → quality_scorer) earns its own
BRANCH path-test.

In v3 every agent was built with `apply_model(..., MODEL_CODER)` and none carried
an explicit `.model_class(...)` call, so all port to model_class="default".
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    send_file_tool,
    send_image_tool,
    send_message_tool,
    send_voice_tool,
    vis_simulation_manager_tool,
)
from src import (
    AgentDefinition,
    AgentTest,
    BranchTest,
    CapabilityTest,
    SubstringEvaluator,
)

ORCHESTRATOR = "vis_simulation"

# ── System prompts (essence carried over from v3) ──

_ORCH_PROMPT = """\
# Vis Simulation — Character Training Data Generator

Generate synthetic conversation data for fine-tuning an LLM to simulate the Vis
character. Each simulation is a 10-message conversation (5 Vis + 5 non-Vis) with
internal thinking and external responses, produced as a pipeline.

## Process
1. Read a random journal excerpt for inspiration.
2. Check existing scenarios to avoid overlap.
3. Dispatch vis_simulation/scenario_generator to create a unique scenario from
   the excerpt.
4. Create the simulation record (scenario_type + description).
5. Dispatch vis_simulation/conversation_simulator to generate the 10 messages.
6. Validate: exactly 10 messages, 5 Vis with thinking_content (50+ words),
   strictly alternating non-Vis -> Vis.
7. Store the messages.
8. Dispatch vis_simulation/character_analyzer to assess depth.
9. Dispatch vis_simulation/quality_scorer for quality, realism, adherence.
10. Store scores, mark the simulation completed, and send a summary via Telegram.

## CRITICAL Rules
- thinking_content is THE MOST IMPORTANT PART (50+ words minimum per Vis message).
- Exactly 10 messages, strictly alternating non-Vis -> Vis.
- Every Vis message must have an actions array with 1+ physical action.
- No placeholder text.
"""

_SCENARIO_GENERATOR_PROMPT = """\
# Scenario Generator

Generate unique scenarios for Vis character simulations based on journal excerpts.

## Scenario Types
- reading — Vis reads something that triggers a reaction
- observing — Vis observes something in the environment
- desiring — Vis wants something (food, sleep, code working)
- reacting — Vis reacts to an event or message
- technical_task — Vis is working on ML/coding
- being_asked — Someone asks Vis something

## Output Format (JSON)
- scenario_type, scenario_description, scenario_assumptions
- emotional_state (dict with 0-1 scores: energy, stress, focus, social_battery)
- interlocutor_type (person/content/environment/self)
- interlocutor_description

## Rules
- Check existing scenarios to avoid overlap.
- Ground scenarios in actual journal content.
- Make emotional states nuanced (not all extremes).
"""

_CONVERSATION_SIMULATOR_PROMPT = """\
# Conversation Simulator

Generate 10-message conversations for Vis character fine-tuning.

## HARD REQUIREMENTS
- Exactly 10 messages, alternating non-Vis -> Vis.
- Exactly 5 Vis messages with thinking_content >= 50 words.
- Every Vis message has an `actions` array with 1+ physical action.
- Every Vis response_content >= 15 words.
- No placeholder text.

## Message Arc (5 pairs)
1. Hook: inciting event -> first gut reaction.
2. Deepening: follow-up -> growing engagement.
3. Core: main challenge -> richest response (longest thinking).
4. Complication: twist/obligation -> conflicted response.
5. Trailing Off: final cue -> characteristic non-resolution.

## Vis's Voice
- Self-interrupting, ADHD spirals; technical + emotional mixing.
- External narrated actions: [vis flicks ear], [vis's tail swishes].
- Sarcastic, self-deprecating but competent.

## Output
JSON array of 10 message objects with: sequence_number, sender,
response_content, thinking_content (Vis only, 50+ words), actions (Vis only,
list of physical actions), trigger_type (non-Vis only).
"""

_CHARACTER_ANALYZER_PROMPT = """\
# Character Analyzer

Analyze character depth by comparing generated messages to the journal.

## CRITICAL: Quote Actual Content
- Quote ACTUAL message content and journal passages; never fabricate quotes.
- Reference specific sequence numbers.

## Analysis Tasks
1. Completeness check — verify all 5 Vis messages have thinking_content.
2. Catalog actual message content with quotes.
3. Investigate the journal for relevant context.
4. Compare the simulation to journal patterns.

## Character Elements to Check
- Self-interruption patterns.
- Self-deprecation mixed with competence.
- ADHD tangents and context-switching.
- Technical competence in ML/coding.
- Procrastination awareness.

## Output (JSON)
character_depth_analysis (multi-paragraph with quotes), topics_investigated,
journal_references ([{topic, quote, relevance}]).
"""

_QUALITY_SCORER_PROMPT = """\
# Quality Scorer

Score simulation quality, realism, and character adherence on a 0-1 scale.

## Scoring Dimensions
- Quality (0-1): writing quality, coherence, engagement.
- Realism (0-1): plausibility of scenario and reactions.
- Character Adherence (0-1): how well Vis's voice is captured.

## CRITICAL: Must quote specific text from messages in notes.

## Scoring Guidelines
- 0.9-1.0: exceptional, publication-ready.
- 0.7-0.9: good, minor issues.
- 0.5-0.7: acceptable but notable weaknesses.
- Below 0.5: significant problems.

## Output
Return scores with detailed notes for each dimension.
"""


def _sim_tool_capability(name: str) -> CapabilityTest:
    # v3 subagents carried vis_simulation_skill (+ subagent_todo, which has no
    # v4 factory). With the manager tool attached they hold a scoped bash
    # permission, so we assert the tool is present rather than bash-absence.
    return CapabilityTest(
        name=name,
        description="Generation/analysis subagent can read/write simulation data via its manager tool.",
        must_have_tools=("vis-simulation-manager",),
    )


def agents() -> list[AgentDefinition]:
    return [
        # ── Generation pipeline orchestrator (dispatch chain) ──
        define_agent(
            ORCHESTRATOR,
            model_class="default",
            short="orchestrate synthetic Vis character training-data generation",
            long=(
                "Runs the simulation pipeline as a chain: scenario_generator ->"
                " conversation_simulator -> character_analyzer -> quality_scorer,"
                " plus journal reads, record/score CRUD, and a Telegram summary. A"
                " router that delegates generation/analysis to subagents."
            ),
            prompt=_ORCH_PROMPT,
            # v3 skills: vis_simulation (manager CRUD) + telegram_notification.
            tools=[
                vis_simulation_manager_tool(),
                send_message_tool(),
                send_voice_tool(),
                send_image_tool(),
                send_file_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="vis-orchestrator-has-manager",
                    description="Orchestrator runs simulation CRUD and sends the summary.",
                    must_have_tools=("vis-simulation-manager",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="pipeline-mentions-thinking-content",
                    prompt="Generate a new Vis character simulation.",
                    evaluators=(
                        SubstringEvaluator(needle="thinking", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Scenario generation ──
        define_agent(
            "vis_simulation/scenario_generator",
            model_class="default",
            short="generate a unique Vis scenario from a journal excerpt",
            long=(
                "Produces a scenario JSON (type, description, assumptions, nuanced"
                " emotional_state, interlocutor info) grounded in journal content and"
                " checked against existing scenarios to avoid overlap."
            ),
            prompt=_SCENARIO_GENERATOR_PROMPT,
            tools=[vis_simulation_manager_tool()],
            capability_tests=[_sim_tool_capability("scenario-generator-has-manager")],
            agent_tests=[
                AgentTest(
                    name="scenario-has-emotional-state",
                    prompt="Generate a scenario from a journal excerpt about debugging at 2am.",
                    evaluators=(
                        SubstringEvaluator(needle="emotional", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Conversation simulation ──
        define_agent(
            "vis_simulation/conversation_simulator",
            model_class="default",
            short="generate a 10-message Vis conversation with thinking content",
            long=(
                "Generates exactly 10 alternating messages (5 Vis with thinking_content"
                " >= 50 words + actions, 5 non-Vis with triggers) following the"
                " Hook->Deepening->Core->Complication->Trailing-Off arc in Vis's voice."
            ),
            prompt=_CONVERSATION_SIMULATOR_PROMPT,
            tools=[vis_simulation_manager_tool()],
            capability_tests=[_sim_tool_capability("conversation-simulator-has-manager")],
            agent_tests=[
                AgentTest(
                    name="conversation-requires-ten-messages",
                    prompt="Simulate a conversation for a reading scenario.",
                    evaluators=(
                        SubstringEvaluator(needle="thinking_content", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Character depth analysis ──
        define_agent(
            "vis_simulation/character_analyzer",
            model_class="default",
            short="analyze character depth by comparing the simulation to the journal",
            long=(
                "Quotes actual message content and journal passages (never fabricated),"
                " checks completeness, and compares the simulation against journal"
                " patterns (self-interruption, ADHD tangents, competence)."
            ),
            prompt=_CHARACTER_ANALYZER_PROMPT,
            tools=[vis_simulation_manager_tool()],
            capability_tests=[_sim_tool_capability("character-analyzer-has-manager")],
            agent_tests=[
                AgentTest(
                    name="analysis-references-journal",
                    prompt="Analyze the character depth of the stored simulation.",
                    evaluators=(
                        SubstringEvaluator(needle="journal", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Quality scoring ──
        define_agent(
            "vis_simulation/quality_scorer",
            model_class="default",
            short="score simulation quality, realism, and character adherence (0-1)",
            long=(
                "Scores three dimensions on a 0-1 scale (quality, realism, character"
                " adherence) and must quote specific message text in its notes."
            ),
            prompt=_QUALITY_SCORER_PROMPT,
            tools=[vis_simulation_manager_tool()],
            capability_tests=[_sim_tool_capability("quality-scorer-has-manager")],
            agent_tests=[
                AgentTest(
                    name="scores-three-dimensions",
                    prompt="Score the stored simulation.",
                    evaluators=(
                        SubstringEvaluator(needle="realism", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """The orchestrator's distinguished generation chain (tested + optimised as a unit)."""
    return [
        BranchTest(
            name="vis_simulation::generation-pipeline",
            description=(
                "Full simulation pipeline: generate scenario, simulate conversation,"
                " analyze character depth, then score quality."
            ),
            entry_agent=ORCHESTRATOR,
            prompt="Generate a new Vis character simulation and score it.",
            path=(
                "vis_simulation/scenario_generator",
                "vis_simulation/conversation_simulator",
                "vis_simulation/character_analyzer",
                "vis_simulation/quality_scorer",
            ),
            evaluators=(
                SubstringEvaluator(needle="score", case_sensitive=False),
            ),
        ),
    ]
