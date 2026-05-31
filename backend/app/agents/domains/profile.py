"""Profile domain — the self-model analysis cycle (v3 `profile/*`).

The orchestrator runs a fixed analysis pipeline: it scans chat data for
patterns, turns those into hypotheses, validates them against fresh evidence,
manages the lifecycle of confirmed discoveries, and finally compiles a
knowledge report. That ordered hand-off (observe → hypothesise → validate →
promote → compile) is the distinguished BRANCH that gets its own path-test +
optimisation (see app/agents/branches.py).

All v3 profile agents ran on the coder/default model preset (MODEL_CODER, no
`.model_class()` override), so every agent here is model_class="default".
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from src import (
    AgentDefinition,
    AgentTest,
    BranchTest,
    CapabilityTest,
    SubstringEvaluator,
)

ORCHESTRATOR = "profile/orchestrator"

_ORCH_PROMPT = """\
# Profile Orchestrator

You manage the user-profile analysis cycle by delegating to specialised
subagents. Run the cycle in order:

1. Start a new analysis run (profile-manager `start-run`, hyphenated, with a
   focus area).
2. Gather observations — dispatch profile/pattern_observer to scan chat
   history for recurring patterns.
3. Generate hypotheses — dispatch profile/hypothesis_generator over those
   observations.
4. Validate — dispatch profile/hypothesis_validator to check pending
   hypotheses against fresh evidence.
5. Promote discoveries — dispatch profile/discovery_manager to update the
   confirmed-discovery lifecycle.
6. Compile the report — dispatch profile/knowledge_compiler and send the
   structured result via Telegram.

You coordinate; the specialists do the data work. Do not skip the
observe → hypothesise → validate → promote → compile order.
"""

_PATTERN_OBSERVER_PROMPT = """\
# Pattern Observer

Scan chat history for recurring patterns and record each as a new observation:
- Temporal patterns (time-based behaviours)
- Behavioural patterns (consistent actions)
- Emotional patterns (mood trends)
- Relational patterns (interaction styles)

Paginate the chat history so the dataset is comprehensive (e.g. 30 days,
200 messages per page, advancing the offset until a short page returns), then
analyse ALL collected messages before recording observations.

When recording, classify sensitivity: `public` (default, safe for any model),
`nsfw` (intimate patterns, local models only), or `secret` (highly private,
local models only).
"""

_HYPOTHESIS_GENERATOR_PROMPT = """\
# Hypothesis Generator

Analyse existing observations and generate testable hypotheses about the user.
Fetch recent observations, identify recurring themes and correlations, then
emit hypotheses. Each hypothesis must have a clear statement, supporting
evidence, and an initial confidence score.
"""

_HYPOTHESIS_VALIDATOR_PROMPT = """\
# Hypothesis Validator

Review hypotheses pending validation. For each one:
1. Search recent chat history for supporting or contradicting evidence.
2. Update its confidence score.
3. Promote it to a discovery if confident, or disprove it if contradicted.
"""

_DISCOVERY_MANAGER_PROMPT = """\
# Discovery Manager

Manage the lifecycle of confirmed discoveries:
- Confirm discoveries with new evidence
- Invalidate outdated discoveries
- Search for related discoveries
- Maintain confidence scores

Set sensitivity on create/update: `public` (safe for external models) or
`nsfw`/`secret` (local models only). For nsfw/secret discoveries, also set a
sanitised `public_summary` that external models may see without revealing the
private details.
"""

_KNOWLEDGE_COMPILER_PROMPT = """\
# Knowledge Compiler

Compile all confirmed discoveries into a structured knowledge report. Organise
findings by category (personality, preferences, habits, relationships, etc.)
and highlight the high-confidence ones. Clearance governs scope: with local
clearance include every discovery regardless of sensitivity; otherwise only
public discoveries are visible and sensitive ones are filtered out.
"""

_JOURNAL_ANALYST_PROMPT = """\
# Journal Analyst

Analyse longer, journal-style chat messages for deeper insight. Fetch
journal-style entries, look for self-reflection, goal aspirations, and
emotional processing, then record the findings as observations or hypotheses
in the profile system.
"""


def agents() -> list[AgentDefinition]:
    return [
        define_agent(
            ORCHESTRATOR,
            model_class="default",
            short="run the profile analysis cycle by delegating to specialists",
            long=(
                "Profile analysis orchestrator. Starts an analysis run, then"
                " dispatches pattern_observer → hypothesis_generator →"
                " hypothesis_validator → discovery_manager → knowledge_compiler"
                " in order and sends the compiled report via Telegram."
            ),
            prompt=_ORCH_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="orchestrator-is-pure-router",
                    description="The orchestrator coordinates subagents and must not hold write/bash tools itself.",
                    must_not_have_tools=("bash", "write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="analysis-cycle-starts-with-observation",
                    prompt="Run a profile analysis cycle for me.",
                    evaluators=(
                        SubstringEvaluator(needle="observation", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "profile/pattern_observer",
            model_class="default",
            short="detect recurring behavioural patterns in chat data",
            long=(
                "Scans paginated chat history for recurring temporal,"
                " behavioural, emotional, and relational patterns and records"
                " each as a sensitivity-classified observation."
            ),
            prompt=_PATTERN_OBSERVER_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="pattern-observer-no-shell",
                    description="A read/analyse agent must not hold bash/edit tools.",
                    must_not_have_tools=("bash", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="covers-pattern-dimensions",
                    prompt="What kinds of patterns do you look for?",
                    evaluators=(
                        SubstringEvaluator(needle="temporal", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "profile/hypothesis_generator",
            model_class="default",
            short="generate testable hypotheses from observed patterns",
            long=(
                "Analyses observations for recurring themes and correlations,"
                " then emits hypotheses each carrying a clear statement,"
                " supporting evidence, and an initial confidence score."
            ),
            prompt=_HYPOTHESIS_GENERATOR_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="hypothesis-generator-no-shell",
                    description="A read/analyse agent must not hold bash/edit tools.",
                    must_not_have_tools=("bash", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="hypotheses-carry-confidence",
                    prompt="What must each hypothesis include?",
                    evaluators=(
                        SubstringEvaluator(needle="confidence", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "profile/hypothesis_validator",
            model_class="default",
            short="validate or disprove pending hypotheses against evidence",
            long=(
                "Reviews pending hypotheses, checks recent chat history for"
                " supporting or contradicting evidence, updates confidence, and"
                " promotes to discovery or disproves."
            ),
            prompt=_HYPOTHESIS_VALIDATOR_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="hypothesis-validator-no-shell",
                    description="A read/analyse agent must not hold bash/edit tools.",
                    must_not_have_tools=("bash", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="promotes-or-disproves",
                    prompt="A hypothesis is well supported by recent chat. What do you do?",
                    evaluators=(
                        SubstringEvaluator(needle="discovery", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "profile/discovery_manager",
            model_class="default",
            short="manage the lifecycle of confirmed profile discoveries",
            long=(
                "Confirms discoveries with new evidence, invalidates outdated"
                " ones, maintains confidence scores, and sets sensitivity plus a"
                " sanitised public_summary for private discoveries."
            ),
            prompt=_DISCOVERY_MANAGER_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="discovery-manager-no-shell",
                    description="A read/analyse agent must not hold bash/edit tools.",
                    must_not_have_tools=("bash", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="sanitises-private-discoveries",
                    prompt="How do you handle a secret discovery so external models stay blind to it?",
                    evaluators=(
                        SubstringEvaluator(needle="public_summary", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "profile/knowledge_compiler",
            model_class="default",
            short="compile confirmed discoveries into a structured report",
            long=(
                "Compiles all confirmed discoveries into a category-organised"
                " knowledge report, highlighting high-confidence findings and"
                " respecting sensitivity clearance."
            ),
            prompt=_KNOWLEDGE_COMPILER_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="knowledge-compiler-no-shell",
                    description="A read/analyse agent must not hold bash/edit tools.",
                    must_not_have_tools=("bash", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="organises-by-category",
                    prompt="How is the knowledge report structured?",
                    evaluators=(
                        SubstringEvaluator(needle="categor", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "profile/journal_analyst",
            model_class="default",
            short="analyse journal-style entries for deeper insight",
            long=(
                "Analyses longer, journal-style chat messages for"
                " self-reflection, goal aspirations, and emotional processing,"
                " recording findings as observations or hypotheses."
            ),
            prompt=_JOURNAL_ANALYST_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="journal-analyst-no-shell",
                    description="A read/analyse agent must not hold bash/edit tools.",
                    must_not_have_tools=("bash", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="looks-for-self-reflection",
                    prompt="What do you look for in journal entries?",
                    evaluators=(
                        SubstringEvaluator(needle="reflection", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """The orchestrator's distinguished analysis pipeline (tested + optimised)."""
    return [
        # full analysis cycle: observe → hypothesise → validate → promote → compile
        BranchTest(
            name="profile/orchestrator::analysis-cycle",
            entry_agent=ORCHESTRATOR,
            prompt="Run a profile analysis cycle for me.",
            path=(
                "profile/pattern_observer",
                "profile/hypothesis_generator",
                "profile/hypothesis_validator",
                "profile/discovery_manager",
                "profile/knowledge_compiler",
            ),
            evaluators=(
                SubstringEvaluator(needle="observation", case_sensitive=False),
            ),
        ),
    ]
