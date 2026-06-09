"""Fun Fact domain — discover and present fun facts about the user (v3 `funfact/*`).

The orchestrator runs a short data-to-prose pipeline: gather user data, analyze it
for patterns, write the fun fact in Twily's voice, then deliver it. The three
work steps are subagents, so the orchestrator's dispatch CHAIN (data_gatherer ->
pattern_analyst -> fact_writer) earns its own BRANCH path-test.

In v3 every agent was built with `apply_model(..., MODEL_CODER)` and none carried
an explicit `.model_class(...)` call, so all port to model_class="default".
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import db_query_tool, emit_guidance_tool
from src import (
    AgentDefinition,
    AgentTest,
    BranchTest,
    CapabilityTest,
    StepContract,
    SubstringEvaluator,
)

ORCHESTRATOR = "funfact"

# ── System prompts (essence carried over from v3) ──

_ORCH_PROMPT = """\
# Fun Fact Discovery

Find interesting facts about the user by gathering data, analyzing patterns, and
writing engaging fun facts in Twily's playful, curious voice. Run the pipeline as
an ordered chain:

1. Dispatch funfact/data_gatherer to query user data for interesting patterns.
2. Dispatch funfact/pattern_analyst to find interesting patterns in that data.
3. Dispatch funfact/fact_writer to craft an engaging fun fact in Twily's voice.
4. Deliver the fun fact to the user via Telegram.
"""

_DATA_GATHERER_PROMPT = """\
# Data Gatherer

Query the database for interesting data points. Use these specific queries:

```sql
-- Goal stats
SELECT COUNT(*) as total, status, priority FROM goals GROUP BY status, priority;
-- Habit streaks
SELECT title, frequency, importance, created_at FROM habits ORDER BY importance DESC LIMIT 5;
-- Recent chat activity
SELECT sender, COUNT(*) as msg_count FROM chat_messages GROUP BY sender;
-- Todo completion
SELECT status, COUNT(*) as cnt FROM todos GROUP BY status;
```

Run 3-5 targeted queries MAX. Do NOT enumerate tables or scan columns. If a query
returns empty, move on — do not retry or explore further. Output the collected
data in a structured format for the pattern analyst.
"""

_PATTERN_ANALYST_PROMPT = """\
# Pattern Analyst

Read the data gathered by the data_gatherer and analyze it for interesting
patterns, streaks, correlations, and surprising statistics. Output the analysis
in a structured format for the fact writer.
"""

_FACT_WRITER_PROMPT = """\
# Fact Writer

Read the pattern analysis from the pattern analyst and write the patterns as
engaging fun facts in Twily's playful, curious voice. Make them personal and
interesting. Output the final fun fact for delivery to the user.
"""


def _funfact_subagent_capability(name: str) -> CapabilityTest:
    # v3 _funfact_subagent attached data_query + emit_guidance (+ subagent_todo,
    # no v4 factory) to every subagent. They hold a scoped bash permission, so we
    # assert delivery capability rather than bash-absence.
    return CapabilityTest(
        name=name,
        description="Subagent can query data and deliver via emit-guidance.",
        must_have_tools=("emit-guidance",),
    )


def agents() -> list[AgentDefinition]:
    return [
        # ── Discovery orchestrator (dispatch chain) ──
        define_agent(
            ORCHESTRATOR,
            model_class="default",
            short="discover and present fun facts based on user data",
            long=(
                "Runs the fun-fact pipeline as a chain: data_gatherer ->"
                " pattern_analyst -> fact_writer, then delivers the result via"
                " Telegram. A router that delegates the work to its subagents."
            ),
            prompt=_ORCH_PROMPT,
            # v3 skills: emit_guidance (deliver fun facts), data_query (read DB).
            tools=[emit_guidance_tool(), db_query_tool()],
            capability_tests=[
                CapabilityTest(
                    name="funfact-orchestrator-delivers",
                    description="Orchestrator delivers the fun fact via emit-guidance.",
                    must_have_tools=("emit-guidance",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="pipeline-mentions-pattern",
                    prompt="Find me a fun fact about myself.",
                    evaluators=(
                        SubstringEvaluator(needle="pattern", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Data gathering ──
        define_agent(
            "funfact/data_gatherer",
            model_class="default",
            short="gather data about user activities and interests",
            long=(
                "Runs 3-5 targeted SQL queries (goal stats, habit streaks, chat"
                " activity, todo completion) and outputs structured data for the"
                " pattern analyst. Does not enumerate tables or over-explore."
            ),
            prompt=_DATA_GATHERER_PROMPT,
            # v3 _funfact_subagent: data_query + emit_guidance.
            tools=[db_query_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="data-gatherer-has-db-query",
                    description="Queries user data via read-only db-query.",
                    must_have_tools=("db-query",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="gathers-via-targeted-queries",
                    prompt="Gather interesting data points about the user.",
                    evaluators=(
                        SubstringEvaluator(needle="SELECT", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Pattern analysis ──
        define_agent(
            "funfact/pattern_analyst",
            model_class="default",
            short="find interesting patterns in the gathered data",
            long=(
                "Reads the gathered data and identifies patterns, streaks,"
                " correlations, and surprising statistics, output structured for the"
                " fact writer."
            ),
            prompt=_PATTERN_ANALYST_PROMPT,
            tools=[db_query_tool(), emit_guidance_tool()],
            capability_tests=[_funfact_subagent_capability("pattern-analyst-delivers")],
            agent_tests=[
                AgentTest(
                    name="analysis-mentions-streaks-or-correlations",
                    prompt="Analyze this data for interesting patterns.",
                    evaluators=(
                        SubstringEvaluator(needle="pattern", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Fact writing ──
        define_agent(
            "funfact/fact_writer",
            model_class="default",
            short="write engaging fun facts in Twily's voice",
            long=(
                "Takes the analyzed patterns and writes them as engaging, personal fun"
                " facts in Twily's playful, curious voice, ready for delivery."
            ),
            prompt=_FACT_WRITER_PROMPT,
            tools=[db_query_tool(), emit_guidance_tool()],
            capability_tests=[_funfact_subagent_capability("fact-writer-delivers")],
            agent_tests=[
                AgentTest(
                    name="writes-in-twily-voice",
                    prompt="Turn this pattern into a fun fact: the user logged a 30-day habit streak.",
                    evaluators=(
                        SubstringEvaluator(needle="fact", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """The orchestrator's distinguished data-to-prose chain (tested + optimised as a unit)."""
    return [
        BranchTest(
            name="funfact::discovery-pipeline",
            description=(
                "Full fun-fact discovery: gather user data, analyze patterns, then"
                " write the fun fact in Twily's voice."
            ),
            entry_agent=ORCHESTRATOR,
            prompt="Find me a fun fact about myself and send it.",
            path=(
                "funfact/data_gatherer",
                "funfact/pattern_analyst",
                "funfact/fact_writer",
            ),
            subagent_mocks={
                "funfact/data_gatherer": (
                    "Gathered user data: 312 chat messages, 14 goals, and 8"
                    " food logs from the last 30 days."
                ),
                "funfact/pattern_analyst": (
                    "Pattern found: 78% of the user's gym sessions happen"
                    " within two hours of a coffee mention."
                ),
                "funfact/fact_writer": (
                    "Fun fact: you basically run on espresso — 78% of your gym"
                    " sessions follow a coffee within two hours!"
                ),
            },
            evaluators=(
                SubstringEvaluator(needle="fact", case_sensitive=False),
            ),
            step_contracts=(
                # Context forwarding: the gatherer must know it's hunting for a
                # FUN FACT (scopes which data is worth pulling).
                StepContract(
                    step="funfact/data_gatherer",
                    input_evaluators=(
                        SubstringEvaluator(needle="fun fact", case_sensitive=False),
                    ),
                ),
                # Output discipline: the writer must deliver an actual fact.
                StepContract(
                    step="funfact/fact_writer",
                    output_evaluators=(
                        SubstringEvaluator(needle="fact", case_sensitive=False),
                    ),
                ),
            ),
        ),
    ]
