"""Persona domain — the conversational core (v3 `persona/*`).

The orchestrator is the fleet's main router: it reads an incoming message and
dispatches to the right specialist (context analysis → thinking → responding),
which is exactly the kind of multi-step BRANCH that gets its own path-test +
optimisation (see app/agents/branches.py).
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

ORCHESTRATOR = "persona/orchestrator"

_ORCH_PROMPT = """\
You are the orchestrator for a personal-assistant persona. You receive a user
message plus context and decide how to handle it. For anything non-trivial you
MUST gather context first (dispatch context_analyzer), then reason
(persona/thinking), then compose the reply (persona/responding). Only trivial
acknowledgements may skip straight to persona/quick_ack.

Never answer a substantive request without first analysing context — a reply
that skips analysis is a failure.
"""


def agents() -> list[AgentDefinition]:
    return [
        define_agent(
            ORCHESTRATOR,
            model_class="analytical",
            short="route a user message to the right persona specialists",
            long=(
                "Main message router. Decides whether a message is trivial"
                " (→ quick_ack) or substantive (→ context analysis → thinking →"
                " responding) and dispatches accordingly."
            ),
            prompt=_ORCH_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="orchestrator-is-pure-router",
                    description="The router must not hold write/bash tools itself.",
                    must_not_have_tools=("bash", "write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="substantive-message-mentions-context-first",
                    prompt="Help me plan my week around my fitness goal.",
                    evaluators=(
                        SubstringEvaluator(needle="context", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "persona/quick_ack",
            model_class="fast",
            short="fast, low-latency acknowledgements",
            long="Emits a brief warm acknowledgement for trivial messages.",
            prompt=(
                "Reply with a single short, warm acknowledgement. No tools, no"
                " analysis — you exist to be fast."
            ),
        ),
        define_agent(
            "persona/thinking",
            model_class="analytical",
            short="reasoning layer for the persona",
            long="Reasons over gathered context to decide what to say/do.",
            prompt=(
                "You are the reasoning layer. Given the user message and the"
                " analysed context, think step by step about the best response"
                " and hand a plan to persona/responding."
            ),
        ),
        define_agent(
            "persona/responding",
            model_class="fast",
            short="compose the final user-facing reply",
            long="Turns the thinking layer's plan into the persona's voice.",
            prompt=(
                "Compose the final reply in the persona's warm, concise voice"
                " from the plan you are given. Do not invent facts not in the"
                " plan or context."
            ),
        ),
    ]


def branches() -> list[BranchTest]:
    """The orchestrator's distinguished paths (tested + optimised as units)."""
    return [
        # substantive message → context → thinking → responding (never skip)
        BranchTest(
            name="persona/orchestrator::substantive-flow",
            entry_agent=ORCHESTRATOR,
            prompt="Help me plan my week around my fitness goal.",
            path=("context_analyzer", "persona/thinking", "persona/responding"),
            evaluators=(SubstringEvaluator(needle="plan", case_sensitive=False),),
        ),
        # trivial greeting → quick_ack short-circuit
        BranchTest(
            name="persona/orchestrator::trivial-ack",
            entry_agent=ORCHESTRATOR,
            prompt="thanks!",
            path=("persona/quick_ack",),
        ),
    ]
