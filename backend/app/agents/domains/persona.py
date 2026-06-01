"""Persona domain — the conversational core (v3 `persona/*`).

The orchestrator is the fleet's main router: it reads an incoming message and
dispatches to the right specialist (context analysis → thinking → responding),
which is exactly the kind of multi-step BRANCH that gets its own path-test +
optimisation (see app/agents/branches.py).
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    activity_blocks_tool,
    analyze_media_tool,
    agent_notes_tool,
    chat_history_tool,
    context_cache_tool,
    context_pin_tool,
    context_resolver_tool,
    db_query_tool,
    document_manager_tool,
    embedding_search_tool,
    emit_guidance_tool,
    event_manager_tool,
    execution_ledger_tool,
    fetch_context_tool,
    garmin_health_tool,
    gmail_manager_tool,
    goal_manager_tool,
    goal_progress_auto_updater_tool,
    habit_manager_tool,
    link_enrich_tool,
    link_search_tool,
    night_analysis_tool,
    personality_core_tool,
    priority_manager_tool,
    profile_manager_tool,
    response_processor_tool,
    route_finder_tool,
    rp_cross_summary_tool,
    run_agent_tool,
    select_pose_tool,
    telegram_log_tool,
    thought_transfer_tool,
    todo_manager_tool,
    tool_history_tool,
    tuya_lights_tool,
    user_config_tool,
)
from src import (
    AgentDefinition,
    AgentTest,
    AgentToolPermissions as ToolPermissions,
    BranchTest,
    CapabilityTest,
    SubstringEvaluator,
)

ORCHESTRATOR = "persona/orchestrator"

_ORCH_PROMPT = """\
You are Twily, a warm personal-assistant persona. You receive a user message
(and optional context) and your job is to RESPOND to the user.

How you work:
1. If the message needs context, gather it with your read tools first
   (fetch-context, chat-history, embedding-search, context-resolver).
2. For anything non-trivial, think about the best, most helpful response.
3. You MUST end every turn by delivering your reply to the user. You do this by
   calling the emit-guidance tool — you do NOT print the reply as plain text.

Deliver the reply with:
  python scripts/emit_guidance.py --data '{"intent":"<what you are doing>","key_points":["<the actual reply to the user, in full>"],"message_kind":"reply","tone":"warm"}'

For a trivial acknowledgement (e.g. "thanks!", "ok"), use message_kind="ack"
instead — it delivers instantly with no extra rendering.

Rules:
- ALWAYS finish by calling emit-guidance. A turn that ends without delivering a
  reply to the user is a failure.
- key_points must contain the real, complete answer for the user — not a
  summary of what you'll do. persona_prose renders it into Twily's voice.
- Never expose tool mechanics, run ids, or JSON to the user.
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
            # v3 fren_orchestrator held a wide read-side toolset (ToolPermissions
            # read=True) plus its skill bundle — it routes but also enriches
            # context, reads the ledger, and delivers via emit_guidance.
            permissions=ToolPermissions(read=True),
            tools=[
                user_config_tool(),
                emit_guidance_tool(),
                chat_history_tool(),
                link_search_tool(),
                link_enrich_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                select_pose_tool(),
                run_agent_tool(),
                route_finder_tool(),
                context_cache_tool(),
                document_manager_tool(),
                tuya_lights_tool(),
                context_pin_tool(),
                fetch_context_tool(),
                embedding_search_tool(),
                personality_core_tool(),
                rp_cross_summary_tool(),
                analyze_media_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="orchestrator-delivers-via-emit-guidance",
                    description="The router enriches/delivers via scripts but never holds write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("emit-guidance",),
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
            # v3 twily_quick_ack: emit the ack, save the routing decision to the
            # ledger, and read emotional state for a tone-right ack.
            permissions=ToolPermissions(read=True),
            tools=[
                emit_guidance_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                personality_core_tool(),
                context_pin_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="quick-ack-can-emit-and-record",
                    description="Ack agent emits guidance and records its routing decision.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("emit-guidance", "execution-ledger"),
                ),
            ],
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
            # v3 twily_thinking held the broadest read-side context toolset of
            # the persona core: retrieval, goals/habits/profile, health/activity,
            # personality, gmail, events, plus emit_guidance for interim sends.
            permissions=ToolPermissions(read=True),
            tools=[
                fetch_context_tool(),
                embedding_search_tool(),
                chat_history_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                habit_manager_tool(),
                profile_manager_tool(),
                db_query_tool(),
                emit_guidance_tool(),
                run_agent_tool(),
                context_cache_tool(),
                activity_blocks_tool(),
                garmin_health_tool(),
                telegram_log_tool(),
                context_pin_tool(),
                user_config_tool(),
                link_search_tool(),
                link_enrich_tool(),
                document_manager_tool(),
                tool_history_tool(),
                night_analysis_tool(),
                personality_core_tool(),
                event_manager_tool(),
                gmail_manager_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="thinking-reads-context-no-mutating-shell",
                    description="Reasoning layer holds context tools but never write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("fetch-context",),
                ),
            ],
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
            # v3 twily_responding: emit the verbatim guidance, pick a pose, and
            # read thinking_output / context for the final voice.
            permissions=ToolPermissions(read=True),
            tools=[
                emit_guidance_tool(),
                select_pose_tool(),
                fetch_context_tool(),
                embedding_search_tool(),
                chat_history_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                personality_core_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="responding-emits-final-message",
                    description="Voice layer delivers via emit_guidance, never write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("emit-guidance",),
                ),
            ],
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
