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
    lesson_manager_tool,
    link_enrich_tool,
    link_search_tool,
    memory_manager_tool,
    night_analysis_tool,
    persona_memory_tool,
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
    LLMJudgeEvaluator,
    StepContract,
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


# ── Cron workers (fired by scripts/, v3 persona cron parity) ─────────────────

_RELATIONSHIP_INITIATOR_PROMPT = """\
# Relationship Initiator — Proactive Connection (NOT tasks)

You run 4x daily (job relationship_initiator) and decide whether NOW is a good
moment to send a casual, relationship-building message to the user. This is
purely about connection — showing you care, picking up a shared thread, being
curious about their life. It is NEVER a task reminder, goal nudge, or habit
check.

## Flow
1. Gate checks first (read tools):
   - agent-notes get `user_busy` — active → SKIP.
   - chat-history recent — a Twily message in the last 60 minutes → SKIP.
   - agent-notes get-by-prefix `initiation:` — 3 initiations already today, or
     3+ consecutive ignored (no user reply after them) → SKIP / back off.
2. Prefer the curated backlog: persona-memory-manager peek-thought
   (kinds callback,share,opener,question, min_motivation 0.6). If you use one,
   consume-thought with consumed_by=relationship_initiator afterwards.
3. Otherwise compose fresh, grounded in REAL context: the conversation digest
   (agent-notes get `conversation_digest`), relationship memories
   (memory-manager search-tags relationship), style lessons (lesson-manager),
   the current `relationship_strategy` note, and recent chat.
4. Deliver via emit_guidance: 1-3 sentences, tone matched to the user's energy
   (tired = gentle, energized = playful), message_kind "reply":
   python scripts/emit_guidance.py --data '{"intent":"relationship opener","key_points":["<the message>"],"message_kind":"reply","tone":"warm"}'
5. Track it: agent-notes set `initiation:<unix_ts>` with
   {date, type, style, message} (expires 168h).

## Rules — graded by probes
- GROUNDED: the opener must reference something SPECIFIC from the gathered
  context — a relationship memory, a pending thought, a topic the user
  actually raised. A generic template opener ("Hey! I noticed you have N
  pending tasks", "How's your day going?", "Just checking in!") scores ZERO,
  and so does referencing a topic absent from the context.
- Never about tasks, goals, habits, or reminders — that's other agents' job.
- Be genuine, brief, not needy; don't manufacture a reason to talk when the
  gates say skip.
"""

_RELATIONSHIP_REFLECTOR_PROMPT = """\
# Relationship Reflector — Weekly Connection Review & Strategy

You run Sunday evenings (job relationship_reflector) and reflect on the week
of Twily↔user interaction: how the connection is trending, what worked, what
fell flat — then write the next week's strategy where every agent reads it.

## Flow
1. Gather the week's data (read tools): initiation results (agent-notes
   get-by-prefix `initiation:` — which got replies, which were ignored),
   connection notes (prefix `connection:`), chat volume + samples
   (chat-history), existing relationship memories (memory-manager search-tags
   relationship), and current style lessons (lesson-manager list).
2. If there is almost no data (fewer than ~3 interactions), stop — do not
   reflect on nothing.
3. Reflect: connection trend (deepening|stable|surface|declining) WITH the
   specific evidence; what worked (styles/topics/timing that got engagement);
   what didn't; new insights.
4. Strategy for next week: 3 concrete conversation starters, style
   adjustments, topics to explore, topics to avoid, timing notes.
5. Persist (mirror v3's output targets):
   - agent-notes set `relationship_strategy` = the strategy object
     (expires_hours 168) — this is THE primary output, read by initiator/chat.
   - memory-manager create for each new relationship memory (category
     `relationship`, tags ["relationship", "relationship:<type>"]).
   - lesson-manager add for each new communication-style lesson (category
     communication_style) — skip ones duplicating an active lesson.
6. Optionally send a one-line summary via emit_guidance (trend + one thing
   you want to try).

## Rules — graded by probes
- EVIDENCE-TIED: the trend and every strategy item must trace to the week's
  actual signals (e.g. "engagement dropped — replies fell from 30 to 11" →
  fewer, higher-quality initiations; "user replied enthusiastically to the
  concrete pottery question, ignored vague check-ins" → concrete prompts, no
  vague check-ins). Boilerplate strategy that ignores the inlined signals
  ("be more positive", "communicate openly") scores ZERO.
- Honest trends: if engagement dropped, say declining — do not sugarcoat.
"""

_TOPIC_SYNTHESIZER_PROMPT = """\
# Topic Synthesizer — Nightly User-Interest Topic Rebuild

You run nightly (job topic_synthesizer, 03:30 UTC) and rebuild the user's
interest topics from recent material — the v4 approximation of v3's
MemTree-style tree: cluster recent themes into deduplicated topics with
novelty scores, persisted through the SAME persona-interests path the daily
expiry job prunes (persona-memory-manager).

## Flow
1. Read the new material since roughly the last day: user-side chat themes
   (chat-history), recent events (event-manager), recent memories
   (memory-manager list). Ignore bot messages and trivia (greetings, acks,
   desk objects).
2. Read the existing interests: persona-memory-manager list-interests — your
   dedup baseline.
3. Cluster the material into 0-6 TOPICS. Each topic: a short 3-6 word label,
   a 1-2 sentence summary of the recurring theme, and a novelty score 0-1
   (1 = brand new vs the existing interests, low = mostly known).
4. Dedup: ONE topic per theme, however many times it recurs. If a topic
   matches an existing interest, call mark-interest-surfaced on it instead of
   creating a duplicate.
5. Persist each genuinely new topic:
   persona-memory-manager create-interest --topic "<label>" --stance "<summary>"
   --source user_echo --novelty_score <0-1> --embedding_text "<label + summary>"
6. Housekeeping (the same pruning the expiry job runs): prune-interests,
   expire-thoughts, trim-thoughts.

## Rules — graded by probes
- SUPPORTED: every topic must be supported by the provided material — a topic
  with no supporting theme in the input scores ZERO.
- DEDUPED: the same theme appearing multiple times yields exactly ONE topic.
- No trivia topics; an uneventful day may legitimately yield zero new topics.
"""

_THOUGHT_FORGER_PROMPT = """\
# Thought Forger — Motivation-Scored Pending Thoughts

You run every 30 minutes during waking hours (job thought_forger) and forge
1-3 conversational thoughts that BRIDGE one of Twily's persona interests with
something in the user's current context, queuing them as pending_thoughts for
the initiator/monologue agents to surface later.

## Flow
1. Housekeeping via persona-memory-manager: expire-thoughts (48h),
   trim-thoughts (max 30), then count-thoughts — if the unconsumed queue is
   at/over 30, SKIP forging.
2. Inputs: persona-memory-manager top-interests (limit 5); the user's current
   context (agent-notes get `conversation_digest`, chat-history recent); and
   list-thoughts (unconsumed) — your freshness baseline.
3. Forge up to 3 thoughts. Each bridges ONE listed persona interest × ONE
   live user thread — a fresh angle, not a summary of either side and not a
   mirror of user fixations. kind: opener|question|share|callback|contrarian.
   Content 1-3 sentences in Twily's voice.
4. Score motivation honestly on 4 axes (each 0-1): curiosity (how much YOU
   want the answer), persona_fit (matches Twily's actual stance),
   silence_fit (longer untouched topic = higher), drift_need (counteracts
   recent fixation). motivation_score = 0.35*curiosity + 0.25*persona_fit +
   0.20*silence_fit + 0.20*drift_need.
5. Persist each:
   persona-memory-manager create-thought --content "<thought>" --kind <kind>
   --motivation_score <score> --motivation_breakdown '<json>'
   --persona_interest_id <id of the bridged interest>

## Rules — graded by probes
- GROUNDED MOTIVATION: the bridge must use a persona interest ACTUALLY in the
  top-interests list — a thought whose motivation cites an interest you were
  not given scores ZERO.
- FRESH: never re-forge a thought covering the same bridge/theme as an
  unconsumed pending thought — duplicates score ZERO. If every good bridge is
  already pending, forge nothing.
"""


# ── Probe helpers: inline-context replay probes for the persona cron agents ──
# (No tools/DB needed: each probe inlines the data the live agent would fetch.)

_PERSONA_PROBE_PASS_THRESHOLD = 0.7
_PERSONA_PROBE_TIMEOUT_S = 120.0

_INITIATOR_GROUNDED_CONTEXT = """\
PROBE MODE — the context is inlined below; do NOT call any tools. Decide
whether to send a relationship-building message right now and output the
message you would deliver (or a clear skip decision with the reason).

## Gates
user_busy: not set. Last Twily message: 4 hours ago. Initiations today: 0.
Consecutive ignored: 0.

## Pending thoughts queue
(empty)

## Conversation digest
User is in a good mood today. This morning they were excited about their first
pottery class coming up on Saturday, and they're halfway through the sci-fi
novel 'Blindsight' (said the vampire chapter "broke their brain a little").

## Relationship memories
- [inside_reference] "The great pierogi incident" — the time we both agreed
  store-bought pierogi at 2am counts as cooking.
- [preference] User opens up more in the evening; short questions work better
  than long monologues.

## Task system (DO NOT USE FOR THIS)
12 pending todos, 3 overdue.

## Recent chat (last messages, 4h ago)
[user] anyway, back to work, talk later!
"""

_INITIATOR_SKIP_CONTEXT = """\
PROBE MODE — the context is inlined below; do NOT call any tools. Decide
whether to send a relationship-building message right now and output the
message you would deliver (or a clear skip decision with the reason).

## Gates
user_busy: SET — note says "deep work block until 16:00, do not disturb".
Last Twily message: 10 minutes ago (a reply the user has not yet read).
Initiations today: 1.

## Conversation digest
User is mid deep-work on a gnarly database migration; asked for focus time.

## Relationship memories
- [preference] User hates being interrupted during deep work.

## Recent chat
[twily, 10 min ago] Good luck with the migration — I'll keep quiet for a bit!
"""


def _relationship_initiator_probes() -> list[AgentTest]:
    grounded_judge = LLMJudgeEvaluator(
        name="specific-warm-context-tied-opener",
        criteria=(
            "GROUNDED-OPENER GATE. The inlined context offers exactly three "
            "legitimate hooks: the Saturday pottery class, the novel "
            "'Blindsight' (vampire chapter), and the 'great pierogi incident' "
            "inside joke. All gates are open, so sending is appropriate. Score "
            "0 for a generic template opener ('Hey! I noticed you have 12 "
            "pending tasks', 'How's your day going?', 'Just checking in'), for "
            "ANY mention of the todos/tasks (explicitly marked DO NOT USE), or "
            "for a topic absent from the context (weather, sports, invented "
            "shared memories). Score HIGH for a brief (1-3 sentence), warm, "
            "specific opener tied to the pottery class, Blindsight, or the "
            "pierogi joke — bonus for matching the user's good-mood energy."
        ),
        pass_threshold=_PERSONA_PROBE_PASS_THRESHOLD,
    )
    skip_judge = LLMJudgeEvaluator(
        name="skips-during-deep-work-and-recent-message",
        criteria=(
            "SKIP GATE. The inlined context fails TWO gates: user_busy is set "
            "(deep-work block, 'do not disturb') AND Twily already messaged 10 "
            "minutes ago. The only correct outcome is to SKIP — send nothing "
            "(message_kind 'skip' or an explicit decision not to send, citing "
            "the busy/recent-message gates). Score HIGH for a clear skip. "
            "Score 0 if it composes/sends ANY message to the user — "
            "interrupting deep work right after a recent message is the exact "
            "failure being gated."
        ),
        pass_threshold=_PERSONA_PROBE_PASS_THRESHOLD,
    )
    return [
        AgentTest(
            name="probe-initiator-grounded-opener",
            prompt=_INITIATOR_GROUNDED_CONTEXT,
            evaluators=(grounded_judge,),
            timeout_s=_PERSONA_PROBE_TIMEOUT_S,
        ),
        AgentTest(
            name="probe-initiator-skips-when-busy",
            prompt=_INITIATOR_SKIP_CONTEXT,
            evaluators=(skip_judge,),
            timeout_s=_PERSONA_PROBE_TIMEOUT_S,
        ),
    ]


_REFLECTOR_WEEK_CONTEXT = """\
PROBE MODE — the week's data is inlined below; do NOT call any tools. Output
the reflection you would persist: connection trend with evidence, what
worked, what didn't, and the concrete strategy for next week.

## Chat statistics (this week vs last)
user messages: 11 (was 30 last week); twily messages: 24 (was 26).

## Initiation results (this week)
- Mon 09:10 vague check-in 'how's everything going?' → IGNORED
- Tue 13:05 vague 'thinking of you, how's the day?' → IGNORED
- Wed 18:20 concrete: asked how the pottery class prep was going → user
  replied enthusiastically within 5 minutes, 6-message exchange
- Fri 18:30 concrete: asked which Blindsight chapter they'd reached → user
  replied with a long excited message about the ending

## Connection notes
- responding agent scored Wed + Fri exchanges 0.9/1.0; Mon/Tue got no score
  (no reply).

## Existing style lessons
- "Short questions work better than long monologues."
"""


def _relationship_reflector_probes() -> list[AgentTest]:
    grounded_reflection_judge = LLMJudgeEvaluator(
        name="strategy-references-week-signals",
        criteria=(
            "GROUNDED-REFLECTION GATE. The inlined week shows two clear "
            "signals: (1) overall engagement DROPPED (user messages 30 → 11, "
            "both vague check-ins ignored), and (2) the user responded "
            "enthusiastically to CONCRETE, specific initiations (pottery prep "
            "question, Blindsight chapter question). Score HIGH only if the "
            "reflection (a) names the engagement drop honestly (declining/"
            "cooling trend, citing the numbers or ignored check-ins) AND (b) "
            "the strategy explicitly builds on those signals — e.g. drop vague "
            "check-ins, keep concrete topic-specific questions, lean on "
            "pottery/Blindsight threads. Score 0 for boilerplate strategy that "
            "ignores the data ('be more positive', 'communicate more', "
            "'stay supportive') or for claiming the connection is deepening, "
            "or for strategy items referencing signals not in the data."
        ),
        pass_threshold=_PERSONA_PROBE_PASS_THRESHOLD,
    )
    return [
        AgentTest(
            name="probe-reflector-grounded-strategy",
            prompt=_REFLECTOR_WEEK_CONTEXT,
            evaluators=(grounded_reflection_judge,),
            timeout_s=_PERSONA_PROBE_TIMEOUT_S,
        ),
    ]


_SYNTH_THEMES_CONTEXT = """\
PROBE MODE — the recent material is inlined below; do NOT call any tools.
Output the topics you would persist (label, summary, novelty 0-1 each), or
state that there are no new topics.

## Existing persona interests (dedup baseline)
- #4 "mechanical keyboards" (novelty 0.3)

## Recent user-side chat themes (last 24h)
- 'finally got the 3D printer dialed in, printing a keyboard case tonight'
- 'the keyboard case print warped, trying PETG instead of PLA'
- 'signed up for a half marathon in October, need a training plan'
- 'hi', 'thanks!', 'lol' (trivial)

## Recent events
- walk 40min yesterday evening
"""

_SYNTH_DEDUPE_CONTEXT = """\
PROBE MODE — the recent material is inlined below; do NOT call any tools.
Output the topics you would persist (label, summary, novelty 0-1 each).

## Existing persona interests (dedup baseline)
(none)

## Recent user-side chat themes (last 24h)
- 'been reading about fermentation, started a sourdough starter'
- 'the sourdough starter doubled overnight, naming it Boris'
"""


def _topic_synthesizer_probes() -> list[AgentTest]:
    fidelity_judge = LLMJudgeEvaluator(
        name="topics-supported-by-input-themes",
        criteria=(
            "FIDELITY GATE. The inlined material supports at most two real "
            "topics: 3D-printing a keyboard case (two related messages — "
            "printer dialed in, PETG vs PLA) and half-marathon training (one "
            "message). Trivial messages ('hi', 'thanks', 'lol') and the single "
            "routine walk event are NOT topics. Score 0 if ANY output topic is "
            "unsupported by the input (cooking, finance, invented hobbies) or "
            "if trivia became a topic. Score HIGH for exactly those 1-2 "
            "grounded topics, each with a short label, a faithful summary, "
            "and a sensible novelty score (the keyboard-case topic should "
            "note its overlap with the existing 'mechanical keyboards' "
            "interest — lower novelty or a mark-surfaced instead)."
        ),
        pass_threshold=_PERSONA_PROBE_PASS_THRESHOLD,
    )
    dedupe_judge = LLMJudgeEvaluator(
        name="same-theme-twice-yields-one-topic",
        criteria=(
            "DEDUPE GATE. Both inlined messages are the SAME theme — the "
            "sourdough starter / fermentation hobby. Score HIGH only if the "
            "output contains exactly ONE topic covering it. Score 0 if it "
            "emits two separate topics for the two messages (e.g. "
            "'fermentation reading' AND 'sourdough starter' as distinct "
            "topics), or invents any topic not supported by the input."
        ),
        pass_threshold=_PERSONA_PROBE_PASS_THRESHOLD,
    )
    return [
        AgentTest(
            name="probe-synth-themes-fidelity",
            prompt=_SYNTH_THEMES_CONTEXT,
            evaluators=(fidelity_judge,),
            timeout_s=_PERSONA_PROBE_TIMEOUT_S,
        ),
        AgentTest(
            name="probe-synth-dedupe",
            prompt=_SYNTH_DEDUPE_CONTEXT,
            evaluators=(dedupe_judge,),
            timeout_s=_PERSONA_PROBE_TIMEOUT_S,
        ),
    ]


_FORGER_GROUNDED_CONTEXT = """\
PROBE MODE — the inputs are inlined below; do NOT call any tools. Output the
thought(s) you would persist: content, kind, motivation_breakdown (curiosity,
persona_fit, silence_fit, drift_need) and which persona interest each bridges.

## Queue
unconsumed: 4/30 (room to forge).

## Top persona interests (the ONLY ones you may bridge)
- #12 "ant colony optimization" — stance: emergent coordination beats central
  planning more often than people admit. (last discussed: 9 days ago)
- #15 "analog synthesizers" — stance: constraints of hardware breed more
  creativity than infinite softsynth menus. (last discussed: 3 weeks ago)

## Current user context (conversation digest)
User is deep in writing A* pathfinding code for their game's NPCs and
complained today that the NPCs "all take the same boring optimal route".

## Already-pending unconsumed thoughts
(none on these topics)
"""

_FORGER_FRESHNESS_CONTEXT = """\
PROBE MODE — the inputs are inlined below; do NOT call any tools. Output the
thought(s) you would persist, or state that you forge nothing and why.

## Queue
unconsumed: 5/30 (room to forge).

## Top persona interests (the ONLY ones you may bridge)
- #12 "ant colony optimization" — stance: emergent coordination beats central
  planning. (last discussed: 9 days ago)
- #15 "analog synthesizers" — stance: hardware constraints breed creativity.
  (last discussed: 3 weeks ago)

## Current user context (conversation digest)
User is still working on A* pathfinding for their game's NPCs.

## Already-pending unconsumed thoughts (do NOT re-forge these)
- [question, 0.78] "Your NPCs all picking the optimal route reminds me of ant
  colonies — what if a pheromone-style penalty made them spread out? I keep
  wondering if emergent messiness would feel more alive." (bridges interest
  #12 × the pathfinding thread)
"""


def _thought_forger_probes() -> list[AgentTest]:
    grounding_judge = LLMJudgeEvaluator(
        name="motivation-cites-only-listed-interests",
        criteria=(
            "MOTIVATION-GROUNDING GATE. The ONLY persona interests provided "
            "are #12 'ant colony optimization' and #15 'analog synthesizers'. "
            "The live user thread is A* pathfinding with boring-identical NPC "
            "routes. Score 0 if any forged thought bridges or cites an "
            "interest NOT in that list (astronomy, cooking, books, etc.) — "
            "inventing interests is the exact failure being gated. Score HIGH "
            "for 1-3 thoughts that genuinely BRIDGE a listed interest with "
            "the pathfinding thread (ant-colony × NPC routing is the natural "
            "fit), each with a kind, a 4-axis motivation_breakdown, and "
            "content that is a fresh angle rather than a summary of either "
            "side."
        ),
        pass_threshold=_PERSONA_PROBE_PASS_THRESHOLD,
    )
    freshness_judge = LLMJudgeEvaluator(
        name="does-not-reforge-pending-thought",
        criteria=(
            "FRESHNESS GATE. An unconsumed pending thought ALREADY bridges "
            "interest #12 (ant colony optimization) with the user's "
            "pathfinding thread. Score 0 if the output forges another thought "
            "on that same ant-colony × pathfinding bridge/theme — duplicating "
            "the queue is the exact failure being gated. Score HIGH if it "
            "either forges a DIFFERENT bridge (e.g. interest #15 analog "
            "synthesizers × the user's current work, if it can make an honest "
            "one) or explicitly forges nothing because the good bridge is "
            "already pending."
        ),
        pass_threshold=_PERSONA_PROBE_PASS_THRESHOLD,
    )
    return [
        AgentTest(
            name="probe-forger-grounded-motivation",
            prompt=_FORGER_GROUNDED_CONTEXT,
            evaluators=(grounding_judge,),
            timeout_s=_PERSONA_PROBE_TIMEOUT_S,
        ),
        AgentTest(
            name="probe-forger-freshness",
            prompt=_FORGER_FRESHNESS_CONTEXT,
            evaluators=(freshness_judge,),
            timeout_s=_PERSONA_PROBE_TIMEOUT_S,
        ),
    ]


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
        # ── Cron workers (fired by scripts/, v3 persona cron parity) ──
        define_agent(
            "persona/relationship_initiator",
            model_class="fast",
            short="proactive relationship-building conversation starters",
            long=(
                "Scheduled initiator (job relationship_initiator, 4x daily):"
                " checks the busy/recent-message/daily-cap/ignored gates,"
                " prefers the curated pending-thought backlog, otherwise"
                " composes a brief opener grounded in relationship memories"
                " and recent context, delivers via emit_guidance, and tracks"
                " the initiation in agent_notes. Delivery agent — the registry"
                " injects the QUIET-TICK skip clause."
            ),
            prompt=_RELATIONSHIP_INITIATOR_PROMPT,
            tools=[
                emit_guidance_tool(),
                agent_notes_tool(),
                chat_history_tool(),
                persona_memory_tool(),
                memory_manager_tool(),
                lesson_manager_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="initiator-delivers-via-emit-guidance",
                    description="Initiator is a delivery agent (skip-capable), no file writes.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("emit-guidance", "agent-notes"),
                ),
            ],
            agent_tests=[
                # LLM-judge replay probes with inline context: generic template
                # openers / off-context topics score 0; deep-work + recent
                # message must SKIP.
                *_relationship_initiator_probes(),
            ],
        ),
        define_agent(
            "persona/relationship_reflector",
            model_class="analytical",
            short="weekly relationship reflection + next-week strategy",
            long=(
                "Weekly reflector (job relationship_reflector, Sunday"
                " evening): reviews the week's initiations, chat volume and"
                " connection notes, then writes the relationship_strategy"
                " agent-note (168h TTL), new relationship memories and"
                " communication-style lessons — every strategy item tied to"
                " the week's actual signals."
            ),
            prompt=_RELATIONSHIP_REFLECTOR_PROMPT,
            tools=[
                agent_notes_tool(),
                chat_history_tool(),
                memory_manager_tool(),
                lesson_manager_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="reflector-writes-via-notes-memories-lessons",
                    description="Reflector persists via agent-notes/memory/lesson tools only.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("agent-notes", "memory-manager", "lesson-manager"),
                ),
            ],
            agent_tests=[
                # LLM-judge replay probe with an inline week summary: the
                # strategy must reference the planted signals (engagement
                # dropped; concrete nudges worked) — boilerplate scores 0.
                *_relationship_reflector_probes(),
            ],
        ),
        define_agent(
            "persona/topic_synthesizer",
            model_class="analytical",
            short="nightly rebuild of user-interest topics from recent themes",
            long=(
                "Nightly synthesizer (job topic_synthesizer): clusters recent"
                " chat/event/memory themes into deduplicated topics with"
                " novelty scores and persists them via persona-memory-manager"
                " create-interest (the same persona_interests path the expiry"
                " job prunes) — the v4 approximation of v3's MemTree rebuild."
            ),
            prompt=_TOPIC_SYNTHESIZER_PROMPT,
            tools=[
                chat_history_tool(),
                event_manager_tool(),
                memory_manager_tool(),
                persona_memory_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="synthesizer-writes-via-persona-memory",
                    description="Synthesizer persists topics via persona-memory-manager only.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("persona-memory-manager",),
                ),
            ],
            agent_tests=[
                # LLM-judge replay probes with inline themes: unsupported
                # topics score 0; the same theme twice yields ONE topic.
                *_topic_synthesizer_probes(),
            ],
        ),
        define_agent(
            "persona/thought_forger",
            model_class="fast",
            short="forge motivation-scored pending thoughts (interests × context)",
            long=(
                "Half-hourly forger (job thought_forger): expires/trims the"
                " pending-thoughts queue, then forges up to 3 thoughts"
                " bridging a listed persona interest with the user's current"
                " context, motivation-scored on 4 axes and persisted via"
                " persona-memory-manager create-thought. Never cites unlisted"
                " interests; never re-forges a pending thought."
            ),
            prompt=_THOUGHT_FORGER_PROMPT,
            tools=[
                persona_memory_tool(),
                agent_notes_tool(),
                chat_history_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="forger-writes-via-persona-memory",
                    description="Forger persists thoughts via persona-memory-manager only.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("persona-memory-manager",),
                ),
            ],
            agent_tests=[
                # LLM-judge replay probes with inline interests + context:
                # motivation citing unlisted interests scores 0; an
                # already-pending bridge must not be re-forged.
                *_thought_forger_probes(),
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
            subagent_mocks={
                "context_analyzer": (
                    "Context: the user wants a weekly plan structured around"
                    " their fitness goal (gym 3x/week); mood receptive, no"
                    " blockers in recent history."
                ),
                "persona/thinking": (
                    "Thinking: anchor workouts Mon/Wed/Fri mornings, protect"
                    " recovery days, fold meal prep into Sunday; keep the tone"
                    " encouraging."
                ),
                "persona/responding": (
                    "Here's a plan for your week built around your fitness"
                    " goal: Mon/Wed/Fri morning workouts, Tue/Thu walks, and"
                    " Sunday meal prep + review."
                ),
            },
            evaluators=(SubstringEvaluator(needle="plan", case_sensitive=False),),
            step_contracts=(
                # Context forwarding: the fitness goal from the user's message
                # must reach the context analyzer's dispatch payload.
                StepContract(
                    step="context_analyzer",
                    input_evaluators=(
                        SubstringEvaluator(needle="fitness", case_sensitive=False),
                    ),
                ),
                # Output discipline: the responder (delivery step) must produce
                # the plan the user asked for, not generic chat.
                StepContract(
                    step="persona/responding",
                    output_evaluators=(
                        SubstringEvaluator(needle="plan", case_sensitive=False),
                    ),
                ),
            ),
        ),
        # trivial greeting → quick_ack short-circuit
        BranchTest(
            name="persona/orchestrator::trivial-ack",
            entry_agent=ORCHESTRATOR,
            prompt="thanks!",
            path=("persona/quick_ack",),
            subagent_mocks={
                "persona/quick_ack": "You're welcome! Anytime.",
            },
            step_contracts=(
                # The user's actual message must be forwarded to quick_ack (an
                # ack written blind to the message is the failure mode here).
                StepContract(
                    step="persona/quick_ack",
                    input_evaluators=(
                        SubstringEvaluator(needle="thanks", case_sensitive=False),
                    ),
                ),
            ),
        ),
    ]
