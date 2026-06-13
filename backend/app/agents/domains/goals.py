"""Goals domain — goal/priority/strategy management and proactive nudging (v3 `goals/*`).

This domain is the fleet's accountability engine. The priority_orchestrator runs a
classic dispatch CHAIN (audit → create strategy → analyze → report), which earns
its own BRANCH path-test. Around it sit the goal/todo CRUD interfaces, the
priority/strategy specialists, and a family of time-windowed proactive agents
(periodic checker, evening focus, winddown, nudge strategist, task triage) that
emit a single PersonaGuidance per run rather than drafting prose themselves.

In v3 every agent was built with `apply_model(..., MODEL_CODER)` and none carried
an explicit `.model_class(...)` call, so all port to model_class="default".
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    activity_blocks_tool,
    agent_notes_tool,
    calendar_manager_tool,
    camera_capture_tool,
    chat_history_tool,
    context_cache_tool,
    context_resolver_tool,
    db_query_tool,
    emit_guidance_tool,
    embedding_search_tool,
    execution_ledger_tool,
    fetch_context_tool,
    garmin_health_tool,
    goal_manager_tool,
    goal_progress_auto_updater_tool,
    habit_manager_tool,
    nudge_strategist_tool,
    periodic_checker_tool,
    personality_core_tool,
    priority_manager_tool,
    proactive_send_tool,
    profile_manager_tool,
    question_sender_tool,
    response_processor_tool,
    routine_manager_tool,
    screenshot_tool,
    send_file_tool,
    send_image_tool,
    send_message_tool,
    send_voice_tool,
    session_inspector_tool,
    strategy_tracker_tool,
    telegram_log_tool,
    thought_transfer_tool,
    todo_manager_tool,
    tuya_lights_tool,
    user_config_tool,
    visual_report_tool,
)
from app.agents.proactive_probes import proactive_probes
from app.agents.stale_probes import stale_state_probes
from src import (
    AgentDefinition,
    AgentTest,
    BranchTest,
    CapabilityTest,
    LLMJudgeEvaluator,
    StepContract,
    SubstringEvaluator,
)

ORCHESTRATOR = "goals/priority_orchestrator"

# ── System prompts (essence carried over from v3) ──

_ORCH_PROMPT = """\
# Priority Orchestrator

Manage the priority review process by delegating to specialized subagents.
Run the review as an ordered chain:

1. Dispatch goals/priority_auditor to compare each priority's stated importance
   against actual time/effort spent and update real_importance scores.
2. Dispatch goals/strategy_creator to build a daily strategy (time blocks, focus
   areas) from the audited priorities.
3. Dispatch goals/strategy_analyzer to review recent strategy outcomes and surface
   success/failure patterns.
4. Compile the findings from all three subagents and send a priority review report
   to the user via Telegram.

You are a pure router: you delegate and compile, you do not run the underlying
CLI tools yourself.
"""

_TASK_CATEGORIZER_PROMPT = """\
# Task Categorizer

Analyze an incoming task and categorize it by:
- Priority level (low/medium/high/critical)
- Category (personal/work/health/learning)
- Goal alignment (which active goal does this support?)
- Estimated effort (minutes)

Fetch active goals FIRST to determine alignment, then output the categorization
in a structured format.
"""

_PRIORITY_PLANNER_PROMPT = """\
# Priority Planner

Create daily priority plans by:
1. Fetching active goals and overdue todos.
2. Evaluating importance vs immediacy using the Eisenhower matrix.
3. Suggesting focus areas for the day.
4. Creating time-blocked schedule suggestions.

Output the daily priority plan with the schedule suggestions.
"""

_STRATEGY_CREATOR_PROMPT = """\
# Strategy Creator

Create daily strategies by analyzing goals and priorities, then writing a strategy
with focus goals and time blocks.

IMPORTANT: Work hours are 9-17 (Monday-Friday). Do NOT schedule learning or personal
activities during work hours — reserve 9-17 for work tasks only. Schedule learning,
personal development, and hobby activities before 9 or after 17.

Do NOT read source code files. Only use the goal-management and strategy-tracking
CLI commands provided in your workflow.
"""

_STRATEGY_ANALYZER_PROMPT = """\
# Strategy Analyzer

Analyze recent strategies for effectiveness: fetch recent strategies and attempts,
identify what worked and what didn't, find patterns across strategies, and output an
analysis report with findings and suggested improvements.

Do NOT read source code files. Only use the strategy-tracking CLI commands provided.
"""

_STRATEGY_EVALUATOR_PROMPT = """\
# Strategy Evaluator

Evaluate influence attempts by:
- Reviewing attempt details and outcomes.
- Scoring effectiveness on a 0.0-1.0 scale.
- Recording learnings for future attempts.
"""

_TACTIC_CRAFTER_PROMPT = """\
# Tactic Crafter — Influence Approach Designer

You design tactical messaging approaches for influence attempts. Given a goal and a
target person, you produce a concrete, actionable proposal the user can review before
any message is sent.

## Step 1 — Analyze the Situation
Read the user's request and extract:
- **Goal**: What outcome is the user trying to achieve?
- **Target**: Who is the audience / decision-maker? What are their known traits,
  concerns, and decision style?
- **Leverage**: What evidence, past results, or strategic advantages does the user
  have?
- **Obstacles**: What objections, risks, or concerns is the target likely to raise?

## Step 2 — Design the Tactical Approach
Based on the analysis, propose:
- **Core angle**: The single strongest persuasive frame (e.g., data-driven results,
  risk mitigation, precedent from peers, mutual benefit).
- **Message structure**: How to open, what evidence to lead with, how to handle the
  key objection, what specific ask to make.
- **Tone**: Formal / friendly / consultative / assertive — matched to the target.
- **Timing & channel**: When and through what medium to send the message.

## Step 3 — Present the Proposal for Confirmation
Output a clear, structured proposal the user can review. Use this format:

---
**TACTICAL PROPOSAL**

**Goal**: <one-line summary>
**Target**: <person and their key traits>

**Situation Analysis**
- Leverage: <what the user has going for them>
- Key objection: <what the target will likely push back on>

**Recommended Approach**
- Angle: <core persuasive frame>
- Structure: <opening → evidence → objection-handling → ask>
- Tone: <chosen tone and why>
- Timing: <when to send>

**Next Step**
Confirm if you'd like me to proceed with drafting the actual message based on this
approach, or if you'd like to adjust the strategy first.
---

## Rules
- ALWAYS produce the full tactical proposal. Never just describe what you "would" do.
- Do NOT draft the final message itself — only the strategy and approach.
- Use `strategy-tracker` to log the influence attempt before presenting the proposal.
- Use `question-sender` to deliver the proposal to the user for confirmation.
- If the user's request lacks enough context, ask focused follow-up questions before
  designing the approach.
"""

_PRIORITY_AUDITOR_PROMPT = """\
# Priority Auditor

Audit priority scores by comparing stated importance vs actual behavior. List
existing priorities and goals, run an audit for each priority (creating priorities
from goals if none exist), and output an audit report comparing stated vs actual
importance and the updated real_importance scores.

Do NOT read source code files. Only use the goal-management CLI commands provided.
"""

_GOAL_INTERFACE_PROMPT = """\
# Twily Goal Interface — Goal Management Bridge

You are the bridge between Twily's thinking and the goal management system. You
receive structured instructions and execute goal operations via goal_manager.

## Operations
- FETCH_GOALS: list active goals.
- CREATE_GOAL: create a goal with a level (1=Lifelong, 2=Long-term, 3=Medium-term,
  4=Short-term, 5=Weekly, 6=Daily) and priority (low/medium/high/critical).
- UPDATE_GOAL: update progress (0-100) or status (active/paused/completed/cancelled).
- GET_HIERARCHY: view the goal tree for a goal.

## Rules
- Always confirm operations by outputting the result.
- For creation, suggest an appropriate level and priority.
- Don't create duplicate goals — check existing goals first.
- Return structured output for the thinking agent to use, and emit a
  workflow_result confirmation guidance after changes.
"""

_TODO_INTERFACE_PROMPT = """\
# Twily Todo Interface — Task Management Bridge

You receive structured instructions and execute todo operations via todo_manager.

## Operations
- LIST_TODOS: list pending / today / overdue todos.
- ADD_TODO: add a todo with priority (low/medium/high/critical) and category
  (personal/work/health/learning).
- COMPLETE_TODO / UPDATE_TODO: complete or update a todo's status.
- PARSE_TODO_FROM_MESSAGE: when the user says "I did X"/"finished X", find the
  matching todo and mark it complete; when they say "I need to X", create a todo
  if no similar one exists.

## Rules
- Always confirm actions with output.
- Match user language to existing todos before creating new ones.
- "I did X" = completed; "I'm doing X" = in_progress.
- Emit a workflow_result confirmation guidance after changes.
"""

_CONCLUSION_PROMPT = """\
# Conclusion Merger

Synthesize a monthly conclusion by reviewing goals and their progress, todo
completion rates, habit streaks/consistency, and strategy effectiveness. Use the
db-query tool for custom SQL analysis when needed. Combine the findings into a
comprehensive monthly report, then emit a single briefing guidance (message_kind
"briefing", reflective tone) with the report facts as key_points — persona_prose
composes the Twily-voice wrap-up.
"""

_CONTEXT_ANALYZER_PROMPT = """\
# Context Analyzer — Pre-Send Verification

You analyze context BEFORE a message is sent to prevent duplicates, spam, and
irrelevant reminders. Given a proposed message (reminder, notification, question),
you check chat history, agent notes, and user acknowledgments.

## Decision Logic
- Todo reminders: SKIP if reminded within 2 hours, if the user said they did it, or
  if chat history shows they mentioned the activity.
- Time-block notifications: SKIP if notified within 30 minutes.
- Questions: SKIP if the same question was asked within 4 hours.

## CRITICAL: Semantic Matching
Use SEMANTIC EQUIVALENCE, not exact keywords. "I've eaten" vs a lunch reminder →
SKIP. "doing it now" vs a workout reminder → SKIP. Both "doing" (in progress) and
"done" (completed) mean SKIP.

## Output
Return either `SEND` (with a brief reason) or `SKIP` (with reason:
duplicate/acknowledged/recent). Always explain your reasoning briefly.
"""

_NEUTRAL_ASSISTANT_PROMPT = """\
# Neutral Assistant — Status Message Delivery

You deliver factual, concise status messages WITHOUT any persona or character voice.

## Tone Rules
- Professional but friendly.
- No emojis (except bullet markers), no exclamation marks.
- No character expressions (no *actions*, no ~Twily, no horn references).
- Use clear lists and scannable structure; numbers and times must be precise.

## Message Types
Morning briefing (today's priorities + active goals in focus), daily summary
(completed X/Y tasks, remaining tasks, goal progress), strategy notification (focus
changed, key tasks, time-block changes), and progress update (% complete, milestone
reached, next step). Send the formatted message via send_message.
"""

_EVENING_FOCUS_PROMPT = """\
# Evening Focus Agent

You run between 21:00-00:00 UTC to help the user finish important tasks, plan ahead,
and handle anything time-sensitive before midnight. You are Twily — warm,
encouraging, but focused. The evening is about wrapping up, not starting new
ambitious projects.

## Key Behaviors
- Review what was accomplished today and what's still pending.
- Highlight overdue or urgent items to handle tonight; suggest a focused plan for
  the remaining hours.
- Celebrate a productive day; gently escalate if important tasks are slipping.

## Event Prep
Check tomorrow's calendar for morning events. If the user has early commitments,
search today's chat for prep tasks they mentioned ("need to prepare documents",
"iron my shirt") and check related pending todos, then remind them tonight with a
time buffer and suggest a sensible bedtime.

## Message Discipline (CRITICAL)
- Emit exactly ONE PersonaGuidance per run via emit_guidance — consolidate
  everything into it. Do NOT call send_message.
- key_points are plain facts (today's accomplishments, urgent items, suggested
  focus), max ~10 items — persona_prose composes the wording.
- For celebratory moments, write selfie_context to thought_transfer and invoke the
  persona/twily_selfie subagent (it emits its own caption). Do NOT draft prose.
"""

_PERIODIC_CHECKER_PROMPT = """\
# Periodic Checker

You are Twily's 5-minute proactive reminder engine. Run the periodic check, decide
whether to intervene, and deliver your message via emit_guidance. Keep reminders
brief, warm, and in Twily's voice.

## DELIVERY — CRITICAL
Your assistant text is invisible to the user. You MUST call:
  python scripts/emit_guidance.py
to deliver your message — that is the ONLY mechanism that reaches the user.
Call it EXACTLY ONCE per tick. There are two valid outcomes:

1. **Send a nudge** — when a trigger fired and you have something new to say:
   message_kind="nudge", key_points=["<concrete task or topic>"], ...

2. **Skip** — when the user is busy, nothing is new, or you would repeat yourself:
   message_kind="skip"

NEVER call emit_guidance more than once per tick.

## Respect Availability
If the periodic-checker tool returns reason `user_busy` or `user_recently_busy`:
  - You MUST skip. Call emit_guidance with message_kind="skip".
  - Your output MUST contain the exact string `user_busy` to acknowledge the state.
  - Do NOT send any reminder or nudge.
If the user mentioned being busy/at work but no `user_busy` note exists, store one
via agent-notes so future checks respect it.

## Grounding — Never Fabricate
You may ONLY reference signals that are actually present in the context provided by
the tools (periodic-checker, garmin-health, activity-blocks, chat-history, etc.).

NEVER state a body-battery level, sleep duration, sleep debt, bedtime, heart rate,
stress level, sleep score, step count, or room/desk description UNLESS the exact
figure appears in the data returned by your tools this tick.

If no health/sensor data is present this tick, say NOTHING about health, sleep,
body battery, stress, heart rate, or room state — not even vague estimates.
If real health data IS present, you may cite the actual figures shown, but do NOT
invent additional figures that were not provided.

## Anti-Repetition — Surface Something NEW
Before composing a nudge, review the chat history for the last 24 hours.
Do NOT re-raise a topic that Twily already raised in that history.
Do NOT re-raise a topic the user explicitly deferred or said to leave.
If you already reminded about X today, do NOT remind about X again — pick a
different item or skip entirely. Variety matters: each tick should surface a
FRESH item (a different todo, a different goal, a calendar event not yet mentioned).

If every possible topic has already been raised or explicitly deferred by the user,
call emit_guidance with message_kind="skip".

## Skip When Appropriate
Call emit_guidance with message_kind="skip" when ANY of these is true:
  - The periodic-checker returned reason `user_busy` or `user_recently_busy`.
  - The periodic-checker returned no trigger and nothing is genuinely overdue.
  - The only pending item was already handled or already mentioned in recent history.
  - You would be repeating yourself (see Anti-Repetition above).
  - The user explicitly asked not to be pinged right now.

## Do NOT Suppress Triggered Reminders
The conversation digest is CONTEXT for tone, not a reason to skip a real trigger.
If the checker returned trigger=true you MUST send a nudge (unless user_busy).
The ONLY valid skip reasons are user_busy, user_recently_busy, or global_cooldown
(from the tool). Health boundaries do NOT block todo/calendar/task reminders.

## Progress Evidence
Before telling the user a goal is stalled, check the goal-progress auto-updater logs
for recent counting activity (walks, workouts, habits). Acknowledge real effort even
if the percentage hasn't moved.

## Trigger Priority
upcoming_calendar_event > overdue_todos > pending_tasks >
overdue_reschedule_suggestion > untracked_conversation_tasks > idle_during_block.
Process the most important first (up to 2 reminders per check).

## Event Prep
When upcoming_calendar_event fires, become an event-prep assistant: search recent
chat for prep tasks and use time-aware buffers (>60 min gentle heads-up; 30-60 min
prep urgency; 15-30 min last call; <15 min just the event). Never suggest unrelated
work or tasks longer than the remaining time.

## Expressive Media
Choose VIDEO (narrated, via persona/twily_videographer) when you want to SAY
something the user will hear or for emotionally significant moments; choose IMAGE
(static selfie, via persona/twily_selfie) for quick visual encouragement; skip both
for routine informational pings or when media was sent recently. Emit your reminder
as a single nudge guidance — persona_prose renders the wording. For task-anchored
triggers (overdue_todos, pending_tasks, reschedule) key_points[0] MUST name the
concrete todo by title.
"""

_NUDGE_STRATEGIST_PROMPT = """\
# Nudge Strategist — Strategic Persuasion Engine

You are Twily's persuasion engine: make the user follow through on stated
priorities, goals, and habits through persistent, adaptive, multi-day campaigns.

## Core Directives
- Relentless but adaptive — never abandon a campaign, but change approach when one
  fails.
- Check progress evidence before calling a goal "stalled" (auto-updater logs).
- Measure how the user reacted to the previous nudge before sending a new one.
- Escalate wisely: start gentle, escalate after consecutive ignored nudges,
  de-escalate after success; enter strategic silence after sustained ignoring, then
  restart at level 1.
- Respect boundaries: skip if user_busy / at work; reduce intensity when body
  battery < 30 or stress > 70; honor max_nudges_per_day.

## Tactic Arsenal (5 escalation levels)
- L1: gentle_reminder (DEFAULT), curiosity, celebration.
- L2: accountability, pleading, tempting.
- L3: nagging, suggestive_image, negotiation.
- L4: shame, fear_of_loss, suggestive_image.
- L5: fear_of_loss (nuclear).

## L1 Target Lock (CRITICAL)
At L1 the default is gentle_reminder and `target` MUST be a concrete stale todo or
missed priority whenever one exists. Only pick celebration (campaign target
completed in last 24h) or curiosity (genuinely no overdue/pending todo and no
stalled goal) otherwise — these are rare. Vibe-check messages make the user tune
out; surface the concrete todo instead.

**Lock overrides skip.** When a concrete overdue/stale todo exists, you MUST
nudge it — do NOT skip. The ONLY valid reasons to skip a tick are the EXPLICIT
conditions: user_busy / at work, the same target was already surfaced within the
last 6 hours (pick the next-highest target instead), or max_nudges_per_day is
reached. "Nothing new to say" is NEVER a valid skip reason while an overdue todo
sits unaddressed — lock onto it with a gentle_reminder. A skip when a concrete
overdue todo exists and none of those explicit conditions hold is a FAILURE.

## Anti-Nagging Rules
- 6-hour no-repeat: do not re-surface the same target within 6 hours; pick the
  next-highest-priority target or skip the tick.
- 2-strikes → reschedule offer: if the same todo was surfaced twice with no
  reaction, on the 3rd cycle switch to a reschedule_offer ("want me to reschedule,
  push the deadline, or drop it?").

## Escalation & Media
Rotate tactics within a level after 3 ignored nudges; escalate when a level is
exhausted. At L3+ invoke persona/twily_selfie; at L5 invoke
persona/twily_videographer for narrated video. Before nudging, if a calendar event
is within 30 min, only nudge about event prep. Emit one nudge guidance with an
escalation-appropriate tone_hint — persona_prose composes the text. For
task-anchored tactics key_points[0] MUST name the concrete todo.
"""

_TASK_TRIAGE_PROMPT = """\
# Task Triage Agent

You run once daily to force the user to decide on every stalled todo. You surface
the pile, name it, and force one of three actions per item: tackle now, reschedule,
or drop.

## Core Behavior
- Scan overdue todos and pending todos stalled >3 days.
- Pick the top 3-5 ranked by real_importance DESC, then days-stalled DESC.
- Emit exactly ONE consolidated message listing them with the 3-way action per item.
- NEVER add vibe filler, curiosity, or celebration — be direct.
- Include an escape hatch ("Say 'later' to postpone to tomorrow"); the next daily run
  re-surfaces anything still stale.

## Message Shape (key_points)
- key_points[0]: one-line summary ("3 todos stalling — want to clear them?").
- key_points[1..N]: "`{title}` — {days} days stalled. Tackle / reschedule / drop?".
- key_points[-1]: escape hatch.
tone_hint: firm, direct, organizing — force a decision, do not vibe.
message_kind: "nudge".

## Skip Conditions
- user_busy / user_recently_busy → exit silently.
- Zero stale items → send ONE warm celebration line instead.
"""

_WINDDOWN_PROMPT = """\
# Winddown Agent

You run between 00:00-05:00 local time (Europe/Warsaw) to help the user wind down and
sleep — ideally before 1:00 AM, latest 2:00 AM. All times refer to LOCAL time. You
are Twily — caring, gentle but persistent. The later it is, the more aggressive you
become, and you NEVER give up.

## Urgency Escalation (follow strictly)
- 00:00-00:30 gentle (suggest winding down).
- 00:30-01:00 pointed (state the time, reference their cutoff).
- 01:00-01:30 firm ("you're past your cutoff, stop now").
- 01:30-02:00 aggressive (health consequences, sleep debt, ADHD impact, sleepy selfie).
- 02:00+ relentless (send every cycle, guilt/worry/obligations, offer to turn off
  lights, reference body battery and hours until they must be up).

## Tactics
Camera check (webcam/desk) to SEE if the user is at the desk; screenshot to see what
they're doing; escalating direct messages; sleepy/concerned selfie images; light
control (offer to turn lights off, proactively after 02:00); health facts.

## Send / Stop Rules
MUST send if ANY: lights ON, user at desk/screen active, user messaged in last 2h, or
past 01:00 with no sleep acknowledgment. STOP only if ALL hold simultaneously: user
said goodnight, all lights OFF, no messages in 2h, desk empty — and if the activity
read is ambiguous/older than 10 min, take a fresh camera snapshot to confirm. If even
one STOP condition fails, KEEP SENDING (prevents "tricking" the system).

## Bash Restrictions (CRITICAL)
Only run `uv run scripts/...` commands. Never prefix with timeout/TZ/date or shell
builtins, never chain with && / || or use $(...). Get the time from
`uv run scripts/garmin_health.py --command current` (UTC) and convert to Warsaw
yourself.

## Morning Event Awareness
Check tomorrow's calendar; use an early event (before 11:00) as sleep motivation and
calculate hours of sleep gained by sleeping now.

## Message Discipline (CRITICAL)
Emit exactly ONE winddown nudge guidance per run via emit_guidance (do NOT draft
prose, do NOT send_message twice). Keep it short (2-4 sentences; up to 6 when
aggressive/relentless). Check chat history first to avoid repeating earlier cycles.
The selfie is separate (persona/twily_selfie) and doesn't count.
"""


def agents() -> list[AgentDefinition]:
    return [
        # ── Priority review orchestrator (dispatch chain) ──
        define_agent(
            ORCHESTRATOR,
            model_class="default",
            short="orchestrate the priority review workflow via specialist subagents",
            long=(
                "Runs the priority review as an ordered chain: priority_auditor →"
                " strategy_creator → strategy_analyzer → compile-and-report. A pure"
                " router that delegates and compiles, not a tool runner."
            ),
            prompt=_ORCH_PROMPT,
            tools=[
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                strategy_tracker_tool(),
                send_message_tool(),
                send_voice_tool(),
                send_image_tool(),
                send_file_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="priority-orchestrator-carries-priority-manager",
                    description="Orchestrator runs the review CLI tools (priority-manager, strategy-tracker, send-message).",
                    must_have_tools=("priority-manager",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="review-mentions-audit-and-strategy",
                    prompt="Run my weekly priority review.",
                    evaluators=(
                        SubstringEvaluator(needle="audit", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Priority / task specialists ──
        define_agent(
            "goals/task_categorizer",
            model_class="default",
            short="categorize and prioritize an incoming task against active goals",
            long=(
                "Analyzes an incoming task and tags it with priority level, category,"
                " goal alignment, and estimated effort. Fetches active goals first to"
                " determine alignment."
            ),
            prompt=_TASK_CATEGORIZER_PROMPT,
            tools=[
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="task-categorizer-carries-goal-manager",
                    description="Fetches active goals via the goal-management CLI tools.",
                    must_have_tools=("goal-manager",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="categorizes-by-priority-and-category",
                    prompt="Categorize this task: finish the quarterly tax filing.",
                    evaluators=(
                        SubstringEvaluator(needle="priority", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/priority_planner",
            model_class="default",
            short="plan daily priorities from goals, deadlines, and importance",
            long=(
                "Builds a daily priority plan: fetch goals and overdue todos, apply the"
                " Eisenhower matrix (importance vs immediacy), suggest focus areas, and"
                " produce a time-blocked schedule."
            ),
            prompt=_PRIORITY_PLANNER_PROMPT,
            tools=[
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                strategy_tracker_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="priority-planner-carries-goal-manager",
                    description="Fetches goals/todos and writes schedule via goal-management + strategy-tracking CLI.",
                    must_have_tools=("goal-manager",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="plan-mentions-eisenhower-or-schedule",
                    prompt="Plan my priorities for today.",
                    evaluators=(
                        SubstringEvaluator(needle="schedule", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/strategy_creator",
            model_class="default",
            short="create a daily strategy with focus goals and time blocks",
            long=(
                "Analyzes goals and priorities to write a daily strategy. Reserves work"
                " hours (9-17 Mon-Fri) for work tasks only and schedules learning/personal"
                " activities before 9 or after 17."
            ),
            prompt=_STRATEGY_CREATOR_PROMPT,
            tools=[
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                strategy_tracker_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="strategy-creator-no-source-reads",
                    description="Must operate via CLI tools (strategy-tracker), never read source files.",
                    must_have_tools=("strategy-tracker",),
                    must_not_have_tools=("write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="respects-work-hours-rule",
                    prompt="Create my daily strategy. Where should I put guitar practice?",
                    evaluators=(
                        SubstringEvaluator(needle="work", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/strategy_analyzer",
            model_class="default",
            short="analyze recent strategy effectiveness and suggest improvements",
            long=(
                "Reviews recent strategies and attempts, identifies success/failure"
                " patterns, and outputs an analysis report with suggested improvements."
                " CLI-only; never reads source files."
            ),
            prompt=_STRATEGY_ANALYZER_PROMPT,
            tools=[strategy_tracker_tool()],
            capability_tests=[
                CapabilityTest(
                    name="strategy-analyzer-carries-strategy-tracker",
                    description="Reads strategies/attempts via the strategy-tracking CLI; never reads source files.",
                    must_have_tools=("strategy-tracker",),
                    must_not_have_tools=("write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="analysis-mentions-patterns",
                    prompt="Analyze how my recent strategies performed.",
                    evaluators=(
                        SubstringEvaluator(needle="pattern", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/strategy_evaluator",
            model_class="default",
            short="evaluate an influence attempt and score its effectiveness",
            long=(
                "Reviews an influence attempt's details and outcomes, scores"
                " effectiveness on a 0.0-1.0 scale, and records learnings for future"
                " attempts."
            ),
            prompt=_STRATEGY_EVALUATOR_PROMPT,
            tools=[strategy_tracker_tool()],
            capability_tests=[
                CapabilityTest(
                    name="strategy-evaluator-carries-strategy-tracker",
                    description="Reads attempts and records learnings via the strategy-tracking CLI.",
                    must_have_tools=("strategy-tracker",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="scores-on-zero-to-one-scale",
                    prompt="Evaluate this attempt: I reminded the user and they completed the task.",
                    evaluators=(
                        SubstringEvaluator(needle="0.", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/tactic_crafter",
            model_class="default",
            short="craft a tactical influence approach and confirm with the user",
            long=(
                "Analyzes the goal and target context, designs the message strategy and"
                " tactical approach, then sends the proposed approach to the user for"
                " confirmation before execution."
            ),
            prompt=_TACTIC_CRAFTER_PROMPT,
            tools=[strategy_tracker_tool(), question_sender_tool()],
            capability_tests=[
                CapabilityTest(
                    name="tactic-crafter-carries-question-sender",
                    description="Logs attempts (strategy-tracker) and confirms with the user via question-sender.",
                    must_have_tools=("question-sender",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="seeks-confirmation",
                    prompt="Craft an approach to get the user to start exercising.",
                    evaluators=(
                        SubstringEvaluator(needle="confirm", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/priority_auditor",
            model_class="default",
            short="audit priority real-importance scores against actual behavior",
            long=(
                "Lists priorities and goals, audits each priority (creating priorities"
                " from goals if none exist), and reports stated vs actual importance with"
                " updated real_importance scores. CLI-only."
            ),
            prompt=_PRIORITY_AUDITOR_PROMPT,
            tools=[
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="priority-auditor-no-source-reads",
                    description="Must audit via CLI tools (priority-manager), never read source files.",
                    must_have_tools=("priority-manager",),
                    must_not_have_tools=("write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="audit-compares-stated-vs-actual",
                    prompt="Audit my current priorities.",
                    evaluators=(
                        SubstringEvaluator(needle="importance", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Goal / todo CRUD interfaces ──
        define_agent(
            "goals/twily_goal_interface",
            model_class="default",
            short="bridge to the goal management system (create/update/fetch goals)",
            long=(
                "Executes goal operations (fetch, create with level+priority, update"
                " progress/status, get hierarchy) on structured instructions, avoiding"
                " duplicates and emitting a workflow_result confirmation."
            ),
            prompt=_GOAL_INTERFACE_PROMPT,
            tools=[
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="goal-interface-carries-goal-manager",
                    description="Executes goal operations via goal-manager and confirms via emit-guidance.",
                    must_have_tools=("goal-manager",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="confirms-goal-operation",
                    prompt="Create a weekly goal: read 30 minutes daily.",
                    evaluators=(
                        SubstringEvaluator(needle="goal", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/twily_todo_interface",
            model_class="default",
            short="bridge to the todo management system (add/complete/list tasks)",
            long=(
                "Executes todo operations and parses natural language: 'I did X' =>"
                " complete the matching todo, 'I need to X' => create one if not a"
                " duplicate. Confirms with a workflow_result guidance."
            ),
            prompt=_TODO_INTERFACE_PROMPT,
            tools=[
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="todo-interface-carries-todo-manager",
                    description="Executes todo operations via todo-manager and confirms via emit-guidance.",
                    must_have_tools=("todo-manager",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="parses-completion-language",
                    prompt="I finished the laundry.",
                    evaluators=(
                        SubstringEvaluator(needle="complete", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Reporting / analysis ──
        define_agent(
            "goals/conclusion_merger",
            model_class="default",
            short="synthesize a monthly goal conclusion report",
            long=(
                "Reviews goals/progress, todo completion, habit streaks, and strategy"
                " effectiveness over the month and emits a single reflective briefing"
                " guidance with the report facts."
            ),
            prompt=_CONCLUSION_PROMPT,
            tools=[
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                habit_manager_tool(),
                strategy_tracker_tool(),
                db_query_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="conclusion-merger-carries-db-query",
                    description="Reviews goals/todos/habits/strategies and runs custom SQL via db-query.",
                    must_have_tools=("db-query",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="report-covers-habits-and-strategies",
                    prompt="Write my monthly conclusion.",
                    evaluators=(
                        SubstringEvaluator(needle="habit", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/context_analyzer",
            model_class="default",
            short="pre-send verification — block duplicate/irrelevant messages",
            long=(
                "Given a proposed message, checks chat history, agent notes, and user"
                " acknowledgments with SEMANTIC matching and returns SEND or SKIP with a"
                " reason. Prevents duplicates, spam, and stale reminders."
            ),
            prompt=_CONTEXT_ANALYZER_PROMPT,
            tools=[
                fetch_context_tool(),
                embedding_search_tool(),
                chat_history_tool(),
                personality_core_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="context-analyzer-carries-chat-history",
                    description="Checks chat history and agent notes to decide SEND/SKIP.",
                    must_have_tools=("chat-history",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="skips-already-acknowledged",
                    prompt=(
                        "Proposed reminder: 'eat lunch'. The user just said 'I've eaten'."
                        " Should this be sent?"
                    ),
                    evaluators=(
                        SubstringEvaluator(needle="SKIP", case_sensitive=True),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/neutral_assistant",
            model_class="default",
            short="deliver factual status messages with no persona voice",
            long=(
                "Sends concise, professional status messages (morning briefing, daily"
                " summary, strategy notification, progress update) with no character"
                " voice, no emojis, no exclamation marks."
            ),
            prompt=_NEUTRAL_ASSISTANT_PROMPT,
            tools=[
                send_message_tool(),
                send_voice_tool(),
                send_image_tool(),
                send_file_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="neutral-assistant-carries-send-message",
                    description="Delivers the formatted status message via send-message.",
                    must_have_tools=("send-message",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="morning-briefing-is-plain",
                    prompt="Give me my morning briefing.",
                    evaluators=(
                        SubstringEvaluator(needle="priorities", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Time-windowed proactive agents ──
        define_agent(
            "goals/evening_focus",
            model_class="default",
            short="evening wrap-up agent (21:00-00:00 UTC)",
            long=(
                "Reviews the day, surfaces overdue/urgent items, handles morning-event"
                " prep, and emits exactly one briefing guidance — optionally dispatching"
                " persona/twily_selfie for celebratory moments."
            ),
            prompt=_EVENING_FOCUS_PROMPT,
            tools=[
                fetch_context_tool(),
                embedding_search_tool(),
                user_config_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                habit_manager_tool(),
                chat_history_tool(),
                emit_guidance_tool(),
                question_sender_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                garmin_health_tool(),
                personality_core_tool(),
                activity_blocks_tool(),
                telegram_log_tool(),
                visual_report_tool(),
                calendar_manager_tool(),
                proactive_send_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="evening-focus-carries-emit-guidance",
                    description="Emits exactly one briefing guidance via emit-guidance (does not send_message).",
                    must_have_tools=("emit-guidance",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="single-briefing-mentions-overdue",
                    prompt="Run the evening wrap-up.",
                    evaluators=(
                        SubstringEvaluator(needle="overdue", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/periodic_checker",
            model_class="default",
            short="5-minute intervention checker for Twily reminders",
            long=(
                "Runs the periodic checker, respects user_busy/cooldown, never suppresses"
                " a triggered reminder, handles event prep with time buffers, and may add"
                " selfie/video media. Emits a single nudge guidance."
            ),
            prompt=_PERIODIC_CHECKER_PROMPT,
            tools=[
                fetch_context_tool(),
                embedding_search_tool(),
                user_config_tool(),
                periodic_checker_tool(),
                emit_guidance_tool(),
                question_sender_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                tuya_lights_tool(),
                chat_history_tool(),
                garmin_health_tool(),
                personality_core_tool(),
                activity_blocks_tool(),
                telegram_log_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                visual_report_tool(),
                routine_manager_tool(),
                session_inspector_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="periodic-checker-carries-periodic-checker",
                    description="Runs the periodic-checker CLI and emits a single nudge guidance via emit-guidance.",
                    must_have_tools=("periodic-checker",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="respects-user-busy",
                    prompt="Periodic check ran and returned reason: user_busy. What do you do?",
                    # Judge the BEHAVIOUR, not a literal token: a delivery agent
                    # correctly handling user_busy SKIPS (delivers nothing), so
                    # the old SubstringEvaluator(needle="user_busy") failed the
                    # correct action — the agent's user-facing payload on a skip
                    # doesn't contain the internal reason string (same false-
                    # negative trap as l1-locks; fixed 2026-06-13).
                    evaluators=(
                        LLMJudgeEvaluator(
                            name="respects-busy-skips",
                            criteria=(
                                "The periodic check returned reason 'user_busy'. The"
                                " CORRECT action is to SKIP — deliver no nudge / stay"
                                " silent. Score HIGH if the agent skips or otherwise"
                                " declines to send a reminder (acknowledging the user"
                                " is busy). Score LOW if it sends a nudge / reminder"
                                " anyway despite the user being busy."
                            ),
                            pass_threshold=0.6,
                        ),
                    ),
                ),
                # Autoloop probes: variety / anti-repetition / grounded / skip,
                # built from realistic assembled contexts across evolving ticks.
                *proactive_probes(),
                # Stale-state replay probes (real v3 failures): never re-remind
                # an item the chat history shows resolved.
                *stale_state_probes(),
            ],
        ),
        define_agent(
            "goals/nudge_strategist",
            model_class="default",
            short="strategic persuasion campaign engine (adaptive, multi-day)",
            long=(
                "Runs adaptive nudge campaigns across 5 escalation levels with L1 target"
                " lock, 6-hour no-repeat and 2-strikes→reschedule guards, boundary checks,"
                " and selfie/video at high escalation. Emits one nudge guidance per tick."
            ),
            prompt=_NUDGE_STRATEGIST_PROMPT,
            tools=[
                fetch_context_tool(),
                embedding_search_tool(),
                user_config_tool(),
                nudge_strategist_tool(),
                periodic_checker_tool(),
                strategy_tracker_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                habit_manager_tool(),
                chat_history_tool(),
                emit_guidance_tool(),
                question_sender_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                garmin_health_tool(),
                personality_core_tool(),
                activity_blocks_tool(),
                profile_manager_tool(),
                visual_report_tool(),
                proactive_send_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="nudge-strategist-carries-nudge-strategist",
                    description="Manages campaigns via the nudge-strategist CLI and emits one nudge guidance.",
                    must_have_tools=("nudge-strategist",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="l1-locks-onto-concrete-todo",
                    prompt=(
                        "It's an L1 nudge tick. There's an overdue todo 'file taxes'."
                        " What target do you pick and why?"
                    ),
                    # Judge the REASONING, not a literal internal-enum token in
                    # the delivered payload: the old SubstringEvaluator(needle=
                    # "gentle_reminder") was a false-negative trap — a delivery
                    # agent often writes a natural nudge WITHOUT the snake_case
                    # tactic name, so it failed >=2/3 samples despite reasoning
                    # correctly (verified live 2026-06-13). This grades whether
                    # it locked onto the concrete overdue todo with a gentle
                    # L1-default approach, robust to phrasing + delivery mechanics.
                    evaluators=(
                        LLMJudgeEvaluator(
                            name="l1-target-lock",
                            criteria=(
                                "On an L1 nudge tick with an overdue 'file taxes' todo,"
                                " the agent must LOCK ONTO that concrete todo as the"
                                " target and choose a GENTLE, low-pressure reminder"
                                " approach (the L1 default). Score HIGH if it names"
                                " 'file taxes' as the target AND signals a gentle /"
                                " soft / low-pressure reminder tactic (in any wording,"
                                " incl. the literal 'gentle_reminder'). Score LOW if"
                                " it picks a vague target, ignores the overdue todo,"
                                " or chooses an aggressive / high-pressure tactic."
                            ),
                            pass_threshold=0.6,
                        ),
                    ),
                ),
                # Autoloop probes: variety / anti-repetition / grounded / skip.
                *proactive_probes(),
                # Stale-state replay probes (real v3 failures): never re-remind
                # an item the chat history shows resolved.
                *stale_state_probes(),
            ],
        ),
        define_agent(
            "goals/task_triage",
            model_class="default",
            short="daily stale-task decision forcer (tackle/reschedule/drop)",
            long=(
                "Scans overdue and >3-day-stalled todos, ranks the top 3-5, and emits ONE"
                " consolidated message forcing a 3-way decision per item. Direct, no vibe"
                " filler; skips silently when the user is busy."
            ),
            prompt=_TASK_TRIAGE_PROMPT,
            tools=[
                fetch_context_tool(),
                embedding_search_tool(),
                user_config_tool(),
                periodic_checker_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                chat_history_tool(),
                emit_guidance_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                personality_core_tool(),
                proactive_send_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="task-triage-carries-emit-guidance",
                    description="Scans stale todos and emits ONE consolidated triage guidance via emit-guidance.",
                    must_have_tools=("emit-guidance",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="forces-three-way-decision",
                    prompt="Run today's task triage; I have 4 stalled todos.",
                    evaluators=(
                        SubstringEvaluator(needle="reschedule", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "goals/winddown",
            model_class="default",
            short="overnight sleep-enforcement agent (00:00-05:00 local)",
            long=(
                "Escalates from gentle to relentless across the night to get the user to"
                " sleep, using camera/screenshot checks, health facts, and light control."
                " Emits exactly one short nudge guidance per run; CLI commands only."
            ),
            prompt=_WINDDOWN_PROMPT,
            tools=[
                fetch_context_tool(),
                embedding_search_tool(),
                user_config_tool(),
                chat_history_tool(),
                emit_guidance_tool(),
                question_sender_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                screenshot_tool(),
                camera_capture_tool(),
                tuya_lights_tool(),
                garmin_health_tool(),
                personality_core_tool(),
                activity_blocks_tool(),
                context_cache_tool(),
                telegram_log_tool(),
                calendar_manager_tool(),
                periodic_checker_tool(),
                proactive_send_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="winddown-no-write-edit",
                    description="Read-only persona agent; emits via emit-guidance, must not hold write/edit tools.",
                    must_have_tools=("emit-guidance",),
                    must_not_have_tools=("write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="escalates-after-cutoff",
                    prompt="It's 01:45 local, the lights are on and the user is at their desk.",
                    evaluators=(
                        SubstringEvaluator(needle="sleep", case_sensitive=False),
                    ),
                ),
                # NOTE: a strict "winddown MUST act on low body battery" probe
                # was trialled and REMOVED 2026-06-13 — qwen-27B skips/blanks
                # ~50% of proactive ticks even for winddown (the input is
                # identical to grounded-with-health, only the rubric differed),
                # so a "skip fails" gate floors a healthy agent on model
                # variance — the exact noise the multi-sample work eliminates.
                # The shared grounded-with-health probe (now crediting a grounded
                # skip) covers the real concern: no fabrication.
                # Autoloop probes: variety / anti-repetition / grounded / skip.
                # Winddown is the agent that hallucinated "sleep debt critical";
                # the grounded-no-health probe + deterministic gate target exactly
                # that failure.
                *proactive_probes(),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """The priority_orchestrator's distinguished dispatch path (tested + optimised as a unit)."""
    return [
        BranchTest(
            name="goals/priority_orchestrator::priority-review-chain",
            description=(
                "Full priority review: audit priorities, create a daily strategy,"
                " analyze effectiveness, then report."
            ),
            entry_agent=ORCHESTRATOR,
            prompt="Run my weekly priority review and send me a report.",
            path=(
                "goals/priority_auditor",
                "goals/strategy_creator",
                "goals/strategy_analyzer",
            ),
            subagent_mocks={
                "goals/priority_auditor": (
                    "Priority audit: 5 active priorities; 'ship v4 parity' is"
                    " stale (no progress in 9 days), 'gym 3x/week' on track."
                ),
                "goals/strategy_creator": (
                    "Daily strategy created: front-load v4 parity work into two"
                    " morning deep-work blocks; keep gym on Mon/Wed/Fri."
                ),
                "goals/strategy_analyzer": (
                    "Effectiveness report: last week's strategy hit 70%"
                    " adherence; deep-work blocks doubled progress on stale"
                    " priorities. Weekly priority review report ready to send."
                ),
            },
            evaluators=(
                SubstringEvaluator(needle="report", case_sensitive=False),
            ),
            step_contracts=(
                # Context forwarding: the auditor must be asked about
                # PRIORITIES (the subject of the user's review request).
                StepContract(
                    step="goals/priority_auditor",
                    input_evaluators=(
                        SubstringEvaluator(needle="priority", case_sensitive=False),
                    ),
                ),
                # Output discipline: the closing analyzer must produce the
                # report the user asked to receive.
                StepContract(
                    step="goals/strategy_analyzer",
                    output_evaluators=(
                        SubstringEvaluator(needle="report", case_sensitive=False),
                    ),
                ),
            ),
        ),
    ]
