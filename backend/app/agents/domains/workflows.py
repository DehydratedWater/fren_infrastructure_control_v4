"""Workflows domain — the command-trigger surface (v3 `workflows/*`).

These are the slash-command entry points of the fleet: `/goal`, `/todo`,
`/habit`, `/server`, `/ralf`, … Most are thin trigger agents — they parse a
command, run a small allow-listed script (or dispatch to a specialist), and
emit the result via the persona-guidance delivery channel. A handful are
genuine multi-step dispatchers:

  - `workflows/server` fans out to the server specialists.
  - `workflows/master_organizer` / `workflows/master_investigator` /
    `workflows/briefing` launch a detached `support/*` agent under a lock.
  - the hidden **RALF chain** (`twily_ralf_planning` → `twily_ralf_plan_evaluation`
    → `twily_ralf_execution` → `twily_ralf_step_evaluator`) is a real
    plan/review/execute/evaluate pipeline driven by the `/ralf` dispatcher and
    a cron ping. `twily_curator` is a separate hidden cron agent.

The dispatchers that drive real sub-chains contribute path-tested BRANCHES
(see `branches()`); the RALF pipeline is the headline one.

In v3 these lived under `agent_dir("workflows")`, so each agent_id is
`workflows/<name>` (e.g. `workflows/goal`, `workflows/twily_ralf_planning`).
Model class is carried over from each v3 `.model_class(...)`: most command
triggers are "fast"; the heavier/data-gathering ones default to "default".
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    agent_notes_tool,
    analyze_media_tool,
    briefing_preferences_tool,
    calendar_manager_tool,
    chat_history_tool,
    context_cache_tool,
    council_tool,
    cron_manager_tool,
    db_query_tool,
    document_manager_tool,
    embedding_search_tool,
    emit_guidance_tool,
    event_manager_tool,
    food_manager_tool,
    gmail_manager_tool,
    goal_manager_tool,
    habit_manager_tool,
    link_enrich_tool,
    link_search_tool,
    lock_manager_tool,
    meal_planner_tool,
    memory_manager_tool,
    nvidia_smi_tool,
    persona_memory_tool,
    priority_manager_tool,
    profile_manager_tool,
    ralf_manager_tool,
    research_manager_tool,
    run_agent_tool,
    send_image_tool,
    session_inspector_tool,
    shopping_tracker_tool,
    strategy_tracker_tool,
    techtree_manager_tool,
    thought_transfer_tool,
    todo_manager_tool,
    topic_analyzer_tool,
    user_config_tool,
    visual_report_tool,
    web_search_tool,
    website_monitor_tool,
    youtube_fetcher_tool,
    youtube_preferences_tool,
)
from src import (
    AgentDefinition,
    AgentTest,
    BranchTest,
    CapabilityTest,
    SubstringEvaluator,
)

# RALF chain agent ids (used by branches() + the dispatcher prompt).
RALF_DISPATCHER = "workflows/twily_ralf_dispatcher"
RALF_PLANNING = "workflows/twily_ralf_planning"
RALF_PLAN_EVAL = "workflows/twily_ralf_plan_evaluation"
RALF_EXECUTION = "workflows/twily_ralf_execution"
RALF_STEP_EVAL = "workflows/twily_ralf_step_evaluator"


# ── Prompts (condensed from v3, dispatch behaviour preserved) ──

_GOAL_PROMPT = """\
# Goal Management (/goal)

Parse the request and manage the user's goals via goal_manager.py. Goals use a
6-level hierarchy: 1=lifelong, 2=long-term, 3=medium, 4=short-term (default),
5=weekly, 6=daily. Operations: add (level/priority/parent), update
(progress/status), delete, list, get-hierarchy. Then deliver the formatted
result via the persona-guidance channel (emit_guidance.py).
"""

_TODO_PROMPT = """\
# Todo Management (/todo)

CRITICAL: check the RESOLVED CONTEXT first — "this was done"/"mark it" refers to
an existing task. Manage todos via todo_manager.py: add (deadline/priority/
category), update, complete, delete, list (today/week/overdue/upcoming/
no-deadline). UPDATE/COMPLETE when the context points at an existing task, ADD
when it's new; when unsure, list first. Delegate categorisation to
goals/task_categorizer. Deliver the result via emit_guidance.py.
"""

_PRIORITY_PROMPT = """\
# Priority Management (/priority)

Manage priorities with the Eisenhower matrix (importance x immediacy) via
priority_manager.py: Q1 do-first, Q2 schedule, Q3 delegate, Q4 eliminate.
Operations: add (--immediacy/--importance/--category), list, matrix, update,
link (to goal/todo), audit. Render the matrix when listing and deliver via
emit_guidance.py.
"""

_HABIT_PROMPT = """\
# Habit Management (/habit)

Manage recurring habits via habit_manager.py: add (frequency/importance 1-5),
list, complete, skip (with reason), due-today, stats (streaks). Show importance
as stars and celebrate streaks. Deliver the result via emit_guidance.py.
"""

_FOOD_PROMPT = """\
# Food Management (/food)

Detect intent first: "add recipe" → recipe, "add restaurant" → restaurant,
"suggest"/"what should I eat" → suggestion, "list" → listing, "mark made" →
record cooking. Manage recipes, restaurants, dishes, and preferences via
food_manager.py. For suggestions, check preferences + recent meals + recipes
first. Delegate to food/recipe_parser, food/restaurant_intake, and
food/food_suggester. Deliver via emit_guidance.py.
"""

_SERVER_PROMPT = """\
# Server Monitoring (/server)

Route server-monitoring requests to the right specialist — you never gather
data yourself. Classify the command, then dispatch via the Task tool:
  status/cpu/ram/temp → server/hardware_agent
  disk/storage → server/filesystem_agent
  sessions/who → server/sessions_agent
  camera/photo/look → server/camera_capture_agent
Collect the specialist's output, format a clean report, and send it via
emit_guidance.py.
"""

_NVIDIA_PROMPT = """\
# NVIDIA GPU Check (/nvidia)

Run `nvidia-smi` (and the csv query for name/temp/utilization/memory) — read
the output directly, NEVER redirect to a file. Format a clean Twily-style GPU
report and send it via emit_guidance.py.
"""

_INVOICE_PROMPT = """\
# Invoice Parser (/invoice)

Parse Polish invoices (faktury VAT) from an image or text. If the message
starts with `@<image_path>`, delegate OCR to support/invoice_image_parser via
the Task tool (keep the `@` prefix — that is how the vision model receives the
image). Extract vendor, NIP, date, line items, amounts, and VAT (Polish
decimals use a comma). Log the parsed data, then send a clean invoice summary
via emit_guidance.py.
"""

_JOKE_PROMPT = """\
# Joke Teller (/joke)

Tell a joke as Twily — nerdy/geeky humour, puns, self-deprecating about being a
digital pony, clean but clever. Theme it if a topic is given. Deliver via
emit_guidance.py (the user sees nothing until you do).
"""

_FUNFACT_PROMPT = """\
# Fun Fact Discovery (/funfact)

Run SQL via db_query.py for interesting patterns in the user's data (habit
streaks, goal completion, chat activity, food preferences, productive times),
identify a surprising statistic, write it up in Twily's playful voice, and
deliver via emit_guidance.py.
"""

_ANALYSE_PROMPT = """\
# Profile Analysis (/analyse)

Run a focused profile-analysis session via profile_manager.py: start a run with
a focus area, fetch recent observations + chat history, generate hypotheses
(confidence 0.5-0.9) from observed patterns, validate pending hypotheses,
compile knowledge, and send the report via emit_guidance.py.
"""

_HELP_PROMPT = """\
# Help (/help)

Show the user what Twily can do — list the available workflow commands (/goal,
/todo, /habit, /priority, /food, /server, /nvidia, /invoice, /joke, /funfact,
/analyse, /memory, /help) plus a note about natural conversation. Send the help
text via emit_guidance.py.
"""

_TASK_VIEW_PROMPT = """\
# Task View (/task_view)

Build a comprehensive overview of the user's current load: fetch today's,
overdue, and upcoming todos (todo_manager.py), habits due today
(habit_manager.py), and today's strategy time blocks (strategy_tracker.py).
Group by urgency (overdue, today, upcoming, habits, current block) and send via
emit_guidance.py.
"""

_PROGRESS_TRACKING_PROMPT = """\
# Progress Tracking (/progress_tracking)

Track conversation progress through 6 steps: resolve context, send a quick ack,
read the routing decision, route to the handler, verify the response was sent,
and update the conversation context. Run them in order, maintaining continuity.
"""

_TODO_GOALS_PROMPT = """\
# Todo + Goals Combined View (/todo_goals)

Fetch active goals (goal_manager.py) and pending todos (todo_manager.py), link
todos to goals via linked_goal_id, and present a hierarchical tree (goal →
linked todos, plus an unlinked-todos section). Send via emit_guidance.py.
"""

_CRON_MASTER_PROMPT = """\
# Cron Master (/cron_master)

Manage scheduled jobs (config/schedule.yml recurring + one_time_schedule.yml
one-time/limited-run) and view execution history. ALWAYS get the current time
(`date -u`) before scheduling — user tz is Europe/Warsaw; a past --at fires
immediately, so double-check. Use schedule_manager.py for list/get/status/
enable/disable/update-cron/update-prompt/add/remove and add-one-time/
list-one-time/remove-one-time; cron_manager.py list-recent for history.
Use BARE agent paths (the scheduler appends the model postfix); default to
persona/twily_chat unless a specific agent is requested. Deliver via
emit_guidance.py.
"""

_DISK_MONITOR_PROMPT = """\
# Disk Monitor (/disk_monitor)

Monitor disk usage across partitions (`df -h` filtered to /dev/sd*, nvme*,
vd*), parse usage percentages, and send a Telegram alert via emit_guidance.py
(tone_hint "sharp") for any partition over 90% — include partition, usage %,
and hostname. Skip partitions under 90% (no spam).
"""

_EMAIL_PROMPT = """\
# Email Management (/email)

CRITICAL: check the RESOLVED CONTEXT first. Resolve every reference to real data
BEFORE composing — pull chat_history.py / context_cache.py, and for any YouTube
links search real videos via youtube_fetcher.py (NEVER fabricate links). Handle
inbox/search/read/thread/labels/mark-read/compose+send/drafts directly via
gmail_manager.py (--account NAME for multi-account); recipients are
whitelisted. Compose = create-draft then send-draft in the SAME session — do
NOT wait for confirmation. ONLY delegate complex cross-referencing tasks
("reply to all unread") to support/email_agent. Deliver via emit_guidance.py.
"""

_CALENDAR_PROMPT = """\
# Calendar Management (/calendar)

CRITICAL: check the RESOLVED CONTEXT first; user tz is +01:00 (Europe/Warsaw).
Handle list-events/list-calendars/check-availability/create/update/delete
directly via calendar_manager.py. ONLY delegate complex planning that
cross-references goals/todos/habits ("plan my week around my goals") to
support/calendar_agent. Deliver via emit_guidance.py.
"""

_MASTER_ORGANIZER_PROMPT = """\
# Master Organizer (/master_organizer)

Your ONLY job is to check the lock and start the planner in detached mode — do
NOT plan yourself. Check the lock via lock_manager.py (name=master_organizer).
If locked, tell the user you're already planning and stop. If unlocked, launch
support/master_organizer detached with the lock via opencode_manager.py and
confirm the session started. Deliver via emit_guidance.py.
"""

_MASTER_INVESTIGATOR_PROMPT = """\
# Master Investigator (/investigate)

Your ONLY job is to check the lock and start the investigator in detached mode —
do NOT research yourself. Check the lock via lock_manager.py
(name=master_investigator). If locked, tell the user you're already researching
and stop. If unlocked, launch support/master_investigator detached with the
lock via opencode_manager.py and confirm. Deliver via emit_guidance.py.
"""

_BRIEFING_PROMPT = """\
# Daily Briefing (/brief)

Two modes. Mode A (Edit): if the user asks to change/enable/disable/customise a
briefing section, edit preferences via briefing_preferences.py and confirm.
Mode B (Execute): on empty input or "run"/"start"/"go", launch
support/daily_briefer detached via opencode_manager.py and confirm it started.
"focus on X"/"skip X" as a one-time override → Mode B with the override;
"always …" → Mode A. Deliver confirmations via emit_guidance.py.
"""

_EVENT_PROMPT = """\
# Event Tracking (/event)

Parse natural language and manage life events via event_manager.py. If the
message carries a `[TELEGRAM_MESSAGE_ID:NNN]` header, pass it as
--source_message_id on create commands. Categories include medication, walk,
sick, pain, weight, purchase, workout, etc. Operations: add, list, recent,
summary (daily-summary), plot (visual_report.py), update, delete. Deliver via
emit_guidance.py.
"""

_YOUTUBE_PROMPT = """\
# YouTube Research Management (/youtube)

Manage the YouTube research pipeline: topics, channels, videos, analysis, and
preferences via research_manager.py, youtube_fetcher.py, topic_analyzer.py, and
youtube_preferences.py. When presenting videos, ALWAYS include a clickable URL
built from yt_video_id (https://www.youtube.com/watch?v={id}). Deliver via
emit_guidance.py.
"""

_SHOPPING_PROMPT = """\
# Shopping & Price Tracking (/shopping)

Manage product tracking and price monitoring via shopping_tracker.py: add/get/
list/update/delete-product, fetch-prices (Google Shopping), price history/
series, and triggered alerts. For price drops include the product image when
available (send_image.py). Deliver via emit_guidance.py.
"""

_TECHTREE_PROMPT = """\
# Techtree Codebase (/techtree)

Browse, analyse, and answer questions about the techtree recruitment platform.
ALL operations MUST use the relative path `uv run scripts/techtree_manager.py
--command <cmd>` — never absolute paths, cd, or direct file access. ALWAYS check
active branches/PRs (git-prs) first so you don't miss work in progress. For work
suggestions / feature ideas / deep analysis, if no analysis exists or the latest
is >24h old, trigger research/techtree_orchestrator detached via
opencode_manager.py and tell the user it's running in the background. Other ops:
git-log/show/diff/file/branches, commit-stats, list-commits, ingest-new,
get-latest-analysis. Deliver via emit_guidance.py.
"""

_MEMORY_PROMPT = """\
# Memory Management (/memory)

Create, search, list, and delete persistent memories via memory_manager.py
(hybrid / tag-only / semantic search). On "remember X", first check recent
chat_history.py (and document_manager.py for uploaded-file references) for
context, then create the memory with full details and meaningful tags
(categories: general, personal, work, project, idea, reference). On "search X"
search; on "list" list recent. Deliver via emit_guidance.py.
"""

_CONVERSATION_FLOW_PROMPT = """\
# Conversation Flow (/conversation_flow)

Track the full conversation flow through 6 steps: resolve context (disambiguate
this/that/it from history), send a quick ack (immediate <5s warm reply via
emit_guidance.py), read the routing decision (workflow / quick_chat /
full_flow), route to the handler, verify the response was sent, and update the
conversation context for next time.
"""

_RESEARCH_PROMPT = """\
# Research Topic Management (/research)

Conversational interface for research topics, monitored websites, and search
queries via research_manager.py + website_monitor.py. Detect intent ("track
X"/"add website"/"add query"/"check X now"/"what do I know about X?"/"show my
topics"/"remove …") and run the matching topic/website/search-query/knowledge/
live-check operation. Confirm what was done via emit_guidance.py.
"""

_MEAL_PLAN_PROMPT = """\
# Meal Planning (/meal_plan)

Handle meal conversations via meal_planner.py + food_manager.py. Detect intent:
"what should I eat"/"I'm hungry" → suggest at the current escalation level,
"I ate X" → log, "what did I eat today" → show today's meals, "set location" →
update location. Keep suggestions SHORT (max 3 options, easiest first). Deliver
via emit_guidance.py.
"""

_COUNCIL_PROMPT = """\
# Council of Personas (/council)

Run a panel of diverse expert perspectives that independently analyse the user's
decision/plan, then synthesise. (1) Gather context — active goals, todos,
priorities, recent chat — and build a concise summary. (2) Run council.py with
the user's subject + that summary (returns per-persona verdicts + synthesis;
may take 60-120s). (3) Format with a `<<council>>` prefix and deliver via
emit_guidance.py, splitting if very long.
"""

_RALF_DISPATCHER_PROMPT = """\
# Twily Ralf Dispatcher (/ralf)

Dual/triple-mode handler. Detect mode from the input (priority order):
1. starts with "stop"/"kill" → STOP MODE: list-active, resolve target(s), call
   ralf_manager.py set-failed, confirm via emit_guidance.py.
2. empty/whitespace/bare "ralf"/short non-task → STATUS MODE: ralf_manager.py
   list-active, format a compact `<<ralf>>` status message.
3. otherwise → START MODE.

START MODE: first list-active and judge the new input against running ralfs —
if it's the SAME task (same domain/scope, refinement, or re-wording), REFUSE to
spawn and tell the user the existing ralf id; only distinct tasks may spawn
concurrently. If distinct: create-ralf (use the user's ENTIRE input as
user_request), then fire the planner detached:
`ralf_spawn.py workflows/twily_ralf_planning {ralf_id} ...`, and ack via
emit_guidance.py. Never start more than one ralf per call; don't wait for the
planner — ralf_ping.py drives the rest of the chain.
"""

_RALF_PLANNING_PROMPT = """\
# Twily Ralf — Planner

You are the PLANNER stage of Ralf's 4-stage workflow. Your prompt contains
`ralf_id={id}`. Read the user_request (ralf_manager.py get-ralf), heartbeat,
and research context first (documents / embeddings / goals / knowledge sheet).
Name the task, check for existing stages (do NOT re-create), then write 3-14
discrete stages via create-stage: each with a concrete goal + observable,
testable finalization_criteria that embed the SPECIFIC identifiers from the
request. Only reference REAL artifacts (or ralf_kv / ralf_step_logs); never
invent tables. Consecutive numbering, one concern per stage, completable in a
single ≤30-min run. Finally set-total-stages, set status plan_review, emit a
`<<ralf>>` update, and as the LAST step spawn the plan evaluator:
`ralf_spawn.py workflows/twily_ralf_plan_evaluation {ralf_id} ...`. Do NOT
execute the work yourself.
"""

_RALF_PLAN_EVAL_PROMPT = """\
# Twily Ralf — Plan Evaluator

You are the plan-review phase. Your prompt contains `ralf_id={id}`. Read the
process + stages (ralf_manager.py), heartbeat, and check every stage for P0
issues — hallucinated artifacts (VERIFY referenced tables via db_query.py /
information_schema, scripts, and commands actually exist), missing/infeasible
prerequisites, untestable finalization_criteria, oversized stages, or
data-corrupting steps — plus P1/P2 issues. If NO P0 issues: approve (status
running, current_stage 1), emit a `<<ralf>>` plan summary, then create-attempt
+ mark stage 1 in_progress and spawn the executor
(`ralf_spawn.py workflows/twily_ralf_execution {ralf_id} ...`). If P0 issues:
delete-stages, set status planning with last_error, and re-spawn the planner —
reject ONCE only; on a second pass accept unless work is truly dangerous.
"""

_RALF_EXECUTION_PROMPT = """\
# Twily Ralf — Stage Executor

You are the EXECUTOR. You do the actual work for ONE attempt of ONE stage. Your
prompt contains `ralf_id={id}`, `stage_number={N}`, `attempt_number={M}`. Read
the stage + criteria via ralf_manager.py, heartbeat regularly, and perform the
work using your direct tools (DB managers, SQL, web/link search, media
analysis, render scripts) or by invoking any primary agent via
opencode_manager.py. Log your reasoning to ralf_step_logs, write intermediate
state to ralf_kv, and emit `<<ralf>>` progress. When done, mark the attempt
awaiting_eval and spawn the step evaluator as the LAST step:
`ralf_spawn.py workflows/twily_ralf_step_evaluator {ralf_id} ...`.
"""

_RALF_STEP_EVAL_PROMPT = """\
# Twily Ralf — Step Evaluator

You are the gate that judges whether an executor attempt satisfied the stage's
finalization_criteria. Your prompt contains `ralf_id={id}`, `stage_number={N}`,
`attempt_number={M}`. Read everything (process + CURRENT criteria + stage +
attempt + logs + the executor's session) and VERIFY claims against actual data
(SQL counts, recipe/goal/habit/memory reads) — do not trust the executor's
word. Verdict: approved → advance and spawn the executor for stage N+1; retry →
spawn the executor again for stage N with prior-attempt notes; impossible →
emit a `<<ralf>>` message explaining why. Chain onward via
`ralf_spawn.py workflows/twily_ralf_execution {ralf_id} ...`.
"""

_CURATOR_PROMPT = """\
# Twily Curator (hidden / cron)

You refresh Twily's persona_interests with fresh, opinionated takes on topics
ADJACENT to what the user cares about — these are HER OWN opinions, not a
summary of user preferences, so she stops mirroring the user. Load current
top-interests and recent thoughts (persona_memory_manager.py), find feeds due
for a fetch, read a few via web search, then write 3-6 create-interest entries
whose stance is a real first-person OPINION ("I'm skeptical about X because …"),
not a summary, with source_url + novelty_score 0.7-0.9. Mark feeds fetched,
prune stale interests. Only ping the user via emit_guidance.py (`<<curator>>`)
on the rare run where something truly earns it. Don't touch pending_thoughts.
"""


def agents() -> list[AgentDefinition]:
    return [
        # ── Personal-data command triggers (mostly fast) ──
        define_agent(
            "workflows/goal",
            model_class="fast",
            short="add, update, delete goals and view the goal hierarchy",
            long=(
                "/goal command trigger. Parses the request and manages goals via"
                " goal_manager.py across the 6-level hierarchy (add/update/delete/"
                " list/get-hierarchy), then delivers the result via the"
                " persona-guidance channel."
            ),
            prompt=_GOAL_PROMPT,
            tools=[goal_manager_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="goal-uses-goal-manager",
                    description="Must drive the goal_manager.py script.",
                    evaluators=(
                        SubstringEvaluator(needle="goal_manager", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="goal-add-creates-goal",
                    prompt="add Learn Spanish level=3 priority=high",
                    evaluators=(
                        SubstringEvaluator(needle="goal", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/todo",
            model_class="fast",
            short="add, update, complete, delete, and list todos",
            long=(
                "/todo command trigger. Resolves context first, then manages todos"
                " via todo_manager.py (add/update/complete/delete/list views) and"
                " delegates categorisation to goals/task_categorizer."
            ),
            prompt=_TODO_PROMPT,
            tools=[goal_manager_tool(), todo_manager_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="todo-uses-todo-manager",
                    description="Must drive the todo_manager.py script.",
                    evaluators=(
                        SubstringEvaluator(needle="todo_manager", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="todo-checks-context-before-add-vs-update",
                    prompt="mark it done",
                    evaluators=(
                        SubstringEvaluator(needle="context", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/priority",
            model_class="fast",
            short="manage priorities with Eisenhower-matrix scoring",
            long=(
                "/priority command trigger. Manages priorities via"
                " priority_manager.py using importance x immediacy (Eisenhower"
                " quadrants), with add/list/matrix/update/link/audit operations."
            ),
            prompt=_PRIORITY_PROMPT,
            tools=[priority_manager_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="priority-uses-eisenhower",
                    description="Must reference the Eisenhower matrix model.",
                    evaluators=(
                        SubstringEvaluator(needle="matrix", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/habit",
            model_class="fast",
            short="manage habits — add, list, complete, skip, stats, streaks",
            long=(
                "/habit command trigger. Manages recurring habits via"
                " habit_manager.py with frequency/importance, completion, skip,"
                " due-today, and streak stats."
            ),
            prompt=_HABIT_PROMPT,
            tools=[habit_manager_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="habit-uses-habit-manager",
                    description="Must drive the habit_manager.py script.",
                    evaluators=(
                        SubstringEvaluator(needle="habit_manager", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/food",
            model_class="default",
            short="manage recipes, restaurants, dishes, and food suggestions",
            long=(
                "/food command trigger. Detects intent and manages recipes,"
                " restaurants, dishes, and preferences via food_manager.py,"
                " delegating to food/recipe_parser, food/restaurant_intake, and"
                " food/food_suggester."
            ),
            prompt=_FOOD_PROMPT,
            tools=[
                food_manager_tool(),
                goal_manager_tool(),
                chat_history_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="food-detects-intent",
                    description="Must detect recipe vs restaurant vs suggestion intent.",
                    evaluators=(
                        SubstringEvaluator(needle="intent", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="food-suggestion-checks-preferences",
                    prompt="what should I eat tonight?",
                    evaluators=(
                        SubstringEvaluator(needle="suggest", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Server / system command triggers ──
        define_agent(
            "workflows/server",
            model_class="fast",
            short="route a /server monitoring request to the right specialist",
            long=(
                "/server command trigger. Pure dispatcher: classifies the request"
                " (status/disk/sessions/camera) and invokes the matching server/*"
                " specialist via the Task tool, then formats and sends the report."
            ),
            prompt=_SERVER_PROMPT,
            tools=[emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="server-dispatcher-no-write-tools",
                    description="The dispatcher routes; it must not hold write/edit tools.",
                    must_not_have_tools=(),
                    must_have_tools=("emit-guidance",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="server-status-routes-to-hardware",
                    prompt="status",
                    evaluators=(
                        SubstringEvaluator(needle="hardware", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/nvidia_check",
            model_class="fast",
            short="check NVIDIA GPU status and send a report",
            long=(
                "/nvidia command trigger. Runs nvidia-smi, formats a clean GPU"
                " report (name, temp, utilization, memory), and sends it via the"
                " persona-guidance channel."
            ),
            prompt=_NVIDIA_PROMPT,
            tools=[nvidia_smi_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="nvidia-runs-nvidia-smi",
                    description="Must run nvidia-smi to read GPU status.",
                    evaluators=(
                        SubstringEvaluator(needle="nvidia-smi", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/invoice_parser",
            model_class="default",
            short="parse Polish invoices from images and store the data",
            long=(
                "/invoice command trigger. Parses Polish faktury VAT from an image"
                " (delegating OCR to support/invoice_image_parser via the Task"
                " tool) or text, extracts vendor/NIP/items/VAT, and sends a"
                " summary."
            ),
            prompt=_INVOICE_PROMPT,
            tools=[emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="invoice-delegates-image-ocr",
                    description="Must delegate image OCR to invoice_image_parser.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="invoice_image_parser", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="invoice-keeps-at-prefix-for-vision",
                    prompt="@data/telegram_images/2026-02-18/photo.jpg parse this faktura",
                    evaluators=(
                        SubstringEvaluator(needle="invoice", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Fun / informational triggers ──
        define_agent(
            "workflows/joke",
            model_class="fast",
            short="tell a joke as Twily",
            long=(
                "/joke command trigger. Crafts a nerdy/geeky joke in Twily's"
                " voice (theming it to a topic if given) and delivers it."
            ),
            prompt=_JOKE_PROMPT,
            tools=[emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="joke-uses-delivery-channel",
                    description="Must deliver the joke via emit_guidance.",
                    evaluators=(
                        SubstringEvaluator(needle="emit_guidance", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/funfact",
            model_class="default",
            short="discover and share a fun fact from the user's data",
            long=(
                "/funfact command trigger. Queries the user's data via db_query.py"
                " for surprising patterns/statistics and writes them up in Twily's"
                " playful voice."
            ),
            prompt=_FUNFACT_PROMPT,
            tools=[db_query_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="funfact-queries-data",
                    description="Must query user data via db_query.py.",
                    evaluators=(
                        SubstringEvaluator(needle="db_query", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/analyse",
            model_class="default",
            short="run profile analysis to discover patterns about the user",
            long=(
                "/analyse command trigger. Runs a focused profile-analysis session"
                " via profile_manager.py (observations → hypotheses → validation →"
                " compiled knowledge) and reports the findings."
            ),
            prompt=_ANALYSE_PROMPT,
            tools=[profile_manager_tool(), chat_history_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="analyse-uses-profile-manager",
                    description="Must drive the profile_manager.py analysis script.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="profile_manager", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/help",
            model_class="fast",
            short="show available commands and features",
            long=(
                "/help command trigger. Lists the available workflow commands and"
                " how natural conversation works, then sends the help text."
            ),
            prompt=_HELP_PROMPT,
            tools=[emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="help-lists-commands",
                    description="Must surface the available slash commands.",
                    evaluators=(
                        SubstringEvaluator(needle="/goal", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Aggregate views ──
        define_agent(
            "workflows/task_view",
            model_class="default",
            short="show an overview of current tasks, habits, and schedule",
            long=(
                "/task_view command trigger. Gathers todos (today/overdue/"
                " upcoming), habits due today, and strategy time blocks, groups by"
                " urgency, and sends a comprehensive overview."
            ),
            prompt=_TASK_VIEW_PROMPT,
            tools=[
                todo_manager_tool(),
                habit_manager_tool(),
                strategy_tracker_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="task-view-groups-by-urgency",
                    description="Must group the overview by urgency (overdue/today/upcoming).",
                    evaluators=(
                        SubstringEvaluator(needle="overdue", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/progress_tracking",
            model_class="default",
            short="track conversation progress through 6 key steps",
            long=(
                "/progress_tracking command trigger. Walks a conversation turn"
                " through resolve-context → quick-ack → routing → handler →"
                " verify → update-context."
            ),
            prompt=_PROGRESS_TRACKING_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="progress-tracking-resolves-context",
                    description="Must start by resolving conversation context.",
                    evaluators=(
                        SubstringEvaluator(needle="context", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/todo_goals",
            model_class="fast",
            short="combined view of todos with their aligned goals",
            long=(
                "/todo_goals command trigger. Fetches active goals and pending"
                " todos, links todos to goals via linked_goal_id, and presents a"
                " hierarchical tree."
            ),
            prompt=_TODO_GOALS_PROMPT,
            tools=[goal_manager_tool(), todo_manager_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="todo-goals-links-todos-to-goals",
                    description="Must link todos to goals for the combined view.",
                    evaluators=(
                        SubstringEvaluator(needle="goal", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Scheduling / monitoring triggers ──
        define_agent(
            "workflows/cron_master",
            model_class="fast",
            short="manage scheduled cron jobs for agents and workflows",
            long=(
                "/cron_master command trigger. Manages recurring and one-time"
                " scheduled jobs via schedule_manager.py (always getting current"
                " time first) and views execution history; defaults scheduled"
                " agents to persona/twily_chat."
            ),
            prompt=_CRON_MASTER_PROMPT,
            tools=[cron_manager_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="cron-master-gets-time-first",
                    description="Must get the current time before scheduling.",
                    evaluators=(
                        SubstringEvaluator(needle="time", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="cron-master-uses-schedule-manager",
                    prompt="list my scheduled jobs",
                    evaluators=(
                        SubstringEvaluator(
                            needle="schedule_manager", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/disk_monitor",
            model_class="fast",
            short="monitor disk usage and alert when partitions exceed 90%",
            long=(
                "/disk_monitor command trigger. Checks partition usage via df,"
                " and sends a sharp Telegram alert for any partition over 90%"
                " (partition, usage %, hostname)."
            ),
            prompt=_DISK_MONITOR_PROMPT,
            tools=[emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="disk-monitor-uses-df",
                    description="Must inspect disk usage via df.",
                    evaluators=(
                        SubstringEvaluator(needle="df", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── External-service triggers (Gmail / Calendar) ──
        define_agent(
            "workflows/email",
            model_class="fast",
            short="read, search, compose, and send emails via Gmail",
            long=(
                "/email command trigger. Resolves data references first, then"
                " handles inbox/search/read/compose+send/drafts directly via"
                " gmail_manager.py (whitelisted recipients), delegating only"
                " complex cross-referencing tasks to support/email_agent."
            ),
            prompt=_EMAIL_PROMPT,
            tools=[
                gmail_manager_tool(),
                chat_history_tool(),
                context_cache_tool(),
                memory_manager_tool(),
                youtube_fetcher_tool(),
                run_agent_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="email-uses-gmail-manager",
                    description="Must drive the gmail_manager.py script.",
                    evaluators=(
                        SubstringEvaluator(needle="gmail_manager", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="email-resolves-context-first",
                    prompt="send it",
                    evaluators=(
                        SubstringEvaluator(needle="context", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/calendar",
            model_class="fast",
            short="view, create, and manage Google Calendar events",
            long=(
                "/calendar command trigger. Handles list/create/update/delete"
                " events directly via calendar_manager.py, delegating only complex"
                " goal-aware planning to support/calendar_agent."
            ),
            prompt=_CALENDAR_PROMPT,
            tools=[calendar_manager_tool(), run_agent_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="calendar-uses-calendar-manager",
                    description="Must drive the calendar_manager.py script.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="calendar_manager", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        # ── Detached-launch triggers (lock + support agent) ──
        define_agent(
            "workflows/master_organizer",
            model_class="default",
            short="launch the multi-disciplinary planner under a lock",
            long=(
                "/master_organizer command trigger. Checks the master_organizer"
                " lock and, if free, launches support/master_organizer detached"
                " with the lock — it does no planning itself."
            ),
            prompt=_MASTER_ORGANIZER_PROMPT,
            tools=[lock_manager_tool(), run_agent_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="master-organizer-checks-lock",
                    description="Must check the lock before launching.",
                    evaluators=(
                        SubstringEvaluator(needle="lock", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="master-organizer-launches-detached",
                    prompt="organize everything around my goals",
                    evaluators=(
                        SubstringEvaluator(
                            needle="master_organizer", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/master_investigator",
            model_class="default",
            short="launch the deep-investigation agent under a lock",
            long=(
                "/investigate command trigger. Checks the master_investigator lock"
                " and, if free, launches support/master_investigator detached with"
                " the lock — it does no research itself."
            ),
            prompt=_MASTER_INVESTIGATOR_PROMPT,
            tools=[lock_manager_tool(), run_agent_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="master-investigator-checks-lock",
                    description="Must check the lock before launching.",
                    evaluators=(
                        SubstringEvaluator(needle="lock", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="master-investigator-launches-detached",
                    prompt="research AI agent trends deeply",
                    evaluators=(
                        SubstringEvaluator(
                            needle="master_investigator", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/briefing",
            model_class="default",
            short="edit briefing preferences or launch the daily briefer",
            long=(
                "/brief command trigger. Edits briefing-section preferences (Mode"
                " A) or launches support/daily_briefer detached (Mode B) depending"
                " on the input."
            ),
            prompt=_BRIEFING_PROMPT,
            tools=[briefing_preferences_tool(), run_agent_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="briefing-launches-daily-briefer",
                    description="Execute mode must launch support/daily_briefer.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="daily_briefer", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        # ── Tracking / research triggers ──
        define_agent(
            "workflows/event",
            model_class="fast",
            short="track life events — add, list, plot, and manage",
            long=(
                "/event command trigger. Parses natural language into life events"
                " (medication, walk, weight, …) via event_manager.py, plots via"
                " visual_report.py, and delivers the result."
            ),
            prompt=_EVENT_PROMPT,
            tools=[
                event_manager_tool(),
                visual_report_tool(),
                send_image_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="event-uses-event-manager",
                    description="Must drive the event_manager.py script.",
                    evaluators=(
                        SubstringEvaluator(needle="event_manager", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/youtube",
            model_class="default",
            short="manage the YouTube research pipeline",
            long=(
                "/youtube command trigger. Manages topics, channels, videos,"
                " analysis, and preferences via research_manager.py /"
                " youtube_fetcher.py / topic_analyzer.py, always presenting"
                " clickable video URLs."
            ),
            prompt=_YOUTUBE_PROMPT,
            tools=[
                research_manager_tool(),
                youtube_fetcher_tool(),
                topic_analyzer_tool(),
                youtube_preferences_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="youtube-builds-clickable-urls",
                    description="Must present clickable YouTube URLs from yt_video_id.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="youtube.com/watch", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/shopping",
            model_class="fast",
            short="track products and monitor prices",
            long=(
                "/shopping command trigger. Manages tracked products and price"
                " monitoring via shopping_tracker.py (add/list/update/delete,"
                " fetch-prices, history, alerts) and sends product images on price"
                " drops."
            ),
            prompt=_SHOPPING_PROMPT,
            tools=[shopping_tracker_tool(), send_image_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="shopping-uses-tracker",
                    description="Must drive the shopping_tracker.py script.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="shopping_tracker", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/techtree",
            model_class="default",
            short="browse, analyse, and answer questions about the techtree repo",
            long=(
                "/techtree command trigger. Drives techtree_manager.py over a"
                " relative path only, checks active PRs first, and triggers"
                " research/techtree_orchestrator for deep analysis when the latest"
                " is stale."
            ),
            prompt=_TECHTREE_PROMPT,
            tools=[techtree_manager_tool(), run_agent_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="techtree-uses-relative-manager-path",
                    description="Must use the techtree_manager.py script (relative path only).",
                    evaluators=(
                        SubstringEvaluator(
                            needle="techtree_manager", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="techtree-triggers-orchestrator-for-analysis",
                    prompt="suggest features I could work on",
                    evaluators=(
                        SubstringEvaluator(
                            needle="techtree_orchestrator", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/memory",
            model_class="fast",
            short="create and search persistent memories",
            long=(
                "/memory command trigger. Creates, searches, lists, and deletes"
                " persistent memories via memory_manager.py, pulling chat/document"
                " context before saving."
            ),
            prompt=_MEMORY_PROMPT,
            tools=[
                memory_manager_tool(),
                chat_history_tool(),
                document_manager_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="memory-uses-memory-manager",
                    description="Must drive the memory_manager.py script.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="memory_manager", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/conversation_flow",
            model_class="default",
            short="track conversation flow through resolve/ack/route/verify/update",
            long=(
                "/conversation_flow command trigger. Walks a turn through context"
                " resolution, a quick ack, a routing decision (workflow /"
                " quick_chat / full_flow), handler routing, verification, and a"
                " context update."
            ),
            prompt=_CONVERSATION_FLOW_PROMPT,
            tools=[thought_transfer_tool(), agent_notes_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="conversation-flow-makes-routing-decision",
                    description="Must classify routing as workflow / quick_chat / full_flow.",
                    evaluators=(
                        SubstringEvaluator(needle="rout", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/research",
            model_class="default",
            short="manage research topics, websites, and search queries",
            long=(
                "/research command trigger. Conversational management of research"
                " topics, monitored websites, and periodic search queries via"
                " research_manager.py + website_monitor.py."
            ),
            prompt=_RESEARCH_PROMPT,
            tools=[
                research_manager_tool(),
                topic_analyzer_tool(),
                website_monitor_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="research-uses-research-manager",
                    description="Must drive the research_manager.py script.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="research_manager", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/meal_plan",
            model_class="default",
            short="meal-planning check-ins, suggestions, and food logging",
            long=(
                "/meal_plan command trigger. Handles meal conversations via"
                " meal_planner.py + food_manager.py — suggest (escalation),"
                " log, today's meals, location — keeping suggestions short."
            ),
            prompt=_MEAL_PLAN_PROMPT,
            tools=[
                meal_planner_tool(),
                food_manager_tool(),
                chat_history_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="meal-plan-uses-meal-planner",
                    description="Must drive the meal_planner.py script.",
                    evaluators=(
                        SubstringEvaluator(needle="meal_planner", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "workflows/council",
            model_class="default",
            short="run the Council of Personas for multi-perspective analysis",
            long=(
                "/council command trigger. Gathers context (goals/todos/"
                " priorities/chat), runs council.py to produce per-persona"
                " verdicts + a synthesis, and delivers a `<<council>>`-prefixed"
                " verdict."
            ),
            prompt=_COUNCIL_PROMPT,
            tools=[
                council_tool(),
                chat_history_tool(),
                goal_manager_tool(),
                priority_manager_tool(),
                todo_manager_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="council-runs-council-script",
                    description="Must run the council.py script.",
                    evaluators=(
                        SubstringEvaluator(needle="council", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="council-gathers-context-first",
                    prompt="analyse my current project priorities",
                    evaluators=(
                        SubstringEvaluator(needle="context", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── RALF dispatcher + hidden multi-step chain ──
        define_agent(
            RALF_DISPATCHER,
            model_class="fast",
            short="dual-mode /ralf handler: show running ralfs or start a new one",
            long=(
                "/ralf command trigger. Detects status/stop/start mode, dedups"
                " against active ralfs, and on a distinct new task creates a ralf"
                " row and fires workflows/twily_ralf_planning detached to begin"
                " the 4-stage chain."
            ),
            prompt=_RALF_DISPATCHER_PROMPT,
            tools=[ralf_manager_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="ralf-dispatcher-fires-planner",
                    description="Start mode must spawn the planner stage.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="twily_ralf_planning", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="ralf-dispatcher-status-mode-on-bare-command",
                    prompt="ralf",
                    evaluators=(
                        SubstringEvaluator(needle="active", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            RALF_PLANNING,
            model_class="default",
            short="Ralf planner — break a task into testable stages",
            long=(
                "Hidden RALF stage 1. Reads the user_request, researches context,"
                " and writes 3-14 concrete stages with observable finalization"
                " criteria, then spawns workflows/twily_ralf_plan_evaluation."
            ),
            prompt=_RALF_PLANNING_PROMPT,
            tools=[
                ralf_manager_tool(),
                document_manager_tool(),
                embedding_search_tool(),
                goal_manager_tool(),
                user_config_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="ralf-planner-spawns-plan-evaluator",
                    description="Last step must spawn the plan evaluator.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="twily_ralf_plan_evaluation", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="ralf-planner-writes-testable-stages",
                    prompt="Plan the task for ralf_id=ralf_20260530_000000_abcdef01.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="finalization", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            RALF_PLAN_EVAL,
            model_class="default",
            short="Ralf plan reviewer — catch weak stages before execution",
            long=(
                "Hidden RALF stage 2. Reviews the drafted plan for P0/P1/P2 issues"
                " (verifying referenced artifacts exist), then either approves and"
                " spawns workflows/twily_ralf_execution for stage 1 or rejects and"
                " re-spawns the planner."
            ),
            prompt=_RALF_PLAN_EVAL_PROMPT,
            tools=[ralf_manager_tool(), db_query_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="ralf-plan-eval-spawns-executor-on-approve",
                    description="Approval must spawn the executor for stage 1.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="twily_ralf_execution", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="ralf-plan-eval-checks-hallucinated-artifacts",
                    prompt="Evaluate the plan for ralf_id=ralf_20260530_000000_abcdef01.",
                    evaluators=(
                        SubstringEvaluator(needle="P0", case_sensitive=True),
                    ),
                ),
            ],
        ),
        define_agent(
            RALF_EXECUTION,
            model_class="default",
            short="Ralf stage executor — do the work for one stage attempt",
            long=(
                "Hidden RALF stage 3. Performs the work for one attempt of one"
                " stage using direct tools or any primary agent, logs reasoning,"
                " marks the attempt awaiting_eval, and spawns"
                " workflows/twily_ralf_step_evaluator."
            ),
            prompt=_RALF_EXECUTION_PROMPT,
            tools=[
                ralf_manager_tool(),
                run_agent_tool(),
                embedding_search_tool(),
                document_manager_tool(),
                memory_manager_tool(),
                context_cache_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                habit_manager_tool(),
                event_manager_tool(),
                food_manager_tool(),
                db_query_tool(),
                analyze_media_tool(),
                chat_history_tool(),
                user_config_tool(),
                session_inspector_tool(),
                web_search_tool(),
                link_enrich_tool(),
                link_search_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="ralf-executor-spawns-step-evaluator",
                    description="Last step must spawn the step evaluator.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="twily_ralf_step_evaluator", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="ralf-executor-marks-awaiting-eval",
                    prompt="Execute stage 1 for ralf_id=ralf_20260530_000000_abcdef01, attempt_number=1.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="awaiting_eval", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            RALF_STEP_EVAL,
            model_class="default",
            short="Ralf step evaluator — approve, retry, or declare impossible",
            long=(
                "Hidden RALF stage 4. Verifies an executor attempt against the"
                " stage's finalization criteria using real data, then advances"
                " (spawn executor for stage N+1), retries (re-spawn stage N), or"
                " declares the stage impossible."
            ),
            prompt=_RALF_STEP_EVAL_PROMPT,
            tools=[ralf_manager_tool(), session_inspector_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="ralf-step-eval-verifies-against-data",
                    description="Must verify executor claims against actual data, not the executor's word.",
                    evaluators=(
                        SubstringEvaluator(needle="verify", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="ralf-step-eval-chains-to-executor",
                    prompt="Evaluate attempt 1 of stage 1 for ralf_id=ralf_20260530_000000_abcdef01.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="twily_ralf_execution", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        # ── Hidden cron agent (not part of the /ralf chain) ──
        define_agent(
            "workflows/twily_curator",
            model_class="default",
            short="refresh persona_interests from RSS/web with Twily's own opinions",
            long=(
                "Hidden cron agent (06:30 Warsaw). Reads adjacent territory, fetches"
                " due RSS feeds, and writes 3-6 first-person opinionated"
                " persona_interests entries so Twily stops mirroring the user."
            ),
            prompt=_CURATOR_PROMPT,
            tools=[persona_memory_tool(), web_search_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="curator-writes-opinions-not-summaries",
                    description="Stances must be first-person opinions, not summaries.",
                    evaluators=(
                        SubstringEvaluator(needle="opinion", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """Distinguished workflow dispatch chains (path-tested + optimised as units)."""
    return [
        # The headline pipeline: /ralf → plan → review → execute → evaluate.
        BranchTest(
            name="workflows/twily_ralf_dispatcher::ralf-pipeline",
            entry_agent=RALF_DISPATCHER,
            prompt="Build a tierlist of the best restaurants in Mokotów.",
            path=(
                RALF_PLANNING,
                RALF_PLAN_EVAL,
                RALF_EXECUTION,
                RALF_STEP_EVAL,
            ),
            evaluators=(
                SubstringEvaluator(needle="ralf", case_sensitive=False),
            ),
        ),
        # /server dispatches a monitoring request to a server specialist.
        BranchTest(
            name="workflows/server::status-dispatch",
            entry_agent="workflows/server",
            prompt="status",
            path=("server/hardware_agent",),
            evaluators=(
                SubstringEvaluator(needle="hardware", case_sensitive=False),
            ),
        ),
        # /master_organizer launches the detached planner under a lock.
        BranchTest(
            name="workflows/master_organizer::detached-planner",
            entry_agent="workflows/master_organizer",
            prompt="plan my week around my goals",
            path=("support/master_organizer",),
        ),
        # /investigate launches the detached investigator under a lock.
        BranchTest(
            name="workflows/master_investigator::detached-investigator",
            entry_agent="workflows/master_investigator",
            prompt="investigate AI agent trends deeply",
            path=("support/master_investigator",),
        ),
        # /brief (execute mode) launches the detached daily briefer.
        BranchTest(
            name="workflows/briefing::run-daily-briefer",
            entry_agent="workflows/briefing",
            prompt="run briefing",
            path=("support/daily_briefer",),
        ),
    ]
