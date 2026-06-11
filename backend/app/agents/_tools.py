"""ToolDefinition factories — v4 port of fren v3's _tools.py catalog.

Each ``*_tool()`` factory returns a compile-time ``ToolDefinition`` that scopes a
bash permission to exactly one ``python scripts/<x>.py`` command (via
``build_tool``). This is the v3-parity wiring: attaching these to an agent's
``extra_tools`` (directly or via a skill) turns a pure-prompt agent into one that
can actually call its tools.

Ported 1:1 from ``fren/agents/_tools.py``: the v3 ``HandlerCls`` argument and its
local import are dropped (v4 ``build_tool`` does not need the handler class);
names, descriptions, script paths, and notes are preserved exactly.

Three v3 factories used ``ToolBuilder.from_command`` instead of ``_build``
(command-scoped, not script-scoped): ``nvidia_smi_tool``, ``date_tool``, and
``run_agent_tool``. These are reproduced here with a raw ``BashToolPermission``.
"""

from __future__ import annotations

from src import (
    BashToolPermission,
    ToolDefinition,
    ToolDefinitionHeader,
    ToolDefinitionLogicBash,
)

from app.agents._tooldefs import build_tool as _build


def _raw_command_def(name: str, description: str, command_pattern: str) -> ToolDefinition:
    """A bash ToolDefinition scoped to a raw command pattern (v3 from_command)."""
    return ToolDefinition(
        header=ToolDefinitionHeader(
            name=name,
            description=description,
            usage_explanation_long=description,
            usage_explanation_short=description,
            rules=[],
        ),
        bash_tool=ToolDefinitionLogicBash(
            permission_bash=BashToolPermission(
                tool_name="bash",
                value="allow",
                allowed_commands=[command_pattern],
            ),
            positive_examples=[command_pattern.rstrip("*")],
            negative_examples=[],
            mode_specific_rules=[],
        ),
    )


def ralf_spawn_tool() -> ToolDefinition:
    return _build(
        "ralf-spawn",
        "Hand the RALF chain to the next stage agent (detached spawn). Usage:"
        " python scripts/ralf_spawn.py <agent_id> ralf_id=<id>"
        " [stage_number=N attempt_number=M]. Call as your LAST step.",
        "scripts/ralf_spawn.py",
    )


def workspace_orientation_tool() -> ToolDefinition:
    """Read-only `ls` — kills the blocked-exploration friction at session start.

    Forensics over the opencode session store (2026-06-11): the most common
    blocked calls were `ls`/`find` on the workspace — models orienting before
    using their scripts (354+192+151+142+120 occurrences for the top ls
    variants alone). Listing is harmless; `find`/`cat`/reads stay blocked.
    """
    return _raw_command_def(
        "list-workspace",
        "List files in the workspace (orientation only — your callable tools"
        " are exactly the scripts documented above; run them with"
        " `python scripts/<name>.py`).",
        "ls*",
    )


# ── Telegram ──


def send_message_tool() -> ToolDefinition:
    return _build("send-message", "Send a Telegram message to the user", "scripts/send_message.py")


def send_image_tool() -> ToolDefinition:
    return _build("send-image", "Send an image with caption via Telegram", "scripts/send_image.py")


def send_voice_tool() -> ToolDefinition:
    return _build(
        "send-voice", "Generate TTS audio and send as Telegram voice message", "scripts/send_voice.py"
    )


def send_file_tool() -> ToolDefinition:
    return _build("send-file", "Send a file via Telegram", "scripts/send_file.py")


def question_sender_tool() -> ToolDefinition:
    return _build(
        "question-sender",
        "Send questions with inline keyboards, rate limiting, dedup",
        "scripts/question_sender.py",
    )


def emit_guidance_tool() -> ToolDefinition:
    return _build(
        "emit-guidance",
        (
            "Deliver your turn's output by emitting a PersonaGuidance JSON. "
            "The script delivers inline and returns delivered_text on stdout. "
            "REPLACES send_message.py. "
            "Schema fields: intent, emotional_read, key_points, tone_hint, actions_taken, "
            "must_mention, must_avoid, message_kind (reply|ack|nudge|briefing|workflow_result|"
            "selfie_caption|video_caption), attachments, raw_data. "
            "Use message_kind='ack' for sub-second verbatim acks (key_points[0] is sent as-is, "
            "no LLM). Use raw_data for list/data queries (rows preserved, formatted by intent)."
        ),
        "scripts/emit_guidance.py",
    )


# ── Context ──


def thought_transfer_tool() -> ToolDefinition:
    return _build(
        "thought-transfer",
        "File-based message passing between agents",
        "scripts/thought_transfer.py",
    )


def execution_ledger_tool() -> ToolDefinition:
    return _build(
        "execution-ledger",
        "DB-backed artifact store for inter-agent coordination (replaces thought_transfer)",
        "scripts/execution_ledger.py",
    )


def context_resolver_tool() -> ToolDefinition:
    return _build(
        "context-resolver",
        "Resolve pronoun/reference ambiguities from conversation history",
        "scripts/context_resolver.py",
    )


def intent_inference_tool() -> ToolDefinition:
    return _build(
        "intent-inference",
        "Detect user intents and verify agent handled them",
        "scripts/intent_inference.py",
    )


def chat_history_tool() -> ToolDefinition:
    return _build(
        "chat-history",
        "Read and query chat message history",
        "scripts/chat_history.py",
        note="If the user asked about past conversations, share what you found via your agent's delivery tool (send_message or emit_guidance).",
    )


# ── Goals ──


def goal_manager_tool() -> ToolDefinition:
    return _build(
        "goal-manager",
        "CRUD for goals with hierarchy and progress tracking",
        "scripts/goal_manager.py",
        note="Confirm goal changes to the user via your agent's delivery tool (send_message or emit_guidance).",
    )


def todo_manager_tool() -> ToolDefinition:
    return _build(
        "todo-manager",
        "CRUD for todos/tasks with deadlines and goal alignment",
        "scripts/todo_manager.py",
        note="Confirm todo changes to the user via your agent's delivery tool (send_message or emit_guidance).",
    )


def habit_manager_tool() -> ToolDefinition:
    return _build(
        "habit-manager",
        "Manage recurring habits with occurrence tracking and streaks",
        "scripts/habit_manager.py",
        note="Confirm habit updates to the user via your agent's delivery tool (send_message or emit_guidance).",
    )


def priority_manager_tool() -> ToolDefinition:
    return _build(
        "priority-manager",
        "Manage priorities with importance/immediacy scoring and audits",
        "scripts/priority_manager.py",
        note="Confirm priority changes to the user via your agent's delivery tool (send_message or emit_guidance).",
    )


def strategy_tracker_tool() -> ToolDefinition:
    return _build(
        "strategy-tracker",
        "Manage daily strategies and influence attempt tracking",
        "scripts/strategy_tracker.py",
    )


def nudge_strategist_tool() -> ToolDefinition:
    return _build(
        "nudge-strategist",
        "Strategic nudge campaign management — analyze priorities, plan campaigns, track effectiveness",
        "scripts/nudge_strategist.py",
    )


# ── Food ──


def food_manager_tool() -> ToolDefinition:
    return _build(
        "food-manager",
        "Manage recipes, restaurants, dishes, and food preferences",
        "scripts/food_manager.py",
        note="Share food suggestions or confirmations with the user via your agent's delivery tool (send_message or emit_guidance).",
    )


def meal_planner_tool() -> ToolDefinition:
    return _build(
        "meal-planner",
        "Daily meal check-ins with escalating suggestions",
        "scripts/meal_planner.py",
        note="Send meal suggestions or check-in results to the user via your agent's delivery tool (send_message or emit_guidance).",
    )


# ── Profile ──


def profile_manager_tool() -> ToolDefinition:
    return _build(
        "profile-manager", "Manage user profile analysis data", "scripts/profile_manager.py"
    )


# ── System ──


def db_query_tool() -> ToolDefinition:
    return _build("db-query", "Run read-only SQL queries against the database", "scripts/db_query.py")


def periodic_checker_tool() -> ToolDefinition:
    return _build(
        "periodic-checker",
        "Lightweight periodic check for intervention triggers",
        "scripts/periodic_checker.py",
    )


def proactive_send_tool() -> ToolDefinition:
    return _build(
        "proactive-send",
        "Tier-aware shared cooldown floor for proactive agents",
        "scripts/proactive_send.py",
    )


def cron_manager_tool() -> ToolDefinition:
    return _build(
        "cron-manager", "Log and query cron and workflow executions", "scripts/cron_manager.py"
    )


def ralf_manager_tool() -> ToolDefinition:
    return _build(
        "ralf-manager",
        "Manage Ralf multi-stage workflow state: processes, stages, attempts, logs",
        "scripts/ralf_manager.py",
    )


def lock_manager_tool() -> ToolDefinition:
    return _build(
        "lock-manager", "Single-instance enforcement via file-based locks", "scripts/lock_manager.py"
    )


def persona_memory_tool() -> ToolDefinition:
    return _build(
        "persona-memory-manager",
        "Manage persona_interests / pending_thoughts / rss_feeds (Twily's own backlog)",
        "scripts/persona_memory_manager.py",
    )


def peek_thought_tool() -> ToolDefinition:
    return _build(
        "peek-thought",
        "Read-only peek at Twily's top pending_thoughts — for drift cues",
        "scripts/peek_thought.py",
    )


def persona_vibe_tool() -> ToolDefinition:
    return _build(
        "persona-vibe",
        "Manage Twily's palette-blend vibe state and rule-scorer audit log",
        "scripts/persona_vibe_manager.py",
    )


# ── Persona ──


def select_pose_tool() -> ToolDefinition:
    return _build(
        "pose-selector", "Select emotional pose for Twily character", "scripts/select_pose.py"
    )


# ── Context (additional) ──


def response_processor_tool() -> ToolDefinition:
    return _build(
        "response-processor",
        "Process user messages to detect task completions and acknowledgments",
        "scripts/response_processor.py",
    )


def agent_notes_tool() -> ToolDefinition:
    return _build(
        "agent-notes",
        "Persistent key-value memory for agents with TTL and scratchpad",
        "scripts/agent_notes.py",
    )


# ── Workflow Master ──


def wm_session_manager_tool() -> ToolDefinition:
    return _build(
        "wm-session-manager",
        "Manage workflow master sessions and history",
        "scripts/wm_session_manager.py",
    )


def wm_file_operations_tool() -> ToolDefinition:
    return _build(
        "wm-file-operations",
        "Safe file operations for workflow creation (restricted directories)",
        "scripts/wm_file_operations.py",
    )


# ── Vis Simulation ──


def ponyxl_prompt_composer_tool() -> ToolDefinition:
    return _build(
        "ponyxl-prompt-composer",
        "Compose structured PonyXL prompts for MLP character images with parametric expressions",
        "scripts/ponyxl_prompt_composer.py",
    )


def render_ponyxl_tool() -> ToolDefinition:
    return _build(
        "render-ponyxl",
        "Render PonyXL images/videos — blocking or non-blocking dispatch to background workers",
        "scripts/render_ponyxl.py",
    )


def rp_scene_composer_tool() -> ToolDefinition:
    return _build(
        "rp-scene-composer",
        "Compose free-form PonyXL prompts for RP adventure scene illustrations (any setting, not MLP-locked)",
        "scripts/rp_scene_composer.py",
    )


def render_rp_scene_tool() -> ToolDefinition:
    return _build(
        "render-rp-scene",
        "Dispatch RP scene illustration to background worker (sends via RP bot, not main bot)",
        "scripts/render_rp_scene.py",
    )


def vis_simulation_manager_tool() -> ToolDefinition:
    return _build(
        "vis-simulation-manager",
        "Manage character simulation training data for fine-tuning",
        "scripts/vis_simulation_manager.py",
    )


# ── Media ──


def analyze_media_tool() -> ToolDefinition:
    return _build(
        "analyze-media",
        "Analyze video files using a vision model — returns visual description + audio transcript. "
        "For images use the Read tool instead. "
        "Handles long videos via 30s chunking with Telegram progress messages. Set bash timeout to 300s.",
        "scripts/analyze_media.py",
    )


# ── Dashboard support ──


def session_inspector_tool() -> ToolDefinition:
    return _build(
        "session-inspector",
        "Browse opencode agent session data, messages, and tool calls",
        "scripts/session_inspector.py",
    )


def report_writer_tool() -> ToolDefinition:
    return _build(
        "report-writer",
        "Write, list, read, and resolve bug/feature reports",
        "scripts/report_writer.py",
    )


def agent_analyzer_tool() -> ToolDefinition:
    return _build(
        "agent-analyzer",
        "Scan agent definitions and build dependency graphs",
        "scripts/agent_analyzer.py",
    )


# ── Research ──


def research_manager_tool() -> ToolDefinition:
    return _build(
        "research-manager",
        "Manage research topics, YouTube channels, and analysis data",
        "scripts/research_manager.py",
    )


def youtube_fetcher_tool() -> ToolDefinition:
    return _build(
        "youtube-fetcher",
        "Fetch YouTube channel videos and transcripts via SearchAPI.io. Set bash timeout to 300s.",
        "scripts/youtube_fetcher.py",
    )


def youtube_preferences_tool() -> ToolDefinition:
    return _build(
        "youtube-preferences",
        "Manage YouTube user preferences and video feedback",
        "scripts/youtube_preferences.py",
    )


def shopping_tracker_tool() -> ToolDefinition:
    return _build(
        "shopping-tracker",
        "Track product prices via Google Shopping with alerts",
        "scripts/shopping_tracker.py",
        note="Share price updates or alerts with the user via your agent's delivery tool (send_message or emit_guidance).",
    )


def web_search_tool() -> ToolDefinition:
    return _build(
        "web-search",
        "Search the web via Google using SearchAPI.io",
        "scripts/web_search.py",
        note="Share interesting findings with the user via your agent's delivery tool (send_message or emit_guidance) — don't just dump raw results.",
    )


def topic_analyzer_tool() -> ToolDefinition:
    return _build(
        "topic-analyzer",
        "Prepare data for topic analysis and save analysis results",
        "scripts/topic_analyzer.py",
    )


def website_monitor_tool() -> ToolDefinition:
    return _build(
        "website-monitor",
        "Check websites for content changes and run periodic search queries. Set bash timeout to 300s.",
        "scripts/website_monitor.py",
    )


def techtree_manager_tool() -> ToolDefinition:
    return _build(
        "techtree-manager",
        "Browse techtree git repo, track commits, manage interests, and analyze code changes",
        "scripts/techtree_manager.py",
    )


def invoice_manager_tool() -> ToolDefinition:
    return _build(
        "invoice-manager",
        "Import, query, and manage parsed invoices",
        "scripts/invoice_manager.py",
    )


def document_manager_tool() -> ToolDefinition:
    return _build(
        "document-manager",
        "Parse, store, and query uploaded documents (PDF, DOCX, TXT, CSV, MD)",
        "scripts/document_manager.py",
    )


# ── Comms ──


def gmail_manager_tool() -> ToolDefinition:
    return _build(
        "gmail-manager",
        "List, read, search, draft, and send emails via Gmail. Set bash timeout to 300s."
        " Tell the user what you found or did via your agent's delivery tool (send_message or emit_guidance).",
        "scripts/gmail_manager.py",
    )


def briefing_preferences_tool() -> ToolDefinition:
    return _build(
        "briefing-preferences",
        "Manage daily briefing section preferences — toggle, reorder, and customize",
        "scripts/briefing_preferences.py",
    )


def calendar_manager_tool() -> ToolDefinition:
    return _build(
        "calendar-manager",
        "Manage Google Calendar events and check availability",
        "scripts/calendar_manager.py",
    )


# ── Standalone commands (not ScriptTool-based) ──


def nvidia_smi_tool() -> ToolDefinition:
    return _raw_command_def("nvidia-smi", "Check NVIDIA GPU status", "nvidia-smi*")


def date_tool() -> ToolDefinition:
    return _raw_command_def("date", "Get current timestamp", "date +%s.%N")


def route_finder_tool() -> ToolDefinition:
    return _build(
        "route-finder",
        "Find agent routes and list reachable capabilities via graph traversal",
        "scripts/route_finder.py",
    )


def event_manager_tool() -> ToolDefinition:
    return _build(
        "event-manager",
        "Track life events with categories, values, and plotting",
        "scripts/event_manager.py",
    )


def user_config_tool() -> ToolDefinition:
    return _build(
        "user-config",
        "Read and write user preferences and agent configuration",
        "scripts/user_config.py",
    )


def lesson_manager_tool() -> ToolDefinition:
    return _build(
        "lesson-manager",
        "Manage agent lessons learned from past mistakes",
        "scripts/lesson_manager.py",
    )


def user_rules_tool() -> ToolDefinition:
    return _build(
        "user-rules",
        "Manage persistent user rules/directives that all agents must follow",
        "scripts/user_rules.py",
    )


def screenshot_tool() -> ToolDefinition:
    return _build(
        "screenshot",
        "Capture desktop screenshot for situational awareness. Returns file path — use Read tool to view the image.",
        "scripts/screenshot.py",
    )


def camera_capture_tool() -> ToolDefinition:
    return _build(
        "camera-capture",
        "Capture a photo from webcam (room view) or desk camera (hands/keyboard view)."
        " Commands: webcam, desk, both. Returns file path(s) — use Read tool to view.",
        "scripts/camera_capture.py",
    )


def context_pin_tool() -> ToolDefinition:
    return _build(
        "context-pin",
        "Manage discussion topics, pinned context, and document references",
        "scripts/context_pin.py",
    )


def context_cache_tool() -> ToolDefinition:
    return _build(
        "context-cache",
        "Central registry of background artifacts (YouTube, research, images, invoices, etc.)",
        "scripts/context_cache.py",
    )


def activity_blocks_tool() -> ToolDefinition:
    return _build(
        "activity-blocks",
        "Structured activity timeline with time-ranged blocks",
        "scripts/activity_blocks.py",
    )


def telegram_log_tool() -> ToolDefinition:
    return _build(
        "telegram-log",
        "Read the user's personal Telegram activity log — hashtags, links, notes",
        "scripts/telegram_log.py",
    )


def garmin_health_tool() -> ToolDefinition:
    return _build(
        "garmin-health",
        "Query Garmin health data: body battery, stress, heart rate, sleep, daily stats",
        "scripts/garmin_health.py",
    )


# ── Home ──


def tuya_lights_tool() -> ToolDefinition:
    return _build(
        "tuya-lights",
        "Control Tuya smart home devices: lights, plugs, switches",
        "scripts/tuya_lights.py",
    )


def memory_manager_tool() -> ToolDefinition:
    return _build(
        "memory-manager",
        "Create, search, and manage persistent memories with tags and embeddings",
        "scripts/memory_manager.py",
    )


def fetch_context_tool() -> ToolDefinition:
    return _build(
        "fetch-context",
        "Unified retrieval from all memory systems — fast heuristic search across chat, memories, embeddings, pins, goals/todos",
        "scripts/fetch_context.py",
    )


def embedding_search_tool() -> ToolDefinition:
    return _build(
        "embedding-search",
        "Cross-table semantic search using embeddings. Set bash timeout to 300s."
        " If the user asked you to find something, share results via your agent's delivery tool (send_message or emit_guidance).",
        "scripts/embedding_search.py",
    )


def link_enrich_tool() -> ToolDefinition:
    return _build(
        "link-enrich",
        "Fetch URL metadata (title, description, og tags) and cache in link_previews for search.",
        "scripts/link_enrich.py",
    )


def link_search_tool() -> ToolDefinition:
    return _build(
        "link-search",
        "Find URLs shared in chat; semantic-search link previews by topic; anchor URLs around a name.",
        "scripts/link_search.py",
        note="When the user asks about a previously-shared link, always try this before giving up.",
    )


def tool_history_tool() -> ToolDefinition:
    return _build(
        "tool-history",
        "Query the tool execution audit log — recent calls, errors, stats",
        "scripts/tool_history.py",
    )


def night_analysis_tool() -> ToolDefinition:
    return _build(
        "night-analysis",
        "Query night analysis findings, reports, and run history",
        "scripts/night_analysis_query.py",
    )


def goal_progress_auto_updater_tool() -> ToolDefinition:
    return _build(
        "goal-progress-auto-updater",
        "Automatically update goal progress from evidence (events, activities, habits, todos). Set bash timeout to 300s.",
        "scripts/goal_progress_auto_updater.py",
    )


def visual_report_tool() -> ToolDefinition:
    return _build(
        "visual-report",
        "Generate visual data reports as PNG images (todo board, goals, habits, priorities, campaigns, events, daily summary)",
        "scripts/visual_report.py",
    )


def run_agent_tool() -> ToolDefinition:
    """Provides the bash permission to invoke agents via opencode_manager.py run."""
    return _raw_command_def(
        "run-agent",
        "Invoke another agent via opencode_manager.py run",
        "uv run scripts/opencode_manager.py run *",
    )


# ── Personality ──


def personality_core_tool() -> ToolDefinition:
    return _build(
        "personality-core",
        "Consult Twily's inner consciousness for emotional processing and response guidance",
        "scripts/personality_core.py",
    )


# ── Council ──


def council_tool() -> ToolDefinition:
    return _build(
        "council",
        "Run the Council of Personas — multi-perspective analysis of decisions and work",
        "scripts/council.py",
    )


# ── Daily Routines ──


def routine_manager_tool() -> ToolDefinition:
    return _build(
        "routine-manager",
        "Manage daily routines checklist — predefined recurring tasks by weekday with time windows",
        "scripts/routine_manager.py",
    )


# ── RP Adventure ──


def rp_adventure_manager_tool() -> ToolDefinition:
    return _build(
        "rp-adventure-manager",
        "Manage RP adventures — create, get, list, update status",
        "scripts/rp_adventure_manager.py",
    )


def rp_character_manager_tool() -> ToolDefinition:
    return _build(
        "rp-character-manager",
        "Manage RP characters — create, get, list, update, load-persona",
        "scripts/rp_character_manager.py",
    )


def rp_world_manager_tool() -> ToolDefinition:
    return _build(
        "rp-world-manager",
        "Manage RP world state aspects — set, get, list",
        "scripts/rp_world_manager.py",
    )


def rp_story_manager_tool() -> ToolDefinition:
    return _build(
        "rp-story-manager",
        "Manage RP story log — append entries, read history, get turn count",
        "scripts/rp_story_manager.py",
    )


def rp_cross_summary_tool() -> ToolDefinition:
    return _build(
        "rp-cross-summary",
        "Bidirectional summaries between main bot and RP bot",
        "scripts/rp_cross_summary.py",
    )


def send_rp_message_tool() -> ToolDefinition:
    return _build(
        "send-rp-message",
        "Send a message to the user via the RP Telegram bot",
        "scripts/send_rp_message.py",
    )


def rp_ban_manager_tool() -> ToolDefinition:
    return _build(
        "rp-ban-manager",
        "Manage anti-cliche ban rules for RP adventures — list, analyze, add, remove",
        "scripts/rp_ban_manager.py",
    )
