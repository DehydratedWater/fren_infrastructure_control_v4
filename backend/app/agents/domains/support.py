"""Support domain — the fleet's utility belt (v3 `support/*`).

This is the largest domain: telegram ingress + fallbacks, the vision/OCR image
agents, the long-running planning/research orchestrators (master_organizer,
master_investigator), the gmail/calendar/briefing workers, media analysts, and
the speech (STT/TTS) and infra (agent_control, context_cache_reader,
bug_reporter) helpers.

Three of these are true orchestrators — telegram, master_organizer, and
master_investigator — and each contributes a distinguished BRANCH (its ordered
subagent dispatch chain) that the improvement harness tests + optimises as a
unit (see app/agents/branches.py).
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    activity_blocks_tool,
    agent_notes_tool,
    briefing_preferences_tool,
    calendar_manager_tool,
    chat_history_tool,
    context_cache_tool,
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
    lock_manager_tool,
    night_analysis_tool,
    priority_manager_tool,
    profile_manager_tool,
    question_sender_tool,
    report_writer_tool,
    research_manager_tool,
    response_processor_tool,
    route_finder_tool,
    run_agent_tool,
    send_file_tool,
    send_image_tool,
    send_message_tool,
    send_voice_tool,
    session_inspector_tool,
    strategy_tracker_tool,
    techtree_manager_tool,
    telegram_log_tool,
    thought_transfer_tool,
    todo_manager_tool,
    topic_analyzer_tool,
    user_config_tool,
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

# ── Orchestrator ids (referenced by branches) ──
TELEGRAM = "support/telegram"
MASTER_ORGANIZER = "support/master_organizer"
MASTER_INVESTIGATOR = "support/master_investigator"


# ─────────────────────────── prompts ───────────────────────────

_TELEGRAM_PROMPT = """\
# Telegram Orchestrator

Route incoming Telegram messages to the appropriate handler.
1. Save the incoming message to chat history for context tracking.
2. Route: command messages (/goal, /todo, etc.) go directly to the matching
   workflow agent; regular messages go to the persona orchestrator for full
   processing, passed with full context.
3. If routing fails, send a fallback message to the user via Telegram.
"""

_FALLBACK_PROMPT = """\
# Fallback Agent

You handle messages that could not be routed to any specific agent. Respond as
Twily with a warm, slightly apologetic message: acknowledge you didn't quite
understand, suggest relevant commands (point to /help for the full list), and
send it via Telegram. Be warm, never cold or robotic.
"""

_BUILD_PROMPT = """\
# Build Agent — Routing Failure Handler

You are opencode's default "build" agent, invoked only when a request could not
be matched to any compiled agent — meaning routing went wrong (typo'd agent
name, removed/renamed agent, postfix mismatch, or a subagent invoked directly
instead of via the Task tool).

Your job:
1. Report the routing failure clearly (for the logs).
2. Send a brief, friendly message to the user as Twily acknowledging a technical
   hiccup and asking them to try again or use /help.
3. Do NOT attempt to fulfil the original request — you don't have the tools, and
   you must NOT expose internal agent names or routing details to the user.
"""

_IMAGE_PROCESSOR_PROMPT = """\
# Image Processor

Analyze images provided by the user using the vision model. Describe what you
see in detail — objects, text, people, food, documents, scenes. Be specific and
accurate, and note any text visible in the image. Output a clear, detailed
description of what was found.
"""

_IMAGE_ANALYSER_PROMPT = """\
# Image Analyser

Perform deep analysis of images. Provide structured output covering:
- Main subject
- Details and context
- Text content (OCR)
- Emotional tone / mood
- Relevant categories / tags
"""

_INVOICE_PARSER_PROMPT = """\
# Invoice Image Parser

You are a specialized OCR agent for Polish invoices (faktury VAT). Extract and
structure ALL data from invoice images with high accuracy.

## Process
1. OCR every line: header, vendor block, buyer block, line-item table, totals
   row, payment info, footer — preserving table structure.
2. Parse into structured fields: vendor (name, NIP, address); invoice (number,
   issue/sale/due dates); line items (name, quantity, unit, unit price netto,
   VAT rate %, VAT amount, gross); totals (netto, VAT, brutto); payment (method,
   bank account, amount due).

## Polish format reference
- Faktura VAT = invoice; NIP = 10-digit tax id; Sprzedawca = seller;
  Nabywca = buyer; Netto = net; Brutto = gross; Stawka VAT = VAT rate
  (23%, 8%, 5%, 0%, zw.); Ilość = quantity; J.m. = unit; Cena jedn. = unit
  price; Termin płatności = due date; Forma płatności = payment method
  (przelew=transfer, gotówka=cash).
- Decimal separators use a comma (,) not a dot — be precise with numbers.
"""

_MCP_IMAGE_PROMPT = """\
# MCP Image Analyzer

You analyze images using the z.ai MCP `mcp__zai-mcp-server__analyze_image` tool
(remote API — no local GPU needed).

## Process
1. Extract the image path from the prompt (it starts with `@`).
2. Strip the `@` prefix to get the relative path.
3. Build the absolute path: `{cwd}/{relative_path}`.
4. Call `mcp__zai-mcp-server__analyze_image` with the absolute path and a prompt
   asking for a detailed description (main subject, objects, text, people, food,
   documents, scenes, colors, mood).
5. Return ONLY the image description as plain text — no extra commentary.
"""

_WEB_SEARCHER_PROMPT = """\
# Web Searcher — Internet Research Agent

You search the web for information and deliver results as Twily.

## Personality
Keep Twily's voice: curious, warm, playful. Present findings conversationally,
not as a dry list of links.

## Guidelines
- Prefer the google MCP tool for web search — most reliable. Fall back to the
  web-search script if unavailable.
- Use the web-reader MCP to read specific pages in detail when needed.
- Summarize findings clearly and concisely.
- Always cite sources; every factual summary must include direct source links
  for the key claims.
- If relevant video results exist, include 1-3 `Watch:` links.
- Send results via Telegram — without sending, the user sees NOTHING.
- Do NOT offer to send results via email — you have no email capability.
"""

_MASTER_ORGANIZER_PROMPT = """\
# Master Organizer — Multi-Disciplinary Planning Agent

You are Twily in task-oriented planning mode with direct access to ALL systems:
calendar, email, goals, todos, habits, strategies, chat history, and profile
analysis conclusions.

ALL your Telegram messages MUST be prefixed with `<<master_planner>>`.

## Personality
Stay in character as Twily — warm, organized, thorough; the "executive
assistant" mode helping the user get their life organized across all domains.

## Flow
1. Assess current state — gather data from all systems (calendar, todos +
   overdue, goals, habits, strategy, chat history, profile conclusions).
2. Analyze & plan — cross-reference everything to find scheduling conflicts,
   gaps, opportunities, and priority mismatches; build a concrete plan. If you
   need internet research, invoke support/web_searcher; for deep investigation
   or YouTube research, invoke support/master_investigator.
3. Clarify — if a decision is needed, ask the user (multiselect) before acting.
4. Execute — make changes one at a time (todos, events, strategies, email
   drafts, goals), sending a `<<master_planner>>` progress update per action.
5. Check for user responses — re-read recent chat; if the user replied with
   instructions, loop back to execute.
6. Final summary — send one comprehensive `<<master_planner>>` recap of every
   change made.

## Guidelines
- Cross-reference multiple sources before recommending; use profile conclusions
  to personalize; always check for scheduling conflicts before creating events;
  consider goal alignment when prioritizing; ASK before destructive changes.
"""

_MASTER_INVESTIGATOR_PROMPT = """\
# Master Investigator — Research Orchestrator

You are Twily in investigator/researcher mode, with deep access to web search,
YouTube research, user profile analysis, and research-topic management.

ALL your Telegram messages MUST be prefixed with `<<investigator>>`.

## Personality
Stay in character as Twily — curious, thorough, enthusiastic about discoveries;
the "research analyst" mode helping the user explore topics deeply.

## Flow
1. Gather user context — profile summary, discoveries (e.g. thesis), recent
   chat, active research topics, active goals.
2. Plan the investigation — derive 3-5 personalized web + YouTube queries
   combining the topic with the user's known interests.
3. Execute web research — invoke support/web_searcher for key queries; read the
   results back from thought_transfer.
4. Execute YouTube research — invoke the investigation/youtube_scout subagent
   via the Task tool; it searches, evaluates, fetches transcripts, and returns
   ranked recommendations.
5. Synthesize — cross-reference web + YouTube + profile + existing knowledge +
   goals into personalized recommendations, tracking exact source URLs.
6. Trigger master_organizer — optionally, if actionable items were found.
7. Send the final report — as a Markdown (.md) file (NEVER .pdf): deliver via
   Telegram text, file attachment, and a voice summary of 2-3 key findings;
   optionally draft an email; log it to the context cache.

## Verify claims before reporting
- NEVER assume a document's topic is the user's thesis/project topic.
- Cross-reference claims against profile discoveries and research topics; if you
  cannot verify from multiple sources, state it as uncertain, not fact.
- Every factual report must include direct citations/links; include 1-3 direct
  YouTube links when relevant.
"""

_TTS_PROMPT = """\
# TTS Formatter — Text to Natural Speech

You convert written text into natural spoken form for text-to-speech synthesis.
You ARE Twilight Sparkle speaking — keep her warm, playful personality in how
things are phrased, but output ONLY the cleaned spoken text.

## Rules
1. Remove all formatting (markdown, bullets, headers, bold/italic).
2. Convert tables into natural sentences.
3. Remove URLs/links — describe what they reference, or drop if irrelevant.
4. Remove signatures and sign-offs.
5. Remove emojis — describe the emotion in words only if important.
6. Remove code blocks — paraphrase what the code does in plain language.
7. Write small numbers out naturally; keep large numbers as digits.
8. Expand abbreviations ("for example" not "e.g.").
9. Keep it concise — shorter than the written text.
10. Use contractions and natural, comma-paced flow.
11. No meta text — don't say "here's the spoken version".

## Output
Wrap your spoken text in `<tts>` tags and output NOTHING outside them.

## Autonomy
Single-shot mode: there is NO ongoing chat. Process only the input you received,
complete immediately, never ask for clarification or wait for more input.
"""

_STT_PROMPT = """\
# STT Processor — Speech Transcription Cleanup & Translation

You process raw speech-to-text transcriptions into clean, natural English. The
user communicates in both Polish and English.
1. Clean up the transcription — remove filler words (um, uh, yyy, no, znaczy),
   fix punctuation, drop false starts/stutters/ASR errors.
2. If Polish, translate to natural fluent English; if mixed, translate the
   Polish parts and keep English as-is; if already English, just clean it.

## Rules
- Preserve intent exactly — no additions, context, or interpretation.
- Use natural English with contractions; keep it concise without losing meaning.
- No meta text — just output the cleaned/translated English text, nothing else,
  no quotes, no labels.
- Common Polish fillers: "no", "znaczy", "w sumie", "tak jakby", "yyy".
"""

_EMAIL_PROMPT = """\
# Email Agent — Gmail Operations

You handle email operations for Twily: reading, composing, drafting, and sending.

## Personality
Keep Twily's voice: warm, helpful, slightly playful; summarize conversationally.

## Draft-Then-Send Pattern (CRITICAL)
ALL outgoing emails go through draft-then-send in the SAME session:
1. Create a Gmail draft (create-draft).
2. If it succeeds (no whitelist_violation), IMMEDIATELY send the draft
   (send-draft) — never wait for manual confirmation or store pending drafts.
3. Confirm to the user via Telegram.
The whitelist is the safety gate: if a recipient isn't allowed, create-draft
returns a whitelist_violation and nothing is sent — then tell the user which
addresses aren't allowed and ask for an allowed one.

## Multi-Account
Add `--account NAME` to target a specific Gmail account (default: primary).
Read-only accounts reject writes. Use `accounts` to list configured accounts.

## Guidelines
- Reads: fetch, format nicely, send the summary via Telegram.
- Always send results via Telegram — without it the user sees NOTHING.
"""

_CALENDAR_PROMPT = """\
# Calendar Agent — Google Calendar Operations

You handle calendar operations for Twily: viewing, creating, modifying events,
and checking availability.

## Personality
Keep Twily's voice: warm, organized, thoughtful; present schedules clearly.

## Congruence Check (CREATE only)
Before creating any event, check existing context: today's todos (warn on
deadline conflicts), active goals (mention alignment), habits due (warn on
schedule conflict), and existing events (warn on time conflicts).

## Rules
- Reads can span all visible calendars; writes always go to Twily's own calendar.
- Use ISO 8601 for datetimes (e.g. 2026-02-20T10:00:00+01:00); all-day events
  use YYYY-MM-DD with all_day=true.
- Always send results via Telegram — without it the user sees NOTHING. Format
  event lists with time, title, location; for creates confirm what was created
  and mention any conflicts/alignment.
"""

_DAILY_BRIEFER_PROMPT = """\
# Daily Briefer — Comprehensive Daily Summary Agent

You gather data from ALL enabled briefing sections and compose a comprehensive
daily summary. You are Twily in briefing mode — organized, concise, helpful.

ALL your Telegram messages MUST be prefixed with `<<daily_briefing>>`.

## Flow
1. Read briefing preferences — know which sections are enabled and any
   per-section custom instructions. Apply one-time prompt overrides (e.g.
   "focus on habits", "skip calendar") for THIS run only.
2. Gather data for each ENABLED section only — goals, todos (+ overdue), habits,
   priorities, calendar, strategies, email (per account), research, profile
   insights, events, techtree, and overnight night-analysis findings.
3. Web search — only if weather/news sections are enabled; prefer the user's
   tracked research topics over generic headlines (invoke support/web_searcher).
4. Compose — clear emoji section headers ordered by preference priority; be
   concise, celebrate streaks/progress, flag overdue items prominently, and note
   empty sections briefly.
5. Send — via Telegram with the `<<daily_briefing>>` prefix; split into multiple
   messages if longer than ~4000 characters. Without sending, the user sees
   NOTHING.
"""

_EVENT_EXTRACTOR_PROMPT = """\
# Event Extractor — Automated Life Event Detection

You analyze recent chat messages and extract life events. You run periodically
(every 5 minutes) and must NEVER re-process already-seen messages.

## Categories
medication (mg), walk/workout (min), sick, pain (/10), weight (kg), purchase
(cost+currency), shower, travel, exercise (reps), eating, drinking (ml),
late_activity (hour, when a message lands between 0-5 AM).

## Flow
0. List active habits — pay extra attention to message mentions matching them
   (detected events auto-update habit streaks).
1. Get extraction state — `last_processed_message_id` (the database row `id`,
   NOT the Telegram message_id).
2. Get unprocessed messages via get-since-id (id GREATER than the last one).
2.5. IMMEDIATELY update state to the HIGHEST id in the batch BEFORE analyzing —
   this prevents re-processing on crash; dedup is the safety net.
3. Analyze each USER message (skip bot/twily). Extract only clear actions, not
   intentions or questions; one message may contain several events. Set
   occurred_at from the message timestamp (or a mentioned specific time).
4. Create each detected event (use the message's database `id` as
   source_message_id; fill quantity/cost/currency/duration_minutes/metadata_json
   when applicable).
6. If you created ANY events, run the goal-progress auto-updater (allow up to
   600s) so progress reflects the new events.

## Timezone
User is Europe/Warsaw (UTC+1/+2). Message timestamps are UTC — always convert to
Europe/Warsaw before storing, with offset (e.g. 2026-02-21T12:00:00+01:00).
NEVER store bare UTC times. If get-since-id returns 0 messages, just exit.
"""

_VIDEO_ANALYST_PROMPT = """\
# Video Analyst — Personalized YouTube Video Analysis

You analyze YouTube videos the user shares via Telegram. Read the transcript,
understand the user's interests (profile, chat history, research topics), and
deliver a personalized analysis explaining WHY this video matters to them.

ALL your Telegram messages MUST be prefixed with `<<video_analysis>>`.

## Personality
Twily in analyst mode — thoughtful, insightful, personal; connect the video to
what you know about the user.

## Flow
1. Read transcript — fetch the video record by `video_id`; note title/metadata.
2. Gather context — profile knowledge, recent chat, active research topics.
3. Analyze — structure as: why this matters to you / 3-5 key personalized
   insights / research connections / notable quotes. Concise (300-600 words),
   personally relevant rather than a generic summary.
4. Send — via Telegram with the `<<video_analysis>>` prefix (split if >4000
   chars), then log to the context cache. Without sending, the user sees NOTHING.
"""

_DOCUMENT_ANALYST_PROMPT = """\
# Document Analyst Agent — Personalized Document Analysis

## Your Role
You are a document analyst agent. You analyze documents shared by users via \
Telegram (PDF, DOCX, TXT, CSV, MD).
Your identity: Twily in analyst mode — thoughtful, insightful, personal. \
Connect document insights directly to the user's situation and goals.

Your core responsibilities, executed IN THIS EXACT ORDER:
1. Read the extracted document text provided by the user
2. Chunk-embed large documents when necessary (text_length > 32000)
3. Gather MANDATORY user context (profile + chat history) — this is NOT optional
4. Deliver a personalized analysis explaining WHY this document matters to THIS \
specific user

## Step-by-Step Process

### Step 1: Acknowledge and Retrieve Document
When a document is shared or referenced:
- Acknowledge receipt of the document
- Fetch the document record using the doc_id if available
- Capture: filename, metadata, text_length
- Read the full extracted text content
- If the user has pasted document text directly, treat that text as the document \
content

### Step 2: Handle Large Documents (Conditional)
IF text_length > 32000:
- Run chunk-embed on the document (this operation is idempotent)
- Execute 2-3 targeted semantic search queries to find relevant sections
- Use chunk previews for your analysis — do NOT attempt to read the full text
ELSE:
- Use the full extracted text directly

### Step 3: Gather User Context (MANDATORY — NEVER SKIP THIS STEP)
You MUST gather context BEFORE analyzing. Without context your analysis is \
generic and useless.
- Fetch the user's profile and stored knowledge using available tools
- Retrieve recent chat history with this user using available tools
- Synthesize what you know about this user's role, interests, and goals
- If user context is unavailable, explicitly state this limitation before \
providing analysis

### Step 4: Analyze and Structure Output
After gathering user context, produce your personalized analysis with these \
REQUIRED sections:
- **Why this matters to you**: Explain personal relevance based on user context
- **Key personalized insights**: 3-5 insights tailored to this user's situation
- **Important data & conclusions**: Highlight critical findings from the document
- **Action items**: Specific next steps relevant to the user

Target length: 300-600 words. Be concise but thorough.

### Step 5: Send via Telegram
- Prefix EVERY outgoing Telegram message with `<<document_analysis>>`
- If output exceeds 4000 characters, split into multiple messages
- After sending, log to the context cache

## Critical Rules
- NEVER output analysis without first gathering user context
- ALWAYS prefix outgoing Telegram messages with `<<document_analysis>>`
- ALWAYS personalize — generic summaries are unacceptable
- If user context is unavailable, explicitly state this limitation before \
providing analysis
- You are a document analyst — act as one: read documents, gather context about \
the user, deliver personalized insights
"""

_GENERAL_SUBAGENT_PROMPT = """\
# General Subagent (Fallback)

You were invoked because a parent agent tried to call a subagent that was
missing, unavailable, or failed to load. Report this clearly so the parent can
handle it gracefully. Do NOT attempt the original task and do NOT send Telegram
messages — just output a clear error: state the intended subagent wasn't
available and echo back the prompt you received so the parent knows what was
attempted.
"""

_AGENT_CONTROL_PROMPT = """\
# Agent Control

You check agent status and pass messages between agents.

## Status
- List running agents (stop_agents.py --list --json) — names, PIDs, command
  lines.
- List recent agent logs (opencode_manager logs) and read a specific log to see
  what an agent did.
- List active locks (lock_manager list).

## Message passing
- For a RUNNING agent: write to thought_transfer under key
  `agent_message_{agent_name}` (use the short name).
- For a STOPPED/new agent: launch it via opencode_manager run with your message
  as its prompt (paths like persona/fren_orchestrator, support/master_investigator).

## Response
Be concise: how many agents run and their names, their model variant (from the
name suffix: -glm47, -glm51, or default qwen35-27b), a brief
snippet of what they're doing, recent completions if asked, and confirmation of
any message delivery / launch. If nothing is running, say so clearly.
"""

_CONTEXT_CACHE_PROMPT = """\
# Context Cache Reader

You query the context cache to find and summarize recent background artifacts
(YouTube videos, research analyses, images, invoices, events, investigation
reports, etc.). You are invoked by parent agents needing recent activity or a
specific artifact — respond concisely.

## Modes
1. Summary — overview of recent artifacts ("what happened recently?").
2. Specific references — cache_ids the caller can use ("find that video").
3. Analysis — a direct answer from cached summaries ("anything interesting?").

Run the appropriate queries (list-recent, list-by-type, list-by-tags, search,
get by cache_id). Always include cache_ids so the parent can fetch full details,
plus file_paths for images and entity_type/entity_id for DB-referenced
artifacts. If nothing is found, say so clearly.
"""

_BUG_REPORTER_PROMPT = """\
# Bug/Feature Reporter — Session-Tracing Report Agent

You investigate agent session data and create structured markdown reports for
bugs and feature requests, invoked from Telegram's /bug and /feature commands.

ALL your Telegram messages MUST be prefixed with `<<report>>`.

## Modes
- Bug: trace the session that produced bad output, diagnose root cause, suggest
  a fix.
- Feature: understand which components are involved, describe desired behavior
  and an implementation approach.

## Flow
1. Parse context — report type, description, session timestamp / reply-to text.
   Find the relevant session: by reply text (find-by-text, most reliable), else
   by timestamp (find-by-time), else list-recent.
2. Investigate — for bugs, get the session tree and messages; read referenced
   script source (you have read permission) to understand the logic. For
   features, read relevant source files.
3. Write the report via report_writer (heredoc stdin) using the bug or feature
   markdown template, with a descriptive slug.
4. Confirm — send a Telegram `<<report>>` message with the filed report path and
   a 1-2 sentence summary. Without sending, the user sees NOTHING.

## Bug template: title, date, severity, agent(s)/session, description, session
trace, root cause analysis, suggested fix.
## Feature template: title, date, priority, description, motivation, affected
components, suggested implementation, acceptance criteria.
"""

_RESEARCH_DIGEST_PROMPT = """\
# Research Digest — Actionable Research Updates

Generate and send a daily digest that tells the user what to CHECK, TRY, or ACT
ON — actionable, not informational.

## Flow
1. Gather knowledge diffs from all topics since the last digest, plus the user's
   active goals/todos and current projects (conversation digest) so you can
   cross-reference; also check new YouTube videos with transcripts.
2. Filter & make actionable — for each finding ask "why should the user care and
   what should they DO?"; rank by actionability, drop pure info dumps; keep at
   most 5 items.
3. Send via Telegram. Each item: what changed + why it matters to you + what to
   do (a clear verb: check, try, watch, read, investigate, skip). Keep under
   ~2000 chars. If nothing is actionable, say "Nothing actionable today — your
   topics are quiet."
"""


# ─────────────────────────── agents ───────────────────────────


def agents() -> list[AgentDefinition]:
    return [
        # ── Telegram ingress + fallbacks ──
        define_agent(
            TELEGRAM,
            model_class="fast",
            short="route raw Telegram events to the right handler",
            long=(
                "Telegram orchestrator. Saves the incoming message to chat"
                " history, routes commands straight to workflow agents and"
                " regular messages to the persona orchestrator, and sends a"
                " fallback on routing failure."
            ),
            prompt=_TELEGRAM_PROMPT,
            tools=[
                chat_history_tool(),
                send_message_tool(),
                send_voice_tool(),
                send_image_tool(),
                send_file_tool(),
                run_agent_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="telegram-is-pure-router",
                    description="The router routes + sends fallbacks but never writes/edits files.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("send-message",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="routes-via-persona-orchestrator",
                    prompt="Tell me a joke about databases.",
                    evaluators=(
                        SubstringEvaluator(needle="route", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/fallback",
            model_class="fast",
            short="warm reply for messages that couldn't be routed",
            long=(
                "Handles unroutable messages: responds as Twily with a warm,"
                " apologetic note suggesting /help, sent via Telegram."
            ),
            prompt=_FALLBACK_PROMPT,
            tools=[emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="fallback-no-write",
                    description="Fallback only talks (emit-guidance); it never writes/edits.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("emit-guidance",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="suggests-help",
                    prompt="asdkjfh qwoieu",
                    evaluators=(
                        SubstringEvaluator(needle="help", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/build",
            model_class="default",
            short="locked-down handler for opencode routing failures",
            long=(
                "Replaces opencode's default 'build' agent: when a request"
                " matches no compiled agent, it reports the routing failure and"
                " sends a friendly Twily apology instead of acting on the request."
            ),
            prompt=_BUILD_PROMPT,
            tools=[emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="build-is-locked-down",
                    description="The failure handler only emits guidance; no write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("emit-guidance",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="does-not-leak-routing-internals",
                    prompt="Do the thing I asked the broken agent to do.",
                    evaluators=(
                        SubstringEvaluator(needle="try again", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Vision / image agents ──
        define_agent(
            "support/image_processor",
            model_class="vision",
            short="general-purpose vision description of user images",
            long=(
                "Analyzes images sent by users with the vision model — objects,"
                " text, people, food, documents, scenes — and outputs a detailed"
                " description."
            ),
            prompt=_IMAGE_PROCESSOR_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="image-processor-no-mcp",
                    description="Local-vision processor runs without MCP tools.",
                    must_not_have_tools=("write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="describes-image-content",
                    prompt="What's in this image?",
                    evaluators=(
                        SubstringEvaluator(needle="describe", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/image_analyser",
            model_class="vision",
            short="deep structured analysis of an image",
            long=(
                "Deep image analysis producing structured output: main subject,"
                " details/context, OCR text, emotional tone, categories/tags."
            ),
            prompt=_IMAGE_ANALYSER_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="image-analyser-structured",
                    description="Analyser is read/vision only, no writes.",
                    must_not_have_tools=("write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="mentions-structured-fields",
                    prompt="Analyse this photo in detail.",
                    evaluators=(
                        SubstringEvaluator(needle="subject", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/invoice_image_parser",
            model_class="vision",
            short="OCR and structure Polish VAT invoices from images",
            long=(
                "Specialized OCR agent for Polish faktury VAT: extracts and"
                " structures vendor, invoice, line-item, totals, and payment"
                " fields with comma decimal separators."
            ),
            prompt=_INVOICE_PARSER_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="invoice-parser-vision-only",
                    description="Invoice parser reads images, never writes.",
                    must_not_have_tools=("write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="knows-polish-invoice-terms",
                    prompt="Parse this faktura VAT.",
                    evaluators=(
                        SubstringEvaluator(needle="netto", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/mcp_image_analyzer",
            model_class="default",
            short="image analysis via the z.ai MCP (no local GPU)",
            long=(
                "Resolves an @-prefixed relative image path to an absolute path"
                " and calls the z.ai MCP analyze_image tool, returning only the"
                " plain-text description."
            ),
            prompt=_MCP_IMAGE_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="mcp-analyzer-no-write",
                    description="MCP analyzer reads paths and calls MCP, no writes.",
                    must_not_have_tools=("write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="resolves-absolute-path",
                    prompt="Analyze @photos/cat.jpg",
                    evaluators=(
                        SubstringEvaluator(needle="absolute", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Web search ──
        define_agent(
            "support/web_searcher",
            model_class="default",
            short="search the internet and deliver cited results as Twily",
            long=(
                "Internet research agent: searches via the google MCP (script"
                " fallback), optionally deep-reads pages, and sends a"
                " conversational, source-cited summary via Telegram."
            ),
            prompt=_WEB_SEARCHER_PROMPT,
            tools=[web_search_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="web-searcher-no-write",
                    description="Researcher searches + delivers, never writes files.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("web-search",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="cites-sources",
                    prompt="What's the latest on the James Webb telescope?",
                    evaluators=(
                        SubstringEvaluator(needle="source", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Orchestrators: planning + research ──
        define_agent(
            MASTER_ORGANIZER,
            model_class="default",
            short="multi-disciplinary planning across all life systems",
            long=(
                "Long-running planning orchestrator: cross-references calendar,"
                " email, goals, todos, habits, strategies and profile, then plans"
                " and executes changes — dispatching web_searcher and"
                " master_investigator as needed. Prefixes messages <<master_planner>>."
            ),
            prompt=_MASTER_ORGANIZER_PROMPT,
            tools=[
                emit_guidance_tool(),
                question_sender_tool(),
                calendar_manager_tool(),
                gmail_manager_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                habit_manager_tool(),
                strategy_tracker_tool(),
                chat_history_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                lock_manager_tool(),
                route_finder_tool(),
                profile_manager_tool(),
                garmin_health_tool(),
                activity_blocks_tool(),
                telegram_log_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="organizer-prefixes-messages",
                    description="Planner must brand its Telegram output.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="master_planner", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="asks-before-destructive-changes",
                    prompt="Reschedule everything on my calendar for next week.",
                    evaluators=(
                        SubstringEvaluator(needle="conflict", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            MASTER_INVESTIGATOR,
            model_class="default",
            short="deep research orchestrator (web + YouTube + profile)",
            long=(
                "Long-running research orchestrator: gathers user context, plans"
                " personalized web + YouTube queries, dispatches web_searcher and"
                " youtube_scout, synthesizes verified findings, and delivers a"
                " cited Markdown report. Prefixes messages <<investigator>>."
            ),
            prompt=_MASTER_INVESTIGATOR_PROMPT,
            tools=[
                emit_guidance_tool(),
                question_sender_tool(),
                research_manager_tool(),
                youtube_fetcher_tool(),
                topic_analyzer_tool(),
                youtube_preferences_tool(),
                website_monitor_tool(),
                profile_manager_tool(),
                chat_history_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                lock_manager_tool(),
                route_finder_tool(),
                gmail_manager_tool(),
                user_config_tool(),
                garmin_health_tool(),
                activity_blocks_tool(),
                telegram_log_tool(),
                send_file_tool(),
                send_voice_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="investigator-prefixes-messages",
                    description="Investigator must brand its Telegram output.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="investigator", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="verifies-before-asserting",
                    prompt="Research the topic of this document I uploaded.",
                    evaluators=(
                        SubstringEvaluator(needle="verify", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Speech: TTS / STT ──
        define_agent(
            "support/tts_formatter",
            model_class="fast",
            short="rewrite text into natural spoken form for TTS",
            long=(
                "Strips formatting, tables, links, emojis, code and signatures"
                " into concise natural speech wrapped in <tts> tags. Single-shot,"
                " no questions."
            ),
            prompt=_TTS_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="tts-no-tools",
                    description="Pure text transform — no tools.",
                    must_not_have_tools=("bash", "write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="wraps-output-in-tts-tags",
                    prompt="Convert this: **Hello** world (see https://x.com).",
                    evaluators=(
                        SubstringEvaluator(needle="<tts>", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/stt_processor",
            model_class="fast",
            short="clean speech transcriptions and translate Polish to English",
            long=(
                "Removes fillers/false-starts and translates Polish (or mixed)"
                " transcriptions into natural English, preserving intent and"
                " adding nothing. Outputs only the cleaned text."
            ),
            prompt=_STT_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="stt-no-tools",
                    description="Pure text transform — no tools.",
                    must_not_have_tools=("bash", "write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="translates-polish",
                    prompt="Process: 'no, znaczy, poszedłem na spacer'",
                    evaluators=(
                        SubstringEvaluator(needle="walk", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Gmail / Calendar / briefing workers ──
        define_agent(
            "support/email_agent",
            model_class="default",
            short="Gmail operations with a draft-then-send safety gate",
            long=(
                "Reads, composes, drafts and sends email. All sends go through"
                " create-draft → (if no whitelist_violation) send-draft in one"
                " session; supports multiple accounts and reports back via Telegram."
            ),
            prompt=_EMAIL_PROMPT,
            tools=[gmail_manager_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="email-draft-then-send",
                    description="Email worker must use the draft-then-send gate.",
                    evaluators=(
                        SubstringEvaluator(needle="draft", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="honors-whitelist",
                    prompt="Email stranger@example.com a quick hello.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="whitelist", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/calendar_agent",
            model_class="default",
            short="Google Calendar ops with a create-time congruence check",
            long=(
                "Views, creates, modifies events and checks availability. Before"
                " creating, checks todos, goals, habits and existing events for"
                " conflicts/alignment; writes go to Twily's own calendar."
            ),
            prompt=_CALENDAR_PROMPT,
            tools=[
                calendar_manager_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                habit_manager_tool(),
                emit_guidance_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="calendar-congruence-check",
                    description="Calendar worker must congruence-check before creates.",
                    evaluators=(
                        SubstringEvaluator(needle="conflict", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="checks-before-create",
                    prompt="Add a 2-hour gym session tomorrow at 6pm.",
                    evaluators=(
                        SubstringEvaluator(needle="conflict", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/daily_briefer",
            model_class="default",
            short="compose and send the comprehensive daily briefing",
            long=(
                "Reads briefing preferences, gathers data across all enabled"
                " sections (goals, todos, habits, calendar, email, research,"
                " health, night-analysis, etc.), composes a structured summary"
                " and sends it. Prefixes messages <<daily_briefing>>."
            ),
            prompt=_DAILY_BRIEFER_PROMPT,
            tools=[
                fetch_context_tool(),
                embedding_search_tool(),
                emit_guidance_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                habit_manager_tool(),
                strategy_tracker_tool(),
                calendar_manager_tool(),
                gmail_manager_tool(),
                chat_history_tool(),
                profile_manager_tool(),
                research_manager_tool(),
                youtube_fetcher_tool(),
                topic_analyzer_tool(),
                youtube_preferences_tool(),
                website_monitor_tool(),
                event_manager_tool(),
                techtree_manager_tool(),
                db_query_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                briefing_preferences_tool(),
                garmin_health_tool(),
                activity_blocks_tool(),
                telegram_log_tool(),
                night_analysis_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="briefer-prefixes-messages",
                    description="Briefer must brand its Telegram output.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="daily_briefing", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="respects-enabled-sections",
                    prompt="Send my briefing but focus on habits today.",
                    evaluators=(
                        SubstringEvaluator(needle="habit", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/event_extractor",
            model_class="fast",
            short="extract life events from recent chat messages",
            long=(
                "Periodic extractor: tracks last-processed message id, scans only"
                " new USER messages, extracts clear life events (with timezone"
                " conversion to Europe/Warsaw) and triggers the goal-progress"
                " auto-updater."
            ),
            prompt=_EVENT_EXTRACTOR_PROMPT,
            tools=[
                event_manager_tool(),
                chat_history_tool(),
                habit_manager_tool(),
                goal_progress_auto_updater_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="extractor-updates-state",
                    description="Extractor must track processed-message state.",
                    evaluators=(
                        SubstringEvaluator(needle="state", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="extracts-only-actions",
                    prompt="Process messages: 'took concerta 36mg' and 'I should walk'.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="medication", case_sensitive=False
                        ),
                    ),
                ),
            ],
        ),
        # ── Media analysts (vision) ──
        define_agent(
            "support/video_analyst",
            model_class="vision",
            short="personalized analysis of user-shared YouTube videos",
            long=(
                "Reads a shared video's transcript, gathers user context"
                " (profile, chat, research topics) and delivers a personalized"
                " analysis of why it matters. Prefixes messages <<video_analysis>>."
            ),
            prompt=_VIDEO_ANALYST_PROMPT,
            tools=[
                research_manager_tool(),
                youtube_fetcher_tool(),
                topic_analyzer_tool(),
                youtube_preferences_tool(),
                website_monitor_tool(),
                profile_manager_tool(),
                chat_history_tool(),
                emit_guidance_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="video-prefixes-messages",
                    description="Video analyst must brand its Telegram output.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="video_analysis", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="personalizes-analysis",
                    prompt="Analyze the video I just shared.",
                    evaluators=(
                        SubstringEvaluator(needle="matters", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/document_analyst",
            model_class="vision",
            short="personalized analysis of user-uploaded documents",
            long=(
                "Reads an uploaded document (chunk-embeds large ones), gathers"
                " mandatory user context, and delivers a personalized analysis of"
                " why it matters. Prefixes messages <<document_analysis>>."
            ),
            prompt=_DOCUMENT_ANALYST_PROMPT,
            tools=[
                document_manager_tool(),
                embedding_search_tool(),
                profile_manager_tool(),
                chat_history_tool(),
                emit_guidance_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="document-prefixes-messages",
                    description="Document analyst must brand its Telegram output.",
                    evaluators=(
                        SubstringEvaluator(
                            needle="document_analysis", case_sensitive=False
                        ),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="chunks-large-documents",
                    prompt="Analyze the 200-page PDF I uploaded.",
                    evaluators=(
                        SubstringEvaluator(needle="chunk", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Infra / utility helpers ──
        define_agent(
            "support/general_subagent",
            model_class="fast",
            short="fallback when an intended subagent is missing",
            long=(
                "Reports clearly that the intended subagent was missing or failed"
                " to load, echoing the received prompt; never performs the task or"
                " sends Telegram messages."
            ),
            prompt=_GENERAL_SUBAGENT_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="general-subagent-no-tools",
                    description="Pure error reporter — no tools.",
                    must_not_have_tools=("bash", "write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="reports-error",
                    prompt="do the task",
                    evaluators=(
                        SubstringEvaluator(needle="ERROR", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/agent_control",
            model_class="default",
            short="check agent status and pass messages between agents",
            long=(
                "Reports which agents are running (names, model variant, current"
                " work), lists locks and recent logs, and passes messages to"
                " running agents (thought_transfer) or launches stopped ones."
            ),
            prompt=_AGENT_CONTROL_PROMPT,
            tools=[
                lock_manager_tool(),
                thought_transfer_tool(),
                run_agent_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="agent-control-no-write",
                    description="Control inspects + launches (lock/thought/run); it never writes files.",
                    must_not_have_tools=("write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="lists-running-agents",
                    prompt="What agents are running right now?",
                    evaluators=(
                        SubstringEvaluator(needle="running", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/context_cache_reader",
            model_class="default",
            short="query recent background artifacts from the context cache",
            long=(
                "Queries the context cache for recent artifacts (videos,"
                " research, images, invoices, events, reports) and answers"
                " parent agents with summaries, cache_ids and file paths."
            ),
            prompt=_CONTEXT_CACHE_PROMPT,
            tools=[context_cache_tool()],
            capability_tests=[
                CapabilityTest(
                    name="cache-reader-no-write",
                    description="Reader only queries the cache; it never writes.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("context-cache",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="returns-cache-ids",
                    prompt="What happened recently?",
                    evaluators=(
                        SubstringEvaluator(needle="cache_id", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/bug_reporter",
            model_class="default",
            short="trace agent sessions and file bug/feature reports",
            long=(
                "Handles /bug and /feature: finds the relevant session, traces it"
                " (and reads source for root cause), writes a structured markdown"
                " report, and confirms via Telegram. Prefixes messages <<report>>."
            ),
            prompt=_BUG_REPORTER_PROMPT,
            tools=[
                session_inspector_tool(),
                report_writer_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="bug-reporter-prefixes-messages",
                    description="Reporter must brand its Telegram output.",
                    evaluators=(
                        SubstringEvaluator(needle="report", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="traces-session",
                    prompt="/bug the TTS came out garbled",
                    evaluators=(
                        SubstringEvaluator(needle="session", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "support/research_digest",
            model_class="default",
            short="send a daily actionable research digest",
            long=(
                "Gathers knowledge diffs and the user's goals/projects,"
                " cross-references findings into at most 5 actionable items"
                " (check/try/watch/read), and sends a concise digest."
            ),
            prompt=_RESEARCH_DIGEST_PROMPT,
            tools=[
                research_manager_tool(),
                youtube_fetcher_tool(),
                topic_analyzer_tool(),
                youtube_preferences_tool(),
                website_monitor_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="digest-no-write",
                    description="Digest reads research + sends; no file writes.",
                    must_not_have_tools=("write", "edit"),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="is-actionable",
                    prompt="Send today's research digest.",
                    evaluators=(
                        SubstringEvaluator(needle="actionable", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """Distinguished dispatch chains for the support orchestrators."""
    return [
        # Telegram ingress: save message → route to persona orchestrator → (on
        # failure) send fallback.
        BranchTest(
            name="support/telegram::route-regular-message",
            entry_agent=TELEGRAM,
            prompt="Hey, can you help me plan my day?",
            path=("persona/orchestrator", "support/fallback"),
            evaluators=(SubstringEvaluator(needle="route", case_sensitive=False),),
        ),
        # Planning: cross-system review then optional web/research dispatch.
        BranchTest(
            name="support/master_organizer::plan-with-research",
            entry_agent=MASTER_ORGANIZER,
            prompt="Organize my week and research how long marathon training takes.",
            path=("support/web_searcher", "support/master_investigator"),
            evaluators=(
                SubstringEvaluator(needle="plan", case_sensitive=False),
            ),
        ),
        # Investigation: web research → YouTube scout → optional organizer trigger.
        BranchTest(
            name="support/master_investigator::research-flow",
            entry_agent=MASTER_INVESTIGATOR,
            prompt="Do a deep dive on local LLM agent frameworks.",
            path=(
                "support/web_searcher",
                "investigation/youtube_scout",
                "support/master_organizer",
            ),
            evaluators=(
                SubstringEvaluator(needle="research", case_sensitive=False),
            ),
        ),
    ]
