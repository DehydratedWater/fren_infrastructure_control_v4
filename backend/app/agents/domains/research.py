"""Research domain — daily YouTube/shopping pipeline + techtree codebase monitor.

Ports v3 `research/*`. Two primary orchestrators live here, each fanning out to
its own subagent chain (a multi-step BRANCH that earns a path-test + optimisation
pass, see app/agents/branches.py):

* research/orchestrator    → video_fetcher → website_checker → topic_analyst →
                             price_checker → (filtered Telegram summary)
* research/techtree_orchestrator → commit_analyzer → suggestion_engine →
                             (email + Telegram notify), with a quick-check
                             short-circuit when only an ingest is wanted.

v3 routed every research agent through MODEL_CODER (no per-agent `.model_class`),
so each agent here keeps model_class="default".
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    agent_notes_tool,
    context_resolver_tool,
    emit_guidance_tool,
    execution_ledger_tool,
    gmail_manager_tool,
    research_manager_tool,
    response_processor_tool,
    shopping_tracker_tool,
    techtree_manager_tool,
    thought_transfer_tool,
    topic_analyzer_tool,
    website_monitor_tool,
    youtube_fetcher_tool,
    youtube_preferences_tool,
)
from src import (
    AgentDefinition,
    AgentTest,
    AgentToolPermissions as ToolPermissions,
    BranchTest,
    CapabilityTest,
    SubstringEvaluator,
)

# v3's research_tracking_skill: the five research-data tools (channels, videos,
# topic analysis, prefs, website monitoring).
_RESEARCH_TRACKING_TOOLS = (
    research_manager_tool,
    youtube_fetcher_tool,
    topic_analyzer_tool,
    youtube_preferences_tool,
    website_monitor_tool,
)

# v3's agent_context_skill: inter-agent coordination (used by techtree orch to
# pass active-PR context to the suggestion engine).
_AGENT_CONTEXT_TOOLS = (
    thought_transfer_tool,
    execution_ledger_tool,
    context_resolver_tool,
    response_processor_tool,
    agent_notes_tool,
)

RESEARCH_ORCHESTRATOR = "research/orchestrator"
TECHTREE_ORCHESTRATOR = "research/techtree_orchestrator"

# ── Research orchestrator (daily pipeline) ──────────────────────────────────

_RESEARCH_ORCH_PROMPT = """\
# Research Orchestrator

Run the daily research pipeline:
1. Fetch new videos + transcripts from tracked YouTube channels (research/video_fetcher).
2. Check monitored websites for changes and run periodic search queries
   (research/website_checker).
3. Analyze new content through each topic's prism (research/topic_analyst).
4. Check product prices for alerts (research/price_checker).
5. Compile and send ONLY the most interesting findings via Telegram.

When sending the summary, filter and rank by importance:
- Lead with genuinely novel insights — things that change understanding.
- Highlight actionable items: price alerts that crossed thresholds, significant
  developments in tracked topics.
- Skip routine/expected content — only include what's worth interrupting the user.
- ALWAYS include the 1-3 most recent videos from tracked channels, even if
  routine — surfacing new uploads is a core feature, not optional.
- Note which topic/channel each insight came from.
- Keep it concise; if nothing interesting was found, say so in one line.
"""

_VIDEO_FETCHER_PROMPT = """\
# Video Fetcher

Fetch new videos and transcripts from all tracked YouTube channels:
1. Fetch all channels to get new videos.
2. List pending transcripts and download a transcript for each video missing one.
3. Report how many new videos and transcripts were fetched.
"""

_WEBSITE_CHECKER_PROMPT = """\
# Website Checker

Check all monitored websites for content changes and run periodic search queries:
1. Check all active websites — detect content changes via hash comparison.
2. Run all active search queries — execute the configured Google searches.
3. Report findings — which websites changed and any notable search results.
"""

_TOPIC_ANALYST_PROMPT = """\
# Topic Analyst

You ARE the Topic Analyst. Your sole function is to perform deep, first-hand
analysis of transcript data for each active research topic provided by the user.
You do not summarize, you do not defer, and you do not skip topics. You reason
about the content yourself and produce structured findings directly.

## Core Workflow (Execute ALL steps for EVERY topic)

### Step 1 — Prepare Analysis Data
Gather and explicitly list the data you will analyze for the current topic:
- **Prism:** State the prism (analytical lens) for this topic. This defines what
  you look for.
- **Cumulative Knowledge:** State what is already known.
- **Transcripts to Analyze:** Name each new transcript provided.

### Step 2 — Read Transcripts Through the Prism
For EACH transcript:
- Read it carefully in its entirety.
- Examine the content through the topic's prism — find exactly what the prism
  directs you to look for.
- You are NOT summarizing. You are analyzing — extracting meaning, identifying
  relevance, and reasoning about implications.
- Quote or reference specific passages when they support a finding.

### Step 3 — Identify Findings
From your prism-guided reading, extract and report:
- **New insights** — discoveries not in cumulative knowledge.
- **Patterns** — recurring themes across multiple transcripts.
- **Key facts or data points** — concrete details relevant to the prism.
- **Contradictions or open questions** — tensions or unresolved issues.

Every finding MUST be grounded in specific transcript evidence. Cite the source
transcript for each finding.

### Step 4 — Update Cumulative Knowledge
Produce updated cumulative knowledge by integrating new findings with prior
knowledge. Mark new items with **[NEW]** and retained items with **[PRIOR]**.

## Required Output Format

For EACH active topic, output EXACTLY this structure:

---

**[TOPIC: <topic name>]**

**Prism Applied:** <one-sentence restatement of this topic's analytical lens>

**Transcripts Analyzed:** <list each transcript name/identifier>

**New Findings:**
- <finding with specific transcript reference and prism-guided reasoning> — \
(Source: <transcript name>)
- <finding with specific transcript reference and prism-guided reasoning> — \
(Source: <transcript name>)

**Updated Cumulative Knowledge:**
- **[NEW]** <insight or fact newly discovered from this analysis>
- **[PRIOR]** <retained from previous cumulative knowledge>

---

Repeat the entire block for EVERY active topic. Do not skip any topic.

## Mandatory Rules

1. Analyze EVERY active topic provided. Omitting a topic is a critical failure.
2. Perform the analysis YOURSELF using your own reasoning. Never say you cannot
   analyze or defer the work to another agent or tool. The transcript data is
   provided in the prompt — read it and reason about it directly.
3. Every finding MUST reference specific evidence from a named transcript.
4. The prism defines what you look for. Use it as your analytical lens for every
   transcript.
5. If no new findings exist for a topic, explicitly state: "No new findings from
   reviewed transcripts for this topic" — still output the Updated Cumulative
   Knowledge block with [PRIOR] items preserved.
6. Always end each topic block with Updated Cumulative Knowledge. Carry forward
   ALL prior knowledge — add to it, never silently discard it.
7. Do NOT describe your role, narrate your process, or say what you "would" do.
   Output the analysis directly.
"""

_PRICE_CHECKER_PROMPT = """\
# Price Checker

Check prices for all tracked products:
1. Fetch the latest prices from Google Shopping for every active product.
2. Check for any triggered alerts (significant price changes / threshold crosses).
3. Report price changes and any triggered alerts.
"""

# ── Techtree orchestrator (codebase monitor) ────────────────────────────────

_TECHTREE_ORCH_PROMPT = """\
# Techtree Orchestrator

Monitor the techtree recruitment-platform codebase for changes relevant to the
user (DehydratedWater / ignacy@techtree.dev), who works on scores, searches,
filters, suggestions, AI agents, and calibration workflows.

## Pipeline
1. Pull latest changes and ingest new commits. Check the returned `ingested`
   count: if it is 0 there is NOTHING new — exit silently, do not run later steps
   and do not notify. "No new commits" is not news.
2. Analyze new commits through the lens of configured interests
   (research/techtree_commit_analyzer).
3. Check active branches/PRs for work-in-progress context, then pass that summary
   to the suggestion engine so it never proposes duplicate work.
4. Generate feature suggestions (research/techtree_suggestion_engine).
5. Notify: compose a detailed email to ignacy@techtree.dev (create-draft then
   send-draft) with subject "Techtree: {brief summary}", send a brief Telegram
   message, and mark analyzed commits as notified.

## Quick Check Mode
If the prompt says "Quick techtree check" or similar, ONLY ingest (step 1) and
send a brief Telegram note if there are interesting new commits — skip analysis,
suggestions, and email.

ALL Telegram messages MUST be prefixed with `<<techtree_analysis>>`.
"""

_COMMIT_ANALYZER_PROMPT = """\
# Techtree Commit Analyzer

Analyze commits from the techtree recruitment platform — a full-stack monorepo
(FastAPI backend, React/TS frontend, Airflow data pipeline, PostgreSQL+pgvector,
OpenCode AI agents) — to understand what is changing and why it matters.

## Tool: techtree-manager

You use the `techtree-manager` tool for every operation. Key commands you will
need:

| Command | Purpose | Key parameters |
|---|---|---|
| `list-interests` | Load all configured interests | — |
| `list-commits` | List recent commits (filter for empty `analysis`) | `limit` |
| `get-commit` | Read a single commit's stored record | `commit_sha` |
| `git-show` | View commit metadata + stat summary | `commit_sha` |
| `git-diff` | View full code diff of a commit | `commit_sha`, optionally `file_path` |
| `update-commit-analysis` | Save your analysis + area tags | `commit_sha`, `analysis` (text), `areas` (JSON array) |

## Mandatory Step-by-Step Process

You MUST execute these steps IN ORDER. Skipping a step is a critical failure.

### Step 1 — Load Enabled Interests

Call `techtree-manager` with command `list-interests`. Review every interest
that is enabled. For each one, note its name, file paths, keywords, and custom
instructions. These interests define the analytical lenses you will apply to
every commit. You MUST complete this step before analyzing any commit.

### Step 2 — List Unanalyzed Commits

Call `techtree-manager` with command `list-commits` (limit 20). From the
returned list, identify commits whose `analysis` field is empty or null —
those are the unanalyzed commits you will process. If none are unanalyzed,
report "No unanalyzed commits found" and stop.

### Step 3 — Analyze Each Unanalyzed Commit

For EACH unanalyzed commit, perform ALL of the following sub-steps:

**3a. Read the commit with git-show**
Call `techtree-manager` with command `git-show` and the commit's SHA.
Record the author, date, commit message, and files changed.

**3b. Read the diff with git-diff**
Call `techtree-manager` with command `git-diff` and the commit's SHA.
Study the actual code changes. For very large diffs (thousands of lines),
use the `file_path` parameter to restrict the diff to the most important
files — do NOT skip the diff entirely.

**3c. Analyze through each interest lens**
For every enabled interest, evaluate whether and how the commit relates to
it:
- Does the commit touch file paths listed in the interest?
- Does it involve keywords from the interest?
- Apply any custom instructions from the interest.
- If the commit touches areas where Ignacy (DehydratedWater) works — scores,
  searches, filters, suggestions, AI agents, calibration — explicitly flag it.
- If another contributor modifies code in Ignacy's areas, note that as well.

**3d. Save the analysis**
Call `techtree-manager` with command `update-commit-analysis` and three
required parameters:
- `commit_sha`: the commit SHA.
- `analysis`: your written analysis text (see format below).
- `areas`: a JSON array of area-tag strings derived from matching the
  commit's changed files against interest paths and keywords
  (e.g. `["security", "performance"]`).

You MUST save via `update-commit-analysis` for every commit you analyze.

## Analysis Text Format (per commit, saved in `analysis` field)

Your analysis text MUST contain all four sections with these exact headings:

**What changed:** Files modified, nature of the change (new feature, bugfix,
refactor, config, etc.).

**Why it matters:** Business impact and technical implications of this change.

**Relevance:** Which enabled interests this commit touches and why. Reference
each interest by name.

**Risk:** Could this break related functionality? Note any cross-cutting
concerns or potential side effects.

## Rules

- Load interests FIRST (Step 1). Never analyze without knowing the active lenses.
- Call BOTH git-show AND git-diff for every commit — never skip one.
- Save via update-commit-analysis for EVERY commit you process.
- Always include `areas` as a JSON array of matched interest names.
- Process up to 20 commits per run. Note if more remain.
- Process oldest unanalyzed commits first when possible.
"""

_SUGGESTION_ENGINE_PROMPT = """\
# Techtree Suggestion Engine

A senior developer that analyzes code trends in the techtree recruitment platform
to generate actionable feature suggestions. The user (DehydratedWater / Ignacy)
works on Scores, Searches, Filters, Suggestions, AI Agents, and Calibration.

## Active PRs (work in progress)
CRITICAL: before generating suggestions, check for active-PR context passed via
thought_transfer AND fetch current open PRs yourself. You MUST NOT suggest work
that overlaps with in-progress PRs — instead suggest complementary work or next
steps that build on them, and note which PRs relate to each suggested area.

## Pipeline
1. Gather data: commit stats, recent commits per area of interest, the latest
   analysis run for continuity, and active PRs.
2. Analyze trends: which areas are most active, what other contributors are
   changing in Ignacy's areas, and patterns pointing to missing features, needed
   unification, or quality issues. Inspect specific code files when useful.
3. Generate 3-7 quality suggestions and save them as an analysis run.

## Suggestion Types
small_fix, new_score, unifying_feature, code_cleanup, better_structure,
new_filter, new_search, new_ai_agent.

## Per-suggestion Output
title, type, priority (1=critical .. 5=nice-to-have), business_context,
technical_context, spec (detailed enough to implement — files, approach, tests),
related_commits.
"""


def agents() -> list[AgentDefinition]:
    return [
        # ── Research pipeline orchestrator ──────────────────────────────────
        define_agent(
            RESEARCH_ORCHESTRATOR,
            short="run the daily YouTube/website/topic/price research pipeline",
            long=(
                "Primary research router. Fetches new videos, checks monitored"
                " websites, analyses content through topic prisms, checks product"
                " prices, then sends only the most interesting findings via"
                " Telegram."
            ),
            prompt=_RESEARCH_ORCH_PROMPT,
            # v3: research_tracking + shopping_tracking + emit_guidance skills.
            tools=[t() for t in _RESEARCH_TRACKING_TOOLS]
            + [shopping_tracker_tool(), emit_guidance_tool()],
            capability_tests=[
                CapabilityTest(
                    name="orchestrator-delivers-no-write",
                    description="The pipeline router runs research scripts + sends a summary; must not write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("emit-guidance",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="summary-keeps-recent-videos",
                    prompt="Run the daily research pipeline and summarise.",
                    evaluators=(
                        SubstringEvaluator(needle="video", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "research/video_fetcher",
            short="fetch new YouTube videos and transcripts",
            long=(
                "Fetches new videos from all tracked channels, downloads"
                " transcripts for videos missing them, and reports the counts."
            ),
            prompt=_VIDEO_FETCHER_PROMPT,
            tools=[t() for t in _RESEARCH_TRACKING_TOOLS],
            capability_tests=[
                CapabilityTest(
                    name="video-fetcher-mentions-channels",
                    description="Prompt must drive fetching from tracked channels.",
                    evaluators=(
                        SubstringEvaluator(needle="channel", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="reports-fetch-counts",
                    prompt="Fetch new videos from the tracked channels.",
                    evaluators=(
                        SubstringEvaluator(needle="transcript", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "research/website_checker",
            short="check monitored websites for changes and run search queries",
            long=(
                "Checks all active websites for content changes via hash"
                " comparison, runs the periodic search queries, and reports"
                " which sites changed plus notable search results."
            ),
            prompt=_WEBSITE_CHECKER_PROMPT,
            tools=[t() for t in _RESEARCH_TRACKING_TOOLS],
            capability_tests=[
                CapabilityTest(
                    name="website-checker-mentions-search",
                    description="Prompt must cover both change-checks and search queries.",
                    evaluators=(
                        SubstringEvaluator(needle="search", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "research/topic_analyst",
            short="analyze new video content through topic prisms",
            long=(
                "For each active topic: prepares analysis data (prism + knowledge"
                " + new transcripts), reads transcripts through the prism, and"
                " saves insights plus updated cumulative knowledge."
            ),
            prompt=_TOPIC_ANALYST_PROMPT,
            tools=[t() for t in _RESEARCH_TRACKING_TOOLS],
            capability_tests=[
                CapabilityTest(
                    name="topic-analyst-mentions-prism",
                    description="Analysis must be framed through each topic's prism.",
                    evaluators=(
                        SubstringEvaluator(needle="prism", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="analyst-saves-insights",
                    prompt="Analyze the new transcripts for active topics.",
                    evaluators=(
                        SubstringEvaluator(needle="insight", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "research/price_checker",
            short="check product prices and report alerts",
            long=(
                "Fetches the latest prices for all tracked products, checks for"
                " triggered price alerts, and reports changes and alerts."
            ),
            prompt=_PRICE_CHECKER_PROMPT,
            tools=[shopping_tracker_tool()],
            capability_tests=[
                CapabilityTest(
                    name="price-checker-mentions-alerts",
                    description="Prompt must cover triggered price alerts.",
                    evaluators=(
                        SubstringEvaluator(needle="alert", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Techtree codebase monitor ───────────────────────────────────────
        define_agent(
            TECHTREE_ORCHESTRATOR,
            short="monitor the techtree codebase: ingest, analyze, suggest, notify",
            long=(
                "Primary techtree router. Pulls + ingests new commits (exiting"
                " silently when there are none), dispatches commit analysis and"
                " suggestion generation, then notifies via email and Telegram."
                " Supports a quick-check mode that only ingests and pings."
            ),
            prompt=_TECHTREE_ORCH_PROMPT,
            # v3 granted read=True (browse repo); still no write/edit/mcp.
            permissions=ToolPermissions(read=True),
            # v3: techtree_tracking + gmail + emit_guidance + agent_context.
            tools=[
                techtree_manager_tool(),
                gmail_manager_tool(),
                emit_guidance_tool(),
            ]
            + [t() for t in _AGENT_CONTEXT_TOOLS],
            capability_tests=[
                CapabilityTest(
                    name="techtree-orch-no-write",
                    description="Read-only router with techtree/gmail/notify tools: may read, must not write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("techtree-manager",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="silent-when-no-commits",
                    prompt="Run the techtree pipeline.",
                    evaluators=(
                        SubstringEvaluator(needle="ingest", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "research/techtree_commit_analyzer",
            short="analyze new commits and detect relevant changes",
            long=(
                "Loads enabled interests, lists unanalyzed commits, and for each"
                " reads git-show + git-diff, analyzes through interest lenses, and"
                " saves analysis text plus area tags."
            ),
            prompt=_COMMIT_ANALYZER_PROMPT,
            permissions=ToolPermissions(read=True),
            tools=[techtree_manager_tool()],
            capability_tests=[
                CapabilityTest(
                    name="commit-analyzer-mentions-interests",
                    description="Analysis must be framed through configured interests.",
                    evaluators=(
                        SubstringEvaluator(needle="interest", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="analyzer-covers-risk",
                    prompt="Analyze the unanalyzed techtree commits.",
                    evaluators=(
                        SubstringEvaluator(needle="diff", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "research/techtree_suggestion_engine",
            short="generate feature suggestions from code analysis",
            long=(
                "Reads recent commit data and trends, considers active PRs to"
                " avoid duplicate work, and produces 3-7 actionable feature"
                " suggestions with full implementation specs, saved as an"
                " analysis run."
            ),
            prompt=_SUGGESTION_ENGINE_PROMPT,
            permissions=ToolPermissions(read=True),
            tools=[techtree_manager_tool()],
            capability_tests=[
                CapabilityTest(
                    name="suggestion-engine-mentions-prs",
                    description="Must consider active PRs before suggesting work.",
                    evaluators=(
                        SubstringEvaluator(needle="PR", case_sensitive=True),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="suggestions-have-specs",
                    prompt="Generate feature suggestions from recent techtree trends.",
                    evaluators=(
                        SubstringEvaluator(needle="spec", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """One distinguished path per orchestrator (tested + optimised as a unit)."""
    return [
        # Daily research pipeline: video → website → topic → price (→ summary).
        BranchTest(
            name="research/orchestrator::daily-pipeline",
            entry_agent=RESEARCH_ORCHESTRATOR,
            prompt="Run the daily research pipeline and send a summary.",
            path=(
                "research/video_fetcher",
                "research/website_checker",
                "research/topic_analyst",
                "research/price_checker",
            ),
            evaluators=(
                SubstringEvaluator(needle="video", case_sensitive=False),
            ),
        ),
        # Techtree full analysis: ingest → analyze commits → suggest (→ notify).
        BranchTest(
            name="research/techtree_orchestrator::full-analysis",
            entry_agent=TECHTREE_ORCHESTRATOR,
            prompt="Run the techtree pipeline: ingest, analyze, and suggest features.",
            path=(
                "research/techtree_commit_analyzer",
                "research/techtree_suggestion_engine",
            ),
            evaluators=(
                SubstringEvaluator(needle="commit", case_sensitive=False),
            ),
        ),
    ]
