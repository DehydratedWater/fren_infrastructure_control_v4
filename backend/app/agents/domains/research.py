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
from src import (
    AgentDefinition,
    AgentTest,
    AgentToolPermissions as ToolPermissions,
    BranchTest,
    CapabilityTest,
    SubstringEvaluator,
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

Analyze new video content through each topic's prism:
1. List all active research topics.
2. For each topic, prepare analysis data (the topic's prism + existing knowledge
   + new transcripts).
3. Read the transcripts carefully through the lens of the topic's prism and
   identify new insights, patterns, and relevant information.
4. Save your analysis and update the cumulative knowledge / key facts.

You ARE the analyst. Read the transcript data and reason about it yourself,
focused on what the topic's prism asks you to look for. Report new findings per
topic.
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

## Job
1. Load all enabled interests (paths, keywords, custom instructions) to know
   what areas to focus on.
2. List commits with empty analysis (process up to ~20 at a time).
3. For each unanalyzed commit: view it (git-show), view the diff (git-diff),
   analyze it through each relevant interest lens, and save the analysis plus
   area tags. Flag commits touching areas where Ignacy works, and note when
   another contributor modifies code Ignacy also works on. For very large diffs,
   restrict git-diff to key files.

## Analysis Format (per commit)
- What changed: files modified, nature of the change.
- Why it matters: business impact, technical implications.
- Relevance: which interests it touches and why.
- Risk: could this break related functionality?
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
            capability_tests=[
                CapabilityTest(
                    name="orchestrator-is-pure-router",
                    description="The pipeline router must not hold write/bash tools itself.",
                    must_not_have_tools=("bash", "write", "edit"),
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
            capability_tests=[
                CapabilityTest(
                    name="techtree-orch-no-write",
                    description="Read-only router: may read, must not write/edit.",
                    must_not_have_tools=("write", "edit"),
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
