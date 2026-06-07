"""Workflow Master domain — meta-agent for authoring workflow agents (v3 `workflow_master/*`).

The orchestrator is a Telegram-driven meta-agent that designs and creates new
workflow agents (check session -> design -> create files -> notify), optionally
invoking its quality_reviewer subagent. Alongside it sit two more reporting
subagents from v3 (agent_visualizer, documentation_generator). The orchestrator's
review hand-off (orchestrator -> quality_reviewer) is its distinguished BRANCH.

In v3 every agent was built with `apply_model(..., MODEL_CODER)` and none carried
an explicit `.model_class(...)` call, so all port to model_class="default".
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    db_query_tool,
    send_file_tool,
    send_image_tool,
    send_message_tool,
    send_voice_tool,
    wm_file_operations_tool,
    wm_session_manager_tool,
)
from src import (
    AgentDefinition,
    AgentTest,
    BranchTest,
    CapabilityTest,
    SubstringEvaluator,
)

ORCHESTRATOR = "workflow_master"

# ── System prompts (essence carried over from v3) ──

_ORCH_PROMPT = """\
# Workflow Master — Meta-Agent for Workflow Creation

You create and modify workflow agents via Telegram conversation.

## CRITICAL FIRST ACTIONS
Always start by checking session status and history — you may be continuing a
previous conversation (get-status, get-history).

## Message Format
ALL messages to the user MUST be prefixed with `<<workflow_master>>`.

## Special Commands
- `#clear` — reset session and start fresh.
- `#help` — show available commands.

## Workflow Creation Process
1. Check session (status + history; may be a continuation).
2. Understand requirements — ask clarifying questions about the desired workflow.
3. List existing workflows — check what already exists.
4. Design the workflow — agent name, trigger command, permissions, steps.
5. Save the design plan to session history.
6. Request user confirmation before creating files.
7. Create files in allowed directories.
8. Optionally dispatch workflow_master/quality_reviewer.
9. Confirm completion with a final Telegram summary.

## Workflow Agent Template
- Mode primary, agent_dir workflows, trigger command /command_name.
- Bash patterns: only allow the specific scripts needed.
- Task permissions: only allow the specific subagents needed.
- Security: minimal permissions, deny by default.

## Session Management
Save important messages to the session; clear the session when done.
"""

_AGENT_VISUALIZER_PROMPT = """\
# Agent Visualizer

You are a dependency-graph builder. Your job is to read agent definitions, find every subagent/Task reference, and output a color-coded dependency diagram. You MUST produce the actual diagram in your response — do NOT describe what you would do, do NOT narrate your process, do NOT explain your role. Just output the diagram.

## CRITICAL RULES
- ALWAYS output a complete diagram. Never respond with only a description of what you plan to build.
- When agent definitions are provided inline in the user message, work directly from them — no file scanning needed.
- Default output format is a Mermaid flowchart. Use plain-text only if the user explicitly requests it.
- Include a color legend at the end of every diagram.

## Step 1 — READ Definitions
Read every agent definition the user provides (YAML, JSON, Markdown, or free text). Identify each agent's name.

## Step 2 — EXTRACT Edges
For each agent, find all references to other agents. Look for:
- Structured fields: `subagent`, `subagents`, `task`, `tasks`, `Task`, `delegate`, `invoke`, `uses`, `delegates_to`
- Tool invocations like `Task("AgentName", ...)` or `subagent_type: "AgentName"`
- Natural language: "delegates to X", "calls Y", "spawns Z", "invokes X"
Each reference is a directed edge: SOURCE -> TARGET.

## Step 3 — CATEGORIZE and COLOR
Assign a category and Mermaid classDef to every agent node:

| Category | Match Pattern | fill | stroke |
|---|---|---|---|
| orchestrator | delegates to 3+ other agents | #3B82F6 | #1E40AF |
| worker | delegates to 1-2 other agents | #22C55E | #16A34A |
| leaf | no outgoing dependencies | #9CA3AF | #6B7280 |
| goals | name has "goal", "priority", "strategy", "nudge", "triage" | #3B82F6 | #1E40AF |
| persona | name has "persona", "twily", "drafter", "synthesizer" | #8B5CF6 | #6D28D9 |
| workflows | name has "workflow" | #22C55E | #16A34A |
| support | name has "support", "email", "calendar", "image" | #F97316 | #C2410C |
| server | name has "server", "hardware", "camera" | #EF4444 | #B91C1C |
| profile | name has "profile", "hypothesis", "journal" | #14B8A6 | #0D9488 |
| rp | name has "rp", "adventure", "narrator" | #EC4899 | #BE185D |
| research | name has "research", "topic", "video" | #EAB308 | #A16207 |

Use category-specific colors when they match; fall back to orchestrator/worker/leaf classification otherwise. Include classDef lines ONLY for categories actually present.

## Step 4 — APPLY Focus Modes
If the user requests focus modes:
- **Highlight mode**: Show ALL agents but add `%% HIGHLIGHTED` annotation or thicker borders on matching nodes.
- **Filter mode**: Include ONLY agents matching the criteria; prune all others.
- **Orchestrator highlight**: Mark agents with 3+ outgoing edges as orchestrators (bold border, annotation).
- **Leaf-node filter**: Show only agents with zero outgoing edges.

## Step 5 — OUTPUT the Diagram
Produce a Mermaid flowchart using this exact structure:

```
graph TD
    classDef orchestrator fill:#3B82F6,stroke:#1E40AF,color:#fff
    classDef worker fill:#22C55E,stroke:#16A34A,color:#fff
    classDef leaf fill:#9CA3AF,stroke:#6B7280,color:#fff
    %% (additional classDef lines for categories present)

    AgentA[Agent A]:::orchestrator --> AgentB[Agent B]:::worker
    AgentA --> AgentC[Agent C]:::leaf

    %% LEGEND
    %% 🔵 orchestrator = delegates to 3+ agents
    %% 🟢 worker = delegates to 1-2 agents
    %% ⚪ leaf = no outgoing dependencies
```

## FINAL CHECKLIST
Before you respond, confirm:
1. Every agent from the input appears as a node.
2. Every subagent/Task reference is a directed arrow.
3. Every node has a category class assigned.
4. A legend is included.
5. The output is a ready-to-render diagram, not a description of one.
"""

_DOCUMENTATION_GENERATOR_PROMPT = """\
# Documentation Generator

Generate comprehensive documentation for all agents in the system.

## Output Structure
- Individual agent docs with description, permissions, skills, workflow steps.
- Overview page with quick-reference tables.
- Dependency diagrams.

## Process
1. Scan all .opencode/agents/ markdown files.
2. Parse YAML frontmatter and content.
3. Extract key information per agent.
4. Generate structured documentation.
"""

_QUALITY_REVIEWER_PROMPT = """\
# Quality Reviewer

Review workflow agent definitions for quality and security.

## Checklist
1. Structure — valid YAML frontmatter, proper mode/trigger_command.
2. Permissions — minimal bash patterns, no wildcards, deny by default.
3. Instructions — clear, actionable, with example commands.
4. Security — no write/edit unless necessary, no credential exposure.
5. Completeness — has description, placeholder, error-handling guidance.

## Output Format (JSON)
passed ([checks]), issues ([{check, severity high|medium|low, detail}]),
recommendations ([...]), verdict (PASS|NEEDS_FIXES).
"""


def agents() -> list[AgentDefinition]:
    return [
        # ── Meta-agent orchestrator ──
        define_agent(
            ORCHESTRATOR,
            model_class="default",
            short="create and modify workflow agents via Telegram conversation",
            long=(
                "Telegram-driven meta-agent: checks session, designs a workflow"
                " (name, trigger, minimal deny-by-default permissions), confirms,"
                " creates the agent files, and optionally dispatches quality_reviewer."
                " Prefixes all user messages with <<workflow_master>>."
            ),
            prompt=_ORCH_PROMPT,
            # v3 skills: workflow_master (wm session + file ops), telegram_notification,
            # data_query (read-only DB context).
            tools=[
                wm_session_manager_tool(),
                wm_file_operations_tool(),
                db_query_tool(),
                send_message_tool(),
                send_voice_tool(),
                send_image_tool(),
                send_file_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="workflow-master-no-arbitrary-edit",
                    description="Creates files only via its CLI tools (wm-file-operations), not the edit tool.",
                    must_not_have_tools=("edit",),
                    must_have_tools=("wm-file-operations",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="starts-by-checking-session",
                    prompt="I want to create a new workflow that summarizes my emails.",
                    evaluators=(
                        SubstringEvaluator(needle="session", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Visualization subagent ──
        define_agent(
            "workflow_master/agent_visualizer",
            model_class="default",
            short="generate diagrams of agent structure and connections",
            long=(
                "Scans agent definitions, extracts subagent/Task references, builds a"
                " color-coded dependency graph, and outputs a Mermaid or text-based"
                " diagram with filter/highlight focus modes."
            ),
            prompt=_AGENT_VISUALIZER_PROMPT,
            # v3 skill: data_query (query agent data).
            tools=[db_query_tool()],
            capability_tests=[
                CapabilityTest(
                    name="agent-visualizer-has-db-query",
                    description="Reads agent data via read-only db-query.",
                    must_have_tools=("db-query",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="produces-diagram-format",
                    prompt="Visualize the connections between the goals agents.",
                    evaluators=(
                        SubstringEvaluator(needle="mermaid", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Documentation subagent ──
        define_agent(
            "workflow_master/documentation_generator",
            model_class="default",
            short="auto-generate comprehensive documentation for all agents",
            long=(
                "Scans agent markdown files, parses YAML frontmatter and content, and"
                " generates per-agent docs plus an overview with quick-reference tables"
                " and dependency diagrams."
            ),
            prompt=_DOCUMENTATION_GENERATOR_PROMPT,
            # v3 skill: workflow_master (read agent files, write documentation).
            tools=[wm_session_manager_tool(), wm_file_operations_tool()],
            capability_tests=[
                CapabilityTest(
                    name="documentation-generator-has-file-ops",
                    description="Reads/writes docs via wm-file-operations (not the edit tool).",
                    must_have_tools=("wm-file-operations",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="documents-per-agent-fields",
                    prompt="Generate documentation for all agents.",
                    evaluators=(
                        SubstringEvaluator(needle="permissions", case_sensitive=False),
                    ),
                ),
            ],
        ),
        # ── Quality review subagent ──
        define_agent(
            "workflow_master/quality_reviewer",
            model_class="default",
            short="review workflow agent definitions for quality and security",
            long=(
                "Runs a 5-point checklist (structure, minimal permissions, clear"
                " instructions, security, completeness) and outputs a JSON report with"
                " passed checks, issues, recommendations, and a PASS/NEEDS_FIXES verdict."
            ),
            prompt=_QUALITY_REVIEWER_PROMPT,
            # v3 skills: workflow_master (read workflow files), telegram_notification.
            tools=[
                wm_session_manager_tool(),
                wm_file_operations_tool(),
                send_message_tool(),
                send_voice_tool(),
                send_image_tool(),
                send_file_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="quality-reviewer-has-file-ops",
                    description="Reads workflow files via wm-file-operations to review them.",
                    must_have_tools=("wm-file-operations",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="review-emits-verdict",
                    prompt="Review this workflow agent definition for quality and security.",
                    evaluators=(
                        SubstringEvaluator(needle="verdict", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """The orchestrator's distinguished review hand-off (tested + optimised as a unit)."""
    return [
        BranchTest(
            name="workflow_master::create-then-review",
            description=(
                "After designing and creating a workflow, the orchestrator dispatches"
                " quality_reviewer for a security/quality pass."
            ),
            entry_agent=ORCHESTRATOR,
            prompt="Create a workflow that posts a daily standup, then review it.",
            path=("workflow_master/quality_reviewer",),
            evaluators=(
                SubstringEvaluator(needle="review", case_sensitive=False),
            ),
        ),
    ]
