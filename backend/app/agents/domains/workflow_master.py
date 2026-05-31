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

Generate diagrams showing agent structure and connections.

## Capabilities
- Filter vs Highlight modes for focusing on specific agents.
- Color-coding by category (goals=blue, persona=purple, workflows=green).
- Arrow connections showing Task tool invocations.
- Output: Mermaid diagram or text-based graph.

## Process
1. Scan all agent definitions.
2. Extract subagent references and Task tool calls.
3. Build the dependency graph.
4. Generate the diagram in the requested format.
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


def _pure_prompt_capability(name: str) -> CapabilityTest:
    return CapabilityTest(
        name=name,
        description="Pure-prompt subagent must not hold write/bash/edit tools itself.",
        must_not_have_tools=("bash", "write", "edit"),
    )


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
            capability_tests=[
                CapabilityTest(
                    name="workflow-master-no-arbitrary-edit",
                    description="Creates files only via its CLI tools, not the edit tool.",
                    must_not_have_tools=("edit",),
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
            capability_tests=[_pure_prompt_capability("agent-visualizer-pure-prompt")],
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
            capability_tests=[_pure_prompt_capability("documentation-generator-pure-prompt")],
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
            capability_tests=[_pure_prompt_capability("quality-reviewer-pure-prompt")],
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
