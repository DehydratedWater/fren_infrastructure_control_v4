"""Authoring helpers — terse, consistent construction of fleet agents.

v3 used open-agent-compiler's fluent `AgentBuilder`. v4 builds the framework's
`AgentDefinition` Pydantic model directly, but ~105 of them would be noisy
without a helper, so `define_agent(...)` fills the fren defaults (mode, todo
strictness, empty permissions for pure-prompt agents) and takes the fields that
actually vary: the prompt, the model_class routing hint, embedded tests, and
sub/peer wiring.

Each agent carries its OWN tests (capability + agent), so the per-agent
improvement loop has a success signal — this is the "every agent is testable
and self-improving" requirement, expressed at authoring time.
"""

from __future__ import annotations

from collections.abc import Sequence

from src import (
    AgentDefinition,
    AgentHeader,
    AgentTest,
    AgentToolPermissions as ToolPermissions,
    CapabilityTest,
    ToolDefinition,
)

# Permissions: pure-prompt agents get an all-deny set (no bash/write/edit/mcp)
# — a deliberate prompt-injection carve-out matching v3's read-only
# orchestration agents.


def pure_prompt_permissions() -> ToolPermissions:
    """No tools: read=write=edit=mcp=False. Safe default for orchestrators."""
    return ToolPermissions()


def define_agent(
    agent_id: str,
    *,
    short: str,
    long: str,
    prompt: str,
    model_class: str = "default",
    tools: Sequence[ToolDefinition] = (),
    capability_tests: Sequence[CapabilityTest] = (),
    agent_tests: Sequence[AgentTest] = (),
    permissions: ToolPermissions | None = None,
    name: str | None = None,
    description: str | None = None,
) -> AgentDefinition:
    """Build one fleet agent.

    `model_class` is the split-profile routing hint (default / fast / analytical
    / vision). `short`/`long` are the usage explanations the framework requires.
    `tools` are the ToolDefinitions (from app/agents/_tools.py) the agent may
    call — they compile into its scoped bash allowlist + tool docs, turning a
    pure-prompt agent into one that can actually act (v3 parity wiring).

    Every tool-carrying agent also gets the workspace-orientation tool (plain
    `ls`): session forensics showed ~1.2k blocked `ls`/`find` calls per night —
    models orient by listing before settling into their scripts. A read-only
    `ls` is harmless; allowing it removes the 1–3 denied calls that opened
    most sessions. Pure-prompt agents stay bash-free.
    """
    if tools:
        from app.agents._tools import workspace_orientation_tool

        tools = list(tools) + [workspace_orientation_tool()]
    return AgentDefinition(
        header=AgentHeader(
            agent_id=agent_id,
            name=name or agent_id,
            description=description or short,
        ),
        usage_explanation_short=short,
        usage_explanation_long=long,
        system_prompt=prompt,
        model_class=model_class,
        extra_tools=list(tools),
        tool_permissions=permissions if permissions is not None else pure_prompt_permissions(),
        capability_tests=tuple(capability_tests),
        agent_tests=tuple(agent_tests),
    )
