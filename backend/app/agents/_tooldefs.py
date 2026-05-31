"""ToolDefinition factory — bridge a ScriptTool/script to a compile-time tool.

v3 wired agents to scripts via `_tools.py` `_build(name, desc, HandlerCls,
"scripts/x.py")` (v1's ToolBuilder.from_handler). This is the v2 equivalent: it
produces a framework `ToolDefinition` whose `bash_tool` carries a
`BashToolPermission` scoped to that one script command, so a compiled agent gets
exactly the bash allowlist it needs — nothing more. Attaching these to an
agent's `extra_tools` (directly or via a skill) is what turns a pure-prompt
agent into one that can actually call its tools (the v3 parity wiring).

The command convention matches how the runner invokes scripts:
`python scripts/<tool>.py ...` from the project root.
"""

from __future__ import annotations

from collections.abc import Sequence

from src import (
    BashToolPermission,
    ToolDefinition,
    ToolDefinitionHeader,
    ToolDefinitionLogicBash,
)


def build_tool(
    name: str,
    description: str,
    script: str,
    *,
    note: str = "",
    rules: Sequence[str] = (),
    examples: Sequence[str] = (),
) -> ToolDefinition:
    """A bash ToolDefinition that allows exactly `python <script> ...`.

    `name` is the tool's reference name (e.g. "priority-manager"); `script` is
    the repo-relative path (e.g. "scripts/priority_manager.py"). The agent that
    carries this tool may run that script and only that script via bash.
    """
    if note:
        description = f"{description} {note}"
    command = f"python {script}"
    pos = list(examples) or [f"{command} --command list"]
    return ToolDefinition(
        header=ToolDefinitionHeader(
            name=name,
            description=description,
            usage_explanation_long=description,
            usage_explanation_short=description,
            rules=list(rules),
        ),
        bash_tool=ToolDefinitionLogicBash(
            permission_bash=BashToolPermission(
                tool_name="bash",
                value="allow",
                allowed_commands=[f"{command}*"],
            ),
            positive_examples=pos,
            negative_examples=[],
            mode_specific_rules=list(rules),
        ),
    )
