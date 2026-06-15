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

import importlib
import inspect
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from src import (
    BashToolPermission,
    ToolDefinition,
    ToolDefinitionHeader,
    ToolDefinitionLogicBash,
)

_SCRIPT_RE = re.compile(r"(scripts/[A-Za-z0-9_]+\.py)")
_IMPORT_RE = re.compile(r"from\s+(app\.tools\.[\w.]+)\s+import")
_VERB_RE = re.compile(r"""command\s*==\s*["']([\w.\-]+)["']""")


def _resolve_script_file(script: str) -> Path | None:
    """Find the on-disk `scripts/<name>.py` across the layouts we compile in:
    the container (`/app/scripts`, `_tooldefs` at `/app/backend/app/agents/`) and
    a host checkout (`<repo>/scripts`). Best-effort — returns None if not found."""
    name = Path(script).name
    roots: list[Path] = []
    try:
        roots.append(Path(__file__).resolve().parents[3])  # repo root / /app
    except IndexError:
        pass
    roots += [Path.cwd(), Path("/app")]
    for r in roots:
        cand = r / "scripts" / name
        if cand.is_file():
            return cand
    return None


@lru_cache(maxsize=None)
def command_vocab(script: str) -> str:
    """Best-effort valid `--command` verbs for a ScriptTool, so a compiled agent
    is told the EXACT verbs instead of guessing a plausible-but-wrong one
    (`list-active` for habit_manager's `list`; `get-extraction-state` for
    event_manager's `get-state`) or burning a turn on `--help` — the runtime
    tool-flail seen in the run traces.

    Source priority: the tool's `Input.command` field description when it
    enumerates verbs (contains '|'); else a scan of the tool module for
    `command == "verb"` dispatch arms (covers tools whose field description is
    bare, e.g. event_manager). Returns "" when neither yields verbs (tools with
    no command field, e.g. send_message / ralf_spawn). Cached per script."""
    try:
        f = _resolve_script_file(script)
        if not f:
            return ""
        m = _IMPORT_RE.search(f.read_text())
        if not m:
            return ""
        mod = importlib.import_module(m.group(1))
        inp = getattr(mod, "Input", None)
        fields = getattr(inp, "model_fields", {}) or {}
        if "command" in fields:
            desc = fields["command"].description or ""
            if "|" in desc:
                return desc.strip()
        verbs = sorted(set(_VERB_RE.findall(inspect.getsource(mod))))
        return "|".join(verbs)
    except Exception:  # noqa: BLE001 — vocab is advisory; never block a compile
        return ""


def script_of_tool(tool: ToolDefinition) -> str:
    """The `scripts/<name>.py` a bash ToolDefinition is scoped to, or "" for a
    non-script tool (raw command / no bash)."""
    try:
        cmds = tool.bash_tool.permission_bash.allowed_commands or []
    except AttributeError:
        return ""
    for c in cmds:
        m = _SCRIPT_RE.search(str(c))
        if m:
            return m.group(1)
    return ""


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
    # CANONICAL invocation is `uv run <script>`: it deterministically resolves
    # the project's uv env (all deps), independent of which `python` happens to
    # be on PATH. Bare `python <script>` is kept ALLOWED as a fallback — it
    # works where the venv is already active (the autoloop forces it via
    # _branch_env; prod runs in the deps-having container) — but the agent is
    # shown the uv form so it prefers the env-guaranteed one.
    uv_command = f"uv run {script}"
    command = f"python {script}"
    pos = list(examples) or [f"{uv_command} --command list"]
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
                # uv run = canonical (env-guaranteed); python = PATH-dependent
                # fallback. Both scoped to exactly this script.
                allowed_commands=[f"{uv_command}*", f"{command}*"],
            ),
            positive_examples=pos,
            negative_examples=[],
            mode_specific_rules=list(rules),
        ),
    )
