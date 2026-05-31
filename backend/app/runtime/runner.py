"""Run a compiled agent — the worker execution layer.

Two backends, selected by `settings.execution_backend`:

- **opencode** (host): shells out to `opencode run --agent <name> --format json`
  with the per-tree `XDG_DATA_HOME`/`PWD` and `stdin=DEVNULL` (the discovered-
  the-hard-way invariants from v2's YT/CES work — opencode hangs on a TTY-less
  stdin and produces no output from `/tmp`). Its JSON event stream yields both
  the assistant text AND the chain of tool / subagent dispatches — exactly the
  trajectory a branch test needs, so this backend is the LIVE promote tier.
- **direct** (host or container): reads the compiled `.md` (model + system
  prompt via the framework's `load_compiled_agent`) and calls the provider over
  httpx. Works anywhere, but only yields text (no tool trajectory).

`parse_opencode_events` is a pure function (no I/O) so the stream parsing is
unit-tested without a live opencode.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import load_compiled_agent
from src.testing.evaluation import ToolCallRecord


@dataclass
class AgentRunResult:
    text: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    raw_stdout: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --- pure event-stream parser (testable without opencode) -------------------

def _tool_name(part: dict) -> str:
    """Best-effort tool/subagent name from a tool-part, across shapes."""
    tool = part.get("tool")
    if isinstance(tool, dict):
        n = tool.get("name")
        if isinstance(n, str) and n:
            return n
    for key in ("tool", "name", "agent"):
        v = part.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def parse_opencode_events(stdout: str) -> tuple[str, list[ToolCallRecord]]:
    """Extract (assistant_text, tool_calls) from opencode's `--format json`.

    Each line is a JSON event with a `part` object. Text parts carry
    `part.text`; tool/subagent parts carry a type marker + a tool/agent name.
    Tolerant of shape drift: unknown lines are skipped, and several plausible
    tool-part shapes are accepted (best-effort on the chain; text is exact).
    """
    text_parts: list[str] = []
    calls: list[ToolCallRecord] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = ev.get("part") if isinstance(ev, dict) else None
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type", ""))
        if isinstance(part.get("text"), str) and "tool" not in ptype:
            text_parts.append(part["text"])
            continue
        # tool / subagent dispatch part — accept a few shapes
        if "tool" in ptype or ptype in ("subagent", "agent", "step"):
            name = _tool_name(part)
            if name:
                args = part.get("args") or part.get("input") or {}
                calls.append(
                    ToolCallRecord(name=name, args=args if isinstance(args, dict) else {})
                )
    text = "\n".join(text_parts) if text_parts else stdout
    return text, calls


# --- opencode backend -------------------------------------------------------

async def run_agent_opencode(
    *, agent_dir: Path, agent_name: str, prompt: str, timeout_s: float = 120,
) -> AgentRunResult:
    env = os.environ.copy()
    env["XDG_DATA_HOME"] = str(agent_dir / ".opencode" / "data")
    env["PWD"] = str(agent_dir)
    cmd = ["opencode", "run", "--agent", agent_name, "--format", "json", prompt]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(agent_dir), env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return AgentRunResult(error=f"timeout after {timeout_s}s")
    except FileNotFoundError:
        return AgentRunResult(error="opencode binary not found")

    stdout = out_b.decode("utf-8", errors="replace")
    text, calls = parse_opencode_events(stdout)
    err = None
    if proc.returncode != 0 and not text.strip():
        err = f"opencode exit {proc.returncode}: {err_b.decode('utf-8', 'replace')[:500]}"
    return AgentRunResult(text=text, tool_calls=calls, raw_stdout=stdout, error=err)


# --- direct backend ---------------------------------------------------------

def _compiled_md_path(agent_dir: Path, agent_name: str) -> Path:
    return agent_dir / ".opencode" / "agents" / f"{agent_name}.md"


async def run_agent_direct(
    *, agent_dir: Path, agent_name: str, prompt: str, timeout_s: float = 120,
    base_url: str | None = None, api_key: str | None = None,
) -> AgentRunResult:
    import httpx

    md_path = _compiled_md_path(agent_dir, agent_name)
    if not md_path.exists():
        return AgentRunResult(error=f"compiled agent not found: {md_path}")
    agent = load_compiled_agent(md_path)
    model = agent.model or os.environ.get("WORKER_MODEL", "zai-coding-plan/glm-4.5-air")

    url = (base_url or os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"))
    key = api_key or os.environ.get("ZAI_API_KEY", "")
    payload: dict[str, Any] = {
        "model": model.split("/", 1)[-1] if "/" in model else model,
        "messages": [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                f"{url.rstrip('/')}/chat/completions", json=payload, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return AgentRunResult(error=f"{type(exc).__name__}: {exc}")
    text = (
        data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(data, dict) else ""
    )
    return AgentRunResult(text=text or "", raw_stdout=json.dumps(data)[:4000])


async def run_agent(
    *, agent_dir: Path, agent_name: str, prompt: str,
    backend: str = "opencode", timeout_s: float = 120,
) -> AgentRunResult:
    """Run a compiled agent under the chosen backend."""
    if backend == "direct":
        return await run_agent_direct(
            agent_dir=agent_dir, agent_name=agent_name, prompt=prompt, timeout_s=timeout_s,
        )
    return await run_agent_opencode(
        agent_dir=agent_dir, agent_name=agent_name, prompt=prompt, timeout_s=timeout_s,
    )
