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
import re
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
    run_id: str = ""
    # The ORDERED, interleaved trajectory of the run as it streamed: narration
    # text, tool invocations, and tool results in chronological (stream) order.
    # Each item is `{"kind": "text"|"tool"|"result", ...}`. Used by the dashboard
    # to replay the agent's actions IN ORDER; the flat `text`/`tool_calls` above
    # stay for back-compat (autoloop scoring consumes those).
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    # The DENIED/blocked tool attempts in this run as (tool_name, reason) pairs —
    # the tool-discipline signal forwarded to the judge + rewriter so the loop
    # learns to stop flailing on forbidden tools. Empty on a clean run.
    blocked: list[tuple[str, str]] = field(default_factory=list)

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


def _tool_call_from_part(part: dict) -> ToolCallRecord | None:
    """Build a ToolCallRecord from a tool/subagent part, or None if unnamed.

    The bash command an agent ran lives in `state.input.command` (not
    `part.args`); we merge it in so the DELIVERY CONTRACT check can see e.g. the
    `python scripts/emit_guidance.py …` call (the only delivery mechanism) and
    parse its payload. `.error` carries the DENIED/blocked or error reason so the
    tool-discipline signal is preserved at the call level.
    """
    name = _tool_name(part)
    if not name:
        return None
    args = part.get("args") or part.get("input") or {}
    state = part.get("state") if isinstance(part.get("state"), dict) else {}
    state_input = state.get("input") if isinstance(state.get("input"), dict) else {}
    if not isinstance(args, dict) or not args:
        args = dict(state_input) if state_input else (args if isinstance(args, dict) else {})
    elif state_input and "command" not in args:
        args = {**args, **state_input}
    reason = str(state.get("error") or "")
    out = str(state.get("output") or "")
    err = None
    if "prevents you from using" in reason or "prevents you from using" in out:
        err = (reason or out)[:200]
    elif reason:
        err = reason[:200]
    return ToolCallRecord(name=name, args=args if isinstance(args, dict) else {}, error=err)


def parse_opencode_trajectory(
    stdout: str,
) -> tuple[str, list[ToolCallRecord], list[dict[str, Any]]]:
    """Parse opencode's `--format json` stream into text + calls + an ORDERED trajectory.

    Each line is a JSON event with a `part` object. Text parts carry
    `part.text`; tool/subagent parts carry a type marker, a tool/agent name and a
    streaming `state` (input, output, status, error). Returns:

      * ``text`` — the joined assistant narration (exact),
      * ``calls`` — the flat ordered list of tool calls (back-compat),
      * ``trajectory`` — the interleaved chronological timeline as a list of
        ``{"kind": "text"|"tool"|"result", ...}`` items in stream order.

    A tool part streams multiple times under one stable ``callID`` (pending →
    completed). We emit a ``tool`` item the FIRST time a callID is seen (to fix
    its position in the timeline) and a ``result`` item once the part finishes
    (carrying output/error/status). Both are refreshed with the latest state, so
    the final command/output/error are what render. Tolerant of shape drift:
    unknown lines are skipped and several tool-part shapes are accepted.
    """
    text_parts: list[str] = []
    calls: list[ToolCallRecord] = []
    trajectory: list[dict[str, Any]] = []
    # callID → (tool_item, result_item|None) so streamed updates patch in place
    seen: dict[str, dict[str, Any]] = {}
    anon = 0  # synthetic key for tool parts without a callID

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
            txt = part["text"]
            text_parts.append(txt)
            trajectory.append({"kind": "text", "text": txt})
            continue

        # tool / subagent dispatch part — accept a few shapes
        if "tool" in ptype or ptype in ("subagent", "agent", "step"):
            rec = _tool_call_from_part(part)
            if rec is None:
                continue
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            status = str(state.get("status") or "")
            output = state.get("output")
            cid = part.get("callID")
            key = cid if isinstance(cid, str) and cid else f"__anon_{anon}"
            command = ""
            if isinstance(rec.args, dict):
                command = str(rec.args.get("command") or "")

            if key not in seen:
                # First sight: append the tool item in stream order. Flat calls
                # list is appended ONCE per call (preserves back-compat counts).
                calls.append(rec)
                tool_item: dict[str, Any] = {
                    "kind": "tool", "name": rec.name, "command": command,
                    "args": dict(rec.args) if isinstance(rec.args, dict) else {},
                    "error": rec.error,
                }
                trajectory.append(tool_item)
                seen[key] = {"tool": tool_item, "result": None, "call_idx": len(calls) - 1}
                if key.startswith("__anon_"):
                    anon += 1
            else:
                # Streamed update for a known call: refresh command/error/args in
                # place and update the flat call record too.
                ti = seen[key]["tool"]
                ti["name"] = rec.name
                if command:
                    ti["command"] = command
                ti["args"] = dict(rec.args) if isinstance(rec.args, dict) else ti.get("args", {})
                if rec.error:
                    ti["error"] = rec.error
                calls[seen[key]["call_idx"]] = rec

            # Emit / refresh a result item once the part carries output, an error
            # or a terminal status.
            has_result = (
                (output is not None and str(output) != "")
                or bool(rec.error) or status in ("completed", "error")
            )
            if has_result:
                res = seen[key]["result"]
                payload = {
                    "kind": "result", "name": rec.name,
                    "output": ("" if output is None else str(output)),
                    "error": rec.error, "status": status,
                }
                if res is None:
                    trajectory.append(payload)
                    seen[key]["result"] = payload
                else:
                    res.update(payload)

    if text_parts:
        text = "\n".join(text_parts)
    elif stdout.lstrip().startswith("{"):
        # No text part found but stdout is the JSON event stream — returning the
        # raw JSON would poison downstream consumers (judges, parsers). The agent
        # simply produced no assistant text (e.g. only tool calls, or a truncated
        # run). Return empty so callers treat it as "no answer".
        text = ""
    else:
        # Non-JSON stdout (e.g. a plain error message) — surface it.
        text = stdout
    return text, calls, trajectory


def parse_opencode_events(stdout: str) -> tuple[str, list[ToolCallRecord]]:
    """Extract (assistant_text, tool_calls) from opencode's `--format json`.

    Back-compat wrapper over :func:`parse_opencode_trajectory` — drops the
    ordered trajectory and returns the flat ``(text, tool_calls)`` the autoloop
    scoring still consumes.
    """
    text, calls, _trajectory = parse_opencode_trajectory(stdout)
    return text, calls


# --- opencode backend -------------------------------------------------------

def subagent_dispatch_chain(stdout: str) -> list[ToolCallRecord]:
    """Extract the ORCHESTRATOR's sub-agent dispatch chain from the event stream.

    v4 orchestrators spawn sub-agents via `bash … opencode_manager.py run --agent
    <name> …` (primary spawn), not the Task tool — so the raw tool calls are all
    `bash` and the real chain is hidden in the bash command. Pull the dispatched
    agent names out so branch/path grading can see the trajectory.
    """
    chain: list[ToolCallRecord] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{") or "--agent" not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = ev.get("part") if isinstance(ev, dict) else None
        if not isinstance(part, dict):
            continue
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        inp = state.get("input") if isinstance(state.get("input"), dict) else {}
        cmd = inp.get("command") or part.get("args", {}).get("command") or ""
        if "--agent" not in cmd:
            continue
        m = re.search(r"--agent\s+[\"']?([A-Za-z0-9_/.-]+)", cmd)
        if m:
            chain.append(ToolCallRecord(name=m.group(1), args={"via": "spawn"}))
    return chain


def blocked_tool_details(stdout: str) -> list[tuple[str, str]]:
    """The DENIED tool attempts in this session as ``(tool_name, reason)`` pairs.

    Agents are compiled with an allow-list (`python scripts/<their tools>.py`).
    When a tool fails, Qwen tends to debug-flail on forbidden commands
    (`pip install`, `which python`, `ls`, `python3 -c …`) — all denied, retried
    repeatedly, wasting the turn and tanking the score. Returns the tool NAME and
    the deny reason for each so the judge / prompt-rewriter learns WHICH forbidden
    tools the model flailed on (not just a count), and can rewrite the prompt to
    avoid them. Name is best-effort across part shapes (falls back to "?").
    """
    out: list[tuple[str, str]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{") or "prevents you from using" not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = ev.get("part") if isinstance(ev, dict) else None
        if not isinstance(part, dict):
            continue
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        reason = str(state.get("output") or state.get("error") or "")
        if "prevents you from using" not in reason:
            continue
        out.append((_tool_name(part) or "?", reason.strip()[:200]))
    return out


def blocked_tool_attempts(stdout: str) -> int:
    """Count tool calls the permission policy DENIED in this session.

    Thin count over :func:`blocked_tool_details` — kept for the existing live
    smoke that asserts the count stays low.
    """
    return len(blocked_tool_details(stdout))


def opencode_errors(stdout: str) -> list[str]:
    """Pull `{"type":"error", ...}` messages out of the JSON event stream.

    opencode reports failures (e.g. `Agent not found`, provider/auth errors) as
    error events, not a non-zero exit. These MUST be surfaced — swallowing them
    as "empty assistant text" is what made a fleet of agent-discovery failures
    masquerade as the model "returning nothing" and score every affected agent 0.
    """
    out: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{") or '"type":"error"' not in line.replace(" ", ""):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "error":
            e = ev.get("error") or {}
            msg = ((e.get("data") or {}).get("message")
                   if isinstance(e, dict) else None) or json.dumps(e)[:200]
            out.append(str(msg))
    return out


async def run_agent_opencode(
    *, agent_dir: Path, agent_name: str, prompt: str, timeout_s: float = 120,
    extra_env: dict[str, str] | None = None,
) -> AgentRunResult:
    env = os.environ.copy()
    env["XDG_DATA_HOME"] = str(agent_dir / ".opencode" / "data")
    env["PWD"] = str(agent_dir)
    # Custom context the compiled agent's own scripts read at runtime
    # (e.g. FREN_RUN_ID / FREN_MSG_HEADER / FREN_CLEARANCE / FREN_MODEL_POSTFIX).
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
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
    text, calls, trajectory = parse_opencode_trajectory(stdout)
    err = None
    # surface opencode error events (agent-not-found, provider errors, …) — never
    # let them pass as "empty text" (see opencode_errors docstring).
    ev_errors = opencode_errors(stdout)
    if ev_errors:
        err = "opencode error: " + " | ".join(ev_errors[:2])
    elif proc.returncode != 0 and not text.strip():
        err = f"opencode exit {proc.returncode}: {err_b.decode('utf-8', 'replace')[:500]}"
    # Surface the denied/blocked tool attempts so the evaluator can forward the
    # tool-discipline signal to the judge + rewriter (close the self-correction
    # loop on flailing). Computed once here; consumers read result.blocked.
    return AgentRunResult(
        text=text, tool_calls=calls, raw_stdout=stdout, error=err,
        blocked=blocked_tool_details(stdout), trajectory=trajectory,
    )


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
