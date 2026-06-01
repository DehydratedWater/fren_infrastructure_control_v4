"""Live wiring for autoresearch — the real promote tier.

`app/agents/improve.py` is tier-agnostic: it builds the loops + fan-out and
takes injected factories. This module supplies the LIVE implementations that
actually call the models:

- `ZaiPromptRewriter` — an LLMMutatorClient that asks z.ai (glm-4.5-air) to
  rewrite an agent's system_prompt to fix its failing tests. THIS is the
  "research" — the model proposing better prompts.
- `live_agent_runner_factory` — compiles ONE candidate agent to a temp tree,
  runs its `agent_tests` through opencode (z.ai / the worker provider), and
  returns (output, tool_calls) so the framework's evaluators can score it.
- `live_branch_invoker_factory_for` — same idea for a branch (orchestrator):
  drive the entry agent through opencode and surface the tool/subagent chain.

A candidate's prompt lives in `version.definition["system_prompt"]`; we rebuild
an AgentDefinition from that dict, compile just it (primary), and run.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from app.agents.config import DEFAULT_WORKER
from app.runtime.runner import run_agent_opencode
from app.settings import get_settings
from src import (
    AgentDefinition,
    AgentRegistry,
    CompilationConfig,
    CompileScript,
    TemplateSlot,
    TemplateTree,
)
from src.testing.evaluation import ToolCallRecord


# --- the LLM "researcher": z.ai rewrites prompts ---------------------------

class ZaiPromptRewriter:
    """LLMMutatorClient backed by z.ai (the worker provider).

    `rewrite(target, guidance, context)` returns a new system prompt. The
    failures that motivated the rewrite are in `context["failures"]`.
    """

    def __init__(self, *, model: str | None = None, timeout_s: float = 120) -> None:
        s = get_settings()
        # worker model id without the provider prefix for the chat endpoint
        wm = model or s.worker_model
        self.model = wm.split("/", 1)[-1] if "/" in wm else wm
        self.api_key = os.environ.get("ZAI_API_KEY", "")
        self.base_url = os.environ.get(
            "ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4",
        ).rstrip("/")
        self.timeout_s = timeout_s

    def rewrite(
        self, target: str, guidance: str, *,
        context: dict[str, Any] | None = None, model: str | None = None,
    ) -> str:
        failures = (context or {}).get("failures") or []
        fail_text = "\n".join(f"- {json.dumps(f)[:300]}" for f in failures[:8]) or "(none recorded)"
        system = (
            "You improve AI-agent system prompts. Given a prompt and the tests it"
            " failed, return an IMPROVED system prompt that would pass them."
            " Preserve the agent's persona, tools, and intent. Do NOT add unrelated"
            " capabilities. Return ONLY the new prompt text — no preamble, no fences."
        )
        user = (
            f"GUIDANCE: {guidance}\n\n"
            f"FAILING TESTS / EVIDENCE:\n{fail_text}\n\n"
            f"CURRENT SYSTEM PROMPT:\n{target}\n\n"
            "Return the improved system prompt only."
        )
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=headers, timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except Exception:  # noqa: BLE001 — a failed rewrite just yields no candidate
            return target
        text = text.strip()
        # Guard: never return an empty prompt (the mutator would dedupe anyway).
        return text or target


# --- compiling + running ONE candidate -------------------------------------

def _compile_one(definition: dict[str, Any], target: Path) -> str:
    """Compile a single candidate agent (primary) into `target`; return its
    spawnable agent name (`<id>-primary`)."""
    agent = AgentDefinition.model_validate(definition)
    agent_id = agent.header.agent_id
    reg = AgentRegistry()
    rid = reg.register_agent(agent_id, agent, DEFAULT_WORKER.preset.to_model_parameters())
    reg.register_template(TemplateTree(
        name="cand",
        slots=[TemplateSlot(name=agent_id, default_agent_id=rid, also_compile_as_primary=True)],
    ))
    reg.create_compilation_config(CompilationConfig(name="prod", template_name="cand"))
    CompileScript(
        target=target, config="prod", factory=lambda: reg,
        variants=[DEFAULT_WORKER], clean=True,
    ).run()
    return f"{agent_id}-primary"


def _run_sync(coro):
    """Run an async coroutine from sync code, even if a loop is already running
    (the loops execute inside run_fleet's threads — each thread has no loop)."""
    return asyncio.run(coro)


def live_agent_runner_factory(definition: dict[str, Any]):
    """Build an AgentRunner that compiles + runs THIS candidate via opencode.

    Returns a callable (definition, AgentTest) -> (output_text, tool_calls).
    Compiles once per candidate (reused across that candidate's tests).
    """
    tmp = Path(tempfile.mkdtemp(prefix="oac_improve_"))
    agent_name = _compile_one(definition, tmp)
    # scripts symlink so any tool the agent calls resolves
    scripts = Path(get_settings().project_root) / "scripts"
    link = tmp / "scripts"
    try:
        if scripts.exists() and not link.exists():
            link.symlink_to(scripts)
    except OSError:
        pass

    def runner(_defn: dict[str, Any], test) -> tuple[Any, list[ToolCallRecord]]:
        prompt = test.prompt or (test.turns[0].prompt if test.turns else "")
        result = _run_sync(run_agent_opencode(
            agent_dir=tmp, agent_name=agent_name, prompt=prompt, timeout_s=180,
        ))
        return result.text, list(result.tool_calls)

    return runner


def live_branch_invoker_factory_for(entry_agent: str):
    """Return a BranchInvokerFactory for `entry_agent` that drives it live."""

    def factory(definition: dict[str, Any]):
        tmp = Path(tempfile.mkdtemp(prefix="oac_branch_"))
        agent_name = _compile_one(definition, tmp)
        scripts = Path(get_settings().project_root) / "scripts"
        link = tmp / "scripts"
        try:
            if scripts.exists() and not link.exists():
                link.symlink_to(scripts)
        except OSError:
            pass

        def invoke(test):
            from src.testing.branch import BranchTrajectory

            prompt = test.prompt or (test.turns[0].prompt if test.turns else "")
            result = _run_sync(run_agent_opencode(
                agent_dir=tmp, agent_name=agent_name, prompt=prompt, timeout_s=240,
            ))
            return BranchTrajectory(output=result.text, tool_calls=list(result.tool_calls))

        return invoke

    return factory
