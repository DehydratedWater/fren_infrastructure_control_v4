"""Live wiring for autoresearch — strong teacher tunes agents for local qwen.

The autoloop has three model roles:

- **Teacher (rewriter + judge)** — a STRONG z.ai model (GLM-5.1 by default). It
  proposes improved system prompts from failing-test evidence AND grades agent
  responses 0..1. This is the intelligence doing the "research".
- **Target (the agent being tuned)** — the LOCAL Qwen-27B (vLLM at
  192.168.0.42:8082). Candidate prompts compile + run on qwen via opencode, so
  prompts are optimised for the model that actually serves them locally.

So: GLM-5.1 teaches → Qwen-27B is the student. `app/agents/improve.py` stays
tier-agnostic; this module supplies the live implementations.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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
from src.testing.evaluation import EvaluationResult, RunContext, ToolCallRecord

_ZAI_BASE = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4").rstrip("/")


def _zai_chat(model: str, messages: list[dict], *, max_tokens: int = 4000,
              temperature: float = 0.4, timeout_s: float = 180) -> str:
    """One z.ai chat completion → assistant text ('' on any failure).

    GLM-5.1 is a reasoning model — give it generous max_tokens so reasoning
    doesn't starve the visible answer.
    """
    key = os.environ.get("ZAI_API_KEY", "")
    payload = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        resp = httpx.post(
            f"{_ZAI_BASE}/chat/completions", json=payload, headers=headers,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception:  # noqa: BLE001
        return ""


# --- teacher 1: the prompt rewriter (GLM-5.1) ------------------------------

class ZaiPromptRewriter:
    """LLMMutatorClient backed by the strong teacher model (GLM-5.1 on z.ai).

    Proposes an improved system prompt from the agent's failing-test evidence.
    Crucially it is told the agent will RUN ON QWEN-27B locally, so it should
    write prompts that a mid-size local model follows reliably (explicit,
    stepwise, unambiguous).
    """

    def __init__(self, *, model: str | None = None, timeout_s: float = 180) -> None:
        self.model = model or get_settings().autoloop_teacher_model
        self.timeout_s = timeout_s

    def rewrite(
        self, target: str, guidance: str, *,
        context: dict[str, Any] | None = None, model: str | None = None,
    ) -> str:
        failures = (context or {}).get("failures") or []
        fail_text = "\n".join(f"- {json.dumps(f)[:400]}" for f in failures[:10]) or "(none recorded)"
        system = (
            "You are a senior prompt engineer improving an AI agent's system"
            " prompt. The agent runs on a MID-SIZE LOCAL MODEL (Qwen-27B), so the"
            " prompt must be explicit, concrete, and unambiguous — spell out the"
            " expected behaviour, output shape, and any required keywords/markers."
            " Given the current prompt and the checks it FAILED, return an improved"
            " prompt that would pass them. Preserve the agent's persona, tools, and"
            " intent; do NOT invent unrelated capabilities. Return ONLY the new"
            " prompt text — no preamble, no markdown fences, no commentary."
        )
        user = (
            f"GUIDANCE: {guidance}\n\n"
            f"FAILED CHECKS / EVIDENCE (what to fix):\n{fail_text}\n\n"
            f"CURRENT SYSTEM PROMPT:\n{target}\n\n"
            "Return the improved system prompt only."
        )
        text = _zai_chat(
            model or self.model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=4000, temperature=0.5, timeout_s=self.timeout_s,
        )
        # strip accidental ``` fences
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text.strip())
        return text or target


# --- teacher 2: the judge (GLM-5.1) ----------------------------------------

class ZaiJudge:
    """JudgeClient backed by the strong teacher model.

    Grades how well an agent response fulfils a stated criterion, 0..1. This is
    the graded signal LLMJudge agent_tests use — always satisfiable and
    improvable, so a prompt rewrite can climb it (unlike brittle substring
    checks). Returns {pass, score, reasoning}.
    """

    def __init__(self, *, model: str | None = None, timeout_s: float = 120,
                 pass_threshold: float = 0.7) -> None:
        self.model = model or get_settings().autoloop_teacher_model
        self.timeout_s = timeout_s
        self.pass_threshold = pass_threshold

    def judge(self, criteria: str, target: Any, *, model: str | None = None) -> dict:
        system = (
            "You are a strict evaluator. Score how well the RESPONSE satisfies the"
            " CRITERION on a 0.0-1.0 scale (1.0 = fully satisfies, 0.0 = ignores"
            " it). Reward responses that clearly fulfil the agent's role; penalise"
            " refusals, empty/echo replies, off-task rambling, or tool-mechanics"
            " leaking into the answer. Respond with STRICT JSON only:"
            ' {"score": <float 0..1>, "reasoning": "<one sentence>"}'
        )
        user = f"CRITERION:\n{criteria}\n\nRESPONSE:\n{str(target)[:4000]}\n\nReturn the JSON."
        raw = _zai_chat(
            model or self.model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=1500, temperature=0.0, timeout_s=self.timeout_s,
        )
        score = 0.0
        reasoning = ""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                score = float(obj.get("score", 0.0))
                reasoning = str(obj.get("reasoning", ""))[:200]
            except Exception:  # noqa: BLE001
                pass
        score = max(0.0, min(1.0, score))
        return {"pass": score >= self.pass_threshold, "score": score, "reasoning": reasoning}


# --- compiling + running ONE candidate (on the LOCAL QWEN target) ----------

def _compile_one(definition: dict[str, Any], target: Path) -> str:
    """Compile a single candidate agent (primary) into `target`, then patch its
    model line to the local-qwen opencode provider so opencode runs it on the
    vLLM. Returns the spawnable agent name (`<id>-primary`)."""
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

    # Re-point the compiled primary at the LOCAL QWEN provider so it's tuned on
    # the model that serves it locally (not z.ai).
    target_model = get_settings().autoloop_target_model
    md = target / ".opencode" / "agents" / f"{agent_id}-primary.md"
    if md.exists():
        txt = md.read_text()
        txt = re.sub(r"^model:.*$", f"model: {target_model}", txt, count=1, flags=re.M)
        md.write_text(txt)
    return f"{agent_id}-primary"


def _run_sync(coro):
    return asyncio.run(coro)


def _prep_candidate(definition: dict[str, Any]) -> tuple[Path, str]:
    tmp = Path(tempfile.mkdtemp(prefix="oac_improve_"))
    agent_name = _compile_one(definition, tmp)
    scripts = Path(get_settings().project_root) / "scripts"
    link = tmp / "scripts"
    try:
        if scripts.exists() and not link.exists():
            link.symlink_to(scripts)
    except OSError:
        pass
    return tmp, agent_name


def _compiled_system_prompt(tmp: Path, agent_name: str) -> str:
    from src import load_compiled_agent

    md = tmp / ".opencode" / "agents" / f"{agent_name}.md"
    if md.exists():
        try:
            return load_compiled_agent(md).system_prompt or ""
        except Exception:  # noqa: BLE001
            return ""
    return ""


def _qwen_direct(system: str, user: str, *, max_tokens: int = 1200) -> str:
    """One completion from the local Qwen with thinking DISABLED.

    opencode's openai-compatible provider silently drops `chat_template_kwargs`,
    so Qwen3.5 keeps thinking and returns empty visible content (the mass-0 bug).
    The autoloop therefore grades via a direct call that DOES send
    enable_thinking=false — proven to return content reliably.
    """
    s = get_settings()
    base = s.local_llm_base_url or "http://192.168.0.42:8082/v1"
    key = s.vllm_api_key if s.vllm_api_key and s.vllm_api_key != "EMPTY" else ""
    payload = {
        "model": "cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # Qwen3.5 wants temperature 1.0 for optimal generation (low temp degrades
        # / can degenerate its output); pair with the model's recommended top-p/k.
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = httpx.post(
            f"{base.rstrip('/')}/chat/completions", json=payload,
            headers=headers, timeout=180,
        )
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def live_agent_runner_factory(definition: dict[str, Any]):
    """AgentRunner for the judge-test autoloop.

    Compiles THIS candidate (so the system prompt is the real, fully-rendered
    one), then runs it on the local Qwen via a DIRECT call with thinking disabled
    — the reliable path. (opencode is kept for branch/trajectory tests where tool
    dispatch is needed.) Returns (output_text, []).
    """
    tmp, agent_name = _prep_candidate(definition)
    system = _compiled_system_prompt(tmp, agent_name) or definition.get(
        "system_prompt", ""
    )

    def runner(_defn: dict[str, Any], test) -> tuple[Any, list[ToolCallRecord]]:
        prompt = test.prompt or (test.turns[0].prompt if test.turns else "")
        text = _qwen_direct(system, prompt)
        if not text.strip():  # rare stochastic blank → one retry
            text = _qwen_direct(system, prompt)
        return text, []

    return runner


def live_branch_invoker_factory_for(entry_agent: str):
    """BranchInvokerFactory that drives `entry_agent` live on the local qwen."""

    def factory(definition: dict[str, Any]):
        tmp, agent_name = _prep_candidate(definition)

        def invoke(test):
            from src.testing.branch import BranchTrajectory

            prompt = test.prompt or (test.turns[0].prompt if test.turns else "")
            result = _run_sync(run_agent_opencode(
                agent_dir=tmp, agent_name=agent_name, prompt=prompt, timeout_s=300,
            ))
            return BranchTrajectory(output=result.text, tool_calls=list(result.tool_calls))

        return invoke

    return factory
