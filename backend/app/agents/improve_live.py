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
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
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

# (teacher GLM calls now run through opencode, not a direct z.ai endpoint)

# The production DELIVERY CONTRACT, stated as a hard rule for BOTH the teacher
# (so a rewrite never drops it) and the compiled candidate (so the model knows to
# obey it). An agent's plain assistant text is INVISIBLE to the user in
# production; the ONLY mechanism that delivers a message is calling
# `python scripts/emit_guidance.py`. The evaluator enforces it; this text teaches
# it. Only applied to DELIVERY agents (those whose allow-list permits
# emit_guidance.py).
DELIVERY_CONTRACT_RULE = (
    "DELIVERY CONTRACT (HARD RULE): The agent's assistant text is INVISIBLE to"
    " the user. It MUST call `python scripts/emit_guidance.py` to deliver its"
    " message to the user — that is the ONLY mechanism that reaches the user."
    " NEVER remove, weaken, or omit the emit_guidance.py delivery instruction"
    " when rewriting — preserve or strengthen it. A prompt that loses it is a"
    " regression that delivers nothing in production."
)


def _zai_chat(model: str, messages: list[dict], *, max_tokens: int = 4000,
              temperature: float = 0.4, timeout_s: float = 180) -> str:
    """One teacher (GLM) completion — RUN THROUGH OPENCODE, never the raw API.

    z.ai's coding-plan is licensed for use via opencode; hammering
    `api.z.ai/chat/completions` directly risks an account ban. So the teacher
    (prompt rewriter, judge, probe writer) runs as an opencode agent on
    `zai-coding-plan/<model>`, exactly like the student runs on qwen — the only
    difference is the model. Returns assistant text ('' on failure).
    """
    model_ref = model if "/" in model else f"zai-coding-plan/{model}"
    ws, agent_name = _ensure_teacher_agent(model_ref)
    # opencode agents take their system prompt from the compiled .md, so fold the
    # caller's system + user messages into a single prompt turn.
    sys_txt = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    usr_txt = "\n\n".join(m["content"] for m in messages if m.get("role") != "system")
    prompt = (f"{sys_txt}\n\n---\n\n{usr_txt}" if sys_txt else usr_txt)
    for attempt in range(3):
        result = _run_sync(run_agent_opencode(
            agent_dir=ws, agent_name=agent_name, prompt=prompt, timeout_s=timeout_s,
        ))
        text = str(result.text).strip()
        if text and not result.error:
            return text
        time.sleep(1.5 * (attempt + 1))
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
            " intent; do NOT invent unrelated capabilities.\n"
            + DELIVERY_CONTRACT_RULE
            + "\nReturn ONLY the new prompt text — no preamble, no markdown fences,"
            " no commentary."
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
    # TOOL-DISCIPLINE guard: stop the model from debug-flailing on forbidden
    # commands (ls/find/pip/which/python3/absolute paths) when a tool errors —
    # those are denied by the allow-list and waste the whole turn (~1800 denied
    # calls observed fleet-wide), tanking the score for no real reason.
    _guard = (
        "\n\nTOOL DISCIPLINE (strict): You may ONLY run the `python scripts/*.py`"
        " tools listed for you, with that EXACT form (relative `scripts/...`,"
        " interpreter `python`). NEVER run ls, find, cat, which, pip, python3,"
        " absolute paths, env inspection, or any install/debug command — they are"
        " blocked and waste the turn. If a tool returns an error, state the failure"
        " in ONE short line and continue with what you have; do NOT retry"
        " variations or try to inspect/fix the environment."
        " Do NOT read source files, glob/list the codebase, or explore the"
        " filesystem to 'understand' your tools — their usage is documented above;"
        " just call them. Pointless reads/globs of files that may not exist waste"
        " the turn."
    )
    # DELIVERY CONTRACT postamble: for an agent whose allow-list permits
    # emit_guidance.py but whose prompt does NOT already instruct emit_guidance,
    # append the STRONG DELIVERY_POSTAMBLE (the exact `python scripts/emit_guidance.py
    # --data '{...}'` invocation, modelled on goals/evening_focus) so optimisation
    # tunes WITH the working delivery contract — the SAME postamble production ships.
    # A weak generic rule did NOT get qwen to comply; this concrete one does. The
    # evaluator then enforces the call actually happens. with_delivery_postamble is
    # idempotent + a no-op for agents that already instruct emit, so it never
    # double-adds.
    from app.agents.improve import with_delivery_postamble, with_skip_clause
    agent = agent.model_copy(update={"postamble": (agent.postamble or "") + _guard})
    agent = with_delivery_postamble(agent)
    agent = with_skip_clause(agent)
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


# --- Approach A: stable opencode project workspace + warm server -----------
# Mirrors v3's `scripts/opencode_manager.py`: candidates are graded by running
# `opencode run` from a STABLE project dir (carrying opencode.json so opencode
# treats it as a project) with FLAT agent names — the two things that make agent
# discovery reliable. An ad-hoc temp dir with nested/slashed names yields
# "Agent not found" ~50-70% of the time, which (silently swallowed as empty
# output) was the real cause of the autoloop's mass-zeros. A warm dedicated
# opencode web server keeps the project + model hot for high-concurrency runs.

_WS_PORT_BASE = int(os.environ.get("AUTOLOOP_OPENCODE_PORT", "4097"))


def _run_ns() -> str:
    """The run-isolation namespace (slug-safe). Empty = base single-run layout."""
    raw = get_settings().autoloop_run_namespace or ""
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", raw).strip("-")


def _ws_port() -> int:
    """Per-namespace opencode server port so parallel runs don't fight over one
    warm server. Deterministic offset from the base port."""
    ns = _run_ns()
    if not ns:
        return _WS_PORT_BASE
    return _WS_PORT_BASE + 1 + (sum(ord(c) for c in ns) % 800)
_ws_lock = threading.Lock()
_ws_ready = False


def _project_root() -> Path:
    return Path(get_settings().project_root)


def _project_opencode_root() -> Path:
    """The dir holding opencode.json (walk up from project_root)."""
    start = _project_root()
    for d in (start, *start.parents):
        if (d / "opencode.json").exists():
            return d
    return start


def _workspace() -> Path:
    # Base (no namespace): run candidates from the real opencode PROJECT ROOT so
    # the agent is discovered and sessions land in <project>/.opencode/data.
    #
    # Namespaced run (FREN_AUTOLOOP_NS set): use a DEDICATED, fully self-contained
    # workspace at <project>/.oac/autoloop_ws/<ns>/ — its own opencode.json,
    # scripts symlink, .opencode/agents (model-specific compiled frontmatter) and
    # .opencode/data. This is what lets N loops of the SAME agent on DIFFERENT
    # models run in parallel: each model's namespace gets its own compiled
    # candidates + opencode server (see _ws_port) + data, so they never collide.
    ns = _run_ns()
    if not ns:
        return _project_opencode_root()
    return _project_opencode_root() / ".oac" / "autoloop_ws" / ns


def _server_healthy(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=2.0).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _ensure_server(ws: Path) -> None:
    """Warm a dedicated opencode web server for the workspace (best-effort).

    `opencode run` does not attach to it, but it keeps the project registered and
    the model hot, mirroring v3's proven high-concurrency setup. A separate port
    from any production server avoids collisions.
    """
    if _server_healthy(_ws_port()):
        return
    env = os.environ.copy()
    env["XDG_DATA_HOME"] = str(ws / ".opencode" / "data")
    try:
        subprocess.Popen(
            ["opencode", "web", "--hostname", "127.0.0.1", "--port", str(_ws_port())],
            cwd=str(ws), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        return
    for _ in range(30):
        if _server_healthy(_ws_port()):
            return
        time.sleep(1)


def _ensure_workspace() -> Path:
    """Create/refresh the stable project workspace once (thread-safe)."""
    global _ws_ready
    ws = _workspace()
    root = _project_opencode_root()
    with _ws_lock:
        (ws / ".opencode" / "agents").mkdir(parents=True, exist_ok=True)
        # opencode.json => opencode treats ws as a project => reliable discovery.
        # For a namespaced ws this is a fresh subdir, so the config + scripts are
        # copied/linked in from the real opencode root (for the base ws they
        # already exist and the copy is a no-op).
        src_cfg = root / "opencode.json"
        dst_cfg = ws / "opencode.json"
        if src_cfg.exists() and src_cfg.resolve() != dst_cfg.resolve():
            shutil.copy(src_cfg, dst_cfg)
        link = ws / "scripts"
        scripts = root / "scripts"
        if scripts.exists() and not link.exists() and scripts.resolve() != link.resolve():
            try:
                link.symlink_to(scripts)
            except OSError:
                pass
        if not _ws_ready:
            # clear last run's candidate agents (once, before any worker compiles)
            for old in (ws / ".opencode" / "agents").glob("cand_*.md"):
                old.unlink(missing_ok=True)
            _ensure_server(ws)
            _warmup_discovery(ws)
            _ws_ready = True
    return ws


def _warmup_discovery(ws: Path) -> None:
    """Prime the warm server's agent discovery before grading starts.

    A freshly (re)started opencode server has a rescan lag: the FIRST dynamically
    added `cand_*.md` isn't found yet (subsequent ones are). Without this, the
    first agent of every run mis-scores 0. Run a throwaway flat candidate until it
    resolves, so real grading never eats the lag.
    """
    target_model = get_settings().autoloop_target_model
    md = ws / ".opencode" / "agents" / "cand_warmup.md"
    md.write_text(
        f"---\nmodel: {target_model}\n---\n"
        "You are a warmup probe. Reply with a single short sentence.\n"
    )
    try:
        for _ in range(8):
            r = _run_sync(run_agent_opencode(
                agent_dir=ws, agent_name="cand_warmup", prompt="say ok", timeout_s=60,
            ))
            if not r.error and str(r.text).strip():
                break
            time.sleep(3)
    finally:
        md.unlink(missing_ok=True)


_TEACHER_AGENTS: dict[str, str] = {}


def _ensure_teacher_agent(model_ref: str) -> tuple[Path, str]:
    """Install a flat teacher agent (a GLM model) into the workspace.

    Named WITHOUT the `cand_` prefix so the per-run candidate cleanup never
    deletes it. Returns (workspace, agent_name)."""
    ws = _ensure_workspace()
    name = "teacher_" + re.sub(r"[^a-zA-Z0-9]", "_", model_ref)
    if model_ref not in _TEACHER_AGENTS:
        md = ws / ".opencode" / "agents" / f"{name}.md"
        md.write_text(
            f"---\nmodel: {model_ref}\n---\n"
            "You are a precise assistant. Do exactly what the user's message asks"
            " and return only the requested output — no preamble, no commentary,"
            " no tool use.\n"
        )
        _TEACHER_AGENTS[model_ref] = name
    return ws, name


def _compile_candidate(definition: dict[str, Any]) -> tuple[Path, str]:
    """Compile a candidate and install it FLAT-named into the workspace.

    Returns (workspace_dir, flat_agent_name)."""
    ws = _ensure_workspace()
    tmp = Path(tempfile.mkdtemp(prefix="oac_cc_"))
    try:
        nested = _compile_one(definition, tmp)  # e.g. "persona/responding-primary"
        src_md = tmp / ".opencode" / "agents" / f"{nested}.md"
        flat = f"cand_{secrets.token_hex(5)}"
        shutil.copy(src_md, ws / ".opencode" / "agents" / f"{flat}.md")
        return ws, flat
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def live_agent_runner_factory(definition: dict[str, Any]):
    """AgentRunner for the judge-test autoloop.

    Scores the candidate the ONLY way that is an actual test of the agent: a full
    `opencode` session on the local Qwen, run to auto-termination, with the real
    runtime (thinking on, tools available). Returns (final_text, tool_calls) — the
    judge then reviews that real output, and the teacher improves the prompt.

    The candidate is installed FLAT-named into a stable opencode project
    workspace (approach A, mirroring v3's opencode_manager) so the agent is
    discovered reliably — an ad-hoc temp dir yields "Agent not found" and a
    mass-zero run.
    """
    ws, agent_name = _compile_candidate(definition)

    def runner(_defn: dict[str, Any], test):
        prompt = test.prompt or (test.turns[0].prompt if test.turns else "")
        # Two distinct retryable conditions: (a) a surfaced opencode error, and
        # (b) Qwen3.x's stochastic EMPTY turn (no text, no tools). Retry both with
        # a short backoff; a real answer breaks early. Discovery is now reliable,
        # so this only soaks up genuine model/transient blanks. The runtime env is
        # passed so the agent's `python scripts/*.py` tools import app deps (else
        # tool agents flail on ModuleNotFoundError and score 0).
        result = None
        for attempt in range(5):
            result = _run_sync(run_agent_opencode(
                agent_dir=ws, agent_name=agent_name, prompt=prompt, timeout_s=300,
                extra_env=_branch_env(),
            ))
            if (str(result.text).strip() or result.tool_calls) and not result.error:
                break
            time.sleep(2)
        # 3-tuple: surface the tool-discipline signal (denied/blocked attempts +
        # the session error) so build_agent_evaluator can forward it to the judge
        # + failures and the loop learns to stop flailing. Backward-compatible —
        # the evaluator also accepts the old (output, calls) 2-tuple.
        signal = {"blocked": list(result.blocked), "error": result.error}
        return result.text, list(result.tool_calls), signal

    return runner


_fleet_compiled = False


def _ensure_fleet_compiled() -> Path:
    """Compile the FULL fleet into the workspace once, so an orchestrator candidate
    can actually spawn its sub-agents (which it does via
    `uv run scripts/opencode_manager.py run --agent <name>`). Without the whole
    fleet present + the real runtime, orchestrators flail on bash and score 0."""
    global _fleet_compiled
    ws = _ensure_workspace()
    with _ws_lock:
        if not _fleet_compiled:
            from app.agents.compile import compile_fleet
            compile_fleet(target=ws, variants=[DEFAULT_WORKER], clean=False)
            _fleet_compiled = True
    return ws


def _branch_env() -> dict[str, str]:
    """The runtime an orchestrator's `python scripts/<tool>.py` and sub-agent
    spawns actually need.

    The orchestrator's bash runs bare `python scripts/X.py`, which otherwise hits
    the SYSTEM python (no pydantic, no app deps, no PYTHONPATH) and fails to import
    `app.tools.*`. Put the autoloop's own deps-having interpreter first on PATH and
    propagate PYTHONPATH so those scripts run exactly like production.
    """
    import os as _os
    import sys

    venv_bin = str(Path(sys.executable).parent)  # interpreter that HAS the v4 deps
    paths = [venv_bin, str(Path.home() / ".local" / "bin"),  # uv
             str(Path.home() / ".opencode" / "bin")]         # opencode
    extra = {
        "PATH": ":".join(paths) + ":" + _os.environ.get("PATH", ""),
        # the run is launched with PYTHONPATH=<backend>:<OpenCodeCompilerV2>;
        # propagate it so `app.tools.*` and `src` import in the subprocess
        "PYTHONPATH": _os.environ.get("PYTHONPATH", "")
        or f"{_project_root()}:{Path(__file__).parents[4]}",
    }
    for k in ("DATABASE_URL", "VLLM_API_KEY", "ZAI_API_KEY",
              "FREN_RUN_ID", "FREN_CLEARANCE"):
        if _os.environ.get(k):
            extra[k] = _os.environ[k]
    extra.setdefault("FREN_RUN_ID", "autoloop")
    return extra


def live_branch_invoker_factory_for(entry_agent: str):
    """BranchInvokerFactory: drive a candidate ORCHESTRATOR live on qwen, with the
    whole fleet present so it can spawn its sub-agents, and capture the spawn
    chain (parsed from the `--agent` bash dispatches) for path grading."""

    def factory(definition: dict[str, Any]):
        ws = _ensure_fleet_compiled()
        _, agent_name = _compile_candidate(definition)

        def invoke(test):
            from src.testing.branch import BranchTrajectory

            from app.runtime.runner import subagent_dispatch_chain

            prompt = test.prompt or (test.turns[0].prompt if test.turns else "")
            result = _run_sync(run_agent_opencode(
                agent_dir=ws, agent_name=agent_name, prompt=prompt,
                timeout_s=600, extra_env=_branch_env(),
            ))
            # the real dispatch chain is the spawned sub-agents, not the raw bash
            chain = subagent_dispatch_chain(result.raw_stdout) or list(result.tool_calls)
            # DELIVERY CONTRACT: subagent_dispatch_chain drops the raw bash tool
            # calls, which would hide the orchestrator's own emit_guidance.py call
            # (with its payload). Preserve any emit_guidance bash call from the raw
            # tool_calls so the branch evaluator can enforce/grade delivery.
            from app.agents.improve import find_emit_guidance_call
            emit_call = find_emit_guidance_call(list(result.tool_calls))
            if emit_call is not None and emit_call not in chain:
                chain = list(chain) + [emit_call]
            # forward the tool-discipline signal so the outcome evaluator labels an
            # errored session (not a blank) and the judge sees the blocked attempts
            return BranchTrajectory(
                output=result.text, tool_calls=chain,
                error=result.error, blocked_tools=list(result.blocked),
            )

        return invoke

    return factory
