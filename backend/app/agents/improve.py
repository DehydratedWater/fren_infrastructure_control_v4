"""Per-agent and per-branch improvement, wired to the framework fleet harness.

This is where the fleet's "every agent self-improves, and every branch is
optimised" requirement is assembled. The framework supplies the loops
(`IterativeLoop`, `build_branch_loop`) and the parallel runner (`run_fleet`);
this module supplies the fren-specific EVALUATORS — how an agent / a branch is
actually run to produce a score.

Tiering (the both-tier decision) lives in the injected invoker factories:
- gate tier: a deterministic mock that reflects the candidate prompt so a round
  is cheap and reproducible;
- promote tier: a live `opencode` run (wired in app/runtime/runner.py).

The harness only promotes a winner that clears the score threshold AND the hard
criteria, into `.oac/promoted/`, where the registry picks it up next compile.
"""

from __future__ import annotations

import json
import re
import statistics
import threading
from pathlib import Path
from typing import Any, Callable

from app.agents.branches import branches
from app.agents.registry import PROJECT_ROOT, all_agents
from src import (
    AgentDefinition,
    AgentTest,
    IdentityMutator,
    ImprovementUnit,
    IterativeLoop,
    LLMPromptRewriter,
    PromptPrefixMutator,
    agent_unit,
    branch_unit,
    build_branch_loop,
    run_fleet,
)
# The improvement Criterion is shadowed at top-level by workflow's Criterion —
# import the optimisation criteria from the improvement package explicitly.
from src.improvement import Criterion, OptimisationCriterion, flailing_note
from src.improvement.mutators import MutationContext
from src.improvement.version import ComponentVersion
from src.testing.branch import BranchInvoker, BranchTrajectory
from src.testing.evaluation import RunContext, ToolCallRecord, evaluate

# How a candidate agent definition is run for ONE agent test → (output, calls).
AgentRunner = Callable[[dict[str, Any], AgentTest], tuple[Any, list[ToolCallRecord]]]
# Build an AgentRunner for a candidate (lets the live tier compile per candidate).
AgentRunnerFactory = Callable[[dict[str, Any]], AgentRunner]
# Build a BranchInvoker for a candidate orchestrator definition.
from src.improvement.branch import (  # noqa: E402
    BranchInvokerFactory,
    branch_component_id,
    build_outcome_branch_loop,
)
from src.improvement.version import ComponentRegistry  # noqa: E402

PASS = OptimisationCriterion(
    name="pass-tests",
    criteria=(Criterion(kind="pass_rate", target=1.0, hard=True),),
)

# --- production DELIVERY CONTRACT -------------------------------------------
# In production an agent's plain assistant text is INVISIBLE to the user. The
# ONLY way a message reaches the user is the agent calling
# `python scripts/emit_guidance.py` (agent → emit_guidance → ledger →
# persona_prose → Telegram). An agent whose allow-list permits emit_guidance is a
# "delivery agent": for it, a candidate that returns text WITHOUT calling
# emit_guidance would deliver NOTHING in production, so it must score 0.0 and the
# evaluator must grade the EMITTED PAYLOAD (the message handed to emit_guidance),
# not the assistant text.
EMIT_GUIDANCE_SCRIPT = "scripts/emit_guidance.py"
_NO_DELIVERY_REASON = (
    "did not call emit_guidance.py — output would be invisible to the user in"
    " production (assistant text is never delivered; only emit_guidance reaches"
    " the user)"
)


def _allowed_commands(definition: dict[str, Any] | Any) -> list[str]:
    """Every bash allowed-command pattern reachable from an agent definition.

    Walks `extra_tools[].bash_tool.permission_bash.allowed_commands` on the
    (possibly model_dump'd) agent definition. Tolerant of both dict dumps and
    pydantic objects so it works on a candidate `version.definition` (a dict) and
    on an `AgentDefinition`.
    """
    if hasattr(definition, "model_dump"):
        definition = definition.model_dump()
    cmds: list[str] = []
    if not isinstance(definition, dict):
        return cmds
    for tool in definition.get("extra_tools") or []:
        bash = (tool or {}).get("bash_tool") if isinstance(tool, dict) else None
        perm = (bash or {}).get("permission_bash") if isinstance(bash, dict) else None
        for c in ((perm or {}).get("allowed_commands") or []) if isinstance(perm, dict) else []:
            if isinstance(c, str):
                cmds.append(c)
    return cmds


def is_delivery_agent(definition: dict[str, Any] | Any) -> bool:
    """True if the agent's compiled tool allow-list permits emit_guidance.py.

    These agents MUST deliver via emit_guidance in production; non-delivery agents
    (e.g. event_extractor — no emit_guidance in its allow-list) keep current
    text-only grading.
    """
    return any(EMIT_GUIDANCE_SCRIPT in c for c in _allowed_commands(definition))


# --- the STRONG delivery postamble ------------------------------------------
# A delivery agent's plain assistant text is INVISIBLE in production — the only
# thing that reaches the user is a `python scripts/emit_guidance.py --data '...'`
# call. The ~36 delivery agents that already DELIVER all carry a "Message
# Discipline (CRITICAL)" block (modelled by goals/evening_focus). The ~38 that do
# NOT instruct emit_guidance in their baseline produce invisible text and score 0.
#
# This postamble is that working block, made MAXIMALLY imperative for a mid-size
# local model (Qwen-27B): it states the invisibility, gives the EXACT CLI with a
# concrete --data example using the real PersonaGuidance fields, and demands the
# emit call be the agent's FINAL action. A prior WEAK generic rule ("it must call
# emit_guidance") did NOT get qwen to comply — this version is concrete + bossy on
# purpose. Keep it in sync with goals/evening_focus's Message Discipline block.
DELIVERY_POSTAMBLE = (
    "\n\n## Message Discipline (CRITICAL — OVERRIDES every earlier instruction)\n"
    "This section OVERRIDES anything above about how to message the user. IGNORE any"
    " earlier instruction to 'send via Telegram', 'send a message', 'split into"
    " multiple messages', or to call send_message — those are obsolete. Your plain"
    " assistant text is INVISIBLE — it is NEVER shown to the user. The ONE AND"
    " ONLY way anything reaches the user is by running scripts/emit_guidance.py."
    " You ALWAYS end your turn by calling scripts/emit_guidance.py exactly once —"
    " either to DELIVER a message, or to SKIP (send nothing). Both are done with the"
    " SAME command; only message_kind differs.\n"
    "\n"
    "WHEN TO SKIP (send nothing) — this is a correct SUCCESS, not a failure:\n"
    "If — per your own instructions above — there is nothing to send this run (no"
    " trigger fired, the user is busy / user_busy / user_recently_busy, in cooldown,"
    " nothing is new since last time, or you would REPEAT something you have already"
    " said recently), then SKIP: call emit_guidance with message_kind=\"skip\" and an"
    " empty key_points array. A skip delivers NOTHING to the user and is the CORRECT,"
    " expected outcome for a quiet tick. Most background ticks should skip. Do NOT"
    " invent a message just to fill the silence; do NOT narrate that you checked and"
    " found nothing (that itself is spam). When in doubt, SKIP.\n"
    "ANTI-REPETITION + TIME-AWARENESS: before you deliver, check the conversation"
    " digest / recent messages and the current time. NEVER resend a message you have"
    " already delivered (even reworded), and NEVER deliver a stale message (e.g. an"
    " overnight 'go to sleep / your shift is done' nudge at midday). If your only"
    " candidate message is a repeat or is stale, SKIP instead.\n"
    "\n"
    "WHEN TO DELIVER: only when there is something genuinely NEW, timely, and not"
    " already said recently — real news, a fired trigger, a direct reply the user is"
    " waiting on. Then:\n"
    "1. Gather only the data you need with your `python scripts/*.py` tools, then"
    " STOP gathering and deliver. Call each tool AT MOST ONCE; if a tool errors or"
    " returns nothing, do NOT retry it or try variations — note it in one line and"
    " move on. Do NOT read source files, do NOT run --help, ls, cat, head, find,"
    " git, or `python -c`, do NOT inspect the environment — those are blocked and"
    " waste the turn. The moment you have enough to say something useful, deliver"
    " (even partial data is fine — emit what you have rather than keep digging).\n"
    "2. CONSOLIDATE everything into ONE message and call emit_guidance EXACTLY ONCE,"
    " as your VERY LAST action. Never call it twice. Never call send_message.\n"
    "3. The --data value is a JSON object with EXACTLY these keys and NO others:\n"
    "   - \"intent\": string, one line describing what you are doing.\n"
    "   - \"key_points\": array of strings — the REAL, COMPLETE content for the"
    " user (what you found / did / want them to do). This is the message body."
    " Do NOT use a \"message\" key; the user-facing text goes in key_points."
    " (For a skip, leave key_points empty.)\n"
    "   - \"message_kind\": one of \"reply\", \"nudge\", \"briefing\","
    " \"workflow_result\", \"ack\", or \"skip\". Use \"ack\" ONLY for a trivial"
    " 'ok/on it'; use \"skip\" when you have nothing to send (see above).\n"
    "   - (optional) \"tone_hint\": string. NO other keys are allowed — do not"
    " invent fields like type/tactic/level/message.\n"
    "   persona_prose composes the final Twily wording from key_points, so put"
    " plain facts there, not pre-written prose. Never expose tool mechanics, run"
    " ids, or JSON to the user.\n"
    "\n"
    "Your FINAL action MUST be EXACTLY one emit_guidance call (relative path,"
    " interpreter `python`, single line, valid JSON, no trailing flags) — either to"
    " DELIVER:\n"
    "  python scripts/emit_guidance.py --data '{\"intent\":\"<one line>\","
    "\"key_points\":[\"<the full content for the user>\"],\"message_kind\":\"reply\"}'\n"
    "or to SKIP (send nothing — a correct quiet tick):\n"
    "  python scripts/emit_guidance.py --data '{\"intent\":\"nothing to send\","
    "\"key_points\":[],\"message_kind\":\"skip\"}'\n"
)


# A blunt PRIMACY directive prepended to the system_prompt for the same broken
# delivery agents. The compiler appends an auto-generated SECURITY POLICY block
# AFTER the postamble, so the postamble is NOT the last thing a small model reads —
# its "your FINAL action" loses recency. Pairing a top-of-prompt directive (primacy)
# with the strong postamble (still present) is what actually got Qwen-27B to stop
# replying in (invisible) plain text and call emit_guidance instead. Kept short so
# it dominates without bloating the prompt.
DELIVERY_PREAMBLE = (
    "DELIVERY RULE — READ FIRST, OVERRIDES EVERYTHING BELOW: You CANNOT message the"
    " user by writing text. Plain text you write is INVISIBLE and thrown away. The"
    " ONLY way to reach the user is scripts/emit_guidance.py. You ALWAYS end your"
    " turn by running it exactly once — either to DELIVER a message:\n"
    "  python scripts/emit_guidance.py --data '{\"intent\":\"<one line>\","
    "\"key_points\":[\"<the full message for the user>\"],\"message_kind\":\"reply\"}'\n"
    "or, when — per your own instructions — there is nothing to send this run (no"
    " trigger, user busy, nothing new, or you would repeat/stale-deliver something),"
    " to SKIP and send NOTHING (a correct, expected quiet tick — NOT a failure):\n"
    "  python scripts/emit_guidance.py --data '{\"intent\":\"nothing to send\","
    "\"key_points\":[],\"message_kind\":\"skip\"}'\n"
    "Do not invent a message to fill silence, and never resend or stale-deliver —"
    " when in doubt, skip. Whatever any instruction below says about 'send',"
    " 'reply', or 'respond', you do it (or skip it) by calling emit_guidance.\n\n"
)


# A COMPACT skip allowance for SELF-INSTRUCTING delivery agents — the ~36 agents
# that already tell themselves to call emit_guidance in their own prompt (so they
# don't get DELIVERY_PREAMBLE) but whose prompt says "emit exactly one guidance per
# run", i.e. they have NO way to stay silent. Scheduled ones (winddown every 15min
# 00-03, evening_focus, meal_planner) would otherwise message every tick. This adds
# the skip path without re-stating the (already-present) emit instruction.
SKIP_CLAUSE = (
    "QUIET-TICK RULE — READ FIRST: You run on a schedule and MOST runs should send"
    " NOTHING. If there is no real trigger, the user is busy, nothing is new since"
    " your last message, or you would repeat or stale-deliver (e.g. an overnight"
    " 'go to sleep' at midday), then SKIP — send nothing — by ending your turn with"
    " exactly:\n"
    "  python scripts/emit_guidance.py --data '{\"intent\":\"nothing to send\","
    "\"key_points\":[],\"message_kind\":\"skip\"}'\n"
    "A skip is a correct SUCCESS, not a failure. Do NOT invent a message to fill"
    " silence; never resend or stale-deliver. Only deliver when there is something"
    " genuinely NEW and timely. When in doubt, SKIP.\n\n"
)


def _prompt_text(definition: dict[str, Any] | Any) -> str:
    """All authored prompt surfaces of an agent (system_prompt + pre/postamble).

    Used to decide whether an agent ALREADY instructs emit_guidance in its own
    prompt (the ~36 working delivery agents) so we don't double-inject the
    postamble for them.
    """
    if hasattr(definition, "model_dump"):
        definition = definition.model_dump()
    if not isinstance(definition, dict):
        return ""
    return "\n".join(
        str(definition.get(k) or "")
        for k in ("system_prompt", "preamble", "postamble")
    )


def prompt_instructs_emit(definition: dict[str, Any] | Any) -> bool:
    """True if the agent's own prompt already tells it to call emit_guidance.

    The working delivery agents reference `emit_guidance` / `emit-guidance` /
    `PersonaGuidance` in their prompt body; the broken ones do not. We only inject
    the postamble for delivery agents whose prompt lacks any such instruction.
    """
    text = _prompt_text(definition).lower()
    return (
        "emit_guidance" in text
        or "emit-guidance" in text
        or "personaguidance" in text
    )


def needs_delivery_postamble(definition: dict[str, Any] | Any) -> bool:
    """A delivery agent whose prompt does NOT already instruct emit_guidance."""
    return is_delivery_agent(definition) and not prompt_instructs_emit(definition)


def with_delivery_postamble(agent: "AgentDefinition") -> "AgentDefinition":
    """Return `agent` with DELIVERY_POSTAMBLE appended to its postamble IFF it is a
    delivery agent that doesn't already instruct emit_guidance; else unchanged.

    Idempotent: if the postamble is already present it is not added again. Used by
    BOTH the production compile (build_registry) and the autoloop candidate compile
    so optimisation tunes WITH the working delivery contract."""
    if not needs_delivery_postamble(agent):
        return agent
    existing = agent.postamble or ""
    if DELIVERY_POSTAMBLE.strip() in existing:
        return agent  # already injected (idempotent)
    # PRIMACY + RECENCY: prepend a blunt directive to the system_prompt (read
    # first, survives the SECURITY POLICY block that the compiler appends after the
    # postamble) AND append the full strong postamble. Both are needed — postamble
    # alone left Qwen replying in invisible plain text (0/3 emit); the pair makes
    # it call emit_guidance.
    base = agent.system_prompt or ""
    new_base = DELIVERY_PREAMBLE + base if DELIVERY_PREAMBLE not in base else base
    return agent.model_copy(update={
        "system_prompt": new_base,
        "postamble": existing + DELIVERY_POSTAMBLE,
    })


def with_skip_clause(agent: "AgentDefinition") -> "AgentDefinition":
    """Prepend the QUIET-TICK skip allowance to a SELF-INSTRUCTING delivery agent.

    Complements with_delivery_postamble: that one handles delivery agents whose
    prompt LACKS an emit instruction (it injects the full preamble, which already
    contains skip). This one handles delivery agents whose prompt ALREADY instructs
    emit but has no skip path (winddown, evening_focus, meal_planner, ...) so they
    can validly stay silent instead of emitting every scheduled tick. No-op for
    non-delivery agents, for agents that get the full preamble, and idempotent."""
    if not is_delivery_agent(agent) or not prompt_instructs_emit(agent):
        return agent
    base = agent.system_prompt or ""
    if "QUIET-TICK RULE" in base or DELIVERY_PREAMBLE in base or '"message_kind":"skip"' in base:
        return agent  # already has a skip path (idempotent)
    return agent.model_copy(update={"system_prompt": SKIP_CLAUSE + base})


_INFRA_ERROR_MARKERS = (
    "timeout", "timed out", "connection", "econnreset", "session",
    "read operation", "cancelled", "canceled", "502", "503", "504",
    "overloaded", "binary not found", "agent_dir", "no such file",
)


def _quantile(values: list[float], q: float) -> float:
    """The q-quantile (0..1) of values via linear interpolation. Robust soft
    floor: tolerant of a few low outliers, sensitive to broad low scores."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= len(xs):
        return xs[-1]
    return xs[lo] + frac * (xs[lo + 1] - xs[lo])


def _is_infra_failure(run_error, output, calls, blocked) -> bool:
    """True when a probe run failed for INFRASTRUCTURE reasons, not agent quality.

    A timed-out / crashed agent session that produced no deliverable, no tool
    calls, and no blocked-tool attempts is vLLM contention or an opencode
    session crash — scoring it 0 (or as a delivery-contract miss) corrupts the
    floor and lets noise drive promotion. High precision so genuine misses are
    still scored: a CLEAN empty turn has run_error=None (→ real 0), and any
    flailing on blocked tools (→ a real signal we DO grade) disqualifies a skip.
    """
    if not run_error:
        return False
    if blocked or calls:
        return False
    if str(output or "").strip():
        return False
    e = str(run_error).lower()
    return any(m in e for m in _INFRA_ERROR_MARKERS)


def find_emit_guidance_call(calls: list[ToolCallRecord]) -> ToolCallRecord | None:
    """The first tool call whose bash command invoked emit_guidance.py, if any.

    The runner captures a bash call's command into `args` (from the opencode
    event's `state.input.command`). A delivery happened iff such a call exists.
    """
    for c in calls or []:
        cmd = str(c.args.get("command") or "") if isinstance(c.args, dict) else ""
        if EMIT_GUIDANCE_SCRIPT in cmd:
            return c
    return None


def emit_is_skip(call: ToolCallRecord | None) -> bool:
    """True if an emit_guidance call is a no-deliver SKIP.

    A conditional background agent that correctly stays silent calls
    `emit_guidance.py --data '{... "message_kind":"skip" ...}'` (or emits with no
    deliverable content). Such a call SATISFIES the delivery contract — it DID
    reach emit_guidance — but delivers nothing to the user. The contract gate must
    therefore NOT score it 0 for 'not messaging'; the judge then grades whether the
    skip was APPROPRIATE for the probe (no trigger → skip is correct/high; a probe
    that clearly warrants a message → skip is wrong/low). Mirrors
    persona_prose.is_skip_guidance on the captured CLI payload.
    """
    if call is None or not isinstance(call.args, dict):
        return False
    cmd = str(call.args.get("command") or "")
    m = re.search(r"--data\s+('([^']*)'|\"([^\"]*)\"|(\S+))", cmd)
    if not m:
        return False
    raw = m.group(2) or m.group(3) or m.group(4) or ""
    try:
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(obj, dict):
        return False
    if obj.get("message_kind") == "skip":
        return True
    # Empty/blank content (no intent, no key_points, no raw_data) is also a skip.
    kp = obj.get("key_points")
    has_kp = isinstance(kp, list) and any(str(x).strip() for x in kp)
    has_content = bool(str(obj.get("intent") or "").strip() or has_kp
                       or str(obj.get("raw_data") or "").strip())
    return not has_content


# A neutral label shown to the judge for a skip run, so the rubric can grade
# whether staying silent was the right call for this probe (instead of seeing an
# empty string and mis-scoring it as a non-answer).
_SKIP_JUDGE_OUTPUT = (
    "[The agent emitted a contract-satisfying SKIP: it called emit_guidance with"
    " message_kind=\"skip\" and delivered NOTHING to the user. This is a correct"
    " outcome ONLY if the request/probe genuinely warranted no message (no trigger,"
    " user busy, nothing new, or a stale/repeat message). Judge whether SKIPPING was"
    " the right call for THIS probe: if the probe clearly warranted a real message,"
    " a skip is WRONG (low); if the probe had no real trigger, a skip is CORRECT"
    " (high).]"
)


def extract_emit_payload(call: ToolCallRecord | None) -> str:
    """The message payload the agent handed to emit_guidance.py.

    emit_guidance is invoked as `python scripts/emit_guidance.py --data '<json>'`
    (the persona/* convention) or with `--message`/a positional arg. Pull the
    human-meaningful deliverable out of the captured command so the JUDGE grades
    what the user would actually receive — not the invisible assistant text. Falls
    back to the whole command tail if no recognised flag is present.
    """
    if call is None or not isinstance(call.args, dict):
        return ""
    cmd = str(call.args.get("command") or "")
    if not cmd:
        return ""
    # `--data '<json>'` (emit_guidance's PersonaGuidance schema) — surface the
    # user-facing fields (key_points / intent / must_mention) the judge cares about.
    m = re.search(r"--data\s+('([^']*)'|\"([^\"]*)\"|(\S+))", cmd)
    if m:
        raw = m.group(2) or m.group(3) or m.group(4) or ""
        try:
            obj = json.loads(raw)
            parts: list[str] = []
            for key in ("intent", "key_points", "must_mention", "actions_taken"):
                v = obj.get(key)
                if isinstance(v, list):
                    parts.extend(str(x) for x in v if x)
                elif v:
                    parts.append(str(v))
            if parts:
                return "\n".join(parts)
        except Exception:  # noqa: BLE001
            return raw
        return raw
    # `--message '<text>'`
    m = re.search(r"--message\s+('([^']*)'|\"([^\"]*)\"|(\S.*))", cmd)
    if m:
        return (m.group(2) or m.group(3) or m.group(4) or "").strip()
    # fall back to everything after the script path (a positional payload)
    idx = cmd.find(EMIT_GUIDANCE_SCRIPT)
    return cmd[idx + len(EMIT_GUIDANCE_SCRIPT):].strip()

# Graded criterion for judge-scored tests: reward higher score_floor (continuous
# 0..1) rather than all-or-nothing. The best-scoring candidate wins; a candidate
# that lifts the score (e.g. 0.4 → 0.8) is promotable even if not perfect.
GRADED = OptimisationCriterion(
    name="lift-judge-score",
    aggregation="weighted",
    criteria=(Criterion(kind="score_floor", target=1.0, weight=1.0),),
)

# Blended criterion (mean + soft quantile floor). The strict score_floor (min)
# gate never promotes a stochastic agent on qwen-27B: 1-2 probes where the
# model skips/blanks ~25-50% of proactive ticks (a MODEL limit, not a quality
# bug) pin the floor at 0 even when the agent is genuinely good overall (mean
# 0.77-0.84). This rewards the mean while a HARD p25 guard still blocks an
# agent that is BROADLY broken (many low probes tank the 25th percentile,
# whereas one flaky probe barely moves it). Verified: nudge_strategist mean
# 0.84 / p25~0.6 promotes; a many-zeros agent (low p25) does not.
BLENDED = OptimisationCriterion(
    name="mean-with-soft-floor",
    aggregation="weighted",
    criteria=(
        Criterion(name="mean", kind="score_mean", target=1.0, weight=1.0),
        # Hard guard: at least the 25th percentile of probes must clear 0.5 —
        # tolerates a couple of model-limited flaky probes, catches broad breakage.
        Criterion(name="soft-floor", kind="score_quantile", target=0.5,
                  weight=0.0, hard=True),
    ),
)
# Back-compat alias (was proactive-only before being adopted fleet-wide).
PROACTIVE_BLEND = BLENDED


def _default_mutators(marker_hint: str = ""):
    """Identity (control) + a prompt-rewriter; a fixed-prefix nudge optional."""
    muts = [IdentityMutator(), LLMPromptRewriter(guidance="Fix the failing tests.")]
    if marker_hint:
        muts.insert(1, PromptPrefixMutator(marker_hint))
    return muts


_PROBE_CACHE: dict[str, str] = {}
_PROBE_LOCK = threading.Lock()
_PROBE_LOADED = False


def _probe_cache_path() -> Path:
    return PROJECT_ROOT / ".oac" / "probe_cache.json"


def _load_probe_cache() -> None:
    """Load synthesised probes from disk ONCE (they're stable per agent role).

    Probe synthesis is a GLM-5.1 call per agent; doing 137 sequentially at
    unit-build time stalled a whole run in setup for ~35 min. Caching to disk +
    a parallel pre-warm (`prewarm_probes`) makes setup instant after the first time.
    """
    global _PROBE_LOADED
    if _PROBE_LOADED:
        return
    with _PROBE_LOCK:
        if _PROBE_LOADED:
            return
        try:
            p = _probe_cache_path()
            if p.exists():
                _PROBE_CACHE.update(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            pass
        _PROBE_LOADED = True


_FALLBACK_PREFIX = "Here is a real task for you:"


def _write_probe_cache() -> None:
    try:
        p = _probe_cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with _PROBE_LOCK:
            # persist only REAL synthesised probes — never the offline fallback
            # (a transient z.ai blip must not poison the cache permanently)
            real = {k: v for k, v in _PROBE_CACHE.items()
                    if not v.startswith(_FALLBACK_PREFIX)}
            p.write_text(json.dumps(real, indent=0))
    except Exception:  # noqa: BLE001
        pass


def prewarm_probes(workers: int = 12) -> int:
    """Synthesise every agent's probe in PARALLEL and persist to disk.

    Run once before an improvement run so the (sequential) unit-build phase
    finds every probe already cached and qwen scoring starts immediately.
    Returns the number newly synthesised.
    """
    from concurrent.futures import ThreadPoolExecutor

    _load_probe_cache()
    todo = [a for a in all_agents() if a.header.agent_id not in _PROBE_CACHE]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(synthesize_probe, todo))
        _write_probe_cache()
    return len(todo)

# Appended to every probe so the model returns a REAL answer, not a meta-demo.
# (Root cause of many 0-scores: the old prompt said "demonstrate your core
# function", so agents narrated/simulated themselves instead of doing the job.)
_ANTI_META = (
    "\n\nReply with your actual response for THIS request only. Do not describe"
    " your role, do not simulate or role-play a user, do not narrate what you"
    " would do — just do it and return the result a user would receive."
)


def synthesize_probe(agent: AgentDefinition) -> str:
    """One concrete, realistic user message that exercises the agent's role.

    Generated once by the teacher (GLM-5.1) and cached. A concrete task is what
    makes the judge signal real — the old self-referential "demonstrate your
    core function" fallback induced meta-commentary that any honest judge scores
    0. Falls back to a direct task string if the teacher is unavailable.
    """
    aid = agent.header.agent_id
    _load_probe_cache()
    if aid in _PROBE_CACHE:
        return _PROBE_CACHE[aid]
    role = (agent.usage_explanation_long or agent.usage_explanation_short
            or agent.header.description or aid)
    msg = ""
    try:
        from app.agents.improve_live import _zai_chat
        from app.settings import get_settings

        sys = (
            "You write ONE realistic user message that triggers the described"
            " agent. CRITICAL: the message must be FULLY SELF-CONTAINED — the"
            " agent has NO other context. If the task needs input (a text to"
            " rewrite, data to analyse, a list, a task to categorise), INCLUDE"
            " realistic example content INLINE in the message. Never say 'this"
            " plan' / 'the above' / 'the task' without actually providing it."
            " Output ONLY the user message — no quotes, no preamble."
        )
        usr = (
            f"AGENT ROLE:\n{role}\n\nWrite the single, self-contained user message"
            " now (include any needed input data inline)."
        )
        msg = _zai_chat(
            get_settings().autoloop_teacher_model,
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            max_tokens=900, temperature=0.4,  # GLM-5.1 reasons; don't starve it
        ).strip().strip('"').strip()
    except Exception:  # noqa: BLE001
        msg = ""
    if not msg:
        msg = f"Here is a real task for you: {role[:160]}. Do it now."
    _PROBE_CACHE[aid] = msg
    return msg


def make_judge_test(agent: AgentDefinition) -> AgentTest:
    """Build a strong, graded role-fulfilment LLMJudge test for `agent`.

    This is the satisfiable, continuous 0..1 signal the autoloop climbs (unlike
    brittle substring checks). The prompt is a CONCRETE task (an authored test
    prompt if present, else a teacher-synthesised user message) plus an
    anti-meta guard; the criterion grades whether the reply actually DID the job.
    """
    from src import AgentTest, LLMJudgeEvaluator

    role = (agent.usage_explanation_long or agent.usage_explanation_short
            or agent.header.description or agent.header.agent_id)
    # ALWAYS synthesise a SELF-CONTAINED probe. Authored agent_test prompts are
    # written for their own (often multi-turn / mocked) harness and routinely
    # reference context that a single-shot judge run never supplies ("rewrite the
    # plan in Sarah's voice") — which left Qwen with an underspecified request and
    # produced empty/flailing output scored 0. A teacher-synthesised task is
    # complete on its own.
    prompt = synthesize_probe(agent).rstrip() + _ANTI_META
    criteria = (
        f"The agent's role is: {role}\n"
        f"The user's request is given to the agent. Score how well the response"
        f" ACTUALLY DOES the agent's job for that request, in the agent's voice."
        f" Score 0.0 if the response merely describes its own role, simulates or"
        f" role-plays a user/conversation, narrates what it 'would' do, refuses,"
        f" echoes the prompt, or leaks tool/JSON mechanics instead of answering."
        f" Reward a concrete, correct, on-task result a user could use as-is.\n"
        f"If instead the response is a bracketed note that the agent acted by"
        f" invoking tools/subagents, judge whether THOSE actions are the right"
        f" ones for this request: correct delegation or tool use for the role"
        f" scores high; flailing on wrong, hallucinated, or blocked tools scores"
        f" low.\n"
        f"TOOL DISCIPLINE: if the response carries a 'TOOL DISCIPLINE' note that the"
        f" agent made DENIED/blocked tool attempts (forbidden by its allow-list) or"
        f" that the session ERRORED, LOWER the score in proportion to the number of"
        f" blocked attempts, and score an errored session near 0 (a failed run, not"
        f" an empty answer)."
    )
    return AgentTest(
        name=f"{agent.header.agent_id}::role-fulfilment",
        prompt=prompt,
        evaluators=(LLMJudgeEvaluator(
            name="role-fulfilment", criteria=criteria, pass_threshold=0.7,
        ),),
    )


def _judge_test_suite(agent: AgentDefinition) -> list[AgentTest]:
    """The graded role-fulfilment judge test PLUS any corpus-grounded pack tests
    PLUS the retrieval suite (for the agents it targets).

    Probe packs (app/agents/probe_packs.py) are generated offline from the real
    v3 chat corpus (`python -m app probe-packs`); when an agent has one, its
    probes ADD graded judge tests so optimisation targets what the user actually
    asks. A missing pack degrades to exactly the old single-test behaviour.
    The retrieval suite (app/agents/retrieval_probes.py) adds exact-ground-truth
    multi-source QA for retrieval/fast_retrieval and persona/responding —
    it requires the seeded autoloop corpus (`python -m app seed-retrieval`).
    """
    from app.agents.probe_packs import pack_tests  # local: avoid import cycle
    from app.agents.retrieval_probes import retrieval_tests

    return ([make_judge_test(agent)] + pack_tests(agent.header.agent_id)
            + retrieval_tests(agent.header.agent_id))


def _test_expectations(agent: AgentDefinition) -> list[dict[str, Any]]:
    """A human-readable description of what each agent_test expects, fed to the
    LLM rewriter as `context.failures` so it knows the target to satisfy."""
    out: list[dict[str, Any]] = []
    for t in agent.agent_tests:
        for e in t.evaluators:
            exp = {
                "test": t.name,
                "prompt": (t.prompt or (t.turns[0].prompt if t.turns else ""))[:160],
                "evaluator": e.kind,
            }
            for attr in ("needle", "expected", "pattern", "path", "criteria"):
                v = getattr(e, attr, None)
                if v is not None:
                    exp[attr] = v
            out.append(exp)
    return out


def build_agent_evaluator(
    agent: AgentDefinition, runner_factory: AgentRunnerFactory,
    *, failures_sink: list[dict[str, Any]] | None = None, judge: Any = None,
    samples: int = 1,
):
    """Score a candidate by running the agent's embedded agent_tests.

    When `failures_sink` is provided, the specific (test, evaluator, output)
    evidence for every FAILED check is appended to it — so the loop's
    MutationContext can feed it to the LLM rewriter (it then knows exactly what
    to fix). The list is cleared each evaluation so it reflects the latest run.

    `samples` runs EACH probe N times and aggregates by median — the principled
    fix for a stochastic agent (qwen temp 1.0): a single unlucky blank/skip can
    no longer floor an otherwise-reliable probe (a 4x repeat of one probe gave
    [1.00, 0.60, 0.00, BLANK]; median=0.6 vs min=0.0). samples=1 (default) is
    byte-identical to the old single-run behaviour.
    """

    # A delivery agent (emit_guidance.py in its allow-list) MUST deliver via
    # emit_guidance; the contract is enforced per-candidate below.
    delivery = is_delivery_agent(agent.model_dump())
    n_samples = max(1, samples)

    def evaluator(version: ComponentVersion) -> dict[str, float]:
        tests = agent.agent_tests
        if failures_sink is not None:
            failures_sink.clear()
        if not tests:
            return {"pass_rate": 1.0}  # nothing to fail
        runner = runner_factory(version.definition)

        def score_once(t) -> tuple[float | None, bool, list[dict]]:
            """One run of one probe → (score|None, ok, failure_records).

            score=None marks an infrastructure failure (excluded from scoring).
            Pure: it appends nothing — the caller aggregates across samples and
            decides what to record."""
            recs: list[dict] = []
            ran = runner(version.definition, t)
            # Backward-compatible unpack: live runner returns a 3-tuple
            # (output, calls, signal); gate-tier / test mocks return (output, calls).
            if len(ran) == 3:
                output, calls, signal = ran
            else:
                output, calls = ran
                signal = {}
            blocked = list(signal.get("blocked") or [])
            run_error = signal.get("error")

            # INFRA SKIP (must precede the delivery-contract check): a timed-out
            # or crashed session that produced NOTHING is an infrastructure
            # failure (vLLM contention, opencode crash) — NOT agent quality.
            # Scoring it 0, or misattributing it as a delivery-contract miss,
            # deflates the floor and lets noise drive promotion. A CLEAN empty
            # turn has run_error=None and still scores a real 0; flailing on
            # blocked tools is still graded.
            if _is_infra_failure(run_error, output, calls, blocked):
                recs.append({
                    "test": t.name, "prompt": (t.prompt or "")[:200],
                    "evaluator": "infra-skip",
                    "criterion": "session error/timeout — excluded from scoring",
                    "score": None, "got_output": "",
                    "judge_reasoning": "infrastructure failure, not an agent miss",
                    "error": str(run_error)[:300],
                })
                return None, False, recs

            # DELIVERY CONTRACT: a delivery agent that did NOT call
            # emit_guidance.py delivered NOTHING in production (its assistant
            # text is invisible). Score a hard 0. If it DID call emit_guidance,
            # grade the EMITTED PAYLOAD (what the user receives).
            if delivery:
                emit_call = find_emit_guidance_call(calls)
                if emit_call is None:
                    recs.append({
                        "test": t.name, "prompt": (t.prompt or "")[:200],
                        "evaluator": "delivery-contract",
                        "criterion": "must call python scripts/emit_guidance.py",
                        "score": 0.0, "got_output": str(output)[:400],
                        "judge_reasoning": _NO_DELIVERY_REASON,
                        "blocked_tools": [n for n, _ in blocked],
                        "blocked_attempts": len(blocked),
                        "error": str(run_error)[:300] if run_error else None,
                    })
                    return 0.0, False, recs
                # A SKIP satisfies the contract (emit_guidance WAS called) but
                # delivers nothing. Hand the judge a neutral skip label so it
                # grades whether staying silent was APPROPRIATE for this probe.
                if emit_is_skip(emit_call):
                    output = _SKIP_JUDGE_OUTPUT
                else:
                    payload = extract_emit_payload(emit_call)
                    if payload.strip():
                        output = payload
            # Trajectory-aware: a handoff/tool agent's real output is its tool
            # calls, not assistant text — show the judge the actions so it can
            # grade their appropriateness (and penalise flailing).
            judge_output = output
            if not str(output).strip() and calls:
                traj = " -> ".join(c.name for c in calls)
                judge_output = (
                    "[The agent produced no prose reply; it acted by invoking"
                    f" tools/subagents in this order: {traj}.]"
                )
            note = flailing_note(blocked, run_error)
            if note:
                judge_output = (str(judge_output or "") + "\n\n" + note).strip()
            ctx = RunContext(output=judge_output, tool_calls=list(calls), judge=judge)
            evs = list(t.evaluators)
            results = [evaluate(e, ctx) for e in evs] if evs else []
            # Grade only NON-skipped checks; a probe whose EVERY check skipped is
            # ungraded → it FAILS (promotion must never ride on ungraded probes).
            considered = [r for r in results if not r.skipped]
            if results and not considered:
                ok = False
                score = 0.0
            else:
                ok = all(r.passed for r in considered) if considered else True
                score = (statistics.fmean([r.score for r in considered])
                         if considered else 1.0)
            for e, r in zip(evs, results):
                if not r.passed or r.score < 1.0 or blocked or run_error:
                    recs.append({
                        "test": t.name, "prompt": (t.prompt or "")[:200],
                        "evaluator": e.kind,
                        "criterion": getattr(e, "criteria", None)
                        or getattr(e, "needle", None)
                        or getattr(e, "expected", None),
                        "score": round(r.score, 2),
                        "got_output": str(judge_output)[:400],
                        "judge_reasoning": r.evidence[:250],
                        "blocked_tools": [n for n, _ in blocked],
                        "blocked_attempts": len(blocked),
                        "error": str(run_error)[:300] if run_error else None,
                    })
            return score, ok, recs

        passes = 0
        # `None` marks an infra-skipped probe (excluded from scoring); aligned
        # 1:1 with `tests` so the per-test by_name metrics stay correct.
        scored: list[float | None] = []
        for t in tests:
            samples_out = [score_once(t) for _ in range(n_samples)]
            real = [(s, ok, rc) for (s, ok, rc) in samples_out if s is not None]
            if not real:
                # Every sample was an infra failure → no quality evidence.
                scored.append(None)
                if failures_sink is not None and samples_out:
                    failures_sink.extend(samples_out[0][2])
                continue
            scores_only = [s for s, _, _ in real]
            # MEDIAN across real samples: robust to a single stochastic blank/
            # skip (the noise diagnosed 2026-06-13). ok by MAJORITY vote.
            agg = statistics.median(scores_only)
            ok = sum(1 for _, o, _ in real if o) * 2 >= len(real)
            scored.append(agg)
            passes += 1 if ok else 0
            if failures_sink is not None:
                # Record the WORST real sample's evidence — the most informative
                # for the teacher (what to fix), without N-fold duplication.
                worst = min(real, key=lambda x: x[0])
                failures_sink.extend(worst[2])
        real = [s for s in scored if s is not None]
        infra_skipped = len(scored) - len(real)
        # All probes infra-skipped (e.g. a total vLLM outage mid-eval) = NO
        # evidence. It must look unpromotable, not perfect — mirror the
        # "every check skipped → FAILS" rule so noise can't promote a candidate.
        if not real:
            # No evidence → unpromotable under ANY criterion (set mean/quantile
            # explicitly so a missing-metric soft-skip can't accidentally pass a
            # blended gate).
            return {"pass_rate": 0.0, "score_floor": 0.0, "score_mean": 0.0,
                    "score_quantile": 0.0, "all_infra_skipped": float(len(scored))}
        metrics = {
            "pass_rate": passes / len(real),
            "score_floor": min(real),
            "score_mean": statistics.fmean(real),
            # 25th-percentile score: a robust 'soft floor'. One flaky / model-
            # limited probe (qwen-27B skips ~50% of proactive ticks) barely
            # moves p25, but broad breakage (many low probes) tanks it — so a
            # blended criterion can promote a genuinely-good agent without
            # shipping one that's widely broken. (See PROACTIVE_BLEND.)
            "score_quantile": _quantile(real, 0.25),
        }
        if infra_skipped:
            metrics["infra_skipped"] = float(infra_skipped)
        # Per-test floors (branch-evaluator convention): without these, a
        # floor-gated agent shows score=0.000 in the summary with no way to
        # tell WHICH probe zeroed it from the snapshots alone. Infra-skipped
        # probes (score None) are omitted — they carry no quality signal.
        for t, s in zip(tests, scored):
            if s is not None:
                metrics[f"score_floor:by_name:{t.name}"] = s
        return metrics

    return evaluator


def build_agent_units(
    runner_factory: AgentRunnerFactory,
    *,
    criterion: OptimisationCriterion = PASS,
    mutators=None,
    max_rounds: int = 2,
    llm: Any = None,
    judge: Any = None,
    only: set[str] | None = None,
    use_judge_test: bool = False,
    samples: int = 1,
) -> list[ImprovementUnit]:
    """One improvement unit per agent.

    `llm` (an LLMMutatorClient) is threaded into each loop's MutationContext so
    the LLMPromptRewriter mutator can actually rewrite prompts. `judge` (a
    JudgeClient) is threaded into the RunContext so LLMJudge tests score live.
    When `use_judge_test` is set, EVERY agent gets a generated graded
    role-fulfilment judge test (so all agents — even those without authored
    tests — are improvable on a continuous signal). `only` restricts to a
    subset of agent ids.
    """
    units: list[ImprovementUnit] = []
    for agent in all_agents():
        if only is not None and agent.header.agent_id not in only:
            continue
        # When using the generated judge test, every agent is improvable;
        # otherwise only those that authored agent_tests.
        if use_judge_test:
            agent = agent.model_copy(update={"agent_tests": _judge_test_suite(agent)})
        elif not agent.agent_tests:
            continue
        baseline = ComponentVersion.of(
            agent.header.agent_id, "agent", agent.model_dump(),
        )
        # Shared failures list: the evaluator writes failed-check evidence into
        # it; the MutationContext reads it so the LLM rewriter knows what to fix.
        # Seed it with the test expectations so the FIRST rewrite is on-target
        # even before the baseline run populates concrete failures.
        failures: list[Any] = _test_expectations(agent)
        ctx = None
        if llm is not None:
            ctx = MutationContext(llm=llm, criterion=criterion, failures=failures)
        loop = IterativeLoop(
            baseline=baseline,
            mutators=mutators or _default_mutators(),
            criterion=criterion,
            evaluator=build_agent_evaluator(
                agent, runner_factory,
                failures_sink=failures if llm is not None else None,
                judge=judge, samples=samples,
            ),
            max_rounds=max_rounds,
            mutation_context=ctx,
        )
        units.append(agent_unit(agent.header.agent_id, loop))
    return units


def make_branch_judge_test(branch) -> AgentTest:
    """A NON-single-shot outcome test for an orchestrator branch.

    Unlike an agent's one-turn judge test, a branch runs the orchestrator as a
    FULL multi-step opencode session (it may take many internal steps and/or spawn
    sub-agents). We grade the OUTCOME — did the final response fulfil the user's
    request — not a brittle dispatch-path match (orchestrators frequently do the
    work themselves rather than dispatch the documented chain, which is fine).
    The expected sub-agent path is offered to the judge as a soft hint only.
    """
    from src import AgentTest, LLMJudgeEvaluator

    task = branch.prompt or (branch.turns[0].prompt if getattr(branch, "turns", None)
                             else "") or branch.name
    path = " -> ".join(branch.path) if getattr(branch, "path", None) else ""
    criteria = (
        f"An ORCHESTRATOR agent received this request: \"{task}\".\n"
        f"Score 0..1 how well its FINAL RESPONSE accomplishes that request for the"
        f" user — a complete, on-task, useful result. The agent may EITHER do the"
        f" work itself OR delegate to sub-agents"
        + (f" (a reasonable plan would involve: {path})" if path else "")
        + "; both are fine as long as the request is fulfilled. Score 0 if it"
        " refuses, stalls, loops, errors out, or returns nothing usable.\n"
        "TOOL DISCIPLINE: if the response carries a 'TOOL DISCIPLINE' note that the"
        " orchestrator made DENIED/blocked tool attempts (forbidden by its"
        " allow-list) or that the session ERRORED, lower the score in proportion to"
        " the blocked attempts, and treat an errored session as a failed run."
    )
    return AgentTest(
        name=f"{branch.name}::outcome",
        prompt=task,
        evaluators=(LLMJudgeEvaluator(
            name="branch-outcome", criteria=criteria, pass_threshold=0.7,
        ),),
    )


def build_branch_evaluator(
    branch_tests: list, invoker_factory: BranchInvokerFactory,
    *, judge: Any = None, failures_sink: list[dict[str, Any]] | None = None,
):
    """Score an orchestrator candidate by running each of its branches as a full
    multi-step session and judging the outcome (+ surfacing the dispatch chain)."""

    def evaluator(version: ComponentVersion) -> dict[str, float]:
        if failures_sink is not None:
            failures_sink.clear()
        if not branch_tests:
            return {"pass_rate": 1.0, "score_floor": 1.0}
        invoke = invoker_factory(version.definition)
        # An orchestrator whose own allow-list permits emit_guidance MUST deliver
        # via it — the same production contract as a per-agent delivery agent.
        delivery = is_delivery_agent(version.definition)
        passes = 0
        scores: list[float] = []
        for bt in branch_tests:
            jt = make_branch_judge_test(bt)
            traj = invoke(bt)
            output = traj.output
            chain = [c.name for c in (traj.tool_calls or [])]

            # DELIVERY CONTRACT: a delivery orchestrator that never called
            # emit_guidance.py delivered nothing in production → hard 0.
            if delivery:
                emit_call = find_emit_guidance_call(list(traj.tool_calls or []))
                if emit_call is None:
                    scores.append(0.0)
                    if failures_sink is not None:
                        failures_sink.append({
                            "test": jt.name,
                            "criterion": "must call python scripts/emit_guidance.py",
                            "score": 0.0,
                            "dispatch_chain": chain[:8],
                            "got_output": str(output)[:400],
                            "judge_reasoning": _NO_DELIVERY_REASON,
                            "blocked_tools": [n for n, _ in
                                              (getattr(traj, "blocked_tools", None) or [])],
                            "blocked_attempts": len(getattr(traj, "blocked_tools", None) or []),
                            "error": (str(getattr(traj, "error", None))[:300]
                                      if getattr(traj, "error", None) else None),
                        })
                    continue
                # A SKIP satisfies the contract but delivers nothing — credit it
                # and let the judge grade whether silence was right for this probe.
                if emit_is_skip(emit_call):
                    output = _SKIP_JUDGE_OUTPUT
                else:
                    payload = extract_emit_payload(emit_call)
                    if payload.strip():
                        output = payload
            # if the orchestrator acted only via dispatch/tools (no prose), show
            # the judge what it DID so it can grade the actions, not an empty turn.
            if not str(output).strip() and chain:
                output = "[orchestrator produced no prose; it acted via: " \
                         + " -> ".join(chain) + "]"
            # TOOL-DISCIPLINE: forward the session error + denied/blocked attempts
            # to the judge (so the rubric fires + an errored run is labelled) and
            # to failures (so the teacher rewrites to avoid those tools).
            blocked = list(getattr(traj, "blocked_tools", None) or [])
            run_error = getattr(traj, "error", None)
            note = flailing_note(blocked, run_error)
            if note:
                output = (str(output or "") + "\n\n" + note).strip()
            ctx = RunContext(output=output, tool_calls=list(traj.tool_calls or []),
                             judge=judge)
            results = [evaluate(e, ctx) for e in jt.evaluators]
            ok = all(r.passed for r in results) if results else True
            passes += 1 if ok else 0
            scores.append(
                statistics.fmean([r.score for r in results]) if results else 1.0
            )
            if failures_sink is not None:
                for e, r in zip(jt.evaluators, results):
                    if not r.passed or r.score < 1.0 or blocked or run_error:
                        failures_sink.append({
                            "test": jt.name,
                            "criterion": getattr(e, "criteria", None),
                            "score": round(r.score, 2),
                            "dispatch_chain": chain[:8],
                            "got_output": str(output)[:400],
                            "judge_reasoning": r.evidence[:250],
                            "blocked_tools": [n for n, _ in blocked],
                            "blocked_attempts": len(blocked),
                            "error": str(run_error)[:300] if run_error else None,
                        })
        return {
            "pass_rate": passes / len(branch_tests),
            "score_floor": min(scores) if scores else 1.0,
        }

    return evaluator


def mock_branch_invoker_factory_for(entry_agent: str):
    """Deterministic GATE-tier BranchInvokerFactory — no model calls.

    Simulates a FAITHFUL orchestrator so a branch's `step_contracts` are
    checkable reproducibly: it dispatches exactly the test's declared `path`,
    forwarding the branch prompt inside each spawn command, and attaches each
    step's `subagent_mocks` text as that call's output. The recording shape
    mirrors the LIVE tier's `subagent_dispatch_chain` exactly —
    ``args={"via": "spawn", "command": "… run --agent <step> '<prompt>'"}`` and
    ``output=<the sub-agent's reply>`` — so a contract that passes here asserts
    the same fields a live run records. The joint output is the LAST step's
    mock (the framework's `mock_chain_invoker` convention; that invoker is NOT
    used because it leaves call args empty, which would fail every
    context-forwarding input contract).

    A delivery orchestrator (emit_guidance in its allow-list) also gets a
    trailing emit_guidance bash call carrying the joint output, mirroring the
    production delivery contract.
    """
    del entry_agent  # one shape fits every entry; kept for factory-for signature

    def factory(definition: dict[str, Any]) -> BranchInvoker:
        def invoke(test) -> BranchTrajectory:
            prompt = test.prompt or (test.turns[0].prompt if test.turns else "")
            calls: list[ToolCallRecord] = []
            for step in test.path:
                cmd = (
                    "uv run scripts/opencode_manager.py run "
                    f"--agent {step} '{prompt}'"
                )
                calls.append(ToolCallRecord(
                    name=step,
                    args={"via": "spawn", "command": cmd},
                    output=test.subagent_mocks.get(step),
                ))
            output: Any = (
                test.subagent_mocks.get(test.path[-1]) if test.path else None
            )
            if output is None:
                output = f"[deterministic] completed branch {test.name}"
            if is_delivery_agent(definition):
                payload = json.dumps(
                    {"intent": "reply", "key_points": [str(output)[:200]]}
                )
                calls.append(ToolCallRecord(
                    name="bash",
                    args={"command":
                          f"python scripts/emit_guidance.py --data '{payload}'"},
                ))
            return BranchTrajectory(output=output, tool_calls=calls)

        return invoke

    return factory


def build_branch_units(
    invoker_factory_for: Callable[[str], BranchInvokerFactory],
    *,
    criterion: OptimisationCriterion = PASS,
    mutators=None,
    max_rounds: int = 2,
    llm: Any = None,
    judge: Any = None,
    only: set[str] | None = None,
    use_judge_test: bool = False,
    samples: int = 1,
) -> list[ImprovementUnit]:
    """One improvement unit per branch.

    `invoker_factory_for(entry_agent)` returns the BranchInvokerFactory for that
    orchestrator (so the consumer can wire mock vs live per entry agent). `llm`
    threads into the MutationContext; `only` restricts to a subset of entry ids.
    """
    by_entry: dict[str, list] = {}
    for b in branches():
        if only is not None and b.entry_agent not in only:
            continue
        by_entry.setdefault(b.entry_agent, []).append(b)

    agent_by_id = {a.header.agent_id: a for a in all_agents()}
    units: list[ImprovementUnit] = []
    for entry_agent, tests in by_entry.items():
        entry_def = agent_by_id[entry_agent].model_dump()
        # CRITICAL: every unit gets its OWN fresh mutators, ComponentRegistry, and
        # MutationContext/failures list. Sharing any of these across the per-entry
        # loop lets one entry's winner inherit another's component identity — the
        # cross-unit contamination that snapshotted every branch winner as the SAME
        # component (and broke promote with a content_hash/definition mismatch).
        if use_judge_test:
            # NON-single-shot grading: run the orchestrator as a full multi-step
            # session per branch and judge the OUTCOME (the only signal that works
            # for orchestrators — a continuous 0..1 the teacher can climb). This is
            # the framework's documented orchestrator default; it namespaces the
            # baseline as `branch:<entry>` (so a branch winner can NEVER collide
            # with the entry agent's own per-agent loop on the shared snapshot dir
            # / promote slot) and gives the loop its own fresh registry.
            failures: list[Any] = []
            ctx = (MutationContext(llm=llm, criterion=criterion, failures=failures)
                   if llm is not None else None)
            loop = build_outcome_branch_loop(
                entry_agent=entry_agent,
                entry_definition=entry_def,
                tests=tests,
                invoker_factory=invoker_factory_for(entry_agent),
                mutators=mutators or _default_mutators(),
                criterion=criterion,
                judge=judge,
                failures_sink=failures if llm is not None else None,
                registry=ComponentRegistry(),
                max_rounds=max_rounds,
                mutation_context=ctx,
            )
        else:
            loop = build_branch_loop(
                entry_agent=entry_agent,
                entry_definition=entry_def,
                tests=tests,
                invoker_factory=invoker_factory_for(entry_agent),
                mutators=mutators or _default_mutators(),
                criterion=criterion,
                registry=ComponentRegistry(),
                max_rounds=max_rounds,
                mutation_context=MutationContext(llm=llm) if llm is not None else None,
            )
        units.append(branch_unit(entry_agent, loop))
    return units


def run_improvement(
    *,
    agent_runner_factory: AgentRunnerFactory,
    branch_invoker_factory_for: Callable[[str], BranchInvokerFactory],
    snapshots_dir: Path | None = None,
    promote_threshold: float | None = 1.0,
    project_root: Path | None = None,
    max_workers: int = 4,
    llm: Any = None,
    judge: Any = None,
    only: set[str] | None = None,
    max_rounds: int = 2,
    include_branches: bool = True,
    criterion: OptimisationCriterion = PASS,
    use_judge_test: bool = False,
    samples: int = 1,
    namespace: str | None = None,
):
    """Run the full fleet improvement (agents + branches) and (optionally) promote.

    `llm` is the LLMMutatorClient that powers prompt rewriting (the research).
    `judge` is the JudgeClient that scores LLMJudge tests live. `criterion`
    selects the scoring goal (use GRADED for judge-scored continuous lift).
    `only` restricts to a subset of agent/branch ids.
    """
    units = build_agent_units(
        agent_runner_factory, llm=llm, judge=judge, only=only,
        max_rounds=max_rounds, criterion=criterion, use_judge_test=use_judge_test,
        samples=samples,
    )
    if include_branches:
        units += build_branch_units(
            branch_invoker_factory_for, llm=llm, judge=judge, only=only,
            max_rounds=max_rounds, criterion=criterion,
            use_judge_test=use_judge_test,
        )
    return run_fleet(
        units,
        snapshots_dir=snapshots_dir,
        project_root=project_root or PROJECT_ROOT,
        promote_threshold=promote_threshold,
        max_workers=max_workers,
        run_label="fleet-improve",
        namespace=namespace or None,
    )
