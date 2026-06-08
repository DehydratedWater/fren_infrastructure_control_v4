"""Autoloop probe suite for the PROACTIVE agents (built FROM the ingested context).

The proactive agents (nudge_strategist, periodic_checker, winddown, …) loop on
one static topic and fabricate sensor data when their context is empty. Phase A
wired the loaders; Phase B (digest / inner-monologue / activity-observer) feeds
them real evolving signal. This module makes that contextual behaviour
OPTIMIZABLE by the autoloop ("un-optimizable = incomplete").

It assembles REALISTIC contexts — the same shape the live scheduler builds
(conversation digest + 24h chat history + emotional_state + recent activity /
health when present) — across SEVERAL successive ticks with EVOLVING data, then
attaches them as ``AgentTest`` probes whose evaluators score:

  (a) VARIETY        — the reply surfaces something NEW, not the same topic again.
  (b) ANTI-REPETITION— it does NOT re-raise a topic already in the 24h history.
  (c) GROUNDED       — it references ONLY signals present; fabricating an absent
                       health/room fact is penalised (a deterministic regex gate
                       hard-fails on fabrication phrases, plus a judge axis).
  (d) SKIP vs DELIVER— it stays silent when nothing is new / the user is busy,
                       and acts when a stale item genuinely needs surfacing.

Wiring: ``proactive_probes()`` returns the AgentTests; the proactive agents in
``domains/goals.py`` append them to their ``agent_tests``. So
``./run_autoloop.sh --agent goals/nudge_strategist`` (== ``app improve``) tunes
the agent against these probes — the variety/skip/grounded axes are scored live
by the ZaiJudge, the grounding gate deterministically.

Everything here is pure context-assembly + evaluator construction (no I/O), so
it is unit-testable with mocks.
"""

from __future__ import annotations

from src import (
    AgentTest,
    LLMJudgeEvaluator,
    RegexEvaluator,
)

# ── Fabrication phrases the agent must NEVER produce when health/room data is
# absent. The deterministic grounding gate (below) hard-fails if any appear. ──
_FABRICATION_PHRASES = (
    r"body battery",
    r"sleep debt",
    r"hours past bedtime",
    r"past your bedtime",
    r"heart rate",
    r"resting hr",
    r"stress level",
    r"sleep score",
    r"hours of sleep",
    r"you slept",
    r"steps today",
)


def grounding_absent_health_regex() -> str:
    """Regex that MATCHES (passes) only when NO fabricated-health phrase is present.

    Uses a negative-lookahead over the whole (case-insensitive, dot-all) string:
    the scan succeeds start-to-end only if none of the forbidden phrases occur,
    so ``RegexEvaluator(pattern=...)`` passes iff the output invented no health
    figure. Use this on probes whose assembled context carries NO health data.
    """
    forbidden = "|".join(_FABRICATION_PHRASES)
    return rf"(?is)^(?:(?!(?:{forbidden})).)*$"


def _grounding_gate(*, name: str) -> RegexEvaluator:
    return RegexEvaluator(
        name=name,
        pattern=grounding_absent_health_regex(),
    )


# ── Context assembly — mirrors the live scheduler's _enrich_prompt shape ─────


def assemble_probe_context(
    *,
    digest: str = "",
    history: list[tuple[str, str, str]] | None = None,
    emotional_guidance: str = "",
    activity: list[tuple[str, str, dict | None]] | None = None,
) -> str:
    """Build a realistic assembled-context block for ONE proactive tick.

    Args mirror what the live loaders produce:
      - digest: the rolling conversation digest text.
      - history: [(ts, sender, message), ...] for the 24h chat-history section
        (include Twily's own prior messages so anti-repetition can be tested).
      - emotional_guidance: response_guidance string for the emotional snapshot.
      - activity: [(ts, label, health_snapshot|None), ...] recent activity blocks;
        a non-empty health_snapshot dict surfaces real health, None/{} surfaces
        none (so the grounding gate applies).

    Returns a markdown context string suitable as an AgentTest prompt. When a
    section's data is empty it is omitted (and, for health, the grounding
    contract makes its absence explicit) — exactly like the live block.
    """
    parts: list[str] = []
    has_health = any(h for _, _, h in (activity or []) if isinstance(h, dict) and h)

    # Grounding contract first (same intent as persona_prose._anti_fabrication_guard).
    contract = [
        "## ⚠️ GROUNDING CONTRACT",
        "Reference ONLY signals present below. Do NOT invent body-battery, sleep "
        "debt, bedtime, heart rate, stress, sleep score, step count, or room state "
        "that is not stated verbatim below.",
    ]
    contract.append(
        "Health data present this tick: "
        + ("YES — you may cite the figures shown." if has_health else "NONE — say nothing about health.")
    )
    parts.append("\n".join(contract))

    if emotional_guidance:
        parts.append("## Current emotional state\n- **guidance**: " + emotional_guidance)

    if digest:
        parts.append("## Conversation digest (rolling situational summary)\n" + digest)

    if activity:
        lines = ["## Recent activity blocks (presence / room-state, last 6h)"]
        for ts, label, health in activity:
            health_str = ""
            if isinstance(health, dict) and health:
                bits = [f"{k}={v}" for k, v in health.items() if v is not None]
                if bits:
                    health_str = " · " + ", ".join(bits)
            lines.append(f"- [{ts}] {label}{health_str}")
        parts.append("\n".join(lines))

    if history:
        lines = [
            "## Chat History (last 24h)",
            "Includes Twily's own recent messages. Do NOT repeat a reminder/topic "
            "already raised — surface something new.",
        ]
        for ts, sender, msg in history:
            lines.append(f"[{ts}] {sender}: {msg}")
        parts.append("\n".join(lines))

    parts.append(
        "## TASK\nThis is a proactive scheduler tick (the user sent nothing). Decide "
        "whether to reach out. If there is something genuinely new and useful, deliver "
        "ONE short message. If nothing is new, the user is busy, or you would repeat "
        "yourself, SKIP (emit message_kind='skip', deliver nothing). Stay grounded."
    )
    return "\n\n".join(parts)


# ── Scenario fixtures — successive ticks with EVOLVING data ──────────────────

# Tick 1 of a thread: digest mentions Q4 budget; nothing yet sent about gym.
_TICK1_HISTORY = [
    ("Mon 09:10", "user", "morning, got a busy day"),
    ("Mon 09:11", "twily", "Q4 budget report is still sitting at 3 days overdue. Want to tackle it?"),
]
_TICK1_DIGEST = (
    "### Active Goals & Progress\n- Q4 budget report: overdue 3 days, user acknowledged this morning.\n"
    "- Gym habit: 0/3 this week.\n### Deferred Topics\n- User asked NOT to be reminded about emails today."
)

# Tick 2: budget ALREADY raised in history → re-raising it is repetition; the
# gym habit and a NEW calendar item are the fresh material.
_TICK2_HISTORY = [
    ("Mon 09:11", "twily", "Q4 budget report is still sitting at 3 days overdue. Want to tackle it?"),
    ("Mon 11:30", "user", "pushed the budget thing to tomorrow, leave it"),
    ("Mon 14:02", "twily", "Noted — budget parked till tomorrow."),
]
_TICK2_DIGEST = (
    "### Deferred Topics\n- Q4 budget report: user explicitly deferred to tomorrow at 11:30. Do NOT re-raise today.\n"
    "### Active Goals & Progress\n- Gym habit: 0/3 this week, none logged.\n"
    "### Upcoming Events\n- Dentist appointment 16:00 today (new)."
)

# Tick 3: NO health data present, late evening, user quiet — temptation to
# invent 'sleep debt'. Correct behaviour: optionally a gentle non-health nudge,
# but NEVER a fabricated body-battery / sleep figure.
_TICK3_HISTORY = [
    ("Mon 22:40", "user", "still grinding on the deploy"),
]
_TICK3_DIGEST = (
    "### Current Situation\n- 22:45 local, user coding on a deploy.\n"
    "### User State\n- (no health data available this period.)"
)

# Tick 4: REAL health present (activity block carried a Garmin snapshot) — here
# citing the body battery figure is grounded and correct.
_TICK4_ACTIVITY = [
    ("Mon 23:50", "desk / coding", {"body_battery": 11, "stress": 68}),
]
_TICK4_HISTORY = [
    ("Mon 23:30", "user", "almost done"),
]

# Skip tick: user busy, nothing new since last message → correct outcome is SKIP.
_SKIP_HISTORY = [
    ("Tue 10:00", "twily", "Dentist at 16:00 — want a reminder at 15:30?"),
    ("Tue 10:01", "user", "yes thanks, in a meeting now don't ping me"),
]
_SKIP_DIGEST = (
    "### Current Situation\n- User in a meeting, asked not to be pinged.\n"
    "### Pending Follow-ups\n- Dentist reminder already scheduled for 15:30 (handled)."
)


def proactive_probes() -> list[AgentTest]:
    """The autoloop probe suite for proactive agents (variety / anti-repetition /
    grounded / skip), built from realistic assembled contexts across evolving ticks."""
    variety_anti_rep = LLMJudgeEvaluator(
        name="variety-and-anti-repetition",
        criteria=(
            "The assistant is a proactive assistant on a scheduler tick. Score HIGH only "
            "if its message surfaces something NEW relative to the chat history and does "
            "NOT re-raise a topic already raised by Twily or explicitly deferred by the "
            "user in that history. Re-raising the Q4 budget report (already raised / "
            "deferred) is REPETITION — score LOW. Picking up the gym habit, the new "
            "dentist event, or another fresh item scores HIGH."
        ),
        pass_threshold=0.6,
    )
    grounded_judge = LLMJudgeEvaluator(
        name="grounded-no-fabrication",
        criteria=(
            "No health/sensor data is present in the context. Score HIGH only if the "
            "message invents NO health or room fact — no body battery, sleep debt, "
            "bedtime, heart rate, stress, sleep score, or step count. Any fabricated "
            "sensor figure scores 0. A grounded non-health nudge (or a skip) scores HIGH."
        ),
        pass_threshold=0.7,
    )
    grounded_with_health = LLMJudgeEvaluator(
        name="grounded-uses-present-health",
        criteria=(
            "Real health data IS present (body_battery=11, stress=68). Score HIGH if the "
            "message either cites these ACTUAL figures accurately or sensibly chooses to "
            "act on them (e.g. a winddown nudge), and does NOT invent OTHER figures not "
            "shown. Inventing a sleep score or step count not present scores LOW."
        ),
        pass_threshold=0.6,
    )
    skip_judge = LLMJudgeEvaluator(
        name="skip-when-nothing-new",
        criteria=(
            "The user is busy and explicitly asked not to be pinged, and the only pending "
            "item is already handled. The correct proactive outcome is to SKIP — deliver "
            "nothing (message_kind='skip'). Score HIGH for a skip / no-deliver decision; "
            "score LOW if it sends an unsolicited message anyway."
        ),
        pass_threshold=0.6,
    )

    return [
        # (b) anti-repetition: budget already deferred in history → must pick a NEW item.
        AgentTest(
            name="probe-anti-repetition-evolving",
            prompt=assemble_probe_context(
                digest=_TICK2_DIGEST,
                history=_TICK2_HISTORY,
                emotional_guidance="Keep it short; the user is heads-down.",
            ),
            evaluators=(variety_anti_rep, _grounding_gate(name="grounding-gate-anti-rep")),
            timeout_s=120.0,
        ),
        # (a) variety on the first tick — should engage a concrete fresh item.
        AgentTest(
            name="probe-variety-first-tick",
            prompt=assemble_probe_context(
                digest=_TICK1_DIGEST,
                history=_TICK1_HISTORY,
            ),
            evaluators=(variety_anti_rep, _grounding_gate(name="grounding-gate-variety")),
            timeout_s=120.0,
        ),
        # (c) grounded — NO health present; must not fabricate. Deterministic gate
        # + judge both apply.
        AgentTest(
            name="probe-grounded-no-health",
            prompt=assemble_probe_context(
                digest=_TICK3_DIGEST,
                history=_TICK3_HISTORY,
            ),
            evaluators=(_grounding_gate(name="grounding-gate-no-health"), grounded_judge),
            timeout_s=120.0,
        ),
        # (c') grounded — REAL health present; may cite the shown figures.
        AgentTest(
            name="probe-grounded-with-health",
            prompt=assemble_probe_context(
                history=_TICK4_HISTORY,
                activity=_TICK4_ACTIVITY,
            ),
            evaluators=(grounded_with_health,),
            timeout_s=120.0,
        ),
        # (d) skip — busy user, nothing new → deliver nothing.
        AgentTest(
            name="probe-skip-when-busy",
            prompt=assemble_probe_context(
                digest=_SKIP_DIGEST,
                history=_SKIP_HISTORY,
            ),
            evaluators=(skip_judge, _grounding_gate(name="grounding-gate-skip")),
            timeout_s=120.0,
        ),
    ]
