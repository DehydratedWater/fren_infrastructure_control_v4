"""Stale-state REPLAY probes — autoloop gates built from REAL v3 failure patterns.

The v3 chat corpus shows four recurring stale-state failures that eroded trust:

  (1) RE-REMINDING RESOLVED ITEMS — the task list lags the chat: the user says
      "Kupiłem już akcje, done" / "I've already bought the stocks", yet the bot
      re-reminds the purchase ("why are you reminding me about stocks?" — twice).
  (2) DOUBLE-COUNTED DOSES — the user logs ONE medication dose, then later
      *refers back* to it ("yeah the atenza I took this morning"); the extractor
      counted two doses.
  (3) DATE/TIMEZONE DRIFT — an event described as "last Tuesday" got stamped
      with today's date ("you're saying I did the hackathon today? it was days
      ago").
  (4) HALLUCINATED SOURCING — invented questions/figures when the source data
      was simply absent.

This module replays those exact patterns as ``AgentTest`` probes so the
autoloop PERMANENTLY gates on them ("un-optimizable = incomplete"):

  - ``stale_state_probes()`` — for goals/periodic_checker and
    goals/nudge_strategist: contexts where the chat history plainly shows an
    item RESOLVED while the digest's task list still carries it stale. The
    graded judge scores 0 for any re-reminder of the resolved item and rewards
    skipping it or acknowledging completion (and surfacing the genuinely fresh
    item instead). Scenarios mix EN/PL like real usage.
  - ``event_extractor_probes()`` — for support/event_extractor: a
    single-dose-mentioned-twice batch (extracting TWO doses scores 0), a
    "last Tuesday" date-drift batch (stamping today's date scores 0), and a
    grounded-absence batch (inventing health-sensor claims with no health data
    present scores 0 — reuses the deterministic
    ``grounding_absent_health_regex`` gate from ``proactive_probes``).

Wiring: the goals probes are appended to the proactive agents' ``agent_tests``
in ``domains/goals.py`` and therefore run under the SAME
``python -m app improve --proactive-probes`` flow (that mode = authored tests
on the proactive agents). The extractor probes are appended to
``support/event_extractor``'s ``agent_tests`` and run via
``python -m app improve --proactive-probes --agent support/event_extractor``
(the mode's hardcoded default agent set lives in ``__main__.py``, owned by
another agent right now; passing ``--agent`` reuses the identical flow).

Everything here is pure context-assembly + evaluator construction (no I/O), so
it is unit-testable with mocks — see ``backend/tests/test_stale_probes.py``.
"""

from __future__ import annotations

from src import (
    AgentTest,
    LLMJudgeEvaluator,
    RegexEvaluator,
)

from app.agents.proactive_probes import (
    assemble_probe_context,
    grounding_absent_health_regex,
)

# Keep thresholds/timeouts consistent with the existing proactive probe suite.
_PASS_THRESHOLD = 0.7
_TIMEOUT_S = 120.0


# ── (1) Resolved-but-stale scenarios for the goals reminder agents ───────────
#
# Each scenario: the chat history PLAINLY shows the item resolved; the digest's
# task list still carries it (sync lag) and offers one genuinely fresh item.

# Scenario A — stocks purchase (Polish, the literal v3 pattern: user complained
# TWICE about being re-reminded after "Kupiłem już akcje").
_STOCKS_HISTORY = [
    ("Tue 09:15", "twily", "Przypominajka: kup akcje pod IKE — termin mija dziś."),
    ("Tue 10:42", "user", "Kupiłem już akcje, done. Nie przypominaj mi o tym więcej."),
    ("Tue 10:43", "twily", "Super, odhaczam zakup akcji ✅"),
]
_STOCKS_DIGEST = (
    "### Active Tasks (task list — may LAG the chat by up to an hour; "
    "cross-check the chat history before reminding)\n"
    "- [todo] Kup akcje pod IKE (due today)\n"
    "- [todo] Review the server backup config (new, untouched)\n"
    "### Current Situation\n- Tue 13:30 local; user at his desk."
)

# Scenario B — apartment payment (English).
_APARTMENT_HISTORY = [
    ("Wed 08:30", "twily", "Heads-up: the apartment payment is due Friday."),
    ("Wed 16:05", "user", "I've already paid for the apartment this morning, you can drop that one."),
    ("Wed 16:06", "twily", "Got it — apartment payment marked done."),
]
_APARTMENT_DIGEST = (
    "### Active Tasks (task list — may LAG the chat; cross-check history)\n"
    "- [todo] Pay apartment installment (due Friday)\n"
    "- [todo] Call the dentist to reschedule Thursday's appointment (new)\n"
    "### Current Situation\n- Wed 18:00 local; user wrapping up work."
)

# Scenario C — completed phone call (mixed PL/EN, like real usage).
_CALL_HISTORY = [
    ("Thu 11:00", "twily", "Pamiętaj żeby zadzwonić do taty w sprawie weekendu."),
    ("Thu 12:10", "user", "btw zadzwoniłem do taty, załatwione — done with that call"),
]
_CALL_DIGEST = (
    "### Active Tasks (task list — may LAG the chat; cross-check history)\n"
    "- [todo] Zadzwonić do taty (weekend plans)\n"
    "- [todo] Hackathon submission — deadline Sunday 23:59 (new, not started)\n"
    "### Current Situation\n- Thu 15:20 local."
)


def _stale_resolution_judge(*, name: str, resolved: str, fresh: str) -> LLMJudgeEvaluator:
    """Graded judge for a resolved-but-stale-task scenario.

    Sharply worded so re-reminding the resolved item is an automatic 0 — the
    exact v3 trust-eroding failure ("why are you reminding me about X? I've
    already done it").
    """
    return LLMJudgeEvaluator(
        name=name,
        criteria=(
            "STALE-STATE GATE. The chat history PLAINLY shows the user already "
            f"completed: {resolved}. The task list in the digest still carries it "
            "only because it lags the chat. Score 0 if the message reminds, nags, "
            "nudges, or asks about that completed item AS IF IT WERE STILL PENDING "
            "in ANY way — this is the exact trust-destroying failure being gated "
            "('why are you reminding me? I've already done it'). Score HIGH if the "
            "message either (a) skips entirely (message_kind='skip'), (b) briefly "
            "acknowledges the item is DONE without re-asking for it, or (c) ignores "
            f"the stale entry and surfaces the genuinely fresh item instead: {fresh}. "
            "Merely mentioning the completed item while clearly treating it as done "
            "is acceptable; treating it as open is an automatic 0."
        ),
        pass_threshold=_PASS_THRESHOLD,
    )


def stale_state_probes() -> list[AgentTest]:
    """Resolved-but-stale replay probes for goals/periodic_checker and
    goals/nudge_strategist (re-reminding a resolved item must score 0)."""
    grounding_gate = RegexEvaluator(
        name="grounding-gate-stale-state",
        pattern=grounding_absent_health_regex(),
    )
    return [
        AgentTest(
            name="probe-stale-resolved-stocks-pl",
            prompt=assemble_probe_context(
                digest=_STOCKS_DIGEST,
                history=_STOCKS_HISTORY,
            ),
            evaluators=(
                _stale_resolution_judge(
                    name="no-rereminder-resolved-stocks",
                    resolved="the stock purchase ('Kupiłem już akcje, done' — "
                             "he bought the IKE stocks and asked not to be "
                             "reminded again)",
                    fresh="reviewing the server backup config",
                ),
                grounding_gate,
            ),
            timeout_s=_TIMEOUT_S,
        ),
        AgentTest(
            name="probe-stale-resolved-apartment-en",
            prompt=assemble_probe_context(
                digest=_APARTMENT_DIGEST,
                history=_APARTMENT_HISTORY,
            ),
            evaluators=(
                _stale_resolution_judge(
                    name="no-rereminder-resolved-apartment",
                    resolved="the apartment payment ('I've already paid for the "
                             "apartment this morning, you can drop that one')",
                    fresh="calling the dentist to reschedule Thursday's appointment",
                ),
                grounding_gate,
            ),
            timeout_s=_TIMEOUT_S,
        ),
        AgentTest(
            name="probe-stale-resolved-call-mixed",
            prompt=assemble_probe_context(
                digest=_CALL_DIGEST,
                history=_CALL_HISTORY,
            ),
            evaluators=(
                _stale_resolution_judge(
                    name="no-rereminder-resolved-call",
                    resolved="calling his dad ('zadzwoniłem do taty, załatwione "
                             "— done with that call')",
                    fresh="the hackathon submission due Sunday 23:59",
                ),
                grounding_gate,
            ),
            timeout_s=_TIMEOUT_S,
        ),
    ]


# ── (2)+(3)+(4) Extractor probes: dose dedup, date drift, grounded absence ───


def assemble_extractor_context(
    *,
    now: str,
    messages: list[tuple[str, str]],
    note: str = "",
) -> str:
    """Build a SELF-CONTAINED probe prompt for support/event_extractor.

    The live agent fetches new messages itself; the probe inlines the batch so
    the test needs no tools/DB. Args:
      - now: the current local date/time line (anchors relative dates).
      - messages: [(local_ts, text), ...] USER messages, ALREADY converted to
        Europe/Warsaw (so timezone handling is not what these probes grade).
      - note: optional extra contract line(s).
    """
    lines = [
        "## Current date/time (Europe/Warsaw)",
        now,
        "",
        "## New USER messages to process (batch already fetched and converted "
        "to Europe/Warsaw — do NOT fetch more; state already updated)",
    ]
    for ts, text in messages:
        lines.append(f"[{ts}] user: {text}")
    lines.append("")
    lines.append(
        "## TASK\nAnalyze ONLY the messages above and list the life events you "
        "would create: one line per event with category, quantity, and "
        "occurred_at (Europe/Warsaw, with offset). Extract clear actions only — "
        "a later REFERENCE to an already-mentioned action is the SAME event, "
        "not a new one. If a message contains no event, extract nothing from it."
    )
    if note:
        lines.append(note)
    return "\n".join(lines)


# Batch A — ONE atenza dose mentioned at 09:02, referred back to at 11:47.
# The v3 failure: the extractor logged TWO 36mg doses for the day.
_DOSE_NOW = "Saturday 2026-06-06 14:00 (+02:00)"
_DOSE_MESSAGES = [
    ("2026-06-06 09:02", "took atenza 36mg at 9"),
    ("2026-06-06 10:15", "gonna try to focus on the backend now"),
    ("2026-06-06 11:47", "yeah the atenza I took this morning is finally kicking in"),
]

# Batch B — date drift: hackathon described as "last Tuesday" on a Saturday.
# The v3 failure: the event was stamped with TODAY's date ("you're saying I did
# the hackathon today? it was days ago"). 2026-06-06 is a Saturday; the
# preceding Tuesday is 2026-06-02.
_DRIFT_NOW = "Saturday 2026-06-06 18:30 (+02:00)"
_DRIFT_MESSAGES = [
    (
        "2026-06-06 18:21",
        "the hackathon last Tuesday went really well btw, we placed 2nd — "
        "barely slept that night",
    ),
]

# Batch C — grounded absence: NO health/sensor data anywhere in the batch; any
# invented health-sensor claim (body battery, sleep score, …) must score 0.
_ABSENT_NOW = "Saturday 2026-06-06 21:10 (+02:00)"
_ABSENT_MESSAGES = [
    ("2026-06-06 20:58", "bought a standing desk mat, 140 zł"),
    ("2026-06-06 21:03", "what should I watch tonight?"),
]


def event_extractor_probes() -> list[AgentTest]:
    """Replay probes for support/event_extractor: dose dedup, date drift, and
    grounded absence (no invented health-sensor claims)."""
    dose_dedup_judge = LLMJudgeEvaluator(
        name="single-dose-no-double-count",
        criteria=(
            "DOSE-DEDUP GATE. The batch contains exactly ONE medication intake: "
            "atenza 36mg taken at 09:00 on 2026-06-06. The 11:47 message ('the "
            "atenza I took this morning') is a REFERENCE BACK to that same dose, "
            "NOT a second intake. Score 0 if the output extracts/creates TWO (or "
            "more) medication events for atenza that day — double-counting a dose "
            "is the exact failure being gated. Score HIGH only if EXACTLY ONE "
            "atenza medication event is extracted, with quantity 36mg and "
            "occurred_at at/around 09:00 Europe/Warsaw on 2026-06-06 (e.g. "
            "2026-06-06T09:00:00+02:00). Wrong time (e.g. 11:47) lowers the score."
        ),
        pass_threshold=_PASS_THRESHOLD,
    )
    date_drift_judge = LLMJudgeEvaluator(
        name="relative-date-no-drift",
        criteria=(
            "DATE-DRIFT GATE. Today is Saturday 2026-06-06. The message describes "
            "a hackathon that happened 'last Tuesday' — i.e. 2026-06-02, DAYS AGO. "
            "Score 0 if the output stamps the hackathon event with TODAY's date "
            "(2026-06-06) or the message timestamp — that is the exact failure "
            "being gated ('you're saying I did the hackathon today? it was days "
            "ago'). Score HIGH if the event's occurred_at is 2026-06-02 (last "
            "Tuesday); choosing to extract no event because the date is in the "
            "past is WORSE than extracting it correctly dated, but still better "
            "than stamping it today."
        ),
        pass_threshold=_PASS_THRESHOLD,
    )
    grounded_absence_judge = LLMJudgeEvaluator(
        name="no-invented-health-claims",
        criteria=(
            "GROUNDED-ABSENCE GATE. The batch contains NO health/sensor data of "
            "any kind — only a purchase (standing desk mat, 140 zł, ~20:58) and a "
            "question (no event). Score 0 if the output invents ANY health-sensor "
            "claim or health event not stated verbatim: body battery, sleep "
            "score/debt, heart rate, stress level, steps, medication, workout, "
            "pain — none of these appear in the batch. Score HIGH if it extracts "
            "exactly one purchase event (cost 140 PLN) and nothing else."
        ),
        pass_threshold=_PASS_THRESHOLD,
    )
    grounding_gate = RegexEvaluator(
        name="grounding-gate-extractor-absent-health",
        pattern=grounding_absent_health_regex(),
    )
    return [
        AgentTest(
            name="probe-extractor-single-dose-dedup",
            prompt=assemble_extractor_context(now=_DOSE_NOW, messages=_DOSE_MESSAGES),
            evaluators=(dose_dedup_judge,),
            timeout_s=_TIMEOUT_S,
        ),
        AgentTest(
            name="probe-extractor-date-drift-last-tuesday",
            prompt=assemble_extractor_context(now=_DRIFT_NOW, messages=_DRIFT_MESSAGES),
            evaluators=(date_drift_judge,),
            timeout_s=_TIMEOUT_S,
        ),
        AgentTest(
            name="probe-extractor-grounded-absence",
            prompt=assemble_extractor_context(now=_ABSENT_NOW, messages=_ABSENT_MESSAGES),
            evaluators=(grounded_absence_judge, grounding_gate),
            timeout_s=_TIMEOUT_S,
        ),
    ]


def stale_probe_registry() -> dict[str, list[AgentTest]]:
    """{agent_id: [AgentTest, ...]} — which agent carries which stale probes.

    Mirrors the wiring in ``domains/goals.py`` / ``domains/support.py`` so
    tests (and tooling) can assert the attachment without importing the full
    domain modules.
    """
    return {
        "goals/periodic_checker": stale_state_probes(),
        "goals/nudge_strategist": stale_state_probes(),
        "support/event_extractor": event_extractor_probes(),
    }
