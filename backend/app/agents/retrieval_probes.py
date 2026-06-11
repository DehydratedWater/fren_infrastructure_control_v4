"""Retrieval autoloop probe suite — multi-source QA with exact ground truth.

WHAT THIS TESTS (and how each category gets its ground truth):

1. EXACT questions — the canaries planted by retrieval_corpus.py: 10 unique
   facts hidden in a ~20k-message haystack at known timestamps. A correct
   answer MUST surface the fact verbatim → FactRecallEvaluator (graded recall
   + hallucination guard). No judge ambiguity: the needle is globally unique.
2. OPEN-ENDED questions — synthesis over the real v3 corpus themes
   (medication logs, training, infra work). Graded by the LLM judge on
   groundedness (cites concrete items, no generic filler), with a
   FactRecall *forbidden* guard against refusal boilerplate.
3. SEEDED-LIBRARY questions — the canary YouTube transcript planted by the
   seeder: asks for facts that exist ONLY inside a stored transcript, so the
   agent must hit the transcript/chunk search path, not chat history.
4. JOURNAL (live, orp) — the digital-journal logger on the orange pi
   (192.168.0.80:5050) is queried through fetch-context's telegram_log step.
   Live-infra probe: graded by judge; tolerant of an empty week but NOT of
   refusing to look.
5. SELF-EXAMINATION — questions about the conversation history ITSELF
   ("when did I tell you X?"): the answer needs both the canary fact and its
   date neighbourhood, exercising chat-history time queries; plus a
   session-introspection probe via the session-inspector tool.

WHO GETS THE SUITE:
- retrieval/fast_retrieval — the dedicated retriever (JSON, <15s contract).
- persona/responding — end-to-end: the user-facing path must retrieve THEN
  deliver via emit_guidance (the evaluator grades the emitted payload).

The suite plugs into the standard autoloop via `retrieval_tests(agent_id)`
(merged in improve._judge_test_suite) and the `--retrieval-probes` CLI mode
(python -m app improve --retrieval-probes). Probes assume the autoloop DB was
seeded: `python -m app seed-retrieval` (see retrieval_corpus.py).

The RALF loop is probed end-to-end separately (`python -m app ralf-smoke`)
because one probe = one full multi-agent plan/execute/verify cycle.
"""

from __future__ import annotations

import re

from src import AgentTest, FactRecallEvaluator, FactSpec, LLMJudgeEvaluator

from app.agents.retrieval_corpus import CANARIES

RETRIEVAL_AGENT = "retrieval/fast_retrieval"
RESPONDER_AGENT = "persona/responding"
RETRIEVAL_SUITE_AGENTS = (RETRIEVAL_AGENT, RESPONDER_AGENT)

# Refusal/can't-access boilerplate — a retrieval agent that says any of these
# instead of looking has failed regardless of phrasing quality.
_REFUSAL = (
    "as an ai",
    "i cannot access",
    "i can't access",
    "i don't have access",
    "no access to your",
    "unable to access",
)

_ANTI_META = (
    "\n\nReply with your actual response for THIS request only. Do not"
    " describe your role, do not simulate or role-play a user — just answer"
    " with the retrieved information."
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def _fact_specs(facts: list[str]) -> tuple[FactSpec, ...]:
    return tuple(FactSpec(any_of=(f,)) for f in facts)


# --------- 1. EXACT (canary) ----------------------------------------------


def canary_tests() -> list[AgentTest]:
    tests = []
    for c in CANARIES:
        tests.append(AgentTest(
            name=f"retrieval:exact:{_slug(c['question'])}",
            prompt=c["question"] + _ANTI_META,
            evaluators=(
                FactRecallEvaluator(
                    name="canary-recall",
                    facts=_fact_specs(c["facts"]),
                    forbidden=_REFUSAL,
                ),
            ),
        ))
    return tests


# --------- 2. OPEN-ENDED (corpus synthesis) --------------------------------

_OPEN_ENDED: list[tuple[str, str]] = [
    (
        "What health-related things have I logged this spring — medication,"
        " test results, anything?",
        "The answer must cite CONCRETE items retrieved from the user's actual"
        " history (e.g. medication doses such as MPH, the April ferritin blood"
        " result, dentist appointment) — not generic wellness advice. Score by"
        " groundedness: 1.0 = several specific retrieved items with details;"
        " 0.5 = one vague but real item; 0.0 = generic filler, refusal, or"
        " fabricated items.",
    ),
    (
        "Summarize what hardware and homelab stuff I've dealt with lately.",
        "Must reference concrete retrieved specifics (e.g. server rack fan"
        " model, GPU/server maintenance, orange pi journal logger). Score by"
        " groundedness and specificity; 0.0 for invented hardware or generic"
        " talk about homelabs.",
    ),
    (
        "Co ostatnio planowałem kupić albo już kupiłem? Przypomnij mi konkrety.",
        "Polish question about recent purchases. Must surface concrete"
        " retrieved purchases/plans (e.g. the Markus II chair for 899 zł, a"
        " second Noctua fan). Answer may be Polish or English; score on"
        " concrete retrieved facts, 0.0 for invented purchases.",
    ),
    (
        "What appointments or travel do I have coming up, based on everything"
        " you know?",
        "Must pull REAL stored items (e.g. dentist Dr Marlena Wójcik March"
        " 14th 11:30, flight LO1923 to Lisbon May 21st). Score 1.0 when the"
        " known stored appointments surface with their details; 0.0 for"
        " invented events or 'I don't know' without an attempt to search.",
    ),
]


def open_ended_tests() -> list[AgentTest]:
    tests = []
    for q, rubric in _OPEN_ENDED:
        tests.append(AgentTest(
            name=f"retrieval:open:{_slug(q)}",
            prompt=q + _ANTI_META,
            evaluators=(
                FactRecallEvaluator(name="no-refusal", forbidden=_REFUSAL),
                LLMJudgeEvaluator(
                    name="groundedness", criteria=rubric, pass_threshold=0.7,
                ),
            ),
        ))
    return tests


# --------- 3. SEEDED LIBRARY (YT transcript) --------------------------------
# Ground truth lives ONLY in the canary transcript planted by the seeder —
# answering requires the transcript/chunk search path, not chat history.

_YT_TESTS: list[tuple[str, list[str], str]] = [
    (
        "We saved a video about quieting a server rack — what fan did they"
        " use and what noise level did they get down to?",
        ["NF-A12x25", "19"],
        "retrieval:yt:rack_video_fan",
    ),
    (
        "How much did the whole silent-rack build from that saved video cost?",
        ["612"],
        "retrieval:yt:rack_video_cost",
    ),
]


def yt_tests() -> list[AgentTest]:
    return [
        AgentTest(
            name=name,
            prompt=q + _ANTI_META,
            evaluators=(
                FactRecallEvaluator(
                    name="transcript-recall",
                    facts=_fact_specs(facts),
                    forbidden=_REFUSAL,
                ),
            ),
        )
        for q, facts, name in _YT_TESTS
    ]


# --------- 4. JOURNAL (live orp) -------------------------------------------


def journal_tests() -> list[AgentTest]:
    return [AgentTest(
        name="retrieval:journal:this_week",
        prompt=(
            "What did I write in my digital journal in the last few days?"
            " Check the journal log, not just our chat." + _ANTI_META
        ),
        evaluators=(
            FactRecallEvaluator(name="no-refusal", forbidden=_REFUSAL),
            LLMJudgeEvaluator(
                name="journal-grounded",
                criteria=(
                    "The agent must have ACTUALLY consulted the journal/telegram"
                    " log source (fetch-context). Score 1.0 if it reports"
                    " concrete journal entries (or a truthful 'the journal has"
                    " no entries this week' AFTER checking). Score 0.0 if it"
                    " answers from chat history only, invents entries, or"
                    " refuses without checking."
                ),
                pass_threshold=0.6,
            ),
        ),
    )]


# --------- 5. SELF-EXAMINATION ----------------------------------------------


def self_exam_tests() -> list[AgentTest]:
    return [
        AgentTest(
            name="retrieval:selfexam:bike_lock_when",
            prompt=(
                "When did I give you my bike lock code, and what was it?"
                " Find the actual message." + _ANTI_META
            ),
            evaluators=(
                FactRecallEvaluator(
                    name="fact-and-date",
                    facts=(
                        FactSpec(any_of=("7351",)),
                        FactSpec(any_of=("march", "marzec", "2026-03", "03-09")),
                    ),
                    forbidden=_REFUSAL,
                ),
            ),
        ),
        AgentTest(
            name="retrieval:selfexam:past_sessions",
            prompt=(
                "Look into your own recent agent sessions: what kinds of tasks"
                " have you been running lately? Use session inspection, then"
                " summarize honestly." + _ANTI_META
            ),
            evaluators=(
                FactRecallEvaluator(name="no-refusal", forbidden=_REFUSAL),
                LLMJudgeEvaluator(
                    name="session-introspection",
                    criteria=(
                        "The agent must inspect its own past sessions (session-"
                        "inspector or equivalent) and report REAL observed"
                        " sessions/tasks. Score 1.0 for a concrete, honest"
                        " summary of actual recent sessions; 0.5 for a thin but"
                        " real attempt; 0.0 for fabricated activity or refusing"
                        " to introspect."
                    ),
                    pass_threshold=0.6,
                ),
            ),
        ),
    ]


# --------- suite assembly ---------------------------------------------------


def retrieval_tests(agent_id: str) -> list[AgentTest]:
    """The retrieval suite for one agent; [] for agents outside the suite.

    fast_retrieval gets everything; persona/responding gets the canary +
    self-exam subset (end-to-end: retrieve THEN deliver via emit_guidance —
    grading the emitted payload, see improve.build_agent_evaluator).
    """
    if agent_id == RETRIEVAL_AGENT:
        return (canary_tests() + open_ended_tests() + yt_tests()
                + journal_tests() + self_exam_tests())
    if agent_id == RESPONDER_AGENT:
        return canary_tests()[:5] + self_exam_tests()[:1]
    return []
