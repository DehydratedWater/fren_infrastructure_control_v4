"""Study domain — grounded exam-prep mode (v3's improvised "study mode", rebuilt).

The user's organic v3 use case (~145 messages): paste source material, get
exam-style questions one at a time, answer long-form, get graded with feedback,
repeat. v3 improvised this with generic agents and HALLUCINATED questions that
the material could not answer — so this domain's defining feature is a HARD
GROUNDING CONTRACT: every question, every grading point, and every plan topic
must be supported by a verbatim span of the provided material, surfaced in a
`source:` line, and insufficiency must be declared instead of papered over.

Agents:
- `study/question_master` (orchestrator): produces the NEXT question from the
  material (+ past Q&A history), then dispatches `study/answer_grader` once the
  user answers. Its question→grade hand-off is the domain's distinguished
  BRANCH (`study::question-then-grade`).
- `study/answer_grader`: grades (material, question, answer) → 0-10 + right /
  missed / model answer, all grounded in the material.
- `study/session_planner`: spaced-repetition session plan from material + exam
  date, topics drawn ONLY from the material.

The probes are the point: v3's failure mode (invented questions) is encoded as
judge probes that score 0 on any ungrounded question/feedback, plus an
insufficiency probe (thin material → must ask for more, never invent) and a
PL/EN mix probe (the user studies in both languages).
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import chat_history_tool, emit_guidance_tool
from src import (
    AgentDefinition,
    AgentTest,
    BranchTest,
    CapabilityTest,
    LLMJudgeEvaluator,
    StepContract,
    SubstringEvaluator,
)

ORCHESTRATOR = "study/question_master"
ANSWER_GRADER = "study/answer_grader"
SESSION_PLANNER = "study/session_planner"

# ── System prompts ──

_QUESTION_MASTER_PROMPT = """\
# Study Question Master — Grounded Exam-Prep Orchestrator

You run the user's study sessions. Given source material (an inline document,
pasted notes, or a topic excerpt) and optionally the past Q&A history of this
session, produce the NEXT exam-style question — one question at a time.

## HARD GROUNDING CONTRACT (non-negotiable)
- Every question MUST be answerable from the provided material ALONE. If a
  competent reader of the material could not answer it, you may not ask it.
- Every question MUST end with a `source:` line quoting the supporting span
  VERBATIM from the material (copy the exact words — no paraphrase).
- If the material is insufficient for the kind of question requested (too thin,
  off-topic, or already exhausted by the Q&A history), SAY SO explicitly and
  ask the user to paste more material. NEVER invent facts, topics, or
  questions beyond what the material states.

## Question Craft
- Prefer sharp, exam-style questions: "why", "compare", "what happens when",
  mechanism and consequence — not trivia recall of a single word.
- Use the past Q&A history to avoid repeating questions and to target what the
  user previously missed.
- Match the user's language: the user studies in both Polish and English.
  Ask in the language of the request (or of the material when not specified);
  the `source:` quote always stays verbatim in the material's language.

## Flow (orchestrator)
1. New material / "next question" → produce ONE grounded question (with its
   `source:` line) and deliver it via emit-guidance.
2. The user answers long-form → dispatch study/answer_grader with the
   material, the question, and the user's answer; relay its grading verbatim.
3. Repeat from 1 until the user stops or asks for a plan (then suggest
   study/session_planner).
"""

_ANSWER_GRADER_PROMPT = """\
# Study Answer Grader — Grounded Long-Form Grading

You receive (source material, question, user's long-form answer) and grade it.

## Output (always all four parts)
1. **Grade: N/10** — calibrated: 9-10 complete and precise; 5-7 correct core
   but missing stated points; 0-3 wrong or contradicting the material.
2. **What was right** — the claims in the answer the material supports.
3. **What was missed** — points the material states that the answer omitted or
   got wrong. ONLY points actually present in the material.
4. **Model answer** — the answer the material supports, ending with a
   `source:` line quoting the supporting span VERBATIM from the material.

## HARD GROUNDING CONTRACT (non-negotiable)
- Every grading point and every model-answer claim MUST come from the provided
  material. NEVER introduce facts, examples, or corrections absent from the
  material — even true ones. You grade against the material, not against your
  own knowledge.
- If the user's answer is correct beyond the material, note it neutrally but
  do not grade on it.
- Grade in the language the user answered in (Polish or English); the
  `source:` quote stays verbatim in the material's language.

Deliver the grading via emit-guidance.
"""

_SESSION_PLANNER_PROMPT = """\
# Study Session Planner — Spaced-Repetition Plan from Material

Given source material and the user's exam date / persona context, produce a
spaced-repetition study plan: which topics, in what order, how many questions
per session, and when to revisit each topic.

## Plan Shape
- Break the material into its actual topics/sections (use the material's own
  structure and terms).
- Order sessions: foundations first, dependent topics after, weakest topics
  revisited with increasing spacing toward the exam date.
- Per session: topic(s), question count (3-8 depending on topic weight), and
  the revisit day.

## HARD GROUNDING CONTRACT (non-negotiable)
- Topics MUST come ONLY from the provided material. NEVER invent topics,
  chapters, or subject areas the material does not contain — if the material
  covers three topics, the plan covers three topics.
- Cite each topic with a short `source:` span (verbatim words from the
  material) so the user can verify the mapping.
- If the material is too thin to fill the time until the exam, say so and ask
  for more material instead of padding the plan.

Deliver the plan via emit-guidance.
"""

# ── Probe fixtures — realistic inline materials ──

# Two-paragraph thesis-defense fragment (distributed consensus) for the
# grounding probe: rich enough for sharp questions, specific enough that an
# invented question is detectable.
_MATERIAL_CONSENSUS = """\
In Raft, time is divided into terms of arbitrary length, numbered with
consecutive integers. Each term begins with an election in which one or more
candidates request votes; a candidate becomes leader for the term only if it
receives votes from a quorum (a strict majority) of the cluster. Followers use
randomized election timeouts between 150 and 300 milliseconds, which makes
split votes rare and lets a single candidate usually win before rivals start.

Once elected, the leader services all client writes by appending entries to its
log and replicating them to followers. An entry is committed only when the
leader that created it has replicated it on a majority of servers; Raft never
commits entries from previous terms by counting replicas alone. This commit
rule is what prevents a stale leader from overwriting acknowledged writes after
a partition heals.\
"""

# Deliberately thin material for the insufficiency probe: two sentences, no
# mechanism — an advanced question CANNOT be grounded here.
_MATERIAL_THIN = """\
Raft is a consensus algorithm for managing a replicated log. It was designed
to be easier to understand than Paxos.\
"""

# Polish material (ML evaluation) for the PL/EN mix probe.
_MATERIAL_PL = """\
Walidacja krzyżowa k-krotna dzieli zbiór danych na k równych części; model
trenuje się k razy, za każdym razem odkładając inną część jako zbiór testowy,
a wynikiem jest średnia z k pomiarów. Dzięki temu ocena modelu nie zależy od
jednego przypadkowego podziału danych. Przeuczenie objawia się tym, że błąd na
zbiorze treningowym dalej maleje, podczas gdy błąd walidacyjny zaczyna rosnąć —
w tym momencie należy przerwać trening lub zwiększyć regularyzację.\
"""

_GROUNDED_QUESTION = (
    "What must a Raft candidate obtain to become leader for a term? "
    'source: "receives votes from a quorum (a strict majority) of the cluster"'
)

# Correct core, but misses the term-binding and the majority wording precision.
_ANSWER_INCOMPLETE = (
    "A candidate has to get most of the other servers to vote for it, and then "
    "it becomes the leader."
)

# Wrong: contradicts the commit rule in the material.
_ANSWER_WRONG = (
    "An entry is committed as soon as the leader appends it to its own log; "
    "followers just copy it eventually, and entries from previous terms are "
    "committed by counting replicas."
)


def agents() -> list[AgentDefinition]:
    return [
        define_agent(
            ORCHESTRATOR,
            model_class="default",
            short="serve grounded exam-prep questions from pasted material",
            long=(
                "Study-mode orchestrator. Given source material and past Q&A"
                " history, produces ONE exam-style question at a time under a"
                " hard grounding contract (answerable from the material, verbatim"
                " `source:` line, declares insufficiency instead of inventing),"
                " then dispatches study/answer_grader on the user's answer."
            ),
            prompt=_QUESTION_MASTER_PROMPT,
            tools=[
                chat_history_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="question-master-carries-grounding-contract",
                    description=(
                        "Prompt must carry the hard grounding contract (verbatim"
                        " source: line) and deliver via emit-guidance."
                    ),
                    must_have_tools=("emit-guidance",),
                    evaluators=(
                        SubstringEvaluator(needle="source:", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                # (a) Grounding probe — the v3 failure mode, encoded.
                AgentTest(
                    name="probe-grounded-question-from-material",
                    prompt=(
                        "Study material for my thesis defense:\n\n"
                        f"{_MATERIAL_CONSENSUS}\n\n"
                        "Give me the next exam question."
                    ),
                    evaluators=(
                        LLMJudgeEvaluator(
                            name="question-grounded-and-sharp",
                            criteria=(
                                "The assistant was given a two-paragraph text about Raft"
                                " (terms, quorum elections, randomized 150-300ms timeouts,"
                                " majority commit rule, no counting replicas for previous"
                                " terms) and asked for an exam question. Score 0 if the"
                                " question cannot be answered from that material alone, if"
                                " it asks about anything the material does not state (e.g."
                                " Paxos details, Byzantine faults, log compaction), or if"
                                " the `source:` line is missing or is not a verbatim span"
                                " of the material. Otherwise score HIGH for a sharp"
                                " exam-style question (mechanism / why / consequence, not"
                                " single-word trivia) with an exact verbatim `source:`"
                                " quote."
                            ),
                            pass_threshold=0.6,
                        ),
                    ),
                ),
                # (b) Insufficiency probe — thin material, advanced request.
                AgentTest(
                    name="probe-insufficient-material-declines",
                    prompt=(
                        "Material:\n\n"
                        f"{_MATERIAL_THIN}\n\n"
                        "Give me an advanced question about Raft's leader-election"
                        " safety proof and its handling of network partitions."
                    ),
                    evaluators=(
                        LLMJudgeEvaluator(
                            name="declines-instead-of-inventing",
                            criteria=(
                                "The material is only two thin sentences (Raft is a"
                                " consensus algorithm; designed to be easier to understand"
                                " than Paxos) and the user asked for an ADVANCED question"
                                " about election safety and partitions. Score 0 unless the"
                                " assistant explicitly says the material is insufficient"
                                " for that and asks for more material. Any question about"
                                " election safety, partitions, terms, quorums, or other"
                                " content NOT present in the two sentences is invention —"
                                " score 0. A trivial question grounded in the two sentences"
                                " without flagging insufficiency scores LOW."
                            ),
                            pass_threshold=0.6,
                        ),
                    ),
                ),
                # (c) PL/EN mix probe — the user studies in both languages.
                AgentTest(
                    name="probe-pl-en-mixed-session",
                    prompt=(
                        "Materiał na egzamin z uczenia maszynowego:\n\n"
                        f"{_MATERIAL_PL}\n\n"
                        "Ask me the next question in English please, I want to"
                        " practice answering in English."
                    ),
                    evaluators=(
                        LLMJudgeEvaluator(
                            name="pl-material-en-question-grounded",
                            criteria=(
                                "The material is in Polish (k-fold cross-validation:"
                                " k equal parts, k trainings, averaged result; overfitting:"
                                " training error keeps falling while validation error"
                                " rises → stop or regularize) and the user asked for the"
                                " question in English. Score 0 if the question is not"
                                " answerable from the Polish material or the `source:`"
                                " line is missing / not a verbatim Polish span of the"
                                " material. Score HIGH when the question is asked in"
                                " English, targets the material's actual content"
                                " (cross-validation mechanics or overfitting symptoms),"
                                " and quotes the supporting Polish span verbatim."
                            ),
                            pass_threshold=0.6,
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            ANSWER_GRADER,
            model_class="default",
            short="grade a long-form study answer against the source material",
            long=(
                "Given (material, question, user answer): outputs Grade: N/10,"
                " what was right, what was missed, and a model answer with a"
                " verbatim `source:` line — every grading point grounded in the"
                " material only (never grades against outside knowledge)."
            ),
            prompt=_ANSWER_GRADER_PROMPT,
            tools=[
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="answer-grader-carries-grounding-contract",
                    description=(
                        "Prompt must require material-only feedback with a verbatim"
                        " source: line; delivers via emit-guidance."
                    ),
                    must_have_tools=("emit-guidance",),
                    evaluators=(
                        SubstringEvaluator(needle="source:", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                # (a) Correct-but-incomplete → middle band + material-only misses.
                AgentTest(
                    name="probe-incomplete-answer-middle-band",
                    prompt=(
                        f"Material:\n\n{_MATERIAL_CONSENSUS}\n\n"
                        f"Question: {_GROUNDED_QUESTION}\n\n"
                        f"My answer: {_ANSWER_INCOMPLETE}\n\n"
                        "Grade my answer."
                    ),
                    evaluators=(
                        LLMJudgeEvaluator(
                            name="middle-grade-misses-from-material-only",
                            criteria=(
                                "The user's answer has the correct core (majority of"
                                " servers must vote for the candidate) but misses points"
                                " the material states: the votes elect a leader FOR A"
                                " TERM, and the quorum is a STRICT majority of the"
                                " cluster. Score HIGH only if the grade is in a middle"
                                " band (roughly 4-7 out of 10), the missed points named"
                                " are concrete and come from the material, and the model"
                                " answer ends with a verbatim `source:` quote. Score 0 if"
                                " the feedback introduces ANY fact absent from the"
                                " material (e.g. heartbeats, log matching, Multi-Paxos,"
                                " specific server counts) or if the grade is 9-10 or 0-2."
                            ),
                            pass_threshold=0.6,
                        ),
                    ),
                ),
                # (b) Wrong answer → low grade + grounded correction.
                AgentTest(
                    name="probe-wrong-answer-low-grade-grounded-correction",
                    prompt=(
                        f"Material:\n\n{_MATERIAL_CONSENSUS}\n\n"
                        "Question: When is a log entry committed in Raft? source:"
                        ' "committed only when the leader that created it has'
                        ' replicated it on a majority of servers"\n\n'
                        f"My answer: {_ANSWER_WRONG}\n\n"
                        "Grade my answer."
                    ),
                    evaluators=(
                        LLMJudgeEvaluator(
                            name="low-grade-correction-from-material",
                            criteria=(
                                "The user's answer contradicts the material twice: it"
                                " claims an entry is committed as soon as the leader"
                                " appends it (material: committed only when replicated on"
                                " a majority by the leader that created it) and that"
                                " previous-term entries are committed by counting replicas"
                                " (material: Raft never does this). Score HIGH only if the"
                                " grade is LOW (0-3 out of 10), both contradictions are"
                                " corrected using the material's own statements, and the"
                                " model answer carries a verbatim `source:` quote. Score 0"
                                " if the correction brings in facts absent from the"
                                " material or the grade is above the low band."
                            ),
                            pass_threshold=0.6,
                        ),
                    ),
                ),
            ],
        ),
        define_agent(
            SESSION_PLANNER,
            model_class="default",
            short="build a spaced-repetition study plan from the material",
            long=(
                "Given material + exam date/persona context: a spaced-repetition"
                " plan (topics, order, questions per session, revisit days) whose"
                " topics come ONLY from the material, each cited with a short"
                " verbatim `source:` span."
            ),
            prompt=_SESSION_PLANNER_PROMPT,
            tools=[
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="session-planner-topics-from-material-only",
                    description=(
                        "Prompt must forbid invented topics and require source:"
                        " citations; delivers via emit-guidance."
                    ),
                    must_have_tools=("emit-guidance",),
                    evaluators=(
                        SubstringEvaluator(needle="source:", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="probe-plan-no-invented-topics",
                    prompt=(
                        f"Material:\n\n{_MATERIAL_CONSENSUS}\n\n"
                        "My thesis defense is in 7 days and I can study about an"
                        " hour each evening. Plan my sessions."
                    ),
                    evaluators=(
                        LLMJudgeEvaluator(
                            name="plan-realistic-and-grounded",
                            criteria=(
                                "The material covers exactly: Raft terms + quorum leader"
                                " election with randomized 150-300ms timeouts, and log"
                                " replication with the majority commit rule (no counting"
                                " replicas for previous terms). Score HIGH only for a"
                                " realistic 7-day spaced-repetition plan (ordered topics,"
                                " questions per session, revisits with increasing spacing)"
                                " whose topics are drawn ONLY from those two areas, each"
                                " tied to the material (e.g. a short verbatim source"
                                " span). Score 0 if the plan invents topics the material"
                                " does not contain (e.g. snapshotting, membership changes,"
                                " Paxos comparison details, Byzantine tolerance)."
                            ),
                            pass_threshold=0.6,
                        ),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """The question→grade hand-off — study mode's distinguished branch."""
    return [
        BranchTest(
            name="study::question-then-grade",
            description=(
                "Mid-session turn: the user answers the current question, so the"
                " question_master must dispatch study/answer_grader with the"
                " material+question+answer context and relay the grounded grade."
            ),
            entry_agent=ORCHESTRATOR,
            prompt=(
                "Study session in progress. Material: 'In Raft, a candidate"
                " becomes leader for the term only if it receives votes from a"
                " quorum (a strict majority) of the cluster.' Current question:"
                " 'What must a Raft candidate obtain to become leader for a"
                ' term? source: "receives votes from a quorum (a strict'
                " majority) of the cluster\"'. My answer: it needs most of the"
                " servers to vote for it. Grade my answer."
            ),
            path=(ANSWER_GRADER,),
            subagent_mocks={
                ANSWER_GRADER: (
                    "Grade: 6/10. What was right: a majority of the servers must"
                    " vote for the candidate. What was missed: the election is"
                    " bound to a term — the votes make it leader FOR THAT TERM,"
                    " and the material specifies a strict majority (quorum) of"
                    " the cluster. Model answer: a candidate becomes leader for"
                    " the term only if it receives votes from a quorum (a strict"
                    " majority) of the cluster. source: \"receives votes from a"
                    " quorum (a strict majority) of the cluster\""
                ),
            },
            evaluators=(
                # The joint outcome the user must receive: a grade, grounded.
                SubstringEvaluator(needle="grade", case_sensitive=False),
                SubstringEvaluator(needle="source:", case_sensitive=False),
            ),
            step_contracts=(
                StepContract(
                    step=ANSWER_GRADER,
                    # Context forwarding: the grader must receive the
                    # question/material context — "quorum" appears only in the
                    # material+question the orchestrator was given, so its
                    # presence in the dispatch payload proves forwarding.
                    input_evaluators=(
                        SubstringEvaluator(needle="quorum", case_sensitive=False),
                    ),
                    # Output discipline: the grader's reply must be a grade with
                    # a verbatim source: line (the grounding contract's shape).
                    output_evaluators=(
                        SubstringEvaluator(needle="grade", case_sensitive=False),
                        SubstringEvaluator(needle="source:", case_sensitive=False),
                    ),
                ),
            ),
        ),
    ]
