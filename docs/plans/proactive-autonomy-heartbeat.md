# Proactive Autonomy Heartbeat

Status: PROPOSED · Owner: fren_v4 · Supersedes the `periodic_checker`,
`winddown`, `evening_focus`, and `night_analysis` cron agents.

## 1. Intent (the role we actually want)

The agent talks with the user, then **does things in the background** — observes
repeating patterns, forges insights, plans strategies, tracks agreements. The
heartbeat is the thing that periodically **wakes the agent up** and asks one
question:

> *"Given everything I know and everything I've been doing, is there a genuine
> reason to reach the user right now — to remind, nudge, share something
> interesting I worked out, execute a plan I made, or route to a specialist —
> and if so, what and how?"*

It is the agent's **agency loop**, not a reminder trigger.

## 2. Current state — verdict: the intent is ~half covered

`periodic_checker` today is a **task/calendar reminder engine**:

- The tool (`app/tools/system/periodic_checker.py`) computes 11 deterministic
  checks — all calendar/todos/routines/time-blocks + two regex conversation
  scans. It reads `todos`, `calendar`, `daily_routines`, `daily_strategies`
  (time-blocks only).
- The agent prompt says it verbatim: *"Twily's 5-minute proactive **reminder**
  engine."* Trigger priority is entirely calendar/todos/tasks.

**Covered:** overdue todos, calendar, pending tasks, routines, reschedule
nudges; crude dropped-thread (Twily's own unfulfilled promises) and task-phrase
detection; availability/cooldown/anti-repetition discipline.

**Missing vs intent:**

1. **The agent's own internal work is never surfaced.** `thought_forger` writes
   motivation-scored `pending_thoughts` every 30 min, but **nothing reads them
   proactively** — only the *reactive* chat path consumes them. Forged insights
   reach the user only if the user starts a conversation, else they expire in
   the daily prune. This is the biggest gap.
2. **Conversation agreements** — no semantic understanding (only `"I need to…"`
   regex). Can't notice "we agreed Tuesday you'd start running."
3. **Procrastination** — only idle-time proxies; no "you said this mattered and
   you've been avoiding it" reasoning.
4. **Autonomy to act/route** — the heartbeat only emits *nudge* or *skip*. It
   can't execute a plan it made or route to a specialist.
5. **Strategies/observations** (`daily_strategies`, `inner_monologue`) aren't
   integrated into the wake-up decision.

Root cause: a **hardcoded deterministic trigger tool feeding a constrained
reminder agent** structurally cannot notice the semantic/novel cases. That is
why a string/cooldown gate is the wrong primitive.

`winddown`, `evening_focus`, `night_analysis` are **the same wake-up loop** with
different time-windows, personas, and actions — currently duplicated as separate
narrow agents.

## 3. Target architecture — one heartbeat, several modes

A single in-process **triage engine** runs every tick. It is NOT an opencode
agent and NOT a tool-loop. It is one structured reasoning call over
deterministically pre-assembled context, followed by deterministic routing.

```
tick(mode):                          # mode = day | evening | winddown | night
  if novelty_pregate(mode) == NOTHING_NEW: return        # conservative, see §3.4
  evidence = assemble_evidence(mode)                      # deterministic Python, §3.2
  decision = triage(evidence, mode)                       # ONE qwen call, thinking ON, §3.1
  route(decision, mode)                                   # deterministic, §3.3
```

### 3.1 The triage call (request/response, NOT LangGraph)

- Runs on the existing `src/interactive` primitive (same tier as first_contact)
  — **no opencode subprocess, no tool-loop**.
- Local qwen, **thinking ON** (the decision is genuinely complex now), **generous
  `max_tokens`** so the reasoning + structured answer never truncate.
- **256k context**: we dump the full evidence picture in one shot — no iterative
  fetch needed. **Multimodal**: include up to 10 recent activity screenshots /
  camera frames as `image_url` parts when relevant (esp. winddown/procrastination).
- **Structured output** (forced via `output_schema`):

  ```jsonc
  {
    "decision": "skip | message | escalate | act",
    "category": "reminder | nudge | share_insight | agreement_followup |
                 procrastination | dropped_thread | plan_execution |
                 winddown | other",
    "urgency": 0-5,                       // drives winddown escalation policy
    "reasoning": "why (thinking summary, for the dashboard + audit)",
    "draft": "the message to send, in Twily voice, when decision=message",
    "route": { "agent": "<id>|null", "action": "<lights_off|camera|...>|null" },
    "confidence": 0.0-1.0
  }
  ```

Why request/response over LangGraph: the flow is linear (assemble→reason→route),
the context is pre-loadable into 256k (no "decide what to fetch next"), it is far
more **testable** (fixture in → assert decision out), and it avoids the tool-loop
flail. Reserve LangGraph for genuine multi-step autonomous *execution* (plan→act→
observe→re-plan) if we build that later.

### 3.2 Evidence assembly (deterministic, the part that fills the gaps)

Assembled in Python before the call — this is where the missing signals enter:

- **Pending thoughts** — top-N `pending_thoughts` by `motivation_score` (unconsumed).
  *(fixes gap #1)*
- **Open agreements / commitments** — extracted/maintained from chat (see §5).
  *(gap #2)*
- **Procrastination signals** — declared-important items vs activity/event
  evidence of avoidance. *(gap #3)*
- **Dropped threads** — user questions/topics with no resolution; Twily promises
  unfulfilled.
- **Strategies & observations** — `daily_strategies`, latest `inner_monologue`,
  `conversation_digest`. *(gap #5)*
- **Deterministic task triggers** — the existing `periodic_checker` tool output,
  fed as **evidence, not as the gate**.
- **Recent activity** — `activity_blocks` summary + recent screenshots/camera
  frames (multimodal).
- **State** — last_user_age, last_bot_age, cooldowns, mode/time-window.

### 3.3 Routing (deterministic Python on the structured decision)

- `skip` → nothing; record reason. (Most day-mode ticks.)
- `message` → render `draft` via `persona_prose` (cheap, no opencode agent) and
  deliver through the existing delivery gate (cooldown/availability respected).
- `escalate` → spawn the full heavy deliberation agent (the current
  `goals/periodic_checker` lineage) with the evidence — for nuanced/novel cases.
- `act` → execute `route.action` (lights off, camera, selfie) or spawn
  `route.agent` (specialist / plan execution). *(fixes gap #4)*

### 3.4 Novelty pre-gate (optional, conservative — NOT semantic)

Skip the LLM call only when **provably nothing changed** since the last tick
across the *union* of all evidence sources (no new msgs, no new pending_thoughts,
no new overdue/agreements, no time-block boundary crossed, no procrastination
timer elapsed). This is input-presence, not judgment — it cannot drop a
semantic case because it fires only when there is literally no new input. Saves
GPU on dead ticks. Always runs the full triage when in doubt.

### 3.5 Modes (unifies winddown / evening / night)

One engine, a `mode` parameter sets policy + available actions:

| mode | window (Warsaw) | persona / policy | extra actions |
|---|---|---|---|
| `day` | 08–21 | reminders, nudges, share-insight, procrastination | route specialists |
| `evening` | 21–24 | evening focus / wind-up | — |
| `winddown` | 00–05 | escalating sleep urgency by `urgency` (gentle→relentless) | camera, lights_off, sleepy selfie |
| `night` | ~02 | reflection / next-day prep (low/no delivery) | write digests |

The winddown escalation ladder becomes the `urgency` field + a mode policy block,
not a separate agent.

## 4. Cron wiring

Replace the separate `periodic_check` / `winddown` / `evening_focus` /
`night_analysis` jobs with `script:scripts/heartbeat.py` whose `--mode` (or
derived from local hour) selects the policy. One code path, time-windowed.
Keep the deterministic per-category cooldowns + availability as **delivery
guards** (so we don't re-ping), never as the wake decision.

## 5. Data / infra to build

- **Agreements/commitments store + extractor** — lightweight: a cheap pass (can
  ride the existing `event_extractor` or a small structured call) that maintains
  open commitments from chat. New table or reuse `agent_notes`.
- **Procrastination signal** — derive from declared-important items + activity
  evidence; expose as an evidence field.
- **Proactive `pending_thoughts` consumption** — the heartbeat reads top
  motivation-scored thoughts and marks `consumed_at`/`consumed_by` when surfaced.
- **Activity frames access** — pull recent `activity_observation` screenshots/
  camera frames for the multimodal context (cap 10).
- **`heartbeat.py`** + `app/agents/heartbeat.py` (engine, mirrors
  `first_contact.py`), mode policies, decision schema.

## 6. Testing — in-depth (hard requirement)

- **Deterministic fixtures**: assembled-evidence → assert decision, per mode and
  per category (golden cases below). The triage being a pure `evidence→decision`
  function makes this clean.
- **Golden cases** (must pass): forged-thought-worth-sharing → `message
  share_insight`; agreement slipping → `escalate`/`message agreement_followup`;
  clear procrastination → `message procrastination`; dropped thread → followup;
  genuine quiet tick → `skip`; user busy → `skip`; winddown past cutoff →
  `act lights_off` + escalating urgency.
- **v3 corpus replay** (port 5452, 19.8k msgs): replay real situations; check the
  heartbeat surfaces what a human would and stays quiet otherwise.
- **Multi-sample eval** (thinking-on is stochastic): run each probe N≥3, aggregate
  by median — a single stochastic blank can't floor a probe. Skip infra-timeout
  samples (don't score them 0). [[feedback-autoloop-infra-noise]]
- **Autoloop-optimizable probes**: ship probes + evaluators so the decision prompt
  is tunable in the loop. [[feedback-autoloop-optimizable-or-incomplete]]
- **Shadow mode first**: run the heartbeat alongside the current agents, **log
  decisions, deliver nothing**, for a day; diff against what the old engine did +
  spot false-skips/false-fires. Only then flip delivery on, per mode.

## 7. Rollout

1. Engine + schema + evidence assembly (pending_thoughts + existing triggers
   first) + fixtures. Shadow mode on `day`.
2. Add agreements + procrastination + dropped-thread signals + multimodal frames.
3. Validate against shadow logs + v3 replay; enable `day` delivery; retire
   `periodic_checker` agent.
4. Add `winddown`/`evening`/`night` modes; validate; retire those agents.

## 8. GPU / cost

Per tick: one in-process thinking call replaces a full opencode+tool-loop agent
run (already thinking-on today) — fewer round-trips, no subprocess. Skip ticks
(most of `day`) avoid all downstream specialist/render work. The novelty pre-gate
removes dead-tick calls entirely. Net: comparable-or-lower per-tick cost, far
higher role coverage. Runs on the local-qwen bg lane (priority 100) so user
replies always preempt it.

## 9. Open questions

- Agreements: dedicated extractor vs fold into `event_extractor`'s pass?
- `act` autonomy scope on day mode — which specialist spawns are allowed without
  confirmation?
- Per-mode tick cadence (day */5, winddown */5, night once) — keep or tune?
