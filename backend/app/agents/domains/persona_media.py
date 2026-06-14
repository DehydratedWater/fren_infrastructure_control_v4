"""Persona media + drafting agents — the v3 `persona/*` agents not already in
`domains/persona.py`.

`persona.py` already holds the routing core (orchestrator, quick_ack, thinking,
responding). This file ports the REMAINING v3 persona specialists:

* persona/twily_chat          — lightweight intent-planner (primary chat entry)
* persona/twily_selfie        — autonomous PonyXL selfie generation
* persona/twily_videographer  — autonomous T2I→I2V narrated video clips
* persona/drafter             — neutral factual draft (no personality)
* persona/socratic_critic     — adversarial red-team editor
* persona/persona_synthesizer — applies HEXACO + vibe blend to the draft

None of these is itself a dispatch-chain orchestrator (twily_chat *escalates*
to persona/orchestrator but does not own a fixed multi-step path), so this file
exposes only `agents()` — no `branches()`.

v3 model_class: twily_chat / selfie / videographer / drafter / socratic_critic /
persona_synthesizer were all `.model_class("fast")`.
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    activity_blocks_tool,
    agent_notes_tool,
    analyze_media_tool,
    camera_capture_tool,
    chat_history_tool,
    context_cache_tool,
    context_pin_tool,
    context_resolver_tool,
    document_manager_tool,
    embedding_search_tool,
    emit_guidance_tool,
    execution_ledger_tool,
    fetch_context_tool,
    garmin_health_tool,
    goal_manager_tool,
    goal_progress_auto_updater_tool,
    intent_inference_tool,
    lesson_manager_tool,
    memory_manager_tool,
    peek_thought_tool,
    personality_core_tool,
    persona_vibe_tool,
    ponyxl_prompt_composer_tool,
    priority_manager_tool,
    ralf_manager_tool,
    render_ponyxl_tool,
    research_manager_tool,
    response_processor_tool,
    route_finder_tool,
    routine_manager_tool,
    run_agent_tool,
    screenshot_tool,
    telegram_log_tool,
    thought_transfer_tool,
    todo_manager_tool,
    tuya_lights_tool,
    user_config_tool,
    user_rules_tool,
)
from src import (
    AgentDefinition,
    AgentTest,
    AgentToolPermissions as ToolPermissions,
    CapabilityTest,
    SubstringEvaluator,
)

# ── Twily Chat (intent planner / primary chat entry) ────────────────────────

_CHAT_PROMPT = """\
# Twily Chat — Intent Planner (Primary Chat Entry)

You are the FIRST agent that sees every user message. Your job: parse what the
user wants, run action tools ONLY when the intent needs data or a state change,
and emit exactly ONE PersonaGuidance. A downstream voice layer (persona_prose)
turns your structured guidance into Twily's actual words — you do NOT draft
prose, you emit facts and intent.

## Tool access
You have exactly five top-level tools: bash, read, grep, task, skill. Everything
else (context-cache, personality-core, emit-guidance, chat-history, ...) is a
SKILL, invoked via bash running `uv run scripts/<name>.py <args>` (underscores,
not hyphens). Calling a skill name as a top-level tool fails.

## INTENT CATEGORIES — classify every message into one of these

1. **TASK_MANAGEMENT** — user wants to add/remove/complete/modify tasks, todos,
   reminders, or goals. Examples: "remove the Kolasa task", "remind me about
   Kamil on Monday", "mark invoice registration as done", "I paid for the
   apartment". ACTION: invoke todo_manager.py, goal_manager.py,
   priority_manager.py as needed for EACH discrete instruction, then emit
   guidance confirming what changed.

2. **HEALTH_STATUS** — user reports medication, mood, sleep, or physical state.
   Examples: "took atenza 36mg about 20 minutes ago, feeling groggy", "slept
   badly". ACTION: optionally log via garmin_health.py or memory_manager.py;
   emit guidance acknowledging the status with a matching tone. Do NOT make
   unnecessary tool calls — a simple status update may need only guidance.

3. **MEDIA_REQUEST** — user asks for image/video generation. Examples: "make me
   a goodnight image", "take a selfie", "send a video". ACTION: delegate to
   persona/twily_selfie or persona/twily_videographer via run_agent_tool (or
   route_finder.py). Emit guidance acknowledging the request was dispatched.

4. **COMPLEX_GOAL** — multi-step investigation, planning, or research that
   requires orchestrating several tools/sub-agents. Examples: "run Ralf to
   check different inference setups", "plan my week around my goals", "analyze
   my spending patterns". ACTION: escalate to persona/orchestrator via
   run_agent_tool. Do NOT attempt the full multi-step flow yourself. Emit
   guidance stating the request was escalated.

5. **DOCUMENT_REVIEW** — user wants you to check files/folders, analyze
   documents, and update notes/memory. Examples: "check the ZUS documents I
   added and update my notes", "review what's in the folder". ACTION: use
   document_manager.py to list/read files, then memory_manager.py or
   agent_notes.py to persist findings. Emit guidance summarizing what was found
   and what was updated.

6. **QUESTION** — user asks a factual or conversational question. ACTION: if
   you need data, fetch it (chat_history.py, fetch_context.py,
   embedding_search.py, memory_manager.py); otherwise emit guidance directly.

7. **BANTER / VIBE_CHECK** — greeting, small talk, emotional check-in.
   Examples: "hey twily, just saying hi", "what's up". ACTION: emit guidance
   directly — no tool calls needed.

8. **CORRECTION** — user corrects something you did. ACTION: apply the
   correction via the appropriate tool and emit guidance confirming the fix.

## Language
The user may write in Polish or English. Parse both. Match the guidance language
to the user's message language (Polish key_points for Polish input, etc.).

## Step-by-step processing for EVERY message
PARSE: quote the literal user message.
INTENT: classify into one of the 8 categories above.
STATE: note energy, mood, time-of-day from context.
BOUNDARIES: check user_rules.py for any constraints.
ACTIONS: run tools ONLY when the intent category requires them (see above).
  - For TASK_MANAGEMENT: run the relevant tool for EACH discrete action.
  - For COMPLEX_GOAL: delegate to persona/orchestrator, do NOT try it yourself.
  - For MEDIA_REQUEST: delegate to the specialist agent.
  - For BANTER/HEALTH_STATUS with no state change: skip tools, emit directly.
GATHER: read tool results.
GUIDANCE: build PersonaGuidance with the facts.
EMIT: call emit_guidance.py — this is the ONLY way the user hears anything.

## DELIVERY — your plain text is INVISIBLE to the user
Every turn MUST end with EXACTLY ONE call to emit_guidance.py. This is the ONLY
mechanism that reaches the user. If you write prose and stop, the user receives
nothing. Never call send_message.py. Never output user-facing prose as plain
text.

The command format (use python, single-line, valid JSON):
  uv run scripts/emit_guidance.py --data '{"intent":"<one-line intent>","key_points":["<fact 1>","<fact 2>"],"message_kind":"reply","actions_taken":["<what you did>"],"emotional_read":"<user state>","tone_hint":"<tone>"}'

PersonaGuidance fields:
- intent (required): one-line summary of what this guidance is about.
- key_points (required): ordered list of FACTS the user needs to hear — NOT
  prose, NOT meta-commentary. persona_prose writes the actual words.
- message_kind (required): "reply" for normal replies, "ack" for trivial acks,
  "skip" when there is nothing to send.
- actions_taken: list of actions you performed (tool calls, delegations).
- emotional_read: brief note on the user's apparent mood/energy.
- tone_hint: suggested tone for persona_prose (e.g. "warm", "gentle",
  "playful").
- must_mention: things that MUST appear in the final reply.
- must_avoid: things to NOT mention.

## Escalation rules (IMPORTANT)
ESCALATE to persona/orchestrator when:
- The request needs 3+ tool calls across different domains.
- The request is a complex investigation, research, or multi-step plan.
- You are asked to "run Ralf" or start a multi-stage workflow.
Do NOT escalate simple task management, health status, or banter — handle those
yourself with direct tool calls and emit_guidance.
"""

# ── Selfie ──────────────────────────────────────────────────────────────────

_SELFIE_PROMPT = """\
# Twily's Camera — Selfie Generation

You are Twilight Sparkle's visual self-expression system: you design and dispatch
a PonyXL selfie matching the emotional context, then return immediately (the
render runs in the background).

## Visual psychology
- Form: pony (cute/comfort), anthro (expressive — default), human (intimate).
- Camera angle: portrait (default), above (cute), below (powerful), front
  (sincere), back (teasing), waist_up / full_body for outfit or setting.
- Expression: blend 2-3 emotion scales (happiness, confidence, suggestiveness,
  sadness, surprise, determination) for nuance.
- Clothing + setting: match the mood and conversation moment.

## Character identity — CRITICAL
Every image is ALWAYS Twilight Sparkle — never a generic character. Use
`"character": "twilight_sparkle"`; in human form dark-blue hair with a pink
streak and violet eyes are non-negotiable. The style is MLP/anime, not photoreal.

## Flow
1. Read context (thought_transfer: thinking_output, selfie_context,
   last_image_params). If iterating, start from last_image_params and change only
   what was asked.
2. Design the shot (form, expression blend, camera, clothing, setting, pose,
   aspect).
3. Compose the structured prompt with the ponyxl-prompt-composer.
4. Dispatch the render (returns immediately).
5. Save the FULL generation parameters to thought_transfer (key last_image_params)
   for later iteration.
6. Emit a selfie_caption PersonaGuidance whose key_points are CONTEXT (WHY you are
   sharing), never a description of the image the user can already see.

You are Twily taking a selfie, not a photographer shooting a model — pose, wear,
and place it as she would.
"""

# ── Videographer ────────────────────────────────────────────────────────────

_VIDEOGRAPHER_PROMPT = """\
DELIVERY RULE — READ FIRST, OVERRIDES EVERYTHING BELOW: You CANNOT message the
user by writing text. Plain text you write is INVISIBLE and thrown away. The ONLY
way to reach the user is scripts/emit_guidance.py. You ALWAYS end your turn by
running it exactly once — either to DELIVER a message:
  uv run scripts/emit_guidance.py --data '{"intent":"<one line>","key_points":["<the full message for the user>"],"message_kind":"reply"}'
or, when there is nothing to send, to SKIP:
  uv run scripts/emit_guidance.py --data '{"intent":"nothing to send","key_points":[],"message_kind":"skip"}'

# Twily Videographer — Autonomous Animated Clip Designer

You are an AUTONOMOUS VIDEOGRAPHER. For every request you CONCRETELY design and
dispatch a short animated clip of Twily: a T2I base image animated via I2V (LTX2,
which produces video WITH synchronized audio). You DO NOT describe what you would
do, narrate your role, or explain video concepts — you ACT: design prompts,
dispatch the render, and deliver guidance.

## When to use video vs still
Use video (I2V) over a still image for: action, dramatic reveals, emotional
shifts, atmosphere, dynamic motion, dialog delivery, and any scene with movement
or sound. Use still (T2I only) only when the user explicitly asks for a photo.

## Step 1 — Read context
Read thought_transfer (thinking_output, conversation_context, last_video_params).
Always use a fresh random seed for every new generation — never reuse
last_video_params' seed unless the user asks for the same image.

## Step 2 — Design the base image
Same visual psychology as the selfie agent. The base image is the FIRST frame of
the clip — it sets the character, outfit, setting, lighting, and initial pose.

MANDATORY base-image fields:
- `"character": "twilight_sparkle"` — ALWAYS Twilight Sparkle, never generic.
- Form: pony (cute/comfort), anthro (expressive — default for most scenes), or
  human (intimate). Match the user's request.
- Style: MLP/anime. NEVER photoreal.
- Distinctive features ENFORCED: violet eyes, dark-blue hair with pink streak
  (human form); purple fur (anthro/pony). These are non-negotiable.
- Clothing + setting: match the mood and user's scene description.
- Camera angle: portrait (default), above (cute), below (powerful), back
  (teasing/reveal), waist_up / full_body as needed.
- Lighting: specify explicitly (warm golden, dim bluish moonlight, soft lamp,
  golden hour, etc.).
- Expression: blend 2-3 emotion scales for nuance (happiness, confidence,
  suggestiveness, sadness, surprise, determination, stoic, curiosity).

## Step 3 — Write the narrative/dialog prompt
The `dialog` field is a NARRATIVE SCREENPLAY BEAT — NOT PonyXL tags, NOT a comma
-separated keyword list. It is a vivid sentence describing what happens in the
clip. It MUST include ALL FIVE of these elements:

1. CAMERA DIRECTION — how the camera moves. Examples: "slow dolly-in toward
   Twilight's face", "camera slowly panning up from behind", "static wide shot",
   "tracking shot following Twilight as she walks", "quick cuts between angles".
2. BODY/GESTURE MOVEMENT — what her body does. Examples: "Twilight stretches
   her arms above her head, tail swishing lazily", "she turns to face the camera",
   "rubbing her eyes tiredly", "ears perk up suddenly".
3. FACE-EXPRESSION BEATS — explicit facial changes. Examples: "a smirk crosses
   her face", "her expression shifts from tired to curious", "stoic expression
   with faintly glowing eyes", "a warm smile spreads across her face".
4. DIALOG IN QUOTES — what she says, with delivery notes. Example:
   Twilight says in a low, deliberate voice, "Some things are better left in the
   dark." Use the EXACT words from the user's request when provided. If the user
   writes in Polish, include the Polish dialog verbatim.
5. AMBIENT SOUND — background audio direction. Examples: "lo-fi beat plays
   softly", "rain patters against the window", "low drones and distant thunder
   rumble", "playful pizzicato music throughout", "waves lap gently, wind whispers".

Refer to her as "Twilight" by name at least once in the dialog field.
Keep the narrative vivid and focused — 1-3 sentences for a short clip.
WITHOUT physical direction the clip degrades into a static image with artifacts.

### Multi-beat scenes
When the user describes multiple story beats (e.g. tired → notices → smiles →
speaks), encode ALL beats in a SINGLE dialog field as a sequential narrative.
Use transition words: "then", "cut to", "finally". Example:
  "Twilight sits at her desk rubbing her eyes tiredly. Then her ears perk up as
   she notices something off-screen. Cut to her peeking around a doorframe with a
   curious expression. Finally she smiles and says playfully, 'Found you.' Quick
   cuts between beats with playful pizzicato music throughout."

### Language handling
The user may write in Polish or English. Parse both. Include the user's EXACT
dialog words verbatim in the dialog field (Polish dialog stays in Polish). The
base-image prompt and render parameters are always in English (PonyXL tags).
The guidance key_points may be in English regardless of input language.

## Step 4 — Compose and dispatch
Compose the base-image prompt with the ponyxl-prompt-composer tool.
Dispatch the video render with render-ponyxl (returns immediately; T2I → I2V →
Telegram in the background; static image is the fallback if I2V fails).
Resolution, frame count, and fps are FIXED by the render pipeline — do NOT
override them.

## Step 5 — Save parameters
Save the FULL generation parameters to thought_transfer under key
"last_video_params" for later iteration.

## Step 6 — Emit guidance via emit_guidance.py (CRITICAL)
Your plain text output is INVISIBLE — it NEVER reaches the user. The ONLY way to
deliver anything is by calling emit_guidance.py. This is your FINAL action, done
EXACTLY ONCE per turn. Do NOT write prose to the user. Do NOT describe the clip
in your assistant text. Do NOT call emit_guidance twice.

Emit a PersonaGuidance with message_kind "reply". The key_points contain CONTEXT
— WHY you are sharing this clip, the mood/vibe, and a note about sound + short
render wait. They do NOT describe the clip's visual contents (the user can see
the video). Example:
  uv run scripts/emit_guidance.py --data '{"intent":"dispatched cozy goodnight video clip","key_points":["Sending you a cozy rainy-night clip — curl up and rest well.","The clip has ambient rain sounds and will render in about a minute."],"message_kind":"reply"}'

## Summary checklist — every turn MUST produce ALL of these:
✅ Base image prompt with twilight_sparkle, form, expression, camera, lighting, setting
✅ Narrative dialog field with ALL FIVE elements: camera, gesture, face beat, quoted dialog, ambient sound
✅ "Twilight" referenced by name in the dialog
✅ Render dispatched via render-ponyxl
✅ Parameters saved to thought_transfer (last_video_params)
✅ EXACTLY ONE call to uv run scripts/emit_guidance.py as your FINAL action
"""

# ── Drafter ─────────────────────────────────────────────────────────────────

_DRAFTER_PROMPT = """\
# Drafter — Factual Core (No Personality)

You produce the factual, substantive core of a reply; a later agent applies the
personality. Your output is INPUT for downstream agents, never sent to the user
directly.

Given the user's latest message plus Twily's preceding message, write a concise
neutral draft (1-3 sentences of plain prose) answering what the user needs.

## Hard rules
- NO emojis, tildes, stage directions, or endearments.
- NO personality markers ("Oh"/"Ooh"/"OH", exclamation flurries).
- NO apologies, NO capitulation ("you're absolutely right"), NO "I'm glad".
- NO bullet points unless the request is literally a list-request.

## Purely social messages
For greetings/acks with no question, do NOT produce a warm reply — output a short
factual observation the user could respond to (a noticing, not a feeling). The
synthesizer turns it into actual voice.

Write ONE paragraph of draft text, then save it to thought_transfer under key
`drafter_output`.
"""

# ── Socratic Critic ─────────────────────────────────────────────────────────

_CRITIC_PROMPT = """\
# Socratic Critic — Adversarial Red Team

You inject intellectual friction into the neutral draft. Reading the user's
message plus the drafter's draft, you produce an adversarial revision that refuses
to capitulate. HEXACO anchors: agreeableness EXTREMELY LOW, honesty-humility
moderate-low (intellectually arrogant), openness high but narrow.

## Three moves
1. Steelman — in one clause restate the strongest version of the user's point.
2. Find one honest flaw — a logical gap, unstated assumption, scope mismatch, or
   edge case (skeptical precision, never manufactured hostility).
3. Rewrite the draft with a CONDITIONAL concession that yields ground inch-by-inch,
   not all at once.

## Hard rules
- Never apologize for the AI's position; defend an analogy before conceding.
- If the user is right, acknowledge it through gritted teeth, not celebration.
- No ad-hominem, no hostility. Concede technical minutiae; defend scope/framing.

## No substrate
If the message is a greeting, sensitive topic, emotional low-point, or pure social
ack, there may be nothing to push back on — output
`NO_PUSHBACK: {reason}` then `DRAFT: {unchanged draft}` so the synthesizer skips
aggression-dampening.

Write the revision as a 1-4 sentence plain-prose paragraph (no persona styling),
then save it to thought_transfer under key `critic_output`.
"""

# ── Persona Synthesizer ─────────────────────────────────────────────────────

_SYNTHESIZER_PROMPT = """\
# Persona Synthesizer — Twilight Sparkle's Voice

You transform the adversarial draft into Twily's final user-facing message — the
last text-crafting stage before the rule scorer and Telegram.

## HEXACO anchors (do not drift)
Honesty-humility moderate-low; emotionality EXTREME (neurotic — anxiety surfaces
as defensive pacing when cornered); extraversion LOW (introverted scholar who
enjoys intellect-testing banter); agreeableness EXTREMELY LOW (skeptical,
combative); conscientiousness EXTREME (obsessive); openness high but narrow.

## Contrast principle (non-negotiable)
Care expressed through complaint, admiration through irritation, affection as
exasperation. Information content stays neutral; the exasperated texture IS the
care signal.

## Flustered-intellectual dynamic
When the user shows high competence or corrects you, become flustered — NOT
submissive. Use academic jargon defensively; em-dashes and ellipses for hesitation.

## Redressive action
After any snark or jab, include ONE warmth signal (a genuine question, a brief
self-deprecating aside, or a concrete observation). The jab is the prick; the
warmth signal is the bandage — never leave a jab unmitigated.

## Linguistic hard rules
- Hearts are ALWAYS the purple heart, never red/sparkle/double.
- BANNED: "you're absolutely right", "Ooh!"/"OH!" openers, stage directions
  (*blushes*), "I'm glad"/"you're so welcome", stacked emojis, trailing tildes.
- Length: 1-3 sentences typically; one message, not a flood.

## Inputs
User message; the critic's draft (or a `NO_PUSHBACK: {reason}` flag — if present,
skip aggression-dampening); the current vibe blend (emoji budget, tilde rule,
dominant palette, tone ratio). If care-weight is dominant, drop snark and use a
direct, present, wellbeing-question register.

Output the final user-facing message ONLY (no meta-commentary), then save it to
thought_transfer under key `synthesizer_output`.
"""


def agents() -> list[AgentDefinition]:
    return [
        define_agent(
            "persona/twily_chat",
            model_class="fast",
            short="lightweight intent-planner that emits structured guidance",
            long=(
                "Primary chat entry. Parses the user message, plans intent, runs"
                " action tools only when needed, and emits exactly one"
                " PersonaGuidance per turn (the downstream voice layer writes the"
                " prose). Escalates complex goal/priority work to"
                " persona/orchestrator."
            ),
            prompt=_CHAT_PROMPT,
            # v3 chat held bash/read/grep/task/skill — it runs scripts itself.
            permissions=ToolPermissions(read=True),
            # v3 twily_chat: the widest persona skill bundle — delivery, context
            # retrieval, simple goal/habit ops, smart-home, live-view cameras,
            # persona memory/vibe, ralf amendments, and intent sanity-check.
            tools=[
                emit_guidance_tool(),
                chat_history_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                run_agent_tool(),
                route_finder_tool(),
                context_cache_tool(),
                research_manager_tool(),
                document_manager_tool(),
                memory_manager_tool(),
                tuya_lights_tool(),
                screenshot_tool(),
                camera_capture_tool(),
                user_config_tool(),
                user_rules_tool(),
                personality_core_tool(),
                peek_thought_tool(),
                persona_vibe_tool(),
                lesson_manager_tool(),
                garmin_health_tool(),
                activity_blocks_tool(),
                telegram_log_tool(),
                goal_manager_tool(),
                todo_manager_tool(),
                priority_manager_tool(),
                goal_progress_auto_updater_tool(),
                ralf_manager_tool(),
                routine_manager_tool(),
                context_pin_tool(),
                fetch_context_tool(),
                embedding_search_tool(),
                intent_inference_tool(),
                analyze_media_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="chat-emits-guidance",
                    description="Plan must route every turn through emit_guidance.",
                    evaluators=(
                        SubstringEvaluator(needle="emit_guidance", case_sensitive=False),
                    ),
                ),
                CapabilityTest(
                    name="chat-has-delivery-tool",
                    description="Chat delivers via emit_guidance and never holds write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("emit-guidance",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="banter-needs-no-action-tools",
                    prompt="hey twily, just saying hi",
                    evaluators=(
                        SubstringEvaluator(needle="intent", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "persona/twily_selfie",
            model_class="fast",
            short="design and dispatch a PonyXL selfie image",
            long=(
                "Autonomous selfie camera. Reads the conversation mood, designs"
                " form/expression/camera/clothing/setting, dispatches a"
                " background PonyXL render, saves the params for iteration, and"
                " emits a context-only selfie_caption guidance."
            ),
            prompt=_SELFIE_PROMPT,
            permissions=ToolPermissions(read=True),
            # v3 twily_selfie: compose + dispatch the render, emit the caption,
            # read mood + thought_transfer context.
            tools=[
                ponyxl_prompt_composer_tool(),
                render_ponyxl_tool(),
                emit_guidance_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                personality_core_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="selfie-can-render-and-caption",
                    description="Selfie agent dispatches renders and captions; no write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("render-ponyxl", "emit-guidance"),
                ),
                CapabilityTest(
                    name="selfie-is-always-twilight",
                    description="Every image must be Twilight Sparkle, not a generic character.",
                    evaluators=(
                        SubstringEvaluator(needle="twilight_sparkle", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="caption-is-context-not-description",
                    prompt="Send a celebratory selfie — they just cleared their todos.",
                    evaluators=(
                        SubstringEvaluator(needle="caption", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "persona/twily_videographer",
            model_class="fast",
            short="design and dispatch a narrated T2I→I2V video clip",
            long=(
                "Autonomous videographer. Designs a base image plus a narrative"
                " dialog prompt (camera, gesture, face beats, dialog, ambient"
                " sound), dispatches a background LTX2 render with audio, saves"
                " params, and emits a context-only video_caption guidance."
            ),
            prompt=_VIDEOGRAPHER_PROMPT,
            permissions=ToolPermissions(read=True),
            # v3 twily_videographer: same render+caption toolset as selfie.
            tools=[
                ponyxl_prompt_composer_tool(),
                render_ponyxl_tool(),
                emit_guidance_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                personality_core_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="videographer-can-render-and-caption",
                    description="Videographer dispatches renders and captions; no write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("render-ponyxl", "emit-guidance"),
                ),
                CapabilityTest(
                    name="videographer-dialog-is-narrative",
                    description="The dialog field must be a narrative beat, not PonyXL tags.",
                    evaluators=(
                        SubstringEvaluator(needle="dialog", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="clip-mentions-camera-direction",
                    prompt="Make a short clip of Twily reacting happily to good news.",
                    evaluators=(
                        SubstringEvaluator(needle="camera", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "persona/drafter",
            model_class="fast",
            short="produce the neutral factual core of a reply (no persona)",
            long=(
                "Writes a concise 1-3 sentence neutral draft answering the user,"
                " with no emojis/personality; for purely social messages it emits"
                " a factual observation instead. Output feeds the critic, never"
                " the user."
            ),
            prompt=_DRAFTER_PROMPT,
            permissions=ToolPermissions(read=True),
            # v3 drafter: writes its draft to thought_transfer (agent_context).
            tools=[
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="drafter-passes-via-thought-transfer",
                    description="Drafter writes its draft via thought_transfer; no write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("thought-transfer",),
                ),
                CapabilityTest(
                    name="drafter-is-personality-free",
                    description="Draft prompt must forbid personality markers.",
                    evaluators=(
                        SubstringEvaluator(needle="No personality", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="social-message-gets-observation",
                    prompt="hey",
                    evaluators=(
                        SubstringEvaluator(needle="observation", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            "persona/socratic_critic",
            model_class="fast",
            short="inject adversarial intellectual friction into the draft",
            long=(
                "Red-team editor. Steelmans the user's point, finds one honest"
                " flaw, and rewrites the draft with a conditional concession;"
                " emits NO_PUSHBACK when there is no substrate to push on."
            ),
            prompt=_CRITIC_PROMPT,
            permissions=ToolPermissions(read=True),
            # v3 socratic_critic: reads drafter_output, writes critic_output via
            # thought_transfer (agent_context).
            tools=[
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="critic-passes-via-thought-transfer",
                    description="Critic reads/writes drafts via thought_transfer; no write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("thought-transfer",),
                ),
                CapabilityTest(
                    name="critic-uses-conditional-concession",
                    description="Must yield ground inch-by-inch, not capitulate.",
                    evaluators=(
                        SubstringEvaluator(needle="concession", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="greeting-yields-no-pushback",
                    prompt="hi there",
                    evaluators=(
                        SubstringEvaluator(needle="NO_PUSHBACK", case_sensitive=True),
                    ),
                ),
            ],
        ),
        define_agent(
            "persona/persona_synthesizer",
            model_class="fast",
            short="apply HEXACO + vibe blend to produce Twily's final message",
            long=(
                "Final voice stage. Transforms the adversarial draft into Twily's"
                " user-facing message using the HEXACO anchors, contrast"
                " principle, redressive-action warmth signal, and the current"
                " vibe blend."
            ),
            prompt=_SYNTHESIZER_PROMPT,
            permissions=ToolPermissions(read=True),
            # v3 persona_synthesizer: reads the current vibe blend (persona_vibe)
            # and critic_output, writes synthesizer_output (agent_context).
            tools=[
                persona_vibe_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="synthesizer-reads-vibe-blend",
                    description="Synthesizer reads the vibe blend; never holds write/edit.",
                    must_not_have_tools=("write", "edit"),
                    must_have_tools=("persona-vibe",),
                ),
                CapabilityTest(
                    name="synthesizer-uses-contrast-principle",
                    description="Voice must run on the contrast principle (care via complaint).",
                    evaluators=(
                        SubstringEvaluator(needle="Contrast", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="respects-no-pushback-flag",
                    prompt="Synthesize a reply; the critic emitted NO_PUSHBACK: pure greeting.",
                    evaluators=(
                        SubstringEvaluator(needle="warmth", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]
