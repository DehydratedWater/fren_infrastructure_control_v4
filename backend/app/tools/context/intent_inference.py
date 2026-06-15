"""Intent Inference — detect user intents from message text."""

from __future__ import annotations

import re

from src import ScriptTool
from pydantic import BaseModel, Field


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for intent matching.

    The single source of truth for how a message is normalised before the
    INTENT_PATTERNS are searched. `_infer` and the bot's fast-path
    (`app.telegram.handlers._try_fast_path` / `_try_media_agent`) both import
    this — previously the fast-path imported a `_normalize` that did not exist,
    so its import raised and the fast-path silently no-op'd (every message fell
    through to the slow orchestrator). Defining it here revives the fast-path.
    """
    return re.sub(r"\s+", " ", str(text)).strip().lower()


# Intent patterns: (regex, intent_type, description)
# Order matters — first match per type wins. More specific patterns first.
INTENT_PATTERNS: list[tuple[str, str, str]] = [
    # ── Task completion (direct + indirect) ──
    (r"\b(i did|i've done|finished|completed|done with|marked.*done)\b", "task_completion", "User completed a task"),
    (
        r"\b(i returned|i sent.*back|already.*(?:returned|sent|gave|dropped off|shipped))\b",
        "task_completion",
        "User completed a return/delivery",
    ),
    (
        r"\b(already.*(?:done|finished|handled|taken care|sorted|fixed|paid|bought|resolved))\b",
        "task_completion",
        "User already handled something",
    ),
    (
        r"\b(i've already|already.*it|sent it back|gave it back|dropped it off|took care of)\b",
        "task_completion",
        "Indirect task completion",
    ),
    # ── Task creation ──
    (
        r"\b(i need to|i have to|i should|remind me|don't let me forget|add.*(?:to my|to the) (?:list|tasks|todos))\b",
        "task_creation",
        "User wants to create a task",
    ),
    (r"\b(oh i should|i gotta|need to remember|i must)\b", "task_creation", "Casual task creation"),
    # ── Task query ──
    (
        r"\b(what.*(?:my|are) (?:tasks|todos|to.?do)|what do i have|my tasks|show.*tasks)\b",
        "task_query",
        "User wants to see tasks",
    ),
    # ── Habit tracking ──
    (
        r"\b(did my|finished my|completed my).*(?:workout|run|exercise|meditation|habit|routine|walk|stretching)\b",
        "habit_completion",
        "User completed a habit",
    ),
    (r"\b(what.*habits|habits.*due|my habits|streak)\b", "habit_query", "User wants to check habits"),
    # ── Goal management ──
    (r"\b(create.*goal|new goal|set.*goal|my goals|goal.*progress)\b", "goal_management", "Goal operations"),
    (r"\b(prioriti|what should i.*(?:do|focus)|eisenhower|what.*important)\b", "priority", "Priority/planning query"),
    # ── Scheduling & reminders ──
    (r"\b(remind me (?:in|at|to)|set.*(?:reminder|alarm|timer))\b", "reminder", "User wants a reminder"),
    (
        r"\b(schedule|cron|at \d{1,2}[:.]\d{2}|in \d+ (?:min|hour|h)|every (?:day|week|monday|morning))\b",
        "scheduling",
        "User wants to schedule something",
    ),
    (
        r"\b(calendar|my events|what.*(?:my|on) (?:schedule|calendar)|free (?:time|slot))\b",
        "calendar",
        "Calendar query",
    ),
    # ── Email ──
    (
        r"\b(send.*email|compose.*email|write.*email|email.*to|check.*(?:email|inbox|mail)|my (?:emails|inbox))\b",
        "email",
        "User wants to handle email",
    ),
    # ── Health & fitness ──
    (
        r"\b(body battery|stress level|heart rate|how.*(?:am i|my) (?:doing|health|energy)|garmin|health.*data)\b",
        "health_query",
        "User wants health data",
    ),
    (r"\b(sleep|how.*(?:did i|was my) sleep|sleep.*(?:score|quality))\b", "health_query", "Sleep data query"),
    (r"\b(steps|distance|calories|workout.*stats|fitness)\b", "health_query", "Fitness stats query"),
    (
        r"\b(took.*(?:pill|med|concerta|atenza|mph|medication)|logged.*(?:med|pill))\b",
        "life_event",
        "Medication logging",
    ),
    (r"\b(i (?:ate|had|eaten)|(?:had|ate) (?:lunch|dinner|breakfast|food|meal))\b", "life_event", "Food/meal logging"),
    (r"\b(went.*walk|did.*(?:exercise|workout|run|gym))\b", "life_event", "Exercise logging"),
    # ── Home automation ──
    (
        r"\b(lights?|bulb|lamp|bright|dim|turn.*(?:on|off)|switch.*(?:on|off))\b",
        "home_automation",
        "Smart light/device control",
    ),
    (r"\b(plug|socket|device|smart.*home|cozy|warm.*light)\b", "home_automation", "Smart device control"),
    # ── Web search / questions ──
    (
        r"\b(what is|how (?:do|does|can)|who is|where is|why (?:is|does)|can you (?:look|find|search))\b",
        "question",
        "User is asking a factual question",
    ),
    (r"\b(search.*(?:for|about|web)|look.*up|google)\b", "web_search", "User wants a web search"),
    # ── YouTube & content ──
    (
        r"\b(youtube|video|channel|subscribe|playlist|transcript)\b",
        "youtube",
        "YouTube/video content management",
    ),
    (r"\b(research.*(?:topic|about)|deep.*research|investigate)\b", "research", "Deep research request"),
    # ── Food & nutrition ──
    (
        r"\b(recipe|restaurant|cook|food.*(?:preference|tracking)|what.*(?:should i|to) eat|meal.*plan)\b",
        "food",
        "Food/recipe/restaurant management",
    ),
    # ── Shopping & products ──
    (
        r"\b(price|product|shopping|track.*price|how much.*cost|buy|purchase|deal|discount)\b",
        "shopping",
        "Product/price tracking",
    ),
    # ── Selfie/image generation ──
    # A GENERATION request = a generation verb NEAR an image noun
    # ("render(ing) an image", "make a picture", "send a pic", "draw me one"),
    # OR an image noun + of/for you|me ("a photo of you", "picture for me"),
    # OR "waiting for / where's the photo". Includes render/rendering/create so
    # "try rendering image for me" routes here, NOT to video. Guards against
    # false positives like "render the report to pdf" / "what is this image"
    # (the latter is image analysis, and attached-image requests skip media
    # routing via has_image). Order matters: this precedes video_gen so an
    # image request never falls through to the videographer.
    (
        r"\b(?:selfie"
        r"|(?:render|rendering|draw|sketch|paint|generate|create|make|take|send|get"
        r"|give|show|want|need|gimme|grab)\b[^.?!\n]{0,30}\b(?:photo|pic|pics|picture"
        r"|image|images|selfie|portrait|drawing|yourself)"
        r"|(?:photo|pic|picture|image|selfie|portrait)\b[^.?!\n]{0,18}\b(?:of|for)\b"
        r"[^.?!\n]{0,12}\b(?:you|yourself|me|us|twily|twilight)"
        r"|(?:waiting for|where['’]?s)\b[^.?!\n]{0,22}\b(?:photo|pic|picture"
        r"|image|selfie))\b",
        "selfie",
        "User wants a selfie/image",
    ),
    # ── Video generation (requires an explicit video noun so bare "render"/
    # "make" don't steal image requests above) ──
    (
        r"\b(?:make|create|generate|render|record|film)\b[^.?!\n]{0,25}"
        r"\b(?:video|animation|clip|gif)\b",
        "video_gen",
        "User wants generated video",
    ),
    # ── Image/video analysis (user sent media) ──
    (
        r"\b(what.*(?:this|that|the) (?:image|photo|picture)|describe.*(?:image|photo))\b",
        "image_analysis",
        "Analyze an image",
    ),
    # ── Memory & knowledge ──
    (
        r"\b(remember|save.*(?:this|that|memory)|don't forget|store.*(?:this|that))\b",
        "memory_save",
        "User wants to save a memory",
    ),
    (
        r"\b(what.*(?:you|do you) (?:remember|know)|recall|saved.*(?:memory|memories|notes))\b",
        "memory_search",
        "User wants to recall a memory",
    ),
    # ── Documents ──
    (r"\b(document|uploaded.*file|pdf|invoice)\b", "document", "Document management or invoice"),
    # ── Profile / personal analysis ──
    (r"\b(what.*(?:you know|do you know).*about me|my profile|my pattern|my behavio)\b", "profile", "Profile analysis"),
    # ── System monitoring ──
    (r"\b(gpu|server|disk.*(?:space|usage)|system.*status|nvidia|cpu|ram)\b", "system_monitoring", "System monitoring"),
    # ── Agent control ──
    (
        r"\b(what.*agents?.*running|tell.*(?:agent|investigator)|pass.*message|kill.*agent|stop.*agent)\b",
        "agent_control",
        "Agent management",
    ),
    # ── Lessons & rules ──
    (r"\b(lesson|what.*(?:you|have you) learned|past mistake)\b", "lessons", "View lessons learned"),
    (r"\b(add.*rule|new rule|always.*(?:do|remember)|never.*(?:do|say))\b", "user_rule", "Create user rule"),
    # ── Daily briefing ──
    (
        r"\b(brief(?:ing)?|morning.*(?:report|summary)|daily.*(?:report|summary)|what.*(?:today|plan))\b",
        "briefing",
        "Daily briefing",
    ),
    # ── Night analysis ──
    (
        r"\b(overnight|night.*analysis|what.*(?:discover|find).*(?:overnight|night|while i slept))\b",
        "night_analysis",
        "Night analysis results",
    ),
    # ── Life events (general) ──
    (r"\b(i (?:bought|traveled|visited|went to)|log.*(?:event|activity))\b", "life_event", "Log a life event"),
]


class Input(BaseModel):
    command: str = Field(default="infer", description="infer|sanity_check")
    message: str = Field(default="", description="User message to analyze")
    actions_taken: str = Field(
        default="",
        description="Comma-separated list of actions already taken (for sanity_check command)",
    )


class Output(BaseModel):
    success: bool = True
    intents: list[dict] = Field(default_factory=list)
    primary_intent: str = ""
    missed_intents: list[dict] = Field(default_factory=list)
    sanity_ok: bool = True
    recommendation: str = ""
    error: str = ""


# Map intent types to acceptable action labels for sanity checking
INTENT_TO_ACTIONS: dict[str, set[str]] = {
    "task_completion": {"todo_complete", "task_complete", "complete"},
    "task_creation": {"todo_create", "task_create", "create"},
    "task_query": {"todo_list", "task_list", "list"},
    "habit_completion": {"habit_complete", "complete"},
    "habit_query": {"habit_list", "list", "respond"},
    "goal_management": {"escalate", "goal_create", "goal_list", "respond"},
    "priority": {"escalate", "priority_check", "respond"},
    "reminder": {"todo_create", "reminder_set", "create"},
    "scheduling": {"escalate", "cron_create", "schedule"},
    "calendar": {"escalate", "calendar_check", "respond"},
    "email": {"email_sent", "email_composed", "escalate"},
    "health_query": {"health_fetch", "garmin_check", "respond"},
    "home_automation": {"lights_control", "device_control", "escalate"},
    "question": {"search", "web_search", "answered", "respond"},
    "web_search": {"search", "web_search", "respond"},
    "youtube": {"escalate", "youtube_search", "respond"},
    "research": {"escalate", "research", "respond"},
    "food": {"escalate", "food_search", "respond"},
    "shopping": {"escalate", "shopping_check", "respond"},
    "selfie": {"selfie", "image", "photo"},
    "video_gen": {"video", "render", "selfie"},
    "image_analysis": {"image_analyzed", "respond"},
    "memory_save": {"memory_create", "save"},
    "memory_search": {"memory_search", "search", "respond"},
    "document": {"escalate", "document_search", "respond"},
    "profile": {"escalate", "profile_search", "respond"},
    "system_monitoring": {"escalate", "system_check", "respond"},
    "agent_control": {"agent_control", "task_invoke", "respond"},
    "lessons": {"lesson_list", "respond"},
    "user_rule": {"rule_create", "respond"},
    "briefing": {"escalate", "briefing", "respond"},
    "night_analysis": {"respond", "search"},
    "life_event": {"event_log", "escalate", "respond"},
}

# Recommendations for missed intents
INTENT_RECOMMENDATIONS: dict[str, str] = {
    "task_completion": "User indicated they completed something — check todos for a match and mark complete.",
    "task_creation": "User mentioned something they need to do — create a todo.",
    "habit_completion": "User completed a habit — find and mark it done.",
    "habit_query": "User asked about habits — list due habits.",
    "goal_management": "User wants goal operations — escalate to orchestrator.",
    "priority": "User asked about priorities — run priority check or escalate.",
    "reminder": "User wants a reminder — create a todo with deadline.",
    "scheduling": "User wants to schedule something — escalate to cron_master.",
    "calendar": "User asked about calendar — escalate to calendar agent.",
    "email": "User wants email — escalate to email agent.",
    "health_query": "User asked about health — fetch fresh Garmin data.",
    "home_automation": "User wants to control a device — use home automation skill.",
    "question": "User asked a question — search for the answer.",
    "web_search": "User wants a web search — use search tools.",
    "youtube": "User mentioned YouTube — escalate for content management.",
    "research": "User wants deep research — escalate to web_searcher.",
    "food": "User mentioned food/recipes — escalate for food management.",
    "shopping": "User mentioned shopping/prices — escalate for product tracking.",
    "selfie": "User wants a selfie — invoke selfie subagent.",
    "video_gen": "User wants a video — invoke videographer subagent.",
    "memory_save": "User wants to save something — use memory manager.",
    "memory_search": "User wants to recall something — search memories.",
    "document": "User mentioned a document — handle or escalate.",
    "life_event": "User logged an activity — record as life event.",
}


class IntentInferenceTool(ScriptTool[Input, Output]):
    name = "intent_inference"
    description = "Detect user intents from message text and verify agent handled them"

    def execute(self, inp: Input) -> Output:
        if inp.command == "infer":
            return self._infer(inp.message)
        if inp.command == "sanity_check":
            return self._sanity_check(inp.message, inp.actions_taken)
        return Output(success=False, error=f"Unknown command: {inp.command}")

    def _infer(self, message: str) -> Output:
        if not message:
            return Output(success=False, error="No message provided")

        lower = _normalize(message)
        intents: list[dict] = []
        seen_types: set[str] = set()

        for pattern, intent_type, description in INTENT_PATTERNS:
            if intent_type not in seen_types and re.search(pattern, lower, re.IGNORECASE):
                intents.append({"type": intent_type, "description": description})
                seen_types.add(intent_type)

        primary = intents[0]["type"] if intents else "casual_chat"

        return Output(
            success=True,
            intents=intents,
            primary_intent=primary,
        )

    def _sanity_check(self, message: str, actions_taken: str) -> Output:
        """Check if the actions taken match the detected intents."""
        inferred = self._infer(message)
        if not inferred.success:
            return inferred

        actions = {a.strip().lower() for a in actions_taken.split(",") if a.strip()}

        missed: list[dict] = []
        for intent in inferred.intents:
            itype = intent["type"]
            expected = INTENT_TO_ACTIONS.get(itype, set())
            if expected and not actions.intersection(expected):
                missed.append(intent)

        sanity_ok = len(missed) == 0
        recommendation = ""
        if missed:
            # Use first missed intent's recommendation
            for m in missed:
                rec = INTENT_RECOMMENDATIONS.get(m["type"])
                if rec:
                    recommendation = rec
                    break
            if not recommendation:
                missed_types = [m["type"] for m in missed]
                recommendation = f"Unhandled intents: {', '.join(missed_types)}. Review and act."

        return Output(
            success=True,
            intents=inferred.intents,
            primary_intent=inferred.primary_intent,
            missed_intents=missed,
            sanity_ok=sanity_ok,
            recommendation=recommendation,
        )


if __name__ == "__main__":
    IntentInferenceTool.run()
