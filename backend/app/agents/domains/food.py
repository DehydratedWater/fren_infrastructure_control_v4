"""Food domain — fridge/recipe/restaurant/meal-planning fleet (v3 `food/*`).

Ported faithfully from v3's open-agent-compiler builders:
- `food/orchestrator` (v3 fridge_inventory_orchestrator) routes a food request to
  the right specialist subagent (suggest → food_suggester, add_recipe →
  recipe_parser, add_restaurant → restaurant_intake) or does a direct DB list.
- `food/meal_planner` — ADHD-friendly escalating meal check-ins (hidden primary).
- subagents: food_suggester, recipe_parser, restaurant_intake, and the
  vision-class product_image_indexer.

The orchestrator's suggest path is a multi-step BRANCH (orchestrator →
food_suggester), so it gets a path-test in branches().
"""

from __future__ import annotations

from app.agents._authoring import define_agent
from app.agents._tools import (
    agent_notes_tool,
    chat_history_tool,
    context_resolver_tool,
    embedding_search_tool,
    emit_guidance_tool,
    execution_ledger_tool,
    fetch_context_tool,
    food_manager_tool,
    meal_planner_tool,
    periodic_checker_tool,
    personality_core_tool,
    proactive_send_tool,
    response_processor_tool,
    thought_transfer_tool,
    user_config_tool,
    web_search_tool,
)
from src import (
    AgentDefinition,
    AgentTest,
    BranchTest,
    CapabilityTest,
    StepContract,
    SubstringEvaluator,
)

ORCHESTRATOR = "food/orchestrator"
FOOD_SUGGESTER = "food/food_suggester"
RECIPE_PARSER = "food/recipe_parser"
RESTAURANT_INTAKE = "food/restaurant_intake"
PRODUCT_IMAGE_INDEXER = "food/product_image_indexer"
MEAL_PLANNER = "food/meal_planner"

_ORCH_PROMPT = """\
# Fridge Inventory Orchestrator

Manage the food system — recipes, restaurants, dishes, and meal suggestions.
Route each request to the right specialist based on user intent.

## Routing
1. Identify intent: suggest / add_recipe / add_restaurant / list / process_image.
2. Dispatch:
   - suggest → food/food_suggester
   - add_recipe → food/recipe_parser
   - add_restaurant → food/restaurant_intake
   - process_image → food/product_image_indexer
   - list → query the food database directly (food-manager) for recipes,
     restaurants, or dishes.
3. Send the result back to the user via emit-guidance.

You are a router: classify intent, hand off to the subagent, then confirm.
"""

_MEAL_PLANNER_PROMPT = """\
# Meal Planner — ADHD-Friendly Meal Check-ins

Help the user remember to eat through gentle, escalating check-ins.

## Escalation Levels
- L0-1: Just ask — "Hey, what are you thinking for lunch?"
- L2: Suggest healthy recipes from the DB + web search for ideas.
- L3: Quick/easy options only (prep ≤ 15 min) — minimize executive-function load.
- L4: Ordering mode — suggest delivery (Lisek if Warsaw) and nearby restaurants.

## ADHD Principles
- Keep messages SHORT and actionable; max 3 options per suggestion.
- Make the easiest option the default. Never guilt-trip about skipped meals.
- If the user hasn't responded by L3, suggest ordering (lowest friction).

## Location Awareness
- Read current_location from food preferences. Warsaw → Lisek + local spots;
  Wroclaw → different set / Polish comfort food + delivery apps. If unknown,
  ask once then remember.

## Flow (each tick)
0. Check the shared proactive-send budget (tier=meal). If can_send=false, exit
   silently — a higher-priority tier just fired and a meal ping would be noise.
1. Get today's check-in status (meal-planner get-today); pick the current slot.
2. Check recent chat (last ~4h) for eating/cooking/ordering mentions. If the
   user already ate this meal, record it and set should_message=false. If a
   twily message in the last 30 min already asked and got no reply,
   should_message=false (let the next slot handle it).
2.5 Escalation-only gate: only when FREN_JOB_ID == meal_escalation. Run the
   periodic checker; if there's an active overdue_todos trigger, set
   should_message=false (reason task_priority_overrides_meal_escalation) and
   record it to thought_transfer. Base check-ins (meal_breakfast/lunch/dinner)
   skip this gate.
3. Decide: pending → ask at L1; asked + no reply at an appropriate slot → bump
   escalation; already responded / duplicate / cooldown active → should_message
   =false. Never override a step-2.5 false. Proceed only if should_message=true.
4. If messaging: try meal-planner suggest-meal ONCE at the current level. If
   recipes/restaurants are empty (normal for an empty DB), do NOT retry — suggest
   2-3 quick meals from your own knowledge per preferences/location. L3+: zero-
   effort options (toast, banana, yogurt, ordering). L4: just suggest delivery.
5. Emit the check-in via emit-guidance (≤3 options). If should_message is false,
   do NOT emit — exit silently. After a successful send, stamp the shared send
   budget (tier=meal) so other agents defer.
"""

_FOOD_SUGGESTER_PROMPT = """\
# Food Suggester

Suggest meals by:
1. Check user food preferences and dietary restrictions.
2. Review recent meals to avoid repetition.
3. Consider available recipes and restaurants.
4. Suggest with clear reasoning, then send the suggestions to the user.
"""

_RECIPE_PARSER_PROMPT = """\
# Recipe Parser

Parse recipe information from user input (structured or freeform text/URLs):
- Extract title, ingredients, instructions, prep/cook times, and servings.
- Detect cuisine type and dietary tags.
- Save the parsed recipe via the food management system.
"""

_RESTAURANT_INTAKE_PROMPT = """\
# Restaurant Intake

Process restaurant information from detailed or brief descriptions:
- Extract name, cuisine types, location, and price range.
- Add associated dishes if mentioned.
- Save the restaurant via the food management system.
"""

_PRODUCT_IMAGE_INDEXER_PROMPT = """\
# Product Image Indexer

Analyze food product images using the vision model to identify:
- Product name and type
- Brand if visible
- Nutritional information
- Suggested categories and tags

Output structured product info: name, type, brand, nutrition, categories, tags.
"""


def agents() -> list[AgentDefinition]:
    return [
        define_agent(
            ORCHESTRATOR,
            model_class="default",
            short="route a food request to the right food specialist",
            long=(
                "Fridge inventory orchestrator. Classifies food intent"
                " (suggest / add_recipe / add_restaurant / list / process_image)"
                " and dispatches to the matching subagent, or lists the food DB"
                " directly, then confirms to the user."
            ),
            prompt=_ORCH_PROMPT,
            tools=[
                food_manager_tool(),
                meal_planner_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="orchestrator-carries-food-manager",
                    description="Lists the food DB directly via food-manager and confirms via emit-guidance.",
                    must_have_tools=("food-manager",),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="suggest-intent-dispatches-suggester",
                    prompt="Suggest something for dinner tonight.",
                    evaluators=(
                        SubstringEvaluator(needle="suggest", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            FOOD_SUGGESTER,
            model_class="default",
            short="suggest meals from preferences, history, and available options",
            long=(
                "Checks food preferences, reviews recent meals to avoid"
                " repetition, considers available recipes/restaurants, and"
                " suggests meals with reasoning."
            ),
            prompt=_FOOD_SUGGESTER_PROMPT,
            tools=[
                food_manager_tool(),
                meal_planner_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="suggester-avoids-repetition-language",
                    description="Prompt must steer away from repeating recent meals.",
                    evaluators=(
                        SubstringEvaluator(needle="repetition", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="suggests-with-reasoning",
                    prompt="What should I eat for lunch? I had pasta yesterday.",
                    evaluators=(
                        SubstringEvaluator(needle="suggest", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            RECIPE_PARSER,
            model_class="default",
            short="parse recipes from text or URLs and save them",
            long=(
                "Extracts title, ingredients, instructions, times, servings,"
                " cuisine type, and dietary tags from structured or freeform"
                " input, then saves via the food manager."
            ),
            prompt=_RECIPE_PARSER_PROMPT,
            tools=[
                food_manager_tool(),
                meal_planner_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="recipe-parser-extracts-ingredients",
                    description="Prompt must require extracting ingredients.",
                    evaluators=(
                        SubstringEvaluator(needle="ingredients", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="extracts-recipe-fields",
                    prompt=(
                        "Save this: Garlic Pasta — boil pasta, fry garlic in oil,"
                        " combine. Serves 2, 15 min."
                    ),
                    evaluators=(
                        SubstringEvaluator(needle="ingredients", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            RESTAURANT_INTAKE,
            model_class="default",
            short="process and store restaurant information",
            long=(
                "Extracts restaurant name, cuisine types, location, price range,"
                " and any mentioned dishes from detailed or brief descriptions,"
                " then saves via the food manager."
            ),
            prompt=_RESTAURANT_INTAKE_PROMPT,
            tools=[
                food_manager_tool(),
                meal_planner_tool(),
                emit_guidance_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="restaurant-intake-extracts-cuisine",
                    description="Prompt must require extracting cuisine types.",
                    evaluators=(
                        SubstringEvaluator(needle="cuisine", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="extracts-restaurant-fields",
                    prompt="Add Sushi Ko — Japanese, Warsaw, mid-range.",
                    evaluators=(
                        SubstringEvaluator(needle="location", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            PRODUCT_IMAGE_INDEXER,
            model_class="vision",
            short="identify food products from images",
            long=(
                "Vision-class subagent. Analyzes a food product image to identify"
                " product name/type, brand, nutrition, and suggested"
                " categories/tags, then outputs structured product info."
            ),
            prompt=_PRODUCT_IMAGE_INDEXER_PROMPT,
            capability_tests=[
                CapabilityTest(
                    name="indexer-is-vision-class",
                    description="Image indexer prompt must reference vision/image analysis.",
                    evaluators=(
                        SubstringEvaluator(needle="image", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="outputs-structured-product-info",
                    prompt="Here is a photo of a cereal box — index it.",
                    evaluators=(
                        SubstringEvaluator(needle="brand", case_sensitive=False),
                    ),
                ),
            ],
        ),
        define_agent(
            MEAL_PLANNER,
            model_class="default",
            short="ADHD-friendly escalating meal check-ins",
            long=(
                "Hidden primary. Runs per-tick: checks the shared send budget,"
                " today's meal status, and chat history, then sends short,"
                " escalating (L0-L4) meal check-ins or exits silently."
            ),
            prompt=_MEAL_PLANNER_PROMPT,
            tools=[
                fetch_context_tool(),
                embedding_search_tool(),
                meal_planner_tool(),
                food_manager_tool(),
                emit_guidance_tool(),
                chat_history_tool(),
                user_config_tool(),
                web_search_tool(),
                personality_core_tool(),
                periodic_checker_tool(),
                thought_transfer_tool(),
                execution_ledger_tool(),
                context_resolver_tool(),
                response_processor_tool(),
                agent_notes_tool(),
                proactive_send_tool(),
            ],
            capability_tests=[
                CapabilityTest(
                    name="meal-planner-keeps-it-short",
                    description="Prompt must encode the ADHD short/escalation principle.",
                    evaluators=(
                        SubstringEvaluator(needle="escalat", case_sensitive=False),
                    ),
                ),
            ],
            agent_tests=[
                AgentTest(
                    name="respects-send-budget",
                    prompt="It's lunchtime — run a meal check-in tick.",
                    evaluators=(
                        SubstringEvaluator(needle="should_message", case_sensitive=False),
                    ),
                ),
            ],
        ),
    ]


def branches() -> list[BranchTest]:
    """The orchestrator's distinguished suggest path (tested + optimised as a unit)."""
    return [
        # suggest intent → food_suggester subagent
        BranchTest(
            name="food/orchestrator::suggest-meal",
            entry_agent=ORCHESTRATOR,
            prompt="Suggest something quick for dinner tonight.",
            path=(FOOD_SUGGESTER,),
            subagent_mocks={
                FOOD_SUGGESTER: (
                    "Suggestion: a quick 15-minute garlic butter shrimp"
                    " stir-fry — perfect for dinner tonight. I can suggest two"
                    " backup options if the fridge is missing shrimp."
                ),
            },
            evaluators=(
                SubstringEvaluator(needle="suggest", case_sensitive=False),
            ),
            step_contracts=(
                # Context forwarding: the meal slot from the user's request must
                # reach the suggester; its reply must stay on that meal.
                StepContract(
                    step=FOOD_SUGGESTER,
                    input_evaluators=(
                        SubstringEvaluator(needle="dinner", case_sensitive=False),
                    ),
                    output_evaluators=(
                        SubstringEvaluator(needle="dinner", case_sensitive=False),
                    ),
                ),
            ),
        ),
        # add_recipe intent → recipe_parser subagent
        BranchTest(
            name="food/orchestrator::add-recipe",
            entry_agent=ORCHESTRATOR,
            prompt="Save this recipe: garlic pasta, serves 2.",
            path=(RECIPE_PARSER,),
            subagent_mocks={
                RECIPE_PARSER: (
                    "Parsed recipe saved: 'Garlic pasta' (serves 2) —"
                    " ingredients: garlic, pasta, olive oil, parmesan; added to"
                    " the recipe book."
                ),
            },
            step_contracts=(
                # The dish named by the user must reach the parser verbatim;
                # the parser must confirm a recipe was handled.
                StepContract(
                    step=RECIPE_PARSER,
                    input_evaluators=(
                        SubstringEvaluator(
                            needle="garlic pasta", case_sensitive=False,
                        ),
                    ),
                    output_evaluators=(
                        SubstringEvaluator(needle="recipe", case_sensitive=False),
                    ),
                ),
            ),
        ),
    ]
