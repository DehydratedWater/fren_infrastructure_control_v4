"""Food Manager — recipes, restaurants, dishes, preferences."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

from src import ScriptTool
from pydantic import BaseModel, Field


def _parse_tag_filter(tags_csv: str, dietary_tags_json: str) -> set[str]:
    """Accept either CSV (--tags 'vegan,quick') or JSON (--dietary_tags '[...]')."""
    out: set[str] = set()
    if tags_csv.strip():
        out.update(t.strip().lower() for t in tags_csv.split(",") if t.strip())
    if dietary_tags_json.strip():
        try:
            parsed = json.loads(dietary_tags_json)
            if isinstance(parsed, list):
                out.update(str(t).lower() for t in parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    return out


class Input(BaseModel):
    command: str = Field(
        description="Recipe: add-recipe|get-recipe|list-recipes|update-recipe|delete-recipe|mark-made|search-recipes "
        "(supports --limit, --tags CSV, --difficulty, --cuisine_type, --meal_type, --max_prep_time); "
        "Restaurant: add-restaurant|get-restaurant|list-restaurants|update-restaurant|delete-restaurant|mark-visited|"
        "list-restaurants-by-location (supports --limit, --location_area); "
        "Dish: add-dish|get-dish|list-dishes|delete-dish (list-dishes supports --limit); "
        "Prefs: get-preferences|set-preference; "
        "Meal: log-meal"
    )
    # Meal log fields
    meal_type: str = Field(default="", description="Meal type (breakfast/lunch/dinner)")
    meal_source: str = Field(default="", description="homemade|ordered|restaurant|skipped")
    max_prep_time: int = Field(default=0, description="Max prep time in minutes for recipe search")
    # IDs
    recipe_id: str = Field(default="", description="Recipe ID")
    restaurant_id: str = Field(default="", description="Restaurant ID")
    dish_id: str = Field(default="", description="Dish ID")
    # Common fields
    title: str = Field(default="", description="Title/name")
    name: str = Field(default="", description="Name (restaurants/dishes)")
    description: str = Field(default="", description="Description")
    notes: str = Field(default="", description="Notes")
    # Recipe fields
    cuisine_type: str = Field(default="", description="Cuisine type")
    meal_type: str = Field(default="", description="Meal type")
    difficulty: str = Field(default="", description="Difficulty level")
    ingredients: str = Field(default="", description="JSON ingredients array")
    instructions: str = Field(default="", description="JSON instructions array")
    dietary_tags: str = Field(default="", description="JSON dietary tags array")
    prep_time: int = Field(default=0, description="Prep time minutes")
    cook_time: int = Field(default=0, description="Cook time minutes")
    servings: int = Field(default=0, description="Number of servings")
    rating: float = Field(default=-1.0, description="Rating")
    source_url: str = Field(default="", description="Source URL")
    # Restaurant fields
    cuisine_types: str = Field(default="", description="JSON cuisine types array")
    location_area: str = Field(default="", description="Location area")
    address: str = Field(default="", description="Address")
    price_range: str = Field(default="", description="Price range")
    # Dish fields
    price: float = Field(default=-1.0, description="Price")
    category: str = Field(default="", description="Category")
    # Preference fields
    key: str = Field(default="", description="Preference key")
    value: str = Field(default="", description="Preference value")
    # List pagination / filters
    limit: int = Field(default=50, description="Max rows to return for list/search commands")
    tags: str = Field(
        default="",
        description="CSV of dietary tags to filter by (e.g. 'vegan,quick') — simpler alias for --dietary_tags JSON",
    )


class Output(BaseModel):
    success: bool = True
    item: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class FoodManagerTool(ScriptTool[Input, Output]):
    name = "food_manager"
    description = "Manage recipes, restaurants, dishes, and food preferences"
    output_note = "If the user requested this, share the result via send_message.py"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.food import DishesRepo, PreferencesRepo, RecipesRepo, RestaurantsRepo

        cmd = inp.command

        # ── Recipes ──
        if cmd == "add-recipe":
            repo = RecipesRepo()
            rid = f"recipe_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            kw = {}
            if inp.description:
                kw["description"] = inp.description
            if inp.cuisine_type:
                kw["cuisine_type"] = inp.cuisine_type
            if inp.meal_type:
                kw["meal_type"] = inp.meal_type
            if inp.difficulty:
                kw["difficulty"] = inp.difficulty
            if inp.ingredients:
                kw["ingredients"] = json.loads(inp.ingredients)
            if inp.instructions:
                kw["instructions"] = json.loads(inp.instructions)
            if inp.dietary_tags:
                kw["dietary_tags"] = json.loads(inp.dietary_tags)
            if inp.prep_time:
                kw["prep_time_minutes"] = inp.prep_time
            if inp.cook_time:
                kw["cook_time_minutes"] = inp.cook_time
            if inp.servings:
                kw["servings"] = inp.servings
            if inp.source_url:
                kw["source_url"] = inp.source_url
            if inp.notes:
                kw["notes"] = inp.notes
            r = await repo.create(rid, inp.title, **kw)
            return Output(success=True, item=r)

        if cmd == "get-recipe":
            r = await RecipesRepo().get(inp.recipe_id)
            return Output(success=bool(r), item=r or {}, error="" if r else "Not found")

        if cmd == "list-recipes":
            rs = await RecipesRepo().list(
                cuisine_type=inp.cuisine_type or None,
                meal_type=inp.meal_type or None,
                limit=inp.limit,
            )
            if inp.difficulty:
                rs = [r for r in rs if (r.get("difficulty") or "").lower() == inp.difficulty.lower()]
            tag_filter = _parse_tag_filter(inp.tags, inp.dietary_tags)
            if tag_filter:
                rs = [r for r in rs if tag_filter.issubset({str(t).lower() for t in (r.get("dietary_tags") or [])})]
            return Output(success=True, items=rs, count=len(rs))

        if cmd == "update-recipe":
            fields = {}
            for k in ("title", "description", "cuisine_type", "meal_type", "difficulty", "notes", "source_url"):
                v = getattr(inp, k)
                if v:
                    fields[k] = v
            if inp.rating >= 0:
                fields["rating"] = inp.rating
            if inp.prep_time:
                fields["prep_time_minutes"] = inp.prep_time
            if inp.cook_time:
                fields["cook_time_minutes"] = inp.cook_time
            if inp.servings:
                fields["servings"] = inp.servings
            if inp.ingredients:
                fields["ingredients"] = json.loads(inp.ingredients)
            if inp.instructions:
                fields["instructions"] = json.loads(inp.instructions)
            if inp.dietary_tags:
                fields["dietary_tags"] = json.loads(inp.dietary_tags)
            r = await RecipesRepo().update(inp.recipe_id, **fields)
            return Output(success=bool(r), item=r or {}, error="" if r else "Not found")

        if cmd == "delete-recipe":
            ok = await RecipesRepo().delete(inp.recipe_id)
            return Output(success=ok)

        if cmd == "mark-made":
            r = await RecipesRepo().update(inp.recipe_id, times_made="times_made + 1")
            return Output(success=bool(r), item=r or {})

        # ── Restaurants ──
        if cmd == "add-restaurant":
            repo = RestaurantsRepo()
            rid = f"rest_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            kw = {}
            if inp.description:
                kw["description"] = inp.description
            if inp.cuisine_types:
                kw["cuisine_types"] = json.loads(inp.cuisine_types)
            if inp.location_area:
                kw["location_area"] = inp.location_area
            if inp.address:
                kw["address"] = inp.address
            if inp.price_range:
                kw["price_range"] = inp.price_range
            if inp.notes:
                kw["notes"] = inp.notes
            if inp.rating >= 0:
                kw["rating"] = inp.rating
            r = await repo.create(rid, inp.name, **kw)
            return Output(success=True, item=r)

        if cmd == "get-restaurant":
            r = await RestaurantsRepo().get(inp.restaurant_id)
            return Output(success=bool(r), item=r or {}, error="" if r else "Not found")

        if cmd == "list-restaurants":
            rs = await RestaurantsRepo().list(limit=inp.limit)
            return Output(success=True, items=rs, count=len(rs))

        if cmd == "update-restaurant":
            fields = {}
            for k in ("name", "description", "location_area", "address", "price_range", "notes"):
                v = getattr(inp, k)
                if v:
                    fields[k] = v
            if inp.cuisine_types:
                fields["cuisine_types"] = json.loads(inp.cuisine_types)
            if inp.rating >= 0:
                fields["rating"] = inp.rating
            r = await RestaurantsRepo().update(inp.restaurant_id, **fields)
            return Output(success=bool(r), item=r or {}, error="" if r else "Not found")

        if cmd == "delete-restaurant":
            ok = await RestaurantsRepo().delete(inp.restaurant_id)
            return Output(success=ok)

        if cmd == "mark-visited":
            r = await RestaurantsRepo().log_visit(inp.restaurant_id)
            return Output(success=bool(r), item=r or {})

        # ── Dishes ──
        if cmd == "add-dish":
            repo = DishesRepo()
            did = f"dish_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            kw = {}
            if inp.description:
                kw["description"] = inp.description
            if inp.price >= 0:
                kw["price"] = inp.price
            if inp.category:
                kw["category"] = inp.category
            if inp.dietary_tags:
                kw["dietary_tags"] = json.loads(inp.dietary_tags)
            if inp.notes:
                kw["notes"] = inp.notes
            if inp.rating >= 0:
                kw["rating"] = inp.rating
            d = await repo.create(did, inp.restaurant_id, inp.name, **kw)
            return Output(success=True, item=d)

        if cmd == "list-dishes":
            ds = await DishesRepo().list_for_restaurant(inp.restaurant_id)
            return Output(success=True, items=ds[: inp.limit], count=min(len(ds), inp.limit))

        if cmd == "get-dish":
            return Output(success=False, error="Use list-dishes with restaurant_id")

        if cmd == "delete-dish":
            ok = await DishesRepo().delete(inp.dish_id)
            return Output(success=ok)

        # ── Preferences ──
        if cmd == "get-preferences":
            prefs = await PreferencesRepo().get()
            return Output(success=True, item=prefs)

        if cmd == "set-preference":
            try:
                val = json.loads(inp.value)
            except (json.JSONDecodeError, TypeError):
                val = inp.value
            prefs = await PreferencesRepo().update(**{inp.key: val})
            return Output(success=bool(prefs), item=prefs or {})

        # ── Extended commands ──
        if cmd == "log-meal":
            from app.db.repos.meal_checkins import MealCheckinsRepo

            repo = MealCheckinsRepo()
            await repo.ensure_today_entries()
            meal = await repo.record_response(
                datetime.now().date(),
                inp.meal_type or "lunch",
                user_response=inp.description or inp.title,
                meal_source=inp.meal_source,
                location=inp.location_area,
            )
            return Output(success=bool(meal), item=meal or {})

        if cmd == "search-recipes":
            # Over-fetch so post-filters still yield up to inp.limit rows
            recipes = await RecipesRepo().list(
                cuisine_type=inp.cuisine_type or None,
                meal_type=inp.meal_type or None,
                limit=max(inp.limit * 4, 50),
            )
            if inp.max_prep_time:
                recipes = [r for r in recipes if (r.get("prep_time_minutes") or 999) <= inp.max_prep_time]
            if inp.difficulty:
                recipes = [r for r in recipes if (r.get("difficulty") or "").lower() == inp.difficulty.lower()]
            tag_filter = _parse_tag_filter(inp.tags, inp.dietary_tags)
            if tag_filter:
                recipes = [
                    r for r in recipes if tag_filter.issubset({str(t).lower() for t in (r.get("dietary_tags") or [])})
                ]
            recipes = recipes[: inp.limit]
            return Output(success=True, items=recipes, count=len(recipes))

        if cmd == "list-restaurants-by-location":
            rs = await RestaurantsRepo().list(limit=max(inp.limit * 4, 100))
            if inp.location_area:
                rs = [r for r in rs if (r.get("location_area") or "").lower() == inp.location_area.lower()]
            rs = rs[: inp.limit]
            return Output(success=True, items=rs, count=len(rs))

        return Output(success=False, error=f"Unknown command: {cmd}")


if __name__ == "__main__":
    FoodManagerTool.run()
