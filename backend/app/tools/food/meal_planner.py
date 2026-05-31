"""Meal Planner — daily meal check-ins with escalating suggestions."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="get-today|get-meal|ask-meal|record-response|suggest-meal|get-history|set-location"
    )
    meal_type: str = Field(default="", description="Meal type: breakfast|lunch|dinner")
    date: str = Field(default="", description="Date (YYYY-MM-DD), defaults to today")
    user_response: str = Field(default="", description="What the user ate")
    meal_source: str = Field(default="", description="homemade|ordered|restaurant|skipped")
    location: str = Field(default="", description="Current location (warsaw|wroclaw|other)")
    escalation_level: int = Field(default=0, description="Escalation level 0-4 for suggestions")
    days: int = Field(default=7, description="Number of days for history")


class Output(BaseModel):
    success: bool = True
    item: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class MealPlannerTool(ScriptTool[Input, Output]):
    name = "meal_planner"
    description = "Daily meal check-ins with escalating suggestions"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    def _parse_date(self, date_str: str) -> date:
        if date_str:
            return date.fromisoformat(date_str)
        return date.today()

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.meal_checkins import MealCheckinsRepo

        cmd = inp.command
        repo = MealCheckinsRepo()

        if cmd == "get-today":
            await repo.ensure_today_entries()
            meals = await repo.get_today()
            return Output(success=True, items=meals, count=len(meals))

        if cmd == "get-meal":
            d = self._parse_date(inp.date)
            meal = await repo.get_by_meal(d, inp.meal_type)
            return Output(success=bool(meal), item=meal or {}, error="" if meal else "Not found")

        if cmd == "ask-meal":
            d = self._parse_date(inp.date)
            await repo.ensure_today_entries()
            meal = await repo.update_status(
                d, inp.meal_type, "asked", asked_at=datetime.now(), escalation_level=inp.escalation_level
            )
            return Output(success=bool(meal), item=meal or {})

        if cmd == "record-response":
            d = self._parse_date(inp.date)
            meal = await repo.record_response(
                d,
                inp.meal_type,
                user_response=inp.user_response,
                meal_source=inp.meal_source,
                location=inp.location,
            )
            return Output(success=bool(meal), item=meal or {})

        if cmd == "suggest-meal":
            return await self._suggest(inp)

        if cmd == "get-history":
            meals = await repo.get_history(days=inp.days)
            return Output(success=True, items=meals, count=len(meals))

        if cmd == "set-location":
            from app.db.repos.food import PreferencesRepo

            prefs = await PreferencesRepo().update(current_location=inp.location)
            return Output(success=bool(prefs), item=prefs or {})

        return Output(success=False, error=f"Unknown command: {cmd}")

    async def _suggest(self, inp: Input) -> Output:
        from app.db.repos.food import PreferencesRepo, RecipesRepo, RestaurantsRepo

        level = inp.escalation_level
        prefs = await PreferencesRepo().get()
        location = inp.location or prefs.get("current_location", "unknown")

        suggestions: dict[str, Any] = {
            "escalation_level": level,
            "meal_type": inp.meal_type,
            "location": location,
        }

        if level <= 1:
            # L1: Just ask — no suggestions needed
            suggestions["message"] = "What are you thinking for " + (inp.meal_type or "your meal") + "?"
            suggestions["recipes"] = []

        elif level == 2:
            # L2: Healthy recipes from DB
            recipes = await RecipesRepo().list(meal_type=inp.meal_type or None, limit=5)
            suggestions["message"] = "Here are some recipe ideas:"
            suggestions["recipes"] = [
                {"title": r["title"], "recipe_id": r["recipe_id"], "prep_time": r.get("prep_time_minutes")}
                for r in recipes
            ]

        elif level == 3:
            # L3: Quick/easy options (prep ≤ 15 min)
            recipes = await RecipesRepo().list(meal_type=inp.meal_type or None, limit=20)
            quick = [r for r in recipes if (r.get("prep_time_minutes") or 999) <= 15]
            if not quick:
                quick = recipes[:3]
            suggestions["message"] = "Quick options (15 min or less):"
            suggestions["recipes"] = [
                {"title": r["title"], "recipe_id": r["recipe_id"], "prep_time": r.get("prep_time_minutes")}
                for r in quick[:5]
            ]

        else:
            # L4: Ordering suggestions
            restaurants = await RestaurantsRepo().list(limit=50)
            local_restaurants = [r for r in restaurants if (r.get("location_area") or "").lower() == location.lower()]
            if not local_restaurants:
                local_restaurants = restaurants[:5]

            delivery = prefs.get("delivery_services", ["lisek"])
            suggestions["message"] = "Time to order! Here are your options:"
            suggestions["restaurants"] = [
                {"name": r["name"], "restaurant_id": r["restaurant_id"], "cuisine": r.get("cuisine_types")}
                for r in local_restaurants[:5]
            ]
            suggestions["delivery_services"] = delivery

        return Output(success=True, item=suggestions)


if __name__ == "__main__":
    MealPlannerTool.run()
