"""Food repositories — recipes, restaurants, dishes, preferences."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class RecipesRepo:
    async def create(self, recipe_id: str, title: str, **kw: Any) -> dict[str, Any]:
        jsonb_keys = ("ingredients", "instructions", "dietary_tags", "nutrition")
        cols = ["recipe_id", "title"]
        vals = [":recipe_id", ":title"]
        params: dict[str, Any] = {"recipe_id": recipe_id, "title": title}
        param_idx = 3
        for k, v in kw.items():
            if v is not None:
                cols.append(k)
                param_key = f"p{param_idx}"
                if k in jsonb_keys:
                    params[param_key] = json.dumps(v)
                    vals.append(f"CAST(:{param_key} AS jsonb)")
                else:
                    params[param_key] = v
                    vals.append(f":{param_key}")
                param_idx += 1
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                INSERT INTO recipes ({", ".join(cols)})
                VALUES ({", ".join(vals)}) RETURNING *
            """,
                params,
            )  # type: ignore[return-value]

    async def get(self, recipe_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM recipes WHERE recipe_id = :rid", {"rid": recipe_id})

    async def list(
        self, *, cuisine_type: str | None = None, meal_type: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        conds: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if cuisine_type:
            conds.append("cuisine_type = :ct")
            params["ct"] = cuisine_type
        if meal_type:
            conds.append("meal_type = :mt")
            params["mt"] = meal_type
        where = " AND ".join(conds) if conds else "1=1"
        async with get_async_session() as s:
            return await fetch_all(
                s, f"SELECT * FROM recipes WHERE {where} ORDER BY created_at DESC LIMIT :limit", params
            )

    async def update(self, recipe_id: str, **fields: Any) -> dict[str, Any] | None:
        jsonb_keys = ("ingredients", "instructions", "dietary_tags", "nutrition")
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"rid": recipe_id}
        param_idx = 1
        for k, v in fields.items():
            if v is not None:
                param_key = f"p{param_idx}"
                if k in jsonb_keys:
                    params[param_key] = json.dumps(v)
                    sets.append(f"{k} = CAST(:{param_key} AS jsonb)")
                else:
                    params[param_key] = v
                    sets.append(f"{k} = :{param_key}")
                param_idx += 1
        async with get_async_session() as s:
            return await fetch_one(
                s, f"UPDATE recipes SET {', '.join(sets)} WHERE recipe_id = :rid RETURNING *", params
            )

    async def delete(self, recipe_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM recipes WHERE recipe_id = :rid RETURNING id", {"rid": recipe_id})
            return r.fetchone() is not None

    async def search_by_dietary(self, tag: str, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM recipes WHERE dietary_tags @> CAST(:tag AS jsonb)
                ORDER BY rating DESC NULLS LAST LIMIT :limit
            """,
                {"tag": json.dumps([tag]), "limit": limit},
            )


class RestaurantsRepo:
    async def create(self, restaurant_id: str, name: str, **kw: Any) -> dict[str, Any]:
        jsonb_keys = ("cuisine_types",)
        params: dict[str, Any] = {"restaurant_id": restaurant_id, "name": name}
        cols = ["restaurant_id", "name"]
        vals = [":restaurant_id", ":name"]
        for k, v in kw.items():
            if v is not None:
                cols.append(k)
                if k in jsonb_keys:
                    v = json.dumps(v)
                    vals.append(f"CAST(:{k} AS jsonb)")
                else:
                    vals.append(f":{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                INSERT INTO restaurants ({", ".join(cols)})
                VALUES ({", ".join(vals)}) RETURNING *
            """,
                params,
            )  # type: ignore[return-value]

    async def get(self, restaurant_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM restaurants WHERE restaurant_id = :rid", {"rid": restaurant_id})

    async def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s, "SELECT * FROM restaurants ORDER BY created_at DESC LIMIT :limit", {"limit": limit}
            )

    async def update(self, restaurant_id: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"rid": restaurant_id}
        for k, v in fields.items():
            if v is not None:
                if k == "cuisine_types":
                    v = json.dumps(v)
                    sets.append(f"{k} = CAST(:{k} AS jsonb)")
                else:
                    sets.append(f"{k} = :{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s, f"UPDATE restaurants SET {', '.join(sets)} WHERE restaurant_id = :rid RETURNING *", params
            )

    async def delete(self, restaurant_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s, "DELETE FROM restaurants WHERE restaurant_id = :rid RETURNING id", {"rid": restaurant_id}
            )
            return r.fetchone() is not None

    async def log_visit(self, restaurant_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE restaurants SET visit_count = visit_count + 1,
                    last_visited_at = NOW(), updated_at = NOW()
                WHERE restaurant_id = :rid RETURNING *
            """,
                {"rid": restaurant_id},
            )


class DishesRepo:
    async def create(self, dish_id: str, restaurant_id: str, name: str, **kw: Any) -> dict[str, Any]:
        params: dict[str, Any] = {"dish_id": dish_id, "restaurant_id": restaurant_id, "name": name}
        cols = ["dish_id", "restaurant_id", "name"]
        vals = [":dish_id", ":restaurant_id", ":name"]
        for k, v in kw.items():
            if v is not None:
                cols.append(k)
                if k == "dietary_tags":
                    v = json.dumps(v)
                    vals.append(f"CAST(:{k} AS jsonb)")
                else:
                    vals.append(f":{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                INSERT INTO dishes ({", ".join(cols)}) VALUES ({", ".join(vals)}) RETURNING *
            """,
                params,
            )  # type: ignore[return-value]

    async def list_for_restaurant(self, restaurant_id: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s, "SELECT * FROM dishes WHERE restaurant_id = :rid ORDER BY name", {"rid": restaurant_id}
            )

    async def delete(self, dish_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM dishes WHERE dish_id = :did RETURNING id", {"did": dish_id})
            return r.fetchone() is not None


class PreferencesRepo:
    async def get(self) -> dict[str, Any]:
        async with get_async_session() as s:
            row = await fetch_one(s, "SELECT * FROM user_food_preferences WHERE id = 1")
            return row or {}

    async def update(self, **fields: Any) -> dict[str, Any] | None:
        jsonb_keys = (
            "dietary_restrictions",
            "favorite_cuisines",
            "disliked_ingredients",
            "allergies",
            "kitchen_equipment",
        )
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {}
        for k, v in fields.items():
            if v is not None:
                if k in jsonb_keys:
                    v = json.dumps(v)
                    sets.append(f"{k} = CAST(:{k} AS jsonb)")
                else:
                    sets.append(f"{k} = :{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s, f"UPDATE user_food_preferences SET {', '.join(sets)} WHERE id = 1 RETURNING *", params
            )
