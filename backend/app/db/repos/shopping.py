"""Shopping tracker repositories — products and price snapshots."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class TrackedProductRepo:
    async def create(
        self,
        product_id: str,
        name: str,
        *,
        search_query: str = "",
        filters: Any = None,
        alert_threshold_percent: float = 5.0,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO tracked_products (product_id, name, search_query, filters, alert_threshold_percent)
                VALUES (:pid, :name, :sq, CAST(:filters AS jsonb), :threshold)
                RETURNING *
                """,
                {
                    "pid": product_id,
                    "name": name,
                    "sq": search_query,
                    "filters": json.dumps(filters or {}),
                    "threshold": alert_threshold_percent,
                },
            )  # type: ignore[return-value]

    async def get(self, product_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM tracked_products WHERE product_id = :pid", {"pid": product_id})

    async def list_active(self, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM tracked_products WHERE status = 'active' ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def list_all(self, *, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM tracked_products ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def update(self, product_id: str, **fields: Any) -> dict[str, Any] | None:
        jsonb_keys = ("filters",)
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"pid": product_id}
        idx = 1
        for k, v in fields.items():
            if v is not None:
                pk = f"p{idx}"
                if k in jsonb_keys:
                    params[pk] = json.dumps(v)
                    sets.append(f"{k} = CAST(:{pk} AS jsonb)")
                else:
                    params[pk] = v
                    sets.append(f"{k} = :{pk}")
                idx += 1
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE tracked_products SET {', '.join(sets)} WHERE product_id = :pid RETURNING *",
                params,
            )

    async def delete(self, product_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s, "DELETE FROM tracked_products WHERE product_id = :pid RETURNING id", {"pid": product_id}
            )
            return r.fetchone() is not None


class PriceSnapshotRepo:
    async def create(
        self,
        snapshot_id: str,
        product_id: str,
        price: float,
        *,
        currency: str = "USD",
        source_title: str = "",
        source_url: str = "",
        raw_api_response: Any = None,
        price_change_percent: float = 0.0,
        alert_triggered: bool = False,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO price_snapshots
                    (snapshot_id, product_id, price, currency, source_title, source_url,
                     raw_api_response, price_change_percent, alert_triggered)
                VALUES (:sid, :pid, :price, :currency, :stitle, :surl,
                        CAST(:raw AS jsonb), :change, :alert)
                RETURNING *
                """,
                {
                    "sid": snapshot_id,
                    "pid": product_id,
                    "price": price,
                    "currency": currency,
                    "stitle": source_title,
                    "surl": source_url,
                    "raw": json.dumps(raw_api_response or {}),
                    "change": price_change_percent,
                    "alert": alert_triggered,
                },
            )  # type: ignore[return-value]

    async def get_latest(self, product_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM price_snapshots WHERE product_id = :pid ORDER BY created_at DESC LIMIT 1",
                {"pid": product_id},
            )

    async def get_history(self, product_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM price_snapshots WHERE product_id = :pid ORDER BY created_at DESC LIMIT :limit",
                {"pid": product_id, "limit": limit},
            )

    async def get_alerts(self, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT ps.*, tp.name AS product_name FROM price_snapshots ps
                JOIN tracked_products tp ON tp.product_id = ps.product_id
                WHERE ps.alert_triggered = TRUE
                ORDER BY ps.created_at DESC LIMIT :limit
                """,
                {"limit": limit},
            )

    async def get_price_series(self, product_id: str, *, limit: int = 90) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT date, price, currency FROM price_snapshots
                WHERE product_id = :pid
                ORDER BY date ASC LIMIT :limit
                """,
                {"pid": product_id, "limit": limit},
            )
