"""Shopping Tracker — Google Shopping price tracking via SearchAPI.io."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import httpx
from src import ScriptTool
from pydantic import BaseModel, Field

_SEARCHAPI_BASE = "https://www.searchapi.io/api/v1/search"


class Input(BaseModel):
    command: str = Field(
        description="add-product|get-product|list-products|update-product|delete-product|"
        "fetch-prices|get-price-history|get-alerts|get-price-series"
    )
    product_id: str = Field(default="", description="Product ID")
    name: str = Field(default="", description="Product name")
    search_query: str = Field(default="", description="Google Shopping search query")
    filters: str = Field(default="", description="JSON filters object")
    alert_threshold_percent: float = Field(default=-1.0, description="Alert threshold %")
    status: str = Field(default="", description="Status (active/paused)")


class Output(BaseModel):
    success: bool = True
    item: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    alerts_count: int = 0
    error: str = ""


class ShoppingTrackerTool(ScriptTool[Input, Output]):
    name = "shopping_tracker"
    description = "Track product prices via Google Shopping with alerts"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.shopping import PriceSnapshotRepo, TrackedProductRepo

        cmd = inp.command

        # ── Product CRUD ──
        if cmd == "add-product":
            repo = TrackedProductRepo()
            pid = f"prod_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            kw = {}
            if inp.search_query:
                kw["search_query"] = inp.search_query
            if inp.filters:
                kw["filters"] = json.loads(inp.filters)
            if inp.alert_threshold_percent >= 0:
                kw["alert_threshold_percent"] = inp.alert_threshold_percent
            p = await repo.create(pid, inp.name, **kw)
            return Output(success=True, item=p)

        if cmd == "get-product":
            p = await TrackedProductRepo().get(inp.product_id)
            return Output(success=bool(p), item=p or {}, error="" if p else "Product not found")

        if cmd == "list-products":
            ps = await TrackedProductRepo().list_all()
            return Output(success=True, items=ps, count=len(ps))

        if cmd == "update-product":
            fields = {}
            if inp.name:
                fields["name"] = inp.name
            if inp.search_query:
                fields["search_query"] = inp.search_query
            if inp.filters:
                fields["filters"] = json.loads(inp.filters)
            if inp.alert_threshold_percent >= 0:
                fields["alert_threshold_percent"] = inp.alert_threshold_percent
            if inp.status:
                fields["status"] = inp.status
            p = await TrackedProductRepo().update(inp.product_id, **fields)
            return Output(success=bool(p), item=p or {}, error="" if p else "Product not found")

        if cmd == "delete-product":
            ok = await TrackedProductRepo().delete(inp.product_id)
            return Output(success=ok)

        # ── Price fetching ──
        if cmd == "fetch-prices":
            return await self._fetch_all_prices()

        # ── Price queries ──
        if cmd == "get-price-history":
            history = await PriceSnapshotRepo().get_history(inp.product_id)
            return Output(success=True, items=history, count=len(history))

        if cmd == "get-alerts":
            alerts = await PriceSnapshotRepo().get_alerts()
            return Output(success=True, items=alerts, count=len(alerts), alerts_count=len(alerts))

        if cmd == "get-price-series":
            series = await PriceSnapshotRepo().get_price_series(inp.product_id)
            return Output(success=True, items=series, count=len(series))

        return Output(success=False, error=f"Unknown command: {cmd}")

    async def _fetch_all_prices(self) -> Output:
        from app.settings import get_settings
        from app.db.repos.shopping import PriceSnapshotRepo, TrackedProductRepo

        api_key = get_settings().searchapi_key
        if not api_key:
            return Output(success=False, error="SEARCHAPI_KEY not configured")

        prod_repo = TrackedProductRepo()
        snap_repo = PriceSnapshotRepo()
        products = await prod_repo.list_active()

        results = []
        alerts_count = 0

        for product in products:
            pid = product["product_id"]
            query = product["search_query"] or product["name"]

            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(
                        _SEARCHAPI_BASE,
                        params={
                            "engine": "google_shopping",
                            "q": query,
                            "api_key": api_key,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except httpx.HTTPError as e:
                results.append({"product_id": pid, "error": str(e)})
                continue

            # Pick best (lowest) price from shopping results
            shopping_results = data.get("shopping_results", [])
            if not shopping_results:
                results.append({"product_id": pid, "error": "No results"})
                continue

            best = min(shopping_results, key=lambda r: r.get("price", float("inf")))
            price = best.get("price", 0.0)
            if not price:
                # Try extracting from extracted_price
                price = best.get("extracted_price", 0.0)
            if not price:
                results.append({"product_id": pid, "error": "No price found"})
                continue

            # Compute change vs latest
            latest = await snap_repo.get_latest(pid)
            change_pct = 0.0
            if latest and latest["price"]:
                old_price = float(latest["price"])
                if old_price > 0:
                    change_pct = round(((price - old_price) / old_price) * 100, 2)

            threshold = float(product.get("alert_threshold_percent", 5.0))
            alert = abs(change_pct) >= threshold

            sid = f"snap_{datetime.now().strftime('%Y%m%d')}_{id(best) % 0xFFFFFFFF:08x}"
            await snap_repo.create(
                sid,
                pid,
                price,
                source_title=best.get("title", ""),
                source_url=best.get("link", ""),
                raw_api_response=best,
                price_change_percent=change_pct,
                alert_triggered=alert,
            )
            if alert:
                alerts_count += 1
            results.append({"product_id": pid, "price": price, "change_pct": change_pct, "alert": alert})

        return Output(success=True, items=results, count=len(results), alerts_count=alerts_count)


if __name__ == "__main__":
    ShoppingTrackerTool.run()
