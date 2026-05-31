"""Invoices repository."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class InvoicesRepo:
    async def create(self, invoice_id: str, raw_data: dict, **kw: Any) -> dict[str, Any]:
        async with get_async_session() as s:
            result = await s.execute(
                text(
                    "INSERT INTO invoices (invoice_id, raw_data, ai_summary)"
                    " VALUES (:invoice_id, CAST(:raw_data AS jsonb), :ai_summary)"
                    " RETURNING *"
                ),
                {
                    "invoice_id": invoice_id,
                    "raw_data": json.dumps(raw_data),
                    "ai_summary": kw.get("ai_summary"),
                },
            )
            return dict(result.fetchone()._mapping)

    async def get(self, invoice_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM invoices WHERE invoice_id = :iid", {"iid": invoice_id})

    async def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(s, "SELECT * FROM invoices ORDER BY created_at DESC LIMIT :limit", {"limit": limit})

    async def search_by_seller(self, seller_name: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM invoices WHERE seller_name ILIKE :sn
                ORDER BY issue_date DESC
            """,
                {"sn": f"%{seller_name}%"},
            )

    async def delete(self, invoice_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(s, "DELETE FROM invoices WHERE invoice_id = :iid RETURNING id", {"iid": invoice_id})
            return r.fetchone() is not None
