"""Invoice manager tool — manage parsed invoices."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="import|get|get-by-number|list|delete")
    invoice_id: str = Field(default="", description="Invoice ID")
    invoice_number: str = Field(default="", description="Invoice number")
    invoice_json: str = Field(default="", description="JSON invoice data")
    summary: str = Field(default="", description="AI summary")
    source_image: str = Field(default="", description="Source image path")
    seller: str = Field(default="", description="Filter by seller")
    date_from: str = Field(default="", description="Filter from date YYYY-MM-DD")
    date_to: str = Field(default="", description="Filter to date YYYY-MM-DD")
    limit: int = Field(default=100, description="Result limit")


class Output(BaseModel):
    success: bool = True
    invoice: dict | None = None
    invoices: list[dict] = Field(default_factory=list)
    count: int = 0
    deleted: bool = False
    error: str = ""


class InvoiceManagerTool(ScriptTool[Input, Output]):
    name = "invoice_manager"
    description = "Import, query, and manage parsed invoices"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.invoices import InvoicesRepo

        repo = InvoicesRepo()

        if inp.command == "import":
            if not inp.invoice_json:
                return Output(success=False, error="--invoice_json required")
            data = json.loads(inp.invoice_json)
            iid = f"inv_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
            inv = await repo.create(
                iid,
                data,
                ai_summary=inp.summary or None,
                source_image_path=inp.source_image or None,
            )

            # Cache the invoice artifact
            try:
                from app.db.repos.context_cache import add_to_cache

                summary_text = inp.summary or f"Invoice imported: {iid}"
                await add_to_cache(
                    "invoice",
                    summary_text,
                    entity_type="invoices",
                    entity_id=iid,
                    file_path=inp.source_image or "",
                    tags=["invoice", "financial"],
                    source_agent="invoice_manager",
                )
            except Exception:
                pass

            return Output(success=True, invoice=inv)

        if inp.command == "get":
            inv = await repo.get(inp.invoice_id)
            return Output(success=True, invoice=inv) if inv else Output(success=False, error="Not found")

        if inp.command == "list":
            if inp.seller:
                invs = await repo.search_by_seller(inp.seller)
            else:
                invs = await repo.list(limit=inp.limit)
            return Output(success=True, invoices=invs, count=len(invs))

        if inp.command == "get-by-number":
            from app.db.session import fetch_one, get_async_session

            async with get_async_session() as s:
                inv = await fetch_one(
                    s,
                    "SELECT * FROM invoices WHERE invoice_data->>'invoice_number' = :num",
                    {"num": inp.invoice_number},
                )
            return Output(success=True, invoice=inv) if inv else Output(success=False, error="Not found")

        if inp.command == "delete":
            deleted = await repo.delete(inp.invoice_id)
            return Output(success=True, deleted=deleted)

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    InvoiceManagerTool.run()
