"""Telegram activity log tool -- read-only access to the user's personal Telegram logging bot.

The user dumps tags (#omega3, #mph, #masturbacja), links, screenshots, and short notes
into a separate Telegram bot. This tool queries that data so agents can see what the user
has been doing outside of the Twily conversation.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
from src import ScriptTool
from pydantic import BaseModel, Field

API_BASE = "http://192.168.0.80:5050"


class Input(BaseModel):
    command: str = Field(description="recent|day|blocks|topics")
    date: str = Field(default="", description="Date YYYY-MM-DD (for day command)")
    block_ids: str = Field(default="", description="Comma-separated block IDs (for blocks command)")
    topic: str = Field(default="", description="Topic name (for topics command)")
    limit: int = Field(default=10, description="Max items to return")


class Output(BaseModel):
    success: bool = True
    data: dict = Field(default_factory=dict)
    error: str = ""


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _yesterday() -> str:
    return (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")


def _simplify_message(msg: dict) -> dict:
    """Extract the useful fields from a raw message."""
    author = msg.get("message_author", {})
    entities = msg.get("entities") or []
    hashtags = [e["extracted_text"] for e in entities if e.get("entity_type") == "hashtag"]
    urls = [e["extracted_text"] for e in entities if e.get("entity_type") == "url"]
    text_urls = [e["extracted_text"] for e in entities if e.get("entity_type") == "text_link"]

    result: dict = {
        "time": msg.get("message_time", ""),
        "author": author.get("nickname") or author.get("full_name", ""),
        "text": msg.get("message_text", ""),
    }
    if hashtags:
        result["hashtags"] = hashtags
    if urls or text_urls:
        result["urls"] = urls + text_urls
    if msg.get("has_photo"):
        result["has_photo"] = True
    if msg.get("is_document"):
        result["has_document"] = True
    if msg.get("photo_text"):
        result["photo_text"] = msg["photo_text"]
    if msg.get("document_text"):
        result["document_text"] = msg["document_text"]
    return result


def _simplify_block_summary(block: dict) -> dict:
    return {
        "block_id": block["message_block_id"],
        "date": block["day_date"],
        "start": block.get("start_time", ""),
        "end": block.get("end_time", ""),
        "message_count": block.get("message_count", 0),
        "authors": block.get("authors_involved", []),
    }


async def _fetch_recent(limit: int) -> dict:
    """Get the newest block and its messages."""
    async with httpx.AsyncClient(base_url=API_BASE, timeout=15) as client:
        # Get newest block summary
        resp = await client.get("/stored_data/fetch_newest_block")
        resp.raise_for_status()
        newest = resp.json()

        block_id = newest["message_block_id"]

        # Fetch its messages
        resp = await client.post(
            "/stored_data/fetch_messages_by_blocks",
            json={"message_block_ids": [block_id]},
        )
        resp.raise_for_status()
        blocks_data = resp.json().get("blocks", [])

        messages = []
        if blocks_data:
            for msg in blocks_data[0].get("messages", [])[-limit:]:
                messages.append(_simplify_message(msg))

        # Also get a few recent blocks for overview
        today = _today()
        resp = await client.post(
            "/stored_data/fetch_blocks_by_date",
            json={"start_date": today, "end_date": today},
        )
        resp.raise_for_status()
        today_blocks = [_simplify_block_summary(b) for b in resp.json().get("blocks", [])]

        return {
            "newest_block": _simplify_block_summary(newest),
            "messages": messages,
            "today_blocks": today_blocks[:limit],
        }


async def _fetch_day(date_str: str, limit: int) -> dict:
    """Get all blocks for a date with formatted conversation."""
    async with httpx.AsyncClient(base_url=API_BASE, timeout=15) as client:
        # Get block summaries
        resp = await client.post(
            "/stored_data/fetch_blocks_by_date",
            json={"start_date": date_str, "end_date": date_str},
        )
        resp.raise_for_status()
        blocks = resp.json().get("blocks", [])

        if not blocks:
            return {"date": date_str, "blocks": [], "formatted": []}

        block_ids = [b["message_block_id"] for b in blocks[:limit]]

        # Get formatted conversation
        resp = await client.post(
            "/stored_data/format_conversation",
            json={"message_block_ids": block_ids},
        )
        resp.raise_for_status()
        formatted = resp.json()

        return {
            "date": date_str,
            "block_count": len(blocks),
            "blocks": [_simplify_block_summary(b) for b in blocks[:limit]],
            "formatted": formatted.get("formatted_blocks", []),
        }


async def _fetch_blocks(block_ids: list[int]) -> dict:
    """Fetch full messages for specific block IDs."""
    async with httpx.AsyncClient(base_url=API_BASE, timeout=15) as client:
        resp = await client.post(
            "/stored_data/fetch_messages_by_blocks",
            json={"message_block_ids": block_ids},
        )
        resp.raise_for_status()
        blocks_data = resp.json().get("blocks", [])

        result = []
        for block in blocks_data:
            result.append(
                {
                    "block_id": block["message_block_id"],
                    "date": block.get("day_date", ""),
                    "start": block.get("start_time", ""),
                    "end": block.get("end_time", ""),
                    "messages": [_simplify_message(m) for m in block.get("messages", [])],
                }
            )
        return {"blocks": result}


async def _fetch_topics(topic: str, limit: int) -> dict:
    """Get topic memory entries."""
    async with httpx.AsyncClient(base_url=API_BASE, timeout=15) as client:
        params: dict = {}
        if limit:
            params["limit"] = limit
        resp = await client.get(
            f"/simple_topic/simple_topic_memory/by_topic/{topic}",
            params=params,
        )
        resp.raise_for_status()
        entries = resp.json()
        return {
            "topic": topic,
            "entries": [
                {
                    "date": e.get("update_date", ""),
                    "text": e.get("text", ""),
                    "author": e.get("author"),
                }
                for e in entries[:limit]
            ],
        }


class TelegramLogTool(ScriptTool[Input, Output]):
    name = "telegram_log"
    description = (
        "Read the user's personal Telegram activity log — hashtags, links, notes, "
        "and screenshots dumped into a separate logging bot"
    )

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        try:
            if inp.command == "recent":
                data = await _fetch_recent(inp.limit)
            elif inp.command == "day":
                date_str = inp.date or _today()
                data = await _fetch_day(date_str, inp.limit)
            elif inp.command == "blocks":
                if not inp.block_ids:
                    return Output(success=False, error="--block_ids required (comma-separated)")
                ids = [int(x.strip()) for x in inp.block_ids.split(",")]
                data = await _fetch_blocks(ids)
            elif inp.command == "topics":
                if not inp.topic:
                    return Output(success=False, error="--topic required")
                data = await _fetch_topics(inp.topic, inp.limit)
            else:
                return Output(success=False, error=f"Unknown command: {inp.command}")
        except httpx.HTTPError as e:
            return Output(success=False, error=f"API request failed: {e}")

        return Output(success=True, data=data)


if __name__ == "__main__":
    TelegramLogTool.run()
