"""Chat history tool — fetch and manage conversation history."""

from __future__ import annotations

import asyncio
from datetime import datetime

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="get-history|get-today|get-range|get-since|get-since-id|check-spam|save-user|save-twily"
    )
    days: int = Field(default=7, description="Number of days for history")
    hours: int = Field(default=0, description="Number of hours for history (overrides days when > 0)")
    message: str = Field(default="", description="Message text for save commands")
    timestamp: float = Field(default=0.0, description="Unix timestamp for get-since")
    message_id: int = Field(default=0, description="Database row ID for get-since-id")
    limit: int = Field(default=50, description="Max messages to return")
    offset: int = Field(default=0, description="Offset for pagination")
    max_chars: int = Field(
        default=0, description="Max total chars in formatted output (0=unlimited). Trims oldest messages first."
    )
    clearance: str = Field(default="", description="Clearance level: full (see all) or public (redact nsfw/secret)")
    from_date: str = Field(default="", description="Start date YYYY-MM-DD (get-range)")
    to_date: str = Field(default="", description="End date YYYY-MM-DD inclusive (get-range)")
    only_with_urls: bool = Field(default=False, description="Filter to messages containing http(s) URLs (get-range)")
    sender: str = Field(default="", description="Filter by sender: user|twily (get-range)")


class Output(BaseModel):
    success: bool = True
    messages: list[dict] = Field(default_factory=list)
    count: int = 0
    formatted: str = ""
    can_send: bool = True
    can_send_overdue_reminder: bool = True
    can_send_question: bool = True
    minutes_ago: float = -1
    error: str = ""


class ChatHistoryTool(ScriptTool[Input, Output]):
    name = "chat_history"
    description = "Access and manage conversation history"
    output_note = "If the user asked about this, share findings via send_message.py"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _try_embed(self, text: str) -> list[float] | None:
        """Best-effort embedding — never fail the save if embedding fails."""
        try:
            from app.settings import get_settings

            if not get_settings().openai_api_key:
                return None
            from app.services.embeddings import get_embedding

            emb = await asyncio.to_thread(get_embedding, text)
            if all(v == 0.0 for v in emb[:10]):
                return None
            return emb
        except Exception:
            return None

    def _get_clearance(self, inp: Input) -> str:
        """Resolve clearance from input or FREN_CLEARANCE env var."""
        import os

        if inp.clearance:
            return inp.clearance
        return os.environ.get("FREN_CLEARANCE", "full")

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.chat import ChatMessagesRepo

        repo = ChatMessagesRepo()
        clearance = self._get_clearance(inp)

        if inp.command == "get-history":
            if inp.hours > 0:
                import math

                since_ts = datetime.now().timestamp() - (inp.hours * 3600)
                days_ceil = math.ceil(inp.hours / 24) + 1
                msgs = await repo.get_history(days=days_ceil, limit=inp.limit, offset=inp.offset, clearance=clearance)
                msgs = [m for m in msgs if m.get("timestamp_unix", 0) > since_ts]
            else:
                msgs = await repo.get_history(days=inp.days, limit=inp.limit, offset=inp.offset, clearance=clearance)
            formatted = self._format(msgs, max_chars=inp.max_chars)
            return Output(success=True, messages=msgs, count=len(msgs), formatted=formatted)

        if inp.command == "get-today":
            today = datetime.now().date()
            msgs = await repo.get_by_date(today, limit=inp.limit, offset=inp.offset, clearance=clearance)
            formatted = self._format(msgs, max_chars=inp.max_chars)
            return Output(success=True, messages=msgs, count=len(msgs), formatted=formatted)

        if inp.command == "get-range":
            from datetime import date as date_cls

            def _parse(s: str) -> date_cls | None:
                if not s:
                    return None
                try:
                    return datetime.strptime(s, "%Y-%m-%d").date()
                except ValueError:
                    return None

            msgs = await repo.get_range(
                from_date=_parse(inp.from_date),
                to_date=_parse(inp.to_date),
                only_with_urls=inp.only_with_urls,
                sender=inp.sender or None,
                limit=inp.limit,
                offset=inp.offset,
                clearance=clearance,
            )
            formatted = self._format(msgs, max_chars=inp.max_chars)
            return Output(success=True, messages=msgs, count=len(msgs), formatted=formatted)

        if inp.command == "get-since":
            msgs = await repo.get_recent(limit=inp.limit, clearance=clearance)
            filtered = [m for m in msgs if m.get("timestamp_unix", 0) > inp.timestamp]
            return Output(success=True, messages=filtered, count=len(filtered))

        if inp.command == "get-since-id":
            msgs = await repo.get_since_id(inp.message_id, limit=inp.limit, offset=inp.offset, clearance=clearance)
            return Output(success=True, messages=msgs, count=len(msgs))

        if inp.command == "check-spam":
            msgs = await repo.get_recent(limit=20)
            twily_msgs = [m for m in msgs if m.get("sender") == "twily"]
            if not twily_msgs:
                return Output(
                    success=True, minutes_ago=-1, can_send=True, can_send_overdue_reminder=True, can_send_question=True
                )
            last_ts = twily_msgs[0].get("timestamp")
            if last_ts:
                now = datetime.now()
                if hasattr(last_ts, "timestamp"):
                    diff = (now - last_ts).total_seconds() / 60
                else:
                    diff = -1
                return Output(
                    success=True,
                    minutes_ago=diff,
                    can_send=diff >= 30 or diff < 0,
                    can_send_overdue_reminder=diff >= 120 or diff < 0,
                    can_send_question=diff >= 240 or diff < 0,
                )
            return Output(success=True, minutes_ago=-1, can_send=True)

        if inp.command == "save-user":
            now = datetime.now()
            embedding = await self._try_embed(inp.message)
            await repo.save(
                sender="user",
                message=inp.message,
                date=now.date(),
                timestamp=now,
                timestamp_unix=now.timestamp(),
                embedding=embedding,
            )
            return Output(success=True)

        if inp.command == "save-twily":
            now = datetime.now()
            embedding = await self._try_embed(inp.message)
            await repo.save(
                sender="twily",
                message=inp.message,
                date=now.date(),
                timestamp=now,
                timestamp_unix=now.timestamp(),
                embedding=embedding,
            )
            return Output(success=True)

        return Output(success=False, error=f"Unknown command: {inp.command}")

    def _format(self, messages: list[dict], *, max_chars: int = 0) -> str:
        now = datetime.now()
        header = f"=== Chat History (current time: {now.strftime('%Y-%m-%d %H:%M')}) ==="
        footer = "=== End History ==="

        # Format all messages newest-first (messages come sorted DESC from DB)
        formatted_lines: list[str] = []
        for m in messages:
            ts = str(m.get("timestamp", ""))[:16]
            sender = m.get("sender", "?")
            text = str(m.get("message", ""))[:300]
            formatted_lines.append(f"[{ts}] {sender}: {text}")

        if max_chars > 0:
            # Keep newest messages, trim oldest until within budget
            overhead = len(header) + len(footer) + 2  # newlines
            budget = max_chars - overhead
            kept: list[str] = []
            total = 0
            for line in formatted_lines:
                if total + len(line) + 1 > budget:
                    break
                kept.append(line)
                total += len(line) + 1
            formatted_lines = kept

        return "\n".join([header, *formatted_lines, footer])


if __name__ == "__main__":
    ChatHistoryTool.run()
