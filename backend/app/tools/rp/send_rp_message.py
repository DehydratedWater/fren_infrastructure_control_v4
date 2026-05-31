"""Send a message via the RP bot (uses BOT_RP_TOKEN, not BOT_TOKEN)."""

from __future__ import annotations

import asyncio

from src import ScriptTool, StreamFormat

from pydantic import BaseModel, Field


class Input(BaseModel):
    message: str = Field(description="Message text to send")


class Output(BaseModel):
    success: bool = True
    parts_sent: int = 0
    error: str = ""


class SendRPMessageTool(ScriptTool[Input, Output]):
    name = "send_rp_message"
    description = "Send a message to the user via the RP Telegram bot"
    stream_format = StreamFormat.TEXT
    stream_field = "message"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._send(inp.message))

    async def _send(self, message: str) -> Output:
        from app.settings import get_settings

        settings = get_settings()
        if not settings.bot_rp_token or not settings.chat_id:
            return Output(success=False, error="BOT_RP_TOKEN or CHAT_ID not configured")

        from telegram import Bot
        from telegram.constants import ParseMode
        from telegram.error import TelegramError

        try:
            import telegramify_markdown

            has_tgmd = True
        except ImportError:
            has_tgmd = False

        message = message.replace("\\n", "\n")

        # Decode unicode escapes (e.g., — → —)
        try:
            message = message.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            # Fallback: just replace common unicode escapes
            import re

            def _replace_unicode(m: re.Match) -> str:
                return chr(int(m.group(1), 16))

            message = re.sub(r"\\u([0-9a-fA-F]{4})", _replace_unicode, message)

        bot = Bot(token=settings.bot_rp_token)
        chat_id = int(settings.chat_id)

        # Split long messages at paragraph boundaries (Telegram 4096 char limit)
        parts = _split_message(message)
        sent = 0

        for part in parts:
            formatted = telegramify_markdown.markdownify(part) if has_tgmd else part
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=formatted,
                    parse_mode=ParseMode.MARKDOWN_V2 if has_tgmd else None,
                )
                sent += 1
            except TelegramError as e:
                if "can't parse entities" in str(e).lower():
                    await bot.send_message(chat_id=chat_id, text=part)
                    sent += 1
                else:
                    return Output(success=False, parts_sent=sent, error=str(e))

        # Log to story — agents should use rp_story_manager for proper logging,
        # but we also save to chat_messages for cross-bot awareness.
        try:
            from datetime import UTC, date, datetime

            from app.db.repos.chat import ChatMessagesRepo

            now = datetime.now(UTC)
            await ChatMessagesRepo().save(
                sender="rp_bot",
                message=message,
                date=date.today(),
                timestamp=now,
                timestamp_unix=now.timestamp(),
                chat_id=str(chat_id),
                content_class="rp",
            )
        except Exception:
            pass

        return Output(success=True, parts_sent=sent)


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split text at paragraph boundaries to stay under Telegram limit."""
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) + 2 > max_len:
            if current:
                parts.append(current.strip())
            current = paragraph
        else:
            current = current + "\n\n" + paragraph if current else paragraph
    if current:
        parts.append(current.strip())
    return parts or [text[:max_len]]


if __name__ == "__main__":
    SendRPMessageTool.run()
