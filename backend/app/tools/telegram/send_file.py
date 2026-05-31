"""Send file to Telegram."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    file_path: str = Field(description="Path to file")
    caption: str = Field(default="", description="Optional caption")


class Output(BaseModel):
    success: bool = True
    error: str = ""


class SendFileTool(ScriptTool[Input, Output]):
    name = "send_file"
    description = "Send a file to Telegram"

    def execute(self, inp: Input) -> Output:
        if not Path(inp.file_path).exists():
            return Output(success=False, error=f"File not found: {inp.file_path}")
        return asyncio.run(self._send(inp.file_path, inp.caption))

    async def _send(self, file_path: str, caption: str) -> Output:
        from app.settings import get_settings

        settings = get_settings()
        if not settings.bot_token or not settings.chat_id:
            return Output(success=False, error="BOT_TOKEN or CHAT_ID not configured")

        from telegram import Bot
        from telegram.error import TelegramError

        bot = Bot(token=settings.bot_token)
        try:
            await bot.initialize()
            with open(file_path, "rb") as doc:
                await bot.send_document(
                    chat_id=settings.chat_id,
                    document=doc,
                    caption=caption or None,
                )
            return Output(success=True)
        except TelegramError as e:
            return Output(success=False, error=str(e))
        finally:
            await bot.shutdown()


if __name__ == "__main__":
    SendFileTool.run()
