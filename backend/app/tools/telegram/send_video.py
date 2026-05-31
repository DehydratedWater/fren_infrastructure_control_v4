"""Send video file to Telegram."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50 MB Telegram bot API limit


class Input(BaseModel):
    file_path: str = Field(description="Path to video file")
    caption: str = Field(default="", description="Optional caption for the video")


class Output(BaseModel):
    success: bool = True
    error: str = ""


class SendVideoTool(ScriptTool[Input, Output]):
    name = "send_video"
    description = "Send a video file to Telegram"

    def execute(self, inp: Input) -> Output:
        p = Path(inp.file_path)
        if not p.is_file():
            return Output(success=False, error=f"File not found: {inp.file_path}")
        if p.stat().st_size > MAX_VIDEO_SIZE:
            return Output(success=False, error=f"File exceeds 50 MB Telegram limit: {p.stat().st_size} bytes")
        return asyncio.run(self._send(inp.file_path, inp.caption))

    async def _send(self, file_path: str, caption: str) -> Output:
        from app.settings import get_settings

        settings = get_settings()
        if not settings.bot_token or not settings.chat_id:
            return Output(success=False, error="BOT_TOKEN or CHAT_ID not configured")

        from telegram import Bot
        from telegram.error import TelegramError
        from telegram.request import HTTPXRequest

        request = HTTPXRequest(read_timeout=120, write_timeout=120, connect_timeout=30)
        bot = Bot(token=settings.bot_token, request=request)
        try:
            await bot.initialize()
            with open(file_path, "rb") as vid:
                await bot.send_video(
                    chat_id=settings.chat_id,
                    video=vid,
                    caption=caption or None,
                )

            # Save to chat history with @path so Twily can see what she sent
            try:
                from datetime import UTC, datetime

                from app.db.repos.chat import ChatMessagesRepo

                repo = ChatMessagesRepo()
                now = datetime.now(UTC)
                # Build relative @path for vision model visibility
                vid_path = Path(file_path)
                try:
                    from app.settings import get_settings as _gs

                    rel = str(vid_path.relative_to(_gs().project_root))
                except (ValueError, Exception):
                    rel = str(vid_path)
                history_msg = f"Twily looks at the video she prepared: @{rel}"
                if caption:
                    history_msg += f"\nCaption: {caption}"
                await repo.save(
                    sender="twily",
                    message=history_msg,
                    date=now.date(),
                    timestamp=now,
                    timestamp_unix=now.timestamp(),
                )
            except Exception:
                pass

            return Output(success=True)
        except TelegramError as e:
            return Output(success=False, error=str(e))
        finally:
            await bot.shutdown()


if __name__ == "__main__":
    SendVideoTool.run()
