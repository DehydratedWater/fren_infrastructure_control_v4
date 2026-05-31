"""Send image to Telegram."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    image_path: str = Field(description="Path to image file")
    caption: str = Field(default="", description="Optional caption")


class Output(BaseModel):
    success: bool = True
    error: str = ""


class SendImageTool(ScriptTool[Input, Output]):
    name = "send_image"
    description = "Send an image to Telegram"

    def execute(self, inp: Input) -> Output:
        if not Path(inp.image_path).exists():
            return Output(success=False, error=f"Image not found: {inp.image_path}")
        return asyncio.run(self._send(inp.image_path, inp.caption))

    async def _send(self, image_path: str, caption: str) -> Output:
        from app.settings import get_settings

        settings = get_settings()
        if not settings.bot_token or not settings.chat_id:
            return Output(success=False, error="BOT_TOKEN or CHAT_ID not configured")

        from telegram import Bot
        from telegram.constants import ParseMode
        from telegram.error import TelegramError

        try:
            import telegramify_markdown

            has_tgmd = True
        except ImportError:
            has_tgmd = False

        bot = Bot(token=settings.bot_token)
        try:
            await bot.initialize()
            with open(image_path, "rb") as photo:
                cap = caption or None
                sent = False
                if cap and has_tgmd:
                    try:
                        converted = telegramify_markdown.markdownify(cap)
                        await bot.send_photo(
                            chat_id=settings.chat_id,
                            photo=photo,
                            caption=converted,
                            parse_mode=ParseMode.MARKDOWN_V2,
                        )
                        sent = True
                    except TelegramError:
                        photo.seek(0)
                if not sent:
                    await bot.send_photo(chat_id=settings.chat_id, photo=photo, caption=cap)

            # Save to chat history with @path so Twily can see what she sent
            try:
                from datetime import UTC, datetime

                from app.db.repos.chat import ChatMessagesRepo

                repo = ChatMessagesRepo()
                now = datetime.now(UTC)
                # Build relative @path for vision model visibility
                img_path = Path(image_path)
                try:
                    from app.settings import get_settings as _gs

                    rel = str(img_path.relative_to(_gs().project_root))
                except (ValueError, Exception):
                    rel = str(img_path)
                history_msg = f"Twily looks at the image she prepared: @{rel}"
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
    SendImageTool.run()
