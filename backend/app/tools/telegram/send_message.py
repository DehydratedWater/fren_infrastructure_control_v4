"""Send text message to Telegram."""

from __future__ import annotations

import asyncio
import re

from src import ScriptTool, StreamFormat
from pydantic import BaseModel, Field


class Input(BaseModel):
    message: str = Field(description="Message text to send")


class Output(BaseModel):
    success: bool = True
    parts_sent: int = 0
    error: str = ""


def _strip_formatting(text: str) -> str:
    """Fast regex-based markdown/formatting removal for TTS."""
    # Code blocks → remove entirely
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Inline code → keep content
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Images → alt text
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Links → keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # Headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Strikethrough
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    # Blockquotes
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Bullet points
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Numbered lists
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


class SendMessageTool(ScriptTool[Input, Output]):
    name = "send_message"
    description = "Send a text message to Telegram"
    stream_format = StreamFormat.TEXT
    stream_field = "message"
    output_note = "If DUPLICATE_DETECTED: message was already sent. STOP. Do NOT retry. Your task is complete."

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._send(inp.message))

    async def _send(self, message: str) -> Output:
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

        # Check for duplicates + capture recent Twily msgs for the rule scorer.
        recent_twily_texts: list[str] = []
        try:
            from difflib import SequenceMatcher

            from app.db.repos.chat import ChatMessagesRepo

            repo = ChatMessagesRepo()
            recent = await repo.get_recent(limit=5)
            twily_msgs = [m for m in recent if m.get("sender") == "twily"]
            recent_twily_texts = [str(m.get("message") or "") for m in twily_msgs]
            if twily_msgs and str(twily_msgs[0].get("message", "")).strip() == message.strip():
                return Output(success=True, parts_sent=0, error="DUPLICATE_DETECTED")
            # Fuzzy near-duplicate: catches vibe-clone messages that survive exact-match
            # dedup (e.g. 3 overlapping cron fires producing ~paraphrased text).
            # 0.75 ratio on lower-cased text; unrelated Twily msgs score <0.5 in practice.
            msg_norm = message.strip().lower()
            if msg_norm:
                for prev in twily_msgs[:3]:
                    prev_norm = str(prev.get("message", "")).strip().lower()
                    if prev_norm and SequenceMatcher(None, prev_norm, msg_norm).ratio() >= 0.75:
                        return Output(success=True, parts_sent=0, error="NEAR_DUPLICATE_DETECTED")
        except Exception:
            pass

        message = message.replace("\\n", "\n")

        # Decode Unicode escape sequences (e.g., \ud83c\udf19 → 🌙)
        # Handle surrogate pairs first (\uD800-\uDFFF), then single escapes
        def _decode_unicode_escapes(text: str) -> str:
            def _replace_surrogates(m: re.Match) -> str:
                hi, lo = int(m.group(1), 16), int(m.group(2), 16)
                return chr(0x10000 + (hi - 0xD800) * 0x400 + (lo - 0xDC00))

            text = re.sub(
                r"\\u([dD][89aAbB][0-9a-fA-F]{2})\\u([dD][cCdDeEfF][0-9a-fA-F]{2})", _replace_surrogates, text
            )
            text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
            return text

        message = _decode_unicode_escapes(message)

        # ── Rule-based style scorer: strip banned phrases, cap emoji/tildes, log events ──
        try:
            from app.db.repos.persona_vibe import StyleEventsRepo
            from app.services.style_scorer import score_and_rewrite

            chat_id_int = int(settings.chat_id)
            polished, violations = await score_and_rewrite(
                message, chat_id=chat_id_int, recent_twily_msgs=recent_twily_texts
            )
            if polished and polished.strip():
                message = polished
            if violations:
                events_repo = StyleEventsRepo()
                # Build events list for bulk log.
                events = [
                    {
                        "violation_type": v.get("violation_type", ""),
                        "details": v.get("details"),
                        "before": v.get("before"),
                        "after": v.get("after"),
                        "enforced": bool(v.get("enforced", True)),
                    }
                    for v in violations
                ]
                await events_repo.log_many(chat_id_int, events)
        except Exception as scorer_exc:
            # Never let the scorer break message delivery.
            import sys

            print(f"[send_message] style_scorer skipped: {scorer_exc}", file=sys.stderr)

        # Prepend header from env var (mode/model indicator) — excluded from TTS and chat history
        import os

        header = os.environ.get("FREN_MSG_HEADER", "")
        display_message = f"{header}\n{message}" if header else message
        tts_message = message  # Original without header for TTS

        # Auto-insert paragraph breaks before section headers. The LLM often
        # outputs "...intro text 💜 **HIGH (6):**\n- first bullet..." with only
        # a SPACE before the header, producing a wall-of-text in Telegram.
        # Insert \n\n before any inline section header that isn't already
        # at paragraph start.
        _header_re = re.compile(
            r"(?P<prefix>[^\n])(?P<gap>[ \t]*|\n)"
            r"(?P<hdr>"
            # **BOLD HEADER:** — starts ALL-CAPS, allows lowercase inside parens
            r"\*\*[A-Z][A-Z \d&/()-]{1,}(?:\([^)]{1,60}\))?[A-Z \d&/()-]*:?\*\*:?"
            r"|#{2,6}\s+\S"  # ## markdown header
            r"|[A-Z][A-Z &/()\d-]{3,60}:(?=\s|$)"  # bare ALL-CAPS HEADER:
            r")"
        )

        def _insert_break(m: re.Match[str]) -> str:
            # Already separated by blank line? leave alone.
            start = m.start()
            if start >= 2 and display_message[start - 1 : start + 1] == "\n\n":
                return m.group(0)
            return f"{m.group('prefix')}\n\n{m.group('hdr')}"

        display_message = _header_re.sub(_insert_break, display_message)
        # Collapse any run of 3+ newlines that our insertion may have created.
        display_message = re.sub(r"\n{3,}", "\n\n", display_message)

        paragraphs = [p.strip() for p in display_message.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [display_message]

        # If any paragraph is still very long (>1500 chars), split on single
        # newlines between bullet-groups as a fallback.
        final_paragraphs: list[str] = []
        for p in paragraphs:
            if len(p) <= 1500:
                final_paragraphs.append(p)
                continue
            # Split before unicode-bullet blocks or numbered-list runs.
            chunks = re.split(r"\n(?=(?:[\u2022\u25AA\u25CF\u25E6\u2023\u2981⦁] ))", p)
            final_paragraphs.extend([c.strip() for c in chunks if c.strip()])
        paragraphs = final_paragraphs

        bot = Bot(token=settings.bot_token)
        try:
            await bot.initialize()
            for i, paragraph in enumerate(paragraphs):
                sent = False
                if has_tgmd:
                    try:
                        converted = telegramify_markdown.markdownify(paragraph)
                        await bot.send_message(
                            chat_id=settings.chat_id,
                            text=converted,
                            parse_mode=ParseMode.MARKDOWN_V2,
                        )
                        sent = True
                    except TelegramError:
                        pass
                if not sent:
                    await bot.send_message(chat_id=settings.chat_id, text=paragraph)
                if i < len(paragraphs) - 1:
                    await asyncio.sleep(0.5)

            # Save to chat history (inherit content_class from env)
            try:
                from datetime import UTC, datetime

                from app.db.repos.chat import ChatMessagesRepo

                repo = ChatMessagesRepo()
                now = datetime.now(UTC)
                content_class = os.environ.get("FREN_CONTENT_CLASS", "public")
                sfw_summary = {
                    "nsfw": "[private conversation]",
                    "secret": "[confidential message]",
                }.get(content_class)
                await repo.save(
                    sender="twily",
                    message=message,
                    date=now.date(),
                    timestamp=now,
                    timestamp_unix=now.timestamp(),
                    content_class=content_class,
                    sfw_summary=sfw_summary,
                )
            except Exception:
                pass

            # Fire TTS as detached background process — don't block message delivery
            self._spawn_background_tts(tts_message, settings)

            return Output(success=True, parts_sent=len(paragraphs))
        except TelegramError as e:
            return Output(success=False, error=str(e))
        finally:
            await bot.shutdown()

    @staticmethod
    def _spawn_background_tts(text: str, settings: object) -> None:
        """Spawn send_voice.py in a detached subprocess for async TTS.

        Passes raw text via temp file — send_voice handles LLM formatting.
        """
        import logging
        import os
        import subprocess
        import tempfile
        from pathlib import Path

        logger = logging.getLogger(__name__)

        if not text or len(text.strip()) < 10:
            return

        project_root = Path(settings.project_root).resolve()
        workspace = project_root / ".agent_workspace"
        workspace.mkdir(exist_ok=True)

        # Write raw text to temp file so send_voice can LLM-format it
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="tts_", dir=str(workspace))
            with os.fdopen(fd, "w") as f:
                f.write(text)
        except Exception as e:
            logger.error("Failed to write TTS temp file: %s", e)
            return

        try:
            subprocess.Popen(
                [
                    "uv",
                    "run",
                    "scripts/send_voice.py",
                    "--text_file",
                    tmp_path,
                    "--voice",
                    "twilight_neutral_v2",
                    "--speed",
                    str(settings.tts_speed),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(project_root),
                env={**os.environ},
            )
        except Exception as e:
            logger.error("Failed to spawn background TTS: %s", e)
            # Clean up temp file on spawn failure
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    SendMessageTool.run()
