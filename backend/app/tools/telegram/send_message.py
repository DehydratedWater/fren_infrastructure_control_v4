"""Send text message to Telegram."""

from __future__ import annotations

import asyncio
import os
import re
import sys

from src import ScriptTool, StreamFormat
from pydantic import BaseModel, Field


class Input(BaseModel):
    message: str = Field(description="Message text to send")


class Output(BaseModel):
    success: bool = True
    parts_sent: int = 0
    error: str = ""
    suppressed: bool = False
    reason: str = ""


def _gate_message(
    message: str, recent_twily_texts: list[str], policy: dict | None,
    *, kind: str = "reply", last_user_age_s: float | None = None,
    last_bot_age_s: float | None = None,
) -> Output | None:
    """The thin delivery-gate seam: None → deliver; Output → suppressed.

    Pure (no I/O) so it is unit-testable in isolation; the verdict comes
    from app.delivery.gate.evaluate_message — the autoloop-optimised
    policy (component_id "policy:delivery_gate"). `kind`/`last_*_age_s` drive
    the proactive background-cooldown (a proactive send is suppressed when the
    user is actively chatting or the bot just spoke). A suppressed message
    reports success=True + suppressed=True so agents treat the task as
    complete instead of retry-spamming. Any gate error → deliver (the
    gate must never block real messages).
    """
    try:
        from app.delivery.gate import evaluate_message

        decision = evaluate_message(
            message, recent_twily_texts, policy,
            kind=kind, last_user_age_s=last_user_age_s, last_bot_age_s=last_bot_age_s,
        )
    except Exception as exc:  # noqa: BLE001 — gate failure must not block delivery
        print(f"[send_message] delivery gate skipped: {exc}", file=sys.stderr)
        return None
    if decision.deliver:
        return None
    # Keep the historical duplicate marker (agents' output_note contract);
    # noop/leak/too_short get explicit SUPPRESSED_* markers.
    error = (
        "DUPLICATE_DETECTED"
        if decision.reason == "duplicate"
        else f"SUPPRESSED_{decision.reason.upper()}"
    )
    print(
        f"[send_message] suppressed ({decision.reason}):"
        f" matched={decision.matched[:120]!r}",
        file=sys.stderr,
    )
    return Output(
        success=True,
        parts_sent=0,
        error=error,
        suppressed=True,
        reason=decision.reason,
    )


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
    output_note = (
        "If DUPLICATE_DETECTED or SUPPRESSED_*: the message was gated "
        "(already sent / no-op / internal leak). STOP. Do NOT retry. "
        "Your task is complete."
    )

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

        # ── Delivery-quality gate (autoloop-optimised "policy:delivery_gate") ──
        # Replaces the old hardcoded dedup (exact + SequenceMatcher>=0.75 vs last
        # 3-5) with the pure policy gate: leak → noop → too_short → dedup. The
        # recent Twily messages are captured once — the gate dedups against them
        # and the rule-based style scorer reuses them below.
        policy: dict | None = None
        try:
            from app.delivery.gate import active_policy

            policy = active_policy()
        except Exception as exc:  # noqa: BLE001 — policy load must not block sends
            print(f"[send_message] active_policy unavailable: {exc}", file=sys.stderr)
        recent_twily_texts: list[str] = []
        last_user_age_s: float | None = None
        last_bot_age_s: float | None = None
        try:
            import time as _time

            from app.db.repos.chat import ChatMessagesRepo

            lookback = int((policy or {}).get("dedup_lookback", 8))
            repo = ChatMessagesRepo()
            # get_recent returns ALL senders; over-fetch so `lookback` Twily
            # messages survive the filter even in a chatty mixed window.
            recent = await repo.get_recent(limit=max(2 * lookback, 10))
            recent_twily_texts = [
                str(m.get("message") or "")
                for m in recent
                if m.get("sender") == "twily"
            ]
            # Ages drive the proactive cooldown (v3 background-cooldown parity):
            # most-recent user message age and most-recent bot message age.
            now = _time.time()
            for m in recent:  # most-recent-first
                ts = m.get("timestamp_unix")
                if ts is None:
                    continue
                age = now - float(ts)
                sender = m.get("sender")
                if sender == "twily" and last_bot_age_s is None:
                    last_bot_age_s = age
                elif sender != "twily" and last_user_age_s is None:
                    last_user_age_s = age
                if last_bot_age_s is not None and last_user_age_s is not None:
                    break
        except Exception:
            pass
        # Proactive runs mark themselves so the cooldown applies ONLY to them;
        # conversational replies are never cooldown-gated. The scheduler/proactive
        # spawn sets FREN_MSG_KIND (else default "reply").
        kind = os.environ.get("FREN_MSG_KIND", "reply")
        gated = _gate_message(
            message, recent_twily_texts, policy,
            kind=kind, last_user_age_s=last_user_age_s, last_bot_age_s=last_bot_age_s,
        )
        if gated is not None:
            return gated

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
                    "python",
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
