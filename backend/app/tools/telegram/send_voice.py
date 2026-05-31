"""Send TTS voice message to Telegram."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field


def _chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text into chunks at sentence boundaries for TTS."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            # Very long sentence — split by commas
            if current:
                chunks.append(current.strip())
                current = ""
            parts = re.split(r"(?<=,)\s+", sentence)
            for part in parts:
                if current and len(current) + len(part) + 1 > max_chars:
                    chunks.append(current.strip())
                    current = part
                else:
                    current = f"{current} {part}" if current else part
        elif current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


class Input(BaseModel):
    text: str = Field(default="", description="Text to convert to speech and send")
    text_file: str = Field(default="", description="Path to file containing text (deleted after read)")
    voice: str = Field(default="twilight_neutral_v2", description="TTS voice name")
    speed: float = Field(default=0.85, description="TTS speech speed multiplier")


class Output(BaseModel):
    success: bool = True
    error: str = ""


class SendVoiceTool(ScriptTool[Input, Output]):
    name = "send_voice"
    description = "Generate TTS audio and send as Telegram voice message"

    def execute(self, inp: Input) -> Output:
        # Resolve text: prefer text_file, fall back to direct text
        text = inp.text
        if inp.text_file:
            try:
                p = Path(inp.text_file)
                text = p.read_text()
                p.unlink(missing_ok=True)
            except Exception:
                pass

        if not text or len(text.strip()) < 10:
            return Output(success=False, error="Text too short for TTS")

        # Format for speech: LLM first, regex fallback
        clean = self._format_for_tts(text)
        if not clean or len(clean.strip()) < 10:
            return Output(success=False, error="Formatted text too short for TTS")

        return asyncio.run(self._send(clean, inp.voice, inp.speed))

    @staticmethod
    def _format_for_tts(text: str) -> str:
        """Format text for TTS. Always tries LLM formatter first, regex fallback."""
        import logging
        import os
        import subprocess

        from app.tools.telegram.send_message import _strip_formatting

        logger = logging.getLogger(__name__)

        # Use dedicated TTS model if set, otherwise inherit from parent env
        tts_postfix = os.environ.get("FREN_TTS_POSTFIX", "") or os.environ.get("FREN_MODEL_POSTFIX", "")
        agent = f"support/tts_formatter{tts_postfix}"
        from app.settings import get_settings

        project_root = get_settings().project_root

        try:
            result = subprocess.run(
                ["opencode", "run", "--agent", agent, text],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=project_root,
                env={
                    **os.environ,
                    "XDG_DATA_HOME": os.path.join(project_root, ".opencode", "data"),
                },
            )
            if result.returncode == 0 and result.stdout.strip():
                raw = result.stdout.strip()
                # Extract content from <tts> tags if present
                import re as _re

                tts_match = _re.search(r"<tts>(.*?)</tts>", raw, _re.DOTALL)
                formatted = tts_match.group(1).strip() if tts_match else raw
                if len(formatted) >= 10:
                    logger.info("TTS: LLM formatter produced %d chars", len(formatted))
                    return formatted
                logger.warning("TTS: LLM formatter output too short (%d chars), using regex", len(formatted))
        except subprocess.TimeoutExpired:
            logger.warning("TTS: LLM formatter timed out, using regex fallback")
        except Exception as e:
            logger.warning("TTS: LLM formatter failed (%s), using regex fallback", e)

        return _strip_formatting(text)

    async def _send(self, text: str, voice: str, speed: float) -> Output:
        from app.settings import get_settings

        settings = get_settings()
        if not settings.bot_token or not settings.chat_id:
            return Output(success=False, error="BOT_TOKEN or CHAT_ID not configured")

        wav_path = await self._generate_voice(text, settings.tts_host, voice, speed)
        if not wav_path:
            return Output(success=False, error="TTS generation failed")

        from telegram import Bot
        from telegram.error import TelegramError

        ogg_path = wav_path.rsplit(".", 1)[0] + ".ogg"
        self._wav_to_ogg(wav_path, ogg_path)
        send_path = ogg_path if Path(ogg_path).exists() else wav_path

        bot = Bot(token=settings.bot_token)
        try:
            await bot.initialize()
            with open(send_path, "rb") as audio:
                await bot.send_voice(chat_id=settings.chat_id, voice=audio)
            return Output(success=True)
        except TelegramError as e:
            return Output(success=False, error=str(e))
        finally:
            await bot.shutdown()
            import contextlib

            with contextlib.suppress(OSError):
                Path(wav_path).unlink(missing_ok=True)
                Path(ogg_path).unlink(missing_ok=True)

    @staticmethod
    def _wav_to_ogg(wav_path: str, ogg_path: str) -> None:
        import contextlib
        import subprocess

        with contextlib.suppress(Exception):
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", ogg_path],
                capture_output=True,
                timeout=120,
            )

    @staticmethod
    def _concat_wav_files(wav_paths: list[str], output_path: str) -> bool:
        """Concatenate multiple WAV files using ffmpeg concat demuxer."""
        import subprocess
        import tempfile

        if len(wav_paths) == 1:
            import shutil

            shutil.copy2(wav_paths[0], output_path)
            return True

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in wav_paths:
                f.write(f"file '{p}'\n")
            list_file = f.name

        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, output_path],
                capture_output=True,
                timeout=120,
            )
            return result.returncode == 0
        except Exception:
            return False
        finally:
            Path(list_file).unlink(missing_ok=True)

    async def _generate_voice(self, text: str, tts_host: str, voice: str, speed: float) -> str | None:
        import logging

        import httpx

        from app.settings import get_settings

        logger = logging.getLogger(__name__)
        chunks = _chunk_text(text)
        url = f"http://{tts_host}/v1/tts/batch"

        # Build batch items — only pad the last chunk to prevent clipping
        items = []
        for i, chunk in enumerate(chunks):
            padded = chunk.rstrip() + (" . . ." if i == len(chunks) - 1 else "")
            items.append({"text": padded, "voice": voice, "speed": speed})

        if len(chunks) > 1:
            logger.info("TTS: split %d chars into %d chunks", len(text), len(chunks))

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(url, json={"items": items})
                resp.raise_for_status()
                data = resp.json()
                resp_items = data.get("items", [])

                # Collect successful output paths
                settings = get_settings()
                wav_paths: list[str] = []
                for item in resp_items:
                    if item.get("success"):
                        container_path = str(item["output_path"])
                        if container_path.startswith("/output/"):
                            wav_paths.append(f"{settings.tts_output_dir}/{container_path[8:]}")
                        else:
                            wav_paths.append(container_path)

                if not wav_paths:
                    return None

                if len(wav_paths) == 1:
                    return wav_paths[0]

                # Concatenate multiple WAV chunks
                combined = wav_paths[0].rsplit(".", 1)[0] + "_combined.wav"
                if self._concat_wav_files(wav_paths, combined):
                    for p in wav_paths:
                        Path(p).unlink(missing_ok=True)
                    return combined

                logger.warning("TTS: WAV concat failed, using first chunk only")
                return wav_paths[0]
        except Exception:
            pass
        return None


if __name__ == "__main__":
    SendVoiceTool.run()
