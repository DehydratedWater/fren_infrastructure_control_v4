#!/usr/bin/env python3
"""Background worker: render PonyXL scene illustration and send via the RP bot.

Ported from v3 scripts/render_and_send_rp.py (fren.* -> app.*). Same as
render_and_send.py but uses BOT_RP_TOKEN instead of BOT_TOKEN, so images land in
the RP bot chat instead of the main Twily chat.

Spawned (detached) by app.tools.vis_simulation.render_rp_scene.RenderRPSceneTool
via `python scripts/render_and_send_rp.py <job.json>`.

The ComfyUI host/port is read from settings.get_comfyui_hosts() inside
render_scene() — not hardcoded here.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _render_scene_sync(**kwargs: object) -> dict:
    """Call app.comfyui.render.render_scene synchronously."""
    from app.comfyui.render import render_scene

    return asyncio.run(render_scene(**kwargs))


def _extract_output_path(render_result: dict) -> str | None:
    """Extract file path from render result. Downloads from remote ComfyUI if needed."""
    from app.comfyui.client import download_output

    output_files = render_result.get("data", {}).get("output_files", [])
    base_url = render_result.get("data", {}).get("base_url", "")
    if not output_files:
        return None
    of = output_files[0]
    filename = of.get("filename", "")
    if not filename:
        return None
    if base_url:
        local = download_output(base_url, of)
        if local:
            print(f"[download] {filename} -> {local}")
            return local
    return None


def _send_photo_rp(file_path: str, caption: str) -> dict:
    """Send photo via the RP bot (BOT_RP_TOKEN)."""
    from app.settings import get_settings

    settings = get_settings()
    if not settings.bot_rp_token or not settings.chat_id:
        return {"success": False, "error": "BOT_RP_TOKEN or CHAT_ID not configured"}

    from telegram import Bot

    bot = Bot(token=settings.bot_rp_token)
    chat_id = int(settings.chat_id)

    async def _send() -> dict:
        try:
            await bot.initialize()
            with open(file_path, "rb") as photo:
                await bot.send_photo(chat_id=chat_id, photo=photo, caption=caption or None)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            await bot.shutdown()

    return asyncio.run(_send())


def _send_error_message_rp(error: str) -> None:
    """Send error notification via the RP bot."""
    from app.settings import get_settings

    settings = get_settings()
    if not settings.bot_rp_token or not settings.chat_id:
        return

    from telegram import Bot

    bot = Bot(token=settings.bot_rp_token)
    chat_id = int(settings.chat_id)

    async def _send() -> None:
        try:
            await bot.initialize()
            await bot.send_message(chat_id=chat_id, text=f"[Illustration Error] {error[:500]}")
        except Exception:
            pass
        finally:
            await bot.shutdown()

    with contextlib.suppress(Exception):
        asyncio.run(_send())


def run_image_mode(job: dict) -> None:
    """Render T2I image and send to Telegram via the RP bot."""
    print(f"[render_and_send_rp] Image mode: {job.get('filename_prefix', 'rp_scene')}")

    result = _render_scene_sync(
        workflow_id=job["workflow_id"],
        positive_prompt=job["positive_prompt"],
        negative_prompt=job["negative_prompt"],
        filename_prefix=job.get("filename_prefix", "rp_scene"),
        seed=job.get("seed"),
        instance_id=job.get("instance_id"),
    )

    if not result.get("success"):
        err = result.get("error", "Unknown render error")
        print(f"[render_and_send_rp] Render failed: {err}", file=sys.stderr)
        _send_error_message_rp(f"Scene illustration failed: {err}")
        return

    output_path = _extract_output_path(result)
    if not output_path or not Path(output_path).exists():
        print("[render_and_send_rp] No output file found", file=sys.stderr)
        _send_error_message_rp("Scene illustration produced no output file")
        return

    elapsed = result.get("data", {}).get("elapsed_seconds", 0)
    print(f"[render_and_send_rp] Done in {elapsed}s -> {output_path}")

    caption = job.get("caption", "")
    send_result = _send_photo_rp(output_path, caption)
    if send_result.get("success"):
        print("[render_and_send_rp] Photo sent to Telegram via RP bot")
    else:
        print(f"[render_and_send_rp] Telegram send failed: {send_result.get('error')}", file=sys.stderr)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: render_and_send_rp.py <job_file.json>", file=sys.stderr)
        sys.exit(1)

    job_path = sys.argv[1]
    try:
        with open(job_path) as f:
            job = json.load(f)
    except Exception as e:
        print(f"Failed to read job file {job_path}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        run_image_mode(job)
    except Exception as e:
        print(f"[render_and_send_rp] Unhandled error: {e}", file=sys.stderr)
        _send_error_message_rp(f"Unhandled error: {e}")
    finally:
        with contextlib.suppress(Exception):
            Path(job_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
