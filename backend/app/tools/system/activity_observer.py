"""Activity observer — periodic camera capture → vision describe → activity_blocks row.

v4's lean replacement for v3's heavy ``scripts/activity_observer.py``. Each cycle:

1. Capture a frame from the room/desk camera (reuses ``CameraCaptureTool``).
2. Describe the frame with the vision-capable local vLLM — a short, factual
   "what's happening in the room / at the desk" observation.
3. Write it as an ``activity_blocks`` row (activity_type='observation') so the
   proactive context loader's "Recent activity blocks" section has live,
   changing material instead of the same static seed.

This gives proactive agents real, varying room-state signal. It is the ONLY
camera→context path, and it never invents: if capture or vision fails, no row is
written (the loader simply has nothing to show, and the grounding contract tells
the agent it has no room-state data).

GROUNDING: the vision prompt asks for a literal description of what is visible.
It does NOT ask for or infer any health / body-battery / sleep figure.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.vllm_resolve import get_llm_endpoint

# Vision-capable endpoint (analytical role, same model analyze_media uses).
VISION_API_URL, VISION_MODEL = get_llm_endpoint()

_OBSERVE_PROMPT = (
    "You are a security-camera describer. In 1-2 plain sentences, state ONLY what is "
    "literally visible in this frame: is a person present at the desk, what are they "
    "apparently doing, and the general room state (lights on/off, screen on/off, tidy/cluttered). "
    "Be factual and concise. Do NOT guess mood, health, time of day, or anything not visible. "
    "If the frame is dark or empty, say so."
)


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _capture_frame(command: str = "webcam") -> str:
    """Capture a single frame. Returns the image path, or "" on failure."""
    try:
        from app.tools.system.camera_capture import CameraCaptureTool, Input

        out = CameraCaptureTool().execute(Input(command=command))
        if not out.success:
            print(f"[activity_observer] capture failed: {out.error}")
            return ""
        return out.webcam_path or out.desk_path
    except Exception as e:
        print(f"[activity_observer] capture error: {e}")
        return ""


async def _describe(image_path: str) -> str:
    """Describe the frame via the vision vLLM. Returns "" on any failure."""
    try:
        from app.tools.media.analyze_media import _encode_image

        b64 = _encode_image(Path(image_path))
    except Exception as e:
        print(f"[activity_observer] encode failed: {e}")
        return ""

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": _OBSERVE_PROMPT},
    ]
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{VISION_API_URL}/chat/completions",
                json={
                    "model": VISION_MODEL,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 512,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            return _strip_thinking(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"[activity_observer] vision call failed: {e}")
        return ""


async def _store_block(description: str, image_path: str) -> bool:
    """Write the observation as an activity_blocks row. Returns True on success."""
    try:
        from app.db.repos.activity_blocks import ActivityBlocksRepo

        now = datetime.now(UTC)
        title = description[:80].strip()
        block = {
            "started_at": now,
            "ended_at": None,
            "activity_type": "observation",
            "title": title,
            "description": description,
            "tags": ["camera", "room_state"],
            "confidence": 0.9,
            "environment": {"image_path": image_path},
            "health_snapshot": {},  # camera carries NO health — never fabricated
        }
        # insert_blocks (not replace_recent) so each observation accumulates as a
        # distinct timeline entry the loader can show as changing material.
        inserted = await ActivityBlocksRepo().insert_blocks(now.date(), [block])
        print(f"[activity_observer] stored observation ({inserted} row): {title}")
        return inserted > 0
    except Exception as e:
        print(f"[activity_observer] store failed: {e}")
        return False


async def run(command: str = "webcam") -> str | None:
    """One observe cycle. Returns the stored description, or None when skipped."""
    print(f"[activity_observer] starting cycle (command={command})")
    image_path = _capture_frame(command)
    if not image_path:
        print("[activity_observer] no frame captured, skipping (no row written)")
        return None

    description = await _describe(image_path)
    if not description:
        print("[activity_observer] no description, skipping (no row written)")
        return None

    print(f"[activity_observer] observed: {description[:120]}")
    await _store_block(description, image_path)
    return description
