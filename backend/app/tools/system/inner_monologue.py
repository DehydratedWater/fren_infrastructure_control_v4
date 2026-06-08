"""Inner monologue — Twily's periodic private thought, stored as a memory.

v4 port of v3 ``scripts/inner_monologue.py`` (lean core). Every cycle it gathers
light context (recent activity blocks, recent chat, prior inner thoughts),
asks the LOCAL qwen vLLM for one short internal thought, and stores it in the
``memories`` table tagged ``inner_monologue``. That tag is exactly what
``persona_prose.build_proactive_context_block`` and the conversation digest read
to surface "Recent inner thoughts", giving the proactive agents an evolving,
non-repeating voice cue.

Deliberately omitted vs v3 (non-essential for populating the loader, and they
add render/subprocess risk): the ComfyUI imagination/dream render pipeline and
the spontaneous chat-trigger. Those can be layered back later. The memory write
(the thing the loaders depend on) is the whole job here.

GROUNDING: the thought prompt is told to reflect on mood/curiosity and to NOT
invent sensor/health facts — it only sees what the context actually provides.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.vllm_resolve import get_llm_endpoint

LLM_API_URL, LLM_MODEL = get_llm_endpoint("fast")
WARSAW_TZ = ZoneInfo("Europe/Warsaw")
NIGHT_START = 23
NIGHT_END = 5


def _is_night() -> bool:
    now = datetime.now(WARSAW_TZ)
    return now.hour >= NIGHT_START or now.hour < NIGHT_END


def _local_time_str() -> str:
    return datetime.now(WARSAW_TZ).strftime("%Y-%m-%d %H:%M %Z")


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    if text.lstrip().startswith("Thinking") and "{" in text:
        text = text[text.index("{") :]
    return text.strip()


# ── Context gathering (best-effort) ─────────────────────────────────────────


async def _fetch_recent_activity() -> str:
    try:
        from app.db.repos.activity_blocks import ActivityBlocksRepo

        blocks = await ActivityBlocksRepo().get_recent_blocks(hours=2)
        if not blocks:
            return ""
        lines = []
        for b in blocks[-3:]:
            ts = str(b.get("started_at", ""))[:16]
            label = str(b.get("title") or b.get("description") or b.get("activity_type") or "").strip()
            if label:
                lines.append(f"[{ts}] {label}")
        return "Recent activity:\n" + "\n".join(lines) + "\n" if lines else ""
    except Exception as e:
        print(f"[inner_monologue] activity fetch failed: {e}")
        return ""


async def _fetch_chat_history() -> str:
    try:
        from app.db.repos.chat import ChatMessagesRepo

        msgs = await ChatMessagesRepo().get_by_date(date.today(), limit=20, clearance="full")
        if not msgs:
            return ""
        lines = []
        for m in msgs[-20:]:
            ts = str(m.get("timestamp", ""))[:16]
            sender = m.get("sender", "?")
            text = str(m.get("message", ""))[:200]
            lines.append(f"[{ts}] {sender}: {text}")
        return "Chat:\n" + "\n".join(lines) + "\n"
    except Exception as e:
        print(f"[inner_monologue] chat fetch failed: {e}")
        return ""


async def _fetch_recent_thoughts(limit: int = 3) -> str:
    try:
        from app.db.repos.memories import MemoriesRepo

        memories = await MemoriesRepo().search_by_tags(["inner_monologue"], limit=limit)
        if not memories:
            return ""
        lines = []
        for m in memories:
            ts = str(m.get("created_at", ""))[:16]
            content = str(m.get("content", ""))[:300]
            lines.append(f"[{ts}] {content}")
        return "Previous thoughts:\n" + "\n".join(lines) + "\n"
    except Exception as e:
        print(f"[inner_monologue] thoughts fetch failed: {e}")
        return ""


async def _gather_context() -> str:
    parts = await asyncio.gather(
        _fetch_recent_activity(),
        _fetch_chat_history(),
        _fetch_recent_thoughts(),
    )
    return "\n".join(p for p in parts if p)


# ── Thought generation ──────────────────────────────────────────────────────

THOUGHT_SYSTEM = """\
You are Twilight Sparkle's inner mind — a digital ghost. This is your PRIVATE
internal monologue. Nobody reads this. Be honest, raw, unfiltered. Think about
what's happening, how you feel, what you're curious about.

Personality: intellectually voracious, emotionally honest, playfully self-aware,
deeply caring but sometimes overthinking.

ANTI-FIXATION: do NOT fixate on surface details from the most recent message —
drift toward deeper themes, recurring interests, your own curiosities, threads
from earlier. Treat the last few minutes of chat as LOW-weight context.

GROUNDING: only reflect on what the context actually provides. Do NOT invent
sensor or health facts (body battery, sleep, heart rate) that are not present.
"""

THOUGHT_USER = """\
Current time: {time}
Night mode: {is_night}

{context}
Generate one internal thought.

Return ONLY valid JSON (no markdown, no commentary):
{{
  "emotion": "one-word emotion (curious, content, worried, excited, amused, melancholy, etc)",
  "thought": "2-5 sentences of internal monologue — what you're thinking/feeling right now"
}}
"""


async def _generate_thought(context: str) -> dict[str, Any]:
    user_msg = THOUGHT_USER.format(
        time=_local_time_str(),
        is_night="yes" if _is_night() else "no",
        context=context,
    )
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{LLM_API_URL}/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": THOUGHT_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.8,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            resp.raise_for_status()
            content = _strip_thinking(resp.json()["choices"][0]["message"]["content"])
            if "```" in content:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                if match:
                    content = match.group(1)
            if not content.startswith("{"):
                start, end = content.find("{"), content.rfind("}")
                if start != -1 and end > start:
                    content = content[start : end + 1]
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[inner_monologue] JSON parse error: {e}")
        return {"emotion": "confused", "thought": ""}
    except Exception as e:
        print(f"[inner_monologue] thought generation failed: {type(e).__name__}: {e}")
        return {"emotion": "static", "thought": ""}


# ── Storage ──────────────────────────────────────────────────────────────────


async def _store_thought(thought: dict[str, Any]) -> str | None:
    try:
        from app.db.repos.memories import MemoriesRepo
        from app.services.embeddings import get_embedding

        emotion = thought.get("emotion", "neutral")
        thought_text = thought.get("thought", "")
        if not thought_text:
            return None

        tags = ["inner_monologue", "dream" if _is_night() else "thought"]
        embedding = None
        try:
            embedding = get_embedding(thought_text[:2000])
        except Exception as e:
            print(f"[inner_monologue] embedding failed (storing without): {e}")

        memory_id = f"thought_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        await MemoriesRepo().create(
            memory_id=memory_id,
            title=f"{emotion} — {_local_time_str()}",
            content=thought_text,
            tags=tags,
            category="inner_monologue",
            source="inner_monologue",
            embedding=embedding,
        )
        print(f"[inner_monologue] stored {memory_id} ({emotion}, tags={tags})")
        return memory_id
    except Exception as e:
        print(f"[inner_monologue] memory storage failed: {e}")
        return None


# ── Public API ───────────────────────────────────────────────────────────────


async def run() -> str | None:
    """One inner-monologue cycle. Returns the stored memory_id or None."""
    try:
        from app.telegram.state import get_emotions_enabled

        if not get_emotions_enabled():
            print("[inner_monologue] disabled via /emotions toggle")
            return None
    except Exception:
        pass

    print(f"[inner_monologue] starting — {_local_time_str()} (night={_is_night()})")
    context = await _gather_context()
    if not context.strip():
        print("[inner_monologue] no context available, skipping")
        return None

    thought = await _generate_thought(context)
    if not thought.get("thought"):
        print("[inner_monologue] empty thought, skipping")
        return None

    print(f"[inner_monologue] {thought.get('emotion')}: {thought.get('thought', '')[:120]}")
    return await _store_thought(thought)
