"""What the user knows about Twily's world — a shared-knowledge ledger.

Her world (Mooring Wells) is PRIVATE by default: the user only knows a thread if
she has introduced it. This module remembers which threads have been *offered*
(she asked if they'd like to hear) or *shared* (she actually told them), so the
chat layer can have her INTRODUCE private things rather than reference them as if
already known, and BUILD ON things the user already learned — and keep that
straight across sessions.

The ledger lives in agent_notes (a small JSON list, long TTL). A tiny thinking-off
local-qwen classifier runs AFTER a reply that drew on her world, recording what she
actually disclosed in that message — so "shared vs private" reflects what she's
genuinely told the user, not just what happened in the sim.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.db.repos.agent_notes import AgentNotesRepo
from app.world.loader import DEFAULT_PACKAGE

logger = logging.getLogger(__name__)

_NOTE_PREFIX = "world_shared_topics"
_PERSIST_HOURS = 24 * 365 * 5  # effectively permanent
_RANK = {"offered": 1, "shared": 2}


def _note_key(world_id: str) -> str:
    return f"{_NOTE_PREFIX}::{world_id}"


async def shared_topics(world_id: str = DEFAULT_PACKAGE) -> list[dict[str, Any]]:
    """The threads the user already knows about, each {key,label,status}."""
    try:
        note = await AgentNotesRepo().get(_note_key(world_id))
    except Exception:  # noqa: BLE001
        return []
    if not note:
        return []
    val = note.get("note_value")
    items = val.get("topics") if isinstance(val, dict) else val
    return [t for t in items if isinstance(t, dict) and t.get("key")] if isinstance(items, list) else []


async def record_disclosures(items: list[dict[str, Any]], world_id: str = DEFAULT_PACKAGE) -> None:
    """Merge newly-disclosed threads into the ledger (status escalates
    offered → shared; never downgrades)."""
    if not items:
        return
    existing = {t["key"]: dict(t) for t in await shared_topics(world_id)}
    changed = False
    for it in items:
        key = str(it.get("key") or "").strip().lower().replace(" ", "_")[:60]
        if not key:
            continue
        status = it.get("status") if it.get("status") in _RANK else "offered"
        label = str(it.get("label") or key).strip()[:120]
        cur = existing.get(key)
        if cur is None:
            existing[key] = {"key": key, "label": label, "status": status}
            changed = True
        else:
            if _RANK[status] > _RANK.get(cur.get("status", "offered"), 1):
                cur["status"] = status
                changed = True
            if label and len(label) > len(cur.get("label", "")):
                cur["label"] = label
                changed = True
    if changed:
        try:
            await AgentNotesRepo().set(
                _note_key(world_id), {"topics": list(existing.values())},
                expires_hours=_PERSIST_HOURS,
            )
        except Exception:  # noqa: BLE001
            logger.exception("world.knowledge: failed to persist disclosures")


_CLASSIFY_SYS = (
    "You track what a USER has been told about Twily's private inner world. Given "
    "(A) a summary of Twily's recent private world-life and (B) a message Twily just "
    "sent the user, list ONLY the world threads Twily actually brought up IN THAT "
    "MESSAGE. For each: a short stable `key` (snake_case), a short human `label`, and "
    "`status`: \"shared\" if she told the user real content about it, or \"offered\" if "
    "she only floated/teased it or asked whether they'd like to hear. If the message "
    "did NOT bring up her private world at all, return an empty list. Return ONLY JSON: "
    '{"disclosures":[{"key":"...","label":"...","status":"shared|offered"}]}'
)


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}


async def classify_and_record(
    reply_text: str, world_life: str, *, world_id: str = DEFAULT_PACKAGE
) -> list[dict[str, Any]]:
    """Detect what Twily disclosed about her world in `reply_text` and record it.
    Best-effort: never raises. Returns the disclosures it recorded."""
    reply_text = (reply_text or "").strip()
    if not reply_text or not (world_life or "").strip():
        return []
    try:
        from app.vllm_resolve import get_llm_endpoint

        base, model = get_llm_endpoint("fast")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _CLASSIFY_SYS},
                {"role": "user", "content": f"(A) Her recent world-life:\n{world_life[:1500]}\n\n"
                                            f"(B) Message she just sent the user:\n{reply_text[:1500]}"},
            ],
            "max_tokens": 400,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{base}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001
        logger.debug("world.knowledge: disclosure classify skipped (LLM error)")
        return []

    disclosures = _extract_json(content).get("disclosures") or []
    disclosures = [d for d in disclosures if isinstance(d, dict) and d.get("key")]
    if disclosures:
        await record_disclosures(disclosures, world_id)
    return disclosures
